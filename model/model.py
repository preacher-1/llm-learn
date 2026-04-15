from typing import Optional, Tuple, List, Union
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast


class MyModelConfig(PretrainedConfig):
    model_type = "mymodel"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_dim: int = 512,
        intermediate_size: int = 1365,  # 512*8/3
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        partial_rotary_factor: float = 0.5,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = True,
        num_experts_per_tok: int = 2,  # 每个token路由到的专家数量
        first_k_dense_replace: int = 1,  # 从第几层开始用MoE替换FFN
        n_routed_experts: int = 8,  # 总的路由专家数量
        n_shared_experts: int = 1,  # 共享专家数量
        routed_scaling_factor: float = 1.4,
        moe_intermediate_size: Optional[int] = None,  # MoE层的intermediate_size，如果为None则在类中计算
        bias_update_speed: float = 0.001,  # 路由分数偏置的更新速度
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_dim = hidden_dim
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.moe_intermediate_size = moe_intermediate_size
        self.bias_update_speed = bias_update_speed
        self.norm_topk_prob = norm_topk_prob

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))  # gamma
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        varience = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(varience + self.eps)  # reverse sqrt
        return self.weight * x


def precompute_freqs(
    head_dim: int,
    seq_len: int,
    theta: int = 1000000,
    partial_rotary_factor: float = 1.0,
):
    # 1. freq = 1/(theta^(i/dim)) for i = 0, 2, 4, ..., dim-2
    dim = int(head_dim * partial_rotary_factor)
    dim = dim // 2 * 2  # 确保dim是偶数

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    # TODO: 实现YaRN
    # 2. 将位置 t 与 freqs 相乘，freqs_i = theta^(-2i/dim)
    t = torch.arange(seq_len, device=freqs.device).float()
    freqs = torch.outer(t, freqs).float()  # (seq_len, dim//2)

    # 3. 填充余弦和正弦值到 (seq_len, hidden_dim) 的矩阵中
    half = head_dim // 2
    freqs_cos = torch.ones(1, seq_len, head_dim, device=freqs.device)
    p_cos = freqs.cos()
    freqs_cos[:, :, : dim // 2] = p_cos
    freqs_cos[:, :, half : half + dim // 2] = p_cos

    freqs_sin = torch.zeros(1, seq_len, head_dim, device=freqs.device)
    p_sin = freqs.sin()
    freqs_sin[:, :, : dim // 2] = p_sin
    freqs_sin[:, :, half : half + dim // 2] = p_sin

    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    q, k: [batch, num_heads, seq_len, head_dim]
    cos, sin: [1, seq_len, head_dim]
    position_ids: [batch, seq_len] 可选
    unsqueeze_dim: 扩展维度位置（1 或 2），默认为1意味着将在num_heads维度上扩展以匹配q/k的形状，对应的q/k形状为[batch, num_heads, seq_len, head_dim]，即在q/k应用transpose(1, 2)之后该函数被调用
    """
    # 1. 扩展维度以匹配 q/k 的广播
    cos = cos.unsqueeze(unsqueeze_dim)  # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(unsqueeze_dim)

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # x: (batch_size, num_kv_heads, seq_len, head_dim)
    bsz, num_kv_heads, seq_len, head_dim = x.shape
    # 使用expand和reshape来重复每个头n_rep次，此处n_rep是在外部计算好的
    if n_rep == 1:
        return x
    x = x.unsqueeze(2)  # (bsz, num_kv_heads, 1, seq_len, head_dim)
    x = x.expand(bsz, num_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(bsz, num_kv_heads * n_rep, seq_len, head_dim)


class Attention(nn.Module):
    def __init__(self, config: MyModelConfig):
        super().__init__()
        self.config = config

        assert (
            config.hidden_dim % config.num_attention_heads == 0
        ), "hidden_dim must be divisible by num_attention_heads"
        self.head_dim = config.hidden_dim // config.num_attention_heads
        self.n_rep = config.num_attention_heads // config.num_key_value_heads

        self.q_proj = nn.Linear(
            config.hidden_dim, config.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_dim, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_dim, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_dim, bias=False
        )

        # 参考Qwen3使用QKNorm
        self.q_norm = RMSNorm(config.num_attention_heads * self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(config.num_key_value_heads * self.head_dim, eps=config.rms_norm_eps)

        # 是否使用flash attention
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and config.flash_attention
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # hidden_states: (batch_size, seq_len, hidden_dim)
        # position_embeddings: (cos, sin) each of shape (seq_len, head_dim)
        # attention_mask: (batch_size, 1, 1, seq_len) or None
        # past_key_values: ((batch_size, num_attention_heads, past_seq_len, head_dim), (batch_size, num_attention_heads, past_seq_len, head_dim)) or None
        bsz, seq_len, _ = hidden_states.shape
        hidden_shape = (bsz, seq_len, -1, self.head_dim)

        # 1. 线性变换得到 q, k, v
        query = self.q_norm(self.q_proj(hidden_states)).view(hidden_shape)
        key = self.k_norm(self.k_proj(hidden_states)).view(hidden_shape)
        value = self.v_proj(hidden_states).view(hidden_shape)

        query = query.transpose(1, 2)  # (bsz, num_heads, seq_len, head_dim)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # 2. 应用旋转位置编码
        cos, sin = position_embeddings  # (seq_len, head_dim)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # 3. 更新 KV Cache
        if past_key_values is not None:
            # 将当前的key和value与过去的key和value拼接起来
            past_key, past_value = past_key_values
            # (bsz, num_kv_heads, past_seq_len + seq_len, head_dim)
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)
        past_key_values = (key, value) if use_cache else None

        # 4. 扩展key和value以匹配query的num_attention_heads
        key = repeat_kv(
            key, self.n_rep
        )  # (bsz, seq_len, num_attention_heads, head_dim)
        value = repeat_kv(value, self.n_rep)

        if self.flash and (seq_len > 1) and (past_key_values is None):
            if attention_mask is None or torch.all(attention_mask == 1):
                # 直接使用原生 Causal 加速 (此时性能最高，可触发纯正 FlashAttention)
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=None, is_causal=True
                )
            else:                
                # 扩充 Padding Mask 为 bool 类型: [bsz, 1, 1, seq_len]
                if attention_mask.dim() == 2:
                    pad_mask = attention_mask.unsqueeze(1).unsqueeze(2).bool() 
                else:
                    pad_mask = attention_mask.bool()
                
                # 构造 Causal Mask (下三角为 True): [1, 1, seq_len, seq_len]
                causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=query.device))
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
                
                # 合并 Mask (利用广播机制，结果为 [bsz, 1, seq_len, seq_len])
                # 只有既是有效 token (pad_mask) 又是当前位置之前的 token (causal_mask) 才为 True
                final_mask = pad_mask & causal_mask
                
                # 传给 SDPA (此时必须设置 is_causal=False，因为 causal 逻辑已经在 final_mask 里了)
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=final_mask, is_causal=False
                )
        else:
            attn_scores = query @ key.transpose(-1, -2) / (self.head_dim**0.5)
            
            # Causal mask 处理
            attn_scores[:, :, :, -seq_len:] += torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=query.device),
                diagonal=1,
            ) 
            
            # Padding mask 处理
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attn_scores = attn_scores.masked_fill(
                    attention_mask == 0, float("-inf")
                )
                
            attn_scores = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)
            attn_output = attn_scores @ value  # (bsz, num_heads, seq_len, head_dim)
            
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        attn_output = self.o_proj(attn_output)  # (bsz, seq_len, hidden_dim)

        return attn_output, past_key_values


class FFN(nn.Module):
    def __init__(self, config: MyModelConfig, intermediate_size: Optional[int] = None):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )

        self.up_gate_proj = nn.Linear(
            self.hidden_dim, self.intermediate_size * 2, bias=False
        )
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_dim, bias=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (batch_size, seq_len, hidden_dim)
        up, gate = self.up_gate_proj(hidden_states).chunk(2, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)


class MoeRouter(nn.Module):
    def __init__(self, config: MyModelConfig):
        super().__init__()
        self.config = config
        self.top_k = config.n_routed_experts
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = nn.Parameter(
            torch.randn(self.n_routed_experts, config.hidden_dim)
        )  # (n_routed_experts, hidden_dim)
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(self.n_routed_experts, dtype=torch.float32),
        )
        # 累积当前 step 内各专家被选中的 token 数
        # persistent=False：不需要保存进 checkpoint，每 step 重置
        self.register_buffer(
            "_expert_load_accum",
            torch.zeros(self.n_routed_experts, dtype=torch.float32),
            persistent=False,
        )
        self.bias_update_speed = config.bias_update_speed

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # hidden_states: (batch_size, seq_len, hidden_dim)
        bsz, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.view(
            bsz * seq_len, -1
        )  # (bsz*seq_len, hidden_dim)

        # 计算路由分数
        scores = nn.functional.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32)
        )  # (bsz*seq_len, n_routed_experts)
        scores = scores.sigmoid()

        # 加上负载均衡偏置
        scores_for_choice = scores + self.e_score_correction_bias
        _, top_k_index = torch.topk(scores_for_choice, k=self.top_k, dim=-1)
        weights = scores.gather(-1, top_k_index)  # (bsz*seq_len, top_k)
        if self.norm_topk_prob:
            weights_sum = weights.sum(dim=-1, keepdim=True)
            weights = weights / (weights_sum + 1e-10)
        weights = weights * self.routed_scaling_factor
        return weights.type_as(hidden_states), top_k_index

    def update_bias(self):
        # 根据当前 step 内各专家被选中的 token 数来更新路由分数偏置
        with torch.no_grad():
            load = self._expert_load_accum  # (n_routed_experts,)
            mean_load = load.mean()
            # 分配固定的正负更新量而不依赖于具体的负载差值大小
            delta = torch.where(
                self._expert_load_accum > mean_load,
                torch.full_like(self._expert_load_accum, -self.bias_update_speed),
                torch.full_like(self._expert_load_accum, self.bias_update_speed),
            )
            self.e_score_correction_bias += delta
            self._expert_load_accum.zero_()  # 重置累积


class MoeExpert(nn.Module):
    """
    单个专家的前馈网络
    """

    def __init__(self, config: MyModelConfig):
        super().__init__()
        self.config = config
        self.n_routed_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_dim
        if config.moe_intermediate_size is not None:
            self.intermediate_size = config.moe_intermediate_size
        else:
            self.intermediate_size = config.intermediate_size // config.n_routed_experts

        self.up_gate_proj = nn.Linear(
            self.hidden_dim, self.intermediate_size * 2, bias=False
        )
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_dim, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # hidden_states: (, hidden_dim)
        # 这里的hidden_states是被路由到当前专家的token组成的一个小批次
        up, gate = self.up_gate_proj(hidden_states).chunk(2, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)


class MoE(nn.Module):
    def __init__(self, config: MyModelConfig):
        super().__init__()
        self.config = config
        self.n_routed_experts = config.n_routed_experts
        self.top_k = config.n_experts_per_tok

        self.router = MoeRouter(config)
        self.experts = nn.ModuleList(
            [MoeExpert(config) for _ in range(self.n_routed_experts)]
        )
        self.shared_experts = FFN(
            config,
            intermediate_size=config.intermediate_size // config.n_shared_experts,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (batch_size, seq_len, hidden_dim)
        shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, shape[-1])  # (bsz*seq_len, hidden_dim)
        weights, top_k_index = self.router(hidden_states)  # (bsz*seq_len, top_k)

        # bitcount 一次计算
        counts = torch.bincount(
            top_k_index.flatten(), minlength=self.n_routed_experts
        )  # (n_routed_experts,)
        if self.training:
            self.router._expert_load_accum += counts.float()

        expert_outputs = torch.zeros_like(hidden_states)
        counts_list = counts.cpu().tolist()
        for i in range(self.n_routed_experts):
            if counts_list[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(top_k_index == i)
            expert_outputs[idx] += expert(hidden_states[idx]) * weights[idx, top, None]
        shared_output = self.shared_experts(hidden_states)
        return (expert_outputs + shared_output).view(shape)


class DecoderLayer(nn.Module):
    def __init__(self, config: MyModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.layer_idx = layer_idx

        self.attention = Attention(config)

        if layer_idx >= config.first_k_dense_replace and config.use_moe:
            self.ffn = MoE(config)
        else:
            self.ffn = FFN(config)

        self.input_layernorm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_dim, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, present_key_values = self.attention(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            use_cache,
        )

        hidden_states = residual + attn_output

        hidden_states = hidden_states + self.ffn(
            self.post_attention_layernorm(hidden_states)
        )

        return hidden_states, present_key_values


class MyModel(nn.Module):
    def __init__(self, config: MyModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.final_layernorm = RMSNorm(config.hidden_dim, config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs(
            config.hidden_dim // config.num_attention_heads,
            config.max_position_embeddings,
            config.rope_theta,
            partial_rotary_factor=config.partial_rotary_factor,
        )
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]]:
        # input_ids: (batch_size, seq_len)
        # past_key_values: 若干层的tuple组成的list，每层是tuple[key, value]
        bsz, seq_len = input_ids.shape

        # 兼容hf transformers
        if hasattr(past_key_values, "layers"):
            past_key_values = None
        past_key_values = past_key_values or [None] * self.num_hidden_layers

        # 取某一层的past_key_values来计算当前输入的起始位置
        start_pos = (
            past_key_values[0][0].shape[2] if past_key_values[0] is not None else 0
        )

        # # 根据起始位置生成位置编码索引，并从预计算的频率中取出对应的cos和sin值
        # position_ids = torch.arange(
        #     start_pos, start_pos + seq_len, device=input_ids.device
        # )
        # position_ids = position_ids.unsqueeze(0)
        # position_embeddings = (
        #     self.freqs_cos[position_ids],
        #     self.freqs_sin[position_ids],
        # )  # (1, seq_len, head_dim) batch维度自动广播
        # 直接在预计算矩阵的 sequence 维度 (dim=1) 上进行切片截取
        position_embeddings = (
            self.freqs_cos[:, start_pos : start_pos + seq_len],
            self.freqs_sin[:, start_pos : start_pos + seq_len],
        )

        hidden_states = self.embed_tokens(input_ids)  # (bsz, seq_len, hidden_dim)
        all_present_key_values = []

        for layer_idx, (layer, pkv) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present_key_values = layer(
                hidden_states,
                position_embeddings,
                attention_mask,
                pkv,
                use_cache,
            )
            all_present_key_values.append(present_key_values)

        hidden_states = self.final_layernorm(hidden_states)

        return hidden_states, all_present_key_values if use_cache else None


class MyModelForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MyModelConfig

    def __init__(self, config: MyModelConfig):
        super().__init__(config)
        self.model = MyModel(config)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **args,
    ) -> CausalLMOutputWithPast:
        hidden_states, present_key_values = self.model(
            input_ids, attention_mask, past_key_values, use_cache, **args
        )
        # 只计算最后输入的token的logits以节省计算
        # 使用内置slice方法
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(
            hidden_states[:, slice_indices, :]
        )  # (bsz, logits_to_keep, vocab_size)

        loss = None
        if labels is not None:
            # 继承PreTrainedModel默认self.loss_function为CrossEntropyLoss
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.vocab_size, **args
            )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=present_key_values,
            hidden_states=hidden_states,
        )
