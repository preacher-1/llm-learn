from typing import Optional, Tuple
from transformers import PretrainedConfig


class MyModelConfig(PretrainedConfig):
    model_type = "mymodel"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_dim: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
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
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

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


def precompute_freqs(hidden_dim: int, seq_len: int, theta: int = 1000000):
    # 1. freq = 1/(theta^(i/dim)) for i = 0, 2, 4, ..., dim-2
    freqs = 1.0 / (theta ** (torch.arange(0, hidden_dim, 2).float() / hidden_dim))
    # TODO: 实现YaRN
    # 2. 将位置 t 与处理好的频率 freqs 相乘，得到每个位置的旋转角度 θ
    t = torch.arange(seq_len, device=freqs.device).float()
    freqs = torch.outer(t, freqs).float()  # (seq_len, dim//2)

    # 3. 计算余弦和正弦值
    freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)  # (seq_len, dim)
    freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)  # (seq_len, dim)

    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    )
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    )
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # x: (batch_size, seq_len, num_heads, head_dim)
    bsz, seq_len, num_heads, head_dim = x.shape
    # 使用expand和reshape来重复每个头n_rep次，此处n_rep是在外部计算好的
    if n_rep == 1:
        return x
    x = x.unsqueeze(2)  # (bsz, seq_len, 1, num_heads, head_dim)
    x = x.expand(bsz, seq_len, n_rep, num_heads, head_dim)
    return x.reshape(bsz, seq_len, n_rep * num_heads, head_dim)


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
        self.q_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)

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
        # position_embeddings: (cos, sin) each of shape (seq_len, hidden_dim)
        # attention_mask: (batch_size, 1, 1, seq_len) or None
        # past_key_values: ((batch_size, num_attention_heads, past_seq_len, head_dim), (batch_size, num_attention_heads, past_seq_len, head_dim)) or None
        bsz, seq_len, _ = hidden_states.shape
        hidden_shape = (bsz, seq_len, -1, self.head_dim)

        query = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
        key = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        value = self.v_proj(hidden_states).view(hidden_shape)

        cos, sin = position_embeddings  # (seq_len, hidden_dim)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        query = query.transpose(1, 2)  # (bsz, num_heads, seq_len, head_dim)
        key = repeat_kv(key, self.n_rep).transpose(
            1, 2
        )  # (bsz, num_heads, seq_len, head_dim)
        value = repeat_kv(value, self.n_rep).transpose(
            1, 2
        )  # (bsz, num_heads, seq_len, head_dim)

        if past_key_values is not None:
            # 将当前的key和value与过去的key和value拼接起来
            past_key, past_value = past_key_values
            key = torch.cat(
                [past_key, key], dim=2
            )  # (bsz, num_key_value_heads, past_seq_len + seq_len, head_dim)
            value = torch.cat(
                [past_value, value], dim=2
            )  # (bsz, num_key_value_heads, past_seq_len + seq_len, head_dim)
            past_key_values = (key, value)

        if (
            self.flash
            and (seq_len > 1)
            and (past_key_values is None)
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=True
            )  # (bsz, num_attention_heads, seq_len, head_dim)
        else:
            attn_scores = query @ key.transpose(-1, -2) / (self.head_dim**0.5)
            attn_scores[:, :, :, -seq_len:] += torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=query.device),
                diagonal=1,
            )  # causal mask
            if attention_mask is not None:
                attn_scores = attn_scores.masked_fill(
                    attention_mask == 0, float("-inf")
                )
            attn_scores = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(
                query.dtype
            )
            attn_output = attn_scores @ value  # (bsz, num_heads, seq_len, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        attn_output = self.o_proj(attn_output)  # (bsz, seq_len, hidden_dim)

        return attn_output, past_key_values
