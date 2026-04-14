## 模型结构
### 参数

### RMSNorm
$$\text{RMSNorm}(\boldsymbol{x}) = \gamma \cdot \frac{\boldsymbol{x}}{\sqrt{\frac{1}{n}\sum{q_i^2}+\epsilon}}$$


### RoPE 
对于第t个token，我们在q和k上分别应用RoPE，具体来说是对于head_dim维度的元素，两两为一组（视作二维向量），对每一组应用旋转矩阵变换，具体如下：

$$
\begin{aligned}
&\boldsymbol{q} = [q_0, q_1, ..., q_{d-1}] \\\\
\boldsymbol{R}_t\boldsymbol{q} &= \scriptsize{\underbrace{\begin{pmatrix} 
\cos t\theta_0 & -\sin t\theta_0 & 0 & 0 & \cdots & 0 & 0 \\\\ 
\sin t\theta_0 & \cos t\theta_0 & 0 & 0 & \cdots & 0 & 0 \\\\ 
0 & 0 & \cos t\theta_1 & -\sin t\theta_1 & \cdots & 0 & 0 \\\\ 
0 & 0 & \sin t\theta_1 & \cos t\theta_1 & \cdots & 0 & 0 \\\\ 
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\\\ 
0 & 0 & 0 & 0 & \cdots & \cos t\theta_{d/2-1} & -\sin t\theta_{d/2-1} \\\\ 
0 & 0 & 0 & 0 & \cdots & \sin t\theta_{d/2-1} & \cos t\theta_{d/2-1} \\\\ 
\end{pmatrix}}_{\boldsymbol{\mathcal{R}}_t} \begin{pmatrix}q_0 \\\\ q_1 \\\\ q_2 \\\\ q_3 \\\\ \vdots \\\\ q_{d-2} \\\\ q_{d-1}\end{pmatrix}} \\\\
&=\begin{pmatrix}q_0 \\\\ q_1 \\\\ q_2 \\\\ q_3 \\\\ \vdots \\\\ q_{d-2} \\\\ q_{d-1} 
\end{pmatrix}\otimes\begin{pmatrix}\cos t\theta_0 \\\\ \cos t\theta_0 \\\\ \cos t\theta_1 \\\\ \cos t\theta_1 \\\\ \vdots \\\\ \cos t\theta_{d/2-1} \\\\ \cos t\theta_{d/2-1} 
\end{pmatrix} + \begin{pmatrix}-q_1 \\\\ q_0 \\\\ -q_3 \\\\ q_2 \\\\ \vdots \\\\ -q_{d-1} \\\\ q_{d-2} 
\end{pmatrix}\otimes\begin{pmatrix}\sin t\theta_0 \\\\ \sin t\theta_0 \\\\ \sin t\theta_1 \\\\ \sin t\theta_1 \\\\ \vdots \\\\ \sin t\theta_{d/2-1} \\\\ \sin t\theta_{d/2-1} 
\end{pmatrix}
\end{aligned}
$$

上式为苏剑林原版的RoPE，其中 $\theta_i = \frac{1}{\theta ^{2i/d}}$ ， $m$ 为token位置， $\theta$ 通常为100000， $d$ 为head_dim。
但是这种设计在实现上不够高效，因为两两相邻位置的切分方式开销较大。Llama等模型采用了一种更高效的方式，即采取从中间切分的方式，让 $q_0$ 和 $q_{d/2}$ ， $q_1$ 和 $q_{d/2+1}$ ，...， $q_{d/2-1}$ 和 $q_{d-1}$ 这样两两配对，本质上与原版相邻位置的切分方式等价，但在 GPU 上更加连续且易于向量化。

对应于代码中的 `q_embed = (q * cos) + (rotate_half(q) * sin)`，公式如下：

$$
\boldsymbol{R}_t\boldsymbol{q} = 
\begin{pmatrix} q_0 \\\\ q_1 \\\\ \vdots \\\\ q_{n/2-1} \\\\ q_{n/2} \\\\ q_{n/2+1} \\\\ \vdots \\\\ q_{n-1} \end{pmatrix} 
\otimes 
\begin{pmatrix} \cos t\theta_0 \\\\ \cos t\theta_1 \\\\ \vdots \\\\ \cos t\theta_{n-1} \\\\ \cos t\theta_0 \\\\ \cos t\theta_1 \\\\ \vdots \\\\ \cos t\theta_{n-1} \end{pmatrix} 
+ 
\begin{pmatrix} -q_{n/2} \\\\ -q_{n/2+1} \\\\ \vdots \\\\ -q_{n-1} \\\\ q_0 \\\\ q_1 \\\\ \vdots \\\\ q_{n/2-1} \end{pmatrix} 
\otimes 
\begin{pmatrix} \sin t\theta_0 \\\\ \sin t\theta_1 \\\\ \vdots \\\\ \sin t\theta_{n-1} \\\\ \sin t\theta_0 \\\\ \sin t\theta_1 \\\\ \vdots \\\\ \sin t\theta_{n-1} \end{pmatrix}
$$

### YaRN

### GQA
对于MHA，每个token的query和key维度相同，切分head后也一致。GQA通过减少kv的维度，让若干head的q共享相同的kv，从而可以在不显著降低性能的同时减少计算量。由于在从x变换到kv时维度少了，因此实现了`repeat_kv`函数手动重复kv，在实际计算的时候仍然让每个head的q都能够与kv进行匹配。
kv cache 在推理的decode阶段使用，通过传递`past_key_values`参数，可以避免重复计算。每次新生成一个token被加入序列后，Attention模块中需要更新kv cache，即将新产生的kv直接concat到`past_key_values`中。在hf transformers中，`past_key_values`被定义为`Cache`类，封装了update方法，并且所有层共用一个`Cache`对象，通过`layer_idx`进行区分。

### FFN
我们采用SwiGLU的实现，对中间线性变换后接一个门控机制，最开始的GLU使用sigmoid激活函数作为非线性激活函数产生门控信号，然后与线性变换结果逐元素相乘控制信息流动，在这里我们使用Swish激活函数代替sigmoid，并且参考主流实现没有添加偏置项。

$$
\text{SwiGLU}(\boldsymbol{x}) = ((\boldsymbol{x}\boldsymbol{W}_1) \odot \text{Swish}(\boldsymbol{x}\boldsymbol{W}_2) )\boldsymbol{W}_3
$$

使用GLU系列激活函数时，为了与经典FFN模块的参数量级一致，原本FFN会把输入维度升维至4倍，而GLU系列激活函数则只升维至 $4\times2\div3=8/3$ 倍，因为GLU相比经典FFN多了一个线性变换权重。

### MoE
相比经典FFN，MoE可以看作显式稀疏的FFN，每次只有一部分专家被选中和激活，激活的专家本身还是一个小的FFN。

本项目的实现中，我们采用了DeepSeek-V3提出的，基于无辅助损失函数的负载均衡策略。具体来说，在路由模块中，输入经过线性变换后，使用sigmoid激活函数而非softmax，随后会被加上一个偏置向量，其长度等于总的专家数量，用来调节每个专家的激活概率。在训练过程中，偏置会随batch被更新，每个batch中会统计各专家激活的token数，分别对高于/低于平均激活token数的专家的偏置进行增加/减少，每次更新的值为一个常量（预设的超参数，ds-V3论文中做了消融实验发现，这种常量更新策略比依赖具体距平差值更新或其它方案综合更好），从而实现负载均衡。

考虑第 $t$ 个token，其具体计算过程如下：

$$
\begin{aligned}
\boldsymbol{out}_t &= \boldsymbol{h}_t + \sum_{i=1}^{\text{numExperts}}g_{i,t}\text{FFN}_i(\boldsymbol{h}_t),\\
g_{i,t} &=\begin{cases}s_{i,t},&s_{i,t}+b_{i}\in\mathrm{Topk}\left(\left\lbrace s_{j,t}+b_{j}\mid 1\leq j\leq N \right\rbrace,K\right),\\ 
0,&\mathrm{otherwise},\end{cases}\\
s_{i,t} &= \text{Sigmoid}(\boldsymbol{h}_t^\top\boldsymbol{W}),\\
\end{aligned}
$$

需要注意的是，用来选择topk专家的是 $s_{i,t}+b_{i}$ ，而加权到FFN输出的是 $s_{i,t}$ ，这是DeepSeek-V3等模型与其它有辅助损失函数的MoE模型的主要区别。

除了无损负载均衡策略，本项目还实现了共享专家，即在路由专家以外还有一个FFN模块，所有token都会经过这个FFN模块。

DeepSeek-V3中还采用了分组路由，把所有路由专家分成若干组，对于每个 token，先确保选择的专家仅在 topk_group 组内，再选择这些组内的 num_experts_per_tok 个专家。但是在本项目实现中，我们并没有采用这种机制。

除此之外，在DeepSeek-V3和GLM-4.5等类似架构的MoE模型中，还额外设置了一个`route_scale`参数，用在加权分数weight已经归一化之后。该参数在GLM-4.5中被设为1，而在DeepSeek-V3中被设为2.5，其具体作用在DeepSeek-V3论文中并没有详细解释。这里引用苏剑林老师在博客中对于该参数的解释（[《MoE环游记：5、均匀分布的反思 》](https://kexue.fm/archives/10945)）：
> 我们将式(2)一般地写成
> 
> $$\boldsymbol{y} = \sum_{i=1}^s \boldsymbol{e}_i + \lambda\sum_{i\in \mathop{\text{argtop}}_{k-s} \boldsymbol{\rho}_{[s:]}} \rho_{i+s} \boldsymbol{e}_{i+s}$$
> 
> 由于Routed Expert带有权重 $\rho_{i+s}$ 而Shared Expert没有，以及Routed Expert的数目通常远大于Shared Expert数目（即 $n−s\gg s$ ）等原因，它们的比例可能会失衡，因此为了让两者不至于被相互埋没，设置合理的 $\lambda$ 尤为重要。对此，我们在[《Muon is Scalable for LLM Training》](https://papers.cool/arxiv/2502.16982)提出，适当的 $\lambda$ 应使得两者在初始化阶段模长接近一致。
> 
> 具体来说，我们假设每个Expert在初始化阶段具有相同的模长（不失一般性，可以直接设为1），并且满足两两正交，然后假设Router的logits服从标准正态分布（即零均值、单位方差，当然如果觉得有必要，也可以考虑其他方差）。这样一来， $s$ 个Shared Expert的总模长就是 $\sqrt{s}$ ，而Routed Expert的总模长是
> 
> $$\lambda\sqrt{\sum_{i\in \mathop{\text{argtop}}_{k-s} \boldsymbol{\rho}_{[s:]}} \rho_{i+s}^2}$$
> 
> 通过让它等于 $\sqrt{s}$ ，我们就可以估计出 $\lambda$ 。由于激活函数、是否重归一化等选择，不同MoE的Router差别可能比较大，所以我们也不设法求解析解，而是直接数值模拟：
> ```python
> import numpy as np
> 
> def sigmoid(x):
>     return 1 / (1 + np.exp(-x))
>
> def softmax(x):
>     return (p := np.exp(x)) / p.sum()
>
> def scaling_factor(n, k, s, act='softmax', renorm=False):
>     factors = []
>     for _ in range(10000):
>        logits = np.random.randn(n - s)
>         p = np.sort(eval(act)(logits))[::-1][:k - s]
>         if renorm:
>             p /= p.sum()
>         factors.append(s**0.5 / (p**2).sum()**0.5)
>     return np.mean(factors)
>
> scaling_factor(162, 8, 2, 'softmax', False)
> scaling_factor(257, 9, 1, 'sigmoid', True)
> ```
> 非常巧的是，这个脚本的模拟结果跟DeepSeek-V2、DeepSeek-V3的设置都很吻合。其中，DeepSeek-V2有 $n=162,k=8,s=2$ ，Softmax激活并且没有重归一化，上述脚本的模拟结果约等于16，而DeepSeek-V2的 $\lambda$ 正好是16；DeepSeek-V3则有 $n=257,k=9,s=1$ ，Sigmoid激活且重归一化，脚本的结果大约是2.83，而DeepSeek-V3的 $\lambda$ 则是2.5。


在具体实现有关模块时，本人参考了DeepSeek-V3和GLM-4.5等模型的实现代码，但并没有发现对路由模块中偏置项`e_score_correction_bias`的具体更新操作，也没有暴露合适的接口与变量来让外部脚本能够“每个batch中统计各专家激活的token数”，因此本项目中的`MoeRouter`模块额外实现了`_expert_load_accum`变量与`update_bias`方法，主要在训练时使用。
