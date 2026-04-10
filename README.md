## 模型结构
### 参数

### RMSNorm
$$\text{RMSNorm}(\boldsymbol{x}) = \gamma \cdot \frac{\boldsymbol{x}}{\sqrt{\frac{1}{n}\sum{q_i^2}+\epsilon}}$$


### RoPE 
对于第t个token，我们在q和k上分别应用RoPE，具体来说是对于head_dim维度的元素，两两为一组（视作二维向量），对每一组应用旋转矩阵变换，具体如下：
$$
\boldsymbol{q} = [q_0, q_1, ..., q_{d-1}]\\
\boldsymbol{R}_t\boldsymbol{q}=\scriptsize{\underbrace{\begin{pmatrix} 
\cos t\theta_0 & -\sin t\theta_0 & 0 & 0 & \cdots & 0 & 0 \\ 
\sin t\theta_0 & \cos t\theta_0 & 0 & 0 & \cdots & 0 & 0 \\ 
0 & 0 & \cos t\theta_1 & -\sin t\theta_1 & \cdots & 0 & 0 \\ 
0 & 0 & \sin t\theta_1 & \cos t\theta_1 & \cdots & 0 & 0 \\ 
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\ 
0 & 0 & 0 & 0 & \cdots & \cos t\theta_{d/2-1} & -\sin t\theta_{d/2-1} \\ 
0 & 0 & 0 & 0 & \cdots & \sin t\theta_{d/2-1} & \cos t\theta_{d/2-1} \\ 
\end{pmatrix}}_{\boldsymbol{\mathcal{R}}_t} \begin{pmatrix}q_0 \\ q_1 \\ q_2 \\ q_3 \\ \vdots \\ q_{d-2} \\ q_{d-1}\end{pmatrix}} \\
=\begin{pmatrix}q_0 \\ q_1 \\ q_2 \\ q_3 \\ \vdots \\ q_{d-2} \\ q_{d-1} 
\end{pmatrix}\otimes\begin{pmatrix}\cos t\theta_0 \\ \cos t\theta_0 \\ \cos t\theta_1 \\ \cos t\theta_1 \\ \vdots \\ \cos t\theta_{d/2-1} \\ \cos t\theta_{d/2-1} 
\end{pmatrix} + \begin{pmatrix}-q_1 \\ q_0 \\ -q_3 \\ q_2 \\ \vdots \\ -q_{d-1} \\ q_{d-2} 
\end{pmatrix}\otimes\begin{pmatrix}\sin t\theta_0 \\ \sin t\theta_0 \\ \sin t\theta_1 \\ \sin t\theta_1 \\ \vdots \\ \sin t\theta_{d/2-1} \\ \sin t\theta_{d/2-1} 
\end{pmatrix}\\
$$
上式为苏剑林原版的RoPE，其中$\theta_i = \frac{1}{\theta ^{2i/d}}$，$m$为token位置，$\theta$通常为100000，$d$为head_dim。
但是这种设计在实现上不够高效，因为两两相邻位置的切分方式开销较大。Llama等模型采用了一种更高效的方式，即采取从中间切分的方式，让 $q_0$ 和 $q_{d/2}$ ，$q_1$和$q_{d/2+1}$ ，...，$q_{d/2-1}$ 和 $q_{d-1}$ 这样两两配对，本质上与原版相邻位置的切分方式等价，但在 GPU 上更加连续且易于向量化。

对应于代码中的 `q_embed = (q * cos) + (rotate_half(q) * sin)`，公式如下：

$$
\boldsymbol{R}_t\boldsymbol{q} = 
\begin{pmatrix} q_0 \\ q_1 \\ \vdots \\ q_{n/2-1} \\ q_{n/2} \\ q_{n/2+1} \\ \vdots \\ q_{n-1} \end{pmatrix} 
\otimes 
\begin{pmatrix} \cos t\theta_0 \\ \cos t\theta_1 \\ \vdots \\ \cos t\theta_{n-1} \\ \cos t\theta_0 \\ \cos t\theta_1 \\ \vdots \\ \cos t\theta_{n-1} \end{pmatrix} 
+ 
\begin{pmatrix} -q_{n/2} \\ -q_{n/2+1} \\ \vdots \\ -q_{n-1} \\ q_0 \\ q_1 \\ \vdots \\ q_{n/2-1} \end{pmatrix} 
\otimes 
\begin{pmatrix} \sin t\theta_0 \\ \sin t\theta_1 \\ \vdots \\ \sin t\theta_{n-1} \\ \sin t\theta_0 \\ \sin t\theta_1 \\ \vdots \\ \sin t\theta_{n-1} \end{pmatrix}
$$

### YaRN

### GQA
对于MHA，每个token的query和key维度相同，切分head后也一致。GQA通过减少kv的维度，让若干head的q共享相同的kv，从而可以在不显著降低性能的同时减少计算量。由于在从x变换到kv时维度少了，因此实现了`repeat_kv`函数手动重复kv，在实际计算的时候仍然让每个head的q都能够与kv进行匹配。
kv cache 在推理的decode阶段使用，通过传递`past_key_values`参数，可以避免重复计算。每次新生成一个token被加入序列后，Attention模块中需要更新kv cache，即将新产生的kv直接concat到`past_key_values`中。在hf transformers中，`past_key_values`被定义为`Cache`类，封装了update方法，并且所有层共用一个`Cache`对象，通过`layer_idx`进行区分。