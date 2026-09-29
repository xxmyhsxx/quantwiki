---
title: Attention 量化误差传播：从 KV 重构到输出与生成
type: concept
tags:
  - attention
  - kv-cache
  - quantization-error
  - sensitivity
sources:
  - raw/papers/2026-09-21/nova-kv/paper.pdf
  - raw/papers/2026-09-22/qjl/paper.pdf
  - raw/papers/2026-09-21/turboquant/paper.pdf
  - raw/papers/2026-09-21/kivi/paper.pdf
updated: 2026-09-29
---

# Attention 量化误差传播：从 KV 重构到输出与生成

KV 重构误差经过 query 点积、softmax 和 Value 加权，才变成 attention 输出误差；后续还要经过投影、残差、其他层和自回归反馈。各阶段的误差指标不能互相替代。本页从一个固定 attention 调用推导这些关系，再说明哪些结论能够延伸到生成。

前置是矩阵乘法、概率期望和二范数；[KV 对象与粒度](kv-cache-quantization-objects-and-granularity.md)解释缓存本身的结构，[加权距离](../fundamentals/mathematics/invertible-transforms-and-kronecker-products.md#7-二阶统计如何定义误差距离)解释矩阵度量。下列恒等式、反例和条件化界是依据 NOVA-KV 附录 E/F、QJL 的估计问题与 TurboQuant 的 MSE/IP 区分所作的教学展开，不是新的模型实验。

## 1. 固定比较对象，才能归因误差

设有 $N$ 个 query、$T$ 个历史位置，头维度为 $d$、Value 维度为 $d_v$：

$$Q\in\mathbb R^{N\times d},\quad K\in\mathbb R^{T\times d},\quad V\in\mathbb R^{T\times d_v},$$
$$Z=QK^\top/\sqrt d,\quad P=\operatorname{softmax}_{row}(Z),\quad O=PV.$$

为研究一次调用的 KV 误差，固定 $Q$、可见位置集合和位置编码，令 $\widehat K=K+E_K$、$\widehat V=V+E_V$。每行 softmax 只在该 query 的可见位置上定义；需要至少一个可见位置。不能对两个包含不同 mask 的 $-\infty$ 张量直接求差后套有限实数范数界。

若整模型运行中 query 也变成 $Q+E_Q$，则额外有

$$\widehat Z-Z=(E_QK^\top+QE_K^\top+E_QE_K^\top)/\sqrt d.$$

因此固定 query 的 KV 诊断，和自由生成中比较两条不同隐藏状态轨迹，回答的是不同问题。位置编码变化、token 删除与缓存精度变化也应分别控制。

## 2. K 误差沿 query 方向进入 logits

只改 K 时，$\Delta Z=QE_K^\top/\sqrt d$。若所有 query 都读取同一组 keys，则

$$\|\Delta Z\|_F^2=\frac1d\operatorname{tr}(E_KM_qE_K^\top),\qquad M_q=Q^\top Q.$$

这说明相同的 K-MSE 可以产生不同 logit 误差。教学例：$q=(10,0)$，两个 key 误差 $e_a=(0.1,0)$、$e_b=(0,0.1)$ 的范数相同，点积误差却分别为 1 和 0。范围或通道幅值只能作为代理，不能代替读取方向。

$M_q$ 是未中心化二阶矩。GQA 的一个 KV head 被多个 query heads 读取时，对总平方误差目标应累加各 head 的 $Q_h^\top Q_h$；若希望不同 heads 等权、不同长度样本等权，必须先定义并归一化相应权重。

**因果可见性会改变精确度量。** 若第 $j$ 个 key 只被集合 $I_j$ 内的 query 读取，可见 logits 的精确平方误差为

$$\frac1d\sum_j e_jM_{q,j}e_j^\top,\qquad M_{q,j}=\sum_{i\in I_j}q_i^\top q_i.$$

对所有 key 使用一个全局 $M_q$ 是全配对代理，或是当前 decode query 对全部有效历史可见时的特例。全配对误差仍可上界可见部分，但不应把它与一般 causal prefill 的精确目标混同。

## 3. softmax 如何改变误差方向

对单行列向量 $p=\operatorname{softmax}(z)$，Jacobian 为

$$J=\operatorname{diag}(p)-pp^\top,\qquad J\mathbf1=0.$$

局部有 $\Delta p=J\Delta z+O(\|\Delta z\|_2^2)$。共同偏移方向被完全消去，而不同 token 之间的相对分数改变归一化后的分配。例如给所有 logits 加 100，概率不变；一个 token 增加而另一个减小，则可显著改变概率。

$J$ 的第 $i$ 行绝对值和是 $2p_i(1-p_i)\le1/2$；$J$ 对称，故谱范数不超过 $1/2$。沿 $z$ 与 $\widehat z$ 的线段积分得到全局界

$$\|\widehat p-p\|_2\le\tfrac12\|\widehat z-z\|_2.$$

这是绝对误差界；不能因此宣称低概率项的相对误差也小，或任务答案不会改变。若每个 logit 误差满足 $|\Delta z_j|\le\eta$，还可由分子和分母的指数界得到

$$e^{-2\eta}\le\widehat p_j/p_j\le e^{2\eta}.$$

要保持原最大 logit 的位置，原第一名与其他项的最小间隔超过 $2\eta$ 是充分条件；间隔小的样本仍容易换序。具体排序问题见 [排序稳定性](quantization-ranking-stability.md)。无偏和小平均 MSE 均不自动给出这个逐项最大误差保证。

## 4. V 误差有抵消，也有叠加

固定 $P$ 时，输出误差为 $PE_V$。单 query 下

$$\delta o=\sum_jp_je_j,\qquad
\|\delta o\|_2^2=\sum_{j,k}p_jp_k\langle e_j,e_k\rangle.$$

交叉项可能为正或负。两个 token 各占 $1/2$ 时，误差 $(a,-a)$ 完全抵消，$(a,a)$ 则完全叠加，逐 token 平方误差总量却相同。因此仅加权各 token 的误差范数会丢掉方向交互。

由平方范数凸性可得确定性上界

$$\left\|\sum_jp_je_j\right\|_2^2\le\sum_jp_j\|e_j\|_2^2.$$

若额外假设误差随机、均值为零且跨 token 不相关，才可把期望写成 $\mathbb E\|\delta o\|^2=\sum_jp_j^2\mathbb E\|e_j\|^2$。真实确定性量化、共享 scale 或相关输入一般不满足这些条件，不能默认把交叉项删除。

K 的小扰动局部造成 $\delta o\approx V^\top J\delta z$（采用单 query 列形式）。即使 attention 变化大，若被重新分配权重的 Values 很相似，输出仍可能变化小；反过来，Values 差异大时较小概率变化也重要。全局可写 $\|(\widehat P-P)V\|_F\le\|\widehat P-P\|_F\|V\|_2$，但该界往往不紧。

## 5. K 与 V 同时量化时的完整展开

令 $\Delta P=\widehat P-P$，则

$$\widehat O-O=\Delta P V+PE_V+\Delta P E_V.$$

前两项分别对应单独扰动 K 和 V，最后一项是交叉误差。这是恒等式；只有扰动足够小，才有理由在一阶分析中忽略交叉项。进一步取范数：

$$\|\widehat O-O\|_F\le\|\Delta P\|_F\|V\|_2+\|PE_V\|_F+\|\Delta P\|_F\|E_V\|_2.$$

这个式子解释为什么“两项单独量化都尚可”不等于组合后仍可，以及为什么 K/V 的代理目标不一定共同最优。输出还可能经过 $W_O$，则误差变为 $(\widehat O-O)W_O$，不同方向继续受到不同放大。

## 6. 无偏、低方差与非线性输出

标量估计的均方误差满足

$$\mathbb E(\widehat a-a)^2=\operatorname{Var}(\widehat a)+(\mathbb E\widehat a-a)^2.$$

无偏消去第二项，未控制第一项；有偏但低方差的估计也可能有更低 MSE。对 softmax 更有 $\mathbb E[\operatorname{softmax}(\widehat z)]\ne\operatorname{softmax}(\mathbb E\widehat z)$，不能把无偏 logits 当成无偏概率。

教学例：两个 logits 原为 $(\log3,0)$，第一项概率为 0.75。仅给第一项加入等概率 $\pm\log3$ 的零均值噪声，得到概率 0.5 或 0.9，期望 0.7。logit 估计无偏，attention 概率已有偏差。

[QJL](../methods/qjl.md)和 [TurboQuant](../methods/turboquant.md)的独立随机投影保证还要求固定被估计向量，或相应独立性。复用同一投影矩阵时，后续 query 可能依赖先前压缩输出；逐点无偏/高概率结论不能直接变成整条自适应轨迹的统一保证。有限个预先固定向量可用 union bound 合并失败概率，但需要相应预算，且不自动解决自适应选择。

## 7. 怎么把这条链用于诊断

| 观察层次 | 固定什么 | 能定位什么 |
| --- | --- | --- |
| K/V 重构 | 原浮点输入、量化组、格式 | 网格、clipping、码本和元数据误差 |
| logits / attention | 相同 query、mask、位置 | 查询方向、softmax 偏移与排序敏感性 |
| attention 输出 | 同一调用的 Q/K/V | Value 方向交互与 K/V 交叉项 |
| teacher-forced 整模型 | 相同输入及后续 token 序列 | 条件计算下的逐层/逐位置传播；隐藏状态仍可能不同 |
| 自由生成与任务指标 | 相同提示、可比采样协议 | 状态和 token 反馈后的实际任务质量 |

这些层次互相补充。先在固定输入上找到第一处可解释差异，再看传播和任务结果；不要把自由生成后完全不同位置的张量 MSE 当成同一局部误差。实现侧还需 [模拟与真实缓存验证](../implementation/quantization-error-diagnosis.md#7-低比特-kv-从模拟到真实执行的对齐)，以排除布局和状态机错误。

配套 [CPU 教学计算](../assets/attention-quantization-error-propagation/check_mechanisms.py) 使用 Python 3 与 NumPy，复算本页误差关系及相关页的二阶统计、码本例子、位打包和简化缓存轨迹；它不调用模型或 GPU，也不能替代真实实现验证。

## 来源身份

- [NOVA-KV，arXiv:2608.04074v1](https://arxiv.org/abs/2608.04074v1)：§3、附录 E/F 的 logit 加权、softmax 范数和输出交叉项。
- [QJL，arXiv:2406.03482v2](https://arxiv.org/abs/2406.03482v2)：非对称内积随机估计；原文精确概率常数的限制见方法页。
- [TurboQuant，arXiv:2504.19874v1](https://arxiv.org/abs/2504.19874v1)：MSE 与无偏内积的目标区分；本页不引用其有问题的有限维球面下界。
- [KIVI，arXiv:2402.02750v2](https://arxiv.org/abs/2402.02750v2)：§3 的 K/V 误差传播与粒度分析。可见性加权、Jacobian 界的展开、抵消与无偏反例为整理者推导。
