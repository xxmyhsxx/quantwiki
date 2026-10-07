---
title: WUSH-KV：联合缓存统计与读取敏感度的可逆变换
type: method
tags:
  - kv-cache
  - equivalent-transform
  - sensitivity
  - calibration
sources:
  - raw/papers/2026-10-03/wush-kv/paper.pdf
  - raw/papers/2026-10-03/wush-kv/source/main.tex
updated: 2026-10-07
---

# WUSH-KV：联合缓存统计与读取敏感度的可逆变换

WUSH-KV 为每层、每个 KV head 离线构造独立的 Key 与 Value 变换。它同时使用缓存向量的二阶统计与读取侧的敏感度，允许正交混合之外的各向异性缩放。Value 变换可折入 V/O 投影；Key 变换在 head normalization 和 RoPE 后执行，新 Key 与对应 Query 需要运行时配对变换，再把缓存按 token/head 量化。

依据 arXiv:2609.38121v1 的 §3–6、附录 A–C 与定向实验说明，结合原页核对式 (1)、(4)–(6) 和表 1–2。本文没有核读作者修改后的 SGLang 实现，也没有运行模型。论文中的 QuEST 理论与模拟实验、采用 OSCAR 量化器的 SGLang 下游结果是不同证据，不能合并为一套已证明的实现。

## 1. 缓存分布与读取方向为什么要共同考虑

采用列 token 约定：一个 KV head 的 $K,V\in\mathbb R^{d\times S}$，读取它的 query heads 为 $Q_g\in\mathbb R^{d\times S}$。GQA 中一个 KV head 被多个 query heads 共享，$g$ 遍历这个共享组；缓存元素数按 KV head 计算，敏感度则要考虑所有读取它的 query heads。

量化某个张量 $X$ 时，在可逆坐标中编码并还原的误差是

$$\varepsilon=T^{-1}\mathcal Q(TX)-X.$$

若它在线性乘积 $W^\top X$ 中被读取，则输出平方误差是

$$
\ell=\|W^\top\varepsilon\|_F^2
=\frac12\sum_j\varepsilon_j^\top H\varepsilon_j,
\qquad M=XX^\top,\quad H=2WW^\top.
$$

M 描述输入在哪些方向有能量，H 描述误差在哪些方向更昂贵；只白化 M 或只控制峰值没有同时处理两件事。Gram 不需要减均值，相关概念见[二阶统计与加权距离](../fundamentals/mathematics/invertible-transforms-and-kronecker-products.md#7-二阶统计如何定义误差距离)。（§3。）

WUSH-KV 主方法选择如下局部代理：

$$
H_K=2\sum_gQ_gQ_g^\top,\qquad
H_V=2\sum_gW_{O,g}W_{O,g}^\top,
$$
$$
T_K=\operatorname{WUSH}(KK^\top,H_K),\qquad
T_V=\operatorname{WUSH}(VV^\top,H_V).
$$

这里 $W_{O,g}\in\mathbb R^{d\times d_{model}}$。H_K 度量未缩放、未掩码的全配对 Q/K 点积平方误差，未加入 softmax 和 Value；H_V 度量各输出投影对 V 的直接读取，未加入 attention 概率以及不同 query heads 合成输出的交叉项。它们是清楚、便宜的双线性代理，不是完整 attention 输出或任务损失的精确 Hessian。（§4.2、式 (4)；附录 D.1 对诊断位置的说明。）

论文的 WUSH-A 改用包含 softmax、因果可见性、Value 差异与输出投影的局部 Hessian；只需前向量即可构造。在零扰动处，重构损失 Hessian 等于 Jacobian 的 $2J^\top J$，不需要语言模型损失反传。具体推导与 GQA 的 head 交叉项见[attention 的局部敏感度](../theory/attention-quantization-error-propagation.md#8-从局部输出导数到-gqa-缓存敏感度)。主实验中它改善幅度有限，作者因此采用简单代理，而非声称简单 H 就等于精确 H。

## 2. WUSH 的闭式构造在平衡什么

对每个 head 宽度 d 的块，令归一化 Hadamard 为 $\mathcal H$，$\gamma\ge0$ 为阻尼比。先做 Cholesky 和对称特征分解：

$$
LL^\top=H+\gamma\frac{\operatorname{tr}H}{d}I,
$$
$$
U\Lambda U^\top
=L^\top\left(M+\gamma\frac{\operatorname{tr}M}{d}I\right)L.
$$

再构造

$$
c=\sqrt{1+\gamma}\,\frac{\sqrt{\operatorname{tr}M}}
{\sqrt{\operatorname{tr}(\Lambda^{1/2})}},\qquad
T=c\mathcal H\Lambda^{-1/4}U^\top L^\top.
$$

阻尼后的矩阵必须正定；零迹的退化矩阵不会仅因增加这个迹比例阻尼就变为正定。该构造还要求存在对应阶的 Hadamard。论文以这些作为定义条件，实验使用 $\gamma=10^{-2}$。M/H 各自乘任意正标量都不改变 T，所以求和与统一比例平均不会改变该构造；不等权重新加权样本则可能改变方向。（§3、式 (1)、定义 1。）

无阻尼且 M/H 正定时，有

$$
TMT^\top=c^2\mathcal H\Lambda^{1/2}\mathcal H^\top,
\qquad
B_T:=T^{-\top}HT^{-1}
=c^{-2}\mathcal H\Lambda^{1/2}\mathcal H^\top.
$$

两侧被调整成比例相同的形状，Hadamard 进一步使各自的对角值相等。它没有把两个矩阵都变成 I；B_T 的对角相等也不意味着非对角项为零。c 保持整体 $\|TX\|_F=\|X\|_F$，不是保持每个 token 的范数。

这一构造的意义可从无裁剪的舍入噪声模型看出：逐组步长与 $\|Tx_j\|_2$ 成正比，舍入误差近似为独立、零均值均匀噪声。期望输出损失因而正比于

$$\operatorname{tr}(TMT^\top)\operatorname{tr}(T^{-\top}HT^{-1}).$$

附录 B.2 用 Cauchy–Schwarz 证明该乘积至少为 $\left(\sum_i\sigma_i(M^{1/2}H^{1/2})\right)^2$，无阻尼 WUSH 达到它。纯正交 T 保持两个 trace，不能一般地降低这个乘积；M 与 H 成比例时才已经达到下界。这解释允许缩放的收益，但最优的是指定噪声模型下的 trace 目标，不是有限网格的确定性任务损失。

## 3. 变换放在哪里，如何保持未量化计算

Key 变换成 $T_KK$，对应 Query 必须是 $T_K^{-\top}Q$：

$$K^\top Q=(T_KK)^\top(T_K^{-\top}Q).$$

只有正交变换时逆转置才等于原矩阵，不能把一般可逆 T 同乘到 K/Q。Value 的配对是

$$W_{O,g}^\top V=(T_V^{-\top}W_{O,g})^\top(T_VV).$$

用论文的权重形状，可离线保存 $\widehat W_V=W_VT_V^\top$、$\widehat W_{O,g}=T_V^{-\top}W_{O,g}$。V 已在线性投影中产生变换坐标，后续每个读取该 KV head 的 O 块都必须配套变化。（§4.2、式 (3)。）

K 侧在 normalization/RoPE 后，因而保留稠密在线变换。不能简单折入投影：任意 T 通常既不与位置旋转交换，也不保持固定 RMSNorm 的分母和逐通道增益。附录 A 在非退化 RoPE 频率、一般固定学习增益的条件下，联合可交换约束只剩成对符号翻转，它不能重新分配幅值。没有 normalization、频率退化或允许重参数化增益时，限制会不同，不能把该结论说成所有模型只允许符号翻转。

另一个选择是缓存 pre-RoPE 的 $T_KR_s^\top k_s$，但读取位置 s 的 query 补偿变成 $T_K^{-\top}R_s^\top q_t$，随历史位置变化。WUSH-KV 选择 post-RoPE，是为了每个新 K/Q 只做固定变换，并保持简单的历史缓存读取；代价是每层每 token 约 $(n_{kv}+n_q)d^2$ 次乘加。Value 两侧可折入，Key 的运行成本仍需后端计时验证。

## 4. 量化器与“近最优”的准确范围

理论与局部误差/PPL 实验使用 QuEST 形式：一个 token 的一个 head 共用尺度，非零 x 的步长和编码为

$$
\Delta=\frac{2\alpha_b}{2^b-1}\frac{\|x\|_2}{\sqrt d},\qquad
i=\operatorname{clip}\left(\left\lfloor x/\Delta\right\rfloor,
-2^{b-1},2^{b-1}-1\right),\quad
\mathcal Q(x)=\Delta(i+1/2).
$$

这是 $2^b$ 个 bin center，偶数级网格不含零；零向量专门返回零。$\alpha_b$ 来自标准高斯固定尺度网格的 MSE 最优端点，不是逐组在线搜索。它与[尺度敏感性页](../theory/calibration-and-range-selection.md#10-极低位宽为什么更依赖尺度选择)中 W2 的三值网格不同，不能混用解析尺度常数。（§4.3、式 (5)。）

定理 2、附录 B 给出的保证有明确条件：M/H 正定，理想 WUSH 使用 $\gamma=0$；舍入建模为独立均匀噪声，裁剪保留真实残差；WUSH 的归一化坐标尾部由高斯尾界控制，裁剪误差没有过度对齐 B_T 的敏感方向，相关界常数不随维度和位宽变坏。竞争者限于 B_T 对角值相等的 sensitivity-balanced 可逆变换。

在这些条件下，$b\to\infty$ 时，WUSH 的模型期望损失至多为竞争者的 $1+O(\alpha_b^{-2})$ 倍。附录对 $\alpha_b>1$ 给出有限位宽系数

$$1+\frac{1+3C_{tail}C_{align}}{\alpha_b^2-1}.$$

这不是任意 2 bit 数据、任意变换或真实 attention 任务的近最优保证；实验阻尼也不同于理想无阻尼条件。下游采用的非对称百分位量化器更不直接受这个 QuEST 定理覆盖。

## 5. 流式缓存、窗口与真实成本

算法 1 离线逐序列积累 K/V Gram 和 query 统计，最后为每层每 head 构造 T；无需保存完整校准激活。推理中的缓存已处于变换坐标，算法 2 先以全精度追加新 chunk，计算当前 attention，再批量量化可移出的历史条目：

- 前 $S_{sink}$ 个位置长期保持全精度。
- 其余位置中最新 $S_{keep}$ 个形成滚动全精度窗口。
- 窗口外的中段按 $S_{flush}$ 的整数倍量化，可能暂留少于一个 flush 的额外全精度条目。

两个窗口不再重叠后，长期全精度条目为 $S_{sink}+S_{keep}$。短序列占比更高；不能只拿下游任务最大上下文下的“不到 1%”描述所有长度。论文也说明 SGLang 集成的存储更新发生在 attention 前，但当前 token/prefill chunk 仍用 BF16；这是实现与算法表述需要核对的次序差别。（§4.4、§5.3。）

L 层缓存有 $2Ln_{kv}dS$ 个 K/V 元素，还要保存 $Ln_{kv}$ 个固定 d×d Key 变换。若某时刻有 r 个全精度位置，忽略元数据的平均编码位宽为 $b+(16-b)r/S$；再加逐组尺度/零点、页对齐和暂存。这里的 2 bit 是量化中段整数编码的标称位宽，不能据此说整机显存或缓存真实字节数缩小八倍。计账方式见[KV 格式与生命周期](../theory/kv-cache-quantization-objects-and-granularity.md#11-从位宽到可执行存储格式)。

## 6. 论文结果与比较条件

三层实验分别回答不同问题：

| 实验层次 | 条件 | 支持的结果 |
| --- | --- | --- |
| attention 误差，§5.1 | Qwen3-8B 全部 36 层；128 条 FineWeb-Edu 校准、32 条独立评测，各 1,024 token；同一 QuEST，无全精度窗口，干净 BF16 轨迹 | 联合 K/V 2 bit 的层间几何平均输出相对平方误差：H 0.325、OSCAR 0.309、WUSH 0.208、WUSH-A 0.195；局部代理改善 |
| PPL，§5.2、表 1 | 128×32,768 校准；Qwen3-8B 使用 YaRN factor 4；WikiText-2 按 2,048 分段、16-token chunk；窗口 16/128/16 | QuEST 下 WUSH 的 4/3/2 bit PPL 为 9.73/9.82/10.51，BF16 9.72；2 bit OSCAR 13.74、H 258.29 |
| 下游，§5.3、表 2 | SGLang；三种 Qwen3 模型，四项生成任务，三 seeds；OSCAR 百分位非对称量化；窗口 64/256/8 | 任务准确率、输出长度和预算耗尽率的实际作者评测；不属于 QuEST 定理的直接验证 |

H 基线只使用归一化 Hadamard 变换，同一实验内匹配当前量化器；未复现完整 TurboQuant 码本或 QJL residual sketch。H 在 2 bit 上的失败不能直接归因给完整 TurboQuant。作者将 PPL 实验的失败联系到第一层单坐标尖峰被 Hadamard 分散后，接近 2 bit 网格中点，造成稠密误差。

下游量化器对 transformed token/head 组先按绝对值的 $\kappa$ 分位数裁剪，再由裁剪后的 min/max 设置 affine scale 和 offset，映射到 $0\ldots2^b-1$。K 的 $\kappa=0.96$；V 在 4B/8B 为 0.92、32B 为 0.96，沿用 OSCAR 参数，没有针对 WUSH 重调。

表 2 的代表值如下，均为任务分数均值 ± 三 seeds 标准差，不表示显著性检验。

| 模型、任务 | BF16 | OSCAR INT2 | WUSH-KV INT2 |
| --- | --- | --- | --- |
| Qwen3-8B，AIME 2025 | 67.8 ± 1.9 | 34.4 ± 8.4 | 46.7 ± 3.3 |
| Qwen3-8B，LiveCodeBench v6 | 47.8 ± 2.2 | 20.4 ± 0.3 | 35.2 ± 0.7 |
| Qwen3-32B，AIME 2025 | 74.4 ± 5.1 | 63.3 ± 3.3 | 60.0 ± 5.8 |
| Qwen3-32B，LiveCodeBench v6 | 61.9 ± 0.3 | 27.8 ± 1.4 | 43.1 ± 2.3 |

8B 四项任务均是 WUSH 均值更高；32B 的 AIME、MATH-500、GPQA 均是 OSCAR 均值更高。因此结论是条件下有竞争力、部分退化任务改善明显，不能说所有任务领先。8B LiveCodeBench 的平均生成量从 OSCAR 73.8k 降到 WUSH 53.9k，预算耗尽率从 71% 降到 26%；这说明评测还受停止与生成长度影响，不能只报准确率或标称压缩率。

附录 D.2 给出重要控制：固定 8B 的 WUSH/FineWeb-Edu 校准，换成 QuEST 后 AIME 由 46.7 ± 3.3 降到 28.9 ± 6.9；尽管作者的局部 L2 测量更偏好 QuEST，下游生成质量更偏好 OSCAR 量化器。重构误差改善与生成行为改善之间仍有缺口。

校准预算也不同：WUSH 为约 4.2M FineWeb-Edu token，公开 OSCAR 8B 变换约用 8k GPQA token；把 OSCAR 换为等大小 FineWeb-Edu 并未改善其结果。主表使用公开 GPQA 变换与共同评测流程，作者说明未精确复现并行预印本全部分数。这是可用的协议内比较，仍不是完全等数据或等校准成本对照。

§5.2 的一次 Qwen3-8B 校准在单张 L40S 上约 12 分钟、峰值 39 GiB，跨位宽复用。§6 把系统优化与吞吐评测列为未来工作，本文也未核读真实布局、融合 kernel 或测量在线稠密 K/Q 变换成本。长上下文附录说明 WUSH 在最长长度有优势，同时 MRCR 相对 BF16 仍有明显损失；这不等于无损长上下文。

## 来源身份

| 来源 | 固定版本 | 本页用途 |
| --- | --- | --- |
| [WUSH-KV: KV Cache Quantization with Data-Adaptive Transforms](https://arxiv.org/abs/2609.38121v1) | arXiv:2609.38121v1 | §3–4 与附录 A–C：构造、位置、量化与理论；§5–6、表 1–2、附录 D.1–D.2 及长上下文说明：协议与证据边界 |
