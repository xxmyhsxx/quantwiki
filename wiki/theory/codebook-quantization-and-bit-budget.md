---
title: 码本量化：标量、向量、加性表示与位宽预算
type: concept
tags:
  - vector-quantization
  - weight-quantization
  - data-format
  - memory
sources:
  - raw/papers/2026-09-21/quip-sharp/source.eprint
  - raw/papers/2026-09-21/aqlm/source.eprint
  - raw/papers/2026-09-21/squeezellm/source.eprint
  - raw/papers/2026-09-21/spqr/source.eprint
  - raw/repositories/2026-09-21/squeezellm/source/squeezellm/quant.py
updated: 2026-09-22
---

# 码本量化：标量、向量、加性表示与位宽预算

码本定义“哪些值可以被表示”，索引定义“这一组权重选哪个值”。低比特并不要求可表示值等间距，也不要求每个标量独立编码。理解 QuIP#、AQLM 和 SqueezeLLM，需要同时说明重构空间、选择目标、码本存储和推理解码。

## 1. 从均匀标量走到向量码本

[均匀量化](../fundamentals/quantization/uniform-quantization-and-groups.md) 的码字可以通过 scale 与 zero-point 生成，不必逐项存表。非均匀标量码本保存 $K$ 个数，$\widehat w=C[z]$；向量码本则保存 $K$ 个 $g$ 维向量，$\widehat v=C[z]\in\mathbb R^g$。若 $K=2^B$，主体码率为 $B/g$ bit/weight。

| 表示 | 一次索引表示什么 | 代表实例 |
| --- | --- | --- |
| 均匀标量 | 一个等间距数值 | GPTQ 等方法的常见网格 |
| 非均匀标量 | 一个可学习代表值 | SqueezeLLM 每输出行 LUT |
| 固定结构向量 | 一组有相关结构的数值 | QuIP# 的 8 维 E8P |
| 可学习加性向量 | 多个码本向量的和 | AQLM |

教学例子：预算为 1 bit/weight、两个数一组。标量方案各选两个值，其四种组合构成矩形；向量方案可把四个码字放在任意四个二维位置，更好覆盖斜向分布。这是更多表示自由度，不是任意任务上的必然优势；向量码本还要付出存储和搜索代价。

## 2. 选最近码字时，“距离”是什么

普通向量量化选择 $z=\arg\min_k\|v-C[k]\|_2^2$。若目标是层输出重构，输入统计会改变距离。对行误差 $e$，目标为 $eGe^\mathsf T$，$G=XX^\mathsf T$；$G$ 的非对角项使不同坐标误差能够抵消。

教学反例：$G=\begin{bmatrix}1&0.9\\0.9&1\end{bmatrix}$，两个候选误差为 $e_a=(1,0)$、$e_b=(1,-1)$。欧氏平方误差分别为 1 和 2，而输出代理误差分别为 1 和 0.2。选择欧氏最近候选与选择最小输出误差候选可能相反。

[QuIP#](../methods/quip-sharp.md) 用块间二阶反馈后再查固定向量码本；[AQLM](../methods/aqlm.md) 则在码本与索引优化中显式使用整个输入 Gram。它们都使用向量表示，却没有执行完全相同的距离搜索。[SqueezeLLM](../methods/squeezellm.md) 的标量加权聚类又采用任务梯度构造的对角敏感性，不能与上述输入 Gram 混同。

## 3. 残差量化与加性量化的区别

加性表示为 $\widehat v=\sum_{k=1}^M C_k[z_k]$。残差向量量化从 $r_0=v$ 出发，依次选择 $C_k[z_k]$ 并令 $r_k=r_{k-1}-C_k[z_k]$；前面的决定通常不再回改。它实现简单，但早期选择未必适合后续可用码字。

AQLM 使用残差 K-means 初始化，随后交替更新索引和码本。原文 §3 的目标展开含码本间、输入组间交叉项，因此最终过程超出了单次贪心残差编码。QuIP# 的 3/4 bit 实验则用带尺度的固定格码本顺序量化残差。两者都能写成向量和，优化算法仍然不同。

一个码本有 $2^{MB}$ 项时可以直接表达全部组合；使用 $M$ 个各 $2^B$ 项码本仅需线性数量的存储，但将表示限制为码字之和。它在表示自由度、码本内存和解码加法之间取舍，不是无代价保留任意大码本的全部能力。（AQLM §3；QuIP#“Scaling E8 to Higher Bitrates”。）

## 4. 为什么 E8P 能用很小的底表

格是由离散规则生成的点集。QuIP# 的 E8P 从 $E_8$ 的平移与符号对称性构造 65,536 个 8 维码字；只存 256 个绝对值模式并紧凑打包，用符号、偶校验和平移位恢复其余结构。其底表为 1 KiB，跨层共享；不是把 65,536 个任意向量压进 1 KiB。

[QuIP# 方法页](../methods/quip-sharp.md) 展开 8+7+1 bit 编码。这里的一般认识是：结构约束同时限制可学习自由度并降低解码成本。论文中“格的球堆积密度高”有助于解释设计，但本身不是任意有限分布、有限码本和神经网络任务损失上的最优证明。

## 5. 三类格式怎样计账

下表按单个 $m\times n$ 矩阵、整组无 padding、理想紧凑存储计账；浮点表或尺度按注明的 16 bit 计算。真实文件还需加未量化参数、对齐与容器，运行峰值另加缓存和工作区。

| 表示 | 平均 bit/weight | 含义 |
| --- | --- | --- |
| AQLM，组宽 $g$，$M$ 个 $2^B$ 项 FP16 码本 | $MB/g+16gM2^B/(mn)+16/n$ | 索引 + 层码本 + 行尺度 |
| SqueezeLLM，逐输出行 $2^b$ 项 FP16 表 | $b+16\cdot2^b/n$ | 主体索引 + 每行 LUT，不含稀疏 |
| SpQR，主体 $b_w$、一级统计 $b_q$、两级组宽 $\beta_1,\beta_2$ | $b_w+2b_q/\beta_1+64/(\beta_1\beta_2)$ | 主体 + 量化 scale/zero + 四个 FP16 二级参数，不含稀疏 |

AQLM 的 $1\times16$ 与 $2\times8$ 在 $g=8$ 时主体都为 2 bit/weight，但 FP16 码本分别为 1 MiB 与 8 KiB。码本开销对小矩阵占比更高，不能只引用大模型平均值。SpQR 主体码保留所有位置时，例外另外保存值和索引，不能用简单的 $(1-p)b+p\cdot16$ 代替整个格式。

这些公式分别依据 AQLM“Estimating model size”、SqueezeLLM 的逐行表设计与 SpQR 表示章节推导。部署核算必须换入真实 dtype；例如 [SqueezeLLM 的固定代码片段](../methods/squeezellm.md) 使用 FP32 表与稀疏值，不能原样套入 FP16 理想公式。[GGUF](../implementation/gguf-block-quantization-formats.md) 展示另一个按实际块字节核算的例子，具体格式不能与论文码本互换。

## 6. 从编码走向计算

压缩权重可在使用前分块解码，也可借助输入相关查表。对 AQLM 的一个输入组 $x_j$，定义 $T_{j,k,z}=C_k[:,z]^\mathsf Tx_j$，则

$$y_i=s_i\sum_{j,k}T_{j,k,z_{ijk}}.$$

这是实数算术中的代数改写；实际浮点求和顺序会改变末位误差。临时表需要 $({n}/{g})M2^B$ 项，码本太大时构表和访存会抵消收益。GPU 还可能选择查表恢复向量后乘加，具体路径由实现决定。

小 batch decode 的低权重复用使读取节省更有价值；batch 增大时，计算与解码组织的重要性上升。判断方法时应沿“编码大小 → 解码方式 → 访存与算术 → 质量和整段推理测量”走通，不能把压缩比直接当作速度比。

## 来源身份

- [QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks](https://arxiv.org/abs/2402.04396v2)，arXiv:2402.04396v2；BlockLDLQ、E8P 与高位宽残差编码。
- [Extreme Compression of Large Language Models via Additive Quantization](https://arxiv.org/abs/2401.06118v4)，arXiv:2401.06118v4；§3、位宽核算与推理章节。
- [SqueezeLLM: Dense-and-Sparse Quantization](https://arxiv.org/abs/2306.07629v4)，arXiv:2306.07629v4；非均匀量化与逐行 LUT。
- [SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression](https://arxiv.org/abs/2306.03078v1)，arXiv:2306.03078v1；两级元数据与例外存储。
- [SqueezeAILab/SqueezeLLM](https://github.com/SqueezeAILab/SqueezeLLM/tree/a5fd71f353bf569feb7b55737c1c1493e78e8f31)，commit `a5fd71f353bf569feb7b55737c1c1493e78e8f31`；`quant.py` 中 LUT、稀疏值和索引的 dtype。
