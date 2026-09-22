---
title: QuIP#：Hadamard 非相干处理与 E8P 格码本
type: method
tags:
  - ptq
  - weight-quantization
  - rotation
  - vector-quantization
sources:
  - raw/papers/2026-09-21/quip-sharp/paper.pdf
  - raw/papers/2026-09-21/quip-sharp/source.eprint
updated: 2026-09-22
---

# QuIP#：Hadamard 非相干处理与 E8P 格码本

QuIP# 沿用 [QuIP](quip.md) 的输入二阶重构目标，将随机正交处理换成快速 Hadamard 变换，将逐标量舍入换成 BlockLDLQ 配合 8 维格码本，再通过小规模微调恢复模型输出。关键是同时设计误差、码本大小和解码代价；它量化权重，不能仅因“2 bit”就理解为硬件 INT2 乘法。

## 1. 随机 Hadamard 改变什么

设 $P_n$ 为归一化正交 Hadamard 矩阵，$S_n$ 为独立随机 $\pm1$ 对角矩阵，输入侧变换为 $R_n=P_nS_n$。对 $W\in\mathbb R^{m\times n}$ 及 $H=\mathbb E[xx^\mathsf T]$，

$$\widetilde W=R_mWR_n^\mathsf T,\qquad
\widetilde H=R_nHR_n^\mathsf T.$$

当维度为 2 的幂时，快速蝶形算法将向量变换从 QuIP 两因子的 $O(n\sqrt n)$ 降到 $O(n\log n)$。Hadamard 的符号和归一化不能漏掉；非 2 幂维度需使用可构造的因子，例如 $28672=1024\times28$，不能直接套任意长度的蝶形。原文另给 RFFT 备选，不表示所有形状都有同一实现代价。

原文“Randomized Hadamard Transform”给出高概率非相干界：$\mu_H=\sqrt{2\log(2n^2/\delta)}$、$\mu_W=2\log(4mn/\delta)$，分别以至少 $1-\delta$ 概率成立。它改善的是界及计算组织，并不保证每个随机种子都降低某个具体张量的最大值。理论定义与旋转反例见 [正交旋转](../theory/orthogonal-rotation-and-hadamard-quantization.md)。

## 2. BlockLDLQ 把什么从标量推广到向量

将输入列每 $g$ 个组成一块，写 $H=TDT^\mathsf T$，其中 $T$ 为单位块上三角矩阵，$D$ 为块对角矩阵。依次量化

$$
\widehat W_{:,G_k}=\mathcal Q_C\left(W_{:,G_k}+
(W_{:,G_{<k}}-\widehat W_{:,G_{<k}})(T-I)_{G_{<k},G_k}\right).
$$

$\mathcal Q_C$ 对每一输出行的一组 $g$ 个数选一个向量码字；块外误差仍通过二阶反馈传递。原始 LDLQ 每次固定一列，这里每次固定一个向量块，避免把向量码本错误拆成多个独立标量量化器。$D$ 的块一般不是对角阵，不能把实现中的欧氏最近码字描述成精确求解任意块内 Hessian 度量。

论文 BlockLDLQ 定理假定量化器对任意输入的误差二阶矩满足 $\mathbb E[ee^\mathsf T]\preceq\sigma^2I$，据此得到带非相干参数的代理误差界。有限 E8P 码本对任意无界输入不自动满足这一统一条件，因此该定理不是有限 E8P 在所有权重上的无条件性能保证。

## 3. E8P 为什么是 2 bit，却有 65,536 个码字

8 个权重共享 16 bit 索引，所以码率是 $16/8=2$ bit/weight；并非每个权重独立从四个数中选择。背景见 [码本量化](../theory/codebook-quantization-and-bit-budget.md)。原文“The E8P Codebook”从格

$$
E_8=\left(\mathbb Z^8\cup(\mathbb Z^8+\tfrac12)\right)
\cap\{x:\mathbf1^\mathsf Tx\in2\mathbb Z\}
$$

构造有限、近球形的码本。高密度格点有利于覆盖近似球形的分布；Hadamard 后权重近似高斯是设计动机和经验近似，不能把有限模型权重当作独立标准高斯。

直接保存 $2^{16}\times8$ 个 FP16 数需 1 MiB。E8P 利用对称性将每个 16 bit 码字拆为：

- 8 bit 选择 256 个绝对值模式之一；它们来自 227 个范数不大于 $\sqrt{10}$ 的模式及 29 个范数为 $\sqrt{12}$ 的补充模式。
- 7 bit 保存符号，最后一个符号由模式与偶校验条件确定。
- 1 bit 选择整体 $+1/4$ 或 $-1/4$ 平移。

底表自身还能紧凑编码，论文报告占 1 KiB；逻辑上的 256×8 模式表不能直接按 FP16 数组计算成这一大小。解码必须同时处理模式、符号、平移和尺度，不能把索引当成 8 个均匀 INT2 数。码本的规则结构与跨层共享减轻缓存压力，是它与大规模逐层学习码本的重要差别。

3 bit 使用 2 bit E8P 加 1 bit 格码本量化残差；4 bit 用两次 2 bit E8P。每级先按其尺度归一化残差，选码字后恢复尺度并累加。这是顺序残差向量量化，不等于 [AQLM](aqlm.md) 交替优化多个可学习码本与联合索引。

## 4. 微调变量与执行边界

原文“Fine-Tuning During Quantization”及附录算法分两阶段：逐块记录浮点输出，每量化一层就冻结该层离散权重编码，优化尚未量化层与允许更新的参数，降低块输出 MSE；全部线性层处理后，再以教师输出分布约束整个模型，优化归一化参数、变换向量和 LM head 等剩余参数。

符号向量在微调时放松成实数并以 FP16 保存。此后这些向量不必满足 $\pm1$，所以最终模型不再完全属于初始随机正交变换的理论条件；效果要由校准与验证衡量。对一个 $m\times n$ 矩阵，两侧 FP16 向量增加 $16(m+n)/(mn)$ bit/weight；4096×4096 时为 0.0078125。主体 2 bit、变换向量、共享码本和其他未量化参数仍应分别计账。

推理需要输入变换、码本解码和乘加、输出变换。小 batch decode 常受权重读取限制，固定小码本的优势要通过实际访存与解码实现兑现；不能直接假设现有 AWQ/Marlin kernel 能接收 E8P。通用执行边界见 [量化矩阵乘法](../implementation/quantized-matmul-scaling-execution.md)。

## 5. 证据：区分码本、微调与模型规模

v2 的 Llama-2 实验用 RedPajama 6144 段、原生长度 4096 估计 Hessian；块内微调用另设训练/验证划分。WikiText2、上下文 4096、Llama-2-7B 的 2 bit 消融为：

| 配置 | PPL，越低越好 |
| --- | ---: |
| FP16 | 5.12 |
| RHT + 标量网格，无微调 | 11.2 |
| RHT + E8P，无微调 | 8.22 |
| 完整 QuIP#，含微调 | 6.19 |

这是同一论文中的逐步消融，分别支持向量码本和微调的贡献；不能把旧的无微调结果与另一方法微调后的结果混成最终排名。（Experiments，context length 4096 表。）

论文讨论的“3 bit 优于 4 bit”涉及**给定模型存储预算下跨模型规模的质量曲线**，不是同一模型降低位宽必然提升质量。其 RTX 4090 生成实验另区分 FlashAttention 与 Hugging Face 路径；本页没有据不同实现的吞吐数字作横向排名，也没有运行原模型。

## 来源身份

[QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks](https://arxiv.org/abs/2402.04396v2)，arXiv:2402.04396v2。使用方法、实验、微调算法与实现细节附录；未将全部附录证明或官方 kernel 复现计入本次范围。
