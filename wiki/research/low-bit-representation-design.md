---
title: 低比特表示的设计空间：坐标、码本与例外预算
type: concept
tags:
  - weight-quantization
  - rotation
  - vector-quantization
  - sparsity
sources:
  - raw/papers/2026-09-21/quip
  - raw/papers/2026-09-21/quip-sharp
  - raw/papers/2026-09-21/aqlm
  - raw/papers/2026-09-21/spqr
  - raw/papers/2026-09-21/squeezellm
  - raw/papers/2026-10-07/dbw
  - raw/repositories/2026-10-07/dbw
  - raw/papers/2026-10-07/stagq
updated: 2026-10-15
---

# 低比特表示的设计空间：坐标、码本与例外预算

低比特表示要在有限存储下保留模型质量，并让保存的参数能够有效参与推理。可选的手段不只有舍入：QuIP、QuIP#、AQLM、SpQR 和 SqueezeLLM 改变坐标、码本或例外预算；DBW 进一步改变主体因子的宽度；StagQ 则让同一份编码支持多个读取精度。本页比较这些选择改变了什么、付出什么代价，帮助判断哪些设计能够配合，哪些约束不能混用。

## 1. 将可优化变量分开

先考虑直接编码完整权重矩阵的表示。对权重 $W$，一种便于讨论的抽象是

$$\widehat W=\Phi^{-1}\big(\operatorname{Decode}(z;C,s)+R\big).$$

$\Phi$ 表示可能存在的可逆坐标或尺度变换，$z$ 为主体离散编码，$C,s$ 为码本及量化参数，$R$ 是在该表示空间中的稀疏修正。它是对前五种方法的整理，不表示它们都使用全部变量；没有变换时 $\Phi=I$，没有例外时 $R=0$。稀疏性依赖所选坐标：在旋转空间稀疏，不意味着恢复到原坐标后仍稀疏。

这不是覆盖所有表示的统一公式。DBW 编码的是两个因子，需共同选择因子宽度与位宽；StagQ 的各精度重建则必须来自同一编码的前缀，不能把每个精度的码字和尺度当作独立变量。两者分别在第 6、7 节展开。共同的比较顺序是先明确表示对象与共享约束，再计算误差、存储和执行成本：

```mermaid
flowchart LR
  A[原权重与使用需求] --> B[完整矩阵或因子表示]
  B --> C[坐标、宽度与主体编码]
  B --> D[例外与修正预算]
  H[单一精度或共享前缀约束] --> C
  H --> D
  C --> E[完整解码与输出误差]
  D --> E
  E --> F[模型质量与真实存储]
  E --> G[推理访存与计算代价]
```

| 方法 | 主要改变什么 | 随之需要考虑什么 |
| --- | --- | --- |
| [QuIP](../methods/quip.md) | 坐标与二阶反馈 | 变换后的量化误差与还原成本 |
| [QuIP#](../methods/quip-sharp.md) | 快速变换、格码本和微调变量 | 规则码本的表达能力与解码成本 |
| [AQLM](../methods/aqlm.md) | 可学习的加性码本与联合索引 | 主体误差、码本存储与访问成本 |
| [SpQR](../methods/spqr.md) | 小组、两级元数据与稀疏例外 | 局部适配收益及元数据、索引开销 |
| [SqueezeLLM](../methods/squeezellm.md) | 任务敏感性加权的代表值与例外 | 任务代理误差及非均匀解码成本 |
| [DBW](../methods/dbw.md) | 主体因子宽度与成对位宽 | 两侧因子总存储及两次矩阵乘成本 |
| [StagQ](../methods/stagq.md) | 多个精度共享编码与元数据 | 省下多份表示，同时限制各精度独立拟合 |

## 2. 不同目标会把预算投向不同位置

以层输出重构为目标，令 $W,\widehat W\in\mathbb R^{m\times d}$ 为原权重及近似权重，$E=\widehat W-W$，$X\in\mathbb R^{d\times n}$ 的每列为一个固定校准输入，$G=XX^{\mathsf T}$。则输出平方误差总和为

$$\|EX\|_F^2=\operatorname{tr}(EGE^{\mathsf T}).$$

$G$ 是输入的未中心化二阶乘积总和，不是默认取均值或中心化的协方差；非对角项保留通道间误差相互抵消或加强的影响。这是固定输入、线性输出下的恒等式，不能直接当作整个模型的任务损失。

SqueezeLLM 使用的任务敏感性目标则为 $\sum_i f_i e_i^2$：$e_i=\widehat w_i-w_i$ 是第 $i$ 个参数的误差，$f_i$ 是样本任务梯度平方的平均值，即经验 Fisher 矩阵的第 $i$ 个对角元。它忽略任务损失的一阶项，以经验 Fisher 近似 Hessian，再舍去跨参数交互；这些近似不由模型已训练自动保证。两种目标对“重要”的定义不同，不能仅凭都用了二阶信息就合并。（SqueezeLLM v4，Methodology。）

由此得到三个推论：

1. 相同权重 MSE，不保证相同层输出误差或任务质量。
2. 校准集改变后，输入相关性或任务梯度改变，码本与例外选择也可能改变。
3. 单层目标降低不等于整模型误差降低，所以 QuIP#、AQLM 另有块级或全模型微调；这种收益不能全部归功于码本形状。

这些是由各方法目标整理出的分析，不是跨模型实验证明。具体推导见 [层重构](../theory/layer-reconstruction-second-order-compensation.md) 与 [码本距离](../theory/codebook-quantization-and-bit-budget.md)。

## 3. 例外保留本质上是一次局部预算分配

在第 1 节的直接编码表示中，给定候选例外集合 $S$，应比较“允许保留 $S$ 后重新优化主体”的误差与原误差，而不是只按原始权重大小排序。以存储为预算的问题可写成

$$\min_{S,C,z,\Phi}\mathcal E(\widehat W)
\quad\text{s.t.}\quad
B_{\rm codes}+B_{\rm metadata}+B_{\rm exceptions}+B_{\rm transform}\le B.$$

例外可能带来两重收益：保护自身、释放主体网格或码本的容量。SpQR 排除某权重后重新拟合组统计，SqueezeLLM 将尾部和敏感值移出共享代表值竞争，都体现了后者。

各例外收益通常不独立相加：移走第一个极值后，第二个值对范围的作用可能改变；若主体重新优化，误差也重新分布。因此简单按“单点收益/bit”排序只是启发式，不应冒充全局最优 knapsack 解。分配对象由整层细化到零散权重后，索引和非规则执行成本也增加，见 [混合精度分配](../theory/mixed-precision-allocation.md)。

## 4. 为什么这些选择不能随意叠加

随机旋转会扩散原坐标中的离群值，也会扩散原有稀疏例外；原坐标选出的稀疏结构未必能在旋转后廉价执行。加性码本可以降低主体误差，但大码本增加缓存压力；稀疏保留可能缩小所需码本，又增加索引、归约和访存。更多连续微调变量提高适配能力，同时可能离开最初正交或固定码本理论的条件。

DBW 与 StagQ 还改变了比较的前提：前者把存储对象换成两侧因子，原矩阵的平均位宽不能直接充当因子位宽；后者让多个精度共享表示，分别优化各精度得到的解不一定满足前缀约束。因此，比较压缩率前要先对齐保存对象，比较质量前要明确是服务一个精度还是多个精度。

因此“把 QuIP#、AQLM、SpQR 的优点全部组合”不是已经验证的方法。一个具体可检验的问题是：在相同总字节预算下，给主体增加一个小码本与增加少量高精度例外，哪种在目标 workload 中更合算？要同时观察模型质量、码本访问、稀疏密度与行分布，不能仅看局部权重误差。

## 5. 怎样检验一个新组合

先固定模型、校准输入和质量评测协议，再以实际保存字节比较候选。分别关闭坐标变换、额外码本、例外、微调，区分收益来源；各消融如果改变文件大小，应另设相同预算的比较。

执行上先验证完整解码一致性，尤其是尺度、码字和稀疏修正是否重复相加；再区分 batch 1 decode、prefill 和批处理。查表或稀疏分支的微基准不能代替整段生成或服务收益。质量与服务指标由 [模型质量评测](../implementation/model-quality-evaluation.md) 和 [服务性能评测](../implementation/serving-performance-evaluation.md) 承接。

这里给出的是后续研究的可检验设计，不报告已经执行组合实验。

## 6. 主体也可以改变因子宽度

前面的坐标变换、码本和例外都围绕主体表示工作；[DBW](../methods/dbw.md)增加另一种选择：用两个可训练因子替换整个主权重，每对分量的正位宽控制量化网格，0 bit 减少两次矩阵乘共享的中间宽度。它不是可逆坐标变换，也不是量化主体之外另加一个低秩残差。相同的原矩阵 bpw 可以对应不同因子宽度/位宽组合，必须按两侧因子元素数共同计费。（DBW v1 §2.2–2.3。）

SVD 只初始化因子；训练和量化后的方向不必保持相互正交，保留 $r$ 对分量只保证有效矩阵秩不超过 $r$。因此不能把最佳截断 SVD 的解析结论直接推广到学习后表示。该方法页进一步区分理想载荷、word packing 与每次前向的浮点恢复，避免从“低秩+低 bit”直接推算完整速度。

## 7. 多精度前缀使主体与例外一起承担约束

[StagQ](../methods/stagq.md)用一对共享元数据支持多个 refinement 前缀，把 bit-plane 和 precision-dependent 原点折成普通 affine 整数；新增 plane 不新增一份组尺度。这个取舍限制每精度的独立网格自由度，也让小 group 的元数据成本保持较低。固定元数据的载荷设计与运行 kernel 的重打包/每组查表仍须分别解释。（StagQ v1 §2–3、附录 G。）

该方法的 pre-grid guard 先向内截断极值后重拟合主体，post-grid guard 再保护两位残差：前者可释放组内其他值的网格分辨率，后者只补固定网格留下的损失。保护位置对所有精度固定，后续 plane 不覆盖；因此不能在深精度丢掉记录，也不能把相同两位残差直接加到已细化主体上。完整保护值的定义见方法页。这将第 3 节“保护自身与释放主体容量”的区别扩展到一个主体服务多个精度的情况。（StagQ v1 式 (4)、§3.5–3.6。）

## 来源身份

- QuIP: 2-Bit Quantization of Large Language Models With Guarantees（`https://arxiv.org/abs/2307.13304v2`），arXiv:2307.13304v2；坐标处理、谱界及有限网格边界。
- Differentiable Bit-Widths: Co-optimizing Pruning and Quantization via SVD for Ultra-Efficient LLM Compression（`https://arxiv.org/abs/2610.06026v1`），`arXiv:2610.06026v1`；主体因子宽度、成对位宽与两阶段执行。
- MMAI-Laboratory/DBW（`https://github.com/MMAI-Laboratory/DBW/tree/a45f26e1d153ab6bc64dd27d056a3710a33f2e7a`），`a45f26e1d153ab6bc64dd27d056a3710a33f2e7a`；`real_dbw.py::get_uv/forward` 的因子恢复与方向归一化。
- StagQ: Constraint-Driven Multi-Precision Weight Quantization for LLMs（`https://arxiv.org/abs/2610.05977v1`），`arXiv:2610.05977v1`；共享 pair 的前缀主体、两阶段 guard 与固定 side record。
- QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks（`https://arxiv.org/abs/2402.04396v2`），arXiv:2402.04396v2；RHT、E8P 与微调。
- Extreme Compression of Large Language Models via Additive Quantization（`https://arxiv.org/abs/2401.06118v4`），arXiv:2401.06118v4；联合索引、码本优化和存储。
- SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression（`https://arxiv.org/abs/2306.03078v1`），arXiv:2306.03078v1；小组元数据、动态敏感性与例外。
- SqueezeLLM: Dense-and-Sparse Quantization（`https://arxiv.org/abs/2306.07629v4`），arXiv:2306.07629v4；任务敏感性与主体/例外分工。
