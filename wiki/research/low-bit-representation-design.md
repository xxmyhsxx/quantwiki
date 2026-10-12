---
title: 低比特表示的设计空间：坐标、码本与例外预算
type: concept
tags:
  - weight-quantization
  - rotation
  - vector-quantization
  - sparsity
sources:
  - raw/papers/2026-09-21/quip/source.eprint
  - raw/papers/2026-09-21/quip-sharp/source.eprint
  - raw/papers/2026-09-21/aqlm/source.eprint
  - raw/papers/2026-09-21/spqr/source.eprint
  - raw/papers/2026-09-21/squeezellm/source.eprint
  - raw/papers/2026-10-07/dbw/paper.pdf
  - raw/repositories/2026-10-07/dbw/source/dbw/real_dbw.py
  - raw/papers/2026-10-07/stagq/paper.pdf
updated: 2026-10-07
---

# 低比特表示的设计空间：坐标、码本与例外预算

极低位宽量化不只是在相同网格上改进舍入。QuIP、QuIP#、AQLM、SpQR 和 SqueezeLLM 共同揭示：可以改变表示的坐标、允许的码字集合、误差优化方式，以及少量例外所获得的预算。本页把这些选择整理为一般问题，不将五种方法简化成精度排行榜。

## 1. 将可优化变量分开

对权重 $W$，一种便于讨论的抽象是

$$\widehat W=\Phi^{-1}\big(\operatorname{Decode}(z;C,s)+R\big).$$

$\Phi$ 表示可能存在的坐标或尺度变换，$z$ 为主体离散编码，$C,s$ 为码本及量化参数，$R$ 是在该表示空间中的稀疏修正。它是整理者的统一表述，不表示五种方法都同时使用全部变量；没有变换时 $\Phi=I$，没有例外时 $R=0$。稀疏性依赖所选坐标：在旋转空间稀疏，不意味着恢复到原坐标后仍稀疏。

```mermaid
flowchart LR
  A[原权重与校准数据] --> B[选择表示坐标]
  B --> C[主体编码与码本]
  B --> D[高精度例外预算]
  C --> E[完整解码与输出误差]
  D --> E
  E --> F[模型质量与真实存储]
  E --> G[推理访存与计算代价]
```

- [QuIP](../methods/quip.md) 主要改变坐标与二阶反馈，使谱结构在量化坐标下更有利。
- [QuIP#](../methods/quip-sharp.md) 同时改变快速变换、向量编码和微调变量，并用规则格结构限制码本成本。
- [AQLM](../methods/aqlm.md) 学习可加的码本及激活感知索引，把预算用于主体表示的自由度。
- [SpQR](../methods/spqr.md) 用小组和两级元数据适配局部分布，把额外预算投向难以补偿的例外。
- [SqueezeLLM](../methods/squeezellm.md) 以任务敏感性拉动非均匀代表值，再把少量尾部和高敏感权重移出主体。

## 2. 不同目标会把预算投向不同位置

以层输出重构为目标时，损失为 $\operatorname{tr}(EGE^\mathsf T)$；通道间误差可以抵消。以任务 Fisher 对角近似为目标时，损失为 $\sum_i f_ie_i^2$；每个参数有独立权重，但跨参数交互被舍弃。两者对“重要”的定义不同，不能仅凭都用了二阶信息就合并。

由此得到三个推论：

1. 相同权重 MSE，不保证相同层输出误差或任务质量。
2. 校准集改变后，输入相关性或任务梯度改变，码本与例外选择也可能改变。
3. 单层目标降低不等于整模型误差降低，所以 QuIP#、AQLM 另有块级或全模型微调；这种收益不能全部归功于码本形状。

这些是由各方法目标整理出的分析，不是跨模型实验证明。具体推导见 [层重构](../theory/layer-reconstruction-second-order-compensation.md) 与 [码本距离](../theory/codebook-quantization-and-bit-budget.md)。

## 3. 例外保留本质上是一次局部预算分配

给定候选例外集合 $S$，应比较“允许保留 $S$ 后重新优化主体”的误差与原误差，而不是只按原始权重大小排序。以存储为预算的一般问题可写成

$$\min_{S,C,z,\Phi}\mathcal E(\widehat W)
\quad\text{s.t.}\quad
B_{\rm codes}+B_{\rm metadata}+B_{\rm exceptions}+B_{\rm transform}\le B.$$

例外可能带来两重收益：保护自身、释放主体网格或码本的容量。SpQR 排除某权重后重新拟合组统计，SqueezeLLM 将尾部和敏感值移出共享代表值竞争，都体现了后者。

各例外收益通常不独立相加：移走第一个极值后，第二个值对范围的作用可能改变；若主体重新优化，误差也重新分布。因此简单按“单点收益/bit”排序只是启发式，不应冒充全局最优 knapsack 解。分配对象由整层细化到零散权重后，索引和非规则执行成本也增加，见 [混合精度分配](../theory/mixed-precision-allocation.md)。

## 4. 为什么这些选择不能随意叠加

随机旋转会扩散原坐标中的离群值，也会扩散原有稀疏例外；原坐标选出的稀疏结构未必能在旋转后廉价执行。加性码本可以降低主体误差，但大码本增加缓存压力；稀疏保留可能缩小所需码本，又增加索引、归约和访存。更多连续微调变量提高适配能力，同时可能离开最初正交或固定码本理论的条件。

因此“把 QuIP#、AQLM、SpQR 的优点全部组合”不是已经验证的方法。一个具体可检验的问题是：在相同总字节预算下，给主体增加一个小码本与增加少量高精度例外，哪种在目标 workload 中更合算？要同时观察模型质量、码本访问、稀疏密度与行分布，不能仅看局部权重误差。

## 5. 怎样检验一个新组合

先固定模型、校准输入和质量评测协议，再以实际保存字节比较候选。分别关闭坐标变换、额外码本、例外、微调，区分收益来源；各消融如果改变文件大小，应另设相同预算的比较。

执行上先验证完整解码一致性，尤其是尺度、码字和稀疏修正是否重复相加；再区分 batch 1 decode、prefill 和批处理。查表或稀疏分支的微基准不能代替整段生成或服务收益。质量与服务指标由 [模型质量评测](../implementation/model-quality-evaluation.md) 和 [服务性能评测](../implementation/serving-performance-evaluation.md) 承接。

这里给出的是后续研究的可检验设计，不报告已经执行组合实验。五篇论文支持各自的组件和配置，尚不能替代这些组合的实证。

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
