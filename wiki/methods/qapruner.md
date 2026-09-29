---
title: QAPruner：量化敏感度与视觉 token 选择的耦合
type: method
tags:
  - vlm
  - sparsity
  - ptq
  - sensitivity
sources:
  - raw/papers/2026-09-21/qapruner/paper.pdf
updated: 2026-09-29
---

# QAPruner：量化敏感度与视觉 token 选择的耦合

QAPruner v1 把 token 的模拟 INT4 重构误差和通道范围加入语义剪枝分数，在固定视觉 token 预算下重新选择保留集合。目标是改善已量化模型的任务质量；它不学习新的量化器，也不是联合优化权重、scale 与删除集合的求解器。

本页研读完整正文及两张结果表；所存 12 页后两页主要为参考文献，没有补充算法附录。未取得可核验作者实现或运行模型。前置见 [视觉 token 与量化对象](../fundamentals/model/vision-language-model-tokens-and-quantization.md)，与纯剪枝的关系见 [CoRePrune](coreprune.md)和 [LayerPos](layerpos.md)。

## 1. 剪枝与量化改变的是不同对象

对线性层 $Y=XW$，量化得到 $Q(X)Q(W)$；token 剪枝则先选择序列行的子集，并改变后续 attention 的交互。用浮点模型选出的集合未必适合低比特模型，因为幸存 token 的误差、数值范围以及后续表示都可能变化。

论文以 Q-VLM 为 PTQ 基础、CDPruner 为剪枝基线，主实验 W4A4。它主张语义选择可能漏掉数值上重要的离群 token，并给出 ScienceQA 实例和同预算准确率对照。这里应区分三个问题：token 本身难量化、删除它是否损害模型、保留它能否减轻其他 token 的误差。本文的评分直接估计第一个问题，后两个问题主要靠经验验证。

尤其在每 token 内独立分组且特征不变时，删掉另一 token 不会改变该 token 的局部 scale。只有共享统计或后续网络交互改变时，才可能改变其量化结果。因此“离群 token 决定整个 tensor 的范围”不能作为逐 token 分组量化下的无条件因果解释；保护离群 token 也不等于普遍扩大范围更好。

## 2. 全局 PTQ 描述与敏感度模拟不是同一个量化器

§3.1 用组内 min–max 非对称网格描述 PTQ：

$$s=\frac{\max T-\min T}{2^b-1},\quad Z=\min T,\quad
Q(T)=s\,\operatorname{clip}\!\left(\operatorname{round}\frac{T-Z}{s},0,2^b-1\right)+Z.$$

这里原文称为 zero-point 的 $Z$ 是实数域最小值偏移，不是通用整数仿射格式中的整数零点；公式输出已经反量化。它没有完整规定零范围组处理、真实打包和整数内核。

§3.2 的选择代理改用**对称** INT4。将第 $i$ 个 token 特征 $v_i\in\mathbb R^D$ 按通道分成 $M=D/G$ 组，示例 $G=128$，计算

$$s_{im}=\frac{\max|v_{im}|}{7},\quad
\widehat v_{im}=s_{im}\operatorname{round}\left(\frac{v_{im}}{s_{im}+\epsilon}\right),\quad
E_i=\|v_i-\widehat v_i\|_2.$$

$E_i$ 是二范数，不是平方误差或平均 MSE；$\epsilon$ 只在除数中，故也不能无条件将其改写成标准带 clipping 的部署算子。将分组维度写成整数还隐含 $G$ 整除 $D$，尾组规则未说明。

前文将视觉编码器输出 $v_i$ 投影为 $x_i=g(v_i)$，后文又以 $v_i$ 表示所评分特征；没有实现时无法确认全部实验评分点都在 projector 前还是后。应保留这一符号/接口缺口，不能把对称模拟误差当作实际 Q-VLM 每个目标张量的精确量化误差。

## 3. 两个数值分数怎样与语义分数合并

第二个指标是 token 内全部通道的范围

$$R_i=\max_jv_{ij}-\min_jv_{ij}.$$

它不是最大绝对值，也不是跨 token 的通道离群统计。两个指标有互补性，但原文所谓 orthogonal metrics 不是统计独立或几何正交的证明。整理者例子：忽略 $\epsilon$，单组 $(-7,0,7)$ 在对称 INT4 下完全可表示，$E=0$ 而 $R=14$；范围大不必难量化。将每个通道同时加常数会保持 $R$，却可能改变对称网格上的 $E$。

分别在当前 $N$ 个视觉 token 上 min–max 归一化：

$$\widetilde E_i=\frac{E_i-\min E}{\max E-\min E},\quad
\widetilde R_i=\frac{R_i-\min R}{\max R-\min R},\quad
S_i^Q=\frac{\widetilde E_i+\widetilde R_i}{2}.$$

再融合原剪枝方法的语义分数 $S_i^P$：

$$S_i=\alpha S_i^P+(1-\alpha)S_i^Q,\qquad S=\operatorname{TopK}_i S_i.$$

高 $E/R$ 提高保留优先级，并不提高该 token 的实际位宽。$\alpha=1$ 退回语义选择，$\alpha=0$ 只保留数值代理；只有两种分数尺度可比，$\alpha$ 才有清晰的权衡含义。论文没有充分交代 $S^P$ 的归一化、实际 $\alpha$、常量分数的零分母处理或 batch 内/跨样本边界。极端 token 还会改变整组 min–max 尺度，所以其他 token 的融合排序可能间接变化。

图 2 给出 TopK 流程，但基线 CDPruner 原本使用集合多样性选择，本文未完整展开如何把它变成所用 $S^P$，或是否保留其集合优化步骤。这是复现实验的必要接口，不能按熟悉的 CDPruner 实现自行填补。

## 4. 与条件可删除性和位置策略怎样衔接

[CoRePrune](coreprune.md) 估计固定 attention 状态下的删除集合扰动，包含 Value 方向交互和重归一化；QAPruner 的 $E_i/R_i$ 则是单 token 数值代理。高重构误差不证明删除损失大，单 token 分数相加也不刻画联合删除。这两条路线可以提出组合实验，但当前论文没有验证与 CoRePrune 的组合收益。

[LayerPos](layerpos.md) 进一步提醒：即使所选 token 相同，稀疏/连续位置编号也会改变 logits。本论文未详细报告剪枝层、position IDs 与 decode 缓存约定，因此不能自动假定其收益已排除了位置策略影响。PTQ 校准后再剪枝与先剪枝再重校准也不是同一对照；论文“基于量化模型”的叙述不足以复原完整校准顺序及统计是否重算。

## 5. 结果支持有限任务上的选择改善

表 1 均为 ScienceQA，而 NAT/SOC/LAN、TXT/IMG/NO 是任务子集，不是六个独立基准。选取 32-token 设置：

| 模型 | 原 token 数 | 浮点 dense | W4A4 dense | W4A4＋原剪枝 | QAPruner |
| --- | --- | --- | --- | --- | --- |
| LLaVA-1.3-7B | 256 | 89.60 | 80.36 | 78.61 | 80.85 |
| LLaVA-1.3-13B | 256 | 91.09 | 84.39 | 82.69 | 84.27 |
| LLaVA-1.5-7B | 576 | 67.79 | 60.62 | 61.05 | 61.59 |

7B 的 78.61→80.85 是 **2.24 个百分点**，不是相对提升 2.24%；80.85 高于量化 dense 的 80.36，却仍明显低于浮点 89.60。32/256=12.5%，32/576≈5.56%，不能给所有模型沿用 12.5% 标签。

“六种剪枝配置中五种超过量化 dense”与表一致，但 13B/32 的 84.27 低于 84.39；LLaVA-1.5 的原剪枝本身也已略高于量化 dense，因此超过 dense 并非新评分独有的效果。没有误差条或重复种子报告，不能给微小差值附加统计显著性。

表 2 中 Group-wise AbsMax 和 Outlier Intensity 单独均为 80.78，联合为 80.85，仅再增 0.07 个百分点。这支持联合指标在该设置的最高观测均分，没有证明二者强协同或普遍最优；G1/G7 列的划分与汇总规则也未完整定义。少量定位实例不足以单独证明“保留离群 token 导致改善”的因果链。

## 6. 成本与可复现范围

按公式，评分需要遍历 $N\times D$ 特征做分组量化模拟、范数、极值和归一化，再选择 TopK；这是整理者对操作的计数，不是作者测速。论文没有完整硬件、batch、延迟、峰值内存或真实 W4A4 内核测量，不能将减少视觉 token 直接换算成端到端加速。

本页已能解释评分和所报告的条件化收益；实际 $\alpha$、语义评分接口、特征采集点、校准顺序与位置策略仍影响精确复现。它们作为明确缺口保留，而非补造一份“作者算法实现”。

## 来源身份

- [QAPruner: Quantization-Aware Vision Token Pruning for Multimodal Large Language Models](https://arxiv.org/abs/2604.02816v1)，arXiv:2604.02816v1；§3、图 2、§4、表 1–2。图表和关键公式回查 PDF 页图；范围/网格反例及条件区分为整理者分析，未复现实验。
