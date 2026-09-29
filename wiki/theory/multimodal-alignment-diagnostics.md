---
title: 多模态对齐诊断：共同方向、主角谱与任务相关性
type: concept
tags:
  - vlm
  - evaluation
  - sensitivity
sources:
  - raw/papers/2026-09-26/the-alignment-illusion/paper.pdf
updated: 2026-09-28
---

# 多模态对齐诊断：共同方向、主角谱与任务相关性

视觉与文字表征相似，可能因为它们表达相关内容，也可能因为共享网络把不同输入压向相同方向。*The Alignment Illusion in Multimodal Large Language Models* v1 通过视觉干预区分这两种解释。它不是量化算法，也没有证明 PA gap 是量化误差的最优目标；它为 [模态重要性与重构目标](../research/mbq-vlmq-comparison.md) 提供的是诊断指标的使用边界。

## 1. 相似度为什么需要反事实检查

视觉编码器和 projector 产生 $V^{(0)}\in\mathbb R^{n_v\times d}$，文字嵌入为 $H^{(0)}\in\mathbb R^{n_t\times d}$；两者进入共享 Transformer。视觉/文字 token 的输入不同，使用的注意力、MLP、归一化和残差路径却共享。路径背景见 [视觉语言模型中的 token](../fundamentals/model/vision-language-model-tokens-and-quantization.md)。

论文在同一组 1,000 道 MMBench 多选题、13 个模型上设置：

| 设置 | 改动 | 主要区分什么 |
| --- | --- | --- |
| ORIG | 原始视觉输入 | 正常参考 |
| NOISE | projector 输出替换为高斯向量，逐 token 匹配原范数 | 保留尺度，移除结构化视觉内容 |
| IRR | 换成另一题的自然图像，固定配对 | 保留自然结构，破坏题图相关性 |
| SHUF | 打乱原视觉 token 顺序 | 内容与顺序的作用 |
| TEXT | 去掉视觉 token | 文字先验参考 |

13 个模型跨 LLaVA-OV、LLaVA-OV-1.5、Qwen2-VL、Qwen2.5-VL、InternVL3，0.5B–72B。NOISE 中的范数匹配是显式操作，IRR 则是实验观察到范数接近；不能将两者都写成精确等范数约束。噪声插在 projector 后，也不是对图像像素加噪。（原文方法与 evaluation protocol。）

作者报告 NOISE 相对 ORIG 准确率下降 38–50 个百分点，但 CKA、SVCCA、MIR 和最大主角余弦无法稳定地区分两者。这支持“高几何相似度不足以证明内容被正确使用”，没有否定这些指标在所有表征比较中的价值。

## 2. 主角谱究竟比较什么

分别对视觉与文字 token 矩阵取前 $k$ 个 PCA 方向，得到正交基 $U_V,U_H\in\mathbb R^{d\times k}$。对小矩阵作 SVD：

$$U_V^TU_H=P\operatorname{diag}(\sigma_1,\ldots,\sigma_k)Q^T,\qquad1\ge\sigma_1\ge\cdots\ge\sigma_k\ge0.$$

$\sigma_i=\cos\theta_i$ 是子空间主角余弦；$U_VP_{:,i}$ 与 $U_HQ_{:,i}$ 是对应的原空间方向。它比较子空间是否重合，不需要将视觉 token 与文字 token 一一配对。PCA 的中心化、取样和有效秩仍是实现约定；本次无作者代码，未验证具体实现。论文默认 $k=30$，可复用实现还必须满足可用秩，不能对很短输入强取 30 维。（主角谱定义，PDF 第 3 页。）

最大主角余弦 $\sigma_1$ 只问“是否存在一对高度相似方向”。即使余下方向完全不同，只要共享一维，就可以达到 1。它也不是权重矩阵的奇异值；下面的 $s_i(W)$ 是另一个对象。

## 3. 共享 down-projection 怎样产生共同方向

列向量约定下，MLP 输出可写为 $W_{out}h$，$W_{out}\in\mathbb R^{d\times d_h}$。若权重 SVD 为 $U\Sigma R^T$，则

$$W_{out}h=\sum_i s_i u_i(r_i^Th).$$

当第一项对两类输入都占优势时，两类输出会靠近 $u_1$。关键条件不仅是 $s_1>s_2$，还包括输入在 $r_1$ 方向有足够投影；若 $r_1^Th=0$，再大的 $s_1$ 也无作用。残差、归一化及后续算子也会改变最终状态。因此“权重谱各向异性”本身不是完整的全网因果证明。

原文的几何界假设输出子空间分别存在单位方向 $\tilde u_X,\tilde u_Y$，与共同方向 $u_1$ 的夹角不超过 $\theta_X,\theta_Y$。由夹角三角不等式和最大主角的变分定义，

$$\sigma_1(WX,WY)\ge|\tilde u_X^T\tilde u_Y|\ge\cos(\theta_X+\theta_Y).$$

最后一界在角和大于 $\pi/2$ 时仅为平凡的非正下界。它说明共同方向足以产生高相似度，**没有证明所有共享权重必然产生这个条件**。（PDF 第 6 页命题与证明附录。）

论文配套证据包括：固定 projector 表征与逐层文字状态的比较、旁路 MLP/attention 的干预、训练权重与随机初始化对照、主角基在 down-projection 主要左奇异子空间上的投影能量。旁路 MLP 的影响在所有 13 模型上更大，作者报告中位影响约为旁路 attention 的 3.5 倍。旁路会同时改变网络计算，所以这些是组合证据，不能把一个干预孤立解释为唯一原因。

## 4. PA gap 的用途与退化反例

论文定义

$$\Delta_{PA}^{(l)}=\sigma_1^{(l)}-\sigma_2^{(l)},\qquad
\overline\Delta_{PA}=\frac1L\sum_l\Delta_{PA}^{(l)}.$$

在作者观察到 $\sigma_1$ 保持较高的情况下：NOISE 的次要共同方向变弱，使 gap 变大；ORIG 保持多个共同方向，使 gap 较小。要解释 gap，必须连同前两项乃至全谱一起看。

**整理者反例：**谱 $(1,1,0)$ 和 $(0,0,0)$ 的 gap 都为 0，前者共享至少两维，后者完全不重合。因此“小 gap”没有无条件的语义对齐或视觉结构保证。一般的多维共享权重偏置也可能同时抬高前两项。

渐进干预使用 $V_\alpha=\alpha V_{ORIG}+(1-\alpha)Z$，$\alpha=0,0.1,\ldots,1$。噪声 $Z$ 逐 token 匹配原范数，但混合向量本身的范数并不自动恒定。作者在内侧 80% 层的聚合口径下，报告 PA gap 与任务准确率的平均绝对 Pearson 相关为 0.894，方向在 13 模型中一致。这里预期是 gap 随质量提升而减小，不能只报绝对相关而隐去符号。（原文渐进破坏实验。）

## 5. 保留结构，不等于保留任务信息

IRR 把结构与题目相关性拆开。作者报告跨模型中位 PA gap 为 ORIG 0.130、IRR 0.213、NOISE 0.324，几何结构依次减弱；但 IRR 的准确率中位数 36.1% 低于 NOISE 的 39.0%。IRR 相对 TEXT 的逐模型差值中位数为 -5.0 个百分点。不同聚合方式不可混用：两组中位数之差不必等于配对差值中位数。

这说明 PA gap 更适合诊断“结构化视觉信息怎样进入语言模型”，不能独立判断这些信息是否对当前问题有用。SHUF 在这组多选题上的小变化也不能推广成空间定位任务不需要位置。

作者还将等范数噪声分别注入主角基张成空间及其正交补，观察到前者影响更大。原文附录使用全部 $k$ 列 $\Pi=U_VP$；由于 $P$ 为方阵正交矩阵，

$$\Pi\Pi^T=U_VPP^TU_V^T=U_VU_V^T.$$

**整理者核查：**该 in-band 干预实际上覆盖整个视觉 top-$k$ PCA 子空间，不能仅凭这项实验断言是其中“跨模态最共享的少数方向”独有的因果效应。要隔离后者，需要仅选高余弦子集并加入匹配维度对照；本次没有开展该实验。

## 6. 怎样用于量化研究

下面是由本论文导出的诊断建议，而非作者已验证的量化方案：

1. 若量化前后 CKA 或 $\sigma_1$ 接近，先问共同方向是否掩盖了次要信息损失，查看完整主角谱。
2. 在相同题图、模板和 token 预算下，对比原图、噪声、无关图与文字基线；同时报告任务结果，区分结构保持与相关性保持。
3. 若提出 PA gap 加权的量化损失，仍需验证它与真实误差及任务目标的关系，并用消融排除指标被直接优化后失去诊断意义。

这与 [量化后的排序稳定性](quantization-ranking-stability.md) 共同提醒：内部几何、输出一致性和任务正确性是不同对象。不能因为几何指标有解释力，就替代 [模型质量评测](../implementation/model-quality-evaluation.md)。

本页已读正文与评测、干预、几何证明等必要附录，并回查核心主角定义和界的页图；未复算全部模型表格、未运行模型，也无公开实现核验。当前证据主要来自一个 MMBench 子集及所列 projector 型架构。

## 来源身份

- [The Alignment Illusion in Multimodal Large Language Models](https://arxiv.org/abs/2609.30210v1)，arXiv:2609.30210v1；2026-09-26 归档。PA-gap 退化反例和全基投影等价式为整理者分析。
