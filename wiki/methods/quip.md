---
title: QuIP：非相干处理与 LDLQ 二阶量化
type: method
tags:
  - ptq
  - weight-quantization
  - rotation
  - second-order
sources:
  - raw/papers/2026-09-21/quip/paper.pdf
  - raw/papers/2026-09-21/quip/source.eprint
updated: 2026-09-22
---

# QuIP：非相干处理与 LDLQ 二阶量化

QuIP 把两个问题结合起来：怎样沿输入相关性补偿舍入误差，以及怎样改变坐标，使这种补偿在极低位宽下更有效。它先对权重与输入二阶统计做随机正交处理，再用 LDLQ 顺序量化。对象是权重，主要考察 W2/W3/W4 配合浮点激活；“2 bit”描述主体权重编码，不是整数乘法类型，也不是模型无损保证。

## 1. 目标与非相干性分别约束什么

设线性层权重 $W\in\mathbb R^{m\times n}$，输入为列向量 $x$。论文的代理目标为

$$
\ell(\widehat W)=\mathbb E\|(\widehat W-W)x\|_2^2
=\operatorname{tr}((\widehat W-W)H(\widehat W-W)^\mathsf T),
\qquad H=\mathbb E[xx^\mathsf T].
$$

这里 $H$ 是校准输入的未中心化二阶矩；与 [二阶补偿页](../theory/layer-reconstruction-second-order-compensation.md) 对未归一化重构损失采用 $2XX^\mathsf T$ 的记法相差归一化和常数。它不是最终任务损失的 Hessian。

QuIP 的非相干性有两种定义（原文“Adaptive Rounding with Linear Feedback”）：

- 对 $H=Q\Lambda Q^\mathsf T$，存在特征向量矩阵满足 $|Q_{ij}|\le\mu_H/\sqrt n$。含义是特征方向不集中在少数坐标轴上；并非 $H$ 对角化，也并非输入通道独立。
- 对权重，$|W_{ij}|\le\mu_W\|W\|_F/\sqrt{mn}$。它约束最大元素相对整体能量的集中程度，使有限网格不必被少量极端元素撑大。

二者解释责任不同：前者进入误差补偿的谱界，后者帮助控制量化范围。随机旋转保留谱，却改变谱方向与量化坐标的对齐关系。

## 2. LDLQ 怎样把历史误差传给后续列

用与原文等价、统一三角方向的记法，取

$$H=TDT^\mathsf T,\qquad T=I+U,$$

其中 $T$ 为单位上三角矩阵，$U$ 为严格上三角矩阵，$D$ 为非负对角矩阵。逐列执行

$$
\widehat W_{:,k}=\mathcal Q\left(W_{:,k}+
(W_{:,<k}-\widehat W_{:,<k})U_{<k,k}\right).
$$

每次先把过去的舍入误差按 $U$ 投射到当前列，再量化当前的修正值；各输出行共享相同反馈矩阵。令 $E=\widehat W-W$，$\eta$ 是量化器对“已修正输入”产生的局部误差，则

$$E=\eta T^{-1},\qquad \ell=\operatorname{tr}(\eta D\eta^\mathsf T).$$

这是理解 LDL 分解作用的关键：相关的原始二次型变成以 $D$ 加权的局部舍入误差。原文“Equivalence to OPTQ”证明，在对应量化顺序、网格与处理条件下，LDLQ 与 OPTQ/GPTQ 的误差反馈等价；QuIP 的新增点不能概括为又发明了一次 GPTQ 补偿。

## 3. 理论保证支持到哪里

对**无界整数网格**和只依赖 $H$ 的固定线性反馈族，论文证明 LDLQ 的最坏情形及指定均匀分布平均误差最优。这不等于每个给定 $W$ 上的全局离散最优。

在 $\mu_H$ 非相干条件下，关键谱界为

$$
\operatorname{tr}D\le\frac{\mu_H^2}{n}
\big(\operatorname{tr}H^{1/2}\big)^2.
$$

若 $H$ 的秩为 $k$，由 Cauchy–Schwarz 可进一步得到 $\operatorname{tr}D\le(\mu_H^2 k/n)\operatorname{tr}H$。它说明低秩或谱集中与非相干性共同创造收益；仅有低秩并不够。教学反例是对角 $H$：此时 $D=H$，没有可用于跨坐标补偿的相关结构。

实际低比特量化会裁剪到有限整数范围。原文“Finite-Grid Quantization”明确给出裁剪后 LDLQ 不再保持上述最优性的反例，并为受约束反馈与随机舍入设计另一个理论方案；实验使用的普通 QuIP 不能直接继承那个方案的全部有限网格保证。

## 4. 离线流程与推理变换

原文“Incoherence Processing”、算法 1–3 的主线如下：

1. 从校准输入估计 $H$ 并加阻尼。先做对角平衡 $W\leftarrow WA$、$H\leftarrow A^{-1}HA^{-1}$，其中 $A_{ii}=(H_{ii}/(W^\mathsf TW)_{ii})^{1/4}$；这是额外启发式，退化通道需要数值处理。
2. 采样随机正交 $U_o,V_o$，得到 $\widetilde W=U_oWA V_o^\mathsf T$ 与相应 $\widetilde H$。原始 QuIP 用两个较小正交矩阵的 Kronecker 乘积，并加入随机置换。
3. 以 $s=\rho\|\widetilde W\|_F/\sqrt{mn}$ 定范围，映射到 $[0,2^b-1]$，执行带裁剪的 LDLQ。可选贪心坐标更新继续降低代理误差；不能把这个选项与所有结果混同。
4. 保存压缩权重以及恢复所需的尺度和变换信息。推理的数学形式为 $y\approx U_o^\mathsf T\widehat{\widetilde W}(V_oA^{-1}x)$，这里 $\widehat{\widetilde W}$ 已恢复为实际数值单位。

随机种子可以减少变换的存储，却不会消除执行成本。两个均衡 Kronecker 因子的向量变换约需 $O(n\sqrt n)$，仍须与权重解码、输入变换和输出变换一起实现。[正交旋转页](../theory/orthogonal-rotation-and-hadamard-quantization.md) 解释浮点等价与量化误差为何能同时成立；[QuIP#](quip-sharp.md) 进一步改变变换与码本以降低这些代价。

## 5. 实验怎样支持机制

QuIP v2 的 OPT-30B 消融使用 C4 的 128 段、每段 2048 token 校准；逐块推进时用量化前缀产生后续输入。在同一 WikiText2 评测中，FP16 为 9.56；2 bit OPTQ 为 71.70，QuIP 为 11.48；最近舍入从无非相干处理的 41547.8 降至带处理的 12.04。该表支持“坐标处理在此配置下是主要增量”，也显示处理后 LDLQ 仍比最近舍入更好。（Experiments，OPT-30B 各种量化与处理组合表；这里 2048 指校准长度。）

代价不能省略：作者在 A6000、OPT-66B、batch 1、生成长度 128 的实现中报告 QuIP 81 ms/token、OPTQ 53 ms/token。表头虽写 throughput，单位表示每 token 时间；这组结果说明早期正交处理实现有明显开销，不能用更低位宽直接宣称更快。以上均为论文报告，本页没有运行模型或 GPU kernel。

## 来源身份

[QuIP: 2-Bit Quantization of Large Language Models With Guarantees](https://arxiv.org/abs/2307.13304v2)，arXiv:2307.13304v2。使用正文、算法与相关理论/实验补充材料；公式依据归档 TeX 核对，本文统一了三角分解和损失尺度的记法。
