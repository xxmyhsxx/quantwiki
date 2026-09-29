---
title: 曲率加权量化误差：Fisher、双侧度量与量化集合
type: concept
tags:
  - sensitivity
  - second-order
  - quantization-error
sources:
  - raw/papers/2026-09-23/rgsq/paper.pdf
  - raw/papers/2026-09-26/casa-cross-layer-sensitivity/paper.pdf
  - raw/papers/2026-09-22/hawq-v2/paper.pdf
  - raw/papers/2026-09-26/quantization-price-prediction/paper.pdf
updated: 2026-09-28
---

# 曲率加权量化误差：Fisher、双侧度量与量化集合

相同大小的量化扰动可能有不同任务代价。曲率度量为不同方向赋予不同权重，但必须先区分被近似的损失、统计来源、坐标约定及允许的量化候选。本页以 RGSQ v1 §III–IV 为实例展开，并修正其列向量化因子顺序；推导中额外限制与反例属于整理者分析。

## 1. 任务 Hessian、模型 Fisher 与经验梯度外积

一般任务损失满足局部展开

$$L(\theta+e)-L(\theta)=g^\top e+\tfrac12e^\top H e+o(\|e\|^2).$$

只有在相应驻点、期望消去或其他有依据的条件下，才能忽略一阶项。$H$ 也不必半正定；实际任务准确率还是离散指标，不是这个局部光滑展开本身。

另一种对象是以当前模型分布为参考的前向 KL：

$$D_{KL}(p_\theta\|p_{\theta+e})=\tfrac12e^\top F e+o(\|e\|^2),\qquad
F=\mathbb E_{u\sim p_\theta}[\nabla\log p_\theta(u)\nabla\log p_\theta(u)^\top].$$

在可交换求导与积分等正则条件下，score 期望为零，Fisher 等于负的期望对数似然 Hessian。这不要求当前模型在真实标签数据上是损失驻点，因为展开对象不同。

校准数据和真实/生成标签的梯度外积平均通常是经验 Fisher 代理；若标签并非从当前模型分布采样，就不能仅凭外积形式把它与上式 Fisher 恒等。RGSQ 的模态统计还做了因子化与加权融合，必须保留这些近似层次。

## 2. 线性层的双侧度量来自哪里

对单 token 列向量 $x$ 和输出梯度 $g$，有 $\nabla_W L=gx^\top$。列优先展开得到 $\operatorname{vec}(gx^\top)=x\otimes g$，单样本外积为 $(xx^\top)\otimes(gg^\top)$。用两个期望的乘积近似联合期望，得到

$$F_W\approx A\otimes S,\quad A=\mathbb E[xx^\top],\quad S=\mathbb E[gg^\top].$$

它忽略输入与输出梯度二阶乘积之间的统计依赖，不是一般恒等式。跨 token 或跨层项也需另行建模。

对 $E\in\mathbb R^{d_o\times d_i}$，

$$\operatorname{vec}_{col}(E)^\top(A\otimes S)\operatorname{vec}_{col}(E)
=\operatorname{tr}(S E A E^\top)
=\|S^{1/2}EA^{1/2}\|_F^2.$$

行优先展开对应 $S\otimes A$，两种记号都可用，但不能混用。RGSQ 式 14 的中间乘积遵循列优先，末行却交换了因子，本页依恒等式修正。Kronecker 与 reshape 的基本关系见 [可逆变换与 Kronecker 乘积](../fundamentals/mathematics/invertible-transforms-and-kronecker-products.md)。

当 $S=I$、$A=XX^\top/N$，恢复层输出 MSE。因此欧氏输出重构并非权重空间所有方向等价；[二阶重构](layer-reconstruction-second-order-compensation.md) 中的输入 Gram 矩阵已经表达方向差异。

## 3. 阻尼与模态融合都会改变度量

因子是未中心化二阶矩，均值项不能随意删去。PSD 因子可以有零空间，零代价不表示该方向在真实任务中绝对安全，可能只是校准未覆盖。

对 $A_\delta=A+\delta I$、$S_\delta=S+\delta I$，代价变为

$$\operatorname{tr}(S E A E^\top)
+\delta\operatorname{tr}(E A E^\top)
+\delta\operatorname{tr}(S E E^\top)
+\delta^2\|E\|_F^2.$$

这说明阻尼不只提供可逆性，也明确增加了误差惩罚。应记录绝对阻尼、相对特征值下界和对角近似分别做了什么。

RGSQ 先用模态系数分别混合 A/S，再做乘积。它产生跨模态因子配对，通常不同于 $\sum_m\pi_m(A_m\otimes S_m)$。方法不同不等于其中一个自动错误，但必须知道真正求解的是哪个代理。

## 4. 欧氏化必须同时搬运约束

若 A/S 正定，线性映射 $T(W)=S^{1/2}WA^{1/2}$ 可逆；目标范数可以欧氏化，但量化约束也必须变为 $T(\mathcal Q)$。对 $T(W)$ 直接应用原来的整数网格，通常求解了另一个问题。

这与低秩近似有关键区别：可逆左右变换保持矩阵秩，所以 [加权低秩近似](../fundamentals/mathematics/invertible-transforms-and-kronecker-products.md#6-激活加权的低秩近似) 可以保留秩约束；一般混合矩阵不保持逐元素量化网格、分组共享尺度或指定打包格式。

因此检查“可直接套用旧量化器”时，应分别问：目标是否等价、可行集合是否等价、求解器是否解该目标、映回结果是否可部署。RGSQ 的具体反例及论文/代码差异见 [RGSQ](../methods/rgsq.md)。

## 5. 标量敏感度为何存在不可消去的失真

若双侧因子正定，$Q(E)/\|E\|_F^2$ 的极值分别是 $m=\lambda_{min}(A)\lambda_{min}(S)$ 和 $M=\lambda_{max}(A)\lambda_{max}(S)$。用任何正标量 $\alpha\|E\|_F^2$ 代替 $Q$，对全部误差方向的最坏双向乘性失真为 $\max(M/\alpha,\alpha/m)$；最优标量 $\sqrt{mM}$ 也只能把它降至 $\sqrt{\kappa(A)\kappa(S)}$。（CASA Theorem 3.1。）

[HAWQ-V2](../methods/hawq-v2.md) 的平均 trace 在等方向能量或各向同性扰动假设下有解释，但与这个最坏方向最优标量解决的不是同一问题。条件数大表明存在坏方向，不证明有限量化候选一定实现它。对角因子保留坐标异质性，也仍舍弃旋转后的相关方向。[CASA](../methods/casa.md) 进一步说明这种区别如何影响有限预算分配，以及跨层有符号项如何使独立求和失效。

## 6. 从误差度量到配置选择

[Quantization Price](../methods/quantization-price-prediction.md) 把位宽、粒度和变换作为候选误差生成器，以 forward KL 的局部二阶价格比较。其平均输出曲率与平均输入矩相乘仍需因子分离近似；进一步用 trace 乘总误差能量，还会丢掉输出误差方向。共同判断标准是：理论目标、统计代理、实际评分和最终测试是否仍对应同一个对象。

## 来源身份

| 来源 | 版本 | 范围 |
| --- | --- | --- |
| [RGSQ: Riemannian Geometry-Sensitive Quantization for Large Vision-Language Models](https://arxiv.org/abs/2609.25492v1) | arXiv:2609.25492v1 | §III–IV 的统计与目标；因子顺序、阻尼展开和约束搬运为本文核对与推导 |
| [Beyond Scalar Sensitivity](https://arxiv.org/abs/2609.25916v1) | arXiv:2609.25916v1 | Theorem 3.1，标量最坏乘性失真及其量词 |
| [HAWQ-V2](https://arxiv.org/abs/1911.03852v1) | arXiv:1911.03852v1 | §2.1，平均trace的方向假设 |
| [Predicting Quantization Price](https://arxiv.org/abs/2609.28270v1) | arXiv:2609.28270v1 | §3、附录B.1/D.2，因子分离与实际评分 |
