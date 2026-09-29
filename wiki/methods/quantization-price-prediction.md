---
title: Quantization Price：用输出分布代价选择 PTQ 配置
type: method
tags:
  - sensitivity
  - mixed-precision
  - calibration
sources:
  - raw/papers/2026-09-26/quantization-price-prediction/paper.pdf
updated: 2026-09-28
---

# Quantization Price：用输出分布代价选择 PTQ 配置

这篇论文研究的是配置选择：将位宽、粒度、量化格式和等价变换视为不同的误差生成器，由浮点模型的下游曲率给误差定价，再在预算内选择配置。它不是新的整数舍入算法，也不要求对每个完整模型都做一次端到端试验。

本页依据 arXiv:2609.28270v1，覆盖 §2–4、附录 B 的推导、C–D 的实验与实现说明。与 [CASA](casa.md) 相比，其选择变量不限于位宽，理论起点是浮点到量化模型的 forward KL；但实际实验进一步标量化，不能因此理解为已经完整测得所有候选的方向代价。本次无模型复现、无本地作者代码。

## 1. 从配置到误差，再到价格

层参考计算为 $z=Wx$。候选 $\alpha$ 必须给出四件事：浮点等价的坐标 $(W(\alpha),x(\alpha))$、有限格式替代 $\widehat W(\alpha)$、合法后端及成本 $\kappa(\alpha)$。有

$$W(\alpha)x(\alpha)=Wx,\qquad
\Delta(\alpha)=\widehat W(\alpha)-W(\alpha),\quad e_z=\Delta(\alpha)x(\alpha).$$

令 $C(\alpha)=\mathbb E[x(\alpha)x(\alpha)^\top]$，则误差未中心化二阶矩为

$$\Sigma(\alpha)=\Delta(\alpha)C(\alpha)\Delta(\alpha)^\top.$$

原文称其 covariance，但若误差均值非零，它是二阶矩而非中心化协方差；二阶损失需要的正是这个量，不能无依据减去均值项。候选可以先生成局部量化权重，无需组装全部层的组合；因此“部署前预测”不等于完全不生成候选误差。

## 2. 为什么 forward KL 没有一阶项

目标是参考模型分布 $p_0$ 到候选分布 $p_\alpha$ 的

$$J(\alpha)=\mathbb E_x D_{KL}(p_0(\cdot|x)\|p_\alpha(\cdot|x)).$$

在参考点，$-\sum_y p_0(y|x)\nabla\log p_0(y|x)=-\nabla\sum_y p_0(y|x)=0$。因此局部展开从二次项开始；它不要求模型在真实标签训练集上达到驻点。用硬标签 NLL 替换这个目标时，不能沿用同一零梯度论证。（Theorem 3.1、附录 B.1。）

令每个输入的输出曲率为 $H_l(x)$。单层精确二次项首先是

$$\tfrac12\mathbb E_x\operatorname{tr}\big(H_l(x)\Delta_l x_lx_l^\top\Delta_l^\top\big).$$

**附录式 23→24 还做了因子分离**：以平均 $H_l=\mathbb E H_l(x)$ 替代样本相关曲率，得到

$$\rho_l(\alpha_l)=\tfrac12\operatorname{tr}(H_l\Sigma_l(\alpha_l)),\qquad \rho=\sum_l\rho_l.$$

一般 $\mathbb E[H(x)\otimes xx^\top]\ne\mathbb EH(x)\otimes\mathbb E[xx^\top]$。所以它不仅删掉跨层项，也近似了层内输入与曲率的相关性。理论将这些偏差一起纳入 $H_\theta-\widetilde H_\theta$，不能把 $\rho$ 说成无条件精确的二阶损失。

对堆叠权重扰动 $\delta$，论文的余项界为

$$|J-\rho|\le\tfrac12\|H_\theta-\widetilde H_\theta\|_2\|\delta\|_2^2
+\tfrac{M}{6}\|\delta\|_2^3,$$

其中 $M$ 控制沿扰动路径的三阶导数。它解释近似何时可能好，但实际选择器没有计算这些完整矩阵与 $M$，因此不是逐候选已认证的误差条。

**排序还需要间隔。** 若候选 A/B 的余项界分别为 $\epsilon_A,\epsilon_B$，只有 $\rho_B-\rho_A>\epsilon_A+\epsilon_B$ 才能据此保证 $J_A<J_B$。这是由三角不等式得到的整理者推论。预测相关性高不保证相近候选的局部排名，更不直接保证任务准确率。

## 3. 旧指标是怎样的简化

在 $\rho=\tfrac12\operatorname{tr}(H\Delta C\Delta^\top)$ 中置 $H=I$，恢复 $\|\Delta X\|_F^2/(2N)$ 层输出重构；再置 $C=\operatorname{diag}(C)$，得到 $\tfrac12\sum_j C_{jj}\|\Delta_{:,j}\|^2$ 输入通道加权。它们分别丢掉输出方向价格与输入相关项。（Remark 3.2、附录 B.9。）

这种统一记法便于比较，但不能把“现有方法是简化”直接变成“本文实际实现一定更精确”：实现也有自己的统计与标量化。共同数学基础见 [曲率加权误差](../theory/curvature-weighted-quantization-error.md)。

对零均值、互不相关的随机舍入模型，若 $\mathbb E\Delta_{ij}^2\le V_{ij}$，则

$$\mathbb E\rho\le\tfrac12\operatorname{diag}(H)^\top V\operatorname{diag}(C).$$

均匀量化在步长 $\delta_b=2r/(2^b-1)$、理想均匀舍入噪声下用 $V_{ij}=\delta_b^2/12$，价格呈近似 $4^{-b}$ 衰减。这里 $\delta^2/12$ 不是任意确定性 RTN 的最坏误差界；无 clipping 的逐元素最坏平方误差是 $\delta^2/4$。裁剪、偏置、块码本相关误差都需要重新估计矩，不能强行套独立均匀噪声。（式 10–11、附录 B.2。）

## 4. 白化定理优化的是输入度量

等价变换 $\widetilde x=Px,\widetilde W=WP^{-1}$ 给出

$$\rho(P,b)=\tfrac12\operatorname{tr}\big(H\widetilde\Delta_b(P)(PCP^\top)\widetilde\Delta_b(P)^\top\big),\quad
\widetilde\Delta_b(P)=Q_b(WP^{-1})-WP^{-1}.$$

Theorem 3.3 在 $C\succ0,|\det P|=1$ 下证明

$$\min_P\lambda_{max}(PCP^\top)=(\det C)^{1/n}=\lambda_g,$$

且等号要求全部特征值相等，即 $PCP^\top=\lambda_gI$。证明只需“最大特征值不小于特征值几何均值”，以及行列式在该约束下固定。由 $C=U\Lambda U^\top$ 得到 $P=\sqrt{\lambda_g}Q\Lambda^{-1/2}U^\top$、$Q$ 正交。

它没有证明同一 P 最小化真实量化价格，因为 P 同时改变权重、网格残差和执行成本。对角缩放只能改变方差，正缩放保持标准化相关系数；允许负缩放时相关符号会翻转、绝对值仍不变。正交旋转保持 C 的谱，不能白化任意各向异性 C，但能改变有限码本误差和离群值。

附录 B.5 的体积、trace、Frobenius 归一化分别给出白化尺度平方为特征值的几何、算术、调和平均。对随权重齐次缩放的量化器，$P$ 的统一尺度放大可被权重误差的缩小抵消，不能只看 $PCP^\top$ 越小就认为量化越好。

## 5. trace 价格的成立条件与实际实现差别

若候选输入 $C=\sigma^2I$，且输出侧残差矩 $\mathbb E[\Delta\Delta^\top]=\omega I+A$，有

$$\rho=\tfrac12\sigma^2\omega\operatorname{tr}H
+\tfrac12\sigma^2\operatorname{tr}(HA),\quad
|\text{后项}|\le\tfrac12\sigma^2\|H\|_2\|A\|_*.$$

候选噪声接近各向同性，为只保留 $\operatorname{tr}H$ 提供充分依据；其他情形需直接检查省略项。“H 接近对角”不等于其对角元素相等。正文式 17–18 用 $\widehat\sigma^2=\operatorname{tr}C/n$、$\widehat\omega=\|\Delta\|_F^2/m$、$\widehat\tau=\operatorname{tr}\widehat H$，缓存 $\widehat\rho=\widehat\sigma^2\widehat\omega\widehat\tau/2$。

附录 D.2 给出的实测流程更具体，也存在需要保留的差别：

- Llama 的 $\widehat\tau=\mathbb E\|\nabla_z\ell\|^2$ 由前反向求得，**没有用 Hessian-vector probe**。正文建议 Hutchinson 并非这里的实际执行流程。
- 若这些梯度来自真实 token 标签，它是经验梯度外积 trace；与参考分布下的 Fisher/GN 相等需要标签采样/期望条件。原文没有足够细节确认该一致性。
- 变换和粒度候选用 $\widehat\tau\,\mathrm{MSE}(z,\widehat z)/2$，至多 512 个监督 token/层。这保留实际输入作用后的总误差能量，通常不同于正文两个归一化 trace 的乘积；输入与残差近各向同性且归约一致时才对应。
- 对单层候选，固定 $\widehat\tau>0$ 不改变输出 MSE 排名；下游曲率 trace 的额外作用主要出现在跨层预算权衡。不能仅因评分乘了 trace 就声称单层候选已按输出方向区分。

## 6. 预算、协议与证据

配置集合应先排除后端不支持的格式。论文把成本写为 bits+metadata+kernel；实际核算必须统一单位或设不同约束，字节与毫秒不能直接相加。逐层成本之和还假设开销可分，融合和混合格式切换可能破坏它。

实验校准为 WikiText-2 validation 的 32×512，seed 0；KL 对齐为 128×1024。位宽分配选择 {2,3,4}，平均预算 $3.0\pm0.005$，group 128，MILP；共享 GPTQ、true-sequential、act-order、无 MSE search。变换固定 W4/g128，从 identity、AWQ/SmoothQuant-style scaling、Hadamard、归一化对角白化选择；粒度固定 W4，选择 tensor/channel、g32/64/128、64×64 block，并计 FP32 scale。

表 1/附录表 2–3 的结果均为作者报告，选择时间只覆盖核心决策，排除后续量化和评测：

| 设置 | 本文选择时间 / PPL / 五任务均值 | 参照 | 解读 |
| --- | --- | --- | --- |
| Llama-3.2-1B 位宽 | 579 s / 12.22 / 51.24 | AMQ 11635 s / 11.05 / 50.13 | 更快且均分较高，但 PPL 较差 |
| Llama-3.1-8B 位宽 | 5967 s / 10.87 / 49.34 | AMQ 23337 s / 9.31 / 54.84 | 搜索便宜，两个质量指标均较差 |
| Llama-3.2-3B 粒度 | 372 s / 8.46 / 58.13 | 固定 g128：8.81 / 58.38 | PPL 改善，均分下降 |

五任务为 BoolQ、TruthfulQA MC2、PIQA、WinoGrande、WiC。图 1 的 log-scale 相关性在 OPT-125M/Qwen3-0.6B 上，完整价格为 0.9470/0.9249，trace 价格为 0.9483/0.9438；属于受控单/双层候选实验，不等于大模型全组合的排序认证。

变换与粒度实验保存的是 dense fake-quant Hugging Face checkpoint，**不是已打包部署产物**。因此支持配置质量与选择成本，不证明真实低比特延迟、显存或支持面。推理验证需重新连接 [部署后端](../implementation/quantized-llm-deployment-backends.md) 和 [质量评测](../implementation/model-quality-evaluation.md)。

## 7. 附录推广不应直接变成主结论

附录 B.10 的 W/A 分支先忽略乘积误差 $\Delta W\epsilon_x$，再用条件零均值去掉交叉矩；在白化和固定各向同性噪声强度下得到 $a\sigma^2+b/\sigma^2$ 的 AM–GM 平衡。这些强度会随量化尺度变化，不能无条件使用其闭式最优尺度。

该附录称激活一阶项需要零均值才能消失；但若沿用每个输入在参考点的 forward-KL 目标，关于中间激活的梯度同样为零，score cancellation 并不限于权重。条件零均值仍可用于交叉矩简化，却不是该 KL 一阶项为零的必要条件。这是按其目标定义得到的核对，不能直接改用一般监督损失的结论。

附录连续位宽式 55 使用 $c_l2^{-b_l}$，而前述均匀平方误差约为 $4^{-b_l}$。二者是不同失真模型，前者的 log 分配系数为 1，后者为 1/2；加入参数量、有限档位和边界后也需重解预算，不能直接当作同一个通用位宽公式。

## 来源身份

| 来源 | 版本 | 范围 |
| --- | --- | --- |
| [Predicting Quantization Price for Selecting PTQ Configurations Before Deployment](https://arxiv.org/abs/2609.28270v1) | arXiv:2609.28270v1 | §2–4、Theorem 3.1/3.3、表 1–3、附录 B/C/D；B.1 与 D.2 的关键实现说明回查页图 |
