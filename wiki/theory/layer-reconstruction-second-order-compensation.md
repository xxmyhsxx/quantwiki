---
title: 层输出重构与二阶误差补偿
type: concept
tags:
  - reconstruction
  - second-order
  - quantization-error
sources:
  - raw/papers/2026-09-22/efficientqat/paper.pdf
  - raw/repositories/2026-09-22/efficientqat/source/quantize/block_ap.py
  - raw/papers/2026-09-21/quip/source.eprint
  - raw/papers/2026-09-21/quip-sharp/source.eprint
  - raw/papers/2026-09-21/aqlm/source.eprint
  - raw/papers/2026-09-21/spqr/source.eprint
  - raw/papers/2026-09-21/squeezellm/source.eprint
  - raw/papers/2026-09-21/qig/paper.pdf
  - raw/papers/2026-09-21/flatquant/paper.pdf
  - raw/repositories/2026-09-21/flatquant/source/flatquant/train_utils.py
  - raw/papers/2026-09-21/omniquant/paper.pdf
  - raw/papers/2026-09-21/q-vlm/paper.pdf
  - raw/papers/2026-09-21/gptaq/paper.pdf
  - raw/papers/2026-09-21/vlmq/paper.pdf
  - raw/papers/2026-09-21/quantization-white-paper/paper.pdf
  - raw/papers/2026-09-21/gptq/paper.pdf
updated: 2026-09-22
---

# 层输出重构与二阶误差补偿

量化改变权重，但我们关心的是这种改变怎样影响计算结果。层输出重构用一批代表性输入衡量误差；二阶补偿进一步利用输入通道之间的关系，在固定一个量化值后调整其他权重，减少输出变化。

本页从 GPTQ v2 §3–4、式 (1)–(5)展开，补足其中压缩表达的线性代数。推导与教学例子由本项目整理，不是模型复现实验，也不是对所有重构算法的综述。

## 1. 输出误差如何引入输入统计

令 $W\in\mathbb R^{r\times d}$、$X\in\mathbb R^{d\times n}$，其中 $n$ 是汇集的校准输入向量数。线性层输出为 $WX$，量化后的近似权重为 $\widehat W$；行列与输入通道的约定见 [线性层与输入通道](../fundamentals/operators/linear-layer-input-channel.md)。采用 Frobenius 范数明确论文式 (1) 的矩阵平方误差：

$$
L(\widehat W)=\|(W-\widehat W)X\|_F^2
=\sum_{o=1}^{r}\delta_o^{\mathsf T}XX^{\mathsf T}\delta_o,
\qquad \delta_o=(\widehat W_{o,:}-W_{o,:})^{\mathsf T}.
$$

因此各输出行可以独立优化，同时共享 $G=XX^{\mathsf T}$。$G_{ij}=\sum_tX_{it}X_{jt}$ 是未中心化的二阶乘积，不应无条件称为协方差；$G_{ii}$ 衡量该输入通道在校准数据上的平方幅度，非对角项反映不同通道共同变化的影响。

若用权重 MSE，即 $\|W-\widehat W\|_F^2$，等于忽略这些输入统计。只有在 $XX^{\mathsf T}$ 与单位阵成比例等特殊条件下，两种目标才等价。一个权重偏移大，并不必然意味着输出误差大；不同通道造成的输出误差还可能互相抵消。

## 2. Hessian 在这里是什么

对一行权重偏移 $\delta$，

$$
L(\delta)=\delta^{\mathsf T}G\delta
=\tfrac12\delta^{\mathsf T}H\delta,\qquad H=2XX^{\mathsf T}.
$$

Hessian 是函数对参数的二阶导数矩阵；本例的梯度是 $2G\delta$，再求导就是 $2G$。它描述不同方向上误差增长的曲率。**对于固定输入的线性层平方重构目标，这是精确的二次形式。** 它没有计算整个网络交叉熵或下游任务损失的 Hessian。用局部重构替代最终模型质量、有限校准样本替代真实输入分布，才引入了重要的代理与采样限制。

对任意向量 $v$，$v^{\mathsf T}Hv=2\|X^{\mathsf T}v\|_2^2\ge0$，所以 $H$ 半正定；输入通道相关或缺乏覆盖时可能不可逆。GPTQ 使用阻尼 $H_\lambda=H+\lambda I$，其中 $\lambda>0$ 时得到正定矩阵。阻尼同时相当于在局部目标增加 $\frac\lambda2\|\delta\|_2^2$，抑制过大的补偿，因而也改变了优化问题。论文使用平均对角值的 1%，不是一个与数据尺度无关的常数。

Cholesky 分解把正定矩阵写成三角因子的乘积。GPTQ 对逆 Hessian 使用上三角因子 $R$，满足 $H_\lambda^{-1}=R^{\mathsf T}R$，以避免反复更新逆矩阵时积累数值误差。不能把 $R$ 本身当作 $H^{-1}$ 代入原补偿式。

## 3. 固定一个权重后，为什么其他权重可以补偿

设当前仍可调整的坐标集合为 $F$。把其中坐标 $q$ 固定到量化值，要求增量 $\delta_q=a$，其中 $a=Q(w_q)-w_q$。这里 $Q$ 返回量化后反量化的网格值，编码、步长与分组的定义见 [均匀量化与分组](../fundamentals/quantization/uniform-quantization-and-groups.md)。在当前自由变量位于相应受约束连续最优点的条件下，增加的局部误差为二次形式。考虑

$$
\min_{\delta_F}\tfrac12\delta_F^{\mathsf T}H_F\delta_F,
\qquad e_q^{\mathsf T}\delta_F=a.
$$

$e_q$ 是选择第 $q$ 个自由坐标的单位向量。拉格朗日函数取

$$
\mathcal L=\tfrac12\delta_F^{\mathsf T}H_F\delta_F
+\mu(e_q^{\mathsf T}\delta_F-a).
$$

令梯度为零：$H_F\delta_F+\mu e_q=0$。再代入约束，得到

$$
\delta_F=\frac{a}{[H_F^{-1}]_{qq}}(H_F^{-1})_{:,q},
\qquad \Delta L=\frac{a^2}{2[H_F^{-1}]_{qq}}.
$$

这解释了 GPTQ 式 (2) 的补偿方向与分母。OBQ 用 $a^2/[H_F^{-1}]_{qq}$ 选择下一个坐标，省略共同的 $1/2$ 不影响排序。这里的“最优”仅指这一步的连续受约束问题；其他权重随后也必须量化，逐步选择并不保证全体离散权重的全局最优。

量化 $q$ 后要删除对应自由坐标。若 $A=H_F^{-1}$，剩余 Hessian 的逆可以由

$$
(H_{F\setminus\{q\}})^{-1}
=\left(A-\frac{A_{:,q}A_{q,:}}{A_{qq}}\right)_{-q,-q}
$$

得到。先做秩一修正再删行列；直接删 $A$ 的行列一般不等于剩余 Hessian 的逆。原论文式 (3) 最后下标印为 $-p$，上下文没有定义 $p$，本页按其“删除 $q$”的文字说明展开。

## 4. 一个能看见补偿效果的例子

设一行权重 $w=(0.4,0.4)^{\mathsf T}$，网格为整数，且

$$
G=\begin{bmatrix}1&0.5\\0.5&1\end{bmatrix},\qquad
H^{-1}=\begin{bmatrix}2/3&-1/3\\-1/3&2/3\end{bmatrix}.
$$

这可以由 $X=\begin{bmatrix}1&0\\0.5&\sqrt{0.75}\end{bmatrix}$ 得到。先将第一项舍入到 0，$a=-0.4$，补偿给出 $\delta=(-0.4,0.2)$，当前权重变为 $(0,0.6)$。第二项再舍入得到最终 $(0,1)$。

| 策略 | 最终权重 | 权重平方误差 | 层输出平方误差 $\delta^{\mathsf T}G\delta$ |
|---|---|---:|---:|
| 独立最近舍入 | $(0,0)$ | 0.32 | 0.48 |
| 依次补偿再舍入 | $(0,1)$ | 0.52 | 0.28 |

补偿方案的权重 MSE 更大，输出误差却更小。这不是“补偿总能更好”的证明，而是输入相关性为何值得建模的具体例子。如果另一分布的通道相关性改变，这种抵消也可能不再成立。

## 5. 从机制回到方法与验证

[GPTQ](../methods/gptq.md)让所有输出行采用相同列顺序，共享 Hessian 处理，并以列块批量更新提高效率。[AWQ](../methods/awq.md)则通过等价缩放和裁剪选择来改善权重量化；两者都使用输入信息，但可调整的变量与搜索流程不同。

解释或复现时需要分别检查：校准输入覆盖、量化网格、阻尼与数值精度、坐标顺序、补偿后更新输入的范围，以及最终模型质量。把某层重构误差降低直接写成“下游任务一定提高”，超出了这个局部目标能支持的结论。

## 6. 偏置校正与软舍入解决的不同问题

量化白皮书 v1（W）§3.3–3.4 展示了两种与本页相关的思路。本节依据白皮书综述；软舍入一侧的原始来源已单独研读，完整推导、消融与适用条件见 [AdaRound](../methods/adaround.md)，其后续把重构粒度从单层扩展到基本模块的工作见 [BRECQ](../methods/brecq.md)。下面保留白皮书的综述视角与它给出的边界，两处不重复维护整套推导。

**偏置校正。** 固定权重量化误差为 $E_W=\widehat W-W$，输入均值为 $\mu_x$。输出误差均值是 $E_W\mu_x$，因此可取

$$
b'=b-E_W\mu_x.
$$

校正后误差为 $E_W(x-\mu_x)$，在这一个输入分布上均值为零；误差方差与后续非线性影响没有因此消失。均值可以用校准样本估计，也可在附加分布假设下解析估计。（W 式 (25)–(28)。）

若 $U\sim\mathcal N(\mu,\sigma^2)$、$\sigma>0$，则

$$
\mathbb E[\operatorname{ReLU}(U)]
=\sigma\phi(\mu/\sigma)+\mu\Phi(\mu/\sigma),
$$

其中 $\phi,\Phi$ 为标准正态密度与分布函数；$\sigma=0$ 时直接取 $\max(\mu,0)$。正态假设不由 BN 自动保证，标准差必须非负。W §3.3 的 gamma/beta 文字说明有次序问题，不能机械沿用；符号条件见 [BN 统计估计](calibration-and-range-selection.md#4-没有样本时统计假设能替代什么)。W 表 4 在 MobileNetV2 中报告偏置校正改善，但这不是全模型输出完全恢复的证据。

**软舍入。** 独立最近舍入最小化每个权重到网格的距离，不一定最小化层输出误差。W §3.4 用可学习 $V$ 与 $h(V)\in[0,1]$，将相邻整数选择放松为

$$
\widetilde W=\Delta\operatorname{clip}\big(\lfloor W/\Delta\rfloor+h(V),n,p\big),
$$

并优化局部重构误差加上推动 $h$ 靠近 0 或 1 的正则项，例如

$$
\|WX-\widetilde WX\|_F^2+
\lambda\sum_{ij}\big(1-|2h(V_{ij})-1|^\beta\big).
$$

优化中调整正则调度，最终固化舍入决策；白皮书进一步讨论比较 $f(WX)$ 与 $f(\widetilde W\widehat X)$，将非线性和前层量化输入纳入重构。此时不再是第 2 节固定输入的线性二次目标。（W 式 (29)–(35)、图 7。）

W 从任务损失 Taylor 展开转到局部重构使用了近似与代理：一阶项是否可以忽略、跨层耦合怎样舍弃、任务曲率如何近似，都不能由 $H=2XX^{\mathsf T}$ 自动证明。白皮书未充分展开原始 AdaRound 的全部假设，本轮保留这一理解边界，不将其视为任务 Hessian 的精确等式。

W 表 5 的 ResNet18 W4、其余设定按 §3.4，全部层最近舍入为 23.99，局部连续放松为 66.56，最终 AdaRound 配置为 68.60（5 次均值）；它支持舍入决策值得优化，不表示任意数据或模型都达到同等收益。梯度优化局部舍入仍可属于 [PTQ](../fundamentals/quantization/post-training-and-quantization-aware-training.md)；它与 GPTQ 顺序固定权重后解析补偿的变量和流程不同。AdaRound 原文把这组数字中的 66.56 与 68.60 分别对应到「逐层 MSE 目标」与「非对称重构加激活函数」两个设计选择上，具体见 [AdaRound 的非对称重构](../methods/adaround.md#4-松弛与正则)：非对称指教师输入 x 与量化前缀产生的输入 $\hat x$ 不同，不是量化网格的 zero-point。

## 7. 加权重构与非对称输入

本节非对称输入与残差补偿依据 **GPTAQ v3 §4.1、附录 A.1** 的局部研读；token 加权依据 **VLMQ v2 附录 C**。以下用统一符号推导两者的组合，不将 GPTAQ 的知识归为 VLMQ 原创。

前文在 token 位置上等权，并让两侧使用同一个输入。推广时应区分两个独立变化：给不同位置不同代价；允许量化路径输入 $U$ 与浮点参考输入 $F$ 不同。对一行权重 $w$，定义 $\delta=(\widehat w-w)^\mathsf T$、$r=w(F-U)$，令 $T=\operatorname{diag}(t_1,\ldots,t_N)$ 为非负位置因子，可写成

$$
J(\delta)=\|(\delta^\mathsf TU-r)T\|_2^2,
\quad A=UT^2U^\mathsf T,\quad c=UT^2r^\mathsf T,
\quad \nabla J=2A\delta-2c.
$$

这里用 $T$ 避免与前文输入 Gram 矩阵 $G$ 重名。$t_n$ 是平方误差前的因子，真正误差系数为 $t_n^2$；若要系数为 $\lambda_n$，应令 $t_n=\sqrt{\lambda_n}$。外部重要性因子不能不加区分地作为 Hessian 对角元素：token 因子通过 $UT^2U^\mathsf T$ 改变输入通道之间的二阶统计，两个矩阵所在的轴不同。

当 $A$ 正定时，令 $B=A^{-1}$、$v=Bc$。固定当前坐标 $\delta_q=a$ 的解为

$$
\delta=v+\frac{a-v_q}{B_{qq}}Be_q.
$$

若 $F=U$，$r=0$，就恢复带权的单步补偿；若进一步 $T=I$，恢复前文等权形式。相反，非对称输入带来非零线性项，仅更新 Hessian 而忽略 $c$ 不能求得这一目标的最优解。这里仍是单步、固定统计的连续问题，不能保证整个离散量化网络最优。

GPTAQ v3 §4.1 与附录 A.1 提供非对称框架，§4.2 又通过残差的通道分解等改进计算效率；[VLMQ](../methods/vlmq.md) 在此基础上用 token 梯度构造因子，并保留对基础算法和模型配置的限制。两者的高效执行不能仅用以上一个解析式代替。因子退化为零、有效输入秩不足与阻尼改变目标等数值条件，仍需与第 2 节一起检查。

## 8. 为什么还需要考虑多个层的联合重构

单层最优不保证后续输出最优。把张量展平记为状态向量，在局部小扰动近似下，后层误差可写为 $\delta x_{k+1}\approx J_k\delta x_k+e_{k+1}$，其中 $J_k$ 是浮点层在参考输入处的 Jacobian，$e_{k+1}$ 是后层自身引入的量化误差。前层误差经过 $J_k$ 变换后，可能放大或与后层误差抵消；这个教学展开解释了只看 $\|\delta x_k\|$ 为什么可能失去后续信息，并非任意大误差下的精确关系。

[Q-VLM](../methods/q-vlm.md) 式 1–3 用连续多层块的输出误差共同选择量化参数，并以熵分数决定联合搜索范围。这里改变的是优化边界，VLMQ 改变的是块内 token 误差权重，GPTQ 的算法列块则主要改变计算组织，三者不能混称同一分块方法。Q-VLM 的代理与代码边界见方法页；多层联合校准也不自动获得整个网络的全局最优。

[OmniQuant](../methods/omniquant.md) v3 算法 1 给出另一种明确的边界：每次优化一个完整 Transformer block 的量化参数，目标为 $\|F_i(W_i,x_i^{fp})-\widehat F_i(W_i,x_i^q;\theta_i)\|_F^2$。$x_i^{fp}$ 来自浮点前缀，$x_i^q$ 来自已经量化的前缀，两者独立推进。当前块可以尝试补偿前层误差，却不会重新优化前面的块；块内含非线性和残差，不能把单线性层输入 Gram 矩阵当作它的精确 Hessian。这里通过梯度学习裁剪和变换，与 GPTQ 的逐权重解析补偿分别说明。

[FlatQuant](../methods/flatquant.md) v4 式 4 与固定代码 `9d88ffcb` 的 `flatquant/train_utils.py:cali_flat_quant` 提供一个不同实例：教师和量化块接收相同的浮点前缀输入，每完成一块以教师输出推进下一块。它仍是块重构，却没有上述量化前缀输入路径。因此讨论累积误差补偿时，必须记录谁产生输入、谁产生目标；不能仅凭“逐块校准”判断目标相同。三种学习变换的具体比较见 [可学习变换路线比较](../research/omniquant-affinequant-flatquant-comparison.md)。

[EfficientQAT](../methods/efficientqat.md) 的 Block-AP 同样将量化前缀输入与浮点参考目标分开推进，但允许当前块的权重和量化参数共同变化。固定代码 `39175493b2d14617d342a0a7956875e6ac16221b` 的 `quantize/block_ap.py:block_ap` 用缓存目标与 `MSELoss()` 实现这一局部训练。随后 E2E-QP 固定整数编码，整网训练尺度；这一步更换了梯度范围和目标，不是继续逐块最小化同一个 MSE。局部重构提供初始化，最终任务仍需独立验证。（EfficientQAT v3 §3、表 4。）

## 9. 路径归因怎样变成二阶系数

[QIG](../methods/qig.md) v1 §4.3 进一步把量化差异的积分梯度用于 token 加权 GPTQ。沿用本页 token 为列的输入 $X$，若非负误差系数为 $\lambda_n$，则 Gram 矩阵为 $X\operatorname{diag}(\lambda)X^\mathsf T$，Hessian 为其两倍；等价于把第 $n$ 列乘以 $\sqrt{\lambda_n}$。直接乘 $\lambda_n$ 再取 Gram 会把系数平方，不能与前文范数内部的因子混用。

非负性确保半正定，却不保证可逆或模型效果最优。[积分梯度](integrated-gradients-and-quantization-sensitivity.md) 的原始贡献可以有正负，QIG 公开代码取绝对值后才归一化；这一步不继承有符号完整性。归因目标、系数构造、MSE/MAE 选择和输入是否对称要分别核对。其缩放搜索允许 MAE，而上述二阶推导针对平方重构，不能因为同属 QIG 就自动合并；具体论文/代码差别由方法页承接。

## 10. 改变补偿规则、表示集合或例外预算

[QuIP 的 LDLQ](../methods/quip.md) 用三角反馈重新表述顺序补偿，并结合非相干处理分析谱结构；对应条件下它与 GPTQ 的反馈等价。[QuIP#](../methods/quip-sharp.md) 则把一次固定一列推广为一次固定一个向量块，配合固定格码本。

[AQLM](../methods/aqlm.md) 保留同类层输出重构目标，但改变可表示的权重集合，以多个学习码本之和编码，并联合优化离散索引和连续码本；输入 Gram 的非对角项使组间选择耦合。[SpQR](../methods/spqr.md) 把单步可补偿误差用于识别值得高精度保留的权重，同时让实际压缩后的 scale/zero 参与主体量化。

[SqueezeLLM](../methods/squeezellm.md) 的敏感性来自任务梯度外积的对角近似，不是本页固定输入线性重构的精确 Hessian。两类目标的来源、耦合和近似应分别说明；方法之间更一般的联系见 [低比特表示设计空间](../research/low-bit-representation-design.md)。

## 来源身份

下表用于在没有本地资料库时辨识来源；具体论述的章节、公式、图表或代码位置见正文。

| 来源 | 版本或快照 | 说明 |
| --- | --- | --- |
| [EfficientQAT: Efficient Quantization-Aware Training for Large Language Models](https://arxiv.org/abs/2407.11062v3) | `arXiv:2407.11062v3` | 局部重构与整网尺度训练 |
| [OpenGVLab/EfficientQAT](https://github.com/OpenGVLab/EfficientQAT/tree/39175493b2d14617d342a0a7956875e6ac16221b) | `39175493b2d14617d342a0a7956875e6ac16221b` | `block_ap.py` 的输入、目标与 MSE；2026-09-22 获取 |
| [QuIP: 2-Bit Quantization of Large Language Models With Guarantees](https://arxiv.org/abs/2307.13304v2) | `arXiv:2307.13304v2` | 量化机制与适用条件 |
| [QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks](https://arxiv.org/abs/2402.04396v2) | `arXiv:2402.04396v2` | 量化机制与适用条件 |
| [Extreme Compression of Large Language Models via Additive Quantization](https://arxiv.org/abs/2401.06118v4) | `arXiv:2401.06118v4` | 量化机制与适用条件 |
| [SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression](https://arxiv.org/abs/2306.03078v1) | `arXiv:2306.03078v1` | 量化机制与适用条件 |
| [SqueezeLLM: Dense-and-Sparse Quantization](https://arxiv.org/abs/2306.07629v4) | `arXiv:2306.07629v4` | 量化机制与适用条件 |
| [Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients](https://arxiv.org/abs/2603.17809v1) | `arXiv:2603.17809v1` | — |
| [FlatQuant: Flatness Matters for LLM Quantization](https://arxiv.org/abs/2410.09426v4) | `arXiv:2410.09426v4` | — |
| [ruikangliu/FlatQuant](https://github.com/ruikangliu/FlatQuant/tree/9d88ffcb7d2c6bda59fb5c44dad36adc101aadb1) | `9d88ffcb7d2c6bda59fb5c44dad36adc101aadb1` | — |
| [OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models](https://arxiv.org/abs/2308.13137v3) | `arXiv:2308.13137v3` | — |
| [Q-VLM: Post-training Quantization for Large Vision-Language Models](https://arxiv.org/abs/2410.08119v3) | `arXiv:2410.08119v3` | — |
| [GPTAQ: Efficient Finetuning-Free Quantization for Asymmetric Calibration](https://arxiv.org/abs/2504.02692v3) | `arXiv:2504.02692v3` | — |
| [VLMQ: Token Saliency-Driven Post-Training Quantization for Vision-language Models](https://arxiv.org/abs/2508.03351v2) | `arXiv:2508.03351v2` | — |
| [A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295v1) | `arXiv:2106.08295v1` | — |
| [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323v2) | `arXiv:2210.17323v2` | — |
