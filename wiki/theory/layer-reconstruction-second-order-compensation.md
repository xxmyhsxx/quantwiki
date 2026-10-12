---
title: 层输出重构与二阶误差补偿
type: concept
tags:
  - reconstruction
  - second-order
  - quantization-error
sources:
  - raw/papers/2026-09-21/norm-tweaking/paper.pdf
  - raw/repositories/2026-09-21/aqlm/source/aq_engine.py
  - raw/repositories/2026-09-21/aqlm/source/main.py
  - raw/papers/2026-09-22/gptvq/paper.pdf
  - raw/papers/2026-09-22/vptq/paper.pdf
  - raw/papers/2026-10-03/xor-trellis/paper.pdf
  - raw/papers/2026-10-03/quantforge/paper.pdf
  - raw/papers/2026-10-03/reconstruction-loss-scale-rmse/paper.pdf
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
updated: 2026-10-11
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

### 1.1 二阶矩的归约与分片合并

总 Gram 与平均二阶矩的区别影响实际损失归约。本小节以 $\overline G=XX^\mathsf T/n$ 表示平均二阶矩，保留前后章节 $G=XX^\mathsf T$ 的总 Gram 定义。于是 $\operatorname{tr}(E\overline GE^\mathsf T)/r=\|EX\|_F^2/(nr)$，其中 $E\in\mathbb R^{r\times d}$ 是权重误差、$n$ 是输入向量数、$r$ 是输出通道数。它是输出元素的平均平方误差；相对总和乘了常数，固定候选的排序可以相同，但梯度尺度、优化器和停止阈值要按实际接口判断。

两片材料分别有 $n_1,n_2$ 个输入向量，平均二阶矩为 $\overline G_1,\overline G_2$ 时，合并应使用

$$
\overline G=\frac{n_1\overline G_1+n_2\overline G_2}{n_1+n_2}.
$$

这也给出在线累积方式：旧均值乘旧样本比例，再加新批的总乘积除以总样本数。只有两片大小相同，直接取两个均值的等权平均才成立。Transformer 输入展平 batch 与序列后，应按实际 token 向量数计数，是否排除 padding 又是独立的数据接口问题。固定 AQLM 的 `AQEngine.add_batch` 和 `main.py:init_aq_engines_parallel` 是这一规则的实际例子，默认 FP64 累积、选码时另转码本 dtype，见[校准实现](../implementation/aqlm-calibration-export-and-inference.md#1.%20%E6%94%B6%E9%9B%86%E7%9A%84%E6%98%AF%20token%20%E5%8A%A0%E6%9D%83%E7%9A%84%E8%BE%93%E5%85%A5%E4%BA%8C%E9%98%B6%E7%9F%A9)。

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

### 3.1 同列向量为什么仍可使用欧氏最近中心

上面的标量补偿可逐输出行叠起来。对 $W\in\mathbb R^{M\times N}$，输出行共享输入侧 $H_F$，令 $A=H_F^{-1}$。在相同的连续最优点条件下，将一整列固定到候选，$a=\widehat W_{:,q}-W_{:,q}$，解为

$$
\Delta W_F^*=\frac{aA_{q,:}}{A_{qq}},\qquad
\Delta L^*=\frac{\|a\|_2^2}{2A_{qq}}.
$$

若一个向量包含同一输入列的若干输出行，各段误差的平方范数相加，没有输出行之间的交叉项。固定码本、各段索引可独立选择时，共同正数 $1/(2A_{qq})$ 不改变最近中心排序，所以可以对反馈后的向量作欧氏选码。这是 [VPTQ](../methods/vptq.md) 将向量沿输出行分组的关键。（VPTQ v2 §3.1、算法1；此处为明确矩阵维度和约束后的整理者推导。）

若向量改为同一输出行内的多个输入列，子目标一般含非对角矩阵度量，不能直接套用这个标量消去。若使用完整任务 Hessian、显式耦合输出行，也不保证以上分解成立。**向量的维度数相同，不代表它沿相同的计算轴或使用相同距离。**

当前列能欧氏选码，仍不表示各列独立：$A_{q,:}$ 决定误差怎样传播到下一列，下一列的待量化值因而改变。量化后自由集缩小，还要使用更新的条件逆或等价因子；本节不把 VPTQ 论文中不完整的逆矩阵处理伪代码当作已验证实现。

### 3.2 共享码本的加权初始化与选码是两种比较

层输出的二次型可以精确展开为

$$
\operatorname{tr}(\Delta W H\Delta W^\top)
=\sum_q H_{qq}\|\Delta W_{:,q}\|_2^2
+\sum_{q\ne r}H_{qr}\langle\Delta W_{:,q},\Delta W_{:,r}\rangle.
$$

先忽略交叉项、把来自不同列的输出向量放进共享码本，得到加权聚类目标 $\sum_t a_t\|u_t-C[z_t]\|_2^2$，其中 $a_t=H_{q(t)q(t)}$。固定分配集合 $S_k$、总权重为正时，中心更新为

$$
C[k]=\frac{\sum_{t\in S_k}a_tu_t}{\sum_{t\in S_k}a_t}.
$$

教学例：两个一维样本0和2共用一个中心，权重9和1。普通均值为1，加权中心为0.2；对应加权平方误差从10降到3.6。权重改变了不同样本对中心的影响；但为其中某个固定样本选中心时，若它的权重为正，该权重对所有候选相同，不改变欧氏最近中心。

这解释了 VPTQ 的 Hessian 加权初始化与同列欧氏选码为何能够并存。两者还用了不同层次的系数：初始化用完整输入统计的对角 $H_{qq}$；条件补偿后的候选增量代价用 $1/(2[H_F^{-1}]_{qq})$。不能把它们直接当作同一个量。对角初始化忽略了上式的交叉项，二阶反馈则利用列间耦合；前者没有证明完整目标已最优，也不能保证聚类每次找到全局最优。（VPTQ v2 §3.2.1；标量归约、中心公式和算例为整理者展开。）

### 3.3 多输入列的条件度量与码本中心更新

同时固定一行内的输入列集合 $P$ 时，向量约束为 $\delta_P=a$。沿用第3节正定 Hessian 和当前自由变量连续最优点的条件，令 $A=H_F^{-1}$，则

$$
\delta^*=A_{:,P}(A_{P,P})^{-1}a,\qquad
\Delta L^*=\frac12a^\top(A_{P,P})^{-1}a.
$$

可由约束选择矩阵 $S\delta=a$ 的驻点 $H_F\delta+S^\top\lambda=0$ 求得；$SA S^\top=A_{P,P}$。因此，多输入列候选的精确连续补偿代价使用**逆矩阵子块的逆**，一般不是直接取 $H_F$ 的子块，也不是取 $A$ 的子块或其对角倒数。$P$ 只有一个元素时退化为第3节标量式。

[GPTVQ](../methods/gptvq.md#2.%20%E5%A4%9A%E8%BE%93%E5%85%A5%E5%88%97%E9%80%89%E7%A0%81%E4%BD%BF%E7%94%A8%E4%BB%80%E4%B9%88%E8%B7%9D%E7%A6%BB)使用按输入坐标的对角敏感度进行聚类与选码。对角代理比完整矩阵便宜，但各坐标的权重仍可改变最近中心：教学例中两个候选误差为 $(1,0)$ 和 $(0,2)$，欧氏平方误差为1、4；采用 $D=\operatorname{diag}(8,1)$ 后为8、4，选择反转。它与同列输出向量共享一个正标量的情形不同。（GPTVQ v2 §4、式4–6；本节条件解与例子为整理者展开，不等于论文完整实现已核。）

对多样本加权聚类 $\sum_i(x_i-C[z_i])^\top D^{(i)}(x_i-C[z_i])$，固定分配后令中心梯度为零，得到

$$
\left(\sum_{i\in I_m}D^{(i)}\right)C_m
=\sum_{i\in I_m}D^{(i)}x_i.
$$

若和矩阵正定，解唯一；半正定时伪逆给一个最小范数解，未被样本权重覆盖的方向可以不唯一。对角情形是各坐标分别求加权均值；$D^{(i)}=a_iI$ 时才退化为第3.2节的一个样本一个标量权重。固定分配的最优中心并不意味着索引与中心联合问题已全局最优。

另一阶段是**固定全部索引后最小化完整层输出误差**。将全部中心向量化为 $c$，查表是固定线性映射，叠上校准输入可写为 $\operatorname{vec}(Q(c)X)=Tc$。于是

$$
\|WX-Q(c)X\|_F^2=\|y-Tc\|_2^2,
\qquad \nabla_c^2L=2T^\top T\succeq0.
$$

这说明 GPTVQ 附录 A 的直接中心更新是凸二次问题。未使用的中心、相同输入方向等仍会导致不唯一；再对中心舍入、改变索引、同时优化尺度/低秩双因子或把目标换成全模型输出，不能继续套用这个结论。初始化的分向量代理与最终层输出目标也不应合并为同一个优化步骤。

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

其中 $\phi,\Phi$ 为标准正态密度与分布函数；$\sigma=0$ 时直接取 $\max(\mu,0)$。正态假设不由 BN 自动保证，标准差必须非负。W §3.3 的 gamma/beta 文字说明有次序问题，不能机械沿用；符号条件见 [BN 统计估计](calibration-and-range-selection.md#4.%20%E6%B2%A1%E6%9C%89%E6%A0%B7%E6%9C%AC%E6%97%B6%EF%BC%8C%E7%BB%9F%E8%AE%A1%E5%81%87%E8%AE%BE%E8%83%BD%E6%9B%BF%E4%BB%A3%E4%BB%80%E4%B9%88)。W 表 4 在 MobileNetV2 中报告偏置校正改善，但这不是全模型输出完全恢复的证据。

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

W 表 5 的 ResNet18 W4、其余设定按 §3.4，全部层最近舍入为 23.99，局部连续放松为 66.56，最终 AdaRound 配置为 68.60（5 次均值）；它支持舍入决策值得优化，不表示任意数据或模型都达到同等收益。梯度优化局部舍入仍可属于 [PTQ](../fundamentals/quantization/post-training-and-quantization-aware-training.md)；它与 GPTQ 顺序固定权重后解析补偿的变量和流程不同。AdaRound 原文把这组数字中的 66.56 与 68.60 分别对应到「逐层 MSE 目标」与「非对称重构加激活函数」两个设计选择上，具体见 [AdaRound 的非对称重构](../methods/adaround.md#4.%20%E6%9D%BE%E5%BC%9B%E4%B8%8E%E6%AD%A3%E5%88%99)：非对称指教师输入 x 与量化前缀产生的输入 $\hat x$ 不同，不是量化网格的 zero-point。

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

[激活引导补偿与正交残差](../methods/activation-guided-compensation.md) 从另一个角度补足上述非对称目标：固定量化输入后，任意权重产生的输出都受其列空间限制，目标中正交于该空间的部分无法通过权重补偿消除。方法页展开伪逆投影恒等式，并区分改变权重与改变输入变换的作用；这不是额外增加一项启发式正则。

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

## 11. 重构损失的尺度怎样影响优化

前面讨论了重构什么、哪些参数可以补偿，以及误差怎样加权。即使这些选择不变，**损失的归约和梯度尺度也会改变有限步优化的结果**。本节依据 *The Devil Is in the Reconstruction Loss Scale* v1 的 §2–3、§4 与附录 A；公式的梯度条件和分组推导由本页展开。

### 从输出误差到参数更新

学习式 PTQ 通常冻结预训练权重，逐单元训练缩放、裁剪、旋转或适配器等辅助参数。单元可以是一个 Transformer block，也可以是连续多个 block。设当前输出残差为 $E=\widehat X_{i+1}-X_{i+1}$，元素数为 $N$，则

$$
L_{\mathrm{MSE}}=\frac{\|E\|_F^2}{N},\qquad
\nabla_{\widehat X}L_{\mathrm{MSE}}=\frac{2E}{N},\qquad
\|\nabla_{\widehat X}L_{\mathrm{MSE}}\|_F
=2\sqrt{\frac{L_{\mathrm{MSE}}}{N}}.
$$

这是 v1 式 (2)、(5) 的等价展开。相同 $N$ 下，MSE 相差一万倍会使输出梯度范数相差一百倍。作者观察到的跨阶段损失跨度还与浮点 block 本身的输出幅值有关，不能全部归因为量化累积；附录 E 在浮点模型上测量了这一差异。

但是输出梯度不等于参数更新。把当前单元输出与辅助参数展平，令 $J_i=\partial\widehat x_{i+1}/\partial\theta_i$，链式法则给出

$$
\nabla_{\theta_i}L=J_i^\top\nabla_{\widehat x_{i+1}}L.
$$

不同单元的 $J_i$、参数尺度、优化器及其历史状态还会影响更新。v1 用每步 $\|\Delta\Theta_i\|_F/\|\Theta_i\|_F$ 再在阶段内平均，作为实际优化强度的经验代理；恒定输出梯度并不证明各阶段这个比值相同。

### RMSE 归一化了什么

对同一个非零残差，单组 RMSE 为

$$
L_{\mathrm{RMSE}}=\frac{\|E\|_F}{\sqrt N},\qquad
\nabla_{\widehat X}L_{\mathrm{RMSE}}
=\frac{E}{\sqrt N\,\|E\|_F}
=\frac{\nabla_{\widehat X}L_{\mathrm{MSE}}}{2\sqrt{L_{\mathrm{MSE}}}},
\qquad
\|\nabla_{\widehat X}L_{\mathrm{RMSE}}\|_F=\frac1{\sqrt N}.
$$

RMSE 保留这一组内部的梯度方向，去掉其对残差整体幅度的依赖。它没有把损失数值强制变成相同大小，也没有消除 Jacobian 的差异。不同阶段的 $N$ 不同时，梯度范数仍不同。（v1 §3.1，式 (6)–(7)。）

对一个整体取平方根是严格单调变换，因此全局 MSE 与对应单组 RMSE 的最小点相同；改变的是优化路径与有限步结果。**先分组开方再平均则一般改变目标，不能继续套用这个最小点等价关系。** 零残差处范数不可微，以上导数式要求非零残差；若实现加平滑项，恒定范数公式也随之改变，论文未提供可据此确认的实现代码。

### 分组决定哪些误差被一起归一化

将全部元素分成 $M$ 个互不重叠、等大小的组，每组 $n=N/M$ 个元素。v1 式 (8) 定义

$$
L_G=\frac1M\sum_{m=1}^M\frac{\|E_{G_m}\|_2}{\sqrt n}.
$$

所有组残差非零时，每组梯度范数为 $1/(M\sqrt n)$，所以

$$
\|\nabla_E L_G\|_2^2
=\sum_{m=1}^M\frac1{M^2n}
=\frac1N.
$$

这解释了分组为何仍有同样的整体范数，却可改变组与组之间的相对权重。组大小不等时平方范数变为 $M^{-2}\sum_m |G_m|^{-1}$；有零残差组时，还须指定次梯度，不能无条件声称恒为 $1/\sqrt N$。

附录 A.2 采用 $E\in\mathbb R^{B\times T\times C}$，分别表示样本、token、通道，所有分组都在每个样本内部进行：

| 变体 | 一组包含什么 | 损失 |
| --- | --- | --- |
| Sample | 一个样本的全部 token 和通道 | $\frac1B\sum_b\sqrt{\frac1{TC}\sum_{t,c}E_{b,t,c}^2}$ |
| Channel | 一个样本中某通道的全部 token | $\frac1{BC}\sum_{b,c}\sqrt{\frac1T\sum_t E_{b,t,c}^2}$ |
| Token | 一个样本中某 token 的全部通道 | $\frac1{BT}\sum_{b,t}\sqrt{\frac1C\sum_c E_{b,t,c}^2}$ |
| Element | 一个特征元素 | $\frac1{BTC}\sum_{b,t,c}\lvert E_{b,t,c}\rvert$，即平均绝对误差 |

Channel 不是把整个 batch 的同名通道合成一组，Sample 也不是对整个 batch 只开一次方。v1 的重构 batch 分别为 OmniQuant/DuQuant 的 1、FlatQuant 的 4、SliderQuant 的 3；复现时须保留这些归约轴。教学上，残差 $(2,0)$ 的 MSE 为 2、MAE 为 1；$(1.4,1.4)$ 的 MSE 为 1.96、MAE 为 1.4，两个目标对候选的偏好相反。这个算例说明分组改变目标，不是模型实验。

### 放回顺序 PTQ 的流程

1. 沿原方法确定单元、校准输入、教师目标与可训练参数；保留原来的浮点或量化前缀路径，不能因为换损失就改变输入协议。
2. 对输出残差选择归约轴，计算对应 RMSE；量化器、参数、优化器、学习率和训练步数保持成对对照。
3. 在当前单元内更新辅助参数，完成后按原方法冻结并推进下一单元。RMSE 不让已经冻结的早期单元重新获得梯度。
4. 分别记录输出梯度、参数梯度、相对更新和最终质量。前者的恒等式不能代替后面三项的测量。

该改动作用于离线重构目标，保留原量化表示和推理流程。它与第 8 节的扩大联合重构范围，以及 [EfficientQAT](../methods/efficientqat.md) 更改可训练参数和整网阶段，是不同的选择。

### 实验支持的范围

以下均为作者报告。v1 附录 A.1 使用相同的 128 条 WikiText-2 校准样本，每条 2,048 token；保留各基础方法的优化器、学习率和步数。PPL 也在 WikiText-2 报告，六项零样本任务均分为 PIQA、ARC-Easy、ARC-Challenge、Winogrande、HellaSwag、BoolQ；同一语料名不能据此称为严格跨域校准。

| 对照及原表 | MSE | RMSE | 能支持的判断 |
| --- | --- | --- | --- |
| OmniQuant，Qwen3.5-27B，W2A16，表 1，PPL / 六任务均分 | 37.51 / 60.45% | 22.20 / 70.96% | 该低比特配置有较大改善；原模型为 6.90 / 77.10%，仍有明显差距 |
| FlatQuant，Qwen3.5-9B，W4A4，表 2，PPL / 六任务均分 | 10.26 / 72.23% | 9.15 / 74.19% | 换损失的价值不只出现在 OmniQuant；改善幅度受基础方法影响 |
| OmniQuant，Qwen3.5-9B，W4A16，表 4，HumanEval | 75.61% | 75.00% | 四项推理任务的均分改善，不代表每个任务都改善 |
| 同模型 W2A16G128，表 4，四项推理任务均分 | 0.97% | 3.65% | 比原模型 87.35% 仍低很多，换损失不足以恢复该配置 |

表 3 没有一个变体在所有配置中最好。另有一处需要保留：§4.1/4.3 将 Qwen 默认变体写为 Channel，但表 1 的 Qwen3.5-9B W3A16 均分 72.79% 与表 3 的 Token 值相同，Channel 列是 73.17%；v1 未解释此差异，不能把这些表当作逐项一致的变体映射。

表 5 在单 NVIDIA L40、batch 1 上测量整个顺序量化过程：四个对照的时间变化为 −1.19% 至 +2.82%，报告的峰值显存相同。这是离线量化成本，不是推理加速；论文未给这些差值的多次测量不确定性。本节未运行模型或作者代码，附录的完整任务附表也未逐格重算。

## 12. 权重确定后，激活舍入的目标怎样改变

前面的 GPTQ 固定输入、优化权重。[QuantForge 的 HiRes](../methods/quantforge.md#4.%20%E9%98%B6%E6%AE%B5%E4%BA%8C%EF%BC%9A%E8%AE%A9%20A4%20%E9%80%89%E7%A0%81%E8%80%83%E8%99%91%E5%B7%B2%E7%BB%8F%E7%A1%AE%E5%AE%9A%E7%9A%84%20W4) 则在 W4 已确定后，继续选择同一固定格式中的激活码。它说明：最近激活值不一定给出最近输出；已经发生的权重误差也会改变激活舍入的偏好。

固定一个32通道块，令 $x,q_0\in\mathbb R^{32}$ 分别为原激活和最近编码的解码值，$W,\widehat W\in\mathbb R^{m\times32}$ 为浮点参考与已编码权重。记 $d_0=q_0-x$，则当前局部残差为

$$
r_0=\widehat Wq_0-Wx=\widehat Wd_0-(W-\widehat W)x.
$$

只把第 $i$ 项改为同一固定指数下的另一个合法值，增量为 $\delta$，残差变为 $r_0+\delta\widehat W e_i$，因此

$$
\Delta L=2\delta[\widehat W^\top r_0]_i
+\delta^2[\widehat W^\top\widehat W]_{ii}
=2\delta[G_Qd_0-K_Qx]_i+\delta^2[G_Q]_{ii},
\quad
G_Q=\widehat W^\top\widehat W,\quad
K_Q=\widehat W^\top(W-\widehat W).
$$

这是固定块输出平方误差的精确展开。仅用 $d_0^\top G_Qd_0$ 会漏掉权重偏差的线性项；只有 $W=\widehat W$ 等使该项消失的条件下，才恢复纯敏感度加权的激活舍入。（QuantForge v1，式8、附录 B.3。）

教学例取块尺度为1，$x=(1.6,4,0,\ldots)$，参考权重一行为 $W=(1.2,0,4,\ldots)$，已编码 $\widehat W=(1,0,4,\ldots)$，其余项为0。两块最大幅值均为4，符合该编码器的指数0。最近激活码是 $(1.5,4,0,\ldots)$，输出残差为 $1.5-1.2\times1.6=-0.42$；把首项改为同样合法的2，残差变0.08。激活自身的距离增大，输出平方误差却从0.1764降到0.0064，$\Delta L=-0.17$。这个算例解释权重与激活误差的补偿，不是模型实验。

**局部精确不等于全层单调。** 若整层输出残差为各输入块贡献之和 $r=\sum_b r_b$，修改块 $b$ 所引起的残差增量为 $\Delta r_b$，则

$$
\Delta\|r\|_2^2
=\Delta\|r_b\|_2^2
+2\left\langle\sum_{a\ne b}r_a,\Delta r_b\right\rangle.
$$

逐块评分未计入第二项。教学上，两个块残差分别为1和−1时整层误差为0；将第一块局部误差降到0，整层残差反而为−1。全层误差、更深的非线性和任务质量因此仍需独立判断。

### 统计必须对应已经安装的计算状态

HiRes 的三处重测依赖具有更一般的解释价值：坐标变换改变 pre-A4 输入 Gram；固定 W4 改变 $G_Q,K_Q$；attention 校正改变下游 MLP 的输入和目标残差。公式正确但统计来自旧状态，仍是在解另一个问题。这里应记录“谁产生当前输入、哪个版本的权重已经固定、目标来自完整浮点轨迹还是同一当前输入上的浮点模块”，而不只记录校准语料名。（QuantForge §5、算法1、附录 C.5。）

这与第8节的联合重构范围是不同维度：重测不会自动扩大梯度范围或联合优化所有阶段。论文的 fresh/stale 对照固定最终执行顺序与校正形式，只改变拟合统计的取得时刻；它支持所测方法的状态依赖，不构成一般收敛或全网络单调性保证。

## 13. LDL 残余度量与离散路径搜索怎样衔接

LDLQ 的顺序反馈与内部量化器可以采用不同目标。[XOR-Trellis](../methods/xor-trellis.md#4.%20%E6%8A%8A%E6%AE%8B%E4%BD%99%E6%95%8F%E6%84%9F%E6%80%A7%E7%94%A8%E4%BA%8E%E8%B7%AF%E5%BE%84%E8%AF%84%E5%88%86)说明：外层用曲率补偿，不代表内部 trellis 路径已经按相同曲率选择。本节区分分解恒等式、实际搜索代理与动态规划精确性的条件。（v1 §4、附录 D/E。）

### 分解是精确的，代理还需另外定义

对正定 $H=LDL^\top$，若采用标量 LDL、$D$ 为正对角，令 $e=w-\widehat w$、$r=L^\top e$，有

$$
e^\top H e=r^\top D r=\sum_i D_{ii}r_i^2.
$$

这里的对角度量作用在**变换后的残差**上。实际顺序算法把先前量化误差反馈到目标 $z_i$，离散搜索面对的是 $\rho_i=z_i-\widehat z_i$。除非对具体反馈、分块和路径约束证明了 $\rho=L^\top e$，不能直接把 $\sum_iD_{ii}\rho_i^2$ 写成原始 full-Hessian 目标。XOR-Trellis 附录 D.1 明确保留这一区分。

采用块 LDL 时，还要区分块内与块间：$D=\operatorname{blockdiag}(D_1,\ldots,D_B)$，则精确形式为 $\sum_b r_b^\top D_b r_b$。若评分仅取 $d_i=(D_b)_{ii}$，就舍弃了块内 $i\ne j$ 的交叉项，即使另有三角反馈也不能自动收回这些项。附录 E.2 实际分析采用16列块；“D 是块对角”不等于“每块内部是对角”。

### 可加权重怎样保留最优子结构

固定目标、图、边界和正权重 $d_t$，令 $f(s,b)$ 为后继状态、$g(s,b)$ 为重构值。搜索目标与递推为

$$
C(\pi)=\sum_t d_t[z_t-g(s_t,b_t)]^2,
\qquad
V_{t+1}(s')=\min_{s,b:f(s,b)=s'}
\{V_t(s)+d_t[z_t-g(s,b)]^2\}.
$$

汇合到同一状态的两条历史拥有相同未来选择；可加目标下，更贵的历史不可能靠同一未来反超，因此每状态只需保留最便宜的前驱。固定 $d_t$ 改变数值评分，不改变这个理由，也不扩大状态/分支数。（附录 D.4。）

教学上，两条可行路径残差为 $(1,0)$ 和 $(0,2)$，欧氏目标为1和4，偏好前者；若 $d=(8,1)$，代价为8和4，偏好后者。若所有 $d_i=\alpha>0$，则两种目标必有同样的最小点。若代价依赖当前状态没有保存的更早历史，或包含未被状态承接的跨步交叉项，上述递推就不能不加修改地保持精确。

### 近似界只约束所声明的对角目标

对同一可行残差集合，令 $\pi_E,\pi_D$ 分别最小化欧氏目标和正对角目标，$d_{\min}>0$。由

$$
d_{\min}\|\rho\|_2^2\le\rho^\top\operatorname{Diag}(d)\rho
\le d_{\max}\|\rho\|_2^2
$$

可得

$$
\mathcal L_D(\pi_E)
\le d_{\max}\|\rho(\pi_E)\|_2^2
\le d_{\max}\|\rho(\pi_D)\|_2^2
\le\frac{d_{\max}}{d_{\min}}\mathcal L_D(\pi_D).
$$

该界依赖正的最小权重和同一候选集合，只保证对角代理上的比值；它既不证明代理等于完整层误差，也不证明语言任务 PPL 的比值。分位数各向异性 $P_{95}(d)/P_5(d)$ 是经验诊断，不能替换此界中的极值比。（附录 D.2、E.2。）

加权发生在离线路径选择时，$d_t$ 无需存入部署流。改变离线评价目标可以保持在线码格式不变；反过来，这也意味着“运行时没有增加开销”不能证明搜索找到了完整模型损失的最优解。

## 14. 逐通道矩匹配与逐位置重构约束不同的对象

同样比较块输出，目标可以只约束统计而不对齐每个位置。[Norm Tweaking](../methods/norm-tweaking.md) v2式2以逐通道均值与方差之差调整归一化参数，冻结量化线性层。原文统计轴、mask和方差估计器未完整给出；以下是整理者用于区分目标的教学定义，不是作者实现复原。

设 $F,Q\in\mathbb R^{S\times C}$，$S$ 是参与统计的位置数、$C$ 是通道数，定义 $\mu_F^c=S^{-1}\sum_sF_{sc}$ 与 $v_F^c=S^{-1}\sum_s(F_{sc}-\mu_F^c)^2$，$Q$同理。比较

$$L_{\rm moment}=\frac1C\sum_c\left(|\mu_F^c-\mu_Q^c|+|v_F^c-v_Q^c|\right),\qquad L_{\rm MSE}=\frac1{SC}\|F-Q\|_F^2.$$

单通道 $F=(-1,1)^\top,Q=(1,-1)^\top$ 的均值均为0、方差均为1，故 $L_{\rm moment}=0$，但 $L_{\rm MSE}=4$。矩匹配允许位置置换，而token语义、对应关系或因果次序可能已经改变；零矩误差不保证零逐位置误差。

多通道还存在相关性缺口。若 $F$ 两行为 $(-1,-1),(1,1)$，$Q$ 两行为 $(-1,1),(1,-1)$，每个通道的均值/方差仍相同，但通道间协方差由 $+1$ 变成 $-1$。更高阶矩也未受约束，因此只能称为逐通道低阶矩匹配，不能直接称为完整分布匹配。若对应位置MSE为0则这些统计必相同，逆向结论由上述反例否定；有限误差下还要保留幅值与统计估计条件。

优化变量同样重要：只更新norm参数、更新全部权重、学习舍入或插入残差，其可达输出集合不同，不能单靠损失名称比较恢复能力。Norm Tweaking的参考由浮点当前块接收量化前缀输入产生；这与全浮点路径的参考不同，需连同[非对称输入](#7.%20%E5%8A%A0%E6%9D%83%E9%87%8D%E6%9E%84%E4%B8%8E%E9%9D%9E%E5%AF%B9%E7%A7%B0%E8%BE%93%E5%85%A5)一并记录。小参数量会减少优化状态，却仍可能需要穿过固定线性计算的输入反向。

## 来源身份

- Norm Tweaking: High-performance Low-bit Quantization of Large Language Models（`https://arxiv.org/abs/2309.02784v2`），`arXiv:2309.02784v2`；算法1、式2；第14节统计定义与反例为整理者教学展开。

- GPTVQ: The Blessing of Dimensionality for LLM Quantization（`https://arxiv.org/abs/2402.15319v2`），arXiv:2402.15319v2；§4式4–6、附录 A式7；本页§3.3的条件解、向量反例和固定索引凸性为整理者展开。

- VPTQ: Extreme Low-bit Vector Post-Training Quantization for Large Language Models（`https://arxiv.org/abs/2409.17066v2`），arXiv:2409.17066v2；§3.1–3.2.1、算法1；本页§3.1–3.2的规范矩阵推导、加权均值和算例为整理者展开。

- XOR-Trellis: Ultra-Low-Complexity Dequantization and Curvature-Aware Hadamard-Free LLM Quantization（`https://arxiv.org/abs/2610.00432v1`），v1；§4、附录 D/E。第13节的路径评分算例和最优子结构说明为整理者展开。

- QuantForge: Discovering Residual Decompositions for MXFP4 Post-Training Quantization（`https://arxiv.org/abs/2609.34680v1`），v1；§5.2/式8、算法1、附录 B.3/C.5。第12节的合法码数值例和跨块交叉项为整理者展开。

- The Devil Is in the Reconstruction Loss Scale: Rethinking Optimization in LLM Quantization（`https://arxiv.org/abs/2610.00983v1`），arXiv:2610.00983v1；§2–3、式 (2)–(8)、§4 表 1–5 与附录 A/D/E/F。第 11 节的链式法则、分组梯度与最小点关系为整理者展开，作者实验与本地验证分开。

下表用于在没有本地资料库时辨识来源；具体论述的章节、公式、图表或代码位置见正文。

| 来源 | 版本或快照 | 说明 |
| --- | --- | --- |
| EfficientQAT: Efficient Quantization-Aware Training for Large Language Models（`https://arxiv.org/abs/2407.11062v3`） | `arXiv:2407.11062v3` | 局部重构与整网尺度训练 |
| OpenGVLab/EfficientQAT（`https://github.com/OpenGVLab/EfficientQAT/tree/39175493b2d14617d342a0a7956875e6ac16221b`） | `39175493b2d14617d342a0a7956875e6ac16221b` | `block_ap.py` 的输入、目标与 MSE；2026-09-22 获取 |
| QuIP: 2-Bit Quantization of Large Language Models With Guarantees（`https://arxiv.org/abs/2307.13304v2`） | `arXiv:2307.13304v2` | 量化机制与适用条件 |
| QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks（`https://arxiv.org/abs/2402.04396v2`） | `arXiv:2402.04396v2` | 量化机制与适用条件 |
| Extreme Compression of Large Language Models via Additive Quantization（`https://arxiv.org/abs/2401.06118v4`） | `arXiv:2401.06118v4` | 量化机制与适用条件 |
| Vahe1994/AQLM（`https://github.com/Vahe1994/AQLM/tree/e79a896ed6656fe4ed06193d42d004e7d0bbdbb2`） | `e79a896ed6656fe4ed06193d42d004e7d0bbdbb2` | `AQEngine.add_batch` 与 `init_aq_engines_parallel` 的均值累积和合并 |
| SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression（`https://arxiv.org/abs/2306.03078v1`） | `arXiv:2306.03078v1` | 量化机制与适用条件 |
| SqueezeLLM: Dense-and-Sparse Quantization（`https://arxiv.org/abs/2306.07629v4`） | `arXiv:2306.07629v4` | 量化机制与适用条件 |
| Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients（`https://arxiv.org/abs/2603.17809v1`） | `arXiv:2603.17809v1` | — |
| FlatQuant: Flatness Matters for LLM Quantization（`https://arxiv.org/abs/2410.09426v4`） | `arXiv:2410.09426v4` | — |
| ruikangliu/FlatQuant（`https://github.com/ruikangliu/FlatQuant/tree/9d88ffcb7d2c6bda59fb5c44dad36adc101aadb1`） | `9d88ffcb7d2c6bda59fb5c44dad36adc101aadb1` | — |
| OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models（`https://arxiv.org/abs/2308.13137v3`） | `arXiv:2308.13137v3` | — |
| Q-VLM: Post-training Quantization for Large Vision-Language Models（`https://arxiv.org/abs/2410.08119v3`） | `arXiv:2410.08119v3` | — |
| GPTAQ: Efficient Finetuning-Free Quantization for Asymmetric Calibration（`https://arxiv.org/abs/2504.02692v3`） | `arXiv:2504.02692v3` | — |
| VLMQ: Token Saliency-Driven Post-Training Quantization for Vision-language Models（`https://arxiv.org/abs/2508.03351v2`） | `arXiv:2508.03351v2` | — |
| A White Paper on Neural Network Quantization（`https://arxiv.org/abs/2106.08295v1`） | `arXiv:2106.08295v1` | — |
| GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers（`https://arxiv.org/abs/2210.17323v2`） | `arXiv:2210.17323v2` | — |
