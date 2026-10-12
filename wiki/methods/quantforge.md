---
title: QuantForge 与 HiRes：实验约束的搜索与 MXFP4 分阶段重构
type: method
tags:
  - ptq
  - microscaling
  - reconstruction
  - equivalent-transform
  - evaluation
sources:
  - raw/papers/2026-10-03/quantforge/paper.pdf
updated: 2026-10-08
---

# QuantForge 与 HiRes：实验约束的搜索与 MXFP4 分阶段重构

QuantForge 是搜索量化程序的方法：保留有效程序，同时用预先登记的竞争解释和对照实验决定下一处修改，并检查实验结论是否进入实际计算。HiRes 是它发现的固定量化器，依次改变坐标、实现合法 W4/A4 编码、校正 attention 与 MLP 路径。后一步使用前一步安装后的统计，不能把各步当成互不依赖的插件。

本页依据 arXiv:2609.34680v1 正文和必要附录 A–D。作者的质量实验采用 PyTorch 模拟量化；原生 MXFP4 内核的吞吐、显存与部署收益未验证。本轮没有匹配作者源码可供静态核读，以下实现细节均为论文描述，没有运行搜索、模型或 GPU。前置见 [MX 格式](../fundamentals/numeric-formats/fp8-and-mx-data-formats.md)、[可逆变换](../fundamentals/mathematics/invertible-transforms-and-kronecker-products.md)和[二阶重构](../theory/layer-reconstruction-second-order-compensation.md)。

## 1. 固定的量化约束与两项成果

HiRes 针对 decoder 的 Q/K/V/O、gate/up/down 线性投影，两侧操作数沿连续输入特征以32个元素分块。每块使用 E2M1 元素和一个 E8M0 尺度；权重静态量化，激活动态量化。embedding、norm、RoPE 和语言模型 head 保持较高精度，评测关闭 KV cache。（§3、附录 B.1。）

作者的参考编码器对非零块设

$$
e=\operatorname{clip}\!\left(\left\lfloor\log_2\max_i|x_i|\right\rfloor-2,-127,127\right),
\qquad s=2^e.
$$

全零块用 $e=-127$；元素绝对值来自 $s\{0,\frac12,1,\frac32,2,3,4,6\}$。最近码并列时取较大绝对值，符号跟随输入。这些是该比较采用的编码约定，不能用它替代所有 MXFP4 转换实现。

“严格 W4A4”约束目标线性层的合法操作数与编码，不表示全部算子、辅助状态和乘加都只用4位。论文实际保留解码为 BF16 的量化权重，用高精度计算评估合法低比特值；结构校正在 GEMM 外执行，另有 FP32 小矩阵。HiRes 的数值质量与 QuantForge 的搜索效率分别评测，不能用其中一项结果替代另一项。

## 2. QuantForge 怎样让实验改变下一份程序

### 2.1 程序有效与解释成立分别记账

搜索状态包含有效程序集合 $\mathcal P_t$ 和未解决问题集合 $\mathcal R_t$。这里的 algorithmic residual 指附着在程序上的**未决设计解释**，与 HiRes 中的数值输出残差不同。例如收益究竟来自变换方向，还是来自随变换一起改变的块编码？两种解释会导向不同代码。（§4、附录 A。）

一个提案登记候选、父程序、要解决的问题、竞争解释 $\mathcal E$、每种解释在条件 $c$ 下的预测 $P_e(c)$，以及允许的对照 $\mathcal U$。执行前选择成本最小、能区分所有登记解释对的对照集合：

$$
\min_{V\subseteq\mathcal U}\sum_{c\in V}\operatorname{cost}(c)
\quad\text{s.t.}\quad
\forall e\ne e',\ \exists c\in\{\mathrm{primary}\}\cup V:
P_e(c)\ne P_{e'}(c).
$$

相同预测的重复对照未必需要重跑；若允许的对照不能区分解释，先修改提案。候选与对照冻结相同模型、校准、格式、评测器和资源上限，只改变声明的干预；旧结果只有完整执行身份一致时才复用。（式2、附录 A.2。）

这是对**已登记解释及其预测**的判别，不保证解释集合穷尽，也不是对普遍因果机制的证明。程序可因成绩好而保留，同时撤回原来的解释。

### 2.2 结论要进入对应计算

论文将实验结果转成后继程序的要求：

| 观测与判断 | 下一步要求 |
| --- | --- |
| 目标收益与预期特征都支持解释 | 扩展或组合该方向 |
| 只有目标收益，没有预期特征 | 保留有用程序，修正解释 |
| 只有预期特征，没有目标收益 | 检查该机制怎样传到最终目标 |
| 与父程序等效，或仍无法区分解释 | 放弃该干预，或增加有效判别 |
| 约束检查或恢复父程序的检查失败 | 拒绝并恢复父程序 |

后继程序必须在问题所指向的计算上体现变化，系统比较 source diff 并执行定向 probe；文字说明和不影响该计算的改名不能关闭问题。随后在改过的程序上重新测量，打开下一项设计空间。这里的“编译”是把实验结论转成程序约束，不是编译 CUDA。（§4.3–4.4、附录 A.2–A.3。）

附录 D.2 给出局部实例：在 Qwen3-4B 的指定 down 输入上，随机置换并配对权重后，超大块内幅值比的块占比从7.2%增到41.5%。这支持检查块成员分布，而非只看全张量范围。后继程序改为按通道对**所在块共享指数**的贡献选择缩放对象，定向撤回该选择会改变 probe；这个对照仅支持所登记设置，并不证明任意随机置换必然更差。

## 3. HiRes 阶段一：坐标要看实际编码留下什么

### 3.1 参考坐标与配对变换

对一个物理 Block32，记 $q=32$、激活 $X\in\mathbb R^{n\times q}$、权重 $W\in\mathbb R^{m\times q}$。对激活与权重二阶矩分别加 ridge：

$$
A=X^\top X/n+\lambda_A I,\qquad B=W^\top W/m+\lambda_B I,
\qquad
M_0=\tfrac12[\operatorname{DN}(A)+\operatorname{DN}(B^{-1})],
\qquad P_{\rm ref}=\operatorname{Cap}_8(M_0^{-1/2}).
$$

其中 $\operatorname{DN}(M)=M/\det(M)^{1/q}$；$\lambda_A,\lambda_B$ 分别取对应未正则矩阵平均对角值的1%再加 $2^{-30}$。$\operatorname{Cap}_8$ 将正定矩阵的 log 特征值投影到“和为0、绝对值不超过 $\frac12\log8$”的集合，再取指数，因此行列式为1、条件数至多8。两侧几何以相反方向响应坐标伸缩，所以这里组合 $A$ 与 $B^{-1}$，不是简单平均两侧幅值。（§5.1 式4、附录 B.2 式18–19。）

### 3.2 从编码误差构造有限的通道交互

参考几何只描述能量，尚未描述 E2M1/E8M0 的实际舍入。令 $Y=XP_{\rm ref}^\top,V=WP_{\rm ref}^{-1}$，$R_H=H_{32}/\sqrt{32}$。先在 Hadamard 后执行一次严格 anchor 量化，再把误差映回 Hadamard 前：

$$
E_X=[Q_A(YR_H^\top)-YR_H^\top]R_H,\qquad
E_W=[Q_W(VR_H^\top)-VR_H^\top]R_H,
\qquad
K_X=E_X^\top Y/n,\quad K_W=V^\top E_W/m.
$$

附录 B.2 将激活按校准序列奇偶、权重按输出行奇偶分成两份。每份以 $c_i=(Y^\top Y/n)_{ii}$、$h_i=(V^\top V/m)_{ii}$ 构造
$p_{ij}=h_i(K_X)_{ij}-c_j(K_W)_{ij}$：一侧的误差由另一侧的能量加权。每对 $i,j$ 再转成对称和反对称的两个模式，使用三项阈值抑制跨份不一致、两侧归因相抵消、反对称强度超过对称强度的部分。

具体过程是按正曲率代理缩放、soft-threshold、将全部 $2\binom{32}{2}$ 个系数共同投影到半径 $1/8$ 的欧氏球，再逆变换为零对角矩阵 $K$；重建时的额外除数至少为1。因此有

$$
\|K\|_2\le\|K\|_F\le1/8,\qquad
G_0=I+K,\qquad G=|\det G_0|^{-1/q}G_0,\qquad
T=R_HGP_{\rm ref}.
$$

范数界保证 $I+K$ 可逆；行列式归一化只控制整体体积，不把 $G$ 变成正交矩阵。最后配对
$\widetilde X=XT^\top,\widetilde W=WT^{-1}$，精确实数代数下保持 $\widetilde X\widetilde W^\top=XW^\top$；有限精度构造与编码仍引入误差。（式5–6、附录式20–30。）这是一次由 anchor 统计构造的更新，不是在线学习稠密变换。

### 3.3 两侧风险只提出缩放方向，实际重构决定接受

固定 $T$ 后，每个 Block32 内再分八个四通道子组。以变换后、严格编码得到的激活/权重误差构造

$$
r_A=\operatorname{tr}(G_WC_E),\qquad
r_W=\operatorname{tr}(G_EC_X),
$$

其中 $C_X=X_g^\top X_g/n,C_E=E_X^\top E_X/n$，$G_W=W_g^\top W_g,G_E=E_W^\top E_W$；这是未中心化二阶矩。对子组风险加 $\epsilon_b=2^{-12}\operatorname{median}_g(r_{A,bg}+r_{W,bg})$，在各风险加正则后为正的条件下，计算 $\frac18\log_2[(r_A+\epsilon_b)/(r_W+\epsilon_b)]$，减去八组均值，再投影到和为0、幅度不超过 $1/8$ 的集合，得到 $u_b$。（式7、附录式31。）

各分量广播到四个通道，形成 $D=\operatorname{Diag}(2^u)$。**$u$ 是实数缩放参数，后面的 E8M0 共享指数才是整数。** 配对关系变为 $X'=X(DT)^\top,W'=W(DT)^{-1}$。

每块按顺序测试全步 $u_b$、半步 $u_b/2$ 和不缩放；每个候选都重新执行 GPTQ 与激活选码。全步的 W4/A4 块输出重构损失不高于不缩放时就接受，否则试半步，否则回到不缩放；不是选三者中的最小值。最终各块组合确定后再统一编码一次。代理风险只给方向，接受依赖实际两侧编码后的损失；局部接受也不等于全模型单调改善。（附录 B.2，式32。）

## 4. 阶段二：让 A4 选码考虑已经确定的 W4

安装 $DT$ 后重新采集变换坐标下的 **pre-A4 激活**，以其 Gram 作 GPTQ 曲率；权重重构保留实际 Block32 边界，阻尼比0.01。随后固定变换、指数和 W4 权重，动态选择 A4 值。（§5.2、附录 B.1/B.3。）

对一个块的列向量 $x\in\mathbb R^{32}$，$q_0$ 是固定共享指数下的最近合法码向量，$d_0=q_0-x$。令 $W$ 为变换后的 BF16 参考权重，$\widehat W$ 为已编码权重的解码值：

$$
G_Q=\widehat W^\top\widehat W,\qquad
K_Q=\widehat W^\top(W-\widehat W).
$$

将坐标 $i$ 改成合法值 $c$，$\delta=c-[q_0]_i$，其块局部输出平方误差变化为

$$
\Delta(i,c)=2\delta[G_Qd_0-K_Qx]_i+\delta^2[G_Q]_{ii}.
$$

$G_Q$ 衡量激活误差如何穿过已量化权重，$K_Q$ 则考虑权重自身的偏差；省略线性项中的 $K_Qx$ 只得到 sensitivity-only 对照。附录 B.3 证明这是 $\|\widehat Wq-Wx\|_2^2$ 的精确变化，不是整个网络任务损失的精确变化。局部展开及跨块反例见[二阶重构 §12](../theory/layer-reconstruction-second-order-compensation.md#12.%20%E6%9D%83%E9%87%8D%E7%A1%AE%E5%AE%9A%E5%90%8E%EF%BC%8C%E6%BF%80%E6%B4%BB%E8%88%8D%E5%85%A5%E7%9A%84%E7%9B%AE%E6%A0%87%E6%80%8E%E6%A0%B7%E6%94%B9%E5%8F%98)。

每次枚举32个坐标、每个坐标16种 E2M1 位模式（包括正负零），只接受最负的一个候选；没有负值则不改。负值并列取较小坐标号，再取较小无符号码号。每个物理块至多改一个元素，共享指数不变。因此每块改码数上限为1、全元素比例上限为 $1/32=3.125\%$，但仍有持续的在线评分和矩阵读取成本。（式8、附录 B.3/C.4。）

## 5. 阶段三：低维校正要在正确状态上拟合

### 5.1 每个 KV head 的 value 校正

已安装 W4/A4 后，对每层每个 KV head，汇集校准 token 和 head 坐标，拟合两个共享标量：

$$
(a,b)=\arg\min_{a,b}\sum_t\|a v_t+b\mathbf1-v_t^\star\|_2^2.
$$

$v_t$ 来自实际量化路径，$v_t^\star$ 是同一校准序列对应的 BF16 轨迹；它不是为每个坐标单独拟合 scale/bias。若汇集所有标量后的输入方差为零，使用 $(a,b)=(1,0)$。FP32 affine 后舍入到 BF16，再进入原 attention 与量化 O 投影。（式9、附录 B.4 式33。）

### 5.2 attention 校正后，再采 MLP 的乘积误差

从校正后的模型重新采集 gate/up 输出 $g,u$，令 $m=\operatorname{SiLU}(g)\odot u$。此处的参考 $g^\star,u^\star$ 是在**同一个实际层输入**上计算的 BF16 投影，目标为 $m^\star=\operatorname{SiLU}(g^\star)\odot u^\star$，与前述 BF16 轨迹目标有不同的输入约定。

每层只用三个共享系数：

$$
\bar m=\operatorname{SiLU}((1+\beta)g+\alpha\sigma_g)\odot(1+\eta)u,
\qquad\theta=(\alpha,\beta,\eta).
$$

$\sigma_g$ 是全层校准 token/通道池化后的正则标准差：
$\sigma_g^2=[\mathbb E g^2-(\mathbb E g)^2]_++2^{-12}\mathbb E g^2+2^{-20}$。
对 $\theta=0$ 作一阶展开，以 $e=m^\star-m$ 为目标、$\phi'=\operatorname{SiLU}'(g)$，每个标量样本的三维设计行是

$$
h_{tj}=[\,\sigma_g\phi'_{tj}u_{tj},\ g_{tj}\phi'_{tj}u_{tj},\ m_{tj}\,],
\qquad
\theta_c=(\mathbb E_c[\omega h^\top h]+\lambda_c I)^{-1}
\mathbb E_c[\omega h^\top e].
$$

权重 $\omega_j$ 来自 BF16 down 投影第 $j$ 列的能量，先相对均值归一并裁到 $[1/4,4]$，再归一到均值1。校准按序列编号模8和四个512-token区段划成32个 cell，各自作加权 ridge。对三项系数分别取 cell 中位数，并用 $1.4826$ 倍中位绝对偏差除以 $\sqrt{32}$ 作 soft shrink，最后整体缩到 $\ell_1$ 范数不超过 $1/8$；偶数个样本的中位数采用较低的中间顺序值。（附录 B.4 式34–38；本页用 $h$ 避免与尺度矩阵 $D$ 重名。）

这是受限的一阶拟合，不是任意非线性 MLP 的精确恢复。校正结果舍入 BF16 后，仍经过 down 的动态 A4 编码再乘 W4。attention 校正改变了下游输入，所以必须重新测 MLP；固定执行顺序而只换 fresh/stale 拟合状态，才隔离了这项依赖。

## 6. 质量结果与搜索结果采用不同协议

### 6.1 固定 HiRes 的质量

主质量协议使用精确排序的128条 C4-train 序列、每条2048 token、seed0，匹配 BF16 checkpoint，关闭 KV cache。七任务为 WT2/C4 PPL 和 ARC-C、ARC-E、HellaSwag、PIQA、WinoGrande；前四个准确率任务用 acc_norm，WinoGrande 用 acc，ARC-C 为全部299题官方 validation。（§6.1、附录 C.1。）

对每模型、每任务，PPL 损害为 $\mathrm{PPL}_q/\mathrm{PPL}_{16}-1$，准确率损害为 $1-\mathrm{Acc}_q/\mathrm{Acc}_{16}$，保留优于 BF16 时的负数。Fit-7 为七任务损害的“均值与最大值各占一半”；Robust Fit 再对七模型 Fit-7 作相同聚合。它不是平均准确率，也不是所有任务损害的统一百分比。

表1报告 HiRes 的 Robust Fit 0.09300，WUSH 为0.09910；Qwen3-32B 的 Fit-7 为0.027369，对照 WUSH 0.036489。优势不是逐模型逐指标一致：Qwen3-4B 的 Fit-7 为0.0524，差于 WUSH 的0.0479；Qwen3-8B 的五项准确率均值67.4385%，也低于 WUSH 的67.9176%，尽管 Fit-7 更低。（表1、表9、表12。）

基线均被适配到作者共同 MXFP4 约束。例如 BRQ 使用固定归一化 $H_{32}$ 加 GPTQ，随机符号/共享/RNG 选择仍未解决；TORQ、BATQuant 为独立重实现，BATQuant/FOCUS 换用本论文校准材料。这些数字比较的是所述适配，不是各原始系统的统一排名。（附录 C.1。）

表2的 seed0 Fit-3（WT2/C4/ARC-C）显示阶段依赖：

| 状态 | Qwen3-8B | Llama-3-8B |
| --- | --- | --- |
| 严格 RTN | 0.20249 | 0.31653 |
| 加坐标构造 | 0.07828 | 0.16741 |
| 再加离散实现 | 0.06502 | 0.15530 |
| 再加结构校正 | 0.03259 | 0.14830 |
| 完整流程改用旧 MLP 统计 | 0.04173 | 0.15824 |

附录 C.2/C.5 又在两模型、校准 seed $\{0,3,5\}$ 上比较。fresh 相对 stale 的六组 Fit-3 都降低；这支持所测流程的重测顺序，不保证每种量化器或新分布下都有效。后验消融按作者说明没有反馈到最终算法设计。

### 6.2 发现 HiRes 的过程与独立搜索比较

| 协议 | 初始程序与可见目标 | Llama-3.1-8B 的角色 |
| --- | --- | --- |
| 开发 HiRes | 从 GPTQ 出发；Qwen3-4B 的 Fit-3 与诊断指导修改 | 迁移验证；是否通过的汇总信号反馈给提案，详细诊断不反馈 |
| 比较搜索系统 | 四组都从 RTN 出发；只见 Qwen3-4B WT2 PPL 与执行反馈 | 每20次搜索评估独立测 champion；结果不反馈，也不用于选后继 |

两种设置不能合并为“整个发现过程完全没看迁移模型”。其他最终评测模型在固定 HiRes 后才运行。独立框架实验使用相同 GPT-5.6 Sol/high 提案设置、接口和八个配对 seed；ReflectMem 与 QuantForge 都允许65536输入 token，其中记忆最多16384。（附录 A.1、D.1。）

每组每 seed 的240次搜索评估中，QuantForge 分配141次新候选、69次对照和30次实现 probe；其他三组各240个新候选。另有每 seed 12次留出测量不计入240。评估调用数相同不等于 GPU 时间、token 用量或总成本相同。（§6.2、表35。）

| 搜索方法 | 最终留出 PPL 中位数 | 达到预设 PPL≤8 的 seed |
| --- | --- | --- |
| Score-only | 10.03 | 1/8 |
| TextMem | 9.20 | 3/8 |
| ReflectMem | 8.09 | 3/8 |
| QuantForge | 7.80 | 6/8 |

表32中 QuantForge 对 ReflectMem 在7/8配对 seed 较好，均值差−0.250，作者配对 t 区间为[−0.386,−0.114]。移除必须落实结论及执行检查、把30次 probe 还给新候选时，中位 PPL 为8.06；它同时改变实现要求与 probe，不是仅一项独立效应。附录 D.3 的 source-diff 中间对照为7.96，进一步区分检查方式。（表32、35–37、43。）

相同父程序、历史和对照结果的48个配对重放中，改善后继由15/48增到28/48；它们来自八条轨迹各六个选定转移，不是48次互相独立的完整搜索。证据支持这个提案器、接口、预算与任务下的搜索收益，不证明任意科研流程必然受益。搜索的留出、预算与统计单位见[质量评测 §11](../implementation/model-quality-evaluation.md#11.%20%E6%90%9C%E7%B4%A2%E5%8F%8D%E9%A6%88%E3%80%81%E7%95%99%E5%87%BA%E8%AF%84%E6%B5%8B%E4%B8%8E%E8%81%9A%E5%90%88%E6%8C%87%E6%A0%87%E8%A6%81%E5%88%86%E5%88%AB%E5%AE%9A%E4%B9%89)。

## 7. 合法四位值与实际存储、运行成本

附录 C.6 的优化实现为每个目标线性层、每个输入 Block32 保留 FP32 的 $U=DT$、$G_Q$ 上三角含对角以及完整 $K_Q$：

$$
4(32^2+32\cdot33/2+32^2)=10304\ {\rm bytes}.
$$

这些状态按每个线性层分别存，不因输入宽度相同自动共享。构造用逆矩阵、Hessian、校准缓存与 BF16 参考权重副本在量化后释放；额外还有每 KV head 两个校正系数、每层四个 MLP 参数（含 $\sigma_g$）。

表28统计的是**BF16 解码权重加辅助张量**：Qwen3-8B 的辅助状态407.534 MiB，总静态存储15.654 GiB，相对 BF16 的15.256 GiB增加2.61%；不包括激活、工作区和 allocator reserve。格式中32个4位值加8位尺度的4.25 bit/weight账，不能替代这个实际实现的内存账。

作者在单 A100-SXM4-80GB、PyTorch2.9.1/CUDA12.6、batch1、2048输入 token、无生成且关闭 cache 下，以10次预热后30次 CUDA 同步完整 forward 的中位数计时。融合算子包含变换、动态选码、辅助矩阵读取和校正，但 GEMM 使用解码 BF16 权重；Qwen3-8B 为1286→1343 ms，增加4.43%，一次性量化61.6分钟。其他两模型完整 forward 也增加3.97%–4.49%，不是原生 MXFP4 加速。（表29。）

表22另报 eager/reference fake-quant 的 greedy decoding、以及开关 one-code 的吞吐差；它的 backend、工作负载和基线均不同，不能与表29合并成一个加速比。变换统计/拟合用 FP64，构造矩阵与激活选码评分用 FP32，校准跨阶段暂存为 FP16，恢复运行时 FP32 后 BF16舍入；这些路径均为作者描述，尚无本轮代码核对或实测。

## 8. 与现有知识的联系和限制

HiRes 将 [GPTQ](gptq.md) 的固定输入重构接到新的坐标及 pre-A4 曲率，再用已经确定的权重偏差选择激活码；它没有取消局部代理的边界。与 [PRQuant](prquant.md) 增加低比特权重残差和扩大归约维不同，HiRes 改变合法码、加入低维路径校正并保留辅助小矩阵；两者的量化对象和开销不能互换。

算法次序依赖的是**当前已实现状态**。改变前一步后重测，比拿旧误差继续加一个补偿项更有依据；但新统计仍可能来自有限校准，局部线性拟合也不能保证下游质量。

## 来源身份

| 来源 | 固定版本 | 本页范围 |
| --- | --- | --- |
| QuantForge: Discovering Residual Decompositions for MXFP4 Post-Training Quantization（`https://arxiv.org/abs/2609.34680v1`） | arXiv:2609.34680v1，2026-09-28提交 | 正文、附录 A/B 算法及 C/D 的协议、关键消融、成本与搜索比较；TeX 与关键 PDF 公式/表格核对。未逐格重算全部附表，未取得匹配作者源码，未运行模型或搜索。 |
