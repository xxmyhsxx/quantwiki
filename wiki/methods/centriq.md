---
title: CentriQ：逐 token 均值旁路与免校准 DiT 量化
type: method
tags:
  - diffusion
  - ptq
  - rotation
  - activation-quantization
  - low-rank
sources:
  - raw/papers/2026-10-07/centriq/paper.pdf
updated: 2026-10-08
---

# CentriQ：逐 token 均值旁路与免校准 DiT 量化

CentriQ 处理扩散 Transformer（DiT）的一种激活问题：条件调制会使同一 token 的通道整体偏移，固定 Hadamard 可能将这部分均值集中到少数坐标，继续撑大量化范围。方法先移走每个 token 的均值，让中心化残差接受旋转与量化，再通过原权重的行和把均值加回输出；权重则用不依赖激活数据的稳健目标拟合。

本页依据 arXiv:2610.06260v1，2026-10-08 研读正文及附录 A–C，结合 TeX 与 PDF 核公式、表1–7和实现描述。它是预印本；当前材料没有作者实现可供本轮核读，内核部分仍是论文描述。模型质量和延迟均为作者报告，未在本项目运行。前置的线性层方向、旋转和范围选择分别见[线性层与输入通道](../fundamentals/operators/linear-layer-input-channel.md)、[正交旋转](../theory/orthogonal-rotation-and-hadamard-quantization.md)与[校准和范围选择](../theory/calibration-and-range-selection.md)。

## 1. 为什么有 LayerNorm，仍会有逐 token 均值

DiT 通过多步去噪生成图像，Transformer 块的线性层在每一步重复执行。这里量化的是这些投影的权重和在线激活，不是自回归 KV cache；W4A4 也不意味着文本编码器、VAE、归一化和所有旁路均为4位。

令线性层为 $y_t=Wx_t+b$，$W\in\mathbb R^{d_{\rm out}\times n}$，$x_t$ 为列向量。论文所讨论的 adaLN 调制输入为

$$
x_t=(1+\gamma)\odot\widetilde x_t+\beta,\qquad
\bar x_t=\frac{\mathbf1^{\mathsf T}x_t}{n}
=\bar\beta+\frac{\gamma^{\mathsf T}\widetilde x_t}{n},
$$

其中 $\widetilde x_t$ 的通道均值为零。由 timestep、prompt 决定的 $\gamma,\beta$ 可在 token 间共享，但第二项仍随 $\widetilde x_t$ 变化，因此一般不能把整个均值折成静态 bias。MLP 第二个投影接收非线性后的激活，也可能重新带有均值。（§4.1。）

把 $x_t$ 分为中心化部分 $x_t^c$ 和常量方向 $\bar x_t\mathbf1$，其 DC 能量比例为 $n\bar x_t^2/\|x_t\|_2^2$（非零 token）。DC 指通道上的常量分量，不是时间平均。即使它未占据大部分能量，集中后的单个坐标仍可能决定逐 token 网格的范围。图1在最高噪声步骤统计 token 中位数和层类型范围，支持这一特定模型观察，不等于每层、每步的均值都占主导。

归一化 Sylvester Hadamard 将 $\mathbf1$ 映成 $\sqrt n\,e_1$，所以未移除的均值会形成幅值 $\sqrt n|\bar x_t|$ 的坐标。一般因子化 Hadamard 则由行和决定这项分布；正交性只保持能量，不能保证减小峰值。论文引理1讨论行和恒定的 regular Hadamard，其阶数必须满足 $n=4u^2$；这是该结构的必要条件，不是“所有正交旋转均无法处理均值”的结论。（§3–4.1、附录 C.5。）

## 2. 中心化、padding 与均值回注

先在原生 $n$ 个通道上求均值，再中心化：

$$
x_t^c=x_t-\bar x_t\mathbf1,\qquad
r=W\mathbf1,\qquad
Wx_t=Wx_t^c+\bar x_t r.
$$

$r\in\mathbb R^{d_{\rm out}}$ 从**未量化权重**一次性在 FP32 中计算，随后以模型计算精度保存；不能由已经量化的权重行和代替。对一批 token，均值向量与 $r$ 的外积至多 rank 1，这就是论文的高精度均值分支。它不是对量化误差做 SVD，也不需要用校准激活拟合方向。（§4.2 式4、图3。）

设 $P\in\mathbb R^{n'\times n}$ 先补零，$P^{\mathsf T}P=I_n$，归一化 $H\in\mathbb R^{n'\times n'}$ 正交。在线和离线操作分别为

$$
u_t=HPx_t^c,\qquad W_R=WP^{\mathsf T}H^{\mathsf T}.
$$

作者取 $H=(H_m\otimes H_{2^k})/\sqrt{n'}$，$n'=m2^k$，$m\in\{1,12,36,44\}$；这里的两个因子为未归一化 Hadamard。能在原宽度分解就用原宽度，否则补到最小可用宽度。论文用两次小矩阵乘实现因子化变换，操作数为 $O(n'(m+2^k))$，不能直接当作所有后端都实现了 FWHT 的 $O(n'\log n')$。（附录 B.1。）

**先中心化再 padding 是算法的一部分。** 先对补零后的 $n'$ 维向量求均值，会把原均值变成 $n\bar x_t/n'$，在原通道与补零通道之间留下残差。教学例 $x=(3,3)$、补至4维：正确次序得到全零残差；反过来对 $(3,3,0,0)$ 中心化，得到 $(1.5,1.5,-1.5,-1.5)$。它仍需占用量化网格，不能和前者视为同一输入。

量化层为

$$
\widehat y_t=\widehat W_R\,Q_a(u_t)+\bar x_t r+b.
$$

去掉两侧量化时，$W_RHP= W$，上述式子恢复原线性层。固定 $W,H,P$ 后，给一个 token 的全部通道加相同常数 $c$，$x_t^c$、激活码和范围不变，理想关系为

$$
\widehat y(x_t+c\mathbf1)=\widehat y(x_t)+cr.
$$

“精确”指代数上的分解及均值绕过低位宽量化器。附录 C.5 把旁路描述为不贡献误差，但它同时说明 $r$ 以模型精度保存；FP16/BF16 存储、均值归约与 epilogue 仍有舍入。本页不据此声称有限精度计算与原模型逐位相同。中心化移除了显式的 $\bar x_tHP\mathbf1$ 项，剩余范围 $\|HPx_t^c\|_\infty$ 仍依赖 $H$ 与残差，不能把论文结论中的“independent of the rotation”理解为任意旋转效果相同。

## 3. 免校准不等于没有在线统计

激活按**当前 token 的整个旋转后向量**使用一个对称网格。记 $q_\rho(|u_t|)$ 为该向量绝对值的经验分位数，$\Phi$ 为标准正态分布函数，论文规定

$$
\widehat\sigma_t=
\frac{q_\rho(|u_t|)}{\Phi^{-1}((1+\rho)/2)},\qquad
\alpha_t=\frac{\gamma^\star\widehat\sigma_t}{2^{b_a-1}-1},
$$

$$
Q_a(u_t)=\alpha_t\,
\operatorname{clip}\!\left(
\operatorname{round}(u_t/\alpha_t),
-(2^{b_a-1}-1),\,2^{b_a-1}-1
\right).
$$

$\rho=0.9995$，表3给 $\gamma^\star=2.5591$（正文近似为2.56）；4位为 $[-7,7]$ 的15个整数值，2位为 $[-1,1]$ 的3个值。该对称网格没有使用全部 $2^{b_a}$ 个码点。$\widehat\sigma_t$ 是通过半正态分位数估计散布，不是直接计算标准差，也不是把99.95%分位数本身设为裁剪端点。（§4.2、附录 B.3、表3。）

常数来自高斯4位裁剪模型，不证明真实 DiT 激活是高斯，也不证明它对所有位宽最优。中心化的偏移不变性是代数性质；剩余分布是否适合这个尺度是另一项经验判断。全零残差的 $\alpha_t=0$ 需实现约定，论文未交代对应保护规则，本页不补写作者代码。

“免校准”在这里指不先用 prompt 数据拟合每个模型的变换和量化参数。推理时仍计算均值和分位数，权重仍有离线迭代优化；作者也用独立的1,000条 MJHQ 开发 prompt 选择设计，不能写成整个研究过程完全未使用数据。（附录 A。）

## 4. RoDS 权重路径：稳健目标优化了哪些变量

旋转后的 $W_R$ 按输出行和连续64个输入列分组。第 $i$ 行、第 $g$ 组恢复为

$$
\widehat W_{ig}=s_i\sigma_{ig}(Q_{ig}-z_{ig}),\qquad Q_{ig}\in\{0,\ldots,15\}^{64}.
$$

$\sigma_{ig}$ 固定为 min–max 初始化；跨该行所有组共用的 $s_i$、每组 $z_{ig}$ 和整数码接受拟合。目标为

$$
\min_{s,z,Q}\ \sum_{i,g,j}
\left|W_{igj}-s_i\sigma_{ig}(Q_{igj}-z_{ig})\right|^{0.7}.
$$

当残差绝对值增大时，$p|e|^{p-1}$ 在 $p=0.7$ 下减小。相比平方误差，这个目标降低大残差的边际权重，让拟合更偏向组内密集主体；它没有证明主体拟合一定改善生成质量，也不是说大残差没有代价。组尺度仍受初始极值影响，不能将其概括为每组尺度都被重新独立优化。（§4.3 式7。）

求解器交替执行两层更新：

1. 固定码和 shift，用迭代重加权最小二乘（IRLS）更新每行 $s_i$；当前残差 $e$ 的权重为 $(\epsilon+e^2)^{(p-2)/2}$，大残差对下一次加权拟合的影响较小。
2. 固定行尺度，在 $W_g/s$ 上用半二次分裂（HQS）更新 $Q_g,z_g$。引入辅助残差 $E_g$，以 $\beta\|E_g-[W_g/s-(Q_g-z_g)\odot\sigma_g]\|_F^2/2$ 把残差与重构约束连接；轮流做网格舍入、HQQ 式近端收缩近似及由均值偏移得到的 $z_g$ 更新。$\|E_g\|_p^p$ 承担稳健误差目标，二次项用于交替求解，不能把它当作最终权重 MSE 目标。

表3规定 $p=0.7,\beta=10$，外层至多3轮、每轮 IRLS 至多5步、近端迭代20步，相对停止阈值 $10^{-3}$；HQS 的收缩是一次近端近似，原问题非凸且非光滑，未声称全局最优。初始化 $s_i=1$，首个内层迭代对应 RTN。列因子在主 W4A4 设置固定为1，只在表7的 weight-only 实验启用；不能据该表把同一套自由度归给主方法。（§4.3 式8、附录 B.3。）

导出时将 $s_i\sigma_{ig}$ 合成一个16位尺度，另存16位 shift。PixArt 使用 FP16，FLUX/SANA 使用 BF16；shift 是保存为浮点的偏移，不应自行改成整数 zero-point。每组64个4位码加32位元数据，主体为 $4+32/64=4.5$ bit/旋转后权重；$r$、bias、padding 与运行缓冲另计。

## 5. 论文描述的执行与实际成本

附录 B.2 描述基于 Nunchaku 的 W4A4 实现：prologue 读取 token、求均值、中心化、因子化旋转和动态量化；主 GEMM 使用整数码；epilogue 加回均值与偏置。这是作者提供的接口描述，尚无本轮作者源码证据可确认具体 packing、归约顺序、分位数实现、累加精度和运行分派。

还需处理非对称权重的 shift。令导出尺度 $\tau_{ig}=s_i\sigma_{ig}$，将4位权重码移成有符号码 $Q'_{ig}=Q_{ig}-8$，同时令 $z'_{ig}=z_{ig}-8$。对该组激活整数码 $q_{t,g}$，

$$
\tau_{ig}(Q'_{ig}-z'_{ig})^{\mathsf T}(\alpha_t q_{t,g})
=\tau_{ig}\alpha_t(Q'_{ig})^{\mathsf T}q_{t,g}
-C_{ig}\alpha_t p_{t,g},
\quad
C_{ig}=\tau_{ig}z'_{ig},\quad
p_{t,g}=\sum_j q_{t,gj}.
$$

因而论文用激活列 $[\alpha_t p_t\mid\bar x_t\mid1]$ 和权重列 $[-C\mid r\mid b]$ 配对，把每组零点修正、均值和 bias 放入高精度旁路。只有均值部分为 rank 1；包含 $G$ 组修正和 bias 的全部辅助列是 $G+2$ 列，不能统称为只多一次 rank-1 计算。

若旋转宽度由 $n$ 变为 $n'$，低位宽矩阵乘的宽度和主体存储都相应扩大。按论文可用阶数推算，原宽2240会补至2304（$36\times64$）；这只是宽度规则的教学核算，不代表已核每层导出。4.5 bit 是旋转后权重的主体率，不能直接当成原生权重或整模型率。在线还需均值、尺度和分组码和，是否融合决定物化与访存成本。

作者在单张 A100-80GB、FLUX.1-schnell、$1024^2$、batch 1、4个去噪步骤报告：W4A4 为1.69 s/图，BF16 为1.77 s/图。附录 A 说明包含文本编码器与 VAE，2轮预热后对6张图用 CUDA events 求均值；附录 B.2 说明 Transformer 使用 CUDA Graph。按报告值计算延迟约降低4.5%，没有支持4倍端到端加速，也不是本项目测得的结果。W2A4/W4A2 均用模拟量化，没有对应速度结论。（§5.1、§7、附录 A–B。）

## 6. 实验证据怎样支持主张

主评测覆盖 PixArt-Σ、FLUX.1-schnell、SANA-1.6B，均为 $1024^2$。PixArt 与 SANA 各用20步 DPM-Solver/Flow-DPM-Solver、guidance 4.5；FLUX 为4步 flow-matching Euler、guidance 0。MJHQ-5k 与 sDCI 各5,000条 prompt，同一方法组共享 prompt 和 seed。FID 对数据集参考图，ImageReward（IR）衡量图像与提示质量；PSNR/LPIPS 对同 seed 的16位输出，衡量像素/感知保真。生成质量和与浮点模型一致性不能合成一个指标。（附录 A。）

下表摘录原表1的 MJHQ 部分。SVDQuant 行**转引其原论文**；CentriQ 的作者自行运行 RTN/NF4/HQQ，但这些主表基线只使用无变换的逐 token 激活量化。不能将这张表称为全部方法在同一运行环境中的直接复现，也不能据它把差距全部归给 RoDS。

| 模型 | 16位 FID | SVDQuant：FID / IR / LPIPS | CentriQ：FID / IR / LPIPS |
| --- | ---: | --- | --- |
| PixArt-Σ（FP16） | 16.7 | 19.2 / 0.878 / 0.323 | 19.1 / 0.891 / 0.278 |
| FLUX.1-schnell（BF16） | 19.5 | 18.3 / 0.951 / 0.258 | 18.8 / 0.939 / 0.173 |
| SANA-1.6B（BF16） | 16.0 | 19.3 / 0.935 / 0.220 | 16.2 / 1.072 / 0.172 |

CentriQ 在这组对照中 LPIPS 更低，但 PixArt/SANA 的 PSNR 仍由 SVDQuant 更高；FLUX 的 FID/IR 也不是 CentriQ 最优。作者把5k样本下低于16位参考的 FID 解释为估计噪声，不能据此宣称量化提升真实分布质量。

**组件消融更直接解释机制。** 表2在 FLUX/MJHQ 保持 RoDS 权重量化，依次加入旋转、中心化回注、分位数裁剪：

| 配置 | W4A4 FID | W4A2 FID | W2A4 FID |
| --- | ---: | ---: | ---: |
| 无变换 | 40.62 | 437.41 | 363.14 |
| 加旋转 | 25.65 | 455.30 | 35.84 |
| 再加中心化回注 | 18.46 | 444.21 | 21.81 |
| 再加分位数裁剪 | 18.80 | 35.59 | 21.56 |

中心化对 W4A4/W2A4 的改善有直接对照；W4A2 还依赖裁剪，不能把全部收益归给均值分支。W4A4 加裁剪后 FID 从18.46到18.80略升，说明并非每个组件都在每个位宽单调改善该指标。表6固定 CentriQ 激活路径后，FLUX 上 RTN/HQQ/RoDS 的 FID 为18.97/18.92/18.80；PixArt 为32.56/21.04/19.13，权重求解器的增益随模型变化。

GenEval 使用553条 prompt、每条4张图。表5中 BF16 Overall 为0.667；CentriQ 的 W4A4/W2A4/W4A2 为0.670/0.647/0.573。作者运行的 SVDQuant W4A4 为0.682，与表中转引另一预印本的0.624是两条不同来源的记录；OrbitQuant W4A4 的0.703、W2A4的0.604也为转引。因此 CentriQ 没有在所有4位指标上最佳，低于4位的质量仍有损失，且这些低位宽结果没有真实低位内核性能证据。

## 7. 与相邻路线的关系及边界

- 与 [QuaRot](quarot.md) 共享离线权重配对、在线混合的思路；CentriQ 额外隔离输入相关的通道均值。它不否定旋转能缓解坐标离群值，而是先移除一个已知会与固定基底相干的方向。
- 与 [MQuant](mquant.md) 处理旋转后集中坐标的问题有关，但保护对象不同：MQuant 讨论权重旋转带来的集中通道；此处保留每个 token 的均值及原权重行和，不能把两种旁路混用。
- 与 [QERA](qera.md) 等低秩误差修正不同，这个 rank-1 方向由 $\mathbf1$ 和 $W\mathbf1$ 决定；它精确分解原输入贡献，没有用 SVD 拟合量化残差。SVDQuant 的校准平滑与低秩分支在本页按作者对照配置说明。
- 适用证据限于所测三类 adaLN DiT。均值很小时，移除均值可能收益较小；视频、其他条件化结构、NVFP4/MXFP4 和2位真实内核未由这些结果验证。普通 Hadamard、码本、IRLS 各自的性质也不能替代端到端质量证据。

## 来源身份

| 来源 | 版本 | 实际使用范围 |
| --- | --- | --- |
| CentriQ: Calibration-Free Quantization of Diffusion Transformers via Exact Mean Centering（`https://arxiv.org/abs/2610.06260v1`） | arXiv:2610.06260v1 | 正文、附录 A–C、公式1–10、图1–3与表1–7；其他定性图不作统计证据。未核作者代码或运行模型 |
