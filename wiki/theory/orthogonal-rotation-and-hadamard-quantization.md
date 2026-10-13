---
title: 正交旋转与量化：等价条件、离群值和在线代价
type: concept
tags:
  - rotation
  - equivalent-transform
  - outliers
sources:
  - raw/papers/2026-09-21/resq
  - raw/repositories/2026-09-21/resq
  - raw/papers/2026-09-21/turboquant
  - raw/papers/2026-09-21/nova-kv
  - raw/papers/2026-09-22/ostquant
  - raw/papers/2026-09-21/quip
  - raw/papers/2026-09-21/quip-sharp
  - raw/papers/2026-09-21/spinquant
  - raw/papers/2026-09-21/flatquant
  - raw/papers/2026-09-21/quarot
  - raw/papers/2026-09-21/slicegpt
  - raw/papers/2026-09-21/mquant
  - raw/papers/2026-10-03/conqur
  - raw/repositories/2026-09-21/saw-int4
  - raw/papers/2026-09-21/saw-int4
  - raw/papers/2026-10-07/centriq
  - raw/repositories/2026-09-21/quip
updated: 2026-10-13
---

# 正交旋转与量化：等价条件、离群值和在线代价

正交旋转通过混合特征坐标改变数值分布，并在相邻运算中抵消变换，目标是让权重和激活更容易量化。它保持浮点计算等价的条件可以证明，但“变换后量化误差一定更小”不能由等价性推出。本页采用 QuaRot v2 的旋转构造、SpinQuant v4 的可学习旋转、SliceGPT v2 的计算等价条件，以及 MQuant v2 对旋转产生新离群值的分析。QuaRot 与 SpinQuant 已全文研读；SliceGPT 仍为按需局部研读。

## 1. 旋转怎样保持线性计算

采用行 token 约定：$A\in\mathbb R^{N\times d}$，$B\in\mathbb R^{d\times o}$，输出 $Y=AB$。$R\in\mathbb R^{d\times d}$ 为实正交矩阵，满足 $R^\top R=RR^\top=I$，因此

$$
AB=(AR)(R^\top B),\qquad \|aR\|_2=\|a\|_2.
$$

这类文献将正交变换统称为旋转，不要求行列式一定为 $+1$。与 [对角缩放](diagonal-scaling-equivalent-transform.md) 按通道放大或缩小不同，旋转混合坐标并保持二范数；两者都须成对抵消，不能任意穿过非线性算子。矩阵方向与 PyTorch 权重转置见 [线性层与输入通道](../fundamentals/operators/linear-layer-input-channel.md)。（QuaRot §3.1、§3.4、§4）

量化后使用的是 $\widehat{AR}\,\widehat{R^\top B}$，一般不再等于 $AB$。若量化误差分别为 $E_A,E_B$，则

$$
\widehat{AR}\,\widehat{R^\top B}-AB
=E_A R^\top B+AR E_B+E_AE_B.
$$

这是教学展开：变换是否有效要同时考察两侧误差及其对输出的影响，不能只展示某一侧最大值下降。

## 2. 为什么使用快速哈达玛变换

归一化 Walsh–Hadamard 矩阵定义为

$$
H_2=\frac1{\sqrt2}\begin{bmatrix}1&1\\1&-1\end{bmatrix},
\qquad H_{2^k}=H_2\otimes H_{2^{k-1}}.
$$

它是正交矩阵，元素为 $\pm1/\sqrt d$。递归加减可在 $O(d\log_2d)$ 操作内完成向量变换，比显式稠密矩阵乘法的 $O(d^2)$ 更便于在线执行。非二次幂维度需要已有小阶矩阵的 Kronecker 分解或其他经过验证的尺寸处理，不能默认所有维度使用同一个内核。（QuaRot §3.1）

随机符号对角矩阵 $D_s=\operatorname{diag}(s_i)$、$s_i\in\{-1,+1\}$ 可与 $H$ 组合，所得矩阵仍正交。符号放在左侧还是右侧应与行/列向量约定对应：例如列向量使用 $HD_s$，行向量对应 $D_sH^\top$。随机化意在降低信号与固定基底的对齐；不能把其统计性质直接套到每一个确定性在线 $H$ 上。（QuaRot §3.1；MQuant §3.2、附录 A.3）

**教学例子：**$(4,0)H_2=(2\sqrt2,2\sqrt2)$，峰值降低；但 $(1,1)H_2=(\sqrt2,0)$，峰值反而增大。二范数都不变，所以保持能量与分散峰值是两个问题。

[激活引导补偿与正交残差](../methods/activation-guided-compensation.md) 进一步用持续离群通道的相干叠加解释固定与随机 Hadamard 的区别，并给出采样候选集合的概率界。其实际选择依据是验证 PPL，而非直接最小化理论离群项；无裁剪假设与实验裁剪的差别也在该页保留。

## 3. 归一化和残差为什么重要

先去掉可学习仿射参数，定义一行特征的归一化

$$
\operatorname{RMS}_0(x)=\frac{x}{\sqrt{\|x\|_2^2/d+\epsilon}}.
$$

因为正交变换不改变分母，$\operatorname{RMS}_0(xR)=\operatorname{RMS}_0(x)R$。于是相邻块可在旋转坐标中传递残差：输入投影改为 $R^\top B_{in}$，输出投影改为 $B_{out}R$，输出偏置也旋转；embedding、残差各支路和最终 head 必须配套变换。非线性内部输入保持原值，块输出整体旋转，最终 head 抵消旋转。（SliceGPT §3.1、定理 1、附录 A.1；QuaRot §3.4）

带可学习逐通道增益的 RMSNorm 不能直接交换任意 $R$。需要先把增益折叠进后续线性权重；偏置存在时也要保留相应线性项。

LayerNorm 还多一步减均值。令 $C=I-\frac1d\mathbf1\mathbf1^\top$，则采用相同 $\epsilon$ 时

$$
\operatorname{LN}(x)=\operatorname{RMS}_0(xC)\operatorname{diag}(\gamma)+\beta.
$$

$xC$ 是去均值后的向量。一般 $CR\ne RC$，因此不能直接把 LayerNorm 改名为 RMSNorm 后旋转。SliceGPT §2.1、§3.2 将中心化与仿射操作按计算图折叠到相邻线性层，连同 embedding、输出层和残差一起处理。其原文用单位二范数归一化，本页把常数 $\sqrt d$ 放回常用 RMS 定义，并显式保留 $\epsilon$。

理解 Pre-LN 的一个无仿射简化：若原块为 $x+f(\operatorname{LN}_0(x))$，以中心化状态 $z=xC$ 表示，下一状态可写为 $z+f(\operatorname{RMS}_0(z))C$；输出投影和偏置吸收右侧 $C$。这依赖后续确实只需中心化状态，边界输出仍要对应原图。MQuant 附录 A.14 进一步讨论其视觉编码器中的 Post-LN 结构：中心化、增益以及残差支路位置不同，不能直接复制 Pre-LN 的融合步骤。

### 输入相关的均值为什么需要旁路

归一化之后的条件调制或非线性可以重新引入逐 token 均值。[CentriQ](../methods/centriq.md) v1 §4.1–4.2 用 DiT 的 adaLN 说明：通道增益和偏移即使由所有 token 共享，乘上不同的归一化 token 后，均值仍可能随 token 变化，不能将整个贡献折成一份静态 bias。

沿本页行 token 约定，令 $C=I-\mathbf1\mathbf1^{\mathsf T}/d$、$\mu=X\mathbf1/d$，则

$$
XB=(XCR)(R^{\mathsf T}B)+\mu(\mathbf1^{\mathsf T}B).
$$

$C$ 是去均值投影，具有零空间，不是可逆旋转；最后的外积把被移出的方向补回，批量更新至多 rank 1。将 $\mathbf1^{\mathsf T}B$ 从原权重预计算并保留在高精度分支，可让中心化残差独立接受旋转与量化。这个分解不需要从数据拟合低秩方向，也不等于对量化误差做 SVD。

固定 Sylvester Hadamard 会把常量方向集中到一个坐标；移走该方向可以消除显式均值项，但剩余残差的峰值仍依赖旋转。精确算术下的分解、输入整体偏移的不变性，以及 FP16/BF16 旁路的存储/计算舍入须分开。含 padding 时还要在原生通道上先中心化，再补零，具体机制与反例见 [CentriQ 的均值回注](../methods/centriq.md#2.%20%E4%B8%AD%E5%BF%83%E5%8C%96%E3%80%81padding%20%E4%B8%8E%E5%9D%87%E5%80%BC%E5%9B%9E%E6%B3%A8)。（§4.2、附录 B.1、C.5。）

## 4. 离线旋转与在线旋转各处理哪里

QuaRot §4 Stage 1a 将全局残差坐标变换折叠到权重中。Stage 1b 则在 FFN 非线性或门控结果 $A$ 之后、down projection $B$ 之前，执行在线 $AH$，并离线保存 $H^\top B$。全局变换不能自动控制非线性新产生的分布，因此这一步仍有作用；权重可离线变换不等于激活变换也无运行时代价。

注意力还有两组不同等价关系：RoPE 后 Q/K 同乘头内正交矩阵，保持点积；V 的头内旋转可沿注意力加权传到输出，再由 O 投影抵消。前者由于 RoPE 边界保留在线运算，后者可吸收入 V/O 权重。QuaRot 还在线完成 O 输入的跨头 Hadamard，具体维度、缓存写入与 prefill/decode 流程见 [QuaRot 的完整图改写](../methods/quarot.md)。（QuaRot §4 Stage 1c–1d、Stage 2c。）

需要核对变换维度、归一化系数、权重转置、padding 和前后量化位置。浮点图等价后再引入量化，最后核对实际内核；这三个层次的检查见 [量化误差诊断](../implementation/quantization-error-diagnosis.md)。

同一机制用在 KV cache 上会得出不同的结构选择：[SAW-INT4](../methods/saw-int4.md) 用块对角 Hadamard，默认在 K/V 中只旋转 K，decode 的 Q 也配套旋转；块大小必须整除 head 维，并把旋转融进内核。其所测消融中同时旋转 K/V 没有改善均分。KV 侧的完整整理见 [KV cache 量化的对象与粒度](kv-cache-quantization-objects-and-granularity.md)。

同一理想Hadamard也不保证不同浮点执行顺序得到相同码。[SAW所选融合writer](../methods/saw-int4.md#4.2%20%E5%86%99%E5%85%A5%EF%BC%9A%E5%85%88%E6%A0%B8%E8%88%8D%E5%85%A5%EF%BC%8C%E5%86%8D%E6%A0%B8%E7%A0%81%E4%B8%8E%E5%85%83%E6%95%B0%E6%8D%AE)先在FP32中预缩放并作FWHT，旋转结果经BF16舍入后才取范围；邻接未融合支路先做BF16预缩放。所选GQA decode对Q采用配套FP32预缩放/FWHT后回BF16，并在每个非空split各自计算；旋转已融合，不表示每请求只计算一次。精确算术下的正交点积关系与这些舍入、量化和调度条件应分开核对。（`fused_hadamard_int4_kv.py:150–191`；`memory_pool.py:1161–1189`；`decode_attention.py:970–998`。）

同样的旋转坐标也可以在读取端还原。[SAW 所选 FA3 prefill](../methods/saw-int4.md#4.5%20FA3%20prefill%EF%BC%9A%E4%BB%8E%E5%88%86%E9%A1%B5%20INT4%20%E6%81%A2%E5%A4%8D%E5%88%B0%E5%8E%9F%20Q%20%E5%9D%90%E6%A0%87)把量化 K 恢复为 FP32 行 $r$，在 FP32 内逆 Hadamard 后转 BF16，配原 Q；decode 则用旋转后的 Q 配旋转坐标内的 K。归一化对称 Hadamard 满足 $(QH)r^{\mathsf T}=Q(rH)^{\mathsf T}$，说明理想点积为何相同；恢复与逆旋转之间是否先舍入、Q 在何处舍入，则决定实际低精度路径。这里没有证明两个 attention 实现逐位相同。（`flatten_kv_cache.py:190–219`；`decode_attention.py:970–998,1047–1070`。）

### 4.1 先恢复浮点权重与保留量化坐标分别意味着什么

本小节采用列输入约定，令 $W\in\mathbb R^{m\times n}$，U/V为输出/输入维正交矩阵，A为正的可逆对角尺度。处理坐标权重为 $\widetilde W=UWA V^\top$，量化近似为 $Q(\widetilde W)$。两种执行位置的理想关系为

$$W_{restored}=U^\top Q(\widetilde W)VA^{-1},\qquad
W_{restored}x=U^\top Q(\widetilde W)(VA^{-1}x).$$

左边可以离线恢复整个浮点矩阵，继续让普通Linear读取；右边可以保留处理坐标的低位载荷及恢复元数据，在推理时变换输入/输出。实数等价不定义存储相同，也不保证有限精度求和/舍入相同。恢复后的每个元素是多个码值的组合，通常不再位于原小网格；W2量化近似的质量证据不能直接说明保存文件或GPU权重只占2bit。

误差统计也必须成对换坐标。若 $E_t=Q(\widetilde W)-\widetilde W$，原输入二阶矩为H，处理统计为 $H_t=VA^{-1}HA^{-1}V^\top$，则

$$\operatorname{tr}(E_rHE_r^\top)=\operatorname{tr}(E_tH_tE_t^\top),
\qquad E_r=U^\top E_tVA^{-1}.$$

用处理残差E_t配原统计H一般不满足该关系。输出U的正交性和A的可逆性都是前提；阻尼/归一化及FP16恢复也需另记，不把代数恒等式称为逐bit实现证明。

固定QuIP的OPT/Balance路径先postproc还原稠密浮点权重，再保存普通state_dict；而error_compute的局部残差仍在处理坐标。这说明需要沿实际对象/保存/消费路径核对，不能从有quantizer或pack函数推出已交付压缩接口。（固定QuIP（`https://github.com/Cornell-RelaxML/QuIP/tree/ac92cfc7a22f6100009e2caf53bb72257d3f3184`），method.py的preproc/postproc/error_compute、bal.py的fasterquant、opt.py的save/load；[所选实现](../implementation/quip-ldlq-and-coordinate-restoration.md)。）

## 5. 旋转也可能制造新的权重离群值

对归一化 $H$ 的全正首行，忽略输出侧其他变换，

$$
(HB)_{0j}=\frac1{\sqrt d}\sum_{i=0}^{d-1}B_{ij}
=\sqrt d\,\overline B_j.
$$

若 $\sqrt d|\overline B_j|>\max_i|B_{ij}|$，该列在变换后出现比原峰值更大的数。输出侧继续旋转后应检查最终权重，不能仅据这个简化式断言所有层都会出现峰值。一个观察指标是 $\sqrt{do}\|B\|_{\max}/\|B\|_F$：正交变换保持分母，却可能改变分子。（MQuant §3.2、式 6–9、附录 A.3；此处统一归一化和绝对值记法）

[MQuant 的旋转幅度抑制](../methods/mquant.md) 将这种集中通道与其余通道拆开量化，避免它决定整组的量化步长。这个反例说明：旋转是一种改变数值分布的工具，是否有利取决于原分布、所选变换、量化粒度与执行成本。

## 6. 可学习混合与正交约束的取舍

[SpinQuant](../methods/spinquant.md) v4 §3–4 保留可融合的正交结构，学习全局残差 R1 与每层头内 Value R2，以最终语言模型交叉熵选择坐标系；RoPE 后 Q/K 和 FFN 门控后的在线 Hadamard 仍固定。其主流程先 W16A4 学旋转，再用 GPTQ 量化权重。这里同时改变的是优化目标与坐标选择，不能概括为“所有 Hadamard 都可学习”，也不要求每一层张量 MSE 都下降。

[FlatQuant](../methods/flatquant.md) v4 §2–3 使用可学习可逆矩阵替代固定变换，并通过结构化两因子与融合控制开销。图 1/7 中通道包络更平，但训练目标仍是 block 输出 MSE；包络均衡不能代替量化质量验证。

正交矩阵的逆是转置且条件数为 1，一般可逆矩阵允许额外拉伸，也可能放大数值扰动。FlatQuant 的 U/V 因子正交，不意味着含可学习对角值的整个矩阵也正交；相关代数与反例见 [可逆变换的数值条件](../fundamentals/mathematics/invertible-transforms-and-kronecker-products.md)。这提供了比较固定 Hadamard 与学习变换的另一维度，不据此断言某一类普遍更优。

## 7. QSUR 的椭球解释与白化边界

[OSTQuant](../methods/ostquant.md) v1 §3 用置信椭球与量化立方体的体积比 QSUR 解释正交加缩放。它关注分布如何占用网格，但原文式 3–7 用主轴端点代表坐标极值，不能当成一般精确公式。下面按其零均值高斯椭球模型作独立教学推导，避免把这个指标直接当量化误差或训练目标。

令 $\Sigma\succ0$，椭球为 $E=\{\Sigma^{1/2}u:\|u\|_2\le\sqrt c\}$，$c>0$ 是置信水平对应常数。第 $j$ 个坐标的极值是 $\pm\sqrt{c\Sigma_{jj}}$，因为 $e_j^{\mathsf T}\Sigma^{1/2}u$ 的最大值由两个向量对齐取得。采用所有坐标共用的对称范围，立方体边长为 $2\sqrt{c\max_j\Sigma_{jj}}$。记单位球体积 $v_d=\pi^{d/2}/\Gamma(d/2+1)$，则

$$\mathrm{QSUR}_{ellipsoid}=\frac{v_d\sqrt{\det\Sigma}}{2^d(\max_j\Sigma_{jj})^{d/2}}.$$

**主轴端点反例。**取二维特征值 $(4,1)$，特征向量为 45 度旋转。此时 $\Sigma_{11}=\Sigma_{22}=2.5$，$c=1$ 时坐标极值为 $\sqrt{2.5}$；只取长轴端点得到 $\sqrt2$，会低估范围。真实体积比为 $\pi/5\approx0.62832$，该端点近似给出 $\pi/4\approx0.78540$。误差不是由浮点舍入造成。

正交 $R$ 保持 $\det\Sigma$ 和谱，只能改变分母中的坐标方差。若 $\Sigma=U\Lambda U^{\mathsf T}$ 且维度支持归一化 Hadamard $H$，列向量变换 $T=HU^{\mathsf T}$ 得到 $T\Sigma T^{\mathsf T}=H\Lambda H^{\mathsf T}$，各对角值均为 $\operatorname{tr}(\Sigma)/d$。任意正交变换后最大对角值至少为此均值，所以这一构造在上述零均值椭球、共享范围口径下达到最优正交方向。它均衡坐标方差，但不改变特征值，也未把各向异性高斯变为球形高斯。行 token 的变换需转置。

进一步允许缩放，可取 $T=\Lambda^{-1/2}U^{\mathsf T}$，得到白化协方差 $I$。由正定矩阵的行列式不超过对角元素之积，

$$\det\Sigma\le\prod_j\Sigma_{jj}\le(\max_j\Sigma_{jj})^d,$$

可得这个体积比不超过 $v_d/2^d$，白化达到该值。这给 OSTQuant v1 式 8、附录 A.2.3 的白化结论一个条件明确的证明，不能顺带证明其原式 7 的一般正确性。

边界也明确：非零均值和范围定义会改变分母；奇异经验协方差使全维体积为零且无法直接白化；小特征值求逆可能放大扰动。真实模型的重尾、不同量化组、有限网格相位、裁剪、逆变换后权重和多分支共享都未被这个单分布体积比控制。保持一侧 QSUR 高，不代表另一侧量化容易或输出误差小；OSTQuant 实际用整网 KL-Top 学习，而非直接最大化 QSUR。

## 非相干性与码本的进一步联系

[QuIP](../methods/quip.md) 进一步区分权重元素的集中程度与 Hessian 特征向量相对坐标轴的集中程度：正交变换保持谱，却改变二阶误差反馈可以利用的坐标结构。这不能简单替换成“激活最大值下降”，相关定义、谱界和有限网格限制由方法页展开。

[QuIP#](../methods/quip-sharp.md) 将 RHT 与 8 维 E8P 码本结合：变换让权重分布更适合固定码本，规则码本又降低解码存储成本。微调将符号向量放松为实数后，最终参数不再自动满足初始随机正交理论；理解时应分别保留变换保证与微调实证。

## 8. 球面分布与查询加权距离是不同目标

[TurboQuant](../methods/turboquant.md) 用 Haar 随机正交变换把固定单位输入变为均匀球面点，单坐标均值零、方差 $1/d$；有限维坐标受总平方和为 1 的约束，并非独立。单个固定 Hadamard 或随机符号 Hadamard 不自动具有完全相同的边缘分布。因此“保持范数”“减小峰值”“产生特定随机分布”是三种不同主张。

[NOVA-KV](../methods/nova-kv.md) 则希望重构误差按 query 二阶矩 $M_q$ 计费。行向量变换满足 $RR^\top=M_q$ 时，$\|eR\|_2^2=eM_qe^\top$；它一般不保持欧氏范数，却把指定 logit 误差精确变成变换域 MSE。$M_q\succ0$、$R$ 可逆时，K 乘 $R$、Q 乘 $R^{-\top}$ 保持未量化内积；退化统计需先说明阻尼或特征值下限后的目标。精确的代数恒等式与高分辨率最优性、有限精度实现、实际任务质量仍应分开验证。

## 9. 从激活几何目标到闭式旋转更新

[ConQuR](../methods/conqur.md) v2 §3、附录 B 提供另一种选择坐标系的目标。对单位列向量 $\tilde x$，正交旋转保持二范数，峰值至少为 $1/\sqrt d$；所有坐标幅值相等的超立方体顶点达到这个下界。无裁剪逐向量均匀量化的舍入界受峰值或带符号范围控制，因此靠近顶点可作为几何代理，但并非实际量化误差或下游任务的精确目标。

令当前最近顶点为 $z_i=\operatorname{sign}(R\tilde x_i)/\sqrt d$，在零坐标处选一个等距顶点。最小距离可写为

$$
\sum_i\|R\tilde x_i-z_i\|_2^2
=2n-\frac2{\sqrt d}\sum_i\|R\tilde x_i\|_1.
$$

它等价于最大归一化激活的 $\ell_1$ 范数，不是处处光滑目标，也不保证同一个 R 能把任意数据集全部变成等幅向量。共享坐标系需要在不同样本方向间取舍。

固定顶点后，将激活和目标按行堆叠为 $\tilde X,Z$，问题成为 $\min_{R\in O(d)}\|\tilde X R^\top-Z\|_F^2$。若 $Z^\top\tilde X=U\Sigma V^\top$，则 $R=UV^\top$ 是该正交 Procrustes 子问题的解。交替选择顶点与求解 R 可在固定数据集上降低代理目标；逐批更换数据之后，以及量化后的整网任务，仍需独立检验。

它和 SpinQuant 的差别不只是优化器：前者选择激活方向的几何目标，后者以最终模型交叉熵选择旋转。ConQuR 的共享残差 R1 与头内 R2 可折入权重，固定 R3/R4 仍有推理时执行；按批次更新的“在线校准”发生在离线 PTQ 阶段。完整位置、SVD 方向、量化感知前向与成本口径见[方法页](../methods/conqur.md)。

## 10. 先选高精度子空间，再在组内旋转

[ResQ](../methods/resq.md)提供一种将精度分配与坐标变换联用的方式。令 $X\in\mathbb R^{n\times d}$ 为校准激活，$P=[P_l\ P_h]\in\mathbb R^{d\times d}$ 正交，$P_h$ 有 $r$ 列、$P_l$ 有 $d-r$ 列；$R_l,R_h$ 为各组宽度对应的方阵正交变换。使用 $U_l=P_lR_l$、$U_h=P_hR_h$ 后，$\|XU_h\|_F^2=\|XP_h\|_F^2$：组内旋转改变各坐标分布，但不改变受保护子空间或其总能量。任意跨组旋转则不保留这一指定分配。

假设两组量化近似为 $A_l=Q_l(XU_l)$、$A_h=Q_h(XU_h)$，分别具有 $n\times(d-r)$、$n\times r$ 形状。旋转坐标应横向拼接为 $[A_l\ A_h]$，原坐标恢复为 $\widehat X=A_lU_l^\top+A_hU_h^\top$。定义 $E_l=XU_l-A_l$、$E_h=XU_h-A_h$，有

$$\|X-\widehat X\|_F^2=\|E_l\|_F^2+\|E_h\|_F^2.$$

这是正交子空间的平方和，不能写成范数直接相加。例如两个误差范数为3、4，总范数为5；同理，低组能量可以用平方能量相减，低组范数却不能写成总范数减高组范数。trace 恒等式也是 $\|XP_h\|_F^2=\operatorname{tr}(P_h^\top X^\top XP_h)$，不省略平方。

高组怎样选择，需要先明确目标。对固定校准矩阵，取 $G=X^\top X$ 的最大 $r$ 个特征值方向，使 $\|XP_h\|_F^2$ 最大。若定义平方能量代理

$$J(P_h)=a\|XP_l\|_F^2+b\|XP_h\|_F^2
=a\|X\|_F^2-(a-b)\|XP_h\|_F^2,$$

其中固定系数满足 $a>b\ge0$，这个选择也使 $J$ 最小；最大保留能量等于最大的 $r$ 个特征值之和。$a=b$ 时代理不区分子空间，$a<b$ 时推荐方向反转。这是整理者对**指定能量代理**的推导，没有把动态范围、舍入、两侧权重量化或整网任务误差变成固定系数，也不修复ResQ定理4.2原证明中的错误范数等式。

未中心化二阶能量与中心化方差还会选择不同方向。令 $X$ 三行为 $(10,1),(10,-1),(10,0)$，则 $X^\top X=\operatorname{diag}(300,2)$，高能量方向为第一维；减均值后Gram为 $\operatorname{diag}(0,2)$，最大方差方向为第二维。ResQ所选代码累计前者。统计量的语义与归约条件见[Gram、二阶矩与协方差](../fundamentals/mathematics/invertible-transforms-and-kronecker-products.md#Gram%E3%80%81%E4%BA%8C%E9%98%B6%E7%9F%A9%E4%B8%8E%E5%8D%8F%E6%96%B9%E5%B7%AE)，实际基底和投影消费见[实现页](../implementation/resq-basis-and-fake-quantization.md)。

最后，组内随机旋转保持能量不代表每个样本峰值都下降，也不使有限维数据严格高斯。选子空间、改善分布、实际量化误差与任务质量是需要分别核验的结论。（ResQ v2 §4.1–4.2、附录A；固定 `get_basis.py` 的 `full_shared` 分支。）

## 来源身份

| 来源 | 版本或快照 | 说明 |
| --- | --- | --- |
| ResQ（`https://arxiv.org/abs/2412.14363v2`） | arXiv:2412.14363v2 | 高精度子空间、平方能量代理与原证明边界 |
| project-resq（`https://github.com/utkarsh-dmx/project-resq/tree/cb87d0d059455858daaa66f7410da4afebee978a`） | `cb87d0d059455858daaa66f7410da4afebee978a` | 所选未中心化Gram与共享基底 |
| CentriQ（`https://arxiv.org/abs/2610.06260v1`） | arXiv:2610.06260v1 | adaLN 均值、中心化投影、rank-1 回注及 padding/精度边界 |
| SAW-INT4: System-Aware 4-Bit KV-Cache Quantization for Real-World LLM Serving（`https://arxiv.org/abs/2604.19157v1`） | `arXiv:2604.19157v1` | K-only块旋转与服务条件；本页采用既有方法页研读 |
| jindajia/sglang-fork（`https://github.com/jindajia/sglang-fork/tree/0fcc241961f9c79c27f6bad9a456bf10c8554a84`） | `0fcc241961f9c79c27f6bad9a456bf10c8554a84` | 所选SAW融合writer/Q读取及prefill逆旋转的预缩放与舍入顺序 |
| ConQuR: Corner Aligned Activation Quantization via Optimized Rotations for LLMs（`https://arxiv.org/abs/2605.10793v2`） | `arXiv:2605.10793v2` | 顶点距离、正交 Procrustes 与逐批校准；几何代理的适用边界 |
| OstQuant: Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for Better Distribution Fitting（`https://arxiv.org/abs/2501.13987v1`） | `arXiv:2501.13987v1` | QSUR 动机及需修正的椭球范围推导 |
| 固定QuIP实现（`https://github.com/Cornell-RelaxML/QuIP/tree/ac92cfc7a22f6100009e2caf53bb72257d3f3184`） | `ac92cfc7a22f6100009e2caf53bb72257d3f3184` | 所选预处理/postproc、误差统计与默认浮点保存；§4.1等式为教学展开 |
| QuIP: 2-Bit Quantization of Large Language Models With Guarantees（`https://arxiv.org/abs/2307.13304v2`） | `arXiv:2307.13304v2` | — |
| QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks（`https://arxiv.org/abs/2402.04396v2`） | `arXiv:2402.04396v2` | — |
| SpinQuant: LLM quantization with learned rotations（`https://arxiv.org/abs/2405.16406v4`） | `arXiv:2405.16406v4` | — |
| FlatQuant: Flatness Matters for LLM Quantization（`https://arxiv.org/abs/2410.09426v4`） | `arXiv:2410.09426v4` | — |
| QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs（`https://arxiv.org/abs/2404.00456v2`） | `arXiv:2404.00456v2` | — |
| SliceGPT: Compress Large Language Models by Deleting Rows and Columns（`https://arxiv.org/abs/2401.15024v2`） | `arXiv:2401.15024v2` | — |
| MQuant: Unleashing the Inference Potential of Multimodal Large Language Models via Full Static Quantization（`https://arxiv.org/abs/2502.00425v2`） | `arXiv:2502.00425v2` | — |
| TurboQuant（`https://arxiv.org/abs/2504.19874v1`） | arXiv:2504.19874v1 | 引理 1、算法 1；Haar 球面边缘与标量码本 |
| NOVA-KV（`https://arxiv.org/abs/2608.04074v1`） | arXiv:2608.04074v1 | §3、附录 F；查询加权变换及满秩条件 |

## 教学计算材料

保留已有教学计算脚本及当时结果，供核对推导与反例；这些材料不代表模型复现或性能实验。

- [algebra-check.json](../assets/orthogonal-rotation-and-hadamard-quantization/checks/algebra-check.json)
- [verify_algebra.py](../assets/orthogonal-rotation-and-hadamard-quantization/checks/verify_algebra.py)
