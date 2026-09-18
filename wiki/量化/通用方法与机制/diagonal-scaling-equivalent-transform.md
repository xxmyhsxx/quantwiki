---
title: 对角缩放与等价变换
slug: diagonal-scaling-equivalent-transform
sources:
  - raw/papers/quantization/vlm/2026-05-splitq-breaking-modality-heterogeneity-low-bit/paper.pdf
  - raw/papers/quantization/vlm/2026-03-masquant-modality-aware-smoothing-quantization/paper.pdf
  - raw/papers/quantization/ptq/2024-03-affinequant-affine-transformation-quantization/paper.pdf
  - raw/papers/quantization/ptq/2024-10-flatquant-flatness-matters-llm-quantization/paper.pdf
  - raw/papers/quantization/ptq/2023-08-omniquant-omnidirectionally-calibrated-quantization/paper.pdf
  - raw/papers/foundation/2021-04-roformer-enhanced-transformer-rotary/paper.pdf
  - raw/papers/quantization/ptq/2024-04-quarot-outlier-free-4-bit-inference-rotated-llms/paper.pdf
  - raw/papers/quantization/vlm/2024-12-mbq-modality-balanced-quantization-for-large-vision-language-models/paper.pdf
  - raw/papers/quantization/2021-06-white-paper-neural-network-quantization/paper.pdf
  - raw/papers/quantization/2017-12-quantization-training-neural-networks-efficient-integer-arithmetic/paper.pdf
  - raw/papers/quantization/ptq/2022-11-smoothquant-accurate-efficient-ptq/paper.pdf
  - raw/papers/quantization/ptq/2023-06-awq-activation-aware-weight-quantization/paper.pdf
  - raw/repositories/quantization/2026-06-mit-han-lab-llm-awq-d6e797a4/source/awq/quantize/auto_scale.py
  - raw/repositories/quantization/2026-06-mit-han-lab-llm-awq-d6e797a4/source/awq/quantize/qmodule.py
  - raw/repositories/quantization/2026-06-mit-han-lab-smoothquant-c61476d/source/smoothquant/smooth.py
updated: 2026-09-15
---

# 对角缩放与等价变换

对角缩放通过改变中间特征与权重的数值范围，给量化器提供更合适的输入。变换在量化前可以保持原有函数，量化后的误差却会改变。关键问题有两个：怎样选择缩放才能改善目标，以及怎样把变换落实到计算图而不改变其他分支。

本页以 SmoothQuant v7第 3—4 页和 AWQ v6§3.2 为依据。两者是应用案例，不代表所有模型都满足相同统计或融合条件。

## 1. 变换保持了什么

设 $W\in\mathbb{R}^{C_{\mathrm{out}}\times C_{\mathrm{in}}}$，$X\in\mathbb{R}^{C_{\mathrm{in}}\times T}$。选择非零通道缩放 $s_j$ 并令 $D=\operatorname{diag}(s)$，则

$$
WX=(WD)(D^{-1}X).
$$

这个等式来自 $DD^{-1}=I$。它保持的是精确算术下未量化的线性函数；带偏置时在两边保留同一个 $b$ 即可。实际方法通常选择正缩放，使幅度调整及幂指数搜索有明确含义。

对输出通道 $o$、输入通道 $j$，操作是把 $W_{oj}$ 乘以 $s_j$，把输入 $X_{jt}$ 除以 $s_j$。一个缩放值同时作用于该输入通道连接的所有输出权重。行、列约定详见[[linear-layer-input-channel|线性层与输入通道]]。

**教学例子：**令 $W=[1,2]$、$x=[3,4]^{\mathsf T}$，原输出为 11。取 $s=[2,0.5]$ 后，$WD=[2,1]$，$D^{-1}x=[1.5,8]^{\mathsf T}$，结果仍为 11。这个例子只验证代数关系，不证明量化精度会提高。

另一类等价变换是 [[orthogonal-rotation-and-hadamard-quantization|正交旋转]]：通过混合坐标而非逐通道缩放调整分布，并利用转置作为逆变换。它保持二范数，但仍可能增大某些坐标峰值；归一化与残差的融合条件也需要另行检查。（QuaRot v2，§3.1、§3.4、§4）

## 2. 为什么量化后会不同

量化算子 $Q$ 一般不能与任意通道缩放交换：

$$
Q(WD)D^{-1}\ne Q(W).
$$

因为缩放会改变权重相对网格的位置，也可能改变同组权重共享的量化范围与步长。反向缩放还改变各通道的误差在输出中被放大的程度。基础定义见[[uniform-quantization-and-groups|均匀量化与分组]]。

在 AWQ 的分析中，若重要通道放大后组步长近似不变，反向缩放可能降低相应输出误差；若组步长明显增加，其他通道可能变差。AWQ v6 表 2 的消融正显示了这种折中。因此优化的是量化误差的分配，不能把代数等价当作量化无损的证明。

若权重和激活都量化，令 $\widetilde W=WD$、$\widetilde X=D^{-1}X$，两侧反量化误差为 $E_W,E_X$，则输出误差为 $E_W\widetilde X+\widetilde WE_X+E_WE_X$。这是对 SmoothQuant 变换的代数展开，解释了为什么仅让激活范围变小仍不够：迁移到权重侧的误差也必须考虑。

## 3. 参数融合需要什么条件

下列推导是对 SmoothQuant 式 (3) 后的融合说明及 AWQ 官方实现的展开。

**前驱是仿射变换。** 若 $X=AU+c\mathbf1^{\mathsf T}$，可以改成

$$
A'=D^{-1}A,\quad c'=D^{-1}c,\quad W'=WD.
$$

这样前驱直接产生缩放后的特征。只修改 $A$ 而忘记 $c$ 会改变函数。如果 $X$ 还被其他分支消费，就需要同步处理那些消费者，或保留原分支并显式缩放。

**前驱是带逐通道仿射参数的归一化。** 将归一化输出写为 $X=\gamma\odot N(U)+\beta$，可将输出侧参数改为 $\gamma' =\gamma\oslash s$、$\beta'=\beta\oslash s$，同时让后继权重乘回 $D$。这里调整的是归一化之后的仿射参数，不是随意修改归一化输入。AWQ 实现见 auto_scale.py:34，commit `d6e797a42b9ef7778de8ee2352116e0f48a78d61`。

**前驱包含非线性。** 对一般函数 $f$，并没有 $f(u/s)=f(u)/s$。以平方函数和 $u=2,s=2$ 为反例，左右分别为 1 和 2。不能把缩放穿过任意非线性。当前 AWQ 代码提供 `ScaledActivation`，显式计算 $f(u)/s$，再配合后继权重；这可能保留运行时操作。qmodule.py:68。

**存在残差或多消费者。** 对 $X=U+V$，若希望直接产生 $D^{-1}X$，要同时处理两条加法输入，或在加法后显式缩放；只缩放一条分支通常不等价。SmoothQuant 第 4 页也说明残差输入可能需要额外缩放。实际能否融合、融合到哪些参数以及是否增加开销，取决于具体计算图。

SmoothQuant 官方 commit `c61476d728e42ae0d8a35e7e78494edcac3237b5` 的 `smooth.py:19`提供具体例子：共享归一化输入的多个 FC 合并计算通道权重最大值，使用同一组缩放。对 Mixtral 的共享归一化输出，路由 gate 和各 expert 的相关输入投影也一起处理。这体现的是计算图一致性要求，不是可任选一个消费者修改。

## 4. 同一种变换可以服务不同目标

| 维度 | SmoothQuant 的相关设置 | AWQ 论文的相关设置 |
|---|---|---|
| 主要量化对象 | 权重和激活，W8A8 | 权重，W3/W4A16 |
| 核心困难 | 激活异常值使激活量化困难 | 重要输入通道上的权重量化误差影响输出 |
| 缩放统计 | 通道最大绝对激活与权重 | 通道平均绝对激活 |
| 缩放族 | $s_j=\max(\lvert X_j\rvert)^\alpha/\max(\lvert W_j\rvert)^{1-\alpha}$ | $s_j=(\operatorname{mean}_t\lvert X_{jt}\rvert)^\alpha$ |
| 缩放用途 | 在权重与激活之间迁移量化难度 | 权衡各通道的 weight-only 输出误差 |

公式分别来自 SmoothQuant 第 4 页式 (4)、AWQ 第 5 页式 (5)。表中只比较这两个来源的设定，不断言其他实现都使用相同公式或超参数。完整流程、代码及证据分别见 [[smoothquant|SmoothQuant]] 和 [[awq|AWQ]]。

在 SmoothQuant 中记 $a_j=\max|X_j|$、$b_j=\max|W_j|$，alpha=0.5 使变换后对应通道两侧最大值同为 $\sqrt{a_jb_j}$，不意味着所有通道同幅度或全局量化误差最优。该论文的不同模型使用不同 alpha，详见 SmoothQuant 主页面。平滑因子 $s_j$ 也不同于运行时量化步长；逐输入通道缩放和整数 GEMM 外维缩放的区别见[[quantized-matmul-scaling-execution|执行路径页]]。

[[mbq|MBQ]] 提供了另一个实例：共享线性层同时处理视觉与文本位置时，恒等式仍是同一条，但搜索目标会区分模态误差。固定实现的默认 MAE 使用回答/视觉 mask 内的加权和；这会与 token 数量共同决定候选排名。改变重构权重不等于改变等价条件，也不等于为各模态分别量化一套权重。（MBQ v2 §3；代码细节见方法页。）

## 5. 怎样判断一次等价变换是否有效

先区分三项性质：

1. **变换正确性：**关闭量化后，包含偏置与分支的计算图是否仍给出相同结果；浮点环境中使用合理误差容限，不能要求所有运算逐位相同。
2. **量化收益：**加入量化后，目标层误差与模型质量是否改善；改善某个局部代理目标不自动保证下游任务提高。
3. **执行收益：**变换是否被融合，新增缩放、元数据与反量化成本如何影响实际延迟。

这些是后续实现验证需要回答的问题，本页没有运行模型测试。未量化的恒等式是数学事实，某个模型上的精度改善和加速则需要各自的实验。

## 6. 固定 BN 融合与跨层均衡

量化白皮书 v1（W）§2.3.1、§3.2 给出另一组图变换案例。它们与前面的通道缩放共享代数基础，但具有不同的结构条件。

对 $u=Wx+b$，推理 BN 使用固定 $\mu,\sigma^2,\gamma,\beta$。令 $a=\gamma/\sqrt{\sigma^2+\epsilon}$，则

$$
\operatorname{BN}(Wx+b)=W'x+b',\qquad
W'=\operatorname{diag}(a)W,\quad b'=a\odot(b-\mu)+\beta.
$$

这是逐输出通道的仿射合并，保留已有 bias 的一般形式。BN 融合会改变权重各通道幅度，因此应对部署实际使用的融合权重建立网格。LayerNorm/RMSNorm 的统计依赖当前输入，不能把整个归一化层按此式并入固定权重；第 3 节调整归一化输出仿射参数则是另一操作。（W 式 (9)–(12)；Jacob 等 arXiv v1 §3.2、图 C.5–C.8。）

跨层均衡（CLE）考虑两层之间的逐通道正齐次非线性，$y=W_2f(W_1x+b_1)+b_2$。若每个 $s_i>0$，且 $f(su)=sf(u)$，则

$$
W'_1=S^{-1}W_1,\quad b'_1=S^{-1}b_1,\quad W'_2=W_2S
$$

保持未量化函数。ReLU 满足该条件；负缩放不满足。ReLU6 的固定阈值一般也不满足，例如 $u=8,s=2$ 时 $\operatorname{ReLU6}(u/s)=4$，而 $\operatorname{ReLU6}(u)/s=3$。若调整阈值或显式保留缩放，须另外说明实际图，不能直接套用恒等式。

设 $r_i^{(1)}$ 为第一层输出通道权重范围，$r_i^{(2)}$ 为第二层对应输入通道范围，且都为正。缩放后范围为 $r_i^{(1)}/s_i$ 与 $r_i^{(2)}s_i$，令其相等得到

$$
s_i=\sqrt{r_i^{(1)}/r_i^{(2)}},\qquad
r_i'=\sqrt{r_i^{(1)}r_i^{(2)}}.
$$

这是 W 式 (20)–(22) 的两侧范围均衡解释，不是任务损失全局最优的证明；零范围需要特殊处理。与 SmoothQuant 的区别在于这里均衡两层权重范围，而后者用激活和权重统计平滑量化难度。相似的平方根形式不等于相同算法。

W §3.2 还讨论吸收较大偏置：将通道偏置减去 $c\ge0$，后继偏置加上 $W_2c$。这要求相关输入上 $\operatorname{ReLU}(u)=\operatorname{ReLU}(u-c)+c$ 成立；充分条件是 $u\ge c$。以分布统计估计这个条件只能得到近似保持，不是全输入恒等。W 表 3 中 MobileNetV2 的 FP32 从 71.72 到加入相应变换后 71.57，已表明这组实用流程不能整体叫作严格无损。

W 表 3 的 per-tensor W8A8 基线为 0.12，CLE 后 69.91，加入偏置吸收为 70.92，per-channel 基线为 70.65。它支持特定网络下通道范围失衡的重要性；该实验不能保证任意非线性、多分支图或大模型具有同样收益。BN 统计与范围估计的关系见 [[calibration-and-range-selection|校准与范围选择]]，训练过程中统计变化的处理见 [[post-training-and-quantization-aware-training|PTQ 与 QAT]]。

## 7. 平移与注意力位置编码的融合边界

[[omniquant|OmniQuant]] v3 式 3 在缩放之外加入平移。沿用本页列 token 约定，$Y=WX+b\mathbf1^\mathsf T$、$\delta\in\mathbb R^{C_{in}}$，有

$$
X'=D^{-1}(X-\delta\mathbf1^\mathsf T),\quad
W'=WD,\quad b'=b+W\delta,\qquad W'X'+b'\mathbf1^\mathsf T=Y.
$$

若前驱为 $X=\gamma\odot N(U)+\beta$，可将其输出仿射参数改为 $\gamma'=\gamma\oslash s$、$\beta'=(\beta-\delta)\oslash s$，并同步变更所有消费者。原算子没有 bias 时，非零平移可能要求新建偏置或显式运算；能否被后端吸收须另查。“消去独立变换”不等于所有模型均无新增参数或执行成本。

注意力点积也允许成对缩放。采用单 token 列向量 $q,k\in\mathbb R^{d_h}$：$(D^{-1}q)^\mathsf T(Dk)=q^\mathsf Tk$。若在点积前还有位置编码，必须保留其位置。RoFormer v5 §3.1–3.2、式 13–16 定义 RoPE 为各二维坐标对上的位置相关旋转；单对为

$$
R_m=\begin{bmatrix}\cos(m\theta)&-\sin(m\theta)\\\sin(m\theta)&\cos(m\theta)\end{bmatrix},
\qquad (R_mq)^\mathsf T(R_nk)=q^\mathsf T R_m^\mathsf T R_n k.
$$

这里 $m,n$ 是位置，$\theta$ 是该坐标对的频率；多维由这样的旋转块组成。两位置的旋转乘积使点积包含相对位置信息，不是给投影输出加一个固定 bias。

由这个定义可自行推导：若将 $q\mapsto D^{-1}q$、$k\mapsto Dk$ 提前到 RoPE 之前，点积变为

$$
q^\mathsf T D^{-1}R_m^\mathsf T R_nDk.
$$

要对所有 $q,k$ 保持原值，需要 $D$ 与相关相对旋转交换。每个旋转坐标对共享同一个非零缩放是充分条件；任意独立通道缩放一般不满足。实际配对由模型布局决定，不能只按相邻内存下标猜测。

**教学反例：**令相对旋转 $R=\left[\begin{smallmatrix}0&-1\\1&0\end{smallmatrix}\right]$、$q=(1,0)^\mathsf T$、$k=(0,1)^\mathsf T$、$D=\operatorname{diag}(2,1)$。原点积为 $q^\mathsf TRk=-1$，提前缩放后为 $q^\mathsf TD^{-1}RDk=-0.5$。改为 $D=2I$ 则保持 -1。该反例验证融合条件，不衡量具体模型的精度影响。

RoPE 定义来自 RoFormer，上述交换条件与反例是本页推导。[[orthogonal-rotation-and-hadamard-quantization|正交旋转页]] 给出 QuaRot v2 Stage 1d 的相关实例：为避开位置编码阻碍，在 RoPE 后在线变换 Q/K。[[omniquant|OmniQuant]] 的固定代码则在投影层写入 Q/K 缩放，须结合这一条件核对，不能只凭点积处的恒等式宣称整个融合图等价。

## 8. 从对角尺度到一般可逆变换

非对角项可以混合通道，但不能直接继承逐通道尺度的全部融合性质。[[affinequant|AffineQuant]] v1 §3.3 在 W4A4 的 Norm 后限制为对角形式；完整矩阵选项需另保留在线运算。[[flatquant|FlatQuant]] v4 §3 则用两个小矩阵表示变换，并融合在线变换与量化，在更多位置承担这项计算。

共同恒等式、逆转置与可逆性/条件数的区别见 [[invertible-transforms-and-kronecker-products|可逆变换与 Kronecker 乘积]]。实际图中，FlatQuant 将一般 Q/K 配对放在 RoPE 之后；down 前的通道混合留在门控非线性之后，只将可交换的对角尺度合入 up 分支。等价变换的扩展必须同时回答“矩阵是什么”和“放在哪里”。

## 9. 多模态尺度与共享权重的约束

采用 $Y=XW$ 布局时，各模态分别有 $X^{(m)}W=(X^{(m)}S_m^{-1})(S_mW)$。各自浮点等价成立，不代表不同 $S_mW$ 能同时由同一份主体权重表示。[[masquant|MASQuant]] v1 §4 因而固定文字主体 $B=Q_w(S_tW)$，用非文字低秩分支近似 $S_mW-B$。这同时包含尺度差与文字权重量化误差，不能只补 $(S_m-S_t)W$。

共享尺度的问题也不能只看哪个模态整体幅度最大：逐 token 动态量化下，均匀放大/缩小各坐标通常会同步改变步长；真正需要检查的是跨通道相对尺度、裁剪和舍入。MASQuant 的 SQNR 推导依赖高分辨率噪声和理想平滑条件，方法页保留了条件与反例。

[[splitq|SplitQ]] v1 §4.2 则把列通道划成主体、文字特殊与视觉特殊三组，各自配对变换。所有 token 仍须累加三组乘法，不能将列分组误写成按模态丢掉其他列。组变换限制跨组混合；后续补偿分支又改变执行图，不能因浮点恒等式成立就称部署无附加成本。

