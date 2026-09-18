---
title: 正交旋转与量化：等价条件、离群值和在线代价
slug: orthogonal-rotation-and-hadamard-quantization
sources:
  - raw/papers/quantization/ptq/2024-05-spinquant-llm-quantization-learned-rotations/paper.pdf
  - raw/papers/quantization/ptq/2024-10-flatquant-flatness-matters-llm-quantization/paper.pdf
  - raw/papers/quantization/ptq/2024-04-quarot-outlier-free-4-bit-inference-rotated-llms/paper.pdf
  - raw/papers/quantization/ptq/2024-01-slicegpt-compress-large-language-models/paper.pdf
  - raw/papers/quantization/vlm/2025-02-mquant-unleashing-inference-potential/paper.pdf
updated: 2026-09-15
---

# 正交旋转与量化：等价条件、离群值和在线代价

正交旋转通过混合特征坐标改变数值分布，并在相邻运算中抵消变换，目标是让权重和激活更容易量化。它保持浮点计算等价的条件可以证明，但“变换后量化误差一定更小”不能由等价性推出。本页采用 QuaRot v2 的旋转构造、SpinQuant v4 的可学习旋转、SliceGPT v2 的计算等价条件，以及 MQuant v2 对旋转产生新离群值的分析。QuaRot 与 SpinQuant 已全文研读；SliceGPT 仍为按需局部研读。

## 1. 旋转怎样保持线性计算

采用行 token 约定：$A\in\mathbb R^{N\times d}$，$B\in\mathbb R^{d\times o}$，输出 $Y=AB$。$R\in\mathbb R^{d\times d}$ 为实正交矩阵，满足 $R^\top R=RR^\top=I$，因此

$$
AB=(AR)(R^\top B),\qquad \|aR\|_2=\|a\|_2.
$$

这类文献将正交变换统称为旋转，不要求行列式一定为 $+1$。与 [[diagonal-scaling-equivalent-transform|对角缩放]] 按通道放大或缩小不同，旋转混合坐标并保持二范数；两者都须成对抵消，不能任意穿过非线性算子。矩阵方向与 PyTorch 权重转置见 [[linear-layer-input-channel|线性层与输入通道]]。（QuaRot §3.1、§3.4、§4）

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

## 4. 离线旋转与在线旋转各处理哪里

QuaRot §4 Stage 1a 将全局残差坐标变换折叠到权重中。Stage 1b 则在 FFN 非线性或门控结果 $A$ 之后、down projection $B$ 之前，执行在线 $AH$，并离线保存 $H^\top B$。全局变换不能自动控制非线性新产生的分布，因此这一步仍有作用；权重可离线变换不等于激活变换也无运行时代价。

注意力还有两组不同等价关系：RoPE 后 Q/K 同乘头内正交矩阵，保持点积；V 的头内旋转可沿注意力加权传到输出，再由 O 投影抵消。前者由于 RoPE 边界保留在线运算，后者可吸收入 V/O 权重。QuaRot 还在线完成 O 输入的跨头 Hadamard，具体维度、缓存写入与 prefill/decode 流程见 [[quarot|QuaRot 的完整图改写]]。（QuaRot §4 Stage 1c–1d、Stage 2c。）

需要核对变换维度、归一化系数、权重转置、padding 和前后量化位置。浮点图等价后再引入量化，最后核对实际内核；这三个层次的检查见 [[quantization-error-diagnosis|量化误差诊断]]。目标论文全文研读和 Python 入口阅读已完成，不代表模型或后端复现。

同一机制用在 KV cache 上会得出不同的结构选择：[[saw-int4|SAW-INT4]] 用块对角 Hadamard 且只旋转键，块大小必须整除 head 维，并把旋转融进解码内核；其消融显示同时旋转键与值几乎没有额外收益。KV 侧的完整整理见 [[kv-cache-quantization-objects-and-granularity|KV cache 量化的对象与粒度]]。

## 5. 旋转也可能制造新的权重离群值

对归一化 $H$ 的全正首行，忽略输出侧其他变换，

$$
(HB)_{0j}=\frac1{\sqrt d}\sum_{i=0}^{d-1}B_{ij}
=\sqrt d\,\overline B_j.
$$

若 $\sqrt d|\overline B_j|>\max_i|B_{ij}|$，该列在变换后出现比原峰值更大的数。输出侧继续旋转后应检查最终权重，不能仅据这个简化式断言所有层都会出现峰值。一个观察指标是 $\sqrt{do}\|B\|_{\max}/\|B\|_F$：正交变换保持分母，却可能改变分子。（MQuant §3.2、式 6–9、附录 A.3；此处统一归一化和绝对值记法）

[[mquant|MQuant 的旋转幅度抑制]] 将这种集中通道与其余通道拆开量化，避免它决定整组的量化步长。这个反例说明：旋转是一种改变数值分布的工具，是否有利取决于原分布、所选变换、量化粒度与执行成本。

## 6. 可学习混合与正交约束的取舍

[[spinquant|SpinQuant]] v4 §3–4 保留可融合的正交结构，学习全局残差 R1 与每层头内 Value R2，以最终语言模型交叉熵选择坐标系；RoPE 后 Q/K 和 FFN 门控后的在线 Hadamard 仍固定。其主流程先 W16A4 学旋转，再用 GPTQ 量化权重。这里同时改变的是优化目标与坐标选择，不能概括为“所有 Hadamard 都可学习”，也不要求每一层张量 MSE 都下降。

[[flatquant|FlatQuant]] v4 §2–3 使用可学习可逆矩阵替代固定变换，并通过结构化两因子与融合控制开销。图 1/7 中通道包络更平，但训练目标仍是 block 输出 MSE；包络均衡不能代替量化质量验证。

正交矩阵的逆是转置且条件数为 1，一般可逆矩阵允许额外拉伸，也可能放大数值扰动。FlatQuant 的 U/V 因子正交，不意味着含可学习对角值的整个矩阵也正交；相关代数与反例见 [[invertible-transforms-and-kronecker-products|可逆变换的数值条件]]。这提供了比较固定 Hadamard 与学习变换的另一维度，不据此断言某一类普遍更优。
