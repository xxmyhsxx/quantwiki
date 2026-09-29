---
title: RGSQ：模态曲率、加权重构与稀疏旋转的边界
type: method
tags:
  - vlm
  - sensitivity
  - reconstruction
  - rotation
sources:
  - raw/papers/2026-09-23/rgsq/paper.pdf
  - raw/repositories/2026-09-23/rgsq/source/riemannian_quant/metrics.py
  - raw/repositories/2026-09-23/rgsq/source/riemannian_quant/whitening.py
  - raw/repositories/2026-09-23/rgsq/source/riemannian_quant/rotation.py
  - raw/repositories/2026-09-23/rgsq/source/riemannian_quant/quantizer.py
  - raw/repositories/2026-09-23/rgsq/source/riemannian_quant/quant_layers.py
updated: 2026-09-28
---

# RGSQ：模态曲率、加权重构与稀疏旋转的边界

RGSQ 用视觉、文本各自的激活与梯度二阶统计构造加权误差，再搜索稀疏 Givens 旋转，让量化扰动避开代价较高的方向。可复用的核心是“输入激发什么方向、输出误差在哪些方向更昂贵”的双侧度量，而不是把所有已有 PTQ 概括为各向同性。

本页依据 arXiv:2609.25492v1 正文和固定代码 `a69e2d59b544f8a673483d7a197ac27db72428b1` 的定向阅读。原文存在向量化顺序、量化集合等价性及实验表格一致性问题，下文分别保留。代码阅读没有验证模型质量、低比特导出或实际内核性能。

## 1. 从模态加权走到方向加权

对列 token 线性层 $Y=WX$，$W\in\mathbb R^{d_o\times d_i}$，取输出梯度 $G=\partial L/\partial Y$。视觉和文本由 token mask 分开。对模态 $m$ 的 $N_m$ 个位置，论文式 8–11 定义

$$A_m=X_mX_m^\top/N_m,\qquad S_m=G_mG_m^\top/N_m,$$

$$\alpha_m=\mathbb E_{\mathrm{batch}}\left[\frac1{N_m}\sum_{t\in m}\|g_t\|_1\right],
\qquad \pi_m=\frac{\alpha_m}{\sum_n\alpha_n+\epsilon},$$

$$A=\sum_m\pi_m A_m,\qquad S=\sum_m\pi_m S_m.$$

$A$ 是输入侧 $d_i\times d_i$ 二阶矩，$S$ 是输出侧 $d_o\times d_o$ 梯度二阶矩；都未减去均值，不能默认是中心化协方差。梯度仍需反向传播，PTQ 不等于纯前向校准。$\epsilon>0$ 时权重和略小于 1；PSD 性只需非负权重，不依赖和严格等于 1。

模态内平均避免单凭 token 数多就提高 $\alpha_m$，但数据组成改变仍会改变激活和梯度。论文把 $\pi_m\propto\alpha_m N_m$ 称为 token-count–balanced 变体；按其公式它重新引入 token 数权重，并不自动保护少数模态。固定代码 `metrics.py:compute_modality_weights` 的 `balanced` 分支确实执行乘 $N_m$，不能按名称反推另一种算法。

先混合因子再做 Kronecker，与先做各模态 Kronecker 再混合不同：

$$\left(\sum_m\pi_m A_m\right)\otimes\left(\sum_n\pi_n S_n\right)
=\sum_{m,n}\pi_m\pi_n(A_m\otimes S_n).$$

存在 $m\ne n$ 交叉配对，因此这是作者选定的可分代理，不是模态 Fisher 混合的恒等式。经验梯度外积、真正模型 Fisher、任务 Hessian 也不能直接等同；推导与反例见 [曲率加权误差](../theory/curvature-weighted-quantization-error.md)。

## 2. 什么叫双侧误差，原文公式怎样修正

对 $E=W-\widehat W$，定义

$$D(E)=\operatorname{tr}(S E A E^\top)=\|S^{1/2}EA^{1/2}\|_F^2.$$

$A$ 强调常被输入激发的误差方向，$S$ 强调对校准损失更敏感的输出方向。例如 $S=I$、$A=XX^\top/N$ 时，$D=\|EX\|_F^2/N$，已经是各向异性的输入加权重构。故原文表 I 的“Euclidean PTQ 等于所有参数度量为 $I$”不能用来描述 GPTQ 的完整目标。

**原文式 12–14 的记号冲突。** 式 14 使用列优先恒等式 $\operatorname{vec}(UEV)=(V^\top\otimes U)\operatorname{vec}(E)$；依此应有

$$D(E)=\operatorname{vec}_{col}(E)^\top(A\otimes S)\operatorname{vec}_{col}(E).$$

论文最后却写为 $S\otimes A$。后者适用于行优先展开，但须同时修改 vec 恒等式。本页保留无歧义的 trace/Frobenius 形式，代码 `compute_whitened_error` 也是左右矩阵相乘。这里修正的是向量化约定，不是推翻双侧加权目标。

若经验因子奇异，$D$ 是半范数，不能直接宣称非退化黎曼度量。加 $\delta I$ 后可求逆；但这会改变误差代价，尤其是未观测方向。固定代码还可能裁剪特征值、使用对角因子或把 $S$ 视作单位阵，不能把所有配置都当作完整双侧曲率。

## 3. 白化使范数等价，不自动使量化问题等价

定义 $T(W)=S^{1/2}WA^{1/2}$。对任何固定候选 $\widehat W$，

$$D(W-\widehat W)=\|T(W)-T(\widehat W)\|_F^2$$

是精确恒等式。正定时若原量化可行集合为 $\mathcal Q$，正确的变量替换为

$$\min_{\widehat W\in\mathcal Q}D(W-\widehat W)
=\min_{Z\in T(\mathcal Q)}\|T(W)-Z\|_F^2.$$

原文式 15 之后从这个范数等价跳到“任意欧氏量化器可直接套用”。缺少的条件是量化器必须在 **变换后的集合 $T(\mathcal Q)$** 内求解。一般稠密 $S^{1/2},A^{1/2}$ 会把逐元素或分组整数网格变成耦合格点集合，普通 RTN、GPTQ、AWQ 并未自动实现该集合。

**教学反例。** 令右因子 $A^{1/2}=\operatorname{diag}(1,2)$、$S=1$，原网格为共享步长 1 的整数向量。原可行点 $\widehat W=(0,1)$ 在变换后是 $(0,2)$；变换空间普通整数点 $(0,1)$ 映回却是 $(0,1/2)$，不在原网格。换成更一般的稠密因子，还会混合坐标。允许另一套通道尺度会改变这个例子的可行集合，但不能使任意变换天然兼容既定部署格式。

所以应区分两种使用：用 $D$ 给原网格候选打分，可保留该网格；直接量化 $T(W)$ 再逆变换，得到的是另一种表示或浮点模拟权重，需重新说明存储、反变换和内核。白化在这里是度量欧氏化，不是把原激活协方差变成单位阵的同一个操作。

## 4. 稀疏旋转如何改变误差方向

论文式 19–23 对输入通道用正交 $R$：$X'=R^\top X$、$W'=WR$，保持 $WX=W'X'$。$R$ 为 $K$ 层不重叠二维 Givens 旋转的乘积，每对通道用

$$G(\theta)=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}.$$

改变坐标后应同步更新 $A'=R^\top A R$。对固定误差随坐标一起变换，度量代价保持一致；收益来自 **重新量化产生不同误差**，不是仅旋转同一个误差就降低真实代价。

算法 2 以 $h_{m,u}=(A_m)_{uu}\operatorname{tr}(S_m)/d_o$ 构造通道分数，按模态权重融合，再对每组通道对贪心选择角度。候选集合为 $\{0,\pi/16,\ldots,\pi\}$，默认 $K=4$。目标为

$$J(R)=\|S^{1/2}(WR-Q(WR))(R^\top A R)^{1/2}\|_F^2
+\lambda_A\mathbb E_t[\pi_{m(t)}\|x'_t-Q_A(x'_t)\|_2^2].$$

这是局部离散搜索，不保证全局最优或所有扰动都落在最低曲率子空间。更新一个 Givens 对的曲率矩阵只涉及两行两列，完整 $K$ 层更新约 $O(Kd_i^2)$；该计数不包含候选角度全部量化、因子求解及校准数据处理。

权重的旋转可以预计算，但 $R^\top X$ 需要在线执行或有合法图融合。原文 §IV-C 承认在线激活旋转，§V-H 又概括为无额外 runtime operator，不能把后者当作整个 RGSQ 的无条件结论。与 [QuaRot](quarot.md) 或 [SpinQuant](spinquant.md) 比较时，也不能忽略它们已有快速/可融合的变换路径。

## 5. 激活与全局相似度目标的作用

论文式 16 用 $\mathbb E_t[\pi_{m(t)}\|x_t-Q(x_t;s,b)\|_2^2]$ 校准激活。如果 $\mathbb E_t$ 均匀覆盖全部 token，每个模态的总系数仍含 $N_m/N$，不能仅凭 $\pi_m$ 的定义宣称整个激活目标与 token 比例无关。若每个 token 独立选尺度，给它乘一个正标量也不改变该 token 自身最优尺度；共享裁剪/尺度参数时才会形成不同 token 的取舍。

可选 $L_{geom}$ 是浮点与量化样本嵌入的两两余弦相似度矩阵 MSE。它约束某种批内几何，不等于语义或任务质量保证。式 24 将权重误差、激活误差、该正则加权；表 III 默认系数为 1、1、0.2。算法 3 主要描述逐层权重与激活求解，未充分展开全局正则如何共同优化所有变量，不应据此补写端到端训练流程。

## 6. 固定代码实际支持哪些路径

以下只涉及已读函数，不声称全仓库运行验证或论文实验由该 commit 产生。

| 已读位置 | 实际行为与解释影响 |
| --- | --- |
| `metrics.py:compute_modality_weights / compute_unified_metric` | 分别融合 A/S；支持 sensitivity、equal、balanced 等分支，部分统计路径允许 S 缺省。 |
| `whitening.py:compute_whitening_transforms / _symmetric_sqrt` | 求平方根/逆并加入阻尼、相对特征值下界；这些操作改变原经验度量。 |
| `whitening.py:GeometryAwareActivationQuantizer.calibrate_scale` | per-token 分支直接 absmax/qmax；模态加权的候选范围搜索只出现在 per-tensor 分支。不能说每个配置都执行了加权尺度搜索。 |
| `rotation.py:select_rotation_angles` | 该旧搜索函数用传入的固定 A/S 因子评价旋转权重误差，未在函数内更新 $R^\top A R$，也没有式 23 的激活项。 |
| `quantizer.py:extract_quant_params` | 另一条 pseudo-quant 路径先白化，量化白化后权重；可在白化空间旋转、量化后乘 $Q^\top$ 折回，再逆白化，另有候选选择。它不能直接证明原格式整数网格约束成立。 |
| `quant_layers.py:RiemannianQuantLinear.forward` | 旋转输入后走 `F.linear`，权重可在白化空间量化再逆白化；返回浮点张量的模拟路径不证明真实 W4 内核已经执行。 |

因此“支持几何评分”“可得到浮点模拟权重”“存在可部署整数格式”“真实加速”是不同证据层次。当前 ingest 承接前两者的分析，并保留后两者的验证缺口。

## 7. 实验结果及原件中的不一致

论文 §V 使用 ShareGPT4V 的 COCO 图文校准，评估 LLaVA-OneVision 7B/72B、InternVL2 8B/26B、Qwen2-VL 7B/72B。主表为六任务均分：MMMU、SEED、OCRBench、VizWiz、ScienceQA、TextVQA。默认表 III 为权重对称分组、group 128，激活逐 token 对称 A8，GPTQ 阻尼 0.01、block 128、关闭 act-order，曲率阻尼 $10^{-4}$。主文同时称权重 per-output-channel，故实际粒度仍应结合具体代码配置核对。

表 II 在 LLaVA-OneVision-7B W4A8 上报告 FP16 67.5、RTN 56.1、MBQ 63.1、MQuant 66.0、RGSQ 68.1。表 IV 的逐项消融为 56.1 → 加模态度量 64.1 → 加 ROW 66.6 → 加 GGES 68.4，说明作者报告的组件贡献，但最后一项与主表 68.1 不同，不能合并成同一次实验。

还需保留以下来源差异，均已回看原件页图或对应正文：

- 表 VI 将 InternVL2-8B 的 68.1 标为 AWQ、69.6 标为 SQ；表 II 的对应数字分别标为 SQ、MBQ。ROW 的方法标签因此不能直接跨表对齐。
- 表 VII 的六任务 W2A8/W3A8 均分明显高于表 V 部分同模型 FP16 均分，原文未解释设置差异；不能用“距 FP16 仅若干点”的文字与这些表拼成可靠排序。
- 图 3 采用三任务均值，主表采用六任务均值，不能互换。
- 表 IX 的部分基线准确率对应前表的不同位宽行，虽正文称 W4A8，不能据此建立严格同条件成本排行榜。

论文报告 LLaVA-OneVision-7B 校准约 2.1 GPU-hour、峰值 38 GB，硬件单 A800；这是离线成本，不能证明推理吞吐。视频结果是静态图文校准后的初步迁移，未补充视频专门校准，不应升级为通用视频结论。

## 8. 与已有方法的关系

[MBQ 与 VLMQ](../research/mbq-vlmq-comparison.md) 已解释模态/token 重要性怎样进入目标。RGSQ 的额外方向是保留输入与输出通道相关结构，而非仅多加一个模态系数；但其模态混合、经验 Fisher 与 K-FAC 都有近似。普通 GPTQ 的输出重构也有输入曲率，不能被划为参数空间纯 $I$ 度量。

[激活引导补偿](activation-guided-compensation.md) 分解的是固定激活输入下的可达输出空间；本页加权的是参数扰动方向的代价。这两个几何空间和解决的问题不同。任何加权目标都应继续核对真实量化集合、坐标变换和最终执行路径。

## 来源身份

| 来源 | 固定版本 | 范围 |
| --- | --- | --- |
| [RGSQ: Riemannian Geometry-Sensitive Quantization for Large Vision-Language Models](https://arxiv.org/abs/2609.25492v1) | arXiv:2609.25492v1 | §III–V、算法 1–3、式 8–24、表 II–IX；PDF 页眉模板日期不作为版本依据 |
| [RL-MIND/RGSQ](https://github.com/RL-MIND/RGSQ/tree/a69e2d59b544f8a673483d7a197ac27db72428b1) | a69e2d59b544f8a673483d7a197ac27db72428b1 | 上表所列函数，未运行模型或 GPU kernel |
