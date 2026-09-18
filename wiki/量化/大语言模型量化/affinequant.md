---
title: AffineQuant：可学习仿射变换与渐进掩码
slug: affinequant
sources:
  - raw/papers/quantization/ptq/2024-03-affinequant-affine-transformation-quantization/paper.pdf
  - raw/papers/quantization/ptq/2023-08-omniquant-omnidirectionally-calibrated-quantization/paper.pdf
  - raw/repositories/quantization/2026-09-bytedance-affinequant-0f3ba939/source/README.md
  - raw/repositories/quantization/2026-09-bytedance-affinequant-0f3ba939/source/quantize/affinequant.py
  - raw/repositories/quantization/2026-09-bytedance-affinequant-0f3ba939/source/quantize/affine_norm.py
  - raw/repositories/quantization/2026-09-bytedance-affinequant-0f3ba939/source/models/transformation.py
  - raw/repositories/quantization/2026-09-bytedance-affinequant-0f3ba939/source/models/int_llama_layer.py
updated: 2026-09-15
---

# AffineQuant：可学习仿射变换与渐进掩码

AffineQuant 把量化前的可学习逐通道缩放扩展为可逆矩阵，使一个通道可以与其他通道混合；与逆矩阵配对后，浮点线性计算不变，量化后的网格适配可能更好。它沿用 [[omniquant|OmniQuant]] 的块级重构和可学习裁剪，核心增量是更大的变换空间，以及逐步开放非对角元素的 gradual mask。

本页依据 arXiv:2403.12544v1（2024-03-19，ICLR 2024），19 页含附录 A.1–A.7 已全文研读。代码固定为 `0f3ba9392bb0037f30750c4b2dbdd38469a2d7da`，下称“代码”；定向核对 LLaMA 校准与共用变换，未运行模型。更大空间不保证更好结果，也不能在所有算子之间免费融合。

## 1. 对象与问题

论文主要研究 OPT、LLaMA-1/2 的 PTQ，包括 W2/W3/W4A16，以及 W4A4。纯权重配置含 per-channel 和 group size 64/128；W4A4 延续权重 per-channel、激活 per-token 的块校准路线。参数数量、模型和分组不同，会明显改变收益。

[[diagonal-scaling-equivalent-transform|对角缩放]] 可以平衡某通道的激活与权重幅度，却不能让一个输出坐标吸收其他输入坐标的信息。AffineQuant 试图通过非对角项改变权重向量相对量化网格的位置。缩放是这一表示族的特例，但完整 AWQ/OmniQuant 算法还包含自己的目标、裁剪和搜索流程，不能仅凭矩阵包含关系就说“包含且优于整个算法”。

## 2. 核心恒等式与重构目标

为贴合原文，暂用 $X\in\mathbb R^{T\times d}$、$W\in\mathbb R^{d\times m}$、偏置 $b\in\mathbb R^m$；$A\in\mathbb R^{d\times d}$ 可逆、$\delta\in\mathbb R^d$ 按 token 广播：

$$XW+b=(X-\delta)A^{-1}(AW)+(b+\delta W).
$$

这里矩阵负责混合/缩放，平移是另一个变量；“仿射”不能解释成只学习一个偏置。若 PyTorch 按 $m\times d$ 存权重，实际要变换成 $W_{torch}A^\mathsf T$，布局解释见 [[linear-layer-input-channel|线性层的权重方向]]。

纯权重量化的单层动机是

$$\min_A\|XW-XA^{-1}Q(AW)\|_F^2.
$$

$Q$ 在此表示含反量化的实数映射：先按步长/zero-point 舍入、饱和，再还原成实数，定义见 [[uniform-quantization-and-groups|量化网格]]。W4A4 还要量化 $(X-\delta)A^{-1}$；不能把上式当成全部配置的完整前向。（论文 §3.1 式 1–4。）

实际在完整 Transformer block 上学习 $A,\delta$ 和裁剪参数，冻结原权重。代码保留两路输入：教师目标来自浮点前缀，学生输入来自已量化前缀；`aug_loss` 再加入浮点块作用于学生输入的参考目标。这些是对 [[layer-reconstruction-second-order-compensation|输出重构]] 的非线性块级应用，并非直接最小化最终任务损失。

**为什么可能有用，又为什么不保证。**非对角混合能改变多个坐标相对均匀网格的位置；相应逆矩阵也会改变激活幅度和量化误差传播。只有两侧及块输出一起更有利，才产生最终收益。将可逆矩阵逐行归一化并不能一般地得到旋转矩阵；“任意权重都能精确落到网格点”也不是这个有限参数、共享变换问题的保证。定义与反例见 [[invertible-transforms-and-kronecker-products|可逆变换与数值条件]]。

### 先看通道混合怎样改变量化结果

这是解释 §3.1 的教学例子：只量化权重，固定步长 1、zero-point 0、整数范围 $[-4,3]$，$Q$ 为舍入后反量化。取

$$W=\begin{pmatrix}0.49\\0.49\end{pmatrix},\quad
A=\begin{pmatrix}1&0.5\\0&1\end{pmatrix},\quad
A^{-1}=\begin{pmatrix}1&-0.5\\0&1\end{pmatrix}.
$$

原坐标直接量化有 $Q(W)=(0,0)^\mathsf T$；混合后 $AW=(0.735,0.49)^\mathsf T$，于是 $Q(AW)=(1,0)^\mathsf T$。恢复到原坐标的有效权重为 $A^{-1}Q(AW)=(1,0)^\mathsf T$。

| 输入 | 浮点输出 $XW$ | 直接量化 $XQ(W)$ | 配对变换后 $XA^{-1}Q(AW)$ |
|---|---:|---:|---:|
| $X=(1,1)$ | 0.98 | 0 | 1 |
| $X=(1,-1)$ | 0 | 0 | 1 |

第一种输入的绝对输出误差从 0.98 降至 0.02，第二种则从 0 升至 1；恢复坐标后的权重平方误差还从 0.4802 增至 0.5002。**优化要看输入加权后的输出，不能只看权重离网格多近。**这里固定网格、只选一个矩阵，既未优化对角基线，也未模拟激活量化，不构成 AffineQuant 优于 OmniQuant 的实验；它说明为何需要校准数据，以及收益为何依赖数据方向。联合 W/A 的误差展开与逆矩阵梯度见 [[invertible-transforms-and-kronecker-products|可逆变换基础]]。

## 3. Gradual mask 怎样工作

直接更新整个矩阵容易使求逆不稳定。论文先用 SmoothQuant 风格的通道尺度初始化对角线，再逐 epoch 扩大可以学习的带状区域：

$$G^{(e)}_{ij}=\begin{cases}
1&i=j,\\
\alpha&0<|i-j|\le r_e,\\
0&\text{其他},
\end{cases}\qquad A_e^*=A_e\odot G^{(e)}.
$$

$e$ 是当前 epoch，$r_e$ 随进度扩大，$\alpha$ 是稳定因子，$\odot$ 为逐元素乘。前向用有效矩阵 $A_e^*$ 及其逆；反向有 $\partial L/\partial A_e=G^{(e)}\odot\partial L/\partial A_e^*$。因此 mask 同时改变有效非对角幅度和梯度路径；不是普通的“先冻结再解冻所有参数”。（§3.2，式 5–9，图 2。）

代码在第一个 epoch 仅开放对角，末尾开放规定范围：Norm 相关矩阵可扩到整个 hidden dimension；V/O 与 Q/K 的 mask 限在各 attention head 内。不能在不同 head 的 attention 加权之前随意混合 head。（`quantize/affinequant.py` 的 maskqkv/maskfc 构造。）

例如一个 $4\times4$ 的允许混合区域，带宽半径从 0 扩到 1 时，mask 从 $I_4$ 变为

$$G_1=\begin{pmatrix}
1&\alpha&0&0\\
\alpha&1&\alpha&0\\
0&\alpha&1&\alpha\\
0&0&\alpha&1
\end{pmatrix}.
$$

到半径 3 时才开放全部非对角项。如果存储参数 $a_{12}=0.8$、$\alpha=0.01$，当前前向看到的是 0.008，反向回到该参数的梯度也乘 0.01；未开放的位置当前前向与损失梯度均为零。这不等于允许随意丢弃 mask：导出必须使用有效矩阵。各 head 的矩阵应分别应用这一过程，不能把“全部开放”误读为允许跨 head 混合。（示例按式 5–9 展开；具体 epoch 到带宽的取整以固定代码为准。）

它希望维持 $|a_{ii}|>\sum_{j\ne i}|a_{ij}|$ 的严格对角占优，从而保证可逆。**但稳定因子小不构成无条件保证**：附录 A.2 的界依赖累计梯度及非零对角项；对角值本身也会更新。代码采用 AdamW，因梯度自适应归一化，不能把 mask 直接解释成等比例的实际学习率缩减。代码没有把每一步投影到严格对角占优集合；截断小元素也不等于限制条件数。

因此需要区分作者的优化动机、充分条件和实现行为。[[invertible-transforms-and-kronecker-products#2. 可逆、对角占优与条件数|占优余量与条件数]] 给出简短证明和更新反例。这里保留的是影响方法可靠性的条件，不是纠缠论文符号笔误。

## 4. 放在哪里，以及能否离线融合

| 位置 | 做什么 | 条件与代码边界 |
|---|---|---|
| Norm → Q/K/V；Norm → up/gate | 输入侧 $A^{-1}$ 与权重侧 $A$ 配对 | 论文 W4A4 在 Norm 后只学习对角，便于合入 Norm；`use_ln_matrix` 才启用完整矩阵，Norm 保留在线乘法 |
| V → attention 输出 → O | head 内变换 V，并在 O 权重中补偿 | attention 的 token 加权对每个 head 内通道共享；跨 head 混合一般不能提前消去 |
| Q/K 配对 | 一侧逆转置，另一侧正向矩阵，试图保持内积 | 代码可用 `use_matrix`，但变换写进 Q/K 投影权重，发生在 RoPE 之前，需要额外交换条件 |
| MLP 中间 → down | down 权重仍量化 | 本方法未在这里引入一般仿射配对，避免大中间维和非线性边界问题；不是跳过 down 量化 |

（论文 §3.3、§4.1；代码 `models/transformation.py`、`models/int_llama_layer.py`、`quantize/affine_norm.py`。）

Norm 处完整矩阵的实现会保存 `c_inverse` 并执行 `out.matmul(c_inverse)`，所以“全部变换无推理开销”不适用于这个分支。融合到权重的一侧，也不说明激活侧必然可以删除。

Q/K 的风险可用代数表达：若行向量在位置 $t$ 经过旋转 $R_t$，先做 $qA^{-1}$、$kA^\mathsf T$，内积包含 $A^{-1}R_tR_s^\mathsf T A$；它只有在相应交换关系成立时才等于原来的 $R_tR_s^\mathsf T$。普通 head 内矩阵并不自动满足。详见 [[diagonal-scaling-equivalent-transform|等价变换的 RoPE 边界]]；[[flatquant|FlatQuant]] 将相应变换放在 RoPE 之后，是另一种处理方式。

## 5. 一轮校准与实现要求

1. 用校准集收集第一层输入与通道统计，冻结基座权重；论文采用 WikiText-2 训练集的 128 段、每段 2048 tokens，附录 A.7 报告单张 A100。
2. 当前 block 生成教师参考输出；按所选位置初始化对角矩阵、平移与 LWC。论文优化器、epoch、学习率沿用 OmniQuant 的相应方案，不能给所有位宽强行套一个配置。
3. 每个 epoch 更新 mask；每个 batch 从基座权重构造临时变换，计算有效逆矩阵，执行裁剪和模拟量化，得到整个 block 输出。
4. 用 MSE 更新变换及裁剪参数；保留求逆失败、非有限损失等信号。校准损失较低只说明该目标较好，不能自动等同于下游任务更好。
5. 校准后固化有效矩阵并融合合法的算子对，重新计算量化权重；推进学生输入到下一个 block，保存变换和网格参数。不能先丢掉 mask 再直接导出未掩码矩阵。

代码定向核对了以上临时/固化两条路径。`real_quant` 分支调用 AutoGPTQ `QuantLinear.pack`；这说明存在权重打包入口，不证明 W4A4 注意力、激活或 KV cache 已实现全整数执行。论文表中的质量与实际端到端速度应分开，见 [[quantized-matmul-scaling-execution|量化执行路径]]。

## 6. 值得保留的实验

下面均为作者报告，未在本地复现；不把不同表或不同论文的模型基线拼成统一排名。

| 证据与设置 | 结果 | 能支持什么 |
|---|---|---|
| 表 3，LLaMA-2-7B W4A4 | C4 PPL：OmniQuant 18.02 → AffineQuant 15.76；FP16 6.97 | 扩大变换空间可缓解这一配置的误差，但离浮点仍有明显距离 |
| 表 2，LLaMA-13B W4A4，六个 zero-shot 任务平均 | OmniQuant 54.37 → AffineQuant 52.58；FP16 66.33 | 不能声称所有模型、所有任务都改善；PPL 优势不保证 QA 优势 |
| 表 11，LLaMA-7B W2A16 vs W2A16g128 | 无分组 Omni 15.47 → Affine 9.53；g128 则 10.53 → 13.51 | 收益对网格粒度有依赖，非对角学习不是越多越好 |
| 表 5，LLaMA-7B W2A16 | $\alpha=1,0.1$ 出现 NaN；0.01 得 WikiText2 9.53；更小也非单调更好 | 稳定因子确实影响收敛与可用性，不能只视为无关超参数 |
| 表 6，移除 gradual mask | OPT-125M W3A16 WikiText2 32.10 → 53.52；LLaMA-7B W2A16 9.53 → NaN | 支持渐进开放在所测配置中的作用，不证明其充分条件被自动满足 |
| 表 4，OPT-6.7B W4A16，double / float | 校准时间 16.7h / 8.65h，报告显存约 41,414 / 21,189 MB；PPL 11.91 / 11.90 | 更高精度减少合并数值误差，但有明显成本，未显示同比例模型质量收益 |

附录 A.4 的末块重构损失与 WikiText2 PPL 散点相关系数约 0.95/0.96，是固定初始化、改变稳定因子下的 OPT-6.7B/LLaMA-7B W4A4 观察。严格关系是 $\mathrm{PPL}=\exp(\mathrm{CE})$；block MSE 与 CE 之间没有论文式 3 所暗示的普遍比例定律。图 7 的矩阵热图帮助观察非对角项随训练开放，却不能代替逐行占优或条件数检查。

## 7. Strong / weak 与方法选择

**Strong：**它在保持基座冻结的条件下提供通道混合自由度；相比只学尺度，可以探索新的网格对齐方式。渐进 mask 给这类较难的求逆优化提供了实用稳定化手段，低位宽的一些改善和失败消融支持这一价值。

**Weak：**求逆和大矩阵校准有成本，稳定性依赖配置；可逆不等于条件良好。为方便融合而限制变换位置，会让“完整矩阵空间”的理论动机与实际全模型可用空间不同。启用 Norm 完整矩阵又会保留在线计算，且 Q/K 路径需要核对 RoPE。

如果问题是“逐通道缩放不够”，这篇值得作为研究起点；如果问题还包含“每个主要线性层都需要混合且实际要加速”，应继续看 [[flatquant|FlatQuant 的结构化变换与内核融合]]。三者的共同条件和证据边界集中在 [[omniquant-affinequant-flatquant-comparison|OmniQuant、AffineQuant 与 FlatQuant 比较]]。本页不将纯文本模型结果直接迁移成多模态或端云收益。
