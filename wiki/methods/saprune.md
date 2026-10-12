---
title: SAPrune：按层深选择时机与双路径视觉 token 剪枝
type: method
tags:
  - vlm
  - pruning
  - calibration
  - evaluation
sources:
  - raw/papers/2026-10-07/saprune/paper.pdf
updated: 2026-10-08
---

# SAPrune：按层深选择时机与双路径视觉 token 剪枝

SAPrune 用离线 attention 统计选择三个剪枝层，再在每个选定层按当前输入保留一部分视觉 token。保留集合由两部分组成：高动作注意力的核心，以及补充动作/指令响应差异的候选。它压缩序列长度，不提出新的量化格式；training-free 表示不更新模型权重，仍有离线校准和在线选择成本。

本页依据 arXiv:2610.05273v1 的正文与必要附录，重点核对选层规则、双路径公式、复杂度和实验协议。当前材料未取得匹配作者实现，具体接入与性能均未本地运行。视觉 token、删除集合与位置的基础见[视觉语言模型中的 token 与量化对象](../fundamentals/model/vision-language-model-tokens-and-quantization.md#7-剪枝额外改变序列集合深度与位置)。

## 1. 两个选择分别回答什么

这里的 stage 指 Transformer 的浅层、中层和深层，不是机器人任务的接近、抓取或放置阶段。离线步骤为一个骨干选择固定层表；在线步骤在这些层上重新计算保留集合及两条路径的名额。（§3、图3。）

| 选择 | 输入与输出 | 固定与动态部分 |
| --- | --- | --- |
| 何时剪 | 校准 rollout 的逐层 action-to-visual attention → 三个层号 | 固定骨干复用层表，不是每帧重新搜索剪枝层 |
| 保留什么 | 当前层的视觉特征、action/text attention 与保留预算 → token 下标 | 候选、双路径比例与集合随当前输入变化 |

默认每次保留一半视觉 token，形成 $256\to128\to64\to32$。文本不剪；剪枝作用于选定层的输出，只有后续层消费更短序列。先前删除的 token 不会在后面重新召回，最后一层也被排除在候选剪枝层之外。（§3.2–3.3、附录 A 算法1–2。）

## 2. 离线选层：把 attention 变化压成一条曲线

### 2.1 统计量包含什么

在层 $l$、校准步骤 $t$，令 $\alpha_{q,v_i}^{(l,t,h)}$ 为第 $h$ 个头中动作相关 query 对视觉 token $i$ 的注意力。论文先对动作 query 集合 $\mathcal Q_a$ 与 $H$ 个头平均，再取视觉 attention 最大的一部分：

$$
a_i^{(l,t)}
=\frac{1}{|\mathcal Q_a|H}\sum_{q\in\mathcal Q_a}\sum_{h=1}^{H}
\alpha_{q,v_i}^{(l,t,h)},\qquad
S_l(t)=\sum_{i\in\operatorname{Top}_{\lfloor\kappa N\rfloor}(a^{(l,t)})}
a_i^{(l,t)},\qquad
\mu_l=\frac1T\sum_{t=1}^{T}S_l(t).
$$

默认 $\kappa=0.125$，$N$ 为视觉 token 数，$T$ 遍历校准 rollout 的全部步骤。因此 $\mu_l$ 按步骤平均，较长轨迹贡献更多，不是每条轨迹或每个任务等权；图4的任务均值只用于可视化。（§2.2、§3.2，式3–5。）

**这里没有规定先在视觉集合内重新归一化。** $S_l$ 同时受总视觉 attention mass 和视觉内部集中程度影响；它不是纯粹的条件集中度。即便 $S_l$ 不变，最高 attention 所在的 token 仍可能完全换位。后面的跨层差也只是这个标量的变化，不是整张 attention 分布的距离；相关分解与反例见[公共基础 §7.1](../fundamentals/model/vision-language-model-tokens-and-quantization.md#71-attention-总量集中度与所关注对象是三件事)。

### 2.2 从变化曲线得到三个层号

论文将零起始层号 $0,\ldots,L-2$ 分成浅、中、深三个连续相对深度区域 $\mathcal R_1,\mathcal R_2,\mathcal R_3$。先对 $\mu_l$ 做三点移动平均得到 $\widetilde\mu_l$，然后计算

$$
d_l=|\widetilde\mu_{l+1}-\widetilde\mu_l|,\qquad
\tau_s=Q_{0.4}(\{d_l\}),\qquad
\tau_r=Q_{0.8}(\{d_l\}).
$$

$Q$ 为经验分位数。连续 $m=2$ 个差值都低于 $\tau_s$ 时，其起点被列为 plateau；连续高于 $\tau_r$ 的极大区间被列为 redistribution event。按式6–9与算法1，选择流程是：

1. 浅层先限制层号至少为 $l_{\min}=1$；若限制后为空，恢复整个浅层区域。
2. 中、深层寻找与该区域相交的事件，按事件整个区间的 $\sum d_l$ 选出最大者，优先搜索其终点之后的层；没有事件或过滤后为空时，恢复整个区域。算法1明确使用“相交”，并未把跨区域事件裁成区域内一段再计分。
3. 在有效区域内选最早的 plateau 起点。若没有 plateau，则选后续至多 $m$ 个可用差值的局部均值最小处；末端按实际可用项数归一，平局选更早层。

OpenVLA 主设置得到零起始层号 $[2,16,26]$。规则以 attention 标量趋稳作为剪枝时机的代理，并有回退选择，不能改写成“只有语义已经稳定才剪”。选层只记录 attention，不用 reward、成功标签或任务结果；这也不表示整个方法的超参数开发从未查看任务表现，附录 B 的候选池比例就是按 LIBERO 消融选择。（§3.2、附录 A.1/B.5。）

## 3. 在线双路径：集中保留与功能覆盖怎样分配

### 3.1 候选池与 probe

在选定层，当前有 $N_l$ 个视觉特征 $z_i\in\mathbb R^d$，目标保留 $K_l$ 个。论文按动作 attention 建立：

$$
C_l=\operatorname{Top}_{\lfloor1.75K_l\rfloor}(a),\qquad
A_l^{\rm probe}=\operatorname{Top}_{\lfloor0.8K_l\rfloor}(a).
$$

默认三次的候选池大小为224、112、56，probe 大小为102、51、25。覆盖路径也只能从这个 attention 预筛池里选，不能找回池外或前面已删除的视觉信息。（§3.3，式10。）

### 3.2 用子空间保真与集中度分配名额

将 probe 的归一化特征视为特征空间中的列向量，做 thin QR 得到正交列基 $U_l\in\mathbb R^{d\times r}$。对候选池，明确归一化动作 attention：

$$
w_j=\frac{a_j}{\sum_{i\in C_l}a_i},\qquad
R_l=\frac{\sum_{j\in C_l}w_j\|z_j-U_lU_l^\top z_j\|_2^2}
{\sum_{j\in C_l}w_j\|z_j\|_2^2},\qquad
P_l=1-R_l.
$$

$P_l$ 表示 probe 子空间解释的加权候选特征能量。正交投影、非负权重且分母为正时，能量分解给出 $0\le P_l\le1$；这是式11的直接性质，不是动作正确概率。QR 在 hidden-feature 空间构造，不能把 token 行空间的正交基直接用于投影 $z_j$。

候选池内部的 attention 集中度则为

$$
H_l=-\sum_{j\in C_l}w_j\log w_j,\qquad
A_l=1-\frac{H_l}{\log|C_l|}.
$$

在 $|C_l|>1$ 的正常概率分布下，均匀 attention 给 $A_l=0$，完全集中给1。它与选层时未作视觉条件归一化的 $S_l$ 不同。两项共同决定 attention 路径占比：

$$
\eta_l=\sqrt{P_lA_l},\qquad
\lambda_l=0.1+0.8\eta_l,\qquad
K_l^{\rm attn}=\lfloor\lambda_lK_l\rfloor,\qquad
K_l^{\rm cov}=K_l-K_l^{\rm attn}.
$$

只有“高 attention probe 能代表候选子空间”和“候选 attention 集中”同时较高，才多分给 attention 核心；任一为0时回到0.1。教学例：$P_l=0.81,A_l=0.25$ 给 $\eta_l=0.45,\lambda_l=0.46$，$K_l=32$ 时保留14个核心与18个覆盖 token。这只是算式说明，没有模型测量。（式11–13。）

### 3.3 覆盖的是 query 响应，不是空间邻居

attention 路径先取候选池中排名最高的 $K_l^{\rm attn}$ 个 token，初始化保留集合 $R$。覆盖路径先按 action-to-text attention 排序非特殊文本 token，取前 $K_x=6$ 个文本 query，再与 action query 组成 $\mathcal Q_l$。每个候选的功能签名是

$$
r_i=[A(q,i)]_{q\in\mathcal Q_l}\in\mathbb R^{|\mathcal Q_l|},
$$

其中 $A(q,i)$ 为跨头平均 attention；有6个文本 query 时签名维度为7。它比较“这些动作/指令 query 怎样关注该视觉 token”，不是 hidden feature 余弦，也不是在图像坐标上补邻居。（式14–15。）

论文对签名作 $\ell_2$ 归一化，汇总候选池证据 $r_{\rm full}$ 与已留证据 $r_R$，以 $m_R=r_{\rm full}-r_R$ 表示尚未覆盖的部分，再逐个选

$$
i^\star=\arg\max_{i\in C_l\setminus R}
\left[\langle\widehat r_i,\widehat m_R\rangle
-0.3\max_{j\in R}\cos(r_i,r_j)\right].
$$

帽号表示单位方向。第一项鼓励对齐残余响应，第二项惩罚与已留集合重复；每次加入后更新 $R$ 和残余，直到用满覆盖名额。最后合并、去重；若不足 $K_l$，按未用候选的 attention 顺序补齐。（式16、算法2。）

功能签名仍是代理：$(0.8,0.2)$ 与 $(0.4,0.1)$ 归一化后方向相同，但 attention 总量不同；签名也没有包含 Value 或删除后的反事实动作变化。论文没有给这条贪心规则全局最优或成功率保证。正文和算法2只称“汇总”功能证据，未明确 $r_{\rm full}$、$r_R$ 的聚合算子及权重，不能擅自补成已核实的求和代码。

## 4. 预算、接入与实际节省

最终32/256只表示视觉 token 的末端保留率12.5%。剪前各层仍处理较长序列，非视觉 token 也继续保留。附录 A.2 用近似单层代价

$$
C(n)=4nd^2+2n^2d+2ndd_{\rm ff}
$$

描述骨干，其中 $n$ 是总 token 数，$d$ 为隐藏宽度，$d_{\rm ff}$ 为 FFN 宽度。这是论文的计数约定，不是所有 attention/FFN 结构的精确 FLOPs；乘加计数和 gated FFN 等差异不能省略。

将零起始剪枝层 $s_1,s_2,s_3$ 转成已执行层数 $b_j=s_j+1$，记保留的非视觉数为 $M$，$n_0=M+256,n_1=M+128,n_2=M+64,n_3=M+32$，则

$$
F_{\rm backbone}
=b_1C(n_0)+(b_2-b_1)C(n_1)+(b_3-b_2)C(n_2)
+(L-b_3)C(n_3).
$$

对32层及 $[2,16,26]$，各段为3、14、10、5层。还需另算所选 query attention、QR、top-k、功能签名/贪心和 gather；作者报告的总 FLOPs 比例额外包含 routing，而上式只算骨干。公共的逐层预算解释见[视觉 token 基础 §7.2](../fundamentals/model/vision-language-model-tokens-and-quantization.md#72-最终-token-数不等于逐层工作量)。

实际接入至少需要在目标模块拿到 action/text query 对视觉 token 的 attention 行，并使 gather 后的 mask、位置和多图对应保持一致。论文报告 SDPA、eager 和 FlashAttention-2 的兼容结果，未提供本轮可核的具体实现。附录 C 也指出：当 VLM 与 DiT 动作专家分离、action query 仅在下游可用时，需要额外适配；删 VLM token 不会直接消除动作专家的去噪计算。

## 5. 作者结果支持什么

### 5.1 默认预算与统一末端预算

表1采用各基线报告/默认 token 设置，SAPrune 使用256→32。以下均为作者在 **RTX 5090、PyTorch SDPA、无 compile** 下的模型时延，包含剪枝模块开销；不能填入论文未完整列出的 batch、dtype 或计时重复次数。

| 骨干，LIBERO | 平均成功率：完整 → SAPrune | 模型时延 ms：完整 → SAPrune | 作者加速比 | 作者 FLOPs 保留比例 |
| --- | --- | --- | --- | --- |
| OpenVLA | 74.9% → 74.0% | 178.46 → 110.96 | 1.608× | 39.5% |
| OpenVLA-OFT | 95.4% → 94.5% | 103.59 → 60.30 | 1.718× | 37.8% |
| $\pi_0$ | 93.9% → 93.1% | 78.49 → 51.22 | 1.532× | 38.6% |

1.718×应限定这组默认 SDPA 主表；附录表8的 eager OpenVLA-OFT 另报1.730×。表7中 OpenVLA 三次 routing 共 $2.72+1.35+0.69=4.76$ ms，已包括在110.96 ms内；其中功能签名与贪心占3.09 ms，不能重复加到总时延。

表6统一最终32 token 后，SAPrune 在所列方法中质量较高，但并非最快。例如 OpenVLA 的 EfficientVLA（T-only）为57.9% / 99.58 ms，SAPrune 为74.0% / 110.96 ms。末端预算相同仍不代表剪枝深度与累计计算量相同。这组证据支持质量与时延取舍，不能据此宣称速度全面领先。

表5的选层消融也同时改变质量和工作量：单层 $[2]$ 为55.1% / 95.64 ms，三层 $[2,16,26]$ 为74.0% / 110.96 ms。固定剪枝层及最终32 token 的表12中，完整双路径74.0%，移除 attention 核心63.9%，移除功能覆盖67.0%；它支持两条路径在该配置下的作用，不证明每个输入都需要相同比例。

### 5.2 机器人流水线与校准迁移

AgileX Piper 的 $\pi_0$ 真实实验每个任务100次、共四项任务。表4的完整策略平均成功率55.5%，SAPrune 为54.0%；堆叠任务从30%降到25%，不能只用平均值描述保持程度。表中 VLA-ADP 的平均54.5%也高于 SAPrune，不能写成所有剪枝方法中成功率最高。

表9将图像预处理、模型、动作解码、通信与编排/同步相加，policy pipeline 从88.68降到61.97 ms，对应11.28→16.14 Hz；该口径**排除物理执行与底层伺服**，不是机器人完成任务时间。模型部分为78.16→51.45 ms。另一个骨干 $\pi_{0.5}$ 的表3报告 LIBERO E2E 为91.47→59.68 ms、成功率96.6%→95.9%，也不能混入表1的模型时延列。

校准迁移须按实验区分。§3.2列举默认 LIBERO，以及 SIMPLER 30、真实40、混合270条 rollout；附录 B.4.5 及表10另做 OpenVLA 的30/60/120等规模比较，不存在一个适用于所有实验的固定样本数。例如真实30条得到 $[2,17,27]$、LIBERO-Long 47.6%，真实60条得到 $[2,17,26]$、51.1%；默认 LIBERO 层表的该项为50.8%。这些是所测骨干的迁移证据，不代表任意新模型都无需校准。

附录 B.4.6/表11用 **$\pi_{0.5}$、300条标准 LIBERO 校准轨迹**，不使用 LIBERO-Plus 校准或重校准，再对相同10030个变体各评一次：整体成功率85.7%→84.5%，整体按各类样本数加权，不是七类均值简单平均。这与 [PIVOT-Q](pivot-q.md#7-作者结果与版本差异) 的同任务适配协议不同，数字不能直接排名。论文未完整给出主表全部 seed、重复试验与置信区间信息，接近的均值不等于统计等价。

## 6. 可以复用的认识与尚缺的实现证据

SAPrune 将“在何层删”与“在该层保留谁”拆开：前者用跨校准步骤的标量趋势选固定时机，后者用当前输入的 feature 子空间与 query 响应分配预算。这补充了 [CoRePrune](coreprune.md) 对条件可删除性与深度的讨论；[LayerPos](layerpos.md) 说明的位置问题仍需接入代码回答，不能从按顺序 gather 自动推出。它与 PIVOT-Q 的量化后参数恢复也作用于不同变量，论文没有验证两者组合。

当前论文解释足以辨认机制及主要取舍，但不足以声称已还原可运行实现：三个区域的端点、移动平均边缘处理和分位数插值未细化；前文动作 query 集合到在线单 query 的骨干级映射未明确；功能证据聚合、QR 秩亏/零范数、文本不足6个等边界尚缺具体代码。附录 C 还限定 action attention 具有任务相关性的假设；遮挡、细接触、长时程和分离动作专家的适用性需要各自证据。

## 来源身份

| 来源 | 固定版本 | 本页使用范围 |
| --- | --- | --- |
| When and What to Prune? Stage-Aware Visual Token Pruning for Efficient VLA（`https://arxiv.org/abs/2610.05273v1`） | arXiv:2610.05273v1，2026-10-04提交 | 正文机制、式3–24与算法1–2；实验及附录 A/B/C 的相关表格、校准与限制。PDF/TeX 交叉核对；未取得匹配作者实现，未运行模型、机器人或 GPU。 |
