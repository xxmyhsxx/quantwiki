---
title: LC-QAT：仿射整数网格的向量量化训练
type: method
tags:
  - qat
  - vector-quantization
  - optimization
  - weight-quantization
sources:
  - raw/papers/2026-09-21/lc-qat/paper.pdf
updated: 2026-10-11
---

# LC-QAT：仿射整数网格的向量量化训练

LC-QAT 要解决的是：怎样让 2-bit 向量权重在整网训练中同时调整离散选择与连续表示，同时避免每次前向都显式查一个大向量表。它将码字限制成同一个仿射映射作用于整数网格；前向仍执行硬舍入，反向使用代理梯度。良好的 PTQ 初始化、代理坐标和训练数据共同影响结果，不能把收益单独归因于“向量量化完全可微”。

本页依据 `arXiv:2606.10531v1` 正文、式3–12、主要结果及附录 A.1–A.6。本文核读论文与原始页图，未取得匹配作者训练或 CUDA 实现，未运行模型。前置知识见[码本与预算](../theory/codebook-quantization-and-bit-budget.md)、[PTQ/QAT 与代理梯度](../fundamentals/quantization/post-training-and-quantization-aware-training.md)、[块 LDL 反馈](quip-sharp.md)。

## 1. 用少量参数生成一个受约束的向量码本

设权重 $W\in\mathbb R^{m\times n}$，沿输入列方向每 $d$ 个数一组，$d$ 整除 $n$。对组向量 $z\in\{0,\ldots,2^b-1\}^d$，恢复向量为

$$c=Az+B,\qquad A\in\mathbb R^{d\times d},\quad B\in\mathbb R^d.$$

$A,B$ 在同一权重矩阵内共享，而整数向量逐组选择。2-bit、$d=4$ 时有 $4^4=256$ 个整数候选，用 $16+4=20$ 个浮点参数生成；一般任意向量表需要存 $d\,2^{bd}$ 个数。矩阵 $A$ 混合组内各坐标，因此某一整数坐标的改变可以同时改变多个恢复分量。但所有中心必须满足同一个仿射网格结构，不具有任意 256 个四维中心的自由度。方阵 $A$ 也不自动构成低维子空间。

理想紧凑索引加 FP16 的 $A,B$，平均预算为 $b+16(d^2+d)/(mn)$ bit/weight；这只是表示预算，另需计未量化参数、Hadamard 相关状态、对齐及实际容器。训练还维护与 $W$ 同形状的浮点代理权重和优化器状态，推理索引为 2-bit 不表示训练内存为 2-bit。与[加性 AQLM](aqlm.md)相比，本方法的结构收益来自共享仿射映射，不是多个任意向量表相加。

## 2. PTQ 初始化先提供可用的离散表示

式4将初始映射设为

$$
A=s_0R,\quad B=-s_0R\mu\mathbf1_d,\quad
\mu=\frac{2^b-1}{2},\quad
s_0=\sqrt{\frac{12}{2^{2b}-1}},
$$

其中 $R$ 是随机正交矩阵。本页改用 $R$ 区别后面的组数；$s_0$ 也不同于代理权重预处理的尺度。在均匀枚举整数候选的条件下，中心化整数的每维方差为 $(2^{2b}-1)/12$，经缩放与正交映射可得到零均值、单位协方差。这不保证实际选中的权重码分布也均匀。

作者使用 Hadamard 处理离群方向，并用 LDLQ 逐列组量化、顺序补偿重构误差；输入二阶矩如何进入反馈见[二阶重构](../theory/layer-reconstruction-second-order-compensation.md)。初始化先取得整数 $W_z$，后续 QAT 不必从已经严重损坏的标量模型恢复。

图2是 Qwen3-1.7B、WikiText-2 交叉熵在两个方向上的投影，支持作者观察到的较好起点；二维切面不能证明全参数空间的全局最小值或收敛。表4的 Qwen3-8B 初始 QA 均分为 LC-PTQ 67.95、GPTQ 37.53、随机初始化 39.51，比较的是起点，并未证明另外两种初始化在同预算训练后永远无法恢复。

## 3. 硬前向怎样成为整数矩阵乘

暂以未缩放的整数坐标代理 $P_z$ 表示式5的变量：

$$W_z=Q(P_z)=\operatorname{clip}(\operatorname{round}(P_z),0,2^b-1).$$

令组数 $N_g=n/d$，$W_{z,i}\in\mathbb Z^{m\times d}$，$X\in\mathbb R^{N\times n}$，$X_i\in\mathbb R^{N\times d}$，$\mathbf1_m$ 为全1输出列向量。恢复矩阵按块满足

$$
\widehat W^\mathsf T=
\begin{bmatrix}
AW_{z,1}^\mathsf T+B\mathbf1_m^\mathsf T\\
\vdots\\
AW_{z,N_g}^\mathsf T+B\mathbf1_m^\mathsf T
\end{bmatrix},
$$

于是式8可以写成

$$
Y=X\widehat W^\mathsf T
=\sum_{i=1}^{N_g}(X_iA)W_{z,i}^\mathsf T
+\sum_{i=1}^{N_g}(X_iB)\mathbf1_m^\mathsf T.
$$

先把浮点输入乘小矩阵 $A$，主体仍消费整数 $W_z$，偏移项按组产生一个 $N\times1$ 列再广播到所有输出通道。原文式8前的文字写作 $X\widehat W$；按其 $m\times n$ 权重形状与式7，本页明确转置。这个重排消除了显式大表查找，不消除舍入节点。

投影与偏移额外成本分别为 $O(Nnd)$、$O(Nn)$，主体为 $O(Nnm)$。$d\ll m$ 使投影的算术比例较小，但实际速度仍受融合、访存和 kernel 支持影响。激活 $X_iA$ 为浮点，论文的 Int2–FP16 路径属于低比特权重计算；不能改称 W2A2。

## 4. 同时训练代理坐标与仿射映射

训练前向重算整数选择、恢复权重，再以整网损失更新浮点代理及 $A,B$；$A,B$ 的连续导数是普通链式求导，整数节点使用近似反向。相比[PV-Tuning](pv-tuning.md)分开连续更新与合法点搜索，LC-QAT 让潜变量累计梯度并在硬前向中跨越舍入边界。两者的搜索对象、更新时序和理论条件不同。

### 4.1 DGE 的公式与未明确的数值处理

式9–10给出一个区间上的代理

$$
f(x)=\frac\delta2\left[1+\operatorname{sign}\left(\frac{2x}\delta-1\right)
\left|\frac{2x}\delta-1\right|^{1/k}\right],\qquad
f'(x)=\frac1k\left|\frac{2x}\delta-1\right|^{1/k-1}.
$$

实验取 $\delta=1,k=5$。在远离区间中点的位置，上式导数可由链式法则得到；$k>1$ 时在 $x=\delta/2$ 发散。区间内 $k\to\infty$ 的值趋于两端阶跃，中点仍为 $\delta/2$；不是处处光滑且有界的梯度。论文以“smooth gradients”解释其稳定性，图5则报告 DGE 比 STE 的训练曲线更稳定；后者是实验观察，不能代替前者的数学条件。

原文未完整交代多个整数区间的局部坐标归一化、中点防发散、clamp 外梯度等实现细节。没有作者实现时，不补写 epsilon 或把它当成真实舍入导数。公共页[代理函数与坐标变换](../fundamentals/quantization/post-training-and-quantization-aware-training.md#3.3%20%E4%BB%A3%E7%90%86%E5%87%BD%E6%95%B0%E5%92%8C%E5%9D%90%E6%A0%87%E5%8F%98%E6%8D%A2%E8%A6%81%E5%88%86%E5%88%AB%E6%A0%B8%E5%AF%B9)给出为何这些条件影响反向的说明。

### 4.2 预处理的尺度不是码本尺度

式11–12从 PTQ 整数初始化浮点代理：

$$P_x=s_x(W_z-t),\quad s_x=\frac{2a}{2^b-1},\quad
t=\frac{2^b-1}{2},\quad a=\sqrt{\frac6{m+n}}.$$

它把整数范围映射到 $[-a,a]$，作者称用于匹配 Xavier 的量级；实际均值和方差仍取决于已选码分布。图5报告去掉预处理后损失改善受限。

需要注意坐标衔接：若把 $P_x$ 直接代入式5的 $Q$，小范围代理会被舍入为相同的码，不能保持原 $W_z$。图3标有 scaling，但式5与式11之间未给出完整换回整数坐标的公式。本页用不同符号保留该接口缺口。作为**整理者的自洽示例**，在量化前令 $P_z=P_x/s_x+t$，再用 $Q(P_z)$，便可保留初始化码；反向还应乘坐标 Jacobian $1/s_x$。这仅说明所需条件，未宣称作者代码采用此实现。该缺口限制训练复现，不改变仿射码本和前向重排的代数解释。

## 5. 数据效率证据支持到哪里

主实验 $d=4$，基座模型用约4B FineWeb token；PTQ 统计用1024个 RedPajama 样本。训练为16张 A100、batch size 8、梯度累积8、有效 batch 1024，约1900步；附录 A.1 指定训练序列长2048。小模型学习率 $10^{-4}$，8B为 $3\times10^{-5}$，指令训练为 $10^{-5}$，使用 AdamW、warmup/stable/decay 及梯度裁剪1。以下均为作者报告，不是本地复现。

| 比较 | 结果及条件 | 可支持的判断 |
| --- | --- | --- |
| 与 PV 的模型能力比较（表1） | Qwen3-8B：LC-PTQ PPL/六任务均分14.95/67.95，PV 10.65/70.99，LC-QAT 10.23/72.18；FP16为9.72/74.12，正文称使用相同数据量 | 该设置下整网恢复优于所列 PV 基线，仍有浮点差距；不能推成任意 PV 配置的上限 |
| 与 ParetoQ 的数据预算（表2） | LLaMA3-8B：LC-QAT 4B token、QA73.42/PPL9.38；ParetoQ 30B、QA72.07/PPL8.00 | $4/30=13.3\%$，较少额外训练数据取得更高六任务均分，但 PPL 更差；作者“约10%”是概括，非精确比值 |
| 同初始化的训练变量（表5、附表10） | Qwen3-1.7B：LC-PTQ52.40，接 PV55.35，LC-QAT59.75；附录1B/4B/10B LC-QAT为59.75/61.43/62.45 | 表10将与表5相同的任务值列为1B结果，主表61.43对应4B；表5未独立注明token量，不能擅自给PV补齐预算，也不能凭共享起点证明坐标下降普遍无效 |
| 与 BitNet 的指令结果（表3） | LC-QAT1.7B均分50.35，BitNet2B48.07；LC-QAT的MMLU46.77低于53.17，MATH39.60低于43.40 | 均值改善不代表各任务改善；BitNet的4T是从头预训练，LC的表内4B是已有基座上的额外训练，0.1%不是全流程总数据节省 |

六项 QA 为 ARC-E、ARC-C、BoolQ、HellaSwag、PIQA、WinoGrande。指令表包含 OpenBookQA、IFEval、MMLU、GSM8K、MATH、HumanEval。§4.3称指令微调用约2B AM-Qwen3-Distilled token，表3却标 LC-QAT 4B；原文未消解这一差异，本页保留两种口径。ParetoQ 基线沿用已有报告，不是由本项目在同数据流水上重训；作者也说明其数据未公开，不能做对称的数据尺度实验。

附录 $d=4/8$ 的 LC-PTQ 只在 Qwen3-1.7B 上改变组宽与校准集，不能推成所有模型不敏感；14B表在约58M token后由PTQ的72.55恢复到74.32（FP16 77.31），是有限规模扩展证据。

## 6. 推理与训练成本分开判断

附录 A.2 表6在**单张 A100、LLaMA3-8B、batch 1、1024 token、启用 CUDA Graph**下报告 FP16 60.38、AQLM $2\times8$ 70.97、QuIP# E8P 79.70、LC-QAT $d=4$ 101.65 token/s；对应约1.68× FP16与1.43× AQLM。作者描述使用定制融合 kernel 和 fast-hadamard-transform；未取得此实现，不能将其改写为已有通用 Int2 后端的必然速度，也不能推广到其他 batch、硬件或服务指标。

表7的 LC-PTQ 6h加QAT55h、总61h，与ParetoQ417h的比较是**8张 A800上的估算**；不同于上述16张 A100训练配置，亦不等于同机受控实测。论文限制还包括大规模训练的推广，以及额外线性映射与优化 GEMM 的交互尚未充分研究。实际打包、DGE数值边界及预处理坐标接口需要匹配作者实现再核查。

## 来源身份

- LC-QAT: Data-Efficient 2-Bit QAT for LLMs via Linear-Constrained Vector Quantization（`https://arxiv.org/abs/2606.10531v1`），`arXiv:2606.10531v1`。式3–4定义码本与初始化，式5–8给出硬前向，式9–12给出代理与预处理；表1–5及附录 A.1–A.6给出数据、消融与速度条件。
