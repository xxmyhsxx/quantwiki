---
title: OSTQuant：正交与缩放联合学习、QSUR 与 KL-Top
type: method
tags:
  - ptq
  - rotation
  - equivalent-transform
  - optimization
sources:
  - raw/papers/2026-09-22/ostquant/paper.pdf
updated: 2026-09-22
---

# OSTQuant：正交与缩放联合学习、QSUR 与 KL-Top

OSTQuant 冻结预训练权重，联合学习正交矩阵和对角尺度，以整个量化网络的输出分布接近浮点教师。它用 QSUR（量化空间利用率）解释为什么要同时调整分布的方向与轴长，用 WOMI 提供权重感知初始化，再用 KL-Top 聚焦教师概率较高的词。**QSUR 是分析指标，实际优化目标是输出损失；浮点等价也不保证量化后没有过拟合。**

本页依据 arXiv:2501.13987v1，研读正文及相关附录。v1 的椭球极值推导、KL-Top 归一化、RoPE 融合条件和部分实验设置存在缺口，以下区分原文报告、条件化教学推导和待代码核对事项。未读取官方代码或运行模型。

## 1. 可学习的对象与方法位置

[SpinQuant](spinquant.md) 已解释全局残差旋转、头内 V/O 旋转和整网损失优化；[OmniQuant](omniquant.md) 解释可学习尺度与裁剪。OSTQuant 的增量是把正交与缩放配对，在多个位置共同学习，并用浮点教师的 top-k 输出信息替代单一 next-token 标签目标。（§4、图 5。）

采用行 token 约定：$x\in\mathbb R^{n\times d}$、$W_1\in\mathbb R^{d\times m}$、$W_2\in\mathbb R^{m\times o}$。令 $O\in\mathbb R^{m\times m}$ 正交，$S=\operatorname{diag}(s)$ 且 $s_i\ne0$。在线性相邻位置，

$$xW_1W_2=(xW_1OS)(S^{-1}O^{\mathsf T}W_2).$$

量化后实际比较的是 $Q_a(xW_1OS)Q_w(S^{-1}O^{\mathsf T}W_2)$ 与原输出。$Q$ 在这里包含反量化值，不能把两组整数编码直接相乘当成同一式子。顺序很重要：$(OS)^{-1}=S^{-1}O^{\mathsf T}$，通常不能交换 $S$ 与 $O$。v1 §4.1 文字把变换写为 $T=\Lambda O$，式 9 却使用 $O\Lambda$；本页按式 9 的相消顺序统一记号，不隐去这个差异。

**整个变换不一定正交。**$P=OS$ 满足 $P^{\mathsf T}P=S^2$，条件数为 $\max_i|s_i|/\min_i|s_i|$；只有各尺度绝对值相同，才是统一缩放的正交变换。单个 $OS$ 也不能表示任意可逆矩阵，因为列向量仍两两正交。一般可逆族及条件数见 [矩阵基础](../fundamentals/mathematics/invertible-transforms-and-kronecker-products.md)。这些是恒等式的教学推导，不是额外的 OSTQuant 算法。

## 2. QSUR 想解释什么，以及推导哪里受限

§3 定义 $\mathrm{QSUR}=V_X/V_{S_X}$：分子为数据占据的体积，分母为量化范围形成的超立方体体积。原文为高斯 $X\sim\mathcal N(\mu,\Sigma)$ 取置信椭球，令 $c=\chi_d^2(\alpha)$，其体积为

$$V_X=\frac{\pi^{d/2}}{\Gamma(d/2+1)}c^{d/2}\sqrt{\det\Sigma}.$$

这并非直接计算有限样本点集的体积：有限点集在连续空间的体积为零；高斯也没有有界的完整支撑。需要先选置信水平、协方差估计和范围口径。图 3 使用 $\mathrm{QSUR}^{1/d}$ 作为归一化展示，并报告它与九任务准确率保留率相关；相关性不能直接给出量化误差界。

**不能把原文式 3–7 当作任意椭球的精确外接范围。**它们主要从特征向量方向的轴端点估计坐标极值。实际椭球 $x=\mu+\Sigma^{1/2}u,\ \|u\|_2\le\sqrt c$ 的第 $j$ 个坐标上下界为

$$x_j^{\max/\min}=\mu_j\pm\sqrt{c\Sigma_{jj}}.$$

这是对 $e_j^{\mathsf T}\Sigma^{1/2}u$ 用 Cauchy–Schwarz 得到的教学推导；涉及全部特征方向，而非仅最大特征向量。若使用所有坐标共享的上下界，立方体边长应为 $\max_j(\mu_j+\sqrt{c\Sigma_{jj}})-\min_j(\mu_j-\sqrt{c\Sigma_{jj}})$。零均值时的体积比、白化以及二维反例详见 [旋转机制页的 QSUR 分析](../theory/orthogonal-rotation-and-hadamard-quantization.md#7-qsur-的椭球解释与白化边界)。

原文式 21 将单位向量最大分量下界写为 $d^{1/2}$；应使用绝对最大值 $\|q\|_\infty\ge d^{-1/2}$。式 6 的负端符号与式 4/22 不一致，式 7 又没有保留绝对值。即使修正这些排印/符号问题，主轴端点也不能替代坐标极值。附录 A.2.3 的白化结论可在正定、零均值椭球模型下另行证明，但不能据此声称找到了整个量化网络的最优变换。

为什么正交加尺度仍有意义？正交变换改变方向、保持协方差谱；尺度允许改变不同方向的方差。两者配合能把单个正定高斯白化，但网络必须同时处理激活、逆变换后的权重及多个共享分支。主文 Lemma 1 的“Hadamard 后近似球形高斯”也需要分布条件：正交变换不改变协方差特征值，不能把任意各向异性高斯变成真正球形高斯。

## 3. 图中哪些参数学习，哪些仍要在线

以下以主方法图 5、§4.1 为准。图例将 $R_{res}$、$R_{ov}$ 标为可训练正交矩阵；$R_{qk}$、$R_{dn}$ 是 Hadamard。不能把“四组变换”理解成每处都训练一个任意稠密旋转。

| 位置 | 学习变量 | 相消机制和部署边界 |
| --- | --- | --- |
| 全局残差 | $R_{res}$ | embedding、残差支路输出、输入投影与最终 head 配套旋转；无仿射 RMS 归一化保范数，可融合入权重 |
| 每块两个 Norm 后 | $S_{attn},S_{ffn}$ | 尺度放入 Norm 输出增益，逆尺度放入所有消费该输出的投影；不是让一般缩放穿过 RMS 归一化 |
| 各头 V/O | $R_{ov}^h,S_{ov}^h$ | 同一头内先变换 V，再由 O 输入侧抵消；两侧可并入投影权重 |
| Q/K | $S_{qk}$，另加固定 $R_{qk}$ | RoPE 后成对逆尺度保持点积，同乘 Hadamard 也保持点积；尺度提前融合需满足 RoPE 交换条件，Hadamard 保留在线 |
| FFN up/down | $S_{u\mid d}$，另加固定 $R_{dn}$ | 对角尺度能经逐元素乘法放入 up 分支；Hadamard 作用在门控输出后，其逆并入 down 权重 |

全局残差只用正交矩阵，是因为 $\operatorname{RMS}_0(xR)=\operatorname{RMS}_0(x)R$；一般 $OS$ 没有这个性质。原 Norm 的增益和所有分支也必须配套处理，具体代数见 [正交旋转与量化](../theory/orthogonal-rotation-and-hadamard-quantization.md)。

对单头，令 $A_h$ 为 attention 概率矩阵、$V_h=X_hW_v^h$，式 11 的相消为

$$A_h(V_hR_{ov}^hS_{ov}^h)\bigl((S_{ov}^h)^{-1}(R_{ov}^h)^{\mathsf T}W_o^h\bigr)=A_hV_hW_o^h.$$

这里混合头内特征，不任意混合具有不同 $A_h$ 的头；GQA 中共享 K/V 的头如何共享变换，v1 的逐头写法不足以规定全部实现细节。

**RoPE 是需要保留的条件。**§4.1 称 $S_{qk}$ 可并入 Q/K 权重，但“位置编码具有乘法形式”不足以推出这一点。若对角尺度提前到 RoPE 前，它必须与对应位置旋转交换；每个实际 RoPE 坐标对使用相同尺度是充分条件。任意独立通道尺度通常失败，已有 [RoPE 缩放反例](../theory/diagonal-scaling-equivalent-transform.md#7-平移与注意力位置编码的融合边界) 给出证明。未核对代码前，不能声称 OSTQuant 已采用这一约束，也不能由论文文字断言实现一定错误。

FFN 则有更直接的逐元素恒等式。令 $z=\operatorname{SiLU}(g)\odot u$，则 $zS=\operatorname{SiLU}(g)\odot(uS)$。若接着应用正交 Hadamard $H$，需保存 $H^{\mathsf T}S^{-1}W_{down}$，因为 $(zSH)(H^{\mathsf T}S^{-1}W_{down})=zW_{down}$。一般不能把 $H$ 搬进 SiLU。这是按图 5 解释的教学改写，也说明“学到的变换可融合”不等于整套方法没有在线变换成本。

## 4. WOMI 初始化与受约束优化

WOMI（Weight Outlier Minimization Initialization）先从权重统计选择方向。§4.1 将接收残差输入的投影权重按共享输入维组织成 $W_{stack}\in\mathbb R^{N\times d}$，从其输入通道协方差取特征向量 $U_W$，初始化

$$R_{res}=U_WH^{\mathsf T},$$

其中 $H$ 已归一化，尺度初始化为单位矩阵。原文文字称“沿输入通道拼接”，但其 $n\cdot oc\times ic$ 形状对应保留输入宽度、堆叠输出行；实现应以实际布局确认。头内变换同样按头分块做权重感知初始化。（§4.1；附录 A.2.2、A.3.1。）

依据协方差 $U_W\Lambda U_W^{\mathsf T}$ 可推得：旋转后的协方差是 $H\Lambda H^{\mathsf T}$；归一化 Hadamard 的元素绝对值相同，所以对角元素都为 $\operatorname{tr}(\Lambda)/d$。这给出均衡坐标方差的解释，**不保证每个有限样本的最大值下降，不保证激活也同时最优**。表 7 中 LLaMA-3-8B W4A4KV4 的 WOMI/随机 Hadamard 九任务均分仅为 61.40/61.38，不能把完整训练收益全归给初始化。

优化时冻结基座参数，以整个网络的教师/学生输出损失更新变换。论文使用 RiemannAdam：正交变量需在 $O^{\mathsf T}O=I$ 的约束下更新，尺度需保持可逆。普通逐元素 Adam 更新不能自动保持正交，普通尺度更新也不会自动防止过零。Stiefel 切空间与约束更新的基础见 [矩阵基础](../fundamentals/mathematics/invertible-transforms-and-kronecker-products.md#cayley-参数化与约束更新不是同一个步骤)；v1 未完整给出所用流形度量、回缩、动量运输、尺度参数化及舍入代理梯度，不能从优化器名称补写实现。

学习参数少也仍需要整网反向；教师前向、激活保留、优化器状态和量化模拟影响显存。v1 没有给出足够细节来把它等同于 SpinQuant 的“W16 学旋转后再 GPTQ”流程。

## 5. KL-Top：选择哪些教师信息，以及归一化歧义

令每个有效预测位置的浮点教师分布为 $p$、量化学生分布为 $q$，$I=\operatorname{TopK}(p)$。§4.2 式 12–13 写成

$$L_{partial}=\sum_{i\in I}p_i\log\frac{p_i}{q_i}.$$

教师及其 top-k 选择不随学生反向更新。它希望在有限校准数据下保留多个高概率候选的信息，避免只拟合单一 next-token 标签。但 v1 §5.2 又提到比较 top-k 前后 softmax，未明确最终采用的归一化次序。以下两种操作不同，不能默认互换：

| 操作 | 损失含义 | 对学生词表的影响 |
| --- | --- | --- |
| 全词表 softmax 后直接截取求和 | 原式的部分和，保留教师 top-k 的概率质量 | 非完整 KL，可能为负；选集外 logits 仍通过 softmax 分母影响梯度 |
| 选教师 top-k 索引，再分别在这些 logits 上 softmax | 条件分布 $p_i/m_p$ 与 $q_i/m_q$ 的 KL，$m_p=\sum_Ip_i,m_q=\sum_Iq_i$ | 对固定选集外的 logits 没有该损失的直接梯度，也不约束学生落在选集内的总概率质量 |

为说明差异，设学生 logits 为 $a$、$q=\operatorname{softmax}(a)$，教师固定。对第一种写法有

$$\frac{\partial L_{partial}}{\partial a_j}=m_pq_j-p_j\mathbf1_{j\in I}.$$

这是教学推导：截取求和不等于只更新被选 logits。例 $p=(0.6,0.4)$、$q=(0.8,0.2)$、$k=1$，部分和为 $0.6\log(0.6/0.8)\approx-0.17261$；条件分布都只有一个类别，KL 则为 0。二者的数值与梯度完全不同。

全 KL 中单类教师概率小，并不自动表示该类无信息；遗漏类的累计质量、学生错误置信度也重要。作者将尾部解释为噪声，是方法动机与消融支持下的判断，不是对所有模型的定理。top-k 还可能减少损失计算量，但若仍产生全词表 logits、计算完整 softmax 和执行 top-k，就不能据 $k/V$ 直接宣称整体计算或显存按同比例下降。

表 6 在 LLaMA-2-7B 的 W4A4KV4 设置中，$k=1000$ 的九任务均分为 63.18，$k=5/10000$ 为 62.40/62.11；$k=5000$ 的 PPL 5.93 反而优于 $k=1000$ 的 5.96。表 8 中 LLaMA-3-8B 的 SpinQuant 加 KL-Top 后均分 64.10→64.07、PPL 7.35→7.54；OSTQuant 对应 65.13→65.37、6.80→7.29。它支持损失与参数空间有交互，也提醒 PPL 与多任务指标不必同向改善。

## 6. 实际流程与可复现边界

1. 确定 W/A/KV 配置和等价图，先验证关闭量化时的输出一致性；Norm、残差、RoPE 和多消费者都要配套。
2. 按权重协方差做 WOMI，尺度设为 1，放置固定 Q/K 与 down Hadamard。
3. 冻结原模型参数，用浮点教师提供输出分布，在量化学生上整网优化正交/尺度参数；实现时必须明确 KL-Top 归一化和量化反向约定。
4. 融合可融合参数，保留必要在线变换，形成最终量化权重与运行时配置，再验证误差、打包和执行结果。学习完矩阵并不等于已经生成可部署模型。

上述步骤按 §4–5 组织；其中实现检查为整理者提出的验证条件。§5 报告激活 per-token 非对称、权重 per-channel 对称；校准取 WikiText2 的 1000 段、每段 2048 token，batch 8，150 次迭代，cosine 学习率衰减。具体 KV 分组/尺度布局、量化模拟与最终权重算法的衔接、训练集抽样/重叠和温度/归约等细节不能仅从这些描述补全。

**学习率存在原文冲突。**§5 给正交/尺度初始值 0.02/0.03；表 5 定义 LR1 为尺度、LR2 为正交，却给 RiemannAdam 的 7B 最佳值 0.02/0.001；邻近文字又说正交学习率应比尺度大 10 倍。三者不能合成唯一配方，当前保留差异，等待固定代码及配置核验。

## 7. 实验支持与反例

以下均为 v1 作者报告。表 2 使用 lm-evaluation-harness 0.4.4 的九任务平均：BoolQ、HellaSwag、LAMBADA OpenAI、OBQA、PIQA、SIQA、WinoGrande、ARC-e、ARC-c；PPL 为 WikiText2。它与 SpinQuant 原文八任务均值不能跨表直接排名。

| 配置与问题 | 同表结果 | 可以支持的结论 |
| --- | --- | --- |
| LLaMA-3-8B，W4A16KV16 | FP 68.09；OSTQuant 67.80；QuaRot 67.27；SpinQuant 66.54 | OSTQuant 此项保留约 99.57% 的 FP 均分，仍非无损 |
| LLaMA-2-7B，W4A16KV16 | FP 65.21；OSTQuant 64.37 | 保留约 98.71%，不支持所有模型都至少 99.5% 的文字概括 |
| LLaMA-3-8B，W4A4KV4 | FP 68.09；SpinQuant 64.10；OSTQuant 65.37 | 缺口由 3.99 缩至 2.72 个百分点，约缩小 31.8%；不是准确率提高 32 个百分点 |
| LLaMA-2-7B，W4A4KV4 | SpinQuant 62.01/PPL 5.96；OSTQuant 63.18/5.91 | 在本文协议下两指标改善；不替代跨论文配置对齐 |

表 4 是顺序累计消融，LLaMA-2-7B W4A4KV4 从基线 33.51，经 $R_{res}$ 到 54.33，再加入残差尺度到 **53.74**，之后加入 down Hadamard 到 61.75，最终 63.18。尺度不是在每个阶段都改善九任务平均；顺序消融也不能当作每个组件独立贡献的可加分解。

不同表间还保留数值差异：主表/表 4 的 7B 完整配置是 63.18/PPL 5.91；KL-Top 表 6 为 63.18/5.96；表 8 则为 63.11/5.94。没有给出足够设置差异解释这些变化，不任选一个作为所有实验的统一结果。没有多种子置信区间时，小幅差异也不自动说明统计显著。

## 8. 速度、内存与尚未完成的扩展

表 3/9 的 prefill 是 **单个 Transformer block**，RTX 3090、batch 4；附录 A.4 说明 INT4 matmul 用 CUTLASS、attention 用 PyTorch SDPA，测 500 次取中位数。例如 LLaMA-3-8B 形状、长度 2048 的 block 为 FP16 57.470 ms、INT4 23.631 ms，作者列加速 2.432 倍；相应内存 0.513/0.185 GB，列节省 2.774 倍。长度 8192 的节省只约 2.003 倍，不能复述成所有场景都超过 3.5 倍。表格数值有舍入，比例按作者单列值保留。

表 10 另报生成速度；A.4 将 70B 放在单张 A6000，量化后为 14.68 token/s、38.41 GB，FP 基线 OOM。没有可比较的 FP 耗时，就不能给该模型算加速倍数；该表的 batch、上下文与生成长度不足，不能混用前述 3090 block 设置来补齐。

离线优化也另算：主文称 7B/13B 约 20 分钟，表 11 却列 0.3/0.8 小时；8B 为 0.4 小时。引言的 A800/8B 约 20 分钟不能自动作为表 11 所有模型的完整硬件配置。这里报告差异，不把量化准备速度等同于推理加速。

附录 A.5、图 9 是包含更多 RoPE/SiLU 节点的全量化设想，明确将实验留作未来工作；不能把 W4A4KV4 主表解读成每个非线性和中间节点都已经四位执行。

## 9. 与既有可学习路线怎样衔接

沿现有 Wiki，可把“学习什么”分为不同对象：[LSQ](lsq.md) 学量化步长；[OmniQuant](omniquant.md) 学尺度/平移与裁剪；[SpinQuant](spinquant.md) 学受图结构约束的正交旋转；[FlatQuant](flatquant.md) 学局部结构化可逆变换；[LoftQ](loftq.md) 则构造低秩初始化并在后续任务中训练适配器。它们不是同一参数空间的简单换名。

OSTQuant 对这条路径的贡献是正交和尺度的共同选择、权重感知初始化与教师 top-k 整网目标。其优势应由同协议消融判断；QSUR 的高斯几何直觉、浮点等价和部署收益是三类不同证据。本文已补齐它们之间的解释，同时保留 KL-Top 实际归一化、RoPE 尺度约束、优化器和最终量化/导出格式等代码核验缺口。

## 来源身份

| 来源 | 版本或快照 | 说明 |
| --- | --- | --- |
| [OstQuant: Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for Better Distribution Fitting](https://arxiv.org/abs/2501.13987v1) | `arXiv:2501.13987v1` | 本页直接来源；公式、表格差异按此版本保留 |

## 教学计算材料

配套 [check_math.py](../assets/ostquant/check_math.py) 和 [math-validation.json](../assets/ostquant/math-validation.json) 验证本页及 QSUR 机制补充中的恒等式与反例；使用 NumPy CPU 小矩阵，不执行官方优化器、模型或 GPU kernel。
