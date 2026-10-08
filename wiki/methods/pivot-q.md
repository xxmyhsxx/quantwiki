---
title: PIVOT-Q：学生轨迹上的稀疏动作恢复与行为锚点
type: method
tags:
  - ptq
  - vlm
  - calibration
  - low-rank
  - evaluation
sources:
  - raw/papers/2026-10-07/pivot-q/paper.pdf
  - raw/papers/2026-10-07/pivot-q/source/sections/preliminaries.tex
  - raw/papers/2026-10-07/pivot-q/source/sections/motivation.tex
  - raw/papers/2026-10-07/pivot-q/source/sections/method.tex
  - raw/papers/2026-10-07/pivot-q/source/sections/experiments.tex
  - raw/papers/2026-10-07/pivot-q/source/sections/appendix.tex
  - raw/repositories/2026-10-07/pivot-q/source/README.md
  - raw/repositories/2026-10-07/pivot-q/source/docs/experiment_protocol.md
  - raw/repositories/2026-10-07/pivot-q/source/config/pivot_q.yaml
  - raw/repositories/2026-10-07/pivot-q/source/scripts/train_pivot_q.py
  - raw/repositories/2026-10-07/pivot-q/source/pivot_q/selector.py
  - raw/repositories/2026-10-07/pivot-q/source/pivot_q/sparse_selection.py
  - raw/repositories/2026-10-07/pivot-q/source/pivot_q/train.py
  - raw/repositories/2026-10-07/pivot-q/source/pivot_q/full_distill.py
  - raw/repositories/2026-10-07/pivot-q/source/third_party/QuantVLA/gr00t/experiment/pivot_q.py
  - raw/repositories/2026-10-07/pivot-q/source/third_party/QuantVLA/gr00t/model/policy.py
  - raw/repositories/2026-10-07/pivot-q/source/third_party/QuantVLA/gr00t/model/action_head/flow_matching_action_head.py
  - raw/repositories/2026-10-07/pivot-q/source/third_party/QuantVLA/gr00t/quantization/__init__.py
  - raw/repositories/2026-10-07/pivot-q/source/third_party/QuantVLA/gr00t/quantization/duquant_layers.py
  - raw/repositories/2026-10-07/pivot-q/source/third_party/QuantVLA/gr00t/quantization/duquant_preprocess.py
updated: 2026-10-08
---

# PIVOT-Q：学生轨迹上的稀疏动作恢复与行为锚点

PIVOT-Q 在已有量化 VLA 策略上增加恢复训练。机器人执行量化学生的动作，冻结的浮点教师评估同一状态；方法根据当前与随后几步的动作差异，从轨迹各阶段选少量状态蒸馏，再用干净分布的行为锚点限制策略偏移。它主要改变**训练状态、权重和 LoRA 参数**，没有提出新的整数存储格式。

本页依据 arXiv:2610.05745v1 正文及附录 A–F，并核作者仓库 `74d6e1f8994a03eb45aae5c316a743abda2cba15` 的主 GR00T-N1.5 + QuantVLA 路径。论文的行为锚点描述、结果和 seed 口径与固定仓库存在差异，下文分别保留。代码分析为静态阅读，没有导入或执行上游，也未运行模型、机器人或 GPU。前置的量化训练与状态覆盖见 [PTQ/QAT](../fundamentals/quantization/post-training-and-quantization-aware-training.md)和[校准状态选择](../theory/calibration-and-range-selection.md#12-具身策略的状态选择与时序覆盖)。

## 1. 要恢复的是闭环行为

VLA 接收图像、机器人状态和语言指令，输出动作 chunk；执行动作后，环境产生下一次观测。量化误差因此不仅改变当前输出，还会改变后续输入分布。原任务成功率保持得好，不保证视角、光照、背景、初始位置或传感噪声改变后仍保持相同成功率。（§3–4。）

论文在学生访问的状态上比较动作，观察到差异集中于少量状态。附录 B 的 GR00T/QuantVLA 队列中，最大5%的状态差异占总差异72.3%；对数正态拟合只是所测数据的描述，选择算法不依赖这个分布成立。状态相关、参数由数据拟合，作者没有把 KS 距离当作标准独立样本检验的 p 值。

§4 表1的诊断干预直接在 rollout 中用教师动作替换选定学生动作：QuantVLA 平均成功率66.4%，随机替换68.2%，高差异替换73.8%，低差异替换66.2%。它支持检查高差异状态，但这是**教师在线接管的诊断**，不同于训练后只部署学生；不能把该成功率当作 PIVOT-Q 的结果或普遍因果定理。原文未完整交代这组干预的预算和阈值，不能确认为同预算比较。

## 2. 同状态、同噪声，只有学生控制环境

记浮点教师为 $\pi_F$，当前量化学生为 $\pi_Q$，观测为 $o_t$、语言指令为 $x$。生成式动作策略在同一随机输入 $\xi_t$ 下产生动作 chunk，比较将用于执行的第一个动作：

$$
q_t=\frac{1}{D_a}\|a_t^Q-a_t^F\|_2^2.
$$

共同观测和噪声使比较聚焦两个模型的输出差异。环境只执行学生动作，教师不生成另一条环境轨迹；更新学生后，新 rollout 自然访问新的状态。（§3、§5.1。）

固定主代码把 chunk 保存为 `[T, action_horizon, action_dim]`；`action_disagreement` 实际取 `[:,0,:7]` 的**归一化动作**，先转 FP32 再求 MSE。它没有比较整段 chunk。送给 LIBERO 的动作另经反归一化和夹爪处理，因此该 MSE 不是执行器物理单位下的误差。

`train.py::_collect_scored_rollout` 每步分别计算教师与学生 backbone，保存同一初始噪声、两者预测及观测。`sample_actions` 保留动作采样的计算图；训练时重用选定观测对应的学生 backbone 特征、噪声和教师目标，重新前向当前 action head。backbone 特征在无梯度下计算，不能据此写成整模型端到端更新。

## 3. 当前差异与未来差异怎样组成选择分数

先沿已经采集的学生轨迹向后看 $H$ 步。为明确代码的尾部行为，记 $H_t=\min(H,T-1-t)$：

$$
r_t^+=
\frac{\sum_{h=1}^{H_t}\gamma^{h-1}q_{t+h}}
{\sum_{h=1}^{H_t}\gamma^{h-1}}
\quad(H_t>0),\qquad r_{T-1}^+=0.
$$

论文式2给固定 horizon 形式；`selector.py::future_risk` 排除当前 $q_t$，按实际剩余长度截断并重新归一化。它衡量随后观察到的差异，**没有在同一状态强制另一动作并重跑环境**，因而不能解释为当前动作对未来的反事实因果影响。

教学例：$q=(1,4,2,8)$，$H=2,\gamma=0.5$，得到 $r^+=(10/3,4,8,0)$。最后一个状态的当前差异可很大，而未来分量为零；截断归一化也不意味着补造终点以后的状态。

将长度 $T$ 的轨迹分为 $K=4$ 个时间段，段号为 $\lfloor Kt/T\rfloor$，不是人工标注的抓取、移动等子任务。每段内分别给 $q$ 和 $r^+$ 排百分位：

$$
P_t=\alpha\,\operatorname{rank}_{phase}(q_t)
    +\beta\,\operatorname{rank}_{phase}(r_t^+),
\qquad \alpha=\beta=0.5.
$$

这是结合附录 A/F 和代码统一后的符号。正文式3前后对 $\alpha_t^T,\beta_t^T$ 的秩与系数措辞混用；实际代码把百分位放入两个独立的 rank 张量，再乘固定系数。包含 $n>1$ 个状态时，稳定升序排序后的第 $j$ 项取 $j/(n-1)$；并列值按原时间顺序分配不同秩，不是统计中的平均并列秩。

随后每段按 $P$ 降序贪心选4个，分数并列时优先较早时间。初始最小间隔为4，凑不满时从间隔3重新选择，依次放宽到0。**间隔只约束段内**，相邻两段的选择可以紧邻；有效间隔会被记录。主 selector 若段内状态少于4，无法满足配额而报错，不能从目标预算16推断任意短轨迹均能返回16个有效状态。（`_legacy_select_states`、`select_phase`。）

分段可以避免整个预算堆在一个高分片段；段内排名也意味着不同段的绝对 MSE 大小不直接决定预算。它改变了选择目标，不能当作对所有状态按原始误差做全局 top-k。

## 4. 选择之后，训练目标仍需行为约束

在所选集合 $\mathcal I$ 上，先用平均分数归一化、裁剪，再归一化为和为1的损失权重：

$$
w_i=\operatorname{clip}\left(
\frac{P_i}{\max(|\mathcal I|^{-1}\sum_{j\in\mathcal I}P_j,\epsilon)},
c_{\min},c_{\max}\right),\qquad
\bar w_i=\frac{w_i}{\sum_{j\in\mathcal I}w_j}.
$$

主配置取 $\epsilon=10^{-8}$、$[c_{\min},c_{\max}]=[0.5,2]$。裁剪范围属于归一化前的 $w$，不是最终 $\bar w$；例如两个分数 $(0.1,0.9)$ 得到 $w=(0.5,1.8)$、$\bar w=(5/23,18/23)$。该加权强调选定状态，没有用逆选择概率修正，不能视为全状态均匀蒸馏的无偏估计。

$$
\mathcal L_{distill}
=\sum_{i\in\mathcal I}\bar w_i\,
\frac{\|a_i^Q-a_i^F\|_2^2}{D_a},\qquad
\mathcal L=\mathcal L_{distill}+\lambda\mathcal L_{anchor}.
$$

行为锚点从 bank 中抽子集 $\mathcal S$，同样比较学生与教师动作：

$$
\mathcal L_{anchor}
=\frac1{|\mathcal S|}\sum_{s\in\mathcal S}
\frac{\|\pi_Q(s)-\pi_F(s)\|_2^2}{D_a}.
$$

两份材料的状态来源必须区分：

| 来源 | 锚点实际描述或实现 |
| --- | --- |
| 论文 §5.4 | bank 包含浮点策略原训练数据集的状态；每个优化步骤随机抽样 |
| 论文附录 A | 至多256个 clean states 的 replay buffer，每步至多4个，$\lambda=0.1$ |
| 固定主代码 `train.py` Stage 5–6 | 用独立 clean LIBERO 服务采集学生控制的短 rollout，默认最多4步；缓存特征、共同噪声和教师目标，追加到容量256的 deque，每次更新随机抽至多4个 |

主代码没有在这条锚点路径直接读取原训练集。干净分布与原数据集不是同一来源承诺；此外 replay bank 会保留旧学生访问的状态，不全是最新策略的状态。可确认的是“用参考分布约束集中修复带来的偏移”，不能据代码反写论文的取样过程。

## 5. 主 GR00T + QuantVLA 调用链与可训练变量

固定入口为 `config/pivot_q.yaml → scripts/train_pivot_q.py → pivot_q/selector.py → pivot_q/train.py`。启动器传入4段×每段4个、random=0、$H=4,\gamma=0.9,\alpha=\beta=0.5$；不能直接引用 `TrainConfig` 的旧默认2个优先+2个随机。selector 还替换训练模块的目标构造函数，因此 vendored helper 中**包含当前项**的旧 `temporal_quantization_risk` 不是正式 PIVOT-Q 入口的未来分数。

| 阶段 | 已核行为与限定 |
| --- | --- |
| 加载 | 先加载冻结浮点教师，再设置量化环境加载学生；教师保持 eval 且无梯度 |
| 量化范围 | 主配置选择语言模型的 q/k/v/o 与 gate/up/down，以及 action head FFN 投影；排除视觉、norm、embedding、lm_head、attn1 等匹配项；W4A8 不是整个图全部量化 |
| LoRA | 在 action head 中名称含 to_q/to_k/to_v 的 `nn.Linear` 上添加 rank16、alpha32 的适配器；优化器取 `requires_grad` 参数，主体由 PEFT 冻结 |
| 优化 | AdamW，lr $5\times10^{-5}$、weight decay0.01、梯度范数裁剪1；50步 warmup 后 cosine decay；每条轨迹复用同一16个目标状态做5次更新 |
| 计算目标 | `_backward_state_loss` 取首动作前7维；将目标转为预测 dtype，先相减再转 FP32 求平方均值，和评分时先将两边转 FP32 的路径有舍入差别 |
| 部署产物 | 保存 LoRA adapter；后续评测加载最终 checkpoint，教师和训练筛选不随学生部署 |

配置虽给 `lora_dropout=0.05`，模型在这条路径保持 eval，不能直接描述为训练时启用5%随机丢弃。本轮未运行 PEFT 或验证完整梯度图。

`Gr00tPolicy` 调用 `enable_duquant_if_configured`；`DuQuantLinear.forward` 对激活和变换权重做 fake quant 后，调用浮点 `torch.nn.functional.linear`，再按配置恢复输出坐标与 bias。这不是原生 INT4 GEMM 的实现证据。`fake_quantize_sym` 为 round/clamp/乘回 scale，所读函数没有自定义 STE；`sample_actions` 保留计算图也不意味着量化图所有分支都有有效梯度。这里核实恢复训练与模拟低位宽的连接，不宣称复现量化后端或吞吐。

本轮主路径已追到评分、选择、缓存、锚点、更新和模拟量化消费；其他模型的全部 adapter、量化器、子模块和执行后端尚未逐链核读。仓库 README 的 OpenVLA-OFT/QVLA 是目标平均4 bit 的混合权重配置，不能并入统一 W4A8；$\pi_{0.5}$ 使用共享 adapter 的多卡恢复，亦不能套用主 GR00T 的独立 suite 路径。

## 6. 16个状态与7.4%究竟节省什么

论文 §6.1 表2报告每条轨迹选择16个状态，FSD 平均使用199.4–242.0个状态。四组 $16/\overline T$ 的比例约为7.6%、6.9%、8.0%、6.6%；其中最后一组表内 $16/242.0=6.61\%$，正文却写7.0%。四个比值的简单平均约7.28%，不同于正文的7.4%。原文称按总状态数比较，但没有在该表提供消除差异所需的原始总数；本页保留作者7.4%的报告，不将其写成表内精确复算结果。

更关键的是，这个比例只涉及**目标轨迹进入蒸馏的状态数**：

- 仍需执行完整学生 rollout，并逐步查询浮点教师来计算全部 $q_t$，不是只调用教师16次。
- 同一组选中状态做5次 action-head 前向/反向；16不等于整轮16次计算。
- 每步更新还有至多4个 clean anchors，且需额外采集 clean rollout、缓存或重算特征。
- FSD 选择全部状态并均匀加权；固定实现对 dense 特征另有缓存安排，不能只按状态数换算墙钟比例。

因此7.4%既不是总教师查询比例，也不是端到端训练时间或部署加速倍数。代码中的 state、noise、teacher target 和特征缓存成本均属于离线恢复流程。

## 7. 作者结果与版本差异

论文表2在 LIBERO-Plus 的4个 suite、7类环境扰动共560个 episode 上报告如下成功率；位宽指对应量化管线。所有数值均为作者报告，非本项目运行。

| 模型 / 量化器 | 原模型 | 量化后 | FSD | PIVOT-Q |
| --- | --- | --- | --- | --- |
| GR00T-N1.5 / QuantVLA W4A8 | 81.4±0.8 | 66.6±0.9 | 76.9±1.6 | 81.3±0.4 |
| $\pi_{0.5}$ / QuantVLA W4A8 | 78.9±0.6 | 71.9±0.6 | 72.7±0.5 | 72.8±1.3 |
| GR00T-N1.5 / HoloQ-VLA W4A4 | 82.0±0.2 | 68.2±0.5 | 78.0±0.4 | 78.3±1.6 |
| OpenVLA-OFT / QVLA mixed W-only | 63.5±0.3 | 52.7±0.8 | 60.4±0.9 | 60.8±0.3 |

论文这张表中 PIVOT-Q 的四个均值均高于 FSD，但不是每类扰动都领先，均值差也未必显著。例如 $\pi_{0.5}$ 仅高0.1个百分点。固定仓库 README 的 HoloQ 行则为 **FSD 78.9±1.4、PIVOT-Q 78.0±1.5**；GR00T/QuantVLA 的 FSD 标准差也为1.7而非论文1.6。两个固定来源不一致，当前材料未说明如何从一组统计得到另一组，不能拼成一张统一结果表。

评测解释还有两个条件：

1. 论文 §6/表2写三次 independent seeds/runs；仓库 `docs/experiment_protocol.md` 明确训练seed0，用同一个最终 checkpoint 评估2026–2029四个seed，标准差反映评测随机性，不能称为四次独立训练。README表仅称已完成评测seed，不足以独立确认每行到底完成几次。
2. 仓库说明恢复与评测使用相同 task IDs，是同任务的传导式适配（transductive adaptation）。最终评测 episode 与恢复 rollout 分开，但这不能证明未见任务或未见扰动的泛化。方法标题中的 generalization 应随该实际协议理解。

论文表3的 GR00T/QuantVLA 同16状态预算消融：随机分段78.5±0.8、均匀分段79.4±1.0、全局贪心78.8±0.7、去未来项79.6±1.2、去锚点78.8±1.5，完整方法81.3±0.4。它支持这组设置下选择、覆盖与锚点共同起作用，不是单靠稀疏即可恢复。固定代码的 global vulnerability 分支同时改用全局秩、移除段配额及时间间隔，不能将差值全归因于一个控制量。

附录 E 的恢复日志、最终策略诊断和最终成功/失败统计不是同一批轨迹。作者按任务身份将选择诊断关联到最终结果；OpenVLA 有560个最终结果但558个选择记录。恢复数多于退化数支持净改进，不能用选中状态的平均差异确定预测某次成功；[评测页](../implementation/model-quality-evaluation.md#10-策略恢复要分清适配范围与统计单位)说明如何保留这些单位。

## 8. 与 OnPTQ 和原生整数 VLA 的关系

[OnPTQ](onptq.md)与 PIVOT-Q 都重视当前量化策略访问的状态和参考行为约束，但选择信号与更新变量不同：

| 问题 | OnPTQ | PIVOT-Q |
| --- | --- | --- |
| 状态 | 当前学生生成的精确 token 前缀 | 学生动作影响环境后获得的观测 |
| 当前损伤 | 同 token pair 的 margin erosion，加全分布 JS | 同噪声首动作的均方差 |
| 未来信号 | 强制不同 token 后的两条学生分支，由教师比较未来分布 | 同一已采集学生轨迹之后几步的动作差异 |
| 保持参考行为 | 原多模态上下文的 hidden anchor | clean/reference 状态的动作蒸馏 |
| 更新与实现证据 | 论文校准参数，匹配作者代码尚缺 | 主代码 action-head LoRA，模拟量化后浮点执行 |

PIVOT-Q 的作者代码补充了具身策略的实现实例，不能填补 OnPTQ 自身参数与更新配置的缺口。[FoldQuantVLA](foldquantvla.md)则侧重共享输入坐标一致性和原生整数投影；动作恢复效果与实际低位宽部署是需要分别验证的两个问题。

## 来源身份

| 来源 | 固定版本 | 使用范围 |
| --- | --- | --- |
| [Beyond In-Distribution Preservation: Recovering Generalization in Quantized VLAs via Vulnerability-Oriented Tuning](https://arxiv.org/abs/2610.05745v1) | arXiv:2610.05745v1 | 正文与附录 A–F；式1–7、表1–3、补充诊断/协议；关键公式与主表回查 PDF |
| [ruanruan-andy/PIVOT-Q](https://github.com/ruanruan-andy/PIVOT-Q/tree/74d6e1f8994a03eb45aae5c316a743abda2cba15) | `74d6e1f8994a03eb45aae5c316a743abda2cba15` | README/实验协议；主 GR00T+QuantVLA launcher→selector→train，vendored QuantVLA 评分、采样、加载与 fake-quant linear；其余模型详细后端未完成核读 |
