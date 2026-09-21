---
title: PTQ、QAT 与代理梯度
type: concept
tags:
  - ptq
  - qat
  - optimization
sources:
  - raw/papers/2026-09-21/spinquant/paper.pdf
  - raw/papers/2026-09-21/omniquant/paper.pdf
  - https://github.com/OpenGVLab/OmniQuant/blob/feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14/quantize/quantizer.py
  - raw/papers/2026-09-21/qlora/paper.pdf
  - raw/papers/2026-09-21/q-vlm/paper.pdf
  - raw/papers/2026-09-21/quantization-white-paper/paper.pdf
  - raw/papers/2026-09-21/integer-only-quantization/paper.pdf
updated: 2026-09-15
---

# PTQ、QAT 与代理梯度

PTQ 从已训练模型出发完成量化转换，可能包含统计、搜索或局部优化；QAT 在训练前向中模拟量化影响，并通过训练调整模型。两者不是“完全不优化”和“做优化”的二分，也不是无数据与有数据的二分。判断时应说明调整哪些变量、使用什么目标与数据、梯度经过哪些层。

本页依据量化白皮书 v1（W）§3–4，以及 Jacob 等整数推理论文 arXiv v1（J）§3、附录 C–D。两篇的 QAT 实验主要属于特定模型与训练设置，不代表量化训练的全部可能形式。

## 1. 区分范式与可调整变量

| 路线 | 典型变量和目标 | 边界 |
|---|---|---|
| PTQ 范围选择 | 网格端点、步长、零点，张量或输出误差 | 可需要校准输入和候选搜索 |
| PTQ 图变换与校正 | 通道缩放、偏置修正 | 一些步骤利用权重或解析统计，另一些使用真实输入 |
| PTQ 局部优化 | 舍入变量、剩余权重，层重构损失 | 即使使用梯度或连续权重调整，仍不能自动称为端到端 QAT |
| QAT | 量化前向下的模型参数，可能加上网格参数 | 成本与训练目标、数据和可训练参数有关；并不意味着训练计算本身都是低比特 |

W §3.4 将通过软舍入变量优化局部重构的 AdaRound 归入 PTQ。[GPTQ](../../methods/gptq.md)利用局部二阶信息调整待量化权重，[AWQ](../../methods/awq.md)搜索缩放与裁剪，均说明“PTQ 不改变任何权重或不做任何优化”的定义过窄。局部误差的机制见 [层输出重构与二阶补偿](../../theory/layer-reconstruction-second-order-compensation.md)。

这些分类有助于解释成本和证据，不应只靠方法名称决定实现。报告至少说明是否更新浮点参数、是否需要标签/教师、优化哪些层、训练多久，以及量化网格是否在训练中变化。

[OmniQuant](../../methods/omniquant.md) 是完整的可学习 PTQ 实例：冻结基座权重，逐 Transformer block 学习裁剪强度和等价变换参数，用浮点参考输出与量化路径输出的 MSE 校准，不需要回答标签。优化时的变换权重和最终导出权重会变化，冻结的是基座优化变量；需要反向传播，也不能据此称为全模型 QAT。作者在 LLaMA 纯权重量化中只启用裁剪，而权重激活量化中联合优化两类参数，说明可训练变量应服务具体量化问题。（OmniQuant v3 §3、§4.1、算法 1、表 A3。）

[Q-VLM](../../methods/q-vlm.md) 进一步说明，应区分论文中“优化视觉编码器”的意图和实际可训练变量：其 §3.3 主要描述视觉量化函数及联合损失，固定公开入口却是无梯度校准，不能直接说已完成视觉权重微调。另一项 QLoRA v1 §2–3 则明确冻结量化基座并更新 LoRA 适配器，不能仅凭使用 NF4 就把一次量化加载称为 QLoRA 微调。


[SpinQuant](../../methods/spinquant.md) 把“冻结原权重”和“整网反向”组合在一起：学习残差与 Value 的正交旋转，目标是最终 next-token 交叉熵；主流程学习时使用 W16 与低比特激活，之后才以 GPTQ 量化权重。它仍被作者归为 PTQ，但明显超出了仅做局部层重构的成本模型。这说明记录优化变量、梯度传播范围、目标和前向位宽，比只标 PTQ/QAT 更能解释方法。（SpinQuant v4 §3.2、§4.2、表 3。）

## 2. 前向模拟什么，部署执行什么

设潜在浮点权重为 $w$，前向使用

$$
\widehat w=\Delta\left[\operatorname{clip}\left(\operatorname{round}(w/\Delta)+z,q_{\min},q_{\max}\right)-z\right].
$$

潜在参数可以积累小于一个量化步长的更新，下一次前向再投影到网格；如果直接把每一步更新写回整数权重，小更新可能不断消失。W §4.1 采用这类浮点潜变量与代理梯度。它没有证明权重梯度、优化器状态与累加均可低比特化。

模拟节点要对应部署真正保存或转换的张量边界。融合的卷积/线性层、偏置和激活函数可能只在最终输出量化一次；残差两支、BN 融合后的权重也需按实际图处理。J 图 1.1、附录图 C.1–C.8 展示了这些区别。任意给每个浮点算子后面插一个量化节点，可能模拟了另一张图。

J §3 中偏置“不量化”指没有像权重一样施加低比特 fake quant；部署时它仍按输入与权重的步长乘积编码为 int32。完整单位换算和输出再量化见 [整数执行展开](../../implementation/quantized-matmul-scaling-execution.md#6-从整数累加到下一层网格)。

## 3. 为什么需要 STE

舍入的真实导数在区间内部几乎处处为零，在跳变处不可导。直接通过真实导数反传，不能给潜在权重提供通常需要的更新信号。Straight-Through Estimator（STE）在反向使用人为选择的代理梯度；它不是证明舍入函数的导数等于 1。

对 $z=0$、固定整数端点 $n,p$ 的简化量化器

$$
\widehat x=\Delta\operatorname{clip}(\operatorname{round}(x/\Delta),n,p),
$$

常见反向近似为

$$
\frac{\widetilde{\partial}\widehat x}{\partial x}
=\begin{cases}1,& n<x/\Delta<p,\\0,&x/\Delta<n\ \text{或}\ x/\Delta>p.\end{cases}
$$

边界点取值必须由具体实现约定。这里按裁剪前的连续变量选择导数区间，是 W §4.1、图 10 的代理计算规则，不是用有限差分得到的真实量化导数。

若步长也参与学习，对相同计算图应用该代理链式法则，有

$$
\frac{\widetilde{\partial}\widehat x}{\partial\Delta}
=\begin{cases}
n,&x/\Delta<n,\\
\operatorname{round}(x/\Delta)-x/\Delta,&n<x/\Delta<p,\\
p,&x/\Delta>p.
\end{cases}
$$

内部项来自外部乘以 $\Delta$ 的导数与内部除以 $\Delta$ 的导数相减；饱和处只剩端点倍数。若仿射零点用连续潜变量、反向近似其舍入导数，则零点的代理导数在非饱和区为 0，在饱和区为 $-\Delta$。改变参数化、裁剪次序或梯度缩放后，公式也可能变化。（W 式 (36)–(40)。）

步长必须保持正值，零点与部署编码范围也要满足约束。W 讨论的 LSQ 等方法还会调整梯度尺度；本页没有完整研读这些方法的原论文，不把上述基本链式法则写成某个完整训练算法。代理梯度的有效性仍需模型实验，而不是仅检查公式可计算。

代理规则应核对到具体计算节点。OmniQuant 官方 commit `feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14` 的 `quantize/quantizer.py:round_ste` 使用 `(round(x)-x).detach()+x`，前向仍取整，反向近似为恒等；权重零点却使用普通 `.round()`，其梯度为零。其学习步长来自 sigmoid 强度与当前权重极值的组合，不是直接套用一个独立可训练步长。不能把“使用 STE”扩写成所有离散节点都使用同一种反向规则。

代理梯度的选择并非无关紧要。[AdaRound](../../methods/adaround.md) 在同一局部重构目标上比较过直通估计器与带显式正则的软松弛，前者精度为 66.63、后者为 68.60（ResNet18，ImageNet），作者把差距归因于 STE 的有偏梯度限制了受限空间中的优化；[BRECQ](../../methods/brecq.md) 则指出激活无法使用舍入式参数化，因为它们随输入变化，只能学习步长。两处都说明：代理规则要与被代理的离散结构匹配，不能只看反向是否可计算。

## 4. 初始化与训练过程为什么仍然重要

W 图 8、12 的 PTQ/QAT 流程都先考虑部署量化配置、范围初始化和可行的图变换。更好的初始化可以减少训练需要恢复的损失，也可能防止低位宽下训练失败。

W 表 9 的 MobileNetV2 per-tensor W4A8 案例中，基础初始化从 0.10 开始，QAT 后仍为 0.10；加入跨层均衡后初始为 12.99，QAT 后为 70.13；再加偏置校正初始为 46.90，QAT 后为 70.07。它说明这个困难配置中初始化重要，同时说明更高的初始分数并不保证更高的最终分数。表 8 的另一些配置则在训练后缩小了 min–max 与 MSE 初始化的差距。均为作者报告，不能写成 QAT 必须或完全不必做 PTQ 初始化。

J §3.1 延迟启用激活量化，并用指数移动平均跟踪范围，以避免训练初期分布剧烈变化；附录 D 的多项任务使用 500,000 步延迟。该数量依赖长训练协议、batch 与异步工作进程，不能直接移植到另一模型。[校准与范围页](../../theory/calibration-and-range-selection.md)解释统计与数据用途的区别。

## 5. BN 与量化训练图的条件

推理时 BN 统计固定，可以并入前一仿射算子；训练时 batch 统计变化，直接把它当成常数会改变训练过程。J 图 C.7–C.8 使用额外路径估计统计，再模拟融合后的权重量化。W §4.2、表 7 比较静态融合、额外前向估计及保留 BN 等策略，并在相应实验中调学习率。不能只看“多跑一次前向”就判断质量或总成本。

融合的代数和适用条件见 [BN 与跨层均衡](../../theory/diagonal-scaling-equivalent-transform.md#6-固定-bn-融合与跨层均衡)。普通 LayerNorm/RMSNorm 的统计依赖当前输入，不能把整层照 BN 推理公式并入固定权重。基础的图一致性要求共通，具体融合方式由算子决定。

## 6. 怎样比较 PTQ 与 QAT 的结果

W 表 6（PTQ）与表 10（QAT）包含不同训练、量化和重复次数：PTQ 通常报告 5 次，QAT 3 次；后者还选择学习率并针对不同任务训练不同轮数。BERT 的 PTQ 表含部分 16-bit 激活，而对应 QAT 配置为全 8-bit 激活。BERT 这里是编码器任务指标，不等于自回归大模型的困惑度或生成质量。

公平比较需同时列出浮点基线是否经过同等微调、训练/校准数据、量化节点、各层实际位宽、粒度、优化成本与独立评测。QAT 高于原浮点分数，也可能受额外训练影响；仅凭该结果不能证明量化带来正则化收益。W §4.5 的跨模型表格支持特定配置的可行性，不支持“QAT 总是胜出”或“per-channel 总是更好”。

模型质量、低比特存储与实际后端性能分别验证。[误差诊断与验证](../../implementation/quantization-error-diagnosis.md)给出从图一致性到任务评价的检查顺序。本页完成文献解释，没有运行 QAT 或 PTQ 模型复现。

> 来源维护（2026-09-21）：上列固定 commit 的外部代码引用对应已移除的本地快照；保留原版本身份，本轮未重新审查相关代码结论。

## 来源身份

下表用于在没有本地资料库时辨识来源；具体论述的章节、公式、图表或代码位置见正文。

| 来源 | 版本或快照 | 说明 |
| --- | --- | --- |
| [SpinQuant: LLM quantization with learned rotations](https://arxiv.org/abs/2405.16406v4) | `arXiv:2405.16406v4` | — |
| [OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models](https://arxiv.org/abs/2308.13137v3) | `arXiv:2308.13137v3` | — |
| [OpenGVLab/OmniQuant](https://github.com/OpenGVLab/OmniQuant/blob/feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14/quantize/quantizer.py) | `feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14` | 本地快照已移除 |
| [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314v1) | `arXiv:2305.14314v1` | — |
| [Q-VLM: Post-training Quantization for Large Vision-Language Models](https://arxiv.org/abs/2410.08119v3) | `arXiv:2410.08119v3` | — |
| [A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295v1) | `arXiv:2106.08295v1` | — |
| [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877v1) | `arXiv:1712.05877v1` | — |
