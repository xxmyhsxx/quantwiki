---
title: PTQ、QAT 与代理梯度
type: concept
tags:
  - ptq
  - qat
  - optimization
sources:
  - raw/papers/2026-09-21/lora
  - raw/papers/2026-09-22/lrq
  - raw/papers/2026-09-22/signround
  - raw/papers/2026-09-22/flexround
  - raw/papers/2026-09-21/quasar
  - raw/papers/2026-09-21/lc-qat
  - raw/repositories/2026-09-21/aqlm
  - raw/papers/2026-09-22/pv-tuning
  - raw/papers/2026-09-22/loftq
  - raw/papers/2026-09-22/efficientqat
  - raw/papers/2026-09-21/llm-qat
  - raw/papers/2026-09-22/lsq
  - raw/papers/2026-09-21/spinquant
  - raw/papers/2026-09-21/omniquant
  - https://github.com/OpenGVLab/OmniQuant/tree/feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14
  - raw/papers/2026-09-21/qlora
  - raw/papers/2026-09-21/q-vlm
  - raw/papers/2026-09-21/quantization-white-paper
  - raw/papers/2026-09-21/integer-only-quantization
  - raw/papers/2026-09-21/multimodal-qat
  - raw/repositories/2026-09-21/qslaw
updated: 2026-10-13
---

# PTQ、QAT 与代理梯度

PTQ 从已训练模型出发完成量化转换，可能包含统计、搜索或局部优化；QAT 在训练前向中模拟量化影响，并通过训练调整模型。两者不是“完全不优化”和“做优化”的二分，也不是无数据与有数据的二分。判断时应说明调整哪些变量、使用什么目标与数据、梯度经过哪些层。

本页从量化白皮书 v1（W）§3–4，以及 Jacob 等整数推理论文 arXiv v1（J）§3、附录 C–D 出发，结合 LSQ 和 LLM 方法解释优化变量、代理梯度、训练目标与部署图。各方法的效果保留其模型、数据与前向配置。

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

[Q-VLM](../../methods/q-vlm.md) 进一步说明，应区分论文中“优化视觉编码器”的意图和实际可训练变量：其 §3.3 主要描述视觉量化函数及联合损失，固定公开入口却是无梯度校准，不能直接说已完成视觉权重微调。另一项 QLoRA v1 §2–3 则明确冻结量化基座并更新 LoRA 适配器，不能仅凭使用 NF4 就把一次量化加载称为 QLoRA 微调。原始[LoRA](../../methods/lora.md)在固定浮点基座上以任务损失学习加法低秩增量，参数减少本身不构成量化；它的固定$\alpha/r$控制增量幅度，也不同于LRQ的选码除数或量化恢复步长。（LoRA v2 §2、§4.1。）


[SpinQuant](../../methods/spinquant.md) 把“冻结原权重”和“整网反向”组合在一起：学习残差与 Value 的正交旋转，目标是最终 next-token 交叉熵；主流程学习时使用 W16 与低比特激活，之后才以 GPTQ 量化权重。它仍被作者归为 PTQ，但明显超出了仅做局部层重构的成本模型。这说明记录优化变量、梯度传播范围、目标和前向位宽，比只标 PTQ/QAT 更能解释方法。（SpinQuant v4 §3.2、§4.2、表 3。）

[LLM-QAT](../../methods/llm-qat.md) 展示整网输出蒸馏路线：量化学生使用生成文本，匹配浮点教师的词表概率分布，并把 K/V 量化放进训练前向。其“Data-Free”免除的是访问原始训练集的要求，仍有数据生成、教师计算和学生训练成本；主方法按极值计算尺度，并不等于采用 LSQ 学习步长。

[EfficientQAT](../../methods/efficientqat.md) 则把两种训练范围串联：先逐块更新潜在权重与网格参数，拟合浮点块输出；再固定整数权重和零点，整网只训练分组尺度。局部阶段自由度大，整网阶段可训练参数少，但仍需整网反向；两阶段的损失与内存来源不能合并为一个“QAT 成本”。

| 机制实例 | 量化对象与主要变量 | 目标与梯度范围 |
|---|---|---|
| LSQ | CNN 权重、激活；潜在权重与逐层步长 | 任务损失；整网训练，步长采用 STE 和专门的梯度缩放 |
| LLM-QAT | LLaMA 权重、激活、K/V；量化学生权重 | 生成前缀上的浮点教师分布蒸馏；整网训练 |
| EfficientQAT Block-AP | 主实验为权重量化；当前块权重、尺度、零点 | 当前块输出重构；梯度限于当前块 |
| EfficientQAT E2E-QP | 固定低比特整数表示；默认只训练分组尺度 | 最终模型目标；整网传播，尺度改变恢复后的权重 |

这组对照用于辨认优化问题，不是跨模型精度排行榜。量化对象、整数编码是否重新计算、数据监督与梯度传播范围均不同。（LSQ v3 §2；LLM-QAT v1 §2；EfficientQAT v3 §3。）

## 2. 前向模拟什么，部署执行什么

设潜在浮点权重为 $w$，前向使用

$$
\widehat w=\Delta\left[\operatorname{clip}\left(\operatorname{round}(w/\Delta)+z,q_{\min},q_{\max}\right)-z\right].
$$

潜在参数可以积累小于一个量化步长的更新，下一次前向再投影到网格；如果直接把每一步更新写回整数权重，小更新可能不断消失。W §4.1 采用这类浮点潜变量与代理梯度。它没有证明权重梯度、优化器状态与累加均可低比特化。

模拟节点要对应部署真正保存或转换的张量边界。融合的卷积/线性层、偏置和激活函数可能只在最终输出量化一次；残差两支、BN 融合后的权重也需按实际图处理。J 图 1.1、附录图 C.1–C.8 展示了这些区别。任意给每个浮点算子后面插一个量化节点，可能模拟了另一张图。

J §3 中偏置“不量化”指没有像权重一样施加低比特 fake quant；部署时它仍按输入与权重的步长乘积编码为 int32。完整单位换算和输出再量化见 [整数执行展开](../../implementation/quantized-matmul-scaling-execution.md#6.%20%E4%BB%8E%E6%95%B4%E6%95%B0%E7%B4%AF%E5%8A%A0%E5%88%B0%E4%B8%8B%E4%B8%80%E5%B1%82%E7%BD%91%E6%A0%BC)。

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

步长必须保持正值，零点与部署编码范围也要满足约束。[LSQ](../../methods/lsq.md) 将这种步长代理与专门的梯度缩放组合：每层共享尺度的梯度汇聚多个元素，因而用 $1/\sqrt{NQ_P}$ 调整尺度梯度，前向网格保持不变。其推导依赖梯度相关性、饱和比例等启发式假设，激活侧还讨论了前置 BN；不能只把 scale 设为可训练，就认为复现了完整 LSQ，也不能把此缩放当成任意模型的最优规则。（LSQ v3 §2、附录 A/B。）

代理规则应核对到具体计算节点。OmniQuant 官方 commit `feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14` 的 `quantize/quantizer.py:round_ste` 使用 `(round(x)-x).detach()+x`，前向仍取整，反向近似为恒等；权重零点却使用普通 `.round()`，其梯度为零。其学习步长来自 sigmoid 强度与当前权重极值的组合，不是直接套用一个独立可训练步长。不能把“使用 STE”扩写成所有离散节点都使用同一种反向规则。

代理梯度的选择并非无关紧要。[AdaRound](../../methods/adaround.md) 在同一局部重构目标上比较过直通估计器与带显式正则的软松弛，前者精度为 66.63、后者为 68.60（ResNet18，ImageNet），作者把差距归因于 STE 的有偏梯度限制了受限空间中的优化；[BRECQ](../../methods/brecq.md) 则指出激活无法使用舍入式参数化，因为它们随输入变化，只能学习步长。两处都说明：代理规则要与被代理的离散结构匹配，不能只看反向是否可计算。

[SignRound](../../methods/signround.md)则把梯度的sign用于舍入偏移与裁剪参数的有限步更新：离散节点怎样产生代理梯度，与优化器怎样使用该梯度，是两层选择。其v5算法1给出 `loss.backward()` 和SignSGD更新，未展开round/裁剪/零点的反向规则，不能从“梯度取符号”推出硬round可微或作者使用了某个确定STE实现。（§3，式3–6、算法1。）

### 固定整数编码时，不要继续套舍入代理

如果训练直接保留整数 $q,z$，只更新 $s$，前向为 $\widehat w=(q-z)s$，于是 $\partial\widehat w/\partial s=q-z$。这里是精确的连续导数，不再出现重新计算整数编码的 $\operatorname{round}(w/s)-w/s$ 代理项。同组所有权重的梯度共同汇聚到一个尺度，尺度更新会改变恢复权重，但不能重新选择组内整数索引。EfficientQAT v3 §3.3 的 E2E-QP 就是这一参数化；它说明“学习步长”尚不足以确定训练规则，必须同时记录哪些量固定、哪些节点仍在前向中。

### 3.1 离散选码可以使用可验证的代理下降

[PV-Tuning](../../methods/pv-tuning.md) 将连续中心/尺度与离散索引分开更新。离散步骤可以使用恢复权重的真实损失梯度构造代理，再在合法量化集合上搜索；不必把它解释成对整数索引求导。以下由其 v2 §3.2、附录 A.2–A.3 展开，陈述的是满足上界条件时的算法性质。

设当前合法恢复权重为 $y\in\mathcal Q$，$g=\nabla\phi(y)$，$L>0$，且对所考虑的候选满足

$$
\phi(x)\leq q_y(x)
=\phi(y)+g^\top(x-y)+\frac L2\|x-y\|_2^2.
$$

完成平方得

$$
q_y(x)=\phi(y)-\frac{\|g\|_2^2}{2L}
+\frac L2\left\|x-\left(y-\frac gL\right)\right\|_2^2.
$$

因此投影目标是绝对权重 $y-g/L$，不是位移 $-g/L$。若 $\widehat x$ 在合法集合中精确最小化代理，旧点 $y$ 的可行性给出

$$
\phi(\widehat x)\leq q_y(\widehat x)\leq q_y(y)=\phi(y).
$$

这里不要求 $\mathcal Q$ 凸。近似搜索也可使用同一论证，但须确认返回合法点且 $q_y(\widehat x)\leq q_y(y)$；仅知道运行了 beam search 不足以推出该不等式。向量码本的合法集合由整组向量组成，不能把各坐标的独立最近标量拼成一个不存在的码字。

真实损失在代理最优点的值一般**不等于**代理最小值。例如整理者教学例 $\phi(x)=x^2/2$，$y=2$，$L=2$，$\mathcal Q=\{0,1,2\}$，投影目标为1，于是 $\phi(1)=0.5$ 而 $q_2(1)=1$。两者都不高于旧损失2，但不能用等号替代上界。PV-Tuning 附录 A.3 式9正有这处等号问题，以上链条可保留其单调性结论。

固定码本时，合法点有界并不保证一个任意学习率都满足所需上界；同时训练中心后，候选集合和梯度基点又会改变。工程实现若复用旧梯度并同时更新中心与索引，需另核实际目标，不能直接套用精确交替的保证。全局目标下有界与每步损失不增，只能推出损失序列收敛；最小值是否可达、参数是否收敛、是否达到最优是另外的问题。

### 3.2 减少更新坐标为什么可能允许更大步长

令 $U$ 的列为选中坐标的标准基，仅允许 $x=y+Uz$。若限制目标 $z\mapsto\phi(y+Uz)$ 在相关域有有效的二次上界常数 $L_S$，可用 $y-UU^\top g/L_S$ 作为选码目标；未选坐标保持不变。对于常 Hessian 的二次目标，限制曲率为 $U^\top H U$，其谱范数不大于 $\|H\|_2$。

“不大于”不等于严格降低。整理者例子 $\phi(x)=\tfrac12(\sum_i x_i)^2$ 有 $H=\mathbf1\mathbf1^\top$，全空间常数为 $d$，单坐标常数为1；而 $\phi(x)=\tfrac\lambda2\|x\|_2^2$（$\lambda>0$）在任意非空坐标子空间的常数都为 $\lambda$。前一种结构允许更大的单坐标步长，后一种没有这个收益。对非线性模型，不同训练轨迹上的估计也不能当成同一域中嵌套子空间的严格比较。（PV-Tuning v2 §3.3、附录 G；反例与条件为整理者展开。）

PV-Tuning 附录 G 使用十个迭代点的梯度差商、以及十步有限差分 power iteration 估计平滑程度。这些有限探测不认证全域上界，观测最大梯度差商也不能用作理论所需的保守上界。实际用 Adam 幅度选子空间、限制量化后的变化范数，是工程选择；其对质量和稳定性的证据应来自所列实验，不能由小子空间自动推出。

小步长被投影吞掉的机制也有确定条件。例如合法标量为 $\{0,1\}$，从0出发的目标0.1仍会选0。STE潜变量可在多步间累计到越过中点；稀疏PV则试图在部分坐标上直接形成足够大的合法跳变。哪种策略更好取决于目标、表示、步长和搜索规则，PV-Tuning 自身也评估了二者混合。

限制变化量时还要说明基准时点。令 $W^k$ 为旧表示，$\overline W$ 为连续中心/尺度更新后、索引未变的表示，$W^{k+1}$ 为选码后表示，则

$$
W^{k+1}-W^k=(\overline W-W^k)+(W^{k+1}-\overline W).
$$

即使能约束第二项，也不能忽略第一项而声称整步满足同一相对上界。[AQLM固定PV实现](../../implementation/aqlm-pv-training-and-code-updates.md#5.%20%E6%9B%B4%E6%96%B0%E6%AF%94%E4%BE%8B%E4%B8%8E%20trust%20ratio%20%E7%9A%84%E5%AE%9E%E9%99%85%E8%BE%B9%E7%95%8C)恰以 $\overline W$ 测选码变化；其前缀截断还保留首个达到或超过阈值的组，本身也不是硬上界。候选数量、索引变化、解码权重变化与整步变化是不同的限制对象，应沿实际时序核对。（AQLM commit `e79a896ed6656fe4ed06193d42d004e7d0bbdbb2`，`pv_optimizer.py:step`、`beam_search_l2.py:beam_search_optimal_codes`。）

### 3.3 代理函数和坐标变换要分别核对

连续解码可精确求导，不使前面的硬舍入变成可微。[LC-QAT](../../methods/lc-qat.md)用共享仿射网格解码整数向量，再为整数节点选 DGE 代理。其区间函数的导数为 $f'(x)=\tfrac1k|2x/\delta-1|^{1/k-1}$；$k>1$ 时在区间中点发散。训练曲线较稳定是所列实验结果，不能直接写成全域光滑、有界或真实离散损失单调下降。多个区间怎样定位、clamp 外怎样反传、数值奇点怎样处理，必须核实际实现。（LC-QAT v1式9–10、图5；导数边界为整理者核算。）

代理变量的尺度也会影响反向。设整数坐标代理为 $P_z$，改用 $P_x=s(P_z-t)$、$s>0$。若仍要保持相同的硬前向，应使用 $Q(P_x/s+t)$，而不是 $Q(P_x)$；按同一代理反向时还多出 $1/s$ 的 Jacobian。这个整理者推导说明：把潜变量缩到 Xavier 量级，不能自动保持选码，也不能仅凭“归一化”判断训练步长。LC-QAT式11描述缩放初始化，图3含 scaling，但式5与式11的坐标接口未完全给出；本页保留这一实现缺口，不把自洽示例当作作者代码。

### 3.4 重构误差怎样进入 STE 更新的条件界

恒等 STE 将 $r=w+\Delta$ 处的梯度施加给潜在参数 $w$，造成 $e=\nabla\mathcal L(r)-\nabla\mathcal L(w)$。[QUASAR](../../methods/quasar.md#3.%20%E7%90%86%E8%AE%BA%E8%BF%9E%E6%8E%A5%E4%B8%BA%E4%BD%95%E9%9C%80%E8%A6%81%E9%A2%9D%E5%A4%96%E6%9D%A1%E4%BB%B6) v1定理1先用线段积分

$$e=\int_0^1\nabla^2\mathcal L(w+u\Delta)\Delta\,du$$

将两者连接。令 $h_i$ 为非负敏感度，$\widehat S=\sum_i h_i\Delta_i^2$；若沿所考虑候选连线满足 $\|\nabla^2\mathcal L(w+u\Delta)\Delta\|^2\le C\widehat S$，Jensen不等式给出 $\|e\|^2\le C\widehat S$。$L$-smooth与一致的 $h_i\ge h_{\min}>0$ 是充分条件，可取 $C=L^2/h_{\min}$；仅有Adam二阶状态不保证这个正下界或曲率控制。

对无偏条件噪声的恒等STE–SGD、合适固定步长，重构误差进入联合驻点界的平均惩罚；要控制最终部署模型的函数值还需额外全局PL条件及终点误差。主实验AdamW的收益需要实验支持，不能套用SGD保证。局部损失差近似使用 $H$，梯度差平方线性化使用 $H^2$，两者需要上述控制条件；任意把敏感度再平方不构成更准确的Hessian。降低代理上界也不证明每步真实低比特损失单调下降。（QUASAR v1 §5、附录A。）

### 3.5 学习尺度前，先确定它进入哪一侧

“每组一个可学习尺度”不唯一地定义前向。以固定浮点w、基础步长Δ>0和整数零点z为例，下面三种计算图调整的对象不同：

| 前向 | 尺度改变什么 | 非饱和内部的尺度导数/代理 |
| --- | --- | --- |
| $d=a(q-z)$，q/z固定 | 合法码的恢复值 | 真实导数q−z，无需通过舍入 |
| $d=a[\operatorname{round}_{STE}(w/a)+z-z]$ | 舍入与恢复同时变 | 舍入代理为round(w/a)−w/a |
| $d=\Delta[\operatorname{round}_{STE}(w/(\Delta a))+z-z]$ | 进入固定网格的权重坐标 | 舍入代理为−w/a² |

中间行按该简化零点抵消和不含裁剪边界的图展开；实际有clip、尺度参数化、整数码更新或梯度归一化时必须回到相应图。前两行分别对应固定码尺度优化和常见步长学习的区别；[QSLAW所选uniform](../../implementation/qslaw-scale-learning-and-float-export.md#3.%20%E5%89%8D%E5%90%91%E4%B8%8E%E5%8F%8D%E5%90%91%E7%9A%84%E9%87%8F%E4%B8%8D%E5%90%8C)采用第三种：除以Δa，最后只乘Δ，码在每次前向重算。

[FlexRound](../../methods/flexround.md)将这一除法参数化用于固定预训练权重的局部重构，联合学习共享步长与逐元素除法参数：$\hat w=s_1\operatorname{round}(w/(s_1a))$。外侧没有乘回$a$，因此$a$改变选码坐标，$s_1$决定恢复网格。QSLAW所选路径固定基础Δ；二者的目标、共享粒度与可训练变量仍需分别说明。（FlexRound v2 §3.2、命题3.1、附录B。）

第三种硬前向在码不变时真实导数为0，−w/a²来自STE链式法则，不能用有限差分把两者认成一致。整理者例w=1.2、Δ=0.5、z=2，a在1附近的小范围内给d=1，硬函数不变；STE在a=1给−1.2。a=2则给d=0.5，而不是把原恢复值再乘2。它也没有成对的输入补偿，不是自动保持WX的等价通道变换。

学习目标决定这些自由度是否有用；参数数目减少并不免除整网反向或浮点基座成本。若导出只写恢复浮点值到Linear，质量模拟与低位驻留/执行仍是两个验证目标。（QSLAW v1（`https://arxiv.org/abs/2408.03735v1`）式7–8；固定代码quantizer.py、utils.py、replace.py；计算图导数和算例为整理者展开。）

[LRQ](../../methods/lrq.md)进一步用低秩乘积和广播偏置约束对数除数，仍由外侧步长决定恢复网格。低秩约束不传递到逐元素指数后的矩阵或量化权重；完整辅助参数还要计偏置与步长，不能把论文只计两因子的比例当作推理存储减幅。具体目标、预算条件和导出边界见方法页。（LRQ v2 §2.3式2、附录G/J/M。）

## 4. 初始化与训练过程为什么仍然重要

W 图 8、12 的 PTQ/QAT 流程都先考虑部署量化配置、范围初始化和可行的图变换。更好的初始化可以减少训练需要恢复的损失，也可能防止低位宽下训练失败。

W 表 9 的 MobileNetV2 per-tensor W4A8 案例中，基础初始化从 0.10 开始，QAT 后仍为 0.10；加入跨层均衡后初始为 12.99，QAT 后为 70.13；再加偏置校正初始为 46.90，QAT 后为 70.07。它说明这个困难配置中初始化重要，同时说明更高的初始分数并不保证更高的最终分数。表 8 的另一些配置则在训练后缩小了 min–max 与 MSE 初始化的差距。均为作者报告，不能写成 QAT 必须或完全不必做 PTQ 初始化。

[LoftQ](../../methods/loftq.md) v4 §3 把初始化与适配器训练分开：先用量化和残差 SVD 共同构造低比特主体与低秩分支，再冻结主体，按任务损失训练适配器。前一阶段只需要原始权重，后一阶段需要任务数据；不能把“无数据初始化”扩写为“全流程无需数据”。基座不重新量化时，适配器反向不需要穿过舍入，也无需对离线 SVD 求导，但仍有整网激活梯度传播。其低秩因子不是 LSQ 的尺度变量，也不同于 EfficientQAT 默认的整网尺度微调。

J §3.1 延迟启用激活量化，并用指数移动平均跟踪范围，以避免训练初期分布剧烈变化；附录 D 的多项任务使用 500,000 步延迟。该数量依赖长训练协议、batch 与异步工作进程，不能直接移植到另一模型。[校准与范围页](../../theory/calibration-and-range-selection.md)解释统计与数据用途的区别。

## 5. BN 与量化训练图的条件

推理时 BN 统计固定，可以并入前一仿射算子；训练时 batch 统计变化，直接把它当成常数会改变训练过程。J 图 C.7–C.8 使用额外路径估计统计，再模拟融合后的权重量化。W §4.2、表 7 比较静态融合、额外前向估计及保留 BN 等策略，并在相应实验中调学习率。不能只看“多跑一次前向”就判断质量或总成本。

融合的代数和适用条件见 [BN 与跨层均衡](../../theory/diagonal-scaling-equivalent-transform.md#6.%20%E5%9B%BA%E5%AE%9A%20BN%20%E8%9E%8D%E5%90%88%E4%B8%8E%E8%B7%A8%E5%B1%82%E5%9D%87%E8%A1%A1)。普通 LayerNorm/RMSNorm 的统计依赖当前输入，不能把整层照 BN 推理公式并入固定权重。基础的图一致性要求共通，具体融合方式由算子决定。

## 6. 怎样比较 PTQ 与 QAT 的结果

W 表 6（PTQ）与表 10（QAT）包含不同训练、量化和重复次数：PTQ 通常报告 5 次，QAT 3 次；后者还选择学习率并针对不同任务训练不同轮数。BERT 的 PTQ 表含部分 16-bit 激活，而对应 QAT 配置为全 8-bit 激活。BERT 这里是编码器任务指标，不等于自回归大模型的困惑度或生成质量。

公平比较需同时列出浮点基线是否经过同等微调、训练/校准数据、量化节点、各层实际位宽、粒度、优化成本与独立评测。QAT 高于原浮点分数，也可能受额外训练影响；仅凭该结果不能证明量化带来正则化收益。W §4.5 的跨模型表格支持特定配置的可行性，不支持“QAT 总是胜出”或“per-channel 总是更好”。

“数据效率”还要声明分母是已有基座上的额外 QAT、指令微调，还是从头预训练。[LC-QAT](../../methods/lc-qat.md#5.%20%E6%95%B0%E6%8D%AE%E6%95%88%E7%8E%87%E8%AF%81%E6%8D%AE%E6%94%AF%E6%8C%81%E5%88%B0%E5%93%AA%E9%87%8C)的4B/30B比较取得较高QA均分而PPL较差；与BitNet的4B/4T又跨越额外训练与从头训练。预训练基座及各任务的变化不能从预算比例中消失。准确率均值、PPL和总训练成本需要分别判断。（LC-QAT v1表2–3。）

模型质量、低比特存储与实际后端性能分别验证。[误差诊断与验证](../../implementation/quantization-error-diagnosis.md)给出从图一致性到任务评价的检查顺序。本页完成文献解释，没有运行 QAT 或 PTQ 模型复现。

## 来源身份

- LoRA: Low-Rank Adaptation of Large Language Models（`https://arxiv.org/abs/2106.09685v2`），arXiv:2106.09685v2；本页§1连接任务适配、加法增量与固定缩放，原文§2、§4.1。

- LRQ: Optimizing Post-Training Quantization for Large Language Models by Learning Low-Rank Weight-Scaling Matrices（`https://arxiv.org/abs/2407.11534v2`），arXiv:2407.11534v2；本页§3.5连接结构化选码参数，原文式2、附录G/J/M。

- Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs（`https://arxiv.org/abs/2309.05516v5`），arXiv:2309.05516v5；区分代理反向与符号梯度更新，原文§3、算法1。

- FlexRound: Learnable Rounding based on Element-wise Division for Post-Training Quantization（`https://arxiv.org/abs/2306.00317v2`），arXiv:2306.00317v2；本页§3.5连接逐元素除法参数与共享步长，原文机制见§3.2、附录B。

- QSLAW v1（`https://arxiv.org/abs/2408.03735v1`）及所选实现（`https://github.com/xjjxmu/QSLAW/tree/264dfb081c210a8bab6a3fb5707d827ade7b6802`）；§3.5比较尺度位置/整数码更新与STE，导数和算例为整理者展开。

| 来源 | 版本或快照 | 说明 |
| --- | --- | --- |
| LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models（`https://arxiv.org/abs/2310.08659v4`） | `arXiv:2310.08659v4` | 权重分解初始化与固定基座的适配器训练 |
| SpinQuant: LLM quantization with learned rotations（`https://arxiv.org/abs/2405.16406v4`） | `arXiv:2405.16406v4` | — |
| OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models（`https://arxiv.org/abs/2308.13137v3`） | `arXiv:2308.13137v3` | — |
| OpenGVLab/OmniQuant（`https://github.com/OpenGVLab/OmniQuant/blob/feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14/quantize/quantizer.py`） | `feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14` | 本地快照已移除 |
| QLoRA: Efficient Finetuning of Quantized LLMs（`https://arxiv.org/abs/2305.14314v1`） | `arXiv:2305.14314v1` | — |
| Q-VLM: Post-training Quantization for Large Vision-Language Models（`https://arxiv.org/abs/2410.08119v3`） | `arXiv:2410.08119v3` | — |
| A White Paper on Neural Network Quantization（`https://arxiv.org/abs/2106.08295v1`） | `arXiv:2106.08295v1` | — |
| Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference（`https://arxiv.org/abs/1712.05877v1`） | `arXiv:1712.05877v1` | — |

- Learned Step Size Quantization（`https://arxiv.org/abs/1902.08153v3`），arXiv:1902.08153v3；步长代理与梯度缩放。
- LLM-QAT: Data-Free Quantization Aware Training for Large Language Models（`https://arxiv.org/abs/2305.17888v1`），arXiv:2305.17888v1；生成数据、教师输出与联合量化。
- EfficientQAT: Efficient Quantization-Aware Training for Large Language Models（`https://arxiv.org/abs/2407.11062v3`），arXiv:2407.11062v3；分阶段变量、目标与尺度参数化。
- PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression（`https://arxiv.org/abs/2405.14852v2`），arXiv:2405.14852v2；代理选码、子空间与精确理论/工程更新的条件。
- LC-QAT: Data-Efficient 2-Bit QAT for LLMs via Linear-Constrained Vector Quantization（`https://arxiv.org/abs/2606.10531v1`），arXiv:2606.10531v1；代理梯度、潜变量坐标与数据预算边界。
- QUASAR: Lowering the Loss Floor of Quantization-Aware Training with Loss-Aware Reconstruction（`https://arxiv.org/abs/2608.13966v1`），arXiv:2608.13966v1；恒等STE梯度差、重构误差与SGD/PL条件界。
- Vahe1994/AQLM（`https://github.com/Vahe1994/AQLM/tree/e79a896ed6656fe4ed06193d42d004e7d0bbdbb2`），固定commit `e79a896ed6656fe4ed06193d42d004e7d0bbdbb2`；PV连续更新与选码变化的不同基准。
