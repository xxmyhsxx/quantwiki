---
title: 校准数据与量化范围选择
type: concept
tags:
  - calibration
  - clipping
  - sensitivity
sources:
  - raw/papers/2026-09-21/norm-tweaking/paper.pdf
  - raw/papers/2026-09-21/norm-tweaking/source/norm_tweak.tex
  - raw/papers/2026-09-21/quasar/paper.pdf
  - raw/papers/2026-09-21/quasar/source/methodology.tex
  - raw/repositories/2026-09-21/aqlm/source/finetune.py
  - raw/repositories/2026-09-21/aqlm/source/src/datautils.py
  - raw/papers/2026-09-22/pv-tuning/paper.pdf
  - raw/papers/2026-09-22/pv-tuning/source/supplementary.tex
  - raw/papers/2026-10-05/onptq/paper.pdf
  - raw/papers/2026-10-05/onptq/source/iclr2027_conference.tex
  - raw/papers/2026-10-03/scale-sensitivity/paper.pdf
  - raw/papers/2026-10-03/scale-sensitivity/source/main_arxiv.tex
  - raw/papers/2026-09-22/qera/paper.pdf
  - raw/papers/2026-09-21/llm-qat/paper.pdf
  - raw/papers/2026-09-21/luq/paper.pdf
  - raw/papers/2026-09-21/qig/paper.pdf
  - raw/papers/2026-09-21/splitq/paper.pdf
  - raw/papers/2026-09-21/masquant/paper.pdf
  - raw/papers/2026-09-21/quarot/paper.pdf
  - raw/papers/2026-09-21/spinquant/paper.pdf
  - raw/papers/2026-09-21/flatquant/paper.pdf
  - raw/papers/2026-09-21/omniquant/paper.pdf
  - https://github.com/OpenGVLab/OmniQuant/blob/feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14/generate_act_scale_shift.py
  - raw/papers/2026-09-21/mquant/paper.pdf
  - raw/papers/2026-09-21/vlmq/paper.pdf
  - raw/papers/2026-09-21/mbq/paper.pdf
  - raw/papers/2026-09-21/quantization-white-paper/paper.pdf
  - raw/papers/2026-09-21/integer-only-quantization/paper.pdf
  - raw/papers/2026-10-07/centriq/paper.pdf
  - raw/papers/2026-10-07/centriq/source/iclr2027_conference.tex
  - raw/papers/2026-10-07/pivot-q/paper.pdf
  - raw/repositories/2026-10-07/pivot-q/source/pivot_q/selector.py
  - raw/repositories/2026-10-07/pivot-q/source/pivot_q/train.py
  - raw/papers/2026-09-21/smoothquant-plus/paper.pdf
  - raw/papers/2026-09-21/smoothquant-plus/source/rsc.tex
  - raw/papers/2026-09-21/veq/paper.pdf
  - raw/papers/2026-09-21/veq/source/arxiv-main.tex
updated: 2026-10-12
---

# 校准数据与量化范围选择

校准是在有限样本或已有统计上，确定部署所需的量化参数及相关调整。范围选择回答“有限网格用来覆盖哪些数值”；校准还可能为通道缩放、层输出重构或误差补偿收集信息。基础问题对卷积网络和大模型共通，所需样本、统计维度与可接受误差则取决于模型和任务。

本页主要依据 Nagel 等量化白皮书 v1（下称 W）§2.2、§3.1、§3.5，以及 Jacob 等整数推理论文 arXiv v1（下称 J）§3。两篇论文的历史配置是案例，不是当前模型的通用默认值。

## 1. 先确定量化对象、网格与目标

仿射网格由整数端点 $q_{\min},q_{\max}$、正步长 $\Delta$ 和整数零点 $z$ 确定，反量化为 $\widehat x=\Delta(q-z)$。定义和粒度见 [均匀量化与分组](../fundamentals/quantization/uniform-quantization-and-groups.md)。给定希望覆盖的实数范围 $[a,b]$，先将其扩展到包含零，常见初始化为

$$
\Delta=\frac{b-a}{q_{\max}-q_{\min}},\qquad
z=\operatorname{clip}\!\left(\operatorname{round}\!\left(q_{\min}-a/\Delta\right),q_{\min},q_{\max}\right).
$$

实际可表示端点是 $r_{\min}=\Delta(q_{\min}-z)$、$r_{\max}=\Delta(q_{\max}-z)$。整数化零点会移动端点，因此“min–max 初始化”不意味着任意实现都逐点覆盖原始极值。全零等退化范围需要单独处理步长，不能除以零。（W 式 (4)–(7)；J §3.1 的零点对齐。）

**教学例子：**3-bit unsigned 网格，希望覆盖 $[-1,2]$，得到 $\Delta=3/7,z=2$，实际范围 $[-6/7,15/7]$。零可精确表示，但原下界 -1 已在实际范围之外。实现必须明确是接受这个移动，还是重新扩展范围，不能同时声称端点和零都无条件精确。

权重范围可直接由固定权重计算；激活范围需要输入或可用的分布统计。per-tensor、per-channel、per-token、group-wise 决定哪些元素共同影响一个网格。先改变粒度再讨论范围策略，否则容易把不同问题混为一谈。

## 2. 为什么范围越大不一定越好

设 $e=\widehat x-x$。未饱和区的舍入误差不超过半个步长；尾部的饱和误差则随超出端点的距离增长。用经验分布表示，总平方误差可按实际网格范围拆开：

$$
\mathbb E[e^2]=\mathbb E[e^2\mathbf1_{r_{\min}\le x\le r_{\max}}]
+\mathbb E[e^2\mathbf1_{x<r_{\min}\ \mathrm{or}\ x>r_{\max}}].
$$

缩小范围让主体部分网格更密，也会牺牲尾部。其效果取决于频率与数值误差对后续计算的影响。“保留全部离群值”和“裁掉离群值”都不是无条件最优方案。（W §3.1，图 3。）

| 目标或规则 | 如何使用数据 | 能保证什么、遗漏什么 |
|---|---|---|
| Min–max | 使用观测极值初始化范围 | 简单，但对单个极值和样本覆盖敏感；还需处理零点对齐 |
| 张量 MSE | 搜索使 $\sum_i(x_i-\widehat x_i)^2$ 最小的候选范围 | 优化这个张量在这批数据上的误差，未衡量下游敏感性 |
| 层输出重构 | 比较原层与量化层在输入上的输出 | 利用输入统计，仍是局部代理目标 |
| 任务损失或输出分布匹配 | 比较网络输出、标签损失或浮点参考预测 | 更接近所选任务，计算更贵，也更易对用于选择的数据过拟合 |

W §3.1 的 MSE+Xent 案例在最后层使用输出分布相关目标，不等于“所有层都应改成交叉熵”。[层输出重构](layer-reconstruction-second-order-compensation.md)说明 $\|(W-\widehat W)X\|_F^2$ 如何包含输入的二阶统计；[AWQ](../methods/awq.md)采用的分组输出裁剪目标也不能叫作纯权重 MSE。

范围还可以用梯度学习。[OmniQuant](../methods/omniquant.md) v3 §3.2 将当前权重极值 $a,b$ 映射为 $l=\beta a,u=\gamma b$，学习 $\beta,\gamma$，用 Transformer block 输出误差决定裁剪。相对强度使端点跟随等价变换后不断变化的权重范围；它不是直接最小化权重张量 MSE，也不表示按比例删掉权重。若 $\gamma=0.5$，只说明正端点变为当前最大值的一半，不能据此推断有一半元素被裁剪。

### 2.1 选码范围和恢复参数可以分开优化

[QUASAR](../methods/quasar.md)将一组潜在权重 $w_i$ 的裁剪范围用于生成固定码 $q_i$，再独立拟合恢复值 $r_i=sq_i+o$。这里 $o$ 是实数偏移，与第1节整数零点不同；若实际格式要求 $r_i=s(q_i-z)$，还必须满足 $o=-sz$ 和零点的整数约束。

给定非负敏感度 $h_i$，令 $H_0=\sum_i h_i>0$、$\bar q_h=\sum_i h_iq_i/H_0$、$\bar w_h=\sum_i h_iw_i/H_0$。在加权码方差非零时，最小化 $\sum_i h_i(sq_i+o-w_i)^2$ 的闭式解为

$$s^\star=\frac{\sum_i h_i(q_i-\bar q_h)(w_i-\bar w_h)}{\sum_i h_i(q_i-\bar q_h)^2},\qquad o^\star=\bar w_h-s^\star\bar q_h.$$

它来自先对偏移求驻点，再解尺度的加权线性回归；只对固定码、无额外格式限制的仿射恢复最优。总权重为零或所有有效码相同，分式不能直接使用；后者仅确定一个恢复常量，不能辨识唯一尺度与偏移。QUASAR v1未完整给出这些数值分支，不能自行补成作者实现。

教学例：$q=(0,0,1)$、$w=(0,1,4)$、$h=(1,3,1)$，闭式拟合得 $o=0.75,s=3.25$，前两个数仍共享0.75，第三个恢复4。拟合无法把共享同一码的前两个数拆开；需要改变选码候选。多个候选范围先选码、再拟合、比较拟合后的误差，不要求使用新尺度重新选码。若存储参数只允许低精度值，需在可表示参数上重新评价误差，浮点最优值不能直接替代已编码值。（QUASAR v1式3–4、算法1；例子、退化条件和格式约束为整理者展开。）

## 3. 作者结果说明了哪些差异

以下是 W 表 1–2 的作者报告，ImageNet top-1，以百分点表示差值；PTQ 实验报告为 5 次运行均值。权重量化表保持激活 FP32，激活量化表保持权重 FP32，不能拼成一个联合 W4A4 结果。

| 模型与量化对象 | FP32 | per-tensor min–max | per-tensor MSE | 更换另一条件后的结果 |
|---|---:|---:|---:|---|
| ResNet18，W4 | 69.68 | 0.12 | 18.82 | per-channel MSE：54.67 |
| MobileNetV2，W4 | 71.72 | 0.59 | 13.77 | per-channel MSE：27.17 |
| ResNet18，A4 | 69.68 | 18.82 | 31.40 | MSE+Xent：59.07 |
| MobileNetV2，A4 | 71.72 | 0.53 | 13.57 | MSE+Xent：30.94 |

这些结果支持范围目标和粒度会显著影响低位宽误差；它们也显示改善初始化后仍可能距浮点模型很远。不能据此推导某种校准目标在所有模型、位宽与数据上最优。

## 4. 没有样本时，统计假设能替代什么

W §3.1.3 用 BN 参数估计激活范围。对推理时固定统计的 BN，若输入确实具有记录的均值与方差，输出均值为 $\beta$，标准差为

$$
\sigma_{\mathrm{out}}=|\gamma|\frac{\sigma_{\mathrm{in}}}{\sqrt{\sigma_{\mathrm{in}}^2+\epsilon}}.
$$

只有忽略 $\epsilon$ 且统计匹配时才近似 $|\gamma|$；gamma 可以为负，不能一般地当作标准差。还要额外假设分布形状，才能用“均值 ± 若干标准差”估计覆盖率；经过 ReLU 后还要考虑截断与零点质量。W 的简写不应取代这些条件。

这种做法依赖模型中可用的 BN 统计。LayerNorm/RMSNorm 的输入相关归一化没有相同的固定总体统计，不能照抄 BN 范围公式；[等价变换页](diagonal-scaling-equivalent-transform.md)进一步区分固定 BN 融合与归一化输出侧缩放。缺少真实数据时，应降低对部署分布覆盖的确信程度，而不是宣称“无需数据所以没有分布风险”。

### 免校准仍可使用当前 token 的统计

[CentriQ](../methods/centriq.md#3-免校准不等于没有在线统计) v1 §4.2 提供一个不同于固定 BN 统计的例子：先移除当前 token 的均值并旋转，再用其绝对值的0.9995分位数除以半正态分位数，估计散布；固定高斯4位裁剪常数与位宽共同确定动态步长。它没有先用 prompt 数据拟合逐模型参数，但每个 token 仍需在线统计。

这里需要分开三件事：中心化消除均值项是代数关系；半正态换算与高斯裁剪常数是分布模型；下游生成质量是实验结果。固定常数不保证所有真实分布或位宽最优，确定性 Hadamard 也不保证激活成为独立高斯。作者用独立开发集选择设计，因而“免校准”不能改写为“整个方法开发不接触数据”。主表对照、组件消融和低位宽模拟边界见方法页。（附录 A、B.3，表2–3。）

### 生成数据解决访问限制，仍需验证覆盖

[LLM-QAT](../methods/llm-qat.md) 将“无需原始预训练数据”与“无需样本”区分开：用 LLaMA-7B 生成文本，再让量化学生匹配浮点教师输出。前缀采样改变输入覆盖，教师软标签改变监督内容，两者不能合并成一个“无数据”属性。其 v1 表 3 中，W4A6 的 7B 用 WikiText-2 微调得到较低的同域 PPL，却不如混合生成数据的常识任务均分；这说明分布适配与单一任务拟合应分别评估。

该方法包含学生权重训练，不是把生成语料用于范围统计就完成了 LLM-QAT。生成器、教师、随机策略、有效 token 和独立测试都应记录；生成器未掌握的分布也不因合成而自动覆盖。（LLM-QAT v1 §2.1–2.2、§3.1、表 3。）

## 5. 校准、训练、验证与测试怎样分工

校准用于估计网格、统计或局部调整；训练用于拟合模型参数；验证用于选择配置；独立测试用于评价选择后的结果。名称并不决定用途：反复查看测试分数来选裁剪比例，已经使测试集参与了选择。

具体记录应足以说明输入如何产生：数据版本、样本数、抽样方式、预处理、随机性、模型与权重版本，以及输入形状。图像任务记录分辨率和增强；文本任务记录分词器、序列长度、拼接/截断、padding/mask 及实际参与统计的 token。不要直接将训练增强得到的范围当作部署范围。若使用标签或浮点教师输出来选择参数，也说明其来源和用途。

静态激活量化离线固定网格，容易受分布偏移影响；动态量化按当前输入估计网格，增加运行时统计开销，也不能保证未见输入的任务质量。[SmoothQuant](../methods/smoothquant.md)的 O1–O3 提供这两个维度的具体案例。

离线学到的变换与动态激活网格可以并存。[OmniQuant](../methods/omniquant.md) 离线学习通道缩放，默认激活量化仍按当前 token 估计范围。其官方快照 `feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14` 的 `generate_act_scale_shift.py` 还区分两种初始化统计：`get_act_scales` 累积逐通道最大绝对值；`get_act_shifts` 对每段输入的逐通道区间中点 $c_t=(\max X_t+\min X_t)/2$ 作 $m_t=0.99m_{t-1}+0.01c_t$ 的 EMA，初值为首段中点。后者不是样本均值，也不是所有样本合起来的全局 min/max 中点；这些统计只用于初始化，不能代替训练后参数。

OmniQuant v3 表 A10–A11 比较几个文本校准集与样本数量，说明所测 LLaMA-7B 配置的变化有限，但不构成部署分布漂移后的恢复证据。样本增加也未始终改善两个评测集，因此“固定 128 条就足够”只能是实验配置，不能升级为普遍规则。

J §3.1 在训练时对激活范围使用指数移动平均，并在部分配置中延迟启用激活模拟量化；这是处理训练初期分布变化的策略，不能据此给所有 PTQ 任务规定相同样本数或预热步数。训练范式见 [PTQ、QAT 与代理梯度](../fundamentals/quantization/post-training-and-quantization-aware-training.md)。

完成范围选择后，应在未用于选择的数据上查看层误差与任务质量，并核对实际执行的网格是否一致。[量化误差诊断与验证](../implementation/quantization-error-diagnosis.md)说明如何分离图变换、量化和后端实现的问题。本页没有模型实验，不提供校准规模充分性的实证保证。

配置选择还会让同一批数据承担候选排序职责。[Quantization Price](../methods/quantization-price-prediction.md) 用参考模型统计给局部候选评分，节省组装完整模型的搜索成本，但实际梯度来源、MSE 归约以及校准/测试划分都决定代理含义；选择成本与最终量化、评测成本应分别报告。

### 5.1 同一语料不等于同一采样分布

“使用同一个数据集”没有确定每个 token 被训练或校准的相对权重。[PV-Tuning](../methods/pv-tuning.md) v2 附录 H 区分均匀抽文档后抽固定长度片段，与先拼接文档再分成固定长度训练块。前者让每篇合格文档获得近似相同的片段名额，后者让长文档贡献更多块；它改变了损失平均所对应的分布。

整理者教学例：两篇文档分别由1个和9个等长块组成。先以各 $1/2$ 的概率选文档、再在文档内均匀选块，短文档唯一块的概率为 $1/2$，长文档每块为 $1/18$；从十个块中均匀抽取则每块为 $1/10$。若各块平均损失为 $\ell_{ji}$，两种目标分别为

$$
L_{\mathrm{doc}}=\frac12\ell_{11}+\frac1{18}\sum_{i=1}^{9}\ell_{2i},
\qquad
L_{\mathrm{block}}=\frac1{10}\left(\ell_{11}+\sum_{i=1}^{9}\ell_{2i}\right).
$$

这个例子假定等长、不重叠块，排除短文档过滤、边界拼接与尾部丢弃等影响。真实预处理还须说明这些规则、EOS、mask及损失归约；不能只靠“拼接”一词推断全部细节。哪种抽样无偏取决于目标究竟是文档均匀、token均匀还是特定部署分布，不能把 token 均匀称为所有任务的唯一正确目标。

PV-Tuning 报告这项预处理改变在部分模型上改善至多约0.15 PPL，并在比较内匹配预处理；其 §4.1 的表示实验仍保留旧方式用于对齐基线。归因算法效果时，应核每项实验的具体取样方式，不能把数据版本一致当作协议完全一致。

沿[已归档AQLM PV入口](../implementation/aqlm-pv-training-and-code-updates.md#7-训练数据与-kl-的实际接口)还能看到“拼接分块”的具体范围：`prepare_training_dataset` 调用 batched map，`group_texts` 在每次收到的批次内拼接，丢弃该批次不足一个完整块的尾部。若两个map批次各有6个token、块长4，则共保留8个；全局12个token一次分块则可保留12个。这个整理者算例说明批次边界会影响有效样本量。函数本身不再插入EOS；外部已分词数据又会绕过这套预处理，不能仅凭数据集名称断言token覆盖和边界完全相同。（AQLM commit `e79a896ed6656fe4ed06193d42d004e7d0bbdbb2`，`finetune.py`、`datautils.py`。）

### 5.2 生成数据仍有种子先验

[LLM-QAT](../methods/llm-qat.md)用随机起始token后接模型生成，[Norm Tweaking](../methods/norm-tweaking.md) v2进一步限制种子语言。以种子 $s$ 的分布为 $\pi(s)$、种子后续文本 $u$ 的生成分布为 $p_\theta(u\mid s)$，边缘分布为

$$p_{\pi,\theta}(u)=\sum_s\pi(s)p_\theta(u\mid s).$$

这是整理者混合分布记法：限制种子候选、改变其权重、调整条件生成器或解码规则，都会改变校准分布。词表项比例、种子概率、生成文本比例和预训练token比例是不同对象，不能互相替代。

教学例：两类种子等概率，生成目标语言的条件概率分别为0.9和0.2，则总体概率为0.55；只保留第一类种子时变成0.9，而非必然为1。均匀选语言也不等于均匀选其词表项，更不保证按训练语料比例生成。Norm Tweaking表8受限种子改善所列生成基线，但未全面超过各评测域的真实校准数据；不能由此认定生成语料不存在偏差。原模型未覆盖的知识也不会因种子重加权自动得到补足。

### 5.3 使用评测输入校准时，要保留选择边界

没有使用答案标签，也可能已针对评测输入选择量化配置。[SmoothQuant+](../methods/smoothquant-plus.md) v1用HumanEval的164个题目描述统计通道最大值并搜索α，再用同一基准比较校准集和网格间隔。这与在未见输入上作独立测试不同；它支持已知prompt集合的适配，未单独验证新任务或新分布。

需分别记录输入是否参与统计、标签/教师是否参与优化，以及任务分数是否参与最终配置选择。论文称最小化“全模型量化损失”时，还要明确比较张量、归约和量化前缀；仅有单层平方输出差不足以补出最终logit MSE。表3显示校准集有影响，但子集规模/预处理不充分时，不能将差异全部归因于语料名称。（v1 §2.2、§3.1、§3.4、表3–4。）

## 6. 图文校准增加的是数据与目标条件

[视觉语言模型](../fundamentals/model/vision-language-model-tokens-and-quantization.md) 沿用前述网格和误差基础，但同一校准序列含有视觉、提示和回答等不同位置。必须记录图像处理后的 token 数、模板、截断、padding、回答是否进入输入，以及重构 mask 与交叉熵标签的区别；不能把所有非视觉位置一概视作同一监督对象。

MBQ v2 使用 128 个图文样本，VLMQ v2 使用 512 个；这两个数字分别属于各自算法和协议，不是通用“足够样本数”。[MBQ](../methods/mbq.md) 表 4 显示图文校准、目标加权与范数选择不能互相替代；[VLMQ](../methods/vlmq.md) 的梯度又来自局部重构 MSE，不能与回答交叉熵梯度混称同一敏感性。

求和、逐 token 平均、先按模态平均再加权会得到不同目标。大量视觉 token 是否主导统计，应结合实际归约规则检查；不同目标产生的敏感性排序也不宜直接当成跨任务稳定属性。这些条件应在比较方法前明确，后续任务评价仍需使用未参与选择的数据。

分模态还可以改变网格，而不改变误差目标：[MQuant](../methods/mquant.md) 为视觉和文本分别离线估计静态激活范围，避免较宽的视觉范围使小幅文本值被粗量化。参数随层变化，推理时按模态选择；这与 MBQ 的目标加权、VLMQ 的 token 权重是不同维度。其表 10 对比说明，直接共用单一静态范围可能很快却严重损害质量。表 8 的样本数稳定性只适用于所测数据和配置，不构成分布外保证。（MQuant v2，§3.1、表 8/10）

### 6.1 MoE 校准须记录专家覆盖与权重位置

[VEQ](../methods/veq.md) v1将模态统计分两处使用：专家分数 $S_i=\gamma N_i^{text}+\beta N_i^{vis}$ 加在跨专家累计重构损失上；亲和权重 $c_j=p_j\alpha_j$ 则进入某专家的输入二阶矩 $XCX^\top$。β归一化视觉数量，α对文本取γ，对视觉取1，两处不能互换。动机中的COCO128样本与超参数验证的MMMU64样本也不等于主量化校准集。

专家级正标量只有在搜索变量/预算跨专家耦合时才改变取舍；独立优化SiLi的最优点与Li相同。token级权重一般会改变二阶矩的方向，应用sqrt(c)缩放输入才对应c加权。需确认p是非负亲和性而非可负原始logit、当前有效路由token、mask/计数以及damping；零token专家没有获得校准质量约束。完整MoE加权输出的误差还含router系数和跨专家项，该二阶矩属于所选重构代理。

因此总样本数之外，应登记每层/专家的有效token及模态覆盖、敏感性目标与归约、共享搜索粒度、p的生成/归一化和低权重方向处理。一次校准的高频与零频均不能替代部署分布下的覆盖验证。（v1 §3.1–3.3、§4.3；正标量优化与曲率条件为整理者推导。）

## 7. 变换与范围选择的先后关系

[FlatQuant](../methods/flatquant.md) v4 表 18 在同一 LLaMA-3-8B 框架中比较裁剪位置：变换前学习裁剪得到 WikiText2 PPL 7.37，变换后为 6.98。通道混合改变了被裁剪对象的分布，所以范围选择应针对最终待量化的表示；逆变换保持未量化计算，不会恢复已裁掉的信息。

其表 17 用 WikiText2、C4、Pile 校准时质量变化较小，只支持这些文本来源下的观察，不是跨模态或所有部署分布的保证。表 15 的同模型跨量化配置复用亦应与跨模型、跨领域迁移区分。这些边界与 [学习变换路线的目标差别](../research/omniquant-affinequant-flatquant-comparison.md) 一起决定怎样解释校准结果。

[QuaRot](../methods/quarot.md) 的固定旋转本身不训练，默认 GPTQ 阶段仍需旋转后的校准输入；[SpinQuant](../methods/spinquant.md) 则先对最终 next-token CE 学旋转，再固定旋转做 GPTQ。后者表 3 中 W16A4 学习再 GPTQ 优于在 RTN W4A4 噪声下学习后再 GPTQ，说明校准时与最终采用的量化器是否匹配会影响结果。不能把“量化参数在变”和“所有量化误差已被联合优化”视为同一件事。（QuaRot v2 §4；SpinQuant v4 §4.2。）

SpinQuant 表 11 在所测模型中，128 与 800 样本的 PPL 都约 6.2，100 到 200 步也基本饱和。它给出了按收益与成本选校准预算的证据，不是所有任务统一用某个样本数的规则；尤其不能从文本内部比较推断图文分布已经覆盖。

## 8. 校准统计稳定，不等于任务目标一致

[MASQuant](../methods/masquant.md) v1 表 4 中，模态等权的 PPL/准确率为 17.2/56.9，降低视觉权重后为 33.7/58.1：困惑度与任务平均分并未选择同一配置。模态重要性应跟随明确目标，不能由一次最优系数推出普遍的视觉/文字重要性排序。其公开损失分支还采用按总 mask 数归一化及额外视觉系数，方法页分别记录论文与代码条件。

[SplitQ](../methods/splitq.md) v1 表 3 报告不同校准样本数下较高的通道集合重合率，验证的是选列稳定性；不等价于准确率稳定、低秩补偿稳定或跨分布泛化。它的文字百分位评分和视觉最大值都是选通道的统计代理，不是损失梯度。

白化补偿还使用激活二阶矩，见 [激活加权低秩近似](../fundamentals/mathematics/invertible-transforms-and-kronecker-products.md)。校准未覆盖的方向既可能影响数值求逆，也可能在部署数据上突然重要。需要记录变换学习与补偿统计分别使用的数据及模型状态；不能只记录“128 个样本”而省略模态组成、有效 token 数与测试独立性。MASQuant 的公开音频数据路径存在测试来源待核对项，具体证据与未确定部分在其方法页保留。

[QERA](../methods/qera.md) v2 §3 把完整二阶矩与逐通道 RMS 分别用于 exact 和 approx 补偿。统计应保留输入均值的贡献，不能把未中心化二阶矩替换成协方差，也不能把 RMS 替换成标准差。图 3 的样本量增加后 PPL 改善是特定实验观察，不是任务质量单调改善定理。附录 A.6 中，SST2 校准与 WikiText2 校准造成不同训练曲线；作者将原因假设为 padding 比例及预处理差异，尚未隔离验证该因果解释。实际复用需要核对哪些位置进入统计，以及下游输入是否仍对应所估计的分布。

[SmoothQuant+](../methods/smoothquant-plus.md) v1表4还显示，α搜索从0.05细化到0.01后，13B代理loss从0.00399降到0.00393，HumanEval pass@1却从37.80%降到33.54%；34B亦出现同向差异。代理更低与任务更好是两个目标，较细候选网格只能在目标/候选/实现一致时扩大搜索空间，不能自动保证任务质量。这是原文所报消融，不是本轮运行结果。

## 9. 归因目标、数据混合与位宽选择

[QIG](../methods/qig.md) v1 §3、附录 B 将归因目标设为浮点/量化输出差异，沿参考激活积分；这与回答 CE 的单点梯度不是同一个敏感性。附录 C 使用 InfoVQA 校准并评价 OCR 任务，说明特定数据—任务组合的效果，不证明部署后分布漂移已解决。需要记录基线、有效 token、目标归约与权重后处理，不能只记录校准样本数。

[LUQ](../methods/luq.md) v3 §4.1、表 3 的 WikiText-2/TextVQA 混合则改变校准 token 组成；1:1 的 token 比例不等于两种模态在重构损失中各占一半。其 LLaVA 超低比特配置从纯文本到混合校准时 TextVQA 51.8 → 53.4，而 4-bit AWQ 没有相同改善，故比例选择仍受量化器与位宽影响。

校准数据之外，位宽选择也会使用任务信息。LUQ §4.2 用 MME Perception 门槛选择 LLaVA 的低比特层数，最终又报告 MME；若不另分选择集和测试集，就不能把该项当作独立验证。TextVQA 同时出现于校准来源和评价任务时，应核对具体 split 和样本 ID：同名任务不证明泄漏，但也不能仅凭文字描述声称已排除。[混合精度分配](mixed-precision-allocation.md) 将这些选择阶段写入预算与质量流程。

> 来源维护（2026-09-21）：上列固定 commit 的外部代码引用对应已移除的本地快照；保留原版本身份，本轮未重新审查相关代码结论。

## 10. 极低位宽为什么更依赖尺度选择

*Scale Sensitivity* v1 从固定网格的最近舍入（RTN）分析尺度，再用 GPTQ 实验检查趋势能否迁移。它补充的是第 2 节“舍入与裁剪如何权衡”的定量解释，不是所有 GPTQ 配置的最优尺度定理。这里的尺度 $s$ 是量化步长，和[重构损失尺度](layer-reconstruction-second-order-compensation.md#11-重构损失的尺度怎样影响优化)所说的跨阶段损失量级是不同概念。

### 先核对网格，才能使用最优尺度

令位宽 $B\ge2$，$M_B=2^{B-1}-1$，论文采用

$$
Q_{s,B}(w)=s\,\operatorname{clip}\!\left(
\operatorname{round}(w/s),-M_B,M_B\right),\qquad s>0.
$$

它有 $2^B-1$ 个数值级别，包含零；2 bit 对应 $\{-s,0,s\}$，并非四级网格。对称半点舍入远离零。非对称零点、偶数级别、不同端点或舍入规则不能直接照搬下述常数。（v1 §3 与附录 B.1.2。）

固定一行权重 $w\in\mathbb R^d$，校准输入 $X\in\mathbb R^{d\times n}$，记 $G=XX^\top$，则归一化的 RTN 输出重构损失是

$$
\mathcal E(s)=
\frac{(w-Q_{s,B}(w))^\top G(w-Q_{s,B}(w))}
{\operatorname{tr}G},\qquad \operatorname{tr}G>0.
$$

按 trace 归一化去掉输入整体幅度；$G=I$ 时退化为权重逐坐标 MSE。这仍保留输入的方向权重，不是在一般情况下把 $G$ 丢掉。GPTQ 会沿量化顺序改变尚未量化的权重，所以补偿后的误差曲线一般不同；相关补偿见[层输出重构](layer-reconstruction-second-order-compensation.md#3-固定一个权重后为什么其他权重可以补偿)。

### 高斯极限的结论及前提

作者在固定非零 $X$、$w_i$ 独立标准高斯及大有效秩条件下，证明 $\mathcal E(s)$ 对全部 $s>0$ 一致趋近

$$
D_B(s)=\mathbb E_{Z\sim\mathcal N(0,1)}
\left[(Z-Q_{s,B}(Z))^2\right].
$$

条件为 $r_{\mathrm{eff}}(G)=\operatorname{tr}G/\|G\|_{\mathrm{op}}
\ge C\log^8d$，固定 $B$、层宽 $d$ 趋于无穷；当前权重须符合固定输入下的高斯假设，不能默认训练后的相关性仍满足它。大有效秩意味着输入能量没有集中在少数方向，**不要求 $G$ 本身就是单位阵**。随机 MLP 的进一步结果要求比例增长的层宽、各向同性高斯输入、相应增长的校准样本，以及奇对称 Lipschitz 激活；ReLU 不在该结果中。（v1 §4.1 定理 1，附录 A.4。）

极限目标有唯一非退化最小点 $s_B^*$，级别增加时最优步长下降。结论属于高斯极限；有限层的损失曲线、GPTQ 的补偿目标以及最终语言任务不因此获得唯一最优点或相同保证。（v1 §4.2，命题 1。）

### 为什么比较相对曲率

不同位宽或权重幅度的最优步长不同，固定一个绝对步长扰动不公平。定义

$$
\kappa_B=(s_B^*)^2D_B''(s_B^*),\qquad
D_B(s_B^*(1+\epsilon))-D_B(s_B^*)
=\frac{\kappa_B}{2}\epsilon^2+o(\epsilon^2).
$$

$\kappa_B$ 衡量同样百分比的步长误差造成多大的归一化损失。若权重尺度整体变为 $\sigma$，则最优步长变为 $\sigma s_B^*$；在损失再除以 $\sigma^2$ 的共同口径下，相对曲率不变。原文图 4 对 $B=2,\ldots,8$ 的数值求值显示曲率近似指数下降；它不是任意训练层都服从精确指数规律的证明。（v1 定义 1、附录 A.5.2。）

有限 GPTQ 曲线可有编码切换，原文 §5.3 使用

$$
C_{\mathrm{sym}}(\delta)=
\frac{L(\widehat s(1+\delta))+L(\widehat s(1-\delta))
-2L(\widehat s)}{\delta^2}.
$$

光滑极限下该量趋于 $\widehat s^2L''(\widehat s)$；实际应称有限窗口敏感性。实验对每个矩阵选 32 个预定行，使用三次校准抽样。另一方面，五种尺度规则的最大/最小 NLL 差还包含规则本身选出的步长差，不能只凭这个差值声称局部曲率改变。（v1 §5.1 Remark 3、§5.3。）

### 怎样选择尺度，以及旋转改变了什么

附录 B.1.3 比较以下规则。设 $e(s)=Q_{s,B}(w)-w$，$s^e=\max_j|w_j|/M_B$：

| 规则 | 使用的信息与选择目标 |
| --- | --- |
| Min–max | 用极值直接取 $s^e$ |
| Shrink-2.4 | 枚举极值范围的缩小比例，最小化 $\sum_j\lvert e_j(s)\rvert^{2.4}$ |
| WMSE | 搜索 $\sum_j e_j(s)^2$，不使用校准激活 |
| RTNH | 搜索 $e(s)^\top G e(s)$，使用未阻尼的输入 Gram |
| Proxy-static | 对当前补偿权重的固定快照搜索 $\sum_j e_j(s)^2/U_{jj}^2$；$U$ 为阻尼逆 Hessian 的上 Cholesky 因子 |
| Analytic | 取 $s_B^*\widehat\sigma$，$\widehat\sigma$ 为当前行或组的权重 RMS，排除零输入统计列 |

Proxy-static 借用 GPTQ 单步补偿的代价权重，却没有对每个候选执行全部反馈，因而仍是代理。Analytic 先离线求高斯目标导数的零点；该网格的 $s_B^*$ 在 2/3/4 bit 分别约为 1.2240/0.6508/0.3534。RMS 不减均值，不能悄悄换成中心化标准差。

这里的 Hadamard 非相干处理（HIP）以正交 $R$ 改写 $W'=WR,\ X'=R^\top X$，使浮点 $W'X'=WX$；输入 Gram 变为 $R^\top G R$，谱和有效秩不变。它可能改善坐标分布，却不把真实权重变成有保证的独立高斯。旋转与在线成本见[正交旋转](orthogonal-rotation-and-hadamard-quantization.md)。

### 实验支持到哪里

以下为作者报告：五个模型的权重投影按 GPTQ 顺序量化，激活、Embedding 和输出 head 保持 FP16；用 C4 training split 的 128 个随机窗口校准，每个 2,048 token。§5 表 1–2 的代表对照是 Llama-3.1-8B（附录 B.1.1 指定 Instruct checkpoint）、W3、逐输出行尺度：

| 尺度规则 | 无 HIP：WikiText-2 PPL | 有 HIP：WikiText-2 PPL |
| --- | --- | --- |
| Min–max | 41.473 | 15.396 |
| Shrink-2.4 | 14.050 | 9.095 |
| Proxy-static | 11.756 | 9.229 |
| Analytic | 930,372.777 | 9.214 |

解析规则在未经处理的权重上严重失败，HIP 后有竞争力，但这张表并未让它成为最低 PPL 的规则。2 bit 下它仍明显落后于搜索；6 bit 以上差异较小是所测设置的观察，不是所有模型的阈值。

§5.3 与附录 B.4 又在 Llama-3.1-8B-Instruct、W2 的十个选定高误差行进行目标局部对照：HIP 后增加对角缩放或白化，使解析与最佳搜索尺度更接近，却使共同归一化下的最佳搜索重构误差中位数变为 HIP 的 3.63 倍和 20.8 倍。**尺度预测更准、有效秩更大和量化重构更好是不同目标。** 这个结果不表示任意白化都更差，也不是整模型部署性能对照。

本节研读正文与相关实验/理论附录定义，未逐步重证全部附录证明、运行匿名补充代码或复现模型。应用时应先匹配网格、粒度、目标与输入路径，再判断尺度搜索是否值得；不能仅按名义位宽选一个通用高斯常数。

## 11. 自回归校准还需要选择前缀状态

图文样本相同，不表示模型见到的生成状态相同。令 $s_t=(I,x,y_{<t})$，静态校准在固定 $d_{static}$ 上优化，而当前量化策略诱导 $d_{Q_\theta}$，θ 更新后该分布也会变化。[OnPTQ](../methods/onptq.md) v1 §3–4 因此保存当前量化模型 rollout 的精确 token IDs，让浮点/量化模型在同一 prefix 比较；更新校准参数后重新生成状态。它和 dynamic per-token 网格不同：一个选择输入分布，一个估计当前输入的量化范围。

on-policy 覆盖仍不等于合适的目标。OnPTQ 保留原多模态上下文 anchor，并从 top-1 mismatch 或边界被削弱的 near miss 中，用同状态差异和短分支后果筛选、加权关键状态。分支均由当前量化模型继续，浮点模型评估未来分布；这是后果代理，不是任务标签或价值保证。（§4.2–4.4。）

三种生成数据用途因此需区分：固定生成语料改善输入覆盖；学生随训练迭代产生新轨迹以更新状态分布；在候选状态强制不同 token 估计未来差异。只记录“128 个校准样本”会遗漏轨迹长度、刷新轮数和分支预算，也不能证明三个方案成本相同。

需要记录模型/量化初始化、prompt manifest 与 chat template、生成策略、exact prefix IDs、rollout 长度、状态预算、短分支 horizon、静态 anchor 和更新变量。独立评测还应检查自由生成质量、正确性转移与实际成本；OnPTQ 的默认两轮刷新优于五轮，说明更多适配并非单调改善。（§5.4、表 4、附录 C–D。）

## 12. 具身策略的状态选择与时序覆盖

VLA 的动作会改变环境和后续观测，因而校准样本还包含**谁控制采集轨迹**。[PIVOT-Q](../methods/pivot-q.md) v1 §3–5 让量化学生控制环境，浮点教师在同观测、同生成噪声下给参考动作；更新后再采新轨迹。即使原图像和任务集合相同，教师控制与学生控制也可能得到不同状态分布。

三个选择维度需要分开：

| 维度 | PIVOT-Q 的具体例子 | 改变了什么 |
| --- | --- | --- |
| 当前与随后信号 | 首动作 MSE，加同轨迹后续短窗口的折扣 MSE | 找到当前或随后存在差异的状态；没有测反事实行动效果 |
| 时间覆盖 | 每段内排名、分配固定名额，并尽可能保持段内间隔 | 防止所有名额集中在一个片段；不同段的绝对误差不再直接竞争 |
| 保留参考分布 | 目标状态损失加 clean/reference 状态的动作锚点 | 限制集中修复对已有行为的影响；不保证没有负迁移 |

段内百分位只保留次序，去掉误差单位和绝对差距；相同百分位不意味着动作误差或任务影响相同。主代码给并列值按时间分配不同秩，又在名额不足时放宽时间间隔，因此“每段 top-k”还不足以完整描述选择规则。（固定 `74d6e1f8994a03eb45aae5c316a743abda2cba15`，`selector.py`。）

OnPTQ 的两条 token 分支与这里观察到的未来差异也不同：前者改变当前决策后比较未来状态，后者回看一条既有轨迹。两者都是任务后果的代理，都不能代替实际成功率。PIVOT-Q 通过 LoRA 恢复动作，亦不同于只调量化网格；方法比较需同时保留状态来源、选择信号、可训练变量和执行图。

状态数只描述部分预算。若完整 rollout 有 $T$ 个状态、选 $M$ 个、重复更新 $U$ 次、每步另用 $B$ 个 anchors，则目标状态的反向参与次数为 $UM$，锚点至多 $UB$；教师仍需为全轨迹评分。教学例 $T=200,M=16,U=5,B=4$：目标覆盖8%，却有80次目标状态和至多20次锚点参与更新，另有完整轨迹采集/教师评估。不能把8%当作总计算比例。（`train.py` 的采集、选择与更新；算例为整理者计数。）

锚点本身也须记录来源与刷新：PIVOT-Q 正文称原训练数据集，主代码却采 clean LIBERO 的学生短轨迹并存入 replay；这项差异随方法页保留。范围选择、状态选择和行为约束是相连但不同的优化自由度。

## 来源身份

- VEQ v1，来源 `https://arxiv.org/abs/2602.01037v1`，arXiv:2602.01037v1；§6.1引用专家覆盖、两种加权位置及其适用条件。

- SmoothQuant+ v1，来源 `https://arxiv.org/abs/2312.03788v1`，arXiv:2312.03788v1；§5.3与§8引用HumanEval输入校准/配方选择及代理loss与任务分数差异。

- [Norm Tweaking: High-performance Low-bit Quantization of Large Language Models](https://arxiv.org/abs/2309.02784v2)，`arXiv:2309.02784v2`；§3校准生成、表1/8；§5.2混合公式与概率例为整理者展开。

- [QUASAR: Lowering the Loss Floor of Quantization-Aware Training with Loss-Aware Reconstruction](https://arxiv.org/abs/2608.13966v1)，`arXiv:2608.13966v1`；编码候选与固定码加权恢复的区别。

- [Scale Sensitivity in Low-Bit Post-Training Quantization: Curvature of the Quantization Error Landscape](https://arxiv.org/abs/2609.37416v1)，arXiv:2609.37416v1；§3–5、定理 1、命题 1、定义 1、表 1–2/5，以及附录 A.4/A.5.2/B.1/B.4。第 10 节采用作者条件和报告，矩阵改写及变量关系为整理者展开。

下表用于在没有本地资料库时辨识来源；具体论述的章节、公式、图表或代码位置见正文。

| 来源 | 版本或快照 | 说明 |
| --- | --- | --- |
| [CentriQ](https://arxiv.org/abs/2610.06260v1) | arXiv:2610.06260v1 | 免校准与在线分位数、分布假设和开发集的区别 |
| [PIVOT-Q](https://arxiv.org/abs/2610.05745v1)；[作者实现](https://github.com/ruanruan-andy/PIVOT-Q/tree/74d6e1f8994a03eb45aae5c316a743abda2cba15) | arXiv:2610.05745v1；`74d6e1f8994a03eb45aae5c316a743abda2cba15` | 学生环境状态、段内选择、锚点来源与恢复预算；代码限定 GR00T+QuantVLA 主路径 |
| [Calibrate the Decisions That Change the Future](https://arxiv.org/abs/2609.36828v1) | `arXiv:2609.36828v1` | 当前量化前缀、静态 anchor、风险选择与轨迹刷新 |
| [QERA: an Analytical Framework for Quantization Error Reconstruction](https://arxiv.org/abs/2410.06040v2) | `arXiv:2410.06040v2` | 二阶矩、校准规模与 padding 假设 |
| [LUQ: Layerwise Ultra-Low Bit Quantization for Multimodal Large Language Models](https://arxiv.org/abs/2509.23729v3) | `arXiv:2509.23729v3` | — |
| [Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients](https://arxiv.org/abs/2603.17809v1) | `arXiv:2603.17809v1` | — |
| [Breaking Modality Heterogeneity in Low-Bit Quantization for Large Vision-Language Models](https://arxiv.org/abs/2605.19929v1) | `arXiv:2605.19929v1` | — |
| [MASQuant: Modality-Aware Smoothing Quantization for Multimodal Large Language Models](https://arxiv.org/abs/2603.04800v1) | `arXiv:2603.04800v1` | — |
| [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456v2) | `arXiv:2404.00456v2` | — |
| [SpinQuant: LLM quantization with learned rotations](https://arxiv.org/abs/2405.16406v4) | `arXiv:2405.16406v4` | — |
| [FlatQuant: Flatness Matters for LLM Quantization](https://arxiv.org/abs/2410.09426v4) | `arXiv:2410.09426v4` | — |
| [OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models](https://arxiv.org/abs/2308.13137v3) | `arXiv:2308.13137v3` | — |
| [OpenGVLab/OmniQuant](https://github.com/OpenGVLab/OmniQuant/blob/feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14/generate_act_scale_shift.py) | `feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14` | 本地快照已移除 |
| [MQuant: Unleashing the Inference Potential of Multimodal Large Language Models via Full Static Quantization](https://arxiv.org/abs/2502.00425v2) | `arXiv:2502.00425v2` | — |
| [VLMQ: Token Saliency-Driven Post-Training Quantization for Vision-language Models](https://arxiv.org/abs/2508.03351v2) | `arXiv:2508.03351v2` | — |
| [MBQ: Modality-Balanced Quantization for Large Vision-Language Models](https://arxiv.org/abs/2412.19509v2) | `arXiv:2412.19509v2` | — |
| [A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295v1) | `arXiv:2106.08295v1` | — |
| [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877v1) | `arXiv:1712.05877v1` | — |

- [LLM-QAT: Data-Free Quantization Aware Training for Large Language Models](https://arxiv.org/abs/2305.17888v1)，arXiv:2305.17888v1；生成数据与蒸馏目标。

- [PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression](https://arxiv.org/abs/2405.14852v2)，arXiv:2405.14852v2；文档与token采样分布、分实验预处理条件。
- [Vahe1994/AQLM](https://github.com/Vahe1994/AQLM/tree/e79a896ed6656fe4ed06193d42d004e7d0bbdbb2)，固定commit `e79a896ed6656fe4ed06193d42d004e7d0bbdbb2`；PV文本分块与map批次尾部。
