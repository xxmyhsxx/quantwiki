---
title: 校准数据与量化范围选择
slug: calibration-and-range-selection
sources:
  - raw/papers/quantization/vlm/2025-09-luq-layerwise-ultra-low-bit-quantization-mllm/paper.pdf
  - raw/papers/quantization/vlm/2026-03-fine-grained-post-training-quantization-lvlm-qig/paper.pdf
  - raw/papers/quantization/vlm/2026-05-splitq-breaking-modality-heterogeneity-low-bit/paper.pdf
  - raw/papers/quantization/vlm/2026-03-masquant-modality-aware-smoothing-quantization/paper.pdf
  - raw/papers/quantization/ptq/2024-04-quarot-outlier-free-4-bit-inference-rotated-llms/paper.pdf
  - raw/papers/quantization/ptq/2024-05-spinquant-llm-quantization-learned-rotations/paper.pdf
  - raw/papers/quantization/ptq/2024-10-flatquant-flatness-matters-llm-quantization/paper.pdf
  - raw/papers/quantization/ptq/2023-08-omniquant-omnidirectionally-calibrated-quantization/paper.pdf
  - raw/repositories/quantization/2026-07-opengvlab-omniquant-feffe8ea/source/generate_act_scale_shift.py
  - raw/papers/quantization/vlm/2025-02-mquant-unleashing-inference-potential/paper.pdf
  - raw/papers/quantization/vlm/2025-08-vlmq-token-saliency-driven-post-training/paper.pdf
  - raw/papers/quantization/vlm/2024-12-mbq-modality-balanced-quantization-for-large-vision-language-models/paper.pdf
  - raw/papers/quantization/2021-06-white-paper-neural-network-quantization/paper.pdf
  - raw/papers/quantization/2017-12-quantization-training-neural-networks-efficient-integer-arithmetic/paper.pdf
updated: 2026-09-15
---

# 校准数据与量化范围选择

校准是在有限样本或已有统计上，确定部署所需的量化参数及相关调整。范围选择回答“有限网格用来覆盖哪些数值”；校准还可能为通道缩放、层输出重构或误差补偿收集信息。基础问题对卷积网络和大模型共通，所需样本、统计维度与可接受误差则取决于模型和任务。

本页主要依据 Nagel 等量化白皮书 v1（下称 W）§2.2、§3.1、§3.5，以及 Jacob 等整数推理论文 arXiv v1（下称 J）§3。两篇论文的历史配置是案例，不是当前模型的通用默认值。

## 1. 先确定量化对象、网格与目标

仿射网格由整数端点 $q_{\min},q_{\max}$、正步长 $\Delta$ 和整数零点 $z$ 确定，反量化为 $\widehat x=\Delta(q-z)$。定义和粒度见 [[uniform-quantization-and-groups|均匀量化与分组]]。给定希望覆盖的实数范围 $[a,b]$，先将其扩展到包含零，常见初始化为

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

W §3.1 的 MSE+Xent 案例在最后层使用输出分布相关目标，不等于“所有层都应改成交叉熵”。[[layer-reconstruction-second-order-compensation|层输出重构]]说明 $\|(W-\widehat W)X\|_F^2$ 如何包含输入的二阶统计；[[awq|AWQ]]采用的分组输出裁剪目标也不能叫作纯权重 MSE。

范围还可以用梯度学习。[[omniquant|OmniQuant]] v3 §3.2 将当前权重极值 $a,b$ 映射为 $l=\beta a,u=\gamma b$，学习 $\beta,\gamma$，用 Transformer block 输出误差决定裁剪。相对强度使端点跟随等价变换后不断变化的权重范围；它不是直接最小化权重张量 MSE，也不表示按比例删掉权重。若 $\gamma=0.5$，只说明正端点变为当前最大值的一半，不能据此推断有一半元素被裁剪。

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

这种做法依赖模型中可用的 BN 统计。LayerNorm/RMSNorm 的输入相关归一化没有相同的固定总体统计，不能照抄 BN 范围公式；[[diagonal-scaling-equivalent-transform|等价变换页]]进一步区分固定 BN 融合与归一化输出侧缩放。缺少真实数据时，应降低对部署分布覆盖的确信程度，而不是宣称“无需数据所以没有分布风险”。

## 5. 校准、训练、验证与测试怎样分工

校准用于估计网格、统计或局部调整；训练用于拟合模型参数；验证用于选择配置；独立测试用于评价选择后的结果。名称并不决定用途：反复查看测试分数来选裁剪比例，已经使测试集参与了选择。

具体记录应足以说明输入如何产生：数据版本、样本数、抽样方式、预处理、随机性、模型与权重版本，以及输入形状。图像任务记录分辨率和增强；文本任务记录分词器、序列长度、拼接/截断、padding/mask 及实际参与统计的 token。不要直接将训练增强得到的范围当作部署范围。若使用标签或浮点教师输出来选择参数，也说明其来源和用途。

静态激活量化离线固定网格，容易受分布偏移影响；动态量化按当前输入估计网格，增加运行时统计开销，也不能保证未见输入的任务质量。[[smoothquant|SmoothQuant]]的 O1–O3 提供这两个维度的具体案例。

离线学到的变换与动态激活网格可以并存。[[omniquant|OmniQuant]] 离线学习通道缩放，默认激活量化仍按当前 token 估计范围。其官方快照 `feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14` 的 `generate_act_scale_shift.py` 还区分两种初始化统计：`get_act_scales` 累积逐通道最大绝对值；`get_act_shifts` 对每段输入的逐通道区间中点 $c_t=(\max X_t+\min X_t)/2$ 作 $m_t=0.99m_{t-1}+0.01c_t$ 的 EMA，初值为首段中点。后者不是样本均值，也不是所有样本合起来的全局 min/max 中点；这些统计只用于初始化，不能代替训练后参数。

OmniQuant v3 表 A10–A11 比较几个文本校准集与样本数量，说明所测 LLaMA-7B 配置的变化有限，但不构成部署分布漂移后的恢复证据。样本增加也未始终改善两个评测集，因此“固定 128 条就足够”只能是实验配置，不能升级为普遍规则。

J §3.1 在训练时对激活范围使用指数移动平均，并在部分配置中延迟启用激活模拟量化；这是处理训练初期分布变化的策略，不能据此给所有 PTQ 任务规定相同样本数或预热步数。训练范式见 [[post-training-and-quantization-aware-training|PTQ、QAT 与代理梯度]]。

完成范围选择后，应在未用于选择的数据上查看层误差与任务质量，并核对实际执行的网格是否一致。[[quantization-error-diagnosis|量化误差诊断与验证]]说明如何分离图变换、量化和后端实现的问题。本页没有模型实验，不提供校准规模充分性的实证保证。

## 6. 图文校准增加的是数据与目标条件

[[vision-language-model-tokens-and-quantization|视觉语言模型]] 沿用前述网格和误差基础，但同一校准序列含有视觉、提示和回答等不同位置。必须记录图像处理后的 token 数、模板、截断、padding、回答是否进入输入，以及重构 mask 与交叉熵标签的区别；不能把所有非视觉位置一概视作同一监督对象。

MBQ v2 使用 128 个图文样本，VLMQ v2 使用 512 个；这两个数字分别属于各自算法和协议，不是通用“足够样本数”。[[mbq|MBQ]] 表 4 显示图文校准、目标加权与范数选择不能互相替代；[[vlmq|VLMQ]] 的梯度又来自局部重构 MSE，不能与回答交叉熵梯度混称同一敏感性。

求和、逐 token 平均、先按模态平均再加权会得到不同目标。大量视觉 token 是否主导统计，应结合实际归约规则检查；不同目标产生的敏感性排序也不宜直接当成跨任务稳定属性。这些条件应在比较方法前明确，后续任务评价仍需使用未参与选择的数据。

分模态还可以改变网格，而不改变误差目标：[[mquant|MQuant]] 为视觉和文本分别离线估计静态激活范围，避免较宽的视觉范围使小幅文本值被粗量化。参数随层变化，推理时按模态选择；这与 MBQ 的目标加权、VLMQ 的 token 权重是不同维度。其表 10 对比说明，直接共用单一静态范围可能很快却严重损害质量。表 8 的样本数稳定性只适用于所测数据和配置，不构成分布外保证。（MQuant v2，§3.1、表 8/10）

## 7. 变换与范围选择的先后关系

[[flatquant|FlatQuant]] v4 表 18 在同一 LLaMA-3-8B 框架中比较裁剪位置：变换前学习裁剪得到 WikiText2 PPL 7.37，变换后为 6.98。通道混合改变了被裁剪对象的分布，所以范围选择应针对最终待量化的表示；逆变换保持未量化计算，不会恢复已裁掉的信息。

其表 17 用 WikiText2、C4、Pile 校准时质量变化较小，只支持这些文本来源下的观察，不是跨模态或所有部署分布的保证。表 15 的同模型跨量化配置复用亦应与跨模型、跨领域迁移区分。这些边界与 [[omniquant-affinequant-flatquant-comparison|学习变换路线的目标差别]] 一起决定怎样解释校准结果。

[[quarot|QuaRot]] 的固定旋转本身不训练，默认 GPTQ 阶段仍需旋转后的校准输入；[[spinquant|SpinQuant]] 则先对最终 next-token CE 学旋转，再固定旋转做 GPTQ。后者表 3 中 W16A4 学习再 GPTQ 优于在 RTN W4A4 噪声下学习后再 GPTQ，说明校准时与最终采用的量化器是否匹配会影响结果。不能把“量化参数在变”和“所有量化误差已被联合优化”视为同一件事。（QuaRot v2 §4；SpinQuant v4 §4.2。）

SpinQuant 表 11 在所测模型中，128 与 800 样本的 PPL 都约 6.2，100 到 200 步也基本饱和。它给出了按收益与成本选校准预算的证据，不是所有任务统一用某个样本数的规则；尤其不能从文本内部比较推断图文分布已经覆盖。

## 8. 校准统计稳定，不等于任务目标一致

[[masquant|MASQuant]] v1 表 4 中，模态等权的 PPL/准确率为 17.2/56.9，降低视觉权重后为 33.7/58.1：困惑度与任务平均分并未选择同一配置。模态重要性应跟随明确目标，不能由一次最优系数推出普遍的视觉/文字重要性排序。其公开损失分支还采用按总 mask 数归一化及额外视觉系数，方法页分别记录论文与代码条件。

[[splitq|SplitQ]] v1 表 3 报告不同校准样本数下较高的通道集合重合率，验证的是选列稳定性；不等价于准确率稳定、低秩补偿稳定或跨分布泛化。它的文字百分位评分和视觉最大值都是选通道的统计代理，不是损失梯度。

白化补偿还使用激活二阶矩，见 [[invertible-transforms-and-kronecker-products|激活加权低秩近似]]。校准未覆盖的方向既可能影响数值求逆，也可能在部署数据上突然重要。需要记录变换学习与补偿统计分别使用的数据及模型状态；不能只记录“128 个样本”而省略模态组成、有效 token 数与测试独立性。MASQuant 的公开音频数据路径存在测试来源待核对项，具体证据与未确定部分在其方法页保留。

## 9. 归因目标、数据混合与位宽选择

[[qig|QIG]] v1 §3、附录 B 将归因目标设为浮点/量化输出差异，沿参考激活积分；这与回答 CE 的单点梯度不是同一个敏感性。附录 C 使用 InfoVQA 校准并评价 OCR 任务，说明特定数据—任务组合的效果，不证明部署后分布漂移已解决。需要记录基线、有效 token、目标归约与权重后处理，不能只记录校准样本数。

[[luq|LUQ]] v3 §4.1、表 3 的 WikiText-2/TextVQA 混合则改变校准 token 组成；1:1 的 token 比例不等于两种模态在重构损失中各占一半。其 LLaVA 超低比特配置从纯文本到混合校准时 TextVQA 51.8 → 53.4，而 4-bit AWQ 没有相同改善，故比例选择仍受量化器与位宽影响。

校准数据之外，位宽选择也会使用任务信息。LUQ §4.2 用 MME Perception 门槛选择 LLaVA 的低比特层数，最终又报告 MME；若不另分选择集和测试集，就不能把该项当作独立验证。TextVQA 同时出现于校准来源和评价任务时，应核对具体 split 和样本 ID：同名任务不证明泄漏，但也不能仅凭文字描述声称已排除。[[mixed-precision-allocation|混合精度分配]] 将这些选择阶段写入预算与质量流程。
