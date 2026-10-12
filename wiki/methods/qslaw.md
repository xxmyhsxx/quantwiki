---
title: QSLAW：多模态适配中的分组尺度学习与数据 warmup
type: method
tags:
  - vlm
  - weight-quantization
  - qat
  - parameter-efficient-adaptation
sources:
  - raw/papers/2026-09-21/multimodal-qat/paper.pdf
  - raw/repositories/2026-09-21/qslaw/source/llava/quantize/quantizer.py
  - raw/repositories/2026-09-21/qslaw/source/llava/quantize/utils.py
  - raw/repositories/2026-09-21/qslaw/source/llava/quantize/replace.py
  - raw/repositories/2026-09-21/qslaw/source/scripts/pretrain/pretrain.sh
  - raw/repositories/2026-09-21/qslaw/source/scripts/sft/chat_for_13b_v1_5.sh
updated: 2026-10-12
---

# QSLAW：多模态适配中的分组尺度学习与数据 warmup

QSLAW（Quantization-aware Scale LeArning based on multimodal Warmup）研究视觉语言指令适配：给预训练LLM权重的每个组增加一个可学习尺度，在量化前向下训练尺度和多模态适配模块，并先用多模态样本、再加入语言样本。目标是保留量化后模型的下游能力，同时减少可训练参数；主实验为4bit权重量化，没有据此交付低位激活、KV缓存或整数训练。

本页依据arXiv:2408.03735v1的正文方法、实验和关键图表；[所选实现](../implementation/qslaw-scale-learning-and-float-export.md)核固定264dfb081c210a8bab6a3fb5707d827ade7b6802的LLaVA/Vicuna uniform路径。论文结果、原文公式差异和代码执行边界分别保留，未运行作者训练或模型/GPU。

## 1. 为什么语言量化配方可能需要重新适配

模型先将图像送入CLIP视觉编码器，再由projector变成LLM可消费的视觉嵌入，与文本共同进入自回归模型；回答token用交叉熵监督。量化权重改变的是共享语言骨干，对视觉token与文字token的影响不必相同。对象背景见[视觉语言模型中的 token](../fundamentals/model/vision-language-model-tokens-and-quantization.md)。

论文图1对比一个LLaVA-13B block的输入激活：图文输入的离群幅值/出现频率高于纯文本输入。整理者用单层关系说明后果：若权重扰动为E，输出扰动为EX；相同E在较大的输入方向上可被放大。这解释了需要下游监督的动机，不能从一层图直接推出全部VLM都具有同样分布，也不能将所有任务下降归因于激活离群值。

作者以NF4+LoRA的QLoRA作为对照：NF4依参考分布设置非均匀码点，LoRA在基座输出之外增加低秩更新。NF4并不保证任意真实权重恰好等人数落入各桶，LoRA也不是数学上不能补偿离群方向。QSLAW提供的是另一种组级调整自由度和所列实验比较，不是对LoRA能力的普遍否定。更多量化目标见[模态重要性与重构](../research/mbq-vlmq-comparison.md)。

## 2. 学习的是进入舍入的尺度

为避免混淆，将论文组尺度s记为a，量化基础步长记为Δ，整数零点记为z。一个组共享a，预训练W不是逐元素优化变量；论文式7先做 $\widehat W=W/a$，式8再作均匀量化。结合所选uniform代码，可以明确写成

$$q(a)=\operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{W}{\Delta a}\right)+z,0,2^b-1\right),\qquad
\widetilde W(a)=\Delta[q(a)-z].$$

**输出乘的是基础Δ，没有再乘a。** a改变进入舍入的坐标；代码在每次量化前向重新计算整数码，不能把它改写为“固定码q，仅学习恢复步长Δa”。后者是另一种问题，例如[EfficientQAT](efficientqat.md)的E2E-QP。没有反向补偿输入的相应变换，W/a也不自动保持原来的WX，不能等同[AWQ](awq.md)的等价缩放。

论文PDF第4页与TeX式8实际将零点写在外乘之后，即Δq−zp；若zp是整数零点，它与Δ(q−zp)不同。整理者零值检查：Δ=2、zp=1、W=0时，前者得到1，后者得到0。所选代码采用后者；本页保留原文差异，没有静默把原公式当作一致。以上公式描述所选uniform解释，不冒充论文给出的完整梯度规范。

通过回答交叉熵来选择a，模型同时考虑当前数据、projector及后续层，而不是对每个权重独立最小化欧氏误差。在非饱和区，所选代码用round_ste反向得到 $\widetilde{\partial}\widetilde w/\partial a=-w/a^2$；硬前向本身在未跨过舍入边界时几乎处处不变。代理梯度及其条件见[尺度放在哪一侧](../fundamentals/quantization/post-training-and-quantization-aware-training.md#35-学习尺度前先确定它进入哪一侧)。论文没有给出对这种非凸训练的单调下降或全局最优证明。

一组128个权重对应一个尺度时，尺度参数数目约为所选权重数/128；整网仍要传播梯度，可训练参数少不等于没有激活、反向或浮点基座成本。论文表1列84.25M对500.70M，比例约16.83%，属于该实验配置的总可训练参数口径，不能推广为所有模型/LoRA rank的固定比例。

## 3. 数据顺序也是优化条件

作者的消融区分三种尺度学习数据：只用多模态指令、从开始就混合语言与多模态、先进行多模态warmup再混合。论文解释为：只用多模态可能损害骨干语言能力；过早加入语言监督又可能妨碍新projector/尺度适配。这个解释受到所列曲线和消融支持，没有单独证明其为唯一原因。

§3.3规定最初η个迭代只用多模态对，随后加入WikiText；§4.1给尺度学习2个epoch、batch64和语言/下游数据1:1，未给η的可复现数值。论文图2讨论更长的策略曲线，不宜将曲线横轴直接当成主实验训练长度。基础网格由OmniQuant初始化；表4另比较NF4、纯语言OmniQuant及混合校准OmniQuant。

固定脚本依次安排pretrain、mix尺度学习和LoRA指令微调，但mix阶段只显式继承projector，没有传scaling_resume，尺度重新初始化为1；阶段顺序不等于预训练尺度接续。所读Dataset/Trainer未在同次训练按η切换数据，warmup_ratio是学习率调度参数。脚本使用558k预训练及LLaVA-1.5 665k指令数据，论文聊天实验描述LLaVA-80k，projector与Vicuna版本也有差别。这些入口与该v1实验分别记录。

## 4. 如何读质量与适配成本

以下均是作者报告，单位为准确率百分数；差值用百分点。表2的Average是ScienceQA测试集整体准确率，NAT/SOC/LAN、TXT/IMG/NO和年级分组不是八个互不重叠的同权桶，不能再对列作简单平均。

| 表2配置 | 浮点参考 | QLoRA | QSLAW | 可以支持的比较 |
| --- | ---: | ---: | ---: | --- |
| LLaVA-13B，无† | 90.92 | 74.04 | 84.20 | QSLAW比该QLoRA高10.16个百分点，仍低于浮点参考 |
| LLaVA-13B，有† | 90.92 | 86.96 | 91.04 | QSLAW比该QLoRA高4.08个百分点，比浮点参考高0.12 |
| LaVIN-7B | 89.41 | 88.27 | 89.79 | 比QLoRA高1.52、比浮点参考高0.38 |

表2将†说明为更大LoRA rank，不能把有无†合为同一配置；表1的†却表示先进节省显存策略，两表符号含义不同。§4.2.1正文把LaVIN的QLoRA写成88.33及增益1.46，表2写88.27；这里按表保留，并记录正文差异。

| 表1，4张A800 | 可训练参数 | Memory列/GB | 适配时间/小时 | ScienceQA |
| --- | ---: | ---: | ---: | ---: |
| LLaVA，启用表1†策略 | 13B | 71.54 | 8.75 | 90.92 |
| QLoRA | 500.70M | 66.92 | 6.12 | 86.96 |
| QSLAW | 84.25M | 66.52 | 5.76 | 91.04 |

相对于QLoRA，表中时间比为6.12/5.76≈1.063，Memory列比66.92/66.52≈1.006；相对于表1浮点†参考则约1.519及1.075。不能用参数数目比直接代替显存/耗时比，也不能把摘要“最多1.4倍”写成这张表的每个比值。Memory列的跨卡聚合/峰值采样协议未充分给出，本页不自行称为总显存或每卡峰值；这些是适配成本，不是推理GEMM速度。

## 5. 消融说明组合条件，而非尺度单项保证

| LaVIN-7B，表3累计设置 | Average | 相对OmniQuant基线 |
| --- | ---: | ---: |
| OmniQuant | 88.38 | 0 |
| +量化尺度学习 | 84.65 | −3.73个百分点 |
| +混合数据 | 88.68 | +0.30 |
| +多模态warmup | 89.79 | +1.41 |

表3不支持“加尺度总会提升”：纯多模态尺度阶段反而下降，加入数据与warmup才在该设置超过基线。§4.3.1正文又写混合阶段提高0.66及88.54，与表3的88.68/+0.30不一致；88.54在表4另对应OmniQuant-2初始化的LoRA结果，不能自行替换表3。

表4按NF4、OmniQuant-1（语言校准）、OmniQuant-2（混合校准）比较：LoRA为88.27/88.38/88.54，QSLAW为88.98/89.79/89.85；对应增益0.71/1.41/1.31个百分点。这支持在所列初始化下都有收益，不代表所选uniform入口实现了所有NF4实验，或任意初始化都会稳定提升。

聊天图3与图4为代表性回答和GPT-4的1–10分评价，不能当成有充分样本/评审版本与统计区间的完整排行榜。图5训练损失和图6图文token两两余弦提供适配观察；较低训练损失或较高几何相似度本身不能保证视觉内容正确。诊断边界见[多模态对齐](../theory/multimodal-alignment-diagnostics.md)。

## 6. 实际采用前要核什么

该路线适合用下游监督调整组级自由度，并将校准初始化、尺度阶段和指令适配分别评估。必须保留基础Δ/z、分组轴与尺度作用位置，说明整数码是否每步重算；数据顺序、混合比例、训练长度和后续LoRA都会影响比较。

[固定实现页](../implementation/qslaw-scale-learning-and-float-export.md)进一步核了尺度参数、浮点fake quant、Dataset的比例/索引条件和保存入口。源码中的“real quant”路径把恢复浮点值写回普通Linear，没有在该路径打包4bit码或接低位kernel；这限制了可声称的驻留内存和执行方式，没有否定论文独立报告的质量/成本。所有模型、数据与依赖组合的训练、保存重载及性能仍未运行验证。

## 来源身份

- Advancing Multimodal Large Language Models with Quantization-Aware Scale Learning for Efficient Adaptation（`https://arxiv.org/abs/2408.03735v1`），arXiv:2408.03735v1；正文方法/实验、表1–4和图1–6；公式与数据差异按原版保留。
- xjjxmu/QSLAW（`https://github.com/xjjxmu/QSLAW/tree/264dfb081c210a8bab6a3fb5707d827ade7b6802`），264dfb081c210a8bab6a3fb5707d827ade7b6802；本页只用于uniform尺度作用及所选脚本阶段对照，完整登记范围见实现页。
