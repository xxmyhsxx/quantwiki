---
title: 模型质量评测：PPL、任务协议与选择性预测
type: concept
tags:
  - evaluation
  - numerical-error
  - reliability
sources:
  - raw/papers/2026-09-28/same-bit-width-different-outcomes/paper.pdf
  - raw/papers/2026-09-26/the-alignment-illusion/paper.pdf
  - raw/papers/2026-09-23/quantization-retrieval-damage/paper.pdf
  - raw/articles/2026-09-22/perplexity/article.md
  - raw/papers/2026-09-21/quantization-reliable-vqa/paper.pdf
updated: 2026-09-29
---

# 模型质量评测：PPL、任务协议与选择性预测

量化后的张量接近参考，不代表生成任务表现不变；任务准确率相近，也不表示置信度或拒答能力相近。评测需要先声明要判断的行为，再固定协议。本文解释 PPL 与可靠性来源中的通用定义，并给出明确标为教学简化的二值正确性例子。

## 1. PPL 的分母是实际参与预测的 token

在 teacher forcing 下，模型看到真实前文。设被计分的位置集合为 $\mathcal I$，使用自然对数：

$$\mathrm{NLL}=-\sum_{t\in\mathcal I}\log p_\theta(x_t\mid x_{<t}),\qquad
\mathrm{PPL}=\exp\left(\frac{\mathrm{NLL}}{|\mathcal I|}\right).$$

等概率预测的教学例：每个正确目标的概率都是 $1/4$，则 PPL 为 4。一般 PPL 是正确 token 概率倒数的几何平均，不是正确率，也不是生成答案的平均概率。

padding、只提供上下文的重叠 token、被忽略的 prompt labels 不计入 $\mathcal I$。因果 LM 常在内部把 logits 与 labels 错开一位；用 loss 乘有效 token 数还原 NLL 时，必须按**错位后的有效 labels**计数，不能盲乘输入序列长度。

合并多个样本时先累加 NLL 和有效 token 数，再取指数。教学例：两段分别有 2 个、8 个计分 token，平均 NLL 分别为 1、3，合并平均 NLL 为 2.6；不能直接平均两段 PPL，亦不能不加权平均两个 loss。

## 2. 上下文窗口会改变比较口径

Hugging Face 的 Perplexity of fixed-length models 说明：模型最多接收 $L$ 个输入 token 时，分成不重叠块会让每块开头缺少前文。滑动窗口可以保留更多历史；用 stride 折中计算成本，并仅对本轮新增目标计分，避免重复累计重叠区域。

要比较浮点与量化模型，固定 tokenizer、文本预处理、BOS/EOS、文档拼接规则、窗口长度、stride、计分 mask 和数据划分。不同 tokenizer 的 token 数与预测单元不同，PPL 不宜直接排名。若只计回答区域，结果应标为该条件下的回答 NLL/PPL，不能与全文语料 PPL混称。

PPL 给的是“真实上下文上的预测损失”。自由生成会把模型输出接回输入，早期错误可能改变后续轨迹，因而需要任务评测补充。生成过程见 [自回归推理](../fundamentals/model/transformer-autoregressive-inference.md)。

## 3. 任务准确率背后也有算法

以下为本页的协议整理，而非新的模型实验。一次量化对照需要共同固定：

- 输入构造：prompt/chat template、few-shot 样例、图像预处理和 token 预算。
- 解码规则：贪心或采样、温度、随机种子、最大输出长度、停止条件。
- 判定规则：整串匹配、选项解析、数字单位归一化或官方软分；记录解析失败、截断、空回答与拒答怎样计分。
- 汇总规则：按样本还是任务平均，分母是否包含所有请求；丢掉失败样本后再报准确率会改变问题。

例如“B”“答案是 B”“(B)”是否相同应由事先确定的 parser 决定，不能看过量化结果后单独放宽。比较同一批样本可同时报告浮点独对、量化独对和双方皆错，避免均值相近掩盖错误集合变化。多模态数据协议的实例见 [VLM 压缩评测](lvlm-compression-benchmark.md)。

## 4. 校准与错误排序是两个问题

以下先用二值正确性 $y_i\in\{0,1\}$、置信度 $c_i\in[0,1]$ 说明。把样本按置信度分成预先规定的箱 $B_b$：

$$\mathrm{ECE}=\sum_b\frac{|B_b|}{n}
\left|\frac{1}{|B_b|}\sum_{i\in B_b}y_i-
\frac{1}{|B_b|}\sum_{i\in B_b}c_i\right|.$$

ECE 检查同一箱内“声称的概率”与“实际正确率”的差异，依赖分箱数量与边界。它不直接衡量把错误排在后面的能力。教学反例：一半正确、一半错误，全部置信度为 0.5，则 ECE 为 0，但基于这个置信度无法挑出更可靠的子集。

自回归模型的 token 概率、整句联合概率和答案正确概率不是同一个对象。可靠性论文的 MaxProb 用生成 token 的概率乘积，具有长度效应；不能把这个乘积无条件当作已校准的正确概率。外部 Selector 则学习另一种置信度。具体设置见 [量化与可靠性](../theory/quantization-reliability-and-selective-prediction.md)。

## 5. 风险—覆盖率与代价

阈值 $\tau$ 下接受集合为 $A_\tau=\{i:c_i\ge\tau\}$。二值错误损失 $\ell_i=1-y_i$ 时：

$$C(\tau)=|A_\tau|/n,\qquad
R(\tau)=\frac{\sum_{i\in A_\tau}\ell_i}{|A_\tau|}.$$

风险的分母是**已接受样本**，覆盖率的分母是**全部样本**。空集合的风险未定义，需要单独处理。$C@r$ 是满足风险上限 $r$ 的可达覆盖率；置信度有并列值时，一个阈值不能随意把同分的正确样本挑出来。经验风险也未必随阈值严格单调，应说明具体选择方法。

风险—覆盖率曲线下面积（AURC，可靠性论文称 AUC）越小通常越好；不要与越大越好的 ROC-AUC 混淆。离散曲线还需声明排序、并列和积分约定。

在二值正确性教学简化中，接受正确得 1 分、接受错误损失 $c$、拒答得 0，平均效用为

$$\Phi_c(\tau)=\frac{n_{\mathrm{accepted,correct}}-c\,n_{\mathrm{accepted,wrong}}}{n}.$$

例如 100 个样本，接受 80 个，其中 76 对、4 错，则覆盖率 80%、风险 5%；$c=10$ 时效用 0.36。VQA 的官方软分与 Effective Reliability 实现应按原协议核对，不能直接用这套二值例子复算论文表格。阈值应在独立验证集选择后固定到测试集；在测试答案上选最优阈值得到的是事后曲线分析，不能冒充部署表现。

## 6. 将不同层次的判断串起来

先用 [误差诊断](quantization-error-diagnosis.md)排除加载、布局和图变换错误，再按共同 PPL 与任务协议判断模型质量；有可靠性要求时补充校准和选择性预测，最后在同一质量约束下比较 [服务性能](serving-performance-evaluation.md)。这能避免用“Kernel 正确”替代“模型可用”，或用“平均准确率接近”替代“低风险覆盖相同”。

本页没有新增模型分数；公式例子只验证计数、归约和指标边界。已有可靠性页的特定结果仍以对应论文为准。

## 7. 平均质量、排序保持与几何诊断分别回答什么

[检索排序稳定性](../theory/quantization-ranking-stability.md) 把“第一名改变”“浮点原本正确的结果丢失”和 Recall/nDCG 净变化分开：新获得的正确答案会抵消旧答案损失，所以只报平均质量不够。但保持浮点答案也不保证正确，必须保留有标签质量指标。

该论文的校准保证控制 $P(\text{接受}\cap\text{翻转})$，本页的选择性风险控制的是 $P(\text{错误}\mid\text{接受})$。两者既有事件差别，也有分母差别。教学例：100 次请求只接受 10 次，其中 2 次翻转，则联合频率 2%，已接受集合的翻转率却是 20%；不能把前者作为后者的保证。阈值顺序统计量的原文问题与纠正在排序页保留。

[多模态对齐诊断](../theory/multimodal-alignment-diagnostics.md) 则说明共享投影可以让几何相似度上升，而视觉任务能力仍下降。完整主角度谱、图像替换/噪声对照和任务分数应一起看；单个 PA gap 也不是量化后任务质量的替代指标。对于动作策略，可沿 [FoldQuantVLA](../methods/foldquantvla.md) 区分固定噪声下动作一致性、坐标幅值误差和闭环成功率，不能把高余弦或少量成功样本升级成部署可靠性保证。

## 8. 无显著差异不等于满足质量等价

[TTS 量化评测](tts-quantization-evaluation.md) 给出一个具体判据：量化减浮点的 UTMOS 均值差的 paired 95% 区间必须全部位于 $[-0.05,0.05]$，同时 WER 差值区间的上界不超过 0.01。这些阈值属于该论文的英文代理评测，不能自动成为所有任务的标准。

区间包含零只表示数据与零差异相容，宽区间并不能证明差异足够小；均值差小于阈值也不足以满足整个区间都在等价带内。必须配对相同输入、平台与生成预算，并区分逐句均值和按词数加权的语料误差。代理自然度、转录正确性、人工偏好与系统开销分别回答不同问题；选择最优配置后的抽样区间也不自动校正配置搜索偏差。

## 来源身份

- [Perplexity of fixed-length models](https://huggingface.co/docs/transformers/en/perplexity)，Transformers main，2026-09-22 获取，`perplexity-20260922-03be86a2`；定义、固定窗口与滑动窗口示例。
- [Evaluating the Impact of Post-Training Quantization on Reliable VQA with Multimodal LLMs](https://arxiv.org/abs/2602.13289v1)，arXiv:2602.13289v1；使用 §3 的置信度、评测指标和 held-out 阈值设置，二值公式与算例为整理者解释。

- [The Undetected Damage of Quantization on Retrieval and How to Fix It](https://arxiv.org/abs/2609.24322v1)，arXiv:2609.24322v1；排序稳定性、间隔分配与校准保证。
- [The Alignment Illusion in Multimodal Large Language Models](https://arxiv.org/abs/2609.30210v1)，arXiv:2609.30210v1；主角度诊断与视觉干预。

- [Same Bit Width, Different Outcomes: Post-Training Quantization of Text-to-Speech Across Architectures](https://arxiv.org/abs/2609.28974v1)，v1；组件消融、质量判据与真实执行对照。
