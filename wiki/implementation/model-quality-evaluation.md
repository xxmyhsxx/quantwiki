---
title: 模型质量评测：PPL、任务协议与选择性预测
type: concept
tags:
  - evaluation
  - numerical-error
  - reliability
sources:
  - raw/papers/2026-09-21/qwen3
  - raw/papers/2026-09-21/deepseek-r1
  - raw/papers/2026-09-21/llama-3
  - raw/papers/2026-10-03/quantforge
  - raw/papers/2026-09-28/same-bit-width-different-outcomes
  - raw/papers/2026-09-26/the-alignment-illusion
  - raw/papers/2026-09-23/quantization-retrieval-damage
  - raw/articles/2026-09-22/perplexity
  - raw/papers/2026-09-21/quantization-reliable-vqa
  - raw/papers/2026-10-03/decision-flips
  - raw/papers/2026-10-07/pivot-q
  - raw/repositories/2026-10-07/pivot-q
  - raw/papers/2026-09-21/int8-gg
  - raw/papers/2026-09-21/helm
  - raw/papers/2026-09-21/longbench-v2
  - raw/papers/2026-09-21/mmlu-pro
  - raw/papers/2026-09-21/ruler
  - raw/repositories/2026-09-21/lm-evaluation-harness
updated: 2026-10-14
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

### 1.1 按字节计分仍须固定文本和上下文

[HELM](helm-evaluation-protocol.md)在所选语言建模场景采用bits-per-byte（BPB），以统一不同tokenizer的长度单位。整理者定义：同一被计分文字区域有B个原始字节、自然对数NLL时，$\mathrm{BPB}=\mathrm{NLL}/(B\ln2)$；例如8字节区域的NLL=8ln4，BPB=2。合并时先加总NLL和相应字节数，不能平均各文档BPB。

B须对应真正参与计分的文字区域，不能把只供上下文的prefix字节混入分母；字符数也不等于UTF-8字节数。仍须对齐原始文本/编码、预处理、特殊token、文档边界和上下文。统一分母不保证模型概率或任务质量可比；具体token到字节的计数须由实现核对。（HELM v2 §5.1.1/C.1及整理者计数定义。）

## 2. 上下文窗口会改变比较口径

Hugging Face 的 Perplexity of fixed-length models 说明：模型最多接收 $L$ 个输入 token 时，分成不重叠块会让每块开头缺少前文。滑动窗口可以保留更多历史；用 stride 折中计算成本，并仅对本轮新增目标计分，避免重复累计重叠区域。

要比较浮点与量化模型，固定 tokenizer、文本预处理、BOS/EOS、文档拼接规则、窗口长度、stride、计分 mask 和数据划分。不同 tokenizer 的 token 数与预测单元不同，PPL 不宜直接排名。若只计回答区域，结果应标为该条件下的回答 NLL/PPL，不能与全文语料 PPL混称。

PPL 给的是“真实上下文上的预测损失”。自由生成会把模型输出接回输入，早期错误可能改变后续轨迹，因而需要任务评测补充。生成过程见 [自回归推理](../fundamentals/model/transformer-autoregressive-inference.md)。

### 2.1 长度分组和截断不能替代配对长度试验

[LongBench v2](longbench-v2-context-and-scoring.md)的语料长度按words分组，模型窗口按各自tokenizer的tokens限制，超窗从中间截断。必须分别记录原文长度、实际输入tokens、提示/输出预算、保留位置与证据是否留下；“语料长”不能直接说明模型读入同样长的材料。

不同长度组的任务构成也可能不同。要判断增加长度本身的影响，应对同一问题构造可比的长度变体、控制有效证据/干扰位置，并固定模板、生成和计分；只比较两组平均值会混入任务差异。这是整理者的配对设计建议，不是论文已经完成的长度因果试验。

### 2.2 有效长度依赖任务阈值和汇总

有效长度要声明测试档位、任务集合/权重、计分、通过阈值与比较方向。[RULER](ruler-task-complexity-and-effective-context.md)用13配置均值超过85.6%；检索占8/13权重，均值通过不保证每类或每项都通过。最大通过档只给该网格的读数；最高档通过尚未测更长，最低档失败也没定位更短边界。

跨长度再汇总时，档位等权、序号线性权重和token数比例权重回答不同问题。保留各档分数，按明确的使用分布计费/比较；不能把选定阈值和汇总权重当作模型自身不变的窗口规格。（RULER v3 §4/表3及整理者解释。）

### 2.3 Llama 3.1的长上下文适应与检索口径

报告v3的405B在预训练后期以约800B长序列训练tokens分六阶段，从8K扩到128K；每阶段同时要求短上下文评测恢复、对应长度单针检索通过。只用短上下文SFT会退化长上下文能力，作者加入合成长文QA、摘要和repo代码任务，报告其消融中0.1%混合比例较好。短上下文DPO没有退化的观察，以已经具备高质量长上下文能力的SFT checkpoint为条件；128K不能仅由RoPE参数或单针成绩推定。（§3.4 Long Context Pre-Training、§4.3 Long Context。）

最终multi-needle插四针、要求找回其中两针，报告十种长度至128K的平均recall；8B/70B/405B为98.8/97.5/98.1，不能当作全部长文任务的成功率。ZeroSCROLLS用validation，任务分别用exact match、F1和ROUGE-L；InfiniteBench En.QA表中405B为30.5、70B为36.7，未支持正文“405B所有项更好”的概括。它们是模型能力表，不是FP8/BF16量化对照。（§5.2.6 Long Context Benchmarks及表21；均按v3的Llama 3.1身份。）

## 3. 任务准确率背后也有算法

以下为本页的协议整理，而非新的模型实验。一次量化对照需要共同固定：

- 输入构造：prompt/chat template、few-shot 样例、图像预处理和 token 预算。
- 解码规则：贪心或采样、温度、随机种子、最大输出长度、停止条件。
- 判定规则：整串匹配、选项解析、数字单位归一化或官方软分；记录解析失败、截断、空回答与拒答怎样计分。
- 汇总规则：按样本还是任务平均，分母是否包含所有请求；丢掉失败样本后再报准确率会改变问题。

例如“B”“答案是 B”“(B)”是否相同应由事先确定的 parser 决定，不能看过量化结果后单独放宽。比较同一批样本可同时报告浮点独对、量化独对和双方皆错，避免均值相近掩盖错误集合变化。多模态数据协议的实例见 [VLM 压缩评测](lvlm-compression-benchmark.md)。

### 3.1 失败计零、随机补偿和有效样本计分有不同分母

总请求n、有效正确C、解析失败/拒答I时，strict为C/n；给无效请求k选一随机猜测期望分则是(C+I/k)/n；只保留有效请求为C/(n−I)，须注明条件及覆盖率，I=n时未定义。教学例：100请求中70对、20错、10无效，四选一分别70%、72.5%、约77.8%；最后一个没有包含全部请求。

[LongBench v2](longbench-v2-context-and-scoring.md)主表与附表采用前两个口径。补偿是人为评价政策，不是模型实际答对无效样本；应同时给出invalid率并在浮点/压缩对照中共用parser，不能将补偿值与strict混比。

### 3.2 可变选项数与随机兜底要逐例计账

每题只有一个正确选项，题i有k_i个实际选项时，均匀随机选择的总体期望是 $n^{-1}\sum_i1/k_i$，不是 $1/(n^{-1}\sum_i k_i)$。教学例：两题分别4、10选一，期望为17.5%，而平均7选项的倒数约14.3%。[MMLU-Pro](mmlu-pro-options-and-answer-extraction.md)只有83%为十选一，不能据此把整套题或解析失败子集的基线一律设为10%。

有效正确C、失败集合F时，从每题实际选项随机兜底的期望为 $[C+\sum_{i\in F}1/k_i]/n$；一次实际随机抽样还有seed和波动。必须检查抽出的字母属于该题实际选项，保留抽取失败与兜底记录。固定补偿、实际随机抽样和失败计零是三种不同协议；任务名相同也须核parser及聚合。（MMLU-Pro v6 §3–4及整理者计数。）

### 3.3 多目标召回不等于整题正确或输出精确

以下为教学集合计分：非空gold集合G、按预定parser/去重得到的预测集合P，recall为 $|G\cap P|/|G|$，非空P的precision为 $|G\cap P|/|P|$。四目标找回三项是75%召回，整题仍不完整；全含四目标但加十个错误项，召回100%，precision却约28.6%。顺序、重复和输出格式另按任务判断。

逐例平均召回与汇总目标计数也不同：两例分别1/1、0/9，宏平均50%，目标微平均10%。[RULER](ruler-task-complexity-and-effective-context.md)采用目标出现的recall-based计分，但其具体字符串匹配/归一化仍待实现核对；上述集合公式不是对作者parser的运行验证。量化对照应同时保留遗漏、多余项和整题成功，避免单一召回隐藏输出变化。

### 3.4 请求、过滤和聚合须分别核对

生成字符串与候选loglikelihood是不同请求；parser选首/末匹配、取首个重复响应和metric忽略大小写也是不同操作。[harness固定实现](lm-evaluation-harness-request-filter-and-aggregation.md)的MMLU-Pro只启用一条大小写敏感regex，失败计零，后续exact_match归一化不能恢复已漏掉的字母。任务名相同仍须对齐执行协议。

学科内逐例均值a_s、实际样本数n_s时，按样本加权总分是 $\sum_s n_s a_s/\sum_s n_s$；学科等权平均则回答另一问题，PPL等非线性指标另按其计数汇总。保留最终配置、原始/过滤响应、指标键与实际分母，并隔离会复用旧输入或模型输出的缓存；没有新模型调用的缓存命中不能证明本次压缩配置已经评测。（所选harness代码及整理者对照要求。）

### 3.5 多次抽样、单次成功与多数投票

同一题生成K个回复，正确指示为$c_i$时，抽样平均$\widehat{\rm pass@1}=K^{-1}\sum_i c_i$估计单次生成的成功率；在恰好K次尝试的教学口径中，“至少一条正确”为$\mathbf1\{\sum_i c_i>0\}$，跨题平均得到相应pass@K；cons@K则先按parser抽答案、投票选一个，再判该选择是否正确。第二种知道尝试集合里有正确答案，不证明部署能够挑中它；多数票还依赖parser、并列和无效输出处理。

R1报告v2默认max32,768 tokens、temperature0.6/top-p0.95；每题AIME/GPQA64次，MATH/Codeforces16次、LCB8次，pass@1按多次正确指示均值计算。附录最终R1 AIME 2024的pass@1=79.8%、pass@64=90.0%、多数票cons@64=86.7%；主文Zero原图/图注明确cons@16，其86.7%不能套同一cons@64标签。标准pass@K的其他估计器和本文parser实现未核，以上集合式是教学定义。（附录Experiment Setup、An Analysis on CoT Length；Zero原图。）

量化对照需要共同固定每响应长度、样本次数、停止条件、提示与选择规则，并记录**总实际生成tokens**。相同单响应上限不保证总预算相同。原报告的LCB时间段2024-08至2025-01、SWE Verified agentless/Aider diff等协议也属于结果身份；o1-1217引用外部官方结果不能证明全部由作者按同协议重跑。10-gram去污染仍未排除改写污染。训练阶段和输出蒸馏的对象区别见[推理后训练](../fundamentals/model/reasoning-post-training-and-distillation.md)。

### 3.6 Qwen3的模式、总输出与任务预算

Qwen3 v1评测两模式时还改变采样配方，不能当作只改flag的单变量对照：

| 模式 | temperature / top-p / top-k | presence penalty |
| --- | --- | --- |
| thinking | 0.6 / 0.95 / 20 | Creative Writing v3/WritingBench为1.5，其余未列 |
| non-thinking | 0.7 / 0.8 / 20 | 1.5 |

普通max output length为32768，AIME'24/25另为38912；这与[思考段预算](../fundamentals/model/reasoning-post-training-and-distillation.md#5.1%20%E6%80%9D%E8%80%83%E9%A2%84%E7%AE%97%E6%8E%A7%E5%88%B6%E8%BE%93%E5%87%BA%E9%98%B6%E6%AE%B5%EF%BC%8C%E4%B8%8D%E4%BF%9D%E8%AF%81%E6%AF%8F%E7%B1%BB%E4%BB%BB%E5%8A%A1%E5%8F%97%E7%9B%8A)分开。AIME每年30题、每题64次取平均正确率，估计pass@1；GPQA每题10次平均。Codeforces经CodeElo、最多8次独立尝试，不能把这三种协议统称pass@64；表21括号才另列pass@64。（§4.6–4.7。）

LCB v5为2024.10–2025.02，thinking提示去掉“只返回程序”限制；BFCL v3的Qwen3用FC格式、多轮用YaRN部署到64K，部分baseline从排行榜FC/Prompt取较高值，其余按Prompt。提示、输出约束和执行格式共同属于结果身份。预训练119语言不表示每种能力均已评：INCLUDE44语言/MMMLU14语言只抽各10%数据，Belebele评80语言并排42未优化语言。（§4.6、附录A.1.2。）

更多思考也有任务反例。RULER表23用YaRN factor4、thinking budget8192：235B非thinking平均95.0/128K90.6，thinking92.2/86.0；32B93.7/91.0，8B89.1/84.4均是非thinking/thinking均分。作者提出思考“可能干扰检索”，仍是解释假设；这不是量化对照或所有长文任务有效长度证明。对照需固定模型、模式、外推、采样、思考/答案预算与实际tokens。（附录A.1.1、表23。）

## 4. 校准与错误排序是两个问题

以下先用二值正确性 $y_i\in\{0,1\}$、置信度 $c_i\in[0,1]$ 说明。把样本按置信度分成预先规定的箱 $B_b$：

$$\mathrm{ECE}=\sum_b\frac{|B_b|}{n}
\left|\frac{1}{|B_b|}\sum_{i\in B_b}y_i-
\frac{1}{|B_b|}\sum_{i\in B_b}c_i\right|.$$

ECE 检查同一箱内“声称的概率”与“实际正确率”的差异，依赖分箱数量与边界。它不直接衡量把错误排在后面的能力。教学反例：一半正确、一半错误，全部置信度为 0.5，则 ECE 为 0，但基于这个置信度无法挑出更可靠的子集。

自回归模型的 token 概率、整句联合概率和答案正确概率不是同一个对象。可靠性论文的 MaxProb 用生成 token 的概率乘积，具有长度效应；不能把这个乘积无条件当作已校准的正确概率。外部 Selector 则学习另一种置信度。具体设置见 [量化与可靠性](../theory/quantization-reliability-and-selective-prediction.md)。

### 4.1 等量分箱与有符号误差不能混用

等样本量分箱按置信度排序后分组，等宽分箱按概率区间分组；箱数、余数、并列和空箱处理需声明。分类任务的top probability取自类别概率向量，生成整句概率衡量的是另一事件。

每箱先取绝对差，再按样本数加权。教学例：两个各10例的箱，平均置信度为0.9/0.6、准确率为0.7/0.8，有符号差相消为0，ECE却为0.2。HELM的具体配置与原式差异见[协议页](helm-evaluation-protocol.md#4.%20%E7%BD%AE%E4%BF%A1%E5%BA%A6%E6%A6%82%E7%8E%87%E5%92%8C%E5%8F%AF%E9%9D%A0%E6%8E%92%E5%BA%8F%E5%88%86%E5%88%AB%E6%A0%B8%E5%AF%B9)。

## 5. 风险—覆盖率与代价

阈值 $\tau$ 下接受集合为 $A_\tau=\{i:c_i\ge\tau\}$。二值错误损失 $\ell_i=1-y_i$ 时：

$$C(\tau)=|A_\tau|/n,\qquad
R(\tau)=\frac{\sum_{i\in A_\tau}\ell_i}{|A_\tau|}.$$

风险的分母是**已接受样本**，覆盖率的分母是**全部样本**。空集合的风险未定义，需要单独处理。$C@r$ 是满足风险上限 $r$ 的可达覆盖率；置信度有并列值时，一个阈值不能随意把同分的正确样本挑出来。经验风险也未必随阈值严格单调，应说明具体选择方法。

风险—覆盖率曲线下面积（AURC，可靠性论文称 AUC）越小通常越好；不要与越大越好的 ROC-AUC 混淆。离散曲线还需声明排序、并列和积分约定。

在二值正确性教学简化中，接受正确得 1 分、接受错误损失 $c$、拒答得 0，平均效用为

$$\Phi_c(\tau)=\frac{n_{\mathrm{accepted,correct}}-c\,n_{\mathrm{accepted,wrong}}}{n}.$$

例如 100 个样本，接受 80 个，其中 76 对、4 错，则覆盖率 80%、风险 5%；$c=10$ 时效用 0.36。VQA 的官方软分与 Effective Reliability 实现应按原协议核对，不能直接用这套二值例子复算论文表格。阈值应在独立验证集选择后固定到测试集；在测试答案上选最优阈值得到的是事后曲线分析，不能冒充部署表现。

### 5.1 准确率面积和风险面积要使用同一排序网格

以降序置信度排序、二值正确性y_i为教学条件，前r例准确率 $a_r=r^{-1}\sum_{i=1}^r y_i$，r=1,…,n。HELM有限式15为 $\mathrm{SCAA}=n^{-1}\sum_r a_r$，越大越好；若AURC也在同一r/n网格用同样矩形权重，二值风险1−a_r使AURC=1−SCAA。非二值损失、不同积分/端点或排序规则不能沿用此式。

准确率的正面积沿coverage从0到1累积；若改按递增阈值t积分，coverage非增，因此需要负号或反向端点。置信度原子/并列也可能令恰好10%的阈值不存在；前⌊.1n⌋排序截取、部署阈值及同分处理是不同协议。没有接受样本时不能把数学约定的准确率1当作实际可靠性。

## 6. 将不同层次的判断串起来

先用 [误差诊断](quantization-error-diagnosis.md)排除加载、布局和图变换错误，再按共同 PPL 与任务协议判断模型质量；有可靠性要求时补充校准和选择性预测，最后在同一质量约束下比较 [服务性能](serving-performance-evaluation.md)。这能避免用“Kernel 正确”替代“模型可用”，或用“平均准确率接近”替代“低风险覆盖相同”。

## 7. 平均质量、排序保持与几何诊断分别回答什么

[检索排序稳定性](../theory/quantization-ranking-stability.md) 把“第一名改变”“浮点原本正确的结果丢失”和 Recall/nDCG 净变化分开：新获得的正确答案会抵消旧答案损失，所以只报平均质量不够。但保持浮点答案也不保证正确，必须保留有标签质量指标。

该论文的校准保证控制 $P(\text{接受}\cap\text{翻转})$，本页的选择性风险控制的是 $P(\text{错误}\mid\text{接受})$。两者既有事件差别，也有分母差别。教学例：100 次请求只接受 10 次，其中 2 次翻转，则联合频率 2%，已接受集合的翻转率却是 20%；不能把前者作为后者的保证。阈值顺序统计量的原文问题与纠正在排序页保留。

[多模态对齐诊断](../theory/multimodal-alignment-diagnostics.md) 则说明共享投影可以让几何相似度上升，而视觉任务能力仍下降。完整主角度谱、图像替换/噪声对照和任务分数应一起看；单个 PA gap 也不是量化后任务质量的替代指标。对于动作策略，可沿 [FoldQuantVLA](../methods/foldquantvla.md) 区分固定噪声下动作一致性、坐标幅值误差和闭环成功率，不能把高余弦或少量成功样本升级成部署可靠性保证。

## 8. 无显著差异不等于满足质量等价

[TTS 量化评测](tts-quantization-evaluation.md) 给出一个具体判据：量化减浮点的 UTMOS 均值差的 paired 95% 区间必须全部位于 $[-0.05,0.05]$，同时 WER 差值区间的上界不超过 0.01。这些阈值属于该论文的英文代理评测，不能自动成为所有任务的标准。

区间包含零只表示数据与零差异相容，宽区间并不能证明差异足够小；均值差小于阈值也不足以满足整个区间都在等价带内。必须配对相同输入、平台与生成预算，并区分逐句均值和按词数加权的语料误差。代理自然度、转录正确性、人工偏好与系统开销分别回答不同问题；选择最优配置后的抽样区间也不自动校正配置搜索偏差。

## 9. 输出分布一致性要固定前缀和指标口径

Decision Flips v1 §3–8 说明评测压缩后的输出保真时，需要把浮点与压缩模型放到同一前缀，再比较 forward KL、TV 和 greedy top-1 ID。这是条件计算的一致性，不是自由生成轨迹相同，也不表示答案正确。具体定义、Jensen 聚合差异、间隔界与留出试验见[分布距离到 token 翻转](../theory/quantization-ranking-stability.md#8.%20%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E7%9A%84%E5%88%86%E5%B8%83%E8%B7%9D%E7%A6%BB%E6%80%8E%E6%A0%B7%E5%8F%98%E6%88%90%20token%20%E7%BF%BB%E8%BD%AC)。

| 评测目的 | 应报告的量与控制 |
| --- | --- |
| 同一模型、同一语料内比较压缩配置 | 任务前缀上的直接 flip，辅以 KL/TV；保留参考版本、序列和位置分母 |
| 跨语料理解散度大小 | 分语料报告；√平均 KL 与平均逐位置 √KL 不同，不能复用未校准的比例系数 |
| 输出分布距离 | forward/reverse 方向、温度、完整词表或 top-k 加聚合 tail；聚合会低估 TV |
| 决策间隔或微小扰动 | output head dtype、并列与 tie-break；BF16 的精确并列不等于连续的零附近密度 |
| draft 的实际价值 | greedy/采样接受协议、整段一致序列、draft/验证成本与实测延迟 |
| 最终质量与生成稳定性 | 自由生成、任务正确率、输出长度和停止/预算耗尽；一致性指标仅提供补充 |

当两模型已经可以运行，直接数任务 flip 的成本与计算散度同阶，没有必要用论文经验比代替实测。非有限 logits 应单独作为运行失败记录，不能只在有限样本上给出散度后称整个配置有效。论文只在部分真实后端验证，FP8 明确未通过其试验。

### 9.1 Llama 3.1的罕见损坏与reward分布

Llama 3报告v3 §6.2发现：标准benchmark常显示405B的FP8与BF16接近，未设尺度上界时却偶发corrupted responses。作者因此比较两种推理产生的100,000条回复的reward-model score分布，主图26显示最终[选择性FP8配方](../fundamentals/numeric-formats/fp8-and-mx-data-formats.md#3.1%20Llama%203.1%E7%9A%84%E9%80%89%E6%8B%A9%E6%80%A7FP8%E6%8E%A8%E7%90%86)分布接近BF16。平均任务分数可能漏掉罕见自由生成损坏，reward分布提供另一观察，但没有给corruption rate、均值差或显著性检验，不能据直方图证明逐token等价或罕见风险为零。

同版源包的`fp8_appendix.tex`未被`paper.tex`引入，三张补充图未编入当前92页PDF。补充稿报告首尾层保护的影响大于尺度上界、row-wise略优于tensor-wise；直方图仍高度重叠且无量化统计，文中“significant shift”不能当作统计显著性检验通过。这个实例保留原文报告与图的实际证据范围，未核reward-model版本、抽样/解码manifest或匹配kernel。完整质量大表与这些FP8对照分别解释，不能互相替代。

## 10. 策略恢复要分清适配范围与统计单位

动作策略中，同状态、同随机噪声下的动作差异隔离输出变化；闭环成功率则让模型自己的动作改变后续环境。两者相互补充，不能把前者的 MSE 降低直接解释为未见任务上的成功率提高。

[PIVOT-Q](../methods/pivot-q.md)提供了适配边界的实例：固定作者仓库 `74d6e1f8994a03eb45aae5c316a743abda2cba15` 的 `docs/experiment_protocol.md` 指定恢复与评测使用相同 task IDs，最终评测另跑 episode。它支持同任务环境扰动下的传导式适配评测，不等于留出任务或留出扰动的泛化。需要同时声明任务集合、初始状态、环境变化、训练/评测随机性及最终 checkpoint 的选择规则；不能只写“560条测试”。

| 统计对象 | 可以描述什么 | 不能替代什么 |
| --- | --- | --- |
| 同一最终 checkpoint 的多个评测 seed | 该固定策略对评测随机性的变化 | 多次独立恢复训练的不确定性 |
| 恢复过程中多个 rollout 的直方图 | 不同适配阶段及策略访问分布的混合 | 同一最终模型在固定状态集上的差异 |
| 按任务身份关联选择分数与最终成功 | 两类记录在任务上的关联 | 同一最终评测轨迹中的因果效应 |
| 最终配对的失败→成功和成功→失败 | 净收益和退化集合 | 单凭平均成功率即可保证无退化 |

PIVOT-Q 论文表2称三次 independent runs/seeds；仓库协议却指定训练seed0、同一最终 checkpoint 的四个评测seed。引用时必须保留各自说法，不能把仓库协议当作论文统计的已证实解释。附录 E 的 OpenVLA 另有560个最终结果、558个选择记录，成功率分母与选择散点分母也不能互换。

对同一组 $N$ 个配对 episode，记恢复数为 $n_{F\to S}$、退化数为 $n_{S\to F}$，成功率差是 $(n_{F\to S}-n_{S\to F})/N$。论文 GR00T/QuantVLA 的单 seed 配对实例为 $100\times(101-18)/560\approx14.82$ 个百分点；它与跨 seed 的均值差是不同统计，且明确存在18个退化实例。这里仅复算作者计数，没有开展新的机器人实验。

## 11. 搜索反馈、留出评测与聚合指标要分别定义

方法搜索不只选择参数，也可能选择算法结构。是否“留出”应由**结果是否影响提案、接受或最终选择**判断，而非只看评测是否由另一个进程执行。[QuantForge](../methods/quantforge.md#6.2%20%E5%8F%91%E7%8E%B0%20HiRes%20%E7%9A%84%E8%BF%87%E7%A8%8B%E4%B8%8E%E7%8B%AC%E7%AB%8B%E6%90%9C%E7%B4%A2%E6%AF%94%E8%BE%83) 的 HiRes 开发会收到 Llama-3.1-8B 迁移验证的接受信号；独立搜索框架比较则把同模型的所有结果与搜索隔离。前者属于开发验证信息，后者才支持该实验声明的独立迁移终点。（v1 附录 A.1。）

预算也要明确单位。一次新候选、一个对照和一个定向执行 probe 可以都记为一次 evaluator call，但不保证各自耗时相同。QuantForge 的240次搜索调用另有12次留出观测；真实成本还包括提案调用、token、校准、失败/重试及硬件时间。首次生成达标候选的索引与定期评测确认达标的索引也不同；只对成功 seed 取中位数，会排除未达标运行，不能代替全体运行的成功概率或时间分布。（附录 D.1。）

### 平均与最差表现怎样共同进入分数

QuantForge 先把每模型 $m$、每任务 $j$ 归一到该模型的 BF16 参考：

$$
d_{mj}=
\begin{cases}
\mathrm{PPL}_{q,mj}/\mathrm{PPL}_{16,mj}-1,&\text{PPL任务},\\
1-\mathrm{Acc}_{q,mj}/\mathrm{Acc}_{16,mj},&\text{准确率任务}.
\end{cases}
$$

分母需为正；优于参考时保留负数。Fit-7 取任务维度的均值与最大值各一半，Robust Fit 再对七模型的 Fit-7 作同样聚合：

$$
F_m=\tfrac12\operatorname{mean}_j d_{mj}+\tfrac12\max_j d_{mj},
\qquad
R=\tfrac12\operatorname{mean}_m F_m+\tfrac12\max_m F_m.
$$

它对最差任务、再对最差模型增加权重，不是平均准确率，也不是对全部模型—任务单元一次性取最大值。任务集合、模型集合、参考分数和两层权重都是指标的一部分。（QuantForge §6.1，式12–13。）

教学上，两个任务损害为 $(0,0.3)$ 的方案，均值0.15、均值/最大值混合分0.225；损害为 $(0.18,0.18)$ 的方案均值更差，却以0.18获得更好的混合分。因此某个聚合分改善，不保证每项任务或另一种汇总都改善。论文 Qwen3-8B 的 Fit-7 与五准确率任务平均值也给出这种口径差异；应保留原始终点而非只留一个排名。

对于配对搜索，seed 是完整搜索结果的统计单位。同一轨迹中抽出的多个转移可用于比较局部后继，但存在共享历史；不能把48个转移当成48次独立搜索。是否独立、是否反馈与按何种维度汇总，需要分别报告。

## 12. 生成质量、保真度和提示协议怎样分开评测

生成模型的量化评测至少有三个不同问题：图像是否满足提示/偏好、是否接近参考模型在同输入下的图像、某项专门能力是否保留。CLIP或学习偏好分数、PSNR/SSIM/LPIPS、OCR文字正确率分别提供证据；高参考保真度不保证最高质量，高整体评分也可能遗漏错误文字。（Ideogram INT8/GGUF v2（`https://arxiv.org/abs/2606.12280v2`），§4–7、表1–4；[方法与条件](../methods/ideogram-int8-gguf.md)。）

评测输入应包含实际消费的提示，而不止原始prompt ID。对每条输入保存最终caption、模板/schema、扩写方式、seed、步数、分辨率和采样预设，各量化版本共享同一组输入元组。扩写不仅改变JSON语法，也可能增加内容；若每个版本各自扩写，同一原始问题不再保证相同条件。论文的expander是按公开规格重建，不能称为原厂生产API。

以固定prompt的单图评分 $m_i$ 为例，令 $f_q$、$f_r$ 分别为量化/参考生成模型，N为配对prompt数。相同输入元组 $u_i$ 下，配对差为 $d_i=m_i(f_q(u_i))-m_i(f_r(u_i))$，平均差为 $N^{-1}\sum_i d_i$。LPIPS则消费两幅图：比较两个量化版本时，分别计算各版本图像对同一FP8参考图的LPIPS，再对两个标量相减。按配对prompt行一起重采样，保留同输入关联；若每条prompt仅一个固定seed，所得区间估计固定seed下prompt抽样变化，不能升级成跨seed稳定性。子集指标另报分母：该论文整体评分为300条，机器可检查文字只63条，不能将63条OCR正确率当作全部300条文字正确率。

参考身份也需固定。该案例以已保存FP8权重反量化为BF16的输出作为参考，没有原始BF16 checkpoint；BF16计算精度不改变参考权重已经经过FP8压缩的事实。保真度结果因此支持接近该FP8参考，不能证明原始训练模型完全保留。

CI含零与质量等价按[第8节](#8.%20%E6%97%A0%E6%98%BE%E8%91%97%E5%B7%AE%E5%BC%82%E4%B8%8D%E7%AD%89%E4%BA%8E%E6%BB%A1%E8%B6%B3%E8%B4%A8%E9%87%8F%E7%AD%89%E4%BB%B7)区分；未设置等价界限、未校正多重比较、缺少逐图CI时保留限制。最终配方的质量表也不能替代消融归因：Ideogram的层保护消融在raw提示下完成，JSON协议只评测完整配方，因此保护贡献比例不能直接移植到JSON协议。

## 来源身份

- Qwen3 Technical Report（`https://arxiv.org/abs/2505.09388v1`），`arXiv:2505.09388v1`；§4.6–4.7与附录A.1.1/A.1.2，模式采样、AIME/Codeforces/BFCL、RULER和语言覆盖条件；未核完整manifest/模板/运行。

- DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning（`https://arxiv.org/abs/2501.12948v2`），`arXiv:2501.12948v2`；Experiment Setup、CoT Length及Zero原图，抽样/投票/时间和引用基线条件，未核parser或运行。

- The Llama 3 Herd of Models（`https://arxiv.org/abs/2407.21783v3`），`arXiv:2407.21783v3`；长上下文训练/后训练及评测、§6.2图26；同版源包未编入PDF的FP8补充稿/三图另行注明；未复现训练、质量或reward评分。

- QuantForge: Discovering Residual Decompositions for MXFP4 Post-Training Quantization（`https://arxiv.org/abs/2609.34680v1`），v1；§6.1–6.2、附录 A.1/D.1/D.2，指标聚合、搜索预算及信息反馈边界。

- Perplexity of fixed-length models（`https://huggingface.co/docs/transformers/en/perplexity`），Transformers main，2026-09-22 获取，`perplexity-20260922-03be86a2`；定义、固定窗口与滑动窗口示例。
- Evaluating the Impact of Post-Training Quantization on Reliable VQA with Multimodal LLMs（`https://arxiv.org/abs/2602.13289v1`），arXiv:2602.13289v1；使用 §3 的置信度、评测指标和 held-out 阈值设置，二值公式与算例为整理者解释。

- The Undetected Damage of Quantization on Retrieval and How to Fix It（`https://arxiv.org/abs/2609.24322v1`），arXiv:2609.24322v1；排序稳定性、间隔分配与校准保证。
- The Alignment Illusion in Multimodal Large Language Models（`https://arxiv.org/abs/2609.30210v1`），arXiv:2609.30210v1；主角度诊断与视觉干预。

- Same Bit Width, Different Outcomes: Post-Training Quantization of Text-to-Speech Across Architectures（`https://arxiv.org/abs/2609.28974v1`），v1；组件消融、质量判据与真实执行对照。
- How Divergence Becomes Decision Flips in Compressed Language Models（`https://arxiv.org/abs/2610.00694v1`），arXiv:2610.00694v1；§3–8 的同前缀输出保真、指标聚合和推测解码评测。
- PIVOT-Q（`https://arxiv.org/abs/2610.05745v1`），arXiv:2610.05745v1，§6/表2及附录 E；固定实验协议（`https://github.com/ruanruan-andy/PIVOT-Q/blob/74d6e1f8994a03eb45aae5c316a743abda2cba15/docs/experiment_protocol.md`）：传导式适配、训练/评测 seed 和最终 checkpoint。

- Ideogram INT8/GGUF研究（`https://arxiv.org/abs/2606.12280v2`），arXiv:2606.12280v2；§3–7、表1–4，生成质量/参考保真度/OCR、共享JSON输入、配对统计与消融协议边界。

- Holistic Evaluation of Language Models，`arXiv:2211.09110v2`；BPB、C.2与式3/6–7/12–15，固定提示/计分及分箱和面积边界，具体身份与阅读范围见HELM页。

- LongBench v2，`arXiv:2412.15204v2`；words/token窗口、截断、两阶段CoT和主表/补偿计分，阅读范围与原件差异见协议页。

- MMLU-Pro，`arXiv:2406.01574v6`；实际选项数与论文随机兜底，阅读范围和表/图差异见协议页。

- RULER，`arXiv:2404.06654v3`；任务/长度汇总、有效长度阈值与recall-based计分，原件差异与阅读范围见协议页。

- lm-evaluation-harness，Summer-Summer fork固定`62617a8cbd7f3beddc2a9c029536e0e387976beb`；所选MMLU-Pro配置与共用执行/过滤/聚合，实际静态与夹具范围见实现页。
