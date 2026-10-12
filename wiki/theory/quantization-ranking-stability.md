---
title: 量化后的排序稳定性：间隔、检索损害与精度分配
type: concept
tags:
  - evaluation
  - reliability
  - mixed-precision
sources:
  - raw/papers/2026-09-23/bos-sink-topology/paper.pdf
  - raw/papers/2026-09-23/quantization-retrieval-damage/paper.pdf
  - raw/papers/2026-10-03/decision-flips/paper.pdf
updated: 2026-10-07
---

# 量化后的排序稳定性：间隔、检索损害与精度分配

量化模型与浮点模型的平均任务分数接近，并不表示二者对每个输入返回相同结果。检索尤其容易出现这种差异：候选文档分数接近，小扰动即可交换第一名，而新旧文档可能都在 nDCG 的统计窗口中。本页依据 *The Undetected Damage of Quantization on Retrieval and How to Fix It* v1 的正文、校准协议和理论附录，解释排序稳定条件、部署判别和两种干预。论文研究编码器的 W3/W4 权重量化，不能直接替代 KV 量化或向量索引压缩的评测。

## 1. 三种损害需要不同分母

设同一查询在浮点和量化系统中返回的第一名为 $a$ 和 $a'$：

- **一致性损失**：全部查询中 $a\ne a'$ 的比例，不需要相关性标签。
- **原有正确结果丢失**：在浮点第一名原本相关的查询中，量化后第一名不再相关的比例，需要标签。
- **平均质量变化**：Recall 或 nDCG 的差值；新获得的正确结果会抵消丢失的正确结果。

例如，教学上原来 100 个查询答对 60 个，量化后其中 6 个变错，同时原来错误的查询有 5 个变对，净准确率只下降 1 个百分点，但原有正确结果损失率为 $6/60=10\%$。结果变化也可能只是两个相关文档交换位置，不能把所有 flip 都称为任务失败。是否改变下游生成答案还需要实际 RAG 评测，论文没有验证所有这类后果。

论文的 CLIP ViT-L/14、Flickr30k 4,000 图像、RTN W4/group-128 对照报告：top-1 改变 24.6%，浮点原有正确答案丢失 10.9%，Recall@1 相对下降约 1.3%。这些分母不同，不能相减。Qwen3-Embedding-8B 在四个 BEIR 语料上的相应均值为 nDCG@10 相对下降 3.1%，原有相关 top-1 丢失 15.6%。这是作者结果，没有本地模型复现。（原文“mean quality metric”与 CLIP 实验段。）

## 2. 从分数误差到排序边界

令 $z\in\mathbb R^N$ 为浮点分数，$z'=z+\delta$ 为量化分数，$\|\delta\|_\infty\le\epsilon$。$\epsilon$ 是**最终分数扰动**的界，不是权重舍入步长。原文诊断用实际两模型输出之差计算它，并未从权重量化步长直接推导全网络界。

浮点第一名为 $a$，候选 $j$ 与它的间隔为 $g_j=z_a-z_j$。直接相减得到

$$z'_a-z'_j=g_j+\delta_a-\delta_j\ge g_j-2\epsilon.$$

因此最小间隔，即 top-1/top-2 间隔 $g_2>2\epsilon$，足以保证唯一第一名不变；反之 $g_2<2\epsilon$ 时，可以构造 $\delta_a=-\epsilon,\delta_{c_2}=+\epsilon$ 使第一名翻转。后者是**存在最坏扰动**，不是实际量化必然翻转。

原文以无并列的严格排名定义可达排列，在此约定下用 $g_2\ge2\epsilon$；等号只能把两者推到相同分数，不能严格反超。真实部署若有有限精度并列与 tie-break，等号不能保证所选 ID 不变，所以实现判据应保留严格余量或明确并列规则。top-$k$ 集合的相同论证要检查第 $k$ 与 $k+1$ 名之间的间隔，保护第一名不等于保护前五名。（原文排名定义与理论附录“Boundary cases”。）

定义 contender set 为 $C_\epsilon=\{j\ne a:g_j<2\epsilon\}$。它统计可能反超的候选，但只有真正落在第一名 $2\epsilon$ 范围内的候选才可成为新第一名；“相邻候选间隔很小”的传递链不能让范围外候选越过第一名。原文关于非传递性的文字应在这个限制内理解。

**教学反例。** 对 $z=(1,0.95,0.2)$、$\epsilon=0.04$，第二名可反超，第三名不能；即使扰动能调整排名的其他位置，第三名与第一名的直接间隔仍为 0.8。

## 3. 为什么间隔要与扰动一起看

分离比 $r=g_2/(2\epsilon)$ 无量纲，正比例缩放分数会同时缩放分子分母；共同平移不影响两者。logit 与余弦相似度可以通过此比值进行稳定性分析，但对分数作非线性 softmax 后，实际扰动和间隔都改变，不能沿用原比值。

分类训练通常鼓励正确类领先竞争类；检索训练约束正负样本，却不普遍保证部署语料中最高两个候选之间有大间隔。不是“所有检索损失都不产生间隔”的定理。论文采用相同 checkpoint 的分类/检索对照，并补充 Qwen 线性 probe 对照，减少架构与训练变化的混淆。

在其 W4/group-128 汇总样本中，作者报告超过稳定阈值的分类输入约 85.8%，检索约 3.5%；不同量化器仍有差异，不能据此说优化量化器无用。间隔预测的是浮点一致性，浮点本身是否正确是另一件事。

若额外假定各分数扰动独立、均匀分布于 $[-\epsilon,\epsilon]$，令 $a_j=g_j/(2\epsilon)$，条件于第一名扰动并积分可得

$$P(\text{stable})=\int_0^1\prod_{j\in C_\epsilon}\min(u+a_j,1)\,du.$$

推导中 $u=(\delta_a+\epsilon)/(2\epsilon)$，乘积来自条件独立。这说明候选数量相同、间隔不同也会有不同风险。真实网络误差通常相关，因此这是论文的建模假设，而不是权重量化保证。（理论附录 flip probability。）

## 4. 部署只运行量化模型时怎样判别

诊断时可运行两模型，但部署不能每次先运行浮点再决定是否需要浮点。论文因此在无标签校准集上计算

$$S(x)=\max_j\bigl(\delta_{a'}(x)-\delta_j(x)\bigr),\qquad a'=\arg\max z'(x).$$

共同的分数平移在差分中抵消，$S\le2\epsilon$。若第一名改变，则量化间隔 $g'_2<S$：原来的第一名至少是量化第一名的一个竞争者，二者原始分数差严格不利于 $a'$，因此其最终领先量小于两者扰动差。最大值必须遍历候选，不能只比较原始前两名。

选校准阈值 $\hat\tau$ 后，部署接受 $g'_2\ge\hat\tau$ 的输入。正确的有限样本上分位数构造是：将 $n$ 个 $S$ **升序**排列，取

$$k=\lceil(n+1)(1-\alpha)\rceil,\qquad\hat\tau=S_{(k)};$$

若 $k=n+1$，用 $+\infty$。交换性及固定打分规则下，$P(S_{\rm test}>\hat\tau)\le\alpha$，而“接受且 flip”蕴含这个事件，所以

$$P(\text{accepted}\cap\text{flip})\le\alpha.$$

**原文待纠正处：**PDF 第 4 页把上述第 $k$ 个顺序统计量写成“k-th largest”。按其数值 $k$，这会取近下尾的阈值，与上面的保证不一致。本页给出的是依据顺序统计量的整理者纠正；未获得作者实现，不能判定其实验代码是否同样写反。

这个保证是对校准集与新样本的边际联合概率，不是每个固定校准集都保证，也不是 $P(\text{flip}\mid\text{accepted})\le\alpha$。后者最多由联合界除以覆盖率得到，低覆盖时可很松。论文另报 conditional violation 的实测值，应当作经验结果。模型及打分规则若用同一校准集适配，交换性论证也不能直接套用；宜使用独立的阈值校准集。分布外流量没有同样保证。选择性风险的分母详见 [模型质量评测](../implementation/model-quality-evaluation.md)。

## 5. 检索侧：给真正影响间隔的层多分位宽

论文的步骤是：

1. 在最多 128 个无标签校准查询上，用浮点模型记录 top-1/top-2 文档身份及其差值。
2. 每次仅把一个线性层用 RTN W3/group-128 量化，其余保持浮点，重新计算**同两个文档**的分数差；此差值可以变负，不能重新取量化 top-2。
3. 定义层敏感度 $s_l=\operatorname{median}_x|g_2(x)-g_2^{(l)}(x)|$，按 $s_l/n_l$ 排序，其中 $n_l$ 为参数量。
4. 在 W3 基础上依次把高得分层提升至 W4，直到参数加权位宽预算用完；分配可与 GPTQ、AWQ 等其他量化器组合。

这是单层干预与启发式贪心分配，不是考虑全部跨层相互作用的最优解。3.5 bit 表示参数加权预算，不能在层大小不同时简单量化“一半层”。权重元数据、索引和双模型部署成本仍需另计，见 [混合精度分配](mixed-precision-allocation.md)。

原文将校准计为 $L+1$ 次遍历查询集。对于双编码器，还要核对文档向量是否随量化重新计算；不能把它解释成与语料大小无关的完整成本承诺。校准查询与评测查询隔离，但 GPTQ/AWQ/AdaRound 的校准文档来自索引语料，这个 transductive 条件保留。（原文 allocation 方法与 calibration protocol。）

作者在六种文本 embedding 模型、四个 BEIR 语料、3.5 bit 预算下报告，间隔分配恢复约 60–73% 的 W3→W4 一致性收益；分母为 W3 与 W4 的 top-1 改变率之差，不是最终准确率，也不是吞吐提升。对比是固定量化器下比较分配准则。

## 6. 分类侧：回退比例与可靠性阈值是两种选择

分类中，先运行量化模型，间隔低于阈值的输入再运行浮点。论文的修复实验用量化间隔第 25 百分位作为阈值，目标是回退约四分之一输入；这与上节 conformal 阈值的概率控制目标不同。

作者在浮点→量化准确率损失至少 0.5 个百分点的任务上报告，回退 25% 恢复约 85–93% 的损失。其“47% 浮点计算成本”采用外部 GPTQ 的 4.5 倍速度进行估算：$1/4.5+0.25\approx0.472$。不是本系统实测延迟，未计双模型驻留、路由、排队和批处理变化。

## 7. 可复用结论与边界

评测应同时保留质量、一致性、原有正确结果损失和实际资源代价。用 [已有 VQA 可靠性](quantization-reliability-and-selective-prediction.md) 的置信度/弃答指标解释检索是不充分的，两者关心的事件不同。

本文支持 W3/W4 权重 PTQ、有限语料精确相似度检索中的观察。未验证激活量化、ANN 索引误差、实际 RAG 质量、漂移下概率保证或本地性能。正文与必要理论/协议附录已读，关键阈值回查 PDF；未逐项重算全部附录表格，也未取得论文声称的补充代码。

这个边界论证也适用于 [attention sink 选头图谱](attention-sink-map-transfer.md)：全局 Spearman 很高，top-k 成员仍可能变化。但 Jaccard 是交并比，不是成员保留率，层内归一化距离也不保留绝对 attention 总量。排序变化只是诊断信号；没有下游任务或干预证据，不能把它直接称为任务损失。

## 8. 语言模型的分布距离怎样变成 token 翻转

*How Divergence Becomes Decision Flips in Compressed Language Models* v1 将排序问题放到下一个 token 的完整输出分布。它比较浮点参考与压缩模型在**同一个前缀**下是否选择同一 token，不测量两条自由生成轨迹逐字对齐，也不把浮点选择当成正确答案。以下依据 §3–8、必要协议与推导附录；未核读作者实现或运行模型。

### 8.1 比较对象与三个指标

令位置 t 的浮点/压缩分布为 $p_t,q_t$，两者都 teacher-force 同一序列。序列可以来自浮点 greedy continuation，也可以是自然文本。以位置等权平均，使用自然对数：

$$
D_t=\sum_vp_t(v)\log\frac{p_t(v)}{q_t(v)},\qquad
V_t=\frac12\sum_v|p_t(v)-q_t(v)|,
$$
$$
\overline D=\mathbb E_tD_t,\quad
\overline V=\mathbb E_tV_t,\quad
f=\mathbb E_t\mathbf1[\arg\max p_t\ne\arg\max q_t].
$$

f 是 greedy top-1 改变率。$\overline D$ 是 dense→compressed 的 forward KL，$\overline V$ 是平均 total variation；它们不等于目标 token 的 NLL 或任务正确率。本文所说 flip 不含“原来正确、现在错误”的额外标签条件，分母区别与 §1 一致。（原文 §3、式 (1)–(4)。）

### 8.2 为什么会观察到平方根，而不是严格的指数定理

在温度 1 下，设压缩造成小 logit 位移 $\Delta z_t$。softmax 输出的 KL 局部展开为

$$D_t=\frac12\operatorname{Var}_{p_t}(\Delta z_t)+O(\|\Delta z_t\|^3).$$

KL 是二阶量；原 top-1/top-2 的 logit 间隔变化 $\Delta g_t=\Delta z_t(i_1)-\Delta z_t(i_2)$ 则是一阶量。若很多位置的间隔接近零，小扰动会让其中一部分跨越边界，因此 flip 计数可随位移幅度一阶增加，形成 $f\approx\kappa\sqrt{\overline D}$。

这个计数还需要零附近间隔密度、扰动正负近似对称及其与间隔的依赖足够温和。更低位宽的高阶项、第三名反超、有限精度并列与 flip 饱和都会破坏简化比例。论文测得指数接近而不精确等于 1/2；约 0.25 nats 以后曲线会弯折，2 bit 权重在其额外极端试验中已接近全翻转，不能延长幂律。（§4、附录 A.2。）

### 8.3 先平均再开方，会引入语料相关系数

只要各分母非零，以下分解是恒等式：

$$
\frac f{\sqrt{\overline D}}
=\frac f{\overline V}\,
\underbrace{\frac{\mathbb E_tV_t}{\mathbb E_t\sqrt{D_t}}}_{c}\,
\underbrace{\frac{\mathbb E_t\sqrt{D_t}}{\sqrt{\mathbb E_tD_t}}}_{J}.
$$

由 Pinsker，$V_t\le\sqrt{D_t/2}$，所以 $c\le1/\sqrt2$；由 Jensen，$J\le1$，D_t 相同时取 1。J 反映误差是否集中在少数 token，不能仅从平均 KL 恢复。

**教学例子：**若 100 个位置都为 $D_t=0.01$，平均 KL 为 0.01、$J=1$；若一个位置为 1、其余为零，平均 KL 仍为 0.01，但 $J=0.1$。这里比较的是聚合方式，不是构造两个模型证明 flip 已不同。论文表 2 的 42 个 FP32-head references 中，GSM8K 的 J 中位数为 0.38，自然 WikiText 为 0.75；解释了为什么跨语料不能直接搬用同一个 $\kappa$。（§5、式 (7)、表 2。）

平均每位置 $\sqrt{D_t}$ 可消去这一 Jensen 因子，但还没有消去 $c$ 或决策几何的条件。低平均 KL 仍可能把错误集中在少数关键位置。

### 8.4 TV 的严格界与近一经验比是两件事

令浮点概率间隔 $\delta_t=p_t(i_1)-p_t(i_2)$。若 greedy 选择改变，则

$$V_t\ge\delta_t/2.$$

证明很直接：新第一名 j 的 $q_t(j)\ge q_t(i_1)$，原领先间隔只能靠 i1 丢失概率与 j 获得概率填平；各项都不超过 TV。设 $F(h)$ 是 $\delta_t<h$ 的位置比例，对任意 h>0 分开小间隔位置与其余位置，再用 Markov 得到

$$f\le F(h)+2\overline V/h.$$

这个分布无关界没有推出 $f\approx\overline V$。**教学反例：**单个位置 $p=(0.5001,0.4999)$、$q=(0.4999,0.5001)$，TV 仅 0.0002，greedy 已翻转；$p=(0.9,0.1)$、$q=(0.6,0.4)$ 的 TV 为 0.3，却没有翻转。单个微小间隔与一次大概率移动都足以让两者比例偏离一。（§5、引理 1；附录 A.2。）

在论文主样本的 802 个配置中，$f/\overline V$ 中位数为 1.049，95% 配置在 0.89–1.29。它是一个经验集中范围，既不是置信区间，也不是所有压缩器的保证。论文用有限 token 集、各坐标独立高斯 logit 噪声及零附近连续的 top-2 间隔密度解释小噪声极限：

$$
\lim_{\sigma\to0}\frac f{\overline V}
=\frac{\sqrt2\rho_g(0)}{\mathbb E_th(p_t)},\qquad
h(p)=\sum_i p_i\sqrt{1-2p_i+\sum_jp_j^2}.
$$

分子是靠近决策边界的密度，分母反映概率质量的移动；没有理由对任意分布都正好相等。真实压缩器只在所测方向与这一噪声近似相似，命题 1 不直接覆盖所有有限量化误差。

另有一个容易误读的温度检验：同时把两模型 logits 除以温度 $\vartheta>0$，不改变 greedy ID，却改变 TV。唯一 top-1 时，$\vartheta\to0$ 使两边分布都变成 one-hot，TV 逐位置趋于 flip 指示量，因此非零 f 时比例趋于一。这不是在温度 1 下的普遍等式；并列与 tie-break 仍要单独控制。（附录 A.3。）

### 8.5 原文覆盖、留出与失败条件

主研究为 19 个 checkpoint、6 个模型家族，802 个配置，54 个模型/语料 references；9 种扰动族包括权重量化、剪枝、KV 量化、删层和噪声。主要在 H100/BF16 上模拟压缩，数据设计不平衡：45% 配置来自开发 mix，RTN 与其单层变体占 35%。统计拟合以 reference 或 model 聚类，不把同一 reference 内配置当成独立样本。（§3。）

| 证据 | 论文结果 | 应保留的条件 |
| --- | --- | --- |
| 固定 reference 的比较，§4–5 | 多数扰动族共享近平方根曲线；逐 reference 标定后，TV/√KL 留出配置的中位相对误差为 1.6%/2.4% | 标定系数属于模型与语料；不能说 KL 在所有比较中无用 |
| 留出 Python/MBPP 域，§6 | 8 个已见模型的指数为 0.48–0.57，flip/TV 为 0.98–1.05 | 新语料而非新模型；KL 的若干系数/族偏移预注册预测失败 |
| 新模型与真实 kernel，§6 | Mistral-Nemo 的 code 比例中位数为 0.91，Granite 为 0.987，存在低于一的情况 | 三个新模型；跨语料 TV 预测误差也可超过预注册 10% |
| MoE/QAT 扩展，§6、附录 A.6 | 小范围扩展仍见主要规律 | 事后扩展；两个 MoE 的 router 保持 BF16；不能外推所有 MoE/QAT 配方 |

BF16 logits 的离散网格会造成精确并列，主参考的 top-1/top-2 并列率中位数 1.16%；间隔分析另用 42 个 FP32 output-head references。把并列当成连续零附近密度会使推导失效。部分生成原始文本的模型会循环，作者保留自然文本参考并筛查退化生成；同前缀并不能替代语料质量检查。

散度使用 dense top-64 token 加聚合 tail，聚合会低估完整 TV；作者核对的中位低估很小，TV 的 95th-percentile 放大因子仍为 1.07。原始非有限 logits 无法计算这些指标，真实 FP8 的 12 次试验中 10 次失败，论文据此明确 FP8 未验证。本文不将排除失败的统计当作 FP8 支持证据。（§3、§6、附录 A.1/A.7。）

### 8.6 一致性何时有用，何时必须另测任务

greedy 推测解码需要 draft 与 target 一致；采样版在同一状态下的标准拒绝采样接受率是 $1-\mathrm{TV}(p,q)$。后者是逐状态的采样恒等式，前者的 greedy 一致率却不能直接替换成 1−TV。估计整轮接受还要考虑错误沿序列聚集；论文的独立轮模型对全部 43 个 target/draft 配对都高估接受率，保留 teacher-forced 一致序列后预测更好。（§7、表 3。）

真实 vLLM 两个 Qwen3 target 的评测中，TV 预测 greedy 接受率的平均相对误差为 1.1–2.4%，但按权重字节近似计算成本时，TV 选出的 draft 仅达到最佳实测加速的 85–100%。一致性还没有计入 draft kernel、验证批次和调度成本，不能据此选最快模型。

最终任务则需要别的证据：论文 §8 在 Gemma-3-4B、相近 GSM8K KL 与 flip 下，magnitude pruning 损失 33.3 个准确率百分点，Wanda 却增加 4.8 个百分点。同样数量的决策改变，其方向与后果可以完全不同；teacher forcing 也没有覆盖早期改变对未来状态的反馈。评测应直接报告任务内 flip，并配合[模型质量协议](../implementation/model-quality-evaluation.md#9-输出分布一致性要固定前缀和指标口径)。

## 来源身份

- The Undetected Damage of Quantization on Retrieval and How to Fix It（`https://arxiv.org/abs/2609.24322v1`），arXiv:2609.24322v1；2026-09-23 归档。作者报告、排序推导与本页纠正分别标注。

- Global Ranks Survive, Selected Heads Shift（`https://arxiv.org/abs/2609.23585v1`），v1；全局排序、top-k 集合及校准图谱的区别。
- How Divergence Becomes Decision Flips in Compressed Language Models（`https://arxiv.org/abs/2610.00694v1`），arXiv:2610.00694v1；§3–8、附录 A 的定义、分解、概率间隔界、噪声条件与评测范围。§8 的离散分布与 Jensen 数值例子是整理者教学推导，没有模型实验。
