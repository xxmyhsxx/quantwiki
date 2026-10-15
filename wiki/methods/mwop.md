---
title: MWOP：按模态剪注意力路径与FFN执行宽度
type: method
tags:
  - vlm
  - pruning
  - attention
  - optimization
sources:
  - raw/papers/2026-10-03/mwop
updated: 2026-10-15
---

# MWOP：按模态剪注意力路径与FFN执行宽度

MWOP保留视觉/文字token集合，减少每个位置实际执行的attention交互和FFN通道。它按头选择模态路径，按模态选择共享FFN的执行通道，再用LoRA恢复质量。剪的是计算宽度；投影维度、KV宽度和参数存储不会因此按相同比例下降。

本页依据arXiv:2610.01434v1，主文与必要附录的选择、恢复、执行和测量说明；作者代码链接没有归档，本页的kernel行为属于论文描述，未核源码或GPU。序列剪枝、位置与量化对象复用[视觉token公共解释](../fundamentals/model/vision-language-model-tokens-and-quantization.md)。

## 1. 一颗attention head包含不同的模态交互

把query模态写在前、key/value模态写在后：

| 路径 | 实际读取关系 | 论文记号 |
| --- | --- | --- |
| VV | 视觉query读取视觉KV | V2V |
| TV | 文字query读取视觉KV | T2V |
| TT | 文字query读取文字KV | T2T |

仅在视觉先于用户文字且使用因果mask的简化序列中，视觉query不能读取未来用户文字，第四块VT为零。System及视觉之前的文字仍须按实际序列处理。这里T2V按query/KV标签命名，不能反过来解释为视觉query读文字。

每个层—头—路径是独立选择单元：一颗头的VV可删除，TV/TT仍保留。稀疏执行跳过被删的QK与AV乘法，但仍保留Q/K/V/O投影、GQA映射和KV cache宽度；不能按删路径比例计算权重或缓存节省。（§3、附录D.5。）

## 2. Taylor代理衡量贡献门控，不包含重归一化

令一条路径的softmax后贡献为 $O_b=A_bV$，完整头输出为 $O$。固定原attention权重，门控 $g_bO_b$从 $g_b=1$改到0，给出一阶损失变化代理。论文对teacher-forced答案交叉熵使用

$$S_{\ell,h,b}=
\frac1{|\mathcal D_{\mathrm{cal}}|}\sum_{x}
\frac{\left|\left\langle O_{\ell,h,b}(x),
\nabla_{O_{\ell,h}(x)}\mathcal L_x\right\rangle\right|}
{|\mathcal Q_b(x)|}.$$

内积先跨query/维度求和，再取绝对值，最后按样本平均；它不是逐元素绝对梯度均值。该分析单独去掉某块加法贡献，保留权重不重归一化。

实际推理却在**softmax前**屏蔽被删路径，保留因果key集合 $\mathcal A(i)$后计算

$$o_i=\frac{\sum_{j\in\mathcal A(i)}e^{s_{ij}}v_j}
{\sum_{j\in\mathcal A(i)}e^{s_{ij}}}.$$

这还改变分母，因此Taylor分数没有精确包含实际删路径的全部效应。论文描述的Triton kernel只遍历保留key区间，跨区间共享一个online softmax状态；最终质量仍由实际mask与恢复后评测决定。（附录A.1、D.5。）

## 3. 共享FFN按输入模态执行不同通道

对模态 $\mu$的token，保留中间通道集合 $\mathcal I_\mu$，只算匹配的gate/up行、非线性与down列，输出仍为原hidden宽度：

$$\widehat{\mathrm{FFN}}_\mu(x)=
\Phi_{\mathcal I_\mu}(x)\,W_{\mathrm{out}}[\mathcal I_\mu,:].$$

共享权重不要求 $\mathcal I_V=\mathcal I_T$。FFN评分对本模态token的中间激活×答案loss梯度先求和、取绝对值，再跨样本平均。Attention剪后会改变FFN输入，所以固定attention mask后要重算两模态分数。

实际主配置让视觉位置使用compact子权重，文字位置使用完整FFN；空视觉通道集意味着零FFN更新，残差和token仍在。完整文字权重与compact视觉权重同时保存还会增加存储。LLaVA配置“50%视觉剪枝”是跨层全局执行预算，并非每层都减半；文字通道全保留。（§5、附录B.1、D.2/D.5。）

## 4. 搜索传递mask，每个恢复候选从同一权重开始

流程顺序为VV→TV→TT→视觉FFN→文字FFN。Attention先在初始模型 $M_0$评分；各阶段用0、0.1…1比例网格在1000样本selection set作无训练筛选，再为候选执行恢复。进入FFN阶段时，在 $M_0$上施加已选attention mask，重算两模态通道分数一次。

跨阶段保留的是累计mask；每个候选都从 $M_0+$当前mask独立恢复一epoch，不继承上一阶段LoRA权重。LLaVA恢复数据282,293条，训练projector与LoRA（$r=128,\alpha=256$），视觉encoder和基础语言权重冻结。完整benchmark也参与anchor及最终点选择，因此这些报告不是完全留出的单次测试。（附录D.3、Algorithm1、D.6–D.7。）

“All Tasks” attention校准为四任务类各256样本，按 $r(u)=\min_c r_c(u)$合排名（1为最重要），保护任一类需要的单元；它与单类256样本的预算不同。FFN评分准则对照另用跨任务平均排名，不能把所有排名实验都概括为同一个min-rank操作。（附录A.5/B.1。）

## 5. 任务保留率与decoder prefill速度各有分母

LLaVA-OneVision-7B的单独MWOP配置，作者报告56.8%理论decoder FLOPs、99.7% ATR及237.0→152.1ms prefill；Qwen2.5-VL-7B为67.4%、99.5%及49.8→38.8ms。ATR先对每benchmark分数除以恢复后dense参考，再在四任务类内/类间等权归约；99.7%不是样本答对率，主表参考也不同于训练前分析。（§6、Table3–5、附录C/D。）

速度实验为A100-SXM4-40GB、batch1 BF16 decoder prefill，匹配长度的synthetic embedding-table输入，dense用FlashAttention-2。编译、权重pack、调优与Graph捕获在计时前完成，CUDA Events测重放；五个window均值再取中位。视觉encoder、生成decode与在线token选择不在该时间内，单独attention/FFN计时之和也不自动等于独立测得的total。

理论FLOPs保留投影、有效交互、FFN及last-token head等项，排除norm/RoPE/softmax和搬运；在线剪枝开销另算。与token剪枝组合的最大约2.9×属于另一配置，不能代替单独MWOP约1.6×的结果。较少交互、较窄FFN与更短序列是可组合的不同操作，仍需分别核实际长度、选择开销和质量。

## 来源身份

| 来源 | 固定版本 | 核读与使用范围 |
| --- | --- | --- |
| MWOP: Modality-aware Width-wise Operation Pruning for Efficient MLLMs（`https://arxiv.org/abs/2610.01434v1`） | `arXiv:2610.01434v1` | 完整主文叙述、表头/定义与主要结果；必要附录注意力/FFN评分、校准、算法1、配置/恢复与稀疏推理/计时，PDF4–11/30/31/33关键原件核读。未逐格复核全部附表单任务值、外链作者代码或GPU；贡献门控代理与推理归一化的差异明确保留。 |
