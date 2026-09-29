---
title: AKVQ-VL：按文本与枢纽 token 保护多模态 KV
type: method
tags:
  - vlm
  - kv-cache
  - mixed-precision
  - rotation
sources:
  - raw/papers/2026-09-21/akvq-vl/paper.pdf
updated: 2026-09-29
---

# AKVQ-VL：按文本与枢纽 token 保护多模态 KV

AKVQ-VL 把多模态 attention 的层间差异转成 KV 精度策略：文本显著层保护文本，枢纽显著层保护少量 pivot，同时保护近期 token；其余历史 KV 采用经过 Hadamard 变换的 2-bit 量化。它保留全部 token，和视觉 token 剪枝解决不同的问题。

本页研读 v1 全部 7 页正文与图表。没有可核验实现或模型运行结果；pivot 检测和缓存组织中原文未给足的细节保留为复现缺口。前置见 [视觉 token 与量化](../fundamentals/model/vision-language-model-tokens-and-quantization.md)、[旋转与 Hadamard](../theory/orthogonal-rotation-and-hadamard-quantization.md)。

## 1. TSA / PSA 是观测模式，不是所有 VLM 的固定层规律

论文用 MileBench 多图 prompt 分析 attention，区分三个现象：近期 token 的局部性；早期部分层对文本的较高平均注意力，即 Text-Salient Attention（TSA）；后续层少量位置吸收大量注意力，即 Pivot-Token-Salient Attention（PSA）。与只保护开头 sink 的规则相比，VLM 的 pivot 还可位于视觉序列内部。

图 4 对每模态 token 的平均 attention 做比较，排除前五个 token 以减弱 sink 影响；LLM 对照使用相同位置分组，那里没有真正的“视觉 token”。平均每 token 分数高，不等于该模态总 attention 质量一定更高，也不构成其每个 token 都不可量化的证明。

表 I 中 LLaVA-v1.5-7B、v1.6-vicuna-7B 的 0–1 层列为 TSA，2–31 为 PSA；v1.6-mistral-7B 没有 TSA，0–31 都列为 PSA。Qwen2-VL 出现在模式表中，但没有进入表 II 的质量对照。v1.5-13B 一行也写到 31，论文没有解释是否仅统计部分层，因此不要由此生成该模型全部层的执行配置。

## 2. 保护对象如何变成混合精度

| 层的模式 | token 类别 | KV 精度 |
| --- | --- | --- |
| TSA | recent | 原始 16 bit |
| TSA | 其余文本 | 4 bit |
| TSA | 其余视觉 | 2 bit |
| PSA | pivot 与 recent 的并集 | 原始 16 bit |
| PSA | 其他历史 token | 2 bit |

这是按图 2 和 §III-B/C 综合还原的策略；不要把所有层的文本一律设成 4 bit。recent 与 pivot 重合时只计一次。实验取 recent=128、pivot=15，2-bit clipping ratio=0.8，4-bit=1.0，量化组大小 128。

TSA 可直接利用模态索引，无须在运行时取出完整 attention matrix。PSA 则利用 Transformer 残差输出中的 massive activations 作为 pivot 的代理：检测大幅值激活所在 token，将位置用于后续 attention 层。它与“按当前 attention TopK”不同，适配不显式返回分数的 attention 路径。

但原文没有明确给出 massive activation 的阈值或排序公式、跨 channel/head 聚合、在哪一层探测和何时刷新、如何处理已量化后才被识别的 pivot。15 是保留数量配置，不能擅自补成“取某个 L2 范数 Top15”的作者算法。对新架构复现时，这些都需要作者代码或进一步实验。

## 3. 动态非对称量化与 clipping 的边界

对一个 token 的量化组，令截断范围为 $[a,c]$，$n$ 为位宽，原文公式为

$$s=\frac{c-a}{2^n-1},\qquad z=-\operatorname{round}(a/s),$$
$$q=\operatorname{clip}(\operatorname{round}(x/s)+z,0,2^n-1),\qquad\widehat x=s(q-z).$$

这里 $q$ 是整数码，$z$ 是整数零点，$\widehat x$ 才是重构值。clipping 缩小范围可减小网格间距，但会增加尾部截断误差；2 bit 仅四个码点，对该取舍更敏感。论文报告 ratio，却没有充分定义两个截断端点怎样由 ratio 得到，不能默认就是绝对值对称 clipping 或分位点 clipping。scale/zero 的存储 dtype、零范围组与打包布局也未明确。

## 4. Hadamard 在什么位置保持等价

设归一化 WHT 矩阵 $H$ 满足 $HH^\top=I$，维度为二次幂。对 RoPE 后的 query/key 做相同变换：

$$(QH)(KH)^\top=QK^\top.$$

因此能把 K 的单通道尖峰分散到多个通道，同时保持未量化 attention logits。必须在 RoPE 后使用这条直接等价式；一般不能交换位置旋转与任意 $H$。WHT 的快速实现约需 $O(d\log d)$ 操作，Q/K 在线变换有成本，不能因为代数等价就说没有运行开销。

V 不经过 RoPE，可以把 $W_V'=W_VH$ 和 $W_O'=H^\top W_O$ 离线融合：

$$S(XW_VH)(H^\top W_O)=SXW_VW_O.$$

多 head 时这是匹配 head 布局的块变换，GQA 的共享 KV/head 映射也要一致。量化后不再严格等价，旋转的意义是改善量化网格面对的分布。

原文称“outlier-free”，更准确的可复用结论是所测 K 的极值显著降低，而非确定性 WHT 对所有输入都消除离群值。反例是 $x=\mathbf1$，其 WHT 会把能量集中到一个坐标，最大值变为 $\sqrt d$；是否改善取决于原始结构。图 5 的观测不能升级成普遍保证。

## 5. 不能把它按纯 2 BPE 计费

设某层 $T$ 个 token，去重后 16-bit 数量为 $n_{16}$、4-bit 数量为 $n_4$，只算 payload 就有

$$b_{\rm payload}=2+14\frac{n_{16}}T+2\frac{n_4}T.$$

还要加量化组的 scale/zero、pivot 索引、混合精度布局与分配开销。PSA 中如果 128 recent 与 15 pivot 完全不重合，$T=4096$ 时仅 payload 就约 2.489 BPE，$T=500$ 时约 6.004 BPE；这是预算示例，不是原文实测。recent 和固定数量 pivot 的比例随长上下文下降，但 TSA 文本数量会随生成增长，因此文本 4-bit 的开销不能一律忽略。

这也说明“16 到 2 bit”不等于总显存缩小 8 倍。模型权重、视觉编码器、临时 workspace 和保护带都还存在；短 prompt、长生成时的缓存组成又不同。

## 6. 实验与可归因范围

表 II 评测四种 LLaVA、MileBench 的 12 个任务，覆盖 MHA 与一种 GQA 模型。多数配置明显优于该文的 RTN/KIVI/SKVQ 对照，但不是逐项无损。例如 v1.5-7B 的 Webpage QA 从 FP16 59.0 到 57.0，v1.5-13B 的 Textbook QA 从 55.0 到 51.0；v1.5-7B 的 Scene Transition 则从 73.0 到 78.0。没有重复运行置信区间，不能把后者解释为可靠的量化泛化增益。

Scene Transition 的递增加组件消融为 RTN 7.5 → WHT 20.5 → 再加 TSA 41.0 → 再加 PSA 78.0。它说明整套组合在这个任务有效；不是完整因子消融，不能宣称 PSA 在任意模型独立贡献 37 点。clipping、recent、pivot 数量也主要在 Document/Webpage QA 上选择，迁移到其他任务需要重验。

SmoothQuant 对照在本实验中被用于 K/V 缩放、$\alpha=0.5$；不应等同于其原始完整激活量化配方。KIVI residual 和 SKVQ window 都设为 128；即便标称均为 2 bit，不同保护集合和粒度仍使有效预算不同。

效率实验是 **4×V100 16GB、LLaVA-v1.5-7B、约 500 输入 token**，逐渐增大 batch 至 OOM。表 IV 的 FP16 batch 40 为 1028.38 token/s，AKVQ batch 130 为 2526.37 token/s，比例约 2.46，允许 batch 比例 3.25。它证明接近相同总显存限制时能服务更多并发，不能读成相同 batch 的单请求延迟改善 2.46×。两行峰值显存约 63GB 是不同 batch 的比较；文中 2.13× 显存改善来自图 9 的批量曲线，不能用表 IV 两行相除。原文未给足输出长度和完整计时边界，不能直接与 NOVA 的长上下文 decode-only 数字比较。

## 7. 与已有知识的关系

[KVQuant](kvquant.md)保护开头 sink 并处理通道离群；AKVQ-VL 的增量是模态/层相关的保护以及非开头 pivot。[attention-sink-map-transfer](../theory/attention-sink-map-transfer.md)讨论的 sink 拓扑提醒我们：attention 集中、激活异常和任务重要性有关联，但不能混为同一个指标。

与 [QAPruner](qapruner.md) 的区别是，AKVQ-VL 分配 KV 精度而不删除序列位置；与 [CoRePrune](coreprune.md) / [LayerPos](layerpos.md) 组合时还需重新验证 pivot 位置、recent 定义和位置编码，原文没有证明这些方法可以直接叠加。

## 来源身份

- [AKVQ-VL，arXiv:2501.15021v1](https://arxiv.org/abs/2501.15021v1)：§III 与图 2–7 支撑机制；§IV、表 II–IV、图 8–9 支撑结果。预算公式和 WHT 反例是整理者推导；检测实现细节保留为未确认。
