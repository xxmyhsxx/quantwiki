---
title: 视觉语言模型中的 token 与量化对象
type: concept
tags:
  - vlm
  - calibration
  - weight-quantization
  - activation-quantization
sources:
  - raw/papers/2026-09-21/splitq/paper.pdf
  - raw/papers/2026-09-21/masquant/paper.pdf
  - raw/papers/2026-09-21/q-vlm/paper.pdf
  - raw/papers/2026-09-21/mquant/paper.pdf
  - raw/papers/2026-09-21/mbq/paper.pdf
  - raw/papers/2026-09-21/vlmq/paper.pdf
  - raw/repositories/2026-09-21/mbq/source/qmllm/models/qwen2_vl/qwen2_vl.py
updated: 2026-09-15
---

# 视觉语言模型中的 token 与量化对象

视觉语言模型的量化依然使用共同的数值表示、舍入、裁剪、误差重构和低比特计算基础。新的问题来自图像与文本怎样进入计算图、哪些位置受到什么损失监督，以及最终需要完成什么任务。本页以 MBQ v2、VLMQ v2、MQuant v2 所研究的视觉编码器—连接器—语言骨干结构为范围，不代表所有多模态架构。

## 1. 图像如何成为语言骨干的输入

一个简化计算链是：

$$
I\xrightarrow{\text{视觉编码器}}V\in\mathbb R^{N_v\times d_v}
\xrightarrow{\text{连接器}}Z_v\in\mathbb R^{N_v'\times d},
\qquad
(t_1,\ldots,t_{N_t})\xrightarrow{\text{文本嵌入}}Z_t\in\mathbb R^{N_t\times d}.
$$

按模型模板将 $Z_v$ 与 $Z_t$ 组织成序列，再由语言骨干生成回答。连接器负责特征维度或序列组织的转换，可能是线性/MLP，也可能包含重采样等结构；$N_v'$ 不一定等于视觉编码器的原始 token 数。（MBQ 补充 §7；VLMQ 附录 B）

这里的 token 是序列中的一个向量位置。视觉 token 不是词，也不必对应单一、固定大小的原图区域；图像切分、分辨率、裁块与连接器都会影响数量。文字 token 也不等于自然语言的一个字或词。因此实验需同时记录图像预处理与展开后的序列长度。

“模态不同”不等于语言骨干为它们分别使用不同权重。进入共享层后，同一个线性变换会作用于各位置；这里的对象和通道关系见 [线性层与输入通道](../operators/linear-layer-input-channel.md)。经过注意力混合，一个视觉位置的表示也可能包含上下文信息，位置标签不是深层特征语义的严格分解。

## 2. 量化对象不只是一份模型名称

| 部分/张量 | 量化问题 | 本轮证据范围 |
|---|---|---|
| 视觉编码器的权重与激活 | 图像特征与分辨率变化怎样影响误差 | MBQ 表 6 有独立视觉编码器量化消融；不能据此推广全部视觉编码器 |
| 连接器 | 视觉信息如何进入语言隐藏空间，误差如何传入骨干 | MQuant 算法 1 将其纳入量化，但未提供独立连接器收益消融 |
| 语言骨干权重 | 一套共享权重应怎样兼顾不同 token 的输出误差 | [MBQ](../../methods/mbq.md) 与 [VLMQ](../../methods/vlmq.md) 的主要研究对象 |
| 语言骨干激活 | 动态范围、离群值、粒度和运行时量化 | MBQ 有 W4A8，VLMQ 主实验是 weight-only，MQuant 使用分模态静态网格；不能混为同一设置 |
| KV cache | 随序列缓存的键值状态，误差和显存随长度变化 | 两篇没有给出专门 KV cache 算法，不能把权重量化成果当作 KV 量化成果 |

KV cache 作为量化对象的键值差异、粒度选择、流式约束与系统限制见 [KV cache 量化的对象与粒度](../../theory/kv-cache-quantization-objects-and-granularity.md)；该页不区分模态，视觉 token 与文本 token 共用同一套缓存结构。

W3A16 描述权重与激活位宽，不完整描述视觉编码器、连接器、KV cache 和未量化算子。[均匀量化与分组](../quantization/uniform-quantization-and-groups.md) 定义的 scale、zero-point 与 group 仍需在每种对象上明确指定。

[Q-VLM](../../methods/q-vlm.md) §3.3 把视觉端量化作为改变后续语言层分布的一种方式，目标是减轻联合校准负担。这说明视觉端误差不仅影响视觉表示本身，还会影响语言层的校准目标与成本；但论文设计、视觉权重是否更新，以及公开代码是否启用视觉量化，应分别核对。

## 3. 为什么 token 数量会影响校准

假设校准重构对所有 token 累积平方误差：

$$
J=\sum_{n=1}^{N}\|\widehat y_n-y_n\|_2^2
=\sum_{n\in v}\|D_n\|_2^2+\sum_{n\in t}\|D_n\|_2^2.
$$

若视觉位置有 100 个、回答位置只有 10 个，且每个位置的误差大小相同，那么视觉组对这个目标的贡献约是回答组的 10 倍。但这些重构误差未必按相同倍数影响回答质量；目标中 token 的数量优势与任务上的重要性是两件事。

由此产生三个不同问题：

- **表示差异：** 两组激活的幅度和方向分布是否不同？它决定量化范围及二阶统计可能怎样变化。
- **数量差异：** 相同归约规则下，哪组被计入更多次？它取决于 mask、序列长度、求和或均值。
- **敏感性差异：** 相同扰动对指定目标造成多大影响？它取决于任务、层、梯度方向和扰动大小。

三者不能互相替代。PCA 中分开不等于对任务不重要；数量多不证明全都冗余；平均梯度小也不意味着可以任意删去这些 token。（MBQ 图 1、§3；VLMQ 图 2–4、§3–4）

[MBQ](../../methods/mbq.md) 用模态组的梯度统计调节缩放搜索目标；[VLMQ](../../methods/vlmq.md) 用局部模块梯度构造 token 加权二阶重构。它们都是修改代理目标的例子，不是视觉 token 剪枝方法。更通用的目标区别见 [层输出重构与二阶误差补偿](../../theory/layer-reconstruction-second-order-compensation.md)。

## 4. 校准序列、回答标签与 mask

图文校准样本可能包含“图像 + 用户提示 + 已知描述/回答”。在教师强制的自回归损失中，用当前位置预测下一个回答 token；提示和图像位置可不直接计入交叉熵，但会通过注意力影响回答，因而仍可能收到梯度。

例如序列为 `[图像位置，提示，回答1，回答2]`：`labels != -100` 标的是被保留的回答标签位置，计算下一 token 损失时还有移位。不能直接认为“loss 的非零位置”“取重构误差的回答位置”“最终影响回答的所有位置”是同一个集合。

MBQ 固定代码 a4d460df 的 Qwen2-VL 接入分别构造 `vision_mask` 和 `caption_mask`，后者来自回答标签；VLMQ 按模型模板构造校准样本，并丢弃视觉段被截断的样本。由此需要区分 padding mask、视觉位置 mask、回答监督 mask；更新图像展开方式后，不能沿用未对齐的 token 下标。（MBQ `qwen2_vl.py` 的标签与 mask 处理；VLMQ 附录 E.1）

[校准数据与量化范围选择](../../theory/calibration-and-range-selection.md) 在这里需额外记录：图文是否配对、描述来自何处、视觉预处理、模板、回答是否参与输入、各 mask、截断规则以及实际有效 token 数。使用回答进行离线校准本身不等于泄漏；关键是回答的来源与测试集是否独立，以及是否根据测试结果反复选择配置。

物理存储位置与语义位置也要区分：[MQuant 的 AIFS](../../methods/mquant.md) 可以把视觉 token 集中到序列前部，但必须携带原 position IDs，并把注意力 mask 的行列同时置换。否则普通下三角 mask 会改变谁能看到谁。该例说明，布局改变可以保持计算等价，条件是因果关系、padding 与 cache 映射一起维护。（MQuant v2，§3.1、附录 A.2/A.4）

## 5. 为什么不能把单算子加速当作多模态端到端收益

一次图像问答通常包含视觉编码、连接器、语言 prefill，以及逐 token 的 decode。prefill 一次处理已有图文上下文；decode 基于历史状态逐步生成。不同阶段的矩阵形状、计算与访存占比不同，低比特内核的相对收益也可能不同。

可用下面的时间分解理解测量口径：

$$
T_{\rm total}=T_{\rm preprocess}+T_{\rm vision}+T_{\rm connector}
+T_{\rm prefill}+\sum_{k=1}^{K}T_{{\rm decode},k}+T_{\rm other}.
$$

若未来引入端云协同，网络传输、排队和云端计算也需要单独计入；上述两篇论文没有提供这些成本的实测结论。只有语言线性层变快时，其余部分不会按同一倍数缩短。低位宽存储、反量化与 kernel 的关系见 [量化矩阵乘法的缩放与执行路径](../../implementation/quantized-matmul-scaling-execution.md)。

模型质量同样需要分任务检查。OCR、文档问答、一般图像描述和纯文本推理可能受不同误差影响；一个汇总平均分不能说明所有能力都保留。关于正负结果和可比条件，可继续阅读 [MBQ 与 VLMQ 的条件化比较](../../research/mbq-vlmq-comparison.md)。

当评测范围扩展到可信度维度时，还需要区分任务精度与社会性指标。已有的 LVLM 压缩基准把静态权重与动态 KV cache 分开、并同时报告识别、推理、幻觉、偏见与毒性等指标，其设计见 [VLM 压缩评测框架](../../implementation/lvlm-compression-benchmark.md)。

## 6. 模态行、特殊通道与被量化模块

设全部 token 的激活是 $X\in\mathbb R^{T\times d}$。[MASQuant](../../methods/masquant.md) v1 §4 按模态选择行、分别使用尺度；[SplitQ](../../methods/splitq.md) v1 §4.2 用模态统计选择列，再把全部 token 的列划为三组。后者有 $XW=\sum_jX[:,C_j]W[C_j,:]$，每个 token 都保留各组贡献；只有其 MAC 分支再选择文字行。模态来源、列统计标签与运行时路径是三层概念。

MASQuant 的主要实验只量化语言部分，Qwen2.5-Omni 限于 Thinker；SplitQ 主实验也以语言部分为中心，附录 A.4 才增加视觉编码器 W4A4 的有限测试。因此“多模态量化”可能指多模态输入进入共享语言层后的量化，不应自动解释为视觉、连接器、语言、音频生成和缓存全部低比特化。比较前应逐模块列出对象，而不只记模型名称。

## 来源身份

下表用于在没有本地资料库时辨识来源；具体论述的章节、公式、图表或代码位置见正文。

| 来源 | 版本或快照 | 说明 |
| --- | --- | --- |
| [Breaking Modality Heterogeneity in Low-Bit Quantization for Large Vision-Language Models](https://arxiv.org/abs/2605.19929v1) | `arXiv:2605.19929v1` | — |
| [MASQuant: Modality-Aware Smoothing Quantization for Multimodal Large Language Models](https://arxiv.org/abs/2603.04800v1) | `arXiv:2603.04800v1` | — |
| [Q-VLM: Post-training Quantization for Large Vision-Language Models](https://arxiv.org/abs/2410.08119v3) | `arXiv:2410.08119v3` | — |
| [MQuant: Unleashing the Inference Potential of Multimodal Large Language Models via Full Static Quantization](https://arxiv.org/abs/2502.00425v2) | `arXiv:2502.00425v2` | — |
| [MBQ: Modality-Balanced Quantization for Large Vision-Language Models](https://arxiv.org/abs/2412.19509v2) | `arXiv:2412.19509v2` | — |
| [VLMQ: Token Saliency-Driven Post-Training Quantization for Vision-language Models](https://arxiv.org/abs/2508.03351v2) | `arXiv:2508.03351v2` | — |
| [thu-nics/MBQ](https://github.com/thu-nics/MBQ/tree/a4d460dfb4b1c07b5d1f3ddda6e86d1c90d6e7f1) | `a4d460dfb4b1c07b5d1f3ddda6e86d1c90d6e7f1` | — |
