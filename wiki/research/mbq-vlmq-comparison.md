---
title: MBQ 与 VLMQ：重要性怎样进入量化目标
type: comparison
tags:
  - vlm
  - sensitivity
  - reconstruction
  - calibration
sources:
  - raw/papers/2026-09-21/luq/paper.pdf
  - raw/papers/2026-09-21/qig/paper.pdf
  - raw/papers/2026-09-21/splitq/paper.pdf
  - raw/papers/2026-09-21/masquant/paper.pdf
  - raw/papers/2026-09-21/mquant/paper.pdf
  - raw/papers/2026-09-21/mbq/paper.pdf
  - raw/papers/2026-09-21/vlmq/paper.pdf
  - raw/repositories/2026-09-21/mbq/source/qmllm/methods/mbq/quantize/auto_scale.py
  - raw/repositories/2026-09-21/mbq/source/qmllm/methods/mbq/quantize/pre_quant.py
updated: 2026-09-15
---

# MBQ 与 VLMQ：重要性怎样进入量化目标

两种方法都认为，直接平等处理全部 token 的重构误差，可能没有准确表达视觉语言任务的需求。但从“估计什么重要”到“实际改了什么”，二者采取了不同路线。不能仅用“MBQ 模态级、VLMQ token 级”概括，更不能跨论文用一个平均分决定优劣。

比较版本为 MBQ arXiv:2412.19509v2 与 VLMQ arXiv:2508.03351v2；MBQ 实现细节另限于官方 commit a4d460df。共同的模型计算背景见 [视觉语言模型中的 token 与量化对象](../fundamentals/model/vision-language-model-tokens-and-quantization.md)。

## 1. 逐项比较机制

| 维度 | MBQ | VLMQ |
|---|---|---|
| 主要调整对象 | 通道等价缩放的候选选择 | 二阶重构中的 token 误差代价与权重补偿 |
| 重要性来源 | 校准回答交叉熵对中间输出的梯度 | 局部带残差注意力模块重构 MSE 的梯度 |
| 聚合粒度 | 模态组统计；固定代码还按 attention/MLP 使用梯度比与中位数下限 | 每投影、每 token 聚合通道梯度，Q/K/V/O 分别处理 |
| 误差目标 | 默认加权 MAE；代码是 mask 内加权和再除共同元素数 | 残差先右乘 $G$ 再求平方范数，实际系数为 $g_n^2$ |
| 求解方式 | 在受限尺度族中做网格搜索，随后量化 | 默认 GPTAQ 非对称校准与二阶补偿，部分 INT2 改用 GPTQ |
| 位宽范围 | W3A16、W4A8 为主，另有高位宽消融 | 主实验 weight-only INT2/3/4，分组和无分组均出现 |
| 重要性在推理时的作用 | 通过已确定的缩放/量化参数体现 | 通过已确定的量化权重体现 |
| 没有做的事情 | 未按 token 动态分配权重位宽 | 未剪枝 token、未按 token 动态分配权重位宽 |

MBQ 的完整搜索、归一化和实现见 [MBQ](../methods/mbq.md)；VLMQ 的 $G^2$、非对称残差与约束解见 [VLMQ](../methods/vlmq.md)。前者从 [对角缩放](../theory/diagonal-scaling-equivalent-transform.md) 入手，后者从 [重构与二阶补偿](../theory/layer-reconstruction-second-order-compensation.md) 入手，关注相近问题不意味着优化同一个目标。

## 2. 两篇关于 token 权重的结果是否矛盾

MBQ §4.3.2 报告直接按 token 梯度重加权，相比模态平衡在 OCRBench 下降约 1.5。VLMQ 的 token 级方法却在多项设置上改善了基础算法。当前证据不足以将其解释为“谁推翻了谁”，因为至少同时改变了：

1. 梯度对应的损失：全网回答交叉熵还是局部模块重构。
2. 梯度被观察和聚合的位置：哪些投影、是否包含输出投影与残差。
3. 因子进入目标的方式：L1 误差系数还是平方误差前的因子。
4. 校准样本数、模板、有效 token 与量化配置。
5. 量化算法：缩放搜索还是 GPTQ/GPTAQ 权重补偿。

其中任何一个都可能影响结果。因此“更细粒度总是更好”与“模态平均总是更稳”都不是已确立规律。若要研究粒度，应在相同基础算法和数据下只改变粒度，并保持归一化可比；这是从现有证据导出的比较要求，不是本轮已经执行的实验。

## 3. 哪些结果可以直接比较

**优先使用同一论文中控制条件较完整的表格。** MBQ 表 4 区分 Pile/COCO 与加权损失；VLMQ 表 6–9 区分因子、反向范围、增强层及基础算法。但“同表”也不自动消除代码版本、超参数选择和统计不确定性的缺口。

VLMQ 表 2 在 Qwen2-VL-7B INT3g128 上列出 MBQ 72.68、GPTAQ 73.68、VLMQ 74.40，这是 VLMQ 作者同一评测表中的比较。它不是我们复现的结果，也不能与 MBQ 自身论文的综合平均值直接拼接。相同表中 Qwen2-VL-2B 的 VLMQ 62.90 还低于 GPTAQ 63.44，负面结果应一并保留。

本页只使用条件可辨、数字可核对的比较；影响排名的原文问题在方法页简要说明。

**校准与评价不是同一阶段。** 校准误差下降不自动意味着任务质量提升，更不意味着端到端时延下降。VLMQ 表 3 中单项 +16.45 个百分点与平均 +1.88 同时伴随其他任务下降；MBQ 表 3 中某个设置与 RTN 持平。误差到质量的验证链见 [量化误差诊断与验证](../implementation/quantization-error-diagnosis.md)。

## 4. 梯度与注意力能否互相替代

注意力权重表示一次注意力运算如何组合值向量；梯度表示指定损失对中间量变化的局部响应。二者不是同一个数学对象。大注意力权重可能作用于小值向量，也可能被后续层抵消；梯度也可能因选择的损失、饱和或当前参考状态而很小。

VLMQ 表 6 在无分组 INT3、Qwen2-VL-7B 上报告：常数权重 69.03，FastV 型注意力 68.29，PACT 型注意力 67.07，梯度 69.70。它足以说明，直接移用这两种注意力分数在该量化目标下不可靠；不足以排除其他注意力估计、不同归一化或其他任务上的有效性。

另一方面，MBQ 的平均梯度加权“上界”缺充分条件，VLMQ 的一阶展开也不能证明绝对梯度均值和平方权重全局最优。科研中应把“代理指标的动机”“同条件消融”“任务级验证”分别呈现，不因使用梯度就自动赋予理论保证。

[MQuant](../methods/mquant.md) 提供另一个比较维度：它按模态分别校准激活网格，并通过序列重排改善执行布局，不依赖这里的梯度重要性目标。因而“模态感知”至少应区分误差目标、量化网格与运行布局，不能只按名称归为同一种方法；MQuant 的组件与运行证据见其 §3.1、表 7/10，跨方法质量仍需统一位宽、对象和协议。

## 5. 哪些认识可以进入后续研究

- **已有工作已覆盖重要性加权重构。** 仅提出“感知模态/token 重要性并加权量化误差”，还不足以支持新的创新主张。需要更具体的问题、差异和证据。
- **重要性加权不等于精度分配。** 两篇主要通过固定低比特配置下的搜索或补偿改进精度，没有解决预算约束下逐层选位宽的问题。要走向混合精度，需要另行明确代价模型、选择变量及硬件支持。
- **校准任务会影响重要性。** 两篇的图像描述校准不能自动代表 OCR、文档理解、部署后分布变化等所有需求。[校准数据与范围选择](../theory/calibration-and-range-selection.md) 应保留分布和目标条件。
- **离线校准可以较重，部署路径仍需验证。** MBQ 有 RTX 4090 分阶段数据，VLMQ 有 H100 校准成本和 RTX 5090 线性层数据；两者均没有测量云端样本选择、参数下发或部署后再校准的完整端云系统。

这些是可复用的知识边界，不是对某个研究方向的自动裁定。后续是否研究任务条件下的敏感性、混合精度或端云校准，需要结合资源、基线和新的文献证据讨论。

## 6. 从目标加权走向表示与补偿

[MASQuant](../methods/masquant.md) v1 §4 将模态差别写入各自尺度，并以低秩残差适配共享文字权重；[SplitQ](../methods/splitq.md) v1 §4 先拆列通道，再分别处理权重和激活误差。因此“模态感知”还需增加两个比较维度：是否改变可用表示，以及是否引入实际辅助计算。

这不取消目标选择问题：MASQuant 表 4 的模态权重在 PPL 与任务平均分之间有取舍；SplitQ 表 5 的文字补偿优于视觉补偿只在指定模型、位宽和三个任务上成立。它们提供不同机制的条件化证据，不能与 MBQ/VLMQ 的梯度代理合并成“文字总比视觉重要”的通则。两篇独立方法页保留公式、消融、辅助位宽与实现证据，跨论文排名仍需统一协议。

## 7. 从误差归因走向预算分配

[QIG](../methods/qig.md) v1 §3 将局部梯度进一步扩展为量化差异的路径归因，并用 token 系数指导缩放或二阶统计。比较 MBQ/VLMQ/QIG 时，应分别控制目标（CE、局部重构或量化差异）、路径与基线、粒度、后处理以及基础量化器。QIG 表 1 的 SFT 分数细化后反而退化，进一步限制了“越细越好”的概括；原始归因的数学性质也不自动保证后处理后的优化收益。

[LUQ](../methods/luq.md) v3 §3.3 则补上前文尚未解决的层间预算问题：用簇频率熵固定排序，前缀层使用 BiLLM，其余用 GPTQ 4 bit。它的选择变量是层配置，不能把这一熵分数直接当作 MBQ/VLMQ/QIG 的 token 损失系数。是否把两类方法结合，需要先说明预算、底层量化器和评测条件；当前文献没有验证任意组合都能叠加增益。

## 来源身份

下表用于在没有本地资料库时辨识来源；具体论述的章节、公式、图表或代码位置见正文。

| 来源 | 版本或快照 | 说明 |
| --- | --- | --- |
| [LUQ: Layerwise Ultra-Low Bit Quantization for Multimodal Large Language Models](https://arxiv.org/abs/2509.23729v3) | `arXiv:2509.23729v3` | — |
| [Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients](https://arxiv.org/abs/2603.17809v1) | `arXiv:2603.17809v1` | — |
| [Breaking Modality Heterogeneity in Low-Bit Quantization for Large Vision-Language Models](https://arxiv.org/abs/2605.19929v1) | `arXiv:2605.19929v1` | — |
| [MASQuant: Modality-Aware Smoothing Quantization for Multimodal Large Language Models](https://arxiv.org/abs/2603.04800v1) | `arXiv:2603.04800v1` | — |
| [MQuant: Unleashing the Inference Potential of Multimodal Large Language Models via Full Static Quantization](https://arxiv.org/abs/2502.00425v2) | `arXiv:2502.00425v2` | — |
| [MBQ: Modality-Balanced Quantization for Large Vision-Language Models](https://arxiv.org/abs/2412.19509v2) | `arXiv:2412.19509v2` | — |
| [VLMQ: Token Saliency-Driven Post-Training Quantization for Vision-language Models](https://arxiv.org/abs/2508.03351v2) | `arXiv:2508.03351v2` | — |
| [thu-nics/MBQ](https://github.com/thu-nics/MBQ/tree/a4d460dfb4b1c07b5d1f3ddda6e86d1c90d6e7f1) | `a4d460dfb4b1c07b5d1f3ddda6e86d1c90d6e7f1` | — |

## 教学计算材料

保留已有教学计算脚本及当时结果，供核对推导与反例；这些材料不代表模型复现或性能实验。

- [math-check-results.json](../assets/mbq-vlmq-comparison/checks/math-check-results.json)
- [verify_math.py](../assets/mbq-vlmq-comparison/checks/verify_math.py)
