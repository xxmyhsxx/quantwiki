---
title: 医学LLaVA压缩：输出相似度删层、恢复微调与AWQ
type: method
tags:
  - vlm
  - pruning
  - weight-quantization
  - evaluation
sources:
  - raw/papers/2026-09-21/medical-mllm-compression
updated: 2026-10-15
---

# 医学LLaVA压缩：输出相似度删层、恢复微调与AWQ

这项研究在皮肤病图文问答上组合三种操作：删除语言骨干的整层、用领域监督恢复、再做AWQ权重量化。删层减少网络深度，微调调整幸存参数，量化改变权重表示；三者的质量和成本需按阶段分别评价。

本页依据Compression Strategies for Efficient Multimodal LLMs in Medical Contexts v3。已有[AWQ机制](awq.md)与[LoRA适配](lora.md)直接复用；这里重点解释删层依据与恢复顺序，所有数字均为论文报告，未核作者实现或运行。

## 1. 删除语言层，以生成答案的变化评分

研究对象含CLIP编码器、多模态projector和32层语言骨干，主要剪语言Transformer block，保留token embedding、输出head及原文编号1、2、32层。这里删除的是计算层，不是[视觉token](../fundamentals/model/vision-language-model-tokens-and-quantization.md)的序列行。

对每个候选层 $\ell$，先单独移除它，在校准图文问答上比较完整模型 $f$与删层模型 $f_{-\ell}$的**生成输出**。用all-MiniLM-L6-v2将答案映成句向量 $g(y)$，再计算

$$S_\ell=\frac1{|\mathcal D_{\mathrm{cal}}|}\sum_{x\in\mathcal D_{\mathrm{cal}}}
\cos\!\left(g(f(x)),g(f_{-\ell}(x))\right).$$

高 $S_\ell$表示删除该层后答案在这个embedding空间里较接近原答案，按方法段的解释，该层较不重要。它测的是任务输出的相似性，不是相邻隐藏层的余弦，也不保证原答案正确。多层合删后还有交互，单层排序不能直接给出累计损失。（§3.2。）

原文实验段又写“minimum sum/lowest importance”优先移除，与上述相似度方向没有明确对应；校准样本量、生成配置及每轮是否重算全层评分也未给齐。本页保留方法解释，不将其补写成已核对的完整排序实现。

## 2. 恢复发生在最终量化之前

论文先用领域数据适配LLaVA，再分析并删除层，随后用相同训练集做恢复SFT，最后量化。两次QLoRA配置分别是初次适配 $r=8,\alpha=32$，删层后恢复 $r=16,\alpha=8$；不能把它们并成一套超参数。（§3.1、§4.4。）

领域数据来自DermNet的八个类别，训练1737图，测试39图。恢复阶段仍需训练资源：小模型推理占用下降，不意味着恢复训练也只占同样显存。论文只提供所列模型的时间/显存结果，没有完整adapter目标模块、优化器、步数与checkpoint配置，故不从 $r$单独计算整个模型的可训练参数量。

最终AWQ按激活选通道缩放并在组内量化；§4.5声明比较方法group size为64。它需要压缩后模型的校准分布与实际导出/消费匹配，不能把本案例解释成“保留显著权重为FP16”或新的激活量化方法。论文没有给齐量化器与kernel配置，具体执行格式仍按[部署路径](../implementation/quantized-llm-deployment-backends.md)核对。

## 3. 对照同时记录质量、显存和速度

原文Table4的阶段对照为：

| 方案 | 作者报告的质量分 | VRAM（GB） | 延迟（ms/token） |
| --- | ---: | ---: | ---: |
| 原LLaVA，16bit | 80.12 | 13.4 | 154 |
| 只量化，4bit | 64.72 | 4.5 | 146 |
| 删层后直接量化 | 25.00 | 3.9 | 122 |
| 删层→SFT→量化 | 54.25 | 3.9 | 122 |

同表说明恢复SFT改善了压缩模型的任务分，却没有恢复到原始质量；两种删层量化方案的延迟相同，也不能将分数恢复读成额外提速。原文Discussion称4.2×，与154/122这组延迟不符；Table2/3的5.6B与Table4的5.85B模型标识也未闭合，不能拼成同一精确配置。测试设备为Kaggle T4，batch、长度、预热与计时边界未充分说明。

质量分来自Qwen-2.5-32B judge：39图各问三题，对生成答案参照reference作1–4评级，求均值再百分化；原文未说明完整换算公式。该分数不是诊断正确率，句向量相似也不能替代[独立任务与可靠性评价](../implementation/model-quality-evaluation.md)。少量案例支持此领域/数据下的阶段对照，不构成临床验证或跨任务排序。

## 来源身份

| 来源 | 固定版本 | 核读与使用范围 |
| --- | --- | --- |
| Compression Strategies for Efficient Multimodal LLMs in Medical Contexts（`https://arxiv.org/abs/2507.21976v3`） | `arXiv:2507.21976v3` | 完整active正文、附录A/B与表1–5；PDF10/15/16原页回查阶段结果、judge与迭代案例。保留评分方向、参数身份与速度说法差异；未核外部代码、数据划分复现或GPU运行。 |
