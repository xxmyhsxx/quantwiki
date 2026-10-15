---
title: 量化知识地图
type: index
sources: []
---

# 量化知识地图

从当前问题选择入口，再沿正文中的必要关联深入。本页供快速浏览，只保留代表入口；具体方法和实现可按名称在知识目录中查找。

## 从问题进入

| 你想理解什么 | 从这里开始 |
| --- | --- |
| 模型在算什么，量化改变哪里 | [自回归主干与模型差异](fundamentals/model/transformer-autoregressive-inference.md)、[视觉语言模型的量化对象](fundamentals/model/vision-language-model-tokens-and-quantization.md) |
| 量化的基本表示与方法区别 | [均匀量化与分组](fundamentals/quantization/uniform-quantization-and-groups.md) → [AWQ、GPTQ 与 SmoothQuant](methods/awq-gptq-smoothquant-comparison.md) |
| 校准、变换与精度预算怎样选择 | [校准与范围](theory/calibration-and-range-selection.md)、[可学习变换比较](research/omniquant-affinequant-flatquant-comparison.md)、[混合精度分配](theory/mixed-precision-allocation.md) |
| 极低比特怎样表示与保存 | [码本与预算](theory/codebook-quantization-and-bit-budget.md) → [表示设计](research/low-bit-representation-design.md) |
| KV、MoE、MLA 为什么需要分别分析 | [KV 对象与粒度](theory/kv-cache-quantization-objects-and-granularity.md)、[专家路由](fundamentals/model/mixture-of-experts-and-routing.md)、[MLA 缓存](fundamentals/model/multi-head-latent-attention.md) |
| 量化产物怎样进入推理系统 | [部署框架与后端](implementation/quantized-llm-deployment-backends.md) → [服务内存与批处理](implementation/serving-memory-and-batching.md) → [nano-vllm 执行链](implementation/nano-vllm-inference-execution.md) |
| GPU 如何执行低精度算子 | [布局与接口](fundamentals/operators/tensor-layout-and-kernel-contracts.md) → [计算模式](fundamentals/operators/gpu-kernel-computation-patterns.md) → [反量化内核](implementation/weight-only-dequant-kernels.md) |
| 误差、质量与运行收益怎样判断 | [误差诊断](implementation/quantization-error-diagnosis.md)、[模型质量](implementation/model-quality-evaluation.md)、[服务性能](implementation/serving-performance-evaluation.md) |

## 查找与维护

知识按解释职责放在 `fundamentals/`、`theory/`、`methods/`、`research/`、`implementation/`。未列入本页不表示没有覆盖；已知方法或框架时，直接按名称查找对应正文。

资料版本和已读范围见 [研读目录](READING.md)，写作与引用约定见 [维护规范](README.md)，实际变动见 [维护记录](LOG.md)。项目定位与当前画像由根目录的 `SELF.md` 说明。
