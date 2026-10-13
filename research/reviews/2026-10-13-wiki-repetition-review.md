# Wiki 重复性审查覆盖记录

起始版本：`d1e700a`；日期：2026-10-13。按解释作用检查重复，保留机制、推导、条件、证据与反例，不进行 ingest。

第 1 批来源区已完成（`ac1e459`）：删除 53 页通用引言，25 张全空说明表缩为两列；来源区清理不等同于正文主题复核。后续清单按主归属计数，每个知识页只列一次，必要关联页可复读。

“完成”表示本组正文或指定差异已核读、发现的问题已处理并复核；原件核验按具体疑点开展，不表示全部科学主张重新审计。问题、理由、检查与遗留范围见 `wiki/LOG.md`。

| 批次 | 范围 | 页数 | 状态 | 提交 |
| --- | --- | ---: | --- | --- |
| 2 | 经典方法与实现 | 7 | 完成 | da89912 |
| 3 | 公共机制 | 12 | 完成 | 5dd99d2 |
| 4a | 旋转与可学习变换 | 10 | 完成 | acb89fb |
| 4b | 码本与表示方法 | 11 | 完成 | acb89fb |
| 4c | 码本实现与新表示 | 8 | 完成 | acb89fb |
| 4d | 混合精度与预算 | 13 | 完成 | cf7a765 |
| 4e | KV cache与注意力误差 | 15 | 完成 | d9d1355 |
| 4f | MoE与MLA | 12 | 完成 | 97c2339 |
| 4g | 硬件与算子契约 | 22 | 完成 | 8bf7759 |
| 4h | 部署与服务 | 17 | 完成 | 28888a8 |
| 4i | 评测与质量 | 12 | 完成 | aaf746d |
| 4j | 多模态目标与状态 | 18 | 完成 | 191a755 |
| 4k | 训练与误差恢复 | 17 | 完成 | — |

正文已覆盖 174 / 174 页；第 5 批阅读路径与全库收尾待完成。

## 2 · 经典方法与实现

[awq](../../wiki/methods/awq.md)、[gptq](../../wiki/methods/gptq.md)、[smoothquant](../../wiki/methods/smoothquant.md)、[awq-implementation](../../wiki/implementation/awq-implementation.md)、[gptq-implementation](../../wiki/implementation/gptq-implementation.md)、[smoothquant-implementation](../../wiki/implementation/smoothquant-implementation.md)、[awq-gptq-smoothquant-comparison](../../wiki/methods/awq-gptq-smoothquant-comparison.md)。

## 3 · 公共机制

[calibration-and-range-selection](../../wiki/theory/calibration-and-range-selection.md)、[layer-reconstruction-second-order-compensation](../../wiki/theory/layer-reconstruction-second-order-compensation.md)、[curvature-weighted-quantization-error](../../wiki/theory/curvature-weighted-quantization-error.md)、[diagonal-scaling-equivalent-transform](../../wiki/theory/diagonal-scaling-equivalent-transform.md)、[uniform-quantization-and-groups](../../wiki/fundamentals/quantization/uniform-quantization-and-groups.md)、[post-training-and-quantization-aware-training](../../wiki/fundamentals/quantization/post-training-and-quantization-aware-training.md)、[adaround](../../wiki/methods/adaround.md)、[brecq](../../wiki/methods/brecq.md)、[calibquant](../../wiki/methods/calibquant.md)、[conqur](../../wiki/methods/conqur.md)、[lsq](../../wiki/methods/lsq.md)、[omniquant](../../wiki/methods/omniquant.md)。

## 4a · 旋转与可学习变换

[quarot](../../wiki/methods/quarot.md)、[spinquant](../../wiki/methods/spinquant.md)、[flatquant](../../wiki/methods/flatquant.md)、[affinequant](../../wiki/methods/affinequant.md)、[ostquant](../../wiki/methods/ostquant.md)、[mquant](../../wiki/methods/mquant.md)、[centriq](../../wiki/methods/centriq.md)、[orthogonal-rotation-and-hadamard-quantization](../../wiki/theory/orthogonal-rotation-and-hadamard-quantization.md)、[omniquant-affinequant-flatquant-comparison](../../wiki/research/omniquant-affinequant-flatquant-comparison.md)、[invertible-transforms-and-kronecker-products](../../wiki/fundamentals/mathematics/invertible-transforms-and-kronecker-products.md)。

## 4b · 码本与表示方法

[codebook-quantization-and-bit-budget](../../wiki/theory/codebook-quantization-and-bit-budget.md)、[low-bit-representation-design](../../wiki/research/low-bit-representation-design.md)、[quip](../../wiki/methods/quip.md)、[quip-sharp](../../wiki/methods/quip-sharp.md)、[aqlm](../../wiki/methods/aqlm.md)、[gptvq](../../wiki/methods/gptvq.md)、[vptq](../../wiki/methods/vptq.md)、[qtip](../../wiki/methods/qtip.md)、[spqr](../../wiki/methods/spqr.md)、[squeezellm](../../wiki/methods/squeezellm.md)、[sphquant](../../wiki/methods/sphquant.md)。

## 4c · 码本实现与新表示

[quantforge](../../wiki/methods/quantforge.md)、[xor-trellis](../../wiki/methods/xor-trellis.md)、[fluxbin](../../wiki/methods/fluxbin.md)、[quip-ldlq-and-coordinate-restoration](../../wiki/implementation/quip-ldlq-and-coordinate-restoration.md)、[quip-sharp-e8p-packing-and-inference](../../wiki/implementation/quip-sharp-e8p-packing-and-inference.md)、[spqr-metadata-outliers-and-cuda-format](../../wiki/implementation/spqr-metadata-outliers-and-cuda-format.md)、[aqlm-pv-training-and-code-updates](../../wiki/implementation/aqlm-pv-training-and-code-updates.md)、[aqlm-calibration-export-and-inference](../../wiki/implementation/aqlm-calibration-export-and-inference.md)。

## 4d · 混合精度与预算

[mixed-precision-allocation](../../wiki/theory/mixed-precision-allocation.md)、[hawq-v2](../../wiki/methods/hawq-v2.md)、[cmpq](../../wiki/methods/cmpq.md)、[coopq](../../wiki/methods/coopq.md)、[llm-mq](../../wiki/methods/llm-mq.md)、[dbw](../../wiki/methods/dbw.md)、[stagq](../../wiki/methods/stagq.md)、[resq](../../wiki/methods/resq.md)、[resq-basis-and-fake-quantization](../../wiki/implementation/resq-basis-and-fake-quantization.md)、[ramp](../../wiki/methods/ramp.md)、[luq](../../wiki/methods/luq.md)、[prquant](../../wiki/methods/prquant.md)、[casa](../../wiki/methods/casa.md)。

## 4e · KV cache与注意力误差

[kv-cache-quantization-objects-and-granularity](../../wiki/theory/kv-cache-quantization-objects-and-granularity.md)、[attention-quantization-error-propagation](../../wiki/theory/attention-quantization-error-propagation.md)、[attention-sink-map-transfer](../../wiki/theory/attention-sink-map-transfer.md)、[kivi](../../wiki/methods/kivi.md)、[qjl](../../wiki/methods/qjl.md)、[kvquant](../../wiki/methods/kvquant.md)、[kitty](../../wiki/methods/kitty.md)、[nova-kv](../../wiki/methods/nova-kv.md)、[saw-int4](../../wiki/methods/saw-int4.md)、[vidkv](../../wiki/methods/vidkv.md)、[wush-kv](../../wiki/methods/wush-kv.md)、[turboquant](../../wiki/methods/turboquant.md)、[akvq-vl](../../wiki/methods/akvq-vl.md)、[oscar-mixed-kv-cache-execution](../../wiki/implementation/oscar-mixed-kv-cache-execution.md)、[flashinfer-paged-kv-write-and-nvfp4](../../wiki/implementation/flashinfer-paged-kv-write-and-nvfp4.md)。

## 4f · MoE与MLA

[mixture-of-experts-and-routing](../../wiki/fundamentals/model/mixture-of-experts-and-routing.md)、[multi-head-latent-attention](../../wiki/fundamentals/model/multi-head-latent-attention.md)、[moe-routing-and-quantized-expert-execution](../../wiki/implementation/moe-routing-and-quantized-expert-execution.md)、[moe-expert-parallel-execution](../../wiki/implementation/moe-expert-parallel-execution.md)、[deepep-moe-communication-execution](../../wiki/implementation/deepep-moe-communication-execution.md)、[deepgemm-moe-expert-execution](../../wiki/implementation/deepgemm-moe-expert-execution.md)、[w4a16-moe-expert-execution](../../wiki/implementation/w4a16-moe-expert-execution.md)、[mla-cache-and-attention-execution](../../wiki/implementation/mla-cache-and-attention-execution.md)、[vllm-mla-context-parallel-execution](../../wiki/implementation/vllm-mla-context-parallel-execution.md)、[vllm-mla-packed-cache-and-dsa-execution](../../wiki/implementation/vllm-mla-packed-cache-and-dsa-execution.md)、[dsa-sparse-device-execution](../../wiki/implementation/dsa-sparse-device-execution.md)、[veq](../../wiki/methods/veq.md)。

## 4g · 硬件与算子契约

[arithmetic-intensity-and-roofline](../../wiki/fundamentals/hardware/arithmetic-intensity-and-roofline.md)、[gpu-execution-and-memory-hierarchy](../../wiki/fundamentals/hardware/gpu-execution-and-memory-hierarchy.md)、[floating-point-and-accumulation](../../wiki/fundamentals/numeric-formats/floating-point-and-accumulation.md)、[fp8-and-mx-data-formats](../../wiki/fundamentals/numeric-formats/fp8-and-mx-data-formats.md)、[gpu-kernel-computation-patterns](../../wiki/fundamentals/operators/gpu-kernel-computation-patterns.md)、[linear-layer-input-channel](../../wiki/fundamentals/operators/linear-layer-input-channel.md)、[tensor-layout-and-kernel-contracts](../../wiki/fundamentals/operators/tensor-layout-and-kernel-contracts.md)、[triton-language-and-compilation](../../wiki/fundamentals/operators/triton-language-and-compilation.md)、[tensor-core-quantized-gemm](../../wiki/implementation/tensor-core-quantized-gemm.md)、[w8a8-quantization-gemm-kernels](../../wiki/implementation/w8a8-quantization-gemm-kernels.md)、[weight-only-dequant-kernels](../../wiki/implementation/weight-only-dequant-kernels.md)、[marlin-batched-w4a16-gemm](../../wiki/implementation/marlin-batched-w4a16-gemm.md)、[block-scaled-matmul-execution](../../wiki/implementation/block-scaled-matmul-execution.md)、[quantized-matmul-scaling-execution](../../wiki/implementation/quantized-matmul-scaling-execution.md)、[cuda-operator-build-and-integration](../../wiki/implementation/cuda-operator-build-and-integration.md)、[gpu-async-copy-pipelines](../../wiki/implementation/gpu-async-copy-pipelines.md)、[gpu-kernel-performance-analysis](../../wiki/implementation/gpu-kernel-performance-analysis.md)、[kernel-configuration-and-autotuning](../../wiki/implementation/kernel-configuration-and-autotuning.md)、[kernel-correctness-and-benchmarking](../../wiki/implementation/kernel-correctness-and-benchmarking.md)、[gguf-block-quantization-formats](../../wiki/implementation/gguf-block-quantization-formats.md)、[gated-activation-cuda-triton-kernels](../../wiki/implementation/gated-activation-cuda-triton-kernels.md)、[rmsnorm-cuda-triton-kernels](../../wiki/implementation/rmsnorm-cuda-triton-kernels.md)。

## 4h · 部署与服务

[transformer-autoregressive-inference](../../wiki/fundamentals/model/transformer-autoregressive-inference.md)、[vllm-inference-execution](../../wiki/implementation/vllm-inference-execution.md)、[vllm-attention-operator-design](../../wiki/implementation/vllm-attention-operator-design.md)、[sglang-inference-execution](../../wiki/implementation/sglang-inference-execution.md)、[sglang-attention-operator-design](../../wiki/implementation/sglang-attention-operator-design.md)、[tensor-parallel-quantization](../../wiki/implementation/tensor-parallel-quantization.md)、[serving-memory-and-batching](../../wiki/implementation/serving-memory-and-batching.md)、[serving-performance-evaluation](../../wiki/implementation/serving-performance-evaluation.md)、[quantized-llm-deployment-backends](../../wiki/implementation/quantized-llm-deployment-backends.md)、[qserve](../../wiki/methods/qserve.md)、[qserve-implementation](../../wiki/implementation/qserve-implementation.md)、[atom](../../wiki/methods/atom.md)、[atom-mixed-gemm-and-paged-kv](../../wiki/implementation/atom-mixed-gemm-and-paged-kv.md)、[flexq](../../wiki/methods/flexq.md)、[flexq-bitplanes-and-group-restoration](../../wiki/implementation/flexq-bitplanes-and-group-restoration.md)、[disaggregated-quantization](../../wiki/methods/disaggregated-quantization.md)、[ideogram-int8-gguf](../../wiki/methods/ideogram-int8-gguf.md)。

## 4i · 评测与质量

[model-quality-evaluation](../../wiki/implementation/model-quality-evaluation.md)、[quantization-error-diagnosis](../../wiki/implementation/quantization-error-diagnosis.md)、[helm-evaluation-protocol](../../wiki/implementation/helm-evaluation-protocol.md)、[longbench-v2-context-and-scoring](../../wiki/implementation/longbench-v2-context-and-scoring.md)、[mmlu-pro-options-and-answer-extraction](../../wiki/implementation/mmlu-pro-options-and-answer-extraction.md)、[ruler-task-complexity-and-effective-context](../../wiki/implementation/ruler-task-complexity-and-effective-context.md)、[lm-evaluation-harness-request-filter-and-aggregation](../../wiki/implementation/lm-evaluation-harness-request-filter-and-aggregation.md)、[lvlm-compression-benchmark](../../wiki/implementation/lvlm-compression-benchmark.md)、[tts-quantization-evaluation](../../wiki/implementation/tts-quantization-evaluation.md)、[quantization-reliability-and-selective-prediction](../../wiki/theory/quantization-reliability-and-selective-prediction.md)、[quantization-ranking-stability](../../wiki/theory/quantization-ranking-stability.md)、[quantization-price-prediction](../../wiki/methods/quantization-price-prediction.md)。

## 4j · 多模态目标与状态

[vision-language-model-tokens-and-quantization](../../wiki/fundamentals/model/vision-language-model-tokens-and-quantization.md)、[multimodal-alignment-diagnostics](../../wiki/theory/multimodal-alignment-diagnostics.md)、[integrated-gradients-and-quantization-sensitivity](../../wiki/theory/integrated-gradients-and-quantization-sensitivity.md)、[mbq-vlmq-comparison](../../wiki/research/mbq-vlmq-comparison.md)、[mbq](../../wiki/methods/mbq.md)、[vlmq](../../wiki/methods/vlmq.md)、[masquant](../../wiki/methods/masquant.md)、[q-vlm](../../wiki/methods/q-vlm.md)、[qig](../../wiki/methods/qig.md)、[qslaw](../../wiki/methods/qslaw.md)、[qslaw-scale-learning-and-float-export](../../wiki/implementation/qslaw-scale-learning-and-float-export.md)、[onptq](../../wiki/methods/onptq.md)、[pivot-q](../../wiki/methods/pivot-q.md)、[foldquantvla](../../wiki/methods/foldquantvla.md)、[saprune](../../wiki/methods/saprune.md)、[coreprune](../../wiki/methods/coreprune.md)、[layerpos](../../wiki/methods/layerpos.md)、[qapruner](../../wiki/methods/qapruner.md)。

## 4k · 训练与误差恢复

[efficientqat](../../wiki/methods/efficientqat.md)、[llm-int8](../../wiki/methods/llm-int8.md)、[zeroquant](../../wiki/methods/zeroquant.md)、[llm-qat](../../wiki/methods/llm-qat.md)、[lc-qat](../../wiki/methods/lc-qat.md)、[quasar](../../wiki/methods/quasar.md)、[norm-tweaking](../../wiki/methods/norm-tweaking.md)、[pv-tuning](../../wiki/methods/pv-tuning.md)、[loftq](../../wiki/methods/loftq.md)、[qera](../../wiki/methods/qera.md)、[activation-guided-compensation](../../wiki/methods/activation-guided-compensation.md)、[rgsq](../../wiki/methods/rgsq.md)、[sharq](../../wiki/methods/sharq.md)、[sharq-sparse-residual-and-shared-scales](../../wiki/implementation/sharq-sparse-residual-and-shared-scales.md)、[smoothquant-plus](../../wiki/methods/smoothquant-plus.md)、[splitq](../../wiki/methods/splitq.md)、[entropy-and-dependence](../../wiki/fundamentals/mathematics/entropy-and-dependence.md)。
