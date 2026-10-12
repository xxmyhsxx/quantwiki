# 全库 Wiki 分批审查

起始版本：`7602bd9`；清单建立日期：2026-10-13。范围为 174 个知识页和 4 个导航、管理页，相关图像和教学材料随所属页面核对。

重点检查解释深度、跨页一致性、事实与来源，以及重复防御性表达。每批按“读正文与关联页 → 列问题和理由 → 定向查证 → 修复 → 内容复核与必要机械检查 → 本地提交”闭合后，再进入下一批。不进行 ingest、模型复现或性能实验。

“已完成”表示本批页面职责与核心解释已核读、发现的问题已处理；来源核验是针对核心主张与疑点的定向核对，不表示所有上游代码或论文证明均重新审计。第 1 批沿用上一轮六页抽查，第 2 批沿用已完成的重构推导复核，第 3 批沿用已完成的 GPTQ 方法复核；相关正文在本轮开始时未变化。

当前完成 26 批、159 个知识页。`wiki/research/mbq-vlmq-comparison.md` 已通读，原有表格排版改动保留在工作区，本轮未修改或暂存该页。

后续批次最多八页；若一批包含独立计算链或问题较多，再拆分。没有问题的页面无需为凑修改而改写。新出现的跨页问题可回修已审页，并记录影响。

## 第 01 批 · 经典缩放与 AWQ

状态：已完成；6 页。提交：`b1c975d`。

- [diagonal-scaling-equivalent-transform](../../wiki/theory/diagonal-scaling-equivalent-transform.md)
- [awq](../../wiki/methods/awq.md)
- [smoothquant](../../wiki/methods/smoothquant.md)
- [awq-gptq-smoothquant-comparison](../../wiki/methods/awq-gptq-smoothquant-comparison.md)
- [quantized-matmul-scaling-execution](../../wiki/implementation/quantized-matmul-scaling-execution.md)
- [awq-implementation](../../wiki/implementation/awq-implementation.md)

## 第 02 批 · 基础与误差分析

状态：已完成；8 页。提交：`4f5c0c4`。

- [linear-layer-input-channel](../../wiki/fundamentals/operators/linear-layer-input-channel.md)
- [floating-point-and-accumulation](../../wiki/fundamentals/numeric-formats/floating-point-and-accumulation.md)
- [uniform-quantization-and-groups](../../wiki/fundamentals/quantization/uniform-quantization-and-groups.md)
- [entropy-and-dependence](../../wiki/fundamentals/mathematics/entropy-and-dependence.md)
- [invertible-transforms-and-kronecker-products](../../wiki/fundamentals/mathematics/invertible-transforms-and-kronecker-products.md)
- [curvature-weighted-quantization-error](../../wiki/theory/curvature-weighted-quantization-error.md)
- [calibration-and-range-selection](../../wiki/theory/calibration-and-range-selection.md)
- [layer-reconstruction-second-order-compensation](../../wiki/theory/layer-reconstruction-second-order-compensation.md)

## 第 03 批 · 经典方法与实现衔接

状态：已完成；5 页。提交：`c9f4c3c`。

- [gptq](../../wiki/methods/gptq.md)
- [llm-int8](../../wiki/methods/llm-int8.md)
- [zeroquant](../../wiki/methods/zeroquant.md)
- [gptq-implementation](../../wiki/implementation/gptq-implementation.md)
- [smoothquant-implementation](../../wiki/implementation/smoothquant-implementation.md)

## 第 04 批 · 模型基础与注意力误差

状态：已完成；6 页。提交：`59c0c71`。

- [mixture-of-experts-and-routing](../../wiki/fundamentals/model/mixture-of-experts-and-routing.md)
- [multi-head-latent-attention](../../wiki/fundamentals/model/multi-head-latent-attention.md)
- [transformer-autoregressive-inference](../../wiki/fundamentals/model/transformer-autoregressive-inference.md)
- [vision-language-model-tokens-and-quantization](../../wiki/fundamentals/model/vision-language-model-tokens-and-quantization.md)
- [attention-quantization-error-propagation](../../wiki/theory/attention-quantization-error-propagation.md)
- [attention-sink-map-transfer](../../wiki/theory/attention-sink-map-transfer.md)

## 第 05 批 · PTQ 舍入、重构与补偿 1

状态：已完成；8 页。提交：`9d3c405`。

- [quantization-error-diagnosis](../../wiki/implementation/quantization-error-diagnosis.md)
- [activation-guided-compensation](../../wiki/methods/activation-guided-compensation.md)
- [adaround](../../wiki/methods/adaround.md)
- [brecq](../../wiki/methods/brecq.md)
- [calibquant](../../wiki/methods/calibquant.md)
- [conqur](../../wiki/methods/conqur.md)
- [loftq](../../wiki/methods/loftq.md)
- [norm-tweaking](../../wiki/methods/norm-tweaking.md)

## 第 06 批 · PTQ 舍入、重构与补偿 2

状态：已完成；2 页。提交：`552c91d`。

- [qera](../../wiki/methods/qera.md)
- [sphquant](../../wiki/methods/sphquant.md)

## 第 07 批 · 可学习变换与旋转 1

状态：已完成；8 页。提交：`b97babf`。

- [resq-basis-and-fake-quantization](../../wiki/implementation/resq-basis-and-fake-quantization.md)
- [affinequant](../../wiki/methods/affinequant.md)
- [flatquant](../../wiki/methods/flatquant.md)
- [omniquant](../../wiki/methods/omniquant.md)
- [ostquant](../../wiki/methods/ostquant.md)
- [quarot](../../wiki/methods/quarot.md)
- [resq](../../wiki/methods/resq.md)
- [rgsq](../../wiki/methods/rgsq.md)

## 第 08 批 · 可学习变换与旋转 2

状态：已完成；4 页。提交：`a492005`。

- [smoothquant-plus](../../wiki/methods/smoothquant-plus.md)
- [spinquant](../../wiki/methods/spinquant.md)
- [omniquant-affinequant-flatquant-comparison](../../wiki/research/omniquant-affinequant-flatquant-comparison.md)
- [orthogonal-rotation-and-hadamard-quantization](../../wiki/theory/orthogonal-rotation-and-hadamard-quantization.md)

## 第 09 批 · 码本、极低比特表示与恢复 1

状态：已完成；8 页。提交：`497a44b`。

- [aqlm-calibration-export-and-inference](../../wiki/implementation/aqlm-calibration-export-and-inference.md)
- [aqlm-pv-training-and-code-updates](../../wiki/implementation/aqlm-pv-training-and-code-updates.md)
- [flexq-bitplanes-and-group-restoration](../../wiki/implementation/flexq-bitplanes-and-group-restoration.md)
- [quip-ldlq-and-coordinate-restoration](../../wiki/implementation/quip-ldlq-and-coordinate-restoration.md)
- [quip-sharp-e8p-packing-and-inference](../../wiki/implementation/quip-sharp-e8p-packing-and-inference.md)
- [sharq-sparse-residual-and-shared-scales](../../wiki/implementation/sharq-sparse-residual-and-shared-scales.md)
- [spqr-metadata-outliers-and-cuda-format](../../wiki/implementation/spqr-metadata-outliers-and-cuda-format.md)
- [aqlm](../../wiki/methods/aqlm.md)

## 第 10 批 · 码本、极低比特表示与恢复 2

状态：已完成；8 页。提交：`251e5b3`。

- [dbw](../../wiki/methods/dbw.md)
- [flexq](../../wiki/methods/flexq.md)
- [fluxbin](../../wiki/methods/fluxbin.md)
- [gptvq](../../wiki/methods/gptvq.md)
- [pv-tuning](../../wiki/methods/pv-tuning.md)
- [qtip](../../wiki/methods/qtip.md)
- [quantforge](../../wiki/methods/quantforge.md)
- [quip-sharp](../../wiki/methods/quip-sharp.md)

## 第 11 批 · 码本、极低比特表示与恢复 3

状态：已完成；8 页。提交：`abdead1`。

- [quip](../../wiki/methods/quip.md)
- [sharq](../../wiki/methods/sharq.md)
- [spqr](../../wiki/methods/spqr.md)
- [squeezellm](../../wiki/methods/squeezellm.md)
- [stagq](../../wiki/methods/stagq.md)
- [vptq](../../wiki/methods/vptq.md)
- [xor-trellis](../../wiki/methods/xor-trellis.md)
- [low-bit-representation-design](../../wiki/research/low-bit-representation-design.md)

## 第 12 批 · 码本、极低比特表示与恢复 4

状态：已完成；1 页。提交：`0231604`。

- [codebook-quantization-and-bit-budget](../../wiki/theory/codebook-quantization-and-bit-budget.md)

## 第 13 批 · 敏感性与混合精度分配 1

状态：已完成；8 页。提交：`3fa65f8`。

- [casa](../../wiki/methods/casa.md)
- [cmpq](../../wiki/methods/cmpq.md)
- [coopq](../../wiki/methods/coopq.md)
- [hawq-v2](../../wiki/methods/hawq-v2.md)
- [layerpos](../../wiki/methods/layerpos.md)
- [llm-mq](../../wiki/methods/llm-mq.md)
- [luq](../../wiki/methods/luq.md)
- [quantization-price-prediction](../../wiki/methods/quantization-price-prediction.md)

## 第 14 批 · 敏感性与混合精度分配 2

状态：已完成；2 页。提交：`6cf06b6`。

- [splitq](../../wiki/methods/splitq.md)
- [mixed-precision-allocation](../../wiki/theory/mixed-precision-allocation.md)

## 第 15 批 · QAT 与学习式量化

状态：已完成；8 页。提交：`181b43d`。

- [post-training-and-quantization-aware-training](../../wiki/fundamentals/quantization/post-training-and-quantization-aware-training.md)
- [qslaw-scale-learning-and-float-export](../../wiki/implementation/qslaw-scale-learning-and-float-export.md)
- [efficientqat](../../wiki/methods/efficientqat.md)
- [lc-qat](../../wiki/methods/lc-qat.md)
- [llm-qat](../../wiki/methods/llm-qat.md)
- [lsq](../../wiki/methods/lsq.md)
- [qslaw](../../wiki/methods/qslaw.md)
- [quasar](../../wiki/methods/quasar.md)

## 第 16 批 · 多模态目标、行为与压缩 1

状态：已完成；8 页。提交：`b0caf1c`。

- [lvlm-compression-benchmark](../../wiki/implementation/lvlm-compression-benchmark.md)
- [tts-quantization-evaluation](../../wiki/implementation/tts-quantization-evaluation.md)
- [akvq-vl](../../wiki/methods/akvq-vl.md)
- [centriq](../../wiki/methods/centriq.md)
- [coreprune](../../wiki/methods/coreprune.md)
- [foldquantvla](../../wiki/methods/foldquantvla.md)
- [masquant](../../wiki/methods/masquant.md)
- [mbq](../../wiki/methods/mbq.md)

## 第 17 批 · 多模态目标、行为与压缩 2

状态：已完成；8 页。提交：`0da3cbd`。

- [mquant](../../wiki/methods/mquant.md)
- [onptq](../../wiki/methods/onptq.md)
- [pivot-q](../../wiki/methods/pivot-q.md)
- [prquant](../../wiki/methods/prquant.md)
- [q-vlm](../../wiki/methods/q-vlm.md)
- [qapruner](../../wiki/methods/qapruner.md)
- [qig](../../wiki/methods/qig.md)
- [ramp](../../wiki/methods/ramp.md)

## 第 18 批 · 多模态目标、行为与压缩 3

状态：已完成；8 页。提交：`1b290e3`。

- [saprune](../../wiki/methods/saprune.md)
- [saw-int4](../../wiki/methods/saw-int4.md)
- [veq](../../wiki/methods/veq.md)
- [vlmq](../../wiki/methods/vlmq.md)
- [mbq-vlmq-comparison](../../wiki/research/mbq-vlmq-comparison.md)
- [integrated-gradients-and-quantization-sensitivity](../../wiki/theory/integrated-gradients-and-quantization-sensitivity.md)
- [multimodal-alignment-diagnostics](../../wiki/theory/multimodal-alignment-diagnostics.md)
- [quantization-ranking-stability](../../wiki/theory/quantization-ranking-stability.md)

## 第 19 批 · 多模态目标、行为与压缩 4

状态：已完成；1 页。提交：`20b79a2`。

- [quantization-reliability-and-selective-prediction](../../wiki/theory/quantization-reliability-and-selective-prediction.md)

## 第 20 批 · KV cache 与长上下文 1

状态：已完成；8 页。提交：`b121e17`。

- [flashinfer-paged-kv-write-and-nvfp4](../../wiki/implementation/flashinfer-paged-kv-write-and-nvfp4.md)
- [oscar-mixed-kv-cache-execution](../../wiki/implementation/oscar-mixed-kv-cache-execution.md)
- [kitty](../../wiki/methods/kitty.md)
- [kivi](../../wiki/methods/kivi.md)
- [kvquant](../../wiki/methods/kvquant.md)
- [nova-kv](../../wiki/methods/nova-kv.md)
- [qjl](../../wiki/methods/qjl.md)
- [turboquant](../../wiki/methods/turboquant.md)

## 第 21 批 · KV cache 与长上下文 2

状态：已完成；3 页。提交：`6b584b8`。

- [vidkv](../../wiki/methods/vidkv.md)
- [wush-kv](../../wiki/methods/wush-kv.md)
- [kv-cache-quantization-objects-and-granularity](../../wiki/theory/kv-cache-quantization-objects-and-granularity.md)

## 第 22 批 · GPU 基础与 Kernel 1

状态：已完成；8 页。提交：`d117e5b`。

- [arithmetic-intensity-and-roofline](../../wiki/fundamentals/hardware/arithmetic-intensity-and-roofline.md)
- [gpu-execution-and-memory-hierarchy](../../wiki/fundamentals/hardware/gpu-execution-and-memory-hierarchy.md)
- [fp8-and-mx-data-formats](../../wiki/fundamentals/numeric-formats/fp8-and-mx-data-formats.md)
- [gpu-kernel-computation-patterns](../../wiki/fundamentals/operators/gpu-kernel-computation-patterns.md)
- [tensor-layout-and-kernel-contracts](../../wiki/fundamentals/operators/tensor-layout-and-kernel-contracts.md)
- [triton-language-and-compilation](../../wiki/fundamentals/operators/triton-language-and-compilation.md)
- [block-scaled-matmul-execution](../../wiki/implementation/block-scaled-matmul-execution.md)
- [cuda-operator-build-and-integration](../../wiki/implementation/cuda-operator-build-and-integration.md)

## 第 23 批 · GPU 基础与 Kernel 2

状态：已完成；8 页。提交：`f7da7d4`。

- [gated-activation-cuda-triton-kernels](../../wiki/implementation/gated-activation-cuda-triton-kernels.md)
- [gguf-block-quantization-formats](../../wiki/implementation/gguf-block-quantization-formats.md)
- [gpu-async-copy-pipelines](../../wiki/implementation/gpu-async-copy-pipelines.md)
- [gpu-kernel-performance-analysis](../../wiki/implementation/gpu-kernel-performance-analysis.md)
- [kernel-configuration-and-autotuning](../../wiki/implementation/kernel-configuration-and-autotuning.md)
- [kernel-correctness-and-benchmarking](../../wiki/implementation/kernel-correctness-and-benchmarking.md)
- [marlin-batched-w4a16-gemm](../../wiki/implementation/marlin-batched-w4a16-gemm.md)
- [rmsnorm-cuda-triton-kernels](../../wiki/implementation/rmsnorm-cuda-triton-kernels.md)

## 第 24 批 · GPU 基础与 Kernel 3

状态：已完成；3 页。提交：`1fd21f5`。

- [tensor-core-quantized-gemm](../../wiki/implementation/tensor-core-quantized-gemm.md)
- [w8a8-quantization-gemm-kernels](../../wiki/implementation/w8a8-quantization-gemm-kernels.md)
- [weight-only-dequant-kernels](../../wiki/implementation/weight-only-dequant-kernels.md)

## 第 25 批 · 部署、内存与批处理 1

状态：已完成；8 页。提交：`8273c70`。

- [atom-mixed-gemm-and-paged-kv](../../wiki/implementation/atom-mixed-gemm-and-paged-kv.md)
- [qserve-implementation](../../wiki/implementation/qserve-implementation.md)
- [quantized-llm-deployment-backends](../../wiki/implementation/quantized-llm-deployment-backends.md)
- [serving-memory-and-batching](../../wiki/implementation/serving-memory-and-batching.md)
- [serving-performance-evaluation](../../wiki/implementation/serving-performance-evaluation.md)
- [sglang-attention-operator-design](../../wiki/implementation/sglang-attention-operator-design.md)
- [sglang-inference-execution](../../wiki/implementation/sglang-inference-execution.md)
- [tensor-parallel-quantization](../../wiki/implementation/tensor-parallel-quantization.md)

## 第 26 批 · 部署、内存与批处理 2

状态：已完成；6 页。提交：随本批清单更新。

- [vllm-attention-operator-design](../../wiki/implementation/vllm-attention-operator-design.md)
- [vllm-inference-execution](../../wiki/implementation/vllm-inference-execution.md)
- [atom](../../wiki/methods/atom.md)
- [disaggregated-quantization](../../wiki/methods/disaggregated-quantization.md)
- [ideogram-int8-gguf](../../wiki/methods/ideogram-int8-gguf.md)
- [qserve](../../wiki/methods/qserve.md)

## 第 27 批 · MoE、MLA 与设备执行链 1

状态：待审；8 页。

- [deepep-moe-communication-execution](../../wiki/implementation/deepep-moe-communication-execution.md)
- [deepgemm-moe-expert-execution](../../wiki/implementation/deepgemm-moe-expert-execution.md)
- [dsa-sparse-device-execution](../../wiki/implementation/dsa-sparse-device-execution.md)
- [mla-cache-and-attention-execution](../../wiki/implementation/mla-cache-and-attention-execution.md)
- [moe-expert-parallel-execution](../../wiki/implementation/moe-expert-parallel-execution.md)
- [moe-routing-and-quantized-expert-execution](../../wiki/implementation/moe-routing-and-quantized-expert-execution.md)
- [vllm-mla-context-parallel-execution](../../wiki/implementation/vllm-mla-context-parallel-execution.md)
- [vllm-mla-packed-cache-and-dsa-execution](../../wiki/implementation/vllm-mla-packed-cache-and-dsa-execution.md)

## 第 28 批 · MoE、MLA 与设备执行链 2

状态：待审；1 页。

- [w4a16-moe-expert-execution](../../wiki/implementation/w4a16-moe-expert-execution.md)

## 第 29 批 · 评测协议

状态：待审；6 页。

- [helm-evaluation-protocol](../../wiki/implementation/helm-evaluation-protocol.md)
- [lm-evaluation-harness-request-filter-and-aggregation](../../wiki/implementation/lm-evaluation-harness-request-filter-and-aggregation.md)
- [longbench-v2-context-and-scoring](../../wiki/implementation/longbench-v2-context-and-scoring.md)
- [mmlu-pro-options-and-answer-extraction](../../wiki/implementation/mmlu-pro-options-and-answer-extraction.md)
- [model-quality-evaluation](../../wiki/implementation/model-quality-evaluation.md)
- [ruler-task-complexity-and-effective-context](../../wiki/implementation/ruler-task-complexity-and-effective-context.md)

## 第 30 批 · 导航、来源研读记录与维护规范

状态：待审；4 页。

- [INDEX](../../wiki/INDEX.md)
- [READING](../../wiki/READING.md)
- [LOG](../../wiki/LOG.md)
- [README](../../wiki/README.md)

