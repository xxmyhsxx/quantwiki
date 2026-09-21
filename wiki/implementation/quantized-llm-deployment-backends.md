---
title: 量化模型的部署框架与后端支持
type: implementation
tags:
  - llm
  - deployment
  - data-format
  - serving
sources:
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/kernels/linear/__init__.py
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/layers/quantization/auto_awq.py
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/layers/quantization/__init__.py
  - raw/repositories/2026-09-21/tensorrt-llm/source/examples/quantization/README.md
  - raw/repositories/2026-09-21/mlc-llm/source/docs/compilation/configure_quantization.rst
  - raw/repositories/2026-09-21/llama-cpp/source/tools/quantize/README.md
  - raw/repositories/2026-09-21/omniserve/source/omniserve/modeling/layers/quantized_linear/w4a8_linear.py
updated: 2026-09-22
---

# 量化模型的部署框架与后端支持

量化方法给出的是算法与一串参数；要让模型真正跑起来，还需要格式、加载路径、内核与运行场景四段配置同时对上。同一个算法名在不同框架里可能对应不同内核，同一位宽标签在不同框架里可能对应不同的量化粒度。本页把这条链路拆开，并记录四个主流推理框架在固定快照下实际暴露的量化支持面。

本页依据 vLLM（快照 `568afb3a`）、TensorRT-LLM（快照 `d567f924`）、MLC-LLM（快照 `6474f7db`）、llama.cpp（快照 `b820cc8e`）的官方代码或文档，以及 [QServe](../methods/qserve.md) 论文与实现中对运行时要求的描述。所有内容为单一快照的只读核对，本轮没有构建、导出或运行任何框架。

部署时还需核对 [张量并行的分片与量化元数据](tensor-parallel-quantization.md)。后端能够加载后，按 [模型质量协议](model-quality-evaluation.md)与 [服务性能协议](serving-performance-evaluation.md)分别确认质量和收益，不能只用“支持该格式”作为完成依据。

## 1. 部署链路的环节

量化模型上线至少要经过四段，任何一段不匹配都会退回未量化路径或直接报错：

1. **量化与校准**：确定位宽、粒度、对称性与尺度来源，产出量化权重与元数据；
2. **导出格式**：把权重、尺度、零点、分组信息与全部元数据序列化成某个具体格式；
3. **框架加载**：框架按格式名或自动探测加载配置，决定用哪套内核；
4. **运行场景**：prefill 还是 decode、批量大小、序列长度、是否量化 KV，决定实际收益落在哪里。

[量化矩阵乘法的缩放与执行路径](quantized-matmul-scaling-execution.md) 解释第 4 段的通用原理；[GGUF 块量化存储格式](gguf-block-quantization-formats.md) 是第 2 段的一个完整实例；本页关注第 2、3 段在不同框架之间的差异。

## 2. 各框架的支持面

| 框架 | 主要存储与格式 | 典型位宽配置 | 加载方式 |
|---|---|---|---|
| llama.cpp | GGUF，含 k-quant／i-quant 与 MXFP4／NVFP4 块 | 由块结构与层间混合决定 | 单文件加载，按张量类型分派 |
| vLLM | 框架内定义的多种量化配置 | 由方法名决定，含 W4A16、W8A8、FP8、MXFP4 等 | 方法名或自动探测 |
| TensorRT-LLM | 量化 toolkit 产出 checkpoint，再编译为引擎 | qformat 指定，含 FP8、INT8 SQ、INT4 AWQ、W4A8 | 量化后需 `trtllm-build` 编译 |
| MLC-LLM | 编译产物，量化模式写在编译配置里 | 权重 q3f16／q4f16／q4f16_awq，W/A 为 FP8 | 先编译再运行，可插入校准步骤 |

这张表本身就是一条结论：**量化方法的部署成本取决于框架把它表达为什么**。llama.cpp 把量化固定在文件里，vLLM 把它表达为一个可探测的方法，TensorRT-LLM 把它当作编译期配置，MLC-LLM 把它当作编译选项并支持独立的校准运行。

## 3. vLLM：方法名清单

vLLM 在 `vllm/model_executor/layers/quantization/__init__.py` 中定义了可用量化方法的字面量集合。固定快照 `568afb3a` 下的清单包括：

```text
awq, auto_awq, fp8, fbgemm_fp8, fp_quant, modelopt, modelopt_fp4,
modelopt_mxfp8, modelopt_mixed, auto_gptq, gptq, gptq_marlin,
awq_marlin, humming, compressed-tensors, bitsandbytes, experts_int8,
quark, moe_wna16, torchao, inc, mxfp4, gpt_oss_mxfp4, deepseek_v4_fp8,
online（含 fp8_per_tensor、fp8_per_block 等在线量化简写）
```

两个值得注意的命名模式：

**量化方法名不与内核一一对应。** 此快照非 CPU 的 AWQ checkpoint 可把 `awq`、`awq_marlin` 等选择统一为 `auto_awq`，随后按配置和层形状选择普通 AWQ 或 MPLinear 路径；后者再选择可实现的 kernel。`AutoAWQMarlinLinearMethod` 的类名也不保证最终调用 Marlin。性能记录需保留最终后端、环境开关和输入规模；具体加载与执行条件见 [AWQ 实现](awq-implementation.md#5-跨引擎vllm)，不能从注册名称清单直接推断运行路径。

**格式名与生态来源并不统一。** `compressed-tensors`、`modelopt`、`quark`、`bitsandbytes`、`torchao`、`inc` 对应各自的工具链与序列化格式，`fp8` 与 `fbgemm_fp8` 则区分了 FP8 的两条实现路径。看到「vLLM 支持 X」时，需要进一步确认是哪条内核路径。

## 4. TensorRT-LLM：量化格式与自动搜索

TensorRT-LLM 通过 `examples/quantization/quantize.py` 的 `--qformat` 指定方案（快照 `d567f924` 的文档），各选项语义为：

| `qformat` | 权重处理 | 激活处理 |
|---|---|---|
| `nvfp4` | 块大小 16 的 NVFP4 | 全局尺度校准 |
| `fp8` | 逐张量 FP8 | 逐张量校准 |
| `fp8_pc_pt` | 逐通道 FP8 | 逐 token 校准并量化 |
| `int8_sq` | 平滑后逐通道 INT8 | 逐张量校准 |
| `int4_awq` | 重缩放后按块 INT4（block size 由参数给出） | 不量化 |
| `w4a8_awq` | 同 `int4_awq` | 逐张量校准 |
| `int8_wo`、`int4_wo` | 量化推迟到引擎构建阶段 | 不量化 |
| `full_prec` | 不量化 | 不量化 |

`int4_awq` 与 `w4a8_awq` 的对照说明了一件事：同一个权重量化算法，可以因为激活是否量化而成为两个部署选项。把论文里的「AWQ 是 weight-only 方法」直接对应到某个框架配置时，这一层区分必须保留。

KV cache 通过独立的 `--kv_cache_dtype` 选择 `int8`、`fp8` 或不量化，说明 KV 位宽在部署里是与权重、激活并列的第三个自由度——与 [混合精度分配](../theory/mixed-precision-allocation.md) 中「对象不同、预算不同」的判断一致。

框架还提供量化方案的自动搜索：`--autoq_format` 在 `fp8`、`int4_awq`、`w4a8_awq`、`int8_sq` 中挑选，`--auto_quantize_bits` 给出平均权重位宽的约束。文档说明 `int8_sq` 与 `fp8` 不能同时使用，且实际选择由优化问题求解给出。这相当于把 [逐层精度分配](../theory/mixed-precision-allocation.md) 的做法内置进框架，只是其候选集合与代价模型由框架固定。

## 5. MLC-LLM：把校准变成独立步骤

MLC-LLM 的量化模式写成短代码（快照 `6474f7db` 的文档）。权重侧格式为 `qAfB(_id)`：`A` 是权重位数、`B` 是激活存储位数、`_id` 区分算法（对称、非对称、AWQ 等）。当前可选 `q0f16`、`q0f32`、`q3f16_1`、`q4f16_1`、`q4f32_1` 与标注为不稳定的 `q4f16_awq`。权重与激活都量化时，CUDA 上提供 `e4m3_e4m3_f16` 与 `e5m2_e5m2_f16`，即两种 FP8（见 [FP8 与 MX 数值格式](../fundamentals/numeric-formats/fp8-and-mx-data-formats.md)），层输出保持 FP16 后重新量化为 FP8。文档把默认分组量化算法的来源指向 k-bit inference scaling laws 与 LUT-GEMM 两篇工作。

它的校准流程值得单独记录，因为这是本页四个框架中唯一把校准暴露为独立运行阶段的：

1. 以校准模式（如 `e4m3_e4m3_f16_max_calibrate`）生成配置并转换权重；
2. 用 `mlc_llm calibrate` 在 ShareGPT 一类数据上运行校准模型，**把统计结果就地写回权重文件**；文档说明该阶段需要关闭 CUDA graph；
3. 用目标量化格式重新生成配置并编译模型，权重无需再次转换。

注意第 2 步是「运行模型并写回量化参数」，不是离线统计脚本。[校准数据与量化范围选择](../theory/calibration-and-range-selection.md) 要求的样本、掩码与用途记录在这里同样是复现的前置条件。

## 6. llama.cpp：格式即部署

llama.cpp 的量化完全体现在 GGUF 文件里：格式名（`Q4_K_M`、`IQ1_M` 等）直接决定块结构与元数据，运行时的后端（CPU、CUDA、Metal、Vulkan）按张量类型分派。量化工具提供逐张量覆盖与重要性矩阵输入，细节见 [GGUF 块量化存储格式](gguf-block-quantization-formats.md)。它的特点是把「选择哪个格式、哪些层降精度」交给使用者在导出时决定，而不是运行时的配置项。

与本页其他框架相比，这条路径的部署资产是文件本身，跨语言与跨设备的可移植性来自格式的标准化程度。

## 7. 加载之外：运行时对量化的额外要求

这一节的约束都来自服务系统的内存与调度结构：缓存被切成固定大小的块、批量按迭代粒度调整、显存不足时需要抢占。这些机制本身的工作原理见 [推理服务的内存管理与批处理](serving-memory-and-batching.md)；理解它们才能判断某个量化方案是「能加载」还是「能在服务里跑出收益」。 vLLM 从 token 调度、KV 分配到紧凑输入的源码链见 [vLLM 推理执行](vllm-inference-execution.md)；Attention 的缓存契约、后端接口、分段归约与图执行见 [vLLM 算子设计](vllm-attention-operator-design.md)。 SGLang 的对应执行链从 [Radix 缓存与调度](sglang-inference-execution.md)进入 [KV 索引和 Attention 算子](sglang-attention-operator-design.md)，解释前缀命中、缓存保护与 extend/decode 形状怎样影响实际工作。

**KV cache 的量化参数存放在哪里。** [QServe](../methods/qserve.md) 采用与 vLLM、TensorRT-LLM 相同的分页 KV 布局，但因为 KV 位宽更低而使用逐 head 动态量化，于是把每个 head 的 fp16 尺度与零点放在分页中量化 KV 特征之后，支持运行时更新。这说明「KV4」不只是一个位宽，还包含一套页内布局约定；换成静态逐张量量化可以省掉动态估计，但精度条件随之变化。

**同一模型需要不同的配置。** QServe 在 A100 上使用逐通道权重、在 L40S 上使用按组权重，原因是两块 GPU 的 CUDA 核心承担反量化开销的能力不同。部署配置与硬件绑定，不是模型属性。

**推理时不再重算的部分要写清。** 各类校准统计、搜索出的缩放与裁剪阈值在部署时都已固化；而令牌级动态量化（如逐 token 激活尺度、逐 head KV 尺度）仍需运行时统计。把二者混为一谈会导致对开销的错误预期，判断依据见 [执行路径](quantized-matmul-scaling-execution.md)。

## 8. 实践中的最小链路

按上述拆分，量化一个模型并部署通常包含这些决策点，缺一项都可能在导出或加载阶段失败：

1. 明确对象与场景：权重、激活、KV cache 分别用什么位宽，目标是显存、吞吐还是延迟；
2. 选择量化方法与校准数据，记录版本、样本与掩码；
3. 确认目标框架接受的格式名，并核对位宽、粒度、对称性是否在该框架的支持范围内；
4. 若框架需要编译（TensorRT-LLM、MLC-LLM），把量化参数与编译配置一并保存；
5. 在同一硬件、批量、序列长度与测量口径下比较基线与量化版本，并单独报告 KV 与权重部分的变化。

第 3 步最容易出错：方法支持与内核支持是两件事，格式可加载也不等于走的是高效路径。

## 9. 局限与未验证

- 本页只读取了各框架的量化方法清单、工具文档与配置说明，未构建或运行任何框架；「支持」指该快照下存在对应条目，不代表其在所有硬件上可用或高效。
- 各框架的版本迭代较快，格式清单与默认配置随版本变化；引用时应固定快照或版本号。
- 未核对 vLLM 各方法对应的具体内核选择逻辑、TensorRT-LLM 的插件实现、MLC-LLM 的后端代码生成，也未验证跨框架的格式互转是否保真。
- 未收集各框架在统一硬件与场景下的对照性能数据，本页不给出性能排序。

## 来源身份

下表用于在没有本地资料库时辨识来源；具体论述的章节、公式、图表或代码位置见正文。

| 来源 | 版本或快照 | 说明 |
| --- | --- | --- |
| [vllm-project/vllm](https://github.com/vllm-project/vllm/tree/568afb3a13806beb53bb2e6bd518269357b237c0) | `568afb3a13806beb53bb2e6bd518269357b237c0` | — |
| [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/tree/d567f924b729dc81465f2d2f762cabb893e28602) | `d567f924b729dc81465f2d2f762cabb893e28602` | — |
| [mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm/tree/6474f7dbfa9bf18fdfd3ef468b227c0706eb9008) | `6474f7dbfa9bf18fdfd3ef468b227c0706eb9008` | — |
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/tree/b820cc8e6f94843d32f92c8ebd7db837dae2bd8b) | `b820cc8e6f94843d32f92c8ebd7db837dae2bd8b` | — |
| [mit-han-lab/omniserve](https://github.com/mit-han-lab/omniserve/tree/02b2925aa6fa3b92b06316a1524b7f38922cd9c8) | `02b2925aa6fa3b92b06316a1524b7f38922cd9c8` | — |
