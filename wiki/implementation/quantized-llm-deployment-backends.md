---
title: 量化模型的部署框架与后端支持
type: implementation
tags:
  - llm
  - deployment
  - data-format
  - serving
sources:
  - raw/articles/2026-09-24/tensorrt-llm-quantization
  - raw/articles/2026-09-24/vllm-v1-architecture
  - raw/articles/2026-09-21/vllm-release-notes
  - raw/repositories/2026-09-21/vllm
  - raw/repositories/2026-09-21/tensorrt-llm
  - raw/repositories/2026-09-21/mlc-llm
  - raw/repositories/2026-09-21/llama-cpp
  - raw/repositories/2026-09-21/omniserve
  - raw/repositories/2026-09-21/sglang
updated: 2026-10-15
---

# 量化模型的部署框架与后端支持

量化方法给出的是算法与一串参数；要让模型真正跑起来，还需要格式、加载路径、内核与运行场景四段配置同时对上。同一个算法名在不同框架里可能对应不同内核，同一位宽标签在不同框架里可能对应不同的量化粒度。本页把这条链路拆开，并记录多个主流推理框架在固定快照下实际暴露的量化支持面。

本页依据 SGLang（快照 `2f730e29`）、vLLM（快照 `568afb3a`）、TensorRT-LLM（快照 `d567f924`）、MLC-LLM（快照 `6474f7db`）、llama.cpp（快照 `b820cc8e`）的官方代码或文档，以及 [QServe](../methods/qserve.md) 论文与实现中对运行时要求的描述。各框架按各自固定快照核对；另补vLLM v0.26.0发布声明及TensorRT-LLM量化文档§4.1的`a6f41514`版本。这里核文档与注明的源码，没有构建、导出或运行任何框架。

部署时还需核对 [张量并行的分片与量化元数据](tensor-parallel-quantization.md)。后端能够加载后，按 [模型质量协议](model-quality-evaluation.md)与 [服务性能协议](serving-performance-evaluation.md)分别确认质量和收益，不能只用“支持该格式”作为完成依据。

多个引擎组成在线服务后，框架/格式之外还要核请求路由、阶段副本和 KV 传输依赖；[分布式服务编排](distributed-llm-serving-orchestration.md)说明这些责任及文档版本配套，具体格式/加载/内核仍按本页的固定来源判断。

## 1. 部署链路的环节

部署链路包含四段，需逐段核对参数与接口：

1. **量化与校准**：确定位宽、粒度、对称性与尺度来源，产出量化权重与元数据；
2. **导出格式**：把权重、尺度、零点、分组信息与全部元数据序列化成某个具体格式；
3. **框架加载**：框架按格式名或自动探测加载配置，决定用哪套内核；
4. **运行场景**：prefill 还是 decode、批量大小、序列长度、是否量化 KV，决定实际收益落在哪里。

[量化矩阵乘法的缩放与执行路径](quantized-matmul-scaling-execution.md) 解释第 4 段的通用原理；[GGUF 块量化存储格式](gguf-block-quantization-formats.md) 是第 2 段的一个完整实例；本页关注第 2、3 段在不同框架之间的差异。

ModelOpt的recipe、校准尺度、搜索恢复状态与导出checkpoint各有什么作用，见[ModelOpt PTQ与消费接口](modelopt-ptq-calibration-and-checkpoint.md)。该页用固定vLLM静态FP8实例解释加载后准备，支持范围仍按各自版本核对。

### 1.1 统一配置、模块 prefix 与初始化分片

固定官方架构文档用 `VllmConfig` 统一承载engine、executor、worker、runner和模型需要的配置，模型constructor接收 `vllm_config` 与 `prefix`。Prefix标识checkpoint里的模块位置，可以让不同子模块采用不同量化配置；以所选Llama/AutoAWQ为例，prefix传给线性层的 `get_quant_method`，被排除的模块返回非量化方法。共享配置结构不等于不同进程共享同一个Python内存对象。（文档Class Hierarchy；源码 `models/llama.py:LlamaForCausalLM`、`layers/linear.py:LinearBase`、`quantization/auto_awq.py:AutoAWQConfig.get_quant_method`。）

初始化先建立本rank所需的参数表示，避免先在每张GPU构造完整浮点权重再切分。普通列并行的 `ColumnParallelLinear` 先确定输出分片尺寸，再由量化方法 `create_weights` 建立存储；loader根据参数轴加载对应分片，详细码值/尺度/零点分片见[张量并行](tensor-parallel-quantization.md)。这是参数存储设计的理由，不是对全部加载瞬时内存的保证；文档405B/810GB/16卡例子为理想容量说明，并非本轮测量。

这里的“初始化量化”不表示loader重跑AWQ校准搜索，也不表示全部转换都在constructor完成。所选 `model_loader/utils.py:process_weights_after_loading` 仍调用量化方法处理加载后的repack/quantization；具体AWQ格式链见[AWQ实现](awq-implementation.md)。同文件 `initialize_model` 对旧constructor还保留warning和参数猜测兼容分支，文档的keyword-only说明不能推成所有旧签名模型必然被拒收。

## 2. 各框架的支持面

| 框架 | 主要存储与格式 | 典型位宽配置 | 加载方式 |
|---|---|---|---|
| llama.cpp | GGUF，含 k-quant／i-quant 与 MXFP4／NVFP4 块 | 由块结构与层间混合决定 | 单文件加载，按张量类型分派 |
| SGLang | 框架注册的量化配置，如 GPTQ、W8A8、QoQ | 位宽、分组、对称性及设备约束按具体配置 | 方法配置 → 参数加载/重排 → 内核 |
| vLLM | 框架内定义的多种量化配置 | 由方法名决定，含 W4A16、W8A8、FP8、MXFP4 等 | 方法名或自动探测 |
| TensorRT-LLM | ModelOpt checkpoint；旧版文档另有 checkpoint → 引擎流程 | 格式、W/A/KV组合与设备按对应版本核对 | `a6f41514`默认PyTorch后端直接加载；`d567f924`所述引擎路径经`trtllm-build` |
| MLC-LLM | 编译产物，量化模式写在编译配置里 | 权重 q3f16／q4f16／q4f16_awq，W/A 为 FP8 | 先编译再运行，可插入校准步骤 |

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

**量化方法名不与内核一一对应。** 此快照非 CPU 的 AWQ checkpoint 可把 `awq`、`awq_marlin` 等选择统一为 `auto_awq`，随后按配置和层形状选择普通 AWQ 或 MPLinear 路径；后者再选择可实现的 kernel。`AutoAWQMarlinLinearMethod` 的类名也不保证最终调用 Marlin。性能记录需保留最终后端、环境开关和输入规模；具体加载与执行条件见 [AWQ 实现](awq-implementation.md#5.%20%E8%B7%A8%E5%BC%95%E6%93%8E%EF%BC%9AvLLM)，不能从注册名称清单直接推断运行路径。

**格式名与生态来源并不统一。** `compressed-tensors`、`modelopt`、`quark`、`bitsandbytes`、`torchao`、`inc` 对应各自的工具链与序列化格式，`fp8` 与 `fbgemm_fp8` 则区分了 FP8 的两条实现路径。看到「vLLM 支持 X」时，需要进一步确认是哪条内核路径。

同一tag的v0.26.0发布说明还列出 `head_dtype` 配置FP32 `lm_head`（含LoRA路径），以及sampler不再把logits升为FP32（Highlights/Engine Core/Hardware & Performance，#48390/#48525/#48641）。两项改动分别作用于输出head与采样阶段，head精度不能替代采样dtype记录，取消sampler转换也不能反推head精度。作者称FP32 head改善generation head精度，但没有给出统一任务、模型配置或收益幅度。

### 3.1 已核对到实现的算法与表示

下表限定 vLLM `568afb3a` 和 SGLang `2f730e29` 的内置代码，并区分算法名与消费格式。有论文、有官方仓库、有注册项、能够实际运行，是四种不同证据；本表不宣称已完成最后一项。

| 已有方法与代码 | vLLM 的已核对路径 | SGLang 的已核对路径 | 接入时最重要的边界 |
| --- | --- | --- | --- |
| [GPTQ](gptq-implementation.md)，IST-DASLab/gptq | gptq/gptq_marlin/auto_gptq → AutoGPTQConfig → 后端选择器；该入口接受对称 4/8-bit | 原生 GPTQ 配置声明 2/3/4/8-bit，Marlin 对称 4/8-bit；仍受形状/设备约束 | 原始仓库保存入口是专用 3-bit 格式，不能当作通用服务 checkpoint；g_idx、零点版本和 TP 要匹配 |
| [SmoothQuant/W8A8](smoothquant-implementation.md)，smoothquant + torch-int | compressed-tensors 的 INT8 W8A8 scheme；静态 tensor 或动态 token 激活 | w8a8_int8 Dense 使用动态逐 token 激活、逐输出通道权重 | 两者都不是在 loader 中重跑 SmoothQuant；官方 OPT/torch-int 导出与服务格式不直接等价 |
| [QServe/QoQ](qserve-implementation.md)，OmniServe | 该快照内置注册表没有 qoq 项 | qoq → QoQLinearMethod → 两种 W4A8 GEMM，g=-1/128、FP16、SM80+ | 文件必须已经按内核布局重排；这里只实现 Linear 消费，不等于论文整套 KV4 系统 |

AWQ 的官方代码、格式转换与内核链已在[既有实现页](awq-implementation.md)展开。内置注册表没有某个论文名，也不能据此否定其可经其他表示或插件接入，需另核对转换与加载证据。

其中 vLLM 的 GPTQ 分派修正了“两个配置名就是两种固定内核”的简化：最终后端还受能力检查、逐层设置、设备和显式后端选项影响。SGLang 原生 GPTQ 甚至可能随 token 数切换到先重建 FP16 权重再用 cuBLAS，故性能记录需要保存实际执行路径。

发布说明另报告限制Marlin的FP4 MoE权重重排峰值及NVFP4 MoE权重加载峰值（Quantization，#47851/#46276）。这针对加载和布局准备阶段，不能当作运行常驻显存或workspace结论，也不能推广到AWQ整数W4A16。已有[AWQ专家链](w4a16-moe-expert-execution.md)选择非对称 `kU4/G128`，逐专家调用 `awq_marlin_repack`；它与这里的浮点FP4/NVFP4分别保留证据。发布条目未给峰值字节数、模型规模或具体限制机制。

## 4. TensorRT-LLM：量化配置与消费条件

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

### 4.1 PyTorch加载入口、KV配置与支持矩阵

固定`a6f41514`的量化文档以默认 **PyTorch backend** 的`LLM(model=...)`加载ModelOpt预量化checkpoint。上一节`d567f924`的量化脚本和引擎构建属于另一固定文档流程，不能写成这个直接调用之前统一必需的步骤；短示例也没有展开加载器内部编译或布局准备。Producer的校准/导出职责见[ModelOpt PTQ](modelopt-ptq-calibration-and-checkpoint.md)，消费端是否接受某个组合仍需按本版本核对。（Usage、Running Pre-quantized Models。）

| KV配置 | 本版本文档要求 | 判断边界 |
| --- | --- | --- |
| `KvCacheConfig(dtype='fp8')` | 即使checkpoint未默认启用FP8 KV，也可手动设置 | 文中未给这种入口的尺度默认值或校准替代机制 |
| `KvCacheConfig(dtype='nvfp4')` | 先用ModelOpt离线生成对应checkpoint，且当前只支持FP8 **权重/激活主体** | NVFP4 KV不能据格式名字推广到任意NVFP4或混合精度body |

`dtype='nvfp4'`配置的是GPU上的 **active KV cache**。文档另将只在Host/Disk存储Pages时用NVFP4压缩、GPU活动缓存保持普通runtime dtype的功能指向KV Cache Compression；该外链未保存，因此这里不推导转移、恢复流程或收益。数值量化对象及粒度复用[KV基础](../theory/kv-cache-quantization-objects-and-granularity.md)。（NVFP4 KV Cache。）

模型支持矩阵与硬件支持矩阵约束不同，需同时核对。保存硬件表给SM100/103的NVFP4 KV标Y，SM120未标支持；Ampere的FP8 KV标Y，三个FP8 GEMM粒度列却均未标支持，说明缓存dtype不能代替矩阵乘能力。SM90与SM100/103的block-FP8还采用不同尺度编码，见[块缩放接口](block-scaled-matmul-execution.md#TensorRT-LLM%E7%9A%84block-FP8%E5%B0%BA%E5%BA%A6%E4%B8%8E%E7%A1%AC%E4%BB%B6)。同为Blackwell或FP8的标签不足以确定可消费的数据。（Hardware Support Matrix及其note。）

这些表的各列单独标Y，不证明任意联合配置；`.`未给独立图例，本文仅解释为“该保存矩阵未标支持”。例如recipe目录/硬件表列出W4A8 GPTQ，但模型表没有对应标Y的模型，不能据此补出一个支持实例。多模态表默认vision component为FP16，方法支持由language component决定，不表示整模型统一低精度。直接示例的FP8模型名没有明确per-tensor/rowwise/block粒度，也不能用名字补表项。（Model Support Matrix及note。）

本条完整核读144行正文和两份矩阵，来源README登记完整`a6f4151497372f14a5261113a12be2b406e9b5fc`；站点短标识`fca831e`另行保留。示例clone没有pin ModelOpt producer，不能自行认定与本Wiki所读`f2f0d695`配套。匹配loader、checkpoint字段、GEMM/attention kernel、外链及实际运行未核。

## 5. MLC-LLM：把校准变成独立步骤

MLC-LLM 的量化模式写成短代码（快照 `6474f7db` 的文档）。权重侧格式为 `qAfB(_id)`：`A` 是权重位数、`B` 是激活存储位数、`_id` 区分算法（对称、非对称、AWQ 等）。当前可选 `q0f16`、`q0f32`、`q3f16_1`、`q4f16_1`、`q4f32_1` 与标注为不稳定的 `q4f16_awq`。权重与激活都量化时，CUDA 上提供 `e4m3_e4m3_f16` 与 `e5m2_e5m2_f16`，即两种 FP8（见 [FP8 与 MX 数值格式](../fundamentals/numeric-formats/fp8-and-mx-data-formats.md)），层输出保持 FP16 后重新量化为 FP8。文档把默认分组量化算法的来源指向 k-bit inference scaling laws 与 LUT-GEMM 两篇工作。

它的校准流程值得单独记录，因为它明确把校准暴露为独立运行阶段：

1. 以校准模式（如 `e4m3_e4m3_f16_max_calibrate`）生成配置并转换权重；
2. 用 `mlc_llm calibrate` 在 ShareGPT 一类数据上运行校准模型，将参数写入 `--output` 指定的缓存目录。代码启用 `update_if_exists=True`；只有该目录指向原有权重缓存时才是就地更新。文档说明该阶段需要关闭 CUDA graph；
3. 用目标量化格式重新生成配置并编译模型，权重无需再次转换。

注意第 2 步是「运行模型并写回量化参数」，不是离线统计脚本。[校准数据与量化范围选择](../theory/calibration-and-range-selection.md) 要求的样本、掩码与用途记录在这里同样是复现的前置条件。

## 6. llama.cpp：格式即部署

llama.cpp 在导出时按 `Q4_K_M`、`IQ1_M` 等策略选择各张量的量化类型；GGUF 保存实际类型、块数据与元数据，运行时后端（CPU、CUDA、Metal、Vulkan）按张量类型分派。量化工具提供逐张量覆盖与重要性矩阵输入，细节见 [GGUF 块量化存储格式](gguf-block-quantization-formats.md)。它把「选择哪个格式、哪些层降精度」交给使用者在导出时决定。

与本页其他框架相比，这条路径的部署资产是文件本身，跨语言与跨设备的可移植性来自格式的标准化程度。

## 7. 加载之外：运行时对量化的额外要求

这一节的约束都来自服务系统的内存与调度结构：缓存被切成固定大小的块、批量按迭代粒度调整、显存不足时需要抢占。这些机制本身的工作原理见 [推理服务的内存管理与批处理](serving-memory-and-batching.md)；理解它们才能判断某个量化方案是「能加载」还是「能在服务里跑出收益」。 vLLM 从 token 调度、KV 分配到紧凑输入的源码链见 [vLLM 推理执行](vllm-inference-execution.md)；Attention 的缓存契约、后端接口、分段归约与图执行见 [vLLM 算子设计](vllm-attention-operator-design.md)。 SGLang 的对应执行链从 [Radix 缓存与调度](sglang-inference-execution.md)进入 [KV 索引和 Attention 算子](sglang-attention-operator-design.md)，解释前缀命中、缓存保护与 extend/decode 形状怎样影响实际工作。

**KV cache 的量化参数存放在哪里。** [QServe](../methods/qserve.md) 采用与 vLLM、TensorRT-LLM 相同的分页 KV 布局，但因为 KV 位宽更低而使用逐 head 动态量化，于是把每个 head 的 fp16 尺度与零点放在分页中量化 KV 特征之后，支持运行时更新。这说明「KV4」不只是一个位宽，还包含一套页内布局约定；换成静态逐张量量化可以省掉动态估计，但精度条件随之变化。

v0.26.0发布说明列出 `speculative_config` 独立的 `kv_cache_dtype`，以及MLA的 `kv_cache_dtype_skip_layers`（Engine Core/Quantization，#48787/#47309）。包含推测模型或逐层覆盖时，需要说明target、draft与受覆盖层各自的KV dtype，不能只保存一个无对象归属的“KV位宽”。这些名称表示独立配置轴，发布条目没有交代默认值、继承、允许组合、尺度粒度或缓存layout。

**同一模型需要不同的配置。** QServe 在 A100 上使用逐通道权重、在 L40S 上使用按组权重，原因是两块 GPU 的 CUDA 核心承担反量化开销的能力不同。部署配置与硬件绑定，不是模型属性。

**推理时不再重算的部分要写清。** 各类校准统计、搜索出的缩放与裁剪阈值在部署时都已固化；而令牌级动态量化（如逐 token 激活尺度、逐 head KV 尺度）仍需运行时统计。把二者混为一谈会导致对开销的错误预期，判断依据见 [执行路径](quantized-matmul-scaling-execution.md)。

## 8. 实践中的最小链路

按上述拆分，量化一个模型并部署通常包含这些决策点，缺一项都可能在导出或加载阶段失败：

1. 明确对象与场景：权重、激活、KV cache 分别用什么位宽，目标是显存、吞吐还是延迟；
2. 选择量化方法与校准数据，记录版本、样本与掩码；
3. 确认目标框架接受的格式名，并核对位宽、粒度、对称性是否在该框架的支持范围内；
4. 若使用引擎构建或编译路径，把量化参数与编译配置一并保存；
5. 在同一硬件、批量、序列长度与测量口径下比较基线与量化版本，并单独报告 KV 与权重部分的变化。

第 3 步最容易出错：方法支持与内核支持是两件事，格式可加载也不等于走的是高效路径。

## 9. 局限与未验证

已核对的算法与格式路径见对应实现页；其余方法、TensorRT-LLM 插件、MLC-LLM 后端代码生成和跨框架格式互转尚未核验。各框架缺少统一场景的对照性能数据，不能据支持面排序。

## 来源身份

- vLLM Release v0.26.0（`https://github.com/vllm-project/vllm/releases/tag/v0.26.0`），snapshot-2026-08-20，获取于2026-08-20T02:37:51+08:00；head/sampler精度、独立KV配置和FP4 MoE加载/重排峰值声明。

| 来源 | 版本或快照 | 说明 |
| --- | --- | --- |
| Quantization — TensorRT-LLM（`https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html`） | `source-commit-a6f4151497372f14a5261113a12be2b406e9b5fc`；获取于2026-09-24T02:12:20+08:00 | 正文144行全文；默认PyTorch加载、W/A/KV条件、模型/硬件矩阵及尺度编码；匹配实现、外链和运行未核。 |
| vLLM V1 Architecture Overview（`https://docs.vllm.ai/en/latest/design/arch_overview/`） | `vllm-main-f9ad9dd6b495f7492fa975e0ac48164b090d99ad`；获取于2026-09-24T02:12:20+08:00 | Class Hierarchy的统一配置、prefix及初始化分片理由；实际加载后处理/兼容分支另按568afb源码核对。 |
| vllm-project/vllm（`https://github.com/vllm-project/vllm/tree/568afb3a13806beb53bb2e6bd518269357b237c0`） | `568afb3a13806beb53bb2e6bd518269357b237c0` | — |
| NVIDIA/TensorRT-LLM（`https://github.com/NVIDIA/TensorRT-LLM/tree/d567f924b729dc81465f2d2f762cabb893e28602`） | `d567f924b729dc81465f2d2f762cabb893e28602` | — |
| mlc-ai/mlc-llm（`https://github.com/mlc-ai/mlc-llm/tree/6474f7dbfa9bf18fdfd3ef468b227c0706eb9008`） | `6474f7dbfa9bf18fdfd3ef468b227c0706eb9008` | — |
| ggml-org/llama.cpp（`https://github.com/ggml-org/llama.cpp/tree/b820cc8e6f94843d32f92c8ebd7db837dae2bd8b`） | `b820cc8e6f94843d32f92c8ebd7db837dae2bd8b` | — |
| mit-han-lab/omniserve（`https://github.com/mit-han-lab/omniserve/tree/02b2925aa6fa3b92b06316a1524b7f38922cd9c8`） | `02b2925aa6fa3b92b06316a1524b7f38922cd9c8` | — |
| sgl-project/sglang（`https://github.com/sgl-project/sglang/tree/2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc`） | `2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc` | 注册项与所选算法/格式路径 |
