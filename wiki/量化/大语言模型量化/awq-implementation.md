---
title: AWQ 的实现核对：从量化脚本到 vLLM 与 SGLang
slug: awq-implementation
sources:
  - raw/repositories/quantization/2026-06-mit-han-lab-llm-awq-d6e797a4/source/awq/quantize/qmodule.py
  - raw/repositories/quantization/2026-06-mit-han-lab-llm-awq-d6e797a4/source/awq/quantize/pre_quant.py
  - raw/repositories/quantization/2026-06-mit-han-lab-llm-awq-d6e797a4/source/awq/quantize/auto_scale.py
  - raw/repositories/quantization/2026-06-mit-han-lab-llm-awq-d6e797a4/source/awq/quantize/auto_clip.py
  - raw/repositories/quantization/2026-06-mit-han-lab-llm-awq-d6e797a4/source/awq/quantize/quantizer.py
  - raw/repositories/quantization/2026-06-mit-han-lab-llm-awq-d6e797a4/source/awq/kernels/csrc/quantization/gemv_cuda.cu
  - raw/repositories/quantization/2026-06-mit-han-lab-llm-awq-d6e797a4/source/awq/kernels/csrc/quantization/gemm_cuda_gen.cu
  - raw/repositories/serving/2026-08-vllm-project-vllm-568afb3a/source/vllm/model_executor/layers/quantization/auto_awq.py
  - raw/repositories/serving/2026-06-sgl-project-sglang-2f730e29/source/python/sglang/jit_kernel/awq_dequantize.py
updated: 2026-09-17
---

# AWQ 的实现核对：从量化脚本到 vLLM 与 SGLang

方法页回答 AWQ 为什么这样设计、证据支持什么；本页回答代码层面的事情：**量化脚本产出什么形状的权重、打包约定是什么、内核从哪里进入、同一个 AWQ checkpoint 在 vLLM 与 SGLang 里各走哪条路径**。这些内容决定一份 AWQ 权重能不能被目标引擎加载，以及加载后走的是哪类内核。

方法机制、论文数值与证据边界见 [[awq|AWQ：激活感知的权重量化]]。本页只做代码核对，全部结论到 code-read 级，未运行任何脚本或内核。

## 1. 三个快照与读取范围

| 快照 | 固定版本 | 本页读取范围 |
|---|---|---|
| mit-han-lab/llm-awq | `d6e797a42b9ef7778de8ee2352116e0f48a78d61` | `awq/quantize/` 的 qmodule、pre_quant、auto_scale；`awq/kernels/csrc/` 的目录结构 |
| vllm-project/vllm | `568afb3a13806beb53bb2e6bd518269357b237c0` | `model_executor/layers/quantization/auto_awq.py` 与相关 marlin/awq_triton 工具 |
| sgl-project/sglang | `2f730e29`（2026-06 快照） | `srt/layers/quantization/awq/`、`jit_kernel/awq_dequantize.py` |

未读取：TinyChat 的 serving 与 stream_generators、AWQ 内核的 CUDA 实现细节（只核对目录与调用入口）、两套启动脚本。因此本页不评价任何内核的性能或数值正确性。

## 2. 原仓库：量化脚本链

一条完整的 AWQ 量化流程由三个脚本组成，彼此通过「先缓存输入、再搜索、后应用」耦合：

1. **缓存校准激活。** `pre_quant.py` 按 block 挂 forward hook 收集线性层输入，并把未施加缩放的 block 输出作为下一块的输入（方法页 §5.3 已记录该顺序）。
2. **搜索缩放。** `auto_scale.py` 枚举指数候选，对每个候选按计算图规则应用缩放、模拟量化、比较参考模块输出误差，选最优。方法页已记录该实现对 α 网格、缩放下限与归一化的具体处理。
3. **搜索并应用裁剪，再选择输出路径。** `auto_clip.py` 按分组找裁剪阈值；随后要么导出「缩放＋裁剪」记录供后续模拟量化，要么进入真正打包为整数权重的路径。方法页 §5.1 已记录裁剪候选与跳过规则。

## 3. 权重产物与打包约定

`qmodule.py` 定义了 AWQ 权重的**实际存储形状**，这也是跨引擎移植的关键：

| 项 | 实现（`qmodule.py`） | 含义 |
|---|---|---|
| `qweight` | int16 缓冲，形状 `(out_features/interleave, in_features//int16_pack_num*interleave)` | int16 只是打包容器；每元素实际占 `w_bit` 位 |
| 打包因子 | `pack_num = 32 // w_bit`、`int16_pack_num = 16 // w_bit` | 无符号打包口径；同一位宽在 32 位与 16 位容器下的分组数不同 |
| `scales` / `qzeros` | 宽度由 `calculate_zeros_width(in_features, group_size, pack_num)` 决定 | 分组元数据也要打包，且宽度向上对齐到 `pack_num` 的整数倍 |
| `scale_zeros` | `zeros * scales`（`from_linear` 内） | 反量化时可把零点与尺度合成一个因子，减少主循环运算 |

`pack_intweight(unpacked_qweight, interleave, kstride)` 是离线重排函数：把 `(N, K)` 的未打包权重按 `32` 个一组的粒度重排后再打包。**这一步定义了 AWQ 自己的元素顺序**，它不是通用的「低半字节在前」约定——下一节会看到，这正是不兼容的来源。`assert in_features % group_size == 0` 之类的整除约束同样写在这里。

## 4. 内核入口

`qmodule.py` 的 forward 按输入规模分派：

- 输入 token 数小于阈值时调用 `awq_inference_engine.gemv_forward_cuda_new(...)`（该文件第 207 行附近）；
- 否则调用 `gemm_forward_cuda_new(...)`（第 218 行附近）。

内核源码位于 `awq/kernels/csrc/`：老路径在 `quantization/`（`dequantize.cuh`、`gemv_cuda.cu`、`gemm_cuda_gen.cu`），另一套在 `quantization_new/`（含各自的 `gemm/`、`gemv/` 子目录与 `dispatch_utils.cuh`）。同目录下还有 attention、layernorm、position_embedding 与 w8a8 内核，说明该仓库本身就是一个可运行推理端，而不只是量化脚本集合。

这两个内核的入口签名、反量化位技巧（`lop3`／`sub.f16x2`／`fma.rn.f16x2`）、线程分块与 group_size 模板分派，见 [[weight-only-dequant-kernels|权重量化反量化内核的契约]]。

这与 [[quantized-matmul-scaling-execution|执行路径]] 的分类对应：AWQ 属于「打包权重 + 运行中反量化 + 浮点矩阵乘」，而 GEMV/GEMM 的分派正是对「解码受带宽限制、预填充受算力限制」这一差异的响应。

## 5. 跨引擎：vLLM

vLLM 的 `auto_awq.py` 指出了移植的真正障碍，并给出转换函数。文件注释写明：**AWQ 在 int32 值内部使用非标准的打包顺序**，`_convert_awq_to_standard_format` 因此要做三件事：

1. 把 `qweight` 从 `(K, N // pack)`、按第 1 维打包，转成 `(K // pack, N)`、按第 0 维打包；
2. 在解包时按 `reverse_order` 重排每个 int32 内的元素顺序（注释称 conv 到标准位序）；
3. 对 `qzeros` 做同类的位序修正与重新打包。

也就是说：**同一个算法、同一组数值，checkpoint 能不能被另一个引擎直接加载，取决于位打包约定而非量化公式。** 转换函数的参数里还显式带有 `pack_factor = 32 // size_bits` 与 `packed_dim`，说明两边的容器/轴向假设不同。

加载之后，vLLM 内部还有 AWQ 原生路径与 marlin 路径之分（对应部署页里并列的 `awq` 与 `awq_marlin` 方法名），后者要经 marlin 工具函数再打包一次，见 [[marlin-batched-w4a16-gemm|Marlin]] 的格式约定与 [[quantized-llm-deployment-backends|部署框架与后端支持]]。

## 6. 跨引擎：SGLang

SGLang 把 AWQ 放在 `srt/layers/quantization/awq/`（`awq.py`、`awq_triton.py`、`schemes/`），并同时注册 `awq` 与 `awq_marlin` 两个方法名。内核侧走 JIT：`jit_kernel/awq_dequantize.py` 用 `_jit_awq_dequantize_module` 编译 `csrc/gemm/awq_dequantize.cuh`，对外暴露 `awq_dequantize(output, qweight, scales, qzeros)`。也就是说 SGLang 的 AWQ 路径同样是「先反量化再计算」，与 vLLM 的原生路径形态一致，区别在于内核由 Triton/JIT 提供而非预编译扩展。

## 7. 论文-代码对齐：本页补充的部分

方法页已记录三处论文与实现的差别（分析式与执行式不同、裁剪候选数与默认值、零通道下限）。代码核对再补两条：

- **论文的等价缩放与实现的打包约定是两层东西。** 论文只规定 `W → WD`、`X → D⁻¹X`；而 `interleave/kstride` 重排、`int16` 容器、`zeros*scales` 合成都是实现层的发明，任何跨引擎复用都必须按实现层核对。
- **仓库自带量化器之外的路径。** 该仓库还包含 `w8a8_linear.py`（`W8A8OF16LinearStaticScale` 与动态输入尺度版本），说明仓库同时支持 W8A8 执行，不能把 llm-awq 仅当作 W4A16 工具。

## 8. 可复用的实现要点

做同类实现或迁移一份 AWQ 权重时，按这个顺序核对：

1. **位打包约定**（容器类型、每容器元素数、元素顺序、打包轴），而不是先看量化公式；
2. **元数据宽度与对齐**（分组的尺度/零点是否也要打包、宽度是否向上对齐）；
3. **反量化发生的位置**（内核内融合还是独立 kernel）；
4. **分派条件**（GEMV 与 GEMM 的切换阈值、batch 与形状约束、整除断言）；
5. **加载端是否要求标准顺序**（例如 vLLM 的转换函数）；
6. 最后才是超参数与校准数据是否一致。

## 9. 验证状态与待验证

- 全部结论为 code-read：读取了上述快照中的文件与其注释，**没有**构建、运行、导出或加载任何 AWQ 模型，也没有验证转换函数的正确性。
- AWQ 的 CUDA 内核实现、TinyChat 的运行时与打包细节未逐行审查；本页只核对目录结构与调用入口。
- vLLM 与 SGLang 的量化方法名、配置类与 JIT 入口取自各自快照；其它版本的方法名、默认值与转换逻辑可能不同，引用时应固定版本。
- 未核对 vLLM 的 AWQ 原生内核（`awq_triton.py`）与 SGLang 的 `awq_triton.py` 是否逐位等价；两者对同一 checkpoint 的数值一致性属于待验证项。
