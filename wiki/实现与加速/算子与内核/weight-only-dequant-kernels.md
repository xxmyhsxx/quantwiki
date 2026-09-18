---
title: 权重量化反量化内核的契约：AWQ 与 marlin 两种实现
slug: weight-only-dequant-kernels
sources:
  - raw/repositories/quantization/2026-06-mit-han-lab-llm-awq-d6e797a4/source/awq/kernels/csrc/quantization/dequantize.cuh
  - raw/repositories/quantization/2026-06-mit-han-lab-llm-awq-d6e797a4/source/awq/kernels/csrc/quantization/gemv_cuda.cu
  - raw/repositories/quantization/2026-06-mit-han-lab-llm-awq-d6e797a4/source/awq/kernels/csrc/quantization/gemm_cuda_gen.cu
  - raw/repositories/quantization/2026-06-mit-han-lab-llm-awq-d6e797a4/source/awq/kernels/csrc/pybind.cpp
  - raw/repositories/serving/2026-06-sgl-project-sglang-2f730e29/source/python/sglang/jit_kernel/gptq_marlin.py
  - raw/repositories/serving/2026-06-sgl-project-sglang-2f730e29/source/python/sglang/jit_kernel/awq_marlin_repack.py
  - raw/repositories/serving/2026-06-sgl-project-sglang-2f730e29/source/python/sglang/jit_kernel/csrc/gemm/marlin/gptq_marlin.cuh
  - raw/repositories/serving/2026-06-sgl-project-sglang-2f730e29/source/python/sglang/jit_kernel/csrc/gemm/marlin/marlin.cuh
updated: 2026-09-17
---

# 权重量化反量化内核的契约：AWQ 与 marlin 两种实现

W4A16 的执行可以概括成一句话：权重以低比特打包存放，内核在读回它们时反量化到 fp16 再参与矩阵乘。执行路径的分类见 [[quantized-matmul-scaling-execution|量化矩阵乘法的缩放与执行路径]]，Marlin 的批处理设计见 [[marlin-batched-w4a16-gemm|Marlin]]。本页补的是更靠底层的一层：**内核入口要求什么样的张量、怎样把 4 位反量化成 fp16、线程与模板怎样配置**，以及这些约定在不同实现形态（预编译扩展、JIT 编译）下的差别。

比较对象是 AWQ 官方仓库的内核（`llm-awq` `d6e797a4`）与 SGLang 的 JIT marlin 内核（`2f730e29`）。全部结论为 code-read，未编译、未运行、未做性能测量。

## 1. 内核契约包含哪几件事

阅读反量化矩阵乘内核时，需要核对的是四类约定，而不是只看「用了什么指令」：

| 约定 | 具体内容 |
|---|---|
| 入口签名 | 每个张量的形状、dtype、打包粒度 |
| 布局契约 | 权重与元数据在内存中的排列（谁和谁共享一个容器、如何对齐） |
| 线程与分块 | block/grid 维度、每线程工作量、与形状的整除关系 |
| 模板与分派 | 哪些位宽/分组/类型有实例，运行时如何选中 |

下面两节按这四类逐项对照两种实现。

## 2. AWQ：反量化的位技巧

`dequantize.cuh` 的 `dequantize_s4_to_fp16x2(uint32_t)` 把一个 int32 里的 8 个 4 位权重转成 4 个 half2。它不逐元素取位，而是三步打包完成（文件内注释自述思路）：

1. **用 `lop3` 一次完成掩码与或操作。** 四个常量是 `immLut = (0xf0 & 0xcc) | 0xaa`、`BOTTOM_MASK = 0x000f000f`、`TOP_MASK = 0x00f000f0`、`I4s_TO_F16s_MAGIC_NUM = 0x64006400`。它把每组 4 位放进 fp16 的尾数位，同时用魔数给出一个合法指数，于是这个半精度数成为「偏移后的整数」。
2. **只做一次移位。** 代码注释写明：整个序列只需要一条移位指令（`top_i4s = i4s >> 8`），因为寄存器打包格式允许低位与高位两组元素分别处理。
3. **用 `sub.f16x2` 与 `fma.rn.f16x2` 还原。** 两条减法减去魔数（{1024,1024}），两条乘加（`× 1/16` 再 `+ (−64)`）把 16 位容器里的错位元素还原成正确的有符号值。注释解释了为什么用乘加：`sub` 与 `fma` 吞吐相同，因此对高位元素不必先移位。

文件里保留的注释还记录了一次设计变更：作者原本用 `{1032,1032}` 与 `−72`，后改为 `{1024,1024}` 与 `−64`，理由是「不需要映射到 [-8, 7]」。**这说明这类技巧的常量与「是否把权重映射成对称区间」绑定**，抄代码时不能只抄指令序列。

## 3. AWQ：GEMV 与 GEMM 的约定

`gemv_cuda.cu` 的文件头注释直接给出契约：

```text
_in_feats:        [B, IC]
_kernel:          int32 [OC, IC // 8]
_zeros:           int32 [OC, IC // G // 8]
_scaling_factors: half  [OC, IC // G]
```

要点有三：**权重按 8 个一组打包**（`IC // 8`）；**零点与尺度按分组维给出**，且零点本身也被打包（额外除以 8）；**group_size 是模板参数**——主机端按 `group_size == 64` 或 `== 128` 分别启动 `gemv_kernel_g64` 与 `gemv_kernel_g128`，注释特别提醒「无法从尺度张量的形状推断 group size」。此外，`zeros_w` 与 `sf_w` 都用两层 `make_divisible` 对齐（先把组数对齐到 `PACK_FACTOR`，再对齐到 2），源码里还留着一条 TODO 说明对齐处理曾导致地址错位——元数据的宽度对齐是这类格式最容易出错的地方。

线程与分块：`dim3 num_blocks(1, num_out_channels / 4, num_out_feats)`、`dim3 num_threads(32, 4)`，注释说明 `blockDim_x * workload_per_thread = IC`、`blockDim_y * gridDim_y = OC`。也就是说输出通道方向按 4 个一组分给 block 的 y 维，batch 维走 z 维，每个 block 只处理一个输出特征（batch 项）。

`gemm_cuda_gen.cu` 走另一套分块：`num_blocks((num_out_feats + 128 - 1) / 128 * j_factors1 * split_k_iters)`、`threads_per_block(32, 4)`，即输出方向按 128 切块，并带有 split-K 因子——与 [[marlin-batched-w4a16-gemm|Marlin]] 用 split-K/条带划分解决形状不整除的思路同源，只是实现方式更简单。

`qmodule.py` 的 forward 按输入 token 数在两者之间分派（见 [[awq-implementation|AWQ 的实现核对]]），因此**同一份权重需要两套内核同时可用**。

## 4. AWQ：内核不是孤立的

`pybind.cpp` 的注册清单说明这套内核是一个完整推理端的一部分，而不是单一的矩阵乘：除量化 GEMV/GEMM（新旧两套）外，还注册了 layer norm、NeoX 风格 RoPE、单查询注意力、融合 RoPE、W8A8 GEMM 与带偏置版本，以及 `invoke_quant`（fp16→int8 量化）、`rms_norm_general`、`silu_and_mul`、`gelu_and_quant` 等融合算子。

对复现的含义是：**量化内核的收益经常来自与相邻算子的融合**（激活函数与量化合并、归一化与量化合并），单独移植矩阵乘内核而不移植这些融合，端到端差异会被算到「量化没用」上。这与 [[quantization-error-diagnosis|量化误差诊断与验证]] 中「单算子加速不等于端到端收益」是同一件事的两面。

## 5. SGLang：JIT 内核怎样被构造

SGLang 的 marlin 路径不预编译扩展，而是在运行时按签名编译：

```python
@cache_once
def _jit_gptq_marlin_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "gptq_marlin", *args,
        cuda_files=["gemm/marlin/gptq_marlin.cuh"],
        cuda_wrappers=[("gptq_marlin_gemm", f"gptq_marlin_gemm<{args}>")],
    )
```

四点契约含义：**按 dtype 生成模板参数**（`make_cpp_args`），因此每种数据类型对应一次编译；`load_jit` 显式声明源文件与包装函数名；`@cache_once` 保证同一 dtype 只编译一次；`@debug_kernel_api` 是运行时的可观测入口。Python 侧还定义 `_MAX_THREAD_N = 256` 并注明与设备端 `device::marlin::` 一致——**主机与设备共享同一个线程上限常量**，两侧不一致会导致模板不匹配。

## 6. SGLang：repack 的几何

marlin 内核不接受量化器直接产出的布局，需要先重排：

- `gptq_marlin_repack(b_q_weight, perm, out, size_k, size_n, num_bits)`：**必须传入 `perm`**，即一个显式置换张量（来自列顺序与分块规则的组合）；
- `awq_marlin_repack(b_q_weight, size_k, size_n, num_bits)`：不需要 perm，但输出形状按 `tile_size = 16` 与 `pack_factor = 32 // num_bits` 计算为 `(size_k // 16, size_n * 16 // pack_factor)`——把沿输出维的打包转成 16 行一组的 tile 布局；
- 另有 `awq_marlin_moe_repack` 带 `perm` 的专家版本。

这解释了为什么 [[awq-implementation|AWQ 实现页]] 与 [[gptq-implementation|GPTQ 实现页]] 里都出现「加载时还要再转换一次」：AWQ 与 GPTQ 的原生布局都不是 marlin tile 布局，差别只是 GPTQ 需要额外的置换张量。

## 7. SGLang：模板空间与线程配置

`gptq_marlin.cuh` 用一条宏严格定义「哪些组合有实例」：

```text
_GET_IF(W_TYPE, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS,
        M_BLOCK_SIZE_8, GROUP_BLOCKS, NUM_THREADS, IS_ZP_FLOAT)
```

运行时的 `q_type`、`thread_m_blocks`、`thread_n_blocks`、`thread_k_blocks`、`m_block_size_8`、`group_blocks`、`num_threads`、`is_zp_float` 必须同时匹配某个实例，否则取不到内核。源码按用途把实例分成五族并加了注释：`COMMON`（`group_blocks ∈ {-1, 2, 4, 8}`、零点非浮点）、`BIGGROUP`（大分组）、`FZP`（浮点零点）、`ACT`（`group_blocks == 0`，即激活重排情形）、`FP4`（nvfp4/e2m1，`group_blocks == 1`）。

另有两条硬约束写在源码里：`marlin.cuh` 中 `default_threads = 256`；`gptq_marlin.cuh` 检查 `num_threads` 至少 128（4 个 warp），并用 `thread_m_blocks * 16` 计算 M 方向的 tile 高度，对 `__CUDA_ARCH__ < 800` 直接不编译。

**对复现的含义：** 加载一份量化权重时，「位宽、分组大小、零点是否为浮点、是否激活重排」这些配置不仅是数值参数，还决定能否命中模板——配置解析错了不是精度略差，而是直接没有内核可用。这与 [[gptq-implementation|GPTQ 实现页]] 记录的加载期组合处理（`desc_act` 与 `group_size=-1` 互斥）互相印证。

## 8. 三种实现形态对照

| 维度 | AWQ：预编译 CUDA 扩展 | SGLang：JIT CUDA 模板 | 参考：Triton 路径 |
|---|---|---|---|
| 交付形态 | `setup.py` 编译的 `.so`，pybind 注册固定名字 | 运行时按 dtype 编译，`load_jit` 声明源文件与包装 | Python 侧 JIT，内核以 DSL 书写 |
| 变体选择 | 主机端 `if (group_size == 64/128)` 启动不同 kernel | 由模板参数与运行时配置共同匹配 `_GET_IF` | 由 Python 侧参数与编译选项决定 |
| 反量化 | 手写 PTX 位技巧（`lop3`／`sub.f16x2`／`fma.rn.f16x2`） | 在模板中按权重类型实例化（含 dequant 头文件） | 由编译器生成 |
| 布局要求 | 权重量化器自己的打包 + 元数据对齐 | 必须经 repack 进入 16 行 tile 布局（GPTQ 还需 perm） | 视内核实现而定 |
| 调试与移植成本 | 需要重新编译扩展；指令级细节不可见 | 首次运行编译、版本敏感；模板空间显式可见 | 最易读，但性能通常受限 |

这张表的用途不是排名，而是提醒：**「同一个量化格式」在不同引擎下意味着不同的内核契约**，跨引擎复用时至少要重新核对布局、模板命中条件与融合算子是否一起移植。

## 9. 可复用的实现要点

1. 先写清张量契约（形状、打包因子、元数据对齐），再谈指令优化；
2. 反量化的常量与「是否映射到对称区间」绑定，抄指令序列时必须一起抄语义；
3. 元数据宽度对齐是易错点（AWQ 源码里的 TODO 就是证据）；
4. 分清「权重打包」与「内核要求布局」两件事，中间往往需要一次 repack；
5. 模板内核要把「配置 → 实例」的匹配条件写进文档，因为不匹配是硬失败而非降级；
6. 量化内核的收益与相邻融合算子强相关，移植时要连同归一化/激活/量化融合一起评估。

## 10. 验证状态与待验证

- 全部为 code-read：未编译、未运行、未测性能，也未核对任何数值输出；本页不给出任何加速比。
- AWQ 的 `quantization_new/` 另一套内核、`gemm_cuda_gen.cu` 的内层循环细节、以及 `marlin_template.h` 的主循环与流水线实现均未逐行审查。
- SGLang 侧只读了 JIT 包装、repack 几何与模板分派宏；`dequant.h`、`marlin_dtypes.cuh` 与设备端主循环未展开。
- 三种形态对照中的 Triton 一列来自本 Wiki 其它页面的既有记录，本轮没有重新核对 SGLang 或 vLLM 的 Triton 实现细节。
