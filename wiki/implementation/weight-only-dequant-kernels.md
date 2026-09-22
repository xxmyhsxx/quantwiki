---
title: 权重量化反量化内核的契约：AWQ 与 marlin 两种实现
type: implementation
tags:
  - weight-quantization
  - gpu
  - kernels
  - matmul
  - data-format
sources:
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/layers/quantization/awq_triton.py
  - raw/repositories/2026-09-21/sglang/source/python/sglang/srt/layers/quantization/marlin_utils.py
  - raw/repositories/2026-09-21/sglang/source/python/sglang/jit_kernel/csrc/gemm/awq_dequantize.cuh
  - raw/repositories/2026-09-21/llm-awq/source/awq/kernels/csrc/quantization_new/dequantize.cuh
  - raw/repositories/2026-09-21/sglang/source/python/sglang/jit_kernel/csrc/gemm/marlin/gptq_marlin_repack.cuh
  - raw/repositories/2026-09-21/sglang/source/python/sglang/jit_kernel/gptq_marlin_repack.py
  - raw/repositories/2026-09-21/llm-awq/source/awq/kernels/csrc/quantization_new/gemm/gemm_cuda.cu
  - raw/repositories/2026-09-21/llm-awq/source/awq/kernels/csrc/quantization_new/gemv/gemv_cuda.cu
  - raw/repositories/2026-09-21/llm-awq/source/awq/quantize/qmodule.py
  - raw/repositories/2026-09-21/llm-awq/source/awq/kernels/csrc/quantization/dequantize.cuh
  - raw/repositories/2026-09-21/llm-awq/source/awq/kernels/csrc/quantization/gemv_cuda.cu
  - raw/repositories/2026-09-21/llm-awq/source/awq/kernels/csrc/quantization/gemm_cuda_gen.cu
  - raw/repositories/2026-09-21/llm-awq/source/awq/kernels/csrc/pybind.cpp
  - raw/repositories/2026-09-21/sglang/source/python/sglang/jit_kernel/gptq_marlin.py
  - raw/repositories/2026-09-21/sglang/source/python/sglang/jit_kernel/awq_marlin_repack.py
  - raw/repositories/2026-09-21/sglang/source/python/sglang/jit_kernel/csrc/gemm/marlin/gptq_marlin.cuh
  - raw/repositories/2026-09-21/sglang/source/python/sglang/jit_kernel/csrc/gemm/marlin/marlin.cuh
  - raw/repositories/2026-09-21/sglang/source/python/sglang/jit_kernel/csrc/gemm/marlin/marlin_template.h
  - raw/repositories/2026-09-21/sglang/source/python/sglang/jit_kernel/csrc/gemm/marlin/dequant.h
  - raw/repositories/2026-09-21/sglang/source/python/sglang/jit_kernel/csrc/gemm/marlin/marlin_dtypes.cuh
updated: 2026-09-22
---

# 权重量化反量化内核的契约：AWQ 与 marlin 两种实现

W4A16 的执行可以概括成一句话：权重以低比特打包存放，内核在读回它们时反量化到 FP16/BF16 再参与矩阵乘，具体 dtype 由实现路径决定。执行路径的分类见 [量化矩阵乘法的缩放与执行路径](quantized-matmul-scaling-execution.md)，Marlin 的批处理设计见 [Marlin](marlin-batched-w4a16-gemm.md)。本页补的是更靠底层的一层：**内核入口要求什么样的张量、怎样把 4 位反量化成 fp16、线程与模板怎样配置**，以及这些约定在不同实现形态（预编译扩展、JIT 编译）下的差别。

比较对象是 AWQ 官方仓库的内核（`llm-awq` `d6e797a4`）、SGLang 的 JIT AWQ/Marlin（`2f730e29`），并参照 vLLM 的 Triton AWQ（`568afb3a`）。本页是源码核对；未编译或运行 GPU 内核，未做性能测量。位布局的独立教学计算另见 AWQ 实现页。

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
3. **用 `sub.f16x2` 与 `fma.rn.f16x2` 还原。** 两条减法减去魔数（{1024,1024}），两条乘加（`× 1/16` 再 `+ (−64)`）把错位的 4-bit 码还原为 FP16 表示的 0–15；它尚未施加量化 scale/zero-point，不是有符号 INT4 的 −8–7。注释解释了为什么用乘加：`sub` 与 `fma` 吞吐相同，因此对高位元素不必先移位。

文件里保留的注释还记录了一次设计变更：作者原本用 `{1032,1032}` 与 `−72`，后改为 `{1024,1024}` 与 `−64`，理由是「不需要映射到 [-8, 7]」。**这说明这类技巧的常量与「是否把权重映射成对称区间」绑定**，抄代码时不能只抄指令序列。

## 3. AWQ：GEMV 与 GEMM 的约定

本节先解释 `quantization/gemv_cuda.cu` 的**旧版**契约，再区分当前 WQLinear 所走的新路径；同名文件所在目录不能省略。旧文件头注释给出：

```text
_in_feats:        [B, IC]
_kernel:          int32 [OC, IC // 8]
_zeros:           int32 [OC, IC // G // 8]
_scaling_factors: half  [OC, IC // G]
```

要点有三：**权重按 8 个一组打包**（`IC // 8`）；**零点与尺度按分组维给出**，且零点本身也被打包（额外除以 8）；**group_size 是模板参数**——主机端按 `group_size == 64` 或 `== 128` 分别启动 `gemv_kernel_g64` 与 `gemv_kernel_g128`，注释特别提醒「无法从尺度张量的形状推断 group size」。此外，`zeros_w` 与 `sf_w` 都用两层 `make_divisible` 对齐（先把组数对齐到 `PACK_FACTOR`，再对齐到 2），源码里还留着一条 TODO 说明对齐处理曾导致地址错位——元数据的宽度对齐是这类格式最容易出错的地方。

线程与分块：`dim3 num_blocks(1, num_out_channels / 4, num_out_feats)`、`dim3 num_threads(32, 4)`，注释说明 `blockDim_x * workload_per_thread = IC`、`blockDim_y * gridDim_y = OC`。也就是说输出通道方向按 4 个一组分给 block 的 y 维，batch 维走 z 维，每个 block 只处理一个输出特征（batch 项）。

`gemm_cuda_gen.cu` 走另一套分块：`num_blocks((num_out_feats + 128 - 1) / 128 * j_factors1 * split_k_iters)`、`threads_per_block(32, 4)`，即输出方向按 128 切块，并带有 split-K 因子——与 [Marlin](marlin-batched-w4a16-gemm.md) 用 split-K/条带划分解决形状不整除的思路同源，只是实现方式更简单。

### 当前 WQLinear 的新版路径

`qmodule.py:WQLinear.forward` 实际调用 `gemv_forward_cuda_new` / `gemm_forward_cuda_new`，经 pybind 进入 `quantization_new/`，并不进入上述旧内核。新版采用 int16 qweight 与浮点 `scales`、`scaled_zeros=-scale*zero`；反量化为 `q*scale+scaled_zeros`。旧版打包 qzeros 的契约不能拿来解释这些参数。

新版 GEMV 主机端只有 g128 分支，按 M=1–7 实例化；每 block 256 线程，输出通道按 `N_PER_BLOCK=2`、`K_INTERLEAVE=4` 组织。新版 GEMM 在已读主机分支固定 `G=128`，按 token 数选择 CTA/SPLITK 配置。上层构造器允许某个 group size，不保证这条执行路径支持它：必须沿实际调用检查分派，而不能从旧文件的 g64 分支推断新版可运行。新版 GEMV 文件头仍有旧布局注释，判断应以 qmodule 的缓冲、主机端 dtype 检查和实际指针访问为准。

这补齐了 [AWQ 实现页](awq-implementation.md) 中“打包产物 → 导出名 → 内核”的关系。本轮只做静态核对，没有编译或运行新版内核。

### 新版 GEMV：加载、反量化与归约

`gemv_kernel<NPerBlock,Batch,BlockSize,GroupSize,T>` 的实例是 NPerBlock=2、BlockSize=256、GroupSize=128、Batch=M。每个 block 负责 `2×4=8` 个输出通道，grid 为 N/8；四行交织来自序列化格式，不是额外 batch 维。

1. 每线程一次用 128-bit 向量读 32 个 4-bit 码，沿输出 tile 读取对应组的 scale 与 scaled zero。虽然 Python 容器是 int16，设备端可以用 uint32/float4 按位搬运；容器视图不改变量化精度。
2. `quantization_new/dequantize.cuh` 将 4-bit 码转换成浮点 0–15，随后 half2/bfloat162 融合乘加 `q*s+scaled_zero`。局部 shuffle 再对应 pack 时的 K 重排，不能把解包后的顺序直接视为逻辑 K 顺序。
3. 读入激活后，用 `__hfma2` 更新类型为 T 的局部 psum。此 GEMV 不是 Tensor Core MMA，FP16/BF16 局部累加也不能统称 FP32。
4. `warp_reduce` 先将局部和转换为 float，按 XOR 16、8、1 合并同一输出的数据；lane 0/2/4/6 写共享内存。之后跨 8 个 warp 以 float 求和，再转换为输出 T。

所以即使整数码与尺度相同，局部累加精度、归约顺序和最终转换仍可能与浮点 matmul 产生差异；须在实际 GPU 上验证，不能由公式等价推导逐位相同。

### 新版 GEMM：token 数决定哪些工作块

`gemm_forward_cuda_new` 在 WQLinear 的 M≥8 分支使用下表；所有分支 G=128。

| M 范围 | CTA M×N×K | stages | split-K |
|---|---|---:|---:|
| 8–32 | 16×128×128 | 4 | 2 |
| 33–64 | 16×128×128 | 3 | 1 |
| 65–128 | 32×128×128 | 4 | 1 |
| 129–192 | 64×128×64 | 4 | 1 |
| >192 | 64×128×64 | 4 | 切换至 `gemm_w4a16_T2`，不使用前述 SPLITK 模板参数 |

已读数据路径用 `cp.async` 搬运到分阶段共享内存，`ldmatrix` 获取矩阵片段，在 B 的共享内存到寄存器阶段解包并应用组尺度，然后进入 MMA；不需要先写出整个浮点权重矩阵。具体 MMA 按 T 区分：half 路径为 `mma.sync...f16.f16.f16.f16`，bfloat16 路径为 `...f32.bf16.bf16.f32`。后者的 MMA 累加为 FP32，但输出和部分后续归约会转换回 BF16，不能把整条路径视为全程 FP32。

启动代码沿 N 使用 `N // CTA_N`，只在 M 上做向上取整；所以 N=136 虽满足 WQLinear 的 N%8 检查，也不能据此认为 GEMM 已覆盖最后 8 列。这是源码可见的形状边界，不是本轮实测故障。M=8–32 的 split-K 分支还有 semaphore 控制的跨块合并；其同步、初始化、所有尾块和数值行为仍需运行测试，表中的线程配置不是性能保证。

## 4. AWQ：内核不是孤立的

`pybind.cpp` 的注册清单说明这套内核是一个完整推理端的一部分，而不是单一的矩阵乘：除量化 GEMV/GEMM（新旧两套）外，还注册了 layer norm、NeoX 风格 RoPE、单查询注意力、融合 RoPE、W8A8 GEMM 与带偏置版本，以及 `invoke_quant`（fp16→int8 量化）、`rms_norm_general`、`silu_and_mul`、`gelu_and_quant` 等融合算子。

对复现的含义是：**量化内核的收益经常来自与相邻算子的融合**（激活函数与量化合并、归一化与量化合并），单独移植矩阵乘内核而不移植这些融合，端到端差异会被算到「量化没用」上。这与 [量化误差诊断与验证](quantization-error-diagnosis.md) 中「单算子加速不等于端到端收益」是同一件事的两面。

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

### 普通 AWQ 的 JIT 反量化与 Marlin 不同

`awq_dequantize.cuh` 的主机包装启动 16×16 线程块，横轴索引打包输出列，纵轴索引 K 行。每线程处理一个 int32 的 8 个码，按 `row // group_size` 读取一组元数据，计算 `(q-z)*s` 后写连续的 8 个浮点数；FP16 采用前述 1024 位技巧，BF16 有独立常量和编译架构门槛。主机匹配张量的 shape、dtype 和设备，不能据此推断任意 stride 或畸形分组都能运行。

这一 kernel 写出完整 `(K,N)` 浮点矩阵，随后普通 AWQ backend 调用 matmul；Marlin 则在矩阵乘里消费预先重排的整数权重。配置选择及两种转换的发生时机见 [AWQ 的 SGLang 路径](awq-implementation.md#6-跨引擎sglang)。

## 6. SGLang：repack 的几何

marlin 内核不接受量化器直接产出的布局，需要先重排：

- Python 接口 `gptq_marlin_repack(b_q_weight, perm, size_k, size_n, num_bits)` 在内部创建并返回 out；底层包装才接收 out。perm 可为空，C++ 通过 `perm.size(0) != 0` 决定是否执行 act-order 置换；tile 重排本身在两种情况下都存在。该包装明确支持 4/8-bit，不能直接消费原始 GPTQ 3-bit 位流；
- `awq_marlin_repack(b_q_weight, size_k, size_n, num_bits)`：不需要 perm，但输出形状按 `tile_size = 16` 与 `pack_factor = 32 // num_bits` 计算为 `(size_k // 16, size_n * 16 // pack_factor)`——把沿输出维的打包转成 16 行一组的 tile 布局；
- `awq_marlin_moe_repack` 签名虽带 perm，本快照函数体并未使用它，而是逐专家调用普通 AWQ repack；参数出现不等于实际执行了置换。

这解释了为什么 [AWQ 实现页](awq-implementation.md) 与 [GPTQ 实现页](gptq-implementation.md) 里都出现「加载时还要再转换一次」：AWQ 与 GPTQ 的原生布局都不是 marlin tile 布局，GPTQ 是否实际执行额外的列置换还取决于 perm 是否非空。

## 7. SGLang：模板空间与线程配置

`gptq_marlin.cuh` 用一条宏严格定义「哪些组合有实例」：

```text
_GET_IF(W_TYPE, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS,
        M_BLOCK_SIZE_8, GROUP_BLOCKS, NUM_THREADS, IS_ZP_FLOAT)
```

运行时的 `q_type`、`thread_m_blocks`、`thread_n_blocks`、`thread_k_blocks`、`m_block_size_8`、`group_blocks`、`num_threads`、`is_zp_float` 必须同时匹配某个实例，否则取不到内核。源码按用途把实例分成五族并加了注释：`COMMON`（`group_blocks ∈ {-1, 2, 4, 8}`、零点非浮点）、`BIGGROUP`（大分组）、`FZP`（浮点零点）、`ACT`（`group_blocks == 0`，即激活重排情形）、`FP4`（nvfp4/e2m1，`group_blocks == 1`）。

另有两条硬约束写在源码里：`marlin.cuh` 中 `default_threads = 256`；`gptq_marlin.cuh` 检查 `num_threads` 至少 128（4 个 warp），并用 `thread_m_blocks * 16` 计算 M 方向的 tile 高度，对 `__CUDA_ARCH__ < 800` 直接不编译。

**对复现的含义：** 加载一份量化权重时，「位宽、分组大小、零点是否为浮点、是否激活重排」这些配置不仅是数值参数，还决定能否命中模板——配置解析错了不是精度略差，而是直接没有内核可用。这与 [GPTQ 实现页](gptq-implementation.md) 记录的加载期组合处理（单组场景下归一化推理期 desc_act 标志，不是禁止量化时采用 act-order）互相印证。

## 8. 三种实现形态对照

| 维度 | AWQ：预编译 CUDA 扩展 | SGLang：JIT CUDA 模板 | 参考：Triton 路径 |
|---|---|---|---|
| 交付形态 | `setup.py` 编译的 `.so`，pybind 注册固定名字 | 运行时按 dtype 编译，`load_jit` 声明源文件与包装 | Python 侧 JIT，内核以 DSL 书写 |
| 变体选择 | 旧 GEMV 按 g64/g128；新版 GEMV 固定 g128，GEMM 再按 M 分派 | 由模板参数与运行时配置共同匹配 `_GET_IF` | 由 Python 侧参数与编译选项决定 |
| 反量化 | 手写 PTX 位技巧（`lop3`／`sub.f16x2`／`fma.rn.f16x2`） | 在模板中按权重类型实例化（含 dequant 头文件） | 由编译器生成 |
| 布局要求 | 权重量化器自己的打包 + 元数据对齐 | 必须经 repack 进入 16 行 tile 布局（GPTQ 的 act-order 置换可为空） | 视内核实现而定 |
| 调试与移植成本 | 需要重新编译扩展；内联 PTX 可从源码检查，最终指令仍需查看编译产物 | 首次运行编译、版本敏感；模板空间显式可见 | DSL 隐去部分硬件细节，需结合生成代码与测量判断；不能仅凭语言预判性能 |

这张表的用途不是排名，而是提醒：**「同一个量化格式」在不同引擎下意味着不同的内核契约**，跨引擎复用时至少要重新核对布局、模板命中条件与融合算子是否一起移植。

反量化产生的寄存器片段怎样进入 Tensor Core、共享内存如何提供 A 片段，以及 AWQ Triton 的 split-K 和编译检查，继续见 [Tensor Core 与量化 GEMM](tensor-core-quantized-gemm.md)。

## 9. 可复用的实现要点

1. 先写清张量契约（形状、打包因子、元数据对齐），再谈指令优化；
2. 反量化的常量与「是否映射到对称区间」绑定，抄指令序列时必须一起抄语义；
3. 元数据宽度对齐是易错点（AWQ 源码里的 TODO 就是证据）；
4. 分清「权重打包」与「内核要求布局」两件事，中间往往需要一次 repack；
5. 模板内核要把「配置 → 实例」的匹配条件写进文档，因为不匹配是硬失败而非降级；
6. 量化内核的收益与相邻融合算子强相关，移植时要连同归一化/激活/量化融合一起评估。

## 10. GPTQ → 现代 Marlin：repack 后怎样参与计算

本节限定 SGLang 固定 commit 的对称 4-bit GPTQ，即 `kU4B8`，激活为 FP16/BF16。它与原始独立 Marlin、AWQ 非对称 `kU4`、FP4 和 FP8 模板不是同一核验范围。`marlin_template.h`、`dequant.h` 与 `marlin_dtypes.cuh` 将以下各阶段连接起来。

### 10.1 Repack 必须匹配寄存器的消费顺序

`gptq_marlin_repack.cuh` 以 $16\times64$ 权重 tile 为单元。对四个工作 warp 的 lane $\ell$，令 `tc_row=(lane%4)*2`、`tc_col=lane/4`；每个 lane 取 K 偏移 $\{0,1,8,9\}$，以及两列 `warp_id*16+tc_col` 和其加 8 的位置，得到 8 个 u4 码。

这 8 个码按 `pack_idx={0,2,4,6,1,3,5,7}` 装入一个 uint32，存至 tile 内 `lane*4+warp_id`。有 perm 时，K 坐标先映射到原权重行，码的位置由原始 K 下标模 pack factor 决定。repack 保存的是相同量化权重的置换表示，不重新估计尺度或求解 GPTQ。

计算内核每个线程从 shared 读取已经安排好的 `int4`，`matmul` 对一个 packed word 及其右移 8 位的版本分别调用 dequant。把 “16 行 tile 的 shape 对了” 当成完整布局正确还不够：上述 lane、nibble 和元数据置换必须一起对应。

### 10.2 U4B8 的减 8 融在位转换里

`dequant<half2,kU4B8,false>` 用 `lop3` 将 nibble 拼到 FP16 的指数/尾数位中，随后用 half2 subtract 和 FMA 得到 $q-8$。常量 `0x6400` 对应 FP16 的 1024 基底；`SUB=0x64086408` 把偏置 8 合并进减法，高 nibble 路径用 `MUL=0x2c002c00` 和 `ADD=0xd480d480` 恢复同样的整数值。

BF16 用不同指数基底 `0x4300` 与 `SUB=0x43084308`，不能直接复制 FP16 的常量。这里 `has_zp=false` 表示没有独立加载的 zero-point tensor，**不是说 u4 码无需减 8**。GPTQ 对称编码与位转换语义由 `kU4B8` 一起规定。

分组尺度乘入恢复后的 FP16/BF16 B fragment，再执行 `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` 或对应 BF16 指令。`FragC` 是 4 个 FP32 元素；不存在这条路径先物化完整浮点权重矩阵的阶段。

### 10.3 Act-order：加载期重排与前向置换各做什么

加载期权重按 perm repack；前向 `gptq_marlin.cuh` 若 `has_act_order`，先运行 `permute_cols_kernel` 写出 `A_tmp[m,k]=A[m,perm[k]]`。同一置换同时作用于 A 的列和 W 的行，维持矩阵乘语义；只改权重会把输入通道配错。

若拥有完整 K，排序后完整原组已连续，host 在置换 A 后将 `has_act_order=false`，主循环可以使用规则分组。若 K 分片不能恢复完整组，则 `group_blocks=0` 走动态组索引路径：

- 将当前 K tile 的 `g_idx` 搬入 shared，并缓存接下来需要的尺度组区间；
- `fetch_scales_to_registers` 按 MMA 的 K 坐标 $\{0,1,8,9\}$ 为四个元素读取对应组尺度；
- `init_same_group` 用 tile 首尾 group id 判断是否可复用同一尺度。这依赖加载期排序后组号单调，不能对任意乱序 `g_idx` 只比较首尾。

所以“不再启用 act-order 模板”不代表在线输入置换从未发生；完整 K 与 TP 局部 K 的元数据成本也不同。

### 10.4 Shared ring 与两套寄存器缓冲

`fetch_to_shared` 通过 `cp_async4` 搬运打包 B，对 A 的越界行使用 predicated 拷贝。规则组只在对应组边界加载尺度；动态组路径还搬运 `g_idx`。没有新有效 tile 时仍执行 fence，以保持排空阶段的 copy-group 计数。

`start_pipes` 预取 `stages-1` 个 tile，`wait_for_stage` 等待 `stages-2` 并 CTA 同步。`fetch_to_registers` 用 ldmatrix 读取 A，直接读取 B 的 `int4`；寄存器用 `k%2` 轮换。稳态先准备 `k+1` 的片段，在倒数第二个 K 子步提交下一 shared stage、等待并切换，再对当前片段解包、缩放和 MMA。循环结束先 wait 0，随后才能做 shared-memory 归约。

host 按候选 `thread_k/thread_n` 检查 K/N 整除和 shared-memory 容量；这是本主循环整 tile 访问的前提，不存在“任意尾块因为 A 有 predicate 就安全”的推论。流水的共享概念见 [异步拷贝](gpu-async-copy-pipelines.md)。

### 10.5 两层归约决定最终精度

CTA 的工作划分采用跨 K/N 的 stripe，使部分输出列 tile 可能由多个 CTA 合作；它不是每个 CTA 都独立完成整个 K，也不是每个输出都一定需要跨 CTA 合并。

先由 `thread_block_reduce` 在 shared 中按树形合并同一 CTA 的 FP32 部分和，再根据 `slice_count` 与两个 flag 选择：

| 条件 | 跨 CTA 合并 | 不能忽略的数值步骤 |
| --- | --- | --- |
| 只有一个 slice | 直接写结果 | 最终转输出 dtype |
| 多 slice，非 atomic，`use_fp32_reduce=false` | 锁按 `slice_idx` 串行接续，用 C 存中间结果 | 每段结果转输出 dtype，下一个 CTA 再转 FP32 加 |
| 多 slice，非 atomic，`use_fp32_reduce=true` | 同样锁定次序，用 C_tmp 存 FP32 | 避免中间 FP16/BF16 存储舍入，仍有 FP32 加法顺序 |
| 多 slice，`use_atomic_add=true` | 初始化输出和锁后，各 CTA 将结果转输出 dtype 再 atomicAdd | 加法的顺序与低精度中间结果不同 |

源码函数名 `global_reduce_fp16` 对 BF16 模板同样通过 `scalar_t` 转换；不能因函数名断言只存 FP16。Python JIT 包装的 `use_fp32_reduce` 默认是 false，但上层 `marlin_utils.py` 的 `USE_FP32_REDUCE_DEFAULT=True`，因此不能只读低层签名判断常用路径。上层 atomic 选择器要求 CUDA、N<2048、K≥2048，并排除 SM90 前的 BF16；源码注释虽说默认关闭，实际环境开关分支写成 `if not True`，不能按注释解释为已经关闭。host 还要求 `ceil(M_split/64)*N<=2048` 才为该分块实际启用 atomic。

对称 **W4 逐通道**还有一个容易漏掉的顺序：`write_result` 先将 FP32 部分和转输出 dtype，再用 half2/BF16 pair 乘列尺度；规则分组/动态组则在 MMA 前乘 B 的组尺度。前者即使采用 FP32 跨 CTA 临时缓冲，也不能自动避免最终“先转低精度、后缩放”的溢出。W8 逐通道源码另有先缩放 FP32 部分和的分支，不应把该分支套到 W4。

例如 FP32 部分和 70,000 配 FP16 尺度约 0.001：先缩放再转 FP16 可得到约 70，先转 FP16 已成为无穷。这个反例只是说明算术顺序的范围差异，不表示已在某个真实模型上复现故障。

[CPU 教学脚本](../assets/w8a8-quantization-gemm-kernels/check_contracts.py)与[结果](../assets/w8a8-quantization-gemm-kernels/checks-2026-09-22.json)检验 16×64 repack 的坐标覆盖和往返、u4→FP16 的位常量、两路置换恒等式、分段低精度存储与 FP32 合并反例。它不执行上游 JIT/CUDA，也不证明所有模板实例正确。

## 11. 验证状态与待验证

- 上游核对为 code-read：未编译、未运行 GPU、未测性能；新增 CPU 教学算例只检验布局和数值顺序，不构成上游输出验证。
- 已核对 AWQ 新版 GEMV 的主要加载、解包、乘加和归约路径，以及 GEMM 主机配置、搬运/MMA 与部分合并代码；尚未逐行验证全部同步、边界和运行时错误处理。
- SGLang 已读普通 AWQ JIT 反量化，以及现代 Marlin 对称 W4 GPTQ 的 repack、反量化、尺度选择、主流水和两层归约。未逐一展开 AWQ 零点、W8、FP4/FP8 全部模板，也未覆盖全部线程配置的同步与越界正确性。
- 已定向读取 vLLM Triton AWQ 的反量化索引、GEMM K 循环和 split-K 包装；没有执行或证明它与 SGLang/CUDA 版本逐位等价。

## 来源身份

下表用于在没有本地资料库时辨识来源；具体论述的章节、公式、图表或代码位置见正文。

| 来源 | 版本或快照 | 说明 |
| --- | --- | --- |
| [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq/tree/d6e797a42b9ef7778de8ee2352116e0f48a78d61) | `d6e797a42b9ef7778de8ee2352116e0f48a78d61` | — |
| [sgl-project/sglang](https://github.com/sgl-project/sglang/tree/2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc) | `2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc` | — |
| [vllm-project/vllm](https://github.com/vllm-project/vllm/tree/568afb3a13806beb53bb2e6bd518269357b237c0) | `568afb3a13806beb53bb2e6bd518269357b237c0` | Triton AWQ 的设备与包装路径 |
