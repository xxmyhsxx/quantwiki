---
title: DeepGEMM 的 MoE 专家执行：行布局、块尺度与 SM90 计算
type: implementation
tags:
  - moe
  - fp8
  - tensor-core
  - kernels
sources:
  - raw/repositories/2026-10-07/deepgemm-sglang
  - raw/repositories/2026-09-21/sglang
updated: 2026-10-14
---

# DeepGEMM 的 MoE 专家执行：行布局、块尺度与 SM90 计算

所选 SGLang FP8 专家链把接收行整理成 contiguous 或 masked 布局，DeepGEMM 据此选择专家权重，以 TMA 读取 FP8 值和激活尺度，用 WGMMA 计算局部点积，再乘恢复尺度累入 FP32，最终写 BF16。**专家行选择、块尺度修正和有效输出范围是三个不同契约。** 对齐 padding 可以参与设备计算，只有被框架后续消费的有效行代表模型贡献。

本页接续 [MoE EP 框架链](moe-expert-parallel-execution.md)和 [DeepEP 通信链](deepep-moe-communication-execution.md)。前者承接模型 gate、group、参数轴、SiLU 与汇合，后者承接 dispatch/回源；这里解释两次专家 GEMM 的设备边界。前置是[矩阵乘尺度](quantized-matmul-scaling-execution.md)、[Tensor Core 数据流](tensor-core-quantized-gemm.md)和[浮点累加](../fundamentals/numeric-formats/floating-point-and-accumulation.md)。

## 1. 匹配版本与核读条件

### 来源身份

| 来源 | 固定身份与依据 |
| --- | --- |
| SGLang | sgl-project/sglang（`https://github.com/sgl-project/sglang/tree/2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc`），commit `2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc`。pyproject、Docker/CI pin、wrapper、MoE runner、scale/activation helper。 |
| SGLang 定制 DeepGEMM | sgl-project/DeepGEMM（`https://github.com/sgl-project/DeepGEMM/tree/b4296edce3cd07e3b6392c8eb390cee9c41a6b70`），官方 `v0.1.3` 标签解析至 commit `b4296edce3cd07e3b6392c8eb390cee9c41a6b70`。定制 Python、TVM-FFI host/API、JIT 配置及 SM90 设备源码。 |
| 发布包对应 | PyPI 0.1.3（`https://pypi.org/project/sgl-deep-gemm/0.1.3/`） 与 cu129 x86_64 wheel（`https://github.com/sgl-project/whl/releases/download/v0.1.3/sgl_deep_gemm-0.1.3+cu129-py3-none-manylinux2014_x86_64.whl`）。2026-10-07 下载哈希与官方元数据一致；每包 869 个 Python/JIT/CUTLASS 源文件及 VERSION 与固定父树/子模块逐字节相符，预编译 `_C.so` 未做可复现构建证明。 |

SGLang `python/pyproject.toml` 固定 `sgl-deep-gemm==0.1.3`；Docker/CI 的 CUDA 12 配方取 cu129 wheel，CUDA 13 使用 PyPI 包。release workflow 虽默认 checkout `dev`，但允许覆盖 ref，不能用当前 dev 代替发布身份。whl 仓库 release 自身还包含其他项目资产，其 tag commit 也不是 DeepGEMM 源码 commit。这里是默认配方与固定包源码的静态对应，没有确认任何实际部署使用它。

构建脚本 `build_sgl_deep_gemm.sh` 将 **`sgl_deep_gemm/__init__.py`** 放入安装包 `deep_gemm`，并复制公共 Python 子模块与 JIT 头文件；预编译 TVM-FFI host `_C.so`。因此实际入口是定制 init → `csrc/tvm_ffi_api.cpp` → `csrc/apis/gemm.hpp`；该 host 库再 JIT 专家设备 kernel。普通 `deep_gemm/__init__.py` 和处于 `#if 0` 的旧 pybind 注册不能代替这个入口。

选定条件沿用 EP 框架页：CUDA SM90、普通 E4M3 激活/权重、每行 128 通道激活尺度、128×128 权重尺度、BF16 输出；JIT 开启，无 SBO overlap、FP4、UE8M0、psum layout、融合 Mega MoE 或附加 epilogue。SGLang 三个 activation/memory 优化开关沿用该页的 False 条件；LL 进一步固定默认 `Mmax=128`，因此每专家容量 $C=128P$，$P$ 为 EP rank 数。$D,I$ 分别为 hidden/intermediate 宽度，均按 128 整除。

**该包不能承接 wrapper 的所有分支。** SGLang `entrypoint.py` 有向 masked API 传 `enable_overlap/max_block_n/signal` 的 SBO 分支；匹配包定制 init、TVM-FFI 与 host masked API 均不接受这些参数。无 overlap 的本页调用契约相符，SBO 是已确认的静态 API 差异，不能从默认路径推广其支持状态；这里未运行该分支。

## 2. 两种输入布局与两次 GEMM

令本地专家数为 $L$，normal 接收的唯一 token 行数为 $R$，$A$ 为各专家按 128 对齐后行数之和。每次 GEMM 的归约维记 $K_g$，输出宽度记 $N_g$，避免与路由 top-k 混用。第一次 $K_g=D,N_g=2I$，第二次 $K_g=I,N_g=D$。

| 参数 | normal / contiguous | LL / masked |
| --- | --- | --- |
| FP8 激活 `qA` | $[A,K_g]$，按专家区间放置配对行 | $[L,C,K_g]$，每专家前 `masked_m[e]` 行有效 |
| FP32 激活恢复尺度 `S_A` | 逻辑 $[A,K_g/128]$，物理 stride $(1,A)$ | 逻辑 $[L,C,K_g/128]$，TMA stride $(C_aK_g/128,1,C_a)$，$C_a=4\lceil C/4\rceil$ |
| FP8 权重 `qB` | $[L,N_g,K_g]$，K-major | $[L,N_g,K_g]$，K-major |
| FP32 权重恢复尺度 `S_B` | $[L,N_g/128,K_g/128]$，普通连续布局 | $[L,N_g/128,K_g/128]$，普通连续布局 |
| BF16 输出 | $[A,N_g]$ | $[L,C,N_g]$ |
| 专家/行元数据 | int32 `m_indices[A]` | int32 `masked_m[L]`、标量 `expected_m` |

表内 $C=128P$，所以 $C_a=C$。权重的模型布局/加载与 EP 映射见框架页，不能仅凭 weight scale 名中含 `inv` 就在计算式中取倒数；当前 kernel 将两侧传入尺度相乘恢复局部点积。SGLang `deep_gemm.py` 的 `_run_contiguous_gemm/_run_masked_gemm` 与外部 `gemm.hpp` 是形状与调用依据。

normal 的 `ep_scatter` 对每个合法接收专家配对复制 FP8 值与尺度，并把**所有对齐行**的 `m_indices` 写成本地专家 ID。padding 的值/尺度由 empty 分配得到，没有初始化保证。第一 GEMM → BF16 SiLU×up → 每行每 128 通道量化 → 第二 GEMM 均处理这些连续容量行；默认 `SGLANG_OPT_FIX_MEGA_MOE_MEMORY=False`，激活与量化是两步。最后 `ep_gather` 只按有效 top-k 的 `output_index` 读取，乘对应 gate，以 FP32 累加，再转 BF16 $[R,D]$。padding 参与计算不等于进入返回贡献。

LL 无需 scatter，DeepEP 返回的 expert-batched 值/尺度直接进入 masked GEMM。第一 GEMM 输出 BF16，所选 fallback activation 只遍历 `token_index < masked_m[e]`；它把门值升为 FP32 计算 SiLU，再显式转回输入 BF16，然后参与 up 乘法与 group amax。不能把理想 SiLU×up 公式表述为一次无中间舍入的 FP32 运算。每组正下限为 `1e-10`，普通恢复尺度为 amax/448，量化值 clamp 后转 E4M3；第二 GEMM 使用相同有效计数。原模型 gate 留到 DeepEP combine，GEMM 内没有 gate、$\rho$、SiLU、bias 或 residual。依据：SGLang `ep_moe/kernels.py` 的 `silu_and_mul_masked_post_quant_fwd` 及 runner。

### 尺度转置改变布局，保留数值

SGLang normal `tma_align_input_scale`、LL `get_mn_major_tma_aligned_tensor` 与外部 `csrc/apis/layout.hpp` 将激活 scale 整理为 MN-major：相邻行的 FP32 scale 连续，一个 K group 到下一组的步长按 16 字节对齐。外部 `smxx_layout.hpp` 已匹配 stride 时返回同一 tensor，否则转换；设备 `smxx_layout.cuh` 的普通 FP32 分支复制值，没有转 UE8M0 或取 2 的幂。

LL 第一 GEMM 的通信尺度通常已满足这一布局；第二次 activation 产出的 row-major scale 需要转置。转置可遍历容量 $C$，而 activation 只写有效前缀，因此尾部 scale 没有模型数值语义。TMA 对齐、数值 scale 和有效行不能合为一个条件。

## 3. API、配置选择与 JIT

contiguous wrapper 调 `m_grouped_fp8_gemm_nt_contiguous`，masked 调 `fp8_m_grouped_gemm_nt_masked`；定制 init 的兼容名称分别进入 `m_grouped_fp8_fp4_gemm_nt_contiguous/masked`。FP8 路径没有传 FP4 recipe，host 默认 recipe 为 `(1,128,128)`，在 SM90 选择 `sm90_*_1d2d`，这里的 1D/2D 指激活每行尺度与权重二维块尺度。

host 检查 K-major FP8 输入、BF16 row-major 输出与 int32 元数据，转换激活 scale 布局，构造 `GemmDesc` 与 TMA descriptors。SM90 contiguous 默认 `BLOCK_M=128`，与 scatter 的专家对齐一致；masked 候选为 64/128；FP8 `BLOCK_K=128`，`BLOCK_N` 由候选、资源、scale 边界和 cluster 合法性筛选，不能一概写成 128。依据：`csrc/jit_kernels/heuristics/runtime.hpp`、`sm90.hpp`。

masked 的 `expected_m` 进入估计块数/波次的配置比较，实际任务来自 GPU `masked_m`。这是成本 heuristic，**没有逐候选运行计时的 autotune**。normal 默认按总对齐行数 $A$ 选配置，不接收每专家真实计数。SGLang 的 masked 暖编译可用全零 `masked_m` 与正 `expected_m`，进一步说明 hint 与有效计数是独立对象。

`sm90_fp8_gemm_1d2d.hpp` 生成包含设备头的模板实例，默认 `compiled_dims="nk"` 把 $N_g,K_g$ 编入模板，$M$ 保留运行参数；group 数、tile、stage、SM/cluster 与模式也进入代码。`LaunchRuntime::generate` 加入 include hash；`Compiler::build` 按 kernel 名、编译器 signature、flags、生成代码计算 cache 身份，命中则复用，否则生成 cubin，再按当前 CUDA stream launch。

SGLang execution hook 可按 kernel 类型、N/K/group 暖编译一批 M，contiguous M 先按外部 alignment 过滤；缓存目录来自 `SGLANG_DG_CACHE_DIR`，编译器选择可由 `SGL_DG_USE_NVRTC` 覆盖，默认 NVCC。暖编译仍调用 JIT 路径。PDL 的 runtime 开关与 SBO 的 signal 参数是不同接口；当前版本的 `get_compile_mode/set_compile_mode` 未在定制顶层导出，SGLang 用 `hasattr` 保护。

## 4. 调度如何选择专家与行

本核的 math warp-group 是 128 个线程，负责 64 行 WGMMA；所选 `BLOCK_M=64` 使用一组，`BLOCK_M=128` 使用两组。这与传统 warp32 的 `mma.sync` collective 不同。依据：`heuristics/sm90.hpp` 的 math thread 配置与设备核的 `threadIdx.x/128`。

共享 scheduler 使用 persistent CTA：第 $t$ 个迭代的任务号由 $t\times\text{numSMs}+\text{blockIdx.x}$ 得到，再映射到 M/N tile。cluster/swizzle 影响任务排列和 multicast，行的来源由以下布局规则决定。依据：`deep_gemm/scheduler/gemm.cuh` 的 `get_next_block/get_global_idx/is_computation_valid`。

### contiguous：专家 ID 在 tile 首行

默认 `BLOCK_M=128`；scheduler 读取 `m_indices[m_block*128]` 选择该 M tile 的专家，B 的 global 行坐标加 $eN_g$，B scale 加专家块偏移；A、A scale 和输出仍按全局连续行索引，不额外加 expert batch 偏移。

这要求每个 128 行 tile 属于同一专家，SGLang 对齐每专家区间满足条件。`m_indices` 不是任意逐行 gather：若一个 tile 混有不同专家，首行选择不能表达这些行各自的权重。

`is_computation_valid` 可按 math warp-group 的起始行 ID 是否非负跳过 WGMMA，但本 caller 给 padding 也写非负 ID，所以对齐尾部仍被计算。合法配对之外的内容可以不确定，后续 gather 根据有效 `output_index` 消费，保持模型语义。

### masked：有效计数选择 tile，不逐行保护尾部

scheduler 对每个专家分配 $\lceil\text{masked\_m}[e]/\text{BLOCK\_M}\rceil$ 个 M 块，再配 N 块；零计数专家跳过。A/D 的 flatten 坐标为 $eC+m_{\mathrm{block}}\text{BLOCK\_M}$，B 为 $eN_g+n_{\mathrm{block}}\text{BLOCK\_N}$。hint 小于真实有效计数也不裁掉任务。

`is_computation_valid` 只判断 math warp-group 起始偏移是否小于有效计数；**GEMM 没有逐行 masked store**。举一个布局例：$C=128P$，某专家有效 65 行，若选 `BLOCK_M=64`，调度两个 M tile。第二块从 64 起，仍执行 WGMMA，并可写 64–127；只有第 64 行属于有效输出。这个例子解释容量与有效前缀，不是 GPU 实测。

若选 `BLOCK_M=128`、有效行仅 3，第二个 math warp-group 可以跳 WGMMA，但仍等待/释放 pipeline barrier，全 tile 写回也不以有效行逐项 predicate。因此“masked”表示有效任务和结果前缀，不能推出尾部全零、完全不读写或没有开销。框架 activation 和通信 combine 的有效范围才限制后续模型贡献。

本页 $C=128P$ 同时按两种候选 M tile 对齐。DeepEP 的 $C\bmod4=0$ 是 FP32 scale 读取条件，单靠它不能证明任意非 tile 对齐容量跨专家的读写安全；非默认 Mmax 需要另沿完整链核对。

## 5. TMA、WGMMA 与恢复尺度

`sm90_fp8_gemm_1d2d.cuh` 分 producer 与 math 线程，维护多 stage 环形 shared memory。producer 等待 empty barrier，再用 TMA 把当前 tile 的 FP8 A/B 与 FP32 A scale 读入 shared；full transaction barrier 记录相应字节数。math 等 full barrier，消费后释放 empty barrier。multicast 是否可用取决于候选 cluster 与邻 CTA 合法性，不能把每 tile 都写成广播。

B 的 FP32 scale 由 math 线程从 global 读入 shared，**不是随 B 值一起 TMA 读取**。scale 按专家、N 的 128 行块、K 的 128 通道块索引；N tile 跨一个权重 scale 边界时保留最多两行尺度，并按 fragment 所属 N 块选择。A scale 则随每个 K stage 按行读取。依据：共享核的 TMA producer、B-scale loading 与 `num_former_iters` 分支。

WGMMA selector 为 M64、K32 的 E4M3×E4M3→FP32。一个 128 通道 stage 发四次 K32 WGMMA，第一次清局部累加器，后续累加；commit/wait 后，将局部 FP32 点积乘 A/B 恢复尺度，累入独立 FP32 `final_accum`。随后进入下一 K group。**尺度乘法位于每个 128 通道局部点积之后，不能改成全部 K 点积完成后乘一个总 scale。**

以 masked 行 $(e,m)$ 为记号，下面是整理者对执行顺序的概括；contiguous 把 $(e,m)$ 换成全局行和其专家 ID：

$$
Y_{e,m,n}\approx
\operatorname{RN}_{\mathrm{BF16}}\!\left[
\sum_{b=0}^{K_g/128-1}
S_A[e,m,b]\,
S_B[e,\lfloor n/128\rfloor,b]\,
\operatorname{Dot}_{\mathrm{FP32}}\!
\left(qA[e,m,128b:128(b+1)],
qB[e,n,128b:128(b+1)]\right)
\right].
$$

$b$ 是 128 通道 group，$n$ 是输出列，$S_A,S_B$ 是传入的 FP32 恢复乘子，Dot 表示该组四次 WGMMA 的局部累加，RN 表示 round-to-nearest BF16 转换。近似号保留输入量化及浮点累加/加乘的舍入；它不是原 BF16 理想 GEMM 的逐位等式，也不表示显式逐元素反量化为 BF16 后才乘。

最后 `__float22bfloat162_rn` 将 FP32 final accumulator 转 BF16，STSM 写 swizzled shared 输出，经 store fence/named barrier 发 TMA global store。输出 descriptor 按 $A\times N_g$ 或 flatten 的 $LC\times N_g$ 容量定义，对 masked_m 没有逐行知识。TMA global 边界与有效前缀属于不同层，容量内 padding 可以被写入。

本 caller 的 $K_g$ 按 128 整除，contiguous 的 $A$ 与每专家区间按 128 对齐，masked 容量按候选 M tile 对齐；这些条件避免相应 M/K 的容量尾块。$N_g$ 虽按 128 对齐，`BLOCK_N` 仍可能不整除它，核内用 `num_full_iters` 处理 N 尾部的 scale 选择，输出 TMA descriptor 以实际 $N_g$ 定义界限。B descriptor 把专家 N 轴 flatten 为 $LN_g$，不能描述成 TMA 独立 mask 每专家的 local N 尾部；这里仅说明合法输出列的接口，不推广任意不对齐 shape 的硬件 OOB 行为。

论文训练中每K组恢复、Dgrad/Wgrad与缓存/优化器精度的另一套责任，见[DeepSeek-V3 FP8训练](deepseek-v3-fp8-training.md)。本页固定推理源码不证明作者HAI-LLM训练kernel已核，FP32寄存器/类型声明也不能代替硬件实际累加精度的测量。

## 6. 两次专家计算的边界与复核范围

这里贯通了通信 payload → 行/scale 布局 → API/JIT → 专家选择 → raw FP8 点积 → 分组恢复/FP32 累加 → BF16 输出。两次 GEMM 之间的 SiLU、动态重新量化以及返回时的 gate/归约仍由框架和 DeepEP 承接，不能从单次 kernel 输出推出整个 MoE 的精度或吞吐。

本页不覆盖 SM100/FP4/UE8M0、psum/Mega MoE、SBO/graph、多 stream、任意 Mmax 或全部 tail/heuristic 组合、完整寄存器映射及设备指令吞吐。没有安装、JIT、GPU 数值、性能、模型或服务实验。所选默认接口的静态贯通已完成，其他分支按具体问题和任务开展。
