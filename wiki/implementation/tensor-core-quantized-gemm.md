---
title: Tensor Core 与量化 GEMM：从 Tile 到寄存器和编译产物
type: implementation
tags:
  - kernels
  - matmul
  - cuda
  - triton
  - weight-quantization
sources:
  - raw/articles/2026-09-21/nvidia-tensor-cores
  - raw/articles/2026-09-22/ptx-mma-layout
  - raw/repositories/2026-09-21/marlin
  - raw/repositories/2026-09-21/vllm
  - raw/repositories/2026-09-21/sglang
  - raw/repositories/2026-09-21/triton
updated: 2026-10-15
---

# Tensor Core 与量化 GEMM：从 Tile 到寄存器和编译产物

分块矩阵乘把 $C=AB$ 拆成可复用的 tile。本页的 WMMA/mma.sync 路径由一个 warp 共同计算，输入和累加状态分布在线程的寄存器片段中；量化 GEMM 还需把打包权重转换到相应接口。“INT4 权重”并不表示设备执行 INT4 MMA。SM100 的另一类协作与块尺度接口见[CUTLASS 块缩放 GEMM](cutlass-blackwell-block-scaled-gemm.md)。

本页从 warp 共同计算的矩阵块出发，对照 WMMA 隐藏的 fragment 与 Marlin 显式 PTX 的寄存器布局，再沿 vLLM/SGLang 现成 AWQ Triton 的索引、解包、`tl.dot` 和 split-K 走到编译检查。各路径的版本、量化格式、累加精度与分派入口分别说明。

## 1. 一个 warp 共同计算什么

Marlin `mma` 发出：

```ptx
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    {d0,d1,d2,d3}, {a0,a1,a2,a3}, {b0,b1}, {c0,c1,c2,c3};
```

这是一整个 warp 共同计算 $D_{16\times8}=A_{16\times16}B_{16\times8}+C_{16\times8}$，不是每个线程各算一个 16×8 输出。A 每线程 4 个 32 位寄存器、每寄存器两个 half；B 每线程 2 个这样的寄存器；C/D 每线程 4 个 FP32。

元素数可以核对：$32\times8=256$ 个 A 元素，$32\times4=128$ 个 B 元素，$32\times4=128$ 个输出。输入是 FP16、累加为 FP32，后续跨 warp/跨块归约和输出写回仍可能使用其他精度，见 [浮点与累加](../fundamentals/numeric-formats/floating-point-and-accumulation.md)。

这里 `.row.col` 约束 MMA 操作数布局，不直接声明原始权重文件是行主序还是列主序。所有 lane 需按指令规定共同参与；尾部 mask 通常作用于加载与写回，不能随意让部分 lane 跳过 collective 指令。

### 1.1 WMMA 怎样加载、累加和写回一个输出块

CUDA 9/V100 教程的图 1 描绘单个 Volta Tensor Core 的 4×4×4 操作；图 5 的 WMMA 接口则由**完整 warp**共同计算 16×16×16。前者是硬件单元粒度，后者是程序可调用的协作粒度；上面的 `m16n8k16` 又是另一条明确 shape 的 PTX 接口。这里按 CUDA 9 的 preview API 解释所保存示例，不能由它确定其他架构、数据类型或后来接口的支持范围。

示例计算 $C\leftarrow\alpha AB+\beta C$，其中 $A[M,K]$、$B[K,N]$ 是 half，累加片段和 $C[M,N]$ 是 float。`wmma::fragment` 用模板参数声明 A/B/accumulator 角色、16×16×16 shape、类型及 A/B 的行列布局，寄存器中的元素到矩阵坐标的映射由 API 隐藏。它仍然由每个 lane 持有片段，但程序不能给 `fragment.x[i]` 套用下一节显式 PTX 的 lane/坐标表。

这份示例的三个矩阵均按**列主序**保存，未转置且无额外 padding，地址按 `row + col*ld` 计算：

| 矩阵 | leading dimension | A/B 当前 K 起点 $i$ 的块或 C 读写起点 |
| --- | --- | --- |
| A | `lda=M`，相邻列相隔 M 个元素 | `aRow + i*lda`，其中 `aRow=16*warpM` |
| B | `ldb=K`，相邻列相隔 K 个元素 | `i + bCol*ldb`，其中 `bCol=16*warpN` |
| C | `ldc=M`，相邻列相隔 M 个元素 | `aRow + bCol*ldc` |

`warpM/warpN` 选择一个 16×16 输出块，计算顺序为：

1. `fill_fragment(acc_frag, 0)` 产生 FP32 部分和，尚不含原 C。
2. 以 `i+=16` 沿 K 前进，`load_matrix_sync` 分别把 A 的第 i 列起的块、B 的第 i 行起的块装入各自 fragment；`mma_sync(acc_frag,a_frag,b_frag,acc_frag)` 把这次乘积加到同一个输出部分和中。原文 “The inner loop” 的 A rows/B columns 文字与下标不一致，这里按代码的 `aCol=i、bRow=i` 解释。
3. K 循环结束后，将原 C 块加载到 `c_frag`。两个 accumulator fragment 的模板参数相同，各 lane 对自己持有的全部元素统一执行 `alpha*acc_frag.x[i]+beta*c_frag.x[i]`；这种统一逐元素运算无需知道逻辑行列。按特定行列施加不同操作则需要额外的坐标依据。
4. `store_matrix_sync(...,mem_col_major)` 将结果写回同一列主序 C 块。保存代码即使 `beta=0` 仍先加载 C；FP32 累加也保留 FP16 输入造成的数值差异，见 [浮点与累加](../fundamentals/numeric-formats/floating-point-and-accumulation.md)。

这些步骤要求同一 warp 的参与线程使用一致的输出块并共同到达匹配的 WMMA 调用。保存正文省略 host launch，因此只靠设备函数不能核定完整启动配置。其 guard 也仅判断块起点：例如 M=17 时 `aRow=16` 仍满足 `aRow<M`，却没有保护随后的整块 16 行访问；K 尾块和 C 写回也没有逐元素 mask。这是整理者按下标给出的静态反例，未运行。要使用示例，需要另核完整块的有效范围、存储/对齐和 API 要求，不能把起点检查当作任意 M/N/K 的尾块处理。

## 2. lane、寄存器元素与矩阵坐标

以下按 CUDA 11.8 PTX 的 `mma.m16n8k16` **FP16/BF16 输入、FP32 累加**布局整理。设 lane 为 $l$，$g=\lfloor l/4\rfloor$，$t=l\bmod4$。小写 $a_i,b_i,c_i$ 在这里指解开 half2 后的标量元素；不要与上面 PTX 的打包寄存器编号混淆。

| 片段 | 标量编号 | 行 | 列 |
| --- | --- | --- | --- |
| A | $i=0,\ldots,7$ | $g+8(\lfloor i/2\rfloor\bmod2)$ | $2t+(i\bmod2)+8\lfloor i/4\rfloor$ |
| B | $i=0,\ldots,3$ | $2t+(i\bmod2)+8\lfloor i/2\rfloor$ | $g$ |
| C/D | $i=0,\ldots,3$ | $g+8\lfloor i/2\rfloor$ | $2t+(i\bmod2)$ |

教学例：lane 0 的 A 元素依次来自 `(0,0),(0,1),(8,0),(8,1),(0,8),(0,9),(8,8),(8,9)`；B 来自 `(0,0),(1,0),(8,0),(9,0)`；输出负责 `(0,0),(0,1),(8,0),(8,1)`。lane 4 则把 A/输出的基础行与 B 的列移到 1。

一个 lane 自己持有的 A/B 元素不足以独立计算其输出，MMA 在 warp 内按硬件接口组织这些操作数。把一个连续数组平均分给 32 个线程，并不能自动得到上表布局；这就是加载重排与离线 packing 的作用。表格不适用于稀疏 MMA、其他 shape 或 Hopper 的另一套指令接口。

## 3. `ldmatrix` 怎样连接共享内存与 A 片段

Marlin `ldsm4` 使用 `ldmatrix.sync.aligned.m8n8.x4.shared.b16`。`.x4` 表示四个 8×8 的 16 位矩阵，各 lane 最终取得 4 个 32 位寄存器。对于 `.x4`，lane 0–7 提供第一块的八个行起点，8–15 提供第二块，依此类推。**提供行地址的 lane 与拿到整行的线程不是一一对应**：一组四个 lane 各接收一行中的两个 half。

将一个逻辑 16×16 A 分为左上、左下、右上、右下四块，并按此顺序提供行地址，就能得到上一节 A 的四个 half2 片段。实际 shared 中可以带 XOR swizzle；此时提供的是经过 swizzle 的行起点，不能直接用未经变换的二维下标。行起点需满足该 16 字节行片段的对齐要求。

`.sync` 保证 warp 共同执行该指令，不等于替代前面的异步拷贝完成等待或跨 warp 同步。Marlin 的 `fetch_to_shared` → `wait_for_stage` → `fetch_to_registers` 对应 global→shared、等待可见、shared→register；其 `wait_for_stage` 中同时有 `cp.async.wait_group` 与 `__syncthreads()`。阶段复用见 [异步拷贝与多级流水](gpu-async-copy-pipelines.md)。

## 4. 量化权重怎样进入 B 片段

同一源码的 `fetch_to_registers` 直接读出打包的权重整数；`matmul` 调用 `dequant` 得到每线程四个 half，按需要乘 group scale，再送入 `mma`。它没有把完整 FP16 权重先写回 global。INT4 是存储码，设备主乘法为 FP16 Tensor Core。

`dequant` 依赖离线排列，使各半字节解开后正好对应 B 片段的行列。仅把普通顺序的八个 INT4 塞进 int32，虽然字节数相同，仍可能把错误元素送入 MMA。这里应核对“逻辑 $(k,n)$ → packed 字段 → lane/寄存器元素”的完整关系，已有格式分析见 [AWQ/Marlin 反量化](weight-only-dequant-kernels.md)。

若尺度只随输出列变化，可以在完整 K 归约后乘；若尺度随 K group 变化，必须在合适的组内阶段施加，不能把所有组无条件合并后只乘一个 scale。Marlin `matmul` 的 `group_blocks != -1` 分支正是在 B 片段进入 MMA 前缩放；数学原因见 [量化矩阵乘缩放](quantized-matmul-scaling-execution.md)。

## 5. vLLM/SGLang AWQ Triton：从 program 到 K 循环

两份所选 `awq_triton.py:awq_gemm_kernel` 的主线一致，包装输入为 $A[M,K]$、`qweight[K,N/8]`、`qzeros[K/G,N/8]`、`scales[K/G,N]`。N 是解包后的输出维，不能把 packed 列数当 N。

第一维 program ID 线性枚举 $(M,N)$ 输出 tile，第二维枚举 split-K 分片 $z$。设块大小为 $B_M,B_N,B_K$、分片数为 $S$，第 $j$ 轮归约处理

$$k_{\mathrm{start}}=(jS+z)B_K.$$

因此不是每个分片连续吃完一段 K，而是轮转领取 K tile。对 $B_K=32,S=2,K=128$，分片 0 处理 `[0,32)` 与 `[64,96)`，分片 1 处理另外两块。

每轮从 int32 中按 `[0,4,1,5,2,6,3,7]` 对应的 shift 解出 8 个 INT4，并以相同顺序解 zeros；广播本组 scale，形成 $(q-z)s$ 的浮点 B tile，调用 `tl.dot`。这组 shift 是 AWQ 格式约定，不是通用 INT4 规则。

源码按 **K tile 起点**选择一个 group 并广播到整块，所以一个 tile 不能跨越两个采用不同元数据的 group。默认 $B_K=32$ 与支持的正常 group sizes 相容；自行改变块大小时必须重新核对，wrapper 的 shape 断言不替代这个计算前提。矩阵尾部则靠 K/M/N mask 处理。

每个 split-K 分片写一个独立 $[M,N]$ 平面，包装最后 `sum(0)`；这增加临时存储、读写和一次归约。$M/N$ 很小时增加 S 能提供更多 program，但不是无成本增加并行度。

特别注意 `accumulator_dtype = c_ptr.type.element_ty`，`tl.dot(..., out_dtype=accumulator_dtype)`，缓冲由 `scales.dtype` 创建。不能把这段源码直接描述成 Marlin 的 FP32 累加链；改变 S 还可能改变低精度归约顺序。数值检查应同时对照解包后的参考乘法与原接口的舍入契约。

这些函数的存在只证明可读的实现路径；实际模型是否选中它们，由量化配置与后端分派决定，见 [AWQ 实现](awq-implementation.md)。

## 6. `tl.dot` 到硬件指令之间还需要查什么

Triton 所选 NVIDIA `CUDABackend.add_stages` 登记 TTIR → TTGIR → LLVM IR → PTX → cubin。`CompiledKernel.asm` 保存这些阶段产物，`asm['sass']` 可从 cubin 反汇编；`n_regs/n_spills` 在二进制加载时取得。PTX 是虚拟指令层，实际执行指令应回看 SASS。

在具备目标 GPU 和可用环境时，可对一次**已成功编译并启动**的 kernel 返回对象 `k` 做如下定位：

```python
# k = awq_gemm_kernel[grid](...实际且合法的参数...)
print(k.asm['ttgir'])  # tile 与线程布局转换
print(k.asm['ptx'])    # MMA、加载、转换与局部内存操作
print(k.asm['sass'])   # 最终机器指令
print(k.n_regs, k.n_spills, k.metadata.shared)
```

这只是检查接口示意，本次未执行。`tl.dot` 不承诺在所有架构和 dtype 下生成上一节的同一条 MMA；不能把原始 Marlin 的手写 PTX 当作 Triton 已生成的结果，也不能把本地独立保存的 Triton 版本自动视为两库的运行依赖版本。

低精度浮点和块缩放点积的另一条接口见 [FP4/FP8 块缩放执行](block-scaled-matmul-execution.md)：它追踪打包元素、scale 重排与 `tl.dot_scaled`，并明确归档教程的 NVIDIA 硬件入口；不要将本页 SM80/FP16 MMA 的路径直接套用到该实例。

## 7. 一次性能诊断怎样形成可检验解释

先固定 shape、dtype、group、S、tile、warp/stage、GPU 与版本，建立数值参考及基线。以下是待运行时使用的判断路线，不是本次测得的瓶颈：

| 观察 | 候选解释 | 应补的检查 |
| --- | --- | --- |
| 小 M 时很多 SM 无足够工作 | 输出 tile 太少 | 计算 program 数，测试 split-K 的收益是否覆盖额外归约 |
| 增大 tile 后变慢、寄存器/溢出增加 | 局部复用收益被资源成本抵消 | 比较相同工作量的寄存器、local 访问、驻留与实测耗时 |
| global 流量低但延迟高 | 解包、地址计算、同步或依赖链限制 | 在生成代码中定位，再用 profiler 核对相关吞吐/停顿；不能只凭带宽未满下结论 |
| shared 访问代价偏高 | 布局与实际访问片段不匹配 | 按 lane 地址核对 bank，再对照 shared 访问统计 |
| GEMM 主循环变快、总调用未变快 | split-K 归约或其他阶段占比上升 | 将主循环、归约和完整包装分别计时 |

一次只改变能检验假设的配置，候选通过数值检查后再比较性能；profiling 有采集成本，最终延迟用一致的独立计时复测。方法见 [配置与自动调优](kernel-configuration-and-autotuning.md)及 [正确性与测量](kernel-correctness-and-benchmarking.md)，端到端收益还需进入 [服务评测](serving-performance-evaluation.md)。

Marlin/PTX 部分保留既有源码、PTX 坐标规则与 CPU 教学映射的检查范围；本次 WMMA 增量核读归档正文和配图。没有据此编译这些 CUDA/Triton kernel、采集其 IR/SASS 或运行 GPU profiler。所列坐标映射对应明确的 PTX 指令接口。

## 8. CUDA 9 库入口与图表的适用范围

同一历史教程展示的库入口与上面的手写 WMMA 分别核对：

| CUDA 9/V100 时代的路径 | 原文给出的 Tensor Core 使用条件 |
| --- | --- |
| cuBLAS GEMM | 显式设置 `CUBLAS_TENSOR_OP_MATH`；`k、lda、ldb、ldc` 为 8 的倍数，`m` 为 4 的倍数；文档列 half/single 输入输出，保存调用例使用 FP16 矩阵与 FP32 compute。原文称不满足这些规则时回退。 |
| cuDNN convolution | 设置 `CUDNN_TENSOR_OP_MATH`，选择 `ALGO_1`（forward 为 `IMPLICIT_PRECOMP_GEMM`）；输入与输出**通道数**为 8 的倍数，输入/filter/输出为 half。所列 forward 示例用 FP32 compute，batch 为 1。 |

这些是该版本文档的条件；不能据此推导现代库的全部选择规则、把所有维度都约束为 8 的倍数，或断言符合形状就已经核实当前运行路径。原文说明 Tensor Core 与普通路径的结果不保证逐位相同。保存的 cuBLAS 片段还存在 `T_ELEM_IN A` 声明却赋指针、`devPtrA=0` 后索引及未定义变量等问题，cuDNN 示例省略多个对象；它们提供接口阅读线索，不构成已验证可执行程序。

图 3 比较 Tesla V100 Tensor Cores 与 Tesla P100，FP16 输入、FP32 compute，方阵 M=N=K 为 512/1024/2048/4096。图 4 比较 Alexnet、VGG-A、Inception_v3、Resnet50 各网络**卷积层**加速的几何均值，双方 FP16 I/O、FP32 math，V100 使用 Tensor Cores，P100 使用 FMA。它们是作者报告的特定算子/跨代硬件结果；开头 125 TFLOPS 与相对 P100 FP32 的 12× 是峰值口径，均不能代替同硬件开关对比、端到端训练时间或任务质量验证。保存材料未给完整计时协议，本次也未取得外链完整示例、核现代 API 或运行 GPU。

## 来源身份

- Programming Tensor Cores in CUDA 9 | NVIDIA Technical Blog（`https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/`），`snapshot-2026-07-02`，正文获取于 `2026-07-02T02:41:08+08:00`；完整保存正文与 2026-09-21 补采图 1–5，使用 CUDA 9/V100 的 WMMA 数据流、历史库规则和图表条件；原 HTML、外链完整代码/API 及运行未核。

- NVIDIA PTX ISA，CUDA 11.8 归档（`https://docs.nvidia.com/cuda/archive/11.8.0/parallel-thread-execution/index.html`），2026-09-22 获取，`ptx-mma-layout-20260922-fc515705`；Matrix Fragments for mma.m16n8k16 with floating point type、Warp-level matrix load instruction: ldmatrix，使用正文坐标与寄存器规则。
- Marlin（`https://github.com/IST-DASLab/marlin/tree/1f25790bdd49fba53106164a24666dade68d7c90`），`1f25790bdd49fba53106164a24666dade68d7c90`；`marlin_cuda_kernel.cu` 的 `mma`、`ldsm4`、`dequant` 与主循环数据搬运。
- vLLM（`https://github.com/vllm-project/vllm/tree/568afb3a13806beb53bb2e6bd518269357b237c0`），`568afb3a13806beb53bb2e6bd518269357b237c0`；`layers/quantization/awq_triton.py` 的 GEMM 与包装。
- SGLang（`https://github.com/sgl-project/sglang/tree/2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc`），`2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc`；`python/sglang/srt/layers/quantization/awq/awq_triton.py` 的对应路径。
- Triton（`https://github.com/triton-lang/triton/tree/81a46fa0c04526e5df55a018ecfab72ff922f592`），`81a46fa0c04526e5df55a018ecfab72ff922f592`；`python/triton/compiler/compiler.py` 与 `third_party/nvidia/backend/compiler.py`。用于解释检查接口，未验证与所选框架环境的版本兼容性。
