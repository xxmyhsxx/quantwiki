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
  - raw/articles/2026-09-22/ptx-mma-layout/article.md
  - raw/repositories/2026-09-21/marlin/source/marlin/marlin_cuda_kernel.cu
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/layers/quantization/awq_triton.py
  - raw/repositories/2026-09-21/sglang/source/python/sglang/srt/layers/quantization/awq/awq_triton.py
  - raw/repositories/2026-09-21/triton/source/python/triton/compiler/compiler.py
  - raw/repositories/2026-09-21/triton/source/third_party/nvidia/backend/compiler.py
updated: 2026-09-22
---

# Tensor Core 与量化 GEMM：从 Tile 到寄存器和编译产物

分块矩阵乘把 $C=AB$ 拆成可复用的 tile，但 Tensor Core 还要求 warp 中每个线程持有规定的寄存器片段。量化 GEMM 需要把文件中的打包权重转换到这个计算接口；“INT4 权重”并不表示设备执行 INT4 MMA。

本页先用原始 Marlin 的显式 FP16 MMA 解释硬件接口，再沿 vLLM/SGLang 现成 AWQ Triton 的索引、解包、`tl.dot` 和 split-K 走到编译检查。两条路径用于解释不同层次，不声称使用同一种量化格式、累加精度或分派入口。

## 1. 一个 warp 共同计算什么

Marlin `mma` 发出：

```ptx
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    {d0,d1,d2,d3}, {a0,a1,a2,a3}, {b0,b1}, {c0,c1,c2,c3};
```

这是一整个 warp 共同计算 $D_{16\times8}=A_{16\times16}B_{16\times8}+C_{16\times8}$，不是每个线程各算一个 16×8 输出。A 每线程 4 个 32 位寄存器、每寄存器两个 half；B 每线程 2 个这样的寄存器；C/D 每线程 4 个 FP32。

元素数可以核对：$32\times8=256$ 个 A 元素，$32\times4=128$ 个 B 元素，$32\times4=128$ 个输出。输入是 FP16、累加为 FP32，后续跨 warp/跨块归约和输出写回仍可能使用其他精度，见 [浮点与累加](../fundamentals/numeric-formats/floating-point-and-accumulation.md)。

这里 `.row.col` 约束 MMA 操作数布局，不直接声明原始权重文件是行主序还是列主序。所有 lane 需按指令规定共同参与；尾部 mask 通常作用于加载与写回，不能随意让部分 lane 跳过 collective 指令。

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

本轮完成源码、PTX 坐标规则与 CPU 教学映射检查，没有编译这些 CUDA/Triton kernel、采集其 IR/SASS 或运行 GPU profiler。所列映射是明确指令的接口，不是对未运行编译产物的猜测。

## 来源身份

- [NVIDIA PTX ISA，CUDA 11.8 归档](https://docs.nvidia.com/cuda/archive/11.8.0/parallel-thread-execution/index.html)，2026-09-22 获取，`ptx-mma-layout-20260922-fc515705`；Matrix Fragments for mma.m16n8k16 with floating point type、Warp-level matrix load instruction: ldmatrix，使用正文坐标与寄存器规则。
- [Marlin](https://github.com/IST-DASLab/marlin/tree/1f25790bdd49fba53106164a24666dade68d7c90)，`1f25790bdd49fba53106164a24666dade68d7c90`；`marlin_cuda_kernel.cu` 的 `mma`、`ldsm4`、`dequant` 与主循环数据搬运。
- [vLLM](https://github.com/vllm-project/vllm/tree/568afb3a13806beb53bb2e6bd518269357b237c0)，`568afb3a13806beb53bb2e6bd518269357b237c0`；`layers/quantization/awq_triton.py` 的 GEMM 与包装。
- [SGLang](https://github.com/sgl-project/sglang/tree/2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc)，`2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc`；`python/sglang/srt/layers/quantization/awq/awq_triton.py` 的对应路径。
- [Triton](https://github.com/triton-lang/triton/tree/81a46fa0c04526e5df55a018ecfab72ff922f592)，`81a46fa0c04526e5df55a018ecfab72ff922f592`；`python/triton/compiler/compiler.py` 与 `third_party/nvidia/backend/compiler.py`。用于解释检查接口，未验证与所选框架环境的版本兼容性。
