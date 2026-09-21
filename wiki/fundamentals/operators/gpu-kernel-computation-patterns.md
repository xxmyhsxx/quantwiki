---
title: GPU 算子的计算模式：逐元素、归约与分块矩阵乘
type: concept
tags:
  - kernels
  - gpu
  - matmul
  - reduction
sources:
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/layers/batch_invariant.py
  - raw/repositories/2026-09-21/sglang/source/python/sglang/jit_kernel/csrc/elementwise/fused_add_rmsnorm.cuh
  - raw/repositories/2026-09-21/triton/source/python/tutorials/01-vector-add.py
  - raw/repositories/2026-09-21/triton/source/python/tutorials/02-fused-softmax.py
  - raw/repositories/2026-09-21/triton/source/python/tutorials/03-matrix-multiplication.py
  - raw/repositories/2026-09-21/triton/source/python/tutorials/05-layer-norm.py
  - raw/repositories/2026-09-21/cuda-samples/source/cpp/0_Introduction/matrixMul/matrixMul.cu
updated: 2026-09-22
---

# GPU 算子的计算模式：逐元素、归约与分块矩阵乘

算子公式规定结果，kernel 还要决定谁负责计算、数据在哪里复用、何时交换局部结果。逐元素、归约和矩阵乘的差别，主要在这些数据依赖上；融合与分块是在依赖允许的范围内减少搬运和调度成本。

本页以 Triton 官方教程 01/02/03/05 及 NVIDIA `MatrixMulCUDA` 为来源，先读 [张量布局与接口](tensor-layout-and-kernel-contracts.md)，硬件概念见 [GPU 执行与存储](../hardware/gpu-execution-and-memory-hierarchy.md)。公式中的流量和分块算例为整理者推导，不是性能实测。

## 1. 逐元素：每个输出有独立的拥有者

对 $z_i=x_i+y_i$，每个 i 的结果与其他位置无关。向量加法教程把连续下标分给不同 program，在尾块同时保护读写。如果每个有效 i 恰好写一次，且输出不覆盖其他工作单元尚需的输入，就不需要跨 program 归约。

若输入输出都是 s 字节元素，理想主数据流量为 $3ns$：读取两路、写一路；运算量 n 次加法，所以 $I\approx1/(3s)$ FLOP/Byte。FP32 时约为 $1/12$，意味着在大尺寸、带宽主导的理想模型中，减少数据搬运比减少这一次加法更有意义。小向量仍可能受启动开销限制，不能仅由 I 判定真实瓶颈。

增加 BLOCK_SIZE 会让每个 program 多处理元素，却也可能增加寄存器需求或减少可调度 program 数。它没有“越大越快”的单调关系。

## 2. 归约：一个输出依赖多个输入

求和 $s=\sum_{j=0}^{N-1}x_j$ 可以让不同线程先算局部和，再通过树形合并得到结果。并行化改变加法结合顺序，浮点运算通常不能保证逐位相同；累加 dtype 与归约次序都是算子的数值约定。

若一行不能由一个工作单元有效容纳，可以先输出多个部分和，再用后续 kernel 合并，或采用满足条件的原子合并。这会增加中间流量、启动或竞争。普通 block 屏障不能同步整个 grid，不能通过在一个 block 中等待其他普通 block 的共享内存来替代全局归约。

### softmax：稳定化和 mask 必须同时成立

Triton `02-fused-softmax.py:softmax_kernel` 对一行先减最大值：

$$m=\max_jx_j,\qquad u_j=\exp(x_j-m),\qquad y_j=\frac{u_j}{\sum_ku_k}.$$

对有限输入，减去 m 保持实数域的 softmax，并使指数输入不大于零，避免大正数指数溢出。长度 N 扩展到 $B=2^{\lceil\log_2N\rceil}$ 时，失效位置加载为 $-\infty$：它不会改变有效元素最大值，其指数为零，也不会改变分母。最终只写前 N 项。

反例：一行只有 `[-2,-3,-4]`，若补到 4 项时把失效位置填 0 后也纳入 softmax，分母就多了一个 $e^0$，有效输出的和不再为 1。`mask` 不是单纯的越界开关，填充值也参与数学语义。若整行被屏蔽，或有效数据全为 $-\infty$，会出现 $-\infty-(-\infty)$；教程未定义这种注意力 mask 语义，接入时应单独规定，不能声称有限输入推导已覆盖。

教程使用近似 `tl.exp`，所以公式等价也不保证与其他指数实现逐位相同。程序通过 `row_start=program_id`、`row_step=num_programs` 循环分担多行，不必为每一行创建一个独立 program；每次循环仍把整行逻辑块放在片上处理，超长行的资源需求限制其适用范围。

### LayerNorm：mask 之后的运算也可能污染归约

`05-layer-norm.py:_layer_norm_fwd_fused` 计算

$$\mu=\frac1N\sum_jx_j,\quad v=\frac1N\sum_j(x_j-\mu)^2,\quad y_j=\frac{x_j-\mu}{\sqrt{v+\epsilon}}w_j+b_j.$$

源码将输入转为 FP32 累加，以真实 N 为分母。求均值时无效 lane 填零；求方差时还必须把无效 lane 的 $x-\mu$ 重新置零，否则先填零再做减均值，会给每个 padding 项加入 $\mu^2$。源码中的 `tl.where(cols < N, x - mean, 0.)` 正是这个步骤。

教学例：有效数据 `[1,3,5]` 补到 4 项，均值是 3，方差应为 $8/3$。若无效项错误贡献 $(-3)^2$，方差变成 $17/3$。公式中的除数是 N，不是补齐长度 B，也不是统计估计时的 N−1。

本节研读的是前向部分，没有将教程的 backward 或其他归一化算子宣称为已 ingest。

### 推理库中的 RMSNorm：归约范围与数据保留

RMSNorm 使用行内二阶矩 $s=\sum_jx_j^2/D$，再计算 $y_j=x_jw_j/\sqrt{s+\epsilon}$，不减均值。分块只改变统计量的计算方式，不能把整行变成多个各自归一化的小行。若 D=2500、每段 B=1024，三段有效长度为 1024、1024、452，先合并三段平方和，再除以 2500；分别求三个 RMS 后直接拼接通常不等价。

一个实现可以在得到统计量后重新读输入，也可以跨归约保留输入的片上状态。前者增加逻辑读取，后者增加活跃状态与潜在寄存器压力；“一次 kernel”不能推出“只读一次”。残差融合还可能要求返回相加后的中间状态，输出不能只按归一化结果理解。

这些问题在[vLLM 与 SGLang 的 RMSNorm](../../implementation/rmsnorm-cuda-triton-kernels.md)中有具体实现：固定分段的 Triton、整行 Triton、CUB 行归约和显式 warp→block 两级归约。无有效元素的线程也可能必须参加 collective；mask 数据访问与退出线程不是同一操作。

## 3. 矩阵乘：用一个数据块支撑多个输出

对 $C=AB$，每个输出 $C_{mn}=\sum_kA_{mk}B_{kn}$。Triton `03-matrix-multiplication.py` 把输出分成 $B_M\times B_N$ 的 tile，每个 program 保留这一 tile 的累加器，并以 $B_K$ 遍历归约维：

$$C_{\rm tile}\mathrel{+}=A_{B_M\times B_K}B_{B_K\times B_N}.$$

一次 K 分块加载 $B_MB_K+B_KB_N$ 个元素，贡献约 $2B_MB_NB_K$ FLOPs；忽略输出和元数据、假设两路每元素 s 字节，有

$$I_{\rm tile}\approx\frac{2B_MB_N}{s(B_M+B_N)}.$$

这解释了为什么扩大 M/N tile 可以提高块内复用，而仅扩大 K tile 不会在这个简化公式中提高 I：$B_K$ 已约掉。K tile 仍影响循环次数、流水和存储需求，不能据约分认定它不影响性能。

例如 FP16、$B_M=B_N=32$、$B_K=32$，每步读 4096 字节、计算 65536 FLOPs，I=16；M/N 增到 64、K 不变，I=32，但累加器元素从 1024 增至 4096。累加器分布在整个 program 的线程中，并非单个线程持有全部元素；寄存器压力仍可能降低驻留数量。

### 复用需要两次同步

NVIDIA `matrixMul.cu:MatrixMulCUDA` 是教学 CUDA 实现：线程先将 A/B tile 搬入共享内存，第一次 `__syncthreads()` 保证可读取完整 tile；每线程计算一个输出局部和，第二次屏障保证所有消费者读完，下一轮才覆盖共享缓冲。

漏掉第一次可能读取未初始化或旧值；漏掉第二次可能在慢线程还读取旧 tile 时被快线程覆盖。这个示例没有完整尾块保护，不能当作支持任意 M/N/K 的通用 GEMM。

### 三条维度边界分别处理

Triton 示例在 M/N 加载坐标上取模，使冗余输出行列读取合法输入，最终 store 再屏蔽冗余位置；在 K 维使用带 `other=0` 的 mask，防止填充项进入有效输出的点积。两种办法解决不同问题：K 若直接取模重用输入，会给有效输出多加一段乘积。M 或 N 为零也不能继续执行取模方案，需要包装层先处理。

FP32 累加器在 epilogue 中可以先应用激活，再转成 FP16 输出。若参考路径先落盘 FP16 再应用激活，两者的舍入位置不同，融合验证必须包含这个差别。

## 4. 融合与重排分别减少什么

Triton softmax 教程的逐步 PyTorch 实现总计读写 $8MN+4M$ 个元素，理想融合路径只需读写 $2MN$ 个元素。比值 $4+2/N$ 是该数据流模型的流量比，不是实际加速保证，更不是对框架原生 fused softmax 的通用比较。流量换成字节时还要乘元素大小。

融合让中间值留在片上，减少全局读写和 kernel 启动；但较大的中间状态可能增加寄存器使用、spill 或缩小可驻留并行度，因此融合收益需要测量。

矩阵乘教程的 grouped program ID 映射则促使相邻输出 tile 复用 L2 中的 A/B 块。它重新安排工作与数据局部性，不改变矩阵的数学结果，也不承诺 GPU 严格按照 program ID 次序执行；不能把这种排序当作同步依赖。

进一步选择 tile、warp 数和流水阶段时，需要同时计算片上资源、尾部浪费与工作块数量，见 [Kernel 配置选择与自动调优](../../implementation/kernel-configuration-and-autotuning.md)。把加载与当前 tile 计算重叠时，还要显式维护完成等待和缓冲生命周期，见 [异步拷贝与多级流水](../../implementation/gpu-async-copy-pipelines.md)。

## 5. 怎样回到量化算子

[AWQ/Marlin 反量化内核](../../implementation/weight-only-dequant-kernels.md) 把“读整数容器 → 解包 → 应用 scale/zero → 矩阵乘”组合起来。这里的核心问题仍是：位布局如何映射给线程、分组参数在哪里复用、是否先写出完整浮点矩阵，以及累加和归约用什么精度。

先用 [roofline](../hardware/arithmetic-intensity-and-roofline.md) 估计值得优化的数据流，再用 [正确性与性能测量](../../implementation/kernel-correctness-and-benchmarking.md) 判断实际结果。本页的小尺寸地址、mask、分块流量关系已做独立 CPU 教学计算；尚未编译 CUDA/Triton 或验证任何 GPU 性能。

## 来源身份

| 来源 | 版本或快照 | 核对范围 |
| --- | --- | --- |
| [Triton 官方教程与工具](https://github.com/triton-lang/triton/tree/81a46fa0c04526e5df55a018ecfab72ff922f592) | `81a46fa0c04526e5df55a018ecfab72ff922f592` | 正文标明具体文件与函数；未运行 GPU 教程。 |
| [NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples/tree/b7c5481c556c3fe98db060207ecaa41a4b9a9abc) | `b7c5481c556c3fe98db060207ecaa41a4b9a9abc` | `matrixMul.cu:MatrixMulCUDA` 的共享内存分块和同步。 |
| [vLLM](https://github.com/vllm-project/vllm/tree/568afb3a13806beb53bb2e6bd518269357b237c0) | `568afb3a13806beb53bb2e6bd518269357b237c0` | Triton RMSNorm 的固定分段与两遍读取。 |
| [SGLang](https://github.com/sgl-project/sglang/tree/2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc) | `2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc` | JIT 残差 norm 的两级归约与线程参与。未运行 GPU 实现。 |
