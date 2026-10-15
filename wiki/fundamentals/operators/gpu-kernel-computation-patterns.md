---
title: GPU 算子的计算模式：逐元素、归约与分块矩阵乘
type: concept
tags:
  - kernels
  - gpu
  - matmul
  - reduction
sources:
  - raw/articles/2026-09-21/cutlass-efficient-gemm
  - raw/articles/2026-09-24/cuda-softmax-worklog
  - raw/articles/2026-09-21/nvidia-warp-primitives
  - raw/papers/2026-09-21/flashattention
  - raw/repositories/2026-09-21/vllm
  - raw/repositories/2026-09-21/sglang
  - raw/repositories/2026-09-21/triton
  - raw/repositories/2026-09-21/cuda-samples
  - raw/articles/2026-09-21/cuda-tile-kernels
updated: 2026-10-15
---

# GPU 算子的计算模式：逐元素、归约与分块矩阵乘

算子公式规定结果，kernel 还要决定谁负责计算、数据在哪里复用、何时交换局部结果。逐元素、归约和矩阵乘的差别，主要在这些数据依赖上；融合与分块是在依赖允许的范围内减少搬运和调度成本。

本页以 Triton 官方教程 01/02/03/05 及 NVIDIA `MatrixMulCUDA` 为来源，先读 [张量布局与接口](tensor-layout-and-kernel-contracts.md)，硬件概念见 [GPU 执行与存储](../hardware/gpu-execution-and-memory-hierarchy.md)。公式中的流量和分块算例为整理者推导，不是性能实测。

## 1. 逐元素：每个输出有独立的拥有者

对 $z_i=x_i+y_i$，每个 i 的结果与其他位置无关。向量加法教程把连续下标分给不同 program，在尾块同时保护读写。如果每个有效 i 恰好写一次，且输出不覆盖其他工作单元尚需的输入，就不需要跨 program 归约。

若输入输出都是 s 字节元素，理想主数据流量为 $3ns$：读取两路、写一路；运算量 n 次加法，所以 $I\approx1/(3s)$ FLOP/Byte。FP32 时约为 $1/12$，意味着在大尺寸、带宽主导的理想模型中，减少数据搬运比减少这一次加法更有意义。小向量仍可能受启动开销限制，不能仅由 I 判定真实瓶颈。

增加 BLOCK_SIZE 会让每个 program 多处理元素，却也可能增加寄存器需求或减少可调度 program 数。它没有“越大越快”的单调关系。

### CUDA Tile的标量分支与逐元素选择

Writing Tile Kernels快照§2.4.7/§2.4.9.4中，scalar `if`控制整个block的逻辑分支；`ct.where`或C++ `select`按布尔tile逐元素选值。这是控制流与数据选择的区别，不把block逻辑路径理解成硬件只用一个线程。具体控制流仍有语言限制，例如tile代码不能在循环内部return；完整规则未核。

## 2. 归约：一个输出依赖多个输入

求和 $s=\sum_{j=0}^{N-1}x_j$ 可以让不同线程先算局部和，再通过树形合并得到结果。并行化改变加法结合顺序，浮点运算通常不能保证逐位相同；累加 dtype 与归约次序都是算子的数值约定。

若一行不能由一个工作单元有效容纳，可以先输出多个部分和，再用后续 kernel 合并，或采用满足条件的原子合并。这会增加中间流量、启动或竞争。普通 block 屏障不能同步整个 grid，不能通过在一个 block 中等待其他普通 block 的共享内存来替代全局归约。

### Warp 的参与集合不代替尾部数据规则

Warp collective 的成员 mask 与内存访问 mask 作用不同；成员位不会压紧 lane 编号，也不会给缺失源补零。同步条件见 [Warp collective 的成员与内存依赖](../hardware/gpu-execution-and-memory-hierarchy.md#Warp%20collective%20%E7%9A%84%E6%88%90%E5%91%98%E4%B8%8E%E5%86%85%E5%AD%98%E4%BE%9D%E8%B5%96)。

Using CUDA Warp-Level Primitives 的 Listing3 先用 ballot 得到成员，却仍执行 offset 为 16、8、4、2、1 的 shuffle-down 树。若仅 lane0…19 参加，第一轮 lane4 就请求 lane20，而后者未参加该循环。保存片段没有为这个缺失源规定求和贡献，不能直接作为任意长度归约模板。这是整理者按下标与参与集合得到的静态条件反例，未运行 GPU，也未读取完整 API 来裁决返回值的全部版本语义。

若复用完整 32-lane 树得到 lane0 的总和，可在 **32 个硬件 lane 均存在且到达每次 collective** 的条件下，让无有效输入的 lane 携带零参加；真实部分 warp 或稀疏集合则须另核合适算法。保护内存访问、定义 padding 的数学贡献、规定 collective 成员，是三个同时需要满足的问题。

### 机会式聚合 atomic 按同地址群组预留

同文 “Opportunistic Warp-level Programming” 的 `atomicAggInc` 允许用当前恰好一起执行的 lane 组成机会集合，再由 `__match_any_sync(__activemask(),ptr)` 按相同目标地址分组。每组 leader 执行一次 `atomicAdd(ptr,group_size)`，取得旧计数 B 并预留一段；广播 B 后，每个 lane 加上组内低于自身 lane 的成员数，得到自己的返回位置。

整理者例子：同址成员是 lane1、4、7，旧计数为100，计数不溢出且目标缓冲容量足够时，leader 预留 `[100,103)`，三者分别得到100、101、102。不同地址各有自己的 atomic；同址线程若分属不同机会批次，也会分别预留。减少的是每个同地址群组内的原子次数，不能由此保证整个 warp 永远只做一次 atomic，或固定跨群组/原逐线程调用的返回次序。

保存函数含 Unicode 减号并省略 `lane_id()` 等 helper，Listing13 的 legacy shuffle 调用也不完整；本页只解释机制，不将片段当作可直接编译实现。文末 `atomicInc` 的称呼与实际 `atomicAdd` 不同，不能由这个名称引入限值回绕语义。

### softmax：稳定化和 mask 必须同时成立

Triton `02-fused-softmax.py:softmax_kernel` 对一行先减最大值：

$$m=\max_jx_j,\qquad u_j=\exp(x_j-m),\qquad y_j=\frac{u_j}{\sum_ku_k}.$$

对有限输入，减去 m 保持实数域的 softmax，并使指数输入不大于零，避免大正数指数溢出。长度 N 扩展到 $B=2^{\lceil\log_2N\rceil}$ 时，失效位置加载为 $-\infty$：它不会改变有效元素最大值，其指数为零，也不会改变分母。最终只写前 N 项。

反例：一行只有 `[-2,-3,-4]`，若补到 4 项时把失效位置填 0 后也纳入 softmax，分母就多了一个 $e^0$，有效输出的和不再为 1。`mask` 不是单纯的越界开关，填充值也参与数学语义。若整行被屏蔽，或有效数据全为 $-\infty$，会出现 $-\infty-(-\infty)$；教程未定义这种注意力 mask 语义，接入时应单独规定，不能声称有限输入推导已覆盖。

教程使用近似 `tl.exp`，所以公式等价也不保证与其他指数实现逐位相同。程序通过 `row_start=program_id`、`row_step=num_programs` 循环分担多行，不必为每一行创建一个独立 program；每次循环仍把整行逻辑块放在片上处理，超长行的资源需求限制其适用范围。

### 在线分母与并行局部统计使用共同指数基准

Softmax worklog 的在线方案扫描有限非空行时维护最大值 $m$ 与按它定义的指数和 $\ell$。以首元素初始化 $m=x_0、\ell=1$；读入后续元素 x 时，令 $m'=\max(m,x)$，更新

$$\ell'=e^{m-m'}\ell+e^{x-m'}.$$

最大值变大时，旧贡献先转到新基准；因此最大值与分母统计可合并成一遍扫描。得到全行统计后，保存的 Kernel2 仍第二遍读取输入输出概率。它写回 `input[i]`，而 naive 版写 `output[i]`，原地与异地接口不同；重复使用相同输入计时前需恢复被改写的数据。完整状态合并与空状态条件见 [分块怎样保持全行 softmax](../../theory/flash-attention-tiling-and-recomputation.md#2.%20%E5%88%86%E5%9D%97%E6%80%8E%E6%A0%B7%E4%BF%9D%E6%8C%81%E5%85%A8%E8%A1%8Csoftmax)。（文章“Kernel2—Online softmax”；实数等价不保证浮点逐位相同。）

并行版让线程 t 读取列 `t,t+blockDim.x,…`，每个非空子集 $S_t$ 得到 $m_t$ 和 $\ell_t=\sum_{j\in S_t}e^{x_j-m_t}$。先归约出**本行**最大值 $m=\max_tm_t$，再合并

$$\ell=\sum_t e^{m_t-m}\ell_t.$$

这是整理者补出的合并步骤：文章 shared/shuffle 代码只展开到 `global_max`，文字虽说对 local norm 做加法归约，却未列出转基准与完整输出，不能把这些片段当完整 softmax。整理者例子：两个线程分别拥有0与 $\log3$，各自 $\ell_t=1$；共同基准下分母为 $1/3+1=4/3$，概率为1/4、3/4。若直接相加局部分母得到2，再按共同 m 输出，结果为1/6、1/2，总和只有2/3。无有效输入线程还需定义中性状态并满足 collective 成员要求，不能直接在空集合上计算局部最大值。

文章同时把一线程一行改为一 block 一行，使同轮相邻线程读取相邻列；变化涉及工作分配、访存和归约，不能只归因于 shared。其 shared 二分树需相应的2幂 block，保存 `smem[1024]` 也限制线程索引；shuffle 版仍用 shared 和 block 屏障合并各 warp 的结果。`warp_size=4` 是缩小的教学图例，真实32-lane/FULL_MASK的参与条件见 [Warp成员与内存依赖](../hardware/gpu-execution-and-memory-hierarchy.md#Warp%20collective%20%E7%9A%84%E6%88%90%E5%91%98%E4%B8%8E%E5%86%85%E5%AD%98%E4%BE%9D%E8%B5%96)。原文永久锁步与 shuffle 越界“默认值/undefined”的不同说法不作为 API 保证；数学式的求和上界写 N，此处按代码 `col<N` 解释 N 个有效元素。（“Kernel1–4”。）

固定 [CUDA Softmax实现](../../implementation/cuda-softmax-row-reductions.md)沿runner、局部统计重标定、两级归约与再次读取输入走到输出，接续上述计算模式；与文章片段和报告数字分别核对。

### LayerNorm：mask 之后的运算也可能污染归约

`05-layer-norm.py:_layer_norm_fwd_fused` 计算

$$\mu=\frac1N\sum_jx_j,\quad v=\frac1N\sum_j(x_j-\mu)^2,\quad y_j=\frac{x_j-\mu}{\sqrt{v+\epsilon}}w_j+b_j.$$

源码将输入转为 FP32 累加，以真实 N 为分母。求均值时无效 lane 填零；求方差时还必须把无效 lane 的 $x-\mu$ 重新置零，否则先填零再做减均值，会给每个 padding 项加入 $\mu^2$。源码中的 `tl.where(cols < N, x - mean, 0.)` 正是这个步骤。

教学例：有效数据 `[1,3,5]` 补到 4 项，均值是 3，方差应为 $8/3$。若无效项错误贡献 $(-3)^2$，方差变成 $17/3$。公式中的除数是 N，不是补齐长度 B，也不是统计估计时的 N−1。

### 推理库中的 RMSNorm：归约范围与数据保留

RMSNorm 使用行内二阶矩 $s=\sum_jx_j^2/D$，再计算 $y_j=x_jw_j/\sqrt{s+\epsilon}$，不减均值。分块只改变统计量的计算方式，不能把整行变成多个各自归一化的小行。若 D=2500、每段 B=1024，三段有效长度为 1024、1024、452，先合并三段平方和，再除以 2500；分别求三个 RMS 后直接拼接通常不等价。

一个实现可以在得到统计量后重新读输入，也可以跨归约保留输入的片上状态。前者增加逻辑读取，后者增加活跃状态与潜在寄存器压力；“一次 kernel”不能推出“只读一次”。残差融合还可能要求返回相加后的中间状态，输出不能只按归一化结果理解。

这些问题在[vLLM 与 SGLang 的 RMSNorm](../../implementation/rmsnorm-cuda-triton-kernels.md)中有具体实现：固定分段的 Triton、整行 Triton、CUB 行归约和显式 warp→block 两级归约。无有效元素的线程也可能必须参加 collective；mask 数据访问与退出线程不是同一操作。

### CUDA Tile归约的输出形状与跨block合并

同快照§2.4.9.2对`(2,4)`沿axis1求和：C++保留轴，得到`(2,1)`；Python默认得到`(2,)`，需`keepdims=True`才保留长度1。按末轴对齐的广播下，逐行统计应保留或补出该轴，不能只核总元素数；局部tile归约也不会自动合并其他blocks。

原§2.4.7.1的单block `tile_sum`循环累加相同tile内位置。令tile宽为B，则输出仍长B，$acc[r]=\sum_k arr[kB+r]$，并非整个array的标量总和；再归约acc才得到标量。这是从原示例下标展开的整理者说明，原代码按完整tiles读取。

跨block的§2.4.10示例先局部求和，再atomic加到共享输出。atomic只保证每个元素的更新，整tile调用不是一次事务，也不构成grid屏障；输出初值和完成后读取的依赖由调用方维护。Python普通atomic默认ACQ_REL/device scope，C++示例显式relaxed/device，省略C++scope时则为system；不能混用不同接口的默认值。保存的Python `block_sum`已坍缩为一行且注释吞掉后续代码，本页只依据正文与可辨C++操作解释机制，未把它当可执行示例。

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

### Epilogue将计算片段转换为输出布局

CUTLASS Efficient GEMM 的“Epilogue”说明，mainloop 留在各线程寄存器里的输出元素，是按计算复用与指令接口组织的，未必适合合并访问 global。所述 epilogue 先经 shared 在线程间交换元素，再按 striped 映射协作读写 global，同时可施加线性缩放、clamp 等逐元素处理。因此它承担计算布局到输出布局的转换，不能只理解为附加一次 activation。具体寄存器映射可对照 [Tensor Core 片段](../../implementation/tensor-core-quantized-gemm.md)。

具体交换与写回接口随架构和后端而异。若将 K 拆给多个工作单元，还需区分哪些后处理可分配到分片上执行、哪些必须在部分和合并后执行，见 [拆K后的合并与后处理](../../implementation/kernel-configuration-and-autotuning.md#%E6%8B%86K%E5%90%8E%E7%9A%84%E5%90%88%E5%B9%B6%E4%B8%8E%E5%90%8E%E5%A4%84%E7%90%86)。

### CUDA Tile的K循环复用同一累加状态

快照§2.4.9.1把A的`(tm,tk)`与B的`(tk,tn)`送入`ct.mma`，累加到FP32 `(tm,tn)`状态，循环覆盖$\lceil K/t_k\rceil$块。A/B加载均使用边界处理，K尾显式ZERO或masked load，C的M/N尾在写回侧保护，具体规则见[Tile接口](tensor-layout-and-kernel-contracts.md#tile-space%E5%9D%90%E6%A0%87%E4%B8%8E%E9%83%A8%E5%88%86%E5%B0%BE%E5%9D%97%E8%AE%BF%E9%97%AE)。Python示例转成C dtype再写回，C++示例使用half输入、FP32输出。FP32 accumulator描述部分和精度，不能单独证明全部乘法为FP32或所有dtype/rank组合均合法；本项未核完整API或测量后端性能。

## 4. 融合与重排分别减少什么

Triton softmax 教程的逐步 PyTorch 实现总计读写 $8MN+4M$ 个元素，理想融合路径只需读写 $2MN$ 个元素。比值 $4+2/N$ 是该数据流模型的流量比，不是实际加速保证，更不是对框架原生 fused softmax 的通用比较。流量换成字节时还要乘元素大小。

融合让中间值留在片上，减少全局读写和 kernel 启动；但较大的中间状态可能增加寄存器使用、spill 或缩小可驻留并行度，因此融合收益需要测量。

[FlashAttention](../../theory/flash-attention-tiling-and-recomputation.md)将两次矩阵乘与行归一化共同分块：用逐行统计合并key块，避免物化完整概率矩阵；训练反向再重算局部score/概率，减少保存和读取二次规模中间值。它展示了增加算术仍可能减少访存的情形，稠密算术量仍为$O(N^2d)$；IO定理的容量范围与实测速度必须分别核对。（原论文arXiv:2205.14135v2 §3.1–3.2、附录B/C。）

矩阵乘教程的 grouped program ID 映射则促使相邻输出 tile 复用 L2 中的 A/B 块。它重新安排工作与数据局部性，不改变矩阵的数学结果，也不承诺 GPU 严格按照 program ID 次序执行；不能把这种排序当作同步依赖。

进一步选择 tile、warp 数和流水阶段时，需要同时计算片上资源、尾部浪费与工作块数量，见 [Kernel 配置选择与自动调优](../../implementation/kernel-configuration-and-autotuning.md)。把加载与当前 tile 计算重叠时，还要显式维护完成等待和缓冲生命周期，见 [异步拷贝与多级流水](../../implementation/gpu-async-copy-pipelines.md)。

## 5. 怎样回到量化算子

[AWQ/Marlin 反量化内核](../../implementation/weight-only-dequant-kernels.md) 把“读整数容器 → 解包 → 应用 scale/zero → 矩阵乘”组合起来。这里的核心问题仍是：位布局如何映射给线程、分组参数在哪里复用、是否先写出完整浮点矩阵，以及累加和归约用什么精度。

先用 [roofline](../hardware/arithmetic-intensity-and-roofline.md) 估计值得优化的数据流，再用 [正确性与性能测量](../../implementation/kernel-correctness-and-benchmarking.md) 判断实际结果。既有地址、mask 与分块流量示例保留此前的 CPU 教学检查范围；Warp 缺失源/分组预留与 Softmax 共同基准/反例均为整理者静态推导。尚未据此编译 CUDA/Triton 或验证 GPU 性能。

## 来源身份

| 来源 | 版本或快照 | 核对范围 |
| --- | --- | --- |
| Efficient GEMM in CUDA（`https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html`） | `snapshot-2026-07-02`，正文取得于 `2026-07-02T02:41:08+08:00`，3图补采于2026-09-21 | 保存175行/3图；epilogue、两级缓冲、拆K和Hopper设计；后处理反例为整理者推导，未取外链源码或运行。 |
| Learning CUDA by optimizing softmax: A worklog（`https://maharshi.bearblog.dev/optimizing-softmax-cuda/`） | `indexed-text-snapshot-2026-09-24`，取得于 `2026-09-24T19:49:18+08:00` | 保存505行索引文本与所列对应3图；online/并行归约及原地/片段边界，合并公式与两元素反例为整理者推导；原HTML采集时404，未运行。 |
| Using CUDA Warp-Level Primitives（`https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/`） | `snapshot-2026-07-02`，正文取得于 `2026-07-02T02:41:08+08:00`，3图补采于2026-09-21 | 完整保存273行/3图；Listing3参与/缺失源与机会式同址atomic，例子为静态推导；helper/调用省略、API/运行未核。 |
| CUDA Programming Guide：Writing Tile Kernels（`https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-tile-kernels.html`） | `snapshot-2026-07-02` | §2.4.7–2.4.10保存示例/正文；block控制、归约形状/K循环与逐元素atomic，tile_sum下标为整理者展开，Python坍缩示例未执行。 |
| FlashAttention（`https://arxiv.org/abs/2205.14135v2`） | arXiv:2205.14135v2 | §4的分块、融合与反向重计算连接；论文§3.1–3.2、附录B/C，未核作者CUDA或性能。 |
| Triton 官方教程与工具（`https://github.com/triton-lang/triton/tree/81a46fa0c04526e5df55a018ecfab72ff922f592`） | `81a46fa0c04526e5df55a018ecfab72ff922f592` | 正文标明具体文件与函数；未运行 GPU 教程。 |
| NVIDIA CUDA Samples（`https://github.com/NVIDIA/cuda-samples/tree/b7c5481c556c3fe98db060207ecaa41a4b9a9abc`） | `b7c5481c556c3fe98db060207ecaa41a4b9a9abc` | `matrixMul.cu:MatrixMulCUDA` 的共享内存分块和同步。 |
| vLLM（`https://github.com/vllm-project/vllm/tree/568afb3a13806beb53bb2e6bd518269357b237c0`） | `568afb3a13806beb53bb2e6bd518269357b237c0` | Triton RMSNorm 的固定分段与两遍读取。 |
| SGLang（`https://github.com/sgl-project/sglang/tree/2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc`） | `2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc` | JIT 残差 norm 的两级归约与线程参与。未运行 GPU 实现。 |
