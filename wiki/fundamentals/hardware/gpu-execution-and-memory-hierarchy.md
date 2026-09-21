---
title: GPU 的执行模型与存储层次
type: concept
tags:
  - gpu
  - memory
  - kernels
sources:
  - raw/repositories/2026-09-22/kernel-skills/source/skills/cuda/optimize-shared-memory-tiling/SKILL.md
  - raw/articles/2026-09-21/cuda-simt-kernels/article.md
  - raw/repositories/2026-09-21/cuda-samples/source/cpp/6_Performance/transpose/transpose.cu
  - raw/repositories/2026-09-21/cuda-samples/source/cpp/0_Introduction/matrixMul/matrixMul.cu
  - raw/repositories/2026-09-21/cs336-lectures/source/lecture_02.py
  - raw/papers/2026-09-21/marlin/paper.pdf
  - raw/papers/2026-09-21/qserve/paper.pdf
updated: 2026-09-22
---

# GPU 的执行模型与存储层次

量化为什么有时快、有时只是省内存，答案通常不在算法里，而在数据从哪一级存储搬到哪一类计算单元。本页整理这一层的最小必要结构：线程如何被调度、存储分成哪几级、计算单元有哪两类，以及这些结构怎样决定低比特实现的收益与代价。

本页的硬件结构描述依据 MARLIN 论文 §2.1（Ampere 架构、执行单元、异步拷贝，已全文研读），设备数值来自 QServe 论文与 Stanford CS336 的公开讲义（本地快照 `lecture_02.py`，含该讲义的 FLOPs 与 roofline 计算）。本页不重复方法页的实现细节，量化执行路径的分类见 [量化矩阵乘法的缩放与执行路径](../../implementation/quantized-matmul-scaling-execution.md)，性能上界的分析模型见 [算术强度与 roofline 分析](arithmetic-intensity-and-roofline.md)。

## 1. 执行模型：从线程到 SM

论文 §2.1 描述的层次是：GPU 由若干流式多处理器（SM）组成，它们共享全局内存（GMEM）与 L2 缓存；每个 SM 划分为若干分区，分区内包含 warp 调度器、寄存器文件与 L0 指令缓存；同一 SM 内的分区共享 L1 缓存，L1 的一部分可配置为快速的片上暂存器，即共享内存（SMEM）。

调度单位是 warp，由 32 个线程组成、并发执行；线程块是一组 warp，被调度到同一个 SM 上。每个 SM 上能同时驻留的 warp 数取决于硬件上限以及每线程寄存器用量、可用共享内存等资源。这一条在后面反复出现：**寄存器与共享内存的占用决定了有多少 warp 能同时在场，而 GPU 正是靠大量在飞 warp 来隐藏延迟。**

### 从 CUDA 下标到 warp

CUDA 文档快照 §2.3.2 给出线程块下标的线性化：$t=x+D_x(y+D_yz)$，x 最快变化；NVIDIA warp 按连续的线性线程编号组织，每个 warp 32 个线程。因此二维 block `(32,8)` 的一个 warp 对应一行 x，而 `(16,16)` 的一个 warp 横跨两行 y。二维 launch 形状本身不改变算子，但坐标映射会改变哪些地址由同一 warp 访问。

SIMT 允许每线程维护自己的状态和控制流。同一 warp 的分歧路径通常需要分开执行相应活动线程，因此“每线程都能写 if”不代表分支没有效率代价；也不能仅凭 warp 这一分组，就省略共享数据交换所需的同步。普通线程块间没有天然执行先后，block ID 是工作分配标识，不是依赖顺序。

### 存储位置、作用域与同步是不同问题

| 对象 | 通常的可见范围 | 开发时需要理解的限制 |
| --- | --- | --- |
| 寄存器 | 单线程 | 编译器分配；高用量会限制驻留，溢出会增加内存访问 |
| shared memory | 同一普通线程块 | 程序显式管理；跨线程读写须有正确同步 |
| local memory | 单线程逻辑作用域 | 物理上在设备内存，名称 local 不表示片上高速 |
| global memory | 设备上相关线程可寻址 | 可跨 kernel 保存；可寻址不等于并发访问自动有序 |
| L1/L2 cache | 硬件缓存层次 | 缓存命中不替代数据依赖与同步约定 |

依据 CUDA 文档 §2.3.3；这里讨论普通线程块，未展开具有专门启动条件的 cluster/distributed shared memory。限制寄存器用量可能提高驻留数，也可能引入 spill；occupancy 不是单独需要最大化的性能目标。

NVIDIA `matrixMul.cu:MatrixMulCUDA` 在共享 tile 写好后做一次 `__syncthreads()`，读完后再做一次，分别保证消费者可以开始读和生产者可以覆盖下一块。对普通块内屏障，所有需要参与的线程应按一致顺序到达；不能把屏障只放在部分线程满足的 `if (i<n)` 中。尾部线程可为加载填中性值，再一起参与屏障。

`__syncthreads()` 只同步该 block，不能同步整个 grid；原子加提供相应地址的原子更新，也不是整个计算阶段已经完成的屏障。跨块阶段可以拆成有序的 kernel 调用，或使用满足前提的专门协作机制。核对这些依赖时要同时看共享缓冲生命周期，而不只是查找有没有屏障函数。

## 2. 计算单元：CUDA 核心与张量核心

同一份论文把执行单元分为四类：整数单元、特殊功能单元、浮点单元（FPU／CUDA 核心）与张量核心单元（TCU）。张量核心为机器学习负载设计，以矩阵片段的乘加提供高吞吐；矩阵指令的吞吐、延迟和完整 tile 的执行时间是不同概念，不能把一条矩阵指令理解成任意矩阵在一个周期内完成。在 Ampere 架构上，张量核心在 FP16 上的吞吐最高可达 FMA 运算的 16 倍。

Ampere 架构还有两项与量化实现直接相关的能力：

- **结构化稀疏。** 2:4 格式把左矩阵按长度 4 的向量分成组、每组置零两个元素，得到 50% 稀疏但结构化的矩阵，配套的稀疏张量核心（SPTC）承诺相对原张量核心 2 倍、相对 FPU 最多 32 倍的吞吐。数据上需要两个结构：非零值，以及记录每个非零元素在组内位置（2 bit）的元数据。
- **异步拷贝。** 数据可以直接从全局内存搬到共享内存，省去中间的寄存器访问。它有两种变体：`access`（把数据也留在 L1 供后续复用）与 `bypass`（跳过 L1）。这一条决定了后续讨论的「读一遍权重」能否真正只读一遍。

## 3. 量化收益为什么取决于这些结构

**权重读取是受带宽限制场景的瓶颈。** 解码阶段的矩阵乘接近矩阵向量乘，权重读取量远大于激活；低比特减少的正是这部分流量。但收益是否兑现，取决于元数据与解包成本是否把省下的带宽吃掉——块量化格式的尺度、零点、子块尺度都会计入每权重字节数，见 [GGUF 与块量化存储格式](../../implementation/gguf-block-quantization-formats.md)。

**共享内存布局决定加载效率。** 以 Marlin 为例，它要求每个线程用 16 字节（128 bit）的加载宽度，使一个 warp 用一条指令取到 $32\times32=1024$ 个 INT4 权重；对激活侧则必须保证 $16\times16$ 的 fp16 块中若干 16 字节向量落在不同 bank，才能让 `ldmatrix` 无冲突执行，做法是在共享内存中用 XOR 做索引变换。权重可离线重排，激活不行——这一不对称是所有在线变换与重排设计的共同约束。

**寄存器压力决定延迟隐藏能力。** QServe 论文 §3.2 指出，按组量化的 W4A4 需要在主循环内同时保存 FP32 与 INT32 两组部分和寄存器；在输出驻留式数据流下，大矩阵乘本就受寄存器限制，双份寄存器进一步压低同时驻留的 warp 数，削弱隐藏延迟的能力。这解释了为什么「把位宽降一半」可能反而更慢。

**反量化落在哪一类单元上，决定它是零成本还是致命开销。** QServe 论文 §1、§3.2 给出 A100 上的量级：CUDA 核心的峰值算力约为 INT4 张量核心的 2%，因此一次 CUDA 核心运算的代价约等于 50 次 INT4 张量核心运算；在 W4A8 主循环里，地址计算的吞吐也比 INT8 张量核心低 32 倍。当反量化或指针算术不得不落在 CUDA 核心上时，慢的那一类运算就会主导主循环。

### 合并访存和 bank conflict 要分别分析

CUDA 文档 §2.3.4.1 用 32-byte 事务说明全局访存。按其示例，32 个活动线程各读一个 FP32：从 32-byte 对齐地址连续读取，需要覆盖 4 个事务段；若每个线程相隔至少 32 字节，则可能覆盖 32 个段。教学换算中，有效载荷同为 128 字节，后者请求段总量为 1024 字节，有效利用率为 12.5%。跨线程地址的排列可以不同，只要覆盖相同的少数段仍可合并；重点是一个 warp 的一次访问覆盖多少段，而不是单线程自己的循环是否连续。缓存、对齐及实际架构会影响最终 HBM 流量。

shared memory 则需检查 bank 映射。文档 §2.3.4.2.2 的 FP32、32-bank 示例中，行优先数组 `tile[32][32]` 的列访问地址为 $32r+c$ 个 word，bank 编号 $(32r+c)\bmod32=c$，32 个不同 word 落在同一 bank。改为 `tile[32][33]` 后变成 $(33r+c)\bmod32=(r+c)\bmod32$，各行落在不同 bank。这里针对不同地址；同地址广播不能套用这个冲突计数。

NVIDIA `transpose.cu:transposeNoBankConflicts` 采用后一种 padding：合并读取 global → 写 shared tile → 块内同步 → 转置读取 shared → 合并写 global。它同时解决两侧全局访问和局部 bank 冲突；代价是共享内存与同步。`+1` 是这个 FP32 教学布局的解法，不是任意 dtype、访问宽度、Tensor Core 布局都适用的定律。实际 AWQ/Marlin 的交织和 swizzle 见 [反量化内核](../../implementation/weight-only-dequant-kernels.md)。


### 从 bank 编号推到实际访问冲突

Shared-memory tiling Skill 提醒要先计算地址，但只统计重复 bank 编号还不够：需同时看地址属于哪个 32-bit word，以及实际发出的访问指令。依据 CUDA 文档 §2.3.4.2 的 word 广播规则，对齐的连续 half 读取中，第 0/1 个 half 属于同一个 32-bit word，第 2/3 个属于下一个 word；不能由“两个 half 同 bank”直接推出两路冲突。读取同一 word 的广播也不表示并发覆盖同一地址是安全的。

对一个 warp 的 32 个线程，各读取不同 FP32 word、word 地址间隔 q，并采用本节 32-bank 模型时，bank 为 $(b_0+q\ell)\bmod32$，$\ell$ 是 lane 编号。由模运算得到：不同 bank 数为 $32/\gcd(q,32)$，每个 bank 对应 $\gcd(q,32)$ 个不同 word。于是 q=1、3、33 覆盖全部 bank，q=2 对应两路冲突，q=32 对应 32 路。这是带明确条件的教学推导；q=0 的同 word 广播不适用“不同 word”假设。

因此 padding 改变的是行步长与 bank 周期，不是给任何访问加一列就一定有效。向量化会改变每线程访问宽度及请求拆分方式，Tensor Core 的矩阵片段加载又有自己的布局；最终仍需回到实际指令核对。half2 可以用于向量化，不能仅为修复想象中的“连续 half 必有冲突”而使用它。

共享内存的收益也不仅是重复使用同一数据：转置案例利用 shared 将跨步写转换成合并写。反过来，纯逐元素流式计算既无复用也无需重排，额外 shared 暂存往往只增加搬运和同步。选择前先说清减少的是哪一种流量或访问损失。

分块与流水如何改变 shared 预算和驻留能力，继续读 [Kernel 配置选择](../../implementation/kernel-configuration-and-autotuning.md)；异步搬运的完成等待和 buffer 复用条件见 [异步拷贝与多级流水](../../implementation/gpu-async-copy-pipelines.md)。

## 4. 设备参数实例

以下数值是论文与讲义给出的规格，用于建立量级感，不是实测结果：

| 设备 | 峰值算力 | 显存带宽 | 加速器强度（峰值算力 ÷ 带宽） | 来源 |
|---|---|---|---|---|
| A100 | FP16／INT8／INT4 张量核心 312／624／1248 TOPS | 2 TB/s | 约 156／312／624 FLOP/Byte | QServe §3.1 |
| H100 | BF16 约 989.5 TFLOP/s（不含稀疏） | 3.35 TB/s | 约 295 FLOP/Byte | CS336 讲义 `facts.py` 与本页换算 |

三点读法：**算力随数据类型变化**，而带宽不变，因此加速器强度也随精度变化（CS336 讲义明确提示这一点），跨精度的 FLOP/s 不能直接比较；**规格是上限**，实际达到的比例通常远低于 1，见 [算术强度与 roofline](arithmetic-intensity-and-roofline.md) 中的 MFU 定义；**代际差异很大**，异步拷贝与结构化稀疏都是 Ampere 引入的能力，把它们当作所有 GPU 的普遍事实会得到错误结论（Marlin 的结论明确限定在 Ampere 级设备上）。

进一步将这些约定用于代码，可读 [张量布局与接口](../operators/tensor-layout-and-kernel-contracts.md) 和 [归约、分块与融合](../operators/gpu-kernel-computation-patterns.md)。

## 5. 局限与未验证

- 本页描述的是执行模型与规格数值，不是测量结果；本轮没有运行任何 GPU 程序或基准。
- 结构与执行约定取自 MARLIN、QServe 的背景章节和 NVIDIA CUDA 文档、示例，均以 NVIDIA 架构为对象；其它厂商加速器的存储层次与执行模型不同，不能直接套用。
- CS336 讲义为教学材料，其数值（H100 峰值与带宽）来自厂商规格页，本页只用于换算量级；讲义对 MFU、算术强度与 roofline 的完整处理见下一页。

## 来源身份

下表用于在没有本地资料库时辨识来源；具体论述的章节、公式、图表或代码位置见正文。

| 来源 | 版本或快照 | 说明 |
| --- | --- | --- |
| [stanford-cs336/lectures](https://github.com/stanford-cs336/lectures/tree/8b59b50730766695c2ffedd1a79c50cd09b9eb91) | `8b59b50730766695c2ffedd1a79c50cd09b9eb91` | — |
| [MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models](https://arxiv.org/abs/2408.11743v1) | `arXiv:2408.11743v1` | — |
| [QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving](https://arxiv.org/abs/2405.04532v3) | `arXiv:2405.04532v3` | — |
| [CUDA Programming Guide：Writing SIMT Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html) | `snapshot-2026-07-02`，获取于 `2026-07-02T02:31:16+08:00` | §2.3.2–2.3.4 的线程、存储与访存解释 |
| [NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples/tree/b7c5481c556c3fe98db060207ecaa41a4b9a9abc) | `b7c5481c556c3fe98db060207ecaa41a4b9a9abc` | 矩阵乘屏障与转置 padding；未运行 |
| [tensormux/kernel-skills](https://github.com/tensormux/kernel-skills/tree/7b7337a123f8711aa8e3d0452351d8fd30dde4b7) | `7b7337a123f8711aa8e3d0452351d8fd30dde4b7` | shared-memory tiling 的解释需求；bank 判断以官方文档与地址推导为依据 |
