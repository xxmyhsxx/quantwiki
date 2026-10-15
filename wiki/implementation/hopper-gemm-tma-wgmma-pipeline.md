---
title: Hopper BF16 GEMM：TMA、WGMMA 与缓冲复用
type: implementation
tags:
  - kernels
  - matmul
  - tensor-core
  - synchronization
sources:
  - raw/articles/2026-09-24/h100-gemm-worklog
  - raw/articles/2026-09-21/cuda-async-copies
  - raw/repositories/2026-09-24/fast-cu
updated: 2026-10-15
---

# Hopper BF16 GEMM：TMA、WGMMA 与缓冲复用

Hopper GEMM 可以让 producer 用 TMA 将 A/B 块搬进 shared 环形队列，consumer warp-group 用 WGMMA 累加 FP32 结果，最后转为 BF16 写回。贯通这条流水需要跟踪三种完成条件：输入何时可读、计算何时不再使用输入、输出暂存何时可覆盖。

本页以 H100 worklog 的 dense BF16 案例解释对象与依赖，并用已收录的 CUDA TMA 文档补充输出可见性条件。前置概念见[分块矩阵乘](../fundamentals/operators/gpu-kernel-computation-patterns.md)与[流水的生产和消费协议](gpu-async-copy-pipelines.md)。

## 1. shared 输入供应寄存器中的累加结果

案例计算 $C=AB$：A 的逻辑形状为 $[M,K]$、行主序，B 为 $[K,N]$、列主序，C 为 $[M,N]$、列主序；三者保存 BF16 元素，计算中以 FP32 累加。一个 CTA 负责 $B_M\times B_N$ 输出，沿 K 分批读取 $B_M\times B_K$ 的 A 和 $B_K\times B_N$ 的 B。输出累加状态贯穿全部 K 块，输入缓冲则反复复用。（worklog “Setup”“Kernel 2–3”。）

保存的 `m64n16k16.f32.bf16.bf16` 示例由一个 **128 线程 warp-group** 协作：A/B 放在 shared，通过 shared-memory descriptor 交给 WGMMA；FP32 结果分布在线程寄存器中。较大的 CTA 输出由多次 WGMMA 覆盖，其指令形状、线程协作范围和 CTA tile 是不同层次。（“Tensor Core”与 Appendix 代码图。）

这里输入是普通 BF16。已有 [DeepGEMM 专家执行](deepgemm-moe-expert-execution.md)所选路径使用 FP8/K32 WGMMA，并连接分组尺度恢复；共享 TMA/WGMMA 概念不意味着两者读取相同数据格式。

## 2. full 保护读取，empty 保护覆盖

worklog 为 `QSIZE` 个物理 slot 各分配一份 A/B shared 缓冲和 `full/empty` barrier。producer 与 consumer 分别维护当前 slot，按 `idx=(idx+1)%QSIZE` 前进。同一 slot 的一次循环为：

| 阶段 | producer 的操作 | consumer 的操作与依赖 |
| --- | --- | --- |
| 取得可复用 slot | 等待 empty | 此前使用该 slot 的 consumer 已完成读取 |
| 填入下一 K 块 | 发起 A/B 的 TMA，并登记 full 的预期传输量 | 等待 full 同时满足参与者到达与传输完成 |
| 计算当前 K 块 | 可以准备其他空闲 slot | 发出 WGMMA，等待所需计算完成 |
| 释放旧输入 | 下一轮才可覆盖该 slot | 不再读取该 slot 后通知 empty |

启动时，consumer 先对各 slot 的 empty 作到达通知，使 producer 能填第一轮。此时 empty 表示写入许可，缓冲中尚无有效输入。（“Kernel 4”队列、producer、consumer 与 QSIZE=2 时序图。）

producer 图使用

```cpp
barrier_arrive_tx(full[idx], 1,
                 (BK * BM + BK * BN) * sizeof(bf16));
```

它登记一次本方 arrival，以及两份 BF16 输入的预期字节量。**调用已经返回与数据已经到达是两件事**；consumer 的等待还必须覆盖 TMA 完成。CUDA 文档的多维 TMA 示例同样由发起线程调用 `barrier_arrive_tx`，其他参与线程 arrive，随后各线程 wait，才读取 shared。（CUDA §4.11.2 的多维示例。）

WGMMA 又有自己的完成协议。worklog 附录展示 `wgmma.fence` → 多次 WGMMA → `wgmma.commit_group` → `wgmma.wait_group 0`；最后的等待覆盖该 warp-group 已提交的组。输入 TMA 完成只解决可读性，WGMMA fence 也不能替代 TMA 等待。把附录顺序接到前述队列上，释放 empty 的必要条件是：所有使用该 slot 的 consumer 都已完成相关 WGMMA 对输入的读取。consumer 图省略了内部计算步骤，这里是在说明完整依赖。（“Kernel 4”consumer 图与 Appendix batching 图。）

### barrier 世代与代表 arrival

最初的 CUDA 接口示例按一个 producer 线程加一个 128 线程 consumer，设置 129 个 arrival；两个 consumer 时改为 257。后续 PTX wrapper 示意让每个 consumer 组由一个代表通知，计数改为 `1+2=3`。减少通知的前提是代表已经获得本组完成的依据；等待线程数不必等于 arrival 数。（“Kernel 7”。）

示意还在 wait 后执行 `phase^=1`，区分同一 barrier 的复用世代。slot 下标选择物理缓冲，phase 区分该 barrier 的轮次，预期 transaction 字节量记录本轮输入，三者承担不同职责。代表线程的组内协调必须与所用 wrapper 一起设计；把计数改为 3 本身不能建立成员间的完成关系。

## 3. 大输出 tile 同时增加累加状态与输入空间

worklog 将输出从 128×128 扩为 128×256 时，报告单 consumer 路径因寄存器不足触发 WGMMA 串行化警告。完整输出有 32768 个 FP32 累加值，集中到一个 128 线程 warp-group，平均就是每线程 256 个值，尚未计入地址、循环和其他活跃状态。

作者改用两个 consumer，分别处理 64×256 的输出行，各自完成整个 K 归约，使累加载荷平均降到每线程 128 个值。这是沿 M 分担独立输出；两个组之间无需合并 K 的部分和。通用资源预算见[配置选择](kernel-configuration-and-autotuning.md)。（“Kernel 5”警告与分工图。）

更大的 A/B 块同时增加 shared 用量，案例将 QSIZE 从 5 减为 3。之后用 `setmaxnreg` 将 producer 的每线程预算降到 24、consumer 提高到 240，把寄存器容量移向计算角色。这些数值属于该配置；它说明每线程累加压力、CTA 输入缓冲和角色分配必须一起取舍。作者把此处提速猜为较少 register bank conflicts，保存材料没有独立证据确认这一因果解释。

## 4. persistent 复用流水，调度决定跨 tile 局部性

persistent CTA 可以连续计算多个输出 tile。producer 供完当前输出的 K 块后，沿已有队列进度预取下一输出；consumer 写回当前结果时，下一份输入便可能进入 shared。跨输出 tile 继续保留 slot 与 barrier 世代，可避免在另一方仍使用状态时重新初始化。（“Kernel 6”。）

这种循环仍需选择任务顺序。文章最初让每个工作者处理一段连续编号，结果同批任务在矩阵上相距较远；改成空间相近的输出成组后，可以提高 A/B 在 L2 中的复用机会。作者报告初始 persistent 版本退至 400 TFLOP/s，调整空间分组后达到 660 TFLOP/s，说明流水改进可能被较差局部性抵消。后续 Hilbert 顺序进一步调整跨组相邻关系；它提供任务排列，不建立跨 CTA 屏障。launch 的 132 或 128 个 CTA 也不构成固定 SM 绑定。（“Scheduling and L2 Cache”“Kernel 11”；persistent 的一般解释见[工作循环与调度](kernel-configuration-and-autotuning.md#4.%20persistent%20%E6%8F%8F%E8%BF%B0%E5%B7%A5%E4%BD%9C%E5%BE%AA%E7%8E%AF%EF%BC%8C%E4%B8%8D%E6%89%BF%E8%AF%BA%E7%BB%91%E5%AE%9A%20SM)。）

cluster multicast 是显式协作的另一条路径。文章示例把两个沿 M 相邻的输出 CTA 放入一个 cluster：两者需要不同 A 块，但共用相同 B 块，因此可 multicast B；各 CTA 仍独立累加并输出自己的结果。cluster 的协作范围与 L2 缓存机会分别由相应机制提供。（“Kernel 8”。）

## 5. TMA 输出必须保护 shared 暂存的生命周期

“Kernel 10”先将寄存器结果转换到 shared 的 `sC`，再发起 TMA shared→global 输出。单份 sC 可以复用，但多占一份输出暂存也会挤压输入队列的 shared 预算。复用的关键是让旧 TMA 读完 sC，再写入下一输出；新 TMA 读取之前，所有写 sC 的线程还要建立对 TMA 的可见性。

CUDA 多维 TMA 示例提供了这两个方向的具体顺序：

1. 写 shared 的线程各自执行 `fence_proxy_async(space_shared)`，随后 `__syncthreads()`，使参与线程的写入对 TMA 可见，再由选定线程发起输出。
2. 发起线程调用 `cp_async_bulk_commit_group`，以 `cp_async_bulk_wait_group_read<0>` 等待该组完成对 shared 源的读取。此后复用 sC 时，还须将可复用状态传达给下一轮写入的其他线程。

这里 wait 的主体是发起该 bulk 操作的线程；其他线程执行自己的等待，不能替它确认这份操作。`wait_group_read` 的完成条件是 shared 源已读完，供后续 kernel 或主机消费 global 输出还要遵循相应的 kernel/stream 完成依赖。（CUDA §4.11.2 与多维 TMA 示例。）

worklog 的输出图展示 bulk wait → 写 sC → consumer 屏障 → 单线程 store/commit，但没有展开 proxy 可见性、发起者等待与末轮排空的完整组织。上述 CUDA 顺序用于说明所需依赖，不能据此推定博客省略的 wrapper 已实现全部条件。输入 TMA 的 shared barrier、WGMMA 组和输出 TMA 的 bulk 组因此应分别跟踪。

## 6. 作者性能结果的适用范围

主要实验为 H100 SXM、CUDA 12.6/V12.6.68、BF16、A 行主序/B 和 C 列主序、M=N=K=4096，输入服从 normal(0,1)。作者称对 8 次运行取平均并忽略首个 warmup，以 $2MNK/t$ 计算吞吐；保存文字没有明确总调用数是 8 次，还是另加一次 warmup。（“Setup”。）

最终报告为 764 TFLOP/s，相对文中约 716 TFLOP/s 的 cuBLAS 基线快约 7%。这是所选单 GEMM 的作者结果，完整 cuBLAS 选项、计时实现、时钟/功耗、cache 协议、原始样本和数值容限没有在材料中展开。文章的线程内 `clock()` Load/TC/Store 周期统计用于定位阶段，不能直接相加为整 kernel 的墙钟延迟；通用测量口径见[正确性与性能测量](kernel-correctness-and-benchmarking.md)。

## 7. fast.cu 固定源码如何接续这条流水

本节核对晚于文章的 fast.cu 固定 commit。`h100/matmul.cu` 依次枚举 kernel 0–12，当前驱动固定 M=N=K=8192；它没有按形状自动择优。Makefile 的 H100 目标编译到 `sm_90a`。以下主线选 `matmul_11.cuh` 与 12 相对 11 的完整差异。（`run_kernel`、`runKernel11/12`。）

11/12 都固定每 CTA 输出128×256、BK=64、3个输入 slot、384线程，启动128个 CTA，组成64个2-CTA cluster。前128线程承担 producer 角色，其中一个线程发 TMA；后两组各计算64×256输出，每线程持有128个FP32累加值。每个BK块发四条 `m64n256k16`，新输出的首条以 `ScaleD=0` 开始，后续指令累加。A/B descriptor 对应128B swizzle；源码的3D tensor map将连续K按64元素分段，属于这份实现的布局选择。

输入协议比文章示意更具体：每份 full 初始化 arrival=1，由本 CTA 的 producer 登记 A+B 预期字节；empty 初始化为 `2个consumer×2个CTA=4`。各 CTA 加载自己的 A，rank_m=0 multicast B。每个 consumer 在 WGMMA commit/wait 后，分别向 cluster 两个 CTA 的 empty 通知一次，双方的 producer 才能复用对应 slot。slot 绕过3份缓冲时翻转 parity，状态跨输出 tile 保留。（`init_barrier/expect_bytes/arrive_cluster` 与 `matmulKernel11`。）

任务来自 host 预建的 Hilbert 表：设备 `Schedule<2>::next` 依次解码输出坐标，每个 cluster 有128个表槽；producer 与 consumer 使用相同次序。这是静态任务排列。wrapper 仅在 M 变化时重建 tensor maps，N/K 只由 assert 比较，数据指针不在缓存键中；Makefile 的 `NDEBUG` 又关闭这些 assert。当前固定指针/方阵驱动是其调用条件，换指针或形状时需要重新建立相应参数。（`createHilbert/runKernel11/12`。）

### 12 改变结果怎样落到 shared

11 将 FP32 片段转换为 BF16 写 sC，两个 consumer 的256线程汇合，由 thread128 发起一次 CTA 输出。12 给每组的每列64行片段补8个 BF16 元素，用 `stmatrix.x4.trans` 协作写 shared，再由各组128线程独立汇合、各自发 TMA 输出。每 CTA 的 A/B 三阶段占147456B，输出区从11的65536B增至12的73728B，还须加 barrier、调度表及对齐。这是输出布局与协作范围的变化，主算和输入协议沿用11。

这里有实际源码缺口需要与第5节的必要顺序对照：`store_async` 中未见 proxy fence；所有 writer 执行各自的 bulk wait 后直接写 sC，未见发起者等待完成再向其他 writer 建立的写前协调；输出循环尾部也没有最终 bulk wait。写完 sC 后的 consumer 屏障不能补上此前的覆盖依赖。因此本节说明其数据流和配置，不把该固定输出路径作为同步完整的复用模板。（11的492–530行、12的503–523行及 `store_async`。）

### 驱动计时与正确性结果分别解释

H100 基线实际调用 `cublasGemmEx`，FP32 compute、BF16输出、alpha=1/beta=0；目录中的 cuBLASLt 开发片段没有接入该目标。kernel>1 的输出与基线按绝对阈值0.1比较，差异表达式先做BF16减法再转float；失败只打印，仍继续计时并可返回0。kernel1使用另一套行主序地址且被跳过比较。

每个 kernel 先做一次校验调用，再对同一组地址重复8次，以 CUDA event 整段平均；顺序固定为0→12，每项前 sleep 5秒，输入生成、H2D与首次 Hilbert/tensor-map准备在计时外。该口径与文章4096主例分别保留，程序退出0或打印吞吐均不代表同步缺口已经验证消除。新快照的 [GB300 NVFP4 路径](blackwell-nvfp4-tmem-execution.md)使用另一套输入格式、计算状态与计时驱动。

## 来源身份

| 来源 | 固定版本或快照 | 核读与使用范围 |
| --- | --- | --- |
| Pranjal Shankhdhar，Outperforming cuBLAS on H100: a Worklog（`https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog`），发表于 2024-11-29 | `html-body-snapshot-2026-09-24`，取得于 `2026-09-24T19:48:31+08:00` | 完整 599 行正文、85 行 README 与实际 38 个附件；正文哈希与登记一致。40 处图引用缺 swizzle/Hilbert 小图，Conclusion 所称性能 plot 未保存；图片于 2026-10-05 补录。使用 BF16 输入、队列、资源和调度案例；省略的 wrapper、descriptor 及完整尾部路径未由图片补造。后续配套源码按下行版本另核。 |
| CUDA Programming Guide：Asynchronous Data Copies（`https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html`） | `snapshot-2026-07-02`，取得于 `2026-07-02T02:31:16+08:00` | 定向核读保存正文 519–565、1082–1153 行：TMA 方向与完成机制、多维示例的 arrival/字节、proxy fence、发起者 bulk-group 等待与 shared 源复用；本次未重读其余章节或三张 swizzle 图。 |
| pranjalssh/fast.cu（`https://github.com/pranjalssh/fast.cu/tree/2dfe5e26aecfd9e5f27bf9d5837deea01acda24b`） | `2dfe5e26aecfd9e5f27bf9d5837deea01acda24b`，取得于 `2026-09-24T19:52:11+08:00` | 完整身份/根README、Makefile、H100驱动与M11，11→12全部差异；M2/3算子入口及3→11相邻版本差异用于核路线，experimental仅选定producer/consumer与wrapper，sum另作覆盖判断。34文件保存完整性复用既有归档证明，不表示全部分支语义核验；GB300范围见独立页。未编译、运行GPU或复现文章/README性能。 |
