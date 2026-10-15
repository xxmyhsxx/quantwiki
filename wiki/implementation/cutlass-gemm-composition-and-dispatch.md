---
title: CUTLASS GEMM 的组成：类型、运行参数与调度
type: concept
tags:
  - kernels
  - matmul
  - gpu
  - data-layout
sources:
  - raw/articles/2026-09-21/cutlass-gemm-api
updated: 2026-10-15
---

# CUTLASS GEMM 的组成：类型、运行参数与调度

CUTLASS 3 把 GEMM 分成几层协作：主机 handle 准备调用参数，kernel 分配输出任务，collective 完成当前任务的搬运、K 归约与输出处理，CuTe 再将这些操作组织成线程和指令。理解这条链，才能判断一个配置改变的是工作划分、流水还是数值接口。

本页依据 CUTLASS 3.0 GEMM API 的固定文档快照。输出 tile 与 K 归约的基础见 [分块矩阵乘](../fundamentals/operators/gpu-kernel-computation-patterns.md)，元素 stride 见 [张量布局与接口](../fundamentals/operators/tensor-layout-and-kernel-contracts.md)。

## 1. Mainloop和epilogue组成kernel，adapter提供主机入口

文档的组装顺序是：先选择 collective mainloop 和 collective epilogue，将它们作为 `kernel::GemmUniversal` 的类型参数，再用 `device::GemmUniversalAdapter<GemmKernel>` 得到可复用主机 handle。

Mainloop 沿 K 搬入 A/B tile，把乘积累加到同一输出状态；epilogue 处理输出布局与逐元素操作，例如完成 $D=\alpha AB+\beta C$。Kernel 安排输出任务、各协作角色及 collective 的调用顺序。这些是同一设备计算的组成层次，各层并不各自启动一次 kernel。（“CUTLASS GEMM Components”“Collective API”“Kernel API”。）

| 层 | 代表对象 | 在这条计算链中的职责 |
| --- | --- | --- |
| Device | `GemmUniversalAdapter` | 保存主机调用状态，将用户参数转换为设备参数，准备并发起 launch |
| Kernel | `GemmUniversal` | 控制 grid 与输出 tile 调度，安排 collective 的执行和线程角色 |
| Collective | `CollectiveMma`、collective epilogue | 在协作域内组织 copy/MMA、共享数据和同步，完成 K 主循环与输出处理 |
| Tiled MMA / Copy | `TiledMma`、`TiledCopy` | 将小操作按线程与数据铺开，经 `cute::gemm`、`cute::copy` 执行较大的操作 |
| Atom | `Mma_Atom`、`Copy_Atom` | 描述一次硬件操作所需的最小参与线程集合与数据布局 |

例如 warp-specialized collective 可以让 producer 搬运、consumer 做 MMA，并通过共享状态协调依赖。Kernel 则在独立输出任务之间调度；普通 grid 可直接分配任务，persistent kernel 可继续向 work-tile scheduler 领取。每个任务内部的 K 循环仍由 mainloop 处理。具体 Hopper 分工见 [配置与工作划分](kernel-configuration-and-autotuning.md#Hopper%E7%9A%84%E4%B8%A4%E4%B8%AAconsumer%E6%80%8E%E6%A0%B7%E5%88%86%E6%8B%85%E8%BE%93%E5%87%BA)。

多个GEMM怎样组成任务空间、visitor怎样定位problem，见[Grouped GEMM任务调度](kernel-configuration-and-autotuning.md#Grouped%20GEMM%20%E6%80%8E%E6%A0%B7%E5%AE%9A%E4%BD%8D%E5%A4%9A%E4%B8%AA%E9%97%AE%E9%A2%98%E7%9A%84%E8%BE%93%E5%87%BA%20tile)。

## 2. 实现类型与本次调用参数分开

`CollectiveBuilder` 接收架构、operator class、输入/累加类型、layout、Alignment、tile/cluster shape、stage 和 schedule 等模板参数，匹配实现并给出 `CollectiveOp` 类型。Mainloop 和 epilogue 因此在组装 kernel 时确定；本次处理的矩阵、尺寸和参数状态则由调用入口提供。

保存示例使用 `cute::Shape<int,int,int,int>` 作为 `ProblemShape` 类型，表达 `[M,N,K,L]`。这里固定的是形状类型和四个维度的位置，`int` 仍承载本次尺寸值。文档也允许把部分或全部尺寸静态化；需要区分“形状类型是模板参数”和“尺寸值已成为编译期常量”。（“CUTLASS GEMM Components”“Kernel API”。）

用户可见参数称为 `Arguments`，adapter 将它们转换为 kernel 实际接收的 `Params`，并管理其生命周期。初始化后的 handle 可以更新参数，再用于不同矩阵。文档称 adapter 为 stateful、kernel 为 stateless，指的正是参数状态由调用方管理，kernel 从 `Params` 接收它；grid 和内核特定的启动配置由 kernel 控制。（“Device API”。）

## 3. 逻辑坐标、存储布局和MMA类型各有作用

Mainloop 的 `StrideA/B` 是 global tensor 的 rank-3 stride，三个 mode 表示 `[outer,inner,batch]`，每个 mode 还可含层次结构。`SmemLayoutAtomA/B` 则描述铺到 shared 上的 rank-2 `[outer,inner]` 基本布局，不含 pipeline mode；MMA 的线程/值布局又负责把这些数据交给指令。（“Collective Mainloops”。）

以下用**教学坐标**展开乘法：数学 A 为 $A[M,K]$、B 为 $B_{\rm math}[K,N]$，批次下标为 l。将 B 的接口坐标记为

$$B_{\rm api}(n,k,l)=B_{\rm math}(k,n,l),\qquad P(m,n,l)=\sum_k A(m,k,l)B_{\rm api}(n,k,l).$$

这样 A 的 outer 是 M，B 的 outer 是 N，两者 inner 都是 K，可以分别按 `[M,K,L]`、`[N,K,L]` 理解。改变坐标的书写顺序不需要先搬数据转置 B。

例如数学 B 按行连续保存、N=3，`B_math[1,2]` 的元素偏移是 `1*3+2=5`。在 `(n,k)` 坐标中，stride 为 `(1,3)`，同一元素仍是 `2*1+1*3=5`。这说明同一存储可以用不同逻辑轴序描述；具体调用需按对应 API 构造 stride。带类型指针的加法按元素寻址，字节与 packed 容器的换算沿前述布局页处理。

Builder 的 `AlignmentA/B` 也按**元素个数**计：8个 FP16 对应16字节。它是实现选择的输入前提，应由实际指针与布局满足。类型同样要分层：global 的逻辑 `ElementA/B` 可以不同于 MMA 接收的 `TiledMma::ValTypeA/B`，特化可经 TMA 或用户 transform 转型；直接 `CollectiveMma` 的累加类型来自 `TiledMma::ValTypeC`，builder 用 `ElementAccumulator` 指定所需的指令累加类型。（“Collective Mainloops”“Collective Builder for CollectiveMmas”。）

## 4. Dispatch选择特化，Auto推导构建配置

`DispatchPolicy` 用 tag 选择架构与算法特化。文档的 Hopper 策略包含 `Stages`、`ClusterShape`、`arch::Sm90` 和 kernel `Schedule`；mainloop 与 schedule 可以在各自支持的组合内复用。类型名称保持 `CollectiveMma` 或 `GemmUniversal`，具体执行仍由匹配的特化决定。（“Collective Dispatch Policies”。）

`StageCountAuto` 根据每阶段 shared 用量，按 **1 CTA/SM** 的假设利用 shared 预算推导阶段数；`StageCount<N>` 则显式指定。`KernelScheduleAuto` 按给定参数选择可用 schedule，也允许用户指定具体类型。这属于 builder 的构建期选择，与运行时对候选配置计时比较的 autotune 不同；资源和驻留取舍见 [配置与自动调优](kernel-configuration-and-autotuning.md)。

保存文档对其 **3.0 builder** 的支持声明是 beta、SM90 tensorop；同名 kernel/adapter 还可由不同特化承接 2.x 与 3.x 接口。判断一个项目的支持范围时，需要对应它实际使用的版本和特化。（“Collective Builder for CollectiveMmas”“Kernel API”“Device API”。）

## 来源身份

- CUTLASS 3.0 GEMM API — NVIDIA CUTLASS Documentation（`https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html`），`snapshot-2026-07-02`，正文取得于 `2026-07-02T02:41:08+08:00`，配图补采于2026-09-21。完整保存493行与配图，主要使用 Device/Kernel/Collective，Tiled/Atom 按本页概述；坐标例为整理者展开。文档未给完整参数初始化和启动示例，本次未核外链实现、完整 CuTe 教程或运行。
