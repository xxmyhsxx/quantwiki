---
title: Kernel 配置选择与自动调优：资源、工作划分和测量
type: recipe
tags:
  - kernels
  - gpu
  - performance
  - matmul
sources:
  - raw/repositories/2026-09-22/kernel-skills/source/skills/patterns/choose-tile-size-and-work-partitioning/SKILL.md
  - raw/repositories/2026-09-22/kernel-skills/source/skills/triton/optimize-triton-block-parameters/SKILL.md
  - raw/repositories/2026-09-21/triton/source/python/tutorials/02-fused-softmax.py
  - raw/repositories/2026-09-21/triton/source/python/tutorials/03-matrix-multiplication.py
  - raw/repositories/2026-09-21/triton/source/python/tutorials/09-persistent-matmul.py
  - raw/repositories/2026-09-21/triton/source/python/triton/runtime/autotuner.py
  - raw/repositories/2026-09-21/triton/source/python/triton/language/semantic.py
  - raw/articles/2026-09-21/cuda-simt-kernels/article.md
updated: 2026-09-22
---

# Kernel 配置选择与自动调优：资源、工作划分和测量

配置选择是在数据复用、并行工作量和片上资源之间做取舍。更大的 tile 能减少重复加载，却也可能让尾块浪费增多、工作块变少、寄存器或共享内存用量上升。autotune 只能在给定候选和测量条件中选出较快者，不能代替正确性验证或证明全局最优。

本页从 `Choose Tile Size and Work Partitioning`、`Optimize Triton Block Parameters` 两个 Skill 提炼问题，再用 Triton 官方教程和 `runtime/autotuner.py` 核对。Skill 中的推荐数值属于起点，不能脱离 dtype、布局、架构和实际版本变成通用规则。前置解释见 [分块矩阵乘](../fundamentals/operators/gpu-kernel-computation-patterns.md) 与 [roofline](../fundamentals/hardware/arithmetic-intensity-and-roofline.md)。

## 1. 三种资源账要分开算

对 A/B 同为 e 字节元素、输出 tile 为 $B_M\times B_N$、K 步长为 $B_K$ 的常规分阶段 GEMM，估计输入缓冲大小为

$$S_{\rm tile}=eB_K(B_M+B_N),\qquad S_{\rm block}\approx L S_{\rm tile}+S_{\rm extra}.$$

L 是同时保留的缓冲阶段数，额外项包含 padding、barrier 和其他共享状态。这是解释明确使用 L 份 A/B 缓冲的模型；Triton `num_stages` 是编译器流水参数，不能保证所有 kernel 的 shared 字节数都严格等于这个乘式。row-wise softmax 的逻辑行也不能机械记为“一整行都占 shared”；寄存器布局与跨线程归约临时空间由编译结果决定。

FP32 累加器的总载荷约为 $4B_MB_N$ 字节。若它均匀分给 T 个线程，仅累加器就平均需要 $B_MB_N/T$ 个 32-bit 寄存器；实际还有输入片段、地址、循环状态和分配粒度。不能一律除以 32，那会把整个 CTA 当成一个 warp；也不能将一个 128×128 累加器视为单线程拥有它。

估算之后还要用编译产物确认实际资源：Triton softmax 教程就在预编译后读取寄存器数量和 shared 用量，再估计驻留能力。CUDA 文档 §2.3.3 说明，寄存器用量过高可能 spill 到设备内存；共享内存超限与寄存器 spill 是不同的资源问题。

## 2. occupancy 是驻留能力，不是性能分数

设每 SM 寄存器数为 R、shared 容量为 S、线程上限为 $T_{\max}$、block 上限为 $B_{\max}$；每 block 使用 T 个线程、r 个寄存器/线程、$S_b$ shared 字节。忽略分配粒度及其他架构约束时：

$$B_{\rm resident}\le\min\left(\left\lfloor\frac R{rT}\right\rfloor,\left\lfloor\frac S{S_b}\right\rfloor,\left\lfloor\frac{T_{\max}}T\right\rfloor,B_{\max}\right).$$

不使用 shared 时省略对应约束，不能除以零。理论 occupancy 再将驻留 warp 数除以该 SM 的 warp 上限；r、$S_b$ 应尽量使用编译结果，还要检查单 block 上限、架构分配粒度与 launch 的合法性。它不是 profiler 观测到的时间平均 achieved occupancy，更不是正在发射指令的比例。

教学算例：假设某设备 S=128 KiB、R=65536、线程上限 2048、block 上限 16、warp 上限 64。固定 T=128、假设编译得到 r=80；取 $B_M=64,B_N=128,B_K=32,e=2$，忽略额外 shared：

| 缓冲阶段 L | shared/block | shared 允许 block 数 | 综合驻留 block 上界 | 理论 warp 比例上界 |
| --- | ---: | ---: | ---: | ---: |
| 2 | 24 KiB | 5 | 5 | 20/64=31.25% |
| 3 | 36 KiB | 3 | 3 | 12/64=18.75% |

这里寄存器允许 6 个 block，所以 shared 是更紧约束。多一阶段可能减少访存等待，也可能压低驻留；不能只选更高 occupancy 的一行。算例参数是为推导构造的，不对应本次测量到的 GPU；也没有证明 r 在不同编译配置下保持 80。

`num_warps` 增加的是 program 内协作 warp 数，不等于单线程的指令级并行 ILP 增加；更多 warp 还会改变值的分配和资源用量。Skill 中 50% occupancy、70% 峰值效率、至少若干 resident blocks 等数字只能作为特定场景的经验，不是完成优化的统一门槛。

## 3. 局部复用不能掩盖全局工作不足

对普通输出 tile 划分，逻辑工作块数量为

$$G=\lceil M/B_M\rceil\lceil N/B_N\rceil.$$

若 G 很少，很多 SM 可能分不到工作；若 tile 太小，复用下降并增加调度与地址开销。最后一轮工作还可能只占满部分 SM。给空闲 SM 增加不产生有效输出的 block 不会自动提高性能。

用于比较不同 tile 的简单几何利用率是

$$U_{\rm tile}=\frac{MN}{\lceil M/B_M\rceil B_M\;\lceil N/B_N\rceil B_N}.$$

它计算逻辑有效输出面积，不是硬件 warp execution efficiency。M=N=129 时，128×128 tile 需要 4 块，面积利用率约 25.4%；64×64 tile 需要 9 块，约 45.1%。后者尾部浪费较少，但不保证更快，因为复用、流水和指令成本也变化。仅看 `N % tile / tile` 还会把恰好整除的满块误判成零利用率。

若输出块太少而 K 很长，可考虑把 K 切成多个部分，但 split-K 需要归约部分和，会增加中间存储或原子竞争，并改变浮点结合顺序。它是并行度与额外成本的交换，不是对所有小 M 的免费加速；具体分支可对照 [AWQ 内核](weight-only-dequant-kernels.md)。

## 4. persistent 描述工作循环，不承诺绑定 SM

Triton `09-persistent-matmul.py:matmul_persistent` 启动 $\min(\text{NUM\_SMS},G)$ 个 program，设备函数按 `tile_id = start_pid + q*NUM_SMS` 遍历输出 tile。q 是循环次数；每块依然独立完成相应输出，边界 store 有 mask。

“启动 NUM_SMS 个 program”不等于 API 将每个 program 固定绑到不同 SM。它表示控制 launch 的工作块数，再让每块循环处理多个 tile。教程的 softmax 则用估计的每 SM 驻留数量乘 SM 数决定 persistent program 数，说明 persistent 并不统一等于“每 SM 仅一个 block”。

普通 tiled GEMM 与 persistent GEMM 都可以只启动一次 kernel，因此后者不必然减少主机 kernel launch 次数。它改变了设备工作调度、循环和可能的数据复用；工作不均、循环状态和资源占用也可能有成本。本页只核对教程的普通 persistent 分支，不概括 TMA/warp-specialized 分支或所有 persistent 算法。

## 5. 候选配置表达哪些选择

官方矩阵乘教程把 kernel 的 `BLOCK_SIZE_*` 放进 `Config.kwargs`，把编译选项作为 `Config` 的构造器参数：

```python
triton.Config(
    {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128,
     "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
    num_warps=4, num_stages=3,
)
```

这里仅展示配置结构，不是实测推荐：tile 决定工作与数据分块，num_warps 决定 program 内的协作规模，num_stages 指导流水深度。三者相互影响，应一起计入资源和延迟分析。

候选应满足实际 kernel 和后端的约束。该 Triton 版本 `semantic.py:arange` 要求区间长度为 2 的幂，因此使用 `tl.arange(0,B)` 的教程对 B 有这一限制；这不能推广成“所有 CUDA tile 必须是 2 的幂或宽度为 32 的倍数”。一个 warp 可以被映射到多行，宽度 16 不自动表示一半线程空闲。是否使用 Tensor Core 同样不能仅由三个 tile 数值达到 16 来保证，还取决于 dtype、布局、精度与后端支持。

## 6. autotune 的 key、试跑和正确性

固定版本 `Autotuner.run` 将 `key` 指定且出现在参数中的值组成 tuple，并追加张量参数的 dtype。它没有自动将所有 shape、stride、layout、量化组大小都纳入 key。底层 JIT 为新参数编译 kernel，与 tuner 是否重新比较候选，是两件事。

若连续和转置输入、不同 group size 会改变最佳配置，应在接口中传入并将相应区别加入 key 或明确分派。相同 key 下复用某配置只说明免去了调优，不说明该配置适合所有布局。换设备或更改实现后，也应重新确认已有调优结果是否仍适用。

`Autotuner._bench` 运行候选并取得耗时，没有计算框架参考结果或比较输出。部分资源/编译失败会得到无限耗时，但没有报错且数值错误的候选仍可能被选中。因此应先用 [边界用例与数值容限](kernel-correctness-and-benchmarking.md) 筛出正确候选，再调优，最终复查胜出配置；不能用“被 autotune 选中”证明正确。

多次试跑还会重复修改数据。对 `out += partial`，若不重置，测试的第二次调用已不是相同初始状态；清零也只适用于目标输出本来应从零开始的情形。`reset_to_zero`、`restore_value` 或自定义 hook 用于维护这种语义，但自定义 hook 会覆盖默认 hook 的相应行为，必须检查实际执行路径。纯覆盖式 `out = f(input)` 与带状态更新的算子应分别处理。

从 tile 展开到 MMA 的 lane/寄存器、AWQ split-K 与 Triton 编译产物的具体例子见 [Tensor Core 与量化 GEMM](tensor-core-quantized-gemm.md)。它把本页资源估计与真实代码接口连接起来。

## 7. 从经验建议得到可检验的选择

先确定真实 shape 分布、布局、dtype、GPU 和计时范围，再提出少量能区分瓶颈的候选：例如缩小 M/N tile 检查工作不足，减小阶段数检查 shared 限制，改变 work ordering 检查缓存复用。合法性由后端与资源检查确认；资源估计只用于缩小搜索空间。

比较时同时观察有效工作量、编译资源和延迟，按场景权重保留必要的配置分派。多测几个配置能提高覆盖，但不存在必须 6–8 个候选才有效的通用定理。最优性仅限已测试候选、输入和环境；配置差异接近测量波动时，不应把最快单次结果当作稳定优势。调优开销与稳态延迟分别记录，具体计时见 [测量页](kernel-correctness-and-benchmarking.md)。

本轮核对了上述源码，完成资源、尾块与 persistent 分工的 CPU 教学计算；未运行 Triton GPU 编译、autotune 搜索或实际性能测量。

## 来源身份

| 来源 | 固定版本或快照 | 使用范围 |
| --- | --- | --- |
| [tensormux/kernel-skills](https://github.com/tensormux/kernel-skills/tree/7b7337a123f8711aa8e3d0452351d8fd30dde4b7) | `7b7337a123f8711aa8e3d0452351d8fd30dde4b7` | 本页列出的 Skill 提供待解释的问题和经验建议，具体主张经官方材料核对。 |
| [Triton](https://github.com/triton-lang/triton/tree/81a46fa0c04526e5df55a018ecfab72ff922f592) | `81a46fa0c04526e5df55a018ecfab72ff922f592` | 正文所列教程与 runtime 文件；仅代码核对。 |
| [CUDA Programming Guide：Writing SIMT Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html) | `snapshot-2026-07-02`，获取于 `2026-07-02T02:31:16+08:00` | 寄存器、shared memory 与 word/bank 关系。 |
