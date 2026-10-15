---
title: Kernel 配置选择与自动调优：资源、工作划分和测量
type: recipe
tags:
  - kernels
  - gpu
  - performance
  - matmul
sources:
  - raw/articles/2026-09-21/cutlass-efficient-gemm
  - raw/repositories/2026-09-22/kernel-skills
  - raw/repositories/2026-09-21/triton
  - raw/articles/2026-09-21/cuda-simt-kernels
  - raw/articles/2026-09-21/cuda-tile-kernels
  - raw/articles/2026-09-21/nvidia-shared-memory
updated: 2026-10-15
---

# Kernel 配置选择与自动调优：资源、工作划分和测量

配置选择是在数据复用、并行工作量和片上资源之间做取舍。更大的 tile 能减少重复加载，却也可能让尾块浪费增多、工作块变少、寄存器或共享内存用量上升。autotune 只能在给定候选和测量条件中选出较快者，不能代替正确性验证或证明全局最优。

本页从 `Choose Tile Size and Work Partitioning`、`Optimize Triton Block Parameters` 两个 Skill 提炼问题，再用 Triton 官方教程和 `runtime/autotuner.py` 核对。Skill 中的推荐数值属于起点，不能脱离 dtype、布局、架构和实际版本变成通用规则。前置解释见 [分块矩阵乘](../fundamentals/operators/gpu-kernel-computation-patterns.md) 与 [roofline](../fundamentals/hardware/arithmetic-intensity-and-roofline.md)。

## 1. 三种资源账要分开算

对 A/B 同为 e 字节元素、输出 tile 为 $B_M\times B_N$、K 步长为 $B_K$ 的常规分阶段 GEMM，估计输入缓冲大小为

$$S_{\rm tile}=eB_K(B_M+B_N),\qquad S_{\rm block}\approx L S_{\rm tile}+S_{\rm extra}.$$

L 是同时保留的缓冲阶段数，额外项包含 padding、barrier 和其他共享状态。这是解释明确使用 L 份 A/B 缓冲的模型；Triton `num_stages` 是编译器流水参数，不能保证所有 kernel 的 shared 字节数都严格等于这个乘式。row-wise softmax 的逻辑行也不能机械记为“一整行都占 shared”；寄存器布局与跨线程归约临时空间由编译结果决定。

FP32 累加器的总载荷约为 $4B_MB_N$ 字节。若它均匀分给 T 个线程，仅累加器就平均需要 $B_MB_N/T$ 个 32-bit 寄存器；实际还有输入片段、地址、循环状态和分配粒度。不能一律除以 32，那会把整个 CTA 当成一个 warp；也不能将一个 128×128 累加器视为单线程拥有它。

Efficient GEMM 的“Pipelining”在两级保存输入双缓冲：CTA 的 shared tiles 轮流供应当前计算与接收下一轮 global 数据；warp 的寄存器 fragments 则轮流供应当前 MMA 与接收下一次 shared fetch。两级作用域和资源不同，输入片段也不同于一直累计输出的 accumulator；它们都要进入资源账。此处保留所述常规结构，不把两个 buffer 当作所有 CUTLASS/架构的固定 stage 数，具体完成等待见 [异步流水](gpu-async-copy-pipelines.md)。

估算之后还要用编译产物确认实际资源：Triton softmax 教程就在预编译后读取寄存器数量和 shared 用量，再估计驻留能力。CUDA 文档 §2.3.3 说明，寄存器用量过高可能 spill 到设备内存；共享内存超限与寄存器 spill 是不同的资源问题。

### 静态与动态shared都占用每block资源

Using Shared Memory快照用`__shared__ int s[64]`声明编译期容量，用`extern __shared__ int s[]`接收启动时的动态区。`<<<1,n,n*sizeof(int)>>>`第三项是每block请求的动态shared字节，不是该block全部shared的上限；资源账还须计入静态区及实际编译状态，“动态”不表示免占SM资源。（“Static/Dynamic Shared Memory”。）

多个动态数组共用一份动态区，以指针切片分配。原例按int/float/char列总字节，没有给任意dtype的对齐算法；推广时须同时核各切片起点的类型对齐、padding与总容量，这是整理者接口补充。文章的16 banks/half-warp请求限定CC1.x，32 banks/整warp请求限定CC2.0，4/8-byte可配置bank宽度限定CC3.x，64KB L1/shared划分限定CC2.x与3.x，不能当当前设备的统一规格；本页先构造预算，再用编译资源和具体设备确认。

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

### 拆K后的合并与后处理

Efficient GEMM 的“Parallelized Reductions”按协作范围区分：

| 分工 | 部分结果在哪里产生 | 所述合并路径 |
| --- | --- | --- |
| parallel reduction split-K | 多个 CTA 对同一输出计算不同 K 分片 | partitionedK GEMM 先写 workspace，再由第二个 batched reduction kernel 合并分片部分和 |
| sliced-K | 同一 CTA 的多个 warp 计算不同 K 分片 | CTA 内合并各 warp 的部分和；这本身不增加 CTA 数 |

表中的 split-K 采用 workspace 与第二个 kernel，其他实现可有不同合并方式；沿 M/N 分配独立输出、各自完成 K 的 warp 分工也不属于 sliced-K。文档以 K=4096 分20段举例，前19段各204、最后220；此例定义逻辑分片区间，执行时仍需满足 tile/MMA 的 K 维要求；分片区间与 buffer stage 是不同粒度。

整理者推导：若 $P_s$ 是 S 个 K 分片的矩阵部分和，目标为

$$D=\alpha\sum_{s=0}^{S-1}P_s+\beta C.$$

完整的 $\beta C$ 只应计一次；若每个分片都先加完整 $\beta C$ 再求和，会变成 $S\beta C$。bias 同理。非线性也不能默认与部分和求和交换，例如 ReLU(−2+3)=1，而 ReLU(−2)+ReLU(3)=3。拆 K 后仍需计入 workspace、额外流量、同步与浮点归约顺序的成本。

## 4. persistent 描述工作循环，不承诺绑定 SM

Triton `09-persistent-matmul.py:matmul_persistent` 启动 $\min(\text{NUM\_SMS},G)$ 个 program，设备函数按 `tile_id = start_pid + q*NUM_SMS` 遍历输出 tile。q 是循环次数；每块依然独立完成相应输出，边界 store 有 mask。

“启动 NUM_SMS 个 program”不等于 API 将每个 program 固定绑到不同 SM。它表示控制 launch 的工作块数，再让每块循环处理多个 tile。教程的 softmax 则用估计的每 SM 驻留数量乘 SM 数决定 persistent program 数，说明 persistent 并不统一等于“每 SM 仅一个 block”。

普通 tiled GEMM 与 persistent GEMM 都可以只启动一次 kernel，因此后者不必然减少主机 kernel launch 次数。它改变了设备工作调度、循环和可能的数据复用；工作不均、循环状态和资源占用也可能有成本。上述 Triton 例子只核对教程的普通 persistent 分支，不代表 TMA/warp-specialized 分支或所有 persistent 算法。

### Hopper的两个consumer怎样分担输出

同一快照“Hopper Warp Specialization”另述 Hopper 的 producer/consumer warp-group（承担搬运/计算的 warp 协作组）：producer 先等待 shared buffer empty，用 TMA（Tensor Memory Accelerator）从 global 搬入数据；完成后通知 full，consumer 等待、发起 MMA，消费完成再 release，供下一轮复用。它们用显式完成与释放维护依赖，不能把普通 `cp.async` 的等待距离直接搬给 TMA。

| Hopper persistent 设计 | 两个 consumer 的工作 | 希望重叠或节省什么 |
| --- | --- | --- |
| cooperative | 沿 M 将同一输出 tile 分成两半，各组处理对应行 | 减少每组持有的输出状态，为较大 tile 提供选择 |
| ping-pong | Tile Scheduler 分给两组不同输出 tile，producer 按有序序列供给 | 重叠一组的 epilogue 与另一组的 math |

按 M 拆独立行不是 sliced-K 的部分和，按不同输出 tile 交替也不等于单 tile 双缓冲。Tile Scheduler 按 cluster 形状与可用 SM 信息安排工作；persistent block 可在其生命周期处理多个输出 tile，以摊薄 CTA 启动与 prologue，而非必然减少 host kernel launch。上述分工限定 Hopper；文档中的 SM100 warp-specialization 结构显著不同。

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

候选应满足实际 kernel 和后端的约束。该 Triton 版本 `semantic.py:arange` 要求区间长度为 2 的幂，因此使用 `tl.arange(0,B)` 的教程对 B 有这一限制；这不能推广到所有一般CUDA分块算法；本节下述CUDA Tile API另有自己的编译形状规则。一个 warp 可以被映射到多行，宽度 16 不自动表示一半线程空闲。是否使用 Tensor Core 同样不能仅由三个 tile 数值达到 16 来保证，还取决于 dtype、布局、精度与后端支持。

### CUDA Tile hint与指针前提承担不同职责

Writing Tile Kernels快照§2.4.11的hint是附在kernel或特定访存调用上的编译建议，可按架构设置，也可被忽略。`occupancy`推荐每SM的活跃CTA数，不是上文warp驻留百分比；`latency=1…10`表示该访问从轻到重的流量/预取倾向，不是周期数。`allow_tma`只用于tile-space访问，允许支持硬件上的TMA候选不保证最终必用，gather/scatter不接受这一hint。

Python `replace_hints`创建具有独立JIT cache的新kernel对象，提供的是调优候选，不是正确性或速度证明。CUDA Tile的shape/dtype按自身编译规则特化，不能直接套用上文Triton的`num_warps`、`num_stages`或autotuner行为。（§2.4.4–2.4.5、§2.4.11。）

相反，C++示例的`__restrict__`和`assume_aligned(ptr,16)`是必须满足的别名与16-byte地址对齐承诺；错误承诺会导致未定义行为。分配基址对齐不能证明切片后的地址也对齐，TMA仍需核其他布局和硬件条件。编译提示、接口前提、实际编译资源与测量结果要分别记录。（§2.4.12；坐标/边界见[Tile接口](../fundamentals/operators/tensor-layout-and-kernel-contracts.md)。）

## 6. autotune 的 key、试跑和正确性

固定版本 `Autotuner.run` 将 `key` 指定且出现在参数中的值组成 tuple，并追加张量参数的 dtype。它没有自动将所有 shape、stride、layout、量化组大小都纳入 key。底层 JIT 为新参数编译 kernel，与 tuner 是否重新比较候选，是两件事。

若连续和转置输入、不同 group size 会改变最佳配置，应在接口中传入并将相应区别加入 key 或明确分派。相同 key 下复用某配置只说明免去了调优，不说明该配置适合所有布局。换设备或更改实现后，也应重新确认已有调优结果是否仍适用。

`Autotuner._bench` 运行候选并取得耗时，没有计算框架参考结果或比较输出。部分资源/编译失败会得到无限耗时，但没有报错且数值错误的候选仍可能被选中。因此应先用 [边界用例与数值容限](kernel-correctness-and-benchmarking.md) 筛出正确候选，再调优，最终复查胜出配置；不能用“被 autotune 选中”证明正确。

多次试跑还会重复修改数据。对 `out += partial`，若不重置，测试的第二次调用已不是相同初始状态；清零也只适用于目标输出本来应从零开始的情形。`reset_to_zero`、`restore_value` 或自定义 hook 用于维护这种语义，但自定义 hook 会覆盖默认 hook 的相应行为，必须检查实际执行路径。纯覆盖式 `out = f(input)` 与带状态更新的算子应分别处理。

从 tile 展开到 MMA 的 lane/寄存器、AWQ split-K 与 Triton 编译产物的具体例子见 [Tensor Core 与量化 GEMM](tensor-core-quantized-gemm.md)。它把本页资源估计与真实代码接口连接起来。

## 7. 从经验建议得到可检验的选择

先确定真实 shape 分布、布局、dtype、GPU 和计时范围，再提出少量能区分瓶颈的候选：例如缩小 M/N tile 检查工作不足，减小阶段数检查 shared 限制，改变 work ordering 检查缓存复用。合法性由后端与资源检查确认；资源估计只用于缩小搜索空间。

比较时同时观察有效工作量、编译资源和延迟，按场景权重保留必要的配置分派。多测几个配置能提高覆盖，但不存在必须 6–8 个候选才有效的通用定理。最优性仅限已测试候选、输入和环境；配置差异接近测量波动时，不应把最快单次结果当作稳定优势。调优开销与稳态延迟分别记录，具体计时见 [测量页](kernel-correctness-and-benchmarking.md)。

此前整理核对了上述 Triton 源码，并完成资源、尾块与 persistent 分工的 CPU 教学计算；本次 Efficient GEMM 增量核读保存文档/配图，并核对拆 K 后处理的教学算术。

如何从 Nsight 时间线和计数器判断应改变 tile、warp 数还是 stages，以及怎样检验等待减少是否真的缩短延迟，见 [GPU 算子性能分析](gpu-kernel-performance-analysis.md)。

## 来源身份

| 来源 | 固定版本或快照 | 使用范围 |
| --- | --- | --- |
| Efficient GEMM in CUDA（`https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html`） | `snapshot-2026-07-02`，正文取得于 `2026-07-02T02:41:08+08:00`，3图补采于2026-09-21 | 保存175行/3图；epilogue、两级缓冲、拆K和Hopper设计；后处理反例为整理者推导，未取外链源码或运行。 |
| Using Shared Memory in CUDA C/C++（`https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/`） | `snapshot-2026-07-02` | 静态/动态区、launch字节与多数组分区及历史设备条件；完整保存正文已读，对齐推广为整理者说明，未编译/测资源。 |
| CUDA Programming Guide：Writing Tile Kernels（`https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-tile-kernels.html`） | `snapshot-2026-07-02` | §2.4.11–2.4.12 hint与前提，保存正文全读；并非实际编译资源/TMA或性能验证，完整API未核。 |
| tensormux/kernel-skills（`https://github.com/tensormux/kernel-skills/tree/7b7337a123f8711aa8e3d0452351d8fd30dde4b7`） | `7b7337a123f8711aa8e3d0452351d8fd30dde4b7` | 本页列出的 Skill 提供待解释的问题和经验建议，具体主张经官方材料核对。 |
| Triton（`https://github.com/triton-lang/triton/tree/81a46fa0c04526e5df55a018ecfab72ff922f592`） | `81a46fa0c04526e5df55a018ecfab72ff922f592` | 正文所列教程与 runtime 文件；仅代码核对。 |
| CUDA Programming Guide：Writing SIMT Kernels（`https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html`） | `snapshot-2026-07-02`，获取于 `2026-07-02T02:31:16+08:00` | 寄存器、shared memory 与 word/bank 关系。 |
