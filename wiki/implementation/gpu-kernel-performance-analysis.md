---
title: GPU 算子性能分析：从时间线到瓶颈假设
type: recipe
tags:
  - kernels
  - gpu
  - performance
  - evaluation
sources:
  - raw/articles/2026-09-22/nsight-compute/pages/ncu-guide.md
  - raw/articles/2026-09-22/nsight-compute/pages/ncu-cli.md
  - raw/articles/2026-09-22/nsight-systems/article.md
  - raw/articles/2026-09-22/pytorch-performance-tools/pages/torch-profiler.md
  - raw/articles/2026-09-22/pytorch-performance-tools/pages/torch-event.md
  - raw/articles/2026-09-22/pytorch-performance-tools/pages/torch-rms.md
  - raw/repositories/2026-09-21/vllm/source/csrc/libtorch_stable/layernorm_kernels.cu
  - raw/repositories/2026-09-21/sglang/source/python/sglang/jit_kernel/csrc/elementwise/fused_add_rmsnorm.cuh
  - raw/repositories/2026-09-21/sglang/source/python/sglang/srt/layers/quantization/int8_kernel.py
  - raw/repositories/2026-09-21/sglang/source/python/sglang/srt/layers/quantization/w8a8_int8.py
  - raw/repositories/2026-09-21/sglang/source/sgl-kernel/csrc/gemm/int8_gemm_kernel.cu
updated: 2026-09-22
---

# GPU 算子性能分析：从时间线到瓶颈假设

性能分析要建立一条可检验的解释：在给定 GPU、输入和调用方式下，哪些成本限制了完成时间；某项修改怎样改变这些成本；独立计时是否确认了收益。计时只回答“有多快”，硬件计数器与时间线帮助回答“为什么”，对照实验才区分相互竞争的解释。

本页面向 NVIDIA CUDA/Triton。工具命令依据 2026-09-22 获取的官方文档快照，实际选项、指标与架构支持须以目标机器版本为准。先完成 [正确性与计时协议](kernel-correctness-and-benchmarking.md)，再使用这里的方法；没有通过数值和内存检查的候选不能进入性能排名。

## 1. 先确认分析对象，再缩小范围

| 层次 | 先问的问题 | 首选证据 |
| --- | --- | --- |
| 模型/子图 | 这个算子是否位于关键路径，值得优化吗？ | PyTorch Profiler 的算子关联、Nsight Systems 的 CPU/GPU 时间线 |
| 对外算子 | 一次调用实际启动几个 kernel、转换和分配？ | NVTX 范围、CUDA API、kernel 和 memcpy 事件 |
| 设备 kernel | 工作不足、带宽、指令吞吐、依赖延迟还是资源约束？ | Nsight Compute 的 launch、memory、compute、scheduler 等 sections |
| 源码与指令 | 哪段索引、解包、加载或同步产生上述限制？ | 优化构建的 line information、SASS、编译资源及局部对照 |

Nsight Systems 展示执行关系；Nsight Compute 定向收集 kernel 内部计数器。PyTorch Profiler 可记录框架算子、shape、CPU/CUDA 活动，并用 stack 定位调用代码，但不是完整硬件计数器分析的替代。`record_shapes` 和 `with_stack` 自身有开销，日常计时应与诊断采集分开。

多 stream 重叠时，各 kernel duration 相加可能大于整段墙钟时间；一个 kernel 变快也可能只减少被其他工作覆盖的时间。因此先在时间线上确认依赖与关键路径，不能直接把热点汇总表的百分比当成整体可加速比例。

## 2. 第一轮报告只收能改变判断的信息

先看少量 sections，再根据假设扩展，避免一开始对所有调用运行 `--set full`：

| Section / 证据 | 看什么 | 不能直接推出什么 |
| --- | --- | --- |
| `LaunchStats` | grid、block、寄存器、shared-memory 配置 | 大 block 或更多线程一定更快 |
| `Occupancy` | 理论驻留上限、实际平均驻留及限制资源 | occupancy 就是算力利用率 |
| `SpeedOfLight` | 计算和内存单元相对峰值的吞吐概览 | Memory 百分比就是 DRAM 带宽百分比 |
| `MemoryWorkloadAnalysis` | DRAM/L2/L1 的流量、请求、命中、访存流水压力 | 高命中率一定减少总时间 |
| `ComputeWorkloadAnalysis` / `InstructionStats` | 实际使用哪些执行流水和指令 | 标称 W4/W8 已保证使用目标 Tensor Core |
| `SchedulerStats` | active、eligible、issued warps 与空发射槽 | 活跃 warp 都随时可以发射 |
| `WarpStateStats` / `SourceCounters` | 指令依赖、同步等待、采样位置 | 最大 stall 项必然值得优先消除 |

不同工具/GPU 版本的 section 与 metric 名称可能不同，先运行 `ncu --list-sections` 和 `ncu --query-metrics`。阅读完整 metric 名称及单位；`avg`、`sum`、`pct_of_peak_sustained_elapsed` 的分母与聚合范围并不相同。

特别要区分三件事：**occupancy 说明驻留了多少 warp，eligible 说明有多少已准备好，issue 说明实际发射了多少。** 若调度器已经持续发射，其他 warp 的等待未必限制总吞吐。官方 Profiling Guide 的 Scheduler Statistics、Warp State Statistics 都提示，应在发射不足时重点分析 stall。

## 3. 由观测提出候选解释，而不是套优化口诀

下表是整理者依据工具语义与执行原理建立的诊断流程，所有优化方向都需要对照验证。

| 观测组合 | 候选解释 | 下一步区分证据与改动 |
| --- | --- | --- |
| 时间线上 kernel 很短、间隙显著，孤立 kernel 已很快 | 主机提交、同步或调用粒度主导 | 检查 API 与依赖；比较真实普通调用和允许的 Graph/融合方式，保留相同输出与计时范围 |
| 计算与内存吞吐都低，grid 很小 | 可并行工作量不足 | 核对输出 tile 数与 SM 数；改变 tile/工作划分，查看 grid、尾波与耗时是否一起改善 |
| DRAM 吞吐高且字节数接近该路径的合理流量 | 外存带宽可能限制 | 减少重复读取或中间写回；若只是改计算指令，预期收益应有限 |
| DRAM 不忙，但 eligible warp 少、long-scoreboard 多 | 访存延迟或依赖链未被隐藏 | 看等待的消费指令、L1/L2/DRAM 流量与访问模式；检查独立加载、预取距离和并发，不直接判作 DRAM 带宽饱和 |
| shared 请求异常、short-scoreboard 或相关流水压力高 | 共享内存访问、bank conflict 或其他短延迟依赖 | 结合请求/事务与源码定位；改变 swizzle/布局，确认访问数量和延迟的变化 |
| 寄存器多、local-memory 访问增加，增大 tile 后变慢 | spill 或驻留减少 | 查编译资源和 SASS；缩短值的存活区间、减 tile/stages，比较额外访存是否减少 |
| Tensor Core 不忙，整数解包/地址指令密集 | 辅助指令流水或供数路径限制 | 对照解包、索引和 MMA 的指令组成，尝试预排布局或重叠解包；统计实际增加的存储成本 |
| barrier 等待多且同一 CTA 的工作不均 | 到达同步点的时间不一致 | 查屏障前不同 warp 的工作；改善分工。不能为了减少 stall 删除正确性所需的屏障 |

long-scoreboard 常与 L1TEX 路径的未完成依赖有关，不能仅凭名称锁定 HBM；short-scoreboard 也不是 bank conflict 的同义词。采样位置通常落在等待结果的**消费者**上，根因可能是更早的生产者。`not_selected` 表示 warp 已准备好但调度器选择了别的 warp，不应把它当成必须归零的故障。

## 4. Roofline 与实际流量要用同一口径

对某一明确内存层级，算术强度与吞吐上界写成

$$
I=\frac{F}{Q},\qquad P\le\min(P_{\rm peak},B I).
$$

$F$ 是指定口径的工作量，$Q$ 是该层级的字节数。用算法的最少输入输出字节计算的是理想工作模型；用 profiler 的 DRAM 字节计算的是本次执行的外存流量模型。缓存复用、重复加载、padding 和 spill 都会让两者不同。若用 L2 字节，就应该与 L2 带宽比较，不能仍套 HBM 上限。

对于 INT8 GEMM，$2MNK$ 可记为有用整数操作量并报告 TOP/s；对于浮点 Tensor Core 则说明输入/累加类型和 FLOP/s 口径。不要把 INT8 操作率除以 FP16 峰值，或用 dense 工作量与稀疏峰值比较。反量化、缩放与地址计算也消耗指令，但不在简单 $2MNK$ 中。

Roofline 给出资源上界。点远低于上界时，还可能是工作不足、启动、依赖、指令流水或同步限制，不能仅依据落在拐点哪侧给一个最终诊断。理论与实际占用的区别见 [配置选择](kernel-configuration-and-autotuning.md)，强度推导见 [Roofline 基础](../fundamentals/hardware/arithmetic-intensity-and-roofline.md)。

## 5. 案例一：RMSNorm 的分析实验如何设计

以下是**待在目标 GPU 执行的实验设计**，不是已测的性能结论。考虑不带 residual 的 $M\times K$ 输入，输出 dtype 与输入一致，统计使用 FP32。

### 建立工作与流量账

一个理想单遍实现读取 X、写 Y，并读取 K 个权重。在忽略缓存行、统计临时存储且假设权重全局只读取一次的理想情况下，有效字节为

$$
Q_{\min}=2MKb+Kb,
$$

$b$ 为输入/输出元素字节数。它是算法流量下界模型，不是所有 kernel 的实际 HBM 流量。每行一次权重逻辑访问、输入二次读取、临时张量及 spill 要另列，缓存可能承接其中一些访问。

[vLLM 与 SGLang RMSNorm 源码](rmsnorm-cuda-triton-kernels.md)提供两个可以检验的机制：vLLM 某 CUDA 路径归约后再次读取输入；SGLang 的一个 fused-add 分支把向量保存在寄存器。后者还写 residual，语义和输出不同，不能直接用两个耗时排名。应在相同语义下构造或选择相应对照。

### 可复用的实验表

| 输入/变化 | 先验证的假设 | 采集与判断 |
| --- | --- | --- |
| 固定 K，M 从 1、16 到 256/更多 | 小 M 可能缺少足够 CTA，或提交成本占比高 | 先看时间线与 grid，再看吞吐；只有大 M 快不能说明 decode 同样受益 |
| 固定 M，扩大 K；仅加入接口支持的边界形状 | 行归约、寄存器与 padding 成本上升 | 看寄存器、local traffic、eligible warps、延迟，不预设大行一定 bandwidth-bound |
| 改每行 warp 数或向量宽度 | 更高协作并行度能否抵消归约成本 | 同时核对尾块数值、shared/barrier、资源与耗时 |
| 保留输入到寄存器 vs 二次加载 | 减少加载是否值得更多活跃寄存器 | 查看真实流量与 spill；源代码少一次 load 不保证 HBM 少一次读取 |

随附 RMSNorm factory 比较 eager FP32 公式和安装环境中的 `torch.nn.functional.rms_norm`，用冻结的 CPU FP64 结果做数值参考。它是采集入口的教学基线，**没有实现自定义 CUDA/Triton kernel，也不假定 PyTorch 路径一定融合为一个 kernel**。自己的候选可以按第 8 节接口接入；先从时间线确认生成的 kernel 数量。

## 6. 案例二：量化 GEMM 必须分开三种成本

W8A8 的量化表示与设备执行见 [W8A8 算子](w8a8-quantization-gemm-kernels.md)。其对外 Linear 可包含：

$$
T_{\rm linear}\approx T_{\rm quant}+T_{\rm gemm}+T_{\rm gaps/extra}.
$$

这是同一串行调用链上的分解模型。分别孤立测量得到的几个中位数，不能保证相加等于完整调用：缓存、分配、提交间隙和并发条件可能变了。优先用同一 trace 定位阶段，再用独立测量做对照。

配套 SGLang factory 提供三个名字明确的调用：

| 调用 | 输入已准备到哪里 | 计入什么 |
| --- | --- | --- |
| `quant` | 浮点 X 已在 GPU | 激活 INT8 编码、行尺度及包装内分配 |
| `gemm` | $q_x,s_x$ 和权重已准备好 | 实际 scaled-GEMM wrapper，含输出/工作区相关路径 |
| `linear` | 浮点 X 与量化权重已准备好 | 同一次调用里的 quant → scaled GEMM |

它匹配已读 SGLang commit 的 API，但不自动安装框架，也不证明其他版本/架构可运行；记录安装版本并核对设备分派。验证参考针对**量化后的整数表示**，不是原始浮点模型的质量评测。容限是示例输入的起点，不能遇到失败就放宽。CPU 编码参考保留 FP32 的已缩放值，再用 FP64 做半整数远离零舍入，避免直接在 FP32 中加 0.5 把略低于半整数的值推到错误一侧。

继续分析时，按问题选择对照：

- **小 M 与大 M**：先确认 host 是否换了 tile/backend，再比较工作块数量、辅助指令和 Tensor Core 利用；不能把两者当成同一个内核只变大了输入。
- **W4A16 与 W8A8**：W4A16 需计入打包权重、scale、zero point/g_idx、repack 或输入置换；W8A8 还要计入激活量化。若重排只在加载期做，应作为启动成本单列，不能每次都摊入 steady-state GEMM。
- **增加 stages**：记录编译 shared/registers、eligible warp 和耗时。等待减少而整体变慢，可能是驻留下降；occupancy 更低但耗时更好也并不矛盾。
- **split-K/stripe**：同时测主计算与部分和合并，复查 FP16/FP32 临时存储和误差。分出的 CTA 更多不等于完整调用更快；[Marlin 的两层归约](weight-only-dequant-kernels.md)说明了额外语义。
- **融合或改布局**：基线和候选必须完成同一输出；转换从 GPU 搬到加载阶段时，分别报告两种成本，不隐藏转换。

以 FP16 激活/输出、仅一个 FP16 scale/组的**假设 W4 格式**为例，最少持久数据量可记为

$$
Q_{\rm W4}=2MK+\frac{KN}{2}+2MN+\frac{2KN}{g}.
$$

假设 $K$ 可被组大小 $g$ 整除，忽略 zero point、padding、g_idx、重复读取与临时归约。这个公式用于对照预算，不替代具体 Marlin/AWQ/QoQ 格式，也不是 measured DRAM bytes。W8A8 的激活量化则至少引入 INT8 $MK$ 码与 FP32 $4M$ 尺度的写出，以及后续 GEMM 对它们的读取；是否命中 cache 需实测。

## 7. 最小采集命令与报告阅读

以下在已安装 PyTorch CUDA、对应工具和候选依赖的 NVIDIA 主机执行。命令从项目根运行；输出写到新的实验目录。脚本不会自行安装依赖或改 GPU 时钟。CUDA 源码诊断采用正常优化构建并加 `-lineinfo`；不要用改变优化行为的 `-G` 构建代表生产速度。

先完成测量页的正确性检查和独立 benchmark，再采集时间线：

~~~bash
mkdir -p /tmp/kernel-perf-rmsnorm
nsys profile --trace=cuda,nvtx --sample=none \
  -o /tmp/kernel-perf-rmsnorm/timeline \
  python3 wiki/assets/gpu-kernel-performance-analysis/benchmark_operator.py \
  --factory performance_cases:rmsnorm --op torch_rmsnorm --mode trace \
  --config '{"m":16,"k":1024,"dtype":"float16"}'

nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum \
  /tmp/kernel-perf-rmsnorm/timeline.nsys-rep
~~~

脚本先校验和预热，再以 NVTX push/pop 标记 `perf_target`。上述 nsys 命令仍捕获启动/校验等活动，应在 GUI 中定位目标 NVTX 范围并关联其 CUDA launches；统计表是聚合入口，不能代替关键路径时间线。这里是单进程独立算子，不能由此判断整个 vLLM/SGLang 模型的调度间隙。

对目标范围内的 kernel 做少量计数器采集：

~~~bash
ncu --list-sections
ncu --query-metrics

ncu --nvtx --nvtx-include 'perf_target/' --set basic \
  --launch-count 3 -o /tmp/kernel-perf-rmsnorm/kernel \
  python3 wiki/assets/gpu-kernel-performance-analysis/benchmark_operator.py \
  --factory performance_cases:rmsnorm --op torch_rmsnorm --mode trace \
  --trace-iterations 1 --config '{"m":16,"k":1024,"dtype":"float16"}'
~~~

`perf_target/` 的末尾斜线用于 NVTX push/pop 范围匹配，不能照搬 start/end 范围语法。`--launch-count 3` 限制匹配的 kernel launch 数，不是三个完整 Linear；一个 wrapper 产生更多 kernel 时，这只是抽样。先用时间线发现具体名称，再用 `--kernel-name` 等过滤对目标实例采集，避免误把初始化或辅助 kernel 当 GEMM。

第二轮按假设添加本机存在的 `MemoryWorkloadAnalysis`、`SchedulerStats`、`WarpStateStats` 或 Tensor Roofline section；不要把不存在的 metric 当成零值。若遇到计数器权限错误，应记录未取得指标并由管理员按环境处理，不把空报告解释为低利用率。

### 采集方式本身会改变执行

官方 Profiling Guide 的 Replay、Overhead、Reproducibility 说明：

- 默认 kernel replay 可能保存/恢复写入内存、重复启动并串行化 kernel；profile 下用 event/墙钟包住调用会包含工具开销，不能作正常 benchmark。
- cache control 与 clock control 会影响对照条件。默认隔离采集的缓存状态未必等于模型连续执行；应用级 replay 或 range replay 需按依赖与确定性选用，不是随意换开关后继续横比。
- 依赖并发才能前进的 kernel 不宜机械套用孤立 kernel replay。多 stream、通信或多 GPU 路径应先阅读对应工具支持范围。
- 工具内部报告的 kernel duration 与外部整段墙钟也不是同一量，采样/SASS patching 指标和硬件计数器可能来自不同 pass。

因此诊断后必须退出 profiler，以相同 workload 和计时协议复测，并回到真实调用链确认收益。

## 8. 采集脚本的接口与结果

配套 [benchmark_operator.py](../assets/gpu-kernel-performance-analysis/benchmark_operator.py)提供 `check`、`bench`、`trace`、`torch-profiler` 四种模式；[performance_cases.py](../assets/gpu-kernel-performance-analysis/performance_cases.py)提供 RMSNorm 和 SGLang W8A8 示例。两者都不实现/优化新 kernel。

外部 factory 通过 `--factory module:function` 返回：

~~~python
{
    "calls": {"baseline": baseline, "candidate": candidate},
    "validate": validate,       # 失败必须抛出异常；比较全部有意义输出
    "metadata": {...},          # 语义、layout、dtype、配置与候选身份
    "repeat_safe": True,        # 输入不变，输出可反复覆盖
}
~~~

有状态接口必须设 `repeat_safe=False`、提供 `prepare(op_name)` 恢复输入，并使用 `--repeats 1`。prepare 在测量区间外执行，但会影响缓存；不要把它的成本和影响说成不存在。此模板要求所有工作在当前 CUDA stream 上完成；自行启动其他 stream 的候选需显式接回依赖或扩展计时，模板不会自动推断。

结果记录 GPU/软件版本、shape/stride、容限、样本、cache 策略、脚本与 factory 哈希；正确性在采集前后检查。计时样本是多次调用的平均值，其分位数不是服务请求 p95/p99。event 模式保留流上的空闲间隙，不保证等于纯 kernel 指令执行时间；wall 模式包含提交与完成等待。trace 模式明确不输出 benchmark 数字。

文件默认打印，显式 `--output` 才保存 JSON；已有文件被拒绝覆盖。实验记录还应补充 git commit、实际 kernel 名、编译配置、并发进程/功耗时钟条件、原始 profiler 报告及对照改动。一次更快不能证明稳定改进；在相同 shape 矩阵中复测多次进程运行，并确认新配置没有牺牲正确性或重要工作负载。

## 9. 当前验证范围

2026-09-22 在 Darwin arm64 上检查：本机没有可用的 NVIDIA CUDA 采集环境，未运行 nsys/ncu/Compute Sanitizer，也未运行两个 factory 的 CUDA 路径。配套脚本只完成 CLI、Python 语法、无 CUDA 失败路径、计时统计/状态约束及教学预算的 CPU 检查；[检查脚本](../assets/gpu-kernel-performance-analysis/check_analysis.py)与[记录](../assets/gpu-kernel-performance-analysis/checks-2026-09-22.json)保留具体范围。

后续实验将在具备 NVIDIA GPU 和 CUDA 环境的目标设备上执行；当前编辑主机的硬件条件只说明本次检查范围，不限制后续实验环境。目标设备上依次完成 CUDA 路径的正确性检查、Compute Sanitizer 检查、独立计时和 Nsight 分析，并记录实际 GPU、软件版本及输入条件。

本页案例在目标设备执行前属于实验方案。执行后保留原始样本与分析报告，再将有实测依据、可复用的结论反馈到对应知识页。

## 来源身份

| 来源 | 版本/快照与使用范围 |
| --- | --- |
| [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) 与 [CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) | snapshot-2026-09-22；sections、计数器语义、NVTX 过滤、replay/cache/clock 和测量开销 |
| [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/) | snapshot-2026-09-22；CUDA/NVTX 时间线、CLI 与统计报告 |
| [PyTorch Profiler](https://docs.pytorch.org/docs/2.14/profiler.html)、[CUDA Event](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.Event.html)、[RMSNorm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.rms_norm.html) | URL 版本 2.14，snapshot-2026-09-22；API 语义，不代表本机安装或运行该版本 |
| [vLLM](https://github.com/vllm-project/vllm/tree/568afb3a13806beb53bb2e6bd518269357b237c0) | RMSNorm 加载/归约/写回设计，仅源码证据 |
| [SGLang](https://github.com/sgl-project/sglang/tree/2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc) | fused-add RMSNorm 与动态 W8A8 wrapper/设备接口；不代表已测性能 |
