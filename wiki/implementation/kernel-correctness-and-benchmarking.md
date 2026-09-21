---
title: 算子正确性与性能测量：参考结果、误差和计时边界
type: recipe
tags:
  - kernels
  - evaluation
  - performance
  - numerics
sources:
  - raw/repositories/2026-09-21/triton/source/python/triton/runtime/autotuner.py
  - raw/repositories/2026-09-21/triton/source/python/tutorials/01-vector-add.py
  - raw/repositories/2026-09-21/triton/source/python/tutorials/02-fused-softmax.py
  - raw/repositories/2026-09-21/triton/source/python/tutorials/03-matrix-multiplication.py
  - raw/repositories/2026-09-21/triton/source/python/tutorials/05-layer-norm.py
  - raw/repositories/2026-09-21/triton/source/python/triton/testing.py
  - raw/repositories/2026-09-22/kernels/source/kernel-builder/src/init/templates/kernel_cuda/kernel.cu
  - raw/repositories/2026-09-22/kernels/source/kernel-builder/src/init/templates/torch-ext/torch_binding.cpp
updated: 2026-09-22
---

# 算子正确性与性能测量：参考结果、误差和计时边界

开发算子需要分别回答两件事：它是否在声明的输入范围内实现了目标语义，以及在相同条件下是否更快。能编译、一次随机输入接近参考值、某个 shape 延迟更低，各自只支持其中很有限的一部分。

本页依据 Triton 官方教程及固定版本 `python/triton/testing.py`、Hugging Face CUDA 初始化模板，整理可复用的验证方法。它承接 [张量接口](../fundamentals/operators/tensor-layout-and-kernel-contracts.md) 与 [计算模式](../fundamentals/operators/gpu-kernel-computation-patterns.md)；下面的用例设计和反例属于整理者分析，本轮没有运行 GPU 基准。

## 1. 先建立两个层次的参考

数学参考说明需要算什么：轴、转置、广播、mask、epsilon 和输出形状。执行参考则规定实际输入精度、累加精度、输出转换与允许误差。小张量可用较高精度计算定位公式错误；与框架算子比较时，还要核对它使用的精度策略和融合边界。

例如 LayerNorm 的总体方差分母 N，与无偏估计的 N−1 不同；矩阵乘后在 FP32 累加器内激活，与先转 FP16 再激活也不完全相同。不能放宽容限来掩盖语义不一致。量化还要区分编码误差与执行错误，见 [量化误差诊断](quantization-error-diagnosis.md)。

## 2. 用例围绕可能失败的机制选择

| 维度 | 有区分力的用例 | 检查什么 |
| --- | --- | --- |
| 工作块边界 | $B-1,B,B+1$，以及小于一个 block 的输入 | ceil grid、load/store mask 和漏写 |
| 归约维 | 奇数长度、长行、非 2 的幂 | 填充值、真实分母、归约与资源边界 |
| 地址 | 连续、转置、间隔切片；只测试声明支持的布局 | stride、重复 offset、错误连续假设 |
| 数值 | 全零、常量、正负抵消、大幅值与小幅值 | 累加精度、溢出、epsilon、近零误差 |
| 接口 | 空输入、不同 dtype/device、给定输出视图 | 明确处理或拒绝，不能静默错误 |
| 写入和重复调用 | 多次执行、输出预填特殊值；按接口测试别名 | 未初始化累加、漏写、竞争及原地覆盖 |

NaN/Inf、全部被屏蔽的 softmax 行等需要先定义预期，再测试；不是每个算子都必须支持。若只声明 contiguous 输入，就应检查非连续输入是否被拒绝，而不是替实现扩张支持范围。

Triton 向量加法教程只展示一个随机向量的差异，softmax 则测试 1823×781 的不规则形状。这些示例教会一种检查方式，不是完整覆盖证明。HF 模板中“输入连续但没有检查输出连续”的边界，是接口用例有必要的具体原因。

## 3. 容限怎样解释

一种常见逐元素判据为

$$|y_i-\widehat y_i|\le\mathrm{atol}+\mathrm{rtol}|y_i|,$$

其中 $y$ 是参考。atol 控制参考接近零时的绝对误差；rtol 随参考量级放宽。容限应由 dtype、输入范围、归约长度和算法允许的近似决定，不应在看过错误结果后不断放宽直到通过。

最大绝对误差能够暴露局部坏点；相对误差需说明近零分母如何处理；范数误差能看整体偏差，但会掩盖少数错误元素。另行检查 NaN/Inf 和未写位置，不让平均值把它们隐藏。

整理者反例：按 IEEE FP32 逐步舍入，$(2^{24}+1)-2^{24}=0$，而 $2^{24}+(1-2^{24})=1$。同一实数和因结合顺序产生差别；这说明树形归约与串行归约可能不同，并不说明任何大小的差异都能接受。Triton GEMM 的 FP32 累加、LayerNorm 的 FP32 中间统计和 softmax 的近似指数必须分别核对。

形状和数值测试之外，共享内存竞争、越界与同步错误还需要对应设备上的内存/竞争诊断。数值通过不证明不存在数据竞争；本轮未执行这类诊断。

## 4. 异步执行决定了计时边界

Triton 向量加法教程明确指出：主机函数返回张量时，GPU 工作可能仍未完成。直接在 Python 调用前后取时间，可能主要量到排队成本；而在 GPU event 的结束记录完成前读时间，也不能得到可信结果。

| 想测的问题 | 应包含的范围 |
| --- | --- |
| 单个设备 kernel | 预先准备输入输出，隔离该 kernel，使用正确 stream 上的 event 并等待完成 |
| 对外算子调用 | 包装、必要布局转换、分配和全部 kernel；明确哪些成本由设备计时覆盖，哪些需要主机墙钟 |
| 首次使用 | 模块加载、JIT、autotune、分配及首次执行 |
| 稳态执行 | 先完成编译和预热，再重复计时；保留相同输入与缓存策略 |
| 模型端到端 | 相同模型、请求与 batch/长度，包含真实调用链上的成本 |

多个 stream 协同的调用还需显式建立依赖并等待所有相关工作；只在一个流上夹两个 event，不会自动包含其他流里无依赖的工作。首调与稳态都可能有实际价值，应分别报告。

## 5. `do_bench` 实际测了什么

本次读取的 Triton `81a46fa0` 中，`do_bench` 的流程是：先调用 fn 并同步，再估计运行时间，据此将 warmup、rep 的毫秒预算换成次数；预热后，每次测量先调用驱动的 cache 清理，再在 fn 两侧记录设备 event，最后同步并汇总。默认汇总为 mean；显式 quantiles 则按调用者指定的分位数返回。

这意味着：

- `warmup=25, rep=100` 是时间预算，不能读成 25 次和 100 次。
- fn 里若含多个 kernel 或必要转换，它们都会进入相应设备时间；Python 包装及分配的全部成本不能简单等同于 event 时间。
- 清缓存发生在正式单次 event 区间之前。这个测量情境与连续复用热数据不同；也不能仅凭 helper 名字保证某次测量确实完全冷缓存。
- 教程变量 `min_ms/max_ms` 接收的是指定的 20%/80% 分位数，不是样本最小/最大值。

`do_bench_cudagraph` 则捕获多次 fn，重放 graph 并按调用数平均，固定版本中没有相同的逐次清缓存步骤。它用于降低主机提交开销，不能与普通调用的测量混作同一种口径；需确认目标场景实际使用 graph，以及 fn 的状态修改是否允许重复重放。


### 调优会重复执行，性能候选也需要语义检查

改变 tile、warp 数、流水深度或 split-K 后，要验证新的边界和归约路径，不能仅复用旧配置的正确性结果。Triton `Autotuner._bench` 比较耗时，没有自动与参考输出对照；每个候选先满足数值与支持范围，才有比较速度的意义。

对于原地更新、原子累加和会改变输入的 kernel，每次调优试跑都可能改变下一次调用的初始状态。明确调用语义后再选择重置或恢复：从零计算输出可以清零，累加到已有张量则应恢复原始值。否则测量的是不同计算，或正确候选因为输出被多次叠加而被误判。具体选择与 cache key 的关系见 [配置与自动调优](kernel-configuration-and-autotuning.md)。

对异步流水，额外覆盖短于预取深度、最后一个 tile 和多次环形复用；这些输入能区分启动、稳态与收尾是否正确。相关推演见 [缓冲复用与等待](gpu-async-copy-pipelines.md)。CPU 顺序模型能检查设计中的等待遗漏，但不能证明实际 GPU 没有内存竞争。

## 6. 性能数字怎样算和怎样判断

向量加法按有效数据量换算带宽：$\mathrm{GB/s}=3ns/(t\cdot10^9)$，其中 t 用秒。GEMM 常用 $\mathrm{TFLOP/s}=2MNK/(t\cdot10^{12})$，将一次乘加计为两次操作。这些是约定下的有效吞吐，不等于 profiler 测得的 HBM 字节数或所有执行指令数；缓存、重复加载和 padding 都会使两者不同。

比较基线与候选应保持 shape、dtype、布局、精度、设备、缓存和计时范围一致。报告延迟分布、软件版本及配置，而不只给最快一次。融合算子还要包含两边完成同一输出所需的全套工作；省略布局转换可能把成本移出统计范围。

若单 kernel 快 2 倍，但只占原调用链耗时的 20%，其余部分不变且没有额外成本，则整体加速最多为 $1/(0.8+0.2/2)\approx1.11$ 倍。这是教学推导，不是本项目实测。瓶颈方向的理论判断见 [roofline](../fundamentals/hardware/arithmetic-intensity-and-roofline.md)，实际性能仍需测量。

## 7. 本轮验证范围

已核对教程的地址、mask、FP32 累加、包装限制与 benchmark 工具实现；独立 CPU 教学计算覆盖地址映射、尾块覆盖、softmax/LayerNorm padding 反例、分块 GEMM 等价关系和 FP32 结合顺序差异。它们验证正文中的推理，不验证 CUDA/Triton 编译、原实现运行、GPU 数值或速度。

这组基础用于理解并验证真实实现；可继续沿 [AWQ 实现链](awq-implementation.md) 和 [反量化 kernel](weight-only-dequant-kernels.md) 检查输入到内核的完整对应关系。

## 来源身份

| 来源 | 版本或快照 | 核对范围 |
| --- | --- | --- |
| [Triton 官方教程与工具](https://github.com/triton-lang/triton/tree/81a46fa0c04526e5df55a018ecfab72ff922f592) | `81a46fa0c04526e5df55a018ecfab72ff922f592` | 正文标明具体文件与函数；未运行 GPU 教程。 |
| [Hugging Face kernels](https://github.com/huggingface/kernels/tree/5c2cf07f7625e1b8c5fb60bcfb073cd055581cbc) | `5c2cf07f7625e1b8c5fb60bcfb073cd055581cbc` | kernel-builder 初始化模板；模板不是完整输入验证实现。 |
