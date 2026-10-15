# GPU 算子：先确认计算与数据接口

用于理解算子代码、布局、并行分工和低精度执行。先定位输入输出及数据依赖，再沿需要的设备机制阅读，避免从优化名词开始遍历资料。

| 要回答的问题 | 正文入口 | 可补查的词 |
| --- | --- | --- |
| shape、stride 与线程访问怎样对应 | [张量布局与接口](/Users/xiao/Documents/Projects/quantwiki/wiki/fundamentals/operators/tensor-layout-and-kernel-contracts.md) | layout、stride、tile、地址 |
| 逐元素、归约和矩阵乘怎样分工 | [计算模式](/Users/xiao/Documents/Projects/quantwiki/wiki/fundamentals/operators/gpu-kernel-computation-patterns.md) | reduction、program、CTA、复用 |
| CUDA 如何编译接入，Triton 如何表达计算 | [CUDA 接入](/Users/xiao/Documents/Projects/quantwiki/wiki/implementation/cuda-operator-build-and-integration.md)、[Triton 语义](/Users/xiao/Documents/Projects/quantwiki/wiki/fundamentals/operators/triton-language-and-compilation.md) | JIT、mask、广播、launch |
| 低比特权重与尺度如何参与矩阵乘 | [反量化内核](/Users/xiao/Documents/Projects/quantwiki/wiki/implementation/weight-only-dequant-kernels.md)、[量化乘法与尺度](/Users/xiao/Documents/Projects/quantwiki/wiki/implementation/quantized-matmul-scaling-execution.md) | packing、zero、GEMV、W4A16 |
| 具体硬件的流水与块缩放在哪里 | [异步流水](/Users/xiao/Documents/Projects/quantwiki/wiki/implementation/gpu-async-copy-pipelines.md)、[块缩放执行](/Users/xiao/Documents/Projects/quantwiki/wiki/implementation/block-scaled-matmul-execution.md) | TMA、WGMMA、TMEM、FP4、FP8 |

具体算子优先在 `implementation/` 查 `rmsnorm`、`gated-activation`、`attention` 等名称，语义前置查 `fundamentals/operators/`。正确性和计时问题转评测 reference；要求实际实现或优化时交统一入口选择 `wiki-kernel-writing`，查询不自行启动开发流程。
