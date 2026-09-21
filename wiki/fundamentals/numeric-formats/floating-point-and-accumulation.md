---
title: 浮点表示与累加：范围、精度和舍入顺序
type: concept
tags:
  - floating-point
  - numerical-error
  - kernels
sources:
  - raw/articles/2026-09-22/floating-point/article.md
  - raw/repositories/2026-09-21/cs336-lectures/source/lecture_02.py
  - raw/repositories/2026-09-21/cs336-lectures/source/images/fp16.png
  - raw/repositories/2026-09-21/cs336-lectures/source/images/bf16.png
  - raw/repositories/2026-09-21/cs336-lectures/source/images/fp32.png
  - raw/repositories/2026-09-21/marlin/source/marlin/marlin_cuda_kernel.cu
updated: 2026-09-22
---

# 浮点表示与累加：范围、精度和舍入顺序

浮点数把有限位数分给指数与有效数字：指数决定覆盖的量级，尾数决定同一量级内的分辨率。因此“都是 16 bit”不表示 FP16 与 BF16 可以无条件互换；“输出 FP16”也没有说明矩阵乘用什么精度累加。

## 1. 位模式如何表示数值

对正常有限数，设符号位为 $s$，指数编码为 $E$、偏置为 $B$，尾数域整数为 $F$、宽度为 $p$：

$$x=(-1)^s2^{E-B}(1+F/2^p).$$

前面的 1 隐含在正常数编码中。全零指数用于零及非正规数，非正规数改为 $(-1)^s2^{1-B}(F/2^p)$；全一指数保留给无穷大和 NaN。下面的位宽结合 CS336 `tensors_memory` 及其三幅格式图，数值由编码公式计算。

| 格式 | 符号/指数/尾数位 | 偏置 | 最小正正常数 | 最大有限数 | $[1,2)$ 的相邻间距 |
| --- | --- | ---: | --- | --- | --- |
| FP16 | 1/5/10 | 15 | $2^{-14}$ | 65504 | $2^{-10}$ |
| BF16 | 1/8/7 | 127 | $2^{-126}$ | $(2-2^{-7})2^{127}$ | $2^{-7}$ |
| FP32 | 1/8/23 | 127 | $2^{-126}$ | $(2-2^{-23})2^{127}$ | $2^{-23}$ |

BF16 与 FP32 的指数范围相同，但尾数精度明显不同。FP16 的最小正非正规数为 $2^{-24}$；实际计算是否保留某类非正规数，还取决于指令和运行设置，不能只从存储格式保证。

正常数落在 $[2^e,2^{e+1})$ 时，相邻间距为 $2^{e-p}$。因此 FP16 在 1 附近能分辨约 0.001，在 1024 附近相邻值差 1。这是 [Marlin 反量化](../../implementation/marlin-batched-w4a16-gemm.md)用 `0x6400` 把整数嵌入 FP16 的原因：指数编码 25、尾数为 $q$，数值正好为 $1024+q$。

## 2. 舍入模式与中点

NVIDIA 浮点文档的 Operations and Accuracy、Rounding Modes 区分最近偶数舍入、朝零、朝正无穷与朝负无穷。最近偶数只在恰好等距时选择有效位末位为偶数的候选，不是把所有数舍到偶数。

教学例：FP16 的 1 与 $1+2^{-10}$ 之间，中点为 $1+2^{-11}$，最近偶数舍入得到 1。改变运算顺序，使中间值先落在这个中点，再进行下一次转换，可能改变最终结果。相对误差也不能用于接近零的元素单独判错，需要绝对容差兜底，见 [正确性与性能测量](../../implementation/kernel-correctness-and-benchmarking.md)。

## 3. FMA 和归约顺序改变了什么

记 $\operatorname{fl}$ 为一次指定格式的舍入。分开乘加计算

$$\operatorname{fl}(\operatorname{fl}(ab)+c),$$

FMA 计算

$$\operatorname{fl}(ab+c).$$

后者在乘积与加法之间不做那次舍入。它通常减少这一步的误差，但不保证与原来的两步实现逐位一致。NVIDIA 文档 The Fused Multiply-Add 与 Dot Product 的示例专门比较这一差异。

另一个教学反例：每次相加都舍入到 FP16 时，$(2048+1)+1$ 得 2048，而 $2048+(1+1)$ 得 2050。归约树、split-K 或融合改变加法顺序，数学实数等价并不保证浮点逐位等价。树归约也不是每个输入上都必然更准。

## 4. 存储、乘法、累加和写回要分别说明

Marlin 所选 `mma` 明确使用 `f32.f16.f16.f32`：FP16 乘法输入、FP32 累加器。输入已经丢掉的尾数不会由 FP32 累加恢复，但累加过程中通常能保留更多小贡献，最后再按输出类型转换。

检查一个量化 GEMM，至少区分：权重文件中的 INT4 码 → 反量化的 FP16/BF16 → MMA 乘法输入 → 累加器 → split-K 局部结果与最终写回。任一阶段的舍入都可能影响数值，不能只看 Python 输出张量 dtype。

同理，RMSNorm 中以 FP32 求平方和，再转回低精度，与全程低精度归约不同；残差是否先写回 half 也会改变后续统计。现成例子见 [RMSNorm 的两条残差流](../../implementation/rmsnorm-cuda-triton-kernels.md)。更低精度的指数与块尺度扩展见 [FP8 与 MX](fp8-and-mx-data-formats.md)。

这些是格式推导、源码核对与 CPU 教学例子；没有据此宣称某个 GPU 编译配置的逐位行为或性能。

## 来源身份

- [Floating Point and IEEE 754](https://docs.nvidia.com/cuda/floating-point/index.html)，页面版本 v13.4，2026-09-22 获取；本地保存 Floating Point、Dot Product 与 Rounding Modes 正文，快照标识 `floating-point-20260922-16c2d89a`。
- [Stanford CS336 lectures](https://github.com/stanford-cs336/lectures/tree/8b59b50730766695c2ffedd1a79c50cd09b9eb91)，`8b59b50730766695c2ffedd1a79c50cd09b9eb91`，`lecture_02.py:tensors_memory` 与 fp16/bf16/fp32 位域图。
- [Marlin](https://github.com/IST-DASLab/marlin/tree/1f25790bdd49fba53106164a24666dade68d7c90)，`1f25790bdd49fba53106164a24666dade68d7c90`，`marlin_cuda_kernel.cu:dequant` 与 `mma`。
