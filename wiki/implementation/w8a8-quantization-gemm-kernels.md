---
title: W8A8 算子：激活量化、整数 GEMM 与输出校正
type: implementation
tags:
  - activation-quantization
  - kernels
  - matmul
  - deployment
sources:
  - raw/repositories/2026-09-21/sglang/source/python/sglang/srt/layers/quantization/int8_kernel.py
  - raw/repositories/2026-09-21/sglang/source/sgl-kernel/csrc/gemm/int8_gemm_kernel.cu
  - raw/repositories/2026-09-21/sglang/source/sgl-kernel/csrc/cutlass_extensions/gemm/gemm_with_epilogue_visitor.h
  - raw/repositories/2026-09-21/sglang/source/sgl-kernel/csrc/cutlass_extensions/epilogue/epilogue_per_row_per_col_scale.h
  - raw/repositories/2026-09-21/sglang/source/sgl-kernel/CMakeLists.txt
  - raw/repositories/2026-09-21/vllm/source/csrc/libtorch_stable/quantization/w8a8/int8/scaled_quant.cu
  - raw/repositories/2026-09-21/vllm/source/csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_c2x.cuh
  - raw/repositories/2026-09-21/vllm/source/csrc/libtorch_stable/cutlass_extensions/epilogue/scaled_mm_epilogues_c2x.hpp
  - raw/repositories/2026-09-21/vllm/source/CMakeLists.txt
  - raw/repositories/2026-09-21/triton/source/third_party/nvidia/language/cuda/libdevice.py
  - raw/articles/2026-09-22/cuda-libdevice-round/article.md
  - https://github.com/NVIDIA/cutlass/blob/57e3cfb47a2d9e0d46eb6335c3dc411498efa198/include/cutlass/gemm/kernel/default_gemm.h
  - https://github.com/NVIDIA/cutlass/blob/57e3cfb47a2d9e0d46eb6335c3dc411498efa198/include/cutlass/gemm/threadblock/default_mma.h
  - https://github.com/NVIDIA/cutlass/blob/57e3cfb47a2d9e0d46eb6335c3dc411498efa198/include/cutlass/gemm/threadblock/mma_multistage.h
updated: 2026-09-22
---

# W8A8 算子：激活量化、整数 GEMM 与输出校正

W8A8 Linear 的运行成本由至少两部分组成：把浮点激活变成整数与尺度，再执行整数 GEMM 和输出变换。SmoothQuant 的离线平滑改变输入分布，却不会自动消除动态量化的归约、读写与启动成本。本页承接 [SmoothQuant 实现](smoothquant-implementation.md)，展开固定版本的 SGLang 与 vLLM Dense CUDA 路径；[QoQ](qserve-implementation.md) 复用其中的激活量化，但权重恢复和零点位置不同。

## 1. 从数学契约到中间张量

采用推理方向 $X\in\mathbb R^{M\times K}$、$W\in\mathbb R^{K\times N}$，权重按输出通道对称量化。激活可以对称，也可以有逐行零点：

$$
X_{mk}\approx s_{x,m}(q_{x,mk}-z_{x,m}),\qquad
W_{kn}\approx s_{w,n}q_{w,kn}.
$$

令 $C_{mn}=\sum_kq_{x,mk}q_{w,kn}$、$a_n=\sum_kq_{w,kn}$，则量化后矩阵的乘积满足

$$
Y_{mn}=s_{x,m}s_{w,n}(C_{mn}-z_{x,m}a_n)+b_n.
$$

这是量化表示的代数展开；与原始浮点 $XW$ 的差异仍包含量化误差。尺度在 $K$ 上恒定，所以整数主循环可以结束后再缩放。沿 $K$ 的任意分组浮点尺度不能直接套这个 epilogue。

| 阶段 | 主要存储 | 执行职责 |
| --- | --- | --- |
| 激活量化 | 浮点 $X$ → INT8 $q_x$、逐行 FP32 尺度；非对称另有 INT32 零点 | 归约范围、舍入、饱和 |
| GEMM | A row-major；权重视为 column-major B；INT32 accumulator | 沿 $K$ 做整数乘加 |
| epilogue | 行尺度、列尺度、可选零点校正和 bias | INT32 转 FP32、广播运算，最终写 FP16/BF16 |

这是逻辑形状，不保证任意 stride、架构或尾块都能运行。SGLang 底层要求 $K\bmod16=0,N\bmod8=0$，还要通过 CUTLASS 的 `can_implement`；详见实现页的 host 条件。

## 2. 激活量化也有设备级数值契约

### SGLang：一行一个 Triton program

`int8_kernel.py:per_token_quant_int8` 取 `BLOCK=next_power_of_2(K)`，mask 掉补齐元素并填零，以 FP32 计算

$$
a_m=\max(\max_k|X_{mk}|,10^{-10}),\qquad
s_{x,m}=a_m/127,\qquad
q_{x,mk}=\operatorname{round}(X_{mk}\,127/a_m).
$$

warp 数为 `min(max(BLOCK // 256, 1), 8)`，`num_stages=1`。这是行归约程序的配置，不能与 GEMM 的 shared-memory stages 混为一谈。源码逻辑上一次加载行向量后用于归约与量化；是否发生寄存器溢出、最终生成几次访存，必须看编译产物。

`libdevice.round` 的 FP32 入口映射到 `__nv_roundf`；NVIDIA 的固定 CUDA 12.9.1 文档规定半整数向远离零方向舍入。普通 W8A8 的尺度输出为 FP32，全零行因此输出零码与正尺度 $10^{-10}/127$。QoQ 指定 FP16 尺度，这个极小值会在存储时下溢为零，不能把“正下限”扩大成所有输出 dtype 都保证正尺度。

可选 `cal_sum` 同时把**原始浮点输入**的行和写出，输出 dtype 跟随输入；它不是 $\sum q_x$。这直接影响 QoQ 逐通道零点校正的精确性，见其实现页。

### vLLM：CTA 归约与第二遍量化

`scaled_quant.cu` 的动态对称路径按 token 启动 CTA，线程数为 `min(K,256)`；先用向量化读取取得每线程 absmax，再用 CUB BlockReduce 合并、写共享 absmax 和输出尺度，经 CTA barrier 后第二遍读取并量化。这里的两遍是源码明确的数据遍历，不能仅凭带宽估算认为它一定比 Triton 慢。

CUDA 浮点到 INT8 helper 使用 `cvt.rni.sat.s8.f32`，即最近偶数舍入并饱和。全零行单独令 inverse scale 为零，输出 $q_x=0,s_x=0$。因此两框架的量化器不能默认逐位等价：

| 输入或条件 | vLLM 本路径 | SGLang 本路径 |
| --- | --- | --- |
| 已缩放值 $0.5$ | 0 | 1 |
| 已缩放值 $-0.5$ | 0 | -1 |
| 全零行、FP32 尺度 | 0 | $10^{-10}/127$ |
| 静态对称尺度 | 输入除以 scale | 此处讨论的接口是动态逐行量化 |

这些差异可能被最终 FP16/BF16 输出舍入隐藏，也可能在长 $K$ 点积中积累；应先比 $q_x,s_x$，再比 GEMM 输出。

### 非对称：舍入顺序和退化范围

同一 vLLM 文件的静态非对称路径先把 $X/s$ 最近偶数舍入成 INT32，**再加整数零点**，最后饱和到 INT8。不要随手改成 $\operatorname{roundEven}(X/s+z)$：$X/s=0.5,z=1$ 时，前者得到 1，后者得到 2。

动态非对称路径归约 min/max，计算 $s=(x_{\max}-x_{\min})/255$ 及 $z=\operatorname{nearbyint}(-128-x_{\min}/s)$。本次读到的设备函数没有 $x_{\max}=x_{\min}$ 的保护分支。此处只定位代码边界，不宣称已运行出某个 NaN/整数结果；常量行需要结合调用条件、编译后行为与数值测试核验，不能借用动态对称的零行处理来证明它安全。

## 3. SGLang SM80 的主循环：沿真正的模板依赖下钻

`int8_gemm_kernel.cu` 的 SM80 分支给出 INT8 A/B、INT32 accumulator、MMA shape $16\times8\times32$。例如 $M\le16,N\le4096$ 分派到 CTA $16\times64\times128$、warp $16\times64\times64$、6 stages。CTA 的 $K$ 是 warp 的两倍，所以同一输出 tile 中有两个 warp 的 $K$ 分区；这与多个 CTA 在全局内存执行 split-K 不是同一件事。

调用链是：

`DefaultGemm::Mma → DefaultMma::ThreadblockMma → MmaMultistage`，外层 `GemmWithEpilogueVisitor::run_kernel_` 构造带问题边界的 A/B iterator，清空 accumulator，执行 $\lceil K/B_K\rceil$ 次 CTA 归约，再交给 epilogue。

这里使用 SGLang `sgl-kernel/CMakeLists.txt` 指定的 CUTLASS commit `57e3cfb47a2d9e0d46eb6335c3dc411498efa198`。2026-09-22 定向取得并阅读三份模板文件，来源采用完整 commit 外链；未将另一 commit 的本地 CUTLASS 快照冒充构建依赖，也未将这些摘取文件表述为完整依赖收录。

`MmaMultistage` 的阶段关系如下：

1. **预热**：`prologue` 提交 `stages-1` 个 global→shared tile。A/B 通过 predicated iterator 和 `cp_async_zfill` 加载；超出有效问题的访问按该路径补零。每个 stage 提交一次 copy group。
2. **等待与首批寄存器**：`gmem_wait` 执行 `cp_async_wait<stages-2>` 和 `__syncthreads`，再加载首个 warp fragment。等待负责拷贝完成，CTA barrier 负责生产者和消费者之间的可见性。
3. **稳态**：`mac_loop_iter` 预读下一片 A/B 到 `(warp_mma_k+1)%2` 寄存器缓冲，对当前片做 warp MMA，同时分摊下一 global tile 的拷贝。在倒数第二个 warp K 步提交、等待并推进 shared ring；最后一步准备下一轮的首片。
4. **收尾**：global iterator 的有效 mask 被清除后，已经预取的 tile 仍需消费；`gemm_iters` 的循环下界包含流水余量，末尾再次 fence、wait 0 和同步，之后才能复用 shared memory。

由此能解释为何 `stages-2` 不是“只加载这么多 tile”，以及为什么不能在 $K$ 主循环结束时直接复用共享内存。通用同步语义见 [异步拷贝与多级流水](gpu-async-copy-pipelines.md)。

本页追到 threadblock mainloop 和 warp-MMA 调用；没有逐 lane 展开 CUTLASS 全部 iterator，也没有反汇编验证最终机器指令。SM75 两级管线、SM90 TMA/WGMMA 和其他架构的设备主循环不能由这个 SM80 实例代替。

## 4. 输出尺度与 bias 在哪里融合

### SGLang CUTLASS 2 路径

`epilogue_per_row_per_col_scale.h` 的 visitor 在 `begin_epilogue` 读取列尺度和可选 bias，在 `begin_row` 读取行尺度，`visit` 把 INT32 accumulator 转成 FP32 后计算

$$
t=\operatorname{FP32}(C)\,(s_ws_x),\qquad
Y=\operatorname{cast}_{\rm out}(t+b).
$$

最后由输出 iterator 存储。行、列元数据在 fragment 消费时广播，不生成一张完整 $M\times N$ 尺度矩阵，也不先写出完整 FP32 GEMM 再另起算子缩放。

### SGLang SM90 与 vLLM CUTLASS 2 路径

SGLang `int8_gemm_kernel.cu` 的 SM90 EVT 先做 $s_w\operatorname{FP32}(C)$，再乘 $s_x$；带 bias 时第二级用 multiply-add。vLLM `scaled_mm_epilogues_c2x.hpp` 的 `ScaledEpilogue`／`ScaledEpilogueBias` 也按先列尺度、后行尺度构造 visitor 树。

这些源码表达的是同一实数公式，但 FP32 的乘法结合次序和 multiply-add 融合会改变舍入。后端比较不能要求仅凭“INT32 累加、FP32 epilogue”便逐位相同，也不能从一个 CPU 模拟证明实际编译器的 FMA 行为。

vLLM `scaled_mm_c2x.cuh` 还明确选用 INT8 的 `OpMultiplyAddSaturate`、`ThreadblockSwizzleStreamK`；调用参数中的 `kGemmSplitKParallel` 与 batch count 1 不能单独证明启动了多少个 $K$ 分区。其构建默认 CUTLASS tag 为 v4.4.2，且允许外部源目录覆盖。本页核对其 wrapper 与 visitor 源码，没有把上一节 SGLang 依赖的主循环等同于 vLLM 的完整执行。

## 5. vLLM 的激活零点校正不需要完整外积

`ScaledEpilogueBiasAzp` 消费已经折叠的静态 $z_xa_n$；`ScaledEpilogueBiasAzpToken` 消费逐 token 的 $z_{x,m}$ 与预计算 $a_n$，在 visitor 内相乘。两者都先在 INT32 域执行 $C-z_xa$，再进入 FP32 的权重尺度、激活尺度和 bias 变换。

因此动态路径只额外保存 $M$ 个零点与 $N$ 个权重列和，不需要物化 $M\times N$ 外积。INT32 accumulator、列和、校正乘法仍有范围条件；“存在饱和 MMA”不等于所有中间整数运算都不会溢出。应分别检查原始累加和校正项，不能只看抵消后的理论结果是否小。

与 [QoQ 逐通道路径](qserve-implementation.md) 对照时尤其要分清：这里校正的是**激活零点 × 整数权重列和**；QoQ 那条路径校正的是**权重零点尺度 × 原始浮点激活行和**。前者是上述整数表示的精确代数，后者还含激活量化误差差项。

## 6. 怎样做有解释力的正确性与性能检查

先固定 quantizer，再单独检查整数矩阵乘和 epilogue，最后接回模型：

- 同时比较 $q_x,s_x,z_x$，包括半整数、全零行、常量行、接近范围上限以及不同 $K$。
- 用 INT64 参考点积检查 INT32 范围，再用目标精度模拟输出缩放；分别有/无 bias、静态/动态零点。
- 按 host 条件和每种 tile 分派覆盖形状；不能用一个大 prefill shape 证明 decode 或所有尾块安全。
- 分开测 activation quant、GEMM、相邻 Norm/激活与整体 Linear。已有 [RMSNorm](rmsnorm-cuda-triton-kernels.md) 和 [门控激活](gated-activation-cuda-triton-kernels.md) 讲解普通算子，但本页不据此宣称这条调用链已把它们与 INT8 量化融合。

[CPU 教学检查](../assets/w8a8-quantization-gemm-kernels/check_contracts.py)及[固定结果](../assets/w8a8-quantization-gemm-kernels/checks-2026-09-22.json)覆盖舍入、零行尺度、校正代数、浮点结合顺序，并与 QoQ 输出坐标、Marlin repack/部分和作对照。运行 `python3 wiki/assets/w8a8-quantization-gemm-kernels/check_contracts.py`，需要 NumPy；默认打印，显式 `--output` 才写记录。

上述是源码核对与 CPU 教学参考，未编译或执行上游 GPU 内核，未完成模型加载、实际量化精度或性能验证。

激活量化、GEMM 与完整 Linear 怎样分别计时、如何解释小 M 与大 M 的瓶颈变化，见 [量化 GEMM 性能分析案例](gpu-kernel-performance-analysis.md)。其采集 factory 不改变这里的格式与数值契约。

## 来源身份

| 来源 | 固定版本与核对范围 |
| --- | --- |
| [SGLang](https://github.com/sgl-project/sglang/tree/2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc) | 动态 INT8 量化、Dense CUTLASS host/device wrapper、自定义 epilogue |
| [vLLM](https://github.com/vllm-project/vllm/tree/568afb3a13806beb53bb2e6bd518269357b237c0) | CUDA INT8 量化、CUTLASS 2 wrapper 与 epilogue |
| [CUTLASS](https://github.com/NVIDIA/cutlass/tree/57e3cfb47a2d9e0d46eb6335c3dc411498efa198) | SGLang 指定依赖；仅定向读 DefaultGemm、DefaultMma、MmaMultistage，2026-09-22 获取 |
| [Triton](https://github.com/triton-lang/triton/tree/81a46fa0c04526e5df55a018ecfab72ff922f592) | NVIDIA libdevice round 符号映射；不代表已核对运行环境安装版本 |
| [NVIDIA libdevice __nv_roundf](https://docs.nvidia.com/cuda/archive/12.9.1/libdevice-users-guide/__nv_roundf.html) | CUDA 12.9.1 的函数定义，2026-09-22 定向摘录 |
