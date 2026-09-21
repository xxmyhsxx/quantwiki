---
title: RMSNorm 的 CUDA 与 Triton 实现：行归约、残差与舍入
type: implementation
tags:
  - kernels
  - cuda
  - triton
  - reduction
  - numerics
sources:
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/models/llama.py
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/layers/layernorm.py
  - raw/repositories/2026-09-21/vllm/source/vllm/kernels/vllm_c.py
  - raw/repositories/2026-09-21/vllm/source/vllm/ir/ops/layernorm.py
  - raw/repositories/2026-09-21/vllm/source/csrc/libtorch_stable/layernorm_kernels.cu
  - raw/repositories/2026-09-21/vllm/source/csrc/libtorch_stable/type_convert.cuh
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/layers/batch_invariant.py
  - raw/repositories/2026-09-21/sglang/source/python/sglang/srt/models/grok.py
  - raw/repositories/2026-09-21/sglang/source/python/sglang/srt/layers/elementwise.py
  - raw/repositories/2026-09-21/sglang/source/python/sglang/srt/layers/layernorm.py
  - raw/repositories/2026-09-21/sglang/source/python/sglang/jit_kernel/norm.py
  - raw/repositories/2026-09-21/sglang/source/python/sglang/jit_kernel/csrc/elementwise/fused_add_rmsnorm.cuh
updated: 2026-09-22
---

# RMSNorm 的 CUDA 与 Triton 实现：行归约、残差与舍入

RMSNorm 的每个输出依赖整行平方和。它适合用来理解三件事：行内数据怎样分给线程，局部统计怎样合并，以及融合怎样改变中间值的保存位置和舍入。这里研读 vLLM 与 SGLang 的现成实现；各分支的调用条件分别说明，不把仓库中的所有 RMSNorm 当成同一条默认路径。

先读[张量布局与接口](../fundamentals/operators/tensor-layout-and-kernel-contracts.md)与[归约计算模式](../fundamentals/operators/gpu-kernel-computation-patterns.md)。本页的地址例子、流量估算和舍入反例是整理者教学推导；没有运行这些 GPU kernel。

## 1. 先区分数学输出与残差状态

对输入 $X\in\mathbb R^{M\times D}$、权重 $w\in\mathbb R^D$，普通 RMSNorm 为

$$s_m=\frac1D\sum_{j=0}^{D-1}x_{mj}^2,\qquad r_m=(s_m+\epsilon)^{-1/2},\qquad y_{mj}=x_{mj}r_mw_j.$$

这里不减均值，源码中的 `variance` 实际保存二阶矩或其平方和，不能按 LayerNorm 的中心化方差理解。每行只有一个 $r_m$，但必须先读完该行才能得到它。除数是有效宽度 D，不是补齐后的逻辑块宽度，也不是参与归约的线程数。

vLLM `models/llama.py:LlamaDecoderLayer.forward` 将残差独立传递：首次没有 residual 时保存当前 hidden states 并执行普通 norm；之后的 norm 将上一子层输出与 residual 相加，返回归一化输出和更新后的 residual。实数域可写成

$$u=x+\mathrm{residual},\qquad y=\mathrm{RMSNorm}(u,w),\qquad \mathrm{residual}_{out}=u.$$

因此 fused add RMSNorm 有两个有意义的输出，不能只验证 y。vLLM 的 C 实现和下述 SGLang JIT 实现都会改写 input 与 residual；重复试跑必须恢复两路原始输入，否则下一次计算的问题已经改变。

## 2. 沿框架入口确认正在分析哪一条路径

| 已核对入口 | 具体分支 | 本页解释范围 |
| --- | --- | --- |
| vLLM `layers/layernorm.py:RMSNorm` | 常规路径调用 IR op；`kernels/vllm_c.py` 注册了 `vllm_c` 实现，要求无 variance-size override，且有权重时与输入 dtype 相同 | C 实现被选择后的 CUDA 行归约；不宣称 IR 永远选择它 |
| vLLM `RMSNorm.forward_cuda` 的 batch-invariant 分支 | 无 residual 时调用 `batch_invariant.py:_rms_norm_kernel`；有 residual 时仍转交 `ops.fused_add_rms_norm` | 固定 1024 列分段的 Triton 路径，不能把含 residual 的调用也说成这个 Triton kernel |
| SGLang `srt/layers/elementwise.py:fused_rmsnorm` | `models/grok.py` 的前向路径有直接调用；`fused_dual_residual_rmsnorm` 还融合两次 norm 与中间残差 | 模型专用 Triton 例子，不是所有 `RMSNorm.forward_cuda` 的默认实现 |
| SGLang `srt/layers/layernorm.py:RMSNorm.forward_cuda` | 普通路径使用导入的 `sgl_kernel` norm；带 residual 且要求 `cast_x_before_out_mul` 时，在 dtype、宽度等条件满足后调用 JIT fused add RMSNorm，否则回退 native | 深入该条件分支的 CUDA 归约和数值契约，不把它替代为普通 `sgl_kernel` 的内部实现 |

SGLang 的该 JIT 分支要求 FP16/BF16、权重 dtype 匹配，Python 宽度筛选为正数、16 的倍数且不超过 8192；C++ 包装继续要求二维连续 input/residual、匹配 dtype/device 和向量宽度整除。框架条件、包装条件与设备地址计算需要一起看。

## 3. CUDA：先算线程局部和，再合并整行

### vLLM 的每行一个 block

`csrc/libtorch_stable/layernorm_kernels.cu:rms_norm_kernel` 用 `blockIdx.x` 选择行。每个线程读取自己的若干向量，转成 FP32 后累加平方和；`cub::BlockReduce` 合并线程局部和。线程 0 计算倒平方根并写入 shared 标量 `s_variance`，所有线程经过 `__syncthreads()` 后才能读它并生成输出。

这里 shared 保存的是统计量，不是整行输入。设备程序先遍历输入求统计，再读取输入与权重写输出；减少 shared 用量并不等于输入只读一次。主机按隐藏维、dtype 选择向量宽度，并根据 token 数限制 block 大小。源码的“小 block 增加并发”是配置意图，不是本地测量结论。

对 2D 输入，行起点使用 `input_stride_d2`，最后一维按连续元素访问，输出按紧凑行存放；3D/4D 分支再拆分 batch/head 等坐标。包装会在输入最后一维不连续时做连续化。不能仅看到支持行 stride，就推断支持任意最后一维切片。

### SGLang 的显式两级归约

`jit_kernel/csrc/elementwise/fused_add_rmsnorm.cuh:fused_add_rmsnorm_reg_kernel` 更直接展示了协作过程。一个 block 处理一行，每个有效线程持有一组 input、residual、weight 的向量，在寄存器中相加并计算 FP32 局部平方和：

```text
每线程若干元素的平方和
    → 每个 32-thread tile 内 cooperative_groups::reduce
    → 各 warp 的 lane 0 写一个 shared 部分和
    → block 屏障
    → 首 warp 读部分和，其余 lane 补 0，再归约并计算 r
    → block 屏障
    → 各线程读 r，对自己的元素归一化并写回
```

第一道屏障保证首 warp 能读到所有部分和；第二道保证其他 warp 能读到计算完成的 r。warp 内归约解决不了跨 warp 的发布与读取顺序。

该实现把线程数向上取到 32 的倍数。最后几个线程可能没有输入向量，它们将局部和保持为零，**仍然参加归约与两道屏障**，只跳过数据读写。不能把 `threadIdx.x >= vec_hidden_size` 改成提前 return 后，又假定全 block 的 collective 仍然成立。

例如在每线程处理 8 个 half 的分支，D=784 需要 98 个有效线程，实际 block 为 128 个线程。最后 30 个线程贡献零；四个 warp 的和再合并为整行平方和，分母仍是 784。归约缓冲仅保存 warp 统计量，输入向量跨归约保留在寄存器变量中；它减少重新读取，但会增加活跃状态，是否 spill 要看编译结果。

## 4. Triton：逻辑张量与物理线程分开理解

`tl.program_id(0)` 在下面两条路径中选一行，`tl.arange(0,B)` 构造 B 个逻辑列。B 不等于 CUDA 线程数；`tl.sum` 描述逻辑归约，元素在线程间的分配和必要通信由编译器生成。没有显式 `__syncthreads()`，不表示归约不需要协作；也不能把 `tl.sum` 理解成跨不同 program 的全局归约。

| 策略 | vLLM `_rms_norm_kernel` | SGLang `fused_rmsnorm_kernel` |
| --- | --- | --- |
| program 工作 | 一整行，但每次只处理 B=1024 列 | 一整行，B 取不小于 D 的最小 2 次幂 |
| 统计 | 循环各段，段内 `tl.sum` 后累加为行平方和 | 一次 `tl.sum(a*a)` |
| 输出 | 第二个循环重新加载各段输入，再乘权重 | 复用逻辑张量 a，加载权重后输出 |
| 尾部 | 每段按实际列号 mask，失效平方和贡献 0 | 补齐列加载 0，最终 store mask |
| 权衡 | 保持固定逻辑分段，增加一次输入读取 | 较少逻辑读取，长行产生更大的活跃状态 |

vLLM 的 B 固定为 1024，不随 batch 大小改归约分段。D=2500 时，三段有效列数分别为 1024、1024、452，平方和相加后除以 2500。SGLang 同宽度使用 B=4096，1596 个填充位置不参与有效结果。这是工作量与归约结构的对照，不能由源码断言哪条更快，也不能据 batch-invariant 命名声称跨 GPU、编译器或整个模型逐位一致。

SGLang 这条 Triton 实现按 `pid*hidden_dim+offset` 寻址，未传入 stride，因此要求行紧凑、列连续。包装只断言二维，不等于已经检查全部布局前提；`inplace=True` 允许 output=x，也不能据此扩展到与 weight 重叠或任意重叠视图。

## 5. 融合仍要保留数值步骤

设 $R_h$ 表示转成输入的低精度 dtype，F 表示 FP32 算术。实数域相同的残差加法，可以有两条不同的统计路径：

$$u_h=R_h(F(x)+F(r)),\quad s_A=\frac1D\sum F(u_h)^2;\qquad
u_f=F(x)+F(r),\quad s_B=\frac1D\sum u_f^2.$$

vLLM fused CUDA 路径先把残差和保存在 `scalar_t` 或 `_f16Vec`，然后转 FP32 求平方和；向量加法细节见 `type_convert.cuh:operator+=/sum_squares`。SGLang 上述 JIT kernel 先用 FP32 的 `inp_res` 求平方和，另将低精度副本写回 residual；`cast_x_before_out_mul=True` 还保留 FP32 和用于归一化，在乘权重前明确转一次低精度。vLLM IR 的 native 参考也使用 FP32 残差和，并在权重乘法前转换到 weight dtype。参考公式、C 实现和 native 的舍入阶段需要分别核对。

教学反例：half 输入 x=1、r=$2^{-11}$，精确和是 1.00048828125，按 half 最近偶数舍入为 1。因此第一种统计使用 1，第二种使用约 1.0009768009 的平方值；之后即使输出 dtype 相同，也不能要求所有输入逐位一致。这个例子只验证舍入位置能产生差别，不是两个框架的 GPU 对比实验。

普通 RMSNorm 也存在“FP32 归一化后直接乘 FP32 权重再转换”与“先转换归一化值再乘权重”的区别。不能只看到两边都有 FP32 平方和，就认为数值契约完全相同。

SGLang `fused_dual_residual_rmsnorm_kernel` 进一步计算 $z=r+\mathrm{RMSNorm}_1(x)$，同时输出 z 与 $\mathrm{RMSNorm}_2(z)$。源码先将第一次 norm 结果转到 residual dtype，再相加、保存 mid、转 FP32 做第二次统计。它融合了两次行归约，仍保留必要的低精度中间语义；不能改写成一次 norm，也不能把 mid 当成可丢弃临时值。

## 6. 怎样判断这类优化是否有用

忽略缓存与权重复用，普通 norm 的“一次读取 x、一次读取 w、一次写 y”需要约 $3MDe$ 字节，e 为元素字节数；两遍读 x 则约 $4MDe$。这是源码层面理想数据流估算，实际 HBM 流量还受缓存、spill 和布局转换影响。片上保留输入节省一次逻辑读取，却可能降低驻留能力，选择时要结合[资源与工作划分](kernel-configuration-and-autotuning.md)。

有区分力的检查包括：有效宽度与 padding 分母、1024 分段边界、最后一个不足 warp 的向量批次、stride 前提、残差两路写回，以及舍入临界值。对真实实现还要检查 collective 参与和 shared 访问，并分别量包装调用与设备 kernel。方法见[正确性与性能测量](kernel-correctness-and-benchmarking.md)。

本页已用 CPU 教学模型核对覆盖、归约中性值与舍入反例；没有编译 CUDA/Triton、运行模型、执行 GPU race 检查或测性能。PDL、全部 norm 后端、所有 batch-invariant 保证和分布式融合不在本页展开范围。

## 来源身份

| 来源 | 固定版本 | 核对范围 |
| --- | --- | --- |
| [vLLM](https://github.com/vllm-project/vllm/tree/568afb3a13806beb53bb2e6bd518269357b237c0) | `568afb3a13806beb53bb2e6bd518269357b237c0` | Llama/RMSNorm 入口、IR 与 C 注册、CUDA 普通及残差 norm、Triton batch-invariant 无残差路径；具体文件与函数见正文。 |
| [SGLang](https://github.com/sgl-project/sglang/tree/2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc) | `2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc` | Grok 调用、Triton 普通及双 norm、RMSNorm 的条件分派、JIT 残差 norm 的包装和设备函数。 |
