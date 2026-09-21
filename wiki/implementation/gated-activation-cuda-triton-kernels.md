---
title: 门控激活的 CUDA 与 Triton 实现：工作映射、向量化与融合
type: implementation
tags:
  - kernels
  - cuda
  - triton
  - data-layout
  - fusion
sources:
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/models/llama.py
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/layers/activation.py
  - raw/repositories/2026-09-21/vllm/source/csrc/libtorch_stable/activation_kernels.cu
  - raw/repositories/2026-09-21/sglang/source/python/sglang/srt/models/llama.py
  - raw/repositories/2026-09-21/sglang/source/python/sglang/srt/layers/activation.py
  - raw/repositories/2026-09-21/sglang/source/python/sglang/jit_kernel/activation.py
  - raw/repositories/2026-09-21/sglang/source/python/sglang/jit_kernel/csrc/elementwise/activation.cuh
  - raw/repositories/2026-09-21/sglang/source/python/sglang/jit_kernel/include/sgl_kernel/utils.cuh
updated: 2026-09-22
---

# 门控激活的 CUDA 与 Triton 实现：工作映射、向量化与融合

门控激活的每个输出只依赖同一位置的 gate 和 up，没有整行归约，因此可以沿列自由切分。vLLM 与 SGLang 的现成实现展示了三种分工：一行一个 CUDA block、将向量块展平成全局线程任务，以及一个 Triton program 处理一行的一段。它们解决同类数据依赖，但具体数值变体和接口条件必须分别看。

本页衔接[张量布局与接口](../fundamentals/operators/tensor-layout-and-kernel-contracts.md)，与[需要行归约的 RMSNorm](rmsnorm-cuda-triton-kernels.md)对照。地址例子与流量估算为教学推导，不是 GPU 测量结果。

## 1. 从模型中的 MLP 看输入布局

两库 `models/llama.py:LlamaMLP` 都按 `gate_up_proj → SiluAndMul → down_proj` 组织普通 SwiGLU MLP。合并的 gate/up 投影得到最后一维宽度为 2D 的输入，前半为 g，后半为 u：

$$X_m=[g_{m0},\ldots,g_{m,D-1},u_{m0},\ldots,u_{m,D-1}],\qquad
y_{mj}=\mathrm{SiLU}(g_{mj})u_{mj}=\frac{g_{mj}}{1+\exp(-g_{mj})}u_{mj}.$$

这是两段连续向量，不是 `[g0,u0,g1,u1,...]` 的交错布局。连续二维输入的两路元素偏移为 $2mD+j$ 与 $2mD+D+j$，输出为 $mD+j$。若以错误的交错解释加载，同样的 shape 仍会算错。

融合发生在激活与乘法之间：投影产物仍是输入，down projection 仍是后续算子。不能把 `SiluAndMul` 称为整个 MLP 已经融合。

## 2. CUDA 的两种输出拥有者

### vLLM：一个 block 负责一行

`layers/activation.py:SiluAndMul.forward_cuda` 分配宽度 D 的输出，调用 `_C.silu_and_mul`。`csrc/libtorch_stable/activation_kernels.cu:act_and_mul_kernel` 用 `blockIdx.x` 选择行，标量分支让线程 t 处理

$$j=t+kT,\quad j<D,$$

其中 T 是 block 的线程数。不同线程写不同列，没有 shared 中间量或 block 级归约，也不需要为了独立输出插入 `__syncthreads()`。

向量分支改为每线程一次加载一组连续元素，分别读取 gate/up，计算后向量写出。启动宏根据架构、构建条件和 token 数选择候选向量字节宽度，再检查 D 能否被每向量元素数整除；不整除则走标量循环。**最后一个 block 没有超出行数，与最后一行的列能否整除向量宽度，是不同条件。**

向量化减少地址/访存指令组织开销，不减少每个输出必需的两路输入。相邻线程即使每次只加载一个元素也可能合并访存；每线程向量加载也不能自动证明地址对齐、编译指令或实际吞吐。该设备函数没有输入 stride 参数，不能据上层 tensor 接口假定支持任意视图。宏中的 `use_vec` 主要根据维度整除选择，不能将源码注释中的“unaligned fallback”理解为所有指针偏移已被检查。

### SGLang：把所有行的向量片段展平

SGLang `SiluAndMul.forward_cuda` 调 `jit_kernel/activation.py:silu_and_mul`；CUDA 平台导入的是 JIT 包装。包装把输入/输出 view 成二维，`ActivationKernel::launch` 核对维度及 dtype/device，要求 D 能被向量元素数 V 整除，使用 256 线程的 block。

`jit_kernel/csrc/elementwise/activation.cuh:act_and_mul_kernel` 将全局线程号 $t=bT+\mathrm{threadIdx.x}$ 映射为

$$Q=D/V,\qquad m=\lfloor t/Q\rfloor,\qquad q=t\bmod Q.$$

线程 t 负责第 m 行、第 q 个向量片段；向量下标分别是 gate 的 $2mQ+q$、up 的 $2mQ+Q+q$ 和输出的 t。每个向量包含 V 个元素，指针转换已经包含元素字节大小，不能再把向量下标当标量下标使用。

教学例：M=3、D=24、V=8 时 Q=3，只有 9 个有效线程任务。t=4 对应第 1 行、第 1 段，读取标量列 gate 8…15 与 up 8…15，其输入标量偏移分别为 56…63 和 80…87，写输出 32…39。尾部线程通过 `token_id >= num_tokens` 返回；这只能处理任务数量不足一个 block，不能处理 D 不是 V 倍数的向量尾部，后者由包装拒绝。

`utils.cuh` 的 `kMaxVecBytes` 按编译架构选择 16 或 32 字节，V 再除以 dtype 字节数。V 是这里的内存工作粒度，不是 warp 宽度。与每行一个 block 相比，展平任务可以让同一个 block 覆盖多行；有多少有效 warp、多少 block 能驻留，仍取决于具体 M、D 与资源，不能由映射本身得出速度排名。

## 3. Triton：用二维 program 网格切分行与列

vLLM `layers/activation.py:_swiglustep_and_mul_kernel` 是仓库中可直接阅读的 Triton 变体。`SwigluStepAndMul.forward_cuda` 调包装，实际包装要求二维输入和偶数最后维度，启动 grid 为 $(M,\lceil D/1024\rceil)$。每个 program 的两维 ID 分别选择行 m 与列段 p：

$$j=1024p+\mathrm{arange}(0,1024),\quad \mathrm{mask}=(j<D).$$

设备函数分别使用输入和输出的行 stride，列方向依旧隐含连续。program 处理 1024 个逻辑元素，不是启动 1024 个 CUDA 线程。没有行内归约，因此多个 program 可以独立写同一行的互不重叠片段；这与 RMSNorm 将一行直接拆成若干独立归一化片段不同。

该 kernel 的失效 load 没有显式指定 `other`，但失效元素只流向逐元素运算，最后 store 同样被 mask，不进入任何有效输出的归约。若以后增加统计归约，就必须重新定义失效值，不能照搬这条理由。

它的数学语义是激活后裁剪：

$$y=\min(\mathrm{SiLU}(g),L)\cdot\mathrm{clip}(u,-L,L).$$

而 CUDA `SiluAndMulWithClamp` 在激活前裁剪 gate，并允许 sigmoid 系数 alpha 与 up 偏置 beta。取 alpha=1、beta=0、g=2、u=1、L=1，前者为 1，后者为 $\mathrm{SiLU}(1)\approx0.73106$。这两个真实实现可以用于学习工作映射，却不能直接作为同一语义的正确性基线或性能竞赛。

## 4. 融合省什么，仍需保存什么

若将 SiLU 和乘法拆成两个 kernel，并把中间激活写回全局内存，每输出的逻辑流量为：读 g、写 SiLU(g)、读 SiLU(g)、读 u、写 y，共 5e 字节。融合后读 g/u 并写 y，共 3e 字节，理想流量比为 5/3。该推导假设同 dtype、中间量实际落盘，不含投影、权重、缓存及启动开销；不是实际加速比，更不是相对编译器已融合路径的收益。

数值阶段也会改变：vLLM 的 `silu_kernel/packed_silu_kernel` 返回输入类型的激活值，再参与乘法；SGLang JIT 的 `apply_activation_f32` 保留 FP32 激活，乘 FP32 up 后再转输出。用 $R_h$ 表示低精度转换，两条路径分别类似 $R_h(R_h(\mathrm{SiLU}(g))u)$ 与 $R_h(\mathrm{SiLU}(g)u)$。近似指数、编译选项与中间舍入都可能影响差异，不能只核对最后的 dtype。

SGLang JIT 还有可选 `expert_ids` 过滤：当 `expert_ids[token_id//expert_step] == -1` 时直接跳过该行。**跳过不是写零。** 若 out 是新建的 `empty`，相应位置没有由该 kernel 初始化；后续消费者必须按同一有效性语义忽略它们。这个分支不能被当作普通稠密输出用于全张量比较。本页只解释其写入契约，没有覆盖完整 MoE 路由。

## 5. 把源码变成可检验的设计判断

首先检查 gate/up 顺序、最后维的 2D 关系、行列 stride 与向量整除，再检查每个有效输出恰有一个拥有者。逐元素算子不需要跨线程交换数据，但仍受外部调用顺序和别名约束；原地覆盖输入可能破坏其他线程尚未读取的位置，不能因“无归约”就默认任意原地执行安全。

接着区分数学变体与舍入阶段，用 D 在向量宽度和 1024 两侧的尺寸检查 fallback/mask；过滤分支以预填哨兵值检查未写区域。测试方法见[正确性与性能测量](kernel-correctness-and-benchmarking.md)。这类算子适合建立 CUDA/Triton 的工作映射基础，再进入[归约](rmsnorm-cuda-triton-kernels.md)、[Attention](vllm-attention-operator-design.md)和[低比特矩阵乘](weight-only-dequant-kernels.md)。

本页已用 CPU 教学模型核对三种工作映射、门控地址、裁剪次序与过滤未写语义；没有执行原仓库 GPU 实现或测量融合收益。PDL、全部激活变体与 MoE 全链路不在本轮展开范围。

## 来源身份

| 来源 | 固定版本 | 核对范围 |
| --- | --- | --- |
| [vLLM](https://github.com/vllm-project/vllm/tree/568afb3a13806beb53bb2e6bd518269357b237c0) | `568afb3a13806beb53bb2e6bd518269357b237c0` | Llama MLP、SiluAndMul 与 SwigluStepAndMul 包装、CUDA 门控与向量分派、Triton 二维网格。 |
| [SGLang](https://github.com/sgl-project/sglang/tree/2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc) | `2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc` | Llama MLP、CUDA 平台导入、JIT activation 包装、展平向量工作映射与过滤写入。 |
