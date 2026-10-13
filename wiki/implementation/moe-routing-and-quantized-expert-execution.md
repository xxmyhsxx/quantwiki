---
title: MoE 路由与量化专家执行：SGLang/vLLM 的单设备 FP8 主线
type: implementation
tags:
  - moe
  - routing
  - fp8
  - kernels
sources:
  - raw/articles/2026-09-21/flashinfer-release-notes
  - raw/repositories/2026-09-21/sglang
  - raw/repositories/2026-09-21/vllm
updated: 2026-10-14
---

# MoE 路由与量化专家执行：SGLang/vLLM 的单设备 FP8 主线

MoE 的 top-k 结果只是专家 ID 与权重；完整执行还要把 token–专家配对变成矩阵乘的行、按正确粒度读取量化尺度，再把专家输出汇回原 token。本页先沿 SGLang 固定 commit `2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc` 的一条普通设备内路径解释这些接口，再在第 8 节对照 vLLM 固定 commit 的同类 FP8/Triton 分支。本次只核读源码，没有加载模型、编译 kernel 或测量性能。

模型结构与离散路由边界见 [MoE 基础](../fundamentals/model/mixture-of-experts-and-routing.md)；请求如何形成当前模型批次见 [SGLang 推理执行](sglang-inference-execution.md)。这里的 token 数 $M$ 是本轮进入 MoE 的隐藏状态行数，可以来自 prefill、decode 或多个请求，不是某个专家的工作行数。

专家跨 rank 放置时，继续见 [MoE 专家并行执行](moe-expert-parallel-execution.md)。该页另行固定 normal/HT 与 low-latency 分支，追踪 expert 映射、dispatch、通信尺度和返回；本页的单设备排序与归约不能直接替代其通信布局。

权重量化专家的加载与设备链见 [W4A16 MoE 专家执行](w4a16-moe-expert-execution.md)：它接续单设备 AWQ 整数零点的 Marlin 格式、专家配对与两次 GEMM。其 down 输出先转 BF16 再乘已转 BF16 的 gate，普通 SiLU 也保留中间舍入；下文所选 FP8/Triton 的精度顺序仍按对应分支解释。

## 1. 先固定分支，避免把不同后端拼成一条链

第 1–7 节选定 `DeepseekV2MoE.forward_normal` 的普通 CUDA 路径：单设备，MoE TP/EP 均为 1，`moe-a2a-backend=none`、`moe-runner-backend=triton`；使用已序列化的 FP8 checkpoint，普通 E4M3 权重、`weight_block_size=[128,128]`、dynamic activation。MoE 层未被 ignored-layers 排除，没有 LoRA、EPLB、冗余专家、hash 路由、SBO/TBO 或特殊 graph 执行；本页先按非 TMA 的指针载入分支，关闭 fused sum/all-reduce。共享专家融合也关闭，以单独的共享 MLP 解释汇合。[1][4][5][6][10][14]

路由部分采用该类支持的 `topk_method="noaux_tc"`、sigmoid 打分、correction bias、grouped top-k 和 `norm_topk_prob=True` 配置。`DeepseekV2MoE` 是多个模型共用的实现类，类名不等于本页全部行为来自 DeepSeek-V2。尤其这些有偏 sigmoid/group 规则不能替换 [V2 论文的 softmax affinity](../fundamentals/model/mixture-of-experts-and-routing.md#1.%20%E5%85%B1%E4%BA%AB%E4%B8%93%E5%AE%B6%E3%80%81%E8%B7%AF%E7%94%B1%E4%B8%93%E5%AE%B6%E4%B8%8E%20gate%20%E5%88%86%E5%88%AB%E5%81%9A%E4%BB%80%E4%B9%88)；原 V2 配置和其他 backend 应分别核对。[1][2]

`Fp8Config` 要求块量化来自 FP8 serialized checkpoint 且 activation 为 dynamic；当前 CUDA 量化包装还要求归约维可被 activation group size 整除。因此下面的 $D,I$ 均取 128 的倍数。[6][15]

| 阶段 | 本页实际调用 | 交付给下一阶段的对象 |
| --- | --- | --- |
| 模型 FFN | `DeepseekV2MoE.forward_normal` | 当前隐藏状态与 top-k 结果 |
| 路由 | `MoEGate → TopK → select_experts` | `StandardTopKOutput(weights, ids, logits)` |
| 专家层 | `FusedMoE.forward_impl` | `StandardDispatcher.dispatch` 的包装对象 |
| 量化方法 | `Fp8MoEMethod.apply` | `TritonMoeQuantInfo`，含权重、尺度和 block shape |
| 执行后端 | `MoeRunner.run → fused_experts_none_to_triton` | `fused_experts` 的两次专家 GEMM 与汇合 |
| 返回模型 | `StandardDispatcher.combine`，再加共享输出 | `[M,D]` 隐藏状态 |

`triton` 与源码中的 `triton_kernels` 是两个 runner backend 名称，TopK 输出格式和后端代码并不相同；本页选择前者。[2][8][9]

## 2. 路由：用于选专家的分数与输出 gate 分开

输入 $X\in\mathbb R^{M\times D}$，router 参数按 `[E,D]` 保存。`MoEGate` 计算 `[M,E]` logits；一般分支为 `F.linear(X, weight)`，某些小 token 数、维度与设备条件会使用专门 router GEMM。这里的 gate 是普通参数和路由计算，未作为 `FusedMoE` 的 FP8 专家张量创建；专家 FP8 也不意味着 router、所有隐藏状态和最终输出都为 FP8。[1，`MoEGate`]

在选定 noaux_tc 分支，下面写出核读的值语义。设 router logit 为 $r_{t,e}$，纠正偏置为 $b_e$：

$$s_{t,e}=\operatorname{sigmoid}(r_{t,e}),\qquad c_{t,e}=s_{t,e}+b_e.$$

专家按配置划成 $G$ 个组。每组用该组内最高两个 $c$ 的和打分，先选 `topk_group` 个组，再从保留组的专家中按 $c$ 选 $K$ 个。输出 gate 则从**没有加偏置**的 $s$ 取回，并在选中集合内归一化：

$$g_{t,j}=\frac{s_{t,e_{t,j}}}{\sum_{\ell=1}^{K}s_{t,e_{t,\ell}}}.$$

因此 correction bias 可改变被选集合，不能把它直接当作最终专家输出的加性权重。`biased_grouped_topk_impl` 提供可读参考；CUDA 包装根据维度与依赖选择 FlashInfer 或 `sgl_kernel.moe_fused_gate` 等实现。核读的 CUDA gate 同样先 sigmoid、加 bias 用于选择，再取原 sigmoid 值归一化。本页固定 `norm_topk_prob=True`，不将这些优化分支的规则推广到未归一化配置；ties 的具体选择顺序也由实际 backend 决定。[2，`select_experts/biased_grouped_topk_impl/biased_grouped_topk_gpu`][3]

最终 `topk_ids` 为 `[M,K]` INT32，`topk_weights` 为 `[M,K]` FP32。它们同一位置对应同一个 token–专家配对；某些参考 top-k 不按分数排序不影响这个契约。此处没有训练 balance loss 或设备容量 token dropping；之后的 GEMM padding 也不会删除模型上下文 token。[2][3]

`FusedMoE.should_fuse_routed_scaling_factor_in_topk` 在所选 FP8/triton 分支为 false，故 router 不预乘 `routed_scaling_factor`。该系数由 routed 输出汇合阶段施加，后文不能再无条件乘一次。[1][5][10]

## 3. 专家权重：w13 合并两层投影，w2 恢复隐藏维

设路由专家数 $E$，单专家中间宽度 $I$，hidden size 为 $D$。单设备无分片时：

| 参数 | shape | 含义 |
| --- | --- | --- |
| `w13_weight` | `[E,2I,D]` | 每个专家的 gate 与 up 投影，按输出行串接 |
| `w2_weight` | `[E,D,I]` | 每个专家的 down 投影 |
| `w13_weight_scale_inv` | `[E,2I/128,D/128]` | gate/up 分开的 128×128 权重块尺度 |
| `w2_weight_scale_inv` | `[E,D/128,I/128]` | down 的 128×128 权重块尺度 |

普通序列化 FP8 分支将权重保存为 `torch.float8_e4m3fn`，块尺度为 FP32。创建 `w13` scale 时分别为 gate/up 各分配 $\lceil I/128\rceil$ 行块，防止误把串接边界跨成一个块。本页整除条件下正好得到上表形状；一般分配代码里的 `ceil` 不代表整个激活量化路径都接受任意尾部维度。[6，`create_weights`][18]

加载时，`FusedMoE._load_w13` 默认把 `w1/gate_proj` 放入前 $I$ 行、`w3/up_proj` 放入后 $I$ 行；`w2/down_proj` 单独加载。某些 backend 可设置 `load_up_proj_weight_first` 或变换布局，本页没有采用。数学上，一个专家的浮点函数为

$$f_e(x)=\bigl[\operatorname{SiLU}(xW_{1,e}^{T})\odot(xW_{3,e}^{T})\bigr]W_{2,e}^{T}.$$

所以“合并 w13”是一次 GEMM 同时给出 gate/up，并未把这两个输出先求和；后续 SiLU 与逐元素乘法把 $2I$ 恢复成 $I$。[5，`_load_w13/make_expert_params_mapping`][10，`_fused_moe_kernel_sequence`]

形状中的 expert 轴、输出轴和归约轴不能互换。例如 `w13` 的每个 expert 都有自己的二维块尺度，不能把整层所有专家共用一个 scale。这里的 `scale_inv` 是参数名称；在核读 Triton GEMM 中，它作为 FP8 数值恢复的乘子使用，不能因为后缀 `inv` 又取一次倒数。[6][13]

## 4. 单设备 dispatch 不搬行，排序发生在专家 kernel 包装内

`FusedMoE.forward_impl` 依次执行 dispatch、`quant_method.apply` 和 combine。a2a 为 none 时，`StandardDispatcher.dispatch` 在本页分支直接包装原 hidden states、`hidden_states_scale=None` 与 top-k 输出，没有在这里复制 $K$ 份输入，也没有 all-to-all 或物理重排行。[5][7]

`Fp8MoEMethod` 创建 `MoeRunner(TRITON, config)`，`apply` 取得 `TritonMoeQuantInfo`。`MoeRunner` 查到注册的 `fused_func("none","triton")`，直接进入 `fused_experts_none_to_triton → fused_experts`。所以 `pre_permute_standard_to_triton` 的示例/测试分支不是本页执行链上的实际排序入口。[6，`create_moe_runner/get_triton_quant_info/apply`][8][9]

真正的索引组织在 `_prepare_fused_moe_run → moe_align_block_size`。把配对位置展开为

$$p=tK+j,\qquad p\in\{0,\ldots,MK-1\}.$$

按 `topk_ids.flatten()[p]` 统计每个专家的配对数 $m_e$，各自补到 GEMM `BLOCK_SIZE_M=B_M` 的倍数，再按专家汇集配对索引。结果是：[10][11][12]

- `sorted_token_ids`：按专家组织的**配对索引** $p$，不只是原 token 行号 $t$。
- `expert_ids`：每个 $B_M$ 行块对应的专家 ID；一个块只使用一个专家的权重。
- `num_tokens_post_padded`：真实配对加各专家 padding 的总长度，kernel 据此跳过缓冲区额外容量。

下面是整理者构造的索引例子，不是模型运行结果。取 $M=3,K=2,E=3$，`topk_ids=[[2,0],[1,2],[1,0]]`，$B_M=4$：

| 专家 | 原配对位置 p | 补齐后的一种排列 |
| --- | --- | --- |
| 0 | `[1,5]` | `[1,5,6,6]` |
| 1 | `[2,4]` | `[2,4,6,6]` |
| 2 | `[0,3]` | `[0,3,6,6]` |

6 是 $MK$，表示不存在的配对。`expert_ids=[0,1,2]`，padding 后共 12 行；专家内部次序不要求上述固定顺序。第一轮 GEMM 以 $t=\lfloor p/K\rfloor$ 回读原 $X$；padding 通过 $p<MK$ mask 排除。真实输入仍按 `[M,D]` 保存，稀疏计算通过索引间接载入实现。[12，`count_and_sort_expert_tokens_kernel`][13]

由此也能解释一个成本条件：逻辑工作有 $MK$ 个配对，块执行工作约为 $\sum_e B_M\lceil m_e/B_M\rceil$ 行。该式是从源码补齐规则整理的记账，不能直接当作 FLOPs 实测；它说明小批次、分散路由和 tile 大小会影响 padding。存储了许多专家也不等于本批次使用了同样多专家。[10–13]

## 5. 两次 FP8 GEMM：按 K 组恢复尺度，再经过门控激活

`get_triton_quant_info` 传入两个专家权重、对应块尺度、`block_shape=[128,128]` 和 `use_fp8_w8a8=True`。dynamic 分支的 `w13_input_scale/w2_input_scale` 为 None；激活尺度每次由输入计算，不使用一个预先保存的全层常数。[6]

### 5.1 第一轮：原 token 隐藏状态到 gate/up

`invoke_fused_moe_kernel` 在 block 分支对 $X$ 调用 `sglang_per_token_group_quant_fp8(X,128)`。输入保持 `[M,D]`，FP8 输出同形，FP32 activation scale 为 `[M,D/128]`。每 token、每 128 个归约通道一组；同一 token 送到不同专家时，第一轮可以回读同一份已量化输入及尺度。[13][15]

为解释数值语义，记输入组 $q$ 的 scale 为 $a_{t,q}$、FP8 数值为 $\hat X$，权重块恢复尺度为 $s_{e,\lfloor n/128\rfloor,q}$、FP8 权重为 $\hat W$。则输出通道 $n$ 的近似为

$$Y_{p,n}\approx\sum_q a_{t,q}s_{e,\lfloor n/128\rfloor,q}
\sum_{k\in q}\hat X_{t,k}\hat W_{e,n,k},\qquad t=\lfloor p/K\rfloor.$$

这是一条按核读接口写出的值语义，不是硬件每一步舍入的 bit-exact 规范。CUDA activation 包装依条件进入 JIT/AOT 实现；所核读 JIT v2 的普通 E4M3、非 UE8M0 分支以组内绝对最大值和小正下界计算 $a=\max(\max_k|x_k|,10^{-10})/448$，保存 $a$，将 $x/a$ 限幅后转换为 FP8。其 Python 包装与 C++ 数值核分开，不能把读到其中一个实现写成已经确认运行环境采用它。[15–17]

Triton 核按配对索引、专家 ID 与输出块读取 A/B。FP8 `tl.dot(a,b)` 的部分和乘当前 K 组的 A/B scale，再累加到 FP32 accumulator；最后转成输入隐藏状态对应的 BF16/FP16 compute type 写回。因 K 组 scale 可不同，不能先完成整个无尺度点积，再乘一个固定总 scale；其共通原因见 [量化矩阵乘的缩放](quantized-matmul-scaling-execution.md#1.%20%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95%E7%9A%84%E5%A4%96%E7%BB%B4%E4%B8%8E%E5%BD%92%E7%BA%A6%E7%BB%B4)。本页是普通 FP8 dot 加块尺度的 kernel，不等同于 [MXFP4/MXFP8 的 `tl.dot_scaled` 教程](block-scaled-matmul-execution.md)。[13][14]

非 TMA 分支按配对位置 $p$ 写 gate/up 中间值 `[MK,2I]`，所以后续可以用普通连续行调用 SiLU-and-mul。第一轮没有提前乘 router gate，因为本页 `apply_router_weight_on_input=False`；否则 gate 插在非线性之前一般会改变函数，不能认为在任意位置乘 gate 都等价。[10][13]

### 5.2 第二轮：门控后的专家专属行到 hidden size

SiLU-and-mul 将每个配对的 gate/up 两半变成 `[MK,I]`。这些行已经依专家计算而不同，同一原 token 的不同专家配对不能再共享一份中间激活。第二轮重新按行和 128 个 I 通道量化，activation scale 为 `[MK,I/128]`。[10][13][15]

down GEMM 重用相同排序和专家 ID，但 kernel 的 `top_k=1`：此时 A 的原行索引就是配对 $p$，不是再除以模型的 $K$。它用 `w2[e]` 生成 `[M,K,D]` 结果，并在第二轮 accumulator 结束处乘该配对的 `topk_weights[p]`。原始 routed $K$ 仍用于最终 token 汇合；kernel 参数中的 1 只描述当前 A 行的回读规则。[10，第二次 `invoke_fused_moe_kernel`][13]

因此这条主线保留低精度专家权重与乘法输入，内部有 FP32 累加、BF16/FP16 中间输出和动态量化。W8A8 的名称没有把 accumulator、SiLU 中间张量或整条模型隐藏状态限制为 8 bit。

## 6. routed 合并、共享专家与原 token 次序

设 routed scaling factor 为 $\rho$，两次 GEMM 后得到的每配对值已乘 $g_{t,j}$。常规汇合将 `[M,K,D]` 沿 K 求和，再乘 $\rho$：

$$y_t^{(r)}=\rho\sum_{j=1}^{K}g_{t,j}\,\tilde f_{e_{t,j}}(x_t).$$

这里 $\tilde f$ 包含量化和中间舍入误差，不是原浮点专家函数。CUDA 汇合实现根据 token 数、K 和开关选择 torch add、编译后的 sum 或 `moe_sum_reduce`；它们在本页承接同一输出位置语义，源码存在多种实现不构成本地吞吐比较。`StandardDispatcher.combine` 对已经完成汇合的 `[M,D]` 直接返回，不做第二次 top-k 加权或反置换。[7][10]

独立共享专家的 `DeepseekV2MLP` 处理全部 $M$ 行：gate/up 线性层、SiLU-and-mul、down 线性层。其总中间宽度是 `moe_intermediate_size * n_shared_experts`，可把多份共享专家的拼接投影和 down 求和组织成一个较宽 MLP；它不是 router 中再竞争一个 top-k 位置。shared linear 的量化方法由对应 `LinearBase` 前缀/配置选择，本页不把 routed `Fp8MoEMethod` 的 kernel 强套到共享 MLP。[1][6]

`forward_normal` 默认专家 runner 可 inplace 改写输入，所以 `defer_shared = not inplace`：inplace 时先用原输入算 shared，非 inplace 时可等 routed 完成后再算。这个时序保护共享 MLP 对原 $X$ 的读取，不表示共享输出应由 routed 输出作为输入。[1，`forward_normal`]

最终

$$y_t=y_t^{(r)}+y_t^{(s)}.$$

本页 FP8/triton 已在 routed 汇合处施加 $\rho$，`maybe_fuse_routed_scale_and_shared_add` 的非 MXFP4 特殊分支只加 shared，不再缩放共享输出。TP=1 不触发最后的 TP all-reduce。模型残差相加属于 decoder 外层职责，不能在本页 FFN 返回值里自行再加一次 $x_t$。[1][19]

共享专家融合是另一条布局选择。源码支持把共享权重映射到额外 expert slot，在 TopK 中追加始终参与的 ID，使用特殊 gate 抵消后续 routed scaling；是否开启受架构、专家数、量化配置、设备及 overlap 条件限制。本页显式关闭它，只说明它不是“把共享专家也按同一 affinity 自由挑选”；不把独立 shared MLP 和 fused shared slot 计算两遍。[1，`determine_num_fused_shared_experts`][2][3]

### v0.6.17 发布声明中的 FP4 专家接口

固定 FlashInfer v0.6.17 发布说明另报告：统一 `MoELayer` 的 TRTLLM-gen MXFP4 权重支持 MXFP8 激活的 W4A8 和 BF16 激活的 W4A16；预路由 FP4 模式可分别传入连续的 `topk_ids` 与 `topk_weights`；TRTLLM-gen FP4 MoE 增加共享专家融合，以及 MXFP4×MXFP8、NVFP4×NVFP4 的 SiTU 激活。这里的 W4A16 指低精度浮点权重格式，不等于前述 AWQ 整数零点/Marlin 链。发布说明未给这些入口的完整 packing/scale、gate/rho 和融合共享贡献顺序，本页 FP8/Triton 主线只能提供核对问题，不能直接替它们定义执行语义。（来源 v0.6.17 Highlights，Unified MoE API；#4159/#4104/#4239/#4180，仅核发布文本。）

## 7. SGLang 主线的覆盖与验证边界

这次核读已经连接四个过去需要分别查证的对象：router 选择与 gate 的区别；配对索引、逐专家 padding 与原 token 次序；专家权重/二维块尺度与 activation 行组尺度；routed 归约、缩放和独立共享输出。它可以承接 [MoE 基础](../fundamentals/model/mixture-of-experts-and-routing.md#4.%20%E6%80%BB%E5%AE%B9%E9%87%8F%E3%80%81%E5%8D%95%20token%20%E8%AE%A1%E7%AE%97%E4%B8%8E%E6%89%B9%E6%AC%A1%E6%88%90%E6%9C%AC%E4%B8%BA%E4%BD%95%E4%B8%8D%E5%90%8C)中“单 token 激活参数量不等于批次访存/吞吐”的问题。

未覆盖 EP/DeepEP 的通信与物理 expert 映射、EPLB、FP4/INT4/WNA16、其他 runner、LoRA 和完整 weight-loader 的所有 checkpoint 名称重映射；TMA、CUDA graph、并行 overlap 也未作为另一条执行链完整核读。源码对不同分支的能力说明不等于本项目已支持全部组合。

实际验证应围绕所选模型、硬件与配置分别检查：expert ID/gate 和 token 对应、padding mask、scale shape 与 K 分组、两次 GEMM/激活/汇合的数值误差，以及真实配对负载与调度成本。此次没有执行这些检查，没有模型正确性、显存或速度结果；不根据参数激活比例或 FP8 名称推算端到端加速。验证方法接到 [Kernel 正确性与测量](kernel-correctness-and-benchmarking.md)及 [服务评测](serving-performance-evaluation.md)。

另一个版本边界是 FlashInfer 发布说明在 release-v0.6.17 撤回 SM90 CUTLASS MoE backend #3738 及依赖 #4025/#4080（What's Changed，#4411）。它没有说明撤回原因，不能推广成所有 SM90 MoE 或 CUTLASS GEMM 都不可用，也没有改变本页所选的 SGLang/vLLM FP8/Triton 源码范围。

## 8. vLLM 对照：同一 FP8 计算由 factory 与 modular kernel 分工

以下对照固定于 vLLM `568afb3a13806beb53bb2e6bd518269357b237c0`。选择 `DeepseekV2MoE`、CUDA、BF16 输入、DP/TP/EP/PCP 均为 1、无 sequence parallel、`moe_backend="triton"`；同样采用 serialized E4M3 FP8、128×128 权重块、dynamic activation，$D,I$ 均可被 128 整除。路由选择有效分组 $G>1$、sigmoid、noaux_tc correction bias 和 `norm_topk_prob=True`；为固定可读的路由分支，关闭 `VLLM_USE_FUSED_MOE_GROUPED_TOPK`，沿 `grouped_topk` 的 PyTorch 值语义实现。[V1][V5][V8][V9]

此外固定 `apply_router_weight_on_input=False`，沿 `DeepseekV2DecoderLayer` 在 CUDA 上显式传入的 `apply_routed_scale_to_output=True`；无 LoRA/EPLB、冗余专家、DBO、latent 输入变换、routing simulator 或 zero experts；关闭共享专家辅助 stream，以及 DeepGEMM E8M0 activation scales，选普通 FP32 尺度。后文先取 $4MK>E$，确保进入按专家对齐分支。这些条件避免将不同路由、通信、数值格式和小批量路径混成一条链。backend 的 `is_supported_config` 还检查设备原生 FP8 能力；没有以配置类的最小 capability 或这次核读代替硬件运行验证。[V1][V4][V6][V9][V13][V16][V22]

### 8.1. FusedMoE 是构造入口，gate 的实际调用属于 runner

这个版本的 `FusedMoE` 是 factory 函数，返回 `MoERunner`，不是把 dispatcher、权重与量化方法全放在同名 layer 内的旧结构。它建立三个对象：[V2]

- `GroupedTopKRouter` 保存路由规则并交付 gate/ID。
- `RoutedExperts` 拥有 `w13_weight`、`w2_weight`、scale 参数与 `Fp8MoEMethod`。
- `MoERunner` 协调 gate、router、专家调用和独立共享输出；本模型把 `GateLinear` 也交给它。

`DeepseekV2MoE.forward` 因 `MoERunner.is_internal_router=True`，把 hidden states 同时填入 `router_logits` 形参作为占位。真实 gate 计算在 `MoERunner._forward_impl` 内进行，所得 `[M,E]` logits 覆盖占位输入，随后 `_apply_quant_method` 才调用 router。本模型没有先在外层算一次 gate 再让 runner 重算。[V1，`forward`][V6，`is_internal_router/_forward_impl`]

`GateLinear` 继承 `ReplicatedLinear`，构造时明确使用 `quant_config=None`。它按设备、矩阵形状与输出 dtype 选择专用 router GEMM 或普通 linear；因此 FP8 专家 checkpoint 不会自动把 router 变成 FP8 GEMM。这里核对了 gate 的接口与分派条件，没有把每个专用 router kernel 展开成第二条执行链。[V3]

| 阶段 | vLLM 选定调用 | 与 SGLang 主线的接口区别 |
| --- | --- | --- |
| 构造 | `FusedMoE → router + RoutedExperts + MoERunner` | SGLang 的同名 `FusedMoE` 是专家层类 |
| Gate/TopK | `MoERunner._forward_impl → GateLinear → router.select_experts` | SGLang 在模型 `forward_normal` 中形成 TopK 输出对象 |
| 参数与量化方法 | `RoutedExperts.forward_modular → Fp8MoEMethod.apply` | 直接传 `topk_weights/topk_ids`，不传 `StandardTopKOutput` |
| 专家计算 | `FusedMoEKernelModularImpl` 的 prepare/expert/finalize | 分成 `NoDPEPModular` 与 `TritonExperts`，后者负责配对对齐和 GEMM |
| 共享汇合 | runner 得到 shared/routed 后相加 | 独立 shared MLP 的语义相同，调度所有者不同 |

### 8.2. scale 位置由调用端决定，不能只读 factory 默认值

所选 vLLM `grouped_topk` 同样用 $s=\operatorname{sigmoid}(r)$、$s+b$ 选组/专家，再取原 $s$ 作为 gate，并按选中集合归一化。`DeepseekV2MoE` 的构造参数虽然默认为 `apply_routed_scale_to_output=False`，上层 `DeepseekV2DecoderLayer` 却在 CUDA 非 AITER 条件下显式传入 True。实际模型主线必须沿这个调用端：factory 给 router 的 scaling factor 为 1，给 runner 的才是 $\rho$。[V1，`DeepseekV2DecoderLayer.__init__`][V2][V5]

因此本页实际 CUDA 主线交给专家的是第 2 节的 $g$，其行和为 1。down GEMM 在专家输出上乘 $g$，`moe_sum` 归约 $K$；BF16 routed 结果在 runner 的 `_maybe_apply_routed_scale_to_output` 内乘 $\rho$，再与独立 shared 相加。这与所选 SGLang 的施加位置相同，不能再对 top-k weights 或共享输出无条件乘一次。该结论固定于 BF16；FP16 时 runner 和 decoder 有联合防溢出缩放，不能直接套用同一描述。[V6][V13][V16]

若单独按 class/factory 默认 False 构造，则另一种接口行为才是

$$\widetilde g_{t,j}=\rho\,\frac{s_{t,e_{t,j}}}{\sum_{\ell=1}^{K}s_{t,e_{t,\ell}}}=\rho g_{t,j},$$

这时 router 的 weight 行和为 $\rho$，down GEMM 使用已含 $\rho$ 的 $\widetilde g$，runner factor 为 1。实数代数下两种位置都表示 $\rho\sum_jg_{t,j}f_{e_{t,j}}(x_t)$；有限精度下，乘法移过存储转换和归约并不保证逐位一致。默认值说明接口可能性，不能拿它替代本固定版本模型调用端的显式 True。[V2][V5][V6]

### 8.3. no-DP/EP prepare 量化原 token 行，专家模块再组织配对

`Fp8MoEMethod` 以 `(kFp8Static128BlockSym,kFp8Dynamic128Sym)` 请求后端。oracle 在明确选择 triton 且配置受支持时创建 `TritonExperts`；所选 Triton 权重格式转换不重排 `w13/w2`。`RoutedExperts._load_w13` 仍把 gate 放在前 $I$ 行、up 放在后 $I$ 行，权重及 FP32 块 scale 的 shape 与第 3 节相同。`scale_inv` 也仍在 GEMM 中作为恢复乘子，不能重复取倒数。[V7][V8][V9][V16]

`make_fp8_moe_kernel → maybe_make_prepare_finalize` 在无 all-to-all 且 DP=1 时选 `MoEPrepareAndFinalizeNoDPEPModular`。它不复制 $K$ 份 token、不按专家物理重排行，直接把原 $X[M,D]$ 动态量化成 `a1q[M,D]` 与 `a1q_scale[M,D/128]`，返回的替代 top-k ID/weight 为 None，故保留 router 原有结果。此处的 `NoDPEP` 指该 prepare/finalize 选择，不能推广到 DP>1 的 gather/scatter 分支。[V10][V11][V12]

与 SGLang 在 `invoke_fused_moe_kernel` 的包装内量化不同，vLLM 第一次输入量化由 prepare 完成。`moe_kernel_quantize_input → per_token_group_quant_fp8` 每个 token、每 128 个连续输入通道产生一个 FP32 恢复尺度；本页关闭 E8M0 时，CUDA 核取组内最大绝对值与 epsilon 下限，再除以 E4M3 上界 448。它量化的是原 token 行，不是 expert 轴；同一个原 token 的不同专家配对在第一次 GEMM 共享这份输入量化结果。[V14][V15][V21][V22]

然后 `TritonExperts.apply → _prepare_expert_assignment → moe_align_block_size` 才建立 `sorted_token_ids/expert_ids/num_tokens_post_padded`。这些对象仍保存第 4 节的配对索引 $p=tK+j$、逐专家 padding 和每行块的 expert ID，CUDA align/sort 核按专家计数与前缀位置填入配对索引；没有事先物理重排 `a1q`。第一次 GEMM 根据 $p//K$ 读取原 token 行。[V13][V16][V17][V20]

小批量有一条必须单独识别的例外：在 `expert_map is None` 且 $4MK\le E$ 的所选 FP8 条件下，`_prepare_expert_assignment` 不调用 align，返回 `sorted_token_ids=None`、展平的 top-k expert ID，并为每个配对安排一个 $B_M$ 块。kernel 仅让该块首行读取配对，其他行是 sentinel。该分支保持配对语义，却没有“按 expert 排序”；因此不能由一个 token 的 decode 情形推断 vLLM 总会建立排序表。这里没有测量两种分派的成本。[V16，`_prepare_expert_assignment/fused_moe_kernel`]

### 8.4. 两次 GEMM 之间融合 activation 与第二次 FP8 量化

沿对齐主线，第一 GEMM 仍从 FP8 `a1q` 与专家 `w13` 计算 gate/up：每个归约组的 dot 按 activation/weight scale 恢复后累加，写入 `intermediate_cache1[M,K,2I]`。从量化输入推导的 `compute_type` 在此为 BF16，配对输出不预乘 router 权重。[V13][V16]

当 activation 为 gated SiLU、`use_fp8_w8a8=True`、block shape 恰为 `[128,128]`、无 LoRA 且未启用 DeepGEMM E8M0 scales 时，`TritonExperts` 直接调用 `ops.silu_and_mul_per_block_quant`。固定 CUDA 核对每个配对、每个 128 通道组读取 BF16 gate/up，在寄存器中以 FP32 计算 SiLU×up，做组内 amax，再写出 `qintermediate_cache2[MK,I]` 的 FP8 值与 `a2q_scale[MK,I/128]` 的 FP32 尺度。[V13][V18][V19]

此处没有先把完整 `[MK,I]` 的 SiLU×up 结果写入 BF16 cache，再重读量化；SGLang 所选分支分别执行 `silu_and_mul` 与第二次 activation 量化。在两侧均采用 BF16 的比较条件下，vLLM 融合核避免这次中间 BF16 存储舍入，且有自己的最小 scale 与编码规则，所以不能从两者相同的算子名称推断位级相同。若融合条件不满足，vLLM 的 else 分支仍先写 activation cache，再 `moe_kernel_quantize_input`；本页没有把该 fallback 或 LoRA 作为另一条完整验证主线。[V13][V19]

第二 GEMM 复用配对表，因输入已经是 `[MK,I]`，传 `top_k=1` 读取行 $p$；输出写回 `intermediate_cache3[M,K,D]`。它在 FP32 accumulator 上乘 `topk_weights[p]`，在本模型主线即不含 $\rho$ 的 $g$，再转换为 BF16。`ops.moe_sum` 按每个 token 的 $K$ 个配对求和为 `[M,D]`；核读的 CUDA sum 以 FP32 累加后写回 BF16，$\rho$ 仍由后续 runner 施加。[V13][V16][V20]

### 8.5. cache 是临时 workspace，finalize 不再加权一次

vLLM 将两组工作空间交给专家模块，当前 gated SiLU shapes 为 `workspace13[M,K,max(I,D)]` 与 `workspace2[M,K,max(2I,D)]`，另有 `[M,D]` 输出视图。`FusedMoEKernelModularImpl._allocate_buffers` 通过 workspace manager 获取两块可同时使用的缓冲；`_resize_cache` 仅裁取并 view 所需容量。[V12][V13][V14]

- 第一投影的 `intermediate_cache1[M,K,2I]` 和 down 输出的 `intermediate_cache3[M,K,D]` 都使用 `workspace2`，因为 down 开始前已消费完 gate/up。
- 普通 activation cache `[MK,I]` 使用 `workspace13`，最终 routed `[M,D]` 输出复用同一 common workspace；融合 SiLU 量化分支则另行分配 FP8 中间值和 scales，不使用该 BF16 activation view 存放结果。
- `apply` 另建最终输出张量；`TritonExperts.finalize_weight_and_reduce_impl` 返回 `TopKWeightAndReduceNoOP`，因为专家内已乘 gate 并归约。所选 CUDA finalize 只把 fused output 复制到该输出，不能再次乘 top-k weights。[V12][V13][V18][V23]

这些 `cache1/2/3` 名称表示本次 MoE 运算的临时张量和容量复用，不是跨生成 step 保存 hidden states，也不是 KV cache。上面核对的是取 buffer 的接口、shape 和存活次序，没有测量 workspace manager 在整机 serving 中的峰值容量或 graph 内存池行为。

独立共享 MLP 仍对所有原 token 计算，不参与 routed affinity top-k。当前关闭 shared auxiliary stream，`SharedExperts` 选 `NO_OVERLAP`，由 runner 在专家调用前运行并保存当前输出；runner 取出 shared/routed 结果，对 BF16 routed 结果乘 $\rho$ 后与 shared 相加，TP=1 无最后 all-reduce。SGLang 普通 inplace 专家路径必须在原输入被覆盖前计算 shared；vLLM modular apply 则为 routed 返回新输出。两者均不能把 shared 当成普通可遗漏的 routed expert，也不能再在 FFN 内自行添加 decoder residual。[V1][V6][V12][V24]

对照到这里，核心量化公式与配对语义可以复用，接口所有者、$\rho$ 的施加位置、第一次量化所在阶段、activation 融合及临时缓冲生命周期需要逐框架核对。本次对 vLLM 同样只核读固定源码；未执行模型、编译 kernel、验证硬件支持或比较数值误差/速度。EP 通信、其他 FP8 后端、LoRA、CUDA graph 和多 stream 仍未沿另一条调用链完整核验。

## 来源身份

- FlashInfer Release v0.6.17（`https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.6.17`），snapshot-2026-08-20，获取于2026-08-20T02:38:02+08:00；统一 FP4 MoE/预路由/shared/SiTU 声明与 SM90 CUTLASS MoE 撤回，未核 PR、匹配实现或运行。

第 1–7 节的 SGLang 源码来自 sgl-project/sglang 固定 commit（`https://github.com/sgl-project/sglang/tree/2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc`），`2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc`。本地归档身份为 `sgl-project-sglang-2f730e29`；目录日与历史获取日分别登记，目录日期不表示该 commit 的发布日。数字索引对应本页实际核读文件：

| 引用 | 固定源码文件 | 核读位置与责任 |
| --- | --- | --- |
| [1] | `python/sglang/srt/models/deepseek_v2.py` | `MoEGate`、`DeepseekV2MLP`、MoE 构造与 `forward_normal`、shared fusion 判定 |
| [2] | `python/sglang/srt/layers/moe/topk.py` | Standard 输出、grouped/biased 选择、CUDA 分派、融合共享权重说明 |
| [3] | `sgl-kernel/csrc/moe/moe_fused_gate.cu` | sigmoid、correction bias、组筛选、gate 恢复与归一化 |
| [4] | `python/sglang/srt/layers/moe/ep_moe/layer.py` | `get_moe_impl_class`，只核对普通分支选择 FusedMoE |
| [5] | `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` | 方法/runner/dispatcher 创建、w13/w2 装载、forward 与 scale 融合标志 |
| [6] | `python/sglang/srt/layers/quantization/fp8.py` | 配置边界、MoE 张量创建、块权重后处理、Triton quant info 与 apply |
| [7] | `python/sglang/srt/layers/moe/token_dispatcher/standard.py` | 普通 dispatch/combine；未展开 FP4 all-gather 分支 |
| [8] | `python/sglang/srt/layers/moe/moe_runner/runner.py` | fused_func 优先执行与 runner 接口 |
| [9] | `python/sglang/srt/layers/moe/moe_runner/triton.py` | none/triton 注册、quant info、fused_experts 调用 |
| [10] | `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py` | 配置/align、两次 GEMM、SiLU-and-mul、加权与归约 |
| [11] | `python/sglang/srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py` | buffer shape、padding sentinel 与原生调用 |
| [12] | `sgl-kernel/csrc/moe/moe_align_kernel.cu` | 专家计数、前缀和、padding 与配对索引写入 |
| [13] | `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py` | activation 量化、间接载入、K 组 scale、FP32 累加和输出 gate |
| [14] | `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py` | block shape 与默认 tile 配置；未核读/测量全部调优 JSON |
| [15] | `python/sglang/srt/layers/quantization/fp8_kernel.py` | CUDA 组量化包装、尺度 shape/dtype 和整除条件 |
| [16] | `python/sglang/jit_kernel/per_token_group_quant_8bit_v2.py` | custom-op 到固定 C++ 数值核的接口 |
| [17] | `python/sglang/jit_kernel/csrc/gemm/per_token_group_quant_8bit_v2.cuh` | 非 UE8M0 E4M3 的 amax、尺度、限幅与编码；未编译 |
| [18] | `python/sglang/srt/layers/moe/utils.py` | `get_moe_weight_sizes`，普通未打包形状 |
| [19] | `python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py` | 仅核读 `maybe_fuse_routed_scale_and_shared_add` 的普通共享相加职责 |


vLLM 对照来自 vllm-project/vllm 固定 commit（`https://github.com/vllm-project/vllm/tree/568afb3a13806beb53bb2e6bd518269357b237c0`），`568afb3a13806beb53bb2e6bd518269357b237c0`，本地归档身份为 `vllm-project-vllm-568afb3a`。该条目的历史获取日与统一归档日同样分别登记；以下 V 编号只对应第 8 节实际核读的文件与位置。

| 引用 | 固定源码文件 | 核读位置与责任 |
| --- | --- | --- |
| [V1] | `vllm/model_executor/models/deepseek_v2.py` | `DeepseekV2MLP/DeepseekV2MoE` 的 gate、shared、forward，以及 decoder 覆盖 scale 参数 |
| [V2] | `vllm/model_executor/layers/fused_moe/layer.py` | `FusedMoE` factory，router、RoutedExperts、MoERunner 与 routed scaling 分配 |
| [V3] | `vllm/model_executor/layers/fused_moe/router/gate_linear.py` | 普通 router 权重、`quant_config=None` 与 GEMM 分派条件 |
| [V4] | `vllm/model_executor/layers/fused_moe/router/router_factory.py` | 有效 grouped router 选择；未展开 simulator/zero/其他 router |
| [V5] | `vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py` | biased sigmoid 分组选择、归一化后乘 rho、选定非 fused 路由 |
| [V6] | `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` | internal gate、TopK、专家调用、shared 汇合及单设备 dispatch 边界 |
| [V7] | `vllm/model_executor/layers/fused_moe/routed_experts.py` | gate/up 加载顺序与 `forward_modular` 的参数所有权 |
| [V8] | `vllm/model_executor/layers/quantization/fp8.py` | serialized/dynamic 限制、权重/尺度创建、后端与 kernel 初始化、apply |
| [V9] | `vllm/model_executor/layers/fused_moe/oracle/fp8.py` | 明确 Triton backend 的受支持配置检查、原布局保留和 kernel 构造 |
| [V10] | `vllm/model_executor/layers/fused_moe/all2all_utils.py` | 仅核读无 all-to-all 且 DP=1 的 prepare/finalize 选择 |
| [V11] | `vllm/model_executor/layers/fused_moe/prepare_finalize/no_dp_ep.py` | 原 token 行量化、保留 top-k、同步 finalize |
| [V12] | `vllm/model_executor/layers/fused_moe/modular_kernel.py` | prepare/expert/finalize、workspace 分配接口和输出 view 生命周期 |
| [V13] | `vllm/model_executor/layers/fused_moe/experts/triton_moe.py` | FP8 设备检查、两次 GEMM、对齐、activation 融合、workspace shapes 与 NoOP finalize |
| [V14] | `vllm/model_executor/layers/fused_moe/utils.py` | `_resize_cache` view 与 `moe_kernel_quantize_input/_fp8_quantize` |
| [V15] | `vllm/model_executor/layers/quantization/utils/fp8_utils.py` | `per_token_group_quant_fp8` 的 shape、连续性与 CUDA 调用 |
| [V16] | `vllm/model_executor/layers/fused_moe/fused_moe.py` | aligned/naive 配对分派、间接载入、块 scale 与 gate 施加 |
| [V17] | `vllm/model_executor/layers/fused_moe/moe_align_block_size.py` | 配对索引缓冲、padding sentinel 与原生接口 |
| [V18] | `vllm/_custom_ops.py` | `silu_and_mul_per_block_quant` 的 FP8/scales 分配与 C++ 调用 |
| [V19] | `csrc/libtorch_stable/quantization/fused_kernels/fused_silu_mul_block_quant.cu` | FP32 SiLU×up、组 amax、最小尺度、量化和输入条件 |
| [V20] | `csrc/libtorch_stable/moe/moe_align_sum_kernels.cu` | 专家计数/前缀和/配对写入与 FP32 moe_sum；未遍历全部优化变体 |
| [V21] | `csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu` | 普通 E4M3 行组尺度与编码 |
| [V22] | `vllm/utils/deep_gemm.py` | 仅核读 E8M0 scale 的全局启用判定，不展开 DeepGEMM 专家执行 |
| [V23] | `vllm/model_executor/layers/fused_moe/topk_weight_and_reduce.py` | `TopKWeightAndReduceNoOP` 的已加权已归约契约和输出复制 |
| [V24] | `vllm/model_executor/layers/fused_moe/runner/shared_experts.py` | `NO_OVERLAP` 选择、单次 shared 调用及当前输出存取 |

编号、配置和调用定位表示源码核读范围，不表示所有文件的全部分支已经验证。正文的索引/成本例子及值语义公式为整理者解释，未作为上游模型实验数据。
