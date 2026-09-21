---
title: vLLM 算子设计：Attention 后端、分页计算与图执行
type: implementation
tags:
  - serving
  - kernels
  - kv-cache
  - attention
  - performance
sources:
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/models/llama.py
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/layers/attention/attention.py
  - raw/repositories/2026-09-21/vllm/source/vllm/v1/attention/backend.py
  - raw/repositories/2026-09-21/vllm/source/vllm/v1/attention/backends/flash_attn.py
  - raw/repositories/2026-09-21/vllm/source/vllm/v1/attention/backends/triton_attn.py
  - raw/repositories/2026-09-21/vllm/source/vllm/v1/attention/ops/triton_unified_attention.py
  - raw/repositories/2026-09-21/vllm/source/vllm/v1/attention/ops/triton_attention_helpers.py
  - raw/repositories/2026-09-21/vllm/source/vllm/v1/worker/gpu_model_runner.py
  - raw/repositories/2026-09-21/vllm/source/vllm/v1/cudagraph_dispatcher.py
  - raw/repositories/2026-09-21/vllm/source/docs/design/cuda_graphs.md
updated: 2026-09-22
---

# vLLM 算子设计：Attention 后端、分页计算与图执行

推理框架中的 Attention 接收本轮 query，却依赖跨轮保存的 KV 状态。一个可用的算子实现，必须同时理解请求边界、位置、缓存布局、写入时序、后端能力与图执行方式。设备端的计算优化就在这些条件之内进行。

本页接续[vLLM 推理执行](vllm-inference-execution.md)，选择 Llama 模型层、通用 Attention 接口、FlashAttention 接入和 Triton unified attention 的普通因果注意力路径。重点是框架与算子怎样衔接，不罗列全部后端或追踪版本迁移。张量的 shape/stride 基础见[张量布局与接口](../fundamentals/operators/tensor-layout-and-kernel-contracts.md)，归约与分块基础见[GPU 算子的计算模式](../fundamentals/operators/gpu-kernel-computation-patterns.md)。

普通 Llama decoder 在 Attention 两侧还维护残差与 RMSNorm，MLP 则沿 gate/up 投影、门控激活和 down 投影执行。学习这些较小的现成算子，可以先看[RMSNorm 的 CUDA/Triton 行归约](rmsnorm-cuda-triton-kernels.md)和[门控激活的工作映射与融合](gated-activation-cuda-triton-kernels.md)，再回到下面带 KV 状态的 Attention。

## 1. 模型层先组织计算，再交给具体 Kernel

`model_executor/models/llama.py` 的 `LlamaAttention.forward` 是一条可读的主线：

```text
hidden_states [本轮 token 数, hidden]
  → QKVParallelLinear
  → split(Q, K, V)
  → 用 positions 对 Q/K 做 RoPE
  → Attention(Q, K, V)
  → RowParallelLinear 输出投影
```

QKV 投影将相关输出放在同一次线性层调用中；`LlamaMLP` 类似地用 `MergedColumnParallelLinear` 组织 gate/up，再做 `SiluAndMul` 和 down projection。这是模型层对计算和张量的组织，不能据此声称整个模块必定成为一个设备 kernel。

量化配置传入这些线性层，具体格式与内核选择由相应实现承接，已在[AWQ 实现页](awq-implementation.md)展开。Attention 还要带上当前 TP rank 的 query/KV head 数：Llama 初始化逻辑在 KV head 数足够时切分 KV heads，不足时允许跨 TP rank 复制。不能一律把本地 KV heads 写成全局数量除以 TP 后的零或小数。

由此得到一个算子设计原则：先确定模型层传入的是全量张量还是并行分片、融合输出中的哪个切片、是否带量化参数，再讨论 tile 大小。单独看论文中的矩阵形状不足以确定实际接口。

## 2. Attention 的输入还包含模型签名之外的状态

`Attention.forward` 将 Q/K/V 组织为 token、head、head dimension 三个维度；runner 在模型前向前用 `set_forward_context` 安装每层元数据与 slot mapping。Attention 根据 layer name 取得对应的缓存和元数据。因此表面的 `attn(q,k,v)` 并不表示它是仅依赖三个参数的无状态函数。

`CommonAttentionMetadata` 提供公共信息，再由 backend 的 metadata builder 转成具体格式。普通单设备因果路径需要区分：

| 信息 | 解决的问题 | 不能替代什么 |
| --- | --- | --- |
| `query_start_loc` | 本轮紧凑 query 中每个请求的起止 | 不能表示历史 KV 总长度 |
| `seq_lens` | 每个请求本轮可见的 KV 长度 | 不能定位 KV 的物理页 |
| `block_table` | 逻辑 KV 页到物理页的映射 | 不能单独给出本轮哪些 token 要写入 |
| `slot_mapping` | 新 K/V 的写入槽位 | 不能替代历史 KV 的完整读取块表 |
| scale、head 数、stride、mask | 数值解释、布局与可见性 | 不能只靠一个 dtype 推断出来 |

若某请求已有 c 个历史 token，本轮有 q 个 query，第 j 个 query 的绝对位置是 c+j；普通因果注意力允许它访问键位置 $0\le k\le c+j$。所以 q=3、KV 长度=5 时，三个 query 分别能看见前 3、4、5 个位置。直接在一个 3×5 矩阵上使用未经偏移的左上角三角 mask，会错误地丢掉历史上下文。

此关系由 runner 的 positions/seq_lens 构造以及 `triton_attention_helpers.compute_kv_seq_mask` 的绝对位置比较支持。前缀 LM、滑窗和 encoder 的 mask 另有规则，本节不将它们并入普通因果例子。

## 3. KV 写入、读取与编译依赖

对写入和计算分开的 backend，Attention 先经 `unified_kv_cache_update` 写入本轮 K/V，再经 `unified_attention_with_output` 计算输出。以 `FlashAttentionImpl` 的普通 decoder 分支为例：

1. `do_kv_cache_update` 使用 slot mapping，将新 K/V scatter 到分页缓存，传递 KV dtype 与 K/V scale。
2. `forward` 将缓存视图、query 累积长度、KV 序列长度、块表、softmax scale 和因果条件交给 `flash_attn_varlen_func`。
3. 结果写入调用方提供的 output，供后续输出投影使用。

`FlashAttentionBackend.get_kv_cache_shape` 在所读快照中给出的逻辑缓存形状为 `[blocks, kv_heads, block_size, 2*head_dim]`，K/V 沿最后一维打包；`get_kv_cache_stride_order` 还区分物理布局。这个例子说明 **逻辑 shape、物理 stride、K/V 视图是三件事**，不能推广成所有 Attention 后端共用同一连续布局。

新 token 的 Attention 通常要读到刚写好的自身 K/V，写入必须发生在读取之前。这里缓存通过上下文访问，不直接作为普通可变参数出现在两个 custom op 的公共签名里；`unified_kv_cache_update` 返回一个空 tensor，作为 `kv_cache_dummy_dep` 传给 Attention，显式建立供编译器保留的顺序依赖。它并不承载 KV 数据，也不意味着增加一次全设备同步。

`unified_attention_with_output` 注册时声明 output 等被修改参数，并提供 fake implementation 供编译期处理；fake 实现不会运行真实注意力。对算子接入而言，数值结果、外部状态副作用和编译器所见依赖都要正确。只检查单次 eager 输出，无法覆盖被编译调度后读到旧缓存的问题。

某些 backend 将 KV 更新包含在 forward 内部，跨层共享 KV 时也可能跳过重复写入；是否分为两个操作由 `forward_includes_kv_cache_update` 等条件决定，不能要求所有后端照搬这个拆分。

## 4. 后端接口为什么比“选一个最快 Kernel”更宽

`AttentionBackend` 同时提供实现类、metadata builder、缓存 shape/stride 约定和能力验证。`validate_configuration` 检查 head size、计算与 KV dtype、块大小、硬件、attention 类型和滑窗等组合。初始化选出合法后端之后，具体后端内部还可以按本轮工作负载选择执行路径。

这意味着替换 backend 需要保持一组共同契约：缓存写入与读取解释一致；metadata builder 与设备代码解释同一个边界；未支持的能力有明确处理；图执行能力如实声明。将一个只接收连续 Q/K/V 的测试 kernel 直接替换分页后端，缺失的是状态与地址转换，不能只在外围补一次函数调用。

FlashAttention 接口和下面的 Triton kernel 是两条可选后端路径，不是 FlashAttention 再调用本页的 Triton kernel。`backends/triton_attn.py` 将元数据和缓存传给 `ops/triton_unified_attention.py::unified_attention`，后者选择 grid 并启动设备程序。

## 5. Triton 内核怎样连接 GQA、分页和在线 softmax

在普通路径中，设备 grid 的前两维组织 query block 与 KV head；需要沿历史维度分段时再加第三维。设每个 KV head 对应 r 个 query heads，kernel 的一组行同时容纳 query 位置和共享这个 KV head 的不同 query heads：

$$\mathrm{query\_pos}=q_{\rm block}B_Q+\lfloor m/r\rfloor,\qquad
\mathrm{query\_head}=h_{KV}r+(m\bmod r).$$

这里 m 为 tile 内行号，$B_Q$ 为该 tile 容纳的 query 位置数。`kernel_unified_attention` 的 `query_pos`、`query_offset_1` 就体现了这种映射。**教学例子：**r=4、$B_M=16$ 时，行可组成 4 个 query 位置 × 每位置 4 个 query heads。它们复用同一个 KV head 的 K/V tile；decode 只有一个 query 位置时，多余位置由 mask 屏蔽。具体 $B_M$、warp 和 tile 参数属于工作负载选择，不能当作普适最优值。

沿历史 KV 维循环时，逻辑位置 t 先经块表查找物理块，再用块内偏移与 stride 读取 K/V。分页提供物理存储的间接寻址；分块计算则决定一次取多少 K/V、保留多少 query 和累加器。KV 页大小与计算 tile 大小职责不同，仅在特定加载路径存在整除或对齐条件。

若把全部 score 存成 $q\times\ell$ 矩阵再 softmax，会产生大的中间存储。源码按 KV tile 计算 score，并用在线 softmax 只维护每行的最大值 M、指数和 L 与向量累加器 A。对一个新 tile 的 score $s_j$ 和 value $v_j$，下面是普通非量化注意力的等价数学解释：

$$M'=\max(M,\max_j s_j),\qquad \alpha=e^{M-M'},$$
$$p_j=e^{s_j-M'},\qquad L'=\alpha L+\sum_jp_j,\qquad A'=\alpha A+\sum_jp_jv_j.$$

所有 tile 完成后输出 $A/L$。M 变化时，旧 L 和 A 同时乘 $\alpha$，让新旧累计处于同一个指数基准；只重缩放分母会改变结果。因果不可见位置用负无穷 score 排除。`softmax_step` 还处理整行被 mask 的中间 tile，避免直接计算负无穷减负无穷产生 NaN。

对应源码先 `tl.dot(Q,K)`，调用 `softmax_step`，重缩放 `acc`，再累加 `tl.dot(P,V)`。FP32 score/累加器、P 转换到乘法输入 dtype 以及低比特 scale 路径仍会影响浮点误差；数学等价不保证不同 backend 位级一致。这里展开普通路径，不声称验证了该文件的所有 KV 量化、TMA 或稀疏掩码分支。

## 6. decode 为什么有时拆成多段再归约

query 很少、历史很长时，仅按 query block × KV head 分工可能无法提供足够并行工作。所读 `unified_attention` 在单 token query、请求数不超过配置阈值、有临时缓冲且未启用 batch invariance 等条件下，允许使用三维 grid：不同程序并行处理历史 KV 的不同 segment，随后 `reduce_segments` 合并。混合 prefill 或 query 长度大于 1 时走该 launcher 的二维路径。

每段 s 保存局部最大值 $m_s$、指数和 $l_s$ 与**尚未除以 $l_s$ 的向量累计** $a_s$。全局结果必须统一基准：

$$m=\max_s m_s,\qquad
o=\frac{\sum_s e^{m_s-m}a_s}{\sum_s e^{m_s-m}l_s}.$$

另一种等价中间表示是段内归一化输出与 log-sum-exp，[SGLang decode 归约](sglang-attention-operator-design.md)解释其加权公式。两种表示表达相同的全局 softmax，但缓冲语义不同，不能直接互换。

这正是 `reduce_segments` 的重缩放与归约关系，不能将各段的归一化输出直接平均。一个反例是两段各有一个 value，分别为 0 和 10，对应 score 为 0 和 $\log3$：正确输出为 7.5，平均两段输出会得到 5。

分段提供更多并行工作，代价是临时结果写读、额外归约 kernel 和不同的浮点求和次序。历史短或请求数本已足够时，新增开销可能得不偿失。源码中的条件是设计取舍的实例，未测量前不能声称多分段总会更快。与 tile/资源的关系可继续读[配置选择与自动调优](kernel-configuration-and-autotuning.md)。

## 7. 动态批次怎样使用 CUDA Graph

`docs/design/cuda_graphs.md` 将编译与 CUDA Graph 捕获分开：编译关注算子图的生成和优化，CUDA Graph 关注捕获并重放设备工作以减少重复提交开销。runner 准备持久输入缓冲，通过 `CudagraphDispatcher` 将本轮批次匹配到可用捕获规格。

运行模式分为 FULL、PIECEWISE、NONE。FULL 覆盖完整前向；PIECEWISE 让可捕获片段重放，Attention 等未纳入部分按对应路径执行；NONE 不使用 CUDA Graph。`AttentionCGSupport` 区分混合批次、统一 query 长度、单 token decode 等支持能力。dispatcher 在允许且已存在的 key 中优先 FULL，其次 PIECEWISE，没有合适项时通常回到 NONE。

**教学例子：**本轮有 13 个有效 token，如果可用捕获规格是 16，则可将执行缓冲补到 16；有效长度、请求边界和 slot mapping 仍必须表达只有前 13 个是实际工作。`GPUModelRunner._get_slot_mappings` 将 padding 部分设为 -1，`_prepare_inputs` 将补齐的 query 起点保持非递减，防止把补齐部分误写成真实 KV 或解释成额外请求。

因此动态批次并不要求捕获所有可能的序列长度组合：形状规格用于图匹配，实际长度与块表作为本轮数据更新。但合法模式、持久缓冲和 padding 行为必须满足对应 backend 的约定。逻辑上紧凑的 token 批次，与图执行时存在少量 padding，可以同时成立。

这也形成算子优化的另一层收益：某个 kernel 的设备时间略短，却引入捕获不支持的行为，整轮可能更慢；反之支持捕获也不能补救错误的缓存更新或昂贵的临时读写。需要分别测设备计算、整轮执行和服务指标，测量方法见[正确性与性能测量](kernel-correctness-and-benchmarking.md)。

## 8. 从框架契约选择验证用例

针对本页的普通路径，验证应覆盖接口交界，而不仅是一个连续张量上的 Attention 数值：

- 同一输入单独执行与放进长短混合批次，检查请求边界、位置和 mask。
- 未命中与命中前缀的输出，完整 prefill 与分段 prefill 的后续输出，检查复用进度与缓存写入。
- 刚好填满一页和跨页追加，使用不连续物理块号，检查 slot mapping 与 block table。
- 无 padding 与捕获规格 padding，eager 与可用图模式，检查补齐行不修改有效 KV。
- 二维与分段路径，检查统一 softmax 归约并使用合适的数值容限。

这些是由源码契约推导的验证设计。本轮实际完成代码研读和 CPU 教学模型核对，覆盖变长边界、槽位、在线 softmax 与 segment 合并；未编译 Triton/FlashAttention、未运行 vLLM 模型、未验证真实 GPU 图捕获与性能，也未将以上集成用例声称为已运行。

## 来源身份

| 来源 | 固定版本 | 使用范围 |
| --- | --- | --- |
| [vllm-project/vllm](https://github.com/vllm-project/vllm/tree/568afb3a13806beb53bb2e6bd518269357b237c0) | `568afb3a13806beb53bb2e6bd518269357b237c0` | Llama 层组织、Attention 接口与元数据、FlashAttention 接入、Triton 普通分页/在线 softmax/分段路径、CUDA Graph 文档和分派。 |
