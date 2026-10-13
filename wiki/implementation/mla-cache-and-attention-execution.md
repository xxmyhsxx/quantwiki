---
title: MLA 缓存与 Attention 执行：SGLang、FlashInfer 与 vLLM
type: implementation
tags:
  - attention
  - kv-cache
  - serving
  - low-rank
sources:
  - raw/repositories/2026-09-21/sglang
  - raw/repositories/2026-09-21/flashinfer
  - raw/repositories/2026-09-21/vllm
updated: 2026-10-07
---

# MLA 缓存与 Attention 执行：SGLang、FlashInfer 与 vLLM

MLA 的历史状态是联合 KV 潜变量与位置 key；执行时既可以临时展开为多头 K/V，也可以让 query 进入潜空间，直接读取压缩表示。两条计算路径可以使用同一缓存池，却具有不同的矩阵形状、临时内存和舍入位置。本页把 [MLA 的代数与缓存对象](../fundamentals/model/multi-head-latent-attention.md)接到实际模型层、槽位布局和 Attention 调用。 第 1–8 节核读 SGLang 调用端，第 9–14 节核读独立 FlashInfer 库快照，第 15–20 节以 vLLM 普通 CUDA 路径作针对性对照；三个固定来源的版本与实际执行范围分别保留。

公式、容量与索引例子是依据源码整理的解释；本页未安装、编译或运行三个快照，未测量误差、显存或性能。

## 1. SGLang 固定版本与所选路径

本节核对 SGLang commit `2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc` 的普通非 DSA DeepSeek MLA、单设备、FlashInfer backend，沿 eager 的 extend/decode 与 prefix 读取展开。DSA、推测执行、LoRA、DCP/CP 和其他硬件分支不作为本节的执行主线；遇到它们时只用于说明所选分支的条件。

该版本 `python/pyproject.toml` 固定 `flashinfer_cubin==0.6.12`、`flashinfer_python[cu13]==0.6.12`。SGLang 调用端可以独立核读，但另一个 FlashInfer 固定快照不能自动视为它的部署依赖；版本间接口分析也不能作为这两个快照已联合安装或实际运行的证据。

阅读前置是 [SGLang 的请求与 KV 槽位](sglang-inference-execution.md#5.%20%E8%AF%B7%E6%B1%82%E4%BD%8D%E7%BD%AE%E6%80%8E%E6%A0%B7%E6%98%A0%E5%B0%84%E5%88%B0%E5%AE%9E%E9%99%85%20KV)。该页解释请求如何取得 `out_cache_loc`；这里回答这些位置保存的元素是什么，以及读取端怎样解释它们。[普通 MHA/GQA 算子页](sglang-attention-operator-design.md)中的独立 K/V buffer 不直接代替 MLA 的联合表示。

## 2. 从模型维度到缓存和工作张量

采用源码的行张量习惯，令本轮 token 总数为 $N$、本地 query heads 为 $H$，模型隐藏宽度为 $D$，内容 key 宽度为 $d_K$，value 宽度为 $d_V$，位置宽度为 $d_R$，KV latent 宽度为 $d_c$，可选 query latent 宽度为 $d_q$。本节单设备下本地 heads 等于模型 heads。$N$ 是本轮所有请求的 token 总数，不等于请求数；普通 decode 才通常每请求贡献一行。

`DeepseekV2AttentionMLA.__init__` 将两条执行路径分别交给 `attn_mha` 与 `attn_mqa`，但二者使用同一个 layer id。源码名称 `k_nope` 在吸收路径指归一化 KV latent，不能因为名称中有 k 就认作 $d_K$ 维的已展开内容 key。

| 对象 | 形状或宽度 | 在所选路径中的作用 |
| --- | --- | --- |
| `hidden_states` | $[N,D]$ | 本轮模型输入 |
| 融合低秩投影的结果 | $[N,d_q+d_c+d_R]$ | 依次拆成 query latent、KV latent、共享位置 key；不压缩 query 的配置使用单独投影 |
| `q_b_proj` 之后的 query | $[N,H,d_K+d_R]$ | 每头的内容 query 与位置 query |
| 归一化后的 `k_nope` / `kv_a` | $[N,1,d_c]$ 或 $[N,d_c]$ | 所有 heads 共用的 KV latent |
| RoPE 后的 `k_pe` | $[N,1,d_R]$ | 所有 heads 共用的位置 key |
| 展开路径的 K/V | $[N,H,d_K+d_R]$ / $[N,H,d_V]$ | 临时供 MHA attention 使用，不是持久缓存布局 |
| 吸收后的 query | $[N,H,d_c]$，另带 $[N,H,d_R]$ | 与 latent 和位置 key 分别做点积 |
| 潜空间 attention 输出 | $[N,H,d_c]$ | 每头自己的 latent 加权和；随后上投影至 $d_V$ |

融合投影由 `prepare_qkv_latent` 生成，并作为 `LayerCommunicator` 的 `qkv_latent_func` 接入；`forward_absorb_prepare` / `forward_normal_prepare` 再从 attention context 取出并拆分。归一化 query latent 后才能进入 `q_b_proj`；KV latent 则经过 `kv_a_layernorm`。这些 RMSNorm 留在实际前向中，不因下面的矩阵吸收而变成一个离线常数矩阵。（`models/deepseek_v2.py::DeepseekV2AttentionMLA`、`DeepseekV2DecoderLayer.__init__`。）

## 3. 投影吸收在该实现中怎样执行

### 3.1 将内容 key 投影移到当前 query

下面是依据源码形状写出的等价线性代数解释，不是新的模型实验。令第 $i$ 头的 key 上投影为 $U_{K,i}\in\mathbb R^{d_K\times d_c}$，value 上投影为 $U_{V,i}\in\mathbb R^{d_V\times d_c}$，归一化后的历史 latent 行向量为 $c_j$。展开的内容 key 是 $c_jU_{K,i}^{\top}$，内容分数可写为

$$
q^C_{t,i}(c_jU_{K,i}^{\top})^{\top}
=(q^C_{t,i}U_{K,i})c_j^{\top}
=\bar q_{t,i}c_j^{\top}.
$$

因此需要变换当前 query，而非为每一步重新展开所有历史 key。`forward_absorb_prepare` 的普通非量化 BMM 对 $[H,N,d_K]$ 的 query 乘 `w_kc` 的 $[H,d_K,d_c]$，得到 $[H,N,d_c]$，再转置为 $[N,H,d_c]$。RoPE 只施加到另一个 $d_R$ 维分支；FlashInfer 路径把内容和位置张量分开传入，无需在模型侧先拼成一个新 query 张量。

第 $i$ 头的完整 score 是

$$
\ell_{t,j,i}=s\left(\bar q_{t,i}c_j^{\top}+q^R_{t,i}(k^R_j)^{\top}\right).
$$

`DeepseekV2AttentionMLA.__init__` 的基础 $s=(d_K+d_R)^{-1/2}$；配置 YaRN 时还由 `compute_mla_mscale_scaling` 调整。吸收后 query 的内容宽度变成 $d_c$，但不能据此把 scale 改为 $(d_c+d_R)^{-1/2}$。这会改变模型的概率分布，而不是一个布局修正。

### 3.2 Value 上投影发生在历史加权之后

令 $a_{t,j,i}$ 为同一 score/可见范围下的 attention 权重，则

$$
\sum_j a_{t,j,i}(c_jU_{V,i}^{\top})
=\left(\sum_j a_{t,j,i}c_j\right)U_{V,i}^{\top}
=z_{t,i}U_{V,i}^{\top}.
$$

`forward_absorb_core` 给 `attn_mqa` 的 key latent 与 value latent 是同一份 `k_nope`，另用 `k_rope` 参与分数；attention 返回 $[N,H,d_c]$ 的 $z$。普通 BMM 将 $[H,N,d_c]$ 乘 `w_vc` 的 $[H,d_c,d_V]$，得到每头 $d_V$ 输出，转为 $[N,Hd_V]$ 后调用 `o_proj`。

这里的“吸收”是改变计算结合次序：此固定实现仍显式执行 query 的 key BMM、输出的 value BMM 和 `o_proj`，没有由类名推出它们都离线融合成一个 GEMM。潜变量共享也不意味着各头共用一个 $z$，因为每头的 query 与概率不同。

### 3.3 权重加载和量化会改变吸收路径

`DeepseekV2WeightLoaderMixin.post_load_weights` 将 `kv_b_proj` 的逻辑权重 $[H(d_K+d_V),d_c]$ 拆为各头的 key/value 部分，再组织成 `w_kc` / `w_vc`。普通 BMM 分支通过 `transpose` 与 `contiguous` 准备底层存储，因此知道逻辑 shape 仍不足以把权重当作任意连续布局。DeepGEMM 分支采用另一组转置和 scale 布局，不能复用普通 BMM 的物理解释。

该函数还揭示 checkpoint 格式不等于实际 BMM 格式：AWQ packed 的 `kv_b_proj` 可先反量化；某些块缩放 FP8 路径会恢复到 BF16，某些 FP8 路径则转换为单 tensor scale 后调用 `bmm_fp8`。满足专门条件时可保留 DeepGEMM 的块 scale。`forward_absorb_prepare/core` 的 `self.w_scale`、`w_scale_k/v` 与动态 activation scale 属于这些**权重/矩阵乘路径**，不是普通 MLA cache 的 per-token scale。

吸收的实数恒等式不能证明量化展开路径与量化吸收路径逐位相同。矩阵表示、重新量化、中间 dtype 与累计次序都可能不同。

## 4. 持久缓存保存什么，槽位怎样写入

### 4.1 一层一份联合 buffer

普通 `MLATokenToKVPool._create_buffers` 为每层创建

$$[S_{\rm alloc}+P,\ 1,\ d_c+d_R]$$

的 `kv_buffer`。$S_{\rm alloc}$ 是池容量，$P$ 是 page size，额外 padding 包含供虚拟 token 写入的槽位。末轴前 $d_c$ 元素是归一化后的 latent，后 $d_R$ 元素是 RoPE 后的位置 key；大小为 1 的轴不是 $H$ 份潜变量。

`get_key_buffer` 返回完整联合 buffer；`get_value_buffer` 返回其前 $d_c$ 元素的 view。接口中出现两个 getter，不表示底层分配了两份独立 K/V。后续 `attn_mha` 或 `attn_mqa` 都按相同 layer id 取到这一份历史。

**教学容量计算：**若 $d_c=512,d_R=64$，普通 BF16 一 token 的联合载荷是 $576\times2=1152$ 字节；不是再乘 K/V 的 2，也不是再乘 query heads。预分配载荷则为

$$M_{\rm pool}=N_l(S_{\rm alloc}+P)(d_c+d_R)\,\operatorname{sizeof}(\texttt{store\_dtype}).$$

它尚未包括请求映射、allocator、workspace、展开 K/V 和 dtype 转换临时量。有效 token 变少时，预分配池不因此自动缩小。量化预算口径见 [从位宽到可执行存储格式](../theory/kv-cache-quantization-objects-and-granularity.md#11.%20%E4%BB%8E%E4%BD%8D%E5%AE%BD%E5%88%B0%E5%8F%AF%E6%89%A7%E8%A1%8C%E5%AD%98%E5%82%A8%E6%A0%BC%E5%BC%8F)。

### 4.2 写入位置与当前分支的先后关系

`out_cache_loc[n]` 是本轮第 $n$ 行应写到的槽位。`set_mla_kv_buffer` 先按 pool dtype 处理两部分，然后把 latent 与 RoPE key 写入同一槽位的两段。`mem_cache/triton_ops/mla_buffer.py::set_mla_kv_buffer_kernel` 按 `loc * buffer_stride + offset` 写，不要求本轮各请求的槽位连续。

`set_mla_kv_buffer_triton` 这个名字不能作为必用 Triton 的证据：该固定版本在写入 token 数至少 768、硬件/对齐条件允许且没有 DCP 时可调 JIT CUDA 批量写；其余路径调用 Triton。这里核对 wrapper 的选择条件和普通 Triton 的地址语义，没有核读该 JIT CUDA 实现或验证其运行收益。

两种模型路径的写入责任不同：

- 展开 MHA 路径在 `forward_normal_prepare` 已通过 `_set_mla_kv_buffer` 保存 normalized latent 和 rotated key，随后 `attn_mha(..., save_kv_cache=False)`；临时展开 K/V 不覆盖联合缓存。
- 吸收 MLA 路径由 `FlashInferMLAAttnBackend.forward_extend/forward_decode` 根据 `save_kv_cache` 在计算之前写入，然后读取缓存。普通 decode 的完整历史因此包含当前 token。

这个顺序意味着缓存表示会参与当前读取的路径需与“先用未转换的新 K/V 完成计算，再写低精度历史”的路径区分。相同最终 cache shape 不能保证本轮数值路径相同，生命周期的一般问题见 [缓存生命周期与数值路径](../theory/kv-cache-quantization-objects-and-granularity.md#12.%20%E7%BC%93%E5%AD%98%E7%9A%84%E7%94%9F%E5%91%BD%E5%91%A8%E6%9C%9F%E4%BC%9A%E6%94%B9%E5%8F%98%E6%95%B0%E5%80%BC%E8%B7%AF%E5%BE%84)。

## 5. Prefill 与 decode 不是固定的一对实现

`dispatch_attn_forward_method` 先按 ForwardMode 选择配置中的 prefill/decode backend 名称，再经 `AttentionBackendRegistry` 决定模型级的 MHA/MLA 算法。`RadixAttention.forward` 是模型到 backend 的接口；`AttentionBackend.forward` 再将普通 decode 分派给 `forward_decode`、普通 extend 给 `forward_extend`。**模型级展开/吸收**和**backend 的阶段函数**是两层选择。

在单设备 CUDA、普通 eager FlashInfer、未显式关闭 ragged 的主线上，`attention_backend_handler.py::_handle_attention_backend` 给出的条件是：

| 条件 | 模型级路径 | 主要中间表示 |
| --- | --- | --- |
| 普通 decode | 吸收 MLA | 潜空间 query、联合 cache、潜空间输出 |
| 普通 extend，无复用前缀 | 展开 MHA；容量允许时 one-shot，否则 chunked-KV 路径 | 每头完整 K/V |
| 普通 extend，前缀总长度达到 chunk 阈值且未禁用 chunked prefix | 同上，按总序列容量选 one-shot 或逐块展开 | 完整或分块的历史 K/V |
| 普通 extend，其他前缀条件 | 吸收 MLA | 直接读取联合 cache |
| `flashinfer_mla_disable_ragged=True` | 上述普通 FlashInfer 分派选择吸收 MLA | paged latent attention |

上述阈值是整个批次的前缀总长度；one-shot 判断也是批次总序列长度与 chunk capacity 的关系，不是按每个请求独立选择。piecewise 图路径会强制 MLA，本节没有核验完整图执行。

文件 `flashinfer_mla_backend.py` 开头的说明把 ragged 开关的 true/false 用途写反了；这里依据实际 `init_forward_metadata` 和 `_handle_attention_backend` 条件解释。这个差异没有通过修改 raw 消除。

### 5.1 展开 MHA：临时恢复 K/V，保留 latent 缓存

`forward_normal_prepare` 从 normalized latent 调 `kv_b_proj`，拆出每头内容 K/V，向 key 拼入共享的 rotated `k_pe`。无 prefix 的 ragged prefill 只需本轮 token，调用因果 MHA attention。

one-shot 且有 prefix 时，先写新 latent，再按当前所有历史槽位取回 latent/RoPE，展开整个序列；query 仍只有本轮 extend 行。这会在读取端重新执行历史的上投影，不能将 one-shot 名称解释为无需 cache 或无需 prefix。

历史较长而选 chunked-KV 时，`forward_normal_chunked_kv_core` 先算当前扩展段，取得 output 与 LSE；`_chunked_prefix_attn_mha` 按 prefix chunk 索引取 latent、调用 `kv_b_proj`，临时恢复该 chunk 的每头 K/V，算出另一组 output/LSE，再用 `merge_state_v2` 合并。历史 prefix 已在所有新 query 之前，因此 prefix wrapper 使用 `causal=False`，当前扩展段使用 `causal=True`。

两部分合并的是同一个注意力归一化。按 [LSE 分段合并](sglang-attention-operator-design.md#4.%20Decode%EF%BC%9A%E6%8A%8A%E5%8E%86%E5%8F%B2%E7%BB%B4%E6%8B%86%E5%BC%80%EF%BC%8C%E5%86%8D%E6%8C%89%20LSE%20%E5%90%88%E5%B9%B6)，若两段局部结果为 $o_1,o_2$、LSE 为 $z_1,z_2$，全局结果为

$$o=\frac{e^{z_1-m}o_1+e^{z_2-m}o_2}{e^{z_1-m}+e^{z_2-m}},\qquad m=\max(z_1,z_2).$$

不能分别 softmax 后直接相加。分块恢复减少同时驻留的展开 K/V，却增加恢复与调用步骤；收益需按实际长度、设备和批次验证。这里的 prefix chunk 切分用于限制历史展开工作，与调度器把长 prompt 分多轮的 chunked prefill 不是同一个切分。

### 5.2 吸收 MLA：同一缓存供两个阶段读取

`forward_absorb_core` 在所选 FlashInfer backend 下分开传入 `q_nope_out/q_pe` 与 `k_nope/k_pe`。后端的 paged prefill 与 decode 都从 `get_key_buffer` 取得联合 buffer，再按 `layer.v_head_dim=d_c` 切成 latent 部分和位置部分，调用 `BatchMLAPagedAttentionWrapper.run`。输出宽度是 $d_c$，由模型侧继续做 value BMM 与 `o_proj`。

这里 `RadixAttention` 的 `num_kv_heads=1` 是潜空间接口；不能拿它与普通 $d_V$ 宽度 MQA 的缓存字节数直接比较。缓存共享、输出仍逐 query head，这两个事实同时成立。

## 6. 请求映射如何变成 paged MLA 的读取索引

普通 decode 的 `FlashInferMLAIndicesUpdaterDecode.call_begin_forward` 用 `seq_lens` 生成 `kv_indptr`，再由 `create_flashinfer_kv_indices_triton` 从 `req_to_token[req_pool_indices[i], :seq_lens[i]]` 复制实际槽位到 `kv_indices`。decode 的 `q_indptr=[0,1,\ldots,B]`，每请求一行。

paged extend 的 `FlashInferMLAIndicesUpdaterPrefill.call_begin_forward` 则用 `seq_lens-prefix_lens` 构造 `qo_indptr`，用完整 `seq_lens` 构造 KV 区间；传给 MLA wrapper 的 page size 为 1，`causal=True`。这里的 kernel page 是一个槽位，不因 pool 的分配 page size 为 $P$ 就把每个 `kv_indices` 当作 $P$ 个 token 的页编号。

**教学索引例子：**A、B 共享两槽位 `[8,9]`；A 本轮写 `[20,21]`，B 本轮写 `[30]`，并采用 paged MLA extend：

```text
req_to_token[A] = [8,9,20,21]    prefix=2, extend=2, seq_len=4
req_to_token[B] = [8,9,30]       prefix=2, extend=1, seq_len=3
qo_indptr      = [0,2,3]
kv_indptr      = [0,4,7]
kv_indices     = [8,9,20,21,8,9,30]
kv_len_arr     = [4,3]
out_cache_loc  = [20,21,30]
```

两请求可以重复引用 `[8,9]` 的数值，但各有自己的索引区间和 softmax。A 的两行 query 应分别只访问逻辑位置 0–2、0–3，B 的一行访问 0–2；这是因果接口需要保持的语义。此节核到 SGLang 的长度、索引和 `causal` 传参；§10 另按 FlashInfer v0.6.17 解释右对齐，保留它与 SGLang 声明的 0.6.12 依赖版本的区别。

无 prefix 的 ragged 展开路径则把连续 K/V 的分段设为 `kv_indptr=qo_indptr`，没有读取历史 cache 的必要；它仍已写 cache 供后续使用。不能将这条路径的索引范围套给 paged MLA。页式 `plan` 与 `run` 的分离表示元数据准备和设备计算是两个阶段，也不能从 `plan` 成功推出模型执行完成。

## 7. Cache dtype、矩阵乘 scale 与读取临时量

`ModelRunner.configure_kv_cache_dtype` 单独选择缓存 dtype：`auto` 可继承模型 dtype，也可因模型的 `kv_cache_quant_algo=FP8` 选择 FP8。因此 checkpoint 权重 dtype、工作张量 dtype 与 cache dtype 需要分别记录。

普通非 DSA `MLATokenToKVPool.set_mla_kv_buffer` 对 latent/RoPE 执行 `to(self.dtype)`；FP8 的底层 `store_dtype` 是 `uint8`，随后 `view` 为字节写入，读取 getter 再 `view(self.dtype)`。`view` 重新解释同一位模式，`to` 做数值类型转换；这不是一个普通非对称整数码、scale 和零点的格式。

这条普通写入分支没有按 `layer.k_scale/v_scale` 除法编码，也没有追加一组 per-token scale。所选 FlashInfer MLA 的 paged extend/decode 则调用 `get_key_buffer(...).to(q.dtype)`，`run` 未传 cache 的 k/v scale。不能将其他 MHA backend 的缩放规则、FP8 权重 BMM 的 `self.w_scale`，或 DSA 的分块缓存 scale 填入这条实际调用链。

因而应区分至少三种 scale：模型 score 的 $s$；FP8 权重/activation BMM 的反量化尺度；选定 cache 格式自己的尺度。所选普通 cache 路径支持的代码转换可以如实描述，却不证明所有 dtype/backend 组合均可运行或保持质量。

`get_key_buffer(...).to(q.dtype)` 作用于整层池，不只是当前 `kv_indices` 的有效槽位。dtype 不同时可能物化一份整池工作 dtype 副本，dtype 相同时可以复用原张量；读取索引仍决定 attention 使用哪些槽位。源码因此提示应分别核算：持久池容量、转换临时量、attention 实际读取范围、展开 K/V 与 workspace。缓存存为 FP8 不能据此宣称本路径直接在 attention 内核内融合 FP8 反量化，或运行峰值必然减半。

展开 MHA 的 `_get_mla_kv_buffer` 则按选定槽位生成 latent/RoPE 临时张量，再进入 `kv_b_proj`。两种恢复方式的临时量规模不同；没有 profiling 就不能把代码中的全池转换写成测得的瓶颈。

## 8. 已核读范围与验证边界

该 SGLang 声明依赖的 FlashInfer 0.6.12 底层设备程序尚未独立核验；以下核读的是库 v0.6.17 的另一份固定源码。其他未核验实现包括 JIT CUDA 缓存批量写内部、完整图执行、LoRA 修正、DSA 特殊 cache/索引、多设备与所有量化配置组合。后续比较应固定模型参数、cache dtype、backend、Q/KV 长度、prefix、阶段和是否计入转换/plan/workspace，沿 [算子测量](kernel-correctness-and-benchmarking.md)与 [服务评测](serving-performance-evaluation.md)分别报告。

## 9. FlashInfer v0.6.17：plan/run 的接口单位

以下独立核读 FlashInfer v0.6.17、commit `a0a6b019b9b27d49d209f85d028a1ae5a9b347d7` 的标准 `BatchMLAPagedAttentionWrapper`。这是另一份固定源码；上一节 SGLang 的声明依赖是 0.6.12，不能将本节的新增参数、dtype 限制或底层实现直接当作该 SGLang 部署正在使用的版本。

实际类定义在 `flashinfer/mla/_core.py`，`mla/__init__.py` 只导出入口。该 wrapper 的 fa2/fa3 路径接收**已经吸收到潜空间的 query**，输出也是 latent 加权和；它不负责 query 上投影、RoPE、Value 上投影和输出投影，亦不在 run 中追加 KV。模型/框架必须先完成新 token 的写入与槽位映射。

设请求数为 $B$、总 query 行数为 $N=\sum_iQ_i$、每请求有效 KV token 数为 $L_i$、heads 为 $H$、kernel page size 为 $P$。以下是代码调用与设备 launcher 中实际使用的单位：

| 参数 | 形状与单位 | 关键关系 |
| --- | --- | --- |
| `qo_indptr` | $[B+1]$，query 行的累计偏移 | $Q_i=\texttt{qo\_indptr}[i+1]-\texttt{qo\_indptr}[i]$；decode 通常每段长度 1 |
| `kv_indptr` | $[B+1]$，页索引的累计偏移 | 相邻差是该请求引用的页数，不是一般情况下的 token 数 |
| `kv_indices` | 展平的物理页编号 | 请求 $i$ 的索引区间由 `kv_indptr` 指定 |
| `kv_len_arr` | $[B]$，每请求有效 KV token 数 $L_i$ | 最后一页可未满；不是 query 长度 |
| `q_nope/q_pe` | $[N,H,d_c]$ / $[N,H,d_R]$ | 一次 run 可包含多行 query 的增量 prefill |
| `ckv_cache/kpe_cache` | $[\text{num\_pages},P,d_c]$ / $[\text{num\_pages},P,d_R]$ | 全池可被不同请求重复引用；两块可来自联合 cache 的切片 view |
| `out` | $[N,H,d_c]$，dtype 与计划的 Q 一致 | 仍需模型侧的 Value/输出投影 |
| `lse` | $[N,H]$，FP32 | 仅在要求返回时写入，默认对数底为 2 |

`plan` 中 `kv_len_arr` 的 docstring 将它称为 query length，`run` 的 query shape 说明也以 batch_size 简写；实际 C++ 与 kernel 使用 `kv_len_arr` 作为 KV 长度、Q 的首轴作为总 query 行数。这里依据 `MLAPlan`、`BatchMLAPagedAttentionRun/SM90Run` 和设备函数核对单位，不由说明文字补造另一种接口。

**教学页例子：**$P=4$、一个请求有 $L=6$ 个 KV token、当前 $Q=2$ 个 query，其物理页为 `[5,2]`：`qo_indptr=[0,2]`、`kv_indptr=[0,2]`、`kv_indices=[5,2]`、`kv_len_arr=[6]`。逻辑 token 0–3 在页 5，4–5 在页 2 的前两个位置；页 2 后两个位置即使已分配也不属于本请求的有效历史。SGLang 上节传 $P=1$，才使 token 槽位与 kernel 页编号在数值上相同。

### 9.1 plan 保存执行安排，run 才读 Q/KV

`plan` 先按 backend、Q/KV/O dtype、索引 dtype 与 $d_c/d_R$ 取得 JIT module。`jit/attention/modules.py::gen_batch_mla_module` 对 fa2 生成 `batch_mla_*` launcher，对 fa3 生成 `batch_mla_sm90_*` launcher并加入 SM90a 编译选项；类型和维度因此进入模板，不是 run 时任意变化的说明标签。

Python 将 query/KV indptr 和 KV 长度搬到 CPU，用 C++ `MLAPlan` 构造设备工作列表、split 与合并偏移，再保存 `_plan_info`。标准 wrapper 另分配 8 MiB 的设备整数 workspace 与同大小的 pinned host buffer，调用者提供的 float workspace 承载部分 attention 输出等临时状态。名称 float/int workspace 描述用途；存储载体可以是字节 tensor，C++ 按大小和偏移解释。

普通模式保存或复制本次索引到设备；graph 模式复制到调用者预留的稳定地址。graph 模式不表示 `plan` 完全在 GPU 上运行，也不表示长度、工作划分改变后可无条件复用旧计划。长度、indptr、causal 条件和实际 run 的张量必须对应同一次安排；具体 graph 调用时序不在本节的验证范围。

`run` 再核对实际 Q/KV dtype 是否与 plan 一致，取得 stride、输出缓冲和 mask，调用 cached module。C++ 以 plan 选定的类型重新解释指针，因此 dtype 不匹配会影响读写语义；不能仅检查元素个数或字节容量。`plan` 返回、JIT 能选择某个模板、launcher 返回均不是本次已跑过 GPU 用例的证据。

## 10. 非等长因果 attention 采用右下对齐

fa2 的 `mla.cuh::logits_mask_` 与 fa3 的 `mla_hopper.cuh::logits_mask_` 使用条件 `kv_idx + qo_len > kv_len + q_idx` 屏蔽未来位置。对某请求的局部 query 行 $t\in[0,Q_i)$、逻辑 KV 位置 $j$，允许范围是

$$0\le j\le L_i-Q_i+t,$$

并受到有效 KV 与当前 split 范围限制。这将 query 与完整 KV 序列的最后 $Q_i$ 个位置对齐，适合 prefix 已缓存的增量 prefill。它不是从左上角开始的 $j\le t$；后者会让新 query 看不到大部分 prefix。

上节 A 的 $Q=2,L=4$，两行分别允许位置 0–2 与 0–3；B 的 $Q=1,L=3$ 允许 0–2。上述 $P=4,L=6,Q=2$ 页例子的两行分别可见 0–4 与 0–5。页大小和物理页顺序不改变逻辑因果关系。

`scheduler.cuh::packed_causal_kv_end` 用同一右对齐关系限制 query tile 需要处理的 KV 总范围；kernel 仍对 tile 内每行施加具体 mask。规划时缩短某 tile 的总读取范围不能替代行级 mask。heads 会和 query 行一起打包用于工作划分，但 `q_idx` 由 packed index 除以 $H$ 恢复，不能让 head 序号进入时间位置。

普通 decode 每请求一行，若 KV 只包含历史与当前 token，无未来位置，调用 `causal=False` 与该一行的因果范围等价。这个结论依赖缓存可见范围；不自动覆盖多行推测验证、窗口或其他可见性规则。

## 11. plan 怎样切工作，kernel 怎样读取潜变量

### 11.1 split-KV 与请求分页是两个维度

标准 fa2/fa3 使用同一 `scheduler.cuh::MLAPlan`。它把每请求 $Q_iH$ 个 packed query/head 行分 tile，依据设备 SM 数安排持久 CTA 工作组；每 CTA 的 query tile 为 64 个 packed 行，规划器按平均 packed 长度选择每组 1 或 2 个 CTA。这里描述的是该 planner 的调度组，不能仅凭 cluster 变量名等同于一套硬件 CTA-cluster API。

planner 估算各 tile 的有效 KV 工作，选取 `kv_len_limit`，把过长范围切成多个 `[kv_start,kv_end)` 工作项，用最小负载堆分配给工作组，并登记合并的 output/LSE 偏移。未 split 的项可直接写最终输出；split 的项先写 float workspace 中的部分输出和 LSE。工作数与划分因此还取决于 heads、query/KV 长度、causal 与 SM 数，不是“每页启动一个 attention kernel”或“每请求固定同样的 split 数”。

分页描述 KV **存在哪些物理页**，split-KV 描述一次 attention 的历史 **由哪些工作项计算**。同一物理页可以被相邻 split 访问，也可以被不同请求复用；两种划分不能互换。

### 11.2 地址由物理页、页内 token 与 stride 共同决定

对逻辑 token $j$，令 $a=\lfloor j/P\rfloor$、$r=j\bmod P$，物理页号为

$$p=\texttt{kv\_indices}[\texttt{kv\_indptr}[i]+a].$$

内容 latent 通道 $c$ 的元素地址是整理者对 `load_kv` 的等价展开：

$$\texttt{ckv\_base}+p\,\texttt{ckv\_stride\_page}+r\,\texttt{ckv\_stride\_n}+c.$$

位置 key 使用自己的 base 和对应 stride。C++ launcher 显式读取页轴与页内轴 stride；末轴按连续通道加载，并非自动支持任意最后轴间隔的 tensor。联合 `[latent,rope]` 池的两块切片可以保留较大的页 stride，所以逻辑宽度 $d_c$ 不要求物理页距离恰好也是 $Pd_c$。

`load_kv` 按有效长度和 split 范围间接取页，异步搬到 shared memory。fa2 的 `compute_mla_qk` 分别计算位置与 latent 点积、相加后进入一次在线 softmax；`compute_mla_pv` 用同一 latent tile 做概率加权。没有从全局缓存展开成 $H$ 份内容 K/V，也不要求物化完整 $Q\times L$ score 矩阵。共享 latent 是持久表示事实，具体多个 head 的加载复用程度仍由 tile 与工作安排决定，不能据此承诺历史只被读取一次。

fa2 launcher 按 shared-memory 容量选择 KV tile、pipeline stage 和 warpgroup 的 QK 分工，用 MMA 完成标准路径。fa3 的 Hopper 入口固定所选 64×64 tile、两阶段流水，将搬运与计算分配给 producer/consumer warpgroups，计算侧使用 WGMMA。

## 12. 在线归一化、split 合并和 LSE 的底

kernel 用 FP32 score/累加状态维护逐行最大值 $m$、指数和 $d$ 与 latent 加权累计量。`sm_scale` 乘 `log2e`，内部用 `exp2` 表达原本的 $\exp(\ell)$。概率中间表示和 partial output 仍会按所选工作类型舍入，不能把 FP32 累加表述为整条路径全部使用 FP32。

完成一个历史 split 后，`write_o` 写的是局部归一化输出 $o_s$ 和

$$z_s^{(2)}=\log_2\sum_{j\in s}\exp(\ell_j).$$

partial output 的模板类型是计划的 O dtype，本 wrapper 与 Q dtype 一致；partial LSE 为 FP32。它们不是原始 QK logits 或未归一化 value 累计量。fa2 与 fa3 的 cooperative kernel 在所有工作完成后执行 grid synchronization，再由 `DevicePersistentMergeStates` 合并；合并在这个 kernel 的第二阶段中进行，不因逻辑上有两阶段就必然额外启动一个 merge kernel。

`state.cuh::state_t.merge` 对各部分的权重按 LSE 恢复，实数语义为

$$o=\frac{\sum_s2^{z_s^{(2)}-m_2}o_s}{\sum_s2^{z_s^{(2)}-m_2}},\qquad m_2=\max_s z_s^{(2)}.$$

这和自然对数的 LSE 合并等价，但只能使用同一底。默认 `return_lse_base_on_e=False` 返回 base-2 LSE；选 True 时写出的 LSE 乘 $\ln2$，成为

$$z^{(e)}=(\ln2)z^{(2)}=\ln\sum_j\exp(\ell_j).$$

对默认 base-2 LSE 直接使用 $e^{z_s-m}$，会改变分段权重，即使张量 shape 和 dtype 都正确。接入框架的 cascade、跨段或其他输出合并前必须确认所用接口的 LSE 约定；SGLang 展开 MHA 的 ragged 合并与这里的 paged MLA 返回值也不能仅因都名为 lse 就互换。

## 13. 原生 FP8 cache 读取与 SGLang 的 cast 路径不同

本固定 wrapper 的 `plan` 允许 KV dtype 为 FP16、BF16、FP8 E4M3FN；FP8 还受明确限制：**fa3、SM90、BF16 query、$d_c=512,d_R=64$**。其他 1 字节存储如 uint8、FP4、E5M2 不能因字节宽度相同就按此接口解释为 FP8；fa2 的这条标准 wrapper 路径也没有 FP8 KV 支持。

FP8 `run` 要求有限正数的 `ckv_scale` 与 `kpe_scale`，语义为

$$\widehat c=s_c\operatorname{cast}_{\rm real}(c_{\rm FP8}),\qquad
\widehat k^R=s_R\operatorname{cast}_{\rm real}(k^R_{\rm FP8}).$$

这两个量是 cache 的独立 per-tensor 反量化尺度；BF16/FP16 cache 不接收它们。wrapper 不负责估计 scale 或编码 cache，调用者需保证历史写入采用相同约定。

Hopper 的 `repack_fp8_kv_to_bf16` 先把一个 FP8 KV tile 从 shared memory 转成 BF16 staging，再乘各自的 scale，供 QK 与 PV 的 BF16×BF16 WGMMA 使用。scale 本身也转换成该工作类型，实际数值包含类型舍入。它不是 BF16×FP8 混合乘法，也不是为整个全局池建立 BF16 副本；staging 被 tile 流水复用，并受 shared-memory 同步约束。

内容 scale 同时影响 score 中的 latent key 和输出中的 latent value，位置 scale 只影响位置 score。按实数语义，

$$\ell_{t,j,i}=s\left(\bar q_{t,i}(s_c\widetilde c_j)^\top+q^R_{t,i}(s_R\widetilde k^R_j)^\top\right),\qquad
z_{t,i}=\sum_j a_{t,j,i}(s_c\widetilde c_j).$$

因此不能只在 QK 处乘内容 scale、却漏掉 PV，或在 kernel 已反量化的输出上再乘一次。模型的 `sm_scale`、cache scale 与前后 BMM 的权重/activation scale 仍属于不同接口。

与 SGLang 所读调用端比较：其普通 paged MLA 先对 `get_key_buffer` 做整池 `.to(q.dtype)`，再将工作 dtype cache 传入 run；本库的 FP8 路径则让 `kv_data_type=FP8` 与 cache scale 进入设备 kernel。两者能表达相同 latent 对象，却具有不同的转换位置、临时内存和版本要求。对 SGLang 0.6.12 调用端不能因本库 v0.6.17 新接口而改写成已使用 fused FP8 dequant。

## 14. 本次库侧核读范围

同一 Python 模块还包含 CUTLASS/Blackwell、TRTLLM、CuteDSL 与稀疏 MLA 入口，本节没有核读这些设备路径。`backend="auto"` 是基于设备条件的选择：该标准 wrapper 在满足 SM90/CUDA 条件时选 fa3，否则 fa2；不能等同于所有架构和 decode 形状都已选择最优后端。其他入口采用不同输入布局、scale、能力限制或输出协议，不能用本节的两个 kernel 覆盖它们。


## 15. vLLM：同一缓存对象，独立选择 decode 与 prefill 实现

本节核读 vLLM commit `568afb3a13806beb53bb2e6bd518269357b237c0` 的普通非稀疏 DeepSeek V2/V3 MLA、单设备 CUDA 路径。模型层与缓存接口沿公共实现追踪；设备读取选 `TRITON_MLA`，展开 prefill 选 `FlashAttnPrefillBackend`，保留 FP16/BF16 工作 dtype，关闭 prefill query 量化。选择这些路径是为了贯通投影、写入、读取和前缀展开，并不表示本次启动或自动选中了它们。服务端 block table 与 slot mapping 的来历见 [vLLM 推理执行](vllm-inference-execution.md)。

`MLAAttention.__init__` 通过 `get_attn_backend(..., use_mla=True)` 选择 MLA backend，同时调用独立的 `get_mla_prefill_backend`。CUDA 的自动优先级随架构和配置改变：例如 SM90 及较早架构的普通 MLA 列表先尝试 `FLASH_ATTN_MLA`、`FLASHMLA`、`FLASHINFER_MLA`，再到 `TRITON_MLA`；major 12 的列表则将 Triton 放在首位。具体 backend 还必须通过 dtype、block size、功能组合和依赖可用性检查；显式配置和按 KV group 的 override 也会影响选择。因此不能用一个已读 kernel 概括“vLLM 默认 MLA 都这样执行”。（`v1/attention/selector.py::get_attn_backend`、`platforms/cuda.py::_get_backend_priorities/get_valid_backends`。）

Prefill 选择器单独验证工作 dtype、MLA 维度与库可用性；CUDA Hopper 及较早架构的自动列表只含 FlashAttention，Blackwell 的列表还有其他候选。本节只沿 FlashAttention 调用契约继续读。其基类允许 FP16/BF16；具体维度支持也受 FlashAttention 版本影响。对于 DeepSeek 的 $d_K=128,d_R=64,d_V=128$，FA3/SM90 和 FA4 可直接处理不同的 QK/V head width，其他需要 padding 的分支将 V 补零到 QK width，再截回 $d_V$。这里没有核读 FlashAttention 库内部的设备 kernel。（`mla/prefill/selector.py`、`base.py::validate_configuration`、`flash_attn.py::FlashAttnPrefillBackend`。）

## 16. vLLM 的投影与吸收位置

沿用第 2 节的 $N,H,d_K,d_R,d_V,d_c,d_q$。`DeepseekV2MLAAttention` 创建模型投影和 norm；`layers/mla.py::MultiHeadLatentAttentionWrapper.forward` 将本轮 hidden states 变为以下对象：

| 前向步骤 | 得到的张量 | 后续用途 |
| --- | --- | --- |
| 融合 A 投影并拆分 | $[N,d_q]$ 与 $[N,d_c+d_R]$ | query latent 和联合 KV latent/位置分支 |
| query norm、B 投影 | $[N,H,d_K+d_R]$ | 尚未吸收 key 上投影的完整 query |
| KV norm、位置分支 RoPE | $[N,d_c]$ 与 $[N,1,d_R]$ | normalized latent 与 rotated position key；query 的末 $d_R$ 部分也经 RoPE |
| `MLAAttention` 返回、`o_proj` | $[N,Hd_V]$ 再到模型 hidden width | 两种 attention 路径汇合后的模型输出 |

不压缩 query 的配置改用 `q_proj` 和单独 `kv_a_proj_with_mqa`；这不改变普通 KV cache 的对象。模型基础 score scale 仍是 $(d_K+d_R)^{-1/2}$，YaRN 配置可进一步调整。它不会因为 decode query 的宽度变成 $d_c+d_R$ 而自动重算。（`models/deepseek_v2.py::DeepseekV2MLAAttention.__init__`。）

`MLAAttention.forward_impl` 根据 metadata 分开处理 decode token 区间与 prefill token 区间，同一批调用可以包含两者。普通 `TritonMLAMetadataBuilder` 的 query-length 支持为单 token，decode reorder threshold 为 1；这是一项 metadata 分类规则，不是仅看请求名或某个全局 forward mode。未命中 decode 区间的普通 prefill 走展开 MHA。与第 5 节 SGLang 的 batch prefix 阈值不同，本节 vLLM 公共路径没有以那个阈值把长前缀 prefill 切成 paged latent attention。（`MLACommonMetadataBuilder.build`、`MLAAttention.forward_impl`。）

对 decode 的内容 query，普通 CUDA 分支显式执行

$$[H,N_d,d_K]\times[H,d_K,d_c]\longrightarrow[H,N_d,d_c],$$

其中 $N_d$ 是本轮 decode token 数。权重名为 `W_UK_T`；结果转置后和 $[N_d,H,d_R]$ 的 query RoPE 部分交给 `forward_mqa`。Triton backend 再将二者拼成 $[N_d,H,d_c+d_R]$；这与 SGLang/FlashInfer 保留两个 query 参数的接口不同。Attention 得到 $[N_d,H,d_c]$ 的潜空间输出，`_v_up_proj` 用 `W_UV:[H,d_c,d_V]` 做第二个 BMM，再把结果交回模型 wrapper 的 `o_proj`。第 3 节的两项吸收恒等式仍适用，显式 BMM 与输出投影也仍然存在。

`process_weights_after_loading` 从 `kv_b_proj` 取得或反量化出工作 dtype 权重，转为 $[d_c,H,d_K+d_V]$，拆分并组织 `W_UK_T` 和 `W_UV`。所选普通 CUDA 分支保留 FP16/BF16 的 BMM 权重副本；同函数的 ROCm/AITER FP4/FP8 BMM 路径不在本节范围。低比特 checkpoint 因而不能证明这些吸收 BMM 本身也在对应低位宽执行。

## 17. vLLM 联合分页缓存与 FP8 scale

### 17.1 写入先于读取，地址由 slot mapping 给出

`MLACommonBackend.get_kv_cache_shape` 的普通布局是 $[B_{\rm blocks},P,d_c+d_R]$；`num_kv_heads=1` 没有另占一个轴，也没有第二份 V cache。`MLAAttentionSpec.real_page_size_bytes` 在普通格式下给出 $P(d_c+d_R)\operatorname{sizeof}(\mathrm{dtype})$。例如 $d_c=512,d_R=64$ 时，BF16 为每 token 1152 字节，普通 FP8 为 576 字节；联合 scalar scale 由 attention layer 管理，不嵌入每个 token 载荷。此计算不含预留空 block、映射和 workspace；专门的 `fp8_ds_mla` packed 格式采用另一条字节规则，不能套用这个公式。（`v1/kv_cache_interface.py::MLAAttentionSpec`。）

`MLAAttention.forward` 的 direct-call 路径先调用 `do_kv_cache_update`，再 `forward_impl`；opaque 路径通过 `unified_mla_kv_cache_update` 产生一个 dummy tensor 依赖，交给 attention op，表达编译图中的先写后读次序。这个依赖不是模型数值。普通 `MLAAttentionImpl.do_kv_cache_update` 调用 `concat_and_cache_mla`，将 normalized latent 和 rotated position key 写入联合 cache。

CUDA 写入 kernel 对第 $n$ 行取 `slot_mapping[n]`，计算物理 block 和 block 内偏移

$$b=\left\lfloor\frac{\mathrm{slot}_n}{P}\right\rfloor,\qquad r=\mathrm{slot}_n\bmod P,$$

然后按真实 block/entry stride 写末轴的两段。负 slot 表示 padding，不写入；launcher 按 slot mapping 的长度决定本次真实行数，兼容 Q/K 投影含图 padding 的情况。decode 随后读取的有效历史包含当前刚写入的 token。（`csrc/libtorch_stable/cache_kernels.cu::concat_and_cache_mla_kernel/concat_and_cache_mla`。）

### 17.2 普通 FP8 的一个 scale 同时作用于 latent 和位置 key

令 $\sigma=\texttt{layer.\_k\_scale}$。普通 FP8 cache writer 对两部分都执行 scaled conversion；NVIDIA helper 的方向是

$$\widetilde c=Q_{\rm FP8}(c/\sigma),\qquad\widetilde k^R=Q_{\rm FP8}(k^R/\sigma).$$

这里的 $Q_{\rm FP8}$ 是浮点编码/舍入，并使用有限值饱和转换；不是 INT8 的整数零点方案。普通格式没有 per-token/per-channel 内嵌 scale。`set_default_quant_scales` 注册 scalar FP32 buffer，缺少量化权重 scale 的路径在加载后重置为 1；配置可选择 `calc_kv_scales`，它在一次前向计算全张量 absmax 后关闭 `calculate_kv_scales`，不能写成每个 token 都重新估计。（`attention.py::_init_kv_cache_quant/set_default_quant_scales`、`mla_attention.py::process_weights_after_loading/calc_kv_scales`、`quantization/w8a8/fp8/nvidia/quant_utils.cuh::scaled_vec_conversion/scaled_convert`。）

该 `calc_kv_scales` 用 normalized latent 的 absmax 计算 `_k_scale` / `_v_scale`；虽然签名带 `k_pe`，实际没有把其幅值纳入这个统计。但联合 writer 仍对 latent 和 RoPE key 使用 `_k_scale`。所以不能把该统计描述为已分别校准两部分，或据此保证位置分支不会饱和。具体 scale 来自何种 checkpoint、对给定模型的误差怎样，需要另行核验；本节没有做精度实验。

所选 Triton MLA 声明支持 cache `auto/float16/bfloat16/fp8/fp8_e4m3`，kernel block size 是 16 的倍数。CUDA 构造器对 FP8 cache 要求 SM89+，对显式 BF16 cache 要求 SM80+；FP8 cache 会关闭公共层的 query 量化，工作 query 仍保留 FP16/BF16。因而本节路径不能直接用于 A100/SM80 的 FP8 cache，也不能以广义“vLLM 支持 FP8”推导任意 backend/dtype 都可用。这里核对的是声明、检查与代码路径，没有验证本机编译或运行。

## 18. Triton MLA：从 block table 读取一次 latent，分段归一化

`TritonMLAImpl.forward_mqa` 将联合 cache 加一个大小为 1 的 head 轴，并把前 $d_c$ 元素作为 V view。普通单 token decode 下 $N_d$ 等于请求数，输出为 $[N_d,H,d_c]$。`decode_attention_fwd` 按 query heads 与 KV heads 的比值分派；本节沿多 query heads、单 KV head 的 grouped MLA 分支核读。它同时处理一小组 heads 和一个历史 split，而不是每个 head 都必须独立加载一份完整历史。

对请求 $a$ 的第 $j$ 个历史 token，设备读取的物理位置是

$$b_j=\mathrm{block\_table}[a,\lfloor j/P\rfloor],\qquad r_j=j\bmod P.$$

再按实际 page/token/head stride 加末轴偏移。`seq_lens[a]` 控制可读 token 范围；block table 中存在尾页不意味着尾页全为有效 token。Page stride 从 tensor 取得，兼容跨层 view 的间隔，不能硬编码为 $P$ 乘 token stride。对典型 $d_c+d_R=576$，grouped kernel 分别载入 512 维 latent tile 和 64 维位置 tile。（`triton_decode_attention.py::_page_stride/_decode_grouped_att_m_fwd/_fwd_grouped_kernel_stage1`。）

若 cache 是 FP8，kernel 在局部 load 后先转 FP32、乘 $\sigma$，再转为对应 Q 的工作 dtype；latent 与位置 key 都做这一步。两次 dot 合成同一 score，再乘模型的 `sm_scale`。`IS_MLA=True` 时 V 直接取**已经载入且已反量化的 latent tile 的转置**，不再从 V view 发起第二次 cache load。调用端虽同时传 `k_scale` 与 `v_scale`，本路径二者都是 `_k_scale`，实际 latent PV 复用 K 侧转换结果。不能把 `_v_scale` 的存在解释为普通 MLA cache 使用一套独立 V 编码/scale。

这也说明 FP8 描述了历史的存储与读取格式：本 kernel 的 dot 操作数是工作 FP16/BF16，概率在进入 PV dot 前也转换成 V 工作 dtype，softmax 状态与累计输出使用 FP32。它不要求“所有 attention 乘法原生 FP8”。与第 13 节 FlashInfer v0.6.17 fa3 的 shared-memory BF16 staging 类似，两者都在读取后改变工作 dtype；但 scale 协议、硬件检查和局部组织不同。

Triton 的历史 split 数由 `_compute_num_kv_splits` 依据 batch 最大序列长度和设备 SM 数选择：以长度除 512 的整除结果为起点，至少 1、上取到 2 的幂，再限制到 $2\times\mathrm{SM\ count}$；batch-invariant 配置强制 1。每个请求按 $\lceil L_a/S\rceil$ 划分自己的连续 token 区间，$S$ 为所选 split 数。这里的 split 是 attention 历史计算切分，并不是 cache page 数，也不是调度器 prompt chunk 数。

第一阶段 kernel 为每个有效 split 写归一化 latent 输出与一个 LSE 到 FP32 scratch：

$$\mathrm{scratch}:[N_d,H,S,d_c+1],\qquad z_u=m_u+\ln\sum_{j\in u}e^{\ell_j-m_u}.$$

第二阶段 `_fwd_kernel_stage2` 跳过空 split，以 $e^{z_u-M}$ 对这些局部输出加权，再写最终 latent 输出和 LSE。这里是**两个 Triton kernel launch**；第 12 节 FlashInfer fa2/fa3 的 cooperative 实现则在同一 kernel 内同步后合并。Triton 内部的 $z_u$ 与合并使用自然对数/`exp`，不能直接接收 FlashInfer wrapper 默认返回的 base-2 LSE。

`TritonMLAMetadataBuilder` 在 graph warmup 前按最大 batch/最大 split 预留共享 scratch；普通调用从 workspace manager 取得所需 view，未初始化时才直接分配。FP32 scratch 不等于所有输出都保持 FP32：最终 `o` 和 `lse` 都由 `forward_mqa` 按 Q dtype 创建，kernel 写入时转换；LSE shape 是 $[N_d,H]$，而下节 prefill 合并接口使用 $[H,N_p]$。这两项区别都属于接口的一部分，不能因都叫 LSE 而忽略。

## 19. vLLM prefill：展开新 token，再分块恢复历史

`MLACommonBaseImpl.forward_mha` 保留完整 $[N_p,H,d_K+d_R]$ query，用本轮 normalized latent 经 `kv_b_proj` 生成 $[N_p,H,d_K+d_V]$，拆出内容 K 和 V，并向 K 广播拼入 shared RoPE key。`FlashAttnPrefillBackend.run_prefill_new_tokens` 以当前 token 的 query offsets 同时作为 Q/K offsets，使用 `causal=True`；只计算本轮新 token 段。没有 prefix context 时直接得到 prefill 输出。

有已缓存前缀时，`MLACommonMetadataBuilder.build` 计算每请求的 context length：总有效长度减本轮 query length，再生成 `chunked_context`。`_compute_prefill_context` 按 chunk 的 request offsets、起点与 block table，通过 `gather_and_maybe_dequant_cache` 将**该 chunk 的有效历史行**复制到 $[T_{\rm chunk},d_c+d_R]$ 工作区。普通 FP8 在 gather 中用 `_k_scale` 反量化到工作 dtype；随后每个 chunk 再调用 `kv_b_proj`，恢复每头 K/V。这里会物化历史 chunk 的 latent 工作副本与展开 K/V，没有将整个预分配 cache 池 `.to(q.dtype)`。（`cache_kernels.cu::gather_and_maybe_dequant_cache`、`MLACommonBaseImpl._compute_prefill_context`。）

`run_prefill_context_chunk` 对历史段使用 `causal=False`，因为所有历史位置都在当前 query 之前。它返回 output/LSE，逐 chunk 合并，再与本轮因果结果合并；公式仍是第 5.1 节的 LSE 权重归一化，不能简单相加。Prefill 的 LSE 接口采用 $[H,N_p]$；本版本 CUDA/Triton `merge_attn_states` 使用 `exp/log` 合并，要求自然对数约定，并处理无 prefix 的请求/空 context chunk。（`merge_attn_states.py::merge_attn_states`、`triton_merge_attn_states.py::mask_empty_context/merge_attn_states_kernel`、`csrc/libtorch_stable/attention/merge_attn_states.cu::merge_attn_states_kernel`。）

这里还有一个数值生命周期差异：虽然所有本轮 latent 已先写 cache，prefill 的新 token 展开仍使用前向传入的 normalized latent/RoPE key；历史展开才读取、可能反量化 cache。Triton decode 则连当前 token 都通过刚写入的联合 cache 读取。因此普通 FP8 开启后，即便忽略实数下展开与吸收的等价式，当前 token 和历史 token 的舍入位置也不相同；不能仅根据持久 cache dtype 宣称 prefill 全部在使用同一低精度 K/V。

Prefill workspace 的 token 容量由 batch、最大序列长度和 block size 约束，并以 64K token 为常用上限再确保每请求至少一页；这限制同时恢复的历史，但恢复后的每头 K/V 通常远大于 latent。源码 gather 还将普通 head width 限在 320 或 576，不能仅从 decode backend 的 head-size 声明推导所有 prefix prefill 组合都支持。Workspace 预算、临时 K/V、cache 容量和实际读取流量应分别核算，没有用源码估算替代设备峰值测量。

## 20. 三个固定来源的对照与剩余边界

| 问题 | SGLang 所选普通调用端 | FlashInfer v0.6.17 标准 wrapper | vLLM 所选普通 CUDA 路径 |
| --- | --- | --- | --- |
| 持久对象 | normalized latent + rotated key，联合 token pool | 调用者提供 paged latent/RoPE views；不负责模型投影或 cache 写入 | 相同逻辑对象，联合 block cache |
| 吸收 query 接口 | 内容与位置分开传入 | 两个 query 参数 | Triton backend 将两部分拼成工作 query |
| 普通 FP8 cache scale | 普通 pool writer 为 cast；所读 paged 调用整池转 Q dtype、run 不传 cache scale | fa3 特定 FP8 组合要求独立 latent/RoPE scalar scale | writer 除共同 `_k_scale`；Triton load 乘同一 scale，再转 Q dtype；latent 复用作 V |
| 展开与吸收选择 | extend 的 MHA/MLA 受 batch prefix/容量与配置影响 | 本入口执行已吸收的 latent attention | metadata 的普通 prefill MHA / 单 token decode MQA，可同批处理 |
| Split 与 LSE | 调用端提供索引和 wrapper plan；库侧 0.6.12 内部未核读 | plan 生成持久工作表；同 kernel 合并；默认 base-2 LSE，可切自然底 | 长度/SM 数选 splits；两个 kernel；自然 LSE；最终 LSE dtype 随 Q |

这个表比较代码中的对象和执行契约，不提供性能排序。对照结果说明：共享 latent、FP8 cache、paged attention 或 LSE merge 这些名称都不足以确定实际读写位置、scale、工作精度和临时量。

本页 vLLM 部分未核读其他 MLA decode kernel、FlashAttention 库内 kernel、ROCm/AITER、推测执行、LoRA、完整 CUDA graph 或量化输出融合。DCP 与 packed/DSA 的专用接口由下列页面接续。

PCP=1 的普通 vLLM DCP 已在[分布式执行页](vllm-mla-context-parallel-execution.md)接续：全局 token 到本地 page、query head 收集、局部 LSE 加权与输出返回，以及 prefix prefill 的缓存收集/展开分别核读。该页限定 Model Runner V1 与所选 BF16/Triton 路径，不扩大本页单设备核读范围，也不覆盖 PCP 或 packed/DSA。

[vLLM packed 缓存与 DSA](vllm-mla-packed-cache-and-dsa-execution.md)单独接续 V3.2 的 656-byte 主记录、内嵌分组 scale、独立 indexer cache、top-k 逻辑位置与主 cache/workspace 读取。它沿所选单设备 FlashMLA sparse 框架链展开，[DSA 设备执行](dsa-sparse-device-execution.md)再接续匹配 DeepGEMM logits 与 FlashMLA 的 SM90 读取、舍入和归一化；这些专用接口不能套成本页普通连续 FP8 行或 DCP 的恢复/归约。

## 来源身份

| 来源 | 固定版本 | 实际使用范围 |
| --- | --- | --- |
| sgl-project/sglang（`https://github.com/sgl-project/sglang/tree/2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc`） | `2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc` | `deepseek_v2.py` 的 MLA 初始化/投影/分派；backend handler；普通 MHA/MLA forward mixin；`post_load_weights` 的 KV 上投影拆分；普通 MLA pool、读写 wrapper/Triton 地址；Radix/backend 接口；FlashInfer MLA 的 eager metadata、MHA chunk、paged extend/decode 与 indices updater；runner cache dtype 和 pyproject 依赖。具体文件登记于 sources，函数就近标明。 |
| flashinfer-ai/flashinfer（`https://github.com/flashinfer-ai/flashinfer/tree/a0a6b019b9b27d49d209f85d028a1ae5a9b347d7`） | v0.6.17，commit `a0a6b019b9b27d49d209f85d028a1ae5a9b347d7` | 标准 `BatchMLAPagedAttentionWrapper` 的 fa2/fa3 plan/run/JIT/C++ binding、共同工作规划、分页 latent/RoPE 读取、右对齐 mask、在线归一化、cooperative split/LSE 合并与 fa3 原生 FP8 tile 反量化；具体文件见新增 sources，函数就近标明。与 SGLang 所声明 FlashInfer 0.6.12 依赖分开。 |
| vllm-project/vllm（`https://github.com/vllm-project/vllm/tree/568afb3a13806beb53bb2e6bd518269357b237c0`） | `568afb3a13806beb53bb2e6bd518269357b237c0` | 普通 DeepSeek MLA 模型/wrapper、公共投影吸收和权重组织、cache spec、普通 CUDA 写入与 prefix gather、Triton grouped MLA decode 与两阶段合并、FlashAttention prefill 选择/调用与 CUDA/Triton LSE 合并；19 个具体文件登记于 sources。其他 backend 和分布式/稀疏路径不作运行结论。 |
