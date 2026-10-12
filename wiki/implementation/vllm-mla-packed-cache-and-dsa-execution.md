---
title: vLLM MLA packed 缓存与 DSA：字段、选择索引和稀疏读取
type: implementation
tags:
  - attention
  - kv-cache
  - fp8
  - serving
sources:
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/models/registry.py
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/models/deepseek_v2.py
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/layers/mla.py
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/layers/attention/mla_attention.py
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/layers/attention/sparse_mla_attention.py
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/layers/sparse_attn_indexer.py
  - raw/repositories/2026-09-21/vllm/source/vllm/v1/attention/backend.py
  - raw/repositories/2026-09-21/vllm/source/vllm/v1/attention/backends/mla/indexer.py
  - raw/repositories/2026-09-21/vllm/source/vllm/v1/attention/backends/mla/flashmla_sparse.py
  - raw/repositories/2026-09-21/vllm/source/vllm/v1/attention/backends/mla/sparse_utils.py
  - raw/repositories/2026-09-21/vllm/source/vllm/v1/attention/ops/flashmla.py
  - raw/repositories/2026-09-21/vllm/source/vllm/utils/deep_gemm.py
  - raw/repositories/2026-09-21/vllm/source/vllm/utils/torch_utils.py
  - raw/repositories/2026-09-21/vllm/source/vllm/_custom_ops.py
  - raw/repositories/2026-09-21/vllm/source/vllm/config/attention.py
  - raw/repositories/2026-09-21/vllm/source/vllm/v1/kv_cache_interface.py
  - raw/repositories/2026-09-21/vllm/source/csrc/libtorch_stable/cache_kernels.cu
  - raw/repositories/2026-09-21/vllm/source/csrc/libtorch_stable/sampler.cu
  - raw/repositories/2026-09-21/vllm/source/csrc/libtorch_stable/cooperative_topk.cu
  - raw/repositories/2026-09-21/vllm/source/csrc/libtorch_stable/cooperative_topk.cuh
  - raw/repositories/2026-09-21/vllm/source/csrc/libtorch_stable/torch_bindings.cpp
  - raw/repositories/2026-09-21/vllm/source/csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh
  - raw/repositories/2026-09-21/vllm/source/cmake/external_projects/flashmla.cmake
  - raw/repositories/2026-10-07/flashmla/source/flash_mla/flash_mla_interface.py
  - raw/repositories/2026-10-07/flashmla/source/csrc/api/sparse_fwd.h
  - raw/repositories/2026-09-21/vllm/source/tests/kernels/attention/test_deepgemm_attention.py
updated: 2026-10-07
---

# vLLM MLA packed 缓存与 DSA：字段、选择索引和稀疏读取

DeepSeek Sparse Attention（DSA）在 MLA 之外增加一个 indexer：先用较小的查询与独立历史 key 选择 token，再让主 MLA attention 读取这些 token 的联合 latent/RoPE 状态。`fp8_ds_mla` 则规定主缓存的混合精度字段与内嵌 scale。**缓存量化决定每条记录怎样表示，DSA 决定当前 query 选择哪些记录；两者共同约束读取接口。** top-k 没有把未选历史从缓存中删除。

本页接续 [MLA 的联合潜变量与投影吸收](../fundamentals/model/multi-head-latent-attention.md)和[普通 MLA 的单设备执行](mla-cache-and-attention-execution.md#15-vllm同一缓存对象独立选择-decode-与-prefill-实现)，解释普通连续 FP8 行不能承接的字段、scale 和索引边界。[普通 MLA DCP](vllm-mla-context-parallel-execution.md)的跨 rank 分片与归约属于另一项范围。

## 1. 固定版本、入口与所选条件

来源为 vLLM commit `568afb3a13806beb53bb2e6bd518269357b237c0`。主线是默认模型 registry 中的 `DeepseekV32ForCausalLM → deepseek_v2.DeepseekV3ForCausalLM`，模型配置带 `index_topk`；沿 `DeepseekV2MLAAttention`、通用 `Indexer`、`MLAAttention` 和 `FLASHMLA_SPARSE` 继续核读。它与源码中另有的硬件专用 `vllm.models.deepseek_v32` 入口不能仅凭名字等同。（`models/registry.py:91–93`、`deepseek_v2.py::DeepseekV3ForCausalLM/DeepseekV2Model`。）

选择 CUDA SM90、BF16 输入/query、eager、DP/TP/PCP/DCP 均为 1，无推测 decode；主 MLA 为 $d_c=512,d_R=64,H=128$，page 大小 $P=64$，cache dtype 为 `fp8_ds_mla`。选择实际创建 Indexer 的普通 backbone 层，`index_topk_freq=1,index_topk_pattern=None`，不包含跨层选择复用。为贯通稀疏 prefill 与 decode，显式选择 `sparse_mla_force_mqa=True`；普通生成 decode 每请求一个 query。Indexer 示例采用 $d_I=128,I=64$、quant block 128，top-k 数 $S$ 来自 `config.index_topk`，下面以 $S=2048$ 举例。这里的条件用于静态解释，没有启动服务或证明该配置已运行。

`FlashMLASparseBackend` 声明 BF16 工作 dtype、head size 576、所需 kernel page 64 与 capability major 9/10。`ops/flashmla.py` 仅在 `_flashmla_C` 和 `_flashmla_extension_C` 均存在时导入实际接口，否则使用抛错 stub；其 `is_flashmla_sparse_supported` helper 还限定实际 family 90/100，但该 sparse backend 没有覆写组合检查来调用它。不能由 backend 的声明推导所有依赖已可用。SM80/A100 不满足这条 FlashMLA sparse 的架构范围。对该 backend，量化 KV dtype 在 `MLAAttention` 构造时规范化为 `fp8_ds_mla`，不能把用户传入的 `fp8` 字符串直接解读为普通 576-byte FP8 cache。（`flashmla_sparse.py::FlashMLASparseBackend`、`ops/flashmla.py`、`mla_attention.py::_canonicalize_sparse_mla_kv_cache_dtype`。）

## 2. 主 cache 与 indexer cache 是两种对象

令 $T$ 为本轮实际 token 行数，$B_c$ 为主缓存物理 page 数，$B_I$ 为 indexer 的物理 page 数。下面的形状是所选路径中的接口，不将两个池的物理 page 编号视为相同。

| 对象 | 张量或载荷 | 作用 |
| --- | --- | --- |
| 主 cache 输入 | normalized latent $c:[T,512]$、RoPE key $k^R:[T,1,64]$ | 保留主 attention 所需历史 |
| 主 packed cache | `uint8` $[B_c,64,656]$ | 保存 FP8 latent、四个 FP32 scales、BF16 RoPE |
| 主 query | 吸收前 $[T,H,192]$，吸收后 $[T,H,576]$ | 内容 query 乘 $U_K$，再拼位置 query |
| Indexer Q/K/weights | $[T,I,128]$、$[T,128]$、$[T,I]$ | 估计历史 token 的选择分数 |
| Indexer cache | `uint8` $[B_I,64,132]$ | 每 page 的 FP8 key 平面与 FP32 scale 平面 |
| 共享 top-k buffer | `int32` $[T_{\max},S]$ | 本层当前 query 选择的请求内逻辑 token 位置 |
| 主 attention 中间输出 | $[T,H,512]$ | latent 加权和，随后经 $U_V$ 与输出投影 |

主 latent 的 norm、位置 RoPE 与投影吸收见[普通 MLA 执行](mla-cache-and-attention-execution.md#16-vllm-的投影与吸收位置)。Indexer 的 key 则由另一套模型投影与 `LayerNorm` 生成，不是从主 latent 的 512 个 FP8 字节取出前 128 个。Index head 数 $I$ 与主 attention head 数 $H$ 也不是同一轴。（`deepseek_v2.py::Indexer/DeepseekV32IndexerCache`、`layers/mla.py::MultiHeadLatentAttentionWrapper`。）

## 3. `fp8_ds_mla` 的 656-byte 主记录

所选 V3.2 的语义 head width 是 $512+64=576$，物理载荷则为 656 字节。`utils/torch_utils.py` 将这一 cache dtype 映射为 `torch.uint8`；`FlashMLASparseBackend.get_kv_cache_shape` 返回末轴 656。字节容器中的各区段由 writer/reader 分别解释。

| 半开字节区间 | 内容 | 粒度与解释 |
| --- | --- | --- |
| $[0,512)$ | 512 个 E4M3 FP8 latent 值 | 每 token 的 latent 沿通道分为四个 128 维组 |
| $[512,528)$ | 4 个 FP32 scale | 第 $g$ 项恢复 latent 第 $128g$ 到 $128g+127$ 维 |
| $[528,656)$ | 64 个 BF16 RoPE 值 | 保留 16-bit 位置分支，不进行 FP8 编码 |

因此 $656=512+4\times4+64\times2$。它与[普通联合 FP8](mla-cache-and-attention-execution.md#172-普通-fp8-的一个-scale-同时作用于-latent-和位置-key)的差别不只在 scale 粒度：普通格式将全部 576 个元素编码为 FP8，并使用 layer 的外部 scalar scale；packed 格式保留 BF16 位置分支，把动态分组 scale 放在记录内。（`flashmla_sparse.py` 的 V3.2 格式说明、`cache_kernels.cu::concat_and_cache_ds_mla_kernel`。）

同一源码还用 `fp8_ds_mla` 名称表示 DeepSeek V4 的 584-byte 格式，其内容宽度、scale 编码和字段位置不同。本页限定 V3.2，不能将 dtype 名称当作跨模型通用的字节契约。（`kv_cache_interface.py::MLAAttentionSpec.real_page_size_bytes`。）

### 3.1 写入：每条记录的四组 scale 在 kernel 中产生

`MLAAttentionImpl.do_kv_cache_update → _custom_ops.concat_and_cache_mla → _C_cache_ops.concat_and_cache_mla` 按 cache dtype 分派到 DS 专用 CUDA writer。`torch_bindings.cpp:809–813/912` 登记 op schema 和 CUDA 实现。入口检查 $d_c=512,d_R=64$、每行 656 字节与两输入元素大小为 2 字节；本页继续限定 BF16，大小检查本身不能证明任意两字节输入的数值语义相同。

对新 token 的 `slot_mapping[n]`，令

$$b=\left\lfloor\frac{\mathrm{slot}_n}{P}\right\rfloor,\qquad r=\mathrm{slot}_n\bmod P.$$

记录首地址按真实 block/entry stride 计算；负 slot 不写入。Launcher 以 slot mapping 长度确定本次行数。对于所选 NVIDIA 路径，第 $g$ 组的计算是

$$s_{n,g}=\max\left(\frac{\max_{j\in G_g}|c_{n,j}|}{448},\mathrm{FLT\_MIN}\right),\qquad
\widetilde c_{n,j}=Q_{\rm E4M3}\!\left(c_{n,j}/s_{n,g}\right),\quad |G_g|=128.$$

这里是带有限值饱和的浮点编码；448 是所用 [E4M3 格式](../fundamentals/numeric-formats/fp8-and-mx-data-formats.md)的最大有限幅值，`FLT_MIN` 是 FP32 最小正正规数，用于避免零幅值组得到零 scale。Scale 以 FP32 写入，不是整数零点或一个字节的 UE8M0 记录。96 个 threads 中，两 warp 处理四个 latent 组，一个 warp 对 RoPE 做 BF16 bit-copy。外部 `_k_scale` 虽出现在函数签名和调用中，DS writer 的主体没有使用它，不能再把普通 FP8 的 scalar 缩放叠加到这条路径。（`cache_kernels.cu:447–547/842–902`；`nvidia/quant_utils.cuh::scaled_convert`。）

## 4. Indexer 的生成与量化尺度

通用 `Indexer.forward` 从归一化 query latent `qr` 经 `wq_b` 生成 $[T,I,d_I]$ 查询。`hidden_states` 经融合的 `wk_weights_proj` 得到单个共用 key 与 $I$ 个 head weights，key 经过 `LayerNorm`。Q/K 的前 $d_R$ 维分别做 RoPE，其余维保留。这里的「融合投影」仍产生不同的 key 与 weights，并没有使用主 attention 的 $U_K/U_V$。（`deepseek_v2.py::Indexer.__init__/forward`。）

对所选 CUDA/BF16、$d_I=128,d_R=64$，`fused_indexer_q_rope_quant` 融合 query 的 RoPE、每 query/head 的 FP8 编码和 weight 缩放。旋转后的值先经历 BF16 舍入，再用整个 128 维的 absmax 选正的二次幂 scale $s^Q_{t,h}$。所传 weight 为

$$w'_{t,h}=w_{t,h}\,s^Q_{t,h}\,d_I^{-1/2}I^{-1/2}.$$

因此 Q scale 已进入 weights，FP8 indexer 调用的独立 `q_scale` 参数为 `None`；不能因没有单独传入 scale 就认为 Q 没有恢复尺度。该 head weight 没有经过 softmax，也不能当作主 attention 的概率。非融合 query 分支通过 `per_token_group_quant_fp8` 得到 Q/scale，再按相同接口构造 weights；本页不据此宣称所有分支逐位相同。（`sparse_attn_indexer.py::fused_indexer_q_rope_quant/_fused_indexer_q_rope_quant_kernel`、`deepseek_v2.py::Indexer.forward`。）

### 4.1 Indexer cache 的 132-byte 预算不等于逐 token 的连续记录

`DeepseekV32IndexerCache` 是独立注册的缓存层。$d_I=128$、quant block 128 时，每 token 的预算为 $128+4=132$ 字节；张量接口是 $[B_I,P,132]$。实际 writer 则按 **每 page 先放所有 token 的 key，再放所有 token 的 scale** 寻址。设 $A_b$ 为该 indexer page 的起始字节地址，第 $r$ 个 token 的地址是

$$A^{K}_{b,r,j}=A_b+r\,128+j,\qquad
A^{s}_{b,r}=A_b+P\,128+4r.$$

在 $P=64$ 时，前 8192 字节是 64 个 FP8 key，后 256 字节是 64 个 FP32 scales；`132` 是预算和 shape 的宽度，不能用 $A_b+132r$ 找 token 的 key。主 packed cache 则确实每 token 有 656 字节的联合记录，二者的布局不能混用。

`indexer_k_quant_and_cache_kernel` 对本 token 的 128 维 key 算

$$s^K_n=2^{\left\lceil\log_2\left(\max(\max_j|k_{n,j}|,10^{-4})/448\right)\right\rceil},\qquad
\widetilde k_{n,j}=Q_{\rm E4M3}(k_{n,j}/s^K_n).$$

`scale_fmt="ue8m0"` 在此表示将 scale 向上取为二次幂，**writer 仍存四字节 FP32 数值**。它不等同于主 cache 的四组 FP32 动态 scale，也不等同于其他格式的单字节指数存储。负 slot 跳过；插入发生在本层打分之前，当前 token 能被自己的 causal query 看见。（`cache_kernels.cu::indexer_k_quant_and_cache_kernel`、`sparse_attn_indexer.py::sparse_attn_indexer`。）

## 5. 分数、因果边界与共享 top-k buffer

本仓测试参考实现规定逐 head 点积、ReLU、乘 head weight 再对 heads 求和；测试使用浮点参考 Q/K。将这一计算与所读模型的量化和 scale 接口合写为以下整理者表达：

$$R_{t,j}=\sum_{h=1}^{I}w'_{t,h}\,\operatorname{ReLU}\!\left(\sum_{a=1}^{d_I}\widetilde q_{t,h,a}\,\widetilde k_{j,a}\,s^K_j\right),\qquad
\mathcal S_t=\operatorname{TopK}_{j\in\mathcal V_t}(R_{t,j},S).$$

$\mathcal V_t$ 是该 query 可见的请求内历史位置，包含当前 token、排除未来 token。由于 $s^Q>0$，把 Q scale 移到 ReLU 之外的 weight 中保持实数表达；但 FP8/BF16 舍入仍然存在。公式依据 `tests/kernels/attention/test_deepgemm_attention.py::_ref_fp8_mqa_logits/_ref_fp8_fp4_paged_mqa_logits` 与本地 wrapper，不是对外部 DeepGEMM 设备累积精度或实际结果的核验。

两阶段读同一份 indexer key，却采用不同访问接口：

- **Prefill：** `cp_gather_indexer_k_quant_cache` 按 indexer 的 block table，将各请求 key/scale 分别收集到连续 workspace；scale 的四字节 buffer 被 view 为 FP32。Metadata 给每个 query 指定 `cu_seqlen_ks/cu_seqlen_ke`，分别限定请求起点与因果终点。DeepGEMM logits 调用后，`top_k_per_row_prefill` 在同一范围内选取 token，并输出请求内位置。
- **Decode：** key cache 加一个 singleton head 轴成为 $[B_I,P,1,132]$ 的库接口，物理内容不重排。Paged logits 调用接收 Q、weights、逐 query context length、block table 与调度 metadata，随后根据条件选择 cooperative/persistent/普通 top-k。所选单 query decode 的可见长度是已经包含本轮新 token 的序列长度。

Prefill 中令某请求本轮 query 数为 $M$、写入后的完整长度为 $L$、在收集 workspace 的起点为 $a_i$，其本轮第 $u$ 个 query（从 0 计）的合法 logits 区间为

$$[\mathrm{start},\mathrm{end})=[a_i,\ a_i+(L-M)+u+1).$$

这使历史 prefix 与已经可见的新 token 都参与选择，同时排除新段未来位置。Prefill top-k 将选中 workspace 列减去请求起点，输出请求内位置；decode 的单 query 则以长度 $L$ 限定 $[0,L)$。有效候选不足 $S$ 时，未填条目保留 `-1`；不承诺 top-k 输出排序或相等分数的稳定顺序。（`indexer.py` 的 prefill metadata 构造、`sampler.cu::top_k_per_row_prefill`、`cooperative_topk.cuh`。）

两个 logits wrapper 都传 `clean_logits=False`，因此不能依赖整个分配矩阵的区间外元素已经清成 $-\infty$；本地 top-k 的合法范围参数也是契约的一部分。上游测试中的 masked reference 解释数学意图，不能替代真实 wrapper 条件。（`sparse_attn_indexer.py::sparse_attn_indexer`、`indexer.py::DeepseekV32IndexerMetadataBuilder`、`utils/deep_gemm.py`。）

模型为各层传入同一份 $[T_{\max},S]$ buffer。所选每层创建 Indexer 的配置下，本层先把需要的区域置为 `-1`，再写有效 top-k，attention 随后消费；下一层重新覆盖它。源码还支持 skip-topk 层复用已有选择，本页没有采用该配置。`-1` 表示没有选到有效 token，不是物理 page 0。Buffer 是本轮选择结果，不能用它代替随上下文增长的主 cache 或 indexer cache，也不能推导历史 token 已被释放。（`deepseek_v2.py::DeepseekV2Model/DeepseekV2MLAAttention`、`sparse_attn_indexer.py::sparse_attn_indexer`。）

## 6. 从请求内 token 位置到主 cache 或 workspace

Indexer 输出的 $j$ 是请求内逻辑 token 位置。`req_id_per_token` 将每条 query 行对应到请求 $i$；attention 必须用 **主 cache 的** block table 转换它。单设备下有效 token 的物理 slot 是

$$\mathrm{slot}(i,j)=\mathrm{block\_table}_c[i,\lfloor j/P\rfloor]P+(j\bmod P).$$

例如 $P=64$、某请求主 block table 为 `[7,2]`，逻辑位置 65 映射为 $2\times64+1=129$；若 indexer page 编号不同，这不改变所选逻辑 token 的身份。Slot 还需结合主缓存的 656-byte entry stride 才能得到字节地址。这与[runner 的新 token slot mapping](vllm-inference-execution.md#5-从请求列表变成紧凑张量)是同一分页关系的不同使用时点。

`triton_convert_req_index_to_global_index` 对负索引或 block-table 列越界输出 `-1`。Prefill 使用恢复后的 workspace 时，索引变成

$$\mathrm{workspace\_row}(i,j)=\mathrm{workspace\_start}_i+j.$$

这一变换同时可统计每 query 的有效 `topk_length`，但所选 wrapper 保持原列位置，不把它称为排序或压缩到前缀。转换函数不按请求真实 `seq_len` 或每 query 的因果位置再检查 $j$；有效历史和因果范围来自前一阶段的 metadata/top-k。所选合法单设备 top-k 的有效项为连续前缀，因此这个数量符合匹配 FlashMLA prefill 的前缀长度契约；任意内部空洞或 DCP owner 过滤则不能作同样推断，具体条件见[设备读取](dsa-sparse-device-execution.md#51-接口与-invalid-indices)。DCP owner 过滤与另一种 compact wrapper 不在本页单设备主线。（`sparse_utils.py::_convert_req_index_to_global_index_kernel/triton_convert_req_index_to_global_index`。）

## 7. 主 attention：packed decode 与 BF16 sparse prefill

主 cache 在 `MLAAttention.forward` 中先更新，再进入 `forward_impl`。Indexer 已在模型 wrapper 中执行；它的 key 插入与 top-k 先于主 attention 消费共享 buffer。`forward_impl` 对 `fp8_ds_mla` 保留字节容器，不作普通 FP8 的整张量 `.view(fp8_dtype)`。内容 query 经 $U_K$ 进入 512 维潜空间，再与 64 维 RoPE query 拼成 BF16 $[T,H,576]$。（`layers/mla.py`、`mla_attention.py::MLAAttention.forward/forward_impl`。）

下面的 prefill/decode 指 metadata 的 kernel 分组。所选 $H=128$/SM90 的默认 reorder threshold 为 256，并要求 uniform decode；短 extend 可能归入 decode 组，不与业务上「读 prompt/生成 token」一一对应。`sparse_mla_force_mqa=True` 阻止 dense MHA 路由，却不强制所有 prompt query 走 BF16 sparse prefill workspace。（`flashmla_sparse.py::FlashMLASparseMetadataBuilder`、`sparse_mla_attention.py::SparseMLACommonMetadataBuilder.build`。）

### 7.1 Decode 直接交付 packed cache 与物理 indices

所选 $H=128$ 不进入 `H<32` 的 mixed-batch 优化，而使用 separate prefill/decode metadata。单 query decode 将 Q 组织为 $[N_d,1,H,576]$，indices 为 $[N_d,1,S]$。`_fp8_flash_mla_kernel` 调用 `flash_mla_with_kvcache`，传入

- `k_cache=cache.view(uint8).unsqueeze(-2)`，即 $[B_c,P,1,656]$；
- 主 cache 的物理 indices、`is_fp8_kvcache=True`、`head_dim_v=512` 与模型 `softmax_scale`；
- scheduler metadata、dummy block table 与构造的 cache length buffer。

这里真实的请求内索引已经提前转换；不能把 dummy block table 当成请求的真实历史页表，也不能把其最大长度 buffer 当作每 query 的因果长度。稀疏集合的合法性由前面的选择链承接。CUDA 调用的返回 latent 输出再由 `_v_up_proj` 恢复 value 宽度，随后进入模型输出投影；cache FP8 不表示 query、投影或模型输出全部是 FP8。（`flashmla_sparse.py::_forward_fp8_kv_separate_prefill_decode/_fp8_flash_mla_kernel`、`mla_attention.py::forward_impl/_v_up_proj`。）

Decode helper 为设备接口将 head 数补至 64 或 128，必要时零填 query 并截回实际 heads；所选 $H=128$ 无此 padding。返回 LSE 在这一单设备调用中没有作为跨 rank 合并状态返回，`forward_mqa` 返回 `(attn_out,None)`，不能将它接成[普通 DCP 的 O/LSE 归约](vllm-mla-context-parallel-execution.md#6-跨-rank-合并需要恢复同一个-softmax-分母)。

### 7.2 Prefill 恢复联合 latent/RoPE，仍按 top-k 读取

Separate 分支按完整请求长度把 metadata 中的 prefill 请求分组，使每组序列能放入 workspace；各组的 `workspace_start` 从零计。这里 $L_i$ 是写入后的完整 `seq_len`，包含旧 context 和本轮 query。`cp_gather_and_upconvert_fp8_kv_cache` 用主 block table 读取该组的 **全部 $L_i$ 行**，把 656-byte packed 记录恢复为 BF16 $[\sum_iL_i,576]$：

$$\widehat c_{j,a}=\operatorname{BF16}\left(s_{j,\lfloor a/128\rfloor}\operatorname{decode}_{\rm E4M3}(\widetilde c_{j,a})\right),$$

位置 key 从 byte 528 开始 bit-copy 到工作行的第 512 维。恢复的是已量化历史，不能消除写入时的舍入，也不能用单一 `_k_scale` 恢复四个内嵌 scale。Workspace 可含当前 query 的未来位置，gather 范围与因果可见范围不同；实际候选仍受之前生成的 top-k indices 限制。（`cache_kernels.cu::cp_gather_and_upconvert_fp8_kv_cache`；入口要求 BF16 destination 与 width 576。）

然后 `_bf16_flash_mla_kernel → flash_mla_sparse_fwd` 使用 BF16 Q、恢复后的联合 workspace、转换后的 workspace indices、`topk_length` 与模型 score scale，输出 latent 宽度 512，再接 $U_V$。这里没有经 `kv_b_proj` 展开成每头 K/V；**BF16 sparse prefill 与普通 dense MHA prefill 是两条不同路径**。Top-k 可以减少主 sparse attention 的候选集合，但这条 prefill 仍收集完整历史，不能把总恢复流量也写成 $S$ 行。（`flashmla_sparse.py::_forward_fp8_kv_separate_prefill_decode/_bf16_flash_mla_kernel`。）

框架 Hopper BF16 sparse helper 将 head 数按 64 补齐，Blackwell 按 128 补齐；匹配外部 API 实际只接受 64 或 128 heads，不能仅凭补齐规则推断任意倍数都可用。所选 128 heads 无需 padding。（外部 FlashMLA `api/sparse_fwd.h:200–207`。）$H<32$ 时另一个 mixed-batch 分支将 prefill/decode query 合在一个 FP8 decode 接口中，不经过这份 BF16 prefill workspace；这里只登记分支条件，不把 separate 路径的成本套给它。

### 7.3 没有强制 MQA 时，短 prefill 可能回到 dense MHA

`MLAAttention.forward_impl` 在 sparse backend 上仍检查：存在 prefill backend、`prefill_max_seq_len<=topk_tokens` 且 `sparse_mla_force_mqa=False` 时，metadata prefill 组可以进入共享的 `forward_mha`；否则把该组行也归入 `forward_mqa`。因此 DSA backend 名称不能证明所有 prefill 都实际执行 top-k sparse kernel。

Dense MHA 的历史读取也针对 packed dtype 分流到 `cp_gather_and_upconvert_fp8_kv_cache`，然后才经 `kv_b_proj` 展开 K/V。`gather_and_maybe_dequant_cache` 的普通连续 320/576 元素与外部 scalar scale 契约不能替代专用恢复。这条接续解释已有[普通 prefix prefill](mla-cache-and-attention-execution.md#19-vllm-prefill展开新-token再分块恢复历史)在缓存格式上的必要变化；本页主线显式强制 MQA，未将 dense 与 sparse 两种结果做运行对比。（`mla_attention.py::forward_impl/MLACommonBaseImpl._compute_prefill_context`。）

## 8. 存储、选择与误差要分别记账

`MLAAttentionSpec.real_page_size_bytes` 对所选 V3.2 packed 格式直接返回 $656P$，内嵌 16-byte scale 已计入。$P=64$ 时，主 page 为 41,984 字节；同一 latent/RoPE 宽度的普通 BF16 page 是 73,728 字节，普通连续 FP8 page 是 36,864 字节。Packed 比普通 FP8 多 80 字节/token，来自保留 BF16 RoPE 的额外 64 字节与四个 scales 的 16 字节；不能仅以「FP8」推断它们同大小。

Indexer 对 $d_I=128$ 的每 page 另需 $132P=8448$ 字节。只为解释有效历史载荷，在拥有 Indexer 的同一层、相同历史数 $L$ 且不计页尾空位时，两种池共同占 $L(656+132)=788L$ 字节。实际分配还需分别按 page 取整，并考虑 KV group 的 page padding、预留 block、top-k buffer、logits 与恢复 workspace；这个算术不是引擎峰值内存测量。（`deepseek_v2.py::DeepseekV32IndexerCache.get_kv_cache_spec`、`kv_cache_interface.py::MLAAttentionSpec/AttentionSpec`。）

与 [Attention 误差传播](../theory/attention-quantization-error-propagation.md)连接时，应拆开三个作用：

1. Indexer Q/K 的量化改变 $R_{t,j}$，可能改变所选集合 $\mathcal S_t$。
2. 主 latent 的量化同时改变被选 token 的内容分数和 latent value；主 RoPE 分支保持 BF16，却仍包含模型计算本身的舍入。
3. 给定集合上的主 attention 归一化，与全历史 attention 的归一化范围不同；恢复 BF16 不能补回未被选择的 token。

上述区分是根据调用链整理的误差对象，不是该模型的精度结论。判断收益还要计入 indexer 全历史打分、top-k、读取恢复和实际调度；缓存字节或局部候选数不能代替端到端延迟、吞吐与质量证据。

## 9. 核读边界

本页核读默认 registry/通用模型入口、两种持久缓存、Q/weights 与 key 量化、indexer 的 metadata/调用、top-k buffer、请求内到物理/workspace 索引、主 packed CUDA writer/专用恢复，以及 FlashMLA sparse 的 prefill/decode 框架调用。数学与地址例子用于解释静态接口，没有执行上游测试或模型。

`cmake/external_projects/flashmla.cmake` 的默认下载分支固定外部 FlashMLA commit `a8f794d1251cbfd88a5011445dd5582289c727e4`；设置 `FLASH_MLA_SRC_DIR` 时则使用提供的源码目录。构建从外部源码生成 `third_party/flashmla/flash_mla_interface.py`。该文件和库设备 kernel 没有包含在所读 vLLM source 树中。本页主体核到框架调用、writer/恢复与默认依赖版本；现已另行保存该固定外部源码，并在[DSA 设备执行](dsa-sparse-device-execution.md)接续所选 SM90 的 QK/PV、scale 舍入、online softmax、LSE/split 合并、无效候选与调度。本页只直接补核 head 与前缀契约，不扩大为 SM100 或其他设备分支已核。

匹配 DeepGEMM 的 SM90 FP8 logits 设备计算、页平面读取与调度也已在上述设备页接续；其他 DeepGEMM 分支、全部 top-k 算法内部、非默认融合模型、FP4 indexer、V4/其他压缩格式、SGLang DSA、其他 backend、DCP/PCP、LoRA、图执行与多 stream 未由本轮完整核验。没有安装依赖、编译、运行 GPU/服务、测量误差或性能。本页承接两种表示到框架读取的链，设备计算的版本、条件与边界由续页承接。

## 来源身份

| 来源 | 固定版本 | 实际使用范围 |
| --- | --- | --- |
| vllm-project/vllm（`https://github.com/vllm-project/vllm/tree/568afb3a13806beb53bb2e6bd518269357b237c0`） | `568afb3a13806beb53bb2e6bd518269357b237c0` | 上述模型、缓存、indexer、sparse backend、CUDA writer/恢复、CMake 和测试 reference；文件与函数就近定位 |
| vllm-project/FlashMLA（`https://github.com/vllm-project/FlashMLA/tree/a8f794d1251cbfd88a5011445dd5582289c727e4`） | `a8f794d1251cbfd88a5011445dd5582289c727e4` | 本页直接补核 Python 前缀说明与 prefill API head gate；SM90 设备主链见续页 |
