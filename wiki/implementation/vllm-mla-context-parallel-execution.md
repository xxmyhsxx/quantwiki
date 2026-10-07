---
title: vLLM MLA 的 DCP 执行：缓存分片、局部 Attention 与跨 rank 合并
type: implementation
tags:
  - attention
  - kv-cache
  - serving
  - low-rank
sources:
  - raw/repositories/2026-09-21/vllm/source/vllm/config/parallel.py
  - raw/repositories/2026-09-21/vllm/source/vllm/config/vllm.py
  - raw/repositories/2026-09-21/vllm/source/vllm/envs.py
  - raw/repositories/2026-09-21/vllm/source/vllm/distributed/parallel_state.py
  - raw/repositories/2026-09-21/vllm/source/vllm/distributed/communication_op.py
  - raw/repositories/2026-09-21/vllm/source/vllm/distributed/device_communicators/cuda_communicator.py
  - raw/repositories/2026-09-21/vllm/source/vllm/distributed/device_communicators/pynccl.py
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/models/deepseek_v2.py
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/layers/mla.py
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/layers/linear.py
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/layers/attention/mla_attention.py
  - raw/repositories/2026-09-21/vllm/source/vllm/v1/kv_cache_interface.py
  - raw/repositories/2026-09-21/vllm/source/vllm/v1/core/kv_cache_utils.py
  - raw/repositories/2026-09-21/vllm/source/vllm/v1/core/kv_cache_coordinator.py
  - raw/repositories/2026-09-21/vllm/source/vllm/v1/core/single_type_kv_cache_manager.py
  - raw/repositories/2026-09-21/vllm/source/vllm/v1/core/kv_cache_manager.py
  - raw/repositories/2026-09-21/vllm/source/vllm/v1/worker/gpu_worker.py
  - raw/repositories/2026-09-21/vllm/source/vllm/v1/worker/gpu_model_runner.py
  - raw/repositories/2026-09-21/vllm/source/vllm/v1/worker/block_table.py
  - raw/repositories/2026-09-21/vllm/source/vllm/v1/worker/cp_utils.py
  - raw/repositories/2026-09-21/vllm/source/vllm/v1/attention/backend.py
  - raw/repositories/2026-09-21/vllm/source/vllm/v1/attention/backends/utils.py
  - raw/repositories/2026-09-21/vllm/source/vllm/v1/attention/backends/mla/triton_mla.py
  - raw/repositories/2026-09-21/vllm/source/vllm/v1/attention/backends/mla/prefill/selector.py
  - raw/repositories/2026-09-21/vllm/source/vllm/v1/attention/backends/mla/prefill/base.py
  - raw/repositories/2026-09-21/vllm/source/vllm/v1/attention/backends/mla/prefill/flash_attn.py
  - raw/repositories/2026-09-21/vllm/source/vllm/v1/attention/backends/fa_utils.py
  - raw/repositories/2026-09-21/vllm/source/vllm/vllm_flash_attn/flash_attn_interface.py
  - raw/repositories/2026-09-21/vllm/source/cmake/external_projects/vllm_flash_attn.cmake
  - raw/repositories/2026-09-21/vllm/source/vllm/v1/attention/ops/triton_decode_attention.py
  - raw/repositories/2026-09-21/vllm/source/vllm/v1/attention/ops/common.py
  - raw/repositories/2026-09-21/vllm/source/vllm/v1/attention/ops/dcp_alltoall.py
  - raw/repositories/2026-09-21/vllm/source/vllm/v1/attention/ops/merge_attn_states.py
  - raw/repositories/2026-09-21/vllm/source/vllm/v1/attention/ops/triton_merge_attn_states.py
  - raw/repositories/2026-09-21/vllm/source/csrc/libtorch_stable/cache_kernels.cu
  - raw/repositories/2026-09-21/vllm/source/csrc/libtorch_stable/attention/merge_attn_states.cu
updated: 2026-10-07
---

# vLLM MLA 的 DCP 执行：缓存分片、局部 Attention 与跨 rank 合并

MLA 让多个 query heads 共享联合 latent/RoPE 历史；Decode Context Parallel（DCP）再把这份历史按 token 分到多个 rank。一个 head 仍需读取完整可见历史，因此分片地址、局部 softmax 的分母、输出回到哪个 head owner，必须一起解释。本页接续[单设备 MLA 缓存与执行](mla-cache-and-attention-execution.md)和[模型结构与矩阵吸收](../fundamentals/model/multi-head-latent-attention.md)，依据固定 vLLM 源码展开这条分布式链。

## 1. 所选路径与两种并行轴

来源为 vLLM commit `568afb3a13806beb53bb2e6bd518269357b237c0`。主线选择普通非 DSA DeepSeek MLA、CUDA、Model Runner V1（显式 `VLLM_USE_V2_MODEL_RUNNER=0`）、eager、DP=1/PCP=1、TP=2/DCP=2，BF16 query 与 `auto`/BF16 联合 cache，物理页 $P=16$、interleave $I=4$。Decode 每请求本轮恰好一个 query，默认 `ag_rs`；另核 `a2a` 合并和公共 FlashAttention prefix prefill。它是静态研读的教学配置，没有启动服务或验证设备支持组合。

显式 runner 环境值由 `VllmConfig.use_v2_model_runner` 采用；`Worker.init_device` 据此选择 `v1/worker/gpu_model_runner.py`，相反分支为 `v1/worker/gpu/model_runner.py`。本页的输入准备与 metadata 来自前者。（`config/vllm.py::use_v2_model_runner`、`envs.py`、`v1/worker/gpu_worker.py`。）

一般记 TP 大小为 $T$、DCP 大小为 $D$、模型 heads 为 $H$，每 TP rank 的 heads 为 $h=H/T$。PCP=1 时 DCP 复用 TP ranks，配置要求 $T$ 能被 $D$ 整除，不扩大进程 world size。一个 DCP 子组汇集 $H_g=Dh$ 个 heads；只有 $D=T$ 时 $H_g=H$。例如 TP=4/DCP=2 是两个 DCP 子组，不是额外增加两个设备。（`config/parallel.py::ParallelConfig`、`distributed/parallel_state.py::initialize_model_parallel`。）

TP 分配 head 对应的投影权重；DCP 分配历史 token。若各 rank 仅让自己的 $h$ 个 heads 查询自己的历史份额，再拼接结果，每个 head 都会漏掉其他 rank 的历史。所选 decode 通过收集 query heads、局部 attention 和 LSE 合并解决这一交叉关系。

`TritonMLAImpl.can_return_lse_for_decode=True`，公共 CP 兼容性检查要求 DCP decode 能返回 LSE。Triton 的页大小为 16 的倍数；配置还要求 $I\le P$ 且 $I\mid P$。BF16 cache 的 CUDA 构造检查要求 SM80+。公共 builder 在此路径使用单 query decode 分类，未开启 DCP varlen 支持；其他 decode backend、PCP、多 query/推测 decode、packed cache、DSA、ROCm、LoRA 和完整图执行不由此例承诺。（`triton_mla.py`、`attention/backend.py`、`worker/cp_utils.py::check_attention_cp_compatibility`、`mla_attention.py::MLACommonMetadataBuilder`、`config/vllm.py::validate_block_size`。）

## 2. 调度 block 与每 rank 的物理 page

普通联合缓存仍是 $[B_{\rm pool},P,d_c+d_R]$。一条 token 的 normalized latent 与已旋转位置 key 保存在同一行；DCP 不把末轴切成 $d_c/D$。单 KV group 的调度与 prefix hash 单位则扩大为 $PD$ 个全局 token：一个管理 block id 对应该组中每 rank 的一张 $P$ 行本地 page。（`MLAAttentionSpec`、`MLACommonBackend.get_kv_cache_shape`、`kv_cache_utils.py::resolve_kv_cache_block_sizes`、`SingleTypeKVCacheManager.__init__`、`UnitaryKVCacheCoordinator`。）

对没有共享、没有 lookahead、按已用长度恰好分配的请求，$L$ 个全局 token 的教学计费为

$$B_a=\left\lceil\frac{L}{PD}\right\rceil,\qquad
M_{\rm group}=DB_aP(d_c+d_R)b,$$

其中 $b$ 是每元素字节数。各 rank 预留 $B_aP$ 行，组内有效行合计仍是 $L$；上式计入尾页空行，未含页表、scale、通信、attention workspace、权重或其他层。$d_c=512,d_R=64$ 的 BF16 行为 1,152 字节，一张 $P=16$ 的本地 page 为 18,432 字节。分片降低每 rank 的历史份额，不能据 $D$ 直接宣称整机峰值显存缩小 $D$ 倍。

普通复制 KV A 权重路径中，KV 输入投影不按 TP 拆 latent；结合各子组重复的 owner 规则，同一 TP group 的 $T/D$ 个 DCP 子组各保留一份逻辑历史。这个条件下，整个 TP group 的预留 payload 是 $TB_aP(d_c+d_R)b$，有效 payload 为 $(T/D)L(d_c+d_R)b$。（`deepseek_v2.py` 的复制 KV A/fused A 投影、`linear.py`、`mla.py` 与 DCP 分组/slot 规则。）全模型的逻辑缓存容量见[MLA 容量推导](../fundamentals/model/multi-head-latent-attention.md#4-缓存容量怎样计算)。持久页、有效 token 与临时通信张量是不同口径。

## 3. 全局 token 如何找到 owner 和本地槽位

`cp_kv_cache_interleave_size=I` 表示连续 $I$ 个 token 落 rank 0，随后 $I$ 个落 rank 1，循环至 rank $D-1$。$I=1$ 为逐 token 轮转，$I=P$ 为逐页轮转；它不等于 attention 的 split 数。（`config/parallel.py`。）

对请求 $a$ 的零起点全局位置 $p$，在 $I\mid P$、管理页与 kernel 页同为 $P$ 的所选路径，worker 的等价地址公式为

$$r(p)=\left\lfloor\frac pI\right\rfloor\bmod D,\qquad
j(p)=I\left\lfloor\frac p{DI}\right\rfloor+(p\bmod I).$$

$r(p)$ 是 owner，$j(p)$ 是该 owner 内去掉其他 rank token 后的连续本地位置。各 rank 持有自己的本地物理页表 $\mathrm{BT}$，写入槽位为

$$\mathrm{slot}_r(a,p)=\begin{cases}
\mathrm{BT}[a,\lfloor j(p)/P\rfloor]P+(j(p)\bmod P),&r=r(p),\\
-1,&r\ne r(p).
\end{cases}$$

源码先取虚拟块 $\lfloor p/(PD)\rfloor$，再压缩其块内位置；上式是同一运算的整理者改写。非 owner 的 `PAD_SLOT_ID` 是不写本地 cache，不是指向另一设备的地址。源码另有 memory block 到 kernel block 的拆分，本例不包含它。（`worker/block_table.py::BlockTable/_compute_slot_mapping_kernel`。）

**教学例子：**$P=16,D=2,I=4$，请求的本地页表为 `[10,3]`。

| 全局位置 $p$ | owner | owner 内位置 $j$ | 页表索引 |
| --- | --- | --- | --- |
| 0–3 | 0 | 0–3 | 0 |
| 4–7 | 1 | 0–3 | 0 |
| 8–11 | 0 | 4–7 | 0 |
| 12–15 | 1 | 4–7 | 0 |
| 24–27 | 0 | 12–15 | 0 |
| 28–31 | 1 | 12–15 | 0 |
| 32–35 | 0 | 16–19 | 1 |

处理 $p=34$ 时，rank 0 取 $j=18$，在第二张物理 page 3 写 slot $3\times16+2=50$；rank 1 的 slot 为 −1。这是本设备 cache 的行编号，还需 dtype 和 stride 才能得到字节地址；两设备的 page 3 也不是同一块内存。

## 4. 全局长度、有效本地长度与规划上界

Model Runner V1 更新请求页表，用已计算 token 数加 query 内偏移生成全局 `positions`，并令 `seq_lens=已计算长度+本轮 query 长度`。普通单 query decode 的位置为 $L-1$，长度 $L$ 包含当前 token。全局位置交给 block table 生成各 rank 写入槽位。（`gpu_model_runner.py::_update_states/_prepare_inputs`。）

总长度 $L$ 在 rank $r$ 的有效行数是

$$L_r=I\left\lfloor\frac L{DI}\right\rfloor+
\operatorname{clip}(L\bmod(DI)-rI,0,I).$$

每个完整轮转贡献 $I$ 行，最后一轮按 rank 顺序填充，故 $L_r$ 可以不同，且 $\sum_rL_r=L$。它既不是给所有 rank 共用的取整 $L/D$，也不是已分配的 $B_aP$ 行数。`GPUModelRunner._build_attention_metadata` 计算并传入 `dcp_local_seq_lens`；公共 MLA builder 为 decode 保留全局 `dcp_tot_seq_lens`，将 `decode.seq_lens` 换成本地长度。（`backends/utils.py::get_dcp_local_seq_lens`、`mla_attention.py::MLACommonMetadataBuilder.build`。）

为避免同步读取各 rank 的最大值，scratch/split 规划使用上界

$$L_{\max,{\rm local}}^{\rm bound}=I\left\lceil\frac{L_{\max}}{DI}\right\rceil.$$

实际 Triton 读取仍按精确 $L_r$ mask。上例处理 $p=34$ 后，$L=35,L_0=19,L_1=16$，两个管理 block 在每 rank 预留 32 行；规划上界为 20。rank 0 最后一条有效行 $j=18$ 恰是当前 token，空行不参与 attention。（`triton_mla.py::forward_mqa`、`triton_decode_attention.py::_fwd_grouped_kernel_stage1`。）

普通 writer 在 attention 前接受本 rank 的 slot mapping；`concat_and_cache_mla_kernel` 对负 slot 直接返回，对正 slot 联合写 latent/RoPE。所有 rank 可以产生当前 token 的联合输入，只有 owner 写持久页。读取端遍历 $j=0,\ldots,L_r-1$，用 $\mathrm{BT}[a,\lfloor j/P\rfloor]$ 和 $j\bmod P$ 找本地历史；decode kernel 不再次判断全局 owner。RoPE 在写入前按全局位置计算，分片必须保留 latent 与 rotated key 的配对。（`MLAAttention.forward/do_kv_cache_update`、`cache_kernels.cu`、Triton grouped stage 1。）

## 5. Decode 的 query 收集与局部 attention

`MLAAttention.forward_impl` 先在本 TP rank 上用 $W_{UK}^{T}:[h,d_K,d_c]$ 吸收 key 上投影，得到内容 query $[B,h,d_c]$，与位置 query $[B,h,d_R]$ 拼成工作 query。$B$ 在此是 decode token 数，也等于所选单 query 路径的请求数。

随后 `get_dcp_group().all_gather(mqa_q,dim=1)` 沿 head 轴得到 $[B,H_g,d_c+d_R]$。每 rank 收到同一组已吸收的 query heads，在自己的历史份额上计算它们；这一步不收集历史 KV，也不要求复制其他 rank 的 $W_{UK}^{T}$。（`mla_attention.py::forward_impl`，吸收 BMM、DCP all-gather、`forward_mqa` 的相邻分支。）

| 对象 | Shape | 所选 BF16 路径的语义 |
| --- | --- | --- |
| 收集后的 query | `[B,H_g,d_c+d_R]` | BF16，内容投影已吸收 |
| 本 rank split scratch | `[B,H_g,S,d_c+1]` | FP32，非空 split 的归一化 latent 输出与自然 LSE |
| 局部 $O_r$ | `[B,H_g,d_c]` | BF16，在该 rank 的历史内归一化 |
| 局部 $z_r$ | `[B,H_g]` | BF16，自然对数 LSE |

Triton 的 split-KV 再拆的是**本 rank 的局部历史**；$S$ 由局部长度上界与 SM 数选择。最终 O/LSE tensor 用 Q dtype 创建，FP32 scratch 不代表交换的 LSE 一定是 FP32，也不能把 decode 的 $[B,H_g]$ LSE 当作 prefill 的 $[h,N_p]$。（`triton_mla.py::_compute_num_kv_splits/forward_mqa`、`triton_decode_attention.py` 两阶段。）

普通 FP8 cache 分支在本地加载时将 latent 与位置 key 用共同 `_k_scale` 恢复到 Q 工作 dtype；latent 同时参与 PV。所读 Triton 关闭公共 FP8 query 量化，局部 O/LSE 仍随 Q dtype，CUDA FP8 cache 构造检查要求 SM89+。cache 位宽不等于 query 通信或输出位宽；本页主线仍为 BF16，不覆盖 `fp8_ds_mla`。（同一 Triton impl/kernel；普通 scale 规则见[单设备缓存接口](mla-cache-and-attention-execution.md#17-vllm-联合分页缓存与-fp8-scale)）。

## 6. 跨 rank 合并需要恢复同一个 softmax 分母

固定一个请求/head，各 rank 的有效历史集合 $J_r$ 不重叠，合计为完整历史。以真实 score $\ell_j$ 与 normalized latent $c_j$ 定义

$$Z_r=\sum_{j\in J_r}e^{\ell_j},\qquad z_r=\ln Z_r,\qquad
O_r=\frac{\sum_{j\in J_r}e^{\ell_j}c_j}{Z_r}.$$

非空分片的 $O_r$ 已除以不同的 $Z_r$，不能直接求和或按 rank 平均。完整结果是

$$z=\ln\sum_r e^{z_r},\qquad w_r=e^{z_r-z},\qquad
O=\sum_r w_rO_r=\frac{\sum_j e^{\ell_j}c_j}{\sum_j e^{\ell_j}}.$$

实现先取 $m=\max_r z_r$，用 $e^{z_r-m}$ 统一基准再归一化。两个 rank 各有一个 value 0、10，score 为 0、$\ln3$，正确输出为 7.5，直接平均局部输出得到 5。空分片视为 $Z_r=0,z_r=-\infty$，其零贡献还需要实现正确处理局部输出，见 §8。

这是合并源码的实数恒等式解释。不同浮点求和顺序和 BF16 O/LSE 舍入不保证与单设备逐位一致。若 backend 返回 base-2 LSE，须改用 $2^{z_r-z}$ 与 $\log_2$；公共入口把 `impl.lse_base_on_e` 传给合并函数，所选 Triton 使用自然底。（`ops/common.py::_correct_attn_cp_out_kernel`、`ops/dcp_alltoall.py::_dcp_a2a_unpack_combine_kernel`、`attention/backend.py`。）

## 7. AG+RS 与 A2A 怎样返回本 rank 的 heads

PCP=1、默认 `ag_rs` 的 `cp_lse_ag_out_rs` 执行三件事：收集 LSE 成 $[D,B,H_g]$；计算全局 LSE 和本 rank 权重，原地将 $O_r$ 校正为 $w_rO_r$；对校正输出沿 head 轴 SUM reduce-scatter，返回 $[B,h,d_c]$。第二步尚未跨 rank 求和，第三步也不需再除以 $D$。每 rank 最终收回自己最初贡献 query 的 heads，但其结果已覆盖全部历史。（`common.py::_cp_lse_common/correct_attn_out/cp_lse_ag_out_rs`、`GroupCoordinator.reduce_scatter`、CUDA communicator 与 PyNccl 的 SUM 调用。）

所选链的 DCP 通信因此是 **query head all-gather → LSE all-gather → weighted O reduce-scatter**。没有 all-gather 全部 O 到每个设备。helper 可额外返回本 rank 的全局 LSE，公共 MLA 调用未请求它。

`dcp_comm_backend="a2a"` 保留前面的 query head all-gather，仅改变 O/LSE 合并。`dcp_alltoall.py` 顶部文字暗示不再 gather 完整 Q，普通 MLA 的实际公共调用却仍先 gather Q，应以调用分支为准。

每个来源 rank 把 $H_g$ 个 heads 按目标 head owner 分成 $D$ 组，每组 $h$ 个，send/recv 为 $[D,B,h,d_c+p]$。BF16 O 时 $p=2$：先将 LSE 转 FP32，再把其 32 bits 拆成两个 16-bit word、bitcast 成两个 BF16 lanes 传送。这是保存 FP32 位表示，不是把 LSE 数值量化成两个 BF16；已经在局部输出时舍入为 BF16 的精度不会因此恢复。（`dcp_alltoall.py::_dcp_a2a_lse_pack_dim/_dcp_a2a_pack_send_kernel/dcp_a2a_lse_reduce`。）

`dist.all_to_all_single(...,group=cp_group.device_group,async_op=True)` 交换等长块并等待完成。接收端第一个轴变成历史来源 rank；每个接收 rank 解包所有历史分片对**自己 heads**的 O/LSE，用 FP32 accumulator 加权，写回 BF16 $[B,h,d_c]$。可选全局 LSE 为 FP32，当前调用同样未请求。

这条链变为 **query head all-gather → packed O/LSE all-to-all**。配置的 3→2 指这些 DCP attention collective 的次数，不能据此推导传输字节、真实时延或服务收益；本轮只核到 PyTorch/NCCL 调用接口，没有核读 NCCL 内部或测量网络。

两种合并之后才用本 rank 的 $W_{UV}:[h,d_c,d_V]$ 做 value 上投影，生成 $[B,h d_V]$，交给 wrapper 的 `o_proj`。DeepSeek `RowParallelLinear` 在 `reduce_results=True`、TP>1 时另对投影后的 hidden 输出做 TP all-reduce；所选 DP=1 不满足 SP MoE 开启条件，使用这一层内 reduction。构造器也允许 False，不能推广为所有外层融合调用都在此 all-reduce。它汇合不同 head 权重分区的贡献，与 DCP 合并历史分片的 softmax 状态是两项不同职责，不能合并计为同一次通信。（`mla_attention.py::_v_up_proj/process_weights_after_loading`、`mla.py::MultiHeadLatentAttentionWrapper.forward`、`deepseek_v2.py::DeepseekV2DecoderLayer`、`config/parallel.py`、`linear.py::RowParallelLinear.forward`。）

## 8. 空历史分片的数值边界

整体历史非空不保证每 rank 非空。主例 $D=2,I=4$，$1\le L\le4$ 时 rank 1 的 $L_1=0$。Triton stage 2 跳过全部空 split，保留 accumulator=0、指数和=0、最大值=−∞，最后无条件写 `acc/e_sum` 和 `e_max+log(e_sum)`。按该算式，局部 O 为 0/0，LSE 为 −∞；预先 zero O 不会阻止 kernel 覆写。（`triton_decode_attention.py::_fwd_kernel_stage2`，空循环与最终 store。）

AG+RS 将 NaN/+∞ LSE 视为 −∞，计算零 factor 后还执行 `where(factor==0,0,output)`，显式抹去空分片的 NaN O。所有 LSE 无效时，它采用 O=0、LSE=−∞ 的实现约定；这不是对没有有效 key 的 softmax 作数学定义。（`common.py::_correct_attn_cp_out_kernel`。）

A2A 也清理无效 LSE，并把 NaN weight 改为 0，但累加式仍是 `load(O).to(float32)*weight`，没有对应的 O 掩码。与所选 Triton 空分片输出组合时，静态代码存在 `NaN*0` 继续污染累计输出的路径。因此不能将这两个 backend 写成“空分片处理同样已保证”。这是源码算式分析，未在 GPU 复现，也未修改上游快照；其他 MLA backend 的空输出约定需另核。（`dcp_alltoall.py::_dcp_a2a_unpack_combine_kernel`。）

## 9. Prefix prefill 先收集历史 cache，再计算展开 MHA

`TRITON_MLA` 的单 query 边界用于 decode 分类与 kernel；其 impl 继承公共 `forward_mha`，prefill 使用独立 backend。本节只讨论 metadata 已划入 prefill 区间的请求；某些短 extend 可以按 threshold 被归为 decode，不能按调度名称判断执行路径。（`MLACommonMetadataBuilder`、`backends/utils.py::split_decodes_and_prefills`。）

当前新段直接从前向参数的 latent/RoPE 经本 rank `kv_b_proj` 展开 K/V，调用 `run_prefill_new_tokens(causal=True)`。它没有因 cache ownership 截去本轮非 owner 的输入，也没有像 decode 那样收集 query heads。历史长度则取 $L_a^{\rm hist}=\mathrm{seq\_len}_a-\mathrm{query\_len}_a$；有历史时，新段同时返回 LSE。

历史 metadata 将原始 workspace 行容量 $W$ 分给 $N_{\rm ctx}$ 个有 context 的请求，并向下取整，得到全局 chunk 容量 $C=P\lfloor W/(N_{\rm ctx}P)\rfloor$。每 rank 的 padded 历史长度为 $I\lceil L_a^{\rm hist}/(DI)\rceil$，local chunk 容量为 $J=I\lceil C/(DI)\rceil$，第 $n$ 次 local 起点为 $nJ$。builder 要求 $W\bmod D=0,C\bmod D=0$；通信后的重排还检查有效总行数与最大请求长度等于全局 metadata。所选主例的 $C$ 为 16 的倍数，也为 $DI=8$ 的倍数；不据这些公式宣称所有可填入配置的 $D/I/P$ 组合均能完成 prefill。（`mla_attention.py::build_mla_chunked_context_metadata/MLACommonMetadataBuilder.__init__`。）

一个静态边界是 $P=16,D=4,I=8,C=16,L^{\rm hist}=32$：它满足上述 $I\mid P,C\bmod D=0$，但 $J=8$，孤立看每请求的重排算术，四 rank 的首轮有效片段合计 32 行，全局 metadata 却要求 16 行。真实流程还有更早的容量检查：由 $C=16$ 得 $W<32N_{\rm ctx}$，首轮每个有 context 的请求至少预留 8 个 local padded 行，故 `toks`=$8N_{\rm ctx}>W/4$，会先触及 `_context_parallel_compute_prefill_context` 的 `toks<=W/D` assert，尚未到 all-gather/reorg。默认 $I=1$ 的对应算术相符；本教学主例 $I=4,D=2$ 也有完整轮转对齐。这个例子只核对整数算术与源码条件，未复现真实运行，也未声称其他上层条件已全部排除。配置合法性和一条完整执行链的成立需要分别核查。

每个历史 chunk 依次执行：

1. `cp_gather_cache` 按本 rank 页表、local 起点和 padded offsets，把 paged 联合 cache 复制成 $[T_{\rm pad},d_c+d_R]$。BF16 主线源/目标 dtype 相同，复制不作反量化，local 起点不要求整页对齐。
2. DCP `all_gather(dim=0)` 收集为 $[DT_{\rm pad},d_c+d_R]$，**通信前没有展开每头 K/V**。预分配 workspace 为 $[W+W/D,d_c+d_R]$，前 $W/D$ 行供 local gather，后 $W$ 行存收集结果；实际 all-gather 仍返回临时 tensor 再 `copy_`，不能把 workspace 当作全部峰值内存。
3. `reorg_kvcache` 按请求收集各 rank 的有效 local 片段、丢弃 padding，恢复 request-major 的联合历史。它按“请求 → rank 片段”拼接，不一定恢复原 token 的交错时间顺序。历史全部早于 query、无因果 mask，K/V 同时置换且 RoPE 已保存在 key，故实数 attention 不受此排列影响。
4. 完整历史 latent 经本 rank 的 `kv_b_proj` 展开到 $[T_{\rm hist},h,d_K+d_V]$，拆 K/V、拼 RoPE，调用 `run_prefill_context_chunk(causal=False)`。该 rank 对自己的 heads 计算当前历史 chunk 的完整 attention。

（`mla_attention.py::_context_parallel_compute_prefill_context/reorg_kvcache/forward_mha`、`cache_kernels.cu::cp_gather_cache`、`FlashAttnPrefillBackend`。）

此处跨 rank 通信是 **历史 latent cache all-gather**；每 rank 已拿到完整有效历史 chunk，不再跨 rank 归约 O/LSE。历史 chunk 间，以及历史与当前 causal 新段之间，用 §6 同样的分母恢复规则在本 rank 合并。Prefill 最终逐头 O 为 $[N_p,h,d_V]$，LSE 为 $[h,N_p]$、自然底；需要 V padding 的分支可在历史 chunk 合并期间保留较宽 O，公共层最后截回 $d_V$。空历史 chunk 经 `mask_empty_context` 清零 O、LSE 置 −∞。没有历史的请求直接使用新段结果。（公共 `forward_mha`、`ops/merge_attn_states.py`、CUDA/Triton merge/mask。）

普通 FP8 cache 的历史分支在 **本地 gather 时**用 `_k_scale` 反量化到工作 dtype，再通信恢复值。当前新段仍使用前向输入，不先经历相同 cache 舍入。不能把该路径写成通信 FP8 展开 K/V，也不能推广到 packed `fp8_ds_mla`。（`gather_and_maybe_dequant_cache` 与公共 CP prefill。）

FlashAttention 调用端允许 FP16/BF16，DeepSeek 的 $d_K=128,d_R=64,d_V=128$ 在支持列表中。FA3/SM90 和 FA4 原生处理不同 QK/V head width，其他需要 padding 的分支将 V 补到 QK width，最终截回 $d_V$。本 prefill caller 不传 FA 的 CP 参数，历史分布式恢复由 vLLM 外层完成。CUDA 适配层转入 FA2 `_vllm_fa2_C.varlen_fwd`、FA3 `_vllm_fa3_C.fwd` 或 FA4 CuTeDSL 入口；构建默认 pin `vllm-project/flash-attention` commit `caaa4eb59845388a20b1f435ecaafb4bd9517ad8`，可由本地源码目录覆盖。本轮核到适配与 build pin，没有核读该外部 pin 的设备实现或实际构建。（`mla/prefill/base.py/flash_attn.py`、`fa_utils.py`、`vllm_flash_attn/flash_attn_interface.py`、`vllm_flash_attn.cmake`。）

## 10. Prefix 命中、存储和通信成本分别计费

所选单 KV group 的 hash 与完整 cache block 都为 $PD$ 个全局 token；`FullAttentionManager.find_longest_cache_hit/cache_blocks` 也采用放大后的单位。某 rank 已写满 $I$ 或 $P$ 行，不构成独立的全局完整前缀命中。`KVCacheManager.get_computed_blocks` 仍把最大命中长度限制为请求长度减 1，以保留获得 logits 的前向；若 $PD=32$、64 token 全部相同，完整块最多命中 32 token，后 32 token 重算。（`kv_cache_coordinator.py`、`single_type_kv_cache_manager.py`、`kv_cache_manager.py`。）

空间检查还计入命中但引用数为零、原处于可淘汰 free queue 的块：touch 会使其退出可分配队列。复用不新建 page，并不意味着对剩余可分配容量免费。Prefix 复用省掉前缀重算和写入；后续 query 仍需读取各分片的有效历史。（`get_num_blocks_to_allocate`；生命周期接续[vLLM 调度与缓存](vllm-inference-execution.md)）。

所选 decode 将每 rank 的历史缩短，却把参与本地 attention 的 heads 从 $h$ 增至 $Dh$；Triton split scratch 因此按 $[B,Dh,S,d_c+1]$ 计费。AG+RS 还交换 LSE 和输出，A2A 有 send/recv 缓冲；prefill 则保留联合历史 gather、重排与展开 K/V。缓存分片、单 kernel 工作量、通信次数和端到端性能需要分别解释，本页不提供速度或峰值显存结论。

## 11. 核验范围与后续接口

本页核读固定版本的配置/group、V1 block 分配/hash、worker 槽位/local lengths、普通 writer、query head gather、Triton 局部 attention、AG+RS/A2A 合并、value/输出投影，以及公共 DCP prefix prefill 的 gather/reorg/展开/本地合并。CPU 教学算术核对了 owner/local 地址与长度守恒、LSE 加权和完整 softmax 的关系；它不执行 vLLM 原实现。

没有安装依赖、编译、启动模型/服务或运行 GPU/分布式测试，没有测量误差、网络和性能。空分片 A2A 的静态数值路径就近保留，不能当作已复现的服务故障。其他 $D/I/P$ 组合、MRV2、PCP、DSA/packed cache、其他 backend、推测执行及完整图行为仍是边界。特殊缓存格式需要继续核其字节/scale、更新与读取契约；不由本页的普通联合行自动承接。

## 来源身份

| 来源 | 固定版本 | 实际使用范围 |
| --- | --- | --- |
| [vllm-project/vllm](https://github.com/vllm-project/vllm/tree/568afb3a13806beb53bb2e6bd518269357b237c0) | `568afb3a13806beb53bb2e6bd518269357b237c0` | 普通 MLA、PCP=1、Model Runner V1 的 DCP 缓存/metadata、所选 Triton decode、两种 O/LSE 通信、公共 FlashAttention prefix prefill，以及必要的调度、投影和库接口。直接使用文件登记于 sources，函数就近定位；外部 FlashAttention/NCCL 设备实现未核。 |
