---
title: FlashInfer KV 接口：分页写入、单请求 attention 与 NVFP4 尺度
type: implementation
tags:
  - attention
  - kv-cache
  - data-layout
  - fp4
sources:
  - raw/articles/2026-09-21/flashinfer-single-prefill
  - raw/articles/2026-09-21/flashinfer-single-decode
  - raw/articles/2026-09-21/flashinfer-append-paged-kv-cache
  - raw/articles/2026-09-21/flashinfer-kv-cache-layout
  - raw/articles/2026-09-21/flashinfer-nvfp4-kv-cache
updated: 2026-10-14
---

# FlashInfer KV 接口：分页写入、单请求 attention 与 NVFP4 尺度

这些0.6.14固定文档分别说明分页写入、索引/张量轴、单请求attention读取与NVFP4产物的尺度契约。页缓存的存储、数值写入和reader的输入准备各有职责，衔接时须核对具体接口。它们是接口参照，没有实现[CalibQuant](../methods/calibquant.md)或[VidKV](../methods/vidkv.md)的编码、保护和校准；本文未运行FlashInfer或核当前版本。

## 1. 长度指针分别计 token、页与 mask 元素

Ragged张量去掉请求间padding，`data`为[总token,H,D]，`indptr`为[B+1]，首项0、相邻差是本请求token数。分页缓存则以`kv_indptr`的相邻差表示**页数**，`kv_indices`保存物理页编号，`kv_last_page_len`保存每请求末页有效token数。非空请求的长度为 $L_i=P(n_i-1)+m_i$，P为page_size、n_i为页数、1≤m_i≤P；整页满时m_i=P，不能写0。文档要求indptr、页索引和末页长度均为int32。

对请求i的逻辑位置p，教学地址关系为

$$j=\lfloor p/P\rfloor,\quad u=p\bmod P,\quad physical=kv\_indices[kv\_indptr[i]+j].$$

例如P=16，请求页列表[7,2]，追加前长度15、追加3个后长度18，末页有效长度2。新位置15/16/17分别写物理页7的槽15、物理页2的槽0/1；不能把逻辑页号1或p=16当作物理槽。该关系须再结合下面的张量stride；[内存管理与批处理](serving-memory-and-batching.md)解释页分配与共享策略。

## 2. append 写数值，调用方先准备页表

`append_paged_kv_cache(append_key, append_value, batch_indices, positions, paged_kv_cache, kv_indices, kv_indptr, kv_last_page_len, kv_layout='NHD')`原位写缓存，返回None。输入新K/V为[A,H,D]，每一行由[A]形状的`batch_indices`和`positions`绑定请求及其逻辑位置；输入不包含旧K/V。文档用`append_indptr[-1]`描述A，但`append_indptr`不是此函数的形参，须先由ragged新token区间生成逐行绑定。

调用前须分配足够物理页，并令三份页表数组**包含本轮追加后的状态**；接口不替调用方分配、淘汰、回收页，也不按返回值提供新页表。原例新长度45/8/25/22、P=16，对应页数3/1/2/2、末页13/8/9/6。其位置数组从0开始，展示的是写入这些完整新序列；存在旧prefix时应对应旧L起的追加位置，不能每次都从0覆盖。

正文要求空间预备与写入位置一致，但没有完整列出dtype/device、越界、重复地址、alias及CUDA stream同步的错误处理。本页不补出未登记行为；共享prefix页若要写新值，copy-on-write和并发顺序由引擎负责，追加完整性仍需后续读取核实。

## 3. NHD/HND 改的是轴与 stride

| 存储 | NHD | HND |
| --- | --- | --- |
| K/V分别保存的4D张量 | [pages,P,H,D] | [pages,H,P,D] |
| 合并的5D张量 | [pages,2,P,H,D] | [pages,2,H,P,D] |

5D中大小为2的轴表示K/V，`[:,0]`为K、`[:,1]`为V。对连续4D张量，NHD元素offset为`((physical*P+u)*H+h)*D+d`，HND为`((physical*H+h)*P+u)*D+d`；形状里的P和H恰好相等时也不能忽略轴义。改变`kv_layout`字符串不会自动转置或复制数据，真实非连续view还须遵守接口支持的stride。

布局文档称FP16下没有显著差异、低精度时HND更利于GPU实现；它没有提供通用性能定律。append默认NHD，而下述NVFP4转换默认HND，调用时须明确输入实际轴序，不能沿用两个函数的不同默认值。

## 4. mask、MLA 和多层 view 需要各自元数据

Attention的mask是另一种ragged记录。请求i有q_i个query、L_i个key时，布尔mask按[q_i,L_i]的query优先顺序展平，`mask_indptr[1:]=cumsum(q_i L_i)`，单位是布尔元素。文档使用little位序：每个独立打包段内的第j个布尔值进入字节的第j mod 8位；这是mask编码，不是KV数值的一位量化。

补采mask图的请求长度12/5/6，分别占2/1/1个字节，各请求尾部有padding；不能把布尔元素偏移直接当成packed字节偏移，也不能把全部23个布尔值无分段打成3字节就视作图中4字节格式。具体packed mask参数/分段接口仍须对应wrapper核对。

MLA缓存存latent `ckv`与位置分量`kpe`，可共用[pages,P,Dckv+Dkpe]再切片；该表示没有普通KV的head轴，也没有NHD/HND区别。它不等于把显式多head K/V直接拼接，实际读取见[MLA缓存与Attention](mla-cache-and-attention-execution.md)。

多层cascade让不同层用不同qo/kv指针**查看同一底层数组**，无需复制完整缓存。补采图的level0写`kv_page_indptr=[0,3]`，却列4个page_indices[0,1,2,3]，与通常末指针长度不符；level2还列没有本层页的分支与末页0。这些图示边界不能直接套非空请求公式，空层与组合协议须核wrapper，不能照抄成可运行元数据。配图在2026-09-21后补，正文为2026-07-02快照，二者获取时间分开保留。

## 5. NVFP4 产物包含两套尺度布局

`nvfp4_quantize_paged_kv_cache(k_cache, v_cache, kv_layout='HND', k_global_sf=None, v_global_sf=None)`接收BF16/FP16、相同轴序的独立4D K/V，目标是**SM100 trtllm-gen MHA**。它不是普通append函数的任意低比特替换，也不是这份文档所述MLA表示的转换接口。

它返回四项：`((k_fp4,v_fp4),(k_scales,v_scales),k_global_scale,v_global_scale)`。

| 产物 | 文档规定 |
| --- | --- |
| k/v_fp4 | 与输入同layout，D改为D//2，uint8承载两个FP4分量 |
| k_scales | 线性输入layout，D改为D//16，float8_e4m3fn |
| v_scales | 同样D//16与FP8 dtype，但用TRT-LLM的**4-token交织**布局 |
| 两个全局恢复因子 | 返回Python float，分别是1/k_global_sf、1/v_global_sf |

调用方可提供FP32标量张量`*_global_sf`；省略时按FLOAT8_E4M3_MAX/对应缓存amax计算。返回的是其**倒数**，不能把输入量化乘数直接当成reader恢复乘数。K/V各自有全局因子与块FP8尺度；尺度的解释、物理顺序与 reader 必须匹配。文档未给完整V交织地址公式、字节内FP4顺序、非整D/padding、amax=0或非有限值处理；`D//2`和`D//16`不是奇数/余组已受支持的证明。

教学账取D=128、两份K/V：码128字节，块尺度16字节，共144字节/页内token/head，相对BF16的512字节约3.56倍；两全局恢复因子、完整页/尾块、页表与工作缓冲另计。此接口转换分配新产物时还可能与原BF16缓存同时存在，不能用持久记录比代替峰值内存或实时加速。相同NVFP4名称也不保证能直接读取[GGUF块记录](gguf-block-quantization-formats.md)；[块尺度格式](../fundamentals/numeric-formats/fp8-and-mx-data-formats.md#5.%20%E4%BB%8E%E6%A0%87%E5%87%86%E5%88%B0%E5%AE%9E%E9%99%85%E5%AD%98%E5%82%A8%E6%A0%BC%E5%BC%8F)说明元素码、存储块与尺度共享组的区别。

## 6. 单请求 decode 的输入、输出与数值参数

`single_decode_with_kv_cache` 接收一个 query 及该请求的 K/V 张量，计算各 query head 的 attention 输出。它的签名没有页表、页分配或追加位置参数；从页缓存取值的调用方须先准备本入口声明的张量，不能仅凭 `kv_cache` 名称代替第 2 节的 writer 或分页读取协议。

令 L 为本请求提供的 KV 长度，Hq/Hkv 为 query/KV head 数，D 为文档输入的 head 宽度：

| 对象 | 文档形状或所选样例 | 含义 |
| --- | --- | --- |
| q | `[Hq,D]` | 单个 query 的各个 head，没有独立 query 长度轴 |
| k、v | NHD `[L,Hkv,D]`，或 HND `[Hkv,L,D]` | 本请求的 KV；布局选择不能把四维页缓存自动变成这份三维输入 |
| O | 来源等宽样例为 `[Hq,D]` | 交给后续网络层的结果 |
| 可选 LSE | `return_lse=True` 时为 `[Hq]` | 每 head 的归一化统计；底与 dtype 未在保存正文中声明 |

Hq 必须是 Hkv 的倍数，二者不等时使用 GQA，即多个 query heads 共享 KV heads。文档的 FP16 样例取 `L=4096,Hq=Hkv=32,D=128`，默认调用返回 `[32,128]`。但 Returns 的 False 分支另写了 `[qo_len,Hq,head_dim_vo]`，与 True 分支及该样例不一致；正文按二维样例说明使用方式，将三维描述保留为待核差异。V 输入仍写 D，因此也不能只凭返回中的 `head_dim_vo` 名称推定任意独立 value 宽度均受支持。（来源 Parameters、Returns、Examples 与 Note。）

`q_scale/k_scale/v_scale` 是 FP8 输入的可选**标量**校准参数，缺省为 1；`sm_scale` 则控制 softmax 分数，缺省为 $1/\sqrt D$。两类参数的对象不同，不能混成同一个默认因子。`pos_encoding_mode` 默认 NONE，可选 ROPE_LLAMA/ALIBI；RoPE 的 `rope_scale/rope_theta` 默认 1/1e4。`window_left=-1` 表示全长左窗口，正的 soft cap c 将相应 logits x 变为 $c\tanh(x/c)$，未提供时 cap 为 0。具体施加顺序与精度路径仍须核实现。（来源对应参数。）

`use_tensor_cores=False` 为默认值。文档称较大的 GQA 组使用 Tensor Core 可更快，但没有在本条给完整硬件/shape/测量条件，本次不将其当通用速度结论。本签名没有 causal/custom mask 参数，调用方须准备正确的可见 KV 范围并核窗口及位置编码约定；完整 dtype/宽度/stride、空 KV 行为和设备分派未从此页确认。

若返回 O 与 LSE 后还要分段合并，先核 LSE 单位、shape/dtype 和 value scale 的施加位置，再按[递归状态的条件](sglang-attention-operator-design.md#Attention%20state%20%E7%9A%84%E9%80%92%E5%BD%92%E5%90%88%E5%B9%B6%E4%B8%8E%E6%8E%A5%E5%8F%A3%E5%85%BC%E5%AE%B9)组织。这份接口文档不证明它的实际 LSE 是自然底，也没有给上述 NVFP4 分页产物的兼容读取契约。本次未核匹配实现或运行。（页面标识 0.6.14，snapshot-2026-07-02。）

## 7. 单请求 prefill：mask、尺度与 NVFP4 读取

`single_prefill_with_kv_cache` 一次接收 Q 个 query，输出每个 query/head 的 attention 结果。签名同样没有页表、追加位置或持久缓存写入参数；文档称它为 Prefill/Append attention，不能据此认定调用会执行第 2 节的原位追加。

令 Q/L 为 query/KV 长度，Dqk/Dvo 为 QK 点积宽度与 value/输出宽度：

| 对象 | 文档形状 | 使用关系 |
| --- | --- | --- |
| q | `[Q,Hq,Dqk]` | 每个 query 分别读取可见 KV |
| k | NHD `[L,Hkv,Dqk]`，HND `[Hkv,L,Dqk]` | 与 q 共享点积宽度 |
| v | NHD `[L,Hkv,Dvo]`，HND `[Hkv,L,Dvo]` | 决定输出最后一维 |
| O / 可选 LSE | `[Q,Hq,Dvo]` / `[Q,Hq]` | `return_lse=True` 才返回二者；LSE 底与 dtype 仍未说明 |

Hq 必须为 Hkv 的倍数，GQA 的 head 关系沿用第 6 节。这里的文档明确区分 Dqk/Dvo，但没有提供任意宽度组合的支持矩阵。`o_dtype` 省略时取 q 的 dtype；量化输入的格式不替代输出类型选择。（来源 Parameters、Returns、Note。）

### mask 决定每个 query 可以读取哪些位置

`causal` 默认 False。布尔 `custom_mask:[Q,L]` 中 False 表示屏蔽；只有未提供 `custom_mask` 时，文档才说 causal 生效。只给布尔 mask 时接口先打包，产生额外开销；给 `packed_custom_mask` 时，它是 1D uint8，布尔 mask 被忽略。来源指向 `packbits`，却没有在本页展开位序、尾部处理及“只给 packed mask”时与 causal 的完整优先级；不能将第 4 节的分段图直接当成本 wrapper 的全部协议。

原例取 `Q=128,L=4096,Hq=32,Hkv=4,Dqk=Dvo=128`，用 `tril(diagonal=L-Q)` 构造与 causal 调用比较的 mask。按这个例子的对角线，query i 可见 key j 的条件为 $j\le L-Q+i$：首个 query 可读到位置 3968，最后一个可读到 4095。它展示 query 与 KV 尾部对齐的因果范围，不能用普通 Q×L 主对角线替代。来源报告两次 FP16 调用在 `rtol=atol=1e-3` 下 allclose；其中 causal 调用启用了 `use_fp16_qk_reduction=True`，本次未复现，也不把这个容差推广到其他输入。

### FP8 按 head 的尺度与额外校准参数

`scale_q:[Hq]`、`scale_k/scale_v:[Hkv]` 用于 FP8 按 head 量化，省略时为 1；它们不同于 decode 的三个标量参数。另有 `k_scale/v_scale`，描述为 FP8/NVFP4 校准尺度，省略时也为 1。签名将后两者写为 Optional[float]，参数说明则允许 float 或 Tensor；与前一组尺度叠加、优先级和实际运算位置未说明，不能把名字相近的参数当成别名。

`sm_scale` 默认 $1/\sqrt{Dqk}$，控制 softmax 分数；位置编码、RoPE、左窗口和 soft cap 的可选值/缺省规则与第 6 节相同，实际施加顺序仍未核。`use_fp16_qk_reduction` 默认 False，来源称启用后可更快但有精度损失，未给通用性能/误差界。普通 backend 可选 auto/fa2/fa3；auto 按设备架构与 kernel 可用性选择，不表示每个组合都支持三种路径。

### NVFP4 reader 与分页量化产物的衔接

提供 `kv_cache_sf=(k_scales,v_scales)` 时，K/V 应为 uint8 打包 FP4，最后一维按相应逻辑宽度减半；两份块尺度均为 `torch.float8_e4m3fn`，且均要求**线性 row-major 布局**。该文档限定 NVFP4 KV 只支持 fa2。它没有展开精确尺度 shape、共享粒度、字节内码序、尾块及全部设备条件。

第 5 节的分页量化器面向 SM100 trtllm-gen MHA，产生四维页数据，V 尺度采用 4-token 交织；本节 reader 声明的是单请求三维 KV 和两份线性尺度。衔接时须按页表收集本请求数据、核对或重排 V 尺度，再核 FP4 码序和全局恢复因子如何进入 `k_scale/v_scale`；改 `kv_layout` 字符串不能完成这些转换。两份固定文档没有闭合这个转换协议，本次没有实现或验证直接互用。

返回 LSE 后的分段合并继续受第 6 节所述单位及递归状态条件约束。以上为页面标识 0.6.14、snapshot-2026-07-02 的接口阅读；未核匹配实现、完整 dtype/stride/空输入与 mask 全屏蔽行为，未执行 FlashInfer/CUDA。

## 来源身份

- FlashInfer `single_prefill_with_kv_cache`（`https://docs.flashinfer.ai/generated/flashinfer.prefill.single_prefill_with_kv_cache.html`），页面标识0.6.14；snapshot-2026-07-02，获取于2026-07-02T02:41:08+08:00；多query/输出、mask与FP8参数、NVFP4线性尺度和fa2限制。
- FlashInfer `single_decode_with_kv_cache`（`https://docs.flashinfer.ai/generated/flashinfer.decode.single_decode_with_kv_cache.html`），页面标识0.6.14；snapshot-2026-07-02，获取于2026-07-02T02:41:08+08:00；单请求输入/返回、GQA与数值参数，文档返回差异和未核实现保留。
- FlashInfer `append_paged_kv_cache`（`https://docs.flashinfer.ai/generated/flashinfer.page.append_paged_kv_cache.html`），页面标识0.6.14；snapshot-2026-07-02；写入契约与追加示例。
- FlashInfer KV-Cache Layout（`https://docs.flashinfer.ai/tutorials/kv_layout.html`），页面标识0.6.14；snapshot-2026-07-02；布局正文的两处数学表达与4张2026-09-21补采图。
- FlashInfer `nvfp4_quantize_paged_kv_cache`（`https://docs.flashinfer.ai/generated/flashinfer.quantization.nvfp4_quantize_paged_kv_cache.html`），页面标识0.6.14；snapshot-2026-07-02；SM100 trtllm-gen MHA的产物、尺度及全局恢复因子。

分页写入、布局与NVFP4转换这三份来源保留2026-10-12的既有核读范围，2026-10-14接续single decode与prefill。未取得历史HTML或匹配0.6.14源码；完整wrapper/分段打包协议及FlashInfer/CUDA运行仍未核，具体限制由正文与READING分别记录。
