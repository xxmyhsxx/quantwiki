---
title: FlashInfer 分页 KV：写入契约、布局与 NVFP4 尺度
type: implementation
tags:
  - kv-cache
  - data-layout
  - fp4
sources:
  - raw/articles/2026-09-21/flashinfer-append-paged-kv-cache/article.md.txt
  - raw/articles/2026-09-21/flashinfer-kv-cache-layout/article.md.txt
  - raw/articles/2026-09-21/flashinfer-kv-cache-layout/formulas.md.txt
  - raw/articles/2026-09-21/flashinfer-kv-cache-layout/assets/image-efc4f5e9d414.png
  - raw/articles/2026-09-21/flashinfer-kv-cache-layout/assets/image-93515341110b.png
  - raw/articles/2026-09-21/flashinfer-kv-cache-layout/assets/image-76c22bd22ebb.png
  - raw/articles/2026-09-21/flashinfer-kv-cache-layout/assets/image-de541b138748.png
  - raw/articles/2026-09-21/flashinfer-nvfp4-kv-cache/article.md.txt
updated: 2026-10-12
---

# FlashInfer 分页 KV：写入契约、布局与 NVFP4 尺度

这三份0.6.14文档快照把“如何写入已有页”“如何解释索引/张量轴”“NVFP4产物如何交给专用reader”分开。它们是接口参照，没有实现[CalibQuant](../methods/calibquant.md)或[VidKV](../methods/vidkv.md)的编码、保护和校准；本文未运行FlashInfer或核当前版本。

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

## 来源身份

三份FlashInfer 0.6.14官方文档，正文`snapshot-2026-07-02`：`https://docs.flashinfer.ai/generated/flashinfer.page.append_paged_kv_cache.html`、`https://docs.flashinfer.ai/tutorials/kv_layout.html`、`https://docs.flashinfer.ai/generated/flashinfer.quantization.nvfp4_quantize_paged_kv_cache.html`。本次读完整保存正文/例子、布局的数学重录与4张2026-09-21补采图；未取得历史HTML、匹配0.6.14源码/现行网页，未核三接口之外的完整wrapper/分段打包协议或运行FlashInfer/CUDA/SM100。
