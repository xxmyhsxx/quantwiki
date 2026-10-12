---
title: DSA 的设备执行：DeepGEMM logits 与 FlashMLA sparse attention
type: implementation
tags:
  - attention
  - kv-cache
  - fp8
  - kernels
sources:
  - raw/repositories/2026-09-21/vllm/source/cmake/external_projects/flashmla.cmake
  - raw/repositories/2026-09-21/vllm/source/cmake/external_projects/deepgemm.cmake
  - raw/repositories/2026-09-21/vllm/source/vllm/utils/deep_gemm.py
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/models/deepseek_v2.py
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/layers/sparse_attn_indexer.py
  - raw/repositories/2026-09-21/vllm/source/vllm/v1/attention/backends/mla/indexer.py
  - raw/repositories/2026-09-21/vllm/source/vllm/v1/attention/backends/mla/flashmla_sparse.py
  - raw/repositories/2026-09-21/vllm/source/vllm/v1/attention/backends/mla/sparse_utils.py
  - raw/repositories/2026-09-21/vllm/source/csrc/libtorch_stable/cache_kernels.cu
  - raw/repositories/2026-09-21/vllm/source/csrc/libtorch_stable/sampler.cu
  - raw/repositories/2026-09-21/vllm/source/csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh
  - raw/repositories/2026-10-07/deepgemm/source/deep_gemm/__init__.py
  - raw/repositories/2026-10-07/deepgemm/source/csrc/python_api.cpp
  - raw/repositories/2026-10-07/deepgemm/source/csrc/apis/attention.hpp
  - raw/repositories/2026-10-07/deepgemm/source/csrc/jit_kernels/impls/smxx_fp8_fp4_mqa_logits.hpp
  - raw/repositories/2026-10-07/deepgemm/source/csrc/jit_kernels/impls/smxx_fp8_fp4_paged_mqa_logits.hpp
  - raw/repositories/2026-10-07/deepgemm/source/deep_gemm/include/deep_gemm/impls/sm90_fp8_mqa_logits.cuh
  - raw/repositories/2026-10-07/deepgemm/source/deep_gemm/include/deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh
  - raw/repositories/2026-10-07/deepgemm/source/deep_gemm/include/deep_gemm/impls/smxx_clean_logits.cuh
  - raw/repositories/2026-10-07/deepgemm/source/deep_gemm/include/deep_gemm/scheduler/paged_mqa_logits.cuh
  - raw/repositories/2026-10-07/deepgemm/source/deep_gemm/include/deep_gemm/mma/sm90.cuh
  - raw/repositories/2026-10-07/flashmla/source/flash_mla/flash_mla_interface.py
  - raw/repositories/2026-10-07/flashmla/source/csrc/api/api.cpp
  - raw/repositories/2026-10-07/flashmla/source/csrc/api/sparse_fwd.h
  - raw/repositories/2026-10-07/flashmla/source/csrc/api/sparse_decode.h
  - raw/repositories/2026-10-07/flashmla/source/csrc/sm90/prefill/sparse/config.h
  - raw/repositories/2026-10-07/flashmla/source/csrc/sm90/prefill/sparse/phase1.cuh
  - raw/repositories/2026-10-07/flashmla/source/csrc/sm90/helpers.h
  - raw/repositories/2026-10-07/flashmla/source/csrc/sm90/decode/sparse_fp8/config.h
  - raw/repositories/2026-10-07/flashmla/source/csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh
  - raw/repositories/2026-10-07/flashmla/source/csrc/sm90/decode/sparse_fp8/components/dequant.h
  - raw/repositories/2026-10-07/flashmla/source/csrc/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu
  - raw/repositories/2026-10-07/flashmla/source/csrc/smxx/decode/combine/combine.cu
  - raw/repositories/2026-10-07/flashmla/source/tests/quant.py
updated: 2026-10-07
---

# DSA 的设备执行：DeepGEMM logits 与 FlashMLA sparse attention

DSA 的 indexer 和主 attention 有不同的计算职责。**DeepGEMM 对独立 indexer Q/K 计算选择分数；vLLM 限定因果范围并选 top-k；FlashMLA 在选中的主 MLA 状态上重新计算 QK、softmax 和 latent 加权和。** Indexer logits 不是主 attention 的概率，两个 cache 的 FP8 表示与 scale 也不同。

本页接续 [vLLM packed 缓存与 DSA 框架链](vllm-mla-packed-cache-and-dsa-execution.md)：字段、写入、投影和请求内到物理/workspace 地址由该页承接，这里追踪固定外部库的设备读取与计算。普通联合缓存及跨 rank 合并分别见 [MLA 执行](mla-cache-and-attention-execution.md)和 [DCP 执行](vllm-mla-context-parallel-execution.md)。

## 1. 版本、调用链与所选条件

vLLM 固定为 `568afb3a13806beb53bb2e6bd518269357b237c0`；两个 CMake 默认下载 pin 分别为 FlashMLA `a8f794d1251cbfd88a5011445dd5582289c727e4` 与 DeepGEMM `a6b593d2826719dcf4892609af7b84ee23aaf32a`。本页已取得并直接读取这两份外部源码。构建仍允许 `FLASH_MLA_SRC_DIR`、`DEEPGEMM_SRC_DIR` 覆盖，DeepGEMM 还可能复用已有构建源码；默认 pin 不能证明任意部署使用相同版本。

沿用默认 `DeepseekV32ForCausalLM → deepseek_v2.DeepseekV3ForCausalLM`、实际创建 Indexer 的普通层、CUDA SM90、BF16 模型输入、eager、单设备、无推测 decode、`sparse_mla_force_mqa=True`。主 MLA 取 $d_c=512,d_R=64,H=128$，`fp8_ds_mla` page 大小 $P=64$；indexer 取 $I=64,D=128$、FP8 Q/K、FP32 weights/scales，top-k 示例 $S=2048$。普通 decode 每请求一条 query；其他形状仅用于辨明入口约束。

| 阶段 | 实际入口与所选设备分支 | 输出 |
| --- | --- | --- |
| Indexer prefill | vLLM `fp8_fp4_mqa_logits` → DeepGEMM attention API → JIT `smxx_fp8_mqa_logits` → `sm90_fp8_mqa_logits` | FP32 选择 logits |
| Indexer decode | `get_paged_mqa_logits_metadata`、`fp8_fp4_paged_mqa_logits` → SM90 paged scheduler/kernel | FP32 请求内逻辑列 logits |
| 主 sparse prefill | `flash_mla_sparse_fwd` → `sparse_attn_prefill_interface` → `run_fwd_phase1_kernel<576,true>` | BF16 512 维 latent O；另返回 FP32 max/LSE |
| 主 sparse decode | `flash_mla_with_kvcache` → `sparse_decode_fwd` → SM90 V32/H128 persistent kernel → combine | BF16 512 维 latent O、FP32 LSE |

DeepGEMM 从 Python `_C` 导出 attention APIs，由 `csrc/python_api.cpp` 注册，实际参数检查在 `csrc/apis/attention.hpp`，JIT 包装负责 TMA descriptor、特化、编译与 launch。FlashMLA 的 Python 接口由 vLLM 构建改写 import shim，op 在 `csrc/api/api.cpp` 注册。下文是这条固定链的静态源码解释，没有执行 JIT、编译、GPU 或服务。

## 2. DeepGEMM：FP8 点积、ReLU 与 head 加权归约

设量化值为 $\widetilde q_{t,h,d}$、$\widetilde k_{j,d}$，其正尺度为 $s^Q_{t,h}$、$s^K_j$。vLLM 预先把 Q scale 与模型的两个归一化因子并入 weight：

$$w'_{t,h}=w_{t,h}s^Q_{t,h}D^{-1/2}I^{-1/2}.$$

所选设备 kernel 在有效列上的实数表达式为

$$A_{t,h,j}=\sum_{d=0}^{D-1}\widetilde q_{t,h,d}\widetilde k_{j,d},\qquad
R_{t,j}=s^K_j\sum_{h=0}^{I-1}w'_{t,h}\max(A_{t,h,j},0).$$

这与先恢复正 scale 后对每 head 的点积做 ReLU 等价，因为 $\max(sx,0)=s\max(x,0)$ 需要 $s>0$。实际浮点计算还受量化舍入和归约顺序影响，不保证与另一次 BF16/FP32 参考逐位一致。Weights 可以有负值，所以 ReLU 不保证最终 $R$ 非负；这里没有 softmax、V 读取或主 attention 输出。（vLLM `deepseek_v2.py::Indexer`；DeepGEMM 两个 `sm90_fp8_*mqa_logits.cuh` 的 dot、`transform` 与 store。）

API 将 `q_sf.has_value()` 当作 FP4 分支标志；FP8 Q 必须传 `(q_e4m3,None)`。独立 Q scale 已由 weight 承接，不能作为非空第二项再次传入。所选非 paged 输入为 Q `[M,I,D]`、K `[N,D]` 的连续 E4M3，K scales `[N]` 连续 FP32、weights `[M,I]` FP32，start/end 各 `[M]` int32；$I\in\{16,32,64\},D\in\{32,64,128\}$。vLLM wrapper 使用默认 full logits 与 FP32 output。返回的 `[M,N]` 是 padded allocation 的 view，行 stride 可以大于 $N$，不能只看 shape 假定连续。（`attention.hpp::fp8_fp4_mqa_logits/register_apis`；vLLM `utils/deep_gemm.py:499–541`。）

SM90 的 dot 使用 E4M3×E4M3、FP32 累加的 WGMMA；$D=128$ 分四个 K32 片段。每线程先对所属 head fragments 做 `ReLU(dot)*weight` 并归约，乘对应 key 的 FP32 scale，再用 warp XOR shuffle 合并线程间分片。主 attention 后续的 BF16 WGMMA 是另一种运算，不能把二者写成同一种 FP8 attention。（DeepGEMM `mma/sm90.cuh::FP8MMA/FP8MMASelector`；非 paged device `248–304`、paged device `284–343`。）

## 3. DeepGEMM paged 读取与因果输出范围

### 3.1 Cache envelope 内部按 page 分平面

Paged API 接收 Q `[B,1,I,D]`、weights `[B,I]`、uint8 cache `[B_I,P,1,D+4]`、block table `[B,max_pages]`。最后的 `D+4` 描述每页总载荷；真实存储为连续的 $PD$ 个 FP8 key bytes，后接 $P$ 个 FP32 scales。API 从同一 pointer 建两个 view：K 的页内 token stride 为 $D$，scale pointer 从 $PD$ 开始，二者各自保留完整 page stride。

设实际 page stride 为 $J$ 字节，逻辑 token $j$ 的页为 $p=\mathrm{block\_table}[b,\lfloor j/P\rfloor]$，页内位置 $r=j\bmod P$。所选 $P=64,D=128$ 的地址为

$$\begin{aligned}
\mathrm{addr}_K(b,j,d)&=\mathrm{base}+pJ+128r+d,\\
\mathrm{addr}_{s^K}(b,j)&=\mathrm{base}+pJ+8192+4r.
\end{aligned}$$

紧密页 $J=8448$，API 允许更大的整页 stride，但要求其可被 4 整除。JIT 为 key 和 scale 分别构建 TMA descriptor；device 从 block table 取得物理 page 后分别搬运它们。这与 vLLM `indexer_k_quant_and_cache_kernel` 的写入布局吻合。vLLM wrapper 中“每 token 尾部四字节”的 docstring 不能替代这个实际契约。（DeepGEMM `attention.hpp:349–362`、paged JIT `259–274`、paged device `176–230`；vLLM `cache_kernels.cu:598–608`。）

### 3.2 分段调度不会代替每行因果范围

单 query 的 `context_lens` 必须是连续 int32 `[B,1]`，不是 `[B]`；vLLM metadata builder 实际执行 `unsqueeze(-1)`，符合固定库。Metadata 对每请求 $\lceil L_b/256\rceil$ 个 key 段做 prefix sum，再分配给 SM scheduler slots，输出 `[num_sms+1,2]` 的 `(query,split)` 边界。一个 256-key 段由四组 math warpgroup 各处理 64 keys，所选物理 page 64 对应每组一页；Q/KV 使用三 stage，CTA 为 128 个 TMA 线程加 512 个 math 线程。Scheduler 切换 query 时重载 Q，各段写自己的 logits 列，无跨 CTA softmax 合并。（DeepGEMM paged JIT `smxx_paged_mqa_logits_metadata/smxx_fp8_paged_mqa_logits`、`scheduler/paged_mqa_logits.cuh`；vLLM `indexer.py:738–750`。）

非 paged prefill 取每个 Q tile 的 start 最小值与 end 最大值，扫描覆盖各行合法区间并集的 KV tiles；$I=64$ 时每 tile 两条 query、每 KV tile 256 列。设备可以写到某行 start/end 之外的 tile 列。Paged decode 同样按分段/64-key tile 写出，尾部列不保证已按真实 context length 设成 $-\infty$。

vLLM 两个 caller 都传 `clean_logits=False`，再把每行合法范围交给 top-k：令 $a_i$ 为该请求在收集 workspace 的起点，$L$ 为写入后完整长度，$M$ 为本轮 query 数，$u$ 为当前 query 的段内零基位置；prefill 为 $[a_i,a_i+(L-M)+u+1)$，decode 为 $[0,L)$。因此应读成 **logits 计算加 top-k 合同共同排除未来/其他请求列**，不能承诺整张 allocation 已经是一份可直接 softmax 的因果矩阵。（DeepGEMM 非 paged device `load_schedule/store`、paged device `346–349`；vLLM `sparse_attn_indexer.py`。）

固定库的非 paged `clean_logits=True` 会另 launch 清理 kernel，将每行区间外设成 $-\infty$；当前 paged API 强制二维 context lens，却在 clean 分支要求非二维，所以所读版本的 paged clean 选项不可用。所选 caller 的二维 lens 与 `False` 符合实际接口，未发现由 wrapper 旧注释导致的运行调用不匹配。（`attention.hpp:213–215,400–407,435–437`、`impls/smxx_clean_logits.cuh`。）

## 4. 主 attention 的共同数学与数值状态

Top-k 经 vLLM 映射后，FlashMLA 对有效候选集合 $\mathcal S_t$ 使用联合 key $x_j=[\widehat c_j,k^R_j]$：

$$z_{t,h,j}=\alpha\,q_{t,h}^{\mathsf T}x_j,\qquad
O_{t,h}=\frac{\sum_{j\in\mathcal S_t}\exp(z_{t,h,j})\,\widehat c_j}
{\sum_{j\in\mathcal S_t}\exp(z_{t,h,j})}.$$

$q,x$ 为 576 维，latent value $\widehat c$ 为前 512 维；64 维 RoPE 参与 QK，不进入 latent value 的加权输出。模型 score scale $\alpha$ 显式传入，不能由 indexer 的 $D^{-1/2}I^{-1/2}$ 替代。公式是对实现计算对象的数学整理，实际 kernel 另有 BF16/FP32 舍入。

两条 SM90 主 attention 路径都以 BF16 Q/K/P/V 与 FP32 WGMMA accumulator 运算；FP32 score、max 与 exp/sum 状态不意味着 PV 使用 FP32 概率，指数片段会转成 BF16 再参与 PV。Packed FP8 是持久格式，读取后进入 BF16 算术。（FlashMLA prefill/decode `config.h`、`phase1.cuh` 和 `splitkv_mla.cuh`。）

内部将 score 乘 $\log_2e$，用 `exp2f` 维护在线 softmax。对新块最大值 $m'$，旧分母与分子乘 $2^{m-m'}$，新块指数按相同 $m'$ 加入；这里 $m$ 采用 log2 单位。令 $m$ 为当前已处理有效候选的最大 $z\log_2e$，$\ell$ 为以该 $m$ 移位后的指数和。有效行的 prefill max logits 为 $m\ln2$，返回 LSE 会换回自然对数：

$$\mathrm{LSE}=\ln\ell+m\ln2.$$

这与[普通 MLA 分段归约](mla-cache-and-attention-execution.md#18.%20Triton%20MLA%EF%BC%9A%E4%BB%8E%20block%20table%20%E8%AF%BB%E5%8F%96%E4%B8%80%E6%AC%A1%20latent%EF%BC%8C%E5%88%86%E6%AE%B5%E5%BD%92%E4%B8%80%E5%8C%96)需要同一分母的原理相同，具体 LSE 存储单位和空行 sentinel 必须按库分别核对。本页不传 attention sink 或 extra KV。

## 5. FlashMLA BF16 sparse prefill：有效前缀与 CTA 内计算

### 5.1 接口与 invalid indices

vLLM 已把每组完整历史恢复到 BF16 `[N,1,576]` workspace，Q 为 `[M,H,576]`，indices 为 `[M,1,S]`。外部 API 只接受 $H=64$ 或 128；所选128无需补齐。SM90 kernel 要求 `h_kv=1`、$S>0$ 且 $S\bmod128=0$；S2048 满足。框架按64补 head 的规则不能据此泛化为 API 接受任意64倍数。（`flash_mla_interface.py::flash_mla_sparse_fwd`、`api/sparse_fwd.h:113–207`、`phase1.cuh::KernelTemplate::run`。）

`topk_length[t]` 表示只消费 `indices[t,:,:length]` 的**列前缀**。Kernel 按 $\lceil\mathrm{length}/64\rceil$ 安排块，同时逐列要求 `column<length`、`0<=index<N`；无效候选的异步 copy 补零，QK score 置 $-\infty$。它不是“在全部 S 列中找到 length 个有效项”。Tensor shape/dtype 被检查，但 length 内容的 $0\le\mathrm{length}\le S$ 由调用者保证。（`flash_mla_interface.py:205`、`phase1.cuh:80–81,119–130,477–510`、`sm90/helpers.h`。）

vLLM converter 统计 non-invalid count，同时 `COMPACT_TO_FRONT=False` 保持原列。所选合法单设备 prefill 中，短行 top-k 把有效位置写在前缀、尾部填 `-1`；正常长行选满 S，合法映射不产生内部空洞，因而数量与前缀长度相符。这个条件不能推广到任意异常 indices 或 DCP owner 过滤：例如 `[4,-1,7,-1]` 的有效数量是2，传 `length=2` 只读前两列，会漏掉7。负/越界索引能被 mask，不代表 count 可以替代任意稀疏列的范围。（vLLM `sampler.cu::topKPerRowJob:383–400`、`sparse_utils.py`；此例是对接口条件的静态演算。）

### 5.2 两份 head CTA，不是全局 split-KV

实际 dispatch 为 `Fwd_Sm90_Impl::run_ → run_fwd_phase1_kernel<576,true>`，每 query 的每64 heads 用一个384线程 CTA，grid 为 $M(H/64)$。H128 的两个 CTA 划分 heads，各自读取该 query 的完整候选集合；没有跨 CTA 拆候选后的 O/LSE workspace 或全局 combine。S2048 最多对应32个64-candidate块，`topk_length` 减少所需块数。（`sparse_fwd.h:41–46`、`phase1.cuh:575–638`。）

CTA 内两组 math warpgroup 协调在线 max、分母与512维输出，两份 KV shared buffer 交替搬运/计算；D576 在 RoPE 数据用完后复用其 shared 区存概率。QK 累积九个64维片段，PV 将前512维分两份256维处理。概率转 BF16，分子/分母归约为 FP32，归一化后 O 转 BF16。（`config.h::SharedMemoryPlan/TiledMMA_QK/TiledMMA_PV_*`、`phase1.cuh` 的 QK、`reduce_L/store_O`。）

API 另分配 FP32 max logits 与 LSE，有效行返回自然对数单位。无有效候选或 `length=0` 时 O 为0、max logits 为 $-\infty$、LSE 为 $+\infty$ sentinel；不能把这个接口 sentinel 写成数学空集合的 $\ln0=-\infty$。vLLM 只取 O，未用这些统计作跨 rank 合并。以上限定正常有限输入；Python 接口另保留被忽略尾部候选含 NaN 时的限制，不承诺任意 NaN 输入的 mask 行为。（`phase1.cuh:201–228,443–449`；vLLM `_bf16_flash_mla_kernel`。）

框架恢复 workspace、CTA shared buffers、返回 max/LSE 是三种对象。该 prefill 仍先恢复该组完整历史，不能因设备只用 top-k 就把总恢复流量记为 S 行；无全局 split workspace也不表示整条链没有临时内存。

## 6. FlashMLA packed decode：反量化与 persistent split

### 6.1 Sparse API 消费已经转换的物理 slot

输入为 BF16 Q `[B,1,128,576]`、uint8 cache `[B_c,64,1,656]` 与 int32 indices `[B,1,2048]`，输出 O 为 BF16 `[B,1,128,512]`，API 返回 LSE 为 FP32 `[B,128,1]`；内部 LSE 原分配 `[B,1,128]`，返回时 transpose。Python sparse 分支要求 `causal=False,is_fp8_kvcache=True`，直接把 indices 交给 `sparse_decode_fwd`，**不传 block table 或 cache lengths 给 sparse C++ 调用**。vLLM 的 dummy block table/max-length buffer 因而不承担这条 kernel 的分页或因果判断；真正 slot 与合法集合已由 indexer/top-k/converter 承接。（`flash_mla_interface.py:156–165`、`api/sparse_decode.h`；vLLM `_fp8_flash_mla_kernel`。）

实际 SM90 V32 分支要求 $S\bmod64=0$、row stride656，不支持 dynamic `topk_length` 或 extra KV；不能仅凭上层 feature declaration 推断这个 model 分支都可使用。本页 S2048、不传 length/extra KV 符合这些 guard。（`splitkv_mla.cuh::KernelTemplate::run:688–703`。）

### 6.2 同一记录的 FP32 scale，在 decode 先转 BF16

Producer 从 slot 算物理 page/页内位置，读前512字节 E4M3 latent、随后四个 FP32 scales、再读64个 BF16 RoPE。所选 V32 实现先将 scale 转 BF16，再以 BF16 乘法恢复 latent，可写成

$$\widehat c^{\rm decode}_{j,a}=
\operatorname{BF16}\!\left(
\operatorname{BF16}(\operatorname{decode}_{\rm E4M3}(\widetilde c_{j,a}))
\operatorname{BF16}(s_{j,\lfloor a/128\rfloor})\right).$$

FP8 到 BF16 的转换可精确承载这里有限 E4M3 值，额外舍入主要来自 scale 和乘积。这与 vLLM prefill gather 的 float scale 乘法后转 BF16 有不同的舍入位置：

$$\widehat c^{\rm prefill}_{j,a}=
\operatorname{BF16}\!\left(
s_{j,\lfloor a/128\rfloor}\operatorname{decode}_{\rm E4M3}(\widetilde c_{j,a})\right).$$

两条路径恢复同一量化记录，却不能由存储格式相同推断逐位结果相同。vLLM writer 写一般 FP32 `amax/448`；FlashMLA 测试量化 helper 会把 V32 scale 向上取2的幂，不能用该测试构造消去一般尺度下的差异。本轮只确认静态舍入位置，没有测量模型误差或判定服务精度问题。（FlashMLA `splitkv_mla.cuh:545–550`、`components/dequant.h:24–26`、`tests/quant.py:17–18,43–46`；vLLM `cache_kernels.cu:523–529,1209–1223`、`nvidia/quant_utils.cuh:364–372`。）

### 6.3 `-1` mask 与物理读取安全是不同条件

所选 producer 只把 `index==-1` 路由到 dummy 地址：$P=64$ 时为物理 page0、行63。它仍会读该记录，再由 `index!=-1` 的 validity 将 score mask 为 $-\infty$；因此 dummy 记录地址必须可读。无效列的 scale 和 FP8 latent payload 在加载后被清零；RoPE 虽不清零，其 QK score 仍会被覆盖为 $-\infty$。这不要求 dummy 记录预先为零或正常有限，但也不是跳过所有 cache load。

这一 SM90 V32 分支没有对其他负数或越过总 cache 的正 slot 做完整范围检查。vLLM converter 已把负逻辑位置转成 `-1`，但 block table 的物理 page 合法性仍由框架保证。外库契约是合法物理 slot 或 `-1`，不能宣称任意 int32 indices 都安全，也不能用 prefill 的 `0<=index<N` guard 替代它。（`splitkv_mla.cuh:538–539,577–580,594–597,612–614,632` 与 `splitkv_mla.cuh::scale_softmax:54–55`；vLLM `sparse_utils.py::_convert_req_index_to_global_index_kernel`。）

### 6.4 Scheduler 按候选列分段，再按分母合并

H128 的每个 scheduler part 使用两个 CTA 组成 cluster，各处理64 heads；两个 producer 各恢复32个候选，并向 peer 共享完整64-candidate tile。同一 part 的 CTA 按 metadata 循环多个请求/候选块，这里的 persistent 是 CTA 工作复用；它与 prefill 的独立 query/head CTA 不同。（`config.h:19–34`、`splitkv_mla.cuh:207–219,459–460,590–591,620–621,769–777`。）

`get_mla_metadata()` 初始返回 lazy scheduler 对象；首次 sparse 调用按 batch/query/head/page/top-k 配置建立 metadata，后续复用时核对配置。所选无动态长度的调度按 S 列处理，包括尾部 `-1`，而非按有效数量缩短。SM90 metadata 使用64-candidate块，按 SM 数、query数与64-head组数设 scheduler parts，并以固定开销加块数分配连续 request/block 区间。它安排选中列表的工作，不重新决定请求的因果长度。（Python `FlashMLASchedMeta/flash_mla_with_kvcache`、`sparse_decode.h::Decode_Sm90_Impl`、`get_decoding_sched_meta.cu`。）

同一请求未拆分时 kernel 直接写最终 BF16 O 与自然单位 LSE；拆分时写 FP32 局部归一化 O 和 log2 LSE 到 accum buffers，随后 combine 用

$$L_2=\log_2\sum_r2^{L_{2,r}},\qquad
O=\sum_r2^{L_{2,r}-L_2}O_r$$

恢复同一 softmax 分母，最终 O 转 BF16、LSE 转自然单位 $L_2\ln2$。API 分配容量为 `B+num_sm_parts` 的 split accum；combine 对仅一个 split 的请求直接返回。这与 indexer 的独立 logits 列写出不同，主 attention 必须合并分母，不能简单平均局部 O。（`sparse_decode.h` 的 accum/combine 设置、`splitkv_mla.cuh` epilogue、`combine.cu::flash_fwd_mla_combine_kernel`。）

## 7. 如何使用边界与空行结果

| 对象 | 有效范围由谁保证 | 空行或无效项的实际约定 |
| --- | --- | --- |
| DeepGEMM logits | metadata 缩小 tile 工作；vLLM top-k 精确限定每行合法列 | 不保证整行已清成 $-\infty$；不能对未定义的区间外 allocation 直接做 softmax |
| FlashMLA BF16 prefill | 有效候选前缀与逐列 index 范围共同 mask | O=0、max=$-\infty$、LSE=$+\infty$ |
| FlashMLA V32 decode | 合法物理 slot 或 `-1`；前一阶段保证因果集合 | `-1` 经 dummy 读取后 mask；最终无有效候选时 O=0、LSE=$+\infty$ |

Decode 空 split 的内部状态为 O=0、log2 LSE=$-\infty$；所有 split 无效时，最终 LSE 使用 $+\infty$ sentinel。Combine 显式处理全 $-\infty$ 的 max 和零分母；这里的 sentinel 不应作为普通有效 logsumexp 直接参与其他库的 DCP 归约。所选 vLLM 单设备路径没有导出这份 LSE 做跨 rank 通信。（`splitkv_mla.cuh` 的 sum/epilogue、`combine.cu:82–98`；vLLM `forward_mqa`。）

上述空行约定以已分配 cache、合法索引与有限有效 Q/候选为前提。DeepGEMM 空 context 不等于 logits 已清零/清 $-\infty$，paged kernel 的 padding 任务还可能读取 page0。

本页设备链限定 SM90、FP8 indexer、BF16 sparse prefill 与 V32/H128 packed decode；未展开 FP4、SM100/SM120、MODEL1、native multi-query、varlen、DCP/PCP 或其他 backend。选择误差、主 cache 舍入与候选集合变化的关系接续[Attention 误差传播](../theory/attention-quantization-error-propagation.md)，具体质量结论仍需对应模型协议。

## 来源身份

| 来源 | 固定版本 | 实际使用范围 |
| --- | --- | --- |
| vllm-project/vllm（`https://github.com/vllm-project/vllm/tree/568afb3a13806beb53bb2e6bd518269357b237c0`） | `568afb3a13806beb53bb2e6bd518269357b237c0` | 默认依赖 pin、真实 wrapper/caller、Q scale/weight、cache writer/gather、top-k 与 indices 转换 |
| vllm-project/FlashMLA（`https://github.com/vllm-project/FlashMLA/tree/a8f794d1251cbfd88a5011445dd5582289c727e4`） | `a8f794d1251cbfd88a5011445dd5582289c727e4` | Python/注册/API、SM90 sparse prefill、V32/H128 packed decode、scheduler/combine、测试量化构造 |
| deepseek-ai/DeepGEMM（`https://github.com/deepseek-ai/DeepGEMM/tree/a6b593d2826719dcf4892609af7b84ee23aaf32a`） | `a6b593d2826719dcf4892609af7b84ee23aaf32a` | Python/注册/API、JIT、SM90 FP8 非 paged/paged MQA logits、页平面地址、scheduler 与 clean 条件 |
