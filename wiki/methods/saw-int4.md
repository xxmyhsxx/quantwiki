---
title: SAW-INT4：面向真实服务约束的 4 bit KV cache 量化
type: method
tags:
  - llm
  - kv-cache
  - rotation
  - serving
sources:
  - raw/repositories/2026-09-21/saw-int4/source/third_party/sglang-fast-rotation/python/sglang/srt/layers/attention/triton_backend.py
  - raw/repositories/2026-09-21/saw-int4/source/third_party/sglang-fast-rotation/python/sglang/srt/mem_cache/memory_pool.py
  - raw/papers/2026-09-21/saw-int4/source.eprint
  - raw/papers/2026-09-21/saw-int4/source/section3.tex
  - raw/papers/2026-09-21/saw-int4/source/section5.tex
  - raw/papers/2026-09-21/saw-int4/paper.pdf
  - raw/repositories/2026-09-21/saw-int4/source/README.md
  - raw/repositories/2026-09-21/saw-int4/source/docs/bdr_env_vars.md
  - raw/repositories/2026-09-21/saw-int4/source/SUBMODULE_VERSIONS.md
  - raw/repositories/2026-09-21/saw-int4/source/third_party/sglang-fast-rotation/python/sglang/QuantKernel/fused_hadamard_int4_kv.py
  - raw/repositories/2026-09-21/saw-int4/source/third_party/sglang-fast-rotation/python/sglang/srt/mem_cache/kv_quant_kernels.py
  - raw/repositories/2026-09-21/saw-int4/source/third_party/sglang-fast-rotation/python/sglang/srt/layers/attention/triton_ops/decode_attention.py
  - raw/repositories/2026-09-21/saw-int4/source/third_party/sglang-fast-rotation/python/sglang/srt/model_executor/model_runner.py
  - raw/repositories/2026-09-21/saw-int4/source/third_party/sglang-fast-rotation/python/sglang/srt/layers/attention/flashattention_backend.py
updated: 2026-10-08
---

# SAW-INT4：面向真实服务约束的 4 bit KV cache 量化

许多 KV cache 压缩方法在离线精度与压缩率上表现不错，接入生产推理引擎后却拿不到收益。那些引擎围绕分页内存、连续批处理与融合注意力内核构建，留给额外算子与不规则访存的空间很小。SAW-INT4 不再提出更复杂的压缩算法，而是反过来问：在这些约束下，哪些方法还站得住，最简方案能有多简单。

本页依据 SAW-INT4: System-Aware 4-Bit KV-Cache Quantization for Real-World LLM Serving（arXiv:2604.19157v1，预印本、标注 under review，下称论文；论文解释沿用既有正文，本轮定向复核对照与服务段；历史全文/附录研读范围与日期待核实）与官方仓库固定快照 `e51bfa72` 的 README、环境变量参考与子模块版本记录。论文的正文为 PDF 文本抽取，表格与图形按抽取内容核对；2026-10-08 接续 fast-rotation 子模块 `0fcc241961f9c79c27f6bad9a456bf10c8554a84`，核通所选 BF16、GQA、K-only 配置的融合 writer 与 Triton decode 设备链，FA3 prefill 只核调用与写入次序。未运行任何服务或内核。KV cache 量化的共同框架见 [KV cache 量化的对象与粒度](../theory/kv-cache-quantization-objects-and-granularity.md)。

## 1. 问题：离线指标与部署现实之间的落差

论文 §1 的出发点是长上下文与 agent 负载下 KV cache 的绝对体量。作者给出的量级是：Llama 4 Scout 权重只有 218 GB，而在其 1000 万 token 最大上下文下服务单个请求就需要约 1.8 TiB 的 KV cache，比权重本身大一个数量级。

随后作者指出核心矛盾：vLLM、SGLang、TensorRT-LLM 这类引擎依赖 PagedAttention、连续批处理与 FlashAttention 内核，而自回归解码受显存带宽限制，因此额外计算、访存与启动开销需要在实际融合路径中计量。既有压缩方法在这一层面各有冲突，论文把它们分为两类（§2）。

**分页内存布局带来的冲突：**

- token 淘汰只有在整块被清空时才真正减少物理占用，否则产生大量内部碎片；
- 混合精度方法（保留部分未量化 KV）破坏分页池要求的统一类型与统一布局；
- 逐通道缩放需要跨 block 计算共享的尺度，而在分页布局里 token 分布在不连续的块上，这要求低效的跨块访问。

论文明确点名 KIVI 与 Kitty 属于第二类：它们维护有固定容量上限的未量化残差缓冲，形成混合精度 KV cache。在分页池中承载两种精度需要碎片化布局或额外页表，都会破坏既有融合内核的假设。[KIVI](kivi.md) 的做法正是这样——分页友好的替代方案见该页与本节末尾的对照。

**注意力内核带来的冲突：**

- 反量化必须在内核内融合，因此向量量化的码本查表会把全局内存访问引入每步解码；
- PCA/SVD 一类基变换需要在线矩阵向量乘，抬高寄存器压力并破坏分块的效率；
- 逐通道量化常导致 channel-major 访存，破坏 GPU 注意力内核要求的合并访问。

作者据此选择按 token 独立处理的统一布局，以避免跨块统计并维持现有融合路径。这里的“唯一”受其页池和内核假设限制，不构成其他粒度在任何分页引擎中均不可实现的证明；支持其他布局所需的额外元数据、缓冲与内核成本应单独评估。

## 2. 方法：块对角 Hadamard 旋转

朴素逐 token INT4 的问题是通道间范围异质：少数高幅值通道把量化网格撑大，使同一 token 内其余通道的量化误差急剧上升（论文 §3）。缓解方式是正交旋转：

$$\widetilde{\mathbf K}=\mathrm{quant}(\mathbf K\mathbf H),\qquad \widetilde{\mathbf V}=\mathrm{quant}(\mathbf V\mathbf H),$$

$\mathbf H$ 正交，保持 $\ell_2$ 范数并重新分配能量。保持注意力等价还需要配对操作：令 $Q'=QH$、$K'=KH$，则 $Q'K'^T=QK^T$；若同时旋转 V，输出还要乘 $H^T$，即 $(P(VH))H^T=PV$。只有旋转本身在精确算术下等价，随后量化仍产生误差。默认只旋转 K 时也必须旋转 Q，但无需还原 V 输出。

全局稠密旋转代价高，论文改用**块对角旋转（BDR）**：把每个 head 的向量按块大小 $h$ 切分，每块独立旋转，整体变换为分块对角阵 $\mathbf H_{\mathrm{blk}}=\mathrm{diag}(\mathbf H_h,\dots,\mathbf H_h)$。块大小的取舍是：较大的 $h$ 混合更充分、更稳健；较小的 $h$ 并行度更高、更利于内核融合。论文报告后一半收益在实践中并不兑现——解码受显存带宽限制，旋转不是瓶颈，因此过小的块只牺牲质量而不带来端到端收益（第 4 节给出内核证据）。

**两个对照方法（§3）：**

- **Hessian 感知量化。** 用离线校准收集的解码期查询向量构造每层的二阶矩 $\mathbf M_\ell=\frac1{N_\ell}\sum_i \mathbf q_i\mathbf q_i^{\mathsf T}$，它给出键量化误差对分数的影响 $\mathcal L_\ell=\delta\mathbf k^{\mathsf T}\mathbf M_\ell\delta\mathbf k$；取其特征向量矩阵作为该层的学习旋转，推理时与 Hadamard 复合后同时作用于 Q 与 K。目标是让量化噪声偏离对分数敏感的方向。
- **向量量化（KMeans）。** 用码本把整个 head 向量映射到质心，$C$ 为质心数。

## 3. 精度对照

评测在 Qwen3-4B-Thinking-2507 上进行，五个任务为 GPQA-Diamond、HumanEval、LiveCodeBench v6、AIME25、MATH500，最大序列 32k、思考模式，5 次运行的均值与标准差（论文的扩展对照表）：

| 方法 | Mean | 相对 BF16 |
|---|---:|---:|
| BF16 | 75.64 | 0.00 |
| 朴素 INT4 | 0.00 | −75.64 |
| BDR-16 | 54.83 | −20.81 |
| BDR-64 | 72.29 | −3.35 |
| BDR-128 | 73.11 | −2.53 |
| BDR-128（仅旋转 K） | 73.78 | −1.86 |
| Hessian + BDR-128 | 65.52 | −10.41 |
| KMeans $C$=1 | 53.61 | −22.03 |
| KMeans $C$=16 | 70.48 | −5.16 |
| KMeans $C$=256 | 71.64 | −4.00 |
| KMeans $C$=2048 | 72.81 | −2.83 |
| KMeans + BDR-128 | 73.35 | −2.29 |

这组数字支持论文的核心主张，同时也限定了它：**朴素 INT4 在这个设定下完全不可用**（五项全部为 0），说明 4 bit KV 量化本身不是「轻微损失」；旋转把不可能变成可用；在旋转之后，更复杂的方法收益有限——$C$=2048 的向量量化与 KMeans+BDR 都没有明显超过 BDR-128，Hessian 感知反而更差（65.52），作者同时指出它仍需额外校准与离线复杂度，而收益没有兑现；原文未给出精度更差的因果解释。

**原文数字冲突。** 论文正文在描述同一项对照时把 Hessian 感知的均值写作 63.23，而其对照表中该行为 65.52。本页采用表中数值，不替原文统一；引用该结论时应以自己核对到的表格为准。

其余模型（论文的另一张对照表）：Qwen3-8B 上 BF16 为 70.84、朴素 INT4 为 0.00、BDR-64（仅 K）为 69.86、BDR-128 为 69.97；GLM-4.7（358B）上 BF16 为 77.89、**朴素 INT4 已经是 77.21**，各 BDR 配置在 77.12 到 77.95 之间。后一组说明不同模型对该类量化的鲁棒性差别很大：对 GLM-4.7 而言旋转几乎只带来边际改善，「需要复杂方法」这一判断是模型相关的。

消融还给出两条限于所测设置的观察（附录：旋转顺序与目标、质心数）：**只旋转键就足够**，同时旋转键与值在各模型与各块大小下几乎无差别，因此默认只旋转键以降低运行时开销；**块大小对敏感模型很重要**，Qwen3-4B 上取 16 的块大小损失可达 20 点以上，而 128 把差距压到 2 点以内。

## 4. 实现：把旋转融进解码路径

论文 §4.1 的实现要点是「旋转的成本主要来自数据搬运，而不是算术」：若把旋转作为独立的一遍扫描叠加在键缓存上，会增加内核启动与对 $S\times R$ 规模张量的额外全局访存，而解码本就受带宽限制。因此作者把旋转直接融进 Triton 解码内核——内核本就在流式读取查询、键、值分块，于是查询可以即时旋转并立刻进入 $qK$ 点积循环，不产生中间结果。

内核剖析（Qwen3-32B，2×H100，tp=2，batch 32 的单步解码，论文的剖析表）显示：融合旋转与纯 INT4 的总时间几乎相同（529,918 ns 对 533,146 ns），而未融合版本多出一项 9,376 ns 的「内核外旋转」开销（约占 1.73%）。预填充侧的剖析（附录）给出类似结论：融合内核使量化写入与反量化内核本身变慢 1.33 到 1.58 倍，但占整体运行时仅 0.10% 到 0.28%；向量量化的质心分配在解码侧代价显著，注意力内核比纯 INT4 慢 1.4 到 2.5 倍。

作者据此提出一条方法学判断：**微基准与端到端并不一致**。纯 INT4 在孤立的 KV 内核带宽测试中最高，融合旋转略低，但这一差距在端到端延迟中基本消失；反过来，向量量化在压缩率或离线精度上更好，其每个解码步的额外开销却会累积。因此评价 KV cache 方案应看服务栈总账，而不是内核微基准。

仓库侧的配置面（快照 `e51bfa72` 的环境变量文档）与论文一致：`--kv-cache-dtype` 取 `auto`（BF16 基线）或 `int4`；`HADAMARD=1` 表示在 INT4 写入前对 K 做块 Hadamard 旋转并在解码时对 Q 做对应修正；`ROTATE_V=1` 额外旋转 V 并对注意力输出做逆旋转；`HADAMARD_ORDER` 是块大小，必须能整除 head 维度。文档给出的模式矩阵把 BF16、INT4 无旋转、INT4+BDR（仅 K）、INT4+BDR（K+V）四种模式区分开，与论文的默认选择（仅旋转 K）一致。

固定版本的主机侧路径补足了配置背后的执行顺序（fast-rotation 子模块 `0fcc2419`）：

1. `memory_pool.py::set_kv_buffer` 在 INT4 且启用 Hadamard 时检查块大小能否整除 head 维。`SGLANG_FUSE_HADAMARD_INT4_KV` 默认开启，调用融合旋转与写入包装；关闭后则显式 reshape、除以 $\sqrt h$、调用 Hadamard，再进入普通量化写入。
2. `triton_backend.py` 的 decode 路径使用同一开关：融合时将 `fuse_q_hadamard=True` 和块大小传入量化 attention；关闭时在调用前显式旋转 Q。更改写入侧和读取侧必须保持一致。
3. 若 `ROTATE_V=1`，写入端同时旋转 V，decode 返回后再显式对输出做归一化 Hadamard 还原。这条分支不能概括为所有旋转均已融入 attention；仅旋转 K 可避免该输出处理。

以下继续沿这个版本核实际设备支路。代码给出的分派条件、格式与算术顺序属于静态证据，尚不证明指定 GPU 上的运行正确性或性能。

### 4.1 所选配置与实际调用

本节限定 CUDA、BF16 Q/K/V 与输出、TP1、普通 full attention 的非空 decode，取 $H_q=32,H_{kv}=8,D_K=D_V=128$，GQA 比率为4；显式设置 `HADAMARD=1 / HADAMARD_ORDER=128 / ROTATE_V=0 / SGLANG_FUSE_HADAMARD_INT4_KV=1`，使用 affine INT4。为解释主线，取无 SWA、spec、graph、logit cap、额外 attention sink 或温度修正。32/8 是所选接口形状，不表示已加载某个模型验证。

`model_runner.py:1592–1605` 要求 INT4 使用 Triton decode、FA3 prefill，且不支持此配置下的 MLA/spec。开关实际默认值是 Hadamard 关、order16、V 旋转关、FUSE 开；FUSE 开并不会自动启用旋转，order128 也不是默认值。（`memory_pool.py:92–102`；`triton_backend.py:31–40`。）

实际 writer 是 `MHATokenToKVPool.set_kv_buffer` → `quantized_set_kv_int4_hadamard_fused_triton`。K-only 下包装器调用一次融合 K writer，再调用一次普通 V INT4 writer；融合的是 K 的旋转、编码与写入，没有把 K/V 合为一个 kernel。D128/Hkv8 时，每个 program 处理一个 token 的8个 KV head，grid 为 `(T,1)`；K 有9个 autotune 候选，静态阅读不能确定运行时选择哪项。`work_k/work_v` 参数被忽略，不需要全局 FWHT 暂存。（`memory_pool.py:1136–1160`；`fused_hadamard_int4_kv.py:86–119,223–318`。）

### 4.2 写入：先核舍入，再核码与元数据

融合 K writer 先将 BF16 行转为 FP32，乘 $1/\sqrt{128}$，再执行7级 Sylvester FWHT。第 $s$ 级配对索引为 $i\mathbin{\mathrm{xor}}2^s$，低侧求和、高侧求差。旋转结果先转 BF16、再转回 FP32，随后才统计整条128维行的范围；没有把完整旋转结果写到独立全局张量。V writer 则从原 BF16 V 转 FP32，不做旋转。两侧都按 **token × KV head × K/V** 独立统计，Hadamard 块大小与量化统计范围是两个概念，在本例才同为128。（`fused_hadamard_int4_kv.py:67–83,150–201`；`kv_quant_kernels.py:120–158`。）

记编码前的行是 $y$，实际 affine 规则为

$$
m=\min_i y_i,\qquad M=\max_i y_i,\qquad
s=\frac{\max(M-m,10^{-8})}{15},\qquad z=-m/s,
$$

$$
q_i=\operatorname{uint8}(y_i/s+z+0.5),\qquad
b_j=q_j\mathbin{\mathrm{OR}}(q_{j+64}\ll4),\quad 0\le j<64.
$$

这里按源码的加0.5后类型转换描述量化，不替换成 ties-to-even 的 `round`；没有显式对 $z$ 取整，也没有额外裁剪或码的显式 clamp。$z$ 是 FP32 **浮点坐标偏移**，并非 packed 整数零点。每个 byte 的低、高 nibble 分别对应通道 $j,j+64$，不是相邻通道或两个 token。均匀网格与零点的共同解释见 [均匀量化与分组](../fundamentals/quantization/uniform-quantization-and-groups.md)。

一个可精确还原的普通 V 教学例：128维行的四段，每段32通道，依次为 $[-15,-5,5,15]$。此时 $s=2,z=7.5$，码为 $[0,5,10,15]$；前32个 byte 是 `0xA0`，后32个是 `0xF5`，读取 $(q-z)s$ 恢复原值。它同时验证通道两半的 packing 与浮点 $z$，不代表上游 kernel 已运行。

融合与分开旋转也不能仅凭理想 Hadamard 矩阵判定逐位相同。所选融合 K 先在 FP32 中预缩放；邻接 FUSE0 支路先执行 BF16 的 `cache_k / sqrt(order)`，再交外部 Hadamard。预缩放的 BF16 舍入位置不同，可能改变范围、码和元数据；本节没有执行外部 CUDA Hadamard 或证明两支路具体 GPU 字节相同。（`memory_pool.py:1161–1189`；融合文件开头的数值差异说明。）

### 4.3 持久记录、地址与完整预算

每层分配四个独立张量：K/V 的 uint8 码各为 `[C,8,64]`，K/V 的 FP32 元数据各为 `[C,8,2]`，其中 $C=\mathrm{size}+\mathrm{page\_size}$，末轴依次为 $s,z$。writer 通过 `loc[token]` 写物理 slot；decode 用请求的 `kv_indptr/kv_indices` 找历史 slot，不能把逻辑 token 序号当成自然连续的物理地址。raw getter 返回码与元数据，不生成整段 dense K/V。（`memory_pool.py:840–884,1035–1119`；`triton_backend.py:263–278`。）

在这里的连续张量布局中，物理 slot $t$、head $h$、半区内通道 $j$ 的 byte 偏移为 $512t+64h+j$；元数据的 FP32 标量偏移为 $16t+2h+[0,1]$，换成 byte 地址需乘4。实际 kernel 接收 token/head stride，不将这些教学偏移硬编码为任意输入的规则。

| 每 token、KV head 的 K+V | 码 | 元数据 | 合计 |
| --- | ---: | ---: | ---: |
| 所选 INT4 | $2\times64=128$ byte | $2\times2\times4=16$ byte | 144 byte |
| BF16 | $2\times128\times2=512$ byte | — | 512 byte |

因此四个持久缓存张量的有效记录是 **4.5 bit/元素、约3.56倍压缩**，即 $144\times8/(2\times128)$ 与 $512/144=32/9$；4 bit、4倍只指裸码。$L$ 层的预分配载荷为 $L\times C\times8\times144$ byte，含额外页槽，但仍不含请求映射、allocator、attention scratch、模型与运行峰值。本池没有 OSCAR 的 HP/recent 保护窗口或量化 flush 计数，当前写点直接编码 K/V。

### 4.4 GQA 读取：恢复精度与两段归约

实际 decode 链为 `triton_backend.forward_decode` → `decode_attention_fwd_quantized` → `decode_attention_fwd_grouped_quant` → `_decode_grouped_att_m_fwd_quant_int4` → `_fwd_grouped_kernel_stage1_quant_int4` → `_fwd_kernel_stage2`。32/8 进入 GQA grouped 分支；D128 的位置专用分支 `BLOCK_DPE=0`，本节没有拼入 MLA 或 MHA 的设备实现。

stage1 的 grid 为 `(B,8,Kmax)`，`BLOCK_H=16/BLOCK_N=32`。每个 group program 只处理4个有效 Q head，共用一个 KV head；例如 Q heads12–15共用 KV head3，其余 head lanes 由 mask 排除。请求段由 `indptr` 给出，token tile 经 `indices` gather 物理 slot；末 tile 的无效 token 被 mask，score 设为负无穷。Q 在**每个非空 split 内**从 BF16 转 FP32、预乘 $1/\sqrt{128}$、作7级 FWHT，再回 BF16；融合省去独立 Q 旋转 launch，但没有跨 split 共用一次旋转结果。（`decode_attention.py:921–1045,1098–1112,1803–1911`。）

K/V 按低、高 nibble 恢复自然两半：

$$
\widehat y_{\rm low}=\operatorname{BF16}((\operatorname{FP32}(b\mathbin{\mathrm{AND}}15)-z_{\rm FP32})s_{\rm FP32}),
\qquad
\widehat y_{\rm high}=\operatorname{BF16}((\operatorname{FP32}((b\gg4)\mathbin{\mathrm{AND}}15)-z_{\rm FP32})s_{\rm FP32}).
$$

即 **FP32 元数据在 FP32 中参与恢复，结果再转 BF16**，不同于 [OSCAR 所选 GQA reader](../implementation/oscar-mixed-kv-cache-execution.md#6-decode-分层读取与尺度精度)先将 $s,z$ 转 BF16。V 元数据在该 kernel 中复用 K 元数据的 stride；本池的两者 shape/连续布局相同，不能推广为任意不同 stride 也受支持。（`decode_attention.py:1047–1070,1115–1133`。）

QK 由两个 BF16 半区 dot 的 FP32 结果相加，再乘 caller 的 attention scaling；这个缩放与 Q/K FWHT 各自的归一化系数分开。online softmax 的 max、分母与 PV accumulator 为 FP32，但概率 $p$ **先转 BF16 才进入 PV dot**，分母仍累加未转 BF16 的 $p$。stage1 将局部归一化 Value 输出 $O_s$ 与自然对数 LSE $\lambda_s$ 写 FP32 scratch；名为 `attn_logits` 的张量承载 $O_s$，不是完整 score 矩阵。（`decode_attention.py:1070–1190`。）

scratch 由 backend 分配为 FP32 `[B,32,Kmax,128]` 与 `[B,32,Kmax]`，有效 split 数 $k$ 由调度给出，容量 `Kmax` 不等于实际非空段数。取容量8时，两者合计每请求129 KiB，独立于持久 INT4 记录预算。stage1 与 stage2 都按真实历史长度 $N$ 使用同一几何：

$$
a=32\left\lceil\frac{\lceil N/k\rceil}{32}\right\rceil,\qquad
\mathrm{start}_u=ua,\qquad \mathrm{end}_u=\min(ua+a,N).
$$

两阶段仅在 $\mathrm{end}_u>\mathrm{start}_u$ 时分别写、读 scratch，因此空尾 split 不需要预写 LSE sentinel；本版本用 `torch.empty`，没有先填负无穷。若教学例 $N=100,k=3,Kmax=8$，则 $a=64$，只有 `[0,64)`、`[64,100)` 两个非空段，第三段已为空。$k=3$ 是假设的调度输入，未声称运行时 planner 恰好选择它。（`triton_backend.py:304–315`；`decode_attention.py:958–998,1950–1962`。）

stage2 在 FP32 中作稳定 LSE 加权合并，其理想关系为

$$
\Lambda=\operatorname{logsumexp}_s\lambda_s,\qquad
O=\sum_s e^{\lambda_s-\Lambda}O_s.
$$

它不直接平均或相加局部输出；共同推导见 [分段 attention 的合并](../implementation/sglang-attention-operator-design.md#4-decode把历史维拆开再按-lse-合并)。wrapper 用输出 `o.shape[-1]` 给 stage2 推出完整128维，未把 packed V 的64字节末维当成输出维度，也未为此恢复完整历史 V。最终写 BF16 输出；K-only 的 V 保持原坐标，caller 不执行输出逆旋转。attention 核心仍为 stage1/stage2 两次调用，writer 与其他模型算子另算。（`decode_attention.py:1915–2000,2272–2289`；`triton_backend.py:1082–1091`。）

### 4.5 当前 token 与 FA3 prefill 的边界

decode 在 attention 前先调用 `set_kv_buffer` 写当前 K/V，随后从完整 INT4 历史读取，所以当前 token 也已经过量化。FA3 prefill host 同样先写当前 chunk，再取完整缓存交 `flatten_kv_cache_sglang`；FUSE1/K-only 向 helper 传 `fuse_k_hadamard=True / fuse_v_hadamard=False`，FA3 消费原 Q 与 flatten 后的 K/V。不能把 OSCAR 的 dense 当前 chunk 旁路结论套到这里。（`triton_backend.py:1017–1081`；`flashattention_backend.py:769–780,892–951`。）

本轮只核 FA3 的 gate、次序和调用参数，尚未逐行核 flatten device 的 INT4 恢复、页索引、K 逆 Hadamard 与 BF16 输出，也未核 FA3 库内 kernel。这是同固定版本、同配置的下一处必要断点，不能把所选 decode 已核扩写成完整 prefill 或全部后端已核。

## 5. 服务级证据

论文在单张 H100 80GB、Qwen3-8B、SGLang（FA3 预填充与 Triton 解码后端）上比较 BF16、INT4 与 INT4+BDR，分长上下文（平均输入 16384、输出 1024）与短上下文（平均输入 256、输出 1024）两种负载（§4.2 与附录）。

**长上下文下两个吞吐指标会背离。** 逐请求输出速度（TPS$_{\mathrm{req}}$）在高并发时显示 BF16 更快，但这来自显存容量的副作用：论文称 BF16 的 KV 条目占 INT4 的4倍、同等显存下 token 槽位为0.25倍，但该句未明确元数据口径。按本轮所选源码核算，计入 FP32 元数据后，四个缓存张量的完整记录比为第4.3节的约3.56倍，4倍只对应裸码；整进程容量还需另计。作者指出更少槽位迫使引擎以更小的解码批量运行，排队请求不断堆积——BF16 在并发 64 时首 token 延迟超过 44 秒、并发 128 时达 113 秒，而 INT4 为 19 秒与 57 秒。系统吞吐（TPS$_{\mathrm{sys}}$，总输出 token 除以墙钟时间）才是可比指标：INT4 与 INT4+BDR 在**所有**并发级别都高于 BF16，分别为并发 8 时 +10.7%、16 时 +18.7%、32 时 +8.4%、64 时 +41.4%、128 时 +25.4%。BDR 与纯 INT4 的系统吞吐差距在各并发级别不超过 1%。

**短上下文下**显存不再是瓶颈，三种配置表现接近：低并发时 BF16 略快约 3%，并发不低于 64 时 INT4 与 BDR 的系统吞吐优势逐渐显现（并发 128 时分别 +13.5% 与 +11.2%），首 token 延迟相当。

论文还报告在四种模型-硬件组合（Qwen3-4B、Qwen3-8B、Qwen3-32B 在 2×H100，GLM-4.7 358B 在 8×H100）上，INT4+BDR 在多数运行区间匹配或略优于纯 INT4 并优于 BF16；与更复杂方法的完整对照在附录。§2 的跨方法吞吐图还比较了 Kitty 等方案，但图注明确非 SGLang BF16 项使用 Hugging Face `model.generate`，缺少连续批处理与 PagedAttention。该图同时改变算法和服务后端，不能把全部差距归因于 Kitty 的量化复杂度，也不能据此给出同后端的方法速度排名。

## 6. Strong / Weak

**Strong：**把「能否部署」作为第一约束来筛选方法，而不是先比离线精度；给出的方案（token 级 INT4 加块对角 Hadamard 旋转，且只旋转键）足够简单，能用环境变量级的配置表达；把旋转融进解码内核并给出内核级与端到端两套证据，说明代价去哪了；用系统吞吐与逐请求吞吐的背离，指出一个容易误读的评价陷阱。

**Weak：**精度结论依赖「思考模式、32k 序列、5 个推理与代码任务」这一设定，朴素 INT4 的完全崩溃说明该设定对低比特极不友好，换到短回答或检索类任务上各方法的排序未必相同；GLM-4.7 上连朴素 INT4 都接近 BF16，说明结论高度模型相关；Hessian 感知与向量量化的实现对作者版本敏感，本页不把它当作这些方法的一般评价；论文为预印本，未经过同行评审；评测与内核由作者在 SGLang 分支上实现，与上游版本的对应关系未核对。

## 7. 与相邻方法的关系

与 [KIVI](kivi.md) 的对照最有信息量：KIVI 用逐通道键量化、逐 token 值量化加未量化残差窗口换取精度，而论文把「混合精度残差缓冲」列为与分页布局冲突的做法，其替代是旋转加统一的逐 token INT4 布局。这不是说 KIVI 无益，而是说两者的设计目标不同——KIVI 追求 2 bit 级别的压缩，SAW-INT4 在 4 bit 上追求零服务开销。阅读两者时应把「压缩到多少位」与「在什么布局下、付出多少系统代价」分开比较。

与 [QuaRot](quarot.md)、[正交旋转与量化](../theory/orthogonal-rotation-and-hadamard-quantization.md) 的关系是同一机制的延续与收缩：旋转抑制离群值的想法相同，但这里把旋转限制在块对角、只作用于键、并融进解码内核；论文也引用 SpinQuant 说明学习旋转的路线，其自身的 Hessian 感知对照显示在该设定下学习旋转的收益不如简单 Hadamard。

与 [QServe](qserve.md) 都强调「低比特方案必须与系统一起设计」，区别在对象与粒度：QServe 的 KV4 是逐 head 动态量化配合分页页内元数据，本方法强调统一 INT4 布局与旋转的无额外算子化。

## 8. 未验证

- 未运行论文的实现或任何服务，未复现精度、吞吐与剖析数据；模型/性能数值来自论文，第4节字节预算、地址与教学算例来自静态源码及独立计算。
- fast-rotation 子模块已核所选 BF16/GQA4/D128/Hadamard128/K-only/FUSE1 的 writer 与 decode 设备链；FA3 flatten/device、kmeans/Hessian、其他 dtype/order/后端、多设备、graph 与特殊 attention 分支未全面研读，未构建或运行。资料已展开、静态链已核与运行验证分别说明。
- 论文为预印本，其精度与吞吐结论尚待同行评审与独立复现；本页按预印本状态记录。

## 来源身份

下表用于在没有本地资料库时辨识来源；具体论述的章节、公式、图表或代码位置见正文。

| 来源 | 版本或快照 | 说明 |
| --- | --- | --- |
| [SAW-INT4: System-Aware 4-Bit KV-Cache Quantization for Real-World LLM Serving](https://arxiv.org/abs/2604.19157v1) | `arXiv:2604.19157v1` | — |
| [togethercomputer/saw-int4](https://github.com/togethercomputer/saw-int4/tree/e51bfa7291d52cd14b86e4c6ded6c002d0444ff0) | `e51bfa7291d52cd14b86e4c6ded6c002d0444ff0` | — |
| [jindajia/sglang-fork](https://github.com/jindajia/sglang-fork/tree/0fcc241961f9c79c27f6bad9a456bf10c8554a84) | `0fcc241961f9c79c27f6bad9a456bf10c8554a84` | fast-rotation 子模块；第4节所选 writer/GQA decode 设备与必要 FA3 host 边界 |
