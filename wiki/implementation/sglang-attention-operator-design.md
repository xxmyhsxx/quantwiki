---
title: SGLang 算子设计：索引化 KV、Extend 与 Decode 归约
type: implementation
tags:
  - serving
  - kernels
  - kv-cache
  - attention
  - performance
sources:
  - raw/articles/2026-09-21/flashinfer-recursive-attention
  - raw/repositories/2026-09-21/sglang
updated: 2026-10-14
---

# SGLang 算子设计：索引化 KV、Extend 与 Decode 归约

SGLang 的模型层通过 RadixAttention 接入具体 Attention backend。Radix cache 解决哪些历史状态可以复用，backend 把请求映射转换为设备索引，kernel 才执行 QK、softmax 和 PV。前缀树、索引元数据和注意力计算是相连的三个职责，不能把 RadixAttention 这个类名理解为一种固定的设备 kernel。

本页沿[SGLang 推理执行](sglang-inference-execution.md)继续，选择所收录源码的普通 Llama/MHA/GQA 与 Triton extend/decode 路径。张量布局和在线 softmax 的共同基础分别复用[算子接口](../fundamentals/operators/tensor-layout-and-kernel-contracts.md)与[vLLM Attention 设计](vllm-attention-operator-design.md)；本页补足 SGLang 的读取接口、两阶段 extend 和 decode 中间量语义。[MLA 缓存与 Attention 执行](mla-cache-and-attention-execution.md)另解释联合潜变量和位置 key，以及展开/吸收算法的分派，不能把本页独立 K/V 的布局直接套给 MLA。

## 1. 从模型层到 Backend 的分工

`models/llama.py` 的普通 `forward_prepare_native` 执行 QKV projection、拆分 Q/K/V，再按 positions 对 Q/K 做 RoPE；`forward` 将它们连同 `ForwardBatch` 传给 `RadixAttention`，最后做输出投影。权重量化配置如何落到线性层与 Marlin，沿用[AWQ 的 SGLang 实现](awq-implementation.md#6.%20%E8%B7%A8%E5%BC%95%E6%93%8E%EF%BC%9ASGLang)。

`RadixAttention` 保存本地 head 数、head dimension、scale、layer id 等约束，整理 K/V 的形状后调用当前 backend。`AttentionBackend.forward` 对普通 DECODE 调 `forward_decode`，普通 EXTEND 和 GPU MIXED 进入 `forward_extend`；特定平台与模式另有分支。模型前向接口可以相同，但后台不必用同一个设备程序处理所有阶段。

`save_kv_cache` 也属于接口的一部分。普通 Triton 后端在需要保存时先写 KV pool，再执行 attention；共享 KV 等场景可以走其他处理。输出 dtype 正确并不足以证明集成正确，还必须确认新 K/V 写入了本轮 `out_cache_loc`，下一轮能够按请求映射读到它。

## 2. 将请求到 token 的映射展平成变长索引

先考虑 page size=1、单设备、普通全注意力。`TritonAttnBackend.init_forward_metadata` 从 `req_pool_indices` 选请求行，再生成一组类似 CSR 的数组：

- `kv_indptr` 给出各请求在展平 KV 索引数组中的起止位置。
- `kv_indices` 保存实际 KV 槽位，kernel 用它间接读取 K/V 数值。
- extend 的 `qo_indptr` 给出各请求在本轮紧凑 Q/输出中的起止。

这里的“类似 CSR”只描述变长分段的存储组织，并不表示普通因果 Attention 自动变成稀疏注意力。`triton_ops/kv_indices.py::create_flashinfer_kv_indices_triton` 按请求 row 和长度将映射写入索引数组；虽然函数名包含 FlashInfer，这段工具也被 Triton backend 使用，不能按名字误判执行后端。

复用框架页的教学例子，A 的映射为 `[8,9,20,21]`，B 为 `[8,9,30]`，两者 prefix=2，extend 分别为 2、1。在普通 extend 路径中：

```text
qo_indptr: [0,2,3]       Q/A 的切片为 [0:2]，Q/B 为 [2:3]
kv_indptr: [0,2,4]       此路径只索引缓存前缀
kv_indices: [8,9,8,9]    两请求共享数值槽位，但各有自己的索引区间
新 K/V: [A2,A3,B2]      本轮连续张量
```

普通 decode 的 `kv_indptr` 则由完整 `seq_lens` 构造，包含本轮已经写入的 token。若后续两个请求各追加一个 token，新槽位为 22、31，就会得到 `kv_indptr=[0,5,9]`，索引为 `[8,9,20,21,22,8,9,30,31]`。**同名元数据在不同路径中覆盖的 KV 范围不同**，不能拿 extend 的 prefix 长度替代 decode 的完整长度。

page size>1 或不同物理布局还要解释页内位置和 stride，本页例子不把 token slot 直接当成字节地址。`MHATokenToKVPool.set_kv_buffer` 就区分 dtype 转换、存储 dtype 和 HND 等布局；kernel 读取方式必须与写入方式一致。

## 3. Extend：前缀与新 token 分两段读，共用一个 softmax

`TritonAttnBackend.forward_extend` 在普通路径中传入本轮连续的 Q/K/V、缓存 K/V buffer、prefix 索引、query 起点和 mask。`extend_attention.py::_fwd_kernel` 对每个 query tile 分成两段循环：

1. 通过 `kv_indices` 读取已缓存 prefix。普通因果注意力中，prefix 位于所有新 query 之前，因此新 query 都可访问它。
2. 直接读取本轮 `K_Extend/V_Extend`，对扩展段施加三角因果 mask，屏蔽当前 query 之后的新 token。

设 prefix 长度 p、本轮 query 长度 q，第 j 个 query 的绝对位置为 p+j。允许访问的逻辑键位置是 $0\ldots p+j$：其中 $0\ldots p-1$ 由第一段提供，$p\ldots p+j$ 由第二段提供。p=2、q=2 时，两行分别读前缀 2 个加新 token 1 个、前缀 2 个加新 token 2 个。

两个循环之间保留同一组 `e_max`、`deno`、`acc`，以在线 softmax 更新，最后才做一次 `acc/deno`。因此这不是“前缀做一次 softmax，新段做一次 softmax，再把两个输出相加”。后一种写法会破坏两组 score 的相对权重。

拆分的意义在于数据来源不同：历史 prefix 需要间接寻址，本轮新 K/V 已是连续计算结果。刚写入缓存仍服务于后续轮次或其他使用缓存的分支；本路径读取当前连续 K/V 不等于无需维护 KV pool。源码还有统一索引、特殊 mask 等扩展，本节不将两段读取推广为所有 backend 的规定。

若使用量化 KV，缓存前缀和新 K/V 的数值表示也可能不同；所读循环分别处理 cache 的 scale 与当前输入，具体精度仍须按 dtype、量化粒度和调用参数验证。不能在不了解路径的情况下重复反量化，或假设新张量已经采用缓存格式。

## 4. Decode：把历史维拆开，再按 LSE 合并

普通 decode 每请求只有一个新 query。`decode_attention_fwd` 根据 query/KV head 比例选择普通或 grouped 主循环；分段数由 backend 的 `get_num_kv_splits` 准备，可以受序列长度、配置和确定性要求影响。按历史维切分能增加并行工作，但要多保存和归约部分结果。

`_fwd_kernel_stage1` 及 grouped 版本读取各段 KV，写出两个中间量：

$$o_s=\frac{a_s}{l_s},\qquad z_s=m_s+\log l_s.$$

其中 $m_s$ 是段内最大 score，$l_s=\sum_{j\in s}e^{score_j-m_s}$，$a_s=\sum_{j\in s}e^{score_j-m_s}v_j$；$z_s$ 就是该段的 log-sum-exp，以下简称 LSE。公式解释普通非量化且没有额外 sink 的情况。`attn_logits` 的名字不能代替数值语义：这里的相应缓冲存的是段内归一化后的 value 加权结果，并非完整 QK score 矩阵。

`_fwd_kernel_stage2` 对这些部分输出按 LSE 加权：令 $z_{\max}=\max_s z_s$，则

$$o=\frac{\sum_s e^{z_s-z_{\max}}o_s}{\sum_s e^{z_s-z_{\max}}}.$$

这是因为段 s 的未归一化总权重为 $e^{z_s}$。**教学例子：**两段局部输出分别为 2 和 8，LSE 分别为 $\log2$ 和 $\log6$，全局输出为 $(2\times2+6\times8)/(2+6)=6.5$，直接平均会得到 5。

与[vLLM 分段归约](vllm-attention-operator-design.md)中保存未归一化累计 a、最大值 m 和指数和 l 的例子相比，这里保存 o 和 z。二者表达同一个全局 softmax，但中间缓冲不能直接互换。已归一化输出若再次按未归一化累计处理，会多除或漏乘一项权重；量化路径还需核对 value scale 的施加位置，所读 stage2 在最终结果处应用 `v_scale`。

空 segment 不应带入未初始化结果，源码通过实际 segment 起止范围跳过它们。数值等价是实数运算下的解释；段边界、浮点累计顺序和类型转换改变时，不保证逐位一致。

### Attention state 的递归合并与接口兼容

FlashInfer 0.6.14 的 recursive attention 教程将局部归一化输出与 LSE 合称 attention state。沿用本节的记号，完整状态是 $(o,z)$：前述公式产生合并后的 o，合并后的自然对数 LSE 则为

$$z=z_{\max}+\log\sum_s e^{z_s-z_{\max}}.$$

因此，合并得到的 $(o,z)$ 可以继续参加下一层合并，不必回到原始 KV。不同线程块或设备可分别处理历史子集；共享 prefix 与各请求的独有 suffix 也可先分别计算，再恢复同一个 query 的完整归一化。这是在实数下重组未归一化加权和与分母，教程配图展示了同一组 key 的不同合并树；它不保证浮点结果逐位相同。（来源状态定义、二元/多元 merge 与 Applications。）

这个 z 是教程完整 state 的数学组成。本页所选 SGLang `_fwd_kernel_stage2`只写最终 O，没有导出这个合并后的 LSE；数学状态与实际返回接口要分别看。（`triton_ops/decode_attention.py:667–738,794–839`。）

按这些公式表示集合并集，需要各段对应**同一 query/head、相同 score 定义和全局可见性规则，并覆盖互不重叠的 KV 子集**。若 I、J 重叠，公共 key 的未归一化权重会被计两次，结果不再是集合并集的 attention；这是对原文 $I\cup J$ 写法补充的必要条件。共享 prefix 可复用 KV 数据与读取，却不能给不同 query 直接复用同一份 prefix 输出/state。

跨接口合并还要核 LSE 的底、shape/dtype 和 value scale 是否已经施加。上述教程公式用自然 exp/log；若收到 base-2 LSE，应按 $2^z$恢复权重，或先乘 $\ln2$换成自然底，不能直接塞入 $e^z$。具体库侧不同约定可对照[MLA 的 LSE 与合并](mla-cache-and-attention-execution.md#12.%20%E5%9C%A8%E7%BA%BF%E5%BD%92%E4%B8%80%E5%8C%96%E3%80%81split%20%E5%90%88%E5%B9%B6%E5%92%8C%20LSE%20%E7%9A%84%E5%BA%95)；教程的数学单位不能代替各实际 API 的返回契约。空段、全屏蔽行及特殊值也须按具体接口处理，不能作为普通有效状态直接参加公式。

教程报告长上下文、大 batch 的共享 prefix 路径有 30× 算子加速，但保存正文没有完整硬件、长度、batch、基线和计时协议，关联博客未读。本页只承接上述分解机制和这项报告的范围，不把数字作为一般收益或本地性能证据。

## 5. GQA 复用与分段各自解决什么问题

如果 r 个 query heads 共享一个 KV head，普通 head 映射为 $h_{KV}=\lfloor h_Q/r\rfloor$。grouped decode 内核把同一组中的若干 query heads 放在一个 tile 内，共享加载的 K/V；组较大时还需将 heads 切成多个 tile，并屏蔽尾部无效行。其 grid 同时组织请求、head 组和历史 segment。

GQA 分组主要增加 K/V 的复用，split-KV 主要增加沿历史维的并行任务。两者都会影响寄存器、临时结果、访存和归约成本，但不能互相替代。对于请求数少、上下文长的批次，增加历史段可能缓解工作不足；请求数已很大时，更多段可能主要增加临时数据与额外 kernel 开销。实际选择须按设备与真实长度分布测量，基础分析见[配置选择与自动调优](kernel-configuration-and-autotuning.md)。

此外，prefix 命中率高可能减少 extend query 数，却保留较长历史读取。这会改变有效计算形状，所以不能只用“prompt 原始长度”或“请求 batch size”描述算子性能；需要记录 prefix 长度、extend 长度或 decode 的 KV 长度，以及 head 配置。

## 6. 图执行把元数据准备也纳入契约

`ModelRunner._forward_raw` 检查模式和 runner 是否能使用图执行；普通 decode 图 runner 将本轮请求数匹配到捕获 bucket，填充稳定输入缓冲，再准备对应的 backend 元数据。没有适用图路径时，模型仍可走相应非图执行路径。

`AttentionBackend` 将元数据准备分成图外与图内职责：图外准备可以处理主机逻辑、动态形状或不可捕获的操作；图内准备限于可记录的固定形状 GPU 工作。eager 入口可以组合两者。将所有长度处理移入 kernel 并不自动实现图兼容；读取设备标量到 CPU、改变缓冲形状和替换捕获地址都可能改变执行约定。

`DecodeCudaGraphRunner` 的重放准备按 raw batch size 选择 bucket，经 `buffer_registry.fill_from` 填入实际数据，再用构造的 ForwardBatch 视图执行图外 metadata 更新。**教学例子：**实际 3 个 decode 请求可在允许 padding 时使用 4 请求的捕获规格，但第 4 行必须遵循 backend 的虚拟请求约定。`ReqToTokenPool` 专门保留 row 0 作为 padding 行；序列长度的填充值由 backend 提供，不能把另一框架的 -1 槽位规则直接搬过来。

extend 的 piecewise 分支在 `RadixAttention` 中通过注册的 custom op 暴露输出写入和图分割边界，并按 `real_num_tokens` 裁剪实际 query/K/V 和缓存写入位置。这个框架接入层与设备 kernel 同样需要验证，否则 padding 行或旧元数据可能破坏真实请求状态。

本节只核对上述接口、普通 decode runner 的资格/准备以及 extend 的 custom op 边界；没有据此证明所有 backend、breakable/prefill graph 或推测模式都已覆盖。

## 7. 能检验这些设计的用例

围绕实际依赖选择用例，比只测随机连续 Q/K/V 更能发现集成问题：

- 同一后缀配不同历史前缀、同一 token 序列配不同 extra key，检验前缀身份；请求重排后仍按正确 row 取 KV。
- 冷启动与命中前缀、完整 prefill 与 chunked prefill，检验历史/当前两段的 mask 和统一归一化。
- decode 使用不连续槽位、空尾 segment 和不同分段数，检验间接寻址与 LSE 加权。
- 重复执行带 KV 写入的调用前恢复输入状态，避免把前次写入当作本次正确性；eager/graph 和 overlap 开关需要验证相同请求语义。
- 基准同时记录真实 query/KV 长度、命中范围、dtype、后端、图模式与是否包含元数据准备，局部 kernel 时间和服务延迟分别报告。

本轮实际完成源码阅读、索引教学模型与 prefix/extend、LSE 分段合并的 CPU 数学核对；没有运行上述 SGLang 集成用例、编译 GPU kernel 或测性能。测量方法复用[算子正确性与性能测量](kernel-correctness-and-benchmarking.md)。

## 来源身份

| 来源 | 固定版本 | 使用范围 |
| --- | --- | --- |
| FlashInfer Recursive Attention（`https://docs.flashinfer.ai/tutorials/recursive_attention.html`） | 页面标识 0.6.14；snapshot-2026-07-02，获取于 2026-07-02T02:41:08+08:00；合并树配图于2026-09-21补采 | 完整保存正文与配图；状态/merge/应用，补不重叠条件与接口单位边界；未核merge API、关联博客或设备实现。 |
| sgl-project/sglang（`https://github.com/sgl-project/sglang/tree/2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc`） | `2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc` | Llama/RadixAttention、backend 接口、Triton metadata/extend/decode、MHA KV 写入和普通 decode 图准备；具体符号见各节。 |
