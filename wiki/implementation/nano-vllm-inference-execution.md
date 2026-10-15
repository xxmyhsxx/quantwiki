---
title: nano-vllm 离线推理：token 进度、分页缓存与模型执行
type: implementation
tags:
  - serving
  - inference
  - kv-cache
  - memory
sources:
  - raw/repositories/2026-09-21/nano-vllm
updated: 2026-10-15
---

# nano-vllm 离线推理：token 进度、分页缓存与模型执行

nano-vllm 的固定版本把一批文本或 token-ID 提示变成生成结果：等待队列提供请求，调度器选择本轮工作，块管理器提供历史 KV 和写入空间，runner 构造输入，Qwen3 前向及采样产生候选 token，后处理再推进请求。它适合用来跟清一条紧凑执行链；这是**离线批量接口**，没有 HTTP 服务、流式响应或现代 V1 的混合调度主线。

本页依据 `bb823b3e06983d71485a8e1f23715ebd87d98ef8`，除根目录的README、pyproject、example.py与bench.py外，以下模块路径相对于 `nanovllm/`。现代引擎的不同组织见[vLLM 推理执行](vllm-inference-execution.md)；因果模型、下一 token 与 KV 的关系见[自回归推理](../fundamentals/model/transformer-autoregressive-inference.md)，分页基础见[服务内存管理](serving-memory-and-batching.md)。既有静态研读已覆盖全部21个 Python 文件；本轮补组件分工时定向复核相关入口，未导入上游或运行模型。

## 1. 离线调用、轮次执行与返回结果

这条执行链把“决定本轮做什么”与“执行张量计算”分开。请求的逻辑进度保存在 `Sequence`，scheduler 决定进度何时前推；KV 块的分配信息与 GPU 上真正的 KV 数值也由不同组件维护：

| 组件 | 管理的状态或职责 | 怎样与下一环节配合 |
| --- | --- | --- |
| `LLMEngine` | 接收输入，驱动轮次，收集完成结果 | 每轮调用 `schedule → runner.run → postprocess` |
| `Sequence` 与 `Scheduler` | Sequence 保存 token、前向进度和页表；Scheduler 管理 waiting/running、选定本轮工作并处理返回 token | 把选中序列及 prefill/decode 标识交给 runner，返回后更新进度或结束请求 |
| `BlockManager` | 管理物理块 ID、引用计数和前缀 hash，更新序列的块表 | 为 runner 的位置映射提供块号；归还块引用不等于释放 GPU 张量 |
| `ModelRunner` | 分配每个 rank 的 KV 池，准备紧凑输入、位置与页表，调用模型和采样 | 将每层 KV 视图交给 Attention；采样结果返回 scheduler 判断是否接纳 |
| 模型与 `Attention` | 计算本轮隐藏状态，按 slot 写入 K/V，再读取允许的历史 | 产出供 LM head 与采样使用的隐藏状态，不负责请求队列和结束判定 |

因此，“缓存已经分配”“本轮已算出 KV”“采样 token 已加入序列”是三个不同事件。后面的进度、分页和前向章节分别解释这些事件如何衔接。（`engine/llm_engine.py:step`、`engine/scheduler.py:schedule/postprocess`、`engine/block_manager.py`、`engine/model_runner.py:allocate_kv_cache/run`、`layers/attention.py`。）

`LLM` 直接继承 `LLMEngine`。初始化先读本地模型配置，构造 runner，完成权重加载、warmup、KV 池分配及可选 CUDA Graph 捕获，再建立 tokenizer 和 scheduler。模型类在 runner 中固定为 `Qwen3ForCausalLM`，不是由 `AutoConfig` 自动选择任意架构。（`llm.py`；`engine/llm_engine.py:__init__`、`engine/model_runner.py:__init__`。）

`generate(prompts, sampling_params)` 将这一批输入加入 waiting：字符串经 tokenizer 编码，token-ID 列表直接用于序列。循环中的 `step` 调用 `schedule → runner.run → postprocess`，收集结束序列的 completion token。全部队列清空后，按 seq_id 排序，并将每份结果转换成 `{"text": ..., "token_ids": ...}`；源码的返回注解为 `list[str]`，实际返回字典列表。（`llm_engine.py:add_request/step/generate`。）

`example.py` 使用本地 Qwen3-0.6B、单卡 eager、temperature=0.6 和 max_tokens=256；两条用户文本先经 chat template，并指定 `add_generation_prompt=True`。输入模板由调用方准备，不能由引擎接收字符串推断它会自动补齐角色提示。

## 2. 已知 token、KV 进度与本轮工作

一个序列同时有文本历史和前向进度，二者不能混用：

| 量 | 源码字段 | 含义 |
| --- | --- | --- |
| n | `num_tokens` | prompt 与已采样 completion 的已知 token 总数 |
| P | `num_prompt_tokens` | 原始 prompt 长度，抢占重算不改变它 |
| c | `num_cached_tokens` | 当前已完成并可复用的 KV/前向进度 |
| q | `num_scheduled_tokens` | 本轮要计算的 token 数 |
| 页表 | `block_table` | 逻辑块到物理 KV 块的映射 |
| 末 token | `last_token` | 普通 decode 下一轮的输入 |

新序列 c=q=0。若没有前缀命中，prefill 处理 prompt 后，c 推进到 P，采样才将 n 增为 P+1。此时最后一个已知 token 尚未形成 KV。下一轮 decode 输入这个 token，写入它的 KV，推进 c，再追加另一个采样 token，所以普通生成通常保持 **n=c+1**。使请求终止的最后一个采样 token 无需再执行前向。（`sequence.py:__init__/append_token`、`scheduler.py:postprocess`。）

Chunk 尚未覆盖完整已知历史时，后处理只推进 c，不追加候选 token；抢占则清空块表并令 c=0，把完整已知历史留给后续重算。这个版本在 runner 返回后更新 c，不应套用现代 V1 的“调度时就前推 computed”时序。

## 3. Prefill 优先、chunk 预算与抢占重算

`Scheduler.schedule` 先看 waiting，按 `max_num_batched_tokens` 计算本轮剩余 token 预算，按 `max_num_seqs` 限制本轮选择的序列数。只有本轮首个请求允许被切成 chunk；已有候选后，下一个请求若无法完整放进剩余 token 预算，就停止接纳。（`scheduler.py:schedule`。）

初次接纳先调用 `can_allocate/allocate`，为请求**整个已知序列所需的逻辑块**建立页表，再令 q 为未计算长度与剩余预算的较小值。前缀命中可共享物理块，但 chunk 小不代表只为这个 chunk 考虑 KV 容量。`max_num_seqs` 限制的是本轮 selected 数，代码未据已有 running 数将它实现为全局活跃请求上限。

只要本轮选到了 prefill，就立即返回 prefill 批次，decode 请求等待下一轮；没有选到 prefill 时，才从 running 取 decode，每请求 q=1。Decode 分支使用序列数上限，不使用 prefill token 预算。**两类工作不混在同一模型轮次**，与现代 V1 的统一 token 调度不同。

Decode 追加页所需空间不足时，先抢占 running 尾部请求；仍不够则抢占当前请求。`preempt` 释放块引用、清空 c 和页表，把请求放回 waiting 前端；恢复时可以命中尚未被覆盖的完整前缀块，否则计算完整已知历史。所读链没有 CPU swap。源码也未提供完整公平性保证；持续的 prefill 和前端 running 顺序会影响等待，不能从“连续批处理”推出每请求每轮都推进。（`scheduler.py:preempt`。）

## 4. 完整前缀块、引用与物理槽位

默认块大小 B=256，配置要求 B 为256的倍数。每个 `Block` 保存 ID、引用数、hash 和 token 列表；实际 KV 数值另在 GPU 池中。完整块的 hash 递归包含父前缀 hash 与本块 token bytes，命中后还比较本块 token 列表，因此不同前文中的相同局部片段不能只按文本复用。（`config.py`；`block_manager.py:compute_hash/can_allocate`。）

命中检查只遍历 `range(seq.num_blocks - 1)`，**最后逻辑块始终留给本请求计算，即使它已完整**。例如 n=512、B=256，最多复用前256个 token；n=513 时才可能复用前512个。不能把它概括成总是只重算一个末 token。

命中的使用中块增加引用数；命中的空闲缓存块移出 free 队列。未命中部分取得私有块。`hash_blocks` 在本轮返回后只登记刚计算完整的块，partial block 不入前缀缓存。这个版本通过共享完整前缀、保留私有尾块避免写共享尾部，没有 beam/fork 或 partial-tail 的写时复制链。（`block_manager.py:allocate/hash_blocks`。）

`deallocate` 将引用归零的块放回空闲队列，但保留其 hash 身份和 KV 字节；以后真正重新分配时，`_allocate_block` 才移除旧身份并重置元数据。因此请求结束不逐块向 CUDA 释放内存，空闲块在被覆盖前仍有复用价值。

教学跨页例：256-token prompt 完成 prefill 后 c=256，首次采样令 n=257，但页表仍只有 prompt 页。下一轮 decode 的 `may_append` 在 `n % B == 1` 时分配第二页，再写位置256的 KV。它体现了“采样”与“形成 KV”的先后，不能在输出 token 刚追加时就宣称新页已写好。

## 5. 紧凑输入、query 长度与 KV 长度

Prefill 为每个请求取 `token_ids[c:c+q]`，位置是 c 到 c+q−1。本轮 query 长度为 q，可见 KV 长度为 ℓ=c+q：

$$Q_0=K_0=0,\qquad Q_{i+1}=Q_i+q_i,\qquad K_{i+1}=K_i+\ell_i.$$

`cu_seqlens_q` 保存 Q，`cu_seqlens_k` 保存 K；它们分别界定紧凑 query 行和每请求历史，不是同一数组。只要本轮总 K 长度大于总 Q 长度，就准备分页 `block_tables`，既包括跨请求命中，也包括上轮 chunk 留下的历史。完全新的 prefill 没有历史，Attention 直接使用本轮连续 K/V；有历史时使用分页池。（`model_runner.py:prepare_prefill`。）

物理 token 槽位为

$$\operatorname{slot}(p)=\operatorname{block\_table}[\lfloor p/B\rfloor]B+(p\bmod B).$$

教学例：c=256、q=128、已知 n=512，B=256，页表 `[10,3]`。本轮输入位置为256…383，query长度128，KV长度384，Q/K累积端点分别为128/384；新写入槽位为768…895。页10保存已有前256个 token，页3保存后续部分；本轮尚未完成 prompt，不接受采样输出。例子只验证索引关系，未调用原代码。

Decode 每请求只输入 `last_token`，位置 n−1，`context_lens=n`；它描述将本轮 token 写入后允许读取的历史长度。`input_ids/positions` 为 CUDA int64，累积长度、context lengths、slot 和页表为 int32；页表按本轮最大表长补−1，温度单独为 FP32。不能由元数据 dtype 推断 KV 数值精度。（`prepare_decode/prepare_block_tables/prepare_sample`。）

## 6. Qwen3 前向、KV 写入与 Attention

每个 rank 的 KV 池预分配为 `[2,L,Nblocks,B,Hkv/TP,d]`，前两轴区分 K/V 与层；每层获得 `[Nblocks,B,Hkv_local,d]` 的视图。L 是层数，d 是每头维度，Hkv 是模型KV head数，TP是rank数，Hkv_local=Hkv/TP，元素 dtype 来自 `hf_config.dtype`。于是每个物理块的本rank字节数为

$$2LBH_{kv,\mathrm{local}}d\,\mathrm{dtype.itemsize}.$$

这是浮点缓存，没有低比特码、scale 或 zero。当前模型投影也调用浮点 `F.linear`；不能把 README 的推理优化当作量化实现。（`model_runner.py:allocate_kv_cache`、`layers/linear.py`。）

令本轮token总行数 T=Σqᵢ。Embedding 得到这些行后，QKV 合并投影再切成 Q `[T,Hq_local,d]`、K/V `[T,Hkv_local,d]`。当 `qkv_bias=False` 时，代码对 Q/K 做逐 head RMSNorm，再对 Q/K 做 split-half RoPE；V 不旋转。RoPE 将前后半维配对，以 FP32 计算后转回原 dtype。（`models/qwen3.py:Qwen3Attention`、`layers/rotary_embedding.py`。）

Attention 先用 Triton 把本轮 K/V 写到 slot，再选择读取路径。Writer 容许源 token 行有 stride，但要求 head 和每头维度连续；KV 池按 token 槽位保存所有本地 KV heads，Graph padding 的 slot=−1 跳过写入。（`layers/attention.py:store_kvcache`。）

| 模式 | 实际调用 | 读取对象 |
| --- | --- | --- |
| 无历史 prefill | `flash_attn_varlen_func`，`causal=True` | 本轮连续 K/V 与 query/KV累积长度 |
| 有前缀或 chunk 历史的 prefill | 同上，传 `block_table` | 分页池中的历史及刚写入部分 |
| Decode | Q加单query维，`flash_attn_with_kvcache` | 页表、`context_lens` 与分页KV池 |

该页核的是调用契约，未核与这些依赖匹配的 FlashAttention 库内分支或设备产物。Attention 输出展平 heads，经行并行 o_proj；残差/norm 后进入合并 gate/up、SiLU×up、行并行 down，再重复各层并最终归一化。其逻辑见[自回归模型](../fundamentals/model/transformer-autoregressive-inference.md)，gate/up关系见[门控激活](gated-activation-cuda-triton-kernels.md)。当前 RMSNorm 用 FP32 计算均方；相加路径用 FP32 残差和计算归一化，残差存储及归一化结果再转回输入 dtype，与“所有融合步骤都用同一低精度”不同。（`qwen3.py:Qwen3DecoderLayer/Qwen3Model`、`layers/layernorm.py`。）

## 7. 最后 query 行、随机采样与请求完成

Prefill 的 LM head 只选 `cu_seqlens_q[1:]-1`，每请求取本轮最后 query 行；decode 已是一请求一行。TP 时各 rank 产生词表分片 logits，rank0 gather 后拼接，只有 rank0 采样。（`layers/embed_head.py:ParallelLMHead`、`model_runner.py:run`。）

Sampler 将 logits 转 FP32、除温度、softmax 得到 p，再计算 `argmax(p / E)`，E 由独立 Exp(1) 随机数产生并夹住到至少10⁻¹⁰。理想未截断随机变量下，Eᵢ/pᵢ服从速率pᵢ的指数分布；取最小值的概率为pᵢ/Σp，是 categorical 的指数竞赛解释。实际浮点与夹住行为仍以代码为准，它不是选最大 logit 的 greedy。（`layers/sampler.py`；推导为整理者说明。）

未完成的 prefill chunk 也调用 sampler，但 `postprocess` 丢弃其候选 token，只更新 c；**被丢弃的采样仍消耗 RNG**。因此相同 seed 不保证 chunk/full prefill 的自由生成文本逐字一致；比较接口正确性应先固定输入，核 logits 和缓存状态，再控制采样随机流。

完成已知历史后才接受 token。遇 EOS且未ignore，或 completion数达到 `max_tokens` 时，请求结束并释放块引用；最终结果只包含 completion，不含 prompt。`SamplingParams` 禁止接近零温度，当前没有 top-k/top-p 或约束解码接口。（`scheduler.py:postprocess`、`sampling_params.py`。）

## 8. 初始化分片、同机 TP 与 Graph 范围

Loader 读取本地 `*.safetensors`，由 CPU 张量拷入设备参数。Q/K/V与gate/up的checkpoint名字映射到合并参数子片；列并行切输出轴，行并行切输入轴，行并行结果all-reduce，bias只由rank0加入一次；词表也按rank切分。`tie_word_embeddings` 为真时，embedding与head共享参数存储。分片原理见[张量并行](tensor-parallel-quantization.md)；此版本要求词表、query/KV heads等满足对应整除，**没有KV heads少于TP时复制的分支**。（`utils/loader.py`、`layers/linear.py/embed_head.py`、`qwen3.py:Qwen3ForCausalLM`。）

TP采用同机SPMD：主进程spawn其他rank，rank对应CUDA设备，NCCL地址为 `localhost:2333`；rank0用固定名、1 MiB shared memory保存pickle控制消息，events唤醒worker。Prefill/重算发送完整已知历史，decode只需last token、进度和页表。它不是多节点路由或CPU/GPU重叠服务链。（`llm_engine.py`、`model_runner.py:call/read_shm/write_shm`、`sequence.py:__getstate__`。）

Warmup用token ID全为0的非空序列做prefill，正常正预算配置下长度为 `min(max_num_batched_tokens,max_model_len)`；此时还没有KV池，不写缓存。随后从device已用显存及Torch peak/current估计执行峰值，扣除后按块字节计算池容量。各rank独立估算，所读代码没有跨rank取最小块数的协商，共同页表能被全部rank容纳是运行前提。（`warmup_model/allocate_kv_cache`。）

`enforce_eager=False` 才捕获CUDA Graph。图只包含**模型hidden-state前向**，LM head/gather和采样在图外；prefill、eager或decode行数>512用普通调用，其余找不小于实际batch的捕获规格。Replay前更新持久IDs、位置和页表，slot先填−1、context lengths先清零；输出只取真实行。Eager仍使用RMSNorm、RoPE、SiLU和sampler的 `torch.compile`，不能理解为全部解释执行。（`run_model/capture_cudagraph`。）

这些接口还有实际代码边界：

- `max_model_len` 用于warmup和Graph页表宽度，入口与scheduler没有据它校验prompt加生成总长度。
- 捕获规格为 `[1,2,4,8]+range(16,max_bs+1,16)`；例如自定义上限17时，实际batch17没有上取整规格，dispatch没有相应fallback。
- 若waiting首请求连完整已知序列的页表都无法分配、且没有running工作，最终到 `assert scheduled_seqs`，未见明确拒绝响应。
- 序列要求非空token列表；逐请求参数列表通过zip配对，长度不等时没有完整校验。`max_tokens` 也未验证正整数范围，不能当作完善的请求接纳层。
- Loader没有检查全部预期参数均已装载；`rope_scaling` 字典只取其中的 `rope_theta`，不能据字段名推定实现全部RoPE scaling策略。

上述判断来自静态分支与字段使用，不是本轮模型故障复现。模型权重及匹配依赖内部实现未收录/核读，不能从源码和语法检查推出任意配置可运行。

## 9. 输出吞吐的计时边界

`bench.py` 使用Python random seed=0，256条随机token-ID请求，输入/输出长度各100–1024，temperature=0.6、ignore_eos=True、Graph开启、max_model_len=4096。初始化和一次 `"Benchmark: "` 生成在计时前；`time.time()` 包住第二次完整 `generate`，包括序列构造、调度、输入准备、设备执行、采样结果 `.tolist()`、后处理及最终tokenizer decode。

分子为 `sum(sp.max_tokens)`，只计输出token，时间仍包含prompt前向。它不是纯kernel时间或在线SLO基准；进度条中的每轮prefill/decode速率也不是此项整批指标。服务计量区别见[负载、延迟与吞吐](serving-performance-evaluation.md)。

固定README报告RTX4070 Laptop 8GB、Qwen3-0.6B、133,966输出token：nano-vllm为93.41秒/1434.13 token/s，vLLM为98.37秒/1361.84 token/s。这是作者报告，未给vLLM完整版本及环境，不能当作现代vLLM排名或本轮实测。本次核对归档与展开树26文件、21 Python文件AST及槽位教学计算，没有安装、编译、运行模型、TP/Graph或benchmark。

## 来源身份

- GeeeekExplorer/nano-vllm，`bb823b3e06983d71485a8e1f23715ebd87d98ef8`，`https://github.com/GeeeekExplorer/nano-vllm/tree/bb823b3e06983d71485a8e1f23715ebd87d98ef8`；全部21 Python文件、固定README与pyproject，具体主线定位见各节；未运行上游及未核匹配依赖内部设备实现。
