---
title: transformers Qwen3：Q/K 归一化、动态缓存与专家路由
type: implementation
tags:
  - llm
  - inference
  - kv-cache
  - moe
sources:
  - raw/repositories/2026-09-21/transformers
updated: 2026-10-14
---

# transformers Qwen3：Q/K 归一化、动态缓存与专家路由

固定版本的 `Qwen3ForCausalLM` 与 `Qwen3MoeForCausalLM` 接收 token ID 或 embedding，将当前输入逐层变成隐藏状态，再由 LM head 输出词表 logits。两者共用 Q/K 逐 head 归一化、RoPE 和缓存更新关系；MoE 将 FFN 换成按 token 选择的专家。`generate` 在模型前向之外选择下一 token、更新输入并判断结束。

本页核对 Summer-Summer/transformers 的 `hf-4.53.2` 快照；下文路径相对于 `src/transformers/`。逻辑计算见 [自回归基础](../fundamentals/model/transformer-autoregressive-inference.md)，容量与单 token 活跃参数见 [MoE 基础](../fundamentals/model/mixture-of-experts-and-routing.md)。本次没有加载权重或运行模型，配置类默认值不能代替已发布模型的实际配置。

## 1. 当前输入、位置和缓存长度

设 batch 为 $B$，本轮输入长度为 $q$，隐藏宽度为 $D$，历史缓存长度为 $c$。`input_ids` 是 $[B,q]$，embedding 或直接传入的 `inputs_embeds` 是 $[B,q,D]$；两者必须恰好提供一个。模型直接调用时，若 `use_cache=True` 且没有缓存对象，就建立 `DynamicCache`。缺省 `cache_position` 为 $c,\ldots,c+q-1$，缺省 `position_ids` 是其加 batch 轴的视图。（`models/qwen3/modeling_qwen3.py:Qwen3Model.forward`；`models/qwen3_moe/modeling_qwen3_moe.py:Qwen3MoeModel.forward`。）

普通生成先经 `GenerationMixin.prepare_inputs_for_generation` 准备输入：已有缓存时，按 `cache_position` 只取尚未处理的 token；提供二维 attention mask 且没有位置 ID 时，用非 padding 位的累积计数生成 `position_ids`，再裁到本轮长度。因此，带 padding 的生成位置不能直接套用“不看 mask 的直接前向默认位置”。`cache_position` 描述缓存进度，`position_ids` 决定 RoPE 角度，二者职责不同。（`generation/utils.py:prepare_inputs_for_generation/_cache_dependant_input_preparation`。）

模型用这些输入准备因果 mask，并在进入各层前计算一次 cos/sin，供所有 decoder 层使用。稠密模型按 `layer_types` 选择 full 或 sliding mask；MoE 由 `config.sliding_window` 为各层选择同一种 mask。两份配置均默认关闭 sliding window，不应由存在这一接口推定发布模型已使用窗口 attention。

## 2. Q/K 在每个 head 的最后一维上归一化

记 query head 数为 $H_q$，KV head 数为 $H_{kv}$，每头维度为 $d$。投影后的布局依次为：

| 对象 | 进入 attention 前的布局 | 操作 |
| --- | --- | --- |
| Q | $[B,H_q,q,d]$ | 先在 $[B,q,H_q,d]$ 上做 Q RMSNorm，再换轴、RoPE |
| K | $[B,H_{kv},q,d]$ | 同样做 K RMSNorm，再换轴、RoPE、写缓存 |
| V | $[B,H_{kv},q,d]$ | 换轴、写缓存；不做 Q/K norm 或 RoPE |

Q norm 与 K norm 各有一个长度 $d$ 的可学习权重，分别在各自所有 heads 上共享；归一化统计在最后一维内计算，不将多个 head 拼成宽度 $H_qd$ 后统一归一化。Python RMSNorm 主体将统计转 FP32，归一化值转回输入 dtype 后再乘权重；RoPE 的角度和 cos/sin 也先用 FP32 计算，再转回隐藏状态 dtype。这里核的是 Python 主体，没有验证可选 hub kernel 替换。（`Qwen3Attention/Qwen3MoeAttention`；各自 `RMSNorm/RotaryEmbedding`。）

这两个模型类**始终构造并调用 Q/K norm**，`attention_bias` 只控制线性投影 bias。它与 [nano-vllm 的固定 Qwen3 路径](nano-vllm-inference-execution.md#6.%20Qwen3%20%E5%89%8D%E5%90%91%E3%80%81KV%20%E5%86%99%E5%85%A5%E4%B8%8E%20Attention)中 `qkv_bias=False` 才归一化的条件不同。做校准或跨实现对照时，需要跟踪实际模块，而不能只看同一个模型名称。

$d$ 由配置决定：稠密类使用 `head_dim`；MoE 使用该属性存在时的值，否则取 $D/H_q$。投影输出宽度为 $H_qd$ 与 $H_{kv}d$，不能先假定 $H_qd=D$ 再反推实际发布配置。

## 3. 先更新 KV，再读取完整历史

每层得到本轮 K/V 后，若传入缓存对象，就调用 `past_key_value.update`，然后将返回的完整 K/V 交给 attention。默认 `DynamicCache` 每层保存一对 $[B,H_{kv},c+q,d]$ 张量，沿序列轴拼接新增部分。K 已做 RoPE，V 未旋转；缓存保存 KV heads，不为每个 query head 单独存一份。（两类 `Attention.forward`；`cache_utils.py:DynamicCache.update`。）

Eager 路径随后把 K/V 逻辑展开为 $H_q$ heads，计算缩放点积、加 mask，softmax 在 FP32 计算后转回 Q dtype，再与 V 相乘。SDPA 或 FlashAttention 由 `_attn_implementation` 选择接口；本次只核所选适配层，没有验证库内设备分支。请求 `output_attentions=True` 时，这两种适配层警告并返回 `None` 权重，**不会自动退回 eager**；需要 attention 矩阵时应显式选择 eager。（`integrations/sdpa_attention.py`、`integrations/flash_attention.py`。）

`use_cache` 与“已传入一个缓存对象”也须分开。Attention 的更新条件只检查对象非空：直接传入对象同时设 `use_cache=False`，仍会修改该对象。稠密模型此时不在结果中返回缓存，MoE 仍返回对象。禁用自动创建或结果输出，不保证现有对象不发生更新。

稠密类声明支持 static cache；MoE 类声明不支持，通用生成遇到 `cache_implementation='static'` 会报错。MoE 参数注解虽然仍写列表，实际主线调用的是 Cache 的方法，不能据注解推定旧 tuple 缓存可用。上述是固定分支判断，没有复现设备或编译故障。

## 4. 专家子批次、输出权重与参数容量

Attention 残差相加后，稠密 FFN 对每行执行 gate/up、激活相乘、down。MoE 则先把 $[B,q,D]$ 展平为 $[T,D]$，$T=Bq$，由无 bias 的 router 得到 $[T,E]$ logits；$E$ 为专家数。路由过程为：

$$p_{t,e}=\operatorname{softmax}_{e}(r_{t,e}),\qquad
\mathcal J_t=\operatorname{TopK}_{e}(p_{t,e}),$$
$$y_t=\sum_{e\in\mathcal J_t}w_{t,e}f_e(x_t).$$

源码的全专家 softmax 使用 FP32；选择 $k$ 个专家后，`norm_topk_prob=False` 时 $w=p$，为真时除以选中概率之和。配置类默认是假，不能改写为“所有 top-k gate 总和都为 1”；实际 checkpoint 是否覆盖该值，本次未核。（`Qwen3MoeSparseMoeBlock`、`configuration_qwen3_moe.py`。）

实现用 one-hot 构造“专家、top-k 槽位、token 行”的对应，从每个被命中专家取出其输入子批次。专家输出乘该 token 对应的路由权重，再由 `index_add_` 加回原 token 行，最后恢复 $[B,q,D]$。因此一行可能收到多个专家贡献；专家访问顺序不能改变输出的 token 顺序。这条路径没有 shared expert、EP 通信或容量 dropping 分支。`mlp_only_layers` 和 `decoder_sparse_step` 还能使指定层使用稠密 FFN，不能由类名推定每层均为 MoE。

每个专家是无 bias 的三矩阵 MLP。设专家中间宽度为 $F_e$，则按当前模块形状，单个 MoE 子层的参数记账为：

| 口径 | 参数项 |
| --- | --- |
| 全部专家与 router | $3DEF_e+DE$ |
| 单 token 参与的专家与 router | $3DkF_e+DE$ |

这是整理者按模块形状推导，未含 attention、norm、embedding/head、其他层或运行暂存。该实例以 ModuleList 构造全部专家，选择 $k$ 不意味着只需保存 $k$ 份参数；批次涉及的是各 token 选集并集。内存和计算的共同口径见 [MoE 参数预算](../fundamentals/model/mixture-of-experts-and-routing.md#4.%20%E6%80%BB%E5%AE%B9%E9%87%8F%E3%80%81%E5%8D%95%20token%20%E8%AE%A1%E7%AE%97%E4%B8%8E%E6%89%B9%E6%AC%A1%E6%88%90%E6%9C%AC%E4%B8%BA%E4%BD%95%E4%B8%8D%E5%90%8C)。

## 5. 词表输出、语言损失与下一 token

最终 RMSNorm 输出 $[B,q,D]$。LM head 将所选位置投影到词表；`logits_to_keep=0` 的切片保留全部位置，正整数只保留最后若干位置，张量则指定索引。普通 `generate` 会在未显式提供该项时设为 1，减少词表投影位置数，decoder 仍处理本轮所有输入。（两类 `ForCausalLM.forward`；`generation/utils.py:generate`。）

传入 labels 才计算语言损失。默认 `ForCausalLMLoss` 将 logits 转 FP32，使位置 $t$ 对应下一位置的标签，`-100` 忽略；默认按有效标签取均值，提供 `num_items_in_batch` 时改为求和再除以该项。（`modeling_utils.py:loss_function`、`loss/loss_utils.py`。）

MoE 另在 `output_router_logits=True` 时计算平衡辅助项；同时有 labels 才将 `router_aux_loss_coef × aux_loss` 加入任务损失。其统计把返回的各层 router logits 拼接，再计算选择频率与平均概率，有二维 mask 时排除 padding。默认系数 0.001、默认不输出 router logits，这只说明该前向的触发条件，不能证明作者训练使用了这一默认配方。

该辅助统计假设收集到的各层项都是 router 张量：混入稠密层时，decoder 返回的 `None` 仍被加入 tuple，汇总函数未过滤便读取设备、调用 `.to`。因此不能只凭开关认定混合稠密层配置也可用；这是静态分支判断，未运行复现。（`Qwen3MoeForCausalLM/load_balancing_loss_func`；稠密层项见 `Qwen3MoeDecoderLayer/Qwen3MoeModel`。）

普通 greedy/sample 生成循环先执行前向、接续 cache 与 mask，再取最后位置 logits 转 FP32，经 logits processor 后 argmax 或 multinomial 选择 token，追加到 ID 历史。下一轮才输入这个新 token 并形成其 KV；终止的最后一个采样 token 无需再前向。调用 `streamer.put` 发生在选择 token 后，它是通用文本生成接口，不能据此推定多模态语音返回的时序。（`generation/utils.py:_sample/_update_model_kwargs_for_generation`。）

## 6. 模型配置与聊天模板的来源

`Qwen3Config()` 默认 32 层、Q/KV heads 32/32；`Qwen3MoeConfig()` 默认 24 层、$D=2048$，示例注释仍使用 15B-A2B 名称。这些是代码构造默认值，[Qwen3 报告](../fundamentals/model/transformer-autoregressive-inference.md#2.2%20Qwen3%E7%A8%A0%E5%AF%86%E7%BB%93%E6%9E%84%E4%B8%8E%E6%8A%A5%E5%91%8A%E9%85%8D%E7%BD%AE)的 30B-A3B/235B-A22B 表是另一份模型配置证据；没有归档实际 checkpoint config 时，两者不能相互补填。

聊天角色、思考标记和 `enable_thinking` 等模板参数不在这两个模型前向内产生。通用 tokenizer 从显式模板或 `tokenizer.chat_template` 取得模板，将额外 kwargs 交给模板渲染；没有模板时会报错。本归档未核实际发布模板，不能仅凭通用接口证明 `/think`、`/no_think` 或关闭思考后的输出前缀。模式机制另见 [模式控制与蒸馏](../fundamentals/model/reasoning-post-training-and-distillation.md)。

本页实际读取稠密/MoE 的配置、前向与 LM head，动态缓存、普通 greedy/sample 的输入准备与状态接续、必要 loss 与 SDPA/FlashAttention 适配。没有审计全部生成策略、量化 Cache/backend、训练器、权重加载、TP/EP、hub kernel 或设备执行；本源码包也不含 Qwen3-VL 模型实现。

## 来源身份

- Summer-Summer/transformers，`hf-4.53.2`，`37f8b0b53512e6aae0cfd15746c133c101783178`，`https://github.com/Summer-Summer/transformers/tree/37f8b0b53512e6aae0cfd15746c133c101783178`；具体模块与静态核读范围见正文。
