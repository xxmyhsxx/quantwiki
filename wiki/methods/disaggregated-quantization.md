---
title: Disaggregated Quantization：分阶段格式、权重与流式 prefill
type: method
tags:
  - serving
  - qat
  - weight-quantization
  - data-format
sources:
  - raw/papers/2026-09-23/disaggregated-quantization/paper.pdf
  - raw/repositories/2026-09-24/disaggregated-quantization/source/qad/quantizers/dual.py
  - raw/repositories/2026-09-24/disaggregated-quantization/source/qad/quantizers/frozen_decode.py
  - raw/repositories/2026-09-24/disaggregated-quantization/source/qad/training/qad.py
  - raw/repositories/2026-09-24/disaggregated-llama-cpp/source/src/llama-odp.cpp
  - raw/repositories/2026-09-24/disaggregated-llama-cpp/source/src/llama-context.cpp
updated: 2026-09-28
---

# Disaggregated Quantization：分阶段格式、权重与流式 prefill

DQ 将 prefill 与 decode 的量化选择拆开：长提示处理优先利用原生低精度矩阵乘，小batch生成优先减少权重读取。进一步可以给两个阶段训练不同权重，并在单设备上从SSD逐块加载只在prefill使用的副本。它既包含无需重新训练的格式切换，也包含成本显著的量化感知蒸馏，不能统称一种训练免费PTQ。

依据 arXiv:2609.26333v1，以及主仓库 `4a99d6e02817a796bfe3f3d7a5edbf75aeb9f4ba`、ODP fork `a0557fcf59e50a3bcd241e1a7de7b151d315720b` 的定向源码阅读。本文未运行训练、服务或性能测量。前置知识见 [阶段计算瓶颈](../implementation/quantized-matmul-scaling-execution.md#4-为什么-token-数改变瓶颈) 和 [服务内存与批处理](../implementation/serving-memory-and-batching.md)。

## 1. 三层分离分别解决什么

| 方案 | Prefill | Decode | 新增代价/限制 |
| --- | --- | --- | --- |
| 未分阶段NVFP4 | 共享W4A4 | 共享W4A4 | decode也支付激活量化误差与开销 |
| 格式分离NVFP4 | 共享W4A4 | 相同权重、A16 | 需要两种执行kernel，不增加一份独立权重 |
| 格式分离LUT | 紧凑LUT权重再量化成NVFP4，A4 | LUT2/3权重、A16 | prefill信息仍受低位LUT瓶颈限制 |
| 权重完全分离 | 独立训练NVFP4权重、A4 | 独立训练LUT权重、A16 | 多一套checkpoint，训练与驻留内存增加 |
| 完全分离+ODP | SSD逐块流入两块设备缓冲 | 生成时恢复并使用原decode权重 | 额外SSD读取、同步、冷启动和短提示停顿 |

LUT转NVFP4通常改变数值网格，不是无损dtype转换。单纯重转换不能恢复低位LUT已经舍掉的信息；独立prefill权重增加表示容量，正是完全分离区别于格式分离的原因。（§2.3–2.5。）

两阶段不必部署在两台机器：格式/权重是逻辑分离，vLLM/NIXL分服务与本地ODP是不同落地方式。也不能把“prefill总是计算受限、decode总是带宽受限”当成无条件事实；本论文重点是长提示、低并发及batch1生成，大batch与长KV可能改变瓶颈。

## 2. 响应损失怎样训练 prompt 侧权重

QADD 用SFT位置标签决定每个token走哪条路径：prompt/system的忽略标签位置走prefill，assistant位置走decode；在teacher forcing的一次前反向中最小化响应位置上的 $D_{KL}(p_{teacher}\|p_{student})$。teacher固定，主权重FP32、前向fake quant、反向STE。（§2.1、附录A.1。）

路径mask取决于**输入位置t的角色**，而损失用隐藏状态t预测标签t+1。最后一个prompt位置仍走prefill，却负责第一个response token的预测。因此prefill权重至少有两条梯度来源：边界处预测，以及后续响应通过attention访问prompt K/V的梯度。

若简单把“loss只在回答上”实现成detach整个prompt路径，就会破坏上述训练。反过来，冻结decode权重也不应关闭整个decode计算图：冻结的是参数更新，仍需对其输入求梯度，才能训练prefiller。

核心实验用Tülu3的100M非padding token、长度2048、全局batch64、约2450步，AdamW学习率3e-6，100步warmup。两套权重意味着额外master、梯度和优化器状态；Qwen3-8B完全分离使用16张B300，对照8张，不能用“同token预算”推断同训练资源。

## 3. 两套权重怎样共享历史状态

prefill权重产生各层prompt K/V，decode权重的query直接访问这些K/V，随后追加自己的K/V。保持层数、head形状等接口相容，可复用原attention结构，但**形状相同不代表表示分布自然匹配**；联合QADD或针对固定decoder训练prefiller负责适配。

对含线性attention的Qwen3.8-27B，评测还转移recurrent state，不只是传统KV；窄gate投影、递归参数等共享且冻结，不能简单复制所有参数为两份。（附录A.2/A.5。）

多轮存在明确未验证问题：缓存中的旧assistant token由decode权重产生，若把相同历史全部重跑prefill，会得到另一组状态。逐字相同的token历史不保证cache等价；前缀缓存、重算与agent多轮策略需额外验证。论文§5明确没有覆盖高batch、多轮与agent行为。

## 4. 格式位宽与kernel口径

NVFP4使用E2M1值、每16元素一个FP8-E4M3尺度，再加每tensor的FP32全局尺度。LUT2/3沿用两级缩放，但组尺度吸收最大绝对值元素的符号，使该元素归一化到+6，使用非对称且包含0的标量表：

$$C_2=\{-3.6517,0,2.5227,6\},$$
$$C_3=\{-4.7038,-2.8698,-1.3696,0,1.2204,2.5285,4.0473,6\}.$$

这些是按其缩放模型对高斯先验优化的固定表，不是均匀INT2/INT3。计入每16元素8-bit尺度后，分别是2.5、3.5 bit/weight，NVFP4是4.5；再加tensor全局尺度、未量化LM head和其他参数。不能把LUT2直接写成全模型2 bit。（附录A.3。）

作者的decode LUT kernel使用 `(N,b,K/32)` int32位平面、shared-memory小LUT、带符号E4M3组尺度与全局尺度，软件解码后执行weight-only乘法。其性能不要求硬件原生支持LUT算术。NVFP4A16另用Marlin-like混合精度路径，与W4A4 CUTLASS路径不同。（附录C.2。）

## 5. ODP为何能复用空间，什么时候不能隐藏读取

prefill按层顺序消费权重，一层输出计算完后该层权重可被下一层覆盖，而已生成的KV继续保留。两个设备slot轮换，一边计算一边准备下一层；slot从prefill暂时不用的decode权重空间借出，结束前将被覆盖权重从SSD恢复。

理想流水近似为首块装入 + 各阶段计算/装入的较大者 + 尾部恢复和同步。要隐藏下一块传输，应满足有效SSD→host→device读取时间不超过当前层计算时间；总SSD流量还包含被借用权重的恢复。这个成本模型为整理者解释，不是测得的通用延迟公式。

提示变长会增加每层计算但不增加同一checkpoint的读取字节，故更容易摊销传输。DGX Spark实验约8K开始计算超过加载，16K以上核心Qwen组的额外prefill开销低于5%。短提示可能更慢；MoE每步只激活部分参数，若流入大量不活跃专家则计算不足以遮盖读取，原文将ODP主要定位于dense模型。

“零额外设备权重驻留”不等于无额外总资源：SSD多一份prefill、host暂存、KV和工作区、恢复流量仍在，统一内存设备尤其不能把host开销忽略。

## 6. 固定代码核对：支持哪些结论

本次只沿对应入口和关键函数核对，未审查整仓库。

| 位置 | 静态可确认行为 | 边界 |
| --- | --- | --- |
| `qad/quantizers/dual.py` 的 `prefill_mask_from_labels`、`quant_phase` | `labels == -100`构造prefill mask；上下文恢复旧mask | mask需覆盖backward，梯度checkpoint重算否则会走错路径 |
| `DualSharedNVFP4Linear` / `DualSplitNVFP4Linear.forward` | 前者共用weight，仅prefill量化激活；后者两套weight，混合token时分别计算再where选择 | 这是训练模拟，混合mask不等于只执行被选token的稀疏GEMM |
| `training/qad.py` 的训练循环 | mask上下文包含forward和backward；默认loss取`labels_data[:,1:]`，head阶段mask取`[:,:-1]` | 核实了位置角色与causal shift分工；可选全prompt loss不是默认协议 |
| `frozen_decode.py` 的 `NVFP4FrozenDecodeLinear` | 外部decode作为不持久化BF16 buffer，`F.linear`保留输入梯度；仅导出prefill | 训练时不使用紧凑GGUF内核，不能由该路径宣称训练显存是1–3bit |
| `llama-odp.cpp` 的 `alloc`、`backup_donor`、`restore` | 实际donor选output head；备份到磁盘，最后层最后使用slot之后恢复，并使slot缓存失效 | head不可用时存在额外分配分支，因此并非所有运行都零额外设备空间 |
| `read_block_to_host`、`llama_odp_build_mm` | 多线程`pread`读到host，再上传；构图替换选定矩阵并设置校准activation scale | 是分阶段搬运，不能写成GPU直接从SSD零拷贝读取 |
| `llama-context.cpp` 的ODP启用条件 | 仅`ubatch.n_tokens > 1`挂接ODP和预取 | 这是原型的形状路由，不能泛化为多请求混合调度下精确角色识别 |

`llama_odp::alloc`能借出空间仍取决于donor容量；源码注释也明确单ring不是跨设备分片方案。head虽是阶段共享参数，在transformer栈期间暂时闲置，必须在最终logits计算前恢复，而不是等下一轮生成才恢复。

另一个需要保留的论文/代码差别：`NVFP4Lloyd43UpcastBothLinear`及2bit子类是同构对照，前向两阶段都用转换后的NVFP4权重，导出也直接走`NVFP4Linear.export_tensors`。这个导出产物本身不能证明论文概念中的“始终存紧凑LUT、运行时转换”完整路径。附录B.3对性能测试的限定进一步印证应分别核算。

## 7. 哪些性能是实测，哪些是开销估计

论文有三种不同证据，不能合成同一端到端加速：

1. **核心prefill栈**：DGX Spark、16K等上下文，计首transformer层到末层，含attention，排除embedding、RoPE表构建、最终norm/head。Qwen3-8B BF16为3801ms，resident NVFP4加速1.49×，ODP为1.47×；不是完整TTFT。
2. **核心decode**：vLLM batch1，以同168-token prompt生成8与128token，$(t_{128}-t_8)/120$估计每输出token时间；warmup后3次中位数，关闭prefix cache、启用CUDA graph。包含attention/norm/head，不能只用位宽比推导收益。
3. **真实ODP TTFT**：Qwen3.8-27B + IQ1_S原生llama.cpp，8K输入由12.27s到6.90s，1.78×；4K–32K为1.38–1.78×，短提示更慢。一次warmup后3次重复、关闭prompt cache。（§3.2、附录C.1。）

**附录B.3的重要限制：** 非分离LUT2/3的decode计时只是在线性层前多调用FP4激活量化并丢弃输出，估计该额外操作成本；没有测完整LUT→NVFP4权重重转换。prefill同样复用NVFP4计时、排除权重转换开销，weight-only参照用BF16。因此表1的LUT格式速度不是全部所述执行链均已完整落地的证明。

核心ODP栈微基准还用与借出空间等大的序列化字节payload模拟被驱逐权重，并计入恢复；Qwen3-8B的SSD流量3.64+0.20GB、Gemma3-12B为5.64+0.23GB。它与fork实际备份/恢复head是两种层次的验证，不能把模拟payload称为已核验完整模型状态恢复。

## 8. 质量提升的范围与负例

表1的速度/设备量分别来自Qwen3-8B与Gemma3-12B，质量列却是多个模型尺寸的**家族平均**，并对最后五个checkpoint平均，不能把同一行所有数字归到单个模型。

以Qwen3家族LUT2为例：未分离DH/PH=34.8/61.3，格式分离37.2/61.0，完全分离45.5/76.6，weight-only38.4/64.1。这里格式分离主要改善decode-heavy，PH略降；完全分离的prefill容量才有更大改善。正文§3.2对Qwen3 LUT2的PH增幅写5.3，与表中76.6−61.3=15.3不符，本页以原表算术为准并保留差异。

冻结decoder的27B结果更不单调：IQ1_S的MMLU-Pro29.04→61.54、MMMU-Pro24.39→59.65；IQ3_S则83.65→83.02、74.74→71.97。该分支另用约95M文本推理token、每种decoder训练自己的prefiller，不能把其收益当作任意权重对直接拼接的效果。附录B.4中有跨decoder的prefiller超过原配，也有明显下降，配对训练不是普遍最优保证。

27B质量评测在GB300/vLLM上将GGUF decode解量化为BF16，并传递KV及递归状态；图中GGUF压缩尺寸不是该评测后端实际设备分配。其结果是step980单次完整评测，无重复/多checkpoint平均。与原生llama.cpp的速度证据保持分开。

核心QADD误差条只反映同一次训练后期五checkpoint的变化，未覆盖随机种子方差。大模型PTQ格式切换实验13个model-task组合中11个点估计改善、6个逐比较显著，另有小幅下降且不显著；它不是13个同时成立的总体保证。

最后，decode权重与单token路径不变，只意味着每token成本可保持；prefiller改变回答内容与生成长度，故整段生成总成本未必不变。附录B.5同时报告缩短与长尾增长，必须结合TTFT、输出token数和每token延迟评价。

## 来源身份

| 来源 | 版本 | 范围 |
| --- | --- | --- |
| [Disaggregated Quantization: Specializing LLM Prefill and Decode](https://arxiv.org/abs/2609.26333v1) | arXiv:2609.26333v1 | §2–3/5、附录A/B/C、表1/5/7/10/11；表1回查页图 |
| [官方QADD仓库](https://github.com/IST-DASLab/disaggregated-quantization/tree/4a99d6e02817a796bfe3f3d7a5edbf75aeb9f4ba) | `4a99d6e02817a796bfe3f3d7a5edbf75aeb9f4ba` | `dual.py`、`frozen_decode.py`与训练循环所列函数 |
| [ODP llama.cpp fork](https://github.com/IST-DASLab/disaggregated-llama.cpp/tree/a0557fcf59e50a3bcd241e1a7de7b151d315720b) | `a0557fcf59e50a3bcd241e1a7de7b151d315720b` | ODP装入、借用、恢复、矩阵替换与context启用条件；未运行 |
