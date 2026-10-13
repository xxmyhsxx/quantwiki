---
title: 推理服务评测：负载、延迟、吞吐与 SLO
type: concept
tags:
  - serving
  - evaluation
  - performance
sources:
  - raw/papers/2026-09-21/distserve
  - raw/repositories/2026-09-21/vllm
  - raw/repositories/2026-09-21/sglang
updated: 2026-10-14
---

# 推理服务评测：负载、延迟、吞吐与 SLO

Kernel 基准回答一次设备运算有多快；服务基准还包含排队、调度、缓存、采样、序列长度与客户端观测。量化可能省下显存，让系统接纳更多请求，却同时增加每个请求的延迟。因此评测必须共同记录负载、质量和延迟目标。

## 1. 到达率、并发和 batch 不同

到达率是每秒向系统提交多少请求；并发是尚未完成的请求数；一次模型迭代的 batch 是调度器此刻组合的请求或 token。三者不等价。等待中的请求并不都已进入 GPU batch，一个 prefill 请求也可能贡献许多 token 行。

开放负载按指定时间过程产生请求，便于观察排队何时增长；封闭负载维持有限在途请求，完成后再提交，系统变慢会反过来降低提交速度。只说“并发 32”无法还原开放负载的 32 requests/s。两库 benchmark 中的 request rate、并发限制及请求长度应分别记录，不能把 `max_num_seqs` 直接当作实际负载。

比较配置时固定输入/输出长度分布、前缀共享比例、到达过程、是否预热，以及是否允许提前 EOS。缓存命中率高的测试不能直接代表每次冷前缀的工作负载。

## 2. 时间轴决定指标含义

设客户端提交时刻为 $t_0$，观测第一个输出 token 为 $t_1$，最后一个为 $t_L$，输出 token 数为 $L$：

| 指标 | 定义 | 回答的问题 |
| --- | --- | --- |
| TTFT | $t_1-t_0$ | 多久看到第一个输出，包含观测范围内的排队等成本 |
| E2E | $t_L-t_0$ | 整个请求多久完成 |
| TPOT | $(\mathrm{E2E}-\mathrm{TTFT})/(L-1)$，$L>1$ | 首 token 后，每个后续 token 的平均间隔 |
| ITL | 相邻输出 token 的间隔序列 | 输出过程中是否出现停顿 |

这是理想逐 token 流式观测。实际客户端可能以多个 token 一起返回的 chunk 记时间，需核对事件粒度；TTFT 也可能对应第一个有效内容 chunk，不能假定等于某个 GPU kernel 的结束时刻。

vLLM `calculate_metrics` 使用上述 TPOT 关系；$L\le1$ 的请求不进入 TPOT 分布，但在 goodput 判定里将其 TPOT 记为 0。这是该实现的约定，不是“一 token 请求有可测的后续生成速度”。该函数还说明不能用 ITL 条目数替代输出 token 数，因为一次返回可包含多个 token。

TPOT 的 p99 是**每请求平均间隔**的 p99；ITL 的 p99 是被纳入的单次间隔分布的 p99。长输出请求贡献更多 ITL 样本，二者不能互换。相同平均 TPOT 也可能有截然不同的停顿。

## 3. 吞吐要固定分母与计数对象

在约定测量区间 $T$ 内：

$$\text{request throughput}=N_{\mathrm{completed}}/T,\qquad
\text{output token throughput}=\sum_i L_i/T.$$

若另报输入加输出 token/s，应明确命名。失败数、失败类型、实际完成的输出长度要一起记录；只报成功请求延迟可能掩盖高失败率。短测试的启动和排空时间占比大，也不能与长时间稳态区间混比。

量化改变输出或 EOS 时，实际 $L_i$ 可能改变。固定生成长度可用于隔离服务性能，但这是专门的性能协议；实际任务质量仍需按正常停止条件在 [模型质量评测](model-quality-evaluation.md)中判断。

## 4. Goodput 把延迟目标放进吞吐

服务目标（SLO）可以要求 TTFT 不超过 $a$、TPOT 不超过 $b$、E2E 不超过 $d$。令 $I_i$ 表示请求成功且同时满足已指定的所有条件，则请求 goodput 为

$$G=\frac{\sum_i I_i}{T}.$$

vLLM `calculate_metrics` 对配置中的条件取交集，再用 `good_completed / dur_s` 汇总。它是满足延迟目标的请求吞吐，默认不包含答案质量判定；若需要“正确且按时”，必须额外定义正确性条件。

教学例：同样 10 秒内，配置 A 完成 100 个请求、60 个满足 SLO，配置 B 完成 90 个、85 个满足 SLO。A 的吞吐为 10 requests/s，高于 B 的 9；goodput 却为 6，低于 B 的 8.5。因而扩大 batch 获得更高吞吐，不一定改善用户侧结果。

只比较平均延迟也不够：两组平均值相同可以有不同长尾。报告 p50/p95/p99 时写清样本数与分位数算法；样本很少时，p99 主要反映最极端的几个观测。

### 4.1 DistServe 的容量目标与实验条件

DistServe v3 的 per-GPU goodput 是固定工作负载、TTFT/TPOT 阈值和 attainment 目标后，**可承载的最大请求到达率除以配置的全部 GPU 数**。对一个配置 C，整理为

$$\lambda^*(C)=\sup\{\lambda:A_{\rm joint}(C,\lambda)\ge\eta\},\qquad
G_{\rm DS}(C)=\frac{\lambda^*(C)}{N_{\rm GPU}(C)}.$$

$A_{\rm joint}$ 是同一请求同时满足 TTFT 与 TPOT 阈值的比例，$\eta$ 是目标满足率；GPU 分母包含 prefill、decode 和所有副本。这个容量口径不同于上文给定测量区间的合格完成数/T。分别求两项延迟的 P90，也不能自动推出同一请求的联合满足率为90%；分阶段估计还须由完整服务的交集结果复核。论文的 placement/实例与 KV 衔接见[阶段分离机制](serving-memory-and-batching.md#5.1%20DistServe%EF%BC%9A%E4%B8%A4%E9%98%B6%E6%AE%B5%E5%AE%9E%E4%BE%8B%E4%B8%8E%20KV%20%E8%BD%AC%E7%A7%BB)。（来源 §1–2、§4、§6.1；公式为按其定义的整理。）

主文目标 $\eta=90\%$。Rate 扫描固定延迟阈值；SLO Scale 扫描固定 rate，并用同一系数同时缩放 TTFT 与 TPOT 阈值，系数越小要求越严。附录C的99%目标另列，不能与主文倍数混用。表1与图7的负载条件为：

| 应用/模型 | TTFT / TPOT 阈值 | 输入 / 输出长度均值 |
| --- | --- | --- |
| ShareGPT，OPT-13B | 0.25s / 0.1s | 755.5 / 200.3 token |
| ShareGPT，OPT-66B | 2.5s / 0.15s | 同上 |
| ShareGPT，OPT-175B | 4.0s / 0.2s | 同上 |
| HumanEval，OPT-66B | 0.125s / 0.2s | 171.3 / 98.2 token |
| LongBench summarization，OPT-66B | 15s / 0.15s | 1738.3 / 90.7 token |

测试床为四节点共32张 SXM A100 80GB，节点内 NVLink、节点间25Gbps；32张是测试床容量，不表示每项实验均占满32张。所有主结果用 FP16 OPT/MHA，长度从数据集采样、到达由泊松过程合成，SLO 由作者经验设定。LongBench 输入受 OPT 最大2048位置限制，不能当作原始长上下文任务的未截断结果。论文当时 vLLM 基线用 TP=1/4/8；DeepSpeed-MII 因词表整除/OOM 没有 OPT-175B 结果，这些是历史配置边界。（§6.1。）

在上述90%目标下，作者报告 ShareGPT 的可承载率/GPU 相对 vLLM 为2.0–4.6×、相对 MII 为1.6–7.4×；HumanEval/OPT-66B 相对 vLLM 为5.7× rate、1.4×更严 SLO；LongBench/OPT-66B 相对 vLLM 为4.3× rate、12.6×更严 SLO。7.4×和12.6×来自不同任务与基线，rate 与 SLO Scale 也来自不同扫描，不能合为一套配置的收益。（§6.2，图8–9。）

传输成本需绑定 placement 和分母。作者在 OPT-175B/ShareGPT 的同节点配对 placement 中，将所有请求各阶段累计时间求和，报告 KV 传输占比小于0.1%；它不是单请求尾延迟界。正文另称95%以上传输小于30ms，但图10的 OPT-175B 曲线在30ms附近尚未达到0.95并有长尾，图文差异保留，不当作三模型统一分位界。（§6.3，图10。）

另外两处原件差异影响复用：附录B表3的 OPT-66B 为 prefill TP4/PP1、decode TP2/PP2，不能直接对上 §4.2/算法2的共同流水阶段与对应层段同节点叙述，论文未讲清该配置怎样满足传输约束；§6.5 把算法1标作 Low、算法2标作 High，与 §4 的算法标题和 §6.4 对调。本页按算法标题解释机制，不补造 placement 或运行时间对应关系。

§6.4 的 vLLM++ 与 DistServe-High/Low 消融是模拟实验；表2模拟器与系统的 attainment 差异在所列样本中小于2个百分点，不能视为任意负载的保证。上述内容来自 v3 论文与关键 PDF 公式/图表核对，本次未核作者代码、复现服务或测量质量/性能。

## 5. 从量化收益定位到系统限制

下面是诊断路径，不是本次实测结果：

1. 固定模型、质量门槛和 workload，对比浮点/量化的显存、实际 batch 和服务指标。
2. 小批量线性层若受权重流量限制，压缩可能有效；大 batch、长 KV 或多卡通信主导时，收益可能不同。
3. 若单 Kernel 更快而 TTFT/TPOT 不变，继续区分排队、prefill 干扰、cache 访问、采样和通信，不把差异直接归因于量化算法。
4. 逐档提高到达率或并发，观察吞吐是否饱和、延迟及失败是否增加，再比较同一 SLO 下可承载的负载。

执行机制见 [vLLM 调度](vllm-inference-execution.md)、[SGLang 调度](sglang-inference-execution.md)及 [张量并行](tensor-parallel-quantization.md)；计时、预热和 GPU 同步见 [Kernel 测量](kernel-correctness-and-benchmarking.md)。这些层次需要共同解释同一次结果，局部算子毫秒数不能直接换算成服务加速比。

本页核对所选 benchmark 的定义与处理分支，没有启动服务、产生请求或测得性能曲线。

## 来源身份

- DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving（`https://arxiv.org/abs/2401.09670v3`），`arXiv:2401.09670v3`；§1–6、附录A–C，goodput/attainment、表1–3与图7–14的条件和原件差异，未核作者实现或运行。

- vLLM（`https://github.com/vllm-project/vllm/tree/568afb3a13806beb53bb2e6bd518269357b237c0`），`568afb3a13806beb53bb2e6bd518269357b237c0`；`vllm/benchmarks/serve.py:calculate_metrics`、请求生成与 benchmark 参数。
- SGLang（`https://github.com/sgl-project/sglang/tree/2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc`），`2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc`；`python/sglang/benchmark/serving.py` 的请求生成、时延与吞吐汇总。本页具体单 token 与 goodput 分支以 vLLM 所列实现为准，不声称两库所有指标处理相同。
