---
title: 推理服务评测：负载、延迟、吞吐与 SLO
type: concept
tags:
  - serving
  - evaluation
  - performance
sources:
  - raw/repositories/2026-09-21/vllm/source/vllm/benchmarks/serve.py
  - raw/repositories/2026-09-21/sglang/source/python/sglang/benchmark/serving.py
updated: 2026-09-22
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

## 5. 从量化收益定位到系统限制

下面是诊断路径，不是本次实测结果：

1. 固定模型、质量门槛和 workload，对比浮点/量化的显存、实际 batch 和服务指标。
2. 小批量线性层若受权重流量限制，压缩可能有效；大 batch、长 KV 或多卡通信主导时，收益可能不同。
3. 若单 Kernel 更快而 TTFT/TPOT 不变，继续区分排队、prefill 干扰、cache 访问、采样和通信，不把差异直接归因于量化算法。
4. 逐档提高到达率或并发，观察吞吐是否饱和、延迟及失败是否增加，再比较同一 SLO 下可承载的负载。

执行机制见 [vLLM 调度](vllm-inference-execution.md)、[SGLang 调度](sglang-inference-execution.md)及 [张量并行](tensor-parallel-quantization.md)；计时、预热和 GPU 同步见 [Kernel 测量](kernel-correctness-and-benchmarking.md)。这些层次需要共同解释同一次结果，局部算子毫秒数不能直接换算成服务加速比。

本页核对所选 benchmark 的定义与处理分支，没有启动服务、产生请求或测得性能曲线。

## 来源身份

- [vLLM](https://github.com/vllm-project/vllm/tree/568afb3a13806beb53bb2e6bd518269357b237c0)，`568afb3a13806beb53bb2e6bd518269357b237c0`；`vllm/benchmarks/serve.py:calculate_metrics`、请求生成与 benchmark 参数。
- [SGLang](https://github.com/sgl-project/sglang/tree/2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc)，`2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc`；`python/sglang/benchmark/serving.py` 的请求生成、时延与吞吐汇总。本页具体单 token 与 goodput 分支以 vLLM 所列实现为准，不声称两库所有指标处理相同。
