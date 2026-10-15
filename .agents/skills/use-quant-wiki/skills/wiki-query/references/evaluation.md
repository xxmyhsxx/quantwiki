# 误差与评测：先分清比较对象和口径

用于解释不一致、质量下降或性能数字。先区分数值误差、生成行为、任务质量、算子时间和服务指标，再读对应正文。

| 要回答的问题 | 正文入口 | 可补查的词 |
| --- | --- | --- |
| 模拟量化与真实执行为什么不一致 | [误差诊断](/Users/xiao/Documents/Projects/quantwiki/wiki/implementation/quantization-error-diagnosis.md) | fake quant、packing、逐层、舍入 |
| PPL、分布距离与生成任务如何比较 | [模型质量协议](/Users/xiao/Documents/Projects/quantwiki/wiki/implementation/model-quality-evaluation.md) | 同前缀、KL、校准/留出、输出预算 |
| 小误差为什么改变 token 或专家选择 | [排序稳定性](/Users/xiao/Documents/Projects/quantwiki/wiki/theory/quantization-ranking-stability.md) | margin、KL/TV、top-k |
| TTFT、吞吐、显存怎样解释 | [服务性能口径](/Users/xiao/Documents/Projects/quantwiki/wiki/implementation/serving-performance-evaluation.md) | batch、长度、TPOT、SLO、分母 |
| Kernel 的正确性与计时怎样组织 | [算子验证与计时](/Users/xiao/Documents/Projects/quantwiki/wiki/implementation/kernel-correctness-and-benchmarking.md) | tolerance、Sanitizer、warmup、同步 |
| 平均准确率之外怎样判断可靠性 | [可靠性与弃答](/Users/xiao/Documents/Projects/quantwiki/wiki/theory/quantization-reliability-and-selective-prediction.md) | selective、risk、coverage、校准 |

协议与诊断先查 `implementation/`，误差机制查 `theory/`。具体论文的结果和条件回到 `methods/` 的相应页面；查询可以说明哪些证据仍缺，但不把制定验证方案写成已运行，也不为回答指标问题启动实验。
