---
title: 推测解码：候选、目标验证与每轮成本
type: concept
tags:
  - llm
  - inference
  - kv-cache
  - performance
sources:
  - raw/articles/2026-09-24/gpu-mode-lecture-materials
updated: 2026-10-15
---

# 推测解码：候选、目标验证与每轮成本

推测解码先用较便宜的提议过程生成多个候选，再让目标模型并行评分，由验证器决定哪些token可以成为正式输出。它希望用一次较贵的目标调用推进多个token；收益取决于实际接受的连续前缀、额外计算和状态管理成本。

本页依据GPU Mode Lecture22《A Hacker’s Guide to Speculative Decoding in vLLM》解释这条执行链。讲义链接的是旧vLLM `37464a0f745a0204da7443d2a6ef4b8f65e5af12`，与[普通V1执行页](../../implementation/vllm-inference-execution.md)的固定实现分别保留。下文采用每位置一个候选、形成单条序列的情形；树候选需要另外定义分支与attention mask。

## 1. 提议、评分与验证处理不同状态

设 $x$ 是已经提交的前缀，提议器继续生成 $k$ 个候选 $y_1,\ldots,y_k$。候选不是已确认输出；第 $j$ 个候选依赖 $x,y_{<j}$。记提议分布为 $q_j$，目标在相同上下文下的分布为 $p_j$。

| 阶段 | 接收什么 | 产生什么及其用途 |
| --- | --- | --- |
| Propose | 已提交前缀及提议器状态 | 候选ID与提议概率，供目标评分和验证使用 |
| Score | 前缀的上下文、候选与因果关系 | 对应位置的目标概率；一次前向可同时评分多个位置 |
| Verify | 候选ID及两侧概率 | 连续接受的候选前缀，以及拒绝后的纠正token或全部接受后的bonus token |

这些是讲义的职责分工，不表示每一阶段都独立启动一个GPU kernel。提议方法与验证方法也是两个选择轴：draft model、ngram等产生候选，标准拒绝采样、greedy或typical acceptance决定如何接受。讲义列出的是设计空间，具体组合需由相应实现支持。（Lecture22第17–20页。）

标准采样验证在首次拒绝处停止沿用原候选链，并按相应的校正分布产生纠正token；拒绝后的候选依赖了已被替换的上下文，不能直接继续提交。教学例：若 $y_1,y_2$接受、$y_3$拒绝，本轮可以输出 $y_1,y_2,z$，其中 $z$是纠正结果。纠正采样并不等同于忽略提议分布、无条件重抽目标token。若全部 $k$ 个候选接受，则可再由目标的下一位置分布采样一个bonus token。（第21/23页文字与第24页引用图；完整采样器公式和证明不在此讲义展开。）

这里的分布保持指适当拒绝采样协议下仍服从目标分布，并保留浮点数值条件；它不要求同随机种子下逐次生成相同字符串。Greedy一致与随机采样分布正确是不同判断，typical acceptance等有损策略又会改变这个目标。逐状态接受率与draft/target分布距离的关系见[一致性与推测接受](../../theory/quantization-ranking-stability.md#8.6%20%E4%B8%80%E8%87%B4%E6%80%A7%E4%BD%95%E6%97%B6%E6%9C%89%E7%94%A8%EF%BC%8C%E4%BD%95%E6%97%B6%E5%BF%85%E9%A1%BB%E5%8F%A6%E6%B5%8B%E4%BB%BB%E5%8A%A1)。（第13、32页。）

## 2. 每轮推进量与总成本共同决定收益

设连续接受的候选数为 $A\in\{0,\ldots,k\}$。在标准“接受前缀后纠正、全部接受后bonus”的流程中，若没有EOS或输出预算截断，一轮输出数是 $L=A+1$，因此

$$\mathbb E[L]=1+\sum_{j=1}^{k}\Pr(A\ge j).$$

这是整理者按输出流程和非负整数期望展开的关系，不假设不同位置独立。单个位置的平均接受率不能直接代替这些“前 $j$ 个全部通过”的概率；错误在序列中的聚集会改变整轮推进量。

讲义第12页以“轮耗时÷期望输出量”估计摊销延迟。若轮耗时也变化，可写作

$$\bar t_{\rm token}\approx
\frac{\mathbb E[T_{\rm round}]}{\mathbb E[L]}.$$

$T_{\rm round}$应包含提议、目标评分、验证及框架开销。讲义的教学例为普通解码30ms输出1token，推测40ms平均输出2.5token，得到16ms/token；这不是实测，也不表示这批token逐个间隔16ms到达。评估实际到达间隔、排队与服务吞吐，继续按[服务指标](../../implementation/serving-performance-evaluation.md)记录。

高接受率仍可能被昂贵的draft或低效目标评分抵消。批量增大时，普通decode本身能利用更多算力，推测额外工作可用的余量也会变化；讲义第26–28页据batch size选择参与推测的请求，必要时关闭推测。它是负载相关策略，阈值与收益需要在相同模型、长度、batch和服务协议下比较。

## 3. 输出提交与候选KV的有效范围需要配合

目标评分会为候选输入计算KV。为了保留接受部分、减少下轮重复计算，讲义的lookahead设计让scheduler额外预留KV空间，worker将候选KV写入其中，再保留已接受前缀对应的有效状态。（第25页。）

因此预留空间、实际写入与成为有效历史是三个阶段。被拒绝的后缀已经不属于正式上下文；新采样的纠正或bonus token若尚未作为目标输入，也没有由本次评分产生的目标KV。不能将本轮返回的token数直接当成新提交KV的数量。这里是按[自回归输入与缓存关系](transformer-autoregressive-inference.md)作的状态解释；讲义未给完整的逐槽提交、回滚与回收实现。

Attention接口也会改变评分成本。当时的单query PagedAttention只支持每序列一个query，Lecture22第29–31页将多个候选前缀展开成各含一个query的virtual sequences。第30页把 $k=3$画成四个评分项，包含bonus位置；多个评分项重复读取共同的KV，扩大了实际attention批次。支持分页多query的attention可以减少这种展开；这不是分页存储本身必须只支持一个query。

本页的三阶段、成本和状态关系可用于理解推测执行；当前固定V1的具体分派、采样和KV协议仍须沿对应源码核查，不能把历史讲义的对象名接到普通decode最后一行logit选择上。

## 来源身份

| 来源 | 固定身份与时间 | 使用范围 |
| --- | --- | --- |
| Supplementary Material for Lectures — GPU Mode（`https://github.com/gpu-mode/lectures/blob/77a8df418834e5789c12da23e7d2719e0efabef1/README.md`） | 目录commit `77a8df418834e5789c12da23e7d2719e0efabef1`；获取于2026-09-24T19:54:39+08:00；Lecture22为保存的Google Slides PDF导出 | 全36页讲义图文，主要第12–13、17–32页；目录commit与外部PDF保存身份分开。仅核讲义及其旧代码链接，未读取链接源码、采样证明或运行模型/性能测试。 |
