---
title: 模型生成 Kernel 的任务与评测：覆盖率、搜索预算和条件速度
type: concept
tags:
  - kernels
  - evaluation
  - performance
  - llm
sources:
  - raw/papers/2026-09-21/kernelbench
  - raw/papers/2026-09-21/tritonbench
updated: 2026-10-15
---

# 模型生成 Kernel 的任务与评测：覆盖率、搜索预算和条件速度

评价一个生成器，需要同时知道它能解决多少任务、成功程序有多快，以及为找到这些程序用了多少尝试。少数正确程序的高速度、全部任务的成功覆盖和评测系统每秒处理多少候选，是不同统计对象。

本页依据KernelBench v1与TritonBench v1解释任务和指标。单个候选的输入、数值与计时检查复用[算子验证与测量](kernel-correctness-and-benchmarking.md)；如何用外部结果驱动下一份代码，见[CudaForge反馈循环](gpu-kernel-performance-analysis.md)。

## 1. 一个生成任务可以包含多个设备kernel

KernelBench提供PyTorch `Model(nn.Module)`，由构造函数、`forward`及辅助函数规定计算；输出是相同接口的 `ModelNew`。其中 `get_inputs`提供forward输入，`get_init_inputs`提供构造参数。论文附录A的一句respectively把二者顺序写反，本页按同版示例代码的实际角色解释。

生成器需要选择替换哪些操作，并提供调用CUDA的外围集成；协议允许保留部分框架操作。一个任务因此可能由多个kernel、框架调用和分配共同完成，250个任务不等于250个独立CUDA函数。KernelBench v1的三个层次是：

| 层次 | 数量 | 参考程序构成与主要机会 |
| --- | ---: | --- |
| L1 | 100 | 单一原语，常需与已有优化库竞争 |
| L2 | 100 | 操作序列，提供融合和算法重组机会 |
| L3 | 50 | 完整架构，需识别值得替换的子图 |

分层依据原语构成，不是普适难度全序。任务指定shape和精度后随机改变数值，允许面向该契约专门化；协议可以接纳不同编程库，但v1实验实际研究raw CUDA。（KernelBench §3.1–3.2、§6、附录A/C。）

参考程序也有两个角色：其输出提供待匹配的语义，其测量时间提供速度基线。改用Eager或某种编译模式，会改变是否达速；语义参考与性能基线需分别登记。

## 2. 全任务覆盖率与正确子集的速度回答不同问题

令选定任务集有 $N$ 项，$c_i$表示第 $i$项在声明用例/容限下通过检查，$T_{\mathrm{ref},i}$与 $T_{\mathrm{new},i}$为相同协议下的耗时。KernelBench定义

$$s_i=\frac{T_{\mathrm{ref},i}}{T_{\mathrm{new},i}},\qquad
\mathrm{fast}_p=\frac1N\sum_{i=1}^{N}
\mathbf1[c_i\land s_i>p].$$

分母包含全部任务，编译、执行或数值失败都没有分子贡献。按论文定义，$\mathrm{fast}_0$用作正确覆盖率，$\mathrm{fast}_1$要求**严格快于**参考；提高 $p$得到要求更高的联合覆盖曲线，不是平均加速比。分层汇总时，三层任务数为100/100/50，合并全250任务应按任务数加权。（KernelBench §3.3、图3。）

只对正确程序画速度分布，则描述“成功之后有多快”。不同生成器成功的任务集合可能不同，一个只解决少量容易任务的模型也可能有很高的条件速度，不能据此替代全任务覆盖。KernelBench附录B.3/图7的箱线图明确只含正确候选；跨设备图8/9也使用各自正确集合。$\mathrm{fast}_p$的分母则始终是选定任务总数。

这类任务比例也不同于[服务goodput](serving-performance-evaluation.md)：后者还有请求到达、时间或容量约束，不能把“多少任务正确且变快”读成“每秒完成多少请求”。

## 3. 预算内至少成功一次，不等于最后一轮成功

重复采样或反馈修订会给每任务产生多份候选。用 $B$统一表示候选预算，可将KernelBench §5.1的定义整理为

$$\mathrm{fast}_p@B=\frac1N\sum_{i=1}^{N}
\mathbf1\!\left[\exists b\le B:\ c_{ib}\land
\frac{T_{\mathrm{ref},i}}{T_{ib}}>p\right].$$

它按任务统计：预算内至少有一份正确且达速便计成功，分母不是 $NB$。公式不要求尝试独立，也不是pass@k的无偏概率估计。Best-so-far曲线随预算不降，部分来自保留历史成功这一统计定义，不意味着每次修改都改善。

正确性本身也会倒退：附录D.4、Table7的候选第8轮通过，第10轮重新出错；第8轮1.33ms仍慢于该例Eager的1.29ms，因此“出现过正确候选”和“$\mathrm{fast}_1$成功”又是两个判断。评测和交付都应保留候选身份，不能把最后一轮自动当作最好版本。

比较预算还须声明单位。论文Table2匹配的是每任务10次模型调用，没有因此匹配token、编译时间、GPU时间或总墙钟。重复采样与反馈修订的温度/提示也不同，应随方法保留。（§5.1、附录C。）

修订输入包括上一份生成 $G$、编译/执行反馈 $E$及可选profile $P$。编译日志、shape/value不匹配、超时和算子时间分解提供不同信号；它们参与了当前任务上的筛选和改写，属于搜索证据。独立留出评估与预算含义见[生成与反馈的评价](model-quality-evaluation.md#11.%20%E6%90%9C%E7%B4%A2%E5%8F%8D%E9%A6%88%E3%80%81%E7%95%99%E5%87%BA%E8%AF%84%E6%B5%8B%E4%B8%8E%E8%81%9A%E5%90%88%E6%8C%87%E6%A0%87%E8%A6%81%E5%88%86%E5%88%AB%E5%AE%9A%E4%B9%89)。

## 4. TritonBench的任务来源和成功子集改变速度含义

TritonBench v1分为两个通道，生成对象都包含Triton GPU kernel与Python wrapper，但输入要求和性能参考不同：

| 通道 | 任务来源与生成器得到的要求 | 速度参考 |
| --- | --- | --- |
| G，184项 | 从公开Triton实现人工补全、去重和调试；指令由实现生成后核验，包含接口和实现描述 | 已整理的Triton实现 |
| T，166项 | 按PyTorch 2.6.0接口/文档构造常见、少见操作及可连接的组合任务 | 标准PyTorch调用 |

G偏向已有公开实现与常见算子，T的任务分布不同；“组合/融合任务”也不表示每份生成结果已经融合成单个kernel。比较加速前应保留通道、任务和参考身份。（TritonBench §3–4、Table1/2。）

该论文把无错误调用的Call Accuracy、输出正确的Execution Accuracy与正确子集的Speed Up分列，Speed Up单项为 $t_{\mathrm{ref}}/t_{\mathrm{gen}}$。Table4中DeepSeek-R1在T的zero/one-shot结果，Execution为53.01%/45.78%，Speed Up却为1.03×/1.91×：条件速度提高与任务覆盖下降可以同时发生。两列不能互相代替；表内跨任务/shape的具体速度聚合公式未完整给出，不自行补定为某种平均。（§3.5、§4.3、Table4。）

G协议另报告GPU Efficiency：专家按输入注释有效Bytes/FLOPs，用 `do_bench`测时间，算出GB/s或FLOP/s相对A100相应理论峰值的比率，再取尺寸扫描中的峰值。它描述所测实现在哪个尺寸达到的相对吞吐，既不是全部尺寸的平均表现，也不是occupancy或NCU硬件计数器；注释字节数与实测HBM流量仍按[roofline口径](gpu-kernel-performance-analysis.md#4.%20Roofline%20%E4%B8%8E%E5%AE%9E%E9%99%85%E6%B5%81%E9%87%8F%E8%A6%81%E7%94%A8%E5%90%8C%E4%B8%80%E5%8F%A3%E5%BE%84)区分。T结果表没有报告该效率列。（附录B、Figure6/7。）

## 5. 测量协议和评估系统吞吐分别记录

KernelBench v1除跨设备实验外采用L40S、Python3.10、PyTorch2.5.0+cu124、CUDA12.4。每任务用5组随机输入比较输出；forward预热3次后用CUDA Events测100次取均值，速度比取两侧均值之比。原文称wall-clock，附录B.1实际给的是Events协议，应按[事件与主机计时边界](kernel-correctness-and-benchmarking.md)解释。Torch.compile的编译成本被排除，主表另比较Inductor default，其余多项分析主要使用Eager；基线模式是分数身份的一部分。

规模化评估可以并行生成、在CPU预编译，将候选分发到多张GPU，同时保持同卡一次只评估一个候选。KernelBench附录H的编排把生成、编译、执行和profile分成状态，仅执行/profile阶段申请GPU；故障后重建进程以接续任务。它提高的是评估系统吞吐，不改变单候选计时需要隔离的条件。

两篇论文的通过率都绑定其实际测试范围。随机用例、模型文本相似度或能调用，与完整输入域的数值/内存正确性不是同一证据；具体判据与未给出的harness范围按来源记录保留。

## 来源身份

| 来源 | 固定版本 | 核读与使用范围 |
| --- | --- | --- |
| KernelBench: Can LLMs Write Efficient GPU Kernels?（`https://arxiv.org/abs/2502.10517v1`） | `arXiv:2502.10517v1` | 主文与active附录A–H/表1–15；本页主要§3/5、附录A/B/C/D.4/H，关键图表原页回查。未核仓库evaluator、全部生成程序或运行，未补完整容限/seed协议。 |
| TritonBench: Benchmarking Large Language Model Capabilities for Generating Triton Operators（`https://arxiv.org/abs/2502.14752v1`） | `arXiv:2502.14752v1` | 正文、表1–6、附录与四构造prompt；本页主要§3–4与附录B。原件表3个别Call/Execution值冲突保留，未核完整任务/harness、训练或A100运行。 |
