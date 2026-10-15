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
  - raw/papers/2026-09-21/fastkernels
  - raw/papers/2026-09-21/flashinfer-bench
  - raw/papers/2026-09-21/kernelbenchx
updated: 2026-10-15
---

# 模型生成 Kernel 的任务与评测：覆盖率、搜索预算和条件速度

评价一个生成器，需要同时知道它能解决多少任务、成功程序有多快，以及为找到这些程序用了多少尝试。少数正确程序的高速度、全部任务的成功覆盖和评测系统每秒处理多少候选，是不同统计对象。

本页从KernelBench与TritonBench的任务和指标，接续FastKernels、FlashInfer-Bench及KernelBenchX的生产输入、数值契约与统计归约。单个候选的输入、数值与计时检查复用[算子验证与测量](kernel-correctness-and-benchmarking.md)；如何用外部结果驱动下一份代码，见[CudaForge反馈循环](gpu-kernel-performance-analysis.md)。

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

通过率都绑定各协议的实际测试范围。随机用例、模型文本相似度或能调用，与完整输入域的数值/内存正确性不是同一证据；具体判据与未给出的harness范围按来源记录保留。

## 6. 生产输入包含张量取值、状态与通信关系

FastKernels v1从模型配置和真实forward中的操作建立任务：L1原语、L2融合段、L3层、L4完整模型。其执行Tier则是另一维：本地注册任务、用户端到端负载、标准化模型/TP/负载比较；不能把Level与Tier互换。

替换一个模块还要匹配构造参数、权重布局、预处理状态、KV/cache及通信元数据。只按shape生成随机张量可能破坏实际数据分布。例如MoE路由取决于门控输入，随机token、门控前随机张量与真实请求会产生不同专家负载；捕获真实运行中的输入再重放，才能保留这种依赖。任务来自真实操作，也不代表每份测试输入都是真实请求。（FastKernels §3–4、附录B/D。）

分布式语义同样属于契约。论文中的all-reduce案例在单进程身份参考下接受了copy，放到实际跨rank求和便错误。通信任务须保留rank组与集体操作；单rank数值通过不能替代跨rank检查。

FlashInfer-Bench v1用Trace将可复用的数学定义与具体运行记录分开：

| 对象 | 保留什么 | 在评测中的作用 |
| --- | --- | --- |
| Definition | 输入输出、dtype、常量/变量轴及关系、PyTorch数学参考 | 规定同一任务的语义；语言和硬件另存 |
| Workload | 变量轴赋值、输入张量dump/随机生成/标量 | 指定一次要检查的实际情境 |
| Solution | 源文件、调用入口、语言/硬件/依赖条件 | 确定候选实现及可运行环境 |
| Evaluation | 绑定以上三者、状态、误差、时间、环境与时间戳 | 保存某候选在某负载上的测量身份 |

例如GQA的page数与张量shape不足以恢复历史：indptr差值、page索引取值决定每条请求读哪些KV。Trace从ShareGPT驱动的SGLang运行中采集，重要输入保存dump，其他输入可按seed随机生成；随后按batch、长度等性能相关轴去重。所得代表负载适于比较覆盖，并非生产请求频率的自然抽样。（FlashInfer-Bench §3.1–3.2、附录A。）

## 7. 数值门、输入归约与任务归约分别定义

KernelBenchX v2扩展TritonBench-T，形成176任务/15类别，包含六项手工量化任务。候选先通过导入、kernel入口与任务约束检查，再在共享seed下递归比较输出；shape/dtype须完全一致，之后才检验数值。标准正态输入之外，异常值模式以0.1%概率放大50倍，用于暴露只适应常见分布的代码。

量化任务要求scale、离散化与转换等显式逻辑，并禁用指定高层量化API；静态检查通过后，输出还须**同时**满足余弦相似度下界、L1相对误差上界和RMSE上界。前四项W8A8的三阈值为0.95/0.05/0.10，attention_W8A8和linear_W4A16为0.90/0.10/0.15。它们是任务的近似精度约定；完整尺度轴、舍入/裁剪及W4物理packing仍需匹配实现说明，不能由阈值反推。（KernelBenchX §3.2、附录A/C、Table6。）

同一任务有多个输入时，先说明输入层怎样归约。KernelBenchX附录A.4定义

$$s_i=\frac{\sum_j t_{\mathrm{ref},ij}}{\sum_j t_{\mathrm{gen},ij}}.$$

它主要再对正确任务集合 $C$取算术平均 $\frac1{|C|}\sum_{i\in C}s_i$；另可报告全部时间的总比值。这两者权重不同，也都不同于先把每个输入速度比等权平均。失败项影响覆盖率，但不进入该正确子集速度；不同轮次新救回的任务较慢，也会拉低条件均值，不能只由均值下降推断原本正确代码全部退步。

### 生产评测中的三种分母

| 协议 | 覆盖或达速比例的分母 | 速度与质量怎样合并 |
| --- | --- | --- |
| FastKernels §6.2代理主表 | Correct为正确/attempted；blocked另列；附录另报正确/全部88目标 | 速度几何均值与fast@阈值都只在正确目标上统计 |
| FastKernels §5 MacroEval | 有效项目/全部项目；再按任务族等权归约 | 正确性、覆盖与有效子集速度分列，再形成默认乘积 |
| FlashInfer-Bench §4.2 | 每份solution在其全部workloads上算fast_p，再跨solutions等权平均 | 速度参考通常为FlashInfer；缺少对应实现时使用PyTorch |

第一行例如DrKernel：8/38 attempted正确，但全部目标覆盖是8/88，fast@1为正确者中的1/8；它不是前述KernelBench全任务分母的fast_1。FastKernels主表各生成器采用各自公开配置，没有共同调用/token/GPU预算。

MacroEval还需要说明“有效”和“平均”的层次。每次请求的输出差异经任务的good/bad阈值映成质量分数，再依次平均请求、项目与任务族；项目质量达到阈值才有效。有效项目的速度将吞吐比与延迟比混合：

$$s_i=\left(\frac{\mathrm{throughput}_{\mathrm{cand}}}
{\mathrm{throughput}_{\mathrm{ref}}}\right)^\lambda
\left(\frac{\mathrm{latency}_{\mathrm{ref}}}
{\mathrm{latency}_{\mathrm{cand}}}\right)^{1-\lambda},
\quad\lambda=0.5.$$

速度先在每族有效项目内取几何均值，再跨非空有效族等权取几何均值；无有效项目的族由覆盖率反映，不给失败者填零后取对数。默认总分是速度、宏平均质量、宏平均覆盖率三轴乘积，仍需同时报告三轴。它与论文另表46架构端到端吞吐比的算术平均是不同实验；完整质量阈值清单未随论文给出。（FastKernels §5、§6.2、附录C及Table7/8。）

FlashInfer-Bench则对solution $j$的 $N_j$个workloads先算
$f_{p,j}=\frac1{N_j}\sum_i\mathbf1[c_{ji}\land s_{ji}>p]$，
再对 $J$份solution取 $F_p=\frac1J\sum_jf_{p,j}$。$N_j$不同时，这不等于将全部测量行合池，更不是生产时间加权。数学参考中的PyTorch与测速度的FlashInfer也要分开；低精度判据中的匹配比例 $\rho$见[数值验证](kernel-correctness-and-benchmarking.md)，与速度阈值 $p$无关。

## 8. 离线选择怎样进入运行调用

FlashInfer-Bench的apply层把离线Evaluation用于运行分派：先按数值要求过滤记录，以workload特征/shape形成key，再为每个key选择已测最快的solution。部分选中实现预编译，其余按需JIT；在线调用根据参数解析Definition/key，命中便调用对应实现，未命中或关闭功能时走原函数。（§3.5、Figure5。）

这里选择的是**已测过的候选**，不是每次请求现场搜索。key必须保留影响语义与性能的条件；论文没有给出完整ragged key或兼容性失效协议，不据概念图补齐实现。单kernel变快也还要经过实际调用链测量：原文Figure8的替换RMSNorm与端到端结果未显示相对原始FlashInfer的加速，因此本页只采用其选择与fallback机制。

## 来源身份

| 来源 | 固定版本 | 核读与使用范围 |
| --- | --- | --- |
| KernelBench: Can LLMs Write Efficient GPU Kernels?（`https://arxiv.org/abs/2502.10517v1`） | `arXiv:2502.10517v1` | 主文与active附录A–H/表1–15；本页主要§3/5、附录A/B/C/D.4/H，关键图表原页回查。未核仓库evaluator、全部生成程序或运行，未补完整容限/seed协议。 |
| TritonBench: Benchmarking Large Language Model Capabilities for Generating Triton Operators（`https://arxiv.org/abs/2502.14752v1`） | `arXiv:2502.14752v1` | 正文、表1–6、附录与四构造prompt；本页主要§3–4与附录B。原件表3个别Call/Execution值冲突保留，未核完整任务/harness、训练或A100运行。 |
| FastKernels: Benchmarking GPU Kernel Generation in Production（`https://arxiv.org/abs/2605.23215v1`） | `arXiv:2605.23215v1` | 主文与必要附录A–G，主要§3–6与附录B/C/D；关键Figure1–4/Table1–8核读。原件正确性单调性与CI非重叠表述不采用，条件速度和不同实验分别解释；未核全部HF映射、运行框架或完整阈值清单。 |
| FlashInfer-Bench: Building the Virtuous Cycle for AI-driven LLM Systems（`https://arxiv.org/abs/2601.00227v1`） | `arXiv:2601.00227v1` | 正文、附录Trace/生成代码/提示及Figure1–8；本页用§3.1–3.5、§4.2与Figure8。图文端到端性能、算法最后候选及缺省边界未补成保证；未核外部workload/solution库、dispatcher或GPU。 |
| KernelBenchX: A Comprehensive Benchmark for Evaluating LLM-Generated GPU Kernels（`https://arxiv.org/abs/2605.04956v2`） | `arXiv:2605.04956v2` | 正文、附录A–E及Figure1–5/Table1–6；本页用§3、附录A/C的门与聚合。方法名、分类/迭代图表、环境和do_bench次数声明有未闭合差异，不取作精确排名；未核完整任务、harness或设备运行。 |
