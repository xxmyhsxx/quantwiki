# 已收录资料的分批 ingest 计划

日期：2026-10-11。排期基线：本地提交 `1a63223`。

状态（更新于2026-10-14）：2026-10-13 全库重复性维护完成后，按用户确认恢复逐项 ingest；第10批FlexRound、SignRound、LRQ、LoRA四篇选定论文的核心机制、公共连接和实际研读范围已写入 Wiki。第1–9批P1已按既定材料和选定路径完成研读、整合与独立review，包含第2–9批35个队列条目及第1批AQLM接续。实际范围和未核分支由READING维护；条件事项仍保留。第10批选定论文范围完成，未核代码与运行条件保留；第11批PACT、LSQ+与LQ-Nets三篇选定论文范围完成，任务代理、初始化及结构化码值/交替拟合的区别已整合；未核代码与运行条件保留。第12批FlashAttention原论文v2的分块、IO与反向重计算已整合；FlashAttention-2的累计/序列并行/warp分工和原件边界已接续，固定代码的所选普通CUDA前后向已接续，其他分支/运行未核范围保留；FlashInfer计时文档的后端与样本口径已接续；FP4量化产物的尺度参数与后端声明已接续；FP8块缩放GEMM的NT输入与尺度存储已接续；BF16×FP4的权重准备/同后端消费与alpha折叠已接续；FP4 GEMM的后端准备/全局恢复与分派已接续；recursive state的完整LSE/子集条件与单位边界已接续；single decode的普通读取/返回与LSE边界已接续；single prefill的mask、按head尺度与NVFP4读取边界已接续；FlashInfer v0.6.17发布声明的FP4/专家/MLA与Ulysses版本边界已接续，第12批选定来源范围完成；第13批DistServe v3的阶段分离、KV转移与容量/SLO口径已接续；Sarathi-Serve v3的chunk预算、混合轮次/PP气泡与token级TBT容量条件已接续；SGLang v2的结构化程序、Radix/离线DFS、FSM/重新分词、API文本复用及IR/程序计量已接续，历史实现与运行未核；nano-vllm固定离线执行链、完整块/尾块、非混合调度、Qwen3/采样与TP/Graph条件已接续，模型/设备运行未核；vLLM v0.26.0发布说明的配置对象、加载峰值和组合边界已接续，PR/新组合实现与运行未核；vLLM V1官方架构的进程拓扑/客户端、统一配置与初始化分片已接续，文档快照和执行源码分别登记；KV管理源码解读文章的watermark接纳与CPU空闲队列已对照568afb源码接续，原文预算/长度及版本差异保留；Dynamo架构的请求/容量/缓存责任、backend-specific KV交接、阶段池/滚动更新与版本/性能边界已接续；Ray Serve的两级路由、配置/placement、DP组计数/生命周期和decode侧P/D协调已接续，第13批九项选定来源按实际范围完成；第14批ModelOpt的对象/算法、尺度/覆盖、搜索预算/状态、导出与所选静态FP8消费已接续，consumer/设备/实现边界保留；TensorRT-LLM的PyTorch加载、KV组合/存储层及硬件尺度条件已接续；FSDP2的参数生命周期、预取/dtype和full state协作已接续，第14批三项选定来源范围完成；第15批Llama 3 v3的配置/逻辑KV、选择性FP8及质量/资源条件已接续；DeepSeek-V3 v2训练精度/缓存/通信、V3路由与MTP已接续；R1 v2的GRPO/验证奖励、checkpoint与监督数据和蒸馏/抽样预算已接续，第15批三项选定来源范围完成；第16批Qwen2.5-VL v1的动态分辨率/帧率、窗口/合并与MRoPE时间坐标及预算条件已接续；Qwen2.5-Omni v1的TMRoPE/表示、Thinker/Talker和codec窗口、训练/评测边界已接续；Qwen3 v1的稠密/MoE配置、模式/预算与两阶段蒸馏及质量条件已接续；Qwen3-VL v2的多层视觉输入、位置/文字时间戳及训练/评测条件已接续；第16批四报告范围完成，已归档transformers前三模型所选输入/位置、生成与缓存/路由主线已接续；官方Omni工具、demo消费及低显存卸载/分块/精度路径按固定范围接续，外部依赖与运行条件保留；官方Qwen3-VL工具、示例和微调调用按选定路径接续；第16批七份已收录来源的选定范围完成，匹配HF内部实现仍按第5节条件项接续。第17批Attention Is All You Need的三attention、post-LN/位置及训练评测条件已接续；Formal Algorithms的行列/softmax轴、mask-token目标与整段推理及原件边界已接续；GQA的组内K/V转换与继续训练、质量/TPU计时条件已接续；MQA的decode访问/操作模型、batch与共享K/V作用及实验配平/摊销已接续；RMSNorm统计、正尺度/梯度、ε与部分统计条件和训练证据已接续，第17批五项选定来源范围完成；下一项第18批CUDA指南目录及Programming Model。2026-10-13用户授权持续完成当时剩余88条全部队列，按批研读、review、检查、提交与push；当前状态以READING为准（剩49条），不等待逐项继续指令。

## 1. 目标、范围与依据

将已收录资料按知识问题分批研读，补足核心机制、必要前置解释和实现边界。成果应进入可独立阅读的 Wiki；已有解释充分时，核对后登记实际范围和“无需新增”的判断。

本计划覆盖 [READING 的待核队列](../../wiki/READING.md#待逐项核查与-ingest-的已收录材料) 中的 130 个条目：57 篇论文、53 份文章或文档、20 个代码条目。第 1 批接续已登记的 AQLM 实现范围，另计于这 130 个条目之外。队列登记状态不等于从未阅读或知识完全缺失，执行前仍需核对已有正文。

另有 59 条 [已使用、研读范围待核实的记录](../../wiki/READING.md#已使用研读范围待核实)：37 篇论文、6 份文档、16 个代码版本。它们随相关批次核实实际范围，不并入 130 条待核队列。

排期依据是 [SELF 的当前画像和优先方向](../../SELF.md#6-下一步建设与维护)、READING 的确定版本与范围记录，以及已读的 PTQ/QAT、混合精度、低比特表示、校准和质量评测正文。各批的理由是拟核查的问题与预期知识增量，正式 ingest 时还需阅读主材料并复核覆盖。

## 2. 优先级、执行顺序与完成条件

第 1–9 批为当前排期 P1，第 10–25 批为 P2。优先级依据当前知识增量；编号表示默认执行顺序。必要的基础文档、原始论文或实现依赖可前移，仍在原分配批次中登记处理结果。前面批次完成后，依据剩余缺口复评后续顺序。

批次是主题分组，执行时逐篇论文或逐条有明确范围的文档、代码路径推进。每项完成深入研读、覆盖判断、增量整合或“无需新增”确认、独立内容 review、必要修正和本地提交后，再接续下一项。材料较多的批次可按表中列出的接口或问题拆成工作单元。

论文与代码同批表示围绕同一问题核查。开始代码阅读前，先确认固定版本、模型、配置、格式和接口是否对应；无法建立对应关系时，分别登记证据范围。来源归档、静态代码阅读、编译、模型运行及性能测量分别说明。

每项完成应有以下结果：

1. 讲清核心问题、机制、必要公式或推理、关键条件及证据边界，落实必要的公共知识承接。
2. 核对来源身份、实际阅读范围、正文引用和受影响链接，记录仍未核查的分支。
3. 完成独立内容 review，修正影响本项解释的必修问题；机械检查只说明其实际覆盖范围。
4. 按实质变化更新 READING、INDEX、SELF 和 LOG；无需变更的文件保留。用明确文件清单保存本地 Git 提交，推送按用户授权单独执行。

本计划安排知识研读与整理。模型训练、GPU 实验、服务验证和性能测量另按具体运行任务开展。需要另找材料补主论文与 Wiki 都未交代的知识时，按 [wiki-ingest](../../.agents/skills/wiki-ingest/SKILL.md) 第 5 步确认待补材料的类型与范围，再获取。

## 3. 优先批次：P1

| 批次 | 范围 | 理由与拟补增量 |
| --- | --- | --- |
| 1．AQLM 实现接续 | 同一固定快照的原始激活感知校准、格式转换、导出与推理读取主链 | PV 训练主链已有承接，初始量化权重与推理产物之间仍有未核范围。接续选定路径，补初始化、训练产物与部署表示的关系。 |
| 2．LC-QAT | LC-QAT | 延续现有码本与 PV-Tuning 知识，核清线性约束向量表示、初始化、训练变量与数据效率证据。单篇成批，保留核心论文的研读深度。 |
| 3．训练目标与分布校正 | QUASAR、Norm Tweaking | 比较训练中的量化重构目标与量化后分布校正，补清优化变量、误差位置和修复对象，接入 PTQ/QAT、校准和重构正文。 |
| 4．混合精度配置选择 | CMPQ、CoopQ、LLM-MQ、ResQ；ResQ 代码 | 接续现有预算与敏感性框架，补通道分配、层间交互和子空间保护的具体选择机制，核对配置与物理存储、执行成本的关系。 |
| 5．量化与硬件执行 | Atom、FlexQ、SharQ、Ideogram 4.0 INT8/GGUF PTQ；Atom、FlexQ、SharQ 代码 | 围绕不同位宽、表示和计算路径的配合，连接算法、打包、内核与运行条件。分别保留硬件和评测协议，不预设统一加速结论。 |
| 6．查表与低比特表示的实现 | FluxBin；QuIP、QuIP#、SpQR 已归档代码 | 现有码本、旋转和稀疏例外解释已有承接，源码覆盖不均。核查表、表示恢复和例外数据的消费路径；三个仓库分别接续其已有论文。 |
| 7．缩放与模态／专家校准 | QSLAW、VEQ、SmoothQuant+；QSLAW 代码；MASQuant、QIG 解读文章 | 区分尺度学习、模态差异与专家贡献的影响。正文与代码承担核心依据，解读文章用于补充核对，接入多模态比较与校准正文。 |
| 8．多模态低比特 KV | CalibQuant、Plug-and-Play 1.x-Bit KV；FlashInfer 分页写入、KV 布局、NVFP4 KV 文档 | 接续 KV 误差与生命周期知识，补多模态保护、校准和存储接口。FlashInfer 文档作为接口参照，分别核对布局与元数据条件，不预设其直接实现两篇论文的方法。 |
| 9．质量评测协议 | HELM、MMLU-Pro、LongBench v2、RULER；lm-evaluation-harness 代码 | 补任务定义、上下文长度、计分和聚合口径，使量化后的能力变化与方法比较有可检查的协议。 |

优先承接入口：[AQLM PV 实现](../../wiki/implementation/aqlm-pv-training-and-code-updates.md)、[PTQ/QAT 与代理梯度](../../wiki/fundamentals/quantization/post-training-and-quantization-aware-training.md)、[混合精度分配](../../wiki/theory/mixed-precision-allocation.md)、[低比特表示设计](../../wiki/research/low-bit-representation-design.md)、[校准与范围选择](../../wiki/theory/calibration-and-range-selection.md)、[KV 对象与粒度](../../wiki/theory/kv-cache-quantization-objects-and-granularity.md)、[模型质量评测](../../wiki/implementation/model-quality-evaluation.md)。这些是实施前需复核的现有落点，不预设每份资料都要新建页面。

## 4. 后续批次：P2

| 批次 | 范围 | 理由与拟补增量 |
| --- | --- | --- |
| 10．舍入、缩放与低秩适配 | FlexRound、SignRound、LRQ、LoRA | 区分舍入变量、缩放矩阵与适配增量，补优化目标和参数预算之间的关系。 |
| 11．可学习量化参数基础 | PACT、LSQ+、LQ-Nets | 补裁剪、偏移和量化器参数学习的原始依据，与现有 LSQ、STE 知识衔接。 |
| 12．Attention 与 FlashInfer 接口 | FlashAttention、FlashAttention-2；flash-attention 代码；第 8 批之外的 9 份 FlashInfer 文档 | 接续 Attention 的计算与 IO 解释，核对量化、GEMM、状态合并、prefill/decode 和计时接口。按 Attention 与低精度算子分段处理。 |
| 13．服务调度与阶段分离 | DistServe、Sarathi-Serve、SGLang；nano-vllm；vLLM 版本、架构及 KV 管理文章，Dynamo、Ray Serve 文档 | 将量化收益放回服务场景，解释批处理、缓存、阶段调度与吞吐／延迟的关系。 |
| 14．量化工具与分布式接入 | ModelOpt PTQ、TensorRT-LLM 量化、PyTorch Distributed/FSDP2 文档 | 补量化产物、框架支持与分布式状态的接口条件，接续前面的方法与部署分析。 |
| 15．Llama／DeepSeek 模型报告 | Llama 3、DeepSeek-V3、DeepSeek-R1 | 核对模型结构、数值与训练行为的原始依据，判断它们怎样影响量化对象和评测条件。 |
| 16．Qwen 多模态与模型实现 | Qwen2.5-VL、Qwen2.5-Omni、Qwen3、Qwen3-VL；两个 Qwen 仓库及已归档 transformers | 连接视觉、音频、语言和专家模块的路径。归档 transformers 4.53.2 可核前三个模型；Qwen3-VL 仓库可先核工具、示例与微调调用，完整模型内部主链须取得匹配实现后接续。 |
| 17．Transformer 原始基础 | Attention Is All You Need、Formal Algorithms for Transformers、GQA、MQA、RMSNorm | 核对已有正文与原始依据，澄清结构、计算和适用范围；发现实质缺口再补写。 |
| 18．CUDA 编程与 Softmax 实践 | 8 份 CUDA 指南／技术文章／Softmax worklog；cudacodes | 接续线程、访存、同步和归约知识，用具体算子解释它们的作用。 |
| 19．CUTLASS 与 GEMM 设计 | 11 份 CUTLASS／CuTe／H100 GEMM 文档与文章；fast.cu | 补布局、Tile、流水线、调度与硬件条件。按布局和计算流水线分段处理。 |
| 20．Triton 算子设计 | 5 份 Triton 教程；Triton-Puzzles | 从已有语言语义继续进入矩阵乘、Attention、分组与持久化调度，建立可读实现例子。 |
| 21．系统课程材料 | 系统软件研讨课、GPU MODE 讲义、IIT Bombay、Microsoft AI-System、LeetGPU 挑战 | 补跨层系统背景。逐项按实际可读内容登记范围，课程入口与视频简介保留其材料限制。 |
| 22．Kernel 生成与反馈 | CudaForge、KernelBench、TritonBench | 核生成任务、正确性判定、性能反馈和优化循环，补自动 Kernel 优化的解释依据。 |
| 23．Kernel 评测覆盖 | FastKernels、KernelBenchX、FlashInfer-Bench | 比较 workload、评测协议与生产条件，补清基准得分可以支持哪些判断。 |
| 24．视觉 token 与 VLA 执行 | vla.simd、MWOP、Query Independent Variable Rate Visual Token Coding、CoVisco；vla.simd、CoVisco 作者仓库，TensorRT-Edge-LLM 部署仓库及两份模型／部署文档 | 接续现有 VLA、视觉 token 和剪枝知识。作者仓库分别接续对应论文；TensorRT-Edge-LLM 的 DART 路径作为独立部署参照，核清算法、支持矩阵与执行成本。 |
| 25．其他多模态案例 | SpikeMLLM、医疗场景 MLLM 压缩 | 扩展时间编码与领域任务，重点核对新增机制、评价对象和适用条件。 |

较大批次按以下范围推进，每个工作单元内仍逐项完成研读、整合与 review。其余批次以单篇论文及必要实现路径、或单份有明确问题的文档为工作单元。

| 批次 | 建议工作单元 |
| --- | --- |
| 1 | 原始激活感知校准；格式转换与导出；选定推理读取路径。 |
| 4–7 | 每篇方法论文先补核心解释，再核对应的固定代码路径；文章与其他方法的代码分别登记依据。 |
| 8 | 两篇 KV 方法分别研读；FlashInfer 的分页、布局与 NVFP4 接口分别核对。 |
| 9 | 四篇评测论文分别核任务与计分；harness 只选对应任务及共用执行、聚合入口，不默认覆盖所有基准。 |
| 12 | FlashAttention 论文与所选实现；FlashInfer prefill/decode、状态合并；低精度量化、GEMM 与计时接口。 |
| 13 | 三篇调度论文分别研读；nano-vllm 所选主链；框架架构、KV 管理与服务文档按接口接续。 |
| 16 | 各模型报告；归档版本具备的模型实现；两个 Qwen 仓库的调用路径。Qwen3-VL 内部实现作为条件接续项。 |
| 18–19 | CUDA 的执行、访存与同步；Softmax 算例；CuTe 布局；GEMM Tile、流水线与调度。代码各选一条相关算子路径。 |
| 24 | 四篇视觉／VLA 论文分别研读；vla.simd 和 CoVisco 对应实现；TensorRT-Edge-LLM 的 DART 与支持矩阵。 |

## 5. 条件事项与历史记录

OnPTQ 的匹配参数更新配置、WUSH-KV 作者改版及一般可逆变换接入继续作为 P1 条件事项。SAPrune、CentriQ、QuantForge、XOR-Trellis 等已整理论文的匹配作者实现，也按实际材料接续。材料具备后插入相应主题批次，并记录版本与可核范围；搜索未命中只说明当前查找结果。

第 16 批的完整 Qwen3-VL 模型实现需补匹配材料。[归档 transformers 的身份记录](../../raw/repositories/2026-09-21/transformers/README.md)是 `hf-4.53.2`，没有 `qwen3_vl` 模型目录；Qwen3-VL 固定仓库的 Quickstart（`../../raw/repositories/2026-09-21/qwen3-vl/source/README.md.txt#quickstart`）要求 `transformers >= 4.57.0`。先完成现有可读范围，再按待补材料规则确认版本和范围，不能用同批的旧版本替代。

59 条历史范围待核记录随相应主题核实。来源已使用与完整研读分别判断；无法核实的范围继续保留未知。模型运行和性能验证另按具体任务登记。

READING 维护资料的实际处理状态，SELF 维护当前画像与优先方向，本文件保留批次和排期理由。发生实质范围或顺序变化时更新本计划；不另外复制逐项进度表。

## 6. 每批材料与固定身份

以下清单是 `1a63223` 中 130 个队列条目的分配快照。论文版本、文档快照和代码 commit 沿用 READING 登记；不将代码条目默认认定为配套论文的完整实现。实际内容和材料完整性在对应 ingest 中核验。

第 1 批的代码来源为已登记的 [Vahe1994/AQLM](https://github.com/Vahe1994/AQLM/tree/e79a896ed6656fe4ed06193d42d004e7d0bbdbb2)，commit `e79a896ed6656fe4ed06193d42d004e7d0bbdbb2`。已核与未核范围见 [AQLM PV 实现](../../wiki/implementation/aqlm-pv-training-and-code-updates.md)，不计入下表的 130 条。

| 批次 | 类型 | 来源 | 固定身份 |
| --- | --- | --- | --- |
| 2 | 论文 | [LC-QAT: Data-Efficient 2-Bit QAT for LLMs via Linear-Constrained Vector Quantization](https://arxiv.org/abs/2606.10531v1) | `arXiv:2606.10531v1` |
| 3 | 论文 | [QUASAR: Lowering the Loss Floor of Quantization-Aware Training with Loss-Aware Reconstruction](https://arxiv.org/abs/2608.13966v1) | `arXiv:2608.13966v1` |
| 3 | 论文 | [Norm Tweaking: High-performance Low-bit Quantization of Large Language Models](https://arxiv.org/abs/2309.02784v2) | `arXiv:2309.02784v2` |
| 4 | 论文 | [Channel-Wise Mixed-Precision Quantization for Large Language Models](https://arxiv.org/abs/2410.13056v4) | `arXiv:2410.13056v4` |
| 4 | 论文 | [CoopQ: Cooperative Game Inspired Layerwise Mixed Precision Quantization for LLMs](https://arxiv.org/abs/2509.15455v2) | `arXiv:2509.15455v2` |
| 4 | 论文 | [LLM-MQ: Mixed-Precision Quantization for Efficient LLM Deployment](https://neurips2023-enlsp.github.io/papers/paper_4.pdf) | `snapshot-sha256-9a2101ae6befaff5566fc1ed1bd78cbf10ccb4a7e2c3ae2b985e9c3845562dc0` |
| 4 | 论文 | [ResQ: Mixed-Precision Quantization of Large Language Models with Low-Rank Residuals](https://arxiv.org/abs/2412.14363v2) | `arXiv:2412.14363v2` |
| 4 | 代码 | [utkarsh-dmx/project-resq](https://github.com/utkarsh-dmx/project-resq/tree/cb87d0d059455858daaa66f7410da4afebee978a) | `cb87d0d059455858daaa66f7410da4afebee978a` |
| 5 | 论文 | [Atom: Low-bit Quantization for Efficient and Accurate LLM Serving](https://arxiv.org/abs/2310.19102v3) | `arXiv:2310.19102v3` |
| 5 | 论文 | [FlexQ: Efficient Post-training INT6 Quantization for LLM Serving via Algorithm-System Co-Design](https://arxiv.org/abs/2508.04405v2) | `arXiv:2508.04405v2` |
| 5 | 论文 | [Holding the FP8 Quality Ceiling at 8-Bit Weights and Activations: INT8 and GGUF Post-Training Quantization of Ideogram 4.0 for Consumer GPUs](https://arxiv.org/abs/2606.12280v2) | `arXiv:2606.12280v2` |
| 5 | 论文 | [SharQ: Bridging Activation Sparsity and FP4 Quantization for LLM Inference](https://arxiv.org/abs/2606.26587v1) | `arXiv:2606.26587v1` |
| 5 | 代码 | [efeslab/Atom](https://github.com/efeslab/Atom/tree/7e3618b1a7a7c86e1c93cc909b1510c046d76ac6) | `7e3618b1a7a7c86e1c93cc909b1510c046d76ac6` |
| 5 | 代码 | [FlyFoxPlayer/FlexQ](https://github.com/FlyFoxPlayer/FlexQ/tree/a42424d843adffebe4b28ddb62a760cb52b0d2f0) | `a42424d843adffebe4b28ddb62a760cb52b0d2f0` |
| 5 | 代码 | [actypedef/SharQ](https://github.com/actypedef/SharQ/tree/3cc584fe48e32e59c8b4e37323470dcdd606bb11) | `3cc584fe48e32e59c8b4e37323470dcdd606bb11` |
| 6 | 论文 | [FluxBin: Flexible LUT-based Ultra-low-bit LLM Inference by Algorithm-Kernel Synergy](https://arxiv.org/abs/2608.15602v1) | `arXiv:2608.15602v1` |
| 6 | 代码 | [Cornell-RelaxML/QuIP](https://github.com/Cornell-RelaxML/QuIP/tree/ac92cfc7a22f6100009e2caf53bb72257d3f3184) | `ac92cfc7a22f6100009e2caf53bb72257d3f3184` |
| 6 | 代码 | [Cornell-RelaxML/quip-sharp](https://github.com/Cornell-RelaxML/quip-sharp/tree/1d8f873e9a2a8b86b12bb1064c312c5689b77d98) | `1d8f873e9a2a8b86b12bb1064c312c5689b77d98` |
| 6 | 代码 | [Vahe1994/SpQR](https://github.com/Vahe1994/SpQR/tree/543d10e56f921e5eeb10d5f0aaaccd30b6662d0e) | `543d10e56f921e5eeb10d5f0aaaccd30b6662d0e` |
| 7 | 论文 | [Advancing Multimodal Large Language Models with Quantization-Aware Scale Learning for Efficient Adaptation](https://arxiv.org/abs/2408.03735v1) | `arXiv:2408.03735v1` |
| 7 | 论文 | [SmoothQuant+: Accurate and Efficient 4-bit Post-Training WeightQuantization for LLM](https://arxiv.org/abs/2312.03788v1) | `arXiv:2312.03788v1` |
| 7 | 论文 | [VEQ: Modality-Adaptive Quantization for MoE Vision-Language Models](https://arxiv.org/abs/2602.01037v1) | `arXiv:2602.01037v1` |
| 7 | 代码 | [xjjxmu/QSLAW](https://github.com/xjjxmu/QSLAW/tree/264dfb081c210a8bab6a3fb5707d827ade7b6802) | `264dfb081c210a8bab6a3fb5707d827ade7b6802` |
| 7 | 文档 | [(CVPR 2026) 从 SmoothQuant 到 MASQuant：多模态量化的新思路](https://mp.weixin.qq.com/s/CQhaOfG9b2PsxVO04pHe-Q?from=singlemessage&isappinstalled=0&scene=1&clicktime=1782831938&enterid=1782831938) | `snapshot-2026-06-30` |
| 7 | 文档 | [CVPR 2026｜从模态均衡到 token 敏感性建模：一文读懂 QIG 如何提升 LVLM 量化精度](https://mp.weixin.qq.com/s/LaDDYRV_vKbCY910dFJ2cg?from=singlemessage&isappinstalled=0&scene=1&clicktime=1782831805&enterid=1782831805) | `snapshot-2026-06-30` |
| 8 | 论文 | [CalibQuant: 1-Bit KV Cache Quantization for Multimodal LLMs](https://arxiv.org/abs/2502.14882v2) | `arXiv:2502.14882v2` |
| 8 | 论文 | [Plug-and-Play 1.x-Bit KV Cache Quantization for Video Large Language Models](https://arxiv.org/abs/2503.16257v2) | `arXiv:2503.16257v2` |
| 8 | 文档 | [flashinfer.page.append_paged_kv_cache - FlashInfer 0.6.14 documentation](https://docs.flashinfer.ai/generated/flashinfer.page.append_paged_kv_cache.html) | `snapshot-2026-07-02` |
| 8 | 文档 | [KV-Cache Layout in FlashInfer - FlashInfer 0.6.14 documentation](https://docs.flashinfer.ai/tutorials/kv_layout.html) | `snapshot-2026-07-02` |
| 8 | 文档 | [flashinfer.quantization.nvfp4_quantize_paged_kv_cache - FlashInfer 0.6.14 documentation](https://docs.flashinfer.ai/generated/flashinfer.quantization.nvfp4_quantize_paged_kv_cache.html) | `snapshot-2026-07-02` |
| 9 | 论文 | [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110v2) | `arXiv:2211.09110v2` |
| 9 | 论文 | [LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks](https://arxiv.org/abs/2412.15204v2) | `arXiv:2412.15204v2` |
| 9 | 论文 | [MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark](https://arxiv.org/abs/2406.01574v6) | `arXiv:2406.01574v6` |
| 9 | 论文 | [RULER: What's the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654v3) | `arXiv:2404.06654v3` |
| 9 | 代码 | [Summer-Summer/lm-evaluation-harness](https://github.com/Summer-Summer/lm-evaluation-harness/tree/62617a8cbd7f3beddc2a9c029536e0e387976beb) | `62617a8cbd7f3beddc2a9c029536e0e387976beb` |
| 10 | 论文 | [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685v2) | `arXiv:2106.09685v2` |
| 10 | 论文 | [FlexRound: Learnable Rounding based on Element-wise Division for Post-Training Quantization](https://arxiv.org/abs/2306.00317v2) | `arXiv:2306.00317v2` |
| 10 | 论文 | [LRQ: Optimizing Post-Training Quantization for Large Language Models by Learning Low-Rank Weight-Scaling Matrices](https://arxiv.org/abs/2407.11534v2) | `arXiv:2407.11534v2` |
| 10 | 论文 | [Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs](https://arxiv.org/abs/2309.05516v5) | `arXiv:2309.05516v5` |
| 11 | 论文 | [LQ-Nets: Learned Quantization for Highly Accurate and Compact Deep Neural Networks](https://arxiv.org/abs/1807.10029v1) | `arXiv:1807.10029v1` |
| 11 | 论文 | [LSQ+: Improving low-bit quantization through learnable offsets and better initialization](https://arxiv.org/abs/2004.09576v1) | `arXiv:2004.09576v1` |
| 11 | 论文 | [PACT: Parameterized Clipping Activation for Quantized Neural Networks](https://arxiv.org/abs/1805.06085v2) | `arXiv:1805.06085v2` |
| 12 | 论文 | [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135v2) | `arXiv:2205.14135v2` |
| 12 | 论文 | [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691v1) | `arXiv:2307.08691v1` |
| 12 | 代码 | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention/tree/73c992c8ca746935548df620ef3c1b6238fe6e68) | `73c992c8ca746935548df620ef3c1b6238fe6e68` |
| 12 | 文档 | [flashinfer.testing.bench_gpu_time - FlashInfer 0.6.14 documentation](https://docs.flashinfer.ai/generated/flashinfer.testing.bench_gpu_time.html) | `snapshot-2026-07-02` |
| 12 | 文档 | [flashinfer.quantization.fp4_quantize - FlashInfer 0.6.14 documentation](https://docs.flashinfer.ai/generated/flashinfer.quantization.fp4_quantize.html) | `snapshot-2026-07-02` |
| 12 | 文档 | [flashinfer.gemm.gemm_fp8_nt_blockscaled - FlashInfer 0.6.14 documentation](https://docs.flashinfer.ai/generated/flashinfer.gemm.gemm_fp8_nt_blockscaled.html) | `snapshot-2026-07-02` |
| 12 | 文档 | [flashinfer.gemm.mm_bf16_fp4 - FlashInfer 0.6.14 documentation](https://docs.flashinfer.ai/generated/flashinfer.gemm.mm_bf16_fp4.html) | `snapshot-2026-07-02` |
| 12 | 文档 | [flashinfer.gemm.mm_fp4 - FlashInfer 0.6.14 documentation](https://docs.flashinfer.ai/generated/flashinfer.gemm.mm_fp4.html) | `snapshot-2026-07-02` |
| 12 | 文档 | [Attention States and Recursive Attention - FlashInfer 0.6.14 documentation](https://docs.flashinfer.ai/tutorials/recursive_attention.html) | `snapshot-2026-07-02` |
| 12 | 文档 | [Release Release v0.6.17 · flashinfer-ai/flashinfer · GitHub](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.6.17) | `snapshot-2026-08-20` |
| 12 | 文档 | [flashinfer.decode.single_decode_with_kv_cache - FlashInfer 0.6.14 documentation](https://docs.flashinfer.ai/generated/flashinfer.decode.single_decode_with_kv_cache.html) | `snapshot-2026-07-02` |
| 12 | 文档 | [flashinfer.prefill.single_prefill_with_kv_cache - FlashInfer 0.6.14 documentation](https://docs.flashinfer.ai/generated/flashinfer.prefill.single_prefill_with_kv_cache.html) | `snapshot-2026-07-02` |
| 13 | 论文 | [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670v3) | `arXiv:2401.09670v3` |
| 13 | 论文 | [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310v3) | `arXiv:2403.02310v3` |
| 13 | 论文 | [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104v2) | `arXiv:2312.07104v2` |
| 13 | 代码 | [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm/tree/bb823b3e06983d71485a8e1f23715ebd87d98ef8) | `bb823b3e06983d71485a8e1f23715ebd87d98ef8` |
| 13 | 文档 | [Release v0.26.0 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/releases/tag/v0.26.0) | `snapshot-2026-08-20` |
| 13 | 文档 | [NVIDIA Dynamo Architecture and Compatibility Documentation](https://docs.nvidia.com/dynamo/v1.4.2/knowledge-base/overview) | `v1.4.2` |
| 13 | 文档 | [Ray Serve LLM Architecture](https://docs.ray.io/en/v2.58.0/serve/llm/architecture/index.html) | `ray-2.58.0-commit-642b00f8c3c5a01cab7c2e2bff8725fa027b23b4` |
| 13 | 文档 | [vLLM V1 Architecture Overview](https://docs.vllm.ai/en/latest/design/arch_overview/) | `vllm-main-f9ad9dd6b495f7492fa975e0ac48164b090d99ad` |
| 13 | 文档 | [vllm v1 源码精读（三）：KV Cache 管理、Chunked Prefill 与异步架构](https://zhuanlan.zhihu.com/p/2060408485595050568) | `snapshot-2026-09-24; article states vLLM ba22152096b2484faa3579624a253d54804d876d (2026-07-06)` |
| 14 | 文档 | [ModelOpt Post-Training Quantization Guides](https://github.com/NVIDIA/Model-Optimizer/tree/f2f0d6958eb95374d816f3d6256a0be3738231a3) | `commit-f2f0d6958eb95374d816f3d6256a0be3738231a3` |
| 14 | 文档 | [PyTorch Distributed Tutorials and FSDP2](https://docs.pytorch.org/tutorials/distributed.html) | `PyTorch Tutorials 2.14.0+cu130; FSDP tutorial last verified 2024-11-05` |
| 14 | 文档 | [Quantization — TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html) | `source-commit-a6f4151497372f14a5261113a12be2b406e9b5fc` |
| 15 | 论文 | [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948v2) | `arXiv:2501.12948v2` |
| 15 | 论文 | [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437v2) | `arXiv:2412.19437v2` |
| 15 | 论文 | [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783v3) | `arXiv:2407.21783v3` |
| 16 | 论文 | [Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215v1) | `arXiv:2503.20215v1` |
| 16 | 论文 | [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923v1) | `arXiv:2502.13923v1` |
| 16 | 论文 | [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388v1) | `arXiv:2505.09388v1` |
| 16 | 论文 | [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631v2) | `arXiv:2511.21631v2` |
| 16 | 代码 | [QwenLM/Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni/tree/d8a31ca56c0456b6edfcbcbf4bdbb6ae2200ef42) | `d8a31ca56c0456b6edfcbcbf4bdbb6ae2200ef42` |
| 16 | 代码 | [QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL/tree/96588727e44c78b25ba03ea03b8e12f7e64fd0da) | `96588727e44c78b25ba03ea03b8e12f7e64fd0da` |
| 16 | 代码 | [Summer-Summer/transformers](https://github.com/Summer-Summer/transformers/tree/37f8b0b53512e6aae0cfd15746c133c101783178) | `37f8b0b53512e6aae0cfd15746c133c101783178` |
| 17 | 论文 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762v7) | `arXiv:1706.03762v7` |
| 17 | 论文 | [Formal Algorithms for Transformers](https://arxiv.org/abs/2207.09238v1) | `arXiv:2207.09238v1` |
| 17 | 论文 | [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245v3) | `arXiv:2305.13245v3` |
| 17 | 论文 | [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150v1) | `arXiv:1911.02150v1` |
| 17 | 论文 | [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467v1) | `arXiv:1910.07467v1` |
| 18 | 文档 | [CUDA Programming Guide — CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) | `snapshot-2026-07-02` |
| 18 | 文档 | [1.2. Programming Model — CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html) | `snapshot-2026-07-02` |
| 18 | 文档 | [2.4. Writing Tile Kernels — CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-tile-kernels.html) | `snapshot-2026-07-02` |
| 18 | 文档 | [CUDA Refresher: The CUDA Programming Model &#124; NVIDIA Technical Blog](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) | `snapshot-2026-07-02` |
| 18 | 文档 | [Using Shared Memory in CUDA C/C++ &#124; NVIDIA Technical Blog](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/) | `snapshot-2026-07-02` |
| 18 | 文档 | [Programming Tensor Cores in CUDA 9 &#124; NVIDIA Technical Blog](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/) | `snapshot-2026-07-02` |
| 18 | 文档 | [Using CUDA Warp-Level Primitives &#124; NVIDIA Technical Blog](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/) | `snapshot-2026-07-02` |
| 18 | 文档 | [Learning CUDA by optimizing softmax: A worklog](https://maharshi.bearblog.dev/optimizing-softmax-cuda/) | `indexed-text-snapshot-2026-09-24` |
| 18 | 代码 | [Maharshi-Pandya/cudacodes](https://github.com/Maharshi-Pandya/cudacodes/tree/9a459b1007c68430c801389f4eaf31cf31b906a6) | `9a459b1007c68430c801389f4eaf31cf31b906a6` |
| 19 | 文档 | [Blackwell SM100 GEMMs — NVIDIA CUTLASS Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html) | `snapshot-2026-07-02` |
| 19 | 文档 | [CuTe DSL — NVIDIA CUTLASS Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html) | `snapshot-2026-07-02` |
| 19 | 文档 | [Efficient GEMM in CUDA — NVIDIA CUTLASS Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html) | `snapshot-2026-07-02` |
| 19 | 文档 | [CUTLASS 3.0 GEMM API — NVIDIA CUTLASS Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html) | `snapshot-2026-07-02` |
| 19 | 文档 | [Grouped Kernel Schedulers — NVIDIA CUTLASS Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html) | `snapshot-2026-07-02` |
| 19 | 文档 | [Layouts and Tensors — NVIDIA CUTLASS Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/layout.html) | `snapshot-2026-07-02` |
| 19 | 文档 | [Quickstart — NVIDIA CUTLASS Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/quickstart.html) | `snapshot-2026-07-02` |
| 19 | 文档 | [Synchronization primitives — NVIDIA CUTLASS Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/pipeline.html) | `snapshot-2026-07-02` |
| 19 | 文档 | [Tile Iterator Concepts — NVIDIA CUTLASS Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/tile_iterator_concept.html) | `snapshot-2026-07-02` |
| 19 | 文档 | [CUTLASS: Fast Linear Algebra in CUDA C++ &#124; NVIDIA Technical Blog](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/) | `snapshot-2026-07-02` |
| 19 | 文档 | [Outperforming cuBLAS on H100: a Worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog) | `html-body-snapshot-2026-09-24` |
| 19 | 代码 | [pranjalssh/fast.cu](https://github.com/pranjalssh/fast.cu/tree/2dfe5e26aecfd9e5f27bf9d5837deea01acda24b) | `2dfe5e26aecfd9e5f27bf9d5837deea01acda24b` |
| 20 | 文档 | [Fused Attention — Triton documentation](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) | `snapshot-2026-07-02` |
| 20 | 文档 | [Group GEMM — Triton documentation](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html) | `snapshot-2026-07-02` |
| 20 | 文档 | [Introduction — Triton documentation](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html) | `snapshot-2026-07-02` |
| 20 | 文档 | [Matrix Multiplication — Triton documentation](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) | `snapshot-2026-07-02` |
| 20 | 文档 | [Persistent Matmul — Triton documentation](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html) | `snapshot-2026-07-02` |
| 20 | 代码 | [gpu-mode/Triton-Puzzles](https://github.com/gpu-mode/Triton-Puzzles/tree/4d794abed02292081500ebf4b1e35cd5fc5fb019) | `4d794abed02292081500ebf4b1e35cd5fc5fb019` |
| 21 | 文档 | [【2026年更新版】大模型研讨课第九期丨大模型系统软件](https://www.bilibili.com/video/BV1R7jc6qEmn/) | `page-description-snapshot-2026-09-24; BV1R7jc6qEmn` |
| 21 | 文档 | [Supplementary Material for Lectures](https://github.com/gpu-mode/lectures/blob/77a8df418834e5789c12da23e7d2719e0efabef1/README.md) | `77a8df418834e5789c12da23e7d2719e0efabef1` |
| 21 | 文档 | [Systems for Machine Learning — IIT Bombay CS 794](https://www.cse.iitb.ac.in/~mythili/sysml/) | `Spring 2026 course archive; slides exported 2026-09-24` |
| 21 | 文档 | [GPT-2 (120M) Transformer Block challenge announcement](https://x.com/LeetGPU/status/2028270254863090114) | `public-index-post-snapshot-2026-09-24` |
| 21 | 文档 | [System for AI — Microsoft AI-System Course](https://microsoft.github.io/AI-System/) | `repository-commit-2a974e1a483eb5c2f714ba5001ac33d8c58998d5` |
| 22 | 论文 | [CudaForge: An Agent Framework with Hardware Feedback for CUDA Kernel Optimization](https://arxiv.org/abs/2511.01884v2) | `arXiv:2511.01884v2` |
| 22 | 论文 | [KernelBench: Can LLMs Write Efficient GPU Kernels?](https://arxiv.org/abs/2502.10517v1) | `arXiv:2502.10517v1` |
| 22 | 论文 | [TritonBench: Benchmarking Large Language Model Capabilities for Generating Triton Operators](https://arxiv.org/abs/2502.14752v1) | `arXiv:2502.14752v1` |
| 23 | 论文 | [FastKernels: Benchmarking GPU Kernel Generation in Production](https://arxiv.org/abs/2605.23215v1) | `arXiv:2605.23215v1` |
| 23 | 论文 | [FlashInfer-Bench: Building the Virtuous Cycle for AI-driven LLM Systems](https://arxiv.org/abs/2601.00227v1) | `arXiv:2601.00227v1` |
| 23 | 论文 | [KernelBenchX: A Comprehensive Benchmark for Evaluating LLM-Generated GPU Kernels](https://arxiv.org/abs/2605.04956v2) | `arXiv:2605.04956v2` |
| 24 | 论文 | [vla.simd: Efficient CPU Inference for Language-Conditioned Manipulation](https://arxiv.org/abs/2609.24274v1) | `arXiv:2609.24274v1` |
| 24 | 论文 | [MWOP: Modality-aware Width-wise Operation Pruning for Efficient MLLMs](https://arxiv.org/abs/2610.01434v1) | `arXiv:2610.01434v1` |
| 24 | 论文 | [Query Independent Variable Rate Visual Token Coding](https://arxiv.org/abs/2610.00204v1) | `arXiv:2610.00204v1` |
| 24 | 论文 | [CoVisco: Codec-Native Vision Encoder with Native Token Compression for Unified Image-Video Understanding](https://arxiv.org/abs/2609.39924v1) | `arXiv:2609.39924v1` |
| 24 | 代码 | [cair-vinuni/vla.simd](https://github.com/cair-vinuni/vla.simd/tree/7354b6d73b3686395d709202dbb519fd2e101aa0) | `7354b6d73b3686395d709202dbb519fd2e101aa0` |
| 24 | 代码 | [ernie-research/CoVisco](https://github.com/ernie-research/CoVisco/tree/604cd70d53afc39506682d423c07604a38d37b40) | `604cd70d53afc39506682d423c07604a38d37b40` |
| 24 | 代码 | [NVIDIA/TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM/tree/e649dadcbbf049fa4caf2f09f3a3275ca56751db) | `e649dadcbbf049fa4caf2f09f3a3275ca56751db` |
| 24 | 文档 | [vla.simd model bundle](https://huggingface.co/collections/khanhnd61/vlasimd-model-bundle) | `2026-09-28 page snapshot` |
| 24 | 文档 | [TensorRT Edge-LLM: Visual-Token Pruning (DART) and Official Support Matrix](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/features/visual-token-pruning.html) | `0.11.0; docs-build-e649dadcbbf049fa4caf2f09f3a3275ca56751db; snapshot-2026-10-05` |
| 25 | 论文 | [Compression Strategies for Efficient Multimodal LLMs in Medical Contexts](https://arxiv.org/abs/2507.21976v3) | `arXiv:2507.21976v3` |
| 25 | 论文 | [SpikeMLLM: Spike-based Multimodal Large Language Models via Modality-Specific Temporal Scales and Temporal Compression](https://arxiv.org/abs/2604.18610v1) | `arXiv:2604.18610v1` |

## 7. 计划评审

2026-10-11 完成计划自查、独立内容评审与修订复核。独立评审未发现 P1；发现的 1 项 P2 必修问题已修正，修订复核范围内没有剩余 P1/P2 必修项。

已确认的结果：

- 130 条材料与 READING 的标题、类型、来源链接和固定身份逐项一致：57 篇论文、53 份文档、20 个代码条目，无遗漏、重复或版本替换。第 1 批 AQLM 接续另计，59 条历史范围待核记录保持单独口径。
- P1/P2 与默认执行顺序有当前 SELF 和已有正文覆盖依据；必要前置依赖可前移，完成后按实际缺口复评。批次分组不表示论文与同批代码已匹配，也不表示对应知识全部缺失。
- 完成条件涵盖核心解释、证据与研读范围、独立 review、修正、受影响入口维护和本地提交；模型与性能验证保持实际证据范围。

发现与修订：

| 事项 | 影响与处理 |
| --- | --- |
| 第 16 批 Qwen3-VL 的匹配实现缺口（P2 必修） | 归档 transformers 4.53.2 无 Qwen3-VL 模型目录，固定 Qwen3-VL 仓库要求至少 4.57.0。已区分现有可核路径与完整内部模型路径，后者列入第 5 节条件事项；附录身份保持不变。 |
| 第 8、24 批的材料关系 | 已明确 FlashInfer 是接口参照；vla.simd、CoVisco 代码分别对应其论文；TensorRT-Edge-LLM 的 DART 是独立部署参照，不作为同批其他方法的作者实现。 |
| 较大批次的执行范围 | 保留 25 个主题批次，补具体工作单元；逐篇、逐接口或所选代码路径推进，harness 不默认覆盖所有基准。 |

机械检查已确认全部来源分配与固定版本、25 个批次编号、Markdown 表格结构、受影响本地链接与锚点，以及提交范围。

本次评审覆盖计划全文、SELF 新入口、READING 全量身份分配，并定向核对关键 Wiki 落点和固定来源。没有逐篇完成 57 篇论文的科学内容审查，没有全面核查代码分支、编译、模型／GPU 运行或性能复现。本次计划整理不改变 READING 的 ingest 状态。
