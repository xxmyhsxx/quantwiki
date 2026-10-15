# Wiki 的定位、职责与当前画像

本文件说明知识库的目的、范围、工作边界、已有能力与下一步。项目介绍见 [README](README.md)，阅读入口见 [知识地图](wiki/INDEX.md)。

快速查看：[当前知识画像](#5-当前知识画像) · [下一步建设与维护](#6-下一步建设与维护)。

## 1. 定位与核心目的

这是面向大语言模型和多模态模型的量化知识库，当前多模态主线以视觉语言模型为重点。它连接量化原理、模型计算、实现与部署，服务系统学习、研究判断和工程理解，并通过当前画像指导后续建设。

主要成果是能够独立阅读、追溯来源的知识页面。建设价值看内容能否帮助理解和判断问题。

## 2. 知识范围与支持来源

主线包括权重、激活与 KV cache 量化，PTQ/QAT，重构、变换、旋转、码本、补偿、敏感性、混合精度和多模态量化。模型结构、数学、数值格式、GPU、推理系统和评测知识按理解需要建设。经典 CNN、视觉 token 剪枝、TTS/VLA 等相关专题按需补充，不追求通用 AI 百科式覆盖。

| 来源 | 包括什么 | 用来做什么 |
| --- | --- | --- |
| 论文与研究报告 | 量化方法、基础与综述、模型结构、系统与算子、数值格式、评测分析；含正式论文、预印本和技术报告 | 理解核心问题、机制、推导、条件与方法联系 |
| 代码 | 算法与模型实现、量化及导出工具、推理框架、底层算子；如 vLLM、SGLang、TensorRT-LLM、llama.cpp、CUDA、Triton、CUTLASS 等 | 核对实际步骤、数据表示、接口与执行路径 |
| 官方文档、标准、教材与课程 | API、硬件与数值规范、数学和系统基础、教学代码 | 补足定义、前置知识与实现语义 |
| 技术文章与公开讨论 | 工程文章、源码解读、公开评议、作者回应、issue/PR 讨论 | 补充解释线索、版本背景与分歧，保留观点归属 |

来源按相关性、知识增量和可追溯性选择，优先补现有解释缺口。代码围绕具体问题读取，保留版本与核读范围；来源示例不表示全功能覆盖。

## 3. 工作职责与边界

- **资料收录：** 保存可追溯来源，为研读提供材料；保存资料与形成 Wiki 知识分别说明。
- **Ingest 与公共知识建设：** 讲清核心问题、观点、机制、必要推导和条件，将增量整合到已有或合适的新页面，连接可复用知识。
- **源码分析：** 核对关键变量、流程、格式和接口，解释实现与算法的关系。
- **Review 与维护：** 检查理解、表达、依据、解释深度和跨页联系，修正已确认问题。
- **查询与研究支持：** 定位知识、解释问题、比较路线，服务明确的研究目标。

一次 ingest 是否完成，取决于核心解释、必要依赖、来源条件和知识关联是否落实；已有内容充分时，无需新增页面。

**论文研读和 review 聚焦核心观点及我们的整理质量，不为纠正论文数据开展具体实验。** 作者实验按需要引用，保留归属与关键条件。整理者推导、教学例子、源码分析和实际运行分别说明。

Kernel 开发、模型运行、部署验证和性能测量按用户明确提出的具体任务开展。缺少实验复现不构成知识整理未完成。

## 4. 知识组织与文件职责

`raw/` 按来源保存材料，`wiki/` 按知识对象与问题组织解释。一份来源可以更新多页，一页也可以综合多个来源。

- [README](README.md) 介绍项目和阅读入口；SELF 说明职责、边界、画像与方向；[AGENTS](AGENTS.md) 约定 Agent 协作方式。
- [INDEX](wiki/INDEX.md) 提供知识导航；[Wiki 规范](wiki/README.md) 规定页面组织、表达与引用；[raw 规范](raw/README.md) 规定来源保存与版本管理。
- [Skills](.agents/skills/) 承接任务方法，[Wiki LOG](wiki/LOG.md) 与 [raw LOG](raw/LOG.md) 记录实际变动；研究计划、开发代码和运行结果由对应任务工作区承接。

SELF 保留当前判断与少量优先方向，具体知识和详细历史由上述位置承接。

## 5. 当前知识画像

**更新日期：2026-10-15。** 当前知识已连接量化方法、误差分析、模型结构与部分固定实现，覆盖程度见下表。每份来源的版本、实际研读范围和未核分支见[研读目录](wiki/READING.md)，逐批变动见[维护记录](wiki/LOG.md)。

| 主线 | 已有知识与入口 | 主要不足或范围 |
| --- | --- | --- |
| 模型与数值基础 | [稠密 decoder 推理](wiki/fundamentals/model/transformer-autoregressive-inference.md)、MoE/MLA 结构，以及均匀量化与浮点基础已有连贯解释；[MoE 执行](wiki/implementation/moe-routing-and-quantized-expert-execution.md)与[MLA 执行](wiki/implementation/mla-cache-and-attention-execution.md)已接到固定框架与设备接口 | 已贯通所选单设备 FP8 与 [W4A16 专家](wiki/implementation/w4a16-moe-expert-execution.md)、[MoE EP 框架调用链](wiki/implementation/moe-expert-parallel-execution.md)和普通 latent/cache 路径；[普通 MLA DCP](wiki/implementation/vllm-mla-context-parallel-execution.md)与[V3.2 packed/DSA 框架链](wiki/implementation/vllm-mla-packed-cache-and-dsa-execution.md)已补；所选 [DSA SM90 设备链](wiki/implementation/dsa-sparse-device-execution.md)已补；[匹配 DeepEP 通信设备链](wiki/implementation/deepep-moe-communication-execution.md)已补；[SGLang 匹配 DeepGEMM 专家设备链](wiki/implementation/deepgemm-moe-expert-execution.md)已补；其他 sparse 设备分支、EPLB、其他特殊格式、PCP/MRV2/其他 DCP 组合与 backend 仍按具体分支核查 |
| 权重/激活 PTQ 与变换 | [经典方法](wiki/methods/awq-gptq-smoothquant-comparison.md)、[可学习变换](wiki/research/omniquant-affinequant-flatquant-comparison.md)已有机制与比较；[FlexRound](wiki/methods/flexround.md)补逐元素除法选码、共同网格学习及幅值/STE条件，保留局部目标和反例；[SignRound](wiki/methods/signround.md)补舍入阈值、裁剪与符号更新，保留联合优化反例及原式/成本协议差异；[LRQ](wiki/methods/lrq.md)补对数因子/广播项与参数计账，区分离线因子和推理整数码；[QuantForge/HiRes](wiki/methods/quantforge.md)补实际编码误差、单码选择与分阶段重测；[XOR-Trellis](wiki/methods/xor-trellis.md)补局部候选、块内曲率路径与轻量解码边界；补齐[损失尺度](wiki/theory/layer-reconstruction-second-order-compensation.md#11-重构损失的尺度怎样影响优化)、[网格尺度敏感性](wiki/theory/calibration-and-range-selection.md#10-极低位宽为什么更依赖尺度选择)、[ConQuR 几何校准](wiki/methods/conqur.md)；[SmoothQuant+](wiki/methods/smoothquant-plus.md)补W4A16平滑/MLP融合与HumanEval校准，保留代理/任务和单/双卡比较边界 | [路线入口](wiki/INDEX.md#按问题选择量化路线)已连接对象、目标与执行；具体模型/量化器的取舍仍需共同协议 |
| 极低比特表示与码本 | [码本与预算](wiki/theory/codebook-quantization-and-bit-budget.md)、[表示设计](wiki/research/low-bit-representation-design.md)解释较深入；[DBW](wiki/methods/dbw.md)补成对分量 rank/bit 与固定恢复代码，[StagQ](wiki/methods/stagq.md)补共享前缀/元数据/保护记录；[VPTQ](wiki/methods/vptq.md)补同列向量、加权初始化与组合码本预算；[GPTVQ](wiki/methods/gptvq.md)补多输入列度量、中心更新与移动CPU小表 | [QuIP所选OPT/Balance](wiki/implementation/quip-ldlq-and-coordinate-restoration.md)已核逆序LDLQ、预处理/恢复与普通浮点保存，字段/误差坐标、LDLQ_RG+lazy实参及其余分支保留边界；[QuIP#所选E8P12](wiki/implementation/quip-sharp-e8p-packing-and-inference.md)已接到packed保存/加载与单行CUDA、多行浮点展开，形状/训练缓存/其余分支保留范围；[SpQR所选格式](wiki/implementation/spqr-metadata-outliers-and-cuda-format.md)补跨行QQ分组、覆盖/相加与4位物理载荷及CSR/PTCSR消费，保护零码/尾块/批量保留条件；各方法的源码与部署覆盖仍不均，DBW 论文 kernel 与 commit 有差异，StagQ 未取得作者实现，logical rate 不能替代物理读取 |
| 敏感性与混合精度 | [曲率加权误差](wiki/theory/curvature-weighted-quantization-error.md)、[预算与分配](wiki/theory/mixed-precision-allocation.md)已有共同分析；补因子载荷与多精度前缀的不同选择单元；[CMPQ](wiki/methods/cmpq.md)补输入通道分位分配、双类保护与实际预算；[CoopQ](wiki/methods/coopq.md)补联合量化边际、交互代理与求解预算；[LLM-MQ](wiki/methods/llm-mq.md)补候选梯度投影、一阶余项与稀疏预算；[ResQ](wiki/methods/resq.md)补高精度信号子空间与平方能量代理，[所选实现](wiki/implementation/resq-basis-and-fake-quantization.md)补未中心化Gram/融合/浮点缓存 | 具体配置仍需结合候选、层间交互、物理格式与系统成本判断 |
| QAT 与可学习量化 | STE、网格参数与训练范式已有解释；[PACT](wiki/methods/pact.md)补激活上界、任务阈值代理与L2正则，区分LSQ内部网格梯度并保留非负CNN条件；[LSQ+](wiki/methods/lsq-plus.md)补实数偏移、权重统计/激活MSE初值与正负码分配，保留原文推论和实现未决；[LQ-Nets](wiki/methods/lq-nets.md)补二进制基码值、前向QEM与任务STE分工、混合标签域内积和基预算，保留秩/时序与部署条件；[EfficientQAT](wiki/methods/efficientqat.md)含定向源码分析；[PV-Tuning](wiki/methods/pv-tuning.md)补离散/连续联合恢复、代理下降与子空间条件；[LC-QAT](wiki/methods/lc-qat.md)补仿射网格、硬前向重排与代理坐标；[QUASAR](wiki/methods/quasar.md)补每步选码/加权恢复、SGD条件界与格式证据；[Norm Tweaking](wiki/methods/norm-tweaking.md)补固定量化主体/norm调整与矩匹配、生成种子条件 | PV精确理论与Adam/共享梯度/STE混合分别说明；[AQLM固定PV主链](wiki/implementation/aqlm-pv-training-and-code-updates.md)已核目标/梯度/选码/写回，含实际trust范围与数据/保存边界；[原始校准与所选转换/CUDA 2×8](wiki/implementation/aqlm-calibration-export-and-inference.md)已接续；其他后端、全部适配器及部分分布式组合未核；LC-QAT/QUASAR匹配训练/内核未取得，DGE数值处理、缩放坐标接口与部分数据口径保留未决 |
| KV cache 量化 | [对象与粒度](wiki/theory/kv-cache-quantization-objects-and-granularity.md)、[误差传播](wiki/theory/attention-quantization-error-propagation.md)、预算与生命周期已有主线；WUSH-KV 补可逆平衡/GQA 敏感度；MLA 补联合池、scale/转换、paged 读取、分段归约、DCP owner/local lengths/通信，以及 V3.2 混合精度字段、indexer 整页 key/scale、因果 top-k 与主缓存读取 | [OSCAR前置实现](wiki/implementation/oscar-mixed-kv-cache-execution.md)的选定量化器/窗口/读取已核，含裁剪与尺度消费差异；[SAW所选writer/GQA decode](wiki/methods/saw-int4.md#41-所选配置与实际调用)已核two-half INT4、FP32元数据恢复后BF16、Q融合/概率舍入与split合并，FA3 flatten 的页表恢复、K 逆旋转后 BF16 与完整历史临时缓冲已接续，FA3 库内未核；WUSH-KV作者改版及一般可逆T的实际集成尚未匹配；MLA 的所选 packed/DSA 框架与匹配 SM90 设备接口已核，含 prefill/decode scale 舍入差异；其他 FlashMLA/DeepGEMM 分支、格式与硬件/backend 组合仍未全面覆盖 |
| 多模态与生成模型量化 | [VLM 对象](wiki/fundamentals/model/vision-language-model-tokens-and-quantization.md)、[模态目标与方法比较](wiki/research/mbq-vlmq-comparison.md)已有联系；[QSLAW](wiki/methods/qslaw.md)补任务监督下的组尺度和数据warmup，[所选实现](wiki/implementation/qslaw-scale-learning-and-float-export.md)补固定网格/重算码、mix与浮点恢复，采样/保存前提及论文协议差异保留；[OnPTQ](wiki/methods/onptq.md)补量化前缀、分支后果、context anchor 与刷新；[VEQ](wiki/methods/veq.md)补MoE专家目标/亲和曲率，保留搜索耦合、非负权重和评测协议条件 | OnPTQ 的状态与后果证据限于论文指定 Qwen 设置，底层更新配置/代码尚缺；[PIVOT-Q](wiki/methods/pivot-q.md)已补学生环境轨迹、稀疏动作恢复、LoRA/fake quant 主链与传导式评测，其他模型详细后端仍待核；[SAPrune](wiki/methods/saprune.md)补层深校准、动态双路径 token 选择、attention 代理与逐层成本，作者实现未取得；[CentriQ](wiki/methods/centriq.md)已补 adaLN 均值、动态中心化旁路、免校准尺度与 DiT 质量/运行边界，作者实现未取得；[医学删层恢复](wiki/methods/medical-llava-prune-sft-quantize.md)和[脉冲时间编码](wiki/methods/spikemllm.md)已按选定来源接续；[VLA CPU执行](wiki/implementation/vla-cpu-action-chunk-execution.md)、[MWOP](wiki/methods/mwop.md)、[CoVisco](wiki/methods/covisco.md)与[RDTok](wiki/methods/rdtok.md)已补不同计算/表示预算；TTS为其他具体案例 |
| 部署、推理框架与多卡 | [vLLM](wiki/implementation/vllm-inference-execution.md)、[SGLang](wiki/implementation/sglang-inference-execution.md)的加载、调度、缓存与 TP 有主线；MoE/MLA 单设备执行、[MoE EP 映射/通信/回传](wiki/implementation/moe-expert-parallel-execution.md)、[MLA DCP 分片/合并与 prefix prefill](wiki/implementation/vllm-mla-context-parallel-execution.md)、[V3.2 packed 缓存与 DSA 选择/读取](wiki/implementation/vllm-mla-packed-cache-and-dsa-execution.md)、[匹配 SM90 DSA 设备 logits/attention](wiki/implementation/dsa-sparse-device-execution.md)、[匹配 DeepEP 通信设备链](wiki/implementation/deepep-moe-communication-execution.md)、[匹配 DeepGEMM 专家设备链](wiki/implementation/deepgemm-moe-expert-execution.md)和独立 FlashInfer v0.6.17 plan/run/device 路径已补 | 固定版本与调用分支分别登记；其他后端较浅，其他 DeepEP 通信分支/完整外部协议、其他 FlashMLA/DeepGEMM 设备分支、EPLB/其他 EP、分布式 KV/推测执行未全面展开，没有本轮服务/GPU 实测 |
| GPU 算子与低精度执行 | [FlashAttention原论文](wiki/theory/flash-attention-tiling-and-recomputation.md)补精确归一化分块、IO容量与下界范围、反向概率重算，保留FP/稀疏/历史硬件条件；[FA2](wiki/theory/flash-attention-2-work-partitioning.md)补减少非matmul工作、sequence/warp分工与计数边界，保留v1公式差异；[所选CUDA](wiki/implementation/flash-attention-2-cuda-execution.md)补普通接口与低精度转换、seqK反向和det/GQA边界；常见推理算子、W4A16/W8A8 已有实例；所选 FP8 MoE 专家设备链已补，对称 GPTQ 现代 Marlin 已核 repack/位恢复/尺度/流水/归约，非对称 AWQ 已接续[所选 Dense Marlin 整数零点消费](wiki/implementation/weight-only-dequant-kernels.md#11-非对称-awq--vllm-marlin整数零点怎样参与计算)；补齐[CUDA 编译与接入](wiki/implementation/cuda-operator-build-and-integration.md)、[Triton 语言语义](wiki/fundamentals/operators/triton-language-and-compilation.md)及[FP4/FP8 块缩放调用与布局](wiki/implementation/block-scaled-matmul-execution.md) | 已形成基础写算子与追踪执行的阅读链；已核 Dense kU4/G128/A16/no-act-order/SM80+，已接续单设备 BF16 W4A16 MoE/G128多组，其他 AWQ 格式/组长、单组/多设备/其他专家 backend 仍需具体分支核查；具体算子的 backward、新架构专属优化尚未系统覆盖，实际运行按任务开展 |
| 质量、可靠性与性能 | [模型质量](wiki/implementation/model-quality-evaluation.md)、[服务指标](wiki/implementation/serving-performance-evaluation.md)、诊断与可靠性已有解释；场景/提示、长度/选项与过滤/聚合协议已接入；补[KL/TV 到 token 翻转](wiki/theory/quantization-ranking-stability.md#8-语言模型的分布距离怎样变成-token-翻转)及同前缀评测条件；[FlashInfer计时与验证](wiki/implementation/kernel-correctness-and-benchmarking.md)补后端/样本及匹配比例/采样判据；[生成Kernel评测](wiki/implementation/llm-generated-kernel-evaluation.md)补真实输入/状态、任务/负载/族分母与条件速度 | 一致性仍需与任务后果、自由生成反馈和运行成本分开验证；未形成统一实测基准或全部方法的排名 |

## 6. 下一步建设与维护

原88条队列已按实际材料完成研读、整合或无需新增判断及独立评审。各页保留具体核对范围；58条“已使用、研读范围未知”的历史记录单独维护，见[研读目录](wiki/READING.md)。

后续按实际知识缺口推进：

1. **补齐缺失实现材料。** OnPTQ的参数更新配置、WUSH-KV作者改版及一般可逆变换的接入仍需匹配材料。OSCAR前置实现已有解释，WUSH自身的实现范围继续单独记录；其余方法的匹配代码缺口见[计划中的条件事项](research/plans/2026-10-11-ingest-batches.md#5-条件事项与历史记录)。
2. **核实历史研读范围。** 围绕当前任务需要的知识，逐项核实58条历史记录；有依据后更新范围、承接页和遗留问题。
3. **按使用需要加深执行路径。** 优先补表中尚未沿完整链核读的通信、缓存、格式与后端分支。编译、模型质量和设备性能由具体应用任务验证，结果与静态分析分别记录。

主线覆盖、缺口或优先方向发生实质变化时，更新本页；排期及完成记录见[分批计划](research/plans/2026-10-11-ingest-batches.md)与LOG。
