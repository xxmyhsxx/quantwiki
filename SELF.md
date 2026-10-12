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

**更新日期：2026-10-12。** 以 2026-10-03 的主题检索与代表正文核读为基线，结合 DeepSeek-V2、CUDA/Triton、六篇优化与生成行为论文，以及本轮 MoE/MLA 固定源码、DBW/StagQ/CentriQ/PIVOT-Q/SAPrune/QuantForge/XOR-Trellis/VPTQ/GPTVQ/PV-Tuning及AQLM所选PV实现、原始校准/转换/推理与LC-QAT/QUASAR/Norm Tweaking/CMPQ/CoopQ/LLM-MQ/ResQ/Atom/FlexQ/Ideogram INT8-GGUF/SharQ ingest 更新受影响判断。现已连接模型结构到缓存/专家接口，并补 rank/bit 与共享前缀的表示预算；续核所选 MoE EP 的位置映射、normal/HT/LL 通信接口、接收侧计算与输出回传，以及普通 MLA DCP 的缓存分片、局部 attention/通信合并和 prefix prefill；再补默认 V3.2 的 packed 主缓存、独立 DSA indexer 的表示与因果选择、物理/workspace 索引及所选 sparse prefill/decode 框架读取，并接续匹配 DeepGEMM/FlashMLA 的 SM90 logits、packed/BF16 读取、尺度舍入、softmax/split 和无效项接口；再接续匹配默认构建来源的 DeepEP normal/HT 的 rank 去重、hierarchical 组去重/载荷/两段归约，以及 LL 的专家复制、动态 FP8、容量/scale、原 gate 回传、event/hook 与双缓冲。再接续 SGLang 默认 DeepGEMM 0.1.3 的包源码对应、contiguous/masked 行选择、TMA scale、分组 WGMMA/FP32 恢复与 BF16 输出，保留 padding/有效前缀及 SBO API 差异。再补 vLLM Dense AWQ Marlin 的整数零点列序/组边界/共同基底抵消、尺度/MMA/归约/bias，以及单组/SM75/atomic 的版本边界；再接续单设备 BF16 W4A16 MoE 的逐专家 AWQ repack、配对/专家 stride、整数零点与尺度、普通 SiLU/gate 的低精度舍入、CUDA FP32 top-k 汇合与 workspace/finalize。再补OSCAR所选BF16/GQA/affine INT2的码与元数据、sink/recent窗口、flush和QP/HP Attention精度接口，作为WUSH-KV公开前置证据；未取得WUSH作者改版。具体版本、条件和核读范围见对应页面及 READING。

| 主线 | 已有知识与入口 | 主要不足或范围 |
| --- | --- | --- |
| 模型与数值基础 | [稠密 decoder 推理](wiki/fundamentals/model/transformer-autoregressive-inference.md)、MoE/MLA 结构，以及均匀量化与浮点基础已有连贯解释；[MoE 执行](wiki/implementation/moe-routing-and-quantized-expert-execution.md)与[MLA 执行](wiki/implementation/mla-cache-and-attention-execution.md)已接到固定框架与设备接口 | 已贯通所选单设备 FP8 与 [W4A16 专家](wiki/implementation/w4a16-moe-expert-execution.md)、[MoE EP 框架调用链](wiki/implementation/moe-expert-parallel-execution.md)和普通 latent/cache 路径；[普通 MLA DCP](wiki/implementation/vllm-mla-context-parallel-execution.md)与[V3.2 packed/DSA 框架链](wiki/implementation/vllm-mla-packed-cache-and-dsa-execution.md)已补；所选 [DSA SM90 设备链](wiki/implementation/dsa-sparse-device-execution.md)已补；[匹配 DeepEP 通信设备链](wiki/implementation/deepep-moe-communication-execution.md)已补；[SGLang 匹配 DeepGEMM 专家设备链](wiki/implementation/deepgemm-moe-expert-execution.md)已补；其他 sparse 设备分支、EPLB、其他特殊格式、PCP/MRV2/其他 DCP 组合与 backend 仍按具体分支核查 |
| 权重/激活 PTQ 与变换 | [经典方法](wiki/methods/awq-gptq-smoothquant-comparison.md)、[可学习变换](wiki/research/omniquant-affinequant-flatquant-comparison.md)已有机制与比较；[QuantForge/HiRes](wiki/methods/quantforge.md)补实际编码误差、单码选择与分阶段重测；[XOR-Trellis](wiki/methods/xor-trellis.md)补局部候选、块内曲率路径与轻量解码边界；补齐[损失尺度](wiki/theory/layer-reconstruction-second-order-compensation.md#11-重构损失的尺度怎样影响优化)、[网格尺度敏感性](wiki/theory/calibration-and-range-selection.md#10-极低位宽为什么更依赖尺度选择)、[ConQuR 几何校准](wiki/methods/conqur.md)；[SmoothQuant+](wiki/methods/smoothquant-plus.md)补W4A16平滑/MLP融合与HumanEval校准，保留代理/任务和单/双卡比较边界 | [路线入口](wiki/INDEX.md#按问题选择量化路线)已连接对象、目标与执行；具体模型/量化器的取舍仍需共同协议 |
| 极低比特表示与码本 | [码本与预算](wiki/theory/codebook-quantization-and-bit-budget.md)、[表示设计](wiki/research/low-bit-representation-design.md)解释较深入；[DBW](wiki/methods/dbw.md)补成对分量 rank/bit 与固定恢复代码，[StagQ](wiki/methods/stagq.md)补共享前缀/元数据/保护记录；[VPTQ](wiki/methods/vptq.md)补同列向量、加权初始化与组合码本预算；[GPTVQ](wiki/methods/gptvq.md)补多输入列度量、中心更新与移动CPU小表 | [QuIP所选OPT/Balance](wiki/implementation/quip-ldlq-and-coordinate-restoration.md)已核逆序LDLQ、预处理/恢复与普通浮点保存，字段/误差坐标、LDLQ_RG+lazy实参及其余分支保留边界；[QuIP#所选E8P12](wiki/implementation/quip-sharp-e8p-packing-and-inference.md)已接到packed保存/加载与单行CUDA、多行浮点展开，形状/训练缓存/其余分支保留范围；[SpQR所选格式](wiki/implementation/spqr-metadata-outliers-and-cuda-format.md)补跨行QQ分组、覆盖/相加与4位物理载荷及CSR/PTCSR消费，保护零码/尾块/批量保留条件；各方法的源码与部署覆盖仍不均，DBW 论文 kernel 与 commit 有差异，StagQ 未取得作者实现，logical rate 不能替代物理读取 |
| 敏感性与混合精度 | [曲率加权误差](wiki/theory/curvature-weighted-quantization-error.md)、[预算与分配](wiki/theory/mixed-precision-allocation.md)已有共同分析；补因子载荷与多精度前缀的不同选择单元；[CMPQ](wiki/methods/cmpq.md)补输入通道分位分配、双类保护与实际预算；[CoopQ](wiki/methods/coopq.md)补联合量化边际、交互代理与求解预算；[LLM-MQ](wiki/methods/llm-mq.md)补候选梯度投影、一阶余项与稀疏预算；[ResQ](wiki/methods/resq.md)补高精度信号子空间与平方能量代理，[所选实现](wiki/implementation/resq-basis-and-fake-quantization.md)补未中心化Gram/融合/浮点缓存 | 具体配置仍需结合候选、层间交互、物理格式与系统成本判断 |
| QAT 与可学习量化 | STE、网格参数与训练范式已有解释；[EfficientQAT](wiki/methods/efficientqat.md)含定向源码分析；[PV-Tuning](wiki/methods/pv-tuning.md)补离散/连续联合恢复、代理下降与子空间条件；[LC-QAT](wiki/methods/lc-qat.md)补仿射网格、硬前向重排与代理坐标；[QUASAR](wiki/methods/quasar.md)补每步选码/加权恢复、SGD条件界与格式证据；[Norm Tweaking](wiki/methods/norm-tweaking.md)补固定量化主体/norm调整与矩匹配、生成种子条件 | PV精确理论与Adam/共享梯度/STE混合分别说明；[AQLM固定PV主链](wiki/implementation/aqlm-pv-training-and-code-updates.md)已核目标/梯度/选码/写回，含实际trust范围与数据/保存边界；[原始校准与所选转换/CUDA 2×8](wiki/implementation/aqlm-calibration-export-and-inference.md)已接续；其他后端、全部适配器及部分分布式组合未核；LC-QAT/QUASAR匹配训练/内核未取得，DGE数值处理、缩放坐标接口与部分数据口径保留未决 |
| KV cache 量化 | [对象与粒度](wiki/theory/kv-cache-quantization-objects-and-granularity.md)、[误差传播](wiki/theory/attention-quantization-error-propagation.md)、预算与生命周期已有主线；WUSH-KV 补可逆平衡/GQA 敏感度；MLA 补联合池、scale/转换、paged 读取、分段归约、DCP owner/local lengths/通信，以及 V3.2 混合精度字段、indexer 整页 key/scale、因果 top-k 与主缓存读取 | [OSCAR前置实现](wiki/implementation/oscar-mixed-kv-cache-execution.md)的选定量化器/窗口/读取已核，含裁剪与尺度消费差异；[SAW所选writer/GQA decode](wiki/methods/saw-int4.md#41-所选配置与实际调用)已核two-half INT4、FP32元数据恢复后BF16、Q融合/概率舍入与split合并，FA3 flatten 的页表恢复、K 逆旋转后 BF16 与完整历史临时缓冲已接续，FA3 库内未核；WUSH-KV作者改版及一般可逆T的实际集成尚未匹配；MLA 的所选 packed/DSA 框架与匹配 SM90 设备接口已核，含 prefill/decode scale 舍入差异；其他 FlashMLA/DeepGEMM 分支、格式与硬件/backend 组合仍未全面覆盖 |
| 多模态与生成模型量化 | [VLM 对象](wiki/fundamentals/model/vision-language-model-tokens-and-quantization.md)、[模态目标与方法比较](wiki/research/mbq-vlmq-comparison.md)已有联系；[QSLAW](wiki/methods/qslaw.md)补任务监督下的组尺度和数据warmup，[所选实现](wiki/implementation/qslaw-scale-learning-and-float-export.md)补固定网格/重算码、mix与浮点恢复，采样/保存前提及论文协议差异保留；[OnPTQ](wiki/methods/onptq.md)补量化前缀、分支后果、context anchor 与刷新；[VEQ](wiki/methods/veq.md)补MoE专家目标/亲和曲率，保留搜索耦合、非负权重和评测协议条件 | OnPTQ 的状态与后果证据限于论文指定 Qwen 设置，底层更新配置/代码尚缺；[PIVOT-Q](wiki/methods/pivot-q.md)已补学生环境轨迹、稀疏动作恢复、LoRA/fake quant 主链与传导式评测，其他模型详细后端仍待核；[SAPrune](wiki/methods/saprune.md)补层深校准、动态双路径 token 选择、attention 代理与逐层成本，作者实现未取得；[CentriQ](wiki/methods/centriq.md)已补 adaLN 均值、动态中心化旁路、免校准尺度与 DiT 质量/运行边界，作者实现未取得；TTS/VLA 为其他具体案例 |
| 部署、推理框架与多卡 | [vLLM](wiki/implementation/vllm-inference-execution.md)、[SGLang](wiki/implementation/sglang-inference-execution.md)的加载、调度、缓存与 TP 有主线；MoE/MLA 单设备执行、[MoE EP 映射/通信/回传](wiki/implementation/moe-expert-parallel-execution.md)、[MLA DCP 分片/合并与 prefix prefill](wiki/implementation/vllm-mla-context-parallel-execution.md)、[V3.2 packed 缓存与 DSA 选择/读取](wiki/implementation/vllm-mla-packed-cache-and-dsa-execution.md)、[匹配 SM90 DSA 设备 logits/attention](wiki/implementation/dsa-sparse-device-execution.md)、[匹配 DeepEP 通信设备链](wiki/implementation/deepep-moe-communication-execution.md)、[匹配 DeepGEMM 专家设备链](wiki/implementation/deepgemm-moe-expert-execution.md)和独立 FlashInfer v0.6.17 plan/run/device 路径已补 | 固定版本与调用分支分别登记；其他后端较浅，其他 DeepEP 通信分支/完整外部协议、其他 FlashMLA/DeepGEMM 设备分支、EPLB/其他 EP、分布式 KV/推测执行未全面展开，没有本轮服务/GPU 实测 |
| GPU 算子与低精度执行 | 常见推理算子、W4A16/W8A8 已有实例；所选 FP8 MoE 专家设备链已补，对称 GPTQ 现代 Marlin 已核 repack/位恢复/尺度/流水/归约，非对称 AWQ 已接续[所选 Dense Marlin 整数零点消费](wiki/implementation/weight-only-dequant-kernels.md#11-非对称-awq--vllm-marlin整数零点怎样参与计算)；补齐[CUDA 编译与接入](wiki/implementation/cuda-operator-build-and-integration.md)、[Triton 语言语义](wiki/fundamentals/operators/triton-language-and-compilation.md)及[FP4/FP8 块缩放调用与布局](wiki/implementation/block-scaled-matmul-execution.md) | 已形成基础写算子与追踪执行的阅读链；已核 Dense kU4/G128/A16/no-act-order/SM80+，已接续单设备 BF16 W4A16 MoE/G128多组，其他 AWQ 格式/组长、单组/多设备/其他专家 backend 仍需具体分支核查；具体算子的 backward、新架构专属优化尚未系统覆盖，实际运行按任务开展 |
| 质量、可靠性与性能 | [模型质量](wiki/implementation/model-quality-evaluation.md)、[服务指标](wiki/implementation/serving-performance-evaluation.md)、诊断与可靠性已有解释；补[KL/TV 到 token 翻转](wiki/theory/quantization-ranking-stability.md#8-语言模型的分布距离怎样变成-token-翻转)及同前缀评测条件 | 一致性仍需与任务后果、自由生成反馈和运行成本分开验证；未形成统一实测基准或全部方法的排名 |

相对深入的是经典 PTQ/变换、误差与表示机制、KV cache，以及部分框架和 GPU 实现路径。优化尺度与生成行为、MoE/MLA 到所选单设备框架接口，以及 MoE EP 的映射/通信调用/回传、所选普通 MLA DCP 的缓存/合并断点已补，INDEX 已连接对象、目标与执行路线。所选 V3.2 packed 字段、DSA 独立选择缓存、sparse 框架读取与匹配 SM90 设备计算也已补；所选 SGLang 匹配 DeepGEMM 专家设备计算、vLLM Dense AWQ 整数零点的 Marlin 设备消费及所选单设备 W4A16 MoE 的专家配对/低比特恢复与加权归约也已接续。所选OSCAR mixed INT2的实际writer/窗口/分层读取与SAW K-only INT4的融合writer/GQA decode也已补，二者恢复精度和当前chunk路径分别说明，未用前置正交变换替代WUSH实现。剩余缺口集中在尚无实现材料的方法、其他通信分支/分布式/特殊格式及未沿完整链核读的后端。算子知识可用于启动常见推理 kernel 开发；新算子的支持条件、目标环境和运行结果仍由具体任务确定。

已完成的知识建设包括方法解释、公共推导与比较、固定源码分析和评测知识。实际开发、运行或性能成果以对应任务记录为准；上游实现分析不等于本项目已实现同一算法，计划与中间产物不计为完成。

## 6. 下一步建设与维护

分批安排与理由见 [已收录资料的 ingest 计划](research/plans/2026-10-11-ingest-batches.md)。资料的实际处理状态仍由 READING 维护。

当前按用户要求逐步处理所有已收录材料；[研读目录](wiki/READING.md#待逐项核查与-ingest-的已收录材料)接入首轮140条未登记来源队列，目前剩107条，已处理项按实际范围移入研读记录。先核真实覆盖，再逐项 ingest、由独立 Astra 子代理 review、修正和本地提交。已有登记也保留实际阅读范围，不能直接计为全文完成。重要性与执行先后分开：

1. **P1：材料具备后补生成校准与低位宽缓存的实现证据。** 本次官方入口复查仍未定位匹配的OnPTQ参数/更新配置或WUSH-KV作者改版；OSCAR公开前置量化器/窗口/读取已补，WUSH的一般可逆变换接入仍需匹配实现。需要材料后再核查，不从搜索未命中推断代码未发布。
2. **P1：逐项补齐已收录材料的核心解释。** SAW 同配置的 [FA3 prefill 恢复](wiki/methods/saw-int4.md#45-fa3-prefill从分页-int4-恢复到原-q-坐标)断点已补，CentriQ 的论文机制、公共均值旁路和动态范围已补，PIVOT-Q论文与GR00T+QuantVLA主恢复链已补，论文/代码协议差异保留；SAPrune 的论文选层/双路径与公共 attention/逐层预算已补，未取得匹配作者实现；QuantForge/HiRes 的坐标/编码/路径恢复与搜索评测已补，保持模拟执行、辅助存储和原生部署边界；XOR-Trellis 的生成器/局部候选、block-LDL对角路径与综合证据已补，完整打包和部署仍待实现；VPTQ 的同列选码、加权初始化与残差/离群预算已补，保留原文公式和运行协议边界；GPTVQ 的多输入列度量、数据感知中心更新与移动小码本成本已补，保留格式/协议差异；PV-Tuning 的联合恢复、代理下降与数据分布已补，保留精确理论/工程近似及主附表差异；AQLM固定PV主链已核绝对目标、新中心时序、STE与实际变化限制；原始校准、条件转换及所选CUDA 2×8前向读取已接续，保留元数据与舍入边界；LC-QAT仿射表示、代理训练与数据/执行证据已补；QUASAR每步重构与理论/实验条件已补；Norm Tweaking的顺序局部参考、norm变量与统计校正已补；CMPQ的输入通道分配/双类保护与分位预算已补，端点/并列与mask实现仍待核；CoopQ的SPQE/协方差代理和预算求解已补，原文符号/对角/基线衔接保留待实现核；LLM-MQ的候选梯度/整数预算及所选GEMV证据已补，保留缺失附录和实际计费/接口边界；ResQ的信号子空间/平方能量代理与所选Llama/full_shared实现已补，原文证明和浮点模拟/部署差异分别保留；第4批完成；第5批[Atom](wiki/methods/atom.md)的固定离群通道、group整数恢复与[分页KV实现](wiki/implementation/atom-mixed-gemm-and-paged-kv.md)已补，保留模拟/native参数舍入、真实权重导出与效率缓存简化边界；[FlexQ](wiki/methods/flexq.md)的INT6配方、signed位平面/group恢复及[所选实现](wiki/implementation/flexq-bitplanes-and-group-restoration.md)已补，保留式6缺项、有限精度/原生布局、完整导出与性能口径边界；[Ideogram INT8/GGUF](wiki/methods/ideogram-int8-gguf.md)的双分支预算、保护/提示协议和[配对生成评测](wiki/implementation/model-quality-evaluation.md#12-生成质量保真度和提示协议怎样分开评测)已补，保留FP8参考、raw/JSON消融与格式/部署边界；[SharQ](wiki/methods/sharq.md)的动态pair主干/量化后残差与[共享尺度/执行](wiki/implementation/sharq-sparse-residual-and-shared-scales.md)已补，保留共同权重界、原文格式/图表差异及native/模拟/融合边界；第5批完成；第6批[FluxBin](wiki/methods/fluxbin.md)的行列二值分解/显著残差和[输入查表](wiki/theory/codebook-quantization-and-bit-budget.md#62-二值图案查表怎样摊销输入点积)已补，保留固定尺度最优条件、完整预算及曲线/表格协议；[QuIP所选实现](wiki/implementation/quip-ldlq-and-coordinate-restoration.md)的逆序二阶反馈/变换、浮点保存和[误差坐标](wiki/theory/orthogonal-rotation-and-hadamard-quantization.md#41-先恢复浮点权重与保留量化坐标分别意味着什么)已补，保留其余分支/字段/lazy组合/诊断边界；[QuIP#所选实现](wiki/implementation/quip-sharp-e8p-packing-and-inference.md)的格编码、[逻辑码/物理载荷](wiki/theory/codebook-quantization-and-bit-budget.md#42-逻辑码字与物理打包为什么需要分别核对)和保存/单行CUDA/多行展开已补，保留形状/可选缓存/其余分支边界；[SpQR所选实现](wiki/implementation/spqr-metadata-outliers-and-cuda-format.md)的两级分组、[保护协议/物理预算](wiki/theory/codebook-quantization-and-bit-budget.md#52-保护值应覆盖还是相加由保存协议决定)与旧恢复/优化消费已补，保护零码及配置/运行边界分别保留；第6批完成；第7批[QSLAW论文/所选实现](wiki/methods/qslaw.md)已补[尺度位置与STE](wiki/fundamentals/quantization/post-training-and-quantization-aware-training.md#35-学习尺度前先确定它进入哪一侧)、数据阶段及浮点恢复边界；[SmoothQuant+](wiki/methods/smoothquant-plus.md)的通道/MLP融合、[校准选择](wiki/theory/calibration-and-range-selection.md#53-使用评测输入校准时要保留选择边界)与服务证据边界已补；[VEQ](wiki/methods/veq.md)已补[专家覆盖与加权位置](wiki/theory/calibration-and-range-selection.md#61-moe-校准须记录专家覆盖与权重位置)及搜索/曲率条件；下一项MASQuant/QIG解读文章，核补充价值与证据边界，再接第8–9批，再按第1–9批计划接续P1论文与必要代码依赖。先补方法核心机制与必要知识依赖，再沿已有代码核格式/执行边界；每项以实际可读材料确定范围，避免将收录或导航登记计为 ingest。
3. **P2：接续基础文档、教程与其余实现分支。** 当前全部材料已获研读授权；按论文/方法依赖接入相关文档和代码，再处理其余条目与58条历史范围待核记录。视频简介、索引页等材料只能支持其实际内容；PIVOT-Q其他模型的完整恢复/量化链、FA3 库内、其他通信/分页/backend 分支按具体范围登记，真实质量与性能仍另按运行任务开展。

主线覆盖、缺口或优先方向发生实质变化时，更新对应画像与建设顺序，移除已解决缺口。
