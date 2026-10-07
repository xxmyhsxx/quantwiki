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

**更新日期：2026-10-07。** 以 2026-10-03 的主题检索与代表正文核读为基线，结合 DeepSeek-V2、CUDA/Triton、六篇优化与生成行为论文，以及本轮 MoE/MLA 固定源码、DBW/StagQ ingest 更新受影响判断。现已连接模型结构到缓存/专家接口，并补 rank/bit 与共享前缀的表示预算；续核所选 MoE EP 的位置映射、normal/HT/LL 通信接口、接收侧计算与输出回传，以及普通 MLA DCP 的缓存分片、局部 attention/通信合并和 prefix prefill；再补默认 V3.2 的 packed 主缓存、独立 DSA indexer 的表示与因果选择、物理/workspace 索引及所选 sparse prefill/decode 框架读取。具体版本、条件和核读范围见对应页面及 READING。

| 主线 | 已有知识与入口 | 主要不足或范围 |
| --- | --- | --- |
| 模型与数值基础 | [稠密 decoder 推理](wiki/fundamentals/model/transformer-autoregressive-inference.md)、MoE/MLA 结构，以及均匀量化与浮点基础已有连贯解释；[MoE 执行](wiki/implementation/moe-routing-and-quantized-expert-execution.md)与[MLA 执行](wiki/implementation/mla-cache-and-attention-execution.md)已接到固定框架与设备接口 | 已贯通所选单设备 FP8 专家、[MoE EP 框架调用链](wiki/implementation/moe-expert-parallel-execution.md)和普通 latent/cache 路径；[普通 MLA DCP](wiki/implementation/vllm-mla-context-parallel-execution.md)与[V3.2 packed/DSA 框架链](wiki/implementation/vllm-mla-packed-cache-and-dsa-execution.md)已补；外部通信/sparse 设备内核、EPLB、其他特殊格式、PCP/MRV2/其他 DCP 组合与 backend 仍按具体分支核查 |
| 权重/激活 PTQ 与变换 | [经典方法](wiki/methods/awq-gptq-smoothquant-comparison.md)、[可学习变换](wiki/research/omniquant-affinequant-flatquant-comparison.md)已有机制与比较；补齐[损失尺度](wiki/theory/layer-reconstruction-second-order-compensation.md#11-重构损失的尺度怎样影响优化)、[网格尺度敏感性](wiki/theory/calibration-and-range-selection.md#10-极低位宽为什么更依赖尺度选择)、[ConQuR 几何校准](wiki/methods/conqur.md) | [路线入口](wiki/INDEX.md#按问题选择量化路线)已连接对象、目标与执行；具体模型/量化器的取舍仍需共同协议 |
| 极低比特表示与码本 | [码本与预算](wiki/theory/codebook-quantization-and-bit-budget.md)、[表示设计](wiki/research/low-bit-representation-design.md)解释较深入；[DBW](wiki/methods/dbw.md)补成对分量 rank/bit 与固定恢复代码，[StagQ](wiki/methods/stagq.md)补共享前缀/元数据/保护记录 | 各方法的源码与部署覆盖不均；DBW 论文 kernel 与 commit 有差异，StagQ 未取得作者实现，logical rate 不能替代物理读取 |
| 敏感性与混合精度 | [曲率加权误差](wiki/theory/curvature-weighted-quantization-error.md)、[预算与分配](wiki/theory/mixed-precision-allocation.md)已有共同分析；补因子载荷与多精度前缀的不同选择单元 | 具体配置仍需结合候选、层间交互、物理格式与系统成本判断 |
| QAT 与可学习量化 | STE、网格参数与训练范式已有解释；[EfficientQAT](wiki/methods/efficientqat.md)含定向源码分析 | 其他方法的代码深度不同，未覆盖完整训练工具链 |
| KV cache 量化 | [对象与粒度](wiki/theory/kv-cache-quantization-objects-and-granularity.md)、[误差传播](wiki/theory/attention-quantization-error-propagation.md)、预算与生命周期已有主线；WUSH-KV 补可逆平衡/GQA 敏感度；MLA 补联合池、scale/转换、paged 读取、分段归约、DCP owner/local lengths/通信，以及 V3.2 混合精度字段、indexer 整页 key/scale、因果 top-k 与主缓存读取 | WUSH-KV 的真实量化/窗口实现尚缺定向核读；MLA 的所选 packed/DSA 框架接口已核，外部 FlashMLA/DeepGEMM sparse kernel、其他格式与全部硬件/backend 组合仍未全面覆盖 |
| 多模态量化与相关压缩 | [VLM 对象](wiki/fundamentals/model/vision-language-model-tokens-and-quantization.md)、[模态目标与方法比较](wiki/research/mbq-vlmq-comparison.md)已有联系；[OnPTQ](wiki/methods/onptq.md)补量化前缀、分支后果、context anchor 与刷新 | 状态与后果的证据限于论文指定 Qwen 设置；底层更新配置/代码尚缺；TTS/VLA 为具体案例 |
| 部署、推理框架与多卡 | [vLLM](wiki/implementation/vllm-inference-execution.md)、[SGLang](wiki/implementation/sglang-inference-execution.md)的加载、调度、缓存与 TP 有主线；MoE/MLA 单设备执行、[MoE EP 映射/通信/回传](wiki/implementation/moe-expert-parallel-execution.md)、[MLA DCP 分片/合并与 prefix prefill](wiki/implementation/vllm-mla-context-parallel-execution.md)、[V3.2 packed 缓存与 DSA 选择/读取](wiki/implementation/vllm-mla-packed-cache-and-dsa-execution.md)和独立 FlashInfer v0.6.17 plan/run/device 路径已补 | 固定版本与调用分支分别登记；其他后端较浅，DeepEP 外部通信内核、FlashMLA/DeepGEMM sparse 设备内核、EPLB/其他 EP、分布式 KV/推测执行未全面展开，没有本轮服务/GPU 实测 |
| GPU 算子与低精度执行 | 常见推理算子、W4A16/W8A8 已有实例；补齐[CUDA 编译与接入](wiki/implementation/cuda-operator-build-and-integration.md)、[Triton 语言语义](wiki/fundamentals/operators/triton-language-and-compilation.md)及[FP4/FP8 块缩放调用与布局](wiki/implementation/block-scaled-matmul-execution.md) | 已形成基础写算子与追踪执行的阅读链；具体算子的 backward、新架构专属优化尚未系统覆盖，实际运行按任务开展 |
| 质量、可靠性与性能 | [模型质量](wiki/implementation/model-quality-evaluation.md)、[服务指标](wiki/implementation/serving-performance-evaluation.md)、诊断与可靠性已有解释；补[KL/TV 到 token 翻转](wiki/theory/quantization-ranking-stability.md#8-语言模型的分布距离怎样变成-token-翻转)及同前缀评测条件 | 一致性仍需与任务后果、自由生成反馈和运行成本分开验证；未形成统一实测基准或全部方法的排名 |

相对深入的是经典 PTQ/变换、误差与表示机制、KV cache，以及部分框架和 GPU 实现路径。优化尺度与生成行为、MoE/MLA 到所选单设备框架接口，以及 MoE EP 的映射/通信调用/回传、所选普通 MLA DCP 的缓存/合并断点已补，INDEX 已连接对象、目标与执行路线。所选 V3.2 packed 字段、DSA 独立选择缓存与 sparse 框架读取也已补；剩余缺口集中在尚无实现材料的方法、匹配的外部设备内核、其他分布式/特殊格式及未沿完整链核读的后端。算子知识可用于启动常见推理 kernel 开发；新算子的支持条件、目标环境和运行结果仍由具体任务确定。

已完成的知识建设包括方法解释、公共推导与比较、固定源码分析和评测知识。实际开发、运行或性能成果以对应任务记录为准；上游实现分析不等于本项目已实现同一算法，计划与中间产物不计为完成。

## 6. 下一步建设与维护

建议按以下顺序建设，具体任务以用户当前目标为准：

1. **P1：材料具备后补生成校准与低位宽缓存的实现证据。** OnPTQ 的参数/更新配置、WUSH-KV 的量化器/窗口/在线变换仍有论文到代码的缺口；需要取得匹配版本的实现后核查，避免把论文流程当成运行成果。
2. **P1：接续匹配版本的外部设备内核。** MoE EP、普通 vLLM MLA DCP 与 V3.2 packed/DSA 的所选框架链已补；代码库下一项优先取得并核读 vLLM 默认固定的 FlashMLA sparse 源码及匹配 DeepGEMM logits 实现，确认 packed 读取/反量化、QK/PV、softmax、无效索引和调度契约。当前 vLLM 归档只包含这些外部接口和默认依赖版本，不能用框架核读补写设备内部。普通 DCP 的空分片 A2A 与非默认 interleave 条件按正文保留静态边界，具体运行验证随任务开展。PCP/MRV2、外部 DeepEP 通信 kernel、EPLB、其他 DSA/backend 或融合模型随部署问题定向核读，不默认扩展为全部设备/分支验收。
3. **P2：按研究需要扩展 VLA/DiT 与其他后端。** 已收录 Pivot-Q、SAPrune、CentriQ 等专题及其他推理后端，但不自动并入当前主线；任务目标需要时再核读其动作/图像质量条件和真实实现。

主线覆盖、缺口或优先方向发生实质变化时，更新对应画像与建设顺序，移除已解决缺口。
