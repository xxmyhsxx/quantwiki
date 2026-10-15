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
- **源码分析：** 理解实现的职责与整体工作方式，再核对关键变量、流程、格式和接口，解释各部分与方法或系统的关系。
- **Review 与维护：** 检查理解、表达、依据、解释深度和跨页联系，修正已确认问题。
- **查询与研究支持：** 定位知识、解释问题、比较路线，服务明确的研究目标。

一次 ingest 是否完成，取决于读者能否沿正文理解对象的作用、整体思路和关键机制，以及必要依赖、来源条件和知识关联是否落实；源码覆盖与局部细节深度不能单独证明主题已经讲清楚。已有内容充分时，无需新增页面。

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
| 模型与数值基础 | [自回归主干与模型差异](wiki/fundamentals/model/transformer-autoregressive-inference.md)、[MoE 路由](wiki/fundamentals/model/mixture-of-experts-and-routing.md)、[MLA 缓存](wiki/fundamentals/model/multi-head-latent-attention.md)连接结构与量化对象；已有所选框架和设备实现 | 结构说明与固定实现分别保留；其他 sparse、并行和特殊格式分支仍按具体版本核查 |
| 权重/激活 PTQ 与变换 | [经典方法比较](wiki/methods/awq-gptq-smoothquant-comparison.md)、[可学习变换](wiki/research/omniquant-affinequant-flatquant-comparison.md)与[校准和尺度](wiki/theory/calibration-and-range-selection.md)已连接目标、机制与条件 | 具体模型、量化器和执行配置的取舍仍需共同口径，方法名不能代替适用条件 |
| 极低比特表示与码本 | [码本与预算](wiki/theory/codebook-quantization-and-bit-budget.md)、[表示设计](wiki/research/low-bit-representation-design.md)及 QuIP、QuIP#、SpQR 等所选导出/读取路径已有解释 | 算法与部署覆盖仍不均；DBW 论文与代码有差异，StagQ 作者实现未取得；逻辑位宽不能替代物理载荷 |
| 敏感性与混合精度 | [曲率加权误差](wiki/theory/curvature-weighted-quantization-error.md)、[预算与分配](wiki/theory/mixed-precision-allocation.md)连接敏感性、配置交互和表示预算 | 代理指标须结合层间交互、候选配置、物理格式和系统成本判断 |
| QAT 与可学习量化 | [训练与代理梯度](wiki/fundamentals/quantization/post-training-and-quantization-aware-training.md)连接网格学习、离散选码和恢复；[AQLM PV 实现](wiki/implementation/aqlm-pv-training-and-code-updates.md)已有固定主链 | 精确理论与实际优化器分别说明；部分方法的匹配训练/内核、适配器和分布式组合未核 |
| KV cache 量化 | [对象与粒度](wiki/theory/kv-cache-quantization-objects-and-granularity.md)、[误差传播](wiki/theory/attention-quantization-error-propagation.md)连接生命周期、变换和预算；已有所选 MLA、OSCAR、SAW 实现 | WUSH-KV 作者改版与一般可逆变换接入尚未匹配；其他格式、硬件及 backend 组合不由已核分支推定 |
| 多模态与生成模型 | [VLM 对象](wiki/fundamentals/model/vision-language-model-tokens-and-quantization.md)、[模态目标与方法比较](wiki/research/mbq-vlmq-comparison.md)连接输入、校准及生成后果；VLA、DiT、TTS 有具体案例 | OnPTQ 底层更新配置/代码尚缺；SAPrune、CentriQ 作者实现未取得；案例不代表所有模型与任务 |
| 部署、推理框架与多卡 | [部署产物与后端](wiki/implementation/quantized-llm-deployment-backends.md)、[nano-vllm 组件与状态](wiki/implementation/nano-vllm-inference-execution.md)、[vLLM](wiki/implementation/vllm-inference-execution.md)与[SGLang](wiki/implementation/sglang-inference-execution.md)已有执行主线，连接所选 MoE/MLA 多卡与设备路径 | MLC-LLM 已补配置到 Python 引擎交接，编译 pass、代码生成与 C++ 执行未核；其他通信、缓存和后端组合保留局部范围，未据静态分析宣称运行收益 |
| GPU 算子与低精度执行 | [CUDA 接入](wiki/implementation/cuda-operator-build-and-integration.md)、[Triton 语义](wiki/fundamentals/operators/triton-language-and-compilation.md)、[反量化内核](wiki/implementation/weight-only-dequant-kernels.md)与[块缩放执行](wiki/implementation/block-scaled-matmul-execution.md)形成基础阅读链 | 格式、组长、设备及分支条件留在实现页；backward 与新架构专属优化尚未系统覆盖 |
| 质量、可靠性与性能 | [模型质量](wiki/implementation/model-quality-evaluation.md)、[服务指标](wiki/implementation/serving-performance-evaluation.md)与诊断已有解释，区分分布、生成行为、计时与资源口径 | 一致性、任务后果与运行成本分别判断；未形成统一实测基准或全部方法排名 |

2026-10-15 的定向复核发现：部分页面已经积累较多实现细节，但对象的整体作用、组件分工和跨主题联系仍不够集中。已补 MLC-LLM 部署主线、nano-vllm 状态职责和模型结构对照，并调整写作与审查要求；这轮覆盖不代表全库页面均已按新要求复核。上述画像只保留当前能力与主要边界，逐项实现条件和历史进展由正文、READING 与 LOG 承接。

## 6. 下一步建设与维护

原88条队列与58条历史范围记录均已按实际材料完成处理。各来源的研读范围、承接位置和未核分支见[研读目录](wiki/READING.md)；原历史日期无法还原的，以本轮核读日期建立范围。

后续按实际知识缺口推进：

1. **补足整体解释断点。** 后续建设先沿相关入口判断对象的作用、整体工作方式与局部机制是否连接；优先修复影响理解的断点，再决定是否继续下钻。已充分解释的页面不套模板重写。
2. **补齐缺失实现材料。** OnPTQ的参数更新配置、WUSH-KV作者改版及一般可逆变换的接入仍需匹配材料。OSCAR前置实现已有解释，WUSH自身的实现范围继续单独记录；其余方法的匹配代码缺口见对应[研读记录](wiki/READING.md)与正文。
3. **按使用需要加深执行路径。** 优先补表中尚未沿完整链核读的通信、缓存、格式与后端分支。编译、模型质量和设备性能由具体应用任务验证，结果与静态分析分别记录。

主线覆盖、缺口或优先方向发生实质变化时，更新本页；资料状态见[研读目录](wiki/READING.md)，实际变动见[维护记录](wiki/LOG.md)。
