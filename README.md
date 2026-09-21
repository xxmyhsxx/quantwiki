# 量化知识库

以知识页面为主体的量化研究知识库，主要关注大语言模型和多模态模型，连接量化算法的原理、代码实现、模型部署与推理加速。

这里整理了论文研读、技术文档和源码分析形成的知识，包括基础概念与推导、具体方法的流程与比较、量化格式和推理框架的实现细节，以及精度和性能的评测方法。它用于系统学习量化、分析研究问题，也为量化代码编写与部署优化提供参考。

**[浏览 Wiki](wiki/INDEX.md)** · [查看来源资料](raw/README.md)

## 这里有什么

知识库围绕三个相互关联的问题展开：量化怎样影响模型，方法如何控制误差，以及低比特表示怎样在实际系统中执行。

| 内容 | 可以了解什么 | 代表页面 |
| --- | --- | --- |
| 量化基础 | scale、zero-point、舍入、裁剪和分组之间的关系，以及误差从哪里产生 | [均匀量化与分组](wiki/fundamentals/quantization/uniform-quantization-and-groups.md) |
| 算法原理 | 不同方法保护什么、优化什么，关键推导与适用条件是什么 | [AWQ](wiki/methods/awq.md)、[GPTQ](wiki/methods/gptq.md)、[SmoothQuant](wiki/methods/smoothquant.md) |
| 多模态量化 | 图像与文本如何进入模型，视觉编码器、连接器和语言层分别面临什么量化问题 | [视觉语言模型中的 token 与量化对象](wiki/fundamentals/model/vision-language-model-tokens-and-quantization.md) |
| KV cache 量化 | 键和值的误差传播、量化粒度，以及逐 token 追加带来的系统约束 | [KV cache 量化的对象与粒度](wiki/theory/kv-cache-quantization-objects-and-granularity.md) |
| 实现与部署 | 量化脚本生成什么权重格式，打包与元数据如何组织，推理框架怎样加载和选择内核 | [AWQ：从量化脚本到 vLLM 与 SGLang](wiki/implementation/awq-implementation.md) |
| 底层 Kernel | 4 位权重怎样反量化，张量布局、线程配置和模板分派有什么约束 | [AWQ 与 Marlin 反量化内核](wiki/implementation/weight-only-dequant-kernels.md) |
| 诊断与评测 | 怎样区分图变换、量化误差和后端实现问题，精度与性能指标分别说明什么 | [量化误差诊断与验证](wiki/implementation/quantization-error-diagnosis.md) |

此外，Wiki 还包含可学习变换、旋转量化、混合精度、多模态量化方法，以及 GPU 执行、数据格式和部署后端等内容。完整主题与阅读路径见 [Wiki 导航](wiki/INDEX.md)。

## 怎样阅读

**从基础开始学习**，可以先读均匀量化与分组，再进入 AWQ、GPTQ 或 SmoothQuant，沿页面中的链接补充校准、等价变换和误差重构等前置知识。

**研究一种方法**，可以从方法页了解问题、机制、推导和实验依据，再读相关比较页及实现分析。共同机制有独立的知识页面，便于把不同算法放在一起理解。

**关注部署与加速**，可以从实现页追踪量化产物、框架加载与内核调用，再深入数据格式、布局和 GPU 执行。以 AWQ 为例，方法页、实现页和反量化内核页分别连接算法设计、部署路径与底层执行。

## 项目如何组织

`wiki/` 是主要阅读内容，按知识主题组织；`raw/` 保存论文、文章、文档和固定版本代码等来源。知识页面将来源中的机制、实现和证据整理为相互关联的解释，读者可以沿知识关系继续阅读，也可以根据来源定位回查依据。

Wiki 以独立可读为设计目标：必要解释、图示、条件和来源身份随知识保留，理解内容不依赖某次对话或临时实验环境。后续研读和实践产生的新认识会补充或修正已有页面。

| 位置 | 内容 |
| --- | --- |
| [wiki/](wiki/INDEX.md) | 知识页面与阅读导航；规范见 wiki/README.md，变动见 wiki/LOG.md |
| [raw/](raw/README.md) | 来源材料、来源说明与收录规范 |
| [.agents/skills/](.agents/skills) | 辅助知识库维护和工程实践的任务方法 |
| [AGENTS.md](AGENTS.md) | Agent 面对本项目时的协作约定 |

## 建设进展

目前可阅读的内容以文献研读和固定版本源码分析为主。上面的 AWQ 部署路径与反量化内核页面提供代码分析，尚未完成对应的本地运行、数值验证或性能测量。各页分别说明来源报告、推导、代码核对和实际实验的范围，具体实现与支持条件也受所分析版本限制。

项目长期服务于量化科研和工程实现，内容随实际问题逐步扩充。
