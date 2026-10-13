# 量化知识库

面向大语言模型和多模态模型的量化研究知识库，以知识页面连接算法原理、代码实现、部署与推理加速。内容来自论文、技术文档和源码分析，用于系统学习、研究判断和工程参考。

**[浏览 Wiki](wiki/INDEX.md)** · [定位、边界与当前画像](SELF.md) · [来源资料](raw/README.md)

## 这里有什么

| 主题 | 内容与代表入口 |
| --- | --- |
| 量化基础 | scale、舍入、裁剪、分组与误差：[均匀量化与分组](wiki/fundamentals/quantization/uniform-quantization-and-groups.md) |
| 算法原理 | 方法的目标、机制、推导与条件：[AWQ](wiki/methods/awq.md)、[GPTQ](wiki/methods/gptq.md)、[SmoothQuant](wiki/methods/smoothquant.md) |
| 多模态量化 | 视觉 token、模型模块与模态差异：[VLM 量化对象](wiki/fundamentals/model/vision-language-model-tokens-and-quantization.md) |
| KV cache | K/V 误差、粒度与缓存约束：[对象与粒度](wiki/theory/kv-cache-quantization-objects-and-granularity.md) |
| 实现与部署 | 量化产物、框架加载与内核调用：[AWQ 部署路径](wiki/implementation/awq-implementation.md) |
| GPU 与 Kernel | 低比特执行、布局与并行分工：[反量化内核](wiki/implementation/weight-only-dequant-kernels.md) |
| 诊断与评测 | 误差定位、质量指标与性能口径：[误差诊断](wiki/implementation/quantization-error-diagnosis.md) |

完整主题与阅读路径见 [Wiki 导航](wiki/INDEX.md)。

## 怎样阅读

- **学习基础：** 从均匀量化与分组进入经典方法，沿链接补充必要前置知识。
- **研究方法：** 先读方法页，再看共同机制、比较与实现分析。
- **理解部署：** 沿量化产物、框架加载、Kernel 和 GPU 执行逐层阅读。

## 项目如何组织

`wiki/` 组织知识，`raw/` 保存可追溯的来源。页面写作与引用规则见 [Wiki 规范](wiki/README.md)，资料保存规则见 [raw 规范](raw/README.md)。

`wiki/research/` 保存研究问题、机制综合、方法联系和有依据的结论。根目录 [research/](research/README.md) 保留 Wiki 维护计划、审查记录和已有实验材料；新的应用代码、部署配置与运行结果由对应应用工程承接。

项目职责、覆盖情况与下一步由 [SELF](SELF.md) 说明；Agent 协作约定见 [AGENTS](AGENTS.md)，具体任务方法见 [Skills](.agents/skills/)。

## 对外能力包

本项目通过 [use-quant-wiki](.agents/skills/use-quant-wiki/SKILL.md) 发布一个完整能力包：一个入口 Skill 按任务选择包内的知识使用、实际问题研究和算子编写能力。应用接收整个包的镜像，直接读取原 Wiki，在自己的工作区执行方法，并向 Wiki 回传使用反馈。

用户提供外部项目地址并要求接入时，入口实际创建目标 Skill 目录、复制完整能力包并核验安装，交付明确的目标路径和结果。外派能力包保留原 Wiki 的知识绑定，不复制知识正文。

整个包统一管理发布版本和内容校验。知识页面更新不触发包升级；应用开始任务前核对包状态，执行过程中固定所用版本。

入口、包内三项能力及发布安装工具已实现，版本和完整文件清单见 [包清单](.agents/skills/use-quant-wiki/package.json)，使用与维护方式见 [能力包设计](.agents/skills/use-quant-wiki/README.md)。当前工作在本仓库完成；外派和更新已用仓库内临时目标验证，尚未安装到其他真实项目或用户共享目录，也未开展模型部署或 GPU 实测。
