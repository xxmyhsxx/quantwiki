# Wiki 维护与已有实验材料

本目录保留量化 Wiki 的建设计划、审查记录和已有实验材料。研究知识页面位于 [wiki/research/](../wiki/research/)，用于解释问题、综合机制、连接方法并保存有依据的研究认识。

## 当前内容

| 目录 | 内容与职责 |
| --- | --- |
| `plans/` | Wiki 资料入库、内容审查和维护安排。 |
| `reviews/` | 来源与 Wiki 审查的依据、结论和待处理事项。 |
| `experiments/` | 已有实验任务的代码与说明，保留原位置和既有引用。 |

当前入口包括：

- [资料 ingest 计划](plans/2026-10-11-ingest-batches.md)。
- [Wiki 审查安排](plans/2026-10-13-wiki-review-batches.md)。
- [raw 审查记录](reviews/2026-10-05-raw-review.md)。
- [Wiki 重复性审查](reviews/2026-10-13-wiki-repetition-review.md)。
- [AWQ W4A16 已有实验任务](experiments/2026-10-03-awq-w4a16/README.md)。实验的运行状态与验证范围以该任务记录为准，代码存在不代表模型、质量或性能已经通过。

## 与应用工作区的边界

新的应用代码、部署配置、依赖环境、实验记录与运行结果由对应应用或实验工程承接。Wiki 的研究页保存可复用认识，并保留支撑结论的项目版本、条件和证据位置。

本目录的维护计划和审查记录继续服务 Wiki。已有实验材料保留，不因能力包设计而删除、重跑或移入未指定的项目。后续确需迁移时，先确定接收工程，再核对正文、来源登记和导航引用。

根目录 `research/` 不随 [量化 Wiki 能力包](../.agents/skills/use-quant-wiki/README.md) 外派。应用通过包内 Skill 使用知识和方法，通用经验通过反馈交给 Wiki 维护端整理。
