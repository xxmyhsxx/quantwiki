---
name: use-quant-wiki
description: "使用量化 Wiki 回答问题、研究方案或开发 GPU 算子。用户明确要求向项目接入、安装或外派时，安装完整能力包并核验。"
---

# 使用量化 Wiki

## 任务

根据任务选择包内能力，完成后统一回传反馈。

## 输入

当前任务、应用目录及目标约束。应用代码、环境和运行记录属于应用项目；知识库、源包、反馈目录以 `package.json` 的绝对绑定为准，不能从应用目录猜测。当前 Wiki 为 `/Users/xiao/Documents/Projects/quantwiki`。

## 原则

准备、安装、发布和反馈的核验由工具完成。成功后不再追加 `status`、`verify`、目录扫描或清单比对；只有失败或结果矛盾时定向排查并报告阻塞。

## 工作步骤

### 1. 准备本次版本

同任务已有快照时继续使用。新任务将入口所在目录设为 `quantwiki_package`，在应用任务目录中选一个尚不存在的快照绝对路径，设为 `quantwiki_snapshot`：

```sh
python3 -B "$quantwiki_package/scripts/package.py" prepare --output "$quantwiki_snapshot"
```

有更新授权才加 `--allow-update`。工具核验完整包、执行已授权的干净镜像更新并建立快照；已有定制和冲突受保护，更新保留旧包。源端不可达、未发布或未获更新授权时，按返回结果使用有效本地版本并说明限制。

成功后使用返回的 `snapshot` 路径，记录版本、内容标识和限制，读取快照入口。执行期间不切换版本；内部规范及辅助材料都从该快照读取。

### 2. 使用所需能力

按问题读取下列规范，路径相对于任务快照，不默认加载全部：

| 当前需要 | 内部规范 |
| --- | --- |
| 定位、解释、比较或分析已有知识 | [wiki-query](skills/wiki-query/SKILL.md) |
| 查证、推导、研究或验证方案 | [wiki-research](skills/wiki-research/SKILL.md) |
| 实现或优化 CUDA/Triton 算子 | [wiki-kernel-writing](skills/wiki-kernel-writing/SKILL.md) |

知识读取交给 query，不预读整份 INDEX、SELF、READING 或 LOG。查询不读 raw；研究核验来源时明确范围和版本。

公共知识和共享方法的修订交维护端，不调用未下发的维护 Skill。算子知识在迭代前准备，执行中不追加外部 Wiki 查询。

### 3. 回传反馈并交付

任务结束时整合内部能力的结果，只提交一份反馈；续接不重复提交，`task_id` 本身不会去重。使用应用已有记录位置保存 JSON，必需字段为：

- `task_id`、`task_summary`、`capabilities_used`；
- `knowledge_locations`：已读知识的绝对路径数组，可空；
- `result_summary`。

按实际补充限制和建议；实验结论保留代码版本、配置和证据位置，不复制完整业务代码或对话。将 JSON 绝对路径设为 `quantwiki_feedback_input`，执行：

```sh
python3 -B "$quantwiki_snapshot/scripts/package.py" feedback --input "$quantwiki_feedback_input"
```

工具新建唯一的反馈记录，不覆盖已有日志。失败时交付反馈文本并说明未回传，不换目录写入。反馈不会自动修改 Wiki 或源 Skill。

### 接入：明确要求时安装

读取接收方协作约定，确认可访问的接收项目绝对路径，设为 `quantwiki_project`，将 `quantwiki_package` 设为实际包目录：

```sh
python3 -B "$quantwiki_package/scripts/package.py" install --project "$quantwiki_project"
```

共享接入改用 `install --skills-dir`，传入已确认的 Skill 目录，二选一；不猜全局路径。仓库 URL 或服务器地址尚未对应接收目录时，说明缺少的路径或连接信息。

工具创建目录并安装完整包，项目位置为 `.agents/skills/use-quant-wiki/`，共享位置为 `<Skill目录>/use-quant-wiki/`。相同内容可复用；有更新授权才加 `--allow-update`，保留旧包备份。来源冲突、定制或接收位置不明时保留并报告，不接管、不初始化项目 Git 或改写项目说明。

依据返回结果交付位置、版本、安装记录和备份；知识访问另按目标环境确认。纯安装至此结束，不生成使用反馈；同时有应用任务时，从返回的安装路径准备任务。

### 维护：发布源包

维护者按 skill-creator 同步设计与执行，检查全部能力、依赖和引用。将 `quantwiki_package` 设为绑定源包，`quantwiki_version` 设为大于当前版本的正整数：

```sh
python3 -B "$quantwiki_package/scripts/package.py" publish --version "$quantwiki_version"
```

首次发布使用 `--metadata` 提供含 `schema_version: 1`、名称、绑定和能力的 JSON；后续沿用清单。保留三项能力与实际绑定，包内文件变化才升级包。使用端不能发布未知草稿。按 Wiki 约定记录结果；发布不自动安装或推送 Git。

## 输出

先回答问题或给出成果，再补关键依据、条件与限制。使用自然、具体的语言，解释必要术语；删去重复结论、空泛评价和不影响用户判断的内部流程。所用版本与能力简记，详细反馈和实验记录提供位置；算子结果按内部规范交付。

## 校验

核对答案与实际结果、版本和路径一致，区分安装成功、知识可访问和应用实测。发出前删掉不影响理解、判断或行动的句子。
