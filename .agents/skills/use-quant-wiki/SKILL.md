---
name: use-quant-wiki
description: "使用量化 Wiki 支撑量化、模型推理、部署、GPU 算子和评测任务，直接读取原 Wiki，并按需调用包内知识使用、问题研究与算子编写能力；用户提供外部项目地址并要求接入、安装或外派时，实际创建目标目录、安装完整能力包并核验。"
---

# 使用量化 Wiki

## 任务与输入

你是量化 Wiki 能力包的统一入口。根据当前任务使用原 Wiki 和包内方法，完成应用问题，结束后统一回传一份反馈；收到明确的外派要求时实际安装完整包。

从对话和应用工作区取得目标、约束、交付物及已有观察。应用的代码、GPU、服务器与结果路径由当前任务确定。知识库、源包和反馈目录以 `package.json` 的绑定为准；当前知识库是 `/Users/xiao/Desktop/量化`，不要从应用目录猜测 Wiki 位置。

## 工作步骤

### 1. 一次准备本次使用的版本

将 `quantwiki_package` 设置为本入口所在目录的实际绝对路径。

**外派请求：** 取得可读写的接收项目绝对路径，读取其已有协作约定，将路径设置为 `quantwiki_project`，执行：

```sh
python3 -B "$quantwiki_package/scripts/package.py" install --project "$quantwiki_project" --allow-update
```

用户明确要求接入并跟随源包时，同一授权覆盖干净镜像的更新，无需每次再问。工具负责创建缺少的目录、完整复制、保留项目定制内容并核验，默认安装到项目的 `.agents/skills/use-quant-wiki/`。仓库 URL 或服务器地址需要对应的接收路径及访问方式；缺少时说明具体缺项。仅外派时交付实际安装路径、版本和知识可访问性，至此结束，不制造使用反馈；同时需要应用任务时，将 `quantwiki_package` 设为返回的安装目录继续准备。

**应用任务：** 同一任务已有快照时继续使用原快照。新任务在应用已有任务目录中选择尚不存在的快照绝对路径，设置为 `quantwiki_snapshot`，执行一次：

```sh
python3 -B "$quantwiki_package/scripts/package.py" prepare --output "$quantwiki_snapshot" --allow-update
```

工具完成版本检查、已授权的干净镜像更新和完整快照。成功后将返回的 `snapshot` 路径设为 `quantwiki_snapshot`，记录版本及必要限制；失败时按返回原因交付具体阻塞。源端无法确认最新时，可用本地发布包的限制由工具说明。

完整性检查由工具负责。正常结果不再追加 `status`、`verify`、目录遍历或清单对比；更新选用了新入口时，读取所选快照的入口以取得当前规范。只有失败或结果明显不一致时才定向排查。任务执行期间不切换快照。

### 2. 使用所需内部能力和原 Wiki

按当前问题读取以下规范，路径相对于所选快照，不默认加载全部：

| 当前需要 | 内部规范 |
| --- | --- |
| 定位、解释、比较或分析已有知识 | [wiki-query](skills/wiki-query/SKILL.md) |
| 解释现象、比较路线、设计或验证方法 | [wiki-research](skills/wiki-research/SKILL.md) |
| 实现或优化 CUDA/Triton 算子 | [wiki-kernel-writing](skills/wiki-kernel-writing/SKILL.md) |

直接读取绑定 Wiki 的根 `SELF.md`、`wiki/INDEX.md` 及相关正文，按需读取 `wiki/README.md` 和必要关联。已读且仍有效的内容可以复用。普通知识使用不读 `raw/`；研究需要核验来源时明确范围与版本。结论基于实际正文，保留来源条件和未核实范围。

应用代码和运行记录归应用工作区。公共知识和共享方法修订交维护任务；应用不调用未下发的维护 Skill。算子执行遵循包内完整验证流程，在迭代循环中不追加外部 Wiki 查询。

### 3. 统一反馈并交付

用户任务结束后整合内部能力的依据、结果和改进信息，只提交一份反馈。同任务续接不重复反馈。反馈 JSON 使用应用已有任务记录位置，必需字段为 `task_id`、`task_summary`、`capabilities_used`、`knowledge_locations`（已读知识绝对路径数组，可空）和 `result_summary`；限制与建议按实际填写，实验保留代码版本、配置和证据位置。

将 JSON 的实际绝对路径设置为 `quantwiki_feedback_input`，通过所选快照的工具提交一次：

```sh
python3 -B "$quantwiki_snapshot/scripts/package.py" feedback --input "$quantwiki_feedback_input"
```

工具负责唯一新建、写入和核验反馈。成功后不再回读复查；失败时交付反馈文本并说明未回传。反馈不自动修改 Wiki 或源 Skill，不复制完整业务代码或对话。

交付应用所需答案、方案、实现或验证进展，说明关键依据和实际限制。区分安装、知识访问与应用实测，算子结果同时满足内部规范的交付要求。
