---
name: wiki-kernel-writing
description: "在需要将 PyTorch 或高层算子实现为 CUDA/Triton Kernel，或针对指定 GPU、workload、dtype 优化已有 Kernel 时使用。按基线建立、编译、正确性验证、性能测量、按需 Profiling、假设驱动迭代和最终验证的流程工作，保留可追溯实验及当前最佳已验证实现。正常运行时不查询外部 Wiki。"
---

# 算子编写

## 任务

你负责完成 CUDA/Triton GPU 算子的开发与迭代优化。在给定目标硬件、workload、正确性约束、性能指标和优化预算的前提下，通过可复现的实验循环，交付本次探索中找到的最佳已验证实现，而不是只交付能够运行的代码。

适用于从 PyTorch 或其他高层实现开发 Kernel、优化已有 CUDA/Triton Kernel，以及算子融合、调度、数据布局和执行策略优化。任务包含实际编译、正确性检查、Benchmark，以及必要的 Profiling 和多轮迭代。

不自动用于概念解释、代码阅读或总结、没有实际运行与测量的静态评审，以及不涉及 Kernel 的纯框架层、模型层或业务层优化。缺少可执行环境和可信 Reference，且任务也不包含补齐这些条件时，不启动完整开发闭环。

本文件定义开发流程、验证门槛、实验记录和停止条件。具体优化知识由本 Skill 实际随附的 `references/` 提供；外部 Wiki 仅服务于离线知识积累和后续 Skill 迭代。

## 输入

从用户任务、当前仓库、已有测试和运行环境中确认以下信息。已有等价配置或记录时直接复用，不要求用户重新填写，也不为遵守 Skill 创建重复文件。

| 信息 | 需要确认的内容 |
| --- | --- |
| 算子与参考 | 算子语义、可信 Reference implementation、输入输出、调用入口和集成要求。 |
| 执行目标 | target GPU、backend、可用的编译与运行环境。 |
| Workload | 必须支持的 shape、动态维度、dtype、layout 和重要边界条件。 |
| 正确性约定 | 对照方式、测试范围、数值语义和 correctness tolerance。 |
| 性能约定 | Baseline、主评测指标、目标（如已指定）、正式评测入口与统一 Benchmark 协议。 |
| 优化预算 | Candidate 数量、Profiling 次数、GPU 时间或用户指定的其他限制。 |
| 既有积累 | 当前实现、实验历史、best candidate，以及实际存在的随包 References 和脚本。 |

对当前任务必要但尚不明确的信息标记为 `UNKNOWN`，优先从上述材料和环境中解析。仍无法确定时，说明具体缺项及其阻塞的步骤；不得自行猜定目标硬件、必需 workload、误差要求、比较对象或预算。

## 原则

1. **正确性先于性能。** 每个 Candidate 必须依次通过 `Compile → Correctness → Benchmark`。编译或正确性未通过时，不进入性能测试，不作性能结论，也不能成为 best candidate。

2. **结论以真实测量为依据。** “更快”“更慢”“达到目标”“接近硬件上限”或“优化有效”必须有对应的 Benchmark 或 Profiler 证据。经验与诊断可以形成假设，不能替代实验；事实、推测和未解决问题必须区分。

3. **保持评测条件可比。** 不得静默改变 GPU、shape、dtype、layout、tolerance、warmup、测量次数、计时方式、Baseline 或性能指标。经任务要求改变评测条件时，明确记录并重新建立可比结果，不能把新协议结果直接与旧数据混比。

4. **实验与实现可追溯。** 保存每个有意义的 Candidate 的实现版本、parent、改动、假设、验证结果和结论，包括编译失败、错误、无效和性能回退的实验。每个阶段结束即更新记录，不到任务结束时凭记忆补写，也不覆盖已经对应测量结果的旧实现。

5. **以假设驱动迭代，保护 best-so-far。** 初始实现优先简单、正确、可验证；后续每个 Candidate 默认检验一个主要性能假设。组合修改必须说明各项变化和组合理由。`latest candidate` 不等于 `best candidate`；只有通过编译与正确性、满足全部 required workload，并按统一协议证实目标指标更优的 Candidate 才能替换 best。

6. **保持运行时闭包。** 正常执行仅使用当前 `SKILL.md`、本 Skill 实际随附的 `references/` 和 `scripts/`、当前仓库代码与文档、当前任务和实验记录，以及本地可用的编译、测试、Benchmark、Profiler 工具。不查询外部 Kernel Wiki，也不以材料不足为由静默扩大运行时依赖。References 在相关步骤按需读取，不默认全部加载；不存在的文件或工具不能当作可用依赖。无法继续时保留当前 best verified result，记录 unresolved issue，留待离线迭代。

## 工作步骤

按以下主线执行。每次产生新的实现版本，都重新经过对应验证门槛；检查失败时按该步骤的分支处理。

```text
确认任务与环境 → 理解 Reference → 建立 Baseline
→ 设计并实现 Candidate → Compile → Correctness → Benchmark / 更新 best
→ 检查停止条件
   ├─ 继续：按需 Profiling → 诊断与假设 → 下一 Candidate → 重新验证
   └─ 结束：最终 canonical evaluation → 交付结果与限制
```

### Step 1：确认任务、环境与评测约定

读取当前任务、仓库入口、已有配置和实验记录，核实“输入”中的必要信息，确认哪些条件已知、哪些仍是 `UNKNOWN`。优化已有实现时，同时确认当前实现及已有 best 的代码位置和验证记录。

将任务目标、全部 required workload、正确性约定、Baseline、性能指标、评测协议与预算记录在项目已有的任务载体中。多个 workload 同时重要时，明确其要求以及如何判断整体改进，不自行把某一个 shape 的提速当作总体成功。

确认可用的编译、测试和 Benchmark 入口；后文的 **canonical evaluation** 指本任务约定的正式评测入口与协议，不另选一套更有利的测试。确认可用的 Profiler、随包 References 和脚本，但不要因此全量读取辅助材料。

Reference、目标 GPU、required workload、正确性约定、Baseline 和性能指标足够明确后，进入 Step 2。必要条件缺失时，先在任务范围内补齐；无法补齐则记录阻塞，不绕过条件宣称开发或验证完成。

### Step 2：理解 Reference 与算子语义

阅读 Reference implementation、调用入口、测试，以及相关 build/runtime 代码。确认数学语义、输入输出、broadcasting、reduction、动态维度、dtype/layout、边界情况和框架集成要求。

区分“Reference 定义要算什么”和“Kernel 应当怎么实现”。不得机械翻译 Reference 的执行结构，也不能为了便于优化改写必须保持的数值语义。

在任务记录中明确：输入是什么、输出是什么、哪些语义必须保持、哪些 workload 必须支持。这些内容能够指导实现和测试后，进入 Step 3；仍有影响正确性的歧义时，先回到任务材料或测试中核实。

### Step 3：建立可比较的 Baseline

使用任务约定的 Baseline。它可以是 PyTorch eager、`torch.compile`、已有 CUDA/Triton Kernel、库实现或项目指定实现；不要自行换成更容易超过的比较对象。

先按正确性约定验证 Baseline，再按正式 Benchmark 协议测量所有 required workload。正确性检查和性能测量分别遵守 Step 6、Step 7 的方法；任一环节失败时，先处理问题，不带着无效基线进入性能比较。

记录 Baseline 实现位置、硬件、shape、dtype/layout、正确性结果、latency/throughput 等约定指标，以及 Benchmark 协议和实际执行入口。

Baseline 正确性确认且性能测量有效后，进入 Step 4。Baseline 与 best 的职责保持区分：Baseline 是比较基准，best 是当前满足任务约束的最佳已验证实现。Baseline 也满足 Candidate 的交付约束时，可以同时作为初始 best；尚无合格 Candidate 时如实记录，不虚构 best。

### Step 4：设计并实现当前 Candidate

首次实现时，根据 Reference 语义与 backend 选择简单、可编译、可验证、可 Benchmark 的实现，不一次加入大量复杂优化。需要具体实现或优化知识时，只读取当前 Skill 已有且与该问题有关的 Reference，并将其用于本轮设计。

后续迭代使用 Step 9 形成的假设、parent 和 planned change，不在实现过程中加入与本轮假设无关的修改。

为 Candidate 分配 `v001` 或项目等价标识，记录 parent、代码位置、设计说明或性能假设，以及计划改动。实现完成后，将状态从 `planned` 更新为 `implemented`；若包含多个行为变化，逐项记录。

保存可追溯的实现后进入 Step 5。实现版本与其验证记录必须对应，后续代码修改不能沿用修改前的通过状态。

### Step 5：通过 Compile Gate

使用已确认的构建入口编译当前 Candidate，保存执行入口、结果和必要错误信息。

- **通过：** 状态更新为 `compiled`，进入 Step 6。
- **失败：** 保存编译错误并定位原因，修复后重新编译；不进入 Correctness 或 Benchmark，不创建性能结论。

仅修复编译问题、且未改变主要性能假设时，可以沿用 Candidate ID，但要保留修复记录。若结构或主要假设发生实质变化，回到 Step 4 建立可区分的实现记录。无法修复或触及停止条件时，按 Step 10 处理，不无限重复构建。

### Step 6：通过 Correctness Gate

将已编译 Candidate 与可信 Reference 对比，执行任务约定的测试。按适用范围覆盖代表性、边界和不规则 shape、随机输入、重要数值范围及支持的 dtype；全部 required workload 均须满足正确性约定。

记录 `PASS / FAIL`、实际检查范围、最大绝对误差和最大相对误差（适用时），以及失败 workload 和对应输入条件。不能只保留总通过标记而丢失失败位置。

- **通过：** 状态更新为 `verified`，进入 Step 7。
- **失败：** 记录 `incorrect`，优先修复正确性，修复后重新进入 Step 5；不进入 Benchmark，也不擅自放宽 tolerance。

修复导致实现结构明显变化时，创建新的 Candidate 或明确区分本轮变化及其验证结果。无法修复或已触及停止条件时，进入 Step 10，保留之前的已验证结果。

### Step 7：执行 Benchmark 并判断是否更新 best

只对 `verified` Candidate 执行性能测试。使用与 Baseline、当前 best 一致的正式协议，包括 warmup、重复执行、必要的同步和计时汇总。优先使用协议约定的稳健统计量，例如 median，不在比较过程中静默切换统计方式。

测量所有 required workload，记录 Candidate、shape、dtype/layout、latency/throughput、Baseline、speedup、重复次数和协议。确认本轮测量有效后，状态更新为 `benchmarked`。

根据测量结果处理：

| 结果 | 动作 |
| --- | --- |
| 尚无 best | 检查当前 Candidate 是否满足全部任务约束。满足时登记为首个 best，并保留与 Baseline 的比较；不因“首次验证”而声称已超过 Baseline。 |
| 确实优于 best | 仅在全部 required workload 满足要求、统一协议下目标指标更优时替换 best，记录具体改进。 |
| 与 best 接近 | 在预算允许时重复测量，判断差异是否属于 noise；不能确认改进时不替换 best。 |
| 正确但性能回退 | 标记 `regression`，保留实现和实验结论，best 不变；分析该实验提供的信息。 |
| 测量无效或条件不可比 | 保存原因，修复评测问题或重新建立可比数据，不用本次数字作性能结论。 |

`accepted` 可以表示成为新的 best，也可以表示保留为有效改进；始终另外明确当前 best 的标识，不把记录为 accepted 自动等同于更新 best。

更新实验历史后检查 Step 10。仍需优化时进入 Step 8；准备结束时进入 Step 11。下一条无依赖关系的新假设默认从 best 出发，而不是在性能回退的 latest 上继续堆叠修改。

### Step 8：判断是否需要 Profiling，并收集必要证据

不要机械地 Profile 每个 Candidate。满足以下任一条件时，应进行针对性 Profiling：

- 第一版正确 Candidate 明显低于目标，或连续两个有效 Candidate 都未改善 best。
- 出现无法仅凭 Benchmark 解释的明显回退，或当前瓶颈不明确。
- 存在多个竞争假设，Benchmark 无法区分；或怀疑资源利用、内存行为、occupancy、stall 等硬件行为存在问题。

若 Compile、Correctness 或 Benchmark 已足够定位问题，直接进入 Step 9，不增加无必要的 Profiling。需要 Profiling 但工具、依赖或剩余预算不支持时，记录缺口；已有证据仍足以形成可验证假设时使用现有证据，否则按 Step 10 处理。

执行 Profiling 时，优先使用项目已有 Profiler 或当前 Skill 实际随附的脚本。按待解释的问题选取指标，例如 kernel duration、SM utilization、memory throughput、cache behavior、occupancy、register/shared memory usage、warp stall、tensor-core utilization 和 launch configuration，不要求每次收集全部指标。

需要理解指标或诊断方法时，按需读取已有的相关 Reference；不把某个指标直接等同于固定代码修改。

保存可复核的原始输出，在上下文中保留与问题相关的紧凑指标摘要。有可用摘要脚本时使用；没有时直接整理必要指标，不假设存在 summarizer 工具。记录 Profiling 对应的 Candidate、workload 和执行入口后进入 Step 9。

### Step 9：诊断、形成假设并选择下一 Candidate

先区分观测事实与诊断。依据 Benchmark、可用 Profiler 指标和实验历史，说明测量到了什么、具体证据是什么、可能对应什么瓶颈、哪些仍不确定；不要看完 Profiler 就直接改代码。

创建下一 Candidate 前，在当前实验记录中写清以下内容；可以复用项目格式，不要求另建假设文件：

| 内容 | 要回答的问题 |
| --- | --- |
| Observation / Evidence | 当前可观测问题是什么，哪些具体数值或现象支持它？ |
| Diagnosis / Uncertainty | 可能的瓶颈是什么，还有哪些解释无法排除？ |
| Hypothesis / Planned change | 准备通过什么可执行修改验证哪一个主要判断？ |
| Expected effect | 修改后预期哪个可测指标发生怎样的变化？ |
| Possible tradeoff | 可能引入什么代价，或影响哪些其他 workload？ |

有效假设至少包含可观测问题、具体证据、可执行修改和预期变化的可测指标。“优化内存”“试试更大的 block”“让它更快”不构成完整假设。

检查 Experiment History，避免重复等价失败方案。只有周边实现发生实质变化、组合条件不同、上次测量无效或受噪声影响、workload 改变，或上次仅验证部分条件时，才重新尝试，并说明本次差异。

默认一个 Candidate 只验证一个主要假设。多个修改无法独立验证、已有实验证明需要组合，或组合本身就是明确假设时，允许共同测试，但必须列出修改及组合理由。

默认选择 best 为 parent。只有假设明确依赖其他 Candidate 的结构、正在检验组合优化、该分支尚未完成验证，或有充分实验证据支持继续探索时，才从非 best 出发，并记录理由。

多个合理且独立的假设可以分支探索。依次考虑证据支持、能否消除当前最大不确定性、预期收益、验证成本和与既有实验的重复程度；每个分支仍须完整通过三道验证门槛。

假设和 parent 明确、预算允许时，回到 Step 4 创建下一 Candidate。无法形成合格假设时，先补充必要测量或返回 Step 8；现有 Skill 与工具仍无法推进时，记录 unresolved issue，进入 Step 10，不盲试参数或转向外部 Wiki。

### Step 10：检查停止条件

每轮记录更新后检查，编译、正确性或环境问题导致无法继续时也适用。以下任一条件成立即可结束优化，但必须记录依据：

| 条件 | 必须记录的依据 |
| --- | --- |
| 达到目标 | Candidate 正确性通过，实测指标达到预定义目标。 |
| 收敛 | 连续多轮有依据的实验不能改善 best，以及这些实验与结论；不能把随便尝试几次未提升称为收敛。 |
| 预算耗尽 | 已达到的 Candidate 数量、Profiling 次数、GPU 时间或其他约定限制。 |
| 接近硬件或实现边界 | 支持“继续优化预期收益很小”的实际测量证据，不能仅凭感觉判断。 |
| 当前无法继续推进 | 缺失硬件能力或依赖、无法解释的 compiler/runtime 行为、References 未覆盖的特殊问题，或无法形成可验证假设。 |

尚未达到停止条件时，继续 Step 8 或当前失败步骤的修复分支。决定结束时保留 best verified result 和 unresolved issue，进入 Step 11；“停止搜索”不等于“最终验证通过”。

### Step 11：执行最终 canonical evaluation

使用当前 best 的确切实现版本，在约定目标硬件和正式协议下重新执行正确性验证与 Benchmark。全部 required workload 均需重新验证，不能仅挑选表现最好的 shape，也不能直接用历史最佳数字替代最终结果。

核对最终使用的代码、配置、workload 与报告一致，保存执行入口、结果和必要日志。

- **全部通过且结果有效：** 以本次最终测量确认交付结果，按“输出”整理实现、证据、限制和结论。
- **正确性失败或结果不可复现、不可比较：** 记录失败，回到对应步骤定位；预算或环境不允许继续时，交付当前代码与已有证据，明确其未通过本次最终验证。
- **不存在合格 Candidate，或无法完成最终评测：** 明确失败或阻塞阶段，不把未经最终 canonical evaluation 的 Candidate 声明为最终已验证结果。

## 输出

### 运行过程中维护的记录

复用项目现有文件或记录机制，保证任务状态、实验历史和 best 状态可查即可，不强制创建四份独立状态文件或另一套进度检查表。

**任务状态**应包括已确认的任务目标、Reference、目标环境、workload、正确性和性能约定、Baseline、预算，以及仍为 `UNKNOWN` 的信息。

**Experiment History**随各步骤更新。每个有意义的 Candidate 至少保留以下信息；未执行的项目明确标为未执行，不填造结果：

| 字段 | 记录内容 |
| --- | --- |
| `id / parent / implementation` | Candidate 标识、来源及可恢复的代码版本或位置。 |
| `status / change_summary` | 当前执行阶段和改动，失败或回退状态。 |
| `hypothesis` | 初始设计说明，或证据、诊断、主要假设、计划修改、预期效果及取舍。 |
| `compile` | 是否通过、执行入口、错误及修复记录。 |
| `correctness` | 测试范围、PASS/FAIL、适用误差、失败 workload 及证据位置。 |
| `benchmark` | 协议、重复次数、逐 workload 指标、Baseline 比较、speedup 及测量有效性。 |
| `profile` | 是否执行、执行入口、关键指标摘要和原始输出位置。 |
| `conclusion` | 假设得到支持、被否定或仍不确定；是否更新 best；下一步及其依据。 |

阶段可沿用项目等价状态；使用原有状态时按以下顺序记录，未通过的阶段不能跳过：

```text
planned → implemented → compiled → verified → benchmarked
                                              → profiled（按需）
```

本轮结论使用 `accepted / regression / rejected` 或项目等价表达，并区分“仍待验证”和“已经淘汰”。这些结论不替代独立的 best 标识；编译失败、incorrect 和尚未执行的后续阶段必须如实保留。

**Best Candidate State**明确当前 best 的标识、对应实现、正确性与性能证据。没有 best 时明确记录；有 best 时，不因 latest 更新、失败或回退覆盖它。

### 最终交付

交付实际实现或可定位的代码修改、复核所需的测试与 Benchmark 入口，以及实验记录位置。报告至少包含以下内容；使用项目已有等价格式即可：

| 部分 | 内容 |
| --- | --- |
| 目标 | Operator、GPU、backend、required workload、dtype/layout、指标与约束。 |
| Baseline | 实现、正确性结果、逐 workload 性能和评测协议。 |
| Best Candidate | 标识、实现位置、最终正确性结果、逐 workload 性能、speedup，以及是否完成 canonical evaluation。 |
| 关键改动 | 实际做了什么，分别验证了哪个假设。 |
| 关键实验 | Candidate、Hypothesis、Result、Conclusion，包含必要的失败或回退实验。 |
| 结束依据 | 达标、收敛、预算或阻塞等停止原因及证据。 |
| 边界与后续 | 剩余瓶颈、未解决问题、适用限制和有依据的后续方向。 |

报告明确区分实际测量确认的结论、推测和 unresolved issue。只完成部分阶段时，如实交付已有代码与证据，说明未完成项，不宣称已获得最终已验证 Kernel 或性能提升。

## 校验

结束前结合代码、实际日志和实验记录逐项核对，不仅检查文档是否填写：

- [ ] Reference 语义、目标 GPU、backend、全部 workload、dtype/layout 和 correctness contract 已核实；未决项及影响已说明。
- [ ] Baseline 正确性与性能测量有效；所有参与性能比较的 Candidate 均先通过 Compile 和 Correctness。
- [ ] 比较期间评测协议一致；发生协议变化时已重新建立可比数据，未沿用失效状态或旧数字。
- [ ] 每个主要优化有证据、假设、具体修改和结果，组合修改与重复实验均有理由。
- [ ] 成功、失败和回退实验可追溯到对应实现；best 与 latest 分开维护，更新 best 时满足全部 required workload。
- [ ] 已判断 Profiling 是否必要，相关原始输出与摘要可复核；工具或证据不足已如实记录。
- [ ] References 仅在需要时读取，未依赖不存在的文件或工具，未在运行时查询外部 Wiki。
- [ ] 停止条件有明确依据，尚未解决的问题未被写成已解决。
- [ ] 最终交付版本已重新执行正式 Correctness 与 Benchmark，覆盖全部 required workload；无法完成或失败时未声称最终验证通过。
- [ ] 报告与实际实现、日志一致，明确区分已验证结论、推测和未完成项。
