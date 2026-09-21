# Wiki 变动记录

记录实际发生的修改、原因、检查结果与遗留影响。同批变动合并记录；规则见 [README](README.md)，阅读入口见 [INDEX](INDEX.md)。

## 2026-09-22 · SGLang 推理框架与算子设计 ingest

- 结合既有 vLLM、PagedAttention、AWQ 与算子基础，选取 raw 中 SGLang `2f730e29` 的 Scheduler/PrefillAdder、Python RadixCache、请求与 KV pool、批次转换和 overlap 转交，以及 RadixAttention/Triton extend/decode 与普通 decode 图准备。完整版本和来源文件登记在知识页，未展开版本迁移。
- 新增“SGLang 推理执行：Radix 缓存、批次调度与重叠执行”，解释前缀匹配与节点拆分、路径锁与淘汰、接纳预算、请求到槽位映射、EXTEND/DECODE/MIXED，以及 CPU/GPU 重叠如何保留自回归依赖。
- 新增“SGLang 算子设计：索引化 KV、Extend 与 Decode 归约”，解释变长 KV 索引、缓存前缀与当前输入共用 softmax、GQA 工作划分、已归一化段输出与 LSE 的合并，以及图内外元数据和 padding 契约。
- 已有服务页、部署页和两篇 vLLM 页接入新解释；vLLM 页增加另一种分段中间量表示的衔接，不把两套缓冲语义混用。共新增 2 个知识页、修改 4 个已有知识页，更新 INDEX 与本日志；raw 和项目 Skill 未改动。
- 验证：CPU 教学模型通过前缀身份/页对齐、共享路径保护与容量、请求/变长索引、prefix+extend 因果归一化、LSE 分段合并与错误平均反例，以及理想重叠时间关系。63 个知识页、463 条来源登记、856 条本地链接检查通过，无错误、警告或跳过；差异空白检查通过。
- 边界：本轮为所选固定源码的知识整合，教学模型没有执行 SGLang 原实现；未启动服务、编译 GPU kernel、验证实际 overlap/CUDA Graph 或测量性能。HiCache、SWA/Mamba、推测执行、分布式 KV 与全部 Attention 后端没有作为已覆盖成果。

## 2026-09-22 · vLLM 推理框架与算子设计 ingest

- 结合已有 PagedAttention、AWQ 与算子基础页面，选取 raw 中 vLLM `568afb3a` 的 V1 调度、KV 管理、GPU model runner、Attention 接口、FlashAttention 接入、Triton unified attention 与 CUDA Graph 设计材料；完整版本与具体文件登记在知识页。
- 新增“vLLM 推理执行：从 token 调度到变长批次”，解释计算进度与 token 预算、chunked prefill、KV 分配与抢占、前缀身份和引用生命周期，以及 query 边界、历史长度、位置、块表和写入槽位的联系。
- 新增“vLLM 算子设计：Attention 后端、分页计算与图执行”，解释模型层组织、缓存副作用与编译依赖、后端能力契约、GQA 行映射、分页加载、在线 softmax、分段归约和动态图批次的捕获规格匹配。
- 已有服务页保留原始论文口径并链接代码主线，澄清迭代接纳仍受资源约束、逻辑紧凑批次与图执行 padding 可以同时存在；部署页与 INDEX 接入新阅读路径。共新增 2 个知识页、修改 2 个已有知识页，并更新 INDEX 与本日志。raw 和 Skill 未改动。
- 验证：CPU 教学计算通过 token 预算、紧凑批次/槽位、绝对位置因果 mask、跨页分配、GQA 映射、padding 写入模型及在线/分段 softmax 检查；构造反例确认直接平均分段输出不等于统一 softmax。61 个知识页、443 条来源登记、834 条本地链接检查通过，无错误、警告或跳过；差异空白检查通过。
- 边界：本轮完成所选主线的代码研读与知识写入；未启动 vLLM、编译 GPU kernel、运行模型、验证 CUDA Graph 或测量性能。多模态 encoder、Mamba、分布式 KV、完整异步/推测执行与特殊 Attention 分支未作为已覆盖成果。

## 2026-09-22 · 从算子 Skill 展开配置与流水知识

- 选择 kernel-skills 中的 `Choose Tile Size and Work Partitioning`、`Optimize Triton Block Parameters`、`Optimize Shared Memory Tiling`，将其凝练建议展开为通用机制、假设和反例。结合既有 raw 的 Triton 教程/runtime、CUDA 文档及 CuTe SM80 流水代码补充依据，版本仅用于来源追溯，不展开版本迁移或接口兼容问题。
- 新增“Kernel 配置选择与自动调优”“GPU 异步拷贝与多级流水”两页：说明数据复用、累加器与 shared 预算、理论/实际 occupancy、尾块面积利用率、persistent 工作分配、调优的输入状态，以及拷贝组完成与缓冲复用的两个依赖方向。
- 补充 GPU 基础页的 word/bank 区分和步长的模运算推导，避免将连续 half 的同 word 读取直接判为 bank 冲突；计算模式与验证页接入资源选择、带状态的重复试跑及流水启动/收尾检查。共新增 2 个知识页、修改 3 个已有知识页，更新 INDEX 与本日志；raw 与项目 Skill 均未改写。
- 验证：CPU 教学模型通过资源上界、尾块利用率、persistent 完整覆盖、word/bank 映射和流水启动/轮换/排空检查，并检出读早与覆盖未释放缓冲的反例；重复调用的初始状态差异也完成教学核对。59 个知识页、424 条来源登记、812 条本地链接检查通过，无错误、警告或跳过；差异空白检查通过。
- 边界：模型验证的是正文列出的假设与顺序关系，不代表 GPU 内存模型或性能验证。未进行 CUDA/Triton 编译、GPU autotune、竞争诊断或实际基准；TMA 和异步 MMA 等机制未纳入本轮展开。

## 2026-09-22 · 算子开发基础 ingest

- 从既有 raw 选择 Triton `81a46fa0` 官方教程 01/02/03/05 与 testing.py、CUDA 文档 `snapshot-2026-07-02`、CUDA Samples `b7c5481c`，以及 Hugging Face kernels `5c2cf07f` 初始化模板；沿用 CS336 讲义核对 roofline。完整版本、快照时间和具体文件登记在相应知识页。
- 新增 3 页：张量布局与算子接口、GPU 算子的计算模式、算子正确性与性能测量。解释 stride/视图、CUDA 与 Triton 的工作分配、mask 与归约中性值、softmax/LayerNorm 前向、分块 GEMM、融合、数值容限及异步计时。
- 更新 GPU 基础页，补充 warp 下标、存储作用域、两次块内屏障、合并访存与 shared-memory bank 冲突；修正 roofline 页中低位宽改变瓶颈方向的反向表述，区分理论利用率上界与实际 MFU，并收紧由 FLOPs 直接断言实测性能的表述。
- 线性层与量化误差诊断页新增必要衔接，INDEX 增加“学习算子开发”路径，连接基础知识、验证与已有 AWQ/Marlin 实现。共新增 3 个知识页、修改 4 个已有知识页，并更新导航与本日志；raw 原始资料保持不变。
- 验证：独立 Python 标准库 CPU 教学计算的 8 组检查通过，涉及布局、尾块覆盖、warp 映射、访存模型、softmax/LayerNorm padding、非整除分块 GEMM、资源记账和 FP32 舍入反例。57 个知识页、410 条来源登记、792 条本地链接检查通过，无错误、警告或跳过；Git 差异空白检查通过。
- 边界：完成上述基础知识与固定代码片段的 ingest；没有编译或运行 CUDA/Triton，没有验证 GPU 数值、竞争诊断或性能。LayerNorm backward、完整框架集成和架构专用高级优化不在本轮覆盖范围。

## 2026-09-22 · AWQ 代码与服务后端 ingest

- 基于 raw 中 llm-awq `d6e797a4`、vLLM `568afb3a`、SGLang `2f730e29` 的固定快照，增量更新 AWQ 方法页、AWQ 实现页、权重量化反量化内核页与部署后端页，共 4 页；完整版本见页面来源身份表。
- 补齐官方搜索记录、fake/real 量化、TinyChat v2 打包与加载链，区分 CLI 校准配置和函数默认值；记录对称分支、group_size、clip 抽样与 GEMM 输出列分块的源码边界。
- 追踪 vLLM auto_awq 的平台与内核选择、普通 AWQ 的 token 阈值和 Triton/CUDA 分派；对照 SGLang 显式 awq 的选择行为、逐次反量化和 Marlin 加载期重排，分别解释权重、scale、zero-point 的布局转换。
- 内核正文补充新版 GEMV 的加载、解包、局部乘加与归约，以及 GEMM 的 CTA、流水阶段、split-K 和 FP16/BF16 累加差异。方法名称、实现类名、实际执行内核分别说明。
- 验证：独立教学计算通过 AutoAWQ 位序、权重/零点轴转换与 TinyChat 交织打包往返；这些计算没有执行原仓库的 PyTorch 打包器。54 个知识页、391 条来源登记、759 条本地链接检查通过，无错误或警告；Git 差异空白检查通过。
- 范围：完成上述主链的代码研读与知识写入，未编译 GPU 内核、运行模型或测量性能；完整 Marlin 设备主循环、MoE 专家路由、跨引擎数值一致性等仍按正文标明未验证。算子开发 Skill 仓库的收录另记 raw/LOG.md。

## 2026-09-22 · 既有内容复习 ingest

- 从 23 个方法主页面复习核心解释，并对解释缺口、公式疑点和实现描述定向重读本地固定论文版本、作者 TeX 与代码。实际修订 16 个知识页：9 个方法页、5 个实现页、2 个共通知识页；原始资料与目录结构保持不变。已有解释充分的内容复用，不重复建页。
- 实现链：修正 TinyChat AWQ 新旧内核的布局与分派混用，补齐 SGLang 普通 AWQ 与 Marlin 路径差异；更正原始 GPTQ 3-bit 打包、可选列置换与加载配置解释；区分 SmoothQuant 平滑因子、动态激活尺度和 INT8 重缩放系数。
- 重构目标与实验条件：补明 AdaRound 的教师/量化前缀非对称输入及软舍入初始化，澄清 BRECQ 梯度平方与首末层 8-bit 条件；更正 LLM.int8() 零点约定、离群维度选择及计时口径；保留 ZeroQuant 正文 3.1 小时、表格 1.1 小时与成本百分比的冲突。
- KV 与服务：补齐 KIVI 两类缓存窗口、统一 softmax 和 scale/min 偏移；补读 SAW-INT4 已展开子模块的 KV 写入与解码主机路径，明确 Q/K 配对旋转、V 还原和跨后端性能比较限制。补充 KV 字节数推导，修正分页注意力块公式的指数/求和顺序及分页集成条件。
- QServe：区分 W4A8KV4 与 FlatQuant 的 W4A4KV4，补明两级量化误差与 epilogue 输入替换近似；指出 v3 保护范围推导的算术不一致及代码中已注释的 119 范围断言，保留条件与未核验部分；局部注意力内核加速不再写成模型端到端收益。
- 验证：54 个知识页、372 条来源登记、758 条本地链接的结构、元数据、路径及锚点检查通过，无错误或警告；Git 差异空白检查通过。另以教学计算核对 GPTQ 3-bit 跨字布局、QServe 条件整数范围、跨块 softmax 归一化及 KV 字节数边界，均通过。
- 范围：这是针对既有解释的增量 ingest，不代表全部论文附录、所有实验表和代码分支均在本轮重新验证。AWQ 补充材料缺件、尚未研读的设备端分支与既有开放问题仍按各页记录保留；未构建 GPU 内核、运行模型、复现性能或核验跨引擎数值一致性。未提交或推送。

## 2026-09-22 · 检查工具归入 Skill

- 按用户确认，将结构检查脚本作为 wiki-review Skill 的随包工具保存，使用说明由 Skill 维护，项目根目录不恢复 scripts/。
- 同步 tags 必填、非空、命名和重复项检查。54 个知识页、344 条来源登记、758 条本地链接检查通过；缺失、空列表、重复和非法标签的反例均能检出，缺少 raw 的独立 Wiki 场景也通过。

## 2026-09-22 · 补齐知识页标签

- 为 54 个知识页补充 tags，按正文的主要对象、机制和应用场景选取，复用统一的小写英文标签。
- README 明确知识页 tags 必填，通常 2–5 个；导航和管理文件不要求标签。本次不修改知识正文、来源或实质内容更新日期。
- 核验通过：54 页标签无缺失、无页内重复、命名格式一致；正文逐页比对未变。清理目录并补标签后，757 条本地链接与 344 条来源登记的结构检查通过，无错误或警告。

## 2026-09-22 · 清理额外目录

- 按用户确认删除根目录 research/（两份研究记录）和 scripts/（结构检查工具）；这两个目录超出了本次 Wiki 重构需要。保留 wiki/research/ 知识分类。
- 清理项目 README、AGENTS、CLAUDE、Wiki 规范与相关 Skill 中的入口及工具执行要求。下文重构时的验证结果作为历史记录保留。

## 2026-09-22 · 目录与规范重构

- 将原有 54 个知识页按主要解释职责迁入五类目录，保留原文件名、中文标题、正文和实质内容更新日期。
- README 收敛为维护规范，新增 INDEX 承接知识地图与阅读路径；本文件记录变动与验证边界。
- 元数据补充 type，删除与文件名重复的 slug；Wiki 内部链接统一为相对路径 Markdown，保留原显示文字与章节目标。
- 各知识页补充来源身份表，保留标题、明确版本与外部 URL，使没有本地 raw 的读者仍能辨识来源。已移除代码快照保留原 commit 外部引用。
- 两篇综合页按已有正文中的问题抽象归入 research；AWQ/GPTQ/SmoothQuant 的具体方法对照保留在 methods。没有为了目录完整新造研究结论。
- 删除 docs：长期目标与有效写作要求并入 README；具体研究问题与文献反馈转入根目录 research；20 份有用的教学计算脚本和历史结果转入相关页面 assets；重复原文提取、旧链接报告、重复规范与过程稿清理。
- 两个旧检查器合并为项目根 scripts/check_wiki.py，适配新规范。项目 README、AGENTS、CLAUDE 和相关 Skill 的入口同步更新。
- raw 保持本轮开始时的内容；临时重构规范在规则承接与检查通过后删除。现有 Git 暂存与未提交工作保留，本批不提交或改动 Git 跟踪范围。

### 页面承接

目录分布：fundamentals 9 页、theory 8 页、implementation 11 页、methods 24 页、research 2 页。

| 原位置 | 当前位置 |
| --- | --- |
| `基础知识/张量算子与计算图/linear-layer-input-channel.md` | [线性层与输入通道](fundamentals/operators/linear-layer-input-channel.md) |
| `基础知识/数学与数值/entropy-and-dependence.md` | [熵、条件熵与依赖：量化代理指标的解释边界](fundamentals/mathematics/entropy-and-dependence.md) |
| `基础知识/数学与数值/integrated-gradients-and-quantization-sensitivity.md` | [积分梯度与量化敏感性：基线、路径和归因边界](theory/integrated-gradients-and-quantization-sensitivity.md) |
| `基础知识/数学与数值/invertible-transforms-and-kronecker-products.md` | [可逆变换、数值条件与 Kronecker 乘积](fundamentals/mathematics/invertible-transforms-and-kronecker-products.md) |
| `基础知识/硬件与计算基础/arithmetic-intensity-and-roofline.md` | [算术强度与 roofline 分析](fundamentals/hardware/arithmetic-intensity-and-roofline.md) |
| `基础知识/硬件与计算基础/gpu-execution-and-memory-hierarchy.md` | [GPU 的执行模型与存储层次](fundamentals/hardware/gpu-execution-and-memory-hierarchy.md) |
| `实现与加速/数据格式与存储/fp8-and-mx-data-formats.md` | [FP8 与 Microscaling 数值格式](fundamentals/numeric-formats/fp8-and-mx-data-formats.md) |
| `实现与加速/数据格式与存储/gguf-block-quantization-formats.md` | [GGUF 与块量化存储格式](implementation/gguf-block-quantization-formats.md) |
| `实现与加速/框架与部署/quantized-llm-deployment-backends.md` | [量化模型的部署框架与后端支持](implementation/quantized-llm-deployment-backends.md) |
| `实现与加速/框架与部署/serving-memory-and-batching.md` | [推理服务的内存管理与批处理](implementation/serving-memory-and-batching.md) |
| `实现与加速/算子与内核/marlin-batched-w4a16-gemm.md` | [Marlin：批处理 W4A16 GEMM 内核](implementation/marlin-batched-w4a16-gemm.md) |
| `实现与加速/算子与内核/quantized-matmul-scaling-execution.md` | [量化矩阵乘法的缩放与执行路径](implementation/quantized-matmul-scaling-execution.md) |
| `实现与加速/算子与内核/weight-only-dequant-kernels.md` | [权重量化反量化内核的契约：AWQ 与 marlin 两种实现](implementation/weight-only-dequant-kernels.md) |
| `评测与验证/lvlm-compression-benchmark.md` | [VLM 压缩评测框架：从任务精度到可信度](implementation/lvlm-compression-benchmark.md) |
| `评测与验证/quantization-error-diagnosis.md` | [量化误差诊断与验证](implementation/quantization-error-diagnosis.md) |
| `评测与验证/quantization-reliability-and-selective-prediction.md` | [量化与可靠性：选择性预测评测](theory/quantization-reliability-and-selective-prediction.md) |
| `量化/卷积网络量化/adaround.md` | [AdaRound：面向任务损失的自适应舍入](methods/adaround.md) |
| `量化/卷积网络量化/brecq.md` | [BRECQ：块级重构与二阶误差的粒度选择](methods/brecq.md) |
| `量化/大语言模型量化/affinequant.md` | [AffineQuant：可学习仿射变换与渐进掩码](methods/affinequant.md) |
| `量化/大语言模型量化/awq-gptq-smoothquant-comparison.md` | [AWQ、GPTQ 与 SmoothQuant：对象、机制与证据比较](methods/awq-gptq-smoothquant-comparison.md) |
| `量化/大语言模型量化/awq-implementation.md` | [AWQ 的实现核对：从量化脚本到 vLLM 与 SGLang](implementation/awq-implementation.md) |
| `量化/大语言模型量化/awq.md` | [AWQ：激活感知的权重量化](methods/awq.md) |
| `量化/大语言模型量化/flatquant.md` | [FlatQuant：结构化可学习变换与低比特执行](methods/flatquant.md) |
| `量化/大语言模型量化/gptq-implementation.md` | [GPTQ 的实现核对：量化器、打包与两套服务路径](implementation/gptq-implementation.md) |
| `量化/大语言模型量化/gptq.md` | [GPTQ：基于二阶补偿的权重量化](methods/gptq.md) |
| `量化/大语言模型量化/kivi.md` | [KIVI：键按通道、值按 token 的 2 bit KV cache 量化](methods/kivi.md) |
| `量化/大语言模型量化/kv-cache-quantization-objects-and-granularity.md` | [KV cache 量化的对象与粒度](theory/kv-cache-quantization-objects-and-granularity.md) |
| `量化/大语言模型量化/llm-int8.md` | [LLM.int8()：向量级量化与混合精度分解](methods/llm-int8.md) |
| `量化/大语言模型量化/luq.md` | [LUQ：以层激活熵选择超低比特量化层](methods/luq.md) |
| `量化/大语言模型量化/masquant.md` | [MASQuant：分模态平滑与共享权重补偿](methods/masquant.md) |
| `量化/大语言模型量化/mbq-vlmq-comparison.md` | [MBQ 与 VLMQ：重要性怎样进入量化目标](research/mbq-vlmq-comparison.md) |
| `量化/大语言模型量化/mbq.md` | [MBQ：模态平衡的视觉语言模型量化](methods/mbq.md) |
| `量化/大语言模型量化/mquant.md` | [MQuant：模态静态量化、等价重排与旋转幅度抑制](methods/mquant.md) |
| `量化/大语言模型量化/omniquant-affinequant-flatquant-comparison.md` | [OmniQuant、AffineQuant 与 FlatQuant：可学习变换路线比较](research/omniquant-affinequant-flatquant-comparison.md) |
| `量化/大语言模型量化/omniquant.md` | [OmniQuant：可学习裁剪与等价变换的块级校准](methods/omniquant.md) |
| `量化/大语言模型量化/q-vlm.md` | [Q-VLM：熵驱动的联合校准与视觉编码器优化](methods/q-vlm.md) |
| `量化/大语言模型量化/qig.md` | [QIG：以量化误差积分梯度指导 token 加权校准](methods/qig.md) |
| `量化/大语言模型量化/qserve.md` | [QServe：W4A8KV4 量化与系统协同](methods/qserve.md) |
| `量化/大语言模型量化/quarot.md` | [QuaRot：固定旋转、全图等价与四位推理](methods/quarot.md) |
| `量化/大语言模型量化/saw-int4.md` | [SAW-INT4：面向真实服务约束的 4 bit KV cache 量化](methods/saw-int4.md) |
| `量化/大语言模型量化/smoothquant-implementation.md` | [SmoothQuant 的实现核对：平滑融合、INT8 接口与离线边界](implementation/smoothquant-implementation.md) |
| `量化/大语言模型量化/smoothquant.md` | [SmoothQuant：迁移激活量化难度的 W8A8 方法](methods/smoothquant.md) |
| `量化/大语言模型量化/spinquant.md` | [SpinQuant：用最终模型损失学习正交旋转](methods/spinquant.md) |
| `量化/大语言模型量化/splitq.md` | [SplitQ：模态离群通道拆分与权重激活双补偿](methods/splitq.md) |
| `量化/大语言模型量化/vision-language-model-tokens-and-quantization.md` | [视觉语言模型中的 token 与量化对象](fundamentals/model/vision-language-model-tokens-and-quantization.md) |
| `量化/大语言模型量化/vlmq.md` | [VLMQ：token 重要性加权的二阶量化](methods/vlmq.md) |
| `量化/大语言模型量化/zeroquant.md` | [ZeroQuant：逐组权重、逐 token 激活与逐层蒸馏](methods/zeroquant.md) |
| `量化/通用方法与机制/diagonal-scaling-equivalent-transform.md` | [对角缩放与等价变换](theory/diagonal-scaling-equivalent-transform.md) |
| `量化/通用方法与机制/layer-reconstruction-second-order-compensation.md` | [层输出重构与二阶误差补偿](theory/layer-reconstruction-second-order-compensation.md) |
| `量化/通用方法与机制/mixed-precision-allocation.md` | [混合精度分配：选择变量、预算与部署口径](theory/mixed-precision-allocation.md) |
| `量化/通用方法与机制/orthogonal-rotation-and-hadamard-quantization.md` | [正交旋转与量化：等价条件、离群值和在线代价](theory/orthogonal-rotation-and-hadamard-quantization.md) |
| `量化/量化基础/calibration-and-range-selection.md` | [校准数据与量化范围选择](theory/calibration-and-range-selection.md) |
| `量化/量化基础/post-training-and-quantization-aware-training.md` | [PTQ、QAT 与代理梯度](fundamentals/quantization/post-training-and-quantization-aware-training.md) |
| `量化/量化基础/uniform-quantization-and-groups.md` | [均匀量化与分组](fundamentals/quantization/uniform-quantization-and-groups.md) |

### 验证与边界

- 迁移正文与开始时快照逐页对照，允许的变化为链接语法和路径、元数据以及追加的来源身份与教学材料入口。
- 检查通过：54 个知识页、344 条来源登记（326 条本地文件、18 条固定 commit 外部引用）、770 条本地链接；无错误、无孤立知识页提醒，全部知识页从 INDEX 可达。54 页原正文完整承接，231 处独立公式块逐字保持不变。
- 缺少 raw 的独立 Wiki 场景通过，326 条本地来源明确标记为未执行核验；严格模式会拒绝缺失 raw。失效链接、错误锚点、旧 wikilink、非法类型、缺失来源、重复字段、未登记 arXiv 号、未来日期及索引遗漏的反例均被检查器检出。
- 10 份教学计算脚本在临时副本运行通过；原目录中的历史结果文件保持原字节。合计完成 21 项正向、反向和执行验证。
- 本轮未重新 ingest，未把旧研读或教学计算状态升级为本次科学性验证；未运行模型、服务、CUDA/Triton 内核或性能基准。
- 数学样例中的历史结果保留原日期和范围。旧样例脚本中混入的迁移前来源哈希、旧页面身份检查已移除，保留教学计算本身；本次重构的结构与来源检查已完成，检查工具不作为长期依赖。

## 2026-09-14—17 · 既有建设记录承接

以下是旧记录的范围摘要，表示当时的研读与写入，不是本轮重新核验。

| 批次 | 知识承接与历史范围 |
| --- | --- |
| 09-14 公共基础 | [均匀量化与分组](fundamentals/quantization/uniform-quantization-and-groups.md)；整数运算、BN 融合、范围与重构；保留教学反例和代码。 |
| 09-14 AWQ | [AWQ：激活感知的权重量化](methods/awq.md)；论文、TinyChat 定向代码及校准/裁剪；没有模型或性能复现。 |
| 09-14 GPTQ / SmoothQuant | [GPTQ：基于二阶补偿的权重量化](methods/gptq.md)；二阶补偿、平滑与 W8A8；作者结果与实现范围分别保留。 |
| 09-14 MBQ / VLMQ | [MBQ 与 VLMQ：重要性怎样进入量化目标](research/mbq-vlmq-comparison.md)；目标加权及局部 GPTAQ 依赖；原文平均分不一致和梯度启动边界仍见方法页。 |
| 09-14 MQuant / Q-VLM | [MQuant：模态静态量化、等价重排与旋转幅度抑制](methods/mquant.md)；重排、旋转与多模态校准；公开实现与论文范围分别说明。 |
| 09-14 OmniQuant | [OmniQuant：可学习裁剪与等价变换的块级校准](methods/omniquant.md)；裁剪、平移、重构与 STE；数值小例子不等于模型反向验证。 |
| 09-15 可学习变换 | [OmniQuant、AffineQuant 与 FlatQuant：可学习变换路线比较](research/omniquant-affinequant-flatquant-comparison.md)；AffineQuant / FlatQuant、逆变换、Kronecker、双参考目标；未遍历全部模型和部署分支。 |
| 09-15 QuaRot / SpinQuant | [SpinQuant：用最终模型损失学习正交旋转](methods/spinquant.md)；旋转与 Cayley 教学核对；公开复现实现与论文内部实现的差别保留。 |
| 09-15 MASQuant / SplitQ | [MASQuant：分模态平滑与共享权重补偿](methods/masquant.md)；低秩残差、白化与分支拆分；辅助位宽及速度口径分别说明。 |
| 09-15 QIG / LUQ | [QIG：以量化误差积分梯度指导 token 加权校准](methods/qig.md)；归因、非负权重、熵与位宽预算；未验证模型与硬件执行。 |
| 09-16 研究基线 | [AdaRound：面向任务损失的自适应舍入](methods/adaround.md)；LLM.int8()、ZeroQuant、AdaRound、BRECQ；卷积网络方法作为共通机制实例。 |
| 09-16 工程链路 | [量化模型的部署框架与后端支持](implementation/quantized-llm-deployment-backends.md)；FP8/MX、GGUF、Marlin、QServe 与部署接口；OCP 规范原件与部分实现分支未独立核对。 |
| 09-16 KV cache | [KV cache 量化的对象与粒度](theory/kv-cache-quantization-objects-and-granularity.md)；KIVI、SAW-INT4 的粒度、窗口与代码入口；不做跨协议性能排序。 |
| 09-16 VLM 评测 | [VLM 压缩评测框架：从任务精度到可信度](implementation/lvlm-compression-benchmark.md)；压缩、可靠性与选择性预测；模型范围、指标条件和未核验附录分别保留。 |
| 09-17 结构补充 | [GPU 的执行模型与存储层次](fundamentals/hardware/gpu-execution-and-memory-hierarchy.md)；硬件层次、roofline、分页与调度；未进行本地硬件测量。 |
| 09-17 实现与 kernel | [权重量化反量化内核的契约：AWQ 与 marlin 两种实现](implementation/weight-only-dequant-kernels.md)；AWQ/GPTQ/SmoothQuant 路径与 SGLang JIT/Marlin、repack、线程和分派；仅代码阅读。 |

### 仍需独立任务解决的内容问题

- 模型精度、跨引擎数值一致性、实际打包导出及 GPU 性能，尚无本地运行验证。
- 方法页已注明的论文内部数字差异、公开代码与论文配置差异、未读分支和有限实验条件继续保留；资料或依赖文件补齐不等于相应代码已经研读。
- 服务知识尚未完整覆盖 chunked prefill、前缀树共享和 prefill/decode 分离；旧记录中的待扩展项不是空目录或已完成页面。
- 研究线索与实验设想保留在项目研究说明，未因本次迁移升级为已验证结论。
