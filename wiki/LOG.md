# Wiki 变动记录

记录实际发生的修改、原因、检查结果与遗留影响。同批变动合并记录；规则见 [README](README.md)，阅读入口见 [INDEX](INDEX.md)。

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
