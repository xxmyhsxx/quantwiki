# Wiki 分批审核清单

范围：2026-10-15 的 213 个知识页，分为 54 批。覆盖基础 21 页、理论 16 页、方法 97 页、综合研究 3 页、实现 76 页；不包含管理页、Skill 和 raw 原件。原件只在核验相应主张时读取。

当前进度：已完整审核 25/213 页（01–06 批）；按新增的正文改善要求，已改善并复核 21/213 页，完成 5/54 批。当前推进第 06 批正文改善。先完成 01–07 批的审核、必要重写与阅读复核，再统一推送；其余批次按用户安排推进。勾选表示已改善并复核，或确认正文已充分、无需修改；仅有阅读记录或待修建议不计完成。

## 审核要求

按 [Wiki 正文规范](wiki/README.md#正文与解释深度) 和 [wiki-review](.agents/skills/wiki-review/SKILL.md) 执行，重点检查：

- 对象在所属任务中解决什么问题、承担什么作用；整体怎样工作，各部分为何存在、怎样配合。
- 必要前置、公式定义、推理、事实依据、版本及适用条件是否充分；方法收益与系统收益是否混淆。
- 段落和链接是否提供必要信息；重复、晦涩、无用内容，以及跨页冲突、解释断点和真正缺失的知识。

每页完整阅读正文，再按主张和疑点查相关页与原件；在授权范围内直接修正事实、补充解释、重组和去重。改后只依据正文与必要链接复述用途、整体过程和取舍，仍有理解断点就继续修改。结构和链接检查不能代替内容判断；不开展实验、不改 raw。

## 如何使用清单

每次读取当前批次及必要关联页，不把整个清单或 INDEX 作为必读输入。每页只分配一个主审核批次；跨批引用可提前查阅，不因此将目标页标为已审核。长页按章节分段读，整页完成后再勾选。

完成一批后，简记正文变化、必要依据、验证与遗留问题，更新勾选和进度。保留此前有效审核记录，但区分审核与改善验收；未处理的解释问题不能与完成勾选并存。已有清楚内容不为凑数改写。新增、合并或删除页面时同步清单。

批次按主题关系和篇幅划分，不是缺陷排名。优先检查已出现问题及内容较密集的解释链；必要前置可定向先读。

| 批次 | 范围 |
| --- | --- |
| 01–07 | 部署、量化执行与低比特表示 |
| 08–19 | 推理引擎、MoE、MLA、分布式与 KV cache |
| 20–28 | GPU 基础、算子实现与性能解释 |
| 29–42 | 量化基础、算法、训练与误差分析 |
| 43–52 | 模型、多模态、语音、动作与其他应用 |
| 53–54 | 质量与性能评测协议 |

## 01 · 部署产物与后端接入（3 页）

重点：从已训练模型到可运行产物的职责、依赖与兼容条件；优先处理 ModelOpt 已发现的段落混杂。

- [x] [ModelOpt PTQ：配置、校准与部署 checkpoint](wiki/implementation/modelopt-ptq-calibration-and-checkpoint.md)
- [x] [量化模型的部署框架与后端支持](wiki/implementation/quantized-llm-deployment-backends.md)
- [x] [GGUF 与块量化存储格式](wiki/implementation/gguf-block-quantization-formats.md)

改善与复核（2026-10-15）：三页均完成正文阅读、必要修改和改后阅读检查。

- ModelOpt 保留开头，按“选择模块/表示→确定尺度或搜索→导出→后端准备→追溯”重写正文。模型命名、视觉前向、内存/分布式、源格式转换等条件回到对应环节；拆清搜索恢复文件、部署 checkpoint 和运行记录的用途，保留全部尺度公式与关键版本限制。
- GGUF 修正 IQ2_XS 为每 16 权重一个 4-bit 尺度（每 32 权重合存一字节），区分 IQ1/2/3 向量网格与 IQ4 标量表，修正 `--pure` 的适用范围。补清 HF→GGUF→量化的输入关系；部署页同步说明模型架构、后端和多模态 mmproj 的配套条件。
- 改后沿三条阅读问题复核：recipe 怎样变成部署参数、块位宽怎样计费、量化文件交给谁执行。正文可直接接通这些关系，既有清楚的格式表和后端章节保留。

核验依据沿用本批原件检查：ModelOpt `f2f0d695` 量化/导出、input_scale1/headroom、KV 搜索与 export-only；vLLM `568afb3a` 静态 FP8 加载；TensorRT-LLM `a6f41514` 正文/支持矩阵；MLC `6474f7db` 参数计费/JIT；llama.cpp `b820cc8e` 容器、块结构、解码、量化工具与 imatrix。IQ2_XS 定位 `ggml-quants.c:dequantize_row_iq2_xs` 2444–2470 行，`--pure` 定位 `llama-quant.cpp:llama_tensor_get_type`。无运行验证；未全面重核其他算法分派、历史后端流程、全部设备 kernel、外链或 release 声明。改后链接检查 3 页、46 项通过，差异空白检查通过；保留 ModelOpt §2 既有锚点，其他改名标题未发现入链。

提交：`docs: clarify batch 01 quantization deployment flow`，仅三篇正文与本记录；01–07 完成后统一推送。此前审核提交 `088b03a` 的有效证据已纳入本记录。

## 02 · 量化矩阵乘的公共解释（3 页）

重点：逻辑计算、缩放、表示与实际执行怎样衔接，公共页与具体实现如何分工。

- [x] [线性层与输入通道](wiki/fundamentals/operators/linear-layer-input-channel.md)
- [x] [量化矩阵乘法的缩放与执行路径](wiki/implementation/quantized-matmul-scaling-execution.md)
- [x] [块缩放矩阵乘：FP4/FP8 编码、尺度布局与执行接口](wiki/implementation/block-scaled-matmul-execution.md)

改善与复核（2026-10-15）：三页已按解释目标复核。线性层页的通道、矩阵方向与误差例子，块缩放页的数值→字节→布局→点积主线已充分，保留正文。

量化矩阵乘页重组为尺度/零点、执行方式、输出网格、额外计算、瓶颈与证据。原 §15 的分组恢复移到首次解释归约维之后，原 §16 位平面移到执行方式下，原 §17 共同权重误差归入补偿分支；保留独有公式、反例、预算及舍入条件。在线变换、低秩、稀疏残差和图融合按作用归类，移除已由 OmniQuant/FlatQuant/SplitQ 方法页承接的重复性能数字。同步 15 个入链文件的标题锚点，仅改变链接，不将这些页面计入内容审核。

核验复用 AWQ v6、SmoothQuant v7、整数推理 v1 的尺度/零点/再量化/NEON 条件及独立算术；Triton `81a46fa0` 的初始化、scale 排列、FP32 参考与 descriptor，FlashInfer 0.6.14 所选 BF16×FP4/FP8 接口。第06批固定 FlexQ 源码核验支持位平面与分组恢复说明。改后重新核对分组式的 X/W 维度与上下文、整数输出单位、补偿成本和证据归属；链接检查覆盖三页及入链共 18 页、1050 项，通过；差异空白检查通过。其他方法原件、硬件 lowering 与设备执行仍按此前范围保留未核状态，无实验。

提交：`docs: organize batch 02 matmul execution explanations`。此前审核提交 `ce82b5f` 的有效证据已纳入本记录。

## 03 · SharQ、SpQR 与表示综合（5 页）

重点：复核已局部修正的用途、整体流程、公式定义和跨页去重，再检查其余正文。

- [x] [SharQ：动态稀疏 FP4 主干与量化残差](wiki/methods/sharq.md)
- [x] [SharQ 的稀疏残差、共享尺度与所选接口](wiki/implementation/sharq-sparse-residual-and-shared-scales.md)
- [x] [SpQR：敏感权重保留与两级量化元数据](wiki/methods/spqr.md)
- [x] [SpQR 的两级尺度、稀疏保护值与所选 CUDA 格式](wiki/implementation/spqr-metadata-outliers-and-cuda-format.md)
- [x] [低比特表示的设计空间：坐标、码本与例外预算](wiki/research/low-bit-representation-design.md)

改善与复核（2026-10-15）：五页均按用途、整体流程与取舍复核。

SpQR 方法页就地定义 m/n，说明每行指针如何摊销；将末尾堆叠的实现字段改为物理预算和覆盖/相加两个关键部署条件。实现页重写开头，先说明小组主体、两级元数据和保护值如何配合，再沿校准→保存→打包→消费解释；组内预测与最终保护分开，明确 QQ 的含义，并拆清数值恢复、批量接口和测试边界。改后能由正文判断例外究竟保存绝对值还是残差、各载荷怎样计费。SharQ 两页和表示综合页已有完整解释主线，保留有效正文，不重复改写。

**撤回此前断言阻断判断：** 原审核提交 `2a5b7c2` 将两个变量误读为同一对象。固定 `spqr_engine.py:315` 断言检查原始 `group_diag_hessian_inv_cho.ndim == 1`，不是索引后的 `loo_group_diag_hessian_inv_cho`；二维索引结果不违反该断言。已从待修清单撤除，未向正文加入错误结论，也未执行或改动 raw。

其余证据复用 SharQ v1/`3cc584fe` 的残差、共同权重、共享尺度与代表结果，SpQR v1/`543d10e5` 的敏感性、两级统计、保护值与 CSR 转换。重核上述断言及真实变量名；既有 3.625/4.75/7.25 bpw、误差恒等式与非整数零点算例成立。未全面重审 CUDA 同步/边界、模拟舍入/vendor 或服务加载，无运行实验。改后五页链接 25 项和差异空白检查通过。

提交：`docs: clarify batch 03 sparse quantization flow`；除撤回项外沿用此前审核证据。

## 04 · 码本与表示预算（4 页）

重点：主体、元数据、例外、解码成本是否使用同一口径；理论页是否变成方法摘录。

- [x] [码本量化：标量、向量、加性表示与位宽预算](wiki/theory/codebook-quantization-and-bit-budget.md)
- [x] [SqueezeLLM：敏感性加权的非均匀量化与稠密稀疏分解](wiki/methods/squeezellm.md)
- [x] [GPTVQ：多输入列向量、数据感知码本与移动 CPU 解码](wiki/methods/gptvq.md)
- [x] [VPTQ：同列向量量化、二阶反馈与码本成本](wiki/methods/vptq.md)

改善与复核（2026-10-15）：四页已按用途、表示、选码、预算和解码复核。SqueezeLLM、GPTVQ、VPTQ 方法页已经解释目标、步骤、独有配置及实现边界，保留正文。

公共码本页将 Lloyd–Max 推导移到距离之后，固定码率、熵与高分辨率近似归入预算章节；§4 改为结构约束总节，让 E8P 与仿射网格并列。保留推导条件、反例和非负预算限制。VPTQ 的重复表大小交给方法页，GPTVQ 公共算例保留索引/中心精度/附加尺度的区别，删去重复字节展开；同步四个入链文件的锚点，不将其计为内容审核。

改后按“代表值怎样选、结构省了什么、全部存储怎样计费、执行需做什么”重读，相关解释已接通。链接检查覆盖四页与四个入链文件，共 8 页、941 项，通过；差异空白检查通过。

原件核验：SqueezeLLM v4 Methodology 的 Fisher 对角代理、标量聚类及例外动机，固定 `a5fd71f3` 的 `pack2` 零映射扣除与 FP32 稀疏值；GPTVQ v2 方法正文的多输入列/对角距离、伪逆中心更新与位预算、附录 A 的固定索引凸二次目标、配置表和移动吞吐表；VPTQ v2 方法/优化正文的同列欧氏选码、跨列加权初始化、残差/微调，表10三阶段微调及附录 D 的恢复/GEMV 两路径。独立复算带权中心例子、条件二阶解、各码本字节数、覆盖加权预算、Lloyd–Max 例子、Gaussian 信息下界和高分辨率预算的非负截断；这些为数学核对，未执行模型或实验。第03批综合页对 SqueezeLLM 目标的归因得到支持。未逐格核所有实验附表，未核 GPTVQ/VPTQ 匹配作者实现；公共页 LQ-Nets/LC-QAT/KV 的特定原件归因未全面复查，E8P/路径/球面表示分别在05/07批续核。链接检查：4 页、49 项通过；本批互链和二阶目标关系已核，其他存量链接目标全文未逐一阅读。

提交：`docs: organize batch 04 codebook fundamentals`；此前审核提交 `bea2042` 的有效证据已纳入本记录。

## 05 · 码本方法与实现对应（6 页）

重点：方法目标如何对应校准、坐标恢复、打包和推理消费，避免将浮点恢复当成低位执行。

- [x] [QuIP：非相干处理与 LDLQ 二阶量化](wiki/methods/quip.md)
- [x] [QuIP#：Hadamard 非相干处理与 E8P 格码本](wiki/methods/quip-sharp.md)
- [x] [AQLM：加性码本与激活感知的联合编码](wiki/methods/aqlm.md)
- [x] [QuIP 的逆序 LDLQ、坐标恢复与浮点保存路径](wiki/implementation/quip-ldlq-and-coordinate-restoration.md)
- [x] [QuIP# 的 E8P 打包、解码与所选 Llama 路径](wiki/implementation/quip-sharp-e8p-packing-and-inference.md)
- [x] [AQLM 的原始校准、模型转换与 2×8 推理读取](wiki/implementation/aqlm-calibration-export-and-inference.md)

改善与复核（2026-10-15）：六页完成解释复核。三篇方法页和 AQLM 实现页已能独立接通表示、优化、保存与执行，保留正文。两篇 QuIP 实现页重写开头，先交代整体过程和最终产物，再说明核读范围；原 QuIP 页补定义 W/H/A/U/V，拆开网格、逆序反馈、贪心和分块；QuIP# 页拆开逻辑码、物理打包、存储预算与测速范围。独有公式、代码差异与限制全部保留。

改后依据正文复述了三种路径：原 QuIP 默认还原浮点权重，QuIP# 保存压缩码并按输入行数分派，AQLM 分开校准/转换/读取。各自的优化目标、载荷和运行条件已能对应。

- QuIP 的上三角正序说明与源码下三角逆序约定分别自洽；实现页已说明预处理、浮点恢复和默认保存不构成物理 W2 压缩。已有 meta flag 字段差异、贪心裁剪后残差失配和 LDLQ_RG/lazy 位置参数错位均能对应源码，不再当作本次新发现。
- QuIP# 已区分 16 bit 逻辑码、重排的 int64 载荷、1 KiB 底表与 Python 完整参考表，也区分单行融合解码、多行展开及训练缓存。有限 E8P 与统一二阶矩定理条件、实数变换向量与初始正交条件的差别已写明。
- AQLM 已保留全部码本的预算，区分原始激活相关选码、PV 的 L2 选码、块 MSE 与整网 KL；有符号码容器、转换入口条件和大输入稠密恢复没有被写成任意位宽紧密打包或已验收部署。

原件核验：QuIP v2 的线性反馈最优性条件和代表 OPT/PPL/时延表；固定 `ac92cfc7` 的逆序递推、贪心残差、lazy 参数、meta flag、postproc/error_compute 顺序及默认保存。QuIP# v2 的 BlockLDLQ 条件、E8P 模式/符号/平移构造与残差码本、4096 上下文消融表；固定 `1d8f873e` 的 maybe_pack_idxs、单/多行 forward、cache_WH 及 None 变换矩阵访问。AQLM v4 的加性表示、MRF 展开、交替优化/块微调、主表与微调/生成表；固定 `e79a896e` 的 token 加权 XTX、Adam/早停、HF 有符号转换及 args.pt 依赖、PV 条件分支、输入数分发和所选 CUDA 2×8 低精度运算。独立复核坐标恢复、误差展开、贪心教学例和主体/码本/尺度计费。第04批公共页的 E8P 与加性预算得到交叉支持。

未重核全部 Hadamard 常量/概率证明、离线统计及微调调用链、各 CUDA host/同步/归约、beam search 全实现、模型适配器或历史实验协议；页面原有“已核”描述不等于本轮全部重新审计。未运行上游或实验。改后链接检查：6 页、32 项和差异空白检查通过；本批互链与相关公共表示已核，其余目标全文未逐条阅读。

提交：`docs: explain batch 05 quantization artifact flow`；此前审核提交 `ce9f404` 的有效证据已纳入本记录。

## 06 · 因子、多精度与其他编码（4 页）

重点：各编码解决的部署需求、表示约束和真实成本，检查与综合页的一致性。

- [ ] [DBW：在 SVD 分量上联合学习秩与位宽](wiki/methods/dbw.md)
- [ ] [StagQ：共享元数据的前缀权重流与稀疏保护](wiki/methods/stagq.md)
- [ ] [FlexQ：INT6 分组量化与二值 Tensor Core 执行](wiki/methods/flexq.md)
- [ ] [FlexQ 的位平面、组恢复与所选调用接口](wiki/implementation/flexq-bitplanes-and-group-restoration.md)

审核记录（2026-10-15，改善待完成）：第06批四页全文已读。核心表示与预算自洽；FlexQ 方法 §3 从 T/d/o 改用 M/K/N 未定义对应关系，实现页也直接使用这些维度，需补清。已抽核 DBW `a45f26e1` 分量配额、训练目标、word packing 与两次浮点 Linear；StagQ v1 的前缀、保护重建、元数据目标及代表质量/decoder 对照；FlexQ v2 位展开/式6、代表 PPL/消融，`a42424d8` 的 signed 位权、组尺度与归约。链接检查 4 页、21 项通过，无实验。待回补前五批后按同一标准完成正文改善，不据此勾选。

## 07 · 高维与二值表示（4 页）

重点：表示对象、选码和解码的整体思路；数学表示与硬件收益之间是否缺少解释。

- [ ] [QTIP：trellis 高维量化、并行解码与尾咬合](wiki/methods/qtip.md)
- [ ] [XOR-Trellis：局部候选集合、曲率加权路径与轻量解码](wiki/methods/xor-trellis.md)
- [ ] [FluxBin：行列二值分解、显著列残差与输入查表](wiki/methods/fluxbin.md)
- [ ] [SPHQuant：符号、球面方向与半径的独立编码](wiki/methods/sphquant.md)

## 08 · 请求到模型执行（4 页）

重点：请求、调度、缓存、前向和返回之间的关系；区分各框架的版本与职责。

- [ ] [nano-vllm 离线推理：token 进度、分页缓存与模型执行](wiki/implementation/nano-vllm-inference-execution.md)
- [ ] [vLLM 推理执行：从 token 调度到变长批次](wiki/implementation/vllm-inference-execution.md)
- [ ] [SGLang 推理执行：Radix 缓存、批次调度与重叠执行](wiki/implementation/sglang-inference-execution.md)
- [ ] [结构化语言模型程序：依赖、前缀共享与约束解码](wiki/implementation/structured-language-model-program-execution.md)

## 09 · 服务内存与 Attention 接口（4 页）

重点：缓存管理与数值计算的交接；接口字段是否服务于清楚的执行过程。

- [ ] [推理服务的内存管理与批处理](wiki/implementation/serving-memory-and-batching.md)
- [ ] [vLLM 算子设计：Attention 后端、分页计算与图执行](wiki/implementation/vllm-attention-operator-design.md)
- [ ] [SGLang 算子设计：索引化 KV、Extend 与 Decode 归约](wiki/implementation/sglang-attention-operator-design.md)
- [ ] [FlashInfer KV 接口：分页写入、单请求 attention 与 NVFP4 尺度](wiki/implementation/flashinfer-paged-kv-write-and-nvfp4.md)

## 10 · MoE 路由与单设备执行（2 页）

重点：先解释稀疏专家计算的用途，再跟清路由、重排、量化与输出合并。

- [ ] [MoE 与专家路由：稀疏计算、容量与量化对象](wiki/fundamentals/model/mixture-of-experts-and-routing.md)
- [ ] [MoE 路由与量化专家执行：SGLang/vLLM 的单设备 FP8 主线](wiki/implementation/moe-routing-and-quantized-expert-execution.md)

## 11 · MoE 专家并行（1 页）

重点：并行分工、通信与回传的完整关系；长页中通用逻辑和框架特例是否混杂。

- [ ] [MoE 专家并行执行：SGLang/vLLM 的映射、通信与回传](wiki/implementation/moe-expert-parallel-execution.md)

## 12 · MoE 通信与矩阵乘实现（3 页）

重点：通信产物怎样成为专家计算输入；布局细节和性能边界是否有必要且相互一致。

- [ ] [DeepEP 的 MoE 通信执行：去重、FP8 布局与输出回传](wiki/implementation/deepep-moe-communication-execution.md)
- [ ] [DeepGEMM 的 MoE 专家执行：行布局、块尺度与 SM90 计算](wiki/implementation/deepgemm-moe-expert-execution.md)
- [ ] [vLLM W4A16 MoE：AWQ 专家格式、配对读取与低精度加权](wiki/implementation/w4a16-moe-expert-execution.md)

## 13 · MLA 基础与缓存执行（2 页）

重点：压缩缓存的目的、位置分支和执行路径之间的关系，避免读者在多个框架间自行拼接。

- [ ] [MLA：联合潜变量、位置分支与缓存表示](wiki/fundamentals/model/multi-head-latent-attention.md)
- [ ] [MLA 缓存与 Attention 执行：SGLang、FlashInfer 与 vLLM](wiki/implementation/mla-cache-and-attention-execution.md)

## 14 · MLA 上下文并行（1 页）

重点：缓存分片、局部结果和跨 rank 合并是否组成完整解释。

- [ ] [vLLM MLA 的 DCP 执行：缓存分片、局部 Attention 与跨 rank 合并](wiki/implementation/vllm-mla-context-parallel-execution.md)

## 15 · MLA packed 缓存与 DSA（2 页）

重点：缓存字段、候选索引、稀疏读取和设备计算的衔接，核对边界及重复说明。

- [ ] [vLLM MLA packed 缓存与 DSA：字段、选择索引和稀疏读取](wiki/implementation/vllm-mla-packed-cache-and-dsa-execution.md)
- [ ] [DSA 的设备执行：DeepGEMM logits 与 FlashMLA sparse attention](wiki/implementation/dsa-sparse-device-execution.md)

## 16 · 分布式与训练状态（5 页）

重点：区分训练分片、推理并行和服务编排，解释各产物和通信的实际用途。

- [ ] [多 GPU 集合通信：归约、分片与 ring 成本](wiki/fundamentals/hardware/gpu-collectives-and-communication-cost.md)
- [ ] [张量并行与量化：分片、通信和元数据](wiki/implementation/tensor-parallel-quantization.md)
- [ ] [分布式 LLM 服务编排：路由、副本与 KV 交接](wiki/implementation/distributed-llm-serving-orchestration.md)
- [ ] [FSDP2：参数分片、计算期聚合与checkpoint](wiki/implementation/fsdp2-parameter-lifecycle-and-checkpoint.md)
- [ ] [DeepSeek-V3的FP8训练：GEMM、缓存与通信](wiki/implementation/deepseek-v3-fp8-training.md)

## 17 · KV 量化的公共解释（3 页）

重点：缓存对象、误差传播、读取粒度和生成过程的联系；定义与条件是否完整。

- [ ] [KV cache 量化的对象与粒度](wiki/theory/kv-cache-quantization-objects-and-granularity.md)
- [ ] [Attention 量化误差传播：从 KV 重构到输出与生成](wiki/theory/attention-quantization-error-propagation.md)
- [ ] [Attention sink 选头图谱：量化迁移、集合边界与校准](wiki/theory/attention-sink-map-transfer.md)

## 18 · KV 量化与服务实现（4 页）

重点：方法改了存储、更新还是 Attention 计算；质量证据与服务代价是否区分。

- [ ] [KIVI：键按通道、值按 token 的 2 bit KV cache 量化](wiki/methods/kivi.md)
- [ ] [KVQuant：RoPE 前缓存、敏感度码本与稀疏离群值](wiki/methods/kvquant.md)
- [ ] [SAW-INT4：面向真实服务约束的 4 bit KV cache 量化](wiki/methods/saw-int4.md)
- [ ] [OSCAR 的混合 KV 缓存：INT2 格式、窗口更新与分层 Attention](wiki/implementation/oscar-mixed-kv-cache-execution.md)

## 19 · KV 变换与动态精度（5 页）

重点：查询、缓存统计、编码与动态选择分别起什么作用，检查方法间比较条件。

- [ ] [Kitty：动态 K 通道提精度与双层 2-bit 缓存页](wiki/methods/kitty.md)
- [ ] [NOVA-KV：查询加权变换与等体积分组向量量化](wiki/methods/nova-kv.md)
- [ ] [WUSH-KV：联合缓存统计与读取敏感度的可逆变换](wiki/methods/wush-kv.md)
- [ ] [QJL：用二值随机投影直接估计注意力分数](wiki/methods/qjl.md)
- [ ] [TurboQuant：随机旋转码本与残差内积修正](wiki/methods/turboquant.md)

## 20 · GPU 计算与数值基础（4 页）

重点：概念在算子执行中的作用、必要前置和示例，检查基础页是否被具体实现细节淹没。

- [ ] [GPU 的执行模型与存储层次](wiki/fundamentals/hardware/gpu-execution-and-memory-hierarchy.md)
- [ ] [算术强度与 roofline 分析](wiki/fundamentals/hardware/arithmetic-intensity-and-roofline.md)
- [ ] [浮点表示与累加：范围、精度和舍入顺序](wiki/fundamentals/numeric-formats/floating-point-and-accumulation.md)
- [ ] [FP8 与 Microscaling 数值格式](wiki/fundamentals/numeric-formats/fp8-and-mx-data-formats.md)

## 21 · 算子映射、布局与编译（3 页）

重点：从逻辑计算到线程、地址和编译产物，避免术语与接口罗列。

- [ ] [GPU 算子的计算模式：逐元素、归约与分块矩阵乘](wiki/fundamentals/operators/gpu-kernel-computation-patterns.md)
- [ ] [张量布局与算子接口：从逻辑下标到内存地址](wiki/fundamentals/operators/tensor-layout-and-kernel-contracts.md)
- [ ] [Triton 语言与编译语义：从逻辑张量到 GPU Kernel](wiki/fundamentals/operators/triton-language-and-compilation.md)

## 22 · Tensor Core、流水与 CUTLASS（4 页）

重点：各层抽象的职责，异步搬运、计算和缓冲复用怎样配合。

- [ ] [Tensor Core 与量化 GEMM：从 Tile 到寄存器和编译产物](wiki/implementation/tensor-core-quantized-gemm.md)
- [ ] [GPU 异步拷贝与多级流水：等待完成和缓冲复用](wiki/implementation/gpu-async-copy-pipelines.md)
- [ ] [Hopper BF16 GEMM：TMA、WGMMA 与缓冲复用](wiki/implementation/hopper-gemm-tma-wgmma-pipeline.md)
- [ ] [CUTLASS GEMM 的组成：类型、运行参数与调度](wiki/implementation/cutlass-gemm-composition-and-dispatch.md)

## 23 · 低位矩阵乘内核（3 页）

重点：打包、反量化、块尺度和硬件执行之间的关系，公共内容与实现专属条件是否分开。

- [ ] [权重量化反量化内核的契约：AWQ 与 marlin 两种实现](wiki/implementation/weight-only-dequant-kernels.md)
- [ ] [Marlin：批处理 W4A16 GEMM 内核](wiki/implementation/marlin-batched-w4a16-gemm.md)
- [ ] [W8A8 算子：激活量化、整数 GEMM 与输出校正](wiki/implementation/w8a8-quantization-gemm-kernels.md)

## 24 · Blackwell 块缩放执行（2 页）

重点：硬件接口与算法计算的对应，输入尺度、累加与输出的完整流程。

- [ ] [GB300 NVFP4 GEMM：输入双环、TMEM 累加与输出复用](wiki/implementation/blackwell-nvfp4-tmem-execution.md)
- [ ] [CUTLASS SM100 块缩放 GEMM：指令选择、协作分块与尺度布局](wiki/implementation/cutlass-blackwell-block-scaled-gemm.md)

## 25 · 归一化、激活与归约（4 页）

重点：算子在模型中的作用、数值语义及线程分工，区分必要例子和重复基础。

- [ ] [LayerNorm、RMSNorm 与部分统计：归一化和不变性的条件](wiki/fundamentals/operators/layernorm-rmsnorm-and-partial-statistics.md)
- [ ] [RMSNorm 的 CUDA 与 Triton 实现：行归约、残差与舍入](wiki/implementation/rmsnorm-cuda-triton-kernels.md)
- [ ] [CUDA softmax：线程分工、局部统计与两级归约](wiki/implementation/cuda-softmax-row-reductions.md)
- [ ] [门控激活的 CUDA 与 Triton 实现：工作映射、向量化与融合](wiki/implementation/gated-activation-cuda-triton-kernels.md)

## 26 · FlashAttention 原理与实现（3 页）

重点：IO 问题、分块重计算、并行分工与 CUDA 实现是否连贯，公式与状态是否定义齐全。

- [ ] [FlashAttention：精确分块、IO复杂度与反向重计算](wiki/theory/flash-attention-tiling-and-recomputation.md)
- [ ] [FlashAttention-2：累计状态、序列并行与warp分工](wiki/theory/flash-attention-2-work-partitioning.md)
- [ ] [FlashAttention-2 CUDA：普通前向、精度与反向归约](wiki/implementation/flash-attention-2-cuda-execution.md)

## 27 · 算子接入、正确性与评测规范（3 页）

重点：检查文档能否指导判断；区分参考结果、测试覆盖和性能结论，不运行实验。

- [ ] [CUDA 算子的编写、编译与 PyTorch 接入](wiki/implementation/cuda-operator-build-and-integration.md)
- [ ] [算子正确性与性能测量：参考结果、误差和计时边界](wiki/implementation/kernel-correctness-and-benchmarking.md)
- [ ] [模型生成 Kernel 的任务与评测：覆盖率、搜索预算和条件速度](wiki/implementation/llm-generated-kernel-evaluation.md)

## 28 · 性能分析与配置选择（2 页）

重点：瓶颈假设、测量证据与调参选择的关系，清除重复流程或无依据建议。

- [ ] [GPU 算子性能分析：从时间线到瓶颈假设](wiki/implementation/gpu-kernel-performance-analysis.md)
- [ ] [Kernel 配置选择与自动调优：资源、工作划分和测量](wiki/implementation/kernel-configuration-and-autotuning.md)

## 29 · 量化与校准基础（3 页）

重点：量化解决的问题、训练和校准的职责；公式、例子及方法专属内容的归属。

- [ ] [均匀量化与分组](wiki/fundamentals/quantization/uniform-quantization-and-groups.md)
- [ ] [PTQ、QAT 与代理梯度](wiki/fundamentals/quantization/post-training-and-quantization-aware-training.md)
- [ ] [校准数据与量化范围选择](wiki/theory/calibration-and-range-selection.md)

## 30 · 二阶误差与重构（2 页）

重点：精确关系、代理目标与近似假设是否区分；检查长页的必要前置和重复展开。

- [ ] [层输出重构与二阶误差补偿](wiki/theory/layer-reconstruction-second-order-compensation.md)
- [ ] [曲率加权量化误差：Fisher、双侧度量与量化集合](wiki/theory/curvature-weighted-quantization-error.md)

## 31 · AWQ、GPTQ 及实现（5 页）

重点：算法选择如何成为校准、导出与消费路径；比较页与方法页不重复维护实现细节。

- [ ] [AWQ：激活感知的权重量化](wiki/methods/awq.md)
- [ ] [GPTQ：基于二阶补偿的权重量化](wiki/methods/gptq.md)
- [ ] [AWQ 的实现核对：从量化脚本到 vLLM 与 SGLang](wiki/implementation/awq-implementation.md)
- [ ] [GPTQ 的实现核对：校准、导出协议与服务内核选择](wiki/implementation/gptq-implementation.md)
- [ ] [AWQ、GPTQ 与 SmoothQuant：对象、机制与证据比较](wiki/methods/awq-gptq-smoothquant-comparison.md)

## 32 · 平滑、缩放与 W8A8（4 页）

重点：等价变换解决什么问题，怎样落到表示与算子；比较离线和在线成本。

- [ ] [对角缩放与等价变换](wiki/theory/diagonal-scaling-equivalent-transform.md)
- [ ] [SmoothQuant：迁移激活量化难度的 W8A8 方法](wiki/methods/smoothquant.md)
- [ ] [SmoothQuant 的实现核对：平滑融合、INT8 接口与离线边界](wiki/implementation/smoothquant-implementation.md)
- [ ] [SmoothQuant+：全模型平滑搜索与 W4A16 部署](wiki/methods/smoothquant-plus.md)

## 33 · 早期混合精度与系统协同（6 页）

重点：高精度分支、通道重排、量化设置和服务路径是否围绕共同目标解释。

- [ ] [LLM.int8()：向量级量化与混合精度分解](wiki/methods/llm-int8.md)
- [ ] [ZeroQuant：逐组权重、逐 token 激活与逐层蒸馏](wiki/methods/zeroquant.md)
- [ ] [Atom：离群通道重排、分组整数 GEMM 与服务成本](wiki/methods/atom.md)
- [ ] [Atom 的混合整数 GEMM 与分页 KV 读取](wiki/implementation/atom-mixed-gemm-and-paged-kv.md)
- [ ] [QServe：W4A8KV4 量化与系统协同](wiki/methods/qserve.md)
- [ ] [QServe/QoQ 的实现核对：两级编码、离线重排与 SGLang W4A8](wiki/implementation/qserve-implementation.md)

## 34 · 可学习变换与比较（4 页）

重点：共同问题、变量、训练目标和部署限制是否可比，综合页有无追加摘录。

- [ ] [OmniQuant：可学习裁剪与等价变换的块级校准](wiki/methods/omniquant.md)
- [ ] [AffineQuant：可学习仿射变换与渐进掩码](wiki/methods/affinequant.md)
- [ ] [FlatQuant：结构化可学习变换与低比特执行](wiki/methods/flatquant.md)
- [ ] [OmniQuant、AffineQuant 与 FlatQuant：可学习变换路线比较](wiki/research/omniquant-affinequant-flatquant-comparison.md)

## 35 · 旋转的数学基础与固定路线（3 页）

重点：变换的作用、等价条件、量化误差及在线成本之间的联系。

- [ ] [可逆变换、数值条件与 Kronecker 乘积](wiki/fundamentals/mathematics/invertible-transforms-and-kronecker-products.md)
- [ ] [正交旋转与量化：等价条件、离群值和在线代价](wiki/theory/orthogonal-rotation-and-hadamard-quantization.md)
- [ ] [QuaRot：固定旋转、全图等价与四位推理](wiki/methods/quarot.md)

## 36 · 学习旋转与混合子空间（6 页）

重点：训练目标、变换位置、子空间划分和真实执行边界是否明确。

- [ ] [SpinQuant：用最终模型损失学习正交旋转](wiki/methods/spinquant.md)
- [ ] [OSTQuant：正交与缩放联合学习、QSUR 与 KL-Top](wiki/methods/ostquant.md)
- [ ] [ConQuR：顶点对齐、Procrustes 更新与逐批旋转校准](wiki/methods/conqur.md)
- [ ] [ResQ：高精度子空间、组内旋转与混合精度执行](wiki/methods/resq.md)
- [ ] [ResQ 实现：未中心化基底、权重融合与浮点模拟量化](wiki/implementation/resq-basis-and-fake-quantization.md)
- [ ] [激活引导补偿与正交残差：从误差分解到 L2-SmoothRot](wiki/methods/activation-guided-compensation.md)

## 37 · 裁剪、步长与量化训练（6 页）

重点：目标怎样影响离散表示，代理梯度、初始化和适用范围是否解释充分。

- [ ] [PACT：任务驱动的激活裁剪与代理梯度](wiki/methods/pact.md)
- [ ] [LSQ：任务驱动的步长学习与梯度缩放](wiki/methods/lsq.md)
- [ ] [LSQ+：可学习偏移、初始化与有符号激活](wiki/methods/lsq-plus.md)
- [ ] [LQ-Nets：二进制基码值、交替拟合与任务训练](wiki/methods/lq-nets.md)
- [ ] [LC-QAT：仿射整数网格的向量量化训练](wiki/methods/lc-qat.md)
- [ ] [LLM-QAT：生成数据蒸馏与权重、激活、KV 联合量化](wiki/methods/llm-qat.md)

## 38 · 舍入、块重构与恢复（6 页）

重点：各方法优化的对象和粒度，校准、微调与质量恢复证据的衔接。

- [ ] [AdaRound：面向任务损失的自适应舍入](wiki/methods/adaround.md)
- [ ] [BRECQ：块级重构与二阶误差的粒度选择](wiki/methods/brecq.md)
- [ ] [FlexRound：逐元素除法选码与共同网格学习](wiki/methods/flexround.md)
- [ ] [SignRound：舍入偏移、裁剪与符号梯度搜索](wiki/methods/signround.md)
- [ ] [Norm Tweaking：量化后仅调整归一化参数](wiki/methods/norm-tweaking.md)
- [ ] [PRQuant：连续尾块与静态权重残差补偿](wiki/methods/prquant.md)

## 39 · 低秩适配与误差补偿（4 页）

重点：任务适配、初始化和量化补偿的作用区别；变量、预算与部署产物是否清楚。

- [ ] [LoRA：低秩任务适配、训练目标与参数预算](wiki/methods/lora.md)
- [ ] [LoftQ：量化主体与低秩适配器的交替初始化](wiki/methods/loftq.md)
- [ ] [QERA：激活二阶矩加权的低秩量化误差修正](wiki/methods/qera.md)
- [ ] [LRQ：低秩对数缩放、块重构与参数预算](wiki/methods/lrq.md)

## 40 · 联合训练与量化搜索（5 页）

重点：离散选码、连续参数和搜索约束的关系；源码事实与论文目标是否混用。

- [ ] [EfficientQAT：逐块全参数训练与整网尺度微调](wiki/methods/efficientqat.md)
- [ ] [PV-Tuning：连续参数与离散选码的联合微调](wiki/methods/pv-tuning.md)
- [ ] [AQLM 的 PV 微调：梯度、选码目标与权重写回](wiki/implementation/aqlm-pv-training-and-code-updates.md)
- [ ] [QUASAR：训练中的损失感知重构](wiki/methods/quasar.md)
- [ ] [QuantForge 与 HiRes：实验约束的搜索与 MXFP4 分阶段重构](wiki/methods/quantforge.md)

## 41 · 混合精度预算与敏感性（6 页）

重点：选择变量、代价、目标和求解条件是否完整，代理指标是否被误当作最终质量。

- [ ] [混合精度分配：选择变量、预算与部署口径](wiki/theory/mixed-precision-allocation.md)
- [ ] [HAWQ-V2：平均曲率、扰动能量与混合精度搜索](wiki/methods/hawq-v2.md)
- [ ] [CMPQ：按输入通道分配精度与双类离群保护](wiki/methods/cmpq.md)
- [ ] [CASA：方向敏感的位宽分配与跨层交换](wiki/methods/casa.md)
- [ ] [CoopQ：联合量化的边际贡献、交互代理与预算求解](wiki/methods/coopq.md)
- [ ] [LLM-MQ：候选梯度评分、稀疏保护与混合位宽 GEMV](wiki/methods/llm-mq.md)

## 42 · 校准代理与误差诊断（6 页）

重点：指标为什么有用、不能说明什么；推导、诊断建议和证据是否对得上。

- [ ] [熵、条件熵与依赖：量化代理指标的解释边界](wiki/fundamentals/mathematics/entropy-and-dependence.md)
- [ ] [积分梯度与量化敏感性：基线、路径和归因边界](wiki/theory/integrated-gradients-and-quantization-sensitivity.md)
- [ ] [QIG：以量化误差积分梯度指导 token 加权校准](wiki/methods/qig.md)
- [ ] [LUQ：以层激活熵选择超低比特量化层](wiki/methods/luq.md)
- [ ] [Quantization Price：用输出分布代价选择 PTQ 配置](wiki/methods/quantization-price-prediction.md)
- [ ] [量化误差诊断与验证](wiki/implementation/quantization-error-diagnosis.md)

## 43 · 模型结构与生成基础（4 页）

重点：先解释模型组件与生成任务，再说明量化对象；长基础页的主线和前置是否清楚。

- [ ] [Transformer 与自回归推理：从 token 到 KV cache](wiki/fundamentals/model/transformer-autoregressive-inference.md)
- [ ] [transformers Qwen3：Q/K 归一化、动态缓存与专家路由](wiki/implementation/transformers-qwen3-cache-and-moe-execution.md)
- [ ] [推理模型后训练：GRPO、模式控制与蒸馏](wiki/fundamentals/model/reasoning-post-training-and-distillation.md)
- [ ] [推测解码：候选、目标验证与每轮成本](wiki/fundamentals/model/speculative-decoding-proposals-verification-and-cost.md)

## 44 · 视觉语言输入与模型执行（3 页）

重点：图像、token、位置和缓存怎样进入模型，基础页与具体模型实现如何分工。

- [ ] [视觉语言模型中的 token 与量化对象](wiki/fundamentals/model/vision-language-model-tokens-and-quantization.md)
- [ ] [Qwen-VL 的视觉输入准备与微调样本](wiki/implementation/qwen-vl-input-preparation-and-finetuning.md)
- [ ] [Qwen2.5-VL：视觉输入、三轴位置与生成缓存](wiki/implementation/qwen25-vl-visual-input-and-position-execution.md)

## 45 · 多模态量化目标与比较（5 页）

重点：模态差异怎样进入优化目标；比较页是否真正解释共同点和不同条件。

- [ ] [MBQ：模态平衡的视觉语言模型量化](wiki/methods/mbq.md)
- [ ] [VLMQ：token 重要性加权的二阶量化](wiki/methods/vlmq.md)
- [ ] [Q-VLM：熵驱动的联合校准与视觉编码器优化](wiki/methods/q-vlm.md)
- [ ] [MBQ 与 VLMQ：重要性怎样进入量化目标](wiki/research/mbq-vlmq-comparison.md)
- [ ] [多模态对齐诊断：共同方向、主角谱与任务相关性](wiki/theory/multimodal-alignment-diagnostics.md)

## 46 · 多模态平滑与补偿（5 页）

重点：共享权重、模态分支、变换和补偿的整体关系，区分内部机制与任务作用。

- [ ] [MASQuant：分模态平滑与共享权重补偿](wiki/methods/masquant.md)
- [ ] [MQuant：模态静态量化、等价重排与旋转幅度抑制](wiki/methods/mquant.md)
- [ ] [SplitQ：模态离群通道拆分与权重激活双补偿](wiki/methods/splitq.md)
- [ ] [RGSQ：模态曲率、加权重构与稀疏旋转的边界](wiki/methods/rgsq.md)
- [ ] [VEQ：模态与专家亲和性加权的 MoE 量化](wiki/methods/veq.md)

## 47 · 多模态适配与缓存压缩（6 页）

重点：训练、浮点恢复、缓存保护和脉冲编码分别解决什么问题，避免统称推理加速。

- [ ] [QSLAW：多模态适配中的分组尺度学习与数据 warmup](wiki/methods/qslaw.md)
- [ ] [QSLAW 的尺度参数、混合数据与浮点恢复路径](wiki/implementation/qslaw-scale-learning-and-float-export.md)
- [ ] [AKVQ-VL：按文本与枢纽 token 保护多模态 KV](wiki/methods/akvq-vl.md)
- [ ] [CalibQuant：视觉 KV 的逐通道一位量化与分数校准](wiki/methods/calibquant.md)
- [ ] [VidKV：视频 KV 的混合位宽、FFT 与三值保护](wiki/methods/vidkv.md)
- [ ] [SpikeMLLM：模态时间步分配与量化值的脉冲编码](wiki/methods/spikemllm.md)

## 48 · 视觉 token 剪枝与运行接入（5 页）

重点：选择依据、层次、位置和缓存怎样共同变化；压缩比例与实际加速是否分开。

- [ ] [SAPrune：按层深选择时机与双路径视觉 token 剪枝](wiki/methods/saprune.md)
- [ ] [CoRePrune：深度与删除集合条件下的视觉 token 可删除性](wiki/methods/coreprune.md)
- [ ] [QAPruner：量化敏感度与视觉 token 选择的耦合](wiki/methods/qapruner.md)
- [ ] [LayerPos：剪枝后的位置重编号与逐层取舍](wiki/methods/layerpos.md)
- [ ] [视觉token剪枝的运行接入：共同子序列、位置与缓存](wiki/implementation/visual-token-pruning-runtime-integration.md)

## 49 · 视觉压缩的不同对象（4 页）

重点：区分 token、注意力路径、层和表示码率，核对任务收益与实现条件。

- [ ] [CoVisco：原生摘要token、分段通信与两层视觉选择](wiki/methods/covisco.md)
- [ ] [MWOP：按模态剪注意力路径与FFN执行宽度](wiki/methods/mwop.md)
- [ ] [RDTok：不依赖问题的视觉特征码率分配](wiki/methods/rdtok.md)
- [ ] [医学LLaVA压缩：输出相似度删层、恢复微调与AWQ](wiki/methods/medical-llava-prune-sft-quantize.md)

## 50 · 音视频与语音生成（4 页）

重点：媒体准备、模型状态、流式生成与质量评测的关系，清除多页重复执行描述。

- [ ] [多模态感知与流式语音生成：时间对齐和Thinker-Talker](wiki/fundamentals/model/multimodal-perception-and-streaming-speech-generation.md)
- [ ] [Qwen2.5-Omni调用链：媒体准备、返回协议与低显存执行](wiki/implementation/qwen25-omni-multimodal-input-and-audio-output.md)
- [ ] [Qwen2.5-Omni模型执行：音画输入、Talker状态与完整波形返回](wiki/implementation/qwen25-omni-model-generation-execution.md)
- [ ] [TTS 量化评测：组件敏感性、质量判据与真实执行](wiki/implementation/tts-quantization-evaluation.md)

## 51 · 动作模型与量化（4 页）

重点：明确动作任务、策略误差、在线输入与执行窗口，避免直接套用文本生成解释。

- [ ] [FoldQuantVLA：共享坐标一致性与原生整数动作推理](wiki/methods/foldquantvla.md)
- [ ] [PIVOT-Q：学生轨迹上的稀疏动作恢复与行为锚点](wiki/methods/pivot-q.md)
- [ ] [OnPTQ：量化策略轨迹、决策风险与未来分支校准](wiki/methods/onptq.md)
- [ ] [VLA的CPU动作块执行：供给窗口、指令缓存与W8A8](wiki/implementation/vla-cpu-action-chunk-execution.md)

## 52 · 其他应用与分阶段部署（4 页）

重点：生成模型、CPU 图和 prefill/decode 分工的实际目标、条件与收益归因。

- [ ] [CentriQ：逐 token 均值旁路与免校准 DiT 量化](wiki/methods/centriq.md)
- [ ] [Ideogram 4.0 的 INT8/GGUF PTQ：层保护与生成评测协议](wiki/methods/ideogram-int8-gguf.md)
- [ ] [RAMP：输出散度、聚类阈值与 CPU 计算图成本](wiki/methods/ramp.md)
- [ ] [Disaggregated Quantization：分阶段格式、权重与流式 prefill](wiki/methods/disaggregated-quantization.md)

## 53 · 质量、可靠性与排序（4 页）

重点：指标要回答的问题、判据、统计条件与结论边界；通用页和专题页的解释分工。

- [ ] [模型质量评测：PPL、任务协议与选择性预测](wiki/implementation/model-quality-evaluation.md)
- [ ] [量化与可靠性：选择性预测评测](wiki/theory/quantization-reliability-and-selective-prediction.md)
- [ ] [量化后的排序稳定性：间隔、检索损害与精度分配](wiki/theory/quantization-ranking-stability.md)
- [ ] [VLM 压缩评测框架：从任务精度到可信度](wiki/implementation/lvlm-compression-benchmark.md)

## 54 · 具体评测协议与服务指标（6 页）

重点：输入构造、调用、计分和统计口径是否连贯，避免将不同协议的数字直接比较。

- [ ] [HELM：场景、提示协议与多指标评测](wiki/implementation/helm-evaluation-protocol.md)
- [ ] [lm-evaluation-harness：请求、答案过滤与聚合](wiki/implementation/lm-evaluation-harness-request-filter-and-aggregation.md)
- [ ] [MMLU-Pro：题目构建、选项数与答案解析](wiki/implementation/mmlu-pro-options-and-answer-extraction.md)
- [ ] [LongBench v2：实际上下文、推理调用与答案计分](wiki/implementation/longbench-v2-context-and-scoring.md)
- [ ] [RULER：任务复杂度、召回与有效上下文](wiki/implementation/ruler-task-complexity-and-effective-context.md)
- [ ] [推理服务评测：负载、延迟、吞吐与 SLO](wiki/implementation/serving-performance-evaluation.md)
