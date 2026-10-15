# Wiki 变动记录

记录修改、原因、检查结果与遗留影响，同批变动合并记录。各条范围与计数对应当时版本；规则见 [README](README.md)，阅读入口见 [INDEX](INDEX.md)。

## 2026-10-15 · Ingest 明确研究对象的作用

- 将对象与所属任务的关系融入 wiki-ingest 的阅读、写入和解释自检，并明确读取正文写作规范、写清对象与动作、逐段去重和删除无用内容；README 同步说明设计理由。保留固定章节，不要求局部页重讲全系统。复核与 Wiki 规范的一致性及修改差异，未执行新论文入库。

## 2026-10-15 · 整体解释与阅读路径改进

- 将对象的整体作用、工作方式与局部机制的联系融入 Wiki 写作、实现分析和维护条款；AGENTS 保留简短入口，SELF 调整完成判断，wiki-ingest 与 wiki-review 的设计/执行同步修改。沿现有步骤落实，不新增模板、检查器或 Skill，不要求局部页重讲全系统。
- [部署页](implementation/quantized-llm-deployment-backends.md)补 MLC-LLM `6474f7db` 的配置、参数缓存、模型库与 Python 引擎交接，说明 JIT 入口和参数统计口径；定向回查固定配置/转换/编译/引擎源码。未扩展为编译 pass、代码生成、C++ 调度或设备核验，READING 与局限说明同步。
- [nano-vllm](implementation/nano-vllm-inference-execution.md)补组件与状态分工，定向复核 `bb823b3e` 的轮次、调度/后处理、块释放、KV 池与 Attention 读写；[模型基础](fundamentals/model/transformer-autoregressive-inference.md)综合已有解释，比较 GQA、QK-Norm、MoE、MLA 与视觉输入怎样改变量化对象，不新增模型支持或性能结论。
- SELF 压缩已由正文、READING 与历史 LOG 承接的逐项细节，保留主线、入口和关键限制；[INDEX](INDEX.md)保留全部 48 条既有阅读路径，按基础、方法、表示、模型/缓存、系统/算子和质量组织，补模型差异入口。

- 阅读复核：从入口回读三处改动，能沿正文解释 MLC 配套产物、nano-vllm 状态职责及模型差异的量化含义；新增知识链接逐项读取目标。另抽查 AWQ 开头与校准/产物主线、RMSNorm 开头与接口/归约说明，已有解释充分，未套模板改写。两个 Skill 的设计与执行变更逐项对应；这属于内容阅读与静态流程检查，未重新执行论文 ingest 或开展实验，也不代表全库语义审查。
- 机械检查：7 个受影响 Wiki 文件、1628 条本地链接 strict 通过，0 错误/待核；根目录与 Skill 文档 6 文件的 54 个本地链接目标存在；48 条阅读路径完整保留，差异格式检查通过。原有 Obsidian 工作区改动保留。

## 2026-10-15 · Triton-Puzzles研读：练习骨架与已有覆盖

- 完整静态研读固定4d794abed02292081500ebf4b1e35cd5fc5fb019的条目/根README、notebook49cell及全部16内嵌PNG；四个source文件的身份/保存证明复用原登记。4个访存demo有实现，12题kernel仍为TODO，19代码cell无保存execution_count/outputs，未将图示或题目目标当已实现/通过。
- 核对spec注解造数、nelem/B网格、trace/viz与打印式allclose；launcher无诊断返回，不能由failures分支推定已核访存检查。Puzzle12为低nibble优先、G8的scale/整数零点教学格式，参考先物化浮点权重；没有量化/pack或融合kernel，未套成AWQ格式。
- 与[语言](fundamentals/operators/triton-language-and-compilation.md)、[计算模式](fundamentals/operators/gpu-kernel-computation-patterns.md)、[布局](fundamentals/operators/tensor-layout-and-kernel-contracts.md)、[反量化](implementation/weight-only-dequant-kernels.md)和[验证](implementation/kernel-correctness-and-benchmarking.md)对照，已有机制解释充分；保留教材学习用途，不新增重复正文、INDEX或sources。不执行notebook，不补外部triton-viz/依赖或GPU结论。
- READING/SELF/计划同步：207条有依据、22条队列（12论文/7文档/3代码），本轮基线88已处理66条，第20批六项选定来源范围完成；下一项第21批系统课程材料。独立Astra review通过，管理链接检查通过（2文件、955条本地链接），git diff --check通过；结构/来源复用上一项203知识页、728项有效结果。提交本项4管理文件，Obsidian改动保留。

## 2026-10-15 · Triton Fused Attention ingest：梯度拥有者与log2统计

- 完整核读snapshot-2026-07-02保存764行/代码、48行README，包括前向/descriptor/配置、预处理/反向两扫描、包装/测试/性能及保存输出，无本地附件；网页与论文v1、后续CUDA库commit分别登记。
- [FA2分工页](theory/flash-attention-2-work-partitioning.md)第2节后补独立Triton对照：每program完整拥有128-key的dK/dV及128-query的dQ，用两条重算扫描替代所选CUDA跨CTA dQ工作区；log2 M、K乘τlog2e、dQ乘ln2与dK乘τ配套解释，保留已舍入O/dO行统计和FP16中间值，未补出注释所称D预除。
- 等长同头/无dropout、N%128与stride/有限mask条件就近保留。测试跳过FP8 bwd而benchmark只在fwd转换，所以保存FP8反向标签实际FP16；Hopper WS条件按非因果fwd代码、保存表全WS=False/无Flash列、scale与多阶段反向计时口径集中登记。不复制在线softmax/IO推导或新建实现综述。
- INDEX/READING/SELF/计划同步：206条有依据、23条队列（12论文/7文档/4代码），本轮基线88已处理65条；下一项Triton-Puzzles。独立Astra原件/正文及记录review通过，无必修项；结构/来源strict检查203知识页、728项登记，0错误/待核，4个受影响Wiki页1396条本地链接strict及diff检查通过；206/58/23记录与12/7/4分类数相符。未新增测试或执行上游/编译/GPU，提交本项6文件，Obsidian改动保留。

## 2026-10-15 · Triton Persistent Matmul ingest：访问块与输出暂存配置

- 完整核读snapshot-2026-07-02保存730行/代码、48行README，五条普通/host descriptor/device descriptor路径、hook/验证/benchmark与CLI均覆盖，无本地附件。网页范围与既有固定代码证据分别登记。
- [配置页](implementation/kernel-configuration-and-autotuning.md)第5节补候选tile与descriptor block_shape联动、完整FP32累加后沿N半宽写回以缩小epilogue暂存、Hopper且启用WS时的host跳过/device flatten与subtile裁剪条件；复用刚完成的布局接口，不复制普通persistent或显式barrier。第4节原普通分支范围改为准确指向新增配置说明。
- 保存TMA⭕按代码解释为未启用；检查以普通tiled结果为参考、转FP16、atol=1.0且失败仅打印，benchmark循环调用会构造descriptor/分配C的包装，不套用Group GEMM预制仅launch口径；实际shared/stages与生成同步仍按编译与运行证据判断。
- INDEX/READING/SELF/计划同步：205条有依据、24条队列（12论文/8文档/4代码），本轮基线88已处理64条；下一项Triton Fused Attention。独立Astra原件/正文及记录review通过，无必修项；结构/来源strict检查203知识页、727项登记，0错误/待核，4个受影响Wiki页1403条本地链接strict及diff检查通过；205/58/24记录与12/8/4分类数相符。未新增测试或执行上游/编译/GPU，提交本项6文件，Obsidian改动保留。

## 2026-10-15 · Triton Group GEMM ingest：问题表、输入接口与调优key

- 完整核读snapshot-2026-07-02保存561行/代码、48行README，普通与TMA分支均完整保存，无本地附件。定向补核固定Triton81a46fa0的descriptor load/store、make_tensor_descriptor、multiple_of及Autotuner.run；网页身份与源码语义分别登记，未重新执行归档或设备检查。
- [配置页](implementation/kernel-configuration-and-autotuning.md)复用已有全局tile前缀解释，补host问题表到设备静态定位、两分支NUM_SM及group_size key看不到表内shape/stride/payload dtype的例子；保留JIT特化不等于重新调优。预制四同形组对逐组torch调用的计时协议就近说明，不复制历史性能表。
- [布局页](fundamentals/operators/tensor-layout-and-kernel-contracts.md)对照普通FP16硬类型/内维stride1/无mask与TMA B[N,K]/统一dtype/descriptor边界和16B前提，解释B.T.contiguous物理成本；来源实际仅FP16检查，未用FP8分支推定通过。原有CUDA包装段分出具体小标题，保持阅读主线。
- INDEX/READING/SELF/计划同步：204条有依据、25条队列（12论文/9文档/4代码），本轮基线88已处理63条；下一项Triton Persistent Matmul。独立Astra原件/正文及记录review通过，无必修项；两处新增互链改用精确标题锚点后，结构/来源strict检查203知识页、726项登记，0错误/待核，5个受影响Wiki页1412条本地链接strict及diff检查通过；204/58/25记录与12/9/4分类数相符。未新增测试或执行上游/编译/GPU，提交本项7文件，Obsidian改动保留。

## 2026-10-15 · Triton Matmul ingest：分组任务映射与缓存复用

- 完整核读snapshot-2026-07-02保存443行正文/代码、README与唯一2026-09-21补采图，并定向对照已归档81a46fa0源码kernel/wrapper；网页、源码和图像补采身份分别保留。
- [计算模式](fundamentals/operators/gpu-kernel-computation-patterns.md)第4节展开grouped program ID：M方向G行分组、组内M先变化、末组真实高度r解码；用来源9×9图解释90→54为前9输出所需的不同输入块集合，保留真实缓存/调度与同步条件。未减少数学K归约或推定实际访存/速度；已有tile、mask、指针、精度和autotune覆盖复用，不新建页/测试/排行。
- INDEX/READING/SELF/计划同步：203条有依据、26条队列（12论文/10文档/4代码），本轮基线88已处理62条；下一项Triton Group GEMM。独立Astra原件/正文及记录review通过，无必修项；结构/来源strict检查203知识页、724项登记，0错误/待核，4个受影响Wiki页1394条本地链接strict及diff检查通过；203/58/26记录与12/10/4分类数相符。未新增测试或执行上游/编译/GPU，提交本项6文件，Obsidian改动保留。

## 2026-10-15 · Triton Introduction ingest：块级数据流与编译职责

- 完整核读snapshot-2026-07-02保存97行、51行README与两张2026-09-21补采图；网页快照与独立Triton源码commit身份分别登记，未据共同归档说明合称同一版本。
- [语言页](fundamentals/operators/triton-language-and-compilation.md)开头补编译器依据控制流/数据流静态安排迭代块的设计说明，连接已有逻辑张量到线程/存储的分工；program内工作安排与运行时CTA调度分别解释。现有分块GEMM、逻辑shape/线程及语言语义覆盖充分，不新建页或复制伪代码/优化能力清单。
- INDEX/READING/SELF/计划同步：202条有依据、27条队列（12论文/11文档/4代码），本轮基线88已处理61条；下一项Triton Matrix Multiplication。独立Astra原件/正文及记录review通过，无必修项；结构/来源strict检查203知识页、723项登记，0错误/待核，4个受影响Wiki页1381条本地链接strict及diff检查通过；202/58/27记录与12/11/4分类数相符。不新增测试或执行上游/编译/GPU，提交本项6文件，Obsidian改动保留。

## 2026-10-15 · fast.cu ingest：H100固定路径与GB300 TMEM

- 核读固定commit `2dfe5e26aecfd9e5f27bf9d5837deea01acda24b`：H100驱动/M11与11→12全部差异，必要早期rung差异及experimental选段；GB300输入准备/runner/shell/r0完整、r9所选helper/几何/完整K组/尾入口与收尾/输出/调度。34文件保存完整性复用有效归档证明，未升级为全部分支语义核验；快照晚于H100文章。
- [Hopper页](implementation/hopper-gemm-tma-wgmma-pipeline.md)补full=1/empty=4的跨CTA归还、固定几何/Hilbert与M12 padded stmatrix，保留源码实际缺失的输出proxy、写前复用协调及末轮等待、wrapper缓存键和失败仍计时条件。[GB300页](implementation/blackwell-nvfp4-tmem-execution.md)贯通标准尺度、AB/SF两环、SM103a/K96、三槽最后reader和TMEM 0/220共享36列/先drain边缘的复用；不据局部wait证明完整drain同步，标准512B格式只链接已有解释。
- r9路由4096容量/实际异常降级、输出行对齐、小gate实际K尾而非M/N尾、用户shape参考与首个cuBLASLt heuristic、current/triton/sustained计时协议就近说明；未复制十rung速度表，sum仅核覆盖及已有归约不宜沿用的条件，未新建范例。
- INDEX/READING/SELF/计划同步：201条有依据、28条队列（12论文/12文档/4代码），本轮基线88已处理60条；第19批选定来源范围完成，下一项Triton Introduction。独立Astra原件/正文与管理记录review通过，无必修项；结构/来源strict检查203知识页、722项登记，0错误/待核，6个受影响Wiki页1387条本地链接strict及diff检查通过；201/58/28记录与12/12/4分类数相符。本批不运行上游/编译/GPU或新增测试，提交本项8文件，Obsidian改动保留。

## 2026-10-15 · H100 GEMM worklog ingest：TMA、WGMMA 与缓冲复用

- 完整核读599行保存正文、85行README与38个实际附件，正文哈希与登记一致；40处图引用缺swizzle/Hilbert小图，Conclusion性能plot未保存，补采图不声称历史字节同版。关联fast.cu只核README身份，晚于文章，不作为发表时实现证据。
- 新增[Hopper BF16 GEMM执行](implementation/hopper-gemm-tma-wgmma-pipeline.md)，按输入slot→TMA/full→WGMMA完成→empty复用贯通，连接双consumer沿M分工、QSIZE与角色寄存器预算、persistent局部性/multicast及输出sC生命周期；保留作者配置/性能口径与因果猜测归属，不复制11项成绩排行或硬件概述中的规格泛化。
- 定向增核已有CUDA async-copies保存正文519–565/1082–1153行，以官方多维TMA示例说明writer proxy可见性、发起线程bulk-group读取完成及共享源复用，未把该补充推定为博客省略wrapper的实现。异步流水与配置页各加一处必要入口，INDEX/READING/SELF/计划同步：200条有依据、29条队列（12论文/12文档/5代码），本轮基线88已处理59条；下一项fast.cu。
- 独立Astra原件/正文与管理记录review通过，无必修项。结构/来源strict检查202知识页、720项登记，0错误/待核；6个受影响Wiki页1388条本地链接strict及diff检查通过，200条有依据/58条历史范围待核/29条队列与12/12/5分类数一致。本批不新增算术测试或执行上游/GPU；提交范围为本项8文件，Obsidian改动保留。

## 2026-10-15 · CUTLASS历史博客 ingest：线程内外积与寄存器复用

- 完整核读snapshot-2026-07-02保存447行、README与10张2026-09-21补采图；正文/图11项哈希与登记一致。页首只保留2018-05-21更新，主体为preview/v0.1.1链接范围，不把抓取日期或图片路径当首次发布日期。
- 在[计算模式](fundamentals/operators/gpu-kernel-computation-patterns.md)第3节补常规SIMT线程的r×c累加状态及r个A/c个B值的外积复用，连接CTA shared复用、线程寄存器活跃状态与warp-cooperative WMMA的分工；线程逻辑行列可交错分布，未按透明图不可辨编号补造lane布局。
- 现有层次、两级缓冲/资源、epilogue、WMMA和测量解释充分，不复制旧类表或性能表；没有把旧双缓冲的特定屏障安排泛化，也没有为简单外积算术新建测试。
- INDEX/READING/SELF/计划同步：199条有依据、30条队列（12论文/13文档/5代码），本轮基线88已处理58条；下一项H100 GEMM worklog。独立Astra原件/正文及记录review通过，无必修项。结构/来源strict检查201知识页、718项登记，0错误/待核；4个受影响Wiki页1376条本地链接strict及diff检查通过，199/30记录与12/13/5分类数相符。本项六文件提交，既有Obsidian改动保留；没有新增或重复运行测试。

## 2026-10-15 · CUTLASS Tile Iterator ingest：旧接口的推进与偏移

- 完整核读snapshot-2026-07-02保存471行、README及同hash通用层次图，正文/图2项哈希核对；正文开头限定legacy CUTLASS2.x，并说明3.0弃用tile access iterators而转用cute::Tensor/Layout。
- 在[布局与接口](fundamentals/operators/tensor-layout-and-kernel-contracts.md)第2节后接旧iterator的当前tile/每线程Fragment、显式load/store与三种单位：预定序列中的tile、逻辑整tile、Element线性偏移。Mask的实现定义、contiguous存储与tile具体排列分别说明，复用现有TensorCore与低比特格式解释。
- 不复制整套重复概念声明，不把concept当C++20关键字或把弃用写成全部移除；访存Fragment与MMA片段需具体映射，未据概念补造packing、mask补零或设备同步。没有新增测试或运行上游。
- INDEX/READING/SELF/计划同步：198条有依据、31条队列（12论文/14文档/5代码），本轮基线88已处理57条；下一项CUTLASS技术博客。独立Astra原件/正文及记录review通过，无必修项。结构/来源strict检查201知识页、717项登记，0错误/待核；4个受影响Wiki页1367条本地链接strict及diff检查通过，198/31记录数相符。本项六文件提交，既有Obsidian改动保留；没有新增教学算术测试或运行上游。

## 2026-10-15 · CUTLASS Pipeline ingest：阶段复用与双方进度

- 完整核读snapshot-2026-07-02保存112行与README，无附件；正文哈希与登记一致，独立准备与Astra原件核读相符。
- 在[异步流水](implementation/gpu-async-copy-pipelines.md)第2节接入CUTLASS四API的阻塞/通知职责、初始empty与start state、4stage/2producer1consumer参与计数及各自PipelineState推进；显式保留写入完成可见性和PipelineTmaAsync commit可No-Op的条件。配置页Hopper段链接协议，不重复展开。
- 页首区分本协议与后续SM80 cp.async批次规则，原有CPU教学推演标为此前工作；源例PipelineAsync/PipelineSync命名差异集中保留，phase/transaction内部细节不由类名补造。没有新增测试或执行上游。
- INDEX/READING/SELF/计划同步：197条有依据、32条队列（12论文/15文档/5代码），本轮基线88已处理56条；下一项Tile Iterator Concepts。独立Astra原件/正文及记录review通过，无必修项。结构/来源strict检查201知识页、716项登记，0错误/待核；5个受影响Wiki页1368条本地链接strict及diff检查通过，197/32记录数相符。本项七文件提交，既有Obsidian改动保留，没有新增或重复运行测试。

## 2026-10-15 · CUTLASS Quickstart ingest：主机缓冲与实例调用

- 完整核读snapshot-2026-07-02保存794行、README与通用层次图；正文/同图2项哈希匹配，复用本批通用图及独立准备/Astra原件核读证据。
- 在[CUTLASS组成与调用](implementation/cutlass-gemm-composition-and-dispatch.md)补kernel typed stride、DeviceAllocation与A/B/C/D对象流、alpha/beta和输出位置；对照SM75 TensorRef与SM90独立D。Library Handle的首个兼容实例选择、构建过滤和不同include/link依赖也留在此页，复用现有布局/配置/SM100及CUDA接入解释。
- 保存示例未展示输入准备与结果消费，正文保留这些调用责任；Library中D未声明、SM100块缩放operation class前后不一按原件登记。一般CUDA最低前提不扩展为所有架构条件；没有新增无用测试或执行构建。
- INDEX/READING/SELF/计划同步：196条有依据、33条队列（12论文/16文档/5代码），本轮基线88已处理55条；下一项Synchronization primitives。独立Astra原件/正文及记录review通过，无必修项。结构/来源strict检查201知识页、715项登记，0错误/待核；4个受影响Wiki页1356条本地链接strict及diff检查通过，196/33记录数相符。本项六文件提交，既有Obsidian改动保留，没有编译/设备运行或新增算式测试。

## 2026-10-15 · Grouped Scheduler ingest：任务定位与负载分配

- 完整核读snapshot-2026-07-02保存311行、README及7张2026-09-21补采图；正文/图8项哈希核对一致，复用独立准备与Astra原件核读证据。
- 在[配置与工作划分](implementation/kernel-configuration-and-autotuning.md)第4节承接problem tile数/前缀→全局任务号→CTA固定步长→problem/local tile→完整输出，再解释device warp-prefix与host逐CTA tuple表、三角任务/macro/补齐条件及K长尾。CUTLASS组成页加任务定位入口，既有DeepGEMM具体专家映射保持其源码条件。
- 原文thread_idx/tile_idx统一为全局任务号；216/256冲突按四问题尺寸及表使用216。三角前缀公式用整数区间解释；排序/关联元数据、GA100例与约30%作者报告限于其参数，未新增教学算术测试或执行上游。
- INDEX/READING/SELF/计划同步：195条有依据、34条队列（12论文/17文档/5代码），本轮基线88已处理54条；下一项CUTLASS Quickstart。独立Astra正式review闭环通过：已修正device-only复用当前problem区间的说明，并复核上三角macro条件，无剩余必修。结构/来源strict检查201知识页、714项登记，0错误/待核；5个受影响Wiki页1363条本地链接strict及diff检查通过，195/34记录数相符。本项七文件提交，既有Obsidian改动保留；文字修正沿用有效检查，未重跑无关测试。

## 2026-10-15 · CUTLASS Layouts ingest：映射对象与存储容量

- 完整核读snapshot-2026-07-02保存280行、README及2026-09-21补采通用层次图，复用本批正文/图2项哈希一致的核对；现有stride/转置视图解释充分，保留原阅读路径。
- 在[张量布局](fundamentals/operators/tensor-layout-and-kernel-contracts.md)第2节补C++ Layout→TensorRef→TensorView的映射、指针与extent分工，以及二维rank/单一动态stride、显式contains、size/capacity与packed元数据职责。shape(2,3)/pitch5的容量和地址为教学展开，不新增算式测试；CuTe、int4底层与未展开类实现的范围集中在来源说明。
- INDEX/READING/SELF/计划同步：194条有依据、35条队列（12论文/18文档/5代码），本轮基线88已处理53条；下一项Grouped Kernel Schedulers。独立Astra原件/正文及记录review通过，无必修项；结构/来源strict检查201知识页、713项登记，0错误/待核，4个受影响Wiki页1353条本地链接strict及diff检查通过，194/35记录数相符。本项六文件提交，既有Obsidian改动保留，没有重复计算测试或运行上游。

## 2026-10-15 · CuTe DSL目录研读：无独立机制增量

- 完整核读CuTe DSL的snapshot-2026-07-02保存正文18行与README；目录实际只有这两个文件，正文为标题及16个子页链接，哈希与登记一致。
- 定向对照现有CUTLASS组成、张量布局、Triton编译与FlashInfer后端正文；本条目录没有独立机制增量，也不能补足CuTe DSL语言/编译语义。READING登记实际材料范围，不新增知识页或来源登记；未取得外链子页，没有执行或新增测试。
- READING记录说明同步容纳无需新增的判断；SELF/计划只更新进度及下一项Layouts/Tensors。当前193条有依据、36条队列（12论文/19文档/5代码），本轮基线88已处理52条。独立Astra原件/登记review通过，无必修；两页899条本地链接strict和diff检查通过，193/36实际记录数相符。仅四个管理文件变更，复用本批有效结构检查（201知识页、712项来源登记），没有重复运行与改动无关的检查。

## 2026-10-15 · Blackwell SM100 ingest：指令、分块与尺度布局

- 完整核读固定snapshot-2026-07-02保存5009行、表1–20、README与两张2026-09-21补采尺度图；正文/两图3项哈希与登记一致，复用独立准备和Astra原件核读证据。
- 新增[SM100块缩放GEMM](implementation/cutlass-blackwell-block-scaled-gemm.md)，沿输入类型/尺度→AB布局和alignment→指令kind/合法tile→2SM每CTA输出与cluster→512B尺度地址→epilogue D/SFD组织解释。块尺度执行页链接复用，TensorCore首段的寄存器片段说明限定于本页WMMA/mma.sync；INDEX接入。
- 保留MXFP4 NT三种合法tile、整体tile传给两个builder、SM90软件恢复/SM100原生尺度及SM120独立条件；地址公式标按图推导。来源范围集中登记，删去重复防御性提醒，没有新增算式断言、编译或设备测试。
- READING/SELF/计划同步：192条有依据、37条队列（12论文/20文档/5代码），本轮基线88已处理51条；下一项CuTe DSL，先按已收录目录范围判断覆盖。独立Astra正文及收尾记录review均通过，无必修项。结构/来源strict检查201知识页、712项登记，0错误/待核；6个受影响Wiki页1366条本地链接strict通过，diff及192/37记录核对通过。提交范围为本项8文件，既有Obsidian改动保留；没有执行上游或新增运行测试。

## 2026-10-15 · CUTLASS 3 API ingest：类型组装与参数分工

- 完整读取snapshot-2026-07-02保存493行/类型声明及配图、README，复用本批2项正文/图哈希核对；现有矩阵乘、TensorCore、布局/配置/流水正文已有基础，缺少类型组装与参数分工主线。
- 新增[CUTLASS GEMM组成](implementation/cutlass-gemm-composition-and-dispatch.md)，从mainloop/epilogue→kernel→adapter连接一次调用，再解释Arguments/Params、Shape类型与尺寸值、global/shared/MMA布局及Alignment元素单位、dispatch与构建期Auto。配置页和INDEX接入，公共分块/流水机制复用。
- B坐标与同一存储地址示例标教学展开；3.0 builder支持声明和2.x/3.x特化按版本保留，核读范围集中在来源说明，不堆重复提醒；没有新增算式断言或运行测试。
- READING/INDEX/SELF/计划同步：191条有依据、38条队列，本轮基线88已处理50条；第19批下一项Blackwell SM100 GEMMs。独立Astra正式review通过，无必修项；组装/调用主线、单位/类型和教学坐标均已对照原件，新增关系复核有效。结构/来源strict检查200知识页、711项登记，0错误/待核；5个受影响Wiki页1348条本地链接strict通过，diff及191/38（12论文/21文档/5代码）记录核对通过。提交范围为本项7文件，Obsidian改动保留；复用有效来源核验，没有新增运行测试。

## 2026-10-15 · Efficient GEMM ingest：输出布局、拆K与Hopper分工

- 完整读取snapshot-2026-07-02保存175行/所有伪码与3图/README，4正文/图哈希相符；层级图结构可辨，透明底部分标字按正文核对。未取原HTML/外链实现、编译或运行GPU。
- [计算模式](fundamentals/operators/gpu-kernel-computation-patterns.md)补epilogue的寄存器→shared交换→输出布局责任；[配置](implementation/kernel-configuration-and-autotuning.md)补CTA/shared与warp/register两级缓冲、splitK/slicedK协作范围、后处理时序及Hopper cooperative/ping-pong分工。已有分块/复用/普通pipeline与rasterization主体复用，不另建摘要页。
- 保留两kernel/workspace是所述splitK方案而非统一规则、逻辑分片不等于合法MMA、beta/bias只计一次及非线性反例为整理推导；按M拆与按K拆、不同tile交替与单tile双缓冲分别说明。Hopper结构不扩SM100，persistent不承诺host launch减少；旧CPU检查与本次核读区分。
- READING/INDEX/SELF/计划同步：190条有依据、39条队列，本轮基线88已处理49条；第19批按基础依赖先完成本项，下一项CUTLASS 3.0 GEMM API。独立Astra review闭环通过，无剩余必修；公式转义/控制字符、后处理概括与原文分片示例归属已修正并复读。按用户要求删去新增正文的重复防御性提醒，保留具体条件并复用有效检查。结构/来源strict检查199知识页、710项登记，0错误/待核；5个受影响Wiki页1355条本地链接strict通过，diff及190/39（12论文/22文档/5代码）计数通过；分片求和、beta重复计入与ReLU教学算术核对通过，七文件无非预期控制字符。提交范围为本项7文件，Obsidian改动保留；未执行上游、编译或设备任务。

## 2026-10-15 · cudacodes ingest：行Softmax统计、归约与输出

- 核固定9a459b1的所选softmax README/Makefile、6kernel/6声明header/helper、bench/test/PyTorch脚本、两log/plot等softmax内所选21文本/4图及仓库根README；原archive哈希及78普通文件source字节/执行位匹配，未将档案核对当语义全库，未执行二进制/编译/设备。
- 新增[CUDA行Softmax实现](implementation/cuda-softmax-row-reductions.md)，沿默认入口→0–4分工→局部统计/共同基准→两级归约/缓冲复用→再次读输入输出和host消费。复用公共Softmax数学、Warp成员和测量页，计算模式/INDEX连接主线，源码v1独立resd与README/文章原地片段分开。
- 保留默认0/1单CTA与2–4每行CTA的多因素差异、1024/full-warp/shared条件、caller发布/复用屏障、float4行首对齐与scalar tail、混合expf/__expf；v5行映射/归约/返回等静态缺口不当完成实现。test无数值参考、单次CUDA event与PyTorch墙钟、日志缺N/运行身份和画图另组数据均保留，不报告GPU或模型收益。
- READING/INDEX/SELF/计划同步：189条有依据、40条队列，本轮基线88已处理48条；第18批九项选定来源按实际范围完成，接第19批CUTLASS/GEMM。独立Astra内容review无机制、条件或知识关系必修；核读数量按其建议澄清为softmax内所选21文本/4图及仓库根README，由主代理对照实际文件清单核实。结构/来源strict检查199知识页、708项登记，0错误/待核；5个受影响Wiki页1345条本地链接strict通过，diff及189/40（12论文/23文档/5代码）计数通过；launch/shared字节/行首对齐与v5行索引静态算术核对通过。提交范围为本项7文件，Obsidian改动保留；未运行上游或设备。

## 2026-10-15 · Softmax worklog ingest：共同基准与计时范围

- 完整读取indexed-text-snapshot-2026-09-24保存505行、README与对应fixed仓库3图，文章正文哈希匹配；原HTML在采集时404，未声称网页完整恢复，仓库代码整项留后续。
- [计算模式](fundamentals/operators/gpu-kernel-computation-patterns.md)补online状态/两遍、并行局部norm共同基准和两元素反例；连接既有FA合并与Warp参与解释。[测量](implementation/kernel-correctness-and-benchmarking.md)补报告读数/算术和不同计时范围；复用公共归约/同步机制，不另建文章摘要页。
- 保留Kernel2原地与naive异地、shared树2幂/1024容量、warp4教学/真实成员及原文shuffle/锁步差异；shared/shuffle片段未展示的合并标整理推导，报告时间/型号/范围不升级为GPU或端到端结果，历史CPU检查归属区分。
- READING/INDEX/SELF/计划同步：188条有依据、41条队列，本轮基线88已处理47条；第18批下一项cudacodes固定代码。独立Astra正式review通过，无剩余必修；历史CPU范围及本次静态/算术归属已修正复读，初始化已补清。结构/来源strict检查198知识页、707项登记，0错误/待核；5个受影响Wiki页1348条本地链接strict通过，旧标题回链定向检查无命中，diff及188/41（12论文/23文档/6代码）计数通过；作者时间比值/降幅与两元素反例独立算术核对通过。提交范围为本项7文件，Obsidian改动保留；未运行上游/GPU。

## 2026-10-15 · Warp primitives ingest：成员、依赖与分组预留

- 核读snapshot-2026-07-02完整保存273行、所有可辨Listing与3图/README，4正文/图现算哈希相符；正文与2026-09-21图片补采分开，未取原HTML/完整API/外链程序或执行原文。
- [GPU硬件](fundamentals/hardware/gpu-execution-and-memory-hierarchy.md)补mask成员/activemask、寄存器交换与内存依赖、读写分相和Volta/legacy边界；[计算模式](fundamentals/operators/gpu-kernel-computation-patterns.md)补20-lane缺失源反例、零贡献/完整参与条件与同址group的预留/rank，互相连接，复用通用归约和block同步解释。
- 保留Listing6单warp范围和注释表漏9重12、Listing7/8初值/范围、helper/Unicode/legacy调用省略及atomicInc称谓差异；整理者例子为静态推导，不扩成作者完整可运行kernel，不假定mask补零/压lane或同步后持续锁步，不报告GPU/性能。
- READING/INDEX/SELF/计划同步：187条有依据、42条队列，本轮基线88已处理46条；第18批下一项CUDA Softmax worklog。独立Astra正式review通过，无事实、机制或必要知识缺口必修；新增解释、两页关系与元数据已核。结构/来源strict检查198知识页、705项登记，0错误/待核；5个受影响Wiki页1338条本地链接strict通过，diff及187/42（12论文/24文档/6代码）计数通过。提交范围为本项7文件，Obsidian改动保留；没有编译、设备运行或性能验证。

## 2026-10-15 · Tensor Core ingest：WMMA片段与历史入口

- 核读snapshot-2026-07-02完整保存334行、全部保存代码及5图/README，6正文/图现算哈希相符；正文与2026-09-21图片补采分开，未取原HTML/外链完整程序/API或编译/运行。
- 在[Tensor Core与量化GEMM](implementation/tensor-core-quantized-gemm.md)补WMMA不透明fragment、列主序地址与K循环、同模板统一epilogue，连接既有显式PTX lane表和编译检查；保存CUDA9/V100硬件/warp粒度、旧库入口与图表证据条件，不复写通用GEMM机制。
- 保留host省略/共同参与、起点guard的整块局限与M17静态反例、beta0仍读C、A行/B列文字和实际代码差异及残缺cuBLAS；旧cuBLAS倍数/cuDNN通道条件不扩现代API，峰值/跨代GEMM和卷积层几何均值不代端到端或质量验证。
- READING/INDEX/SELF/计划同步：186条有依据、43条队列，本轮基线88已处理45条；第18批下一项Using CUDA Warp-Level Primitives。独立Astra review通过，无剩余必修；READING下一项/计划日期、旧CPU检查归属和网页获取时间已修正并复读关闭。结构/来源strict检查198知识页、703项登记，0错误/待核；4个受影响Wiki页1321条本地链接strict通过，diff及186/43（12论文/25文档/6代码）计数检查通过。提交范围为本项6文件，Obsidian改动保留；未编译或运行设备。

## 2026-10-14 · Shared Memory ingest：反转快照与动态资源

- 核读snapshot-2026-07-02完整保存137行/所有保存示例及README，正文现算哈希相符；没有独立保存配图，未取原HTML/作者GitHub完整程序或编译/运行。
- 在[GPU基础](fundamentals/hardware/gpu-execution-and-memory-hierarchy.md)补n64单block反转的shared旧输入快照和一次屏障，解释其与循环缓冲复用的阶段差别；[配置资源](implementation/kernel-configuration-and-autotuning.md)补静态区与launch第三参数动态区、每block合计资源及多数组分区对齐。既有bank/warp/共享同步主线复用，原地依赖与对齐推广标为整理者说明。
- 保留CC≤1.1的顺序访存动机与≥1.2逆序store反例，旧16/32bank、4/8-byte和64KB划分不作现代规格；n64/未给guard与多block映射、空include/遗漏函数与extern文本差异保留，未称完整可执行代码或固定100×/必然加速。
- READING/INDEX/SELF/计划同步：185条有依据、44条队列，本轮基线88已处理44条；第18批下一项Programming Tensor Cores in CUDA 9。独立Astra正式review通过，无剩余必修；旧bank/宽度/shared划分的CC条件逐项明确并复读关闭，最终记录已核。结构/来源strict检查198知识页、702项登记，0错误/待核；5个受影响Wiki页1326条本地链接strict通过，diff及185/44计数检查通过。提交范围为本项7文件，Obsidian改动保留；未编译、运行模型/设备或测性能。

## 2026-10-14 · CUDA Refresher ingest：硬件版本与二维地址

- 核读snapshot-2026-07-02保存全文103行与4图/图注，5正文/图片现算哈希相符；正文与2026-09-21补采图区分，未取原HTML/外链deviceQuery或完整作者示例。
- [GPU基础](fundamentals/hardware/gpu-execution-and-memory-hierarchy.md)补CC硬件功能版本与Toolkit/API条件的职责；[布局接口](fundamentals/operators/tensor-layout-and-kernel-contracts.md)补MatAdd的C行优先地址/每16-lane x段步长，明确warp跨段、guard与省略host。host/device、线程和存储主体已充分覆盖，不重复重写，地址观察标为整理者推导。
- 不将二维launch当合并访存证明、图3当线程永久绑定core、图4的A10040GB数字当全部型号或实测；原三次拷贝流程不覆盖驻留数据/UM/mapped路径，沿既有边界保留，不添加无增量规格表。
- READING/INDEX/SELF/计划同步：184条有依据、45条队列，本轮基线88已处理43条；第18批下一项Using Shared Memory。独立Astra正式review通过，无剩余必修，新增解释与最终LOG已复读闭环；结构/来源strict检查198知识页、700项登记，0错误/待核；5个受影响Wiki页1323条本地链接strict通过，diff及184/45计数检查通过。提交范围为本项7文件，Obsidian改动保留；未编译、运行模型/设备或测性能。

## 2026-10-14 · Writing Tile Kernels ingest：坐标、归约与提示条件

- 核读snapshot-2026-07-02保存正文1210行/§2.4.1–2.4.12与所有保存示例、图19回查；正文与补采图片现算哈希相符，未取得原HTML、完整API/后端或编译/运行设备。
- 在[布局与接口](fundamentals/operators/tensor-layout-and-kernel-contracts.md)补入口/结构特化、tile-space与element坐标和两语言不同尾块规则；[计算模式](fundamentals/operators/gpu-kernel-computation-patterns.md)补block条件/逐元素选择、归约输出轴、tile_sum逐位置部分和、K累计及逐元素atomic；[配置](implementation/kernel-configuration-and-autotuning.md)补hint与别名/对齐承诺。复用通用归约、GEMM与资源解释，不新建API摘要页。
- 保留ct.load默认UNDETERMINED与gather零填充差异、C++unmasked和masked、两语tile-space全越界共同限制；保留13.3/模块名文本、Python block_sum坍缩、FP32 accumulator/乘法区别、atomic初值与完成依赖、hint不证TMA/资源/性能和切片对齐条件。tile_sum下标明确为整理者展开。
- READING/INDEX/SELF/计划同步：183条有依据、46条队列，本轮基线88已处理42条；第18批下一项CUDA Refresher。独立Astra正式review通过，无剩余必修；两处衔接/边界及最终LOG已复读闭环；旧后续指代和A/B加载与C写回边界按建议补清。结构/来源strict检查198知识页、698项登记，0错误/待核；6个受影响Wiki页1326条本地链接strict通过，diff及183/46计数检查通过。提交范围为本项8文件，Obsidian改动保留；未编译、运行模型/设备或测性能。

## 2026-10-14 · CUDA编程模型 ingest：Tile、cluster与内存范围

- 核读snapshot-2026-07-02保存正文166行/§1.2.1–1.2.3及脚注，README与9正文/图片现算哈希相符；配图/图注回查，图3–6/8–9结构可辨，图2/7透明底部分文字由正文核对；正文快照与2026-09-21图片补采分开登记，未重新取得HTML或整部PDF。
- 在[GPU执行](fundamentals/hardware/gpu-execution-and-memory-hierarchy.md)补CC≥9.0可选cluster、同GPC/同组协作范围与DSM，区分UVA、Unified Memory及mapped访问/局部性；在[布局与接口](fundamentals/operators/tensor-layout-and-kernel-contracts.md)补SIMT/Tile职责、block执行与tile数据、array/不可变tile及编译shape，复用普通warp/访存/同步，不复写已有机制。
- 不把cluster当全grid屏障、统一地址当物理DRAM合并、tile元素数当线程数，或图示补零当全部API默认；CUDA Tile与Triton规则分别保留。高级启动/迁移API、系统类别、编译器布局和设备性能未核。
- READING/INDEX/SELF/计划同步：182条有依据、47条队列，本轮基线88已处理41条；第18批下一项Writing Tile Kernels。独立Astra正式review通过，无剩余必修；GPC定义与最终LOG已复读闭环；GPC名称/SM分组按可选建议补清。结构/来源strict检查198知识页、695项登记，0错误/待核；5个受影响Wiki页1309条本地链接strict通过，diff及182/47计数检查通过。提交范围为本项7文件，Obsidian改动保留；未编译、运行模型/设备或测性能。

## 2026-10-14 · CUDA指南主页研读：目录范围无需新增

- 核对CUDA Programming Guide主页snapshot-2026-07-02，完整保存正文49行及README版本/哈希，当前正文SHA相符；只含五部分组织、读者与章节用途，未取得原HTML或链接指向整部PDF。
- 现有[GPU执行与存储基础](fundamentals/hardware/gpu-execution-and-memory-hierarchy.md)已有知识路径，主页没有独立机制增量；确认无需正文新增，不把目录材料登记成整部指南研读或复制组织说明填充知识页。Programming Model、Tile Kernels等独立来源后续分别核读。
- READING/SELF/计划同步：181条有依据、48条队列，本轮基线88已处理40条；第18批下一项Programming Model。独立Astra正式review通过，无必修，目录范围与最终LOG已复读闭环；复用上一项有效结构/来源检查（知识正文与来源字段无修改），本项2个受影响Wiki页849条本地链接strict、diff及181/48计数检查通过。提交范围为本项4管理文件，Obsidian改动保留；未编译、运行设备或核完整指南/后端。

## 2026-10-14 · RMSNorm ingest：统计、尺度与部分归一化

- 核读v1主文主线、§3–5公式/表1、§6代表实验及附录A设置，PDF3–9关键式和表4/8/10回查；PDF14页/eprint哈希相符，17源成员可读、4保留展开文本字节对应，原图仍在eprint，未编译或运行作者实现。
- 新建[归一化统计与尺度](fundamentals/operators/layernorm-rmsnorm-and-partial-statistics.md)，补中心化/二阶矩与gain/bias、正共同尺度和Jacobian/梯度条件、ε的代数影响、pRMS固定前缀与代表性/非零/旋转边界；原式与整理者等价展开区分。Transformer、实现与旋转页只加用途连接，复用线程/归约/舍入和已有旋转接口。
- 保留V100 Transformer训练步耗时/翻译协议、OE pRMS的63.9%范围、框架切片开销及CIFAR质量反例与12.5%设置；未将6.25%当通用阈值，未把无ε性质套带ε实现、模型训练耗时当现代kernel普遍加速。
- READING/INDEX/SELF/计划同步：180条有依据、49条队列，本轮基线88已处理39条；第17批五项选定来源范围完成，运行/引用实现等未核范围保留；下一项第18批CUDA指南目录及Programming Model。独立Astra正式review通过，无剩余必修；完整正文、用途入口与最终LOG已复读闭环；结构/来源strict检查198知识页、693项登记，0错误/待核；7个受影响Wiki页1350条本地链接strict通过，diff与计数/数学分隔/EOF检查通过。提交范围为本项9文件，Obsidian改动保留；未运行作者代码、模型、训练或设备性能实验。

## 2026-10-14 · MQA ingest：decode流量与摊销条件

- 核读v1完整主文、einsum伪码、访问/操作分析与WMT/LM1B实验，表1–3/关键PDF4–8回查；PDF9页/eprint哈希相符，3源成员与展开文件逐字节对应，未运行或编译作者实现。
- 在[既有Transformer基础](fundamentals/model/transformer-autoregressive-inference.md)补n次incremental的长度/head假设、元素访问与算术量，区分batch摊销权重和共享K/V缩小各请求历史；保留query/输出heads与整体计算，roofline连接指理想时间下界，不把元素/操作直接当设备Byte/FLOP。
- 保留TPUv2八核、batch1024及固定padding的摊销/整批时间；WMT三attention替换、FFN配平与GQA转换分开，dev/test/beam/sacrebleu及lnPPL、LM1B词级PPL口径保留。M/x、o/O、MQA追加轴及尺寸文字问题按原件解释，完整配置仍未补。
- READING/INDEX/SELF/计划同步：179条有依据、50条队列，本轮基线88已处理38条；下一项RMSNorm。独立Astra正式review通过，无剩余必修；流量/roofline、实验配平与计时/质量口径已原件复读。结构/来源strict检查197知识页、692项登记，0错误/待核；4个受影响Wiki页1305条本地链接strict通过，diff与计数/数学分隔/EOF检查通过。提交范围为本项6文件，Obsidian改动保留；未运行作者代码、模型、训练或设备性能实验。

## 2026-10-14 · GQA ingest：checkpoint转换与证据条件

- 核读v3完整主文与附录A，图1–6/表1及PDF1–4对应回查；PDF7页/eprint哈希相符，15源成员可读、12保留展开文件字节对应，未编译或运行作者实现。
- 在[既有Transformer基础](fundamentals/model/transformer-autoregressive-inference.md)补组内K/V mean-pooling与Q保留、非函数恒等转换及继续预训练，α为步数比例；复用已覆盖head映射、KV容量与执行入口，不另建摘要/形状页。
- 保留T5.1.1仅decoder self/cross转换、Adafactor与原配方、mean/first/random和训练步数消融；区分混合任务dev分数、任务长度与微调、逐模型并行/batch的TPUv4 per-sample/per-chip时间。保留CNN非相等、三任务消融、非严格单调与MQA稳定性反例，图3毫秒/表1秒差异按表记录，未证XXL从头/decoder-only恢复。
- READING/INDEX/SELF/计划同步：178条有依据、51条队列，本轮基线88已处理37条；下一项MQA。独立Astra正式review通过，无剩余必修；转换/α、dev与单位/计时、消融/稳定性条件已复读闭环，结构/来源strict检查197知识页、691项登记，0错误/待核；4个受影响Wiki页1301条本地链接strict通过，diff与计数/数学分隔/EOF检查通过。提交范围为本项6文件，Obsidian改动保留；未运行作者代码、模型、训练或设备性能实验。

## 2026-10-14 · Formal Algorithms ingest：记号与训练接口

- 核读v1 §3–8、全部15算法与附录B记号，PDF5–7/11–15关键式/架构/训练生成回查；PDF16页与eprint哈希相符，3成员可读、2保留展开文本字节对应，未编译或核引用模型上游实现。
- 在[既有Transformer基础](fundamentals/model/transformer-autoregressive-inference.md)补token列/行转换和softmax key轴、因果上/下三角由轴决定、next-token与输入mask重构的目标/归约；具体BERT/GPT2 norm/GELU选择与Llama分开，复用共同attention、KV和归一化解释，不建摘要页。
- 说明算法14/15全前缀重算及采样/返回/停止差别，温度等价标为整理者展开；保留PE频率/pair差异、无epsilon、i/l及d_f投影维度问题，不把形式伪码当可运行缓存接口或性能结果。
- READING/INDEX/SELF/计划同步：177条有依据、52条队列，本轮基线88已处理36条；下一项GQA。独立Astra正式review通过，队列论文标题16→15已关闭，正文/原件与计数复读无剩余必修，结构/来源strict检查197知识页、690项登记，0错误/待核；4个受影响Wiki页1298条本地链接strict通过，diff检查通过。提交范围为本项6文件，Obsidian改动保留；未运行作者代码、模型、训练或设备性能实验。

## 2026-10-14 · Transformer原论文 ingest：结构与位置范围

- 核读v7主文主干与必要实验，精读§3–5/§6.1–6.2和有效ms.tex入口，图1–2、表1–3/PE与关键PDF回查；PDF15页及源包hash相符，23成员可读、11保留展开文件逐字节对应，未读全部可视化heads或编译/运行作者实现。
- 在[既有Transformer基础](fundamentals/model/transformer-autoregressive-inference.md)补encoder–decoder三attention、cross Q/K/V与不等长、目标右移、post-LN/dropout/ReLU及base参数；相加sinusoid/共享embedding与QK RoPE分开，复用共同attention/mask/KV基础，不建摘要页或重复FA理论。
- 保留原缩放的独立零均值单位方差假设、层内O1顺序与自由生成区别，PE/heads的dev消融与未证外推、label smoothing指标反例；test checkpoint平均/beam/长度条件、EN-FR表/文41.8/41.0和FLOPs估算分别说明。原件身份、整理者形状/一步方差展开与运行证据分开。
- READING/INDEX/SELF/计划同步：176条有依据、53条队列，本轮基线88已处理35条；下一项Formal Algorithms。独立Astra正式review通过，无必修；base8头、共享embedding及heads消融条件已复读确认。结构/来源strict检查197知识页、689项登记，0错误/待核；4个受影响Wiki页1294条本地链接strict通过，diff检查通过。提交范围为本项6文件，Obsidian改动保留；未运行作者代码、模型、训练或设备性能实验。

## 2026-10-14 · Qwen3-VL仓库 ingest：媒体与微调样本边界

- 核读工具、必要HF/vLLM/web与视频notebook、微调数据/位置/collator/attention/参数/保存与pack和三份配方；root ZIP SHA/CRC及154展开文件逐字节对应。源码保存全核不表示所有评测/cookbooks/网络已读，匹配4.57外部模型未取得。
- 新建[输入准备与微调样本](implementation/qwen-vl-input-preparation-and-finetuning.md)，沿媒体→模板/processor→回答标签/三轴→累计边界→训练参数与保存解释；VLM、旧版VL实现、LoRA及INDEX补用途入口，不用旧4.53.2转填DeepStack/MRoPE内部。
- 保留patch16/对齐32与默认14、文件/列表采样metadata和裁剪参照、显式resize预算；硬ID/start+2/end+2标签、video grid拆分/局部t0、普通截断完整媒体与packed不截断、int32累计边界/FA条件、未cast dtype及未传选项、tokenizer上限未同步、merger/head/LoRA/dropout/LR/保存rank条件。调用、原件、静态算例和运行证据分别登记。
- READING/SELF/计划同步：175条有依据、54条队列，本轮基线88已处理34条；第16批七来源按选定范围收尾，匹配内部条件项保留，下一项第17批Attention Is All You Need。独立Astra正式review通过，无必修；采帧合法下限、标签切片、局部与最终位置、累计边界及参数/保存条件已原件回查。结构/来源strict检查197知识页、688项登记，0错误/待核；7个受影响Wiki页1310条本地链接strict通过，diff与新页EOF检查通过。提交范围为本项9文件，Obsidian改动保留；未运行模型、训练、FA设备、量化、GPU或性能实验。

## 2026-10-14 · Qwen2.5-Omni仓库 ingest：媒体准备与低显存返回

- 核读固定官方仓库的媒体工具、Web demo与四cookbook代表调用、两个低显存适配demo及本地模型的总生成/Token2Wav分块/Euler/dtype/卸载，README只取使用/量化/服务条件；ZIP SHA/CRC与全部24展开文件逐字节对应，不称全网络或训练审计。
- 新建[媒体准备与低显存调用](implementation/qwen25-omni-multimodal-input-and-audio-output.md)，沿模板/媒体→外部processor→返回消费→低显存状态与块裁切解释；基础、独立transformers执行与TTS页补用途入口，INDEX同步，不重复模型主链与报告成绩。
- 保留数组16kHz前提/文件重采样、采样帧数分支与FPS kwargs、两个低显存demo未显式传processor音轨flag；二元/三元返回、48code/24左12右与R mel/vocoder20mel/240样本、内部yield与一次完整wave、预分配与状态驻留分别说明。Thinker低比特、Token2Wav half/Euler及卸载共同改变，README WER/显存不证明纯位宽效应；4.52.3/0.0.4/保存0.0.8与另存fork分别登记，发布实体与运行未核。
- READING/SELF/计划同步：174条有依据、55条队列，本轮基线88已处理33条；下一项官方Qwen3-VL媒体与微调路径。独立Astra正式review通过；WER改按原表数值引用，未补猜表头单位，已复读关闭。结构/来源strict检查196知识页、687项登记，0错误/待核；7个受影响Wiki页1287条本地链接strict通过，diff与新页EOF检查通过。提交范围为本项9文件，Obsidian改动保留；未运行模型、语音、量化、GPU或性能实验。

## 2026-10-14 · transformers Qwen实现：媒体输入、缓存与生成状态

- 定向静态核读归档hf-4.53.2的Qwen2.5-VL/Omni/Qwen3稠密与MoE主线，processor/config、媒体编码与注入、位置、缓存、生成包装及必要loss/适配；zip SHA/CRC通过，4775成员身份与32选中文件字节对应，未称全框架源码审计。
- 新建[VL输入与位置](implementation/qwen25-vl-visual-input-and-position-execution.md)、[Omni模型生成](implementation/qwen25-omni-model-generation-execution.md)、[Qwen3缓存与路由](implementation/transformers-qwen3-cache-and-moe-execution.md)，沿输入对象/状态/输出解释；四个基础页与INDEX只补用途入口，不复制报告训练/质量主文。
- 保留VL每时间网格full、dense mask成本与0.5秒先转整数的静态反例、实例delta和embedding绕过注入；Omni有效mel/mask、音画与codec单位、串行完整返回、独立Talker cache、DiT层级窗口/首head RoPE、FP32/SDPA与10时间点9区间×4RK4调用；Qwen3逐head norm不受bias开关、cache更新与输出开关、top-k配置/专家预算、loss触发与混合稠密层None项前提分别说明。发布checkpoint/template/speaker、Qwen3-VL内部与模型/训练/语音/量化/GPU运行未核。
- READING/SELF/计划同步：173条有依据、56条队列，本轮基线88已处理32条；一份来源分三个执行主线，下一项官方Qwen2.5-Omni仓库。独立Astra正式review通过；视频帧数/FPS条件、混合稠密层aux汇总前提及DiT引用定位已修正并复读。结构/来源strict检查195知识页、686项登记，0错误/待核；10个受影响Wiki页1339条本地链接strict通过，diff检查通过。提交范围为本项12文件，Obsidian改动保留；未运行模型、训练、语音、量化、GPU或性能实验。

## 2026-10-14 · Qwen3-VL ingest：多层视觉输入与时间表达

- 核读v2主编译入口、架构与必要pre/posttrain/数据、消融和needle、视频协议与定向附录提示；图1/3、表1/11/12与PDF1/3/4/24/25回查。PDF42页/源包hash匹配，30成员扣00README为29来源，20展开文件逐字节对应；未扩为全部任务表或实现审计。
- 完善[视觉token基础](fundamentals/model/vision-language-model-tokens-and-quantization.md)，补SigLIP-2/多层merger→前三LLM层残差、序列长度不变与额外成本，分开轴频率分配、文字timestamp和2D/3D输出坐标；SPHQuant补真实校准入口，INDEX说明同步，不新建模型摘要。
- 保留旧版14patch/32层/窗口/时间公式不转填，DeepStack结构式/频率角仅示意，internal15B-A2B/200B/no-posttrain消融与TVQA退化，CLIP/VLM阶段、[0,1000]新坐标、per-token平方根归约缺式、text-only蒸馏/SAPO未给更新式、工具奖励反例、视频API帧预算不齐和needle恒定预算文字/横轴差异。匹配内部实现/量化与运行未核。
- READING/SELF/计划同步：172条有依据、57条队列，本轮基线88已处理31条；第16批四报告范围接续，下一项已归档transformers模型实现。独立Astra正式review通过；新增SPHQuant锚点已按真实标题修正并复读，无剩余必修。结构/来源strict检查192知识页、683项登记，0错误/待核；5个受影响Wiki页1272条本地链接strict通过，diff检查通过。提交范围为本项7文件，Obsidian改动保留；未运行模型、processor、训练、量化、GPU或性能实验。

## 2026-10-14 · Qwen3 ingest：模式预算与两阶段蒸馏

- 核读v1实际编译入口/摘要、Intro/Arch/Conclusion，必要pretrain主链、posttrain四阶段/模板/蒸馏/协议与Discussion、附录A.1.1及必要语言范围；原图1/2、关键表1/2/9/21/22/23及PDF回查。PDF35页/源包hash匹配，29来源成员中15展开文件逐字节对应；未编入experiments残稿排除，未称全部模型/语言表审计。
- 在既有模型/MoE/后训练/质量四页补职责内增量，不建模型摘要页；后训练标题扩大为模式控制与蒸馏，章节与LLM-QAT锚点/INDEX同步。区分Qwen3继续Fusion SFT与R1重新Base、同一权重模式/思考输出预算、教师回复与学生前缀soft KL、无shared的128选8及常驻/批次预算。
- 保留head_dim/发布配置、gate和完整norm/KL/模板未核，32K训练与128K外推、采样配方/总输出差异、8B同off-policy checkpoint接续的1/10 GPUh、阶段退化/ThinkFollow非100%、RULER更多思考退化及语言子集/agent协议。匹配实现、量化、模型/训练/设备性能未运行。
- READING/SELF/计划同步：171条有依据、58条队列，实际表行已核，本轮基线88已处理30条；下一项Qwen3-VL，准备不计完成。独立Astra正式review通过；表22阶段数字的thinking条件、空block与预算指代已补明并复读，图/表原件已核。结构/来源strict检查192知识页、682项登记，0错误/待核；8个受影响Wiki页1310条本地链接strict通过，diff检查通过。提交范围为本项10文件，Obsidian改动保留；未运行模型、trainer、训练、量化、GPU或性能实验。

## 2026-10-14 · Qwen2.5-Omni ingest：音画对齐与语音生成

- 核读v1编译入口/摘要、Intro/Architecture/Pretraining/Posttraining/Conclusion及必要输入/输出评测；图2–4与关键PDF、表1/4/7–10回查。PDF20页/源包hash匹配，28来源成员中16展开文件逐字节对应；未编入的Qwen2 appendix残稿排除，未扩成20页全文逐项审计。
- 新建[多模态感知与语音生成](fundamentals/model/multimodal-perception-and-streaming-speech-generation.md)，沿编码表示/位置→Thinker→隐藏状态与采样文字→Talker code→DiT mel→BigVGAN波形解释；视觉token/TTS/INDEX补用途入口，公共token/质量解释复用。
- 保留输入40ms/音画2秒与输出codec块不同、时间/空间/序列位置分开、隐语义与实际发音依赖、context不证同一物理cache、两回看一前看及缺首包/RTF实测；训练阶段/参数与DPO目标、TTS评测范围/非等价模态比较分别说明。表9/10同标RL中文WER1.42/1.30差异保留；匹配实现、低比特、训练/语音与设备未运行。
- READING/SELF/计划同步：170条有依据、59条队列，实际表行已核，本轮基线88已处理29条；下一项Qwen3，准备不计完成。独立Astra正式review通过；DPO改写、音画/codec窗口与WER百分数及两表差异已原件复核。结构/来源strict检查192知识页、678项登记，0错误/待核；6个受影响Wiki页1272条本地链接strict通过，diff检查通过。提交范围为本项8文件，Obsidian改动保留；未运行模型、训练、语音、量化、GPU或性能实验。

## 2026-10-14 · Qwen2.5-VL ingest：动态长度与时间坐标

- 核读v1完整Approach架构/数据/训练主文、Introduction、video正文/结果表及必要grounding/OCR条件；图1原件与PDF1/3/4/5/14的版本、表1/8回查。PDF23页/源包hash匹配，24来源成员中13展开文件逐字节对应；未扩成全部任务/数据表或实现审计。
- 完善[视觉token基础](fundamentals/model/vision-language-model-tokens-and-quantization.md)，沿resize→14patch→窗口/full编码→四邻拼接/MLP→语言输入解释动态长度，区分视觉2D-RoPE和语言MRoPE、帧数和时间ID。INDEX替换泛泛入口说明，不另建模型摘要页。
- 保留28倍数调整/补齐未核、四full层的二次项、图像/视频结构计数仅教学、坐标回映/时间取整未核、50/50样本与tokens不同、ViT冻结不替代量化校准，以及768帧/24576video tokens/字幕条件；merger减位置不是降位宽。匹配processor/完整窗口/模型/量化/性能未运行。
- READING/SELF/计划同步：169条有依据、60条队列，实际表行已核，本轮基线88已处理28条；下一项Qwen2.5-Omni，准备不计完成。独立Astra正式review通过；FPS改变的相邻间隔与相同实际时间的编码尺度已消歧并复读。结构/来源strict检查191知识页、677项登记，0错误/待核；4个受影响Wiki页1253条本地链接strict通过，diff检查通过。提交范围为本项6文件，Obsidian改动保留；未运行模型、processor、量化、GPU或性能实验。

## 2026-10-14 · R1报告 ingest与第15批收尾

- 核读v2正文GRPO/奖励/阶段与必要Zero结果、附录PPO比较/infra/数据/超参/蒸馏及必要评测协议/完整表；六张源包原图与关键PDF页回查。PDF86页/源包hash匹配，47 regular成员扣编译管理00README为46来源，18展开文件逐字节对应；未称86页全文或全部安全/样例/人评审计。
- 新建[推理后训练](fundamentals/model/reasoning-post-training-and-distillation.md)，补actor/old/reference、组内反馈、KL样本与期望/教学反例、rule reward、Dev2数据/第二SFT重新V3-Base、输出监督/学生起点；质量页只补抽样pass@1/pass@K/cons@K预算与R1例，Transformer/LLM-QAT/INDEX补用途入口。
- 保留ε10只明属于R1首RL、zero-std/mask与实现未核、accuracy/format与语言/general奖励范围、800K概数/804745实表、首RL数据与最终teacher泛称、Zero8.2K预算干预/图cons16和最终R1cons64、阶段非单调及蒸馏非等预算RL对照。匹配trainer/rollout、reward/parser、dtype/打包及训练/评测未核；KL反例仅数学核算。
- READING/SELF/计划同步：168条有依据、61条队列，实际表行已核，本轮基线88已处理27条；第15批Llama/V3/R1按选定来源范围完成，下一项第16批Qwen2.5-VL，准备不计完成。独立Astra正式review通过；问题分布/条件采样公式已修正并复读，600K筛选的V3判定与首RL规则奖励已消歧。结构/来源strict检查191知识页、676项登记，0错误/待核；7个受影响Wiki页1281条本地链接strict通过，diff检查通过。提交范围为本项9文件，Obsidian改动保留；未运行trainer、模型、训练、GPU或评测实验。

## 2026-10-14 · DeepSeek-V3 ingest：FP8训练、路由与MTP

- 核读v2架构/MTP及训练/推理基础设施主链、完整FP8节116行、必要超参/消融/推测和附录B；PDF53页/源包hash匹配，24成员中8展开源码逐字节核对，核心六图与PDF47原件目视。未扩成全部数据/后训练/质量表。
- 新建[FP8训练](implementation/deepseek-v3-fp8-training.md)，解释三GEMM、online组尺度/局部promotion、反向缓存/optim与通信及DualPipe条件；MoE基础并列V3 sigmoid/biased选择、wholebatch更新/小seqaux/node条件，保留V2；模型基础补MTP因果链/目标/可丢模块/有条件的推测证据。FP8/DeepGEMM/consumer/INDEX补入口，consumer标题改为具体所选配置。
- 保留每K128尺度不可移出、H800累加报告与理论算力、E5M6/2幂缓存、FP32master/grad与BF16moments、FP8dispatch/BF16combine、双向流水/拓扑和反例条件。16B/230B相对loss<0.25%不扩成671B完整BF16质量对照；node评分/seqaux记号未明及MTP非全任务改善按原件保留。匹配HAI-LLM/PTX、packed/round和完整训练/设备/推测运行未核。
- READING/SELF/计划同步：167条有依据、62条队列，实际表行已核，本轮基线88已处理26条，下一项DeepSeek-R1，准备不计完成。独立Astra正式review通过；MTP逐位置简写已恢复为投影加序列因果Transformer并复读，表2/§5.4.3引用及变量/梯度缓冲消歧已关闭，V3配置句另复核。结构/来源strict检查190知识页、674项登记，0错误/待核；9个受影响Wiki页1283条本地链接strict通过，diff检查通过。提交范围为本项11文件，Obsidian改动保留；未运行作者代码、训练、GPU或性能实验。

## 2026-10-14 · Llama 3报告 ingest：选择性FP8与质量/资源条件

- 研读v3语言主线、预训练/后训练与语言结果、推理PP/FP8和必要图表，多模态/Guard只定向核对象与稳定性；PDF92页/源包hash匹配，157成员中82展开源码逐字节核对，余图仍保留archive内。主代理复核图25/26/27、microbatch两图及三张未编入PDF的FP8补图，不杜撰附图PDF编号。
- 四个既有页增量：[模型基础](fundamentals/model/transformer-autoregressive-inference.md)补3.1配置及16/40/63GiB逻辑KV教学计算；[FP8](fundamentals/numeric-formats/fp8-and-mx-data-formats.md)补FFN范围/首尾保护/row-wise/尺度上界1200；[质量](implementation/model-quality-evaluation.md)补十万回复reward/罕见损坏和长上下文协议/表文差异；[服务评测](implementation/serving-performance-evaluation.md)补BF16双节点/FP8单节点、microbatch和2×注记。已有格式与公共评测解释复用，不建模型摘要页。
- 保留主文实际3.1身份、BF16训练与FP8推理、尺度上界与裁剪、图证据/统计检验、训练适应与最终能力、资源计账与kernel收益的区别；匹配代码/scale/配置、reward manifest、全安全/多模态表和训练/服务运行未核。
- READING/SELF/计划同步：166条有依据、63条队列，实际表行已核，本轮基线88已处理25条，下一项DeepSeek-V3，准备不计完成。独立Astra正式review通过，九原图及PDF7/39/53/54等编号/条件已回查。结构/来源strict检查189知识页、670项登记，0错误/待核；6个受影响Wiki页850条本地链接strict通过，diff检查通过。提交范围为本项8文件，Obsidian改动保留；未运行模型、训练、量化或服务基准。

## 2026-10-14 · FSDP2教程 ingest与第14批收尾

- 完整研读Distributed目录217行、FSDP2教程411行和三张保存原图，5文件hash/大小匹配；新建[FSDP2参数生命周期](implementation/fsdp2-parameter-lifecycle-and-checkpoint.md)，补持久Shard(0)/临时group、AG/RS/optim、预取与reshard/dtype、DTensor/DCP full state和collective/单writer。TP/ModelOpt/INDEX只补用途入口，AQLM及MLA DCP版本/对象不混用。
- 保存片段torch.save未在rank0 guard内的竞争/空结果风险按静态范围记录，未称当前库运行故障；例程/API/PyTorch库实现、optimizer保存、跨world恢复及训练/通信/保存运行未核。目录链接不代表其他教程已读；3图10-05补录不证9-24字节，README图片改链/source备份说明与现稿不一致，不改raw。
- READING/SELF/计划同步：165条有依据、64条队列，实际表行已核，本轮基线88已处理24条；第14批ModelOpt/TensorRT/FSDP2按选定材料范围完成研读、承接及独立Astra正式review；下一项第15批Llama 3，准备不计完成。结构/来源strict检查189知识页、666项登记，0错误/待核；6个受影响Wiki页1226条本地链接strict通过，diff检查通过。提交范围为本项8文件，Obsidian改动保留；未运行训练、通信、checkpoint或GPU。

## 2026-10-14 · TensorRT-LLM文档 ingest：加载入口与KV条件

- 完整研读固定a6f41514…的144行量化正文及模型/硬件两表，1文件hash/大小匹配；在[部署页](implementation/quantized-llm-deployment-backends.md)补默认PyTorch直接加载、NVFP4 KV离线/FP8主体条件、活动与存储层缓存区别；[块缩放页](implementation/block-scaled-matmul-execution.md)补SM90/SM100/103的FP32/UE8M0尺度声明。旧d567f924引擎路径保留版本，不搬运算法或公共格式解释。
- 模型/硬件单列Y不推组合，矩阵未标支持不写成永久禁用；producer clone未pin，来源完整a6f与站点fca831e分开。匹配loader/导出器、字段/布局、设备kernel、compression外链和运行未核。
- READING/SELF/计划同步：164条有依据、65条队列，实际表行已核，本轮基线88已处理23条；下一项PyTorch Distributed/FSDP2。独立Astra正式review通过；按可选建议将块缩放上级标题扩大为框架，明确来源归属。结构/来源strict检查188知识页、665项登记，0错误/待核；4个受影响Wiki页820条本地链接strict通过，diff检查通过。提交范围为本项6文件，Obsidian改动保留；未运行框架、模型或GPU。

## 2026-10-14 · ModelOpt文档 ingest：校准、搜索与checkpoint消费

- 完整研读固定f2f0d695…的HF PTQ811行与Recipes556行，2文件hash/大小匹配；新增[ModelOpt PTQ](implementation/modelopt-ptq-calibration-and-checkpoint.md)，串起对象/算法、校准尺度/前向覆盖、weight/KV搜索预算和恢复状态、导出及静态FP8 consumer。校准页只补多级尺度，部署/INDEX只补入口。
- 定向核vLLM568afb配置选择/解析及静态FP8参数、加载后最大尺度重标和转置。保留input_scale1全局固定/块动态、headroom与预算对象、mixed-KV stock0.26 export-only、cast范围与回退、VLM/RealQuant/FSDP2/追踪条件。原始YAML/算法/导出实现及外链/运行未核；README相对链接修复说明与当前稿不同，不改raw。
- 独立Astra技术review通过，两处加载顺序及layerwise export归属进一步收紧；内容、来源和必要数学/格式/诊断目标已核。READING/SELF/计划同步：163条有依据、66条队列，本轮基线88已处理22条；下一项TensorRT-LLM量化文档，准备不计完成。
- 结构/来源strict检查188知识页、663项登记，0错误/待核；6个受影响Wiki页的1275条本地链接strict通过，163/66实际表行及管理记录经独立Astra闭环复核通过。提交范围为本项8文件，Obsidian改动保留；没有运行模型、校准/导出或GPU。

## 2026-10-14 · Ray Serve文档 ingest与第13批收尾

- 完整读七份保存正文26/224/514/29/209/206/263行，7文件hash/大小匹配；ASCII图与十处图注已核，原图未保存。接续[服务编排](implementation/distributed-llm-serving-orchestration.md)的模型/副本两级选择、配置/协议/启动、placement与路由策略、DP gang计数/重启及decode侧P/D和connector预热；Dynamo机制保持各自归属。
- 保留CPU server actor、engine、GPU worker与DP组单位；路由事件/估计负载与实际KV身份分开，集中get/set不证复合选择线性化；head轴TP与MLA DCP分别解释。文档后端覆盖、原图缺失、外链/API/literalinclude、匹配代码与运行边界保留，没有导入或执行上游。
- READING/SELF/计划同步：162条有依据、67条队列，本轮基线88已处理21条。第13批九项选定来源完成研读、知识承接与独立review流程，未核实现/设备分支按实际记录保留；下一项第14批ModelOpt PTQ，准备不计完成。
- 独立Astra正式review通过，无必修问题；七稿/ASCII图/图注、Ray增量和EP/MLA DCP阅读目标已核。全库结构/来源strict检查187知识页、660项登记，0错误/待核；3个受影响Wiki页的778条链接strict通过，162/67实际记录与第13批九项队列移出已核。仅提交本项5文件，其他任务的能力包工作已独立提交，本项保留Obsidian改动；未运行服务、模型或GPU。

## 2026-10-14 · Dynamo文档 ingest：路由、容量与KV交接

- 完整研读三份保存正文220/128/329行与四张原图，7文件hash/大小匹配。新建[分布式服务编排](implementation/distributed-llm-serving-orchestration.md)，补引擎外三路径、backend-specific传输元数据/时序、xPyD与滚动更新；内存/部署/INDEX只补入口，复用已有DistServe/Sarathi和评测解释。
- 版本按明确1.4.2依赖行与保存快照声明分别归属，保留v1.5.0/dev混杂、SGLang LoRA状态与分离图模型身份差异；KV可见/元数据/完成及控制目标/ready容量分别解释。README所称source原稿缺失，当前hash一致不证明转换前后；没有改raw或核匹配实现/外链/运行。
- READING/SELF/计划同步：161条有依据、68条队列，本轮基线88已处理20条；第13批下一项Ray，准备不计完成。
- 独立Astra正式review通过，SGLang LoRA归属与延迟比方向修正后已回读，无剩余必修项；三稿/四图、新页全文及必要目标已核。全库结构/来源strict检查187知识页、659项登记，0错误/待核；6个受影响Wiki页的1243条链接strict通过，161/68实际记录与KVBM图中比值已核。仅提交本项8文件，能力包及Obsidian改动保留；未运行服务或GPU。

## 2026-10-14 · vLLM KV管理文章 ingest：容量接纳与空闲队列

- 完整研读snapshot-2026-09-24文章506行及六图，7文件hash/大小与登记一致；作者自述ba221历史代码未核。沿[vLLM推理](implementation/vllm-inference-execution.md)补watermark条件/两次容量检查、CPU块元数据与空闲链表/归零分组顺序，复用已有token进度和紧凑批次解释。
- 定向对照568afb配置、waiting caller、allocate_slots、块池/队列及slot dtype/core client/cache注册；保留bool(running)、检查时空闲数变化、单块O(1)/整批O(n)与预留区别。文章图3超预算、累计KV长度502/755、异步socket图文与布局/dtype/登记时序差异不迁成当前实现结论。
- READING/SELF/计划同步：160条有依据、69条队列，本轮基线88已处理19条；第13批下一项Dynamo、随后Ray，准备不计完成。
- 独立Astra正式review通过，READING下一项状态遗漏修正后已回读闭环，无剩余必修项；完整文章/六图及所选配置、容量/队列代码已核。全库结构/来源strict检查186知识页、658项登记，0错误/待核；3个受影响Wiki页的779条链接strict通过，160/69实际记录和预算/长度/余量教学算术已核。仅提交本项5文件，Obsidian改动保留；未运行框架、模型或GPU。

## 2026-10-14 · vLLM V1架构文档 ingest：进程拓扑与初始化接口

- 完整研读官方f9ad9dd6…固定文档300行及五张原图，6文件hash/大小与登记一致；对照现有568afb同名文档，仅旧API启动段与路径有差别。沿[vLLM推理](implementation/vllm-inference-execution.md)补进程拓扑/条件计数与别名-client边界，沿[部署页](implementation/quantized-llm-deployment-backends.md)补统一配置/prefix、初始化分片和加载后处理。
- 定向核568afb的两份alias、core client、同步/异步入口、serve默认、Llama/Linear/AutoAWQ与loader相关段。保留internal/external/hybrid/multi-port区别、旧包装图和keyword-only兼容、配置共享与跨进程对象区别；理想容量例不作实际峰值。
- READING/SELF/计划同步：159条有依据、70条队列，本轮基线88已处理18条；第13批下一项KV管理源码解读文章，准备不计完成。
- 独立Astra正式review通过，无必修问题；文档/原图与所选客户端、CLI/supervisor及模型构造/加载段已对照。全库结构/来源strict检查186知识页、657项登记，0错误/待核；4个受影响Wiki页链接strict通过，TP/AWQ目标及阅读路径已核。仅提交本项6个成果文件；能力包/README和Obsidian改动保留，不混入本项；未启动进程、模型或GPU。

## 2026-10-14 · vLLM v0.26.0发布说明 ingest：配置对象与组合边界

- 完整研读固定v0.26.0、snapshot-2026-08-20发布正文，137行及SHA与登记一致；现有568afb…源码属于同一tag。仅在[Attention接口](implementation/vllm-attention-operator-design.md)、[部署配置](implementation/quantized-llm-deployment-backends.md)和[DCP边界](implementation/vllm-mla-context-parallel-execution.md)增补相应声明。
- 区分KV group后端、head与sampler精度、target/draft/层级KV dtype、加载峰值与运行内存、普通V1与MRV2/PCP/推测组合；FlashInfer依赖0.6.14与独立0.6.17分开。已有末轴K/V打包与短sparse-MLA回退可复用，缺测量条件的kernel/TPOT数字不加总或排名；未核PR/新组合实现或运行。
- READING/SELF/计划同步：158条有依据、71条队列，本轮基线88已处理17条；第13批下一项vLLM V1官方架构文档，准备不计完成。
- 独立Astra正式review通过，无必修问题；发布原文、既有AWQ/普通DCP与支持范围已对照。全库结构/来源strict检查186知识页、655项登记，0错误/待核；5个受影响Wiki页链接strict通过，AWQ目标及版本/阅读关联已核。仅提交本项7个成果文件，Obsidian改动保留；未运行框架或GPU。

## 2026-10-14 · nano-vllm ingest：离线状态与分页执行

- 完整读取固定bb823b3e…的21个Python文件、README与pyproject，归档SHA/commit及26文件字节对应通过；只做AST和独立槽位教学计算，未导入上游。新增[nano-vllm离线执行](implementation/nano-vllm-inference-execution.md)，沿已知token/KV、本轮调度、块引用/尾块、紧凑输入、浮点Qwen3、采样/完成与TP/Graph解释主线；vLLM、内存和INDEX只加阅读入口。
- 保留prefill选中即让decode等待、首次完整历史分块、最后完整块也重算、chunk采样仍消耗RNG、全局活跃数与本轮序列数的区别。代码的输入/长度/参数校验、Graph规格、rank池协商和loader完整性边界就近说明；作者benchmark与运行证据分开。
- READING/SELF/计划同步：157条有依据、72条队列，本轮基线88已处理16条；第13批下一项vLLM v0.26.0发布说明，准备不计完成。
- 独立Astra正式review通过，warmup的token ID/长度歧义修正后已独立闭环，无剩余必修问题。全库结构/来源strict检查186知识页、652项登记，0错误/待核；6个受影响Wiki页链接strict通过，状态/页表/模型/TP/评测的阅读承接已核。只提交本项8个成果文件，Obsidian改动保留；没有运行模型、服务或GPU。

## 2026-10-14 · SGLang v2 ingest：程序依赖与前缀共享

- 完整研读固定arXiv:2312.07104v2主文与附录A–D，PDF/源包哈希与登记一致，13份展开文本对应原包，15份图保留在原包。新增[结构化程序执行](implementation/structured-language-model-program-execution.md)，串起提示流/分支依赖、跨调用Radix、离线DFS、FSM/重新分词、API文本复用及IR；公共内存/评测页只补生命周期与计量关联，现代Scheduler主线保留。
- 保留最优命中率的离线/容量条件、未知生成与饥饿、原算法计数省略、图14变量标签差异、概率失真与prompt重排语义风险。性能按程序/请求/token分别计数，长输出反例与旧基线保留；没有核匹配历史实现、完整KV失效/API回退或运行服务。
- READING/SELF/计划同步：156条有依据、73条队列，本轮基线88已处理15条；第13批下一项nano-vllm，准备不计完成。
- 独立Astra正式review完成，修正图/定理/附录编号、Vicuna收益归属与API模型名后复核通过；图2同步/数据传递按原件核清。全库结构/来源strict检查185知识页、651项登记，0错误/待核；7个受影响Wiki页链接strict通过，程序→现代缓存与公共计量的阅读路径已核。已有能力包任务的独立提交保留；本项只提交9个成果文件，Obsidian改动排除。未运行程序、服务或GPU。

## 2026-10-14 · Sarathi-Serve v3 ingest：混合预算与token级TBT

- 完整研读固定arXiv:2403.02310v3正式主文与Artifact Appendix，PDF/源包哈希与登记一致；36份展开文件对应原包，25份图保留在原包。沿[内存与批处理](implementation/serving-memory-and-batching.md)接续三种调度轴、剩余预算/顺序、历史KV与气泡；在[服务评测](implementation/serving-performance-evaluation.md)补token级P99 TBT、排队条件、硬件/负载/预算与消融。
- 保留mixed轮次仍慢于pure decode、chunk预算须重profile、TTFT/TBT取舍；核关键PDF表1–4/算法3/图6–14，保留算法省略、3.5×归因与25%开销图文差异；不混用旧稿或不同基线/并行/容量定义，不与DistServe直接排名。未核作者实现或运行服务。
- READING/SELF/计划同步：155条有依据、74条队列，本轮基线88已处理14条；第13批下一项SGLang v2，准备不计完成。
- 独立Astra正式review通过，无必修问题；补清GQA/滑动窗口、Falcon部署指代和作者增补PP基线后已独立闭环。全库结构/来源strict检查184知识页、648项登记，0错误/待核；4个受影响Wiki页链接strict通过，机制/评测互链与roofline/现代调度目标已核。保留Obsidian改动，不纳入本项提交；未核作者实现或运行服务。

## 2026-10-14 · DistServe v3 ingest：阶段分离与容量口径

- 完整研读固定arXiv:2401.09670v3主文/附录A–C，PDF/源包哈希与登记一致，16份展开文件对应源包，20份图/徽章保留在原包。在[内存与批处理](implementation/serving-memory-and-batching.md)补两阶段实例/排队/TP-PP/带宽placement与KV pull；在[服务评测](implementation/serving-performance-evaluation.md)补容量goodput/GPU、联合SLO与扫描/实验条件。
- 保留M/D/1与理想流水条件、经验模拟器、全部GPU分母和LongBench截断，不合并不同任务/基线极值；回查关键PDF原件，保留OPT-66B附录配置、High/Low标签及传输CDF差异。未核作者实现/同步释放、真实传输或运行系统。
- READING/SELF/计划同步：154条有依据、75条队列，本轮基线88已处理13条；第13批下一项Sarathi-Serve，准备不计完成。
- 独立Astra正式review通过，无必修问题；关键公式/算法/图表、原件差异及阅读连接已核。全库结构/来源strict检查184知识页、646项登记，0错误/待核；4个受影响Wiki页链接strict通过。保留Obsidian改动，不纳入本项提交；未核作者实现或运行系统。

## 2026-10-14 · FlashInfer v0.6.17发布说明 ingest与第12批收尾

- 完整研读固定v0.6.17、snapshot-2026-08-20发布文本，哈希与README一致。在[块缩放](implementation/block-scaled-matmul-execution.md)、[专家路由](implementation/moe-routing-and-quantized-expert-execution.md)、[专家并行](implementation/moe-expert-parallel-execution.md)、[MLA](implementation/mla-cache-and-attention-execution.md)与[DCP](implementation/vllm-mla-context-parallel-execution.md)原有机制旁接续相关版本声明，不另建功能摘要页。
- 区分0.6.14接口与新尺度参数、浮点FP4专家与AWQ整数链、调优参数缓存与激活/JIT缓存、作业存活与贡献完整性、标准MLA与Blackwell/稀疏入口、Ulysses布局A2A与DCP状态合并。保留SM90 CUTLASS MoE撤回的具体范围，未核PR或对应实现不作运行结论。
- 第12批选定来源已完成逐项研读、知识承接与内容review流程，其他设备/分支/运行范围按READING保留；当前153条有依据、76条队列，本轮基线88已处理12条。下一项为第13批DistServe，准备工作不计入完成。
- 独立Astra内容review通过，无必修问题；发布声明与五页既有版本/机制上下文已对照。全库结构/来源strict检查184知识页、644项登记，0错误/待核；7个受影响Wiki页链接strict通过，新增锚点与承接用途已核。未安装、编译或运行GPU/分布式系统。

## 2026-10-14 · FlashInfer single prefill文档 ingest：mask与尺度

- 固定FlashInfer页面0.6.14、snapshot-2026-07-02，完整保存正文与README哈希一致。在[KV接口页](implementation/flashinfer-paged-kv-write-and-nvfp4.md)接续多query/输出、尾对齐mask与打包选择、FP8按head尺度、NVFP4线性尺度/fa2；复用decode参数和递归状态解释。
- 区分页内NVFP4 writer的V交织与single reader的两份线性尺度，不推定页数据/全局因子能直接互用；保留packed-only与causal、尺度参数叠加及签名类型差异。READING/SELF/计划同步：152条有依据、77条队列；本轮88已处理11条，第12批尚余发布说明。
- 独立Astra内容review通过，无必修问题；分页NVFP4来源与reader声明已对照。全库结构/来源strict检查184知识页、639项登记，0错误/待核；3个受影响Wiki页链接strict通过，mask尾对齐与前文/递归状态的承接已核。本项仅为固定接口阅读，未核匹配实现、实现转换或运行GPU。

## 2026-10-14 · FlashInfer single decode文档 ingest：读取与返回

- 固定FlashInfer页面0.6.14、snapshot-2026-07-02，完整保存正文与README哈希一致。在[KV接口页](implementation/flashinfer-paged-kv-write-and-nvfp4.md)接续单query/3D KV、GQA/数值参数、输出与LSE；原分页写入、索引及NVFP4范围保留，来源身份按条分行。
- 保留Returns的qo_len/二维样例差异、未声明的LSE底/dtype和输入宽度/精度/设备范围，不推成分页或NVFP4reader。INDEX/READING/SELF/计划同步：151条有依据、78条队列；本轮88已处理10条，2份FlashInfer文档待接续。
- 独立Astra内容review完成，修复来源拆分后的旧条目指代并复核通过，无剩余必修问题。全库结构/来源strict检查184知识页、638项登记，0错误/待核；4个受影响Wiki页链接strict通过，新增关系及递归状态目标上下文已核。未核匹配实现或运行GPU；已有能力包设计/说明改动留在工作区，不属于本项提交。

## 2026-10-14 · FlashInfer状态文档 ingest：递归合并与条件

- 固定FlashInfer页面0.6.14、snapshot-2026-07-02，全文及2026-09-21补采合并树图核读，两文件哈希与登记一致。在[SGLang归约页](implementation/sglang-attention-operator-design.md)复用已有输出加权公式/例子，只补完整自然LSE与递归state、同query/score/mask及互不重叠条件，保留浮点和实际接口单位边界。
- 区分共享KV与复用query相关输出；关联MLA底转换解释，保留原文30×缺完整测量/博客未读。未核merge API/设备实现，不把数学合并写成所选SGLangstage2已导出完整state。READING150条有依据、79条队列；本轮88已处理9条，3份FlashInfer文档待接续。
- 独立Astra内容review通过，无必修问题；定向回核SGLangstage2及调用链，明确它只写O、不导出合并LSE。全库结构/来源strict检查184知识页、637项登记，0错误/待核；3个受影响Wiki页链接strict通过，新增关系与MLA目标上下文已核。未执行FlashInfer/SGLang或GPU。

## 2026-10-14 · FlashInfer FP4 GEMM文档 ingest：布局与恢复

- 固定FlashInfer页面0.6.14、snapshot-2026-07-02，完整保存正文与README哈希一致。[块缩放页](implementation/block-scaled-matmul-execution.md)接续NV/MX FP4两侧数据/尺度、各backend的128×4/8×4与shuffle、TRTLLM原例的B/sf配套转置和全局恢复、auto/PDL及输出范围。
- 保留逻辑K/packed存储K和b_descale描述缺口，不用Triton格式或其他量化入口补helper行为；例子不是通用scale算法/设备能力或性能结论。READING149条有依据、80条队列；本轮88已处理8条，4份FlashInfer文档待接续。
- 独立Astra内容review完成，补回原例先FP32计算globalsf的步骤并收紧非有限输入边界，修复闭环后无必修问题。全库结构/来源strict检查184知识页、636项登记，0错误/待核；3个受影响Wiki页链接strict通过，新增入口与上下文已核。未核匹配实现、编译或运行GPU。

## 2026-10-14 · FlashInfer BF16×FP4文档 ingest：准备与消费

- 固定FlashInfer页面0.6.14、snapshot-2026-07-02，完整保存正文与README哈希一致。[块缩放页](implementation/block-scaled-matmul-execution.md)补BF16×NVFP4的加载时准备、同backend消费、alpha折叠/None与block16/输出；BF16支持和未知设备/布局条件分别保留。
- 复用已有数据/块尺度解释，未将FP4量化或Triton布局补成prepared格式；未读prepare函数页面或匹配实现，不猜折叠算法。READING148条有依据、81条队列；本轮88已处理7条，5份FlashInfer文档待接续。
- 独立Astra内容review通过，无必修问题，补PDL默认值区别。全库结构/来源strict检查184知识页、635项登记，0错误/待核；3个受影响Wiki页链接strict通过，新增入口与上下文已核。未核或运行量化/GEMM实现与GPU。

## 2026-10-13 · FlashInfer FP8 GEMM文档 ingest：NT与尺度存储

- 固定FlashInfer页面0.6.14、snapshot-2026-07-02，完整保存正文与README哈希一致。[块缩放页](implementation/block-scaled-matmul-execution.md)接续FP8 NT输入轴、FP32块尺度、MN/K存储方向、mma_sm及BF16默认输出；复用既有K归约/尺度解释，不新增摘要页。
- 保留groupwise转引未读、尺度精确shape/stride/粒度索引、尾块与设备/精度路径缺失，不把Triton microscaling、descriptor或CuTe量化条件当本接口的实现。READING147条有依据、82条队列；本轮88已处理6条，6份FlashInfer文档待接续。
- 独立Astra内容review通过，无必修问题；全库结构/来源strict检查184知识页、634项登记，0错误/待核，3个受影响Wiki页链接strict通过，新增入口与上下文已核。未核匹配实现或运行GPU。

## 2026-10-13 · FlashInfer FP4量化文档 ingest：数据与尺度产物

- 固定FlashInfer页面0.6.14、snapshot-2026-07-02，完整保存正文与README哈希一致。在[块缩放页](implementation/block-scaled-matmul-execution.md)接续packed FP4数据与sf、global scale逆解释、尺度粒度/编码/布局及CUDA/CuTe组合；Triton的原生硬件前提前置，既有编码/布局解释保留。
- 不用Triton的nibble/scale映射补FlashInfer缺失字段，保留sf精确shape、swizzle、算法、全局/局部尺度关系及匹配实现未核。INDEX/READING/SELF与计划同步：146条有依据、83条队列；本轮88已处理5条，7份FlashInfer文档待接续。
- 独立Astra内容review通过，无必修问题，启用条件按原文收紧。全库结构/来源strict检查184知识页、633项登记，0错误/待核；4个受影响Wiki页链接strict通过，新增入口与上下文已核。无量化实现或GPU运行验证。

## 2026-10-13 · FlashInfer计时文档 ingest：后端与样本口径

- 固定FlashInfer页面0.6.14、snapshot-2026-07-02，完整保存正文与README哈希一致；未保存原HTML。现有[正确性与计时页](implementation/kernel-correctness-and-benchmarking.md)补CUPTI/Graph/Event优先级、返回毫秒列表、调用方median及次数/时间预算区别，不新增摘要页。
- 保留Cold-L2策略正文缺失、弃用版本截断、迭代参数优先关系与多kernel/stream聚合未交代；不把文档描述推成已核源码或实际测量。READING145条有依据、84条队列；本轮88已处理4条，8份FlashInfer文档待接续。
- 独立Astra内容review通过，无必修问题；全库结构/来源strict检查184知识页、632项登记，0错误/待核，3个受影响Wiki页链接strict通过，新增入口与上下文已核。未核匹配计时实现、GPU缓存/采集行为或运行性能。

## 2026-10-13 · FlashAttention代码 ingest：分派、精度与反向

- 固定Dao-AILab/flash-attention `73c992c8ca746935548df620ef3c1b6238fe6e68`，包声明2.8.4；15个相关源码文件与固定Git对象字节一致，不重验全部25846项档案。静态研读普通Python/C++入口、SM80 FP16/d64/causal/dropout0标准前向及seqK反向，必要补GQA/deterministic。
- 新增[普通CUDA执行](implementation/flash-attention-2-cuda-execution.md)：核自动split-KV条件、shape/元素stride、raw-score在线状态/自然LSE、FP16乘法与FP32累计/舍入、空行sentinel、反向三次launch、dQ工作区/atomic/split合并及GQA展开缓冲。两论文页只补独立实现连接，保留版本与框架外部pin差异。
- 用户认可五条实现页解释原则，补入维护规范；FA示例按输入/前向状态/保存/反向梯度/归约重组，补变量角色、首块与mask条件，分派和核验细节后置。块缩放页“先把逻辑计算写清楚”改为“块尺度在 K 归约中的作用”，标题直接指向内容。
- INDEX/READING/SELF及计划同步：144条有依据、85条队列、184知识页；本轮88已处理3条，第12批下一项9份FlashInfer文档。源码按版本字符串分支而非语义版本比较、det仍用atomic及共享KV梯度舍入条件保留，未将静态分派或资源推算当设备/性能结果。
- 独立Astra内容review及独立可读性复核完成，公式可见域、保存范围与默认/deterministic合并时序修复闭环，无剩余必修问题。全库结构/来源strict检查184知识页、631项登记，0错误/待核；8个受影响Wiki页链接strict通过。复用CPU教学检查的分派/grid、48 KiB共享缓冲、8/32 MiB工作区、split分工、exp2换底与字符串版本判断；不构成CUDA或设备执行。split-KV设备/combine、其他接口/架构、完整RNG/构建与模型/设备运行未核。

## 2026-10-13 · FlashAttention-2 ingest：累计与工作分配

- 固定FlashAttention-2 arXiv:2307.08691v1，PDF14页与同版TeX；原PDF/源包SHA-256与登记一致。核读正文§1–5、算法1–2、图2–7和表1，关键公式/图表定位回PDF。
- 新增[FA2](theory/flash-attention-2-work-partitioning.md)，补未归一化片上累计、最终LSE、query外层与序列并行、warp分Q、反向列块/atomic dQ及GQA梯度归约。FA1只加后续连接，复用原有反向/IO与公共资源解释。
- 固定v1的逆缩放、背景归一化、因果方向、dK/D形状与缩放脚注按定义校正并保留原件差异；区分每序列/batch的matmul计数、causal微基准与model MFU、图/文及未来硬件预期，不把数学核对写成发布代码缺陷。READING143条有依据、86条队列、知识页183个；本轮88条已处理2条，第12批下一项固定代码，9份文档仍待接续。
- 独立Astra内容review及阅读路径复核无必修问题，补单头列块所有权与H100图点条件。全库结构/来源183知识页、627项登记，0错误/待核；5个受影响Wiki页链接strict通过。CPU float64教学核72组query外层前向/最终LSE、因果tile条件与逆缩放反例，24个共享KV梯度有限差分坐标及性能计数，最大前向误差6.7e-16、共享梯度误差6.7e-10；不构成发布代码、原子顺序或设备验证。未核发布代码、全部接口或GPU/模型/性能，后续固定代码单独登记。

## 2026-10-13 · FlashAttention ingest：精确分块、IO与反向重计算

- 固定原论文arXiv:2205.14135v2，PDF34文件页；原PDF/源包SHA-256与登记一致。核读正文§1–5、附录B/C，定向核D/E实现与实验条件；核心算法、公式、定理及所引图表以PDF和同版TeX回查。
- 新增[FlashAttention](theory/flash-attention-tiling-and-recomputation.md)，讲清整行softmax的分块合并、K/V外层循环、片上资源与IO推导，及最终统计恢复概率、D点积、dropout同mask重现和反向梯度。模型基础与GPU计算模式只补短连接，复用现有框架执行与roofline解释。
- 保留d²/M流量优势条件、固定容量下N²增长、非逐点最优下界、辅助状态/总内存差别，及短序列反例、各基准mask/dtype/硬件口径和原文GPT数字差异。READING142条有依据、87条队列、知识页182个；第12批只完成原论文，下一项FA2。按用户授权持续处理本轮开始时的88条全队列，历史58条范围待核不混入该基线。
- 独立Astra内容review与阅读路径复核完成，补齐IO流量优势条件及表21硬件/dtype，修正后无阻塞。全库结构/来源检查182个知识页、624项登记，0错误/待核；6个受影响Wiki页链接strict检查通过。CPU float64教学对照144组分块前向/重计算反向及72个有限差分坐标，最大对照误差1.8e-15、梯度误差5.5e-10，并核IO容量边界、合并反例与速度口径；不构成作者实现或运行复现。未核作者CUDA、具体RNG/空mask处理、现代库内部或设备、模型和性能。

## 2026-10-13 · LQ-Nets ingest：基码值、QEM与位运算

- 固定LQ-Nets arXiv:1807.10029v1；核读21页PDF的正文§1–5与补充§1–2，回查式2–8、算法1、表1–4/6–7和关键图；原件/源包SHA-256与条目记录一致，TeX仅PDF合并包装，公式编号按原PDF核对。
- 新增[LQ-Nets](methods/lq-nets.md)，补二进制基生成的结构化标量码值、最近编码/最小二乘、训练前向QEM与外层任务STE的分工，以及混合±1/0-1标签域内积和基/码预算。公共[码本](theory/codebook-quantization-and-bit-budget.md#1.%20%E4%BB%8E%E5%9D%87%E5%8C%80%E6%A0%87%E9%87%8F%E8%B5%B0%E5%88%B0%E5%90%91%E9%87%8F%E7%A0%81%E6%9C%AC)和[训练代理](fundamentals/quantization/post-training-and-quantization-aware-training.md#3.%20%E4%B8%BA%E4%BB%80%E4%B9%88%E9%9C%80%E8%A6%81%20STE)只补必要连接，复用LSQ/LSQ+及位平面解释。
- 保留码值碰撞、秩条件、EMA/本次重构时序未核、端层/不同位宽训练协议与基线范围，以及训练耗时和理论位运算兼容的区别。INDEX、READING、SELF和计划同步：有依据记录141条，待核队列88条，知识页181个；第11批三篇选定论文范围完成，下一项FlashAttention原论文v2。
- 独立Astra内容review及阅读路径复核完成；修正LSQ+对照的权重/激活参数与初始化范围，修正后无阻塞。全库结构/来源检查181个知识页、621项登记，0错误/待核；6个受影响Wiki页链接strict通过。教学计算验证124个选码探针、4步满秩最小二乘及奇异反例、5460个二进制内积模式对、4096个加权位平面组合、基预算与5处百分点差；只核代数，不构成作者实现或运行复现。未逐点重建全部图或对照，未核作者实现、奇异/重复码/EMA具体处理、真实packing/kernel与模型/部署性能。

## 2026-10-13 · LSQ+ ingest：偏移、初始化与有符号激活

- 固定LSQ+ arXiv:2004.09576v1；核读8页PDF正文§1–6及有效TeX，式1–8、图1–2、表1–6回查PDF；原件/源包SHA-256与条目记录一致，TeX注释里的旧设置不作证据。
- 新增[LSQ+](methods/lsq-plus.md)，解释实数偏移/整数零点、四种编码配置、共享任务代理、权重高斯初值/激活MSE初值及任务微调分工。连接[LSQ](methods/lsq.md)、[PACT](methods/pact.md)和公共[代理梯度](fundamentals/quantization/post-training-and-quantization-aware-training.md#3.%20%E4%B8%BA%E4%BB%80%E4%B9%88%E9%9C%80%E8%A6%81%20STE)，保留各自变量、反向与原实验条件。
- 保留均值括注和4位初值例差异、原偏移零梯度推论的上饱和反例、已刊1.8/1.9百分点差异、best/mean/Δacc口径、SE/bottleneck例外与静态bias折叠条件。导航、READING、SELF和计划同步：有依据记录140条，待核队列89条，知识页180个；第11批下一项LQ-Nets。
- 独立Astra内容review与阅读路径复核通过，明确代数换写与合法整数零点的条件；167组局部代理、18组网格重编号、静态bias展开及初始化/百分点计算一致。严格结构/来源/导航及七个受影响Wiki页面的链接检查通过，0错误、0警告。未逐点重建图，未核作者实现、确切训练/初始化/缩放协议、正约束、端层与导出/packing/kernel，未做模型/性能复现。

## 2026-10-13 · PACT ingest：激活裁剪与任务梯度

- 固定PACT arXiv:1805.06085v2；核读15页PDF正文及附录A–C，回查式1–3、图2–3/6–8、表1/2/5的阈值、正则、训练和硬件条件。PDF与源包SHA-256匹配来源记录；源包TeX仅为PDF包装，公式和编号依据原PDF核对。
- 新增[PACT](methods/pact.md)，补非负裁剪/网格、共享上界的任务代理、正则/初始化与参数预算；区别于[LSQ](methods/lsq.md)的内部舍入残差与梯度缩放。LSQ及公共[代理梯度](fundamentals/quantization/post-training-and-quantization-aware-training.md#3.%20%E4%B8%BA%E4%BB%80%E4%B9%88%E9%9C%80%E8%A6%81%20STE)只增必要连接，复用范围和部署基础。
- 保留无量化粒度消融、全精度初值轨迹、正约束未披露、端层位宽反例、ResNet18基线差异，以及DeepMatrix系统建模与真实运行的边界。INDEX、READING、SELF和计划同步：有依据记录139条，待核队列90条，知识页179个；第11批首项PACT完成，下一项LSQ+。
- 独立Astra内容review及阅读路径复核通过，补清阈值梯度与输入反传路径的区别；36组同网格前向/局部代理教学计算和关键百分点差值复核一致。严格结构/来源/导航与六个受影响Wiki页面的链接检查通过，0错误、0警告。未逐点重建曲线或逐项核全部比较表，未核作者实现、实际训练、完整低比特导出/内核或硬件复现。

## 2026-10-13 · LoRA ingest：任务增量与参数预算

- 固定LoRA arXiv:2106.09685v2；核读正文问题/方法/实验与秩分析主线，定向回查附录D.2–D.4、F.2、H.2–H.4的配置、预算与条件，主要编号对照26页PDF。PDF和TeX源包SHA-256与来源条目一致。
- 新增[LoRA](methods/lora.md)，补直接任务训练的低秩加法增量、$\alpha/r$、初始化反传、目标矩阵/任务预算与合并条件；与[LRQ](methods/lrq.md)的对数选码缩放及离线因子去向区分。复用LoftQ/QERA与低秩执行解释，只在公共[PTQ/QAT](fundamentals/quantization/post-training-and-quantization-aware-training.md#1.%20%E5%8C%BA%E5%88%86%E8%8C%83%E5%BC%8F%E4%B8%8E%E5%8F%AF%E8%B0%83%E6%95%B4%E5%8F%98%E9%87%8F)增加范式连接。
- 保留D.4与表15的预算冲突、q/v并非单项最优、小秩/指标/调参条件，以及任务checkpoint与基座、显存、吞吐口径。导航、READING、SELF和计划同步：有依据记录138条，待核队列91条，知识页178个；第10批四篇选定论文范围完成，下一项PACT。
- 独立Astra内容review与阅读路径复核完成，修正表18验证损失并列及LRQ选码固化措辞；五种矩阵/秩预算、FP16字节与吞吐口径复算一致。严格结构/来源/导航检查与五个受影响Wiki页面的链接检查通过，0错误、0警告。未核全部附表、分类任务头完整计数、作者代码与实际训练/合并/性能，不把论文整理写成运行验证。

## 2026-10-13 · LRQ ingest：对数因子与参数预算

- 核读LRQ固定v2正文主线、限制及必要附录，式1–2、图3–4、表7、附录B/F/G/I/J/M的预算/消融/成本与混合配置回查PDF；原件和源包校验值与条目记录一致。
- 新增[LRQ](methods/lrq.md)，明确低秩乘积、广播偏置与逐元素指数的不同秩界；说明块输入路径、初始化和条件性梯度链，完整辅助参数计因子/偏置/步长，表29只计主因子，不作为显存或推理存储比例。公共[尺度位置](fundamentals/quantization/post-training-and-quantization-aware-training.md#3.5%20%E5%AD%A6%E4%B9%A0%E5%B0%BA%E5%BA%A6%E5%89%8D%EF%BC%8C%E5%85%88%E7%A1%AE%E5%AE%9A%E5%AE%83%E8%BF%9B%E5%85%A5%E5%93%AA%E4%B8%80%E4%BE%A7)只补结构化选码连接，复用原有FlexRound/QERA/LoftQ/DBW解释。
- 保留宽度/样本与指标反例、混用FlexRound和后加A/KV的顺序、少参数但更慢的离线成本、局部FFN时延边界。独立review完成，补清整块参数集合并收紧表10方案级/逐组超参的证据边界；四种主因子比例重算一致，严格结构/来源/导航与链接检查0错误、0警告。未核作者实现、全部附表、字节驻留、导出/内核或模型/性能复现。导航、READING、SELF和计划同步：有依据记录137条，队列92条，知识页177个。

## 2026-10-13 · SignRound ingest：阈值搜索与符号更新

- 先按用户最新授权检查并单独保存既有两份Markdown表格格式及五个Obsidian JSON快照；表格单元格/来源/链接未变，JSON合法且未发现凭据，保留客户端历史内容。
- 核读SignRound固定v5正文主线、限制与必要成本证据，式1–6、算法1、表1/3–9回查PDF。新增[方法页](methods/signround.md)，解释加法舍入偏移、端点收缩、阈值区间/移动预算、块输出和当前batch最佳候选；与AdaRound/FlexRound/GPTQ分清参数化及更新规则。公共[代理梯度页](fundamentals/quantization/post-training-and-quantization-aware-training.md#3.%20%E4%B8%BA%E4%BB%80%E4%B9%88%E9%9C%80%E8%A6%81%20STE)只补sign更新与反向规则的连接。
- 标准仿射恢复和条件性STE梯度均标为整理者展开；保留原式未减回零点、学习率措辞与耗时步数差异，以及联合裁剪/更多迭代/方法比较反例。独立内容及阅读路径review完成，修正式1不含V、式3才加入V的定位表达；严格结构/来源/导航与链接检查通过，0错误、0警告。未核作者实现、全部附表、导出/内核或模型/性能复现。INDEX、READING、SELF和计划同步：有依据记录136条、队列93条、知识页176个。

## 2026-10-13 · FlexRound ingest：除法选码与共同网格

- 按用户确认恢复逐项 ingest，先处理第10批FlexRound固定v2。核读正文问题/方法/实验主线、附录B，定向回查消融与语言模型条件、反例；主要公式/命题/表号对照固定PDF。PDF和TeX源包校验值与条目记录一致。
- 新增[FlexRound](methods/flexround.md)，解释共同步长、同形逐元素除数、通道乘积、非对称重构输入、STE梯度和固化整数码；保留共享参数冗余的整理者分析、ImageNet/LLaMA反例、逐通道/逐张量及LoRA适配条件。公共[尺度位置](fundamentals/quantization/post-training-and-quantization-aware-training.md#3.5%20%E5%AD%A6%E4%B9%A0%E5%B0%BA%E5%BA%A6%E5%89%8D%EF%BC%8C%E5%85%88%E7%A1%AE%E5%AE%9A%E5%AE%83%E8%BF%9B%E5%85%A5%E5%93%AA%E4%B8%80%E4%BE%A7)已有同型推导，只补方法连接，不另建重复机制页。INDEX、READING、SELF和计划同步；有依据记录135条，队列94条，知识页175个。
- 独立内容review与阅读路径复核完成，补齐复合除数到实际参数的乘积链，以及W4A16反例的格式条件。严格结构/来源/导航检查和受影响页面链接检查通过，0错误、0警告；未核全部附表、作者实现、有限位宽裁剪/零点/正约束、导出/打包/内核及模型/性能复现，没有将数学固化方向写成真实部署证据。

## 2026-10-13 · 重复性审查：第 5 批，全库收尾

- 核读README/INDEX、READING规范与相关记录、LOG本轮条目及覆盖清单。沿均匀量化→AWQ→实现、对角缩放→SmoothQuant→实现、重构/AdaRound→GPTQ→实现及码本第9→12节复读，删改承接、局部解释与条件连贯；这不表示逐条核验全库存量关系。
- 全库定向检索来源定位说明，修复WUSH-KV末尾误指frontmatter文件的残留，保留OSCAR公开前置证据与未核WUSH集成边界；READING改明上一轮内容审查的时间指代，不升级研读范围或日期。
- 174项分组清单与实际知识页、INDEX逐项相符，无遗漏、重复或额外页；597项来源目录登记及固定版本保留。全库严格结构/来源/链接/锚点/raw存储与差异检查通过，0错误、0警告；没有合页、删除独有机制/推导/条件/反例或进行 ingest。各批已复核并分次提交、推送，原有工作继续留在工作区。

## 2026-10-13 · 重复性审查：第 4 批，训练与恢复（4k）

- 通读QAT/尺度/规范调整、PV/低秩/稀疏恢复、LLM.int8/ZeroQuant、SmoothQuant+/SplitQ及熵基础十七页。仅删除ZeroQuant重复的生成任务总结与QERA页末目标/解析解回顾，具体实验与复用条件保留。
- 固定码/重算码、前缀参考、梯度接口、连续/离散变量与存储消费各有独立作用，不合并为通用训练或补偿描述。独立删改/承接复核、受影响页严格链接及差异检查通过，无 ingest；174个知识页重复性分组审查完成，进入第5批收尾。

## 2026-10-13 · 重复性审查：第 4 批，多模态与状态（4j）

- 通读 VLM对象、对齐/IG公共页、MBQ/VLMQ比较及九个量化/状态方法、四个剪枝方法与QSLAW实现，共十八页。仅删除VLMQ重复的推理无梯度说明及Q-VLM本轮操作记录；MBQ/VLMQ已有表格排版继续保留在工作区，不暂存。
- 回看不同mask、MAE/MSE/G²归约、尺度两侧、反事实短分支/同轨迹后续、坐标与整数证据、删除集合/位置/阶段预算，确认各有独立作用。独立删改复核、严格链接和差异检查通过；其余十六页无确证应删重复，无 ingest。

## 2026-10-13 · 重复性审查：第 4 批，评测与质量（4i）

- 通读模型质量/诊断、HELM/LongBench v2/MMLU-Pro/RULER/harness/LVLM/TTS、可靠性/排序及价格回归十二页。删除可靠性页重复的Qwen2-VL/int3/ECE例子和LVLM评测页重复的无效输出收尾，原始例子、作者指标局限与诊断入口仍保留。
- 不合并PPL/BPB分母、strict/补偿/随机兜底、过滤/缓存、召回/翻转/低风险覆盖等不同协议与事件。独立删改/承接复核、受影响页严格链接与差异检查通过；其余十页未找到确证应删重复，无 ingest。

## 2026-10-13 · 重复性审查：第 4 批，部署与服务（4h）

- 通读模型主干、两库调度/Attention、TP、内存/服务指标、部署后端、QServe/Atom/FlexQ方法与实现、阶段量化及Ideogram十七页。删除SGLang Attention重复的普通MLP导航段，共同基础由其已有vLLM入口承接；删除部署后端表后的逐项复述。
- 保留生命周期、缓存身份、LSE缓冲语义、数据转换、后端分派、尾部与性能条件。独立删改和知识关系复核、受影响页严格链接及差异检查通过，无 ingest；其余十五页无确证需要修改的重复。

## 2026-10-13 · 重复性审查：第 4 批，硬件与算子（4g）

- 通读硬件/数值、张量接口/Triton、矩阵执行/格式、流水与测量二十二页。仅删除反量化、执行路径、Marlin 和 roofline 四处重复的未运行、性能口径或泛例提醒；具体接口、布局、guard、组尺度、尾部、舍入、等待/排空和测量条件保留。
- 不合并 Tensor Core 片段与实际低比特消费，不将块格式、descriptor 或基准与 profiling 混为同一职责。独立回读删改与承接，受影响页严格链接及差异检查通过；未发现本组旧来源编号说明残留，无 ingest 或 GPU 运行。

## 2026-10-13 · 重复性审查：第 4 批，MoE 与 MLA（4f）

- 通读 MoE/MLA 模型基础、路由/EP、DeepEP/DeepGEMM/W4A16、普通 MLA/DCP/packed/DSA 及 VEQ 十二页。相似表述承担不同坐标、通信、缓存、dtype、split/LSE 和归约职责，没有确证应删除的机制性重复，保留现有解释深度。
- 修复 W4A16、普通 MLA 与 DCP 三页仍沿用文件级 `sources` 的定位说明。W4A16 从旧固定登记恢复 28 项仓库相对文件编号表；MLA 保留十九文件核读覆盖事实与 FlashInfer 依赖/核读版本差别。科学正文、编号/行号和来源目录不变，独立定位复核、严格机械及差异检查通过，无 ingest。

## 2026-10-13 · 重复性审查：第 4 批，KV 与注意力（4e）

- 通读 KV/Attention 三个公共页、十个方法页和两个实现页，共十五页；邻近的 SPHQuant/CASA 正文核读按实际对象分别归入码本和混合精度。精简重复的 KIVI 数值回顾及 Kitty/NOVA-KV 页末机制总结，保留误差指标层次、保护轴、窗口/当前 chunk、元数据消费和验证条件。
- 修复 OSCAR 来源目录迁移后失真的“编号对应 frontmatter 文件顺序”说明：从原固定版本登记恢复 22 项正文源码定位表，编号、行号和科学正文不变，`sources` 继续按目录登记。核对固定树文件身份、独立删改及编号复核、受影响页严格检查与差异检查通过；无 ingest 或运行实验。

## 2026-10-13 · 重复性审查：第 4 批，混合精度（4d）

- 通读混合精度公共页、HAWQ-V2/CMPQ/CoopQ/LLM-MQ/DBW/StagQ/ResQ/RAMP/LUQ/PRQuant及ResQ实现十二页，另复用邻近组CASA的正文核读，按实际权重量化对象归属登记，共十三页。
- 只精简CoopQ重复的联合状态总结与StagQ重复的机制罗列；保留效用/成本、候选/预算、协方差代理、因子/子空间/共享前缀、保护字段和部署证据条件。重读承接与复用边界，独立删改复核、局部严格链接及差异检查通过，无 ingest。

## 2026-10-13 · 重复性审查：第 4 批，旋转与码本（4a–4c）

- 通读旋转/变换、码本方法、固定实现与新表示三组共 28 页，并复用邻近组的 SPHQuant 正文核读，覆盖记录按码本归属登记，共 29 页。保持各页职责和核心解释；码本高分辨率分组以第 12 节为主位置，第 9 节保留 NOVA-KV 应用与独立 Gaussian/渐近条件。精简四个旋转方法的页末复述、QuantForge 重测收尾、AQLM 重复 batch 定义及泛提醒。
- 清空二阶重构与旋转来源表中七行仅复述主题的泛说明，保留名称、固定版本、URL 及独有范围。具体预算、保护/恢复协议、图位置、有限精度、失败结果和证据差异不删。
- 新建全库重复性覆盖记录，174 个知识页分组无遗漏或重复。完成独立删改/承接复核、受影响页严格链接与差异检查；没有合页、删除核心推导或进行 ingest。混合精度、KV、MoE/MLA、硬件/评测及剩余专题继续分组推进。

## 2026-10-13 · 重复性审查：第 3 批，公共机制

- 通读校准、二阶重构、曲率、对角缩放、均匀量化、PTQ/QAT 六个公共页及 AdaRound、BRECQ、CalibQuant、ConQuR、LSQ、OmniQuant 六个方法页，并对照第 2 批经典方法。主要解释位置保持现结构；精简四页的范式定义、方法专属推导复述、重复分工、维护历史和无信息增量收尾。
- 保留固定输入重构与任务曲率的区别、归约单位、Gauss–Newton/Fisher 假设、不同 STE 节点、融合/RoPE 条件、状态/范围选择与实际预算。重读承接段落，独立复核删改及阅读连贯性，受影响页严格链接和差异检查通过；没有删除页面、核心公式或教学反例，无 ingest。
- 第 4–5 批待办：其余主题分组与全库阅读路径收尾。

## 2026-10-13 · 重复性审查：第 2 批，经典方法与实现

- 通读 AWQ、GPTQ、SmoothQuant 的方法页与实现页六页，修订其中五页。方法页保留真实产物和适用条件的概述，将重复的打包、dtype 与分派细节指向已有实现；删除页内重复的分工、核验状态、g128 和代码路径提醒，以及维护过程叙述。
- 机制公式、搜索与补偿目标、校准顺序、模型/后端差异、失败结果和来源冲突完整保留。独立 review 发现 AWQ 可选 bias 尚未承接，定向回查固定源码并补入已有产物表。回读承接章节、复核删改差异与新增章节链接，局部严格链接检查通过；没有重新审计全部论文或代码主张，无 ingest 或运行实验。
- 第 3–5 批待办：跨页公共机制、其余主题分组与全库阅读路径收尾。

## 2026-10-13 · 重复性审查：第 1 批，来源区

- 核对 53 页重复的来源表引言及 25 页全空的“说明”列：删除只复述统一引用职责的引言，将全空说明表缩为来源、版本两列。来源名称、固定版本、URL 与非空说明原样保留，包括局部研读、快照身份、归因及所选实现范围。
- 逐页核对差异仅在来源区；正文、frontmatter、知识链接及 597 项来源登记未变，174 个知识页保留。完成独立内容复核、全库严格机械检查与差异检查；本批没有重新核验科学主张或原件，不进行 ingest。
- 第 2–5 批待办：经典方法页内重复、跨页公共机制、其余主题分组及全库阅读路径收尾。原有 review 文件、MBQ/VLMQ 表格排版和 Obsidian 配置继续保留，不纳入本批提交。

## 2026-10-13 · 来源登记按条目归并

- 论文、文章和代码的 `sources` 统一指向来源条目目录，同一固定版本每页只登记一次。同步维护规范、wiki-review 设计与执行说明，以及条目/版本检查；具体证据仍由正文定位。
- 分五类迁移 174 个知识页，登记项从 1,568 减至 597；历史代码引用从 21 个文件引用归并为 7 个固定 commit 根入口。逐页确认正文与其他元数据未变，188 个来源记录的身份和版本保持原样。
- 42 项回归测试与全库严格检查通过，0 错误、0 警告。保留已有工作区改动；未新增来源或进行 ingest。

## 2026-10-13 · 全库分批审查：第 30 批

- 核读 INDEX、READING、LOG 与 README。修正“缓存生命周期”入口误指格式章节，AdaRound/BRECQ/QServe 摘要改为具体解释职责；回修 BRECQ 页首的普遍失效断言，与粒度消融和近似条件保持一致。
- READING 压缩重复的评测未核清单与九批进度，保留研读范围、来源差异及匹配实现缺口；LOG 说明移至页首，历史服务缺口标明当时范围并接入现有正文。SELF 与 ingest 计划同步暂停状态，既有来源记录不升级为完整研读。
- 全部 30 批、174 个知识页及 4 个导航/管理页完成。批次清单与 INDEX 覆盖无遗漏或重复；134/58/95 条研读状态计数一致，全库结构、来源登记、链接/锚点与 raw 存储检查通过。未进行 ingest 或模型/GPU 运行；科学内容核查按各批核心主张与疑点定向开展。

## 2026-10-13 · 全库分批审查：第 29 批

- 通读 HELM、LongBench v2、MMLU-Pro、RULER、harness 与公共模型质量页；回查等量分箱/ECE与SCAA、两次CoT/补偿计分、随机兜底、RULER阈值与权重，以及regex、字符串长度归一、响应缓存和学科聚合。核心定义与跨页分母一致，已有解释深度足够。
- 压缩来源段重复的未审全体/未运行清单与公共页过程尾注；harness 改明保存输入、逐例计分和缓存记录的目的。保留论文与实现的parser差异、原件公式/登记冲突、有限网格、并列、参考身份及配对统计条件。
- 无 ingest 或模型运行；174 个知识页全部完成，剩余 4 个导航、管理页。内容复核、链接与差异检查通过。

## 2026-10-13 · 全库分批审查：第 28 批

- 通读 W4A16 MoE 页，回查 AWQ 专家尺度置换、Marlin 两次 GEMM 的配对/加权参数、低精度 gate、SiLU 中间转换及 FP32 routed sum；与 Dense Marlin 和 FP8/MoE 页的接口、舍入区别一致。
- 删除来源段、教学例和结尾重复的未部署/未运行声明；保留页首静态核读说明、单组形参边界、合法路由与未核分支。格式、地址、缓冲和数值链的深度足够，无需补写机制。
- 无 ingest 或 GPU 运行；累计完成 168 个知识页。内容复核、链接与差异检查通过。

## 2026-10-13 · 全库分批审查：第 27 批

- 核读 MoE 路由/EP、DeepEP/DeepGEMM、普通 MLA、DCP、packed DSA 与设备链八页；回查 LL 容量/scale stride、专家 tile 有效范围、LSE 底数、空分片归约、主缓存恢复与 indexer 页平面。主要机制和跨页接口一致，解释深度足够。
- MLA 集中说明源码解释与未运行条件，删除各阶段重复声明；DSA 空行段保留合法索引、有限输入和已分配缓存条件，移除极端输入清单。压缩单设备 MoE 和 DeepGEMM 的重复尾注，保留固定版本差异、dtype/维度 guard、DCP 空分片 NaN 风险及未核读分支。
- 无 ingest 或 GPU 运行；累计完成 167 个知识页。内容复核、链接与差异检查通过。

## 2026-10-13 · 全库分批审查：第 26 批

- 核读 vLLM 执行/Attention、Atom、DQ、Ideogram 与 QServe 六页；回查缓存 shape/stride、进度与在途计数、分段归约、阶段 mask/backward、Atom 配方及 QServe 表3和延迟分项。具体机制与相邻实现页一致。
- QServe 通道重要性从均值改为 max-abs，统一行 token 权重的缩放方向；QK/SV 两次降幅合计0.06 ms。五任务差值改为百分点，70B表中76.57→76.40与正文0.40%的差异保留；区分 Atom 最终INT8保护与FP16消融。压缩重复优劣、未运行及未审全仓声明，保留保护范围、原件差异和实验协议条件。
- 无 ingest 或模型运行；累计完成 159 个知识页。内容复核、链接与差异检查通过。

## 2026-10-13 · 全库分批审查：第 25 批

- 核读 Atom/QServe 实现、部署支持面、内存调度、服务指标、SGLang 两条执行链与张量并行八页；回查分页量化参数、Radix 锁计数、TP 的 KV 复制与 bias、TPOT/goodput 聚合以及 PagedAttention 的实验解释。执行步骤与接口深度足够。
- 部署页修正接口不匹配必然回退/报错的绝对说法，区分 GGUF 导出策略与张量类型；内存页区分请求供给不足与计算受限，并将论文显存结论限定到其设置。公共 KV 表补清 head 内共享与 token/时间范围是两项粒度，合并重复范围声明。
- 无 ingest 或服务运行；累计完成 153 个知识页。内容复核、链接与差异检查通过。

## 2026-10-13 · 全库分批审查：第 24 批

- 核读 Tensor Core、W8A8 与权重反量化三页；复核 MMA片段/寄存器、AWQ split-K、INT8舍入及校正、Marlin打包与零点/尺度/流水/归约。回查量化器、反量化常量、epilogue类型和工作区分配，所选计算链与相邻页一致。
- `_MAX_THREAD_N=256` 修正为CTA的N方向tile宽度上限，说明它用于FP32归约工作区，线程数由其他参数决定。删除反量化页重复实现要点，核验范围改为两段，保留SGLang/vLLM、对称/非对称与未核模板的实质区别。
- 无 ingest 或 GPU 运行；累计完成 145 个知识页。内容复核、链接与差异检查通过。

## 2026-10-13 · 全库分批审查：第 23 批

- 核读门控激活、GGUF、异步流水、性能分析、配置调优、正确性测量、Marlin 与 RMSNorm 八页；回查编码结构、imatrix 拟合/必需条件、MMA、批次等待、autotune key/hook、计时与归约写回。具体执行链及测量协议深度足够。
- GGUF 修正 K-quant 尺度概括、权重/激活符号和 imatrix 条件；未注明硬件的吞吐表移除CPU归属，并用反例说明速度并不随位宽单调变化。Marlin 修正将矩阵操作数尺寸误读为指令次数的问题，M=16的64列需8次 m16n8k16；压缩重复优劣、边界和未来实验段落。
- 无 ingest 或 GPU 运行；累计完成 142 个知识页。内容复核、链接与差异检查通过。

## 2026-10-13 · 全库分批审查：第 22 批

- 核读 roofline、GPU 执行、FP8/MX、计算模式、布局、Triton 语义、块缩放及 CUDA 接入八页；回查流量计数、FP8 位模式、mask/stride、类型提升、scale permute、硬件入口及扩展包装。关键索引、预算与算法解释一致。
- GPU 页改明激活需在线重排，整数吞吐与强度使用 OP 单位；修正 BLEU 拼写。压缩 roofline/GPU/FP8 重复边界与计算模式、Triton、块缩放的过程尾注，保留实际硬件、格式、包装与未运行条件。
- 无 ingest 或 GPU 运行；累计完成 134 个知识页。内容复核、链接与差异检查通过。

## 2026-10-13 · 全库分批审查：第 21 批

- 核读 VidKV、WUSH-KV 与公共 KV 框架三页；回查 FFT/三值公式与缓存追加、WUSH 构造与 QuEST 网格，复核物理预算及窗口时间线。VidKV 的原件歧义、WUSH 的正定与量化器条件保留，未发现新增算法性错误。
- 公共页同步 KIVI 的1.90个百分点和 softmax 后权重误差，SAW 改为融合降低额外开销、解码独立旋转占1.73%；删去“融合决定成败”及重复边界小节，范围说明对齐已有格式与代码分析。
- 无 ingest 或模型运行；累计完成 126 个知识页。内容复核、链接与差异检查通过。

## 2026-10-13 · 全库分批审查：第 20 批

- 核读 FlashInfer 分页/NVFP4、OSCAR、Kitty、KIVI、KVQuant、NOVA-KV、QJL 与 TurboQuant 八页；回查尺度倒数、INT2 打包与恢复精度、窗口刷新、敏感度尺度平方、查询加权变换及投影/残差估计。
- KIVI 的 GSM8K 差距明确为1.90个百分点，TruthfulQA 指标改善不擅自判作波动；吞吐倍数限定同显存增批条件。删去 KIVI 重复 Strong/Weak、OSCAR 数值接口回顾和 KVQuant 运行尾注，保留真实预算、论文证明问题及具体未核接口。
- 无 ingest 或模型运行；累计完成 123 个知识页。内容复核、链接与差异检查通过。

## 2026-10-13 · 全库分批审查：第 19 批

- 通读选择性预测页，回查 MaxProb/Selector 定义及两张主表。原文“所有位宽都降低 ECE”的概括存在反例：Qwen2-VL int8 HQQ 的 ECE 从3.5升到4.3，正文改按指标分别解释。
- “恢复 bf16 水平”区分 MaxProb 与 Selector 基线；75%内存表述区分裸权重位宽比与峰值显存。阈值等号对齐原文，总体期望风险与有限样本曲线分开；压缩重复提示与运行尾注。
- 无 ingest 或模型运行；累计完成 115 个知识页。内容复核、链接与差异检查通过。

## 2026-10-13 · 全库分批审查：第 18 批

- 核读 SAPrune、SAW-INT4、VEQ、VLMQ、MBQ/VLMQ 比较及积分归因、对齐诊断、排序稳定性八页；回查双路径预算、INT4 恢复顺序、专家加权、$G^2$ 与非对称残差、完整性、全基投影及排序概率条件。
- SAW-INT4 的“零服务开销”改为融合降低额外开销，删去其与 VLMQ 的重复 Strong/Weak 和运行尾注，保留具体代码范围；排序分离比补明非零扰动条件。比较页正文与相邻方法一致，原有排版改动保留且未暂存。
- 无 ingest 或模型运行；累计完成 114 个知识页。内容复核、链接与差异检查通过。

## 2026-10-13 · 全库分批审查：第 17 批

- 核读 MQuant、OnPTQ、Pivot-Q、PRQuant、Q-VLM、QAPruner、QIG 与 RAMP 八页；定向回查分模态布局、决策风险、时序选择、尾部残差、熵分块、剪枝评分、归因分支及聚类阈值。
- Q-VLM 相邻分数平均补明 $b>a$，单层块不使用此式；压缩 Q-VLM/MQuant/QIG 的重复优劣段落，删除 OnPTQ/QIG 的过程尾注。保留论文与代码差异、OCRBench 反例、梯度与裁剪分支以及校准成本等具体条件。
- 无 ingest 或模型运行；累计完成 106 个知识页。内容复核、链接与差异检查通过。

## 2026-10-13 · 全库分批审查：第 16 批

- 核读压缩评测、TTS 评测、AKVQ-VL、CentriQ、CoRePrune、FoldQuantVLA、MASQuant 与 MBQ 八页；定向回查联合压缩表、配对 bootstrap、KV 保护、删除恒等式、均值旁路、INT4 prologue 及模态重构路径。
- 修正逐通道 KV “性能一致更差”的概括，补明 MME 反例；评测表区分离线校准与权重反向更新。合并评测页重复边界，删除 MBQ/MASQuant 的重复 Strong/Weak 及 CentriQ 的 ingest 过程表达，保留独有的指标核对、模型范围和部署条件。
- 无 ingest 或模型运行；累计完成 98 个知识页。内容复核、链接与差异检查通过。

## 2026-10-13 · 全库分批审查：第 15 批

- 核读 QAT 基础、QSLAW 方法/实现、EfficientQAT、LC-QAT、LLM-QAT、LSQ 与 QUASAR 八页；定向回查步长/零点 STE、固定码尺度、仿射网格与 DGE、QUASAR 条件界及 QSLAW 浮点恢复路径。关键推导与跨页表述一致，未发现新增算法性错误。
- 公共页范围说明补明 LSQ/LLM 机制；删除重复的来源维护尾注、QSLAW 运行范围罗列及 QUASAR 过程声明。保留正尺度/奇点、数据与文件接口、具体未核分支、SGD/PL 假设和实际浮点执行条件。
- 无 ingest 或模型运行；累计完成 90 个知识页。内容复核、链接与差异检查通过。

## 2026-10-13 · 全库分批审查：第 14 批

- 通读 SplitQ 和公共混合精度页；回查通道选取、CWS/MAC 与 SVD 门控，复核补偿误差恒等式、加权位宽、前缀/分位搜索及 Shapley 算例。公共机制与已审方法页一致，未发现新增推导错误。
- 公共页范围说明改为层间、通道及因子分配，补齐正文已引用的 CASA/HAWQ-V2 来源身份；SplitQ 删除重复 Strong/Weak 与来源维护尾注，保留固定版本、快照缺失和特定 W4A4 prefill 成本条件。
- 无 ingest 或模型运行；累计完成 82 个知识页。内容复核、链接与差异检查通过。

## 2026-10-13 · 全库分批审查：第 13 批

- 核读 CASA、CMPQ、CoopQ、HAWQ-V2、LayerPos、LLM-MQ、LUQ 与 Quantization Price Prediction 八页；定向回查联合状态评分、曲率/一阶代理、预算选择、聚类熵和位置编号执行路径，未发现新增算法性错误。
- CMPQ、LLM-MQ 的保护值解释改指覆盖/相加协议小节，原落点只是计费；删去 CMPQ、CoopQ 的重复运行尾注及 LUQ 的重复 Strong/Weak 概括，保留选择/评测隔离、部署格式重评等具体缺口。LUQ 项目页日期对应历史快照，不以归档目录日期替换。
- 无 ingest 或模型运行；累计完成 80 个知识页。内容复核、链接与差异检查通过。

## 2026-10-13 · 全库分批审查：第 12 批

- 通读公共码本页，核对标量/向量/加性表示、码本距离、残差语义、物理格式、路径与球面码率，以及 Lloyd–Max、信息下界和高分辨率推导。预算、算例与第 9–11 批方法/实现一致。
- AQLM 预算表补明每输出行沿输入分组、每行一个 FP16 尺度，避免套到实现支持的二维分组；删除来源末尾与正文重复的球面下界和连续代理限制，具体条件仍留在对应推导中。
- 无 ingest 或模型运行；累计完成 72 个知识页。内容复核、链接与差异检查通过。

## 2026-10-13 · 全库分批审查：第 11 批

- 核读 QuIP、SharQ、SpQR、SqueezeLLM、StagQ、VPTQ、XOR-Trellis 与低比特表示设计八页；未发现新增算法性错误。压缩四个方法页重复的运行尾注，将 StagQ 阅读范围移至来源说明，删除表示设计页重复的例外收益限制。
- 定向回查 SqueezeLLM 敏感性目标与稀疏零值修正、StagQ 前缀/保护值与半步定义、VPTQ 同列选码及加权初始化、XOR-Trellis 局部候选和路径目标。QuIP、SharQ、SpQR 的所选实现沿用第 9 批有效核对；预算、来源差异与具体运行限制保留。
- 无 ingest 或模型运行；累计完成 71 个知识页。内容复核、链接与差异检查通过。

## 2026-10-13 · 全库分批审查：第 10 批

- 核读 DBW、FlexQ、FluxBin、GPTVQ、PV-Tuning、QTIP、QuantForge、QuIP# 八页，核心表示、选码目标与跨页承接一致。精简 DBW 末节重复的训练、执行和优化限制，保留表 19 的独有预算差异；压缩 FluxBin、QuantForge、QuIP# 重复运行声明。
- 定向回查 DBW 分配与分阶段损失、GPTVQ 加权聚类及预算、FluxBin 显著列和两分支载荷、QTIP 尾咬合、QuantForge 激活选码及辅助状态、QuIP# 块反馈与微调；FlexQ/PV 的实现核对沿用上一批仍有效的结果。必要数学前提、作者实验协议和未消解的来源差异保留。
- 无 ingest 或模型运行；累计完成 63 个知识页。内容复核、链接与差异检查通过。

## 2026-10-13 · 全库分批审查：第 9 批

- 通读 AQLM 方法及校准/PV 两实现、FlexQ、QuIP、QuIP#、SharQ 与 SpQR 实现八页，核对优化目标、保存对象、位预算、恢复坐标和所选内核消费。AQLM PV 将索引位数记为 b_c，避免与输入块下标 b 冲突。
- 删除七页重复的结尾核验声明，保留首段静态范围及正文中真实导出缺口、形状/精度条件、近似选码、浮点保存、保护值覆盖/相加与计时口径。没有把未运行的代码升级为运行证据。
- 定向回查 AQLM 原始二阶矩/目标/PV 更新顺序、QuIP 逆序反馈与错位参数、E8P 打包、FlexQ signed 位权/组尺度、SharQ 编码残差和 SpQR 保护值转换；AQLM 加性表示与 MRF 推导回查固定论文。至本批覆盖 55 个知识页，未进行 ingest 或模型实验。

## 2026-10-13 · 全库分批审查：第 8 批

- 通读 SmoothQuant+、SpinQuant、三方法比较与公共旋转四页，核对全局/局部目标、图位置、Cayley 更新、查询加权距离与研究结论。SpinQuant 将求和公式改为公开前向的有效 token 平均，区分单批归约和训练采样/梯度累积。
- 公共旋转页同步 SAW-INT4 的 K/Q 配对，补 NOVA 满秩逆变换条件，来源表收回两条误放在教学材料下的论文身份。合并方法/比较/公共页重复过程说明，保留校准与评测重合、近似 Cayley、量化集合与部署范围。
- 回查 SpinQuant 的 CE/Cayley 固定实现及论文、SmoothQuant+ 的搜索与校准表，复核 AffineQuant/FlatQuant 两种前缀推进；公共代数与既有教学计算一致，其余引用沿用明确的历史研读范围。至本批覆盖 47 个知识页，未进行 ingest 或模型实验。

## 2026-10-13 · 全库分批审查：第 7 批

- 通读 ResQ 实现、AffineQuant、FlatQuant、OmniQuant、OSTQuant、QuaRot、ResQ 与 RGSQ 八页，核对变换位置、优化变量和部署路径。修正 FlatQuant/QServe 对比遗漏的量化粒度，明确完整整数累加后的尺度与分组部分和缩放；修正 QuaRot 对 SAW-INT4“只旋转键”的含混概括，补配套 Q 旋转与 decode 融合。
- 明确 FlatQuant 正损失归一化的梯度为原梯度乘 1/L，压缩 AffineQuant、OSTQuant、RGSQ 与 ResQ 实现的重复过程说明，保留可逆条件、RoPE、证明缺陷、格式与硬件范围。
- 定向回查固定论文的核心变换/证明/目标，以及 FlatQuant 校准、AffineQuant mask、ResQ 基底、RGSQ 角度搜索和 SAW-INT4 查询旋转代码。OmniQuant 已移除的本地代码沿用历史核读范围；未重新审计全部实现或实验。至本批覆盖 43 个知识页，未进行 ingest。

## 2026-10-13 · 全库分批审查：第 6 批

- 通读 QERA 与 SPHQuant 两页，回查固定论文的加权 SVD、对角二阶矩条件、方向/半径投影、角度参数化与代表质量表。QERA 和 LoftQ 的变量、主体更新及最优性区别一致，核心解释保留。
- SPHQuant 补固定正角度下限对可表示方向的限制：坐标与球面边界保持正距离，不能任意逼近；压缩 QERA 及 SPHQuant 重复核验声明，保留校准、奇异统计、训练、真实位预算与作者实现证据范围。至本批覆盖 35 个知识页，未进行 ingest 或模型实验。

## 2026-10-13 · 全库分批审查：第 5 批

- 通读误差诊断、激活引导补偿、AdaRound、BRECQ、CalibQuant、ConQuR、LoftQ 与 Norm Tweaking 八页，核对重构变量、投影残差、旋转更新、矩匹配及缓存预算。修正 AdaRound 用同向舍入失效推断耦合及“不更新权重”的含混表述，区分浮点参数、舍入变量与最终量化权重。
- BRECQ 补 Gauss–Newton 被忽略项与 Fisher 采样条件，修正模型必须全由块组成的过强要求，明确检测与准确率差值单位；合并 AdaRound/BRECQ/诊断/LoftQ 重复过程声明，保留局部代理、数据、格式、硬件与未核实现条件。
- 定向回查八页对应的固定论文 TeX 与 Kitty 舍入/尺度代码，重读相关公共重构与曲率段落；来源核查覆盖核心机制和所修改数字，未逐项重审全部实验或运行模型。至本批覆盖 33 个知识页，未进行 ingest。

## 2026-10-13 · 全库分批审查：第 4 批

- 通读稠密 decoder、MoE、MLA、图文 token、Attention 误差与 sink 图谱六页，核对共享头、缓存容量、矩阵吸收、softmax 与集合指标的条件。核心解释可用，不为增加修改而改写推导。
- 修正图文页对 KV 公共页“不区分模态”的过时描述，明确共享缓存组织与按位置选择精度是不同层次；将表中“两篇”明确为 MBQ/VLMQ。删去两处重复运行声明，保留 V2/V3 路由区别与子集 softmax 恢复条件。
- 回查 DeepSeek-V2 v5 架构/设备路由、BOS-Sink v1 指标和现有 Llama 投影组织；其余主张结合现有推导与承接页面核对。至本批覆盖 25 个知识页，批次清单同步完成状态。

## 2026-10-13 · 全库分批审查：第 3 批

- 通读 LLM.int8()、ZeroQuant、GPTQ 实现与 SmoothQuant 实现四页，与已复核的 GPTQ 方法页对照，共覆盖五页。修正经验离群比例的固定化与无来源的后续普及判断，解释行尺度无法隔离离群列的机制，保留作者对剩余特征分布的归因。
- 区分 ZeroQuant 的 20B 推理观察与 LLM.int8() 的训练／微调附录；将 FP16 GEMM 输出的处理明确为反量化，区分 GPTQ 补偿与 AWQ 缩放／裁剪。合并两实现页重复运行声明，保留格式、TP、形状、分派及 CPU 教学材料的范围。
- 回查两篇固定论文的相关段落及现有框架的类型映射、阈值和动态 INT8 量化代码；未新增来源 ingest。至本批，内容核读与复核覆盖 19 个知识页；其余主题及导航、管理页仍待逐批审查。
- 完整批次清单保存在 `research/plans/2026-10-13-wiki-review-batches.md`，178 页均已分配，无遗漏或重复；提前修复来源表不计为整页完成。

## 2026-10-13 · 全库分批审查：第 2 批

- 通读线性层、浮点表示、均匀量化、熵、可逆变换、曲率误差、校准七页，结合已完成的重构推导复核，共覆盖八页。修正曲率页 token 平均与元素 MSE 的归约差异，明确代价符号，压缩校准页重复过程声明。
- 按原始条目的历史获取记录修正五处 `None` 日期，同步 Q-VLM、FlatQuant、LUQ 来源表；这三页本批仅核对相关来源身份，正文审查留在后续主题批次。保留未中心化统计、退化条件、网格/代理/任务区别与作者实验条件。

## 2026-10-13 · 全库分批审查：第 1 批

- 复核等价缩放、AWQ、SmoothQuant、三方法比较、低比特矩乘执行与 AWQ 实现六页；修正分派与阶段瓶颈的过强对应，明确 TinyChat/AutoAWQ 格式入口，补公共倍率归一化的推导及数值下限条件，合并方法页重复核验声明。
- 对照已有固定论文与代码，保留融合条件、网格端点、内核配置和原文实验差异；全库机械基线通过。内容审查仍按批次推进，未进行 ingest。

## 2026-10-12 · 抽查问题修复与内容复核

- 在[重构机制页](theory/layer-reconstruction-second-order-compensation.md)将二维算例前移到标量补偿之后，补齐自由坐标消元到 Cholesky 尾块的推导；[GPTQ](methods/gptq.md)接入这一桥接，说明更新与损失分母的区别。
- [均匀量化](fundamentals/quantization/uniform-quantization-and-groups.md)与[AQLM PV 实现](implementation/aqlm-pv-training-and-code-updates.md)集中范围说明，删去重复过程表述，保留分布/饱和、存储/计算、更新顺序与阈值例外等条件；[公共评测页](implementation/model-quality-evaluation.md)保留分箱与面积解释，HELM 原式差异由[协议页](implementation/helm-evaluation-protocol.md)集中承接。
- 回读修改段落与关联目标，复核 GPTQ v2 算法1/固定实现、PV v2 附录K及 AQLM 选码/分块损失代码。32组正定矩阵、176步消元/补偿和120组即时/延迟更新比较通过，并复算二维例子、ECE与同网格面积关系。本轮为已有知识修复，无新增 ingest。

## 2026-10-12 · harness ingest 与P1收尾评审

- 研读Summer-Summer固定62617a8c的MMLU-Pro配置/十四学科/prompt工具和共用请求、过滤、结果、聚合与缓存路径。新增实现页，公共质量页补执行协议；MMLU-Pro论文页接入代码差异，计划/SELF/READING同步第1–9批闭合与第10批建议。
- 核清generate_until与multiple_choice请求、两个首选层次、单regex失败计零、metric归一化位置、实际样本加权，以及请求/响应缓存身份边界。未运行完整框架、下载数据/模型或执行真实backend/API/GPU/量化评测；其余任务/后端、滚动分词、分布式故障/CI未核。
- 独立Astra核查32组核心主张和12处新增关系，无P1/P2必修；采纳repeats默认来源、示范答案拼接及log_samples字段的澄清。归档SHA与29个选定文件逐字节一致；13个Python文件AST解析、八组抽取/计分与提示/请求/加权/缓存行为夹具通过，未导入完整框架。
- 第2–9批35个计划条目均对应固定身份、研读范围与LOG，第1批AQLM接续亦有范围记录；此为管理核对，沿用各项独立内容评审。当前有依据记录134条、历史待核58条、队列95条（36论文/48文章文档/11代码），根入口135；全库174知识页、1564来源、2992本地链接，0错误/0待核，客户端实际跳转未核。P1既定范围完成，OnPTQ/WUSH等条件材料与未核分支仍保留；下一项建议第10批FlexRound。

## 2026-10-12 · raw 去除重复派生材料

- 删除 5108 份重复读本、公式册、图片清单、PDF 页图及导航等文件，约 1758.7 MiB；保留作者 PDF、TeX/源包、代码版本与必要网页原图。条目说明移除重复摘要、通用说明和已删除文件登记。
- raw 保存规范压缩为 43 行，取消整套重录与页图要求；仅规范纳入 Git，来源档案仍留在本地。原文具体疑点、定位与条件承接到已有研究审查记录，本轮没有重新进行科学核查。
- 同步 FlashInfer 来源登记及阅读范围；现有 lint 增加已知派生材料检查与两项回归用例。机械检查结果在交付时报告。

## 2026-10-12 · RULER ingest 与独立评审

- 研读固定arXiv:2404.06654v3的摘要、§3–8、附录A概述/主模型登记及B–D，回查PDF3–10/17–18/20–23共14页、表1–9所选主试验/配置/模板。新增任务复杂度/召回/有效长度页，公共质量页补任务阈值、长度权重与多目标计分。
- 区分13配置/四类别、目标召回/整题正确、测试档位/声明长度及实际部署权重；保留S-NIAH模板、Mixtral/DBRX登记和API部署范围差异。未审全部37模型、附录E/F、曲线数值/引用或匹配生成/解析代码，未生成数据或执行模型/API/GPU/量化评测。
- 独立Astra核查30组核心主张、14页原始页图与13处新增关系；1项P2受控对照概括已修正并回读，现无P1/P2必修。采纳最大值空集合的可选澄清。两原件SHA与46个实际展开文件逐字节一致；复算表3全部17行均值/序号权重，以及任务计数和多目标指标边界。
- 管理有依据记录133条、历史待核58条、队列96条（36论文/48文章文档/12代码），根入口134。并行raw清理同步来源后，全库173知识页、1546来源、2981本地链接，0错误/0待核；本轮五页1047引用通过，未验证客户端实际跳转。下一项harness。

## 2026-10-12 · MMLU-Pro ingest 与独立评审

- 研读固定arXiv:2406.01574v6的摘要、§3–8、附录A.1–A.2/A.4，回查PDF4–9/13–14/16共9页、表1–7与图2–5。新增构建/选项/解析页，公共质量页补逐题随机基线与兜底期望。
- 区分筛选/新来源/选项变化、5-shot与Gemini例外、正则失败/随机兜底及提示范围；保留表/图GPT-4o和Phi3-medium差异。未审全部题目/题解/24个prompt/错误案例/引用或作者实现，未运行模型/API/GPU/量化评测。
- 独立Astra直接核查26组核心主张、9页原始页图与10处新增关系，无P1/P2必修；采纳唯一正确选项和图5六模型的可选澄清。两原件SHA与3个实际展开文件逐字节一致；核算题目来源/筛选、收益百分点、错题分类及逐例随机期望。
- 管理有依据记录132条、历史待核58条、队列97条（37论文/48文章文档/12代码），根入口133；校正顶部遗留的126条计数。全库172知识页、1545来源、2969本地链接，0错误/0待核，未验证客户端实际跳转。下一项RULER与harness。

## 2026-10-12 · LongBench v2 ingest 与独立评审

- 研读固定arXiv:2412.15204v2的摘要、§3–4、结论/限制、附录B/D/E及C.3开头；回查PDF1–9/24–26共12页、表1–5/图1–4。新增上下文/调用/计分协议页，公共质量页补长度分组的配对条件与失败分母。
- 区分words与tokens、完整语料与截断输入、限时筛选的人类基线、两次CoT/内部推理及strict/compensated；保留结论人类50.1与摘要/主表53.7差异。未审503条数据、完整平台/指南/引用或匹配实现，未执行检索/API/模型/GPU/量化评测。
- 独立Astra直接核查28组核心主张、12页原始页图及13处新增关系，无P1/P2必修。两原件SHA与13个实际展开文件逐字节一致；教学核算通过六类/难度/长度计数、三种失败分母、3.44个百分点CoT增量、表3加权与舍入、RAG块预算。
- 管理有依据记录131条、历史待核58条、队列98条（38论文/48文章文档/12代码），根入口132条。全库171知识页、1543来源、2960本地链接，0错误/0待核；未验证客户端实际跳转。下一项MMLU-Pro、RULER与harness。

## 2026-10-12 · Wiki 章节链接修复

- 将 85 页中 426 处 GitHub 式 slug 引用改为 URL 编码的完整标题，逐项唯一对应 225 个现有章节；原先 128 项中文标点提示是其中一部分。仅改章节地址，保留显示文字、文件路径、目标章节和其余正文。
- 链接 lint 将与完整标题不同的所有 slug 列为兼容待核，覆盖普通编号、英文空格与大小写；34 项回归通过，完整严格检查无错误或待核项，raw 保留说明页仍无点击关系。
- Astra 核对规则和管理导航、AWQ/SAPrune 等代表关系，未发现需删除的无用途链接；本次不代表逐条审过全库知识关系。章节形式依据 Obsidian 官方说明，当前窗口不可用，客户端实际跳转尚未验证。


## 2026-10-12 · HELM 评测协议 ingest 与独立评审

- 研读固定arXiv:2211.09110v2的摘要、§2/3.1–3.2/4/7/11及C.2–C.4/C.6、C.7所选计算、J.1–J.4，回查PDF14–18/27–28/34–35/46/132–133/160–162及表4/7/14。新增HELM协议页，公共质量页补BPB、等量分箱与SCAA/AURC面积条件。
- 区分核心场景覆盖/指标覆盖、固定选例/解码随机性、多选评分/生成解析、有限扰动/部署风险以及原始/去噪/理想计时；保留ECE绝对值、SCAA总体积分方向和默认策略登记差异。未核162页全文、全部专项/排名/引用/污染来源或匹配HELM实现，未访问模型API/运行评测/GPU。
- 独立Astra直接核查28组核心主张、15页原始页图及14处新增知识/管理关系，无P1/P2必修；采纳可选澄清，补明n为评测样本数。两原件SHA与79个实际展开文件逐字节一致；教学检查覆盖510组二值面积、31种等量分箱规模、27组有限扰动、BPB和ECE反例。
- 管理范围记录130条、历史待核58条、队列99条（39论文/48文章文档/12代码），根入口131条。同期章节链接修复后，当前全库170知识页、1541来源、2948本地链接，0错误/0待核；格式与链接通过不替代内容评审或客户端实际跳转。下一项LongBench v2、MMLU-Pro、RULER与harness。

## 2026-10-12 · FlashInfer 分页KV文档 ingest 与独立评审

- 读取三份0.6.14官方文档保存正文/追加例子、布局数学重录与4张补采图；新增分页写入/布局/NVFP4实现页，公共KV页补页表与数值写入职责。区分页/token/mask单位、追加后元数据、NHD/HND、分段mask、MLA/cascade view，以及K/V块尺度/全局恢复因子的方向。
- 保留cascade图指针差异、图片补采时间、V交织/完整wrapper/字节与退化输入协议不足；未取得现行网页或匹配0.6.14代码，未运行FlashInfer/CUDA/SM100。管理范围记录129条、历史待核58条、队列100条（40论文/48文章文档/12代码）；第8批完成，下一项第9批评测协议。
- 独立Astra评审核24组主张及21处新增关系，直接核三正文/例子、数学补充和四图，无P1/P2必修项；采纳mask段内位索引的就近澄清。九份文件指纹通过，教学核页数/末页、跨页地址、120个布局坐标、分段little打包与144字节/全局因子预算；管理129/58/100及130条根入口通过。
- 机械快照169知识页、1539来源、2935本地链接，0错误/128待核；局部五页1039链接为0错误/61待核。未核客户端跳转、存量关系用途或作者执行结果；仅提交本次六文件增量，保留并行迁移和范围外工作。

## 2026-10-12 · raw 无链接存储与论文来源去重

- raw 中 8613 份 Markdown 原文及阅读副本以原名追加 .txt 封存，字节与权限逐文件匹配；29 个原始 Markdown 符号链接的目标文字保留为 .symlink.txt。285 份说明页去掉 3644 处点击，保留来源身份、原始定位和校验记录。raw 原始材料仍按项目约定保存在本地，不进入 Git 分发。
- 89 个 Wiki 页面的 sources 同步 60 条实际存储路径、去掉 158 条同版论文重复载体；不同版本和不同代码实现文件保持独立。同步 2 份研究文档中 33 处旧文件引用，其余入库修改保留。
- 完整机械检查无确定错误，128 处章节跳转兼容提示留待核查；raw 无展开的原文 Markdown，保留说明页无点击关系。未验证客户端图谱显示，也不表述为科学内容审核。

## 2026-10-12 · VidKV ingest 与独立评审

- 研读固定arXiv:2503.16257v2的§3–5、附录A–G，回查PDF1–10、14–18的式1–7、表1–6、图1–7及算法1–3；新增方法页，公共KV页补非整数字母表/FFT坐标预算与旧K重新量化路径。区分量化粒度、保护轴、三值阈值/恢复幅值以及名义/物理位宽。
- 保留FFT公式/算法、V粒度/主附表、保护选择/索引/幅值和打包协议差异；核任务变体、Qwen TempCompass与VATEX损失、FFT/STP消融及测试反馈选γ，不把质量表转成部署字节或实时性能。未取得匹配作者代码、新材料或运行模型/FFT内核/GPU。
- 同步INDEX/READING/SELF，范围记录126条、历史待核58条、队列103条（40论文/51文章文档/12代码）；下一项FlashInfer三个接口及第9批评测协议。
- 独立Astra评审核28组主张及14处新增关系，直接核正式§3–5/附录A–G和上述PDF范围，未发现P1/P2必修项；采纳局部l/l_v/l_t/d定义及V[l,d]澄清。两原件指纹及19个实际展开文件与源包逐字节一致；教学核三组DFT往返/共轭、混合预算、17步token身份、重复量化反例与表数，管理126/58/103及129条根入口通过。
- 并行raw迁移完成后的机械快照：168知识页、1528来源、2915本地链接，0错误/128待核；局部五页1023链接为0错误/61待核。静态结果不代表存量关系、客户端跳转、作者格式或GPU已验证；仅提交本次六文件增量，保留并行迁移与范围外工作。

## 2026-10-12 · CalibQuant ingest 与独立评审

- 研读固定arXiv:2502.14882v2的§3–5及附录A/A.1，回查PDF1–9、11–12的式1–8、表1–8、图1–3、算法1–2；新增方法页，公共Attention页补仿射偏移与V偏置在混合子集归一化中的条件。区分视觉端点码、整数打包、同一量化值的后缩放与logits校准。
- 保留τ/退化区间、混合mask/拼接、范围更新与计分协议不足；核算法字面索引、视频误用DocVQA数、吞吐的批量/型号/比值及单位差异，不修改raw或推断作者实现失败。未取得匹配代码、新材料或运行模型/Triton/GPU。
- 同步INDEX/READING/SELF，范围记录125条、历史待核58条、队列104条（41论文/51文章文档/12代码）；下一项VidKV，再核FlashInfer三个接口及第9批评测协议。
- 独立Astra评审实际核28组主张及12处新增关系，直接读正式§3–5/附录A/A.1和上述PDF范围；发现1项P2公式控制字符，修复并回读闭合，当前无剩余P1/P2。补明τ1=τ2才只平移，以及图3(a) batch64/表5 batch80的差异；未把伪代码缺口推断为实际实现失败。
- 两原件指纹及5个实际展开文件与源包逐字节核对通过；标准库核9组K后缩放、27组V子集质量、48组校准端点、768组字节打包、缓存预算及表数。管理125/58/104、127条根入口与diff检查通过；全库167知识页、1678来源、2936本地链接为0错误/127待核，局部五页993链接为0错误/58待核。静态检查与新增关系评审不代表存量关系、客户端跳转、作者实现或GPU已验证；提交限定本次六文件，保留范围外工作。

## 2026-10-12 · Wiki 正文越界链接清理

- 清理 Wiki 正文 876 处越界点击：来源 URL 改为行内代码，项目根目录 SELF 引用改为文字路径；保留来源标题、版本、URL 与定位信息，sources 登记、原有标题和内部链接保持一致。
- 全 Wiki 链接与结构检查无确定错误；127 处章节跳转兼容提示保留待核。仅完成点击边界清理，尚未逐条审核内部知识关系。

## 2026-10-12 · MASQuant/QIG 解读快照核对与独立评审

- 读取两篇snapshot-2026-06-30解读正文，各5张方法/结果配图，回查MASQuant v1相关方法/表与QIG v1基线/表1–4。机制已有方法页承接；MASQuant补Opt.分组消融与主导模态层/通道条件，QIG补六配置均分约0.48个百分点和W/A/WO基线边界，不新建重复方法页。
- 保留解读文章与原论文/固定代码的证据层级；白化不保证原差异代数秩下降、均分不保证每任务、归因定理/后处理/静态分支条件仍由已有正文承担。受影响方法页来源URL改行内代码，保留标题/版本/定位。未获取现行网页、新材料或运行作者代码/模型/硬件。
- 同步READING/SELF，范围记录124条、历史待核58条、队列105条（42论文/51文章文档/12代码），第7批完成；下一项第8批CalibQuant。
- 独立Astra评审实际核18组主张（MASQuant8/QIG7/来源与关系3），直接读两解读正文、10张研究配图、MASQuant PDF7–8与相关TeX、QIG PDF6–7及方法/基线/表1–4和必要附录B，未发现P1/P2；两处指向/历史核对措辞已澄清并回读。新增6处方法页/LOG关系按用途核读；没有把并行历史链接迁移算成本次语义评审。
- 两文章清单15个文件指纹、两论文4份原件指纹及实际展开13/5文件与源包逐字节核对通过；重算Opt.分组和六配置0.48个百分点。两方法页链接模式23处通过，按表格数据行重核管理124/58/105，根入口125条及本轮diff通过。全库检查当时仍有876越界错误/127锚点待核（166页、1674来源、2931本地链接），并行清理后的结果不由该快照保证。提交仅包含本次五文件增量；未核客户端跳转、现行网页、全部附录/参考文献或重审作者代码、运行模型/GPU。

## 2026-10-12 · VEQ ingest 与独立评审

- 研读固定arXiv:2602.01037v1正文方法/实验、式1–5、表1–3和图1–7，回查PDF1–8；新增方法页，公共校准页补专家覆盖与加权位置。区分AWQ-ME/GPTQ-MA两版本、专家累计误差与均方归约、跨专家耦合与正标量不改独立最优点。
- 解释sqrt(c)输入缩放、非负亲和性、曲率因子2/奇异方向与完整MoE误差代理；核九任务/三任务平均、摘要3.09与表差2.98、TextVQA百分点与相对增幅，保留主校准/梯度/p/软件/评测协议缺口。未取得匹配作者实现、获取新材料或运行量化/模型/SGLang/GPU。
- 同步INDEX/READING/SELF，范围记录122条、历史待核58条、队列107条（42论文/53文章文档/12代码）；下一项MASQuant/QIG解读文章。
- 独立Astra评审核26组关键主张，直接核TeX摘要、引言、方法/实验及PDF1–8的式1–5、表1–3和图1–7，未发现P1/P2必修项；采纳表1 Qwen W4 RTN均分71.01/表列70.89的补充，保留原件。新增16处点击关系按源上下文和目标相关段落核读，未逐条审核存量关系或验证Obsidian实际跳转。
- PDF/eprint指纹及5个实际展开文件与源包逐字节核对通过；标准库教学核专家分数/共享候选、正缩放、12组加权曲率及Hessian因子2、九/三任务均分与增幅。新VEQ页链接模式通过7条链接，管理122/58/107、125条根入口和diff核对通过。按当前新链接规范，全库检查166页、1660来源、2925本地链接，但有884错误/266待核，受影响四个旧页有315错误/58待核，不能记为全库通过；存量外部/越界点击不在本次增量清理范围。参考文献、历史18文件记录和热图逐点未完整复核，未运行作者实现、模型或GPU。

## 2026-10-12 · SmoothQuant+ ingest 与独立评审

- 研读固定arXiv:2312.03788v1正文方法/实验、式1–6、表1–5和图1–7，回查PDF1–11关键布局/主表；新增方法页，公共校准页补评测输入/配方选择与代理loss/任务质量的区别。核通道平滑、双尺度MLP融合、全模型α搜索及数据条件，保留全模型loss定义缺口与更细网格/成绩差异。
- 核作者报告W4A16迁移时量化、主体/元数据/峰值、HumanEval与BabelCode分数、单/双卡吞吐及均值/中位延迟；区分论文描述、教学推导和所选负载证据，保留表间Python、AWQ笼统文字与图示差异。未取得匹配代码/软件pin、获取新材料或运行代码评测、模型/GPU。
- 同步INDEX/READING/SELF，范围记录121条、历史待核58条、队列108条（43论文/53文章文档/12代码）；下一项第7批VEQ。
- 独立Astra评审核26组关键主张，直接核rsc.tex正文方法/实验及原始PDF1–11的公式、主表和图1–7，未发现P1/P2；采纳RTN发生在加载迁移时的就近澄清，明确其与离线平滑的先后关系。
- PDF/eprint指纹及2个展开文件与源包逐字节核对通过；标准库教学核27组MLP融合恒等式、45组通道最大值关系、代理/任务反例、语言/位宽对照及服务统计比例。Wiki机械检查165页、1656来源、2912链接无错误/警告/跳过，管理121/58/108、122条根入口及diff检查通过。未重审历史10文件源包、逐点重建曲线/全部引用或运行作者实现、模型、加载、代码生成/执行及GPU质量/性能测试。 本轮新增来源URL改为行内代码，新增关系按上下文回读；执行该项检查时，旧检查器不支持并行规范更新中的links-only/paths选项，因此该模式未完成，存量外部链接与Obsidian实际跳转仍未核。

## 2026-10-12 · QSLAW 论文与所选实现 ingest 及独立评审

- 研读固定arXiv:2408.03735v1的正文方法/实验、表1–4及图1–6，回查PDF1–8；核固定264dfb081c210a8bab6a3fb5707d827ade7b6802的14个所选文件相关链路。新增方法/实现页，公共QAT页补尺度位置、重算码与STE的区别；保留式8零点、表/正文数字和†含义差异。
- 区分任务监督/重构、组尺度/恢复步长、论文η数据warmup/脚本阶段与学习率warmup、参数数目/实际适配成本，以及fake量化/普通Linear恢复；记录mix比例/索引、尺度正值、文件名/分层结构和中间保存前提。未导入上游、获取新材料、安装、训练或运行模型/GPU。
- 同步INDEX/READING/SELF，范围记录120条、历史待核58条、队列109条（44论文/53文章文档/12代码）；下一项第7批SmoothQuant+。
- 独立Astra评审核26组关键主张，直接回查TeX关键方法/实验/表、PDF第2及4–8页和14文件所选链路；主研读的PDF1–8范围与独立回查范围分别登记。修复1项P2：mix尺度阶段只显式继承projector、dynamic_scale重置为1，不能写成预训练尺度接续；方法/实现页同步并已回读，未发现新增P1/P2。
- PDF/eprint指纹、4个展开源码与源包、ZIP及14个所选代码文件逐字节核对通过，10个Python文件静态语法通过。标准库教学核原式/整数零点、硬导数/STE及尺度位置、M≠T切片和层索引、参数/时间/Memory比例与百分点；Wiki机械检查164页、1652来源、2903链接无错误/警告/跳过，管理120/58/109、119条根入口链接及diff检查通过。未重审历史174文件或运行作者函数/训练/模型/GPU；完整attention/tokenizer/image、NF4/bnb、外部依赖/产物往返与实验复现仍未核。

## 2026-10-12 · SpQR 所选实现 ingest 与独立评审

- 定向读取固定543d10e56f921e5eeb10d5f0aaaccd30b6662d0e的所选Linear校准/逐列补偿、leave-one-out及最终动态保护、跨行二级分组、旧保存/浮点恢复、3/16/16优化转换及CSR/PTCSR融合matvec；容器/测试/计时按必要入口核读，论文只回查method的存储/推理与appendix预算。新增实现页，方法页区分本版本与论文原结果，公共码本页补绝对保护/相加及逻辑/物理预算。
- 核H归一化和U对角/阈值、真实QQ轴及beta命名、保护mask/当前绝对值/稀疏残差、非整数零点的quantize(0)非零条件、64bit字段/重复片段/对齐与4位物理载荷、CSR/PTCSR填充/置换、FP16恢复/FP32累加、逐向量forward、固定设备形状/列索引/同步和计时协议；保留旧产物覆盖与新接口直接相加的静态条件，不推断历史实验失败。
- 同步INDEX/READING/SELF，范围记录118条、历史待核58条、队列111条（45论文/53文章文档/13代码），第6批完成；下一项第7批QSLAW论文/所选实现。未取得新材料或独立Git对象/归档，固定身份/历史885文件依归档登记；未导入/安装/编译/运行上游、模型或GPU，未核完整HF外部加载、全部模型/harness/shape/配置/设备生命周期。
- 独立Astra评审核28组关键主张，直接核18个所选文件的相关链路及论文PDF第8/19页、method.tex:205–247与appendix.tex:168–201，未发现P1/P2必修项；采纳τ字段、m/n维度与公共零码公式三处澄清。所选源码指纹一致；历史885文件/Git对应、完整论文实验表及其余运行范围未重新核验。
- 标准库教学核3072个码的打包/二级恢复、116/128字节预算、四种shape载荷、真实23/填充64条PTCSR、624个置换索引、25组列值字段及覆盖/相加反例，区分序列Gram与token平均、U对角与完整逆对角；13个Python文件静态语法通过，convert_to_hf含当前Python3.9不支持的match语法，继续只读。Wiki机械检查162页、1626来源、2885链接无错误/警告/跳过；管理实际计数118/58/111、115条根入口链接及diff检查通过。这些算例和静态核对没有执行作者函数、模型、CUDA或性能测试。

## 2026-10-12 · QuIP# E8P 所选实现 ingest 与独立评审

- 定向读取固定1d8f873e9a2a8b86b12bb1064c312c5689b77d98的所选Llama/E8P12/rank0/had统计、逆序BlockLDLQ及buffer/微调、模式/符号/平移与tile打包、分层/HF保存加载及单行CUDA、多行FP16恢复；必要Hadamard、评速与旧测试入口按具体范围核读。新增实现页，既有方法页补固定执行边界，公共码本页补逻辑自由度/交错容器/物理字节及临时内存。
- 区分保存协方差/未中心化重构H、默认段数/论文配置、浮点参考前缀/量化前缀、离散坐标更新/块输出微调、逻辑16bit码/int64四码容器、主体预算/变换和共享底表、单行融合/多行展开/训练缓存、FP16操作数/FP32累加与原子汇合，以及默认无cache重复输入/真实生成；保留纯2幂train_mode的None前提、rank0断言与shape/版本限制。
- 同步INDEX/READING/SELF，范围记录117条、历史待核58条、队列112条（45论文/53文章文档/14代码）；下一项第6批SpQR所选实现。未取得新材料/独立Git对象或归档，版本与历史60文件完整性依归档记录；未导入/安装/编译/运行上游、模型/GPU或性能复现。
- 独立Astra评审核26组关键主张，直接回查23个登记文件的相关链路，未发现P1/P2必修项；采纳符号奇偶条件、反馈公式的变换/归一化坐标及计时循环平均三处澄清。matmul_had只核1–125的必要入口/形状，model/llama只核所选模块和调用，全部常量、其他attention/多GPU/shape/3–4bit及完整e2e训练继续保留未核范围。
- 18个所选文件保存本地SHA指纹，未与新Git对象比字节；标准库教学核全部65,536逻辑码及FP16位模板恢复、6272个码的四种shape打包往返、奇偶交错、主体预算、协方差/未中心化矩与欧氏/加权距离反例，以及train cache/rank的静态条件。Wiki机械检查161页、1596来源、2877链接无错误/警告；管理实际计数117/58/112、112条根入口链接及diff检查通过。这些核对未导入或执行作者函数、模型或CUDA，也不验证设备同步/越界或性能。

## 2026-10-12 · QuIP 所选实现 ingest 与独立评审

- 定向读取固定ac92cfc7a22f6100009e2caf53bb72257d3f3184的OPT校准/评测/保存、QuantMethod预处理/恢复/统计、Balance、qfn网格及LDLQ/LDLQ_RG/lazy，等价脚本只核条件；新增实现页，既有方法页区分压缩推理形式与所选浮点保存，公共旋转页补恢复位置和成对误差坐标。
- 核序列数归一化Gram、预处理顺序/阻尼、meta flag字段、四点网格/中点舍入/裁剪、逆序及排序/贪心/lazy条件、FP16返回、默认state_dict与注释pack3，以及postproc后局部残差和H不在同坐标的静态诊断；不把代数等价或文件存在称为模型/压缩/性能验收。
- 同步INDEX/READING/SELF，范围记录116条、历史待核58条、队列113条（45论文/53文章文档/15代码）；下一项第6批QuIP#、SpQR所选实现。未获取新材料、导入/安装/运行上游、编译或执行模型/GPU，未核全部其他模型/方法/legacy 3bit分支。
- 独立Astra评审核26组关键主张，修复2项P2：LDLQ_RG与lazy组合的位置实参错位，以及贪心轮末裁剪后存储残差未同步；同步方法页不保证有限网格迭代单调下降，补所选qfn=b的FP16返回范围。评审直接核10个所选文件相关链路，opt未核benchmark设备函数、vector_balance未核allbal/ADMM，历史论文完整表格、其他分支和运行仍未核。
- ZIP原件SHA与10个所选文件逐字节核对通过；标准库教学核32组坐标/度量、混合坐标反例、恢复网格、逆序LDL恒等式与中点规则、9个Python文件静态语法、6组lazy实参绑定和贪心裁剪残差反例。Wiki机械检查160页、1564来源、2867链接无错误/警告；管理实际计数116/58/113、109条根入口链接及diff检查通过。这些核对未导入或执行作者函数、模型或GPU。

## 2026-10-12 · FluxBin ingest 与独立评审

- 研读固定arXiv:2608.15602v1的行列二值机制、显著列残差、优化证明/算法1–3及必要质量/成本/执行附录，回查PDF方法、主表、证明、预算与代表附表；新增方法页，公共码本页补输入符号LUT、组尺度、构表摊销与混合分支计数。
- 区分固定尺度符号最优/联合非凸最优、ε稳定/无正则精确解、显著性代理/实际量化扰动、主体覆盖/残差叠加、名义基底数/完整平均位宽、读取数/硬件速度、固定block/batch共享内存、峰值/理论载荷，以及不同配置的质量/速度/能耗；保留反馈伪代码与表间协议/数据差异。未取得新材料、核作者实现或运行模型/GPU。
- 同步INDEX/READING/SELF，范围记录115条、历史待核58条、队列114条（45论文/53文章文档/16代码）；下一项第6批QuIP、QuIP#、SpQR所选实现。
- 独立Astra评审核26组关键主张，修复1项P2：分清相对读取项数比例和其倒数的减少倍数；已回读方法与公共段落。另补地址位及block/tile符号定义。评审核正文§3–4、必要附录A–E、算法1–3及代表表1–9/12，回查PDF1/3–8/11–15/17–20/24；其余曲线、完整tile扫参和作者实现仍未核。
- 2个原件SHA及22个展开文件逐字节核对通过；教学核28个固定尺度符号、62个LUT图案、36个分组输出、12个预算、5组表均分及ε/代理反例。Wiki机械检查159页、1546来源、2858链接无错误/警告；管理实际计数115/58/114、106条根入口链接及diff检查通过。这些检查未执行作者模型、GPU或性能复现。

## 2026-10-12 · SharQ 论文/代码 ingest 与独立评审

- 研读固定arXiv:2606.26587v1正文方法/理论/实验、表1–5、图2–4与必要格式附录，回查PDF关键页；核固定3cc584fe48e32e59c8b4e37323470dcdd606bb11的所选NVFP4/Llama、SHARQ_SIM、准备/共享权重、GEMM host/epilogue与vLLM导出/默认计时，匹配CUTLASS只核必要类型/转换。新增方法/实现页，公共执行页补主干量化后残差和共同/不同路径权重的误差分解。
- 区分pair maxabs/标量或能量最优、残差恒等式/固定权重界/原浮点权重误差、共享载荷/尺度解码、4.75/7.25位逻辑预算/物理对齐、BF16中间输出/FP32累加、融合接口/多个kernel、默认extra_fusion=False/类默认True、质量模拟/native和导出/完整加载；保留原文E2M1 NaN、图/正文数字和prefill/端到端边界。未取得新材料或执行上游/模型/GPU。
- 同步INDEX/READING/SELF，范围记录114条、历史待核58条、队列115条（46论文/53文章文档/16代码）；第5批完成，下一项第6批FluxBin，再接QuIP/QuIP#/SpQR所选实现。
- 固定PDF/eprint哈希及6个实际展开源文件、15个主仓文件与ZIP字节核对通过；三份匹配CUTLASS文件只核本地SHA，pin依历史归档记录，本轮无子Git对象/独立归档可重新比字节。标准库教学核算通过pair maxabs/平方和反例、12组共同权重恒等式及条件范数界、不等权重分解、16个有限E2M1编码、中点舍入差、压缩载荷/metadata、共享尺度视图、4.75/7.25位预算及主表差距恢复/百分点、prefill和局部速度比例。全库机械检查158知识页、1530来源、2849本地链接通过，无错误/警告/跳过；根目录104链接/锚点与114/58/115实际记录通过。上述检查不表示作者代码、模型/GPU或性能复现。
- 独立内容评审直接核24组关键主张、固定论文/代表图表与所选准备、共享权重、GEMM/fusion、模拟和导出接口，未发现P1/P2必修项；采纳projection权重分别量化与公共范数定义的两项可选澄清。全部vendor设备主循环/同步、任意shape/模型/专家/VLM/视频、HiF4/MXFP4 native及完整外部vLLM加载/服务继续保留未核范围。

## 2026-10-12 · Ideogram INT8/GGUF ingest 与独立评审

- 研读固定arXiv:2606.12280v2的正文方法/协议、表1–5及代表图，回查PDF1–9与源文件；对照既有llama.cpp固定b820cc8e6f94843d32f92c8ebd7db837dae2bd8b的Q4_K/Q8_0结构、参考编码和类型/量化入口。新增方法页，公共评测页补生成质量/参考保真度/OCR和共享输入/配对统计。
- 区分两DiT分支/完整管线、FP8 checkpoint/BF16计算、动态网格/离线缩放、raw提示保护消融/最终JSON配方、点估计/配对CI/正式等价、名义码宽/完整GGUF参数、磁盘权重/峰值显存/驻留延迟；保留PDF/公开标题和原文Q4_K encoder/Q8_0无损说法的条件/版本差异。未取得作者实现或新材料，未生成图像/运行模型或GPU。
- 同步INDEX/READING/SELF，新增论文范围记录并将既有llama.cpp的本轮三文件GGUF范围从历史待核移入明确记录，保留其余部署范围未核；范围记录112条、历史待核58条、队列117条（47论文/53文章文档/17代码）；下一项SharQ论文与所选实现。
- 固定PDF/eprint哈希及4个实际展开源文件、既有llama.cpp三文件与ZIP字节核对通过；标准库教学核算通过对角缩放恒等式、Q4_K/Q8_0的4.5/8.5位预算、Q8舍入反例、配对/独立重采样单位差异、CI含零/等价界限反例及主表差值/比例。全库机械检查156知识页、1502来源、2833本地链接通过，无错误/警告/跳过；根目录102链接/锚点与112/58/117实际记录通过。上述检查不表示作者代码、模型/GPU或性能复现。
- 独立内容评审直接回查20组关键主张、paper.tex 1–483、PDF第1/3–9页及GGML三文件相关结构/编码/恢复/类型入口，未发现P1/P2必修项；采纳公共评测页中固定prompt评分和LPIPS双图接口的可选澄清。保护名单、JSON消融归因、实际融合/GPU/峰值显存及完整单卡管线继续保留未核范围。

## 2026-10-12 · FlexQ 论文/代码 ingest 与独立评审

- 研读FlexQ v2正文、式1–6/表1–6及代表布局/性能图；核所选a42424d843adffebe4b28ddb62a760cb52b0d2f0的Llama浮点模拟、engine signed位平面/BMMA/组尺度恢复、默认检查与FasterTransformer wrapper/FFN调用。新增方法/实现页，公共执行页补signed位权与二值点积。
- 区分免外部校准/当前输入范围、W6Ax/down A8、unsigned展开/signed二补码、原式6缺项/代码位权、FP16尺度乘积/FP32归约、理想载荷/对齐复制、模拟质量/原生部署、局部kernel/端到端与不同硬件协议；未取得新材料或运行上游/模型/GPU。
- 同步INDEX/READING/SELF，范围记录110条、历史待核59条、队列118条（48论文/53文章文档/17代码）；下一项Ideogram INT8/GGUF，再接SharQ。
- 固定PDF/eprint哈希及11个实际展开文件、20个所选仓库文件与ZIP字节核对通过；标准库教学核算通过20544个signed乘积、12组位布局往返、位权缺项/分组恢复反例、FP16尺度乘积差异、6.125位理想预算及代表表格均分/PPL、绝对6.5504门限和局部GEMV比例。全库机械检查155知识页、1495来源、2819本地链接通过，无错误/警告/跳过；根目录100链接/锚点与110/59/118实际记录通过。上述检查不表示作者代码、模型/GPU或性能复现。
- 独立内容评审直接回查20组关键主张、所选20文件相关入口与PDF第1/6–15页，发现方法/公共公式未定义q/p位宽及a/b向量坐标的一项P2；修订经实际正文回读闭合，当前范围无剩余P1/P2必修项。编译/GPU数值与同步、全部shape/硬件、完整权重导出/加载与框架融合继续保留未核范围。

## 2026-10-12 · Atom 论文/代码 ingest 与独立评审

- 研读Atom v3正文设计/分析、表1–4和代表性能图，回查PDF方法/主表/消融/服务与格式页；所选固定7e3618b1a7a7c86e1c93cc909b1510c046d76ac6的Llama校准/重排/GPTQ/浮点模拟、native混合GEMM、输出量化及paged KV writer/RoPE reader、默认效率脚本。新增方法/实现页，公共执行页补group恢复与有限精度边界。
- 区分固定索引/在线重排、平方能量/任务Hessian、INT4/INT8表示、实数偏移/整数零点、FP16尺度乘积/FP32跨组累加、浮点模拟/物理缓存、质量实验/随机效率模型及不同服务协议。未取得新材料或运行上游、模型/GPU。
- 同步INDEX/READING/SELF，范围记录108条、历史待核59条、队列120条（49论文/53文章文档/18代码）；下一项第5批FlexQ及所选实现。
- 固定PDF/eprint哈希及21个实际展开文件、24个仓库文件与Git blob字节核对通过；标准库教学核算通过配对置换/逐点MLP、分组恢复反例、条件4.25位预算、signed GEMM/unsigned KV、整数零点/实数偏移、FP16尺度乘积差异、native输出abs范围与原signed编码反例，以及代表均分/百分点/PPL和顺序消融比例。全库机械检查153知识页、1465来源、2805本地链接通过，无错误/警告/跳过；根目录98链接/锚点与108/59/120实际记录通过。上述检查不表示作者代码、模型/GPU或性能复现。
- 独立内容评审直接回查20组关键主张及PDF第1/3/5–11页，发现GPTQ普通权重网格与native KV输出abs统计/signed编码两项P2；修订均经源码和实际正文回读闭合，当前范围无剩余P1/P2必修项。编译运行、GPU数值/同步、任意形状与全部vendor/其他模型分支、真实权重导出和完整服务曲线继续保留未核范围。

## 2026-10-12 · ResQ 论文/代码 ingest 与独立评审

- 研读ResQ `arXiv:2412.14363v2` 正文与必要证明/实现附录，回查PDF关键方法/主表/消融/证明页；所选固定 `cb87d0d059455858daaa66f7410da4afebee978a` 的Llama/full_shared基底、融合/重排、GPTQ与量化器、post-RoPE QK及eager浮点缓存。新增方法/实现页，公共旋转页补平方和、未中心化能量/PCA代理与组内旋转边界。
- 修正不同形状分组的拼接表达、Gram方向与范数关系，保留定理4.2原证明的trace/范数相减/期望/元素计数问题；不把能量代理最优冒充实际量化最优。分开脚本1B/W16与论文W4配置、浮点模拟与packed存储、单block性能与整网生成；没有取得新材料、运行上游或模型/GPU。
- 同步INDEX/READING/SELF，范围记录106条、历史待核59条、队列122条（50论文/53文章文档/19代码）。第4批完成，下一项第5批Atom及所选实现。
- 固定PDF/eprint哈希及20个实际展开文件、43个仓库文件与ZIP/CRC字节核对通过；标准库教学核算通过两路乘积/平方误差/组内能量、范数反例、未中心化/中心化Gram、9个能量代理恒等式与系数方向、成对列重排、4.5位名义预算及PPL/百分点比例。全库机械检查151知识页、1429来源、2786本地链接通过，无错误/警告/跳过；根目录96链接/锚点与106/59/122实际记录通过。上述检查不表示作者代码、模型/GPU或性能复现。
- 独立内容评审直接回查20组关键主张、PDF第1/3–8/13–14页及定向附表，发现共享融合调度名、pretraining_tp=1的wrapper消费条件、SELF队列摘要3项P2；修订均经评审回读闭合，当前范围无剩余P1/P2必修项。全部模型/训练、其他attention或TP分支、packed协议与缺失CUTLASS性能实现继续保留未核范围。

## 2026-10-12 · LLM-MQ ingest 与独立评审

- 研读LLM-MQ固定SHA-256 `9a2101ae6befaff5566fc1ed1bd78cbf10ccb4a7e2c3ae2b985e9c3845562dc0` 的全部5页Workshop PDF，核式1–5/图1–2/表1和GEMV/端到端条件。新增方法页，公共曲率页补候选梯度投影、联合一阶三角界、条件余项及排序/鞍点反例。
- 保留一阶展开方向/绝对评分、稀疏值载荷/CSR及元数据、名义位宽/完整预算、单GEMV/INT4端到端、正文百分比/表内百分点。现有文件缺其他模型附录，未取得新材料、核作者实现或运行模型/GPU。
- 同步INDEX/READING/SELF，范围记录104条、历史待核59条、队列124条（51论文/53文章文档/20代码）。下一项第4批ResQ及所选实现，补子空间混合精度与部署路径。
- 固定原件PDF哈希核对通过；标准库教学核算通过方向、内积抵消/排序、4组条件余项与3组鞍点方向，核2/3位载荷、保护值/假设索引/元数据和名义整数预算例；复核六组任务均分、误差/PPL比例及百分点。全库机械检查149知识页、1404来源、2769本地链接通过，无错误/警告/跳过；根目录94链接/锚点与104/59/124实际记录通过。紧密3-bit字节序例只验证整理者计数，不证明作者packing接口；未执行上游或模型/GPU。
- 独立内容评审回查全部五页原件和16组关键主张，发现公共余项界缺梯度对象/联合扰动与范数定义1项P2；补齐后评审回读确认闭合，当前无剩余P1/P2必修项。缺失附录、作者梯度/求解和真实保护/packing接口继续保留未核边界。

## 2026-10-12 · CoopQ ingest 与独立评审

- 研读CoopQ `arXiv:2509.15455v2` 正文SPQE/二次代理/MILP、代表表1–3/图1–2及必要预算/基线附录，回查PDF方法、主表、消融和附录页。新增方法页，公共混合精度页补联合状态的边际量、协方差反例和二进制Shapley代数。
- 保留NLL方向/最小化链、协方差形状/真实交互、Taylor系数/对角、升级/总预算与区间PPL口径；分开Quanto估计/其他后端使用和18小时离线成本，不将原文未澄清处判为实际代码缺陷。未取得新材料或运行作者实现/模型/GPU。
- 同步INDEX/READING/SELF，范围记录103条、历史待核59条、队列125条（52论文/53文章文档/20代码）。下一项第4批LLM-MQ，补候选量化扰动与梯度评分。
- 固定PDF/eprint哈希及4个实际展开源文件字节核对通过；标准库教学核算通过两排列边际/协方差反例、三玩家6排列的Shapley关系、8个二进制配置的线性化等价、4个乘积真值例及符号/预算例，另核三组PPL比例与1.05个百分点差异。全库机械检查148知识页、1402来源、2757本地链接通过，无错误/警告/跳过；根目录93链接/锚点和103/59/125实际记录通过。上述验证未执行作者代码或模型/GPU。
- 独立内容评审回查16组关键主张、原始PDF第1/3–7/11页与公共教学代数，未发现P1/P2必修项；采纳单位向量及降精度前状态的可选澄清后，评审回读确认当前范围闭合。实际符号转换/对角处理、预算扣基线和求解证书继续保留未核边界。

## 2026-10-11 · CMPQ ingest 与独立评审

- 研读CMPQ `arXiv:2410.13056v4` 正文、算法1、关键实验与必要附录A–D，回查输入通道分配、双类保护/重聚类和预算/部署图表。新增方法页，公共混合精度页补有限通道整数名额、分位并列/严格比较与端点反例。
- 保留名义位宽/完整存储、激活范数/候选边际收益、原值覆盖/残差叠加；明确0.45%+0.05%主配置与附录0.5%文字差异、基线反例/不同预算与所选CSR/拆分部署。未核作者实现，未运行模型/GPU。
- 同步INDEX/READING/SELF，范围记录102条、历史待核59条、队列126条（53论文/53文章文档/20代码）。下一项第4批CoopQ，补层间交互与合作评分。
- 固定PDF/eprint哈希及6个实际展开源文件字节通过核对；标准库教学核算通过4个并列/端点计数例、对角与相关输入的2个输出误差例、3个元数据/保护预算及PPL/延迟/峰值比例。全库机械检查147知识页、1398来源、2749本地链接通过，无错误/警告/跳过；根目录92链接/锚点与102/59/126实际记录通过。教学分位例使用明确的线性插值，未执行作者代码或模型/GPU。
- 独立内容评审核验16组关键主张、PDF关键页及公共预算解释，发现概述将3-bit初始化写成普遍最终结果1项P2；修正后评审回读确认闭合，当前无剩余P1/P2必修项。并列/端点的真实处理、保护mask/交叠、打包与LUT量化接口仍待作者实现核查。

## 2026-10-11 · Norm Tweaking ingest 与独立评审

- 研读Norm Tweaking `arXiv:2309.02784v2` 正文、算法1/式1–3、表1–9及必要附录表10–11，回查PDF方法和结果。新增方法页；公共重构页补逐通道矩匹配、位置/协方差反例，校准页补生成种子先验与混合分布。
- 保留量化前缀上的局部浮点参考、冻结线性权重/输入反传、LayerNorm/RMSNorm、式2记法与未给定统计轴；分开词表计数15.75%/文字17%、有限LAMBADA恢复/任务下降、OPT W5A8表注、OmniQuant未明模型，以及各模型校准时间比例。未取得作者实现或运行模型/GPU。
- 同步INDEX/READING/SELF，范围记录101条、历史待核59条、队列127条（54论文/53文章文档/20代码）。第3批论文已整合，下一项第4批混合精度的CMPQ。
- 固定PDF/eprint哈希与4个实际展开源文件字节通过核对；标准库教学核算通过位置置换/协方差反例、种子混合、词表加和与三组校准时间比例。全库机械检查146知识页、1394来源、2738本地链接通过，无错误/警告/跳过；根目录91链接/锚点及101/59/127实际记录通过。验证不表示作者模型、训练或部署复现。
- 独立内容评审回查16组关键主张及PDF第3–7/10页，确认两处公共解释、来源和管理口径一致；词表加法已在教学核算中修正并经评审回读，当前无剩余P1/P2必修项。新出现的范围外`.obsidian/`保留，不纳入本项提交。

## 2026-10-11 · QUASAR ingest 与独立评审

- 研读 QUASAR `arXiv:2608.13966v1` 正文、算法1、定理1/推论1及必要附录A–D，核每步范围选码/固定码加权拟合、EMA代理、STE梯度差的曲率控制与SGD/PL证明链。新增方法页，公共校准页补编码与恢复参数、实数偏移/整数零点与退化条件，训练页补误差到梯度差的条件界。
- 保留理论SGD/主实验AdamW、条件曲率控制/实际EMA、整数/NVFP4/LUT、模拟训练/作者真实NV检查点评测的边界；主表与消融数值分别登记，INT3最低KL不代表最高任务均分，NV表内提升0.5/0.8与正文0.5–2.0差异保留。没有取得作者实现或运行训练/模型/GPU。
- 同步INDEX/READING/SELF，范围记录100条、历史待核59条、队列128条（55论文/53文章文档/20代码）。下一项第3批Norm Tweaking。
- 固定PDF/eprint哈希、18个实际展开文件的源包字节通过核对；标准库教学核算通过加权拟合/正规方程与3个最优性算例、15个范围候选、3.125位LUT预算、5个二次目标梯度差界，以及总步时/重构时间比例。全库机械检查145知识页、1388来源、2721本地链接通过，无错误/警告/跳过；根目录90链接/锚点及100/59/128实际记录通过。验证不表示作者训练、内核或部署复现。
- 独立内容评审回查16组关键主张、必要证明/格式附录及原始PDF页图，核两处公共承接和100/59/128管理口径；未发现P1/P2必修项，当前范围评审闭合。实际EMA时序、退化实现、导出器和硬件支持继续保留为未核范围。

## 2026-10-11 · LC-QAT ingest 与独立评审

- 研读 LC-QAT `arXiv:2606.10531v1` 正文、式3–12和必要附录A.1–A.6，回查原始PDF关键方法与附表页。新增方法页；公共码本页补共享仿射网格的自由度/预算与输入侧重排，训练页补代理函数/坐标变换和数据预算口径。
- 保留DGE中点发散与多区间数值处理、式5/11坐标接口、指令2B/4B差异。分开4B/30B的QA/PPL结论、额外微调/从头预训练、A100吞吐与A800估算时间，未取得作者实现或运行模型/GPU。
- 同步INDEX/READING/SELF；范围记录99条，历史待核59条，队列129条（56论文/53文章文档/20代码）。下一项第3批QUASAR、Norm Tweaking。
- 固定PDF/eprint哈希与登记一致，7个实际展开文件与源包字节一致；标准库教学核算通过仿射前向重排、均匀网格矩、30个坐标反变换、4处远离中点的DGE导数及逼近中点增长、两组速度比例。全库机械检查144个知识页、1374条来源、2706条本地链接通过，无错误/警告/跳过；根目录89条链接/锚点与99/59/129实际记录核对通过。教学与机械项不表示作者训练或推理被复现。
- 独立内容评审核验16组关键主张、公共承接和管理记录，发现公共码本公式的形状定义缺少1项P2；补矩阵对象/分组轴及输入/码块/广播形状后，评审回读确认闭合，当前范围无剩余P1/P2必修项。

## 2026-10-11 · AQLM 原始校准、转换与所选推理 ingest

- 接续固定 AQLM `e79a896ed6656fe4ed06193d42d004e7d0bbdbb2`：原始 main/AQEngine 的 token 二阶矩、残差初始化、连续更新与激活 beam、当前块目标及量化前缀；新实现页连接条件 PV legacy 转换、HF 有符号码容器和 CUDA 2×8 前向两路读取。
- 公共二阶页补平均矩的归约与按实际 token 数合并；同步 AQLM/PV 关联、INDEX、READING、SELF 与计划执行状态。AQLM 保持同一版本范围记录，队列仍为130条；下一项为LC-QAT。
- 保留 legacy CLI 分支、缺失 args 元数据、逐输出组尺度和转换FP16条件；区分低精度中心加法/FMA、FP32局部归约、低精度输出及输出端尺度，与通用权重端恢复的舍入顺序。CPU/Numba、Triton、Turing和其他格式/适配器仍未完整核读；未安装、编译或运行模型/GPU。
- 独立内容评审核验16组关键主张；修正训练/验证目标范围、总Gram/平均二阶矩符号和推理布局定义3项P2，另纠正PV读取器叙述顺序。评审已回读修订，正文范围内无剩余P1/P2必修项。标准库教学核算通过token加权合并、激活候选误差恒等式与15组有符号码恢复；未执行上游。全库机械检查通过143个知识页、1368条来源、2691条本地链接，无错误、警告或跳过；仅证明相应机械项。

## 2026-10-11 · AQLM PV 实现 ingest 收尾复核

- 对固定 AQLM `e79a896ed6656fe4ed06193d42d004e7d0bbdbb2` 的既有 PV 主链逐项回查：梯度传播、连续中心/尺度更新、绝对权重或 STE 缓冲目标、L2 选码及恢复权重写回；补核原记录中的候选组、trust 前缀、FSDP owner、数据/KL、配置与保存边界，没有扩展原始校准或推理实现范围。
- 实现页补输出/输入组数、块索引和 KL 的 microbatch 符号定义，明确 trust 概述不构成硬上界；READING 将 AQLM 仓库行移至固定版本代码表，保留原 ingest 日期并新增本次复核日期。同步关联页、INDEX 与 SELF；有范围记录仍为98条、历史待核59条、队列130条，下一项仍为同快照原始校准及导出/推理接口，其后 LC-QAT。
- 固定归档 SHA-256 与条目登记一致，ZIP CRC 检查通过，51个来源文件与归档逐文件字节一致。独立标准库核算通过：9组尺度归一化、6组距离展开、54组两码本坐标候选；分阶段变化上界反例、trust 前缀越界及等阈值/全保留边界、候选数量取整、map批次丢尾、名义token/累积步数和教师到学生的位置级KL。另以 AST 只读确认入口未转交两项温度参数；这些算例与静态观察不表示执行了 AQLM。
- 独立 review 子代理直接核对16组关键主张及必要跨页范围，回查固定源码、论文v2算法3与附录K；未发现PV主链的P1内容错误。补齐符号定义、修正来源分类与无法证明的历史review归属后，reviewer已回读实际修订，未发现未解决P1/P2必修项。
- 全库机械检查通过：142个知识页、1351条来源（1330本地/21固定外链）、2665条本地链接，无错误、警告或跳过；检查仅覆盖结构、链接/锚点、来源元数据与导航，不表示科学结论或运行正确性已被全库验证。
- 根目录 SELF/README 的86条本地链接与锚点通过，READING 的98/59/130条实际记录与摘要一致。提交范围限定为实现页和8个必要关联文件；相对于本轮开始时保存的字节基线，208份范围外已跟踪或未忽略文件、51份AQLM来源文件保持不变。
- 本轮仍限静态代码研读与知识整理；没有安装、导入/JIT或执行上游，没有模型训练、分布式通信或GPU实验。原始激活感知校准、格式转换/专用推理kernel、全部加载和分布式组合继续保留为未核范围。

## 2026-10-08 · AQLM PV 实现 ingest

- 核读 Vahe1994/AQLM `e79a896ed6656fe4ed06193d42d004e7d0bbdbb2` 根finetune→PV优化器/Adam→aq/L2选码→恢复权重写回，及数据、KL、FSDP owner、保存的必要依赖；新增实现页，闭合论文Alg3绝对目标接口，分开新中心/尺度时序、STE缓冲与实际trust边界。
- 公共训练页补分阶段变化的基准；校准页补map批次丢尾；关联AQLM/PV方法页并同步INDEX/SELF/READING。范围记录98条，历史待核59条，队列剩130条（57论文/53文章文档/20代码）；下一项接续同快照原始校准及导出/推理接口，随后LC-QAT。
- 保留实际前缀越界、候选组取整、CLI未转交参数、README/论文配置、checkpoint及数据/PPL边界。单rank直接更新和owner完整STE缓冲分别解释，多rank无STE组合未声称等价；未执行上游、导入/JIT、训练、分布式通信或GPU。

## 2026-10-08 · PV-Tuning ingest 与 Astra review

- 研读 arXiv:2405.14852v2 正文、算法1–3与必要附录 A–P，回查关键PDF公式与表1–2/7–9；新增方法页，分开精确P/V、代理上界、稀疏选码与工程Adam/共享梯度/STE混合，保留位预算、训练和推理条件。
- 公共训练页补代理下降链、目标与位移区别、近似选码条件及子空间反例；校准页补文档/块采样的概率与目标差异。同步INDEX/SELF/READING；范围记录97条，历史待核59条，队列剩131条（57论文/53文章文档/21代码）；下一项为已有AQLM快照中的PV优化器。
- 保留A.3等式、Alg3接口、有限smoothness估计、trust ratio与更新比例、原文主附表冲突；不将这些未澄清处判为真实代码缺陷。未执行上游、模型训练或GPU推理。
- 独立 Astra 子代理回查固定来源、教学推导及跨页范围；修正READING论文子队列58→57，并补五任务名称与正曲率条件。核心推导、预算及数据归因未发现其他P1/P2必修项。
- 验证通过：固定PDF/eprint哈希，19个展开源文件及10张另存原图同源包逐字节一致，另19文件只保留源包。独立标准库核算精确P/V、代理平方展开与上界反例、子空间曲率、向量合法性、实际跳变/随机舍入均值、文档/块抽样与位预算；未导入上游。全库141个知识页、1331条来源（1310本地/21固定外链）、2646条本地链接通过；171个数学表达式通过KaTeX，SELF的67条链接/锚点通过。原15个未提交文件与范围外181份已跟踪Wiki/SELF文件哈希不变。

## 2026-10-08 · GPTVQ ingest 与 Astra review

- 研读 arXiv:2402.15319v2 正文、算法1–2及必要附录 A–I，回查关键 PDF 公式与表1–7、13–19；新增方法页，解释多输入列向量、坐标加权E/M、固定索引层级中心更新、INT8/INT4小表、SVD/归一化、全模型FT和LoRA。
- 公共重构页补多列条件矩阵度量、对角代理反例、伪逆中心解与固定索引凸二次条件；码本页补索引宽度/中心dtype/查表容量及共享组成本。补VPTQ版本比较入口，同步 INDEX/SELF/READING；范围记录96条，历史待核59条，队列剩132条（58论文/53文章文档/21代码），下一篇 PV-Tuning。
- 保留原文损失/反馈/因子符号、SVD/归一化公式、组形状与bpv配置差异；分开默认与专门VQ质量协议、附表FT基线、任务LoRA以及主文X Elite端到端与附录单层transfer路径。未取得匹配已归档作者仓库，未执行上游、模型、CPU/GPU性能实验。
- 独立 Astra 子代理直接回查固定来源、公共推导和跨页范围，无P1/P2必修项；采纳将SVD维度写明“右因子转置后相乘”的澄清，Astra已重读确认。
- 验证通过：固定PDF/eprint哈希，31个展开源文件与3张另存原图同源包逐字节一致，另9文件只保留在源包。独立标准库有理数计算核多列条件解/驻点/扰动、子块逆与对角代理、坐标加权中心、固定索引最小二乘/未用中心、排序索引重映射、存储和主表比值；未导入上游。全库140个知识页、1317条来源（1296本地/21固定外链）、2627条本地链接通过；456个数学表达式通过KaTeX，SELF的66条链接/锚点通过。原15个未提交文件与范围外179份已跟踪Wiki/SELF文件哈希不变。

## 2026-10-08 · VPTQ ingest 与 Astra review

- 研读 arXiv:2409.17066v2 正文、算法1–2及必要附录 A–D，回查 PDF 关键公式与表2–3、8–11；新增方法页，解释跨输出行的同列向量、逐列反馈、加权初始化、残差/离群列、layer/end-to-end refinement 及表示和运行成本。
- 公共重构页补同列共同曲率的条件推导、向量方向和加权初始化/选码的不同比较对象；码本页补互斥离群覆盖、重叠残差与分组中心表计账。同步 INDEX/SELF/READING；范围记录95条，历史待核59条，队列剩133条（59论文/53文章文档/21代码），下一篇 GPTVQ。
- 保留原文零变化约束、反馈矩阵/系数、残差回加及离群索引括号问题；区分5.32 PPL中的端到端微调、naive backend吞吐、分阶段解码、不同GPU预算和未核packing。没有匹配的已归档作者仓库，未执行上游程序、模型或GPU。
- 独立 Astra 子代理直接回查固定来源、规范推导及跨页记录；补齐三页完整来源身份，并将码本“四倍”写成总量关系。修正后 Astra 重读确认，无未解决P1/P2必修项。
- 验证通过：固定 PDF/eprint 哈希，33个展开源文件及1张另存原图与源包逐字节一致，另1个 bibliography style 仅保留源包。独立标准库有理数核算覆盖条件二阶解/驻点/自由集更新、向量方向反例、对角/交叉项、加权中心、残差相加及位预算/速度比例；未使用上游代码。全库139个知识页、1308条来源（1287本地/21固定外链）、2609条本地链接通过；361个数学表达式通过KaTeX，SELF的65条链接/锚点通过。原15个未提交文件与范围外179份已跟踪Wiki/SELF文件哈希不变。

## 2026-10-08 · XOR-Trellis ingest 与 Astra review

- 研读 arXiv:2610.00432v1 正文与附录 A–F，回查 PDF 关键公式、表1–3/8–10及图3；新增方法页，解释16位状态/2位转移、四组FP4候选、parity/分支重排、局部最优条件与统计边界、D加权和energy ordering。
- 公共码本页补全局边际/条件候选/路径连接的区别及算例；公共重构页补标量LDL恒等式、反馈残差与block-D对角代理的分层、可加动态规划与正对角近似界。同步 INDEX/SELF/READING；有范围记录94条，历史待核59条不变，队列剩134条（60论文/53文章文档/21代码），下一篇 VPTQ。
- 保留E5M3尺度与标准MXFP4不同、开放路径/packing未核、listing返回数值表索引、RHT后加权/部分energy ordering反例。区分14 XOR2与小LUT的逻辑计数、N3P综合面积和未报告的GPU/端到端吞吐。
- 独立 Astra 子代理直接回查固定来源与公共推导，发现并闭合 SELF 队列摘要滞后一处P2；采纳显式写出组尺度重构的建议，最终无未解决P1/P2。未取得匹配完整作者实现，未执行上游、模型、GPU或电路综合。
- 验证通过：固定PDF/eprint哈希，9个展开源文件与5张另存原图同源包逐字节一致，其余13文件保留于源包。独立标准库计算核65,536状态/262,144后继的数值生成、局部集合/分布/相关与二进制秩；九步固定起点的262,144条路径枚举与欧氏/加权动态规划结果一致，教学LDL/候选/位预算核算通过。这些是有限状态与数学检查，不是作者实现或模型复现。全库138个知识页、1299条来源（1278本地/21固定外链）、2588条本地链接通过；340个数学表达式通过KaTeX，SELF的64条链接/锚点通过。原15个未提交文件和范围外178份已跟踪Wiki/SELF文件哈希不变。

## 2026-10-08 · QuantForge/HiRes ingest 与 Astra review

- 研读 arXiv:2609.34680v1 正文、附录 A/B 算法及 C/D 的必要实验与协议；以 TeX/PDF 核 QuantForge 竞争解释/最低成本判别/结论进入代码，以及 HiRes 的 Block32 几何、anchor误差、受限交互与D接受、GPTQ/单码评分、attention→MLP校正及重测依赖。
- 公共重构页补已编码权重下激活选码的精确局部目标、合法码教学例、跨块交叉项和统计状态；评测页补搜索反馈/独立留出、调用预算/达标时间、Fit-7/Robust Fit与统计单位。同步 INDEX/SELF/READING，范围记录增为93条，历史待核59条不变；队列剩135条（61论文/53文章文档/21代码），下一篇 XOR-Trellis。
- 分开 HiRes 开发的GPTQ/Fit-3/迁移接受信号与独立RTN/WT2/无反馈留出比较；保留基线适配、合法操作数与BF16 decoded实现、每块10304B辅助矩阵、实际存储和完整forward增加的口径。未将格式合法性写成原生MXFP4部署收益。
- 独立 Astra 子代理直接回查固定 TeX/PDF、公共推导和跨页范围，未发现必修项；采纳明确单码公式中的权重偏差项、补全 MLP teacher 目标两处措辞建议。未取得匹配作者实现，未运行上游、搜索、模型或 GPU。
- 验证通过：固定 PDF/eprint 哈希，17个已展开源文件及另存的2个原图 PDF 与源包逐字节一致；源包另含未展开的 bibliography style，未把已展开范围写成整个源包。独立教学核算覆盖配对变换、合法单码精确展开、跨块反例、MLP一阶设计、聚合/配对统计、调用预算、辅助存储与时延。全库137个知识页、1293条来源（1272本地/21固定外链）、2573条本地链接通过；287个数学表达式通过 KaTeX，SELF的63条链接/锚点通过。原15个未提交文件与范围外177份已跟踪Wiki/SELF文件哈希不变。

## 2026-10-08 · SAPrune 论文 ingest 与 Astra review

- 研读 arXiv:2610.05273v1 正文与必要附录 A/B/C，以固定 TeX/PDF 核选层、双路径、算法1–2、复杂度与主要实验。新增 SAPrune 方法页，讲清层深阶段、离线层表与在线集合、未作视觉条件归一化的选层统计、QR 子空间/熵分配和动作/文本 query 响应覆盖。
- 公共视觉 token 页补 attention 总量/集中度/对象变化的分解与反例，以及剪枝层输出后生效的逐层工作量；同步 INDEX/SELF。READING 有范围记录增为92条，59条历史范围待核不变；待核队列剩136条（62论文/53文章文档/21代码），下一篇 QuantForge。
- 保留功能证据聚合和骨干 query 映射等未细化实现边界；默认 SDPA 主表与其他 backend、统一末端预算与不同剪枝深度、模型时延与排除物理执行的机器人流水线分别说明。LIBERO-Plus 的标准 LIBERO 校准迁移协议没有套用 PIVOT-Q 的同任务适配结论。
- 独立 Astra 子代理直接回查原件和公共推导，未发现必修项；采纳校准规模实验的章节/表格定位建议。未取得匹配作者实现，未安装、导入或执行上游，未运行模型、机器人或 GPU。
- 验证通过：固定 PDF/eprint 哈希，展开31个源文件与源包逐字节一致；独立教学核算覆盖 attention 分解/对象换位、投影能量、熵/名额、签名方向、分段工作量与时延口径。全库136个知识页、1281条来源（1260本地/21固定外链）、2555条本地链接通过；116个数学表达式通过 KaTeX，SELF的62条链接/锚点通过。原15个未提交文件与范围外177份已跟踪Wiki/SELF文件哈希不变。

## 2026-10-08 · PIVOT-Q 论文与主恢复链 ingest 和 Astra review

- 研读 arXiv:2610.05745v1 正文与附录 A–F，回查 PDF 关键公式和表1–3；新增 PIVOT-Q 方法页，解释学生闭环状态、同噪声首动作误差、排除当前的未来窗口、段内秩/配额/间隔放宽、损失权重归一化、行为锚点与 LoRA 恢复。
- 固定作者 commit `74d6e1f8994a03eb45aae5c316a743abda2cba15`，限定 GR00T+QuantVLA 主配置/launcher→selector→train，追到 vendored QuantVLA 的评分、可带梯度动作采样、模型接入与 fake quant 后浮点 linear；没有将旧 helper 默认配置或原生 INT4 执行套入此路径。其他模型完整后端仍保留局部缺口。
- 公共校准页补具身状态/时间覆盖与训练预算，评测页补同任务适配、训练/评测 seed、轨迹与配对统计单位；同步 INDEX/SELF。READING 增至91条有范围依据，59条历史范围待核不变；从待核队列移出论文和主代码条目，剩137条（63论文/53文章文档/21代码），不将局部代码研读记作整仓库完成；下一篇 SAPrune。
- 保留论文/代码锚点来源、HoloQ结果、seed口径差异，以及论文7.4%与表内状态比值的差异。16个目标状态不能代替完整teacher评分/五次更新/clean anchor成本；论文成功率为作者报告，未运行模型/机器人/性能。
- 独立 Astra 子代理直接回查论文/源码，所核范围无必修项；采纳百分点单位换算、诊断干预预算与阈值未完整披露的两项措辞建议。
- 验证通过：固定 PDF/eprint/代码压缩包哈希，26个展开论文源文件与源包逐字节一致，14个实际使用的源码文件与parent archive及Git blobs一致；独立教学核算覆盖尾部未来分数、裁剪/归一化权重、间隔放宽、训练状态预算及配对成功率。全库135个知识页、1277条来源（1256本地/21固定外链）、2543条本地链接通过；151个数学表达式通过 KaTeX，SELF的61条链接/锚点通过。原15个未提交文件与范围外175份已跟踪Wiki/SELF文件哈希不变。

## 2026-10-08 · CentriQ 论文 ingest 与 Astra review

- 研读 CentriQ arXiv:2610.06260v1 正文与附录 A–C，以固定 TeX/PDF 核公式1–10、图1–3及表1–7。新增方法页，解释 adaLN 的逐 token 均值、中心化投影与原权重行和旁路、先中心化后 padding、动态分位数/高斯常数、RoDS 固定/可调变量和 IRLS/HQS、零点修正与均值旁路的区别。
- 公共旋转页补输入相关均值的 rank-1 回注，校准页补免校准与在线统计/开发集的区别；INDEX 建立 DiT 路径，SELF 更新画像与下一项。READING 有依据记录增为89条、历史范围待核59条不变；CentriQ 从待核队列移入实际研读，剩139条（64论文/53文章文档/22代码）。
- 独立 Astra 子代理直接回查 TeX 与关键 PDF 页，所核内容无必修项；采纳“首个内层迭代对应 RTN”的措辞建议。保留概率界未逐步重证和无作者源码/运行验证的边界。
- 验证通过：固定 PDF/eprint 哈希、展开22个文件与源包逐字节一致；独立教学核算覆盖列/行分解、padding与偏移不变性、非对称零点修正、4.5bit预算及宽度/延迟比。134个知识页、1252条来源（1231本地/21固定外链）、2526条本地链接通过；265个数学表达式通过 KaTeX，SELF 60条本地链接/锚点通过。原15个未提交文件与范围外174份已跟踪 Wiki/SELF 文件哈希不变。
- 明确代数精确不等于 FP16/BF16 旁路零舍入，移除均值项不等于范围与旋转无关；主表 SVDQuant 为转引，CentriQ 作者本地运行的 SVDQuant GenEval 结果单列；W2A4/W4A2 为模拟量化，A100 W4A4 延迟为作者报告。未取得本轮可核的作者实现，未安装、导入、运行上游或开展模型/GPU/服务实验。

## 2026-10-08 · SAW-INT4 prefill 恢复与全量材料队列

- 按用户要求，改由独立 Astra 子代理 review。先复核上一提交，修正公共 KV 页残留的 SAW“全文研读”断言；补 Hessian 原表 Mean 65.52、Drop −10.41 与整理者相减 −10.12 的冲突，保留原表归属，不修改 raw。
- 接续 fast-rotation `0fcc241961f9c79c27f6bad9a456bf10c8554a84`，核所选 BF16/TP1/GQA4/D128/H128/K-only/FUSE1、page_size=1 普通 causal extend 的 flatten helper：完整已用历史长度、页/slot/逻辑输出、mask、FP32 恢复与 K 逆 FWHT 后 BF16、V 直接恢复、原 Q 的坐标配对，以及 `4096*sum(N_b)` 字节 BF16 工作缓冲。FA3 库内未核。
- 同步两份公共理论、INDEX/READING/SELF。当前 raw 清点283条（147论文、69文章/文档、67代码条目）；将其中尚未在知识页来源字段登记的140条接入现有 READING 待核队列（65/53/22），不把来源登记或归档计为完整 ingest。已有88条有范围依据、59条历史范围待核记录继续保留，口径不同不相加。下一篇为 CentriQ，随后 PIVOT-Q/匹配代码与 SAPrune。
- 独立 Astra 复核确认新增恢复链无必修问题，前两项 P2 已解决；283条清点、140条队列和逐项来源关联经其独立复算一致。
- 验证通过：133个知识页、1245条来源（1224本地/21固定外链）、2509条本地链接；252个数学表达式通过 KaTeX，SELF 59条链接/锚点通过。独立教学核算验证页表到逻辑输出、5 token/20 KiB 缓冲、理想 Hadamard 配对及 Hessian 均值/差值。原15个未提交文件与范围外173份已跟踪 Wiki/SELF 文件哈希不变。
- 新增使用的 flatten 文件已与官方固定 commit 逐字节及 SHA-256 比对；原7文件复用此前官方固定下载，重新核对本地哈希。未安装、导入或执行上游，未运行模型/GPU/服务或质量/显存/性能实验。

## 2026-10-08 · SAW-INT4 writer 与 GQA decode ingest 和 review

- 接续现有SAW方法页，固定parent `e51bfa7291d52cd14b86e4c6ded6c002d0444ff0` 与fast-rotation child `0fcc241961f9c79c27f6bad9a456bf10c8554a84`，限定BF16/TP1/GQA4/D128/Hadamard128/K-only/FUSE1。parent HEAD/gitlink与三份配置/版本文件核实，7个实际使用的child源码与官方固定commit逐字节及SHA-256一致；未把缺失child Git元数据写成独立HEAD核验，也未重验全部既有存档。
- 核K的FP32预缩放/7级FWHT→BF16舍入→全行affine，以及V普通INT4 writer；核通道两半nibble/FP32浮点zero/四buffer与144B完整记录、GQA FP32恢复后BF16、每非空split的Q旋转、P BF16/分母FP32、局部O/LSE与共同空段几何、FP32合并→BF16输出。含元数据为4.5bit/元素、相对BF16约3.56倍，裸码4倍和论文服务数据分别保留。
- 同步KV格式/生命周期、旋转舍入、INDEX/READING/SELF；将已固定的child独立列为来源版本；论文定向复核范围单独确认，历史全文/附录未知不变，88条有依据、59条范围待核实。修正SAW与KIVI均用token粒度的旧概括。FA3 host已确认先写当前chunk再flatten，未套OSCAR dense旁路；设备恢复/Hadamard配对为同配置下一项P2，OnPTQ/WUSH匹配材料仍为P1。
- 三组独立内容review回查producer/论文归因、host/来源/记录及reader/精度/split和相关理论桥。修正作者4×元数据口径、Hessian成本被误作下降原因、KIVI固定容量/粒度概括与历史论文范围记录，补准确源码定位和分段attention锚点；修正后各审查范围无剩余必修。
- 验证：7个child文件与官方固定commit、3个parent文件与Git blobs、2个论文tex与固定eprint逐字节一致。独立教学核算验证A0/F5两半码与恢复、144B/4.5bit/32/9预算、GQA共同地址、Sylvester正交关系、N100/k3的有效split和LSE加权合并。全库133个知识页、1242项来源（1221本地/21固定外链）、2506处本地链接检查通过；七个成果文件的227个数学表达式通过KaTeX，SELF的58条本地链接/锚点与空白/末尾换行通过。原有15个未提交文件及范围外173份已跟踪Wiki/SELF文件保持原哈希。
- 本轮为静态源码ingest；FA3 flatten/device与库内、其他格式/配置/backend/多设备未全面核，没有安装、导入/执行上游、JIT/编译、GPU/模型/服务或质量/显存/性能测量。

## 2026-10-08 · OSCAR 混合缓存源码 ingest 和 review

- 复查OnPTQ/WUSH-KV固定v1与官方入口；本次可读范围未定位匹配作者实现。既有WUSH仓库对应前作W4A4，未当作WUSH-KV代码；OnPTQ的θ/优化器/更新步数与WUSH作者改版仍保留为未核。
- 固定OSCAR `41ebcdba3db5f0ce1339c3727caea80df575d437`，raw保存8166个parent blob与指定simple-evals pin的25个blob；完整树/ZIP/文件系统回读核Git blob SHA-1、SHA-256、路径、类型、6个symlink与218个执行位，8191项全部一致，无额外/缺少项或LFS指针。vendored SGLang为parent内容，raw归档与知识核读分别记账。
- 新增[OSCAR混合KV缓存执行](implementation/oscar-mixed-kv-cache-execution.md)，沿BF16/TP1/GQA/D128/G128、affine INT2、sink64/recent256、FP32元数据、旋转暂存与裁剪/量化/打包kernel分开、V absorption、FA3 prefill框架→Triton decode的选定分支。核通道四分区码/浮点zero、普通prefill排序与flush二分裁剪、独立QP/HP arena/slot及预分配预算、flush/remap/当前HP写入、尺度BF16消费与概率舍入、FP32 O/LSE合并及BF16输出逆旋转。
- 同步WUSH前置入口、KV格式/生命周期、INDEX/READING/SELF，新增一个固定来源记录，86条有依据、60条范围待核实。OSCAR未替代WUSH一般可逆变换实现；P1仍待作者匹配材料，下一条材料已具备的P2为SAW-INT4既有host路径后的选定设备核读。

- 三组独立内容review回查producer/旋转/裁剪、host/布局/窗口与reader/精度/split合并。修正真实函数标识、FA3预旋转归属和融合边界、scratch分配责任、0-based裁剪索引及来源定位；补FA3前缀FP32恢复与GQA decode先转s/z BF16的差别，限定桥页reader范围，修正后无剩余必修问题。
- 验证：22个直接使用的源码文件与固定ZIP逐字节及SHA-256一致；独立教学计算核E4码/还原、80字节/6.4倍记录预算、7步二分阈值、512→520位置守恒和LSE加权合并。全库133个知识页、1226项来源（1205本地/21固定外链）、2496处本地链接检查通过；七个成果文件的154个数学表达式通过KaTeX，SELF的57条本地链接/锚点与空白/末尾换行通过。原有15个未提交文件及范围外173份已跟踪Wiki/SELF文件保持原哈希。本轮静态核读，没有安装、导入/执行上游、JIT/编译、模型/GPU/服务或质量/显存/性能实验。

## 2026-10-07 · W4A16 MoE 专家源码 ingest 和 review

- 新增 [W4A16 MoE 专家执行](implementation/w4a16-moe-expert-execution.md)，连接既有 AWQ 加载、Dense Marlin 和 FP8 MoE 页。固定 vLLM `568afb3a13806beb53bb2e6bd518269357b237c0`，限定 Standard MarlinExperts、BF16/SM80+、非对称 kU4/G128、多组、单设备合法路由、普通 SiLU、无 bias/act-order/A8。
- 核逐专家 AWQ repack 与 Dense 标准化/GPTQ repack 的接口差异、w13/w2 参数与元数据布局、w13 scale 置换传入 I 的实际条件；沿配对 p、expert block/sentinel 前缀、专家 stride 接续整数零点/尺度/FP32 MMA 与两次 GEMM。
- 明确普通 SiLU 的两次 BF16 舍入、down accumulator 和 gate 各自先转 BF16再乘、实际 callback 的 CUDA FP32 top-k sum；两次 GEMM 显式非atomic/FP32跨CTA归约，NoOP finalize不重复加权。区分两组激活cache、INT32锁和每次GEMM的FP32部分和。
- 同步 AWQ/FP8 MoE 承接页、INDEX/READING/SELF；复用同一 vLLM 研读记录，85条有依据、60条范围待核实不变。移除所选 W4A16 MoE 的已解决 P1，下一项 P1 为取得 OnPTQ/WUSH-KV 的匹配实现材料；其他 backend/组长/单组/多设备分支按具体问题展开。
- 三组独立内容 review 回查入口/加载/host、配对/专家地址/恢复、数值舍入/汇合/缓冲与跨页范围。修正 P 外但仍在分配容量内的 expert sentinel 措辞、模型汇合锚点、实际 fused_moe 激活来源登记；补设备恢复/归约定位、来源身份 URL 与浮点前置链接，修正后无剩余必修问题。
- 验证：28个本轮直接使用的 vLLM源码文件与固定ZIP逐字节一致；教学计算核配对/读取映射、专家/组地址和BF16 gate舍入差异。全库132个知识页、1190项来源、2481处本地链接检查通过；七个成果文件中126个数学表达式通过KaTeX，SELF的55条本地链接/锚点及差异空白/换行检查通过。原有15个未提交文件和范围外172份已跟踪Wiki/SELF文件保持原哈希。本轮为静态源码 ingest，没有安装、生成/编译、JIT、GPU、模型、服务或性能实验。

## 2026-10-07 · AWQ Marlin 零点设备源码 ingest 和 review

- 接续既有 AWQ 加载与对称 GPTQ Marlin 主线，在[内核契约页](implementation/weight-only-dequant-kernels.md#11.%20%E9%9D%9E%E5%AF%B9%E7%A7%B0%20AWQ%20%E2%86%92%20vLLM%20Marlin%EF%BC%9A%E6%95%B4%E6%95%B0%E9%9B%B6%E7%82%B9%E6%80%8E%E6%A0%B7%E5%8F%82%E4%B8%8E%E8%AE%A1%E7%AE%97)增补固定 vLLM `568afb3a13806beb53bb2e6bd518269357b237c0` 的 Dense `kU4/G128/A16/no-act-order/SM80+`。核 checkpoint→MPLinear 标准化→Marlin 权重/scale/packed zeros 配套布局→host/template→设备消费，限定本分片 K≥256且按128整除、N按8整除；本范围只补 N 至64倍数，没有重新量化模型。
- 核零点64列置换与 nibble interleave、shared ring 组首复制/复用和 lane→列/尺度对应。FP16/BF16 的 q 与整数 z 分别恢复为1024/128的共同基底，相减抵消后在低精度 B fragment 上乘组尺度，再做 FP32 MMA。区分对称 GPTQ 的固定减8、TinyChat scaled-zero 与未选 float-zp 分支。
- 明确 vLLM atomic 环境开关实际生效且默认关闭、FP32跨CTA缓冲默认开启；bias先转换到输出类型再加一次。保留 K=128单组后乘scale、SM75 FP16多组低精度MMA/二阶段流水、atomic条件与不同舍入顺序，未把这些相邻分支混入 SM80+ 多组主线。
- 同步 AWQ 承接页、INDEX/READING/SELF；沿用同一 vLLM 研读记录，85条有依据、60条范围待核实不变。下一项可执行代码库P1为单设备W4A16 MoE的专家配对/两次GEMM/零点尺度/gate汇合，所需固定来源已存在；OnPTQ/WUSH-KV仍待匹配实现材料。
- 三组独立内容 review 回查加载/host、零点流水/位恢复/公式、尺度/MMA/归约/bias 与跨页范围。修正 checkpoint 零点位序的阶段描述，补 AWQ 页的直接设备来源、repack 坐标/打包定位和 stripe 组对齐前提；修正后复核无剩余必修问题，必要前置由已有正文承接。
- 验证：本轮使用的15个 vLLM源码文件与固定ZIP逐字节一致；独立教学计算核256组 q/z 的共同基底抵消、8个零点列映射word与2个教学word。全库131个知识页、1160项来源、2471处本地链接检查通过，六个成果文件中35个数学表达式通过KaTeX，SELF的55条本地链接/锚点与差异空白/换行检查通过；原有15个未提交文件保持原哈希。本轮没有安装、生成/编译selector、JIT、GPU、模型、服务或性能实验，原有工作区改动保留。

## 2026-10-07 · DeepGEMM 专家设备源码 ingest 和 review

- 接续所选 MoE EP/DeepEP 主线，新增 [DeepGEMM 专家设备执行](implementation/deepgemm-moe-expert-execution.md)。核定 SGLang 默认 `sgl-deep-gemm==0.1.3` 与官方 `v0.1.3` tag commit `b4296edce3cd07e3b6392c8eb390cee9c41a6b70`；每个 PyPI/cu129 x86_64 wheel 的 870 项非binary文件与固定父树/子模块相符。6,923 blob 的完整保存与哈希/类型/执行位检查另记 raw，不把host二进制身份或默认依赖当成实际部署证明。
- 核通定制init/TVM-FFI→host/API→JIT/成本heuristic→SM90 scheduler/TMA/WGMMA→BF16输出。区分tile首行专家选择、128K局部点积后的两侧恢复scale、FP32 final accumulator与RN BF16；B scale由普通global/shared读取，A scale使用TMA布局，转置保留普通FP32数值。
- 澄清contiguous padding ID仍非负，masked_m只控制tile/warp-group起始而非逐行store，expected_m只影响配置估计；默认Mmax128保证容量tile对齐，后续有效gather/activation/combine才限定模型贡献。所匹配包未接收wrapper SBO的overlap/signal参数，明确为静态API差异并保留无overlap范围。
- 三组独立内容review回查版本/FFI/JIT、contiguous布局/计算/尾部、masked计数/SiLU舍入/scale与公式；修正前置链接、表格列结构和gate乘法措辞，补math warp-group定义及N尾部/flatten边界。同步两承接页、INDEX/READING/SELF，研读记录85条有依据、60条范围待核实。
- 代码下一项P1为已有材料的Dense非对称AWQ W4A16现代Marlin零点消费；对称GPTQ主流水已有覆盖，故采用最小增量，OnPTQ/WUSH-KV仍待matched实现材料。本次没有安装、JIT、GPU、模型、服务或性能实验，机械检查与索引验证的范围按实际交付说明。

## 2026-10-07 · DeepEP 通信设备源码 ingest 和 review

- 接续所选 MoE EP 框架链，新增 [DeepEP 通信设备执行](implementation/deepep-moe-communication-execution.md)。分别核定 SGLang 默认 CUDA `9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee`、vLLM 默认 CUDA Docker `73b6ea4a439ba03a695563f9fd242c8e4b02b37c`，与独立工具回退、配方补丁、覆盖版本及实际部署状态分开；两固定源码存档与 73 个 blob 核验另记 raw LOG。
- normal/HT 沿 Python→C++→layout/notify/dispatch/combine 核 rank去重、专家计数alignment与接收R分离、局部ID/非本地gate零、普通FP32 scale复制与行连续返回、int64平面、六项handle/prefix/head/返程和FP32→BF16归约。hierarchical核八GPU分组、目的组去重/bitmap fanout、SourceMeta/int32wire/slot stride、十项handle及NVL/RDMA两段BF16舍入。
- LL核每专家复制、BF16发送端per128动态E4M3量化、FP32 restoring scale、消息长度与capacity reservation、每专家P*Mmax容量/有效segment/列连续scale、五项handle与原topk回源FP32 gate归约→BF16；核peer NVLink/IBGDA条件分支、event/hook SEND/RECV、双缓冲signal clean与Buffer回收入口，保留支持集合、ID编译条件、QPdepth及shrink关闭的边界。
- 同步 [EP框架页](implementation/moe-expert-parallel-execution.md)中的已核外部实现、INDEX/READING/SELF；有依据研读记录增至84条，60条范围待核实不变。代码库下一项为匹配SGLang DeepGEMM contiguous/masked专家设备链；OnPTQ/WUSH-KV仍待实现材料。
- 三组独立内容 review 回查版本/normal/LL/hierarchical 与跨页一致性，修正 normal 返程匹配、LL 分配计费与实际消息步长、旧未核边界与正文定位/引用，并补 normal notify 的 CPU 动态计数等待。全库130个知识页、1109项来源、2449处本地链接检查通过；两受影响正文163个数学表达通过KaTeX，73个raw blob及原有15个脏文件哈希核对通过。
- 本轮为静态源码 ingest 与内容 review；没有安装、编译、GPU/模型/网络/故障或性能实验。NVSHMEM外部库/完整WQE完成与竞态协议、其他格式/拓扑/overlap/graph/V2、专家GEMM设备实现保留未核范围；raw本地保存，原有Skill/研究/其他工作区改动保留。

## 2026-10-07 · DSA 设备源码 ingest 和 review

- 接续 packed/DSA 框架页，补齐 vLLM 默认固定 FlashMLA `a8f794d1251cbfd88a5011445dd5582289c727e4` 与 DeepGEMM `a6b593d2826719dcf4892609af7b84ee23aaf32a`。两份父归档与全部指定子模块共 13,784 个 blob，经上游原始归档、官方完整 Git tree、字节/Git blob SHA-1/类型/执行位/缺失额外项核对一致；raw 留在本地资料层，文件与哈希由对应 README 和 raw LOG 承接。
- 新增 [DSA 设备执行](implementation/dsa-sparse-device-execution.md)，沿 Python/注册/API/JIT/device 核读 SM90、I64/D128/P64/单 query 的 FP8 indexer logits，及 H128/MQA576/512 的 BF16 sparse prefill 与 V32 packed decode。贯通 Qscale 融权重、ReLU/head 归约/Kscale、整页 key/scale/TMA、因果 top-k 边界，主 attention 的 QK/PV、online softmax、LSE、persistent scheduler 和 split combine。
- 澄清所选路径的实际契约：DeepGEMM context lens 是二维、cache 是整页平面，cleanFalse 的合法列由 top-k 保证；FlashMLA prefill length 是列前缀，普通合法单设备 top-k 的前缀条件成立；SM90 packed decode 先将 FP32 scale 转 BF16，与 prefill 恢复的舍入位置不同。负一 sentinel 仍读 dummy 地址但清零 latent/scale、mask score，空行最终 LSE 使用正无穷 sentinel。这些是静态接口与数值顺序，不表示已经复现精度或服务缺陷。
- 三组内容 review 定向回查 DeepGEMM、FlashMLA prefill 和 decode 的核心公式、shape/layout、mask、调度、精度与证据；修正 Kscale 乘法/跨线程归约顺序、反量化来源路径与函数定位、dummy 内容的过强条件，并补因果符号、LSE 移位分母、输出 shape 和 decode CTA cluster/persistent 的解释。
- 更新 [packed/DSA 框架页](implementation/vllm-mla-packed-cache-and-dsa-execution.md)的实际 head/前缀契约及外部设备接续；[MLA 执行](implementation/mla-cache-and-attention-execution.md)、[KV 对象与粒度](theory/kv-cache-quantization-objects-and-granularity.md)补必要阅读连接，INDEX/READING/SELF 同步。研读记录增至 82 条有依据、60 条范围待核实；下一项代码库 P1 为所选 MoE EP 的匹配 DeepEP 通信设备链，OnPTQ/WUSH-KV 仍待实现材料。
- 验证：全库机械检查 129 个知识页、1,063 条来源（1,042 本地、21 固定外链）、2,416 条本地链接通过，无错误/警告/跳过；本轮四个知识页的 409 个数学表达式通过 KaTeX，SELF 的 49 条本地链接/锚点与八个成果文件的空白/换行检查通过。175 份基线中仅改七个既有文件、新增一个预定设备页，其余 168 份哈希一致，原有 15 个未提交文件保持原哈希；提交前另核对完整索引快照。
- 本轮没有安装、JIT/编译、运行上游测试、GPU/model/服务或性能测量；SM100/FP4、MODEL1、其他 DSA/backend、多 query、DCP/PCP 与极端输入未全面展开。保留 raw、Skill、研究材料和原有工作区改动，提交与推送分开。

## 2026-10-07 · vLLM MLA packed 缓存与 DSA 源码 ingest 和 review

- 新增 [vLLM MLA packed 缓存与 DSA](implementation/vllm-mla-packed-cache-and-dsa-execution.md)，接续普通 MLA 与 DCP 页，固定 vLLM commit `568afb3a13806beb53bb2e6bd518269357b237c0`。选择默认 registry 的通用 V3.2、SM90/BF16、单设备、page 64、实际创建 Indexer 的层，核 656-byte 主记录、四组 FP32 scale、BF16 RoPE、CUDA writer 与专用恢复，以及独立 indexer 的整页 key/scale、Q scale/weights、因果 top-k、请求内到物理/workspace 位置和所选 sparse prefill/decode 框架链。
- 核清 132-byte indexer 预算不等于逐 token 连续记录；`scale_fmt=ue8m0` 在所选 indexer 路径约束 scale 数值为二次幂，仍用四字节 FP32 存储。主 packed 的内嵌 scale 与普通 MLA 外部 `_k_scale` 分开，专用恢复不能替换为普通 gather；选中 token 不表示未选历史被删除，prefill 恢复完整 context+query，再按因果 top-k 使用。
- 三组独立内容 review 覆盖格式/预算/来源、indexer/选择与 sparse 读取/路由；主 Agent 复核相关旧页增量、入口、画像与范围。修正测试 reference 与量化接口合写的归因、skip-topk 层的覆盖范围、业务阶段与 metadata 分组、helper 与真实导入 gate、prefill 完整长度及外部源码目录 override 条件；补 op registration、因果区间、常数定义与准确来源/锚点。
- 同步四个相关知识页、INDEX/READING/SELF。SELF 将所选 packed/DSA 框架接口标为已补，代码库下一项为取得匹配 FlashMLA sparse 与 DeepGEMM logits 设备源码。CMake 默认固定的外部 FlashMLA commit 与构建生成 interface 分别说明；当前快照不含该接口文件和库内 kernel，框架调用完成不等于 sparse QK/PV、softmax、invalid index 或 SM100 内部 scale 已核验。
- 验证：全库机械检查通过，128 个知识页、1,027 条来源（1,006 本地、21 固定外链）、2,395 条链接，无错误/警告/跳过；五个受影响知识页的 394 个数学表达式通过 KaTeX，SELF 的 47 条本地链接/锚点和差异空白检查通过。三组 review 的修正点复核闭合，15 份既有未提交文件哈希保持一致。
- 本轮没有安装、编译、运行模型/GPU/服务或测量误差与性能。raw、Skill、研究材料和既有工作区改动保留；提交按本轮明确范围处理，推送另按用户授权执行。

## 2026-10-07 · vLLM 普通 MLA DCP 源码 ingest 与 review

- 接续所选单设备 MLA 和 MoE EP 主线，新增 [vLLM MLA DCP 执行](implementation/vllm-mla-context-parallel-execution.md)，固定 commit `568afb3a13806beb53bb2e6bd518269357b237c0`。沿 Model Runner V1、PCP=1、DP=1、所选普通 BF16/Triton 分支核 group、PD 容量/hash、owner/local slot、有效长度、Q head gather、局部 O/LSE、AG+RS/A2A、value/TP 投影，以及 prefix prefill 的 cache gather/reorg/展开与本地合并。
- 同步旧 MLA 执行、模型基础、KV 容量与两个 vLLM 入口、INDEX/READING/SELF；旧页补 FA4 与 FA3/SM90 同属原生 diff-head-width 的条件。decode 的 LSE 仍随 Q dtype；A2A 保留前面的 Q gather，FP32 bit packing 不恢复先前舍入。prefill 先收集联合历史，再对本地 heads 计算，不套用 decode 的跨 rank O/LSE 归约。
- 三组独立内容 review 分别回查缓存/metadata/prefix、decode/通信/投影与 prefill/库接口，合计覆盖新页全文和相关旧页增量；主 Agent 复核公共容量、导航与画像。确认非默认 interleave 例的真实流程有更早的 local workspace 容量 assert，修正触发顺序与隔离重排算术的归属；另澄清最终逐头 O 与 padding 分支内部 chunk O 宽度，修正后独立复核闭合。其他核心主张未确认需修复的错误，必要前置已有正文承接。
- 保留空分片 Triton O 与 A2A `NaN*0` 静态路径，以及非默认 interleave 的容量/行数条件，不写成已复现服务故障。CPU 教学算术核对 495 组 owner/本地地址/长度关系、12 组 LSE 合并关系（最大绝对差约 `8.9e-16`），另核 chunk 容量反例；没有执行 vLLM 原实现。
- 验证：全库机械检查 127 个知识页、1,003 条来源（982 本地、21 固定外链）、2,377 条链接通过，无错误/警告/跳过；本轮六个受影响知识页的 465 个数学表达式通过 KaTeX，SELF 的 45 条本地链接/锚点、差异空白与末尾换行检查通过。205 份基线中变动限于九个已有文件，196 份哈希不变，仅新增一个预定页面，无范围外新文件或删除。
- 没有安装、编译、运行模型/GPU/服务、测量网络或性能，没有提交或推送；raw、Skill、研究材料与既有工作区修改保留。SELF 下一项代码库 P1 为已有 MLA packed 格式与 DSA 读取链，其他部署分支按具体任务开展。

## 2026-10-07 · MoE 专家并行源码 ingest 与 review

- 接续已完成的单设备 MoE 主线，新增 [MoE 专家并行执行](implementation/moe-expert-parallel-execution.md)，沿同一固定 SGLang/vLLM 版本核读 expert 映射、dispatch、接收侧排列/专家计算与 combine 返回。旧单设备页、MoE 基础、TP 与两个框架入口补必要接续，INDEX/READING/SELF 同步实际覆盖。
- SGLang 固定 SM90、JIT DeepGEMM、普通块 FP8：normal 贯通二维接收行、ep_scatter、两次 grouped GEMM、ep_gather 本地加权与 handle 返回；LL 贯通 expert-batched/masked_m、masked GEMM、原 ID/gate 的 combine 和后置 rho/shared。保留 normal Triton combine 暂禁、expected_m 精确表达式、event/hook 与 Python handle 清除不等于通信完成。
- vLLM 固定 DP=P、TP/PCP/SP=1、均分 linear、普通块 FP8：对照 HT+Triton、LL+batched Triton 与 AG/RS；核 EP group、all2all 条件、global/local 与 round-robin 通信编号、scale 量化时点、gate/rho/handle 归属。普通 FP32 scale 叙述显式关闭全局 E8M0；按实际 CUDA decoder 参数而非 factory 默认值解释 routed scaling。
- 公共增量区分 logical/physical 身份与 global/local 作用域、唯一接收 token/专家配对/目的 rank 计数、通信与权重量化、buffer capacity/padding/训练 dropping。logical payload 仅为有前提的教学记账，不表示已核网络协议。外部 DeepEP/DeepGEMM 内核与独立版本、EPLB/其他 backend、图/多 stream/重叠及真实 GPU/服务结果留在未核范围。
- 两组独立框架内容 review 与一组公共/跨页 review 完成，主动回查选定路径、shape/scale、gate/rho/shared、索引、状态与核心记账。修正两处概括：padding 由计数/mask/有效逆映射限制最终结果，可能仍有额外执行；SGLang normal 物理 scatter 与 vLLM HT 的配对索引间接回读分别说明。同步间接回读的来源行范围，明确 expected_m 为按本 rank 行数估算的提示，并修正一处 LaTeX 标识符下划线。修正后复核无剩余必修问题；结论不覆盖外部库内核或全部源文件分支。
- 验证：全库机械检查通过，126 个知识页、967 条来源（946 本地、21 固定外链）、2363 条本地链接，无错误/警告/跳过；六个受影响知识页的 278 个数学表达式通过 KaTeX，SELF 的 43 条本地链接/锚点和受影响文件空白检查通过。204 份基线中九个已有文件按范围更新，其余 195 份保持原哈希；仅新增一页，无范围外新文件或删除。
- SELF 将所选 EP 框架调用链标记为已补，代码库下一项为已有 vLLM 普通 MLA 的 DCP 分片、局部 attention 与跨 rank 合并；OnPTQ/WUSH-KV 实现待匹配材料，外部通信内核/其他 EP 按具体部署问题展开。上述后续方向尚未启动。本轮没有安装、编译、模型/GPU/服务实测，没有提交或推送；raw、Skill、研究材料和既有工作区改动保留。

## 2026-10-07 · MoE、MLA 与预算表示的源码 ingest 和 review

- 按授权顺序完成 SGLang → FlashInfer → vLLM → DBW → StagQ，再统一内容 review。新增 [MLA 执行](implementation/mla-cache-and-attention-execution.md)、[MoE 执行](implementation/moe-routing-and-quantized-expert-execution.md)、[DBW](methods/dbw.md)、[StagQ](methods/stagq.md) 四页；更新九个已有知识页及 INDEX/READING/SELF/本记录，保留原有工作区内容。
- 第一项 SGLang：MLA 贯通投影吸收、联合 latent/RoPE pool、展开/吸收分派、MHA one-shot/chunked prefix、paged 索引、写入/读取与 dtype；MoE 贯通选择分数/gate、dispatch 包装、配对排序/padding、两次 FP8 块尺度 GEMM 与 routed/shared 汇合。选定普通单设备分支，明确 noaux_tc 与 V2 论文路由、普通 cache cast 与 BMM scale 的区别；顶部 ragged 注释与实际条件按代码核定。
- 第二项独立 FlashInfer v0.6.17：补 plan/run、fa2/fa3 JIT/C++/device、物理页与有效长度、右下对齐 causal、split-KV/base-2 LSE 合并和 fa3 原生 FP8 tile 恢复。与 SGLang 声明的 0.6.12 分开；CPU 教学例子核对不计为库运行。
- 第三项 vLLM 对照：MLA 补模型/wrapper、联合 cache scale/写入、Triton grouped decode/自然 LSE 两阶段、历史 gather/反量化与 FlashAttention prefill 调用；MoE 补 factory/internal gate、no-DP/EP prepare、aligned/naive 配对、融合 activation 量化、workspace/finalize/shared。以实际 CUDA decoder 的缩放参数为准，保留 class 默认值与调用端的区别。
- 第四项 DBW v1 与固定作者代码：补成对 SVD 分量、rank/bit 两级预算、Hamilton/STE、LBQ、分阶段训练、INT32 packing/FP16 恢复/两次 linear；公共预算/表示页提取因子计费与主体宽度。保留论文与 commit 在熵损失阶段、优化器/学习率、bit-plane packing/并行 GEMM 和 trunc 口径的差异；表 19 的加权 Bit 与表尾差异未裁决。review 后补整网参数先升 FP32、BF16 前向与保存 dtype 的边界。
- 第五项 StagQ v1：补前缀约束、dyadic bits/affine 原点、跨精度 pair/floor 拟合、两阶段 guard/固定 side record、load 重打包与论文 kernel。保留 2.35 logical / 2.47 physical bpw、合成权重 GEMV/decoder、格式无关融合与真实权重两位局部数值检查的区别；未取得作者实现。
- 三组独立内容 review 覆盖 MLA、MoE、DBW/StagQ 全文，并回查核心公式、真实调用分支/源码、代表数据/表号、版本与适用条件；主 Agent 复核受影响公共页、导航与画像。确认 StagQ 附录文件名与编译编号错位，按 PDF/main.tex 修正最优性 C、质心 D、配置 B、限制 A，并同步公共预算与 READING，已复核无残留；其余未确认需修复的事实错误。内容审查不等于逐格重算全部附表或验证所有源文件分支。
- 同步研读记录与当前画像：READING 有依据记录增为 80 条，另 60 条保持范围待核实；移除过时的未核 MoE/MLA 实现表述，具体框架范围另列。SELF 移除已贯通的单设备缺口；后续 P1 为材料具备后的 OnPTQ/WUSH-KV 实现，以及代码 EP dispatch/返回/通信，MLA DCP/特殊格式按需求随后展开；VLA/DiT/其他后端为 P2。
- 验证：全库机械检查 125 个知识页、924 条来源（903 本地、21 固定外链）通过，无错误/警告/跳过；本轮 13 个知识页的 657 个数学表达式通过 KaTeX；SELF 的 41 条本地链接/锚点、差异空白与新页换行检查通过。146 份基线中改动限于上述 13 个已有文件，另外 133 份哈希保持一致；仅新增四个预定页面，无范围外新文件或删除。
- 本轮为固定资料的 ingest 与内容审查，没有安装依赖、运行模型/GPU、编译 kernel 或复现质量/性能；raw、Skill、研究材料及既有未提交改动保留，没有提交或推送。未核实现分支与论文/代码差异就近保留，后续方向为建议，尚未启动。

## 2026-10-07 · 量化优化、变换与生成行为逐论文 ingest

- 第一篇完成 RMSE v1：在[层输出重构](theory/layer-reconstruction-second-order-compensation.md#11.%20%E9%87%8D%E6%9E%84%E6%8D%9F%E5%A4%B1%E7%9A%84%E5%B0%BA%E5%BA%A6%E6%80%8E%E6%A0%B7%E5%BD%B1%E5%93%8D%E4%BC%98%E5%8C%96)补齐损失尺度、输出/参数梯度、等大小分组与四种归约、离线流程及实验条件。保留非零残差条件、分组改变目标、Qwen W3 表间变体映射差异和单任务反例。
- 第二篇完成 Scale Sensitivity v1：在[校准页](theory/calibration-and-range-selection.md#10.%20%E6%9E%81%E4%BD%8E%E4%BD%8D%E5%AE%BD%E4%B8%BA%E4%BB%80%E4%B9%88%E6%9B%B4%E4%BE%9D%E8%B5%96%E5%B0%BA%E5%BA%A6%E9%80%89%E6%8B%A9)补齐奇数级网格、高斯 RTN 极限、相对尺度曲率、六种选择规则与 HIP/GPTQ 边界；保留解析尺度在原始权重上的失败，以及预测更准而重构更差的局部对照。
- 第三篇完成 [ConQuR v2](methods/conqur.md)，新增方法解释，并补[公共旋转目标](theory/orthogonal-rotation-and-hadamard-quantization.md#9.%20%E4%BB%8E%E6%BF%80%E6%B4%BB%E5%87%A0%E4%BD%95%E7%9B%AE%E6%A0%87%E5%88%B0%E9%97%AD%E5%BC%8F%E6%97%8B%E8%BD%AC%E6%9B%B4%E6%96%B0)：顶点代理与 Procrustes 子问题、共享/头内旋转、逐批量化感知校准、质量与校准成本的条件。
- 第四篇完成 [WUSH-KV v1](methods/wush-kv.md)，展开双侧统计的闭式可逆变换、GQA 局部代理/精确 Hessian、post-RoPE 位置与双窗口；同步公共误差与缓存页，区分 QuEST 近最优条件、模拟结果和 OSCAR 量化器的 SGLang 评测。
- 第五篇完成 Decision Flips v1，整合至[排序稳定性](theory/quantization-ranking-stability.md#8.%20%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E7%9A%84%E5%88%86%E5%B8%83%E8%B7%9D%E7%A6%BB%E6%80%8E%E6%A0%B7%E5%8F%98%E6%88%90%20token%20%E7%BF%BB%E8%BD%AC)和[评测协议](implementation/model-quality-evaluation.md#9.%20%E8%BE%93%E5%87%BA%E5%88%86%E5%B8%83%E4%B8%80%E8%87%B4%E6%80%A7%E8%A6%81%E5%9B%BA%E5%AE%9A%E5%89%8D%E7%BC%80%E5%92%8C%E6%8C%87%E6%A0%87%E5%8F%A3%E5%BE%84)：局部平方根规律、Jensen 因子、TV 严格界与经验比、留出失败条件及推测解码的接受/成本区别。
- 第六篇完成 [OnPTQ v1](methods/onptq.md)，展开量化策略轨迹、边界损伤、量化分支/浮点评估、context anchor 与刷新；补校准状态选择和多模态目标的公共联系。区分单步共同 continuation 的理论界与风险代理、历史与共同流程的基线、正确性 flip 与 token flip，保留底层参数更新配置缺口。
- 同步 [INDEX](INDEX.md)、[READING](READING.md) 与 SELF（`SELF.md`）：新增三方法入口与六条定向阅读路径，建立[按问题选择量化路线](INDEX.md#从问题进入)的导航表；READING 登记固定版本、实际研读范围及未核查部分，SELF 移除已补齐缺口并更新优先顺序。
- 本批研读六篇正文与必要理论、算法和实验附录，结合固定 TeX 与原始 PDF 页图核对核心公式和代表图表；内容复核覆盖本批新增解释与跨页联系，未逐步重证全部附录证明或逐格重算全部任务附表。
- 验证：全库机械检查通过，121 个知识页、808 条来源（787 本地、21 固定外链），无错误、警告或跳过；11 个受影响知识页的 662 个数学表达式通过 KaTeX，SELF 的 40 条本地链接与锚点通过检查，差异空白检查通过。131 份本批未涉及的基线文件哈希保持一致，没有范围外的新文件或删除。
- 本批新增三方法页、更新八个已有知识页及四个导航/记录/画像文件。原始资料、Skill、研究材料和既有工作区改动保留；未运行模型、作者代码或 GPU，未复现质量和性能，不提交或推送。下一项建议落实为 MoE/MLA 到已归档 vLLM/SGLang 的具体表示与接口，尚未启动。

## 2026-10-05 · 资料研读目录

- 新增 [READING](READING.md)，按确定来源版本记录研读范围、Wiki 承接、尚未核查内容与有依据的日期；从 INDEX 增加入口，README 说明维护职责和历史补录边界。
- 初始索引覆盖当前知识页使用的 127 份本地资料和 3 个固定 commit 外链版本。依据既有 LOG 登记 70 条有明确范围的研读记录；另有 60 条仅确认使用关系，保留为范围与日期待核实，不自动登记完整研读或未读。
- 检查器识别 READING 为根层导航索引，检查元数据、链接与锚点，不将其计为知识页；具体知识页仍核对 arXiv 来源登记。118 个知识页、784 条原有来源的结构、链接、来源与导航检查通过，无错误、警告或跳过。
- 本次整理现有记录，没有重新研读全部来源或改变知识结论；raw 规则与原件保持原状，既有工作区改动保留。

## 2026-10-04 · CUDA/Triton 编写知识与块缩放执行补充、Review

- 新增 [CUDA 编写与接入](implementation/cuda-operator-build-and-integration.md)、[Triton 语言与编译语义](fundamentals/operators/triton-language-and-compilation.md)、[FP4/FP8 块缩放矩阵乘](implementation/block-scaled-matmul-execution.md) 三页，分别承接完整编译/调用链、语言规则与 JIT、格式到执行的固定源码分析。CUDA/Triton 使用相同 `x+1` 推理契约的最小教学例子，复用已有布局、计算模式、GEMM、流水和测量解释。
- 复用 Triton `81a46fa0…`、HF kernels `5c2cf07f…` 与 CUDA 官方快照；为缺少的 PyTorch 编译/注册语义收录两页官方文档，保留 2.14 API 与 2026-10-04 教程快照、获取时间和哈希。没有新增论文或覆盖旧来源。
- 内容 review 覆盖三页全文和关联页本次增量，回查设备/stream、schema 与加载模式、FakeTensor 和梯度职责、类型提升、mask/where、编译特化，以及 FP4/E8M0 编码、K packing、B scale 方向、descriptor 约束、FP32 累加和输出转换。按源码展开 packed scale 的 5D 重排与坐标关系，明确 NVFP4 scale 不能按 E8M0 解码、格式字符串不能代替完整输入契约。
- 区分教程的 NVIDIA 原生硬件入口与 `dot_scaled` 的实验性软件模拟，说明 A100 不满足该 descriptor 教程的入口，不将 FP32 参考或上游误差阈值当作模型质量、硬件算术或本地运行证据。CUDA 教学例子明确普通 ATen 本地构建和推理范围，未宣称跨版本稳定 ABI 或训练支持。
- 同步四个关联页、[INDEX](INDEX.md) 与 SELF（`SELF.md`）；移除已补齐的低精度浮点执行实例待办。下一项优先整理问题到量化路线的阅读入口，其后按具体问题贯通 MoE/MLA 框架接口；backward 和新架构优化随实际算子任务补充。
- 验证：118 个知识页、784 条来源的机械检查通过，无错误、警告或跳过；另核对 SELF 的 33 条本地链接与锚点。8 个 Python 代码块通过语法解析；MXFP4 教学例的 1,024 个 scale 坐标与源码 permute 顺序一致。空白检查通过，12 个已有无关改动文件的哈希保持一致。机械与索引检查不替代内容 review。
- 本次完成知识补充、内容 review、画像和下一步更新，没有编译 CUDA/Triton、运行 GPU、复现性能或纠正论文实验数据；raw 原件、Skill、既有研究材料及其他工作区改动保留，不提交或推送。

## 2026-10-04 · DeepSeek-V2 全流程 ingest 示例与内容 review

- 根据 SELF 与相关正文确认 MoE/MLA 公共结构解释不足，选择已有归档的 DeepSeek-V2 v5。研读核心架构、必要训练与服务条件、相关对齐说明，以及 Appendix B/C/D 的结构公式与 attention 对照；按 PDF 页图核对关键公式和表号，采用同版本 TeX 补足抽取损坏的公式，没有逐项审查全部评测附表。
- 新增 [MLA](fundamentals/model/multi-head-latent-attention.md) 与 [MoE 路由](fundamentals/model/mixture-of-experts-and-routing.md) 两个独立公共基础页，展开联合潜变量、矩阵吸收、解耦 RoPE、缓存预算，以及共享/路由专家、设备限制、平衡与 dropping。连接现有 Attention 误差、排序稳定性、校准与执行知识；为稠密 decoder 和 KV 粒度总论补上适用范围与入口，同步 INDEX。
- 内容 review 覆盖两页全文及关联页本次增量，回查核心公式、配置、机制条件与主要依据。明确 MLA 缓存还包含共享位置 key、逐头概率不能合并、实际归一化不能遗漏；保留 V2 公式没有 TopK 后重归一化、平衡损失不保证严格均匀、训练 dropping 与评测不同的条件。将专家数限定为 MoE 层，分别说明总容量、单 token 计算与批次成本；服务吞吐保留 FP8/平均 6-bit 缓存、更大 batch 与 H800 条件，未改写为单独 MLA 的收益。
- 更新 SELF（`SELF.md`） 的受影响画像，移除已解决的 MoE/MLA 基础缺口，将剩余缺口落实为具体框架接口和低精度浮点执行。下一项仍优先复用既有比较页补路线选择入口；下一次来源研读可选已有 Triton 块缩放矩阵乘教程，无需默认继续增加模型报告。
- 验证：全库机械检查通过，115 个知识页、767 条来源（746 本地、21 固定 commit 外链）、1,418 条本地链接，无错误、警告或跳过；另核对 SELF 的 32 条本地链接与锚点。来源公式与新增推导人工核对通过，受影响文件空白与既有无关改动保留检查通过。机械结果不替代内容 review。
- 边界：本次完成知识写入、内容 review、画像与下一步判断；没有运行模型、GPU 或论文数据纠正实验，也未核读 MoE/MLA 实现代码，不将作者报告表述为本地运行结果。保留 raw、Skill、已有研究材料和其他工作区修改，不提交或推送。

## 2026-10-03 · 项目说明精简

- 复评 README、SELF 和 AGENTS，合并重复说明，压缩来源分类、Skill 适用范围复述及重复建设方向。保留项目入口、来源范围、核心工作边界、十条主线的已有覆盖与不足，以及三项下一步。
- 在 AGENTS 补入知识建设原则：按现状判断缺口与选择资料，ingest 后复评缺口、更新受影响画像与优先顺序，主动提出下一项及理由，在已授权范围内推进。
- 保留论文研读与 review 聚焦核心观点、实验与工程任务按具体目标开展的边界；知识正文、来源与其他已有改动保持原状。
- 检查：正文复评及六份关联文档的 374 条本地链接、章节锚点、表格与空白检查通过。

## 2026-10-03 · SELF 职责说明与知识画像

- 新增根 SELF，统一说明核心目的、知识与来源范围、ingest/review/源码分析等职责、论文实验边界、文件分工、当前主线画像及近期建设方向。明确独立实验与工程任务另定目标，不把论文数据纠正和复现列为默认工作。
- 画像依据全库主题检索与公共基础、机制、比较、代表实现的定向正文核读，区分已有解释、源码分析、实际实现、覆盖不足和暂不展开；没有重新核验全部单篇方法或原始来源。确认后续 vLLM/SGLang 已覆盖分段 prefill 与前缀共享，不沿用旧记录将其列为当前缺失。
- README 保留项目简介与阅读入口；AGENTS、Wiki README、INDEX 接入 SELF。Wiki README 将整体定位交由 SELF，保留页面规范，并要求实质覆盖变化同步维护画像。近期建议集中于问题导向的路线入口、MoE/MLA 必要基础、低精度浮点执行实例；这些建议尚未启动。
- 本轮保留已有知识正文、来源、工作区改动和研究材料，没有修改上游代码或 Skill。
- 验证：复用 wiki-review 检查器的 Markdown 与锚点规则，对 SELF、README、AGENTS、Wiki README、INDEX、LOG 六份文档的 380 条本地链接、目标章节、表格宽度、行尾空白与文件末尾换行完成检查，无错误；受影响已跟踪文件的差异空白检查通过。上述检查不表示全库科学主张重新审核。

## 2026-10-03 · 全面 review 后的五处局部修订

- 按已确认的 review 结果修正五处内容：FP8/MX 页区分尺度共享组与存储块，明确 MXFP4 4.25 bpw 与该 ggml NVFP4 4.5 bpw 的块内账；执行页将原生低精度浮点/块缩放点积纳入辨识范围，分别核对乘法输入、累加与输出，避免把高精度输出等同于解码后的高精度 GEMM。
- NOVA-KV 页为全局 query Gram 恒等式补上未掩码全配对条件，说明 causal prefill 的逐 key 可见度量；可靠性页以严格单调变换反例区分概率校准、排序及阈值决策；KIVI 页将融合表述收紧为减少反量化开销的实现方式，不再宣称所有低比特收益必须融合。
- 同步 AWQ 方法页与均匀量化页的执行路径链接，回查相关原论文、固定 ggml/量化代码以及 Attention 误差页。保留现有有效正文和来源，没有修改上游材料。
- 验证：113 个知识页的机械检查通过，761 条来源、1,385 条本地链接，无错误、警告或跳过；4,474 个数学表达式通过 KaTeX，差异空白检查通过。上述计数对应本次内容修订，模型实验另行记录，不用格式检查替代模型证据。

## 2026-09-29 · 五个公共知识主题补深与 review

- 按已确认的五个主题补齐可复用知识：新增 Attention 量化误差传播一页；深化可逆变换页第 7 节、码本页第 10–12 节、KV 对象页第 11–12 节、误差诊断页第 7 节。优先沿用已有内容，未将五个主题机械拆成五页；INDEX 新增连贯阅读路径。
- Attention 展开固定 query 的误差、因果可见性下逐 key 查询度量、softmax Jacobian 与有限扰动界、Value 方向交互、K/V 交叉项、无偏 logits 的非线性反例及自适应查询边界。数学页区分 Gram、未中心化二阶矩与协方差，说明距离欧氏化、白化、PCA、奇异变换和网格约束。
- 码本页补 Lloyd–Max 条件均值推导、固定码率与熵、Gaussian 信息下界和测度条件、高分辨率中心密度、连续预算的负位宽反例及非负约束。区分恒等式、必要条件、渐近代理与实际可达编码，不沿用球面下界错误。
- 工程部分从整数码贯通位面、元数据、完整页字节、预分配和缓存生命周期，再给出参考编码、打包往返、逐步 token 身份、固定 attention、teacher-forced 模型与性能的分层定位方法。定向回查 Kitty 固定 commit 的布局、量化、模拟、更新及模型接入；保留 dtype、舍入、保护窗口和触发时刻的具体条件。
- 内容 review 覆盖五页本批新增正文与相邻论述，并复用第四批已核验的来源结论，回查必要原文与固定代码。补明 causal prefill 的全局 Gram 是全配对代理；修正逐 token V 量化“不会污染”的过度概括；明确平方范数与 MSE 归约、空区间与码本边界条件，校正来源位置和仓库身份。
- 验证：22 项 CPU 教学检查通过，配套脚本随 Attention 页保存，包括 256 个 2-bit 相邻组合、16 个 4-bit 拆位码和 20 步简化缓存轨迹；五页 419 个数学表达式通过 KaTeX。全库机械检查通过：113 个知识页、760 条来源（739 本地、21 固定 commit 外链）、1,382 条本地链接，无错误、警告或跳过；差异空白检查通过。机械检查不替代内容 review。
- 边界：这是公共知识写入与本批内容 review；没有运行实际模型、码本训练、GPU kernel 或性能测试，也未重新审核全库正文。原始来源、上游代码、Skill、此前批次和用户无关改动保留，不提交或推送。

## 2026-09-29 · 第四批五篇 ingest 与内容 review

- 按顺序完成 QAPruner、TurboQuant、NOVA-KV、AKVQ-VL、Kitty 五篇 ingest，新增五个方法页；研读完整正文，定向核对理论证明、系统协议、必要附录、核心公式与图表。同步 KV 粒度、码本预算、正交旋转、视觉 token 四个已有页及 INDEX，连接剪枝代理、失真目标、保护轴、分组码率与读取布局。
- 正式内容 review 覆盖五个新页全文及四个关联页的本批增量，回查关键结论、数字、原始图表和代码。QAPruner 区分对称评分模拟与非对称 PTQ，保留语义评分、融合系数、特征位置与剪枝/校准顺序缺口；不把大重构误差等同于不可删除，也不把局部 scale 的变化泛化为跨 token 因果。
- TurboQuant v1 的球面 Shannon 下界不能直接使用环境空间微分熵；二维、总 2 bit 的共享随机旋转四区间量化器，对每个固定单位输入的期望 MSE 为 1−8/π²≈0.18943，低于所印 0.25。已保存独立复算脚本，保留无偏残差估计与精确方差，不沿用全部有限维近最优保证；同时核对范数公式、IP 常数上标和 2.25/2.5 位算术差别。
- NOVA-KV 区分精确的满秩查询加权距离、高分辨率分组模型与部署 RMS 归一化；补明实际 V 使用 OSCAR 旋转 SQ、learned sink 的均值偏移、校准域、保护带/码本预算和 chunked prefill。保留短上下文编码开销及低于 BF16 的显著任务差距，不把“未检出差异”写成无损。
- AKVQ-VL 明确 TSA/PSA 的层与模型条件、pivot 检测实现缺口、RoPE 后 WHT 和 V/O 融合，按实际 2/4/16-bit 混合计费。2.46×吞吐来自不同容量上限 batch，未误写成相同批量延迟收益。
- Kitty 定向核对官方固定 commit dfd2c07b 的十个登记来源文件，贯通 Qwen3 接入、按页幅值 TopK、位平面打包、FP16 元数据、QK/softmax/SV 和计时入口。确认默认 25% boost 对应 Pro；完整量化页含元数据为约 2.53125 BPE，另加保护缓冲/页表等。登记模拟与系统在舍入、微小 scale、K 页边界、V 老化上的差别，以及打印 HQQ、实际 quanto 的配置差异，未宣称实际执行或质量影响已验证。
- 验证：15 项 CPU 教学代数/布局检查通过，另独立复算 TurboQuant 有限维反例与位宽算术；9 个受影响知识页的 394 个数学表达式通过 KaTeX。全库机械检查通过：112 个知识页、745 条来源（724 本地、21 固定 commit 外链）、1,356 条本地链接，无错误、警告或跳过；差异空白检查通过。
- 边界：没有运行模型、校准/码本训练、编译或执行 GPU kernel、复现性能和统计检验。NOVA 附录 I 只对与结论边界相关的结果定向核对，未逐格重算全部任务表；Kitty 是登记路径的代码核对，不是整仓库审计。作者报告、整理者推导、静态实现差异和未公开条件在正文分别保留。保留前三批及用户已有改动，不改 raw/上游代码/Skill，不提交或推送。

## 2026-09-29 · 第三批四篇 ingest 与内容 review

- 按既定第三批完成 BOS-Sink Topology、CoRePrune、LayerPos、Same Bit Width, Different Outcomes 的正文与必要附录研读，新增选头图谱机制、两篇视觉剪枝方法、TTS 量化评测四页。机制分别覆盖校准均值与集合边界、带重归一化的集合删除失真、逐层位置策略及跨组件/跨后端评测。
- 正式内容 review 覆盖四个新页全文及五个关联页的本批增量，定向回查指标定义、核心公式、实验条件和主结论，核对所用 PDF 图表与固定代码。补正 BOS-Sink 跨 Query 均匀注意力应使用因果调和均值的口径，区分 Jaccard 与成员保留率；CoRePrune 补明固定 logits、负恢复收益、局部代理与物理剪枝阶段，并区分汇总 prefill、decode 和峰值 GPU 内存收益。
- LayerPos 核对官方固定 commit 的单层扫描、三套骨干脚本与 LLaVA 位置路径。记录 7B 区间上界、InternVL 附加层与论文的差异，以及 13B “全 Sparse”脚本只列前 32 层的限制；说明 RoPE 衰减上界不保证逐项单调，综合 98.2% 不代表定位恢复同样比例。代码差异不自动证明论文表格使用了对应配置。
- TTS 核对 RTN/激活模拟、组件 GPTQ、ONNX MatMulNBits 改写和配对 bootstrap；补明 GPTQ 只作用于捕获统计的 Linear、其他组件内模块仍为 RTN，动态激活模拟不等于真实整数执行。保留不满足预设质量区间的近均值配置，以及质量通过但更慢、速度改善但质量未达标的对照。
- 更新视觉 token 基础、排序稳定性、混合精度分配、模型质量评测、矩阵乘执行五个已有页及 INDEX，连接删除集合/位置、选头迁移、质量等价与真实图覆盖。不改 raw、上游实现或项目 Skill，保留工作区既有修改。
- 验证：11 项 CPU 教学代数/指标检查通过；9 个受影响知识页的 305 个数学表达式通过 KaTeX，差异空白检查通过。全库机械检查通过：107 个知识页、721 条来源（700 本地、21 固定 commit 外链）、1,308 条本地链接，无错误、警告或跳过。机械与教学检查不代替科学主张的来源 review。
- 边界：没有运行 VLM/TTS 模型、生成音频、复现剪枝/校准、执行低比特内核或测量性能；CoRePrune 全部任务附表未逐格重算。代码核对限于所登记路径，不是仓库全审；未公开实现细节、论文/代码配置差别、校准隔离与代理指标限制均在页面保留。本批不构成全库内容审查。

## 2026-09-28 · 第二批七篇 ingest 与内容 review

- 按约定完成 Retrieval Damage、Alignment Illusion、PRQuant、RAMP、FoldQuantVLA、KVQuant、QJL 的正文与必要证明/协议附录研读，新增两篇公共机制页及五篇方法页。内容覆盖排序边界与可靠性分母、主角谱与视觉干预、门控 FFN 残差扩宽、CPU 图切分、共享输入折叠、pre-RoPE/敏感度码本及二值投影分数估计。
- 正式内容 review 覆盖七个新页全文和五个关联旧页的增量，按公式条件、原文页图、实验口径、实现证据与读者路径复核并修正。修正检索校准阈值的顺序统计量方向，区分联合概率与条件风险；补充 PA gap 退化反例和全主角基投影的干预边界；保留 RAMP 正文与质心阈值算法的差异。
- QJL 的精确常数存在可计算反例：正交单位 q/k、独立高斯投影、ε=0.1、δ=0.05 时，原引理要求的 m=542 对应实际失败率约 0.06323。保留无偏性、精确方差与数量级，纠正 softmax 线性化和范数依赖的适用边界；标准库复算脚本与固定结果保存在 QJL 配套材料中。未修改原始论文。
- FoldQuantVLA 定向核对固定 commit 的折叠、GPTQ 统计、INT4 打包/激活量化、CUTLASS 整数 GEMM 与插件调用分支；明确 FWHT 融合与 dense rotation 多调用的差别。PRQuant 未提供实际测速，KVQuant 的百万/千万为容量估算，QJL 的主要计时是单层 attention，均未写成已验证的整模型服务收益。
- 同步模型质量评测、MBQ/VLMQ、混合精度分配、矩阵乘执行、KV 粒度总论与 INDEX；修正逐通道缓存必需在线窗口的过度概括，补齐量化坐标、读取接口及真实位宽的联系。
- 验证：19 项 CPU 代数/合成概率检查通过；另精确复算上述 QJL 反例。12 个受影响知识页的 463 个数学表达式通过 KaTeX，差异空白检查通过。全库结构检查通过：103 个知识页、701 条来源（680 本地、21 固定 commit 外链）、1,277 条本地链接，无错误、警告或跳过；机械检查不替代本轮内容 review。
- 边界：没有执行上游量化、模型质量复现、GPU 编译、性能测量或机器人实验；除 FoldQuantVLA 的定向代码阅读外，其他方法的实现及未公开超参数仍保留为证据缺口。本轮不包含第三批论文，也不构成全库内容审查。

## 2026-09-28 · 八篇论文逐篇 ingest 与内容 review

- 按约定顺序完成 Activation-Guided Compensation、RGSQ、HAWQ-V2、CASA、Predicting Quantization Price、QTIP、SPHQuant、Disaggregated Quantization 的正文及关键附录研读、写入与复核，新增八篇方法主页面。分别解释投影残差、模态双侧度量、平均曲率、方向敏感分配、KL配置定价、trellis编码、球面表示与分阶段执行；关键公式和实验表格按需回查页图，不以摘要代替机制。
- 新增“曲率加权量化误差”公共页，串联Fisher与经验代理、Kronecker顺序、阻尼、约束搬运和标量失真。增补既有九个知识页的相关解释与承接入口，更新INDEX的两条阅读路径和八篇顺序；保留既有内容，不改raw、Skill或用户其他改动。
- 内容审查重点：修正RGSQ列向量化因子顺序并保留白化量化集合与表格矛盾；区分HAWQ-V2的QAT条件、CASA连续定理与实际局部搜索保证、Quantization Price完整度量与实验评分；明确QTIP尾咬合近似、SPHQuant实际2.25位及元数据开销；核清DQ共享/独立权重、mask与梯度路径、ODP借用恢复、LUT计时估计和真实TTFT的不同范围。未解释的原文/代码差异就地标注，不强行统一成已证实结论。
- 定向代码核对：RGSQ固定快照的五个度量/白化/旋转/模拟量化文件；DQ主仓库的dual、frozen_decode与训练循环，以及ODP fork的装入/恢复和context启用条件。阅读范围登记到具体文件与函数，未宣称整仓库审查或运行验证。
- 验证：15项CPU教学数值检查通过，覆盖投影恒等式与秩亏情形、双侧度量/vec、阻尼、L2尺度、trace方向反例、有序组合数、标量失真极值、连续预算、白化、球面投影与位宽算术；另以Viterbi和全部128条合法路径穷举核对QTIP图2例子，均得平方误差和0.0307。这些只验证所列数学实例，不代表原论文模型或kernel复现。
- 全库机械检查通过：96个知识页、678条来源登记（657条本地、21条固定外链）、1220条本地链接，无错误、警告或跳过。受影响文件空白检查通过。公式按原件与推导核对；本轮未完成自动公式渲染检查。
- 边界：没有下载模型、运行校准/微调、部署服务或测GPU性能，也没有提交或推送。性能和模型质量数字仍为作者报告；论文理论条件、实验近似、代码差别及尚缺实现证据保留在各方法页。

## 2026-09-22 · 算子测试工具、性能分析与采集示例

- 深化既有正确性与性能测量页：补充 Compute Sanitizer 四类检查、CUDA Events/PyTorch Timer 的计时边界、CUTLASS Profiler 的职责、采样统计、有状态恢复与结果记录。新增 GPU 算子性能分析页，连接时间线、硬件 sections、Roofline、瓶颈假设与对照实验。
- 用 RMSNorm 和量化 GEMM 组织案例：区别输入二次读取/寄存器保留、工作量与驻留、激活量化/GEMM/完整 Linear，以及 W4 元数据与跨 CTA 归约成本。案例是待在 GPU 执行的实验设计，没有填入虚构性能数据。
- 提供教学采集器和两个 factory：RMSNorm 的 eager FP32/PyTorch 路径、SGLang W8A8 的 quant/gemm/linear；独立 check、bench、trace、torch-profiler 模式，保留样本、环境、测量范围与脚本哈希，拒绝无 CUDA 伪测量和已有输出覆盖。
- 新收录 4 份官方文档条目，共 8 个网页正文快照：Nsight Compute、Nsight Systems、Compute Sanitizer、PyTorch 性能工具；保留获取时间、URL 和文本哈希。复用既有 CUTLASS Profiler 与固定版本框架代码，不改写上游源码。
- 内容复核覆盖两份核心页面全文、采集脚本与两份 factory、相关入口增量。重点核对 NVTX push/pop 过滤、计数器与计时的区别、profile 重放/缓存条件、stateful 语义、整数参考与误差容限；修正 CPU 舍入参考在半整数邻近位置的 FP32 加法问题。
- 验证：9 项 CPU 检查通过，包含 CLI/语法、状态约束、无 CUDA 退出/结果保留、统计与教学预算；6 个受影响知识页的 144 个数学表达式通过 KaTeX；全库结构检查通过：87 个知识页、653 条来源（632 本地、21 固定 commit 外链）、1,157 条本地链接，无错误、警告或跳过。内容判断与机械检查分别完成。
- 范围：本次编辑与 CPU 检查在 Darwin arm64 主机完成；没有运行上游 GPU kernel、两个 CUDA factory、Sanitizer 或 Nsight。已确认后续实验将在具备 NVIDIA GPU 和 CUDA 环境的目标设备上执行；当前编辑主机的硬件条件不限制后续实验环境，CPU 检查不能替代 GPU 正确性与性能实测。同步 INDEX、RMSNorm、W8A8、配置调优和 Roofline 入口。


## 2026-09-22 · W8A8、QoQ 与 GPTQ/Marlin 算子深化及 review

- 新增 W8A8 算子页，连接 SGLang/vLLM 激活量化、整数主循环、尺度/bias visitor 与激活零点校正；补充两框架半整数舍入、全零行尺度和 FP32 输出运算次序的差异。
- 在 QServe/QoQ 既有实现页展开 g128 的线程分工、shared swizzle、寄存器解包、多级流水、CTA 内 INT32 合并和 half2 写回；明确列尾块与短 K 预取条件，不能把 M 方向 guard 或组大小检查当成全部边界验证。
- 在权重反量化页展开现代 SGLang Marlin 的对称 W4 GPTQ：16×64 repack、U4B8 常量、输入置换、组尺度缓存、设备主流水及跨 CTA 归约。核对低层/上层 FP32 reduce 默认值差异和 atomic 选择器的实际分支，保留输出 dtype 临时存储与先转换后缩放的数值边界。
- 同步 GPTQ、SmoothQuant 实现入口与 INDEX。SGLang CUTLASS 依赖按 CMake 指定 commit 定向取得并阅读三份模板文件，以固定 commit 外链登记，没有替换已有不同版本的 raw 仓库；补充 CUDA 12.9.1 libdevice roundf 的小范围原文摘录。
- 内容 review 覆盖新增 W8A8 页全文、QoQ/Marlin 本批新增节及两份入口页增量，按量化→搬运→计算→归约→写回回查具体源函数，区分 CUDA predicate 跳过与 CUTLASS 补零、CTA 内 K 分片与跨 CTA 合并、模板参数与真正启用条件。
- 验证：9 项 NumPy CPU 教学检查通过，脚本与结果保存于 W8A8 页 assets；5 个受影响实现页的 212 个数学表达式通过 KaTeX；全库结构检查通过：86 个知识页、635 条来源（614 本地、21 固定 commit 外链）、1,135 条本地链接，无错误、警告或跳过。没有重跑未改变的上一批教学脚本。
- 边界：未编译或运行上游 GPU 内核、未复现模型加载/精度/性能；CUTLASS 的 SM80 threadblock 主流水已追踪，SM90 TMA/WGMMA、全部 lane iterator、vLLM 各架构主循环及所有 Marlin 数据类型仍不能视为完整核验。上述数值反例是教学参考，不是已复现的模型故障。


## 2026-09-22 · GPTQ、SmoothQuant/W8A8、QServe/QoQ 实现深化与 review

- 按“优先 vLLM/SGLang 有明确消费路径、且已有代码库”的范围，深化 GPTQ、SmoothQuant 两个既有实现页，新增 QServe/QoQ 实现页。全部使用现有固定 commit；更新三个方法页入口、部署映射与 INDEX，不改原始材料。
- GPTQ 补齐 Llama 校准顺序、Hessian 归约、原始 3-bit 保存限制、CUDA GEMV、服务格式、TP 尺度和 SGLang fallback；修正 vLLM 配置别名不等于固定内核、act-order 与推理组排序的关系。SmoothQuant 区分两次校准、浮点 fake quant、真实 OPT INT8 图及服务格式，深入 SGLang Triton/CUTLASS 与 vLLM compressed-tensors 包装。
- QServe/QoQ 连接上游 fake-quant 输入、converter、tile/nibble 与元数据重排、SGLang 加载、两种零点符号、字节恢复及 INT8 MMA；核对原始输入和校正的近似，保留保护范围未被 converter 强制、TP/尾块未验证、Linear 支持不等于整套 KV4 系统等边界。
- 内容 review 覆盖三个实现页全文、部署页和三个方法页的本批增量，按量化/导出/加载/执行顺序回看具体函数、形状、符号与分派条件。修正 SGLang Marlin 自动提升的用户选项条件，避免把配置声明、实际内核支持和模型运行混成同一证据。
- 验证：13 项 NumPy CPU 教学检查通过，三个脚本及固定日期 JSON 结果保存在各实现页 assets；132 个数学表达式通过 KaTeX；全库结构检查通过，85 个知识页、618 条来源（600 条本地、18 条固定 commit 外链）、1,117 条本地链接，无错误/警告/跳过。git diff 空白检查通过；机械检查不代替内容 review。
- 边界：没有执行上游量化器、模型导出/加载、GPU 编译或性能基准。教学检查只验证所列布局与代数条件；本批不是全部方法实现闭环，也不宣称跨框架 checkpoint 已可直接互换。

## 2026-09-22 · OSTQuant 可学习正交与缩放 ingest 和 review

- 核对现有 SpinQuant、正交旋转和尺度机制覆盖，选择既有 raw 的 OSTQuant v1（2501.13987）作为可学习变换路线的下一篇。新增方法页，解释各图位置的学习变量、正交与尺度相消、WOMI 初始化、整网教师目标、KL-Top 与实际执行条件。
- 更新正交旋转机制、对角缩放、SpinQuant 3 个已有知识页及 INDEX。对 QSUR 的零均值正定椭球模型独立推导坐标极值、最优正交方向与白化上界；提供主轴端点反例。补明 RoPE 尺度交换条件，以及损失与参数空间的对照，不改 raw、Skill 或无关知识页。
- 内容 review 覆盖新增页全文和 3 页增量，回查 12 个 PDF 页图，包含核心公式、方法图、主结果、消融、成本与未来扩展。保留原文变换次序、椭球范围/符号、top-k 归一化、学习率与跨表数值差异；区分 QSUR 指标与训练目标、浮点等价与泛化、单 block prefill 与整模型生成。
- 验证：6 项 NumPy float64 CPU 教学检查通过，覆盖变换/逆的顺序和条件数、RMS/RoPE 边界、椭球极值/WOMI/白化、FFN 缩放、部分和与条件 KL 的有限差分梯度，以及报告指标换算。4 个受影响知识页的 301 个数学表达式通过 KaTeX；全库结构检查通过：84 个知识页、579 条来源登记（561 条本地、18 条固定 commit 外链）、1,102 条本地链接，无错误、警告或跳过。脚本和结果保存于 Wiki assets，内容 review 与机械检查分别完成。
- 边界：没有读取官方代码、运行流形优化器/模型校准、验证打包格式或测 GPU 性能。KL-Top 的最终归一化、RoPE/GQA 尺度约束、训练和最终量化算法的衔接仍需固定实现核验；v1 的全节点量化是未来工作，不作为已实现成果。

## 2026-09-22 · LoftQ 量化与低秩联合初始化 ingest 和 review

- 使用既有 raw 的 LoftQ v4（2310.08659），新增 LoftQ 方法页，解释量化主体与残差低秩分支的交替更新、截断 SVD、因子尺度、初始化与任务微调的区别，以及与 QERA 固定主体的激活加权目标的联系。
- 更新 QERA、PTQ/QAT、混合精度分配、量化执行 4 个已有知识页，维护 INDEX 的低秩阅读路径。区分层数位宽标签与完整格式预算、存储保留比例与训练显存、单矩阵 CPU 初始化与模型或推理成本；未修改 raw 或项目 Skill。
- 内容 review 覆盖新增页全文与 4 页增量，回查 13 个 PDF 页图，核对算法、结果、消融和附录成本。保留 NF 简化公式的端点问题、图 3 与表 7 的位宽标注冲突，以及正文与表格的收益差异；明确多轮不应累加历史补偿，量化子步骤近似时不能由 SVD 最优性推出整轮单调下降。
- 验证：6 项 NumPy CPU 教学检查通过，覆盖截断最优性、整轮误差反例、残差替换与迭代配对、初始化梯度、因子与适配器尺度、位宽及存储记账。5 个受影响知识页的 304 个数学表达式通过 KaTeX 解析；结构检查通过：83 个知识页、575 条来源登记（557 条本地、18 条固定 commit 外链）、1,083 条本地链接，无错误、警告或跳过。教学脚本及结果保存在 Wiki assets，内容 review 与机械检查分别完成。
- 边界：未读取 LoftQ 官方代码、运行模型微调或测量 GPU 性能，未验证实际导出与打包格式；论文实验数值均为作者报告，教学计算不构成模型复现。

## 2026-09-22 · QERA 低秩量化误差修正 ingest 和 review

- 使用既有 raw 的 QERA v2（2410.06040），研读正文与必要附录，新增 QERA 方法页。解释固定量化主体、层输出二阶目标、加权截断 SVD、exact 与 RMS 对角近似、可逆性及分布条件，以及 PTQ 补偿与 QPEFT 初始化的区别。
- 复用已有激活加权低秩推导，补充未中心化二阶矩、协方差与 RMS 的区别；校准页补充样本量曲线与 padding 因果边界，执行页补充离线成本与旁路存储记账，MASQuant 加入残差定义的联系。更新 4 个已有知识页，并维护 INDEX 的低秩阅读路径；未改写 raw、其他算法或项目 Skill。
- 内容 review 覆盖新增方法页全文与 4 页增量，回查 16 个 PDF 页图，覆盖定理、算法、主结果／附录、校准与成本。保留 2 位 RoBERTa 的 exact/approx 描述差异、Phi-3.5 PPL 冲突、70B/80B 列名冲突，以及附录 A.3 的平方根与逆奇异值写法问题；用条件明确的教学推导承接，未擅自修正原件或宣布 CALDERA 原文已核验。
- 验证：8 项 NumPy CPU 教学检查通过，覆盖经验目标、加权截断与尾和、两类反例、非零均值、替代因子与逆奇异值、奇异统计／阻尼和旁路预算。5 个受影响知识页的 411 个数学表达式通过 KaTeX 解析；结构检查通过：82 个知识页、569 条来源登记（551 条本地、18 条固定 commit 外链）、1,067 条本地链接，无错误、警告或跳过。内容 review 与机械检查分别完成。
- 边界：未读取 QERA 官方代码、运行模型微调／PTQ、验证实际导出格式或测 GPU 性能；A.11 热图仅定性抽查。论文未明确的校准细节、原文数值分歧及解析最优性范围保留在方法页；教学脚本不构成模型复现。

## 2026-09-22 · LSQ、LLM-QAT 与 EfficientQAT 逐篇 ingest 和 review

- 按 LSQ v3（1902.08153）、LLM-QAT v1（2305.17888）、EfficientQAT v3（2407.11062）的顺序逐篇研读、写入和内容复核，新增三个方法主页面。重点解释步长代理与梯度缩放、生成前缀和软目标蒸馏、W/A/KV 量化、非对称块重构、固定整数编码后的整网尺度训练。
- 补充已有 PTQ/QAT 基础、校准与范围、KV 对象与粒度、层输出重构四页，加入训练对象、目标和梯度范围的对照；更新 INDEX 的 QAT 阅读路径。保留原有内容及本批之前的工作区改动。
- 为 EfficientQAT 收录官方固定 commit `39175493b2d14617d342a0a7956875e6ac16221b` 的完整代码归档，48 个源码文件与归档逐一比对一致；定向阅读七个训练、量化、标签和打包文件。补清块重构两路输入、E2E 语言模型标签与浮点 matmul，以及论文理想位宽与容器打包开销的差别。
- 内容 review 覆盖三篇新增页面全文及四个已有页面的本批增量，回查所用核心公式、主结果与消融表的 PDF 页图。保留并说明 LLM-QAT 背景多位量化公式、EfficientQAT 零点梯度、附录 PPL/位宽数字等原文不一致；限定 LSQ 的 CNN 实验、蒸馏前提和 BN 启发式，保留 LLM-QAT 的 MMLU/W4A4 退化与 EfficientQAT 的多模态反例。没有把精度恢复、紧凑存储、局部 GEMV 和端到端性能混为一个结论。
- 验证：6 项 CPU 教学检查通过，覆盖 LSQ 的真实/代理导数区别、教师交叉熵恒等式、固定编码的共享尺度导数、饱和零点导数、元数据预算和 3 bit 打包空位；七个受影响知识页的 289 个数学表达式通过 KaTeX 解析。全库结构检查通过：81 个知识页、565 条来源登记（547 条本地、18 条固定 commit 外链）、1,045 条本地链接，无错误、警告或跳过。机械检查不代替上述内容 review。
- 边界：未执行三篇论文的模型训练、低比特部署或 GPU 性能复现；EfficientQAT 代码为所列固定版本的阅读证据，不声称就是全部论文实验的运行版本。论文与代码仍未充分规定的细节在对应页面明确保留。

## 2026-09-22 · QuIP、QuIP#、AQLM、SpQR 与 SqueezeLLM 论文 ingest

- 使用已有 raw 的 QuIP v2（2307.13304）、QuIP# v2（2402.04396）、AQLM v4（2401.06118）、SpQR v1（2306.03078）与 SqueezeLLM v4（2306.07629），研读方法、实验及必要附录，以归档 TeX 核对公式和表格。新增五篇方法主页面，解释非相干性、LDL/BlockLDLQ、E8P、联合加性编码、动态敏感例外、两级元数据及任务 Fisher 加权聚类。
- 新增“码本量化：标量、向量、加性表示与位宽预算”和“低比特表示的设计空间：坐标、码本与例外预算”。区分固定结构与学习码本、残差编码与联合搜索、输入重构与任务敏感性、主体位宽与完整格式，并提出有条件的组合研究问题。
- 定向核对 SqueezeLLM 固定代码 a5fd71f3 的例外提取、聚类输入与 LUT 打包：明确零映射补偿防止重复相加，以及 FP32 表/稀疏值与论文 FP16 紧凑口径的区别。登记三个具体来源文件；未扩展为整仓库或 CUDA 复现。
- 为已有旋转、二阶补偿、混合精度和 GGUF 四页补充理解入口，更新 INDEX 的极低比特阅读路径。本批新增 7 个知识页、修改 4 个已有知识页，并维护 INDEX 与本日志；复用来源，没有新增 raw 条目、项目脚本或过程报告。
- 验证：13 项 CPU 教学检查通过，覆盖标量/块 LDL 反馈、正交代理不变性、距离反例、条件补偿代价、加权簇中心、Fisher 均方区别、AQLM 交叉项与查表恒等式、贪心残差反例、格式位宽、稀疏零映射及 E8P 绝对值模式数量。受影响页面的 417 个公式经 KaTeX 解析通过，1 个 Mermaid 图解析通过。全库 78 个知识页、548 条来源登记（530 条本地、18 条固定 commit 外链）、1017 条本地链接检查通过，无错误、警告或跳过；差异空白检查通过。
- 边界：完成论文知识整合、所列代码片段阅读和教学计算；没有执行原模型校准、微调、完整编码器/解码器或 GPU kernel。正文性能与模型质量数字均标为作者报告，教学检查不代表模型或硬件复现。

## 2026-09-22 · 全库审查后的纠错与知识衔接

- 根据 65 页全库审查结果，修正 GGUF IQ1_M 总尺度的分散编码、q4 元数据字节口径、Marlin FP16 指数与位技巧、QServe 丢失转义的公式，以及可靠性页与自身 ECE 表格矛盾的总结。同步收紧 decode/带宽与 KV 离线校准的入口表述，澄清 VLM 评测页的 AWQ 泛化推断和 2 bit 比较条件。
- 新增 6 页：Transformer 与自回归推理、浮点表示与累加、模型质量评测、服务性能评测、张量并行与量化、Tensor Core 与量化 GEMM。补齐模型形状/缓存、数值舍入、PPL/风险覆盖率、TTFT/TPOT/goodput、分片通信，以及 lane/寄存器/ldmatrix/MMA 与 AWQ Triton split-K 的解释。
- 复用已有 vLLM、SGLang、Marlin、Triton、CS336 与可靠性论文；新增收录 NVIDIA 浮点、CUDA 11.8 PTX 片段布局及 Hugging Face PPL 三份官方文档的所用正文。保留完整版本、公开来源与快照身份，不保存原始 HTML。
- 维护已有推理框架、部署、诊断、反量化、自动调优与数值页的相关入口，更新 INDEX。实际修改 15 个已有知识页，新增 6 页；不新增项目脚本、Skill 或过程报告。
- 验证：21 项 CPU 教学/静态检查通过，覆盖 FP16 舍入与常量、IQ1_M 的 65,536 种尺度位模式、KV 容量、MMA 坐标与 ldmatrix 四块映射、AWQ 打包与 split-K、TP/GQA，以及评测计数与反例。受影响页面中已检查的 488 个公式经 KaTeX 转换通过，3 个 Mermaid 图解析通过；官方文档主读本摘要值与 README 一致。全库 71 个知识页、516 条来源登记（498 条本地、18 条固定 commit 外链）、953 条本地链接检查通过，无错误、警告或跳过；最终差异空白检查通过。
- 边界：本轮完成知识写入与所列验证；没有启动服务、执行原始 CUDA/Triton kernel、多 GPU collective 或 GPU profiler，也没有得到这些 kernel 的实际编译产物。编译检查和性能诊断部分提供有源码依据的执行方法，不冒充已经测得的结果。本轮新增的验证脚本和解析依赖放在临时目录。

## 2026-09-22 · 从推理库算子学习 CUDA 与 Triton

- 按本轮重点，选取已有 raw 中 vLLM `568afb3a` 与 SGLang `2f730e29` 的现成门控激活、RMSNorm 和残差融合实现，沿模型入口、包装分派与设备函数核对。版本仅用于定位；raw 与 Skill 未修改。
- 新增“门控激活的 CUDA 与 Triton 实现”，解释 gate/up 两段布局、每行 block、展平向量线程和二维 program 网格，补明向量整除、激活前后裁剪、中间舍入与过滤行未写语义。
- 新增“RMSNorm 的 CUDA 与 Triton 实现”，解释行统计、CUB 与显式两级归约、无有效向量线程的参与、固定分段/整行读取取舍，以及两路残差输出和低精度舍入阶段。区分 SGLang 模型专用 Triton、条件 JIT 路径与普通后端；不将代码存在等同于模型必然执行。
- 深化张量接口、计算模式和验证方法 3 页，两篇推理框架 Attention 页接入较小算子的学习入口，并更新 INDEX。共新增 2 个知识页、修改 5 个已有知识页，更新导航与本日志。
- 验证：CPU 教学模型通过三种工作映射与地址覆盖、向量尾部和 warp 部分和、1024 分段与 padding、独立分段归一化反例、half 残差与激活中间舍入、裁剪次序及过滤哨兵检查。65 个知识页、491 条来源登记、881 条本地链接检查通过，无错误、警告或跳过；差异空白检查通过。
- 边界：完成第一批执行映射、访存、归约与融合知识 ingest；未编译或运行 CUDA/Triton 原实现、执行 GPU 内存竞争诊断或测性能。GEMM/Tensor Core 深化与 profiling 为后续批次，PDL、全部后端和 MoE 路由没有作为已完成范围。

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

### 当时记录的后续问题

- 模型精度、跨引擎数值一致性、实际打包导出及 GPU 性能，尚无本地运行验证。
- 方法页已注明的论文内部数字差异、公开代码与论文配置差异、未读分支和有限实验条件继续保留；资料或依赖文件补齐不等于相应代码已经研读。
- 当时服务知识对 chunked prefill、前缀树共享和 prefill/decode 分离覆盖不足；后续已有 [vLLM 调度](implementation/vllm-inference-execution.md)、[SGLang 缓存](implementation/sglang-inference-execution.md)和[分阶段量化](methods/disaggregated-quantization.md)承接，具体范围见各页正文。
- 研究线索与实验设想保留在项目研究说明，未因本次迁移升级为已验证结论。

## 2026-10-14 · 能力包统一

- 与 wiki_backbone 使用相同包工具和测试，日常收敛为 prepare、按需能力、统一反馈，发布 v3；保留绑定和三项能力，更新保留旧包。两边各 31 项测试及临时接入、更新和查询流程通过；不增加旧记录兼容，未运行模型或 GPU。

## 2026-10-15 · 系统课程短材料研读：页面说明与挑战公告

- 完整研读大模型系统软件课程页面的README和16行上传者说明，固定为page-description-snapshot-2026-09-24 / BV1R7jc6qEmn。保存的是官方元数据API返回的说明，未含视频/字幕/讲义；课程事件与页面发布时间分别保留，未将160GB、8ND、性能倍数或6D称呼扩成通用结论。
- 完整研读LeetGPU GPT-2 (120M) Transformer Block公告的README与7行公开索引文本，固定为public-index-post-snapshot-2026-09-24。它只说明挑战方向与语言/设备示例，没有模型规格、实现、测试或性能结果，未沿外链补采。
- 对照[训练状态](implementation/fsdp2-parameter-lifecycle-and-checkpoint.md)、[roofline](fundamentals/hardware/arithmetic-intensity-and-roofline.md)、[阶段分离](implementation/distributed-llm-serving-orchestration.md)、[Transformer](fundamentals/model/transformer-autoregressive-inference.md)与[算子验证](implementation/kernel-correctness-and-benchmarking.md)，已有解释可承接；两项只登记实际范围，不新增正文、INDEX或sources。
- READING/SELF/计划同步：209条有依据、20条队列（12论文/5文档/3代码），本轮基线88已处理68条；第21批其余三份课程讲义继续研读。独立Astra review通过，管理链接检查通过（2文件、967条本地链接），git diff --check通过；结构/来源复用203知识页、728项有效结果，不执行上游或增加测试。提交范围为4管理文件，Obsidian改动保留。

## 2026-10-15 · IIT Bombay系统讲义研读与集合通信整合

- 通读Spring 2026导出快照七份讲义（第2–8讲，共369页）的可提取文本；重点核第6讲collective定义与第8讲ring/网络分层。整理者视觉核11页：第6讲15/20/21/22，第8讲19/33–37/42；独立Astra另核第6/8讲144页渲染总览与第8讲34–36/38放大页。其余图表未逐页视觉核，课程引用论文、当前库和性能结果未据此认定已核。
- 新增[集合通信与ring成本](fundamentals/hardware/gpu-collectives-and-communication-cost.md)：用统一rank/块定义区分SUM归约、收集与交换；解释reduce-scatter→all-gather、两阶段各p−1轮、每rank发送/接收与全组计量，再关联逻辑算法、transport、物理路径及channel/小片流水。通信量和延迟模型标为整理者推导，明确张量dtype、均匀链路、收发重叠和忽略项。
- 原图2p−1个primitive步骤与2(p−1)次链路传输分开解释；不吸收讲义中的当前NCCL LL概括、ZeRO-2 activation或GELU整列依赖等不准确泛化。TP/FSDP仅补公共解释链接，保留其固定实现与状态生命周期，不重复扩写训练策略。
- READING/INDEX/SELF/计划同步：210条有依据、19条队列（12论文/4文档/3代码），本轮基线88已处理69条；下一项GPU Mode与Microsoft讲义。独立Astra review通过，结构/来源检查通过（204知识页、729项来源），受影响链接检查通过（6文件、1432条本地链接），git diff --check通过。不运行课程代码或增加算术测试；提交8文件，Obsidian改动保留。

## 2026-10-15 · Microsoft系统讲义研读：跨块进度与缓冲生命周期

- 完整核来源README，固定repository-commit-2a974e1a483eb5c2f714ba5001ac33d8c58998d5；材料准备逐页通读第5–10讲273页可提取文字，并视觉核39页：第7讲25–28，第9讲19–24/39–43，第10讲30–39/50–63。多数图像正文未逐页核，不把文本遍历算273页完整内容研读。根代理回查第9讲20/39–42原页，引用按PDF物理页定位。
- [硬件页](fundamentals/hardware/gpu-execution-and-memory-hierarchy.md)补NNFusion任务融合图的进度死锁：驻留block等待全部Matmul结束，却占满可用资源，尚未调度的依赖任务无法执行。区分数据可见性与执行机会，不推定任意grid自旋屏障或完整软件执行器已可用。
- [计算模式](fundamentals/operators/gpu-kernel-computation-patterns.md)仅补最后使用/同时存活容量与[MoE工作区](implementation/moe-routing-and-quantized-expert-execution.md)的连接；现有实例已说明gate/up消费后复用down缓冲，不另开内存规划专题。保留历史材料身份，不吸收量化数例不一致或把未作答练习当实现证据。
- READING/SELF/计划同步：211条有依据、18条队列（12论文/3文档/3代码），本轮基线88已处理70条；下一项GPU Mode讲义。独立Astra review通过，结构/来源检查通过（204知识页、731项来源），受影响链接检查通过（4文件、1006条本地链接），git diff --check通过。未执行课程/NNFusion代码，不新增测试；提交6文件，Obsidian改动保留。

## 2026-10-15 · GPU Mode讲义研读：推测执行与EVT布局连接

- 材料准备完整读README40行、固定目录253行及三份PDF（54/36/95共185页）的可提取文本/逐页渲染图览；根代理复读Lecture22全36页文本，重点核18/24/30原图及Lecture23第24、42/44–47/49–51原页。目录commit77a8df418834e5789c12da23e7d2719e0efabef1与外部PDF保存身份分开；未沿目录取代码或其他课程材料。
- 新增[推测解码](fundamentals/model/speculative-decoding-proposals-verification-and-cost.md)：候选/目标概率→验证→接受前缀与纠正/bonus，区分输出与有效KV；以A+1推进量和期望尾和解释每轮成本，公式为带终止条件的整理者展开。lookahead与旧单query虚拟序列限定在讲义设计，不接成568afb3a普通V1源码；新版完整采样/提交协议及性能未核。
- [CUTLASS组成](implementation/cutlass-gemm-composition-and-dispatch.md)补SM90 EVT节点、(m,1) bias沿N广播、builder与subtile回调；[布局](fundamentals/operators/tensor-layout-and-kernel-contracts.md)补(4,(2,2))/(2,(1,8))的逻辑分解与存储映射。保留2024-06-07/CUTLASS3.5与伪码范围，不据简化指令/consumer片段改写现有正确解释。量化基础已有充分覆盖，不复制历史性能表或把2/3bit模拟记作原生kernel。
- INDEX与vLLM/一致性/W8A8按理解需要连接；READING/SELF/计划同步212条有依据、17条队列（12论文/2文档/3代码），本轮基线88已处理71条，第21批五项范围完成，下一项第22批。独立Astra review通过，结构/来源检查通过（205知识页、734项来源），受影响链接检查通过（9文件、1484条本地链接），git diff --check通过。未执行上游、未新增测试；提交11文件，Obsidian改动保留。

## 2026-10-15 · CudaForge v2研读：反馈分流与候选预算

- 完整研读固定arXiv:2511.01884v2的LaTeX六节和必要附录prompt/指标表/案例及任务子集；材料准备视觉核Figure1–9与Table1–8，33页PDF仅作定位与关键原页回查，不将文本提取计作逐页视觉阅读。源包保存证明复用原登记，未改raw或取外链实现。
- [性能分析](implementation/gpu-kernel-performance-analysis.md)补任务Model→ModelNew、外部验证后分流修复/性能反馈、最新提示状态与跨轮最佳正确候选；解释离线代表任务正确样本的相关指标筛选与在线3–4项/一项改动。Algorithm2单kernel/runtime求相关与正文十样本粒度不一致，按正文说明意图，不补造完整实现；既有NCU/瓶颈/计时解释直接复用。
- [测量页](implementation/kernel-correctness-and-benchmarking.md)补任务覆盖、候选选择和搜索预算；作者全250/子集25、默认10/扩展30轮与Correct范围分开，只用自身成本0.30美元/26.5分钟并保留RTX6000名称冲突。Perf/失败聚合、H200图文与外部成本单位等未闭合内容不拼成排行榜或成本倍数，原件条件继续保留。
- INDEX/READING/SELF/计划同步213条有依据、16条队列（11论文/2文档/3代码），本轮基线88已处理72条；接续KernelBench/TritonBench。独立Astra review通过，结构/来源检查通过（205知识页、736项来源），受影响链接检查通过（5文件、1460条本地链接），git diff --check通过。未执行作者框架、生成或GPU实验，不新增测试；提交7文件，Obsidian改动保留。

## 2026-10-15 · KernelBench与TritonBench：生成任务与指标分母

- 按用户最新要求加速，以同主题合批整合，独立评审改用gpt-6.1-sol，新代理fork_turns:none。KernelBench v1材料准备完整读active主文/附录A–H与表1–15，视觉核关键12页；TritonBench v1完整读正文/表1–6/附录/四构造prompt，视觉核Figure1–7/Table1–6与未引用流程图。固定源包证明复用原登记，无外链补采或raw修改。
- 新增[生成Kernel任务与评测](implementation/llm-generated-kernel-evaluation.md)：区分任务/Module与设备kernel、全任务fast_p与正确子集速度、预算内至少一次与最后一轮；Table7第8轮正确但慢于Eager、第10轮重新错误作为实际反例。G/E/P与生成/CPU编译/GPU隔离复用已有候选验证和CudaForge状态解释，Events不泛称完整主机wall-clock。
- TritonBench G184/T166的来源、接口与速度参考分别记录；用Table4覆盖下降而条件速度上升说明分母差异，G的尺寸扫描峰值GPU Efficiency与平均表现、occupancy和NCU流量分开，未扩大到T也有该列。原文Call/Execution冲突、聚合/容限与harness未给范围保留。
- INDEX与测量/分析页连接；READING/SELF/计划同步215条有依据、14条队列（9论文/2文档/3代码），本轮基线88已处理74条，第22批范围完成。独立Sol review通过，结构/来源检查通过（206知识页、738项来源），受影响链接检查通过（6文件、1473条本地链接），git diff --check通过。未运行生成器/评测器/GPU，未新增测试；提交8文件，Obsidian改动保留。

## 2026-10-15 · 生产Kernel评测：真实输入、数值契约与归约

- 联合研读FastKernels v1、FlashInfer-Bench v1、KernelBenchX v2的固定主文与必要/全部附录，关键图表原件回查；源包身份复用已有版本证明。准备报告分别保留实际视觉范围、完整数据/harness缺失与原文差异，没有沿外链补采或修改raw。
- [生成Kernel评测](implementation/llm-generated-kernel-evaluation.md)补模型来源与真实输入、状态/通信关系、Trace四对象；区分FastKernels代理主表、MacroEval与46架构端到端实验，解释有效族几何平均和三轴乘积；FlashInfer按workload→solution的宏平均保留基线身份与apply离线选择/fallback。
- KernelBenchX补shape/dtype前置门、标准/异常输入、六项量化三指标联合阈值，以及任务内多输入总时间比和跨正确任务算术平均。[验证页](implementation/kernel-correctness-and-benchmarking.md)补低精度匹配比例、采样TVD+mask与空KV合法LSE特殊值；不重复计时/反馈基础，也不把静态门、有限测试和算法图扩成实现保证。
- 原件FastKernels CI/条件速度强表述、FlashInfer Figure8性能与末候选、KernelBenchX图文/名称/环境和do_bench runs差异集中登记，不据此做生成器排名。READING/INDEX/SELF/计划同步218条有依据、11条队列（6论文/2文档/3代码），本轮基线88已处理77条，第23批范围完成；接续第24批。独立Sol review修正apply图号后通过；结构/来源检查通过（206知识页、742项来源），受影响链接检查通过（5文件、1470条本地链接），git diff --check通过。未执行上游/模型/GPU，不新增测试；提交7文件，Obsidian改动保留。

## 2026-10-15 · 多模态压缩案例：删层恢复与脉冲时间编码

- 为并行加速先完成第25批两论文，接续第24批准备。完整研读医学压缩v3的active正文/附录A/B及表1–5，视觉核PDF10/15/16；完整研读SpikeMLLM v1主文/附录§6–10、两等价证明/算法与表1–13，视觉核PDF3/4/8/15。版本证明复用原README，未取外链/改raw；文本提取不记为全部图像已读。
- 新增[医学删层恢复](methods/medical-llava-prune-sft-quantize.md)：完整答案→MiniLM相似度、整层删除/恢复/权重量化的对象与顺序，两次QLoRA配置、域数据与judge分别说明。保留原件排序方向、参数身份及4.2×与延迟表冲突；百分化1–4评价不改写成诊断正确率。
- 新增[SpikeMLLM](methods/spikemllm.md)：标准单位脉冲与polar位权编码、尺度/去极端级和符号条件、non-polar差异、MED模态/层预算；用线性展开区分离散表示等价与FP16任务质量。MAC/AC/位宽代理、PE位门控与RTL/周期模型对A800实测分别解释，25.8×按功率比处理，不作GPU算法加速或同任务能效结论。
- VLM公共入口与INDEX/READING/SELF/计划同步220条有依据、9条队列（4论文/2文档/3代码），本轮基线88已处理79条；第25批选定来源范围完成，第24批四论文/三代码/两文档继续。独立Sol review通过；结构/来源检查通过（208知识页、744项来源），受影响链接检查通过（6文件、1492条本地链接），git diff --check通过。未执行作者软件/RTL/模型或GPU，不新增测试；提交8文件，Obsidian改动保留。

## 2026-10-15 · 视觉预算与VLA接入：原88条队列收尾

- 第24批九项按固定版本完成：vla.simd v1/代码7354b6d7/collection快照、MWOP v1、RDTok v1、CoVisco v1/代码604cd70d、Edge DART文档0.11.0/同版代码e649dadc。各来源README与有效保存证明复用；论文主文/必要附录及关键原件、所列代码主线实际核读，未将保存文件数或提取文本等同全图/全库语义覆盖。
- 新增[VLA CPU执行](implementation/vla-cpu-action-chunk-execution.md)、[MWOP](methods/mwop.md)、[CoVisco](methods/covisco.md)、[RDTok](methods/rdtok.md)、[DART运行接入](implementation/visual-token-pruning-runtime-integration.md)五页。分别讲动作窗口/缓存/量化、模态执行宽度与恢复、native摘要/输入输出两层选择、保留位置的特征码率DP、共同keep-list/base-draft位置接口。collection只有八仓库索引，已有机制可承接，无需新增正文或sources。
- 保留独有条件与反例：MWOP贡献gate不含实际softmax重归一化，配置选取使用full benchmark；CoVisco selector仅在所用模式执行、codec外部缺口/40%soft目标/eval shift；RDTok有限source目标最优不扩成真实packing或未来任务最优；VLA供给率不当反馈频率，DART共同成功题集与token ratio不当实测全任务速度。未执行上游、训练/模型/机器人/GPU或新增测试。
- 公共VLM/FoldQuantVLA与INDEX/READING/SELF/计划同步：**229条有依据、58条历史范围未知、0条待核；原88条全部按实际范围完成。** 对照 `1bb2565028724f0d3445b846af462a66f5ee0f26`的29论文/48文档/11代码逐项核身份：此前79+本批9唯一覆盖全部88，无漏失/重复销项/其他来源混入；18处版本显示补充保留原固定身份，58条历史整行未改变。修正首轮140历史计数与SELF剩余摘要。
- 独立Sol五页内容及整批管理/阅读路径review通过，修正CoVisco selector适用范围与视觉输入长度措辞。全库机械检查通过（213知识页、752项来源、4185条本地链接；138外链仅登记未访问），errors/warnings均0；基线覆盖复核为141/58/88→229/58/0，原88全部唯一对应，来源集合及58条历史整行保持一致；git diff --check通过。提交12文件，保留Obsidian改动。

## 2026-10-15 · 全库最终复核：表达、事实、深度与链接

- 独立 Sol 分组完整审读全部213个知识页：97方法、76实现、40基础/理论/研究；根代理复核管理入口和修订。事实检查定向回查固定原件中的关键公式、数字与执行接口；代表阅读路径另核目标正文，未把全页阅读记作全部主张逐项原件核验。
- 修正 MBQ 将裸权重内存估计写成显存下降、ZeroQuant 将多步 LKD 写成一次反向、VLM 评测把分组收益限定到2bit，以及 AWQ 旧 GEMV 混称输入行/输出通道的问题。BRECQ 明确 LSQ 式步长梯度、舍入变量学习率及未采用的梯度缩放；可靠性结果按所测模型、位宽和具体指标表述。
- 删去 AffineQuant/变换比较的空泛场景提醒、算子页的未来实验安排和 MoE 页的维护过程措辞；码本页由现有 E8P 实现章节承接重复位域细节，保留逻辑码/物理载荷、主体预算、临时内存差异及独有反例。未新增页面、修改 raw 或开展 ingest。
- SELF、READING 和原计划移除重复粘贴的逐批履历，保留当前画像、后续缺口、历史入口与全部原表。逐行比对确认画像表、研读表和计划表未改变；229条有依据、58条历史范围未知、0条待核状态保持。修订经独立 Sol 复核。
- 全库严格机械检查通过：213知识页、752项来源、4190条本地链接，errors/warnings均0；138条外部引用仅登记，未逐条访问。git diff --check通过。未执行上游、模型或GPU，不新增测试；提交范围为11知识页和4管理文件，保留Obsidian改动。

## 2026-10-15 · 历史范围核实第1批：经典量化与数值格式

- 核实原58条中的16条：11篇经典量化论文，以及FP8、Microscaling和浮点/PPL/PTX三份文档。读取固定来源身份、承接正文和关键原件；已有方法解释可复用，BiLLM/GPTAQ/整数推理/白皮书按实际局部依赖登记。新核读记为2026-10-15，旧日期无法还原时不猜填全文历史。
- PPL页补保存HF示例的静态计数问题：重叠窗口首标签已忽略，错位后仍须按有效labels计数；直接减batch size会少计。执行路径对白皮书的非对称权重开销改用正确小节名称定位。数值格式与PTX的定义、特殊值、尺度/转置及warp坐标已有承接，无需新增页。
- READING逐条补范围、主承接与未核部分：245条有依据、42条历史范围未知，原88条队列仍为0。固定身份保留；SELF与原计划同步当前剩余。
- 独立Sol复核通过；全库严格机械检查通过（213知识页、752项来源、4208条本地链接，错误/警告均0），git diff --check通过。本批6文件，不新增测试或运行上游，保留Obsidian改动。

## 2026-10-15 · 历史范围核实第2批：变换与旋转

- 核实14条：8篇变换/旋转及其数学依赖论文、PyTorch正交参数化快照、OmniQuant/AffineQuant/FlatQuant/QuaRot/SpinQuant五个固定代码版本。逐条核身份、必要原件与所选主线；可逆/条件数、占优证明、RoPE交换、旋转与截断、局部MSE与最终CE、浮点模拟与部署条件已有承接，无需新增方法解释。
- OmniQuant原raw快照已移除，本轮从固定commit临时取得同版材料，实际核LLaMA校准、LWC/LET、QK/cache和pack入口。三页删去已失效的“本轮未重审”维护句，在来源表更新实际范围；原固定URL、版本与raw移除事实保留。READING补OmniQuant、QuaRot、SpinQuant当前方法入口。
- 当前259条范围明确、28条历史范围未知，原88条队列仍为0。新读日期与旧历史分开，所选代码不扩大为全部模型、后端或设备运行。
- 独立Sol复核通过；全库严格机械检查通过（213知识页、752项来源、4225条本地链接，错误/警告均0），git diff --check通过。本批7文件，保留Obsidian改动。

## 2026-10-15 · 历史范围核实第3批：多模态与归因

- 核实18条：10篇多模态/IG/QLoRA论文、LUQ项目页与MIT熵讲义、6个固定代码版本。核对方法页与公共解释，再按实际依赖读核心原件/所选代码。IG完整性、模态缩放与白化残差、两套静态网格、加权重构、NF4/尺度二次量化等已有解释充分，无需新增正文。
- 六个代码条目分别保留实际范围：MASQuant浮点主体/低秩旁路、MBQ梯度统计/候选搜索、MQuant首通道与单网格、QIG归因/目标/浮点执行、QVLM校准及单/多token分流。SplitQ同SHA临时归档确认仅README/LICENSE/.gitignore，没有算法实现；以文件内容审查完成该项，不计为算法代码已核。
- READING补准MBQ、SplitQ、VLMQ方法页及NF4公共入口，固定身份与其余承接保留。当前277条范围明确、10条历史范围未知，原88条队列仍为0；SELF与原计划同步。
- 独立Sol复核通过，临时审查编号改为具体来源名；全库严格机械检查通过（213知识页、752项来源、4247条本地链接，错误/警告均0），git diff --check通过。本批4个管理文件，保留Obsidian改动。

## 2026-10-15 · 历史范围核实第4批：系统评测与全58条收尾

- 核实最后10条：KIVI、LVLM评测、Marlin、PagedAttention、QServe、可靠性6论文及KIVI/Marlin/MLC-LLM/TensorRT-LLM四个固定代码版本。核必要方法/协议/表格与所选代码；各页已有机制和条件可复用。Marlin论文对IST-DASLab仓库有明确原文脚注，按同一论文/所选实现关系保留。
- 部署页修正MLC写回条件：CLI将独立的--output传给save_params，只有指向已有权重缓存时属于就地更新；不由update_if_exists推定写入模型目录。KIVI窗口/元数据、Marlin接口、PagedAttention共享、可靠性指标与基线继续按固定版本保留。
- 原58条按16+14+18+10四批全部处理：37论文、6文档、15代码；新核读范围和日期逐条登记，未核表格、附录、设备或其他模型分支继续明确。当前287条范围明确、历史待核0、原88条队列0；删除SELF已完成的历史核实待办，保留实际知识缺口，旧排期和历史日志继续可追溯。
- 独立Sol内容与最终记录复核通过；对照c55e67d，原229条明确行逐字保留，58个固定身份各唯一迁移，总身份集合不变。全库严格机械检查通过（213知识页、752项来源、4257条本地链接，错误/警告均0），git diff --check通过。本批5文件；全程未运行模型/GPU或新增测试，Obsidian改动保留。

## 2026-10-15 · 删除根目录research工作区

- 按用户要求删除根目录research：旧计划/审查记录与AWQ实验代码共11个跟踪文件，并清理目录内派生缓存。未迁移或重建替代目录；被删版本可从Git历史追溯。
- 知识页没有以这些文件登记来源或支撑正文；当前材料缺口已由SELF、READING和相应正文承接。wiki/research知识分类、213知识页及287条研读记录保持原样；已有日志中的路径保留为历史叙述。
- README、SELF、Wiki规范及研读目录清理旧入口，明确研究/开发/实验工作区位于仓库外。能力包README同步边界，随包说明变动对应更新发布版本与清单。
- 独立Sol复核通过；全库严格机械检查通过（213知识页、752项来源、3512条本地链接，错误/警告均0），git diff --check通过。能力包v4发布时已验证10文件清单；未改工具代码或更新外部镜像，不新增测试。提交18文件，保留Obsidian改动。

## 2026-10-15 · 精简导航与按需读取 Wiki

- INDEX 从335行、43271字元缩为28行、1865字元，只保留主要问题的代表入口；删除逐页目录与重复阅读路径，不删知识正文。根说明与Wiki维护规范同步：普通新页面通过正文关联和检索进入，不再逐项扩充索引。修复一处历史日志指向已删除标题的链接。
- wiki-query 增加量化方法、模型与缓存、部署系统、GPU算子、误差与评测五份短 reference；只承担选路和检索提示，不复制完整目录。已知页面直接进入，否则选择相关 reference，先查文件，再查标题和命中段落，按解释缺口补读；未列入导航不视为知识缺失。外层入口和research同步取消预读整份INDEX、SELF等管理材料的要求。
- 实际按新路径检查三类查询：已知AWQ实现页可定位SGLang普通AWQ的独立反量化与矩阵乘；部署reference可进入MLC的配置、权重、模型库及运行时职责链，并保留JIT与固定版本边界；models reference未列出PIVOT-Q，仍可用“稀疏动作”检索方法目录，读到学生轨迹恢复、行为锚点和所选状态比例的成本边界。查询定位无需加载完整INDEX；这三例不是全任务检索质量评测。
- 全库严格机械检查通过（213知识页、752项来源，错误/警告均0）；能力包原有31项测试通过。发布本地v6清单，共15个载荷文件；隔离临时项目安装及prepare返回ready，快照包含全部五份reference，临时文件随后清理。未更新真实外部镜像，未运行模型或GPU。

## 2026-10-15 · 精简全部 Skill 的表达

- 重写8项Skill的README与执行规范，共16份文档。README说明用途、设计选择和边界，执行规范保留动作、条件与失败处理；合并重复要求，删去空泛说明、旧方案痕迹和无用栏目，改写抽象术语。文档总字符由95015减至25222；数量仅记录变化，不作为质量判据。
- skill-creator取消固定章节模板，在原有写作和检查步骤中加入信息增量、去重、具体表达与保留必要条件的要求；AGENTS将简洁表达统一用于文档、Skill和对话。算子规范明确当前没有随附references或脚本，去掉accepted的双重含义。
- 对照原版检查职责与关键分支，走查仅授权README、查询未命中、材料不可读、重复入库、限定链接检查、独有内容合并及算子失败/回退/预算结束等场景；这些是文档走查，不代表实际研究或GPU运行。元数据、74处本地链接与锚点及差异格式检查通过。
- 能力包原有31项测试通过，本地发布v7；临时安装与prepare成功，15个包文件和5份查询reference完整，Wiki绑定保持不变。未更新真实外部安装，未运行模型或GPU。

## 2026-10-15 · 恢复 Skill 标准章节并改进内容取舍

- 修正上一轮取消固定章节的做法：8项Skill恢复标准的任务、输入、原则、工作步骤、输出、校验章节，README恢复设计章节。将已有内容归位，保留精简结果，不恢复重复段落。
- skill-creator要求先判断内容是否必要，再决定保留、合并、删除或移动，最后改措辞；新增规则先判断需求及已有职责，涉及边界调整先核实现。Research开头直接说明研究任务，版本、保存位置及反馈分别放到相关位置；保留源码执行状态和因果归因等具体判断要求。AGENTS替换原有表达条款，不另叠规则。
- 核对prepare与feedback实现后保留必要使用边界：工具准备和核验包快照，不自动限制模型读写。对照文档检查仅方案、知识已充分、材料或环境受阻、公共知识修改建议等分支；这些检查不等于真实研究效果评测。
- 8项Skill章节与元数据、22份文档的74处本地链接及锚点检查通过。能力包本地发布v8，隔离临时安装和prepare通过，15个包文件及5份查询reference完整。工具代码未变，未更新真实外部安装，未运行模型或GPU。

## 2026-10-15 · 按内容用途精简 Skill 与 README

- 逐段检查8项Skill及其README，并检查项目、Wiki和raw的README。删除README重复的执行表、章节清单、记录字段和未来设想，合并执行规范中的重复检索、边界与记录要求；保留固定章节、证据条件和失败处理。16份Skill文档由27362字元减至24322；raw规范各段有具体用途，未修改。
- skill-creator把内容取舍落实为删段检查，写作时明确对象、动作和条件；能力包入口区分用户答案与内部反馈，研究及算子交付提供结果和记录位置，不逐轮复述过程。用户级表达规范同步用于所有回复与进度说明。
- 对照差异核对授权、来源不可读、仅方案、实验失败、最佳版本保留及最终验证等分支。8项Skill结构和元数据、26份文档的142处本地链接检查通过；v9临时安装与prepare通过，15个包文件和5份查询reference完整。未做模型效果对照实验，未更新真实外部安装。
