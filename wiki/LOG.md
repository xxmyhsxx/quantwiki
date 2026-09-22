# Wiki 变动记录

记录实际发生的修改、原因、检查结果与遗留影响。同批变动合并记录；规则见 [README](README.md)，阅读入口见 [INDEX](INDEX.md)。

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

### 仍需独立任务解决的内容问题

- 模型精度、跨引擎数值一致性、实际打包导出及 GPU 性能，尚无本地运行验证。
- 方法页已注明的论文内部数字差异、公开代码与论文配置差异、未读分支和有限实验条件继续保留；资料或依赖文件补齐不等于相应代码已经研读。
- 服务知识尚未完整覆盖 chunked prefill、前缀树共享和 prefill/decode 分离；旧记录中的待扩展项不是空目录或已完成页面。
- 研究线索与实验设想保留在项目研究说明，未因本次迁移升级为已验证结论。
