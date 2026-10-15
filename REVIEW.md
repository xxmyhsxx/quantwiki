# Wiki 分批审核清单

范围：2026-10-15 的 213 个知识页，分为 54 批。覆盖基础 21 页、理论 16 页、方法 97 页、综合研究 3 页、实现 76 页；不包含管理页、Skill 和 raw 原件。原件只在核验相应主张时读取。

当前进度：完整审核 0/213 页，完成 0/54 批。下一批：01。此前四页已修正指定问题，仍需检查其余正文，未记为整页通过。

## 审核要求

按 [Wiki 正文规范](wiki/README.md#正文与解释深度) 和 [wiki-review](.agents/skills/wiki-review/SKILL.md) 执行，重点检查：

- 对象在所属任务中解决什么问题、承担什么作用；整体怎样工作，各部分为何存在、怎样配合。
- 必要前置、公式定义、推理、事实依据、版本及适用条件是否充分；方法收益与系统收益是否混淆。
- 段落和链接是否提供必要信息；重复、晦涩、无用内容，以及跨页冲突、解释断点和真正缺失的知识。

每页完整阅读正文，再按主张和疑点查相关页与原件。结构和链接检查不能代替内容判断；本次不开展实验。

## 如何使用清单

每次读取当前批次及必要关联页，不把整个清单或 INDEX 作为必读输入。每页只分配一个主审核批次；跨批引用可提前查阅，不因此将目标页标为已审核。长页按章节分段读，整页完成后再勾选。

完成一批后，在该批下面简记主要发现、已修正与待修问题、核验范围及提交；同步页勾选、总进度与下一批。看过标题、抽查过段落或修过一处问题，都不算整页完成；无问题也不为凑数改写。新增页面加入相应批次，页面合并或删除时同步清单。

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

- [ ] [ModelOpt PTQ：配置、校准与部署 checkpoint](wiki/implementation/modelopt-ptq-calibration-and-checkpoint.md)
- [ ] [量化模型的部署框架与后端支持](wiki/implementation/quantized-llm-deployment-backends.md)
- [ ] [GGUF 与块量化存储格式](wiki/implementation/gguf-block-quantization-formats.md)

## 02 · 量化矩阵乘的公共解释（3 页）

重点：逻辑计算、缩放、表示与实际执行怎样衔接，公共页与具体实现如何分工。

- [ ] [线性层与输入通道](wiki/fundamentals/operators/linear-layer-input-channel.md)
- [ ] [量化矩阵乘法的缩放与执行路径](wiki/implementation/quantized-matmul-scaling-execution.md)
- [ ] [块缩放矩阵乘：FP4/FP8 编码、尺度布局与执行接口](wiki/implementation/block-scaled-matmul-execution.md)

## 03 · SharQ、SpQR 与表示综合（5 页）

重点：复核已局部修正的用途、整体流程、公式定义和跨页去重，再检查其余正文。

- [ ] [SharQ：动态稀疏 FP4 主干与量化残差](wiki/methods/sharq.md)；已有局部修正，待整页复核
- [ ] [SharQ 的稀疏残差、共享尺度与所选接口](wiki/implementation/sharq-sparse-residual-and-shared-scales.md)；已有局部修正，待整页复核
- [ ] [SpQR：敏感权重保留与两级量化元数据](wiki/methods/spqr.md)
- [ ] [SpQR 的两级尺度、稀疏保护值与所选 CUDA 格式](wiki/implementation/spqr-metadata-outliers-and-cuda-format.md)
- [ ] [低比特表示的设计空间：坐标、码本与例外预算](wiki/research/low-bit-representation-design.md)；已有局部修正，待整页复核

## 04 · 码本与表示预算（4 页）

重点：主体、元数据、例外、解码成本是否使用同一口径；理论页是否变成方法摘录。

- [ ] [码本量化：标量、向量、加性表示与位宽预算](wiki/theory/codebook-quantization-and-bit-budget.md)；已有局部修正，待整页复核
- [ ] [SqueezeLLM：敏感性加权的非均匀量化与稠密稀疏分解](wiki/methods/squeezellm.md)
- [ ] [GPTVQ：多输入列向量、数据感知码本与移动 CPU 解码](wiki/methods/gptvq.md)
- [ ] [VPTQ：同列向量量化、二阶反馈与码本成本](wiki/methods/vptq.md)

## 05 · 码本方法与实现对应（6 页）

重点：方法目标如何对应校准、坐标恢复、打包和推理消费，避免将浮点恢复当成低位执行。

- [ ] [QuIP：非相干处理与 LDLQ 二阶量化](wiki/methods/quip.md)
- [ ] [QuIP#：Hadamard 非相干处理与 E8P 格码本](wiki/methods/quip-sharp.md)
- [ ] [AQLM：加性码本与激活感知的联合编码](wiki/methods/aqlm.md)
- [ ] [QuIP 的逆序 LDLQ、坐标恢复与浮点保存路径](wiki/implementation/quip-ldlq-and-coordinate-restoration.md)
- [ ] [QuIP# 的 E8P 打包、解码与所选 Llama 路径](wiki/implementation/quip-sharp-e8p-packing-and-inference.md)
- [ ] [AQLM 的原始校准、模型转换与 2×8 推理读取](wiki/implementation/aqlm-calibration-export-and-inference.md)

## 06 · 因子、多精度与其他编码（4 页）

重点：各编码解决的部署需求、表示约束和真实成本，检查与综合页的一致性。

- [ ] [DBW：在 SVD 分量上联合学习秩与位宽](wiki/methods/dbw.md)
- [ ] [StagQ：共享元数据的前缀权重流与稀疏保护](wiki/methods/stagq.md)
- [ ] [FlexQ：INT6 分组量化与二值 Tensor Core 执行](wiki/methods/flexq.md)
- [ ] [FlexQ 的位平面、组恢复与所选调用接口](wiki/implementation/flexq-bitplanes-and-group-restoration.md)

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
