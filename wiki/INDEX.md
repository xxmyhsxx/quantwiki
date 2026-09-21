---
title: 量化知识地图
type: index
sources: []
---

# 量化知识地图

从问题选择阅读路径。页面按主要职责归入五类目录，同一知识可以出现在不同阅读路径中。维护规范见 [README](README.md)，变动记录见 [LOG](LOG.md)。

## 从哪里开始

- **理解模型与数值**：[Transformer 与自回归推理](fundamentals/model/transformer-autoregressive-inference.md) → [浮点表示与累加](fundamentals/numeric-formats/floating-point-and-accumulation.md)。
- **建立共同基础**：[线性层与输入通道](fundamentals/operators/linear-layer-input-channel.md) → [均匀量化与分组](fundamentals/quantization/uniform-quantization-and-groups.md) → [PTQ、QAT 与代理梯度](fundamentals/quantization/post-training-and-quantization-aware-training.md) → [校准数据与量化范围选择](theory/calibration-and-range-selection.md)。
- **理解量化方法**：[AWQ](methods/awq.md) → [GPTQ](methods/gptq.md) → [SmoothQuant](methods/smoothquant.md) → [AWQ、GPTQ 与 SmoothQuant](methods/awq-gptq-smoothquant-comparison.md)。
- **理解变换与研究抽象**：[对角缩放与等价变换](theory/diagonal-scaling-equivalent-transform.md) → [OmniQuant](methods/omniquant.md) → [AffineQuant](methods/affinequant.md) → [FlatQuant](methods/flatquant.md) → [OmniQuant、AffineQuant 与 FlatQuant](research/omniquant-affinequant-flatquant-comparison.md) → [QuaRot](methods/quarot.md) → [SpinQuant](methods/spinquant.md)。
- **研究多模态量化**：[视觉语言模型中的 token 与量化对象](fundamentals/model/vision-language-model-tokens-and-quantization.md) → [MBQ](methods/mbq.md) → [VLMQ](methods/vlmq.md) → [MBQ 与 VLMQ](research/mbq-vlmq-comparison.md) → [QIG](methods/qig.md) → [LUQ](methods/luq.md)。
- **学习算子开发**：[张量布局与接口](fundamentals/operators/tensor-layout-and-kernel-contracts.md) → [GPU 执行与存储](fundamentals/hardware/gpu-execution-and-memory-hierarchy.md) → [逐元素、归约与分块矩阵乘](fundamentals/operators/gpu-kernel-computation-patterns.md) → [roofline](fundamentals/hardware/arithmetic-intensity-and-roofline.md) → [正确性与性能测量](implementation/kernel-correctness-and-benchmarking.md) → [AWQ 反量化内核](implementation/weight-only-dequant-kernels.md)。
- **从算法走向部署**：[量化矩阵乘法的缩放与执行路径](implementation/quantized-matmul-scaling-execution.md) → [AWQ 的实现核对](implementation/awq-implementation.md) → [GPTQ 的实现核对](implementation/gptq-implementation.md) → [SmoothQuant 的实现核对](implementation/smoothquant-implementation.md) → [权重量化反量化内核的契约](implementation/weight-only-dequant-kernels.md) → [量化模型的部署框架与后端支持](implementation/quantized-llm-deployment-backends.md)。
- **从推理库学习 CUDA 与 Triton**：[工作映射与张量接口](fundamentals/operators/tensor-layout-and-kernel-contracts.md) → [vLLM/SGLang 门控激活](implementation/gated-activation-cuda-triton-kernels.md) → [RMSNorm 行归约与残差融合](implementation/rmsnorm-cuda-triton-kernels.md) → [Attention 算子设计](implementation/sglang-attention-operator-design.md)。
- **深入矩阵乘与多卡执行**：[Tensor Core 与量化 GEMM](implementation/tensor-core-quantized-gemm.md) → [配置与自动调优](implementation/kernel-configuration-and-autotuning.md) → [张量并行与量化](implementation/tensor-parallel-quantization.md)。
- **优化算子配置与流水**：[GPU 算子的计算模式](fundamentals/operators/gpu-kernel-computation-patterns.md) → [Kernel 配置选择与自动调优](implementation/kernel-configuration-and-autotuning.md) → [异步拷贝与多级流水](implementation/gpu-async-copy-pipelines.md) → [正确性与性能测量](implementation/kernel-correctness-and-benchmarking.md)。
- **理解 KV cache 与服务**：[KV cache 量化的对象与粒度](theory/kv-cache-quantization-objects-and-granularity.md) → [KIVI](methods/kivi.md) → [SAW-INT4](methods/saw-int4.md) → [推理服务的内存管理与批处理](implementation/serving-memory-and-batching.md)。
- **理解推理框架与算子设计**：[内存管理与批处理](implementation/serving-memory-and-batching.md) → [vLLM 的 token 调度与变长批次](implementation/vllm-inference-execution.md) → [vLLM Attention 后端、分页计算与图执行](implementation/vllm-attention-operator-design.md) → [AWQ 实现](implementation/awq-implementation.md) → [正确性与性能测量](implementation/kernel-correctness-and-benchmarking.md)。
- **理解 SGLang 的缓存与算子**：[Radix 缓存、批次与重叠执行](implementation/sglang-inference-execution.md) → [KV 索引、Extend 与 Decode 归约](implementation/sglang-attention-operator-design.md) → [AWQ 的 SGLang 路径](implementation/awq-implementation.md#6-跨引擎sglang)。
- **判断误差与运行收益**：[量化误差诊断与验证](implementation/quantization-error-diagnosis.md) → [模型质量评测](implementation/model-quality-evaluation.md) → [量化与可靠性](theory/quantization-reliability-and-selective-prediction.md) → [服务性能评测](implementation/serving-performance-evaluation.md)。

## 全部知识页

### 公共基础

| 页面 | 主要问题 |
| --- | --- |
| [Transformer 与自回归推理](fundamentals/model/transformer-autoregressive-inference.md) | 从 token、QKV、残差和 LM head 解释 prefill/decode、GQA 与 KV 复用。 |
| [浮点表示与累加](fundamentals/numeric-formats/floating-point-and-accumulation.md) | FP16/BF16/FP32 的范围、有效数字、FMA 和归约顺序怎样影响结果。 |
| [张量布局与算子接口：从逻辑下标到内存地址](fundamentals/operators/tensor-layout-and-kernel-contracts.md) | shape、stride、dtype 和执行接口怎样决定正确读写。 |
| [GPU 算子的计算模式：逐元素、归约与分块矩阵乘](fundamentals/operators/gpu-kernel-computation-patterns.md) | 从数据依赖理解并行分工、mask、复用和融合。 |
| [线性层与输入通道](fundamentals/operators/linear-layer-input-channel.md) | 线性层把一组输入特征加权组合成输出特征。 |
| [熵、条件熵与依赖：量化代理指标的解释边界](fundamentals/mathematics/entropy-and-dependence.md) | 熵描述一个指定概率分布的不确定性。 |
| [可逆变换、数值条件与 Kronecker 乘积](fundamentals/mathematics/invertible-transforms-and-kronecker-products.md) | 量化前可以改变坐标，使数值更适合有限网格；另一侧施加逆变换，保持浮点计算。 |
| [算术强度与 roofline 分析](fundamentals/hardware/arithmetic-intensity-and-roofline.md) | 「这个算子受什么限制」是决定优化方向的第一问题。 |
| [GPU 的执行模型与存储层次](fundamentals/hardware/gpu-execution-and-memory-hierarchy.md) | 量化为什么有时快、有时只是省内存，答案通常不在算法里，而在数据从哪一级存储搬到哪一类计算单元。 |
| [FP8 与 Microscaling 数值格式](fundamentals/numeric-formats/fp8-and-mx-data-formats.md) | 整数网格不是低精度表示的唯一形式。 |
| [视觉语言模型中的 token 与量化对象](fundamentals/model/vision-language-model-tokens-and-quantization.md) | 视觉语言模型的量化依然使用共同的数值表示、舍入、裁剪、误差重构和低比特计算基础。 |
| [PTQ、QAT 与代理梯度](fundamentals/quantization/post-training-and-quantization-aware-training.md) | PTQ 从已训练模型出发完成量化转换，可能包含统计、搜索或局部优化；QAT 在训练前向中模拟量化影响，并通过训练调整模型。 |
| [均匀量化与分组](fundamentals/quantization/uniform-quantization-and-groups.md) | 均匀量化用等间距的有限数值表示原来的实数或浮点数。 |

### 理论、机制与分析

| 页面 | 主要问题 |
| --- | --- |
| [积分梯度与量化敏感性：基线、路径和归因边界](theory/integrated-gradients-and-quantization-sensitivity.md) | 积分梯度（Integrated Gradients，IG）把一个标量输出相对参考输入的变化分配给输入坐标。 |
| [量化与可靠性：选择性预测评测](theory/quantization-reliability-and-selective-prediction.md) | 多模态模型常常「很有把握地答错」。 |
| [KV cache 量化的对象与粒度](theory/kv-cache-quantization-objects-and-granularity.md) | 权重可以离线反复优化，激活随输入即时产生，KV cache 与两者都不同：它随输入在推理中追加，数值在线产生；策略可以离线校准，历史保留、重算与回收受服务成本约束。 |
| [对角缩放与等价变换](theory/diagonal-scaling-equivalent-transform.md) | 对角缩放通过改变中间特征与权重的数值范围，给量化器提供更合适的输入。 |
| [层输出重构与二阶误差补偿](theory/layer-reconstruction-second-order-compensation.md) | 量化改变权重，但我们关心的是这种改变怎样影响计算结果。 |
| [混合精度分配：选择变量、预算与部署口径](theory/mixed-precision-allocation.md) | 混合精度分配决定不同层、模块或张量用什么量化配置。 |
| [正交旋转与量化：等价条件、离群值和在线代价](theory/orthogonal-rotation-and-hadamard-quantization.md) | 正交旋转通过混合特征坐标改变数值分布，并在相邻运算中抵消变换，目标是让权重和激活更容易量化。 |
| [校准数据与量化范围选择](theory/calibration-and-range-selection.md) | 校准是在有限样本或已有统计上，确定部署所需的量化参数及相关调整。 |

### 具体方法与对照

| 页面 | 主要问题 |
| --- | --- |
| [AdaRound：面向任务损失的自适应舍入](methods/adaround.md) | 权重量化的最后一步通常是最舍入：把每个浮点权重放到最近的网格点上。 |
| [BRECQ：块级重构与二阶误差的粒度选择](methods/brecq.md) | 逐层重构把每层的输出误差压到最小，但在位宽继续下降时反而不再有效：单层看上去接近无损，整网输出却偏得很远。 |
| [AffineQuant：可学习仿射变换与渐进掩码](methods/affinequant.md) | AffineQuant 把量化前的可学习逐通道缩放扩展为可逆矩阵，使一个通道可以与其他通道混合；与逆矩阵配对后，浮点线性计算不变，量化后的网格适配可能更好。 |
| [AWQ、GPTQ 与 SmoothQuant：对象、机制与证据比较](methods/awq-gptq-smoothquant-comparison.md) | 这三种方法都利用少量校准数据改善模型量化，但量化对象、可调整变量与实现路线并不相同。 |
| [AWQ：激活感知的权重量化](methods/awq.md) | AWQ 用校准激活判断哪些输入通道需要更多保护，再通过等价缩放与权重裁剪减轻低比特量化的损失。 |
| [FlatQuant：结构化可学习变换与低比特执行](methods/flatquant.md) | FlatQuant 在量化前学习通道混合，让权重与激活更适合均匀网格；同时用两个小矩阵的 Kronecker 乘积表示大变换，并融合在线变换和量化，控制运行开销。 |
| [GPTQ：基于二阶补偿的权重量化](methods/gptq.md) | GPTQ 对训练好的模型做逐层权重量化。 |
| [KIVI：键按通道、值按 token 的 2 bit KV cache 量化](methods/kivi.md) | KV cache 让推理避免重算历史，代价是显存随批量与上下文线性增长。 |
| [LLM.int8()：向量级量化与混合精度分解](methods/llm-int8.md) | 把 8 bit 量化推向十亿参数规模时，会出现一种在小模型上观察不到的现象：隐藏状态里少数特征维度的幅值远大于其他维度，而且它们系统性地反复出现。 |
| [LUQ：以层激活熵选择超低比特量化层](methods/luq.md) | LUQ 将多模态模型的语言骨干做层间混合精度：先用校准激活的聚类熵给层排序，低熵层优先用 BiLLM 的超低比特权重量化，其余层用 GPTQ 4 bit，再按质量或容量要求选择降位宽的层数。 |
| [MASQuant：分模态平滑与共享权重补偿](methods/masquant.md) | MASQuant 解决一个具体冲突：视觉、文字、音频希望使用不同的逐通道尺度，但共享语言层通常只保存一套主体权重。 |
| [MBQ：模态平衡的视觉语言模型量化](methods/mbq.md) | MBQ 在校准阶段用梯度估计视觉和回答文本的敏感性，再用模态加权的输出误差选择通道缩放。 |
| [MQuant：模态静态量化、等价重排与旋转幅度抑制](methods/mquant.md) | MQuant 把视觉与文本的分布差异落实为两组静态激活网格，再用保持注意力语义的序列重排降低执行开销。 |
| [OmniQuant：可学习裁剪与等价变换的块级校准](methods/omniquant.md) | OmniQuant 冻结原始浮点权重，学习决定量化结果的裁剪强度和等价变换参数。 |
| [Q-VLM：熵驱动的联合校准与视觉编码器优化](methods/q-vlm.md) | Q-VLM 关注“哪些层应放在一起校准”。 |
| [QIG：以量化误差积分梯度指导 token 加权校准](methods/qig.md) | QIG 的问题是：图文 token 在共享语言层中不断交互，只按“视觉/文字”给一个平均重要性，可能掩盖模态内部的差别；直接照搬任务梯度或注意力，也未必测量了量化造成的误差。 |
| [QServe：W4A8KV4 量化与系统协同](methods/qserve.md) | 权重量化在单用户场景常能带来接近理论上限的加速，但云侧服务需要同时处理多请求：计算强度上升，瓶颈从权重读取转向计算与反量化。 |
| [QuaRot：固定旋转、全图等价与四位推理](methods/quarot.md) | QuaRot 先把模型改写到适合量化的坐标系，再量化权重、激活和 KV cache。 |
| [SAW-INT4：面向真实服务约束的 4 bit KV cache 量化](methods/saw-int4.md) | 许多 KV cache 压缩方法在离线精度与压缩率上表现不错，接入生产推理引擎后却拿不到收益。 |
| [SmoothQuant：迁移激活量化难度的 W8A8 方法](methods/smoothquant.md) | SmoothQuant 面向同时量化权重与激活的 W8A8 推理。 |
| [SpinQuant：用最终模型损失学习正交旋转](methods/spinquant.md) | SpinQuant 的出发点是：不同正交旋转在浮点下等价，在量化后却可能相差很大。 |
| [SplitQ：模态离群通道拆分与权重激活双补偿](methods/splitq.md) | SplitQ 把多模态量化冲突拆成两个层次：少数模态离群通道先单独处理；剩余共享通道再分别缓解权重量化误差与激活量化误差。 |
| [VLMQ：token 重要性加权的二阶量化](methods/vlmq.md) | VLMQ 用局部注意力模块重构损失的梯度估计各 token 的重要性，再把这些因子放进线性层的二阶重构目标。 |
| [ZeroQuant：逐组权重、逐 token 激活与逐层蒸馏](methods/zeroquant.md) | 8 bit 量化到十亿参数规模时，精度损失主要来自两个数值事实：不同 token 的激活范围相差很大，不同输出行的权重范围也相差很大。 |

### 研究抽象

| 页面 | 主要问题 |
| --- | --- |
| [MBQ 与 VLMQ：重要性怎样进入量化目标](research/mbq-vlmq-comparison.md) | 两种方法都认为，直接平等处理全部 token 的重构误差，可能没有准确表达视觉语言任务的需求。 |
| [OmniQuant、AffineQuant 与 FlatQuant：可学习变换路线比较](research/omniquant-affinequant-flatquant-comparison.md) | 这三篇共同研究：冻结基座权重，以少量校准数据学习量化前的表示与网格，减少 Transformer block 输出误差。 |

### 工程实现与验证

| 页面 | 主要问题 |
| --- | --- |
| [模型质量评测](implementation/model-quality-evaluation.md) | PPL 的有效计数、任务协议与风险—覆盖率怎样定义和比较。 |
| [推理服务评测](implementation/serving-performance-evaluation.md) | 负载、TTFT、TPOT、吞吐与 SLO 下的 goodput 如何共同解释量化收益。 |
| [张量并行与量化](implementation/tensor-parallel-quantization.md) | 列/行分片、GQA 复制、collective 与 group/scale/zero 如何配合。 |
| [Tensor Core 与量化 GEMM](implementation/tensor-core-quantized-gemm.md) | 从 tile、lane 和寄存器走到 ldmatrix、MMA、AWQ Triton 与编译诊断。 |
| [vLLM 推理执行：从 token 调度到变长批次](implementation/vllm-inference-execution.md) | 请求如何分配 token 与 KV 空间，再变成算子的紧凑输入。 |
| [vLLM 算子设计：Attention 后端、分页计算与图执行](implementation/vllm-attention-operator-design.md) | 分页状态与元数据如何进入 Attention，设备端怎样复用、归约并适配图执行。 |
| [SGLang 推理执行：Radix 缓存、批次调度与重叠执行](implementation/sglang-inference-execution.md) | 前缀复用如何连接缓存保护、请求映射、批次与跨轮结果。 |
| [SGLang 算子设计：索引化 KV、Extend 与 Decode 归约](implementation/sglang-attention-operator-design.md) | 历史前缀和当前输入如何进入 Attention，分段输出怎样按 LSE 合并。 |
| [门控激活的 CUDA 与 Triton 实现](implementation/gated-activation-cuda-triton-kernels.md) | 从两段输入布局理解独立输出、向量任务、二维 program 网格与融合边界。 |
| [RMSNorm 的 CUDA 与 Triton 实现](implementation/rmsnorm-cuda-triton-kernels.md) | 从整行统计理解线程协作、分段读取、残差写回与舍入位置。 |
| [Kernel 配置选择与自动调优：资源、工作划分和测量](implementation/kernel-configuration-and-autotuning.md) | 如何权衡数据复用、资源预算、并行工作量与调优成本。 |
| [GPU 异步拷贝与多级流水：等待完成和缓冲复用](implementation/gpu-async-copy-pipelines.md) | 数据何时可以读取，缓冲何时可以覆盖，怎样安全地重叠搬运和计算。 |
| [算子正确性与性能测量：参考结果、误差和计时边界](implementation/kernel-correctness-and-benchmarking.md) | 怎样选择验证用例，并正确解释 GPU 计时和吞吐。 |
| [GGUF 与块量化存储格式](implementation/gguf-block-quantization-formats.md) | 论文里的量化方案要落到文件，还需要回答一组工程问题：权重怎么分块、每块保存哪些元数据、文件如何描述张量与模型、量化工具怎样按张量选择格式、以及校准统计在哪里参与。 |
| [量化模型的部署框架与后端支持](implementation/quantized-llm-deployment-backends.md) | 量化方法给出的是算法与一串参数；要让模型真正跑起来，还需要格式、加载路径、内核与运行场景四段配置同时对上。 |
| [推理服务的内存管理与批处理](implementation/serving-memory-and-batching.md) | 量化把每个权重、每个缓存元素压小，但能同时服务多少请求，取决于服务系统怎么用这些省下来的空间。 |
| [Marlin：批处理 W4A16 GEMM 内核](implementation/marlin-batched-w4a16-gemm.md) | 权重量化的加速通常以「生成阶段受权重读取带宽限制」为前提。 |
| [量化矩阵乘法的缩放与执行路径](implementation/quantized-matmul-scaling-execution.md) | 同样写着低比特量化，实际可能是浮点模拟、压缩权重配合即时反量化，或整数输入参与矩阵乘法。 |
| [权重量化反量化内核的契约：AWQ 与 marlin 两种实现](implementation/weight-only-dequant-kernels.md) | W4A16 的执行可以概括成一句话：权重以低比特打包存放，内核在读回它们时反量化到 fp16 再参与矩阵乘。 |
| [VLM 压缩评测框架：从任务精度到可信度](implementation/lvlm-compression-benchmark.md) | 多模态模型压缩的评测常停在几个 VQA 分数上。 |
| [量化误差诊断与验证](implementation/quantization-error-diagnosis.md) | 量化后结果变差，可能来自图变换错误、网格与分布不匹配、局部误差传播，也可能来自整数后端与模拟规则不一致。 |
| [AWQ 的实现核对：从量化脚本到 vLLM 与 SGLang](implementation/awq-implementation.md) | 方法页回答 AWQ 为什么这样设计、证据支持什么；本页回答代码层面的事情：量化脚本产出什么形状的权重、打包约定是什么、内核从哪里进入、同一个 AWQ checkpoint 在 vLLM 与 SGLang 里各走哪条路径。 |
| [GPTQ 的实现核对：量化器、打包与两套服务路径](implementation/gptq-implementation.md) | GPTQ 的论文讲的是「固定一个权重后怎样补偿剩下的」，而工程上要回答的是另外三件事：校准与量化参数在哪里算出来、权重被打成什么位布局、服务框架拿到这份 checkpoint 后认不认。 |
| [SmoothQuant 的实现核对：平滑融合、INT8 接口与离线边界](implementation/smoothquant-implementation.md) | SmoothQuant 的公式只有一行（`WX = (WD)(D⁻¹X)`），但工程上要回答的是：平滑因子在哪里算、按什么统计算、融合到哪些参数上、以及服务框架拿到这份 checkpoint 时还要不要做别的事。 |
