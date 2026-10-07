---
title: 量化知识地图
type: index
sources: []
---

# 量化知识地图

从问题选择阅读路径。页面按主要职责归入五类目录，同一知识可以出现在不同阅读路径中。整体定位、职责边界、当前画像与下一步见 [SELF](../SELF.md)，维护规范见 [README](README.md)，变动记录见 [LOG](LOG.md)。

按资料查看读到哪里、知识整合到哪里及尚未核查内容，见 [资料研读目录](READING.md)。

## 按问题选择量化路线

先确定量化对象和目标，再沿对应机制查看可调整变量、条件与实际执行。下表是阅读入口，具体配置仍需共同协议比较。

| 目标或约束 | 机制与路线入口 | 实现或评测入口 |
| --- | --- | --- |
| 降低权重占用，激活保持高精度 | [AWQ/GPTQ 的不同重构目标](methods/awq-gptq-smoothquant-comparison.md)、[极低比特表示设计](research/low-bit-representation-design.md) | [权重反量化 kernel 契约](implementation/weight-only-dequant-kernels.md) |
| 权重与激活联合量化 | [校准与网格](theory/calibration-and-range-selection.md)、[缩放](theory/diagonal-scaling-equivalent-transform.md)、[旋转](theory/orthogonal-rotation-and-hadamard-quantization.md)、[可学习变换](research/omniquant-affinequant-flatquant-comparison.md) | [尺度位置与矩阵乘执行](implementation/quantized-matmul-scaling-execution.md) |
| 降低长上下文或多请求的缓存占用 | [KV 对象/粒度](theory/kv-cache-quantization-objects-and-granularity.md)、[误差传播](theory/attention-quantization-error-propagation.md)、[WUSH-KV 的双侧统计](methods/wush-kv.md) | [缓存生命周期](theory/kv-cache-quantization-objects-and-granularity.md#11-从位宽到可执行存储格式)、[服务内存与批处理](implementation/serving-memory-and-batching.md) |
| 在总预算内选择层或对象的精度 | [混合精度分配](theory/mixed-precision-allocation.md)、[曲率方向](theory/curvature-weighted-quantization-error.md)、[CASA 的配置交互](methods/casa.md) | [服务性能协议](implementation/serving-performance-evaluation.md) |
| 保留多模态信息与生成行为 | [模态目标比较](research/mbq-vlmq-comparison.md)、[OnPTQ 的状态与后果](methods/onptq.md) | [正确性、分布与生成评测](implementation/model-quality-evaluation.md) |
| 判断输出分布偏移与原模型不一致 | [间隔、KL/TV 与 token 翻转](theory/quantization-ranking-stability.md#8-语言模型的分布距离怎样变成-token-翻转)、[可靠性与弃答](theory/quantization-reliability-and-selective-prediction.md) | [相同前缀与最终任务的协议](implementation/model-quality-evaluation.md#9-输出分布一致性要固定前缀和指标口径) |
| 确认方案能否导出、运行并获得收益 | [后端支持边界](implementation/quantized-llm-deployment-backends.md)、[格式与执行](implementation/quantized-matmul-scaling-execution.md) | [模拟与真实执行诊断](implementation/quantization-error-diagnosis.md)、[计时与性能测量](implementation/kernel-correctness-and-benchmarking.md) |

## 从哪里开始

- **理解秩、位宽与前缀预算**：[混合精度计费](theory/mixed-precision-allocation.md) → [DBW 的成对分量](methods/dbw.md) / [StagQ 的共享前缀](methods/stagq.md) → [主体与例外的表示选择](research/low-bit-representation-design.md)；分清名义精度、保存字段、实际读取和训练/推理工作张量。
- **贯通 MoE 路由与执行**：[专家选择、容量与误差](fundamentals/model/mixture-of-experts-and-routing.md) → [配对排序、FP8 专家与共享输出](implementation/moe-routing-and-quantized-expert-execution.md) → [EP 映射、通信与输出回传](implementation/moe-expert-parallel-execution.md) → [DeepEP 设备去重、FP8 布局与回源](implementation/deepep-moe-communication-execution.md) → [DeepGEMM 专家行选择、scale 与 SM90 计算](implementation/deepgemm-moe-expert-execution.md)；分清模型 gate、本地参数轴、接收行/专家配对、通信尺度和汇合职责。
- **贯通 MLA 结构与执行**：[联合潜变量与位置分支](fundamentals/model/multi-head-latent-attention.md) → [SGLang 的请求与槽位](implementation/sglang-inference-execution.md) → [投影吸收、联合缓存与 Attention 读取](implementation/mla-cache-and-attention-execution.md)，再按问题进入 [vLLM DCP 分片与跨 rank 合并](implementation/vllm-mla-context-parallel-execution.md)或[packed 缓存与 DSA 选择/读取](implementation/vllm-mla-packed-cache-and-dsa-execution.md)，再接[DSA 设备 logits 与 sparse attention](implementation/dsa-sparse-device-execution.md)；分别核对持久状态、工作张量、通信轴、候选集合与精度转换。
- **理解重构的优化行为**：[层输出重构与二阶补偿](theory/layer-reconstruction-second-order-compensation.md) → [损失尺度、分组 RMSE 与参数更新](theory/layer-reconstruction-second-order-compensation.md#11-重构损失的尺度怎样影响优化)；区分改变重构对象、损失归约和可训练参数。
- **理解低位宽的尺度选择**：[校准与范围选择](theory/calibration-and-range-selection.md) → [高斯极限、相对曲率与 GPTQ 尺度](theory/calibration-and-range-selection.md#10-极低位宽为什么更依赖尺度选择) → [旋转的等价条件与代价](theory/orthogonal-rotation-and-hadamard-quantization.md)。
- **选择旋转的校准目标**：[正交变换的等价条件](theory/orthogonal-rotation-and-hadamard-quantization.md) → [SpinQuant 的模型损失](methods/spinquant.md) / [ConQuR 的顶点距离与 Procrustes 更新](methods/conqur.md)；结合可融合位置、量化器和校准成本判断。
- **选择 KV 的误差度量与变换**：[K/V 的传播路径](theory/attention-quantization-error-propagation.md) → [NOVA-KV 的 query 加权](methods/nova-kv.md) / [WUSH-KV 的分布与敏感度平衡](methods/wush-kv.md) → [格式、窗口与实际执行](theory/kv-cache-quantization-objects-and-granularity.md#11-从位宽到可执行存储格式)。
- **解释分布误差与生成决策**：[间隔与排序稳定性](theory/quantization-ranking-stability.md) → [KL、TV 与 token 翻转](theory/quantization-ranking-stability.md#8-语言模型的分布距离怎样变成-token-翻转) → [同前缀与实际任务评测](implementation/model-quality-evaluation.md#9-输出分布一致性要固定前缀和指标口径)。
- **选择生成校准的状态与目标**：[校准数据与前缀覆盖](theory/calibration-and-range-selection.md#11-自回归校准还需要选择前缀状态) → [OnPTQ 的决策与分支后果](methods/onptq.md) → [静态 token 权重与生成状态的比较](research/mbq-vlmq-comparison.md#9-从静态-token-重要性到生成状态与后果)。

- **理解模型与数值**：[Transformer 与自回归推理](fundamentals/model/transformer-autoregressive-inference.md) → [浮点表示与累加](fundamentals/numeric-formats/floating-point-and-accumulation.md)。
- **理解 MoE/MLA 与量化对象**：[稠密 decoder 主干](fundamentals/model/transformer-autoregressive-inference.md) → [MoE 的专家选择与容量](fundamentals/model/mixture-of-experts-and-routing.md) / [MLA 的联合缓存与位置分支](fundamentals/model/multi-head-latent-attention.md) → [排序稳定性](theory/quantization-ranking-stability.md) / [KV cache 量化](theory/kv-cache-quantization-objects-and-granularity.md)。
- **建立共同基础**：[线性层与输入通道](fundamentals/operators/linear-layer-input-channel.md) → [均匀量化与分组](fundamentals/quantization/uniform-quantization-and-groups.md) → [PTQ、QAT 与代理梯度](fundamentals/quantization/post-training-and-quantization-aware-training.md) → [校准数据与量化范围选择](theory/calibration-and-range-selection.md)。
- **理解量化训练**：[PTQ、QAT 与代理梯度](fundamentals/quantization/post-training-and-quantization-aware-training.md) → [LSQ 的步长学习](methods/lsq.md) → [LLM-QAT 的生成数据蒸馏](methods/llm-qat.md) → [EfficientQAT 的分阶段训练](methods/efficientqat.md)。
- **理解低秩误差修正**：[激活加权低秩近似](fundamentals/mathematics/invertible-transforms-and-kronecker-products.md#6-激活加权的低秩近似) → [LoftQ 的交替初始化](methods/loftq.md) → [QERA 的解析补偿](methods/qera.md) → [MASQuant 的模态残差](methods/masquant.md) / [SplitQ 的双补偿](methods/splitq.md) → [低秩分支的执行成本](implementation/quantized-matmul-scaling-execution.md#12-低秩辅助分支不是免费或统一位宽)。
- **理解量化方法**：[AWQ](methods/awq.md) → [GPTQ](methods/gptq.md) → [SmoothQuant](methods/smoothquant.md) → [AWQ、GPTQ 与 SmoothQuant](methods/awq-gptq-smoothquant-comparison.md)。
- **理解变换与研究抽象**：[对角缩放与等价变换](theory/diagonal-scaling-equivalent-transform.md) → [OmniQuant](methods/omniquant.md) → [AffineQuant](methods/affinequant.md) → [FlatQuant](methods/flatquant.md) → [OmniQuant、AffineQuant 与 FlatQuant](research/omniquant-affinequant-flatquant-comparison.md) → [QuaRot](methods/quarot.md) → [SpinQuant](methods/spinquant.md) → [OSTQuant 的正交与缩放联合学习](methods/ostquant.md)。
- **理解极低比特表示**：[二阶重构](theory/layer-reconstruction-second-order-compensation.md) → [QuIP](methods/quip.md) → [码本量化](theory/codebook-quantization-and-bit-budget.md) → [QuIP#](methods/quip-sharp.md) → [AQLM](methods/aqlm.md) → [SpQR](methods/spqr.md) / [SqueezeLLM](methods/squeezellm.md) → [坐标、码本与例外预算](research/low-bit-representation-design.md)。
- **研究多模态量化**：[视觉语言模型中的 token 与量化对象](fundamentals/model/vision-language-model-tokens-and-quantization.md) → [MBQ](methods/mbq.md) → [VLMQ](methods/vlmq.md) → [MBQ 与 VLMQ](research/mbq-vlmq-comparison.md) → [QIG](methods/qig.md) → [LUQ](methods/luq.md)。
- **学习算子开发**：[张量布局与接口](fundamentals/operators/tensor-layout-and-kernel-contracts.md) → [GPU 执行与存储](fundamentals/hardware/gpu-execution-and-memory-hierarchy.md) → [CUDA 编写、编译与接入](implementation/cuda-operator-build-and-integration.md) / [Triton 语言与编译语义](fundamentals/operators/triton-language-and-compilation.md) → [逐元素、归约与分块矩阵乘](fundamentals/operators/gpu-kernel-computation-patterns.md) → [正确性与性能测量](implementation/kernel-correctness-and-benchmarking.md)。
- **从算法走向部署**：[量化矩阵乘法的缩放与执行路径](implementation/quantized-matmul-scaling-execution.md) → [AWQ 的实现核对](implementation/awq-implementation.md) → [GPTQ 的实现核对](implementation/gptq-implementation.md) → [SmoothQuant 的实现核对](implementation/smoothquant-implementation.md) → [QServe/QoQ 的实现核对](implementation/qserve-implementation.md) → [权重量化反量化内核的契约](implementation/weight-only-dequant-kernels.md) → [量化模型的部署框架与后端支持](implementation/quantized-llm-deployment-backends.md)。
- **从推理库学习 CUDA 与 Triton**：[工作映射与张量接口](fundamentals/operators/tensor-layout-and-kernel-contracts.md) → [vLLM/SGLang 门控激活](implementation/gated-activation-cuda-triton-kernels.md) → [RMSNorm 行归约与残差融合](implementation/rmsnorm-cuda-triton-kernels.md) → [Attention 算子设计](implementation/sglang-attention-operator-design.md)。
- **深入矩阵乘与多卡执行**：[Tensor Core 与量化 GEMM](implementation/tensor-core-quantized-gemm.md) → [W8A8 量化、GEMM 与校正](implementation/w8a8-quantization-gemm-kernels.md) → [配置与自动调优](implementation/kernel-configuration-and-autotuning.md) → [张量并行与量化](implementation/tensor-parallel-quantization.md)。
- **贯通低精度浮点执行**：[FP8 与 Microscaling 格式](fundamentals/numeric-formats/fp8-and-mx-data-formats.md) → [K 尺度与执行路径](implementation/quantized-matmul-scaling-execution.md) → [FP4/FP8 块缩放的编码、布局与调用](implementation/block-scaled-matmul-execution.md) → [Tensor Core 与编译诊断](implementation/tensor-core-quantized-gemm.md)。
- **优化算子配置与流水**：[GPU 算子的计算模式](fundamentals/operators/gpu-kernel-computation-patterns.md) → [Kernel 配置选择与自动调优](implementation/kernel-configuration-and-autotuning.md) → [异步拷贝与多级流水](implementation/gpu-async-copy-pipelines.md) → [正确性与性能测量](implementation/kernel-correctness-and-benchmarking.md)。
- **建立算子测试与性能分析流程**：[正确性、Sanitizer 与计时](implementation/kernel-correctness-and-benchmarking.md) → [时间线、硬件指标与瓶颈定位](implementation/gpu-kernel-performance-analysis.md) → [配置选择与对照实验](implementation/kernel-configuration-and-autotuning.md)。
- **贯通量化的公共知识**：[二阶统计与加权距离](fundamentals/mathematics/invertible-transforms-and-kronecker-products.md#7-二阶统计如何定义误差距离) → [Attention 误差传播](theory/attention-quantization-error-propagation.md) → [码本与率失真](theory/codebook-quantization-and-bit-budget.md#10-从失真目标推导标量量化器) → [缓存格式与生命周期](theory/kv-cache-quantization-objects-and-granularity.md#11-从位宽到可执行存储格式) → [模拟到实际执行](implementation/quantization-error-diagnosis.md#7-低比特-kv-从模拟到真实执行的对齐)。
- **理解 KV cache 与服务**：[KV cache 量化的对象与粒度](theory/kv-cache-quantization-objects-and-granularity.md) → [KIVI](methods/kivi.md) → [KVQuant](methods/kvquant.md) / [QJL](methods/qjl.md) → [SAW-INT4](methods/saw-int4.md) → [推理服务的内存管理与批处理](implementation/serving-memory-and-batching.md)。
- **理解推理框架与算子设计**：[内存管理与批处理](implementation/serving-memory-and-batching.md) → [vLLM 的 token 调度与变长批次](implementation/vllm-inference-execution.md) → [vLLM Attention 后端、分页计算与图执行](implementation/vllm-attention-operator-design.md) → [AWQ 实现](implementation/awq-implementation.md) → [正确性与性能测量](implementation/kernel-correctness-and-benchmarking.md)。
- **理解 SGLang 的缓存与算子**：[Radix 缓存、批次与重叠执行](implementation/sglang-inference-execution.md) → [KV 索引、Extend 与 Decode 归约](implementation/sglang-attention-operator-design.md) → [AWQ 的 SGLang 路径](implementation/awq-implementation.md#6-跨引擎sglang)。
- **判断误差与运行收益**：[量化误差诊断与验证](implementation/quantization-error-diagnosis.md) → [模型质量评测](implementation/model-quality-evaluation.md) → [量化与可靠性](theory/quantization-reliability-and-selective-prediction.md) → [服务性能评测](implementation/serving-performance-evaluation.md)。
- **理解误差方向与配置选择**：[激活引导补偿](methods/activation-guided-compensation.md) → [曲率加权误差](theory/curvature-weighted-quantization-error.md) / [RGSQ](methods/rgsq.md) → [HAWQ-V2](methods/hawq-v2.md) → [CASA](methods/casa.md) → [Quantization Price](methods/quantization-price-prediction.md)。
- **连接编码与阶段执行**：[码本量化](theory/codebook-quantization-and-bit-budget.md) → [QTIP](methods/qtip.md) → [SPHQuant](methods/sphquant.md) → [Disaggregated Quantization](methods/disaggregated-quantization.md)。

- **诊断平均分之外的损害**：[排序稳定性与间隔](theory/quantization-ranking-stability.md) → [多模态对齐诊断](theory/multimodal-alignment-diagnostics.md) → [模型质量评测](implementation/model-quality-evaluation.md)。
- **连接低比特方案与真实执行**：[PRQuant 的残差拼接](methods/prquant.md) → [RAMP 的 CPU 图成本](methods/ramp.md) → [FoldQuantVLA 的共享输入与整数投影](methods/foldquantvla.md)。

- **理解视觉 token 删除**：[视觉 token 基础](fundamentals/model/vision-language-model-tokens-and-quantization.md) → [CoRePrune 的条件可删除性](methods/coreprune.md) → [LayerPos 的位置策略](methods/layerpos.md) → [QAPruner 的量化感知选择](methods/qapruner.md)。
- **核验校准图谱迁移**：[排序稳定性](theory/quantization-ranking-stability.md) → [BOS-Sink 图谱](theory/attention-sink-map-transfer.md)。
- **评估跨架构低比特部署**：[混合精度分配](theory/mixed-precision-allocation.md) → [TTS 敏感性与质量判据](implementation/tts-quantization-evaluation.md) → [缩放与真实执行](implementation/quantized-matmul-scaling-execution.md)。

- **比较低比特 KV 的目标与布局**：[QJL](methods/qjl.md) → [TurboQuant](methods/turboquant.md) → [NOVA-KV](methods/nova-kv.md) → [Kitty](methods/kitty.md)；多模态保护见 [AKVQ-VL](methods/akvq-vl.md)。

## 全部知识页

### 公共基础

| 页面 | 主要问题 |
| --- | --- |
| [Transformer 与自回归推理](fundamentals/model/transformer-autoregressive-inference.md) | 从 token、QKV、残差和 LM head 解释 prefill/decode、GQA 与 KV 复用。 |
| [MoE 与专家路由](fundamentals/model/mixture-of-experts-and-routing.md) | 共享与路由专家、设备限制、负载、总容量与量化路由误差怎样连接。 |
| [MLA：联合潜变量、位置分支与缓存表示](fundamentals/model/multi-head-latent-attention.md) | 联合压缩、矩阵吸收、解耦 RoPE 如何改变量化对象与缓存预算。 |
| [浮点表示与累加](fundamentals/numeric-formats/floating-point-and-accumulation.md) | FP16/BF16/FP32 的范围、有效数字、FMA 和归约顺序怎样影响结果。 |
| [张量布局与算子接口：从逻辑下标到内存地址](fundamentals/operators/tensor-layout-and-kernel-contracts.md) | shape、stride、dtype 和执行接口怎样决定正确读写。 |
| [Triton 语言与编译语义](fundamentals/operators/triton-language-and-compilation.md) | 逻辑张量、广播、mask、类型与 JIT 特化怎样影响设备代码。 |
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
| [Attention sink 选头图谱](theory/attention-sink-map-transfer.md) | 全局相关、集合边界、域迁移及重校准的证据范围。 |
| [量化后的排序稳定性](theory/quantization-ranking-stability.md) | 分数间隔、原有正确结果损失、校准保证与检索精度分配。 |
| [多模态对齐诊断](theory/multimodal-alignment-diagnostics.md) | 共享投影、主角度谱、视觉干预与几何代理的边界。 |
| [曲率加权量化误差](theory/curvature-weighted-quantization-error.md) | Fisher 与经验代理的区别、双侧度量、阻尼及量化集合的坐标变换。 |
| [码本量化：标量、向量、加性表示与位宽预算](theory/codebook-quantization-and-bit-budget.md) | 码字集合、激活感知距离、残差编码和真实存储开销怎样联系。 |
| [积分梯度与量化敏感性：基线、路径和归因边界](theory/integrated-gradients-and-quantization-sensitivity.md) | 积分梯度（Integrated Gradients，IG）把一个标量输出相对参考输入的变化分配给输入坐标。 |
| [量化与可靠性：选择性预测评测](theory/quantization-reliability-and-selective-prediction.md) | 多模态模型常常「很有把握地答错」。 |
| [Attention 量化误差传播](theory/attention-quantization-error-propagation.md) | 从查询加权误差、softmax 和 Value 交互，理解局部重构为何不能直接代表生成质量。 |
| [KV cache 量化的对象与粒度](theory/kv-cache-quantization-objects-and-granularity.md) | 权重可以离线反复优化，激活随输入即时产生，KV cache 与两者都不同：它随输入在推理中追加，数值在线产生；策略可以离线校准，历史保留、重算与回收受服务成本约束。 |
| [对角缩放与等价变换](theory/diagonal-scaling-equivalent-transform.md) | 对角缩放通过改变中间特征与权重的数值范围，给量化器提供更合适的输入。 |
| [层输出重构与二阶误差补偿](theory/layer-reconstruction-second-order-compensation.md) | 量化改变权重，但我们关心的是这种改变怎样影响计算结果。 |
| [混合精度分配：选择变量、预算与部署口径](theory/mixed-precision-allocation.md) | 混合精度分配决定不同层、模块或张量用什么量化配置。 |
| [正交旋转与量化：等价条件、离群值和在线代价](theory/orthogonal-rotation-and-hadamard-quantization.md) | 正交旋转通过混合特征坐标改变数值分布，并在相邻运算中抵消变换，目标是让权重和激活更容易量化。 |
| [校准数据与量化范围选择](theory/calibration-and-range-selection.md) | 校准是在有限样本或已有统计上，确定部署所需的量化参数及相关调整。 |

### 具体方法与对照

| 页面 | 主要问题 |
| --- | --- |
| [QAPruner：量化与 token 选择](methods/qapruner.md) | 误差/范围代理、语义融合、删除重要性与局部 scale 边界。 |
| [DBW：SVD 分量的秩与位宽](methods/dbw.md) | 成对分量预算、Hamilton/STE、训练/导出与固定代码的浮点恢复路径。 |
| [StagQ：共享前缀与稀疏保护](methods/stagq.md) | dyadic refinement 的 affine 解码、共享 pair、pre/post guard 与逻辑/读取 rate。 |
| [TurboQuant：旋转码本与残差修正](methods/turboquant.md) | 球面边缘、MSE 与无偏内积、有限维下界及位宽核对。 |
| [NOVA-KV：查询加权变换与分组 VQ](methods/nova-kv.md) | 精确加权距离、等体积分组、校准与编码/读取成本。 |
| [AKVQ-VL：多模态 token 精度保护](methods/akvq-vl.md) | TSA/PSA、pivot 代理、Hadamard 与实际混合位宽。 |
| [Kitty：动态通道提精度与缓存页](methods/kitty.md) | 幅值选通道、双 2-bit 位面、模拟/内核差异与真实字节。 |
| [CoRePrune：条件可删除性](methods/coreprune.md) | 删除失真、集合方向交互、深度刷新与两阶段视觉剪枝。 |
| [LayerPos：逐层位置策略](methods/layerpos.md) | 稀疏/连续编号、定位校准和 prefill/decode 的坐标一致性。 |
| [PRQuant：置换与低比特残差拼接](methods/prquant.md) | 门控 FFN 的等价重排、尺度位置、残差与实际扩宽成本。 |
| [RAMP：CPU 混合精度与图成本](methods/ramp.md) | 单层 logits 敏感度、质心阈值和融合损失怎样影响选择。 |
| [FoldQuantVLA：共享输入折叠与整数投影](methods/foldquantvla.md) | 变换顺序、动态范围、代码执行路径与动作/闭环证据。 |
| [KVQuant：pre-RoPE 与敏感度码本](methods/kvquant.md) | 静态通道尺度、非均匀码本、稀疏离群值和真实缓存位宽。 |
| [QJL：二值投影与分数估计](methods/qjl.md) | 非对称无偏估计、概率保证边界、范数开销与单层测速。 |
| [激活引导补偿与正交残差](methods/activation-guided-compensation.md) | 权重补偿的可达空间、随机符号残差界与 L2 缩放的代理推导。 |
| [RGSQ：模态曲率与稀疏旋转](methods/rgsq.md) | 双侧误差、模态融合及白化量化的成立边界与代码差异。 |
| [HAWQ-V2：trace 与混合精度](methods/hawq-v2.md) | 如何估计平均曲率、压缩配置搜索并结合 QAT；方向假设与 Pareto 边界。 |
| [CASA：方向敏感分配与跨层交换](methods/casa.md) | 标量失真界、对角因子 MCKP、跨层上界提案及校准接受的保证边界。 |
| [Quantization Price：部署前配置定价](methods/quantization-price-prediction.md) | forward KL 的局部价格、trace 简化条件、配置预算与实验实现差别。 |
| [QTIP：trellis 高维量化](methods/qtip.md) | 路径表示、Viterbi、并行位窗口解码、尾咬合与有效位宽。 |
| [SPHQuant：球面方向与半径编码](methods/sphquant.md) | 半径投影、角度训练、实际2.25位布局与吞吐/内存取舍。 |
| [Disaggregated Quantization：分阶段格式与权重](methods/disaggregated-quantization.md) | QADD、两阶段状态衔接、ODP借用恢复及真实计时边界。 |
| [LLM-QAT：生成数据蒸馏与联合量化](methods/llm-qat.md) | 前缀覆盖、软标签与 W/A/KV 量化怎样共同进入训练。 |
| [QERA：激活二阶矩加权的低秩量化误差修正](methods/qera.md) | 固定量化主体下的最佳低秩修正、对角近似条件与初始化／推理成本。 |
| [LoftQ：量化主体与低秩适配器的交替初始化](methods/loftq.md) | 先减去旧补偿再量化、重算残差 SVD，最后冻结主体做任务适配。 |
| [OSTQuant：正交与缩放联合学习、QSUR 与 KL-Top](methods/ostquant.md) | 哪些变换可以学习与融合，分布体积指标、教师损失和量化效果怎样区分。 |
| [EfficientQAT：逐块全参数训练与整网尺度微调](methods/efficientqat.md) | 局部自由度、固定整数编码与整网目标怎样降低训练成本。 |
| [LSQ：任务驱动的步长学习与梯度缩放](methods/lsq.md) | 舍入位置、共享尺度梯度与训练协议怎样共同决定网格。 |
| [QuIP：非相干处理与 LDLQ](methods/quip.md) | 谱方向、线性反馈与有限网格的保证边界。 |
| [QuIP#：RHT、BlockLDLQ 与 E8P](methods/quip-sharp.md) | 快速变换、格编码与微调如何兼顾质量和解码成本。 |
| [AQLM：激活感知加性量化](methods/aqlm.md) | 多个学习码本、联合索引和块内微调怎样工作。 |
| [SpQR：敏感权重与两级元数据](methods/spqr.md) | 动态例外选择、小组网格和稀疏修正怎样计账。 |
| [SqueezeLLM：加权非均匀量化](methods/squeezellm.md) | 任务 Fisher、标量 LUT 与稠密稀疏分解如何配合。 |
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
| [SAW-INT4：面向真实服务约束的 4 bit KV cache 量化](methods/saw-int4.md) | 块旋转如何融入 INT4 写入、GQA decode 与 prefill 恢复；持久记录、舍入位置和临时缓冲如何区分。 |
| [SmoothQuant：迁移激活量化难度的 W8A8 方法](methods/smoothquant.md) | SmoothQuant 面向同时量化权重与激活的 W8A8 推理。 |
| [SpinQuant：用最终模型损失学习正交旋转](methods/spinquant.md) | SpinQuant 的出发点是：不同正交旋转在浮点下等价，在量化后却可能相差很大。 |
| [ConQuR：顶点对齐与逐批旋转校准](methods/conqur.md) | 激活几何代理、Procrustes 闭式子问题、量化感知前向与可融合边界。 |
| [WUSH-KV：缓存统计与敏感度的可逆变换](methods/wush-kv.md) | 闭式平衡、GQA 读取度量、post-RoPE 配对、双窗口与两套量化器的边界。 |
| [OnPTQ：生成决策与未来分支校准](methods/onptq.md) | 同量化前缀、风险选择、context anchor 与轨迹刷新；代理理论、正确性 flip 和执行条件。 |
| [SplitQ：模态离群通道拆分与权重激活双补偿](methods/splitq.md) | SplitQ 把多模态量化冲突拆成两个层次：少数模态离群通道先单独处理；剩余共享通道再分别缓解权重量化误差与激活量化误差。 |
| [VLMQ：token 重要性加权的二阶量化](methods/vlmq.md) | VLMQ 用局部注意力模块重构损失的梯度估计各 token 的重要性，再把这些因子放进线性层的二阶重构目标。 |
| [ZeroQuant：逐组权重、逐 token 激活与逐层蒸馏](methods/zeroquant.md) | 8 bit 量化到十亿参数规模时，精度损失主要来自两个数值事实：不同 token 的激活范围相差很大，不同输出行的权重范围也相差很大。 |

### 研究抽象

| 页面 | 主要问题 |
| --- | --- |
| [低比特表示的设计空间](research/low-bit-representation-design.md) | 坐标、码本、目标和例外预算构成哪些可检验选择。 |
| [MBQ 与 VLMQ：重要性怎样进入量化目标](research/mbq-vlmq-comparison.md) | 两种方法都认为，直接平等处理全部 token 的重构误差，可能没有准确表达视觉语言任务的需求。 |
| [OmniQuant、AffineQuant 与 FlatQuant：可学习变换路线比较](research/omniquant-affinequant-flatquant-comparison.md) | 这三篇共同研究：冻结基座权重，以少量校准数据学习量化前的表示与网格，减少 Transformer block 输出误差。 |

### 工程实现与验证

| 页面 | 主要问题 |
| --- | --- |
| [OSCAR 混合 KV 缓存执行](implementation/oscar-mixed-kv-cache-execution.md) | affine INT2码/浮点元数据、BF16 sink/recent更新、QP/HP索引、读取舍入与LSE合并怎样配套。 |
| [MLA 缓存与 Attention 执行](implementation/mla-cache-and-attention-execution.md) | 潜变量/RoPE 的联合槽位、投影吸收、展开与潜空间路径、读取索引及精度转换怎样衔接。 |
| [vLLM MLA 的 DCP 执行](implementation/vllm-mla-context-parallel-execution.md) | 全局 token 到本地 page、query head 收集、局部 LSE 合并、AG+RS/A2A 返回及 prefix prefill 的缓存收集怎样衔接。 |
| [vLLM MLA packed 缓存与 DSA](implementation/vllm-mla-packed-cache-and-dsa-execution.md) | 656-byte 主记录、独立 indexer 的整页 key/scale、因果 top-k、物理槽位与 BF16 sparse prefill 怎样衔接。 |
| [DSA 的设备执行](implementation/dsa-sparse-device-execution.md) | DeepGEMM 的 FP8 logits、因果选择范围，以及 FlashMLA 的 packed/BF16 读取、online softmax 与 split 合并怎样执行。 |
| [MoE 路由与量化专家执行](implementation/moe-routing-and-quantized-expert-execution.md) | top-k、配对排序与 padding、FP8 专家块尺度、两次 GEMM 和共享输出怎样连接。 |
| [W4A16 MoE 专家执行](implementation/w4a16-moe-expert-execution.md) | AWQ 专家 repack、配对/专家读取、整数零点与尺度、SiLU/gate 舍入及 FP32 top-k 汇合怎样衔接。 |
| [MoE 专家并行执行](implementation/moe-expert-parallel-execution.md) | SGLang/vLLM 的 group、expert 映射、normal/HT 与 LL 布局、通信尺度、gate/rho 与输出回传怎样配合。 |
| [DeepEP 通信设备执行](implementation/deepep-moe-communication-execution.md) | normal/HT 的 rank/组去重、LL 的专家复制、FP8 消息/scale/handle、BF16 回传与 FP32 gate 归约怎样执行。 |
| [DeepGEMM 专家设备执行](implementation/deepgemm-moe-expert-execution.md) | SGLang 匹配版本的 contiguous/masked 行选择、TMA scale、分组 WGMMA、padding 与 BF16 输出怎样配合。 |
| [TTS 量化敏感性与评测](implementation/tts-quantization-evaluation.md) | 组件消融、统计等价、模拟与真实低比特图及系统成本。 |
| [模型质量评测](implementation/model-quality-evaluation.md) | PPL 的有效计数、任务协议与风险—覆盖率怎样定义和比较。 |
| [推理服务评测](implementation/serving-performance-evaluation.md) | 负载、TTFT、TPOT、吞吐与 SLO 下的 goodput 如何共同解释量化收益。 |
| [张量并行与量化](implementation/tensor-parallel-quantization.md) | 列/行分片、GQA 复制、collective 与 group/scale/zero 如何配合。 |
| [Tensor Core 与量化 GEMM](implementation/tensor-core-quantized-gemm.md) | 从 tile、lane 和寄存器走到 ldmatrix、MMA、AWQ Triton 与编译诊断。 |
| [CUDA 算子的编写、编译与接入](implementation/cuda-operator-build-and-integration.md) | 从完整设备代码和包装走到编译、dispatcher、FakeTensor 与接口检查。 |
| [FP4/FP8 块缩放矩阵乘](implementation/block-scaled-matmul-execution.md) | 元素编码、K 组尺度、packed scale 布局与点积接口如何衔接。 |
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
| [GPU 算子性能分析：从时间线到瓶颈假设](implementation/gpu-kernel-performance-analysis.md) | 怎样选择 profiling 工具、解释计数器，并用 RMSNorm/量化 GEMM 对照检验瓶颈。 |
| [W8A8 算子：激活量化、整数 GEMM 与输出校正](implementation/w8a8-quantization-gemm-kernels.md) | 两框架的舍入、零行尺度、CUTLASS 主流水和零点校正怎样影响数值与成本。 |
| [权重量化反量化内核的契约：AWQ 与 marlin 两种实现](implementation/weight-only-dequant-kernels.md) | AWQ/TinyChat 与现代 Marlin 的打包、元数据和设备消费；区分对称 GPTQ 固定偏置与非对称 AWQ 整数零点，以及尺度、MMA、归约和输出顺序。 |
| [VLM 压缩评测框架：从任务精度到可信度](implementation/lvlm-compression-benchmark.md) | 多模态模型压缩的评测常停在几个 VQA 分数上。 |
| [量化误差诊断与验证](implementation/quantization-error-diagnosis.md) | 量化后结果变差，可能来自图变换错误、网格与分布不匹配、局部误差传播，也可能来自整数后端与模拟规则不一致。 |
| [AWQ 的实现核对：从量化脚本到 vLLM 与 SGLang](implementation/awq-implementation.md) | 方法页回答 AWQ 为什么这样设计、证据支持什么；本页回答代码层面的事情：量化脚本产出什么形状的权重、打包约定是什么、内核从哪里进入、同一个 AWQ checkpoint 在 vLLM 与 SGLang 里各走哪条路径。 |
| [GPTQ 的实现核对：校准、导出协议与服务内核选择](implementation/gptq-implementation.md) | 连接 Llama 校准、原始 3-bit 保存、服务格式、vLLM 后端选择与 SGLang 原生/Marlin；保留 TP 与 fallback 条件。 |
| [QServe/QoQ 的实现核对：两级编码、离线重排与 SGLang W4A8](implementation/qserve-implementation.md) | 从上游 fake-quant 产物追到 nibble 与尺度重排、SGLang QoQ 加载、字节级恢复及 INT8 MMA；区分 W4A8 Linear 与完整 KV4 系统。 |
| [SmoothQuant 的实现核对：平滑融合、INT8 接口与离线边界](implementation/smoothquant-implementation.md) | 区分平滑、fake quant 与 OPT INT8 导出，追踪 SGLang 动态 W8A8 和 vLLM compressed-tensors 的尺度、布局与执行边界。 |
