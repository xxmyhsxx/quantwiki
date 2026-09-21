---
title: 均匀量化与分组
type: concept
tags:
  - data-format
  - granularity
  - rounding
  - clipping
sources:
  - raw/papers/2026-09-21/qlora/paper.pdf
  - raw/papers/2026-09-21/integer-only-quantization/paper.pdf
  - raw/papers/2026-09-21/quantization-white-paper/paper.pdf
  - raw/papers/2026-09-21/awq/paper.pdf
  - raw/repositories/2026-09-21/llm-awq/source/awq/quantize/quantizer.py
  - raw/papers/2026-09-21/gptq/paper.pdf
  - raw/papers/2026-09-21/smoothquant/paper.pdf
updated: 2026-09-14
---

# 均匀量化与分组

均匀量化用等间距的有限数值表示原来的实数或浮点数。位宽控制可用编码数量，量化步长控制相邻数值间隔，zero-point 决定整数编码与实数零的对应关系。理解这三者后，才能区分数值误差、模型质量和实际低比特存储。

本页依据量化白皮书 v1 §2–4，以及 Jacob 等整数推理论文 arXiv v1 §2–3 展开。AWQ 代码仅作为一个具体应用，不代表所有量化器的默认实现。

## 1. 量化编码与反量化

设位宽为 $b$，整数范围为 $[q_{\min},q_{\max}]$，步长为 $\Delta>0$，整数零点为 $z$。一种常用仿射量化定义是

$$
q=\operatorname{clip}\left(\operatorname{round}\left(\frac{x}{\Delta}\right)+z,q_{\min},q_{\max}\right),
\qquad \widehat x=\Delta(q-z).
$$

$q$ 是整数编码，$\widehat x$ 是反量化后的近似实数。若 $z$ 在允许的整数范围内，$x=0$ 可映射到 $q=z$ 并恢复为零。该性质对零填充等计算有用。$\Delta$ 通常用浮点数存储，因此低比特编码之外还有元数据成本。

对 unsigned $b$-bit 编码，常用 $q_{\min}=0,q_{\max}=2^b-1$。对 signed 编码，一种常用范围为 $[-2^{b-1},2^{b-1}-1]$。不同量化器也可能选择窄范围，必须写出实际端点，不能只写“INT4”。

整数均匀网格不是低比特表示的唯一形式。浮点格式（FP8 的 E4M3 与 E5M2）与块内共享尺度的 Microscaling 格式另行定义可表示点与元数据，见 [FP8 与 Microscaling 数值格式](../numeric-formats/fp8-and-mx-data-formats.md)；把量化结果落到文件时，块结构、二级尺度与重要性矩阵的组织方式见 [GGUF 块量化存储格式](../../implementation/gguf-block-quantization-formats.md)。两者都属于“位宽标签之外”的格式信息，比较方法或复现结果前必须先对齐。

**教学例子：**取 $b=3$、整数范围 $[0,7]$、$\Delta=0.5$、$z=3$，可表示实数范围为 $[-1.5,2]$。$x=0.7$ 编码为 $q=4$，反量化为 0.5；$x=2.4$ 饱和到 $q=7$，反量化为 2。前一个主要体现舍入误差，后一个体现超范围裁剪。此例不表示 AWQ 模型实验。

## 2. 对称、非对称与误差

白皮书将 $z=0$ 的简化形式称为对称量化；它避免累加时处理整数偏移，但 signed 范围的正负端点并不一定在实数上完全对称。非对称量化允许非零 $z$，能更灵活地利用整数范围表示偏移分布。符号命名和实际端点应一起说明。

在未触发饱和时，最近舍入的误差满足

$$
|\widehat x-x|\leq\frac{\Delta}{2}.
$$

若额外假设归一化舍入残差近似均匀分布于 $[-0.5,0.5]$，则有符号均值约为 0，平均绝对误差约为 $\Delta/4$，均方误差约为 $\Delta^2/12$。后两式是均匀残差模型下的推导，不是所有实际权重分布的保证；一旦发生饱和，单纯的半步长界也不足以描述总误差。

范围如何从样本得到、整数零点对齐为什么可能移动原端点，见 [校准与范围选择](../../theory/calibration-and-range-selection.md)。本页的 $q_{\min},q_{\max}$ 指整数编码端点，白皮书部分公式用相似记号表示实数端点，阅读时须区分。

固定编码数量时，扩大范围通常意味着增大步长：能容纳更多极端值，却让常见值附近的网格变稀；缩小范围则降低区间内舍入误差，却让尾部值承担更多裁剪误差。判断哪种更好，要看优化的目标。最小化 $\|\widehat W-W\|$、最小化 $\|(\widehat W-W)X\|$、保持下游任务分数，分别是不同问题。

AWQ 的裁剪搜索采用输入相关的分组输出误差，具体归约维度见 [AWQ 的裁剪目标](../../methods/awq.md#51-裁剪目标不是纯权重-mse)，不能直接称为“让权重 MSE 最小”。两种误差目标为什么不同、输入相关性如何影响误差，可进一步读 [层输出重构与二阶误差补偿](../../theory/layer-reconstruction-second-order-compensation.md)。

## 3. 量化粒度

量化粒度描述哪些数值共享同一组 $(\Delta,z)$。以 $W\in\mathbb{R}^{C_{\mathrm{out}}\times C_{\mathrm{in}}}$ 为例：

| 粒度 | 共享范围示例 | 主要取舍 |
|---|---|---|
| per-tensor | 整个权重矩阵 | 参数少，但极端值可能拉大整个矩阵的步长 |
| per-output-channel | 每一个输出行 | 不同输出行可采用不同范围 |
| group-wise | 每行沿输入维度划分连续的 $g$ 个权重 | 进一步适应局部分布，同时增加元数据和处理复杂度 |

这里的分组方向是具体约定，不是 group-wise 一词的唯一含义。论文或代码必须说明沿哪个维度分组以及剩余元素如何处理。AWQ 当前所读代码要求输入维度能被 group size 整除。C0 `quantizer.py:61`，commit `d6e797a42b9ef7778de8ee2352116e0f48a78d61`。

**教学例子：**若 $W$ 为 $2\times8$、group size=4，则有 4 组量化参数；若按输入通道进行等价缩放，则有 8 个通道缩放值。这些缩放值与 4 组量化步长承担不同职责，不能互相替代。矩阵维度见[线性层与输入通道](../operators/linear-layer-input-channel.md)。

忽略对齐和其他开销，若每组有 $g$ 个 $b$-bit 权重，加上共 $m$ bit 的元数据，平均存储成本为

$$
b_{\mathrm{effective}}=b+\frac{m}{g}.
$$

例如仅作示意，取 $b=4,g=128,m=32$，平均为 4.25 bit/weight，FP16 到此格式的理想比值为 $16/4.25\approx3.76$。这不是 AWQ checkpoint 的实测压缩率：实际还包含对齐、未量化参数、激活、KV cache 与后端格式。

## 4. W4A16 与模拟量化

W4A16 表示相应权重采用 4-bit 表示、激活保持 16-bit；它不意味着乘法和累加全部使用 INT4。AWQ 论文中的 TinyChat 会在运算时把压缩权重反量化到 FP16，再参与相应矩阵计算。AWQ v6§4.2。

模拟量化常把浮点权重映射到网格后，仍以浮点张量保存 $\widehat W$。它可以帮助评估低比特取值带来的质量变化，却没有自动减少物理存储，也没有证明后端实际运行了高效内核。实际压缩还需要编码打包、保存元数据以及读取这些数据的运算实现；三类路径的区别和代码实例见 [量化的实际执行路径](../../implementation/quantized-matmul-scaling-execution.md#3-三类执行路径)。

PTQ 是从训练好的模型出发做量化；它可以使用校准数据、搜索甚至局部优化，并不等于完全无数据或不调整参数。QAT 则在训练前向中考虑量化影响并训练模型。范式、可调整变量与代理梯度的区别见 [PTQ、QAT 与代理梯度](post-training-and-quantization-aware-training.md)。AWQ 属于前者，使用校准输入进行统计和候选比较，不执行权重梯度训练。这里仅说明术语与 AWQ 的关系，不评价两类方法在所有模型上的优劣。AWQ v6 §2、§3.2。

## 5. 应用到 AWQ 时必须区分的两种公式

AWQ 论文式 (1) 使用 $\Delta=\max(|w|)/2^{b-1}$ 和不显式写饱和的形式来分析误差；所读官方代码默认使用含 zero-point、有限整数范围和数值下限的仿射量化器。前者是理解机制的分析表达，后者是该代码版本的执行规则。

因此，复现时不能仅按论文的一条简化公式重写量化器并声称等同官方实现。也不能用当前代码默认值补作论文所有历史实验的设置。具体差异、版本和证据已在 [AWQ 主页面](../../methods/awq.md)展开。

## 6. 静态、动态与不同含义的分组

对激活而言，**静态量化**在校准时确定步长并在推理复用；**动态量化**根据运行时输入重新确定步长。它们与 per-token/per-tensor 这类粒度是两个分析维度。[SmoothQuant 的 O1–O3 配置](../../methods/smoothquant.md#4-量化配置与算子范围)中，O1/O2 分别是动态 per-token/per-tensor，O3 是静态 per-tensor；三者的通道平滑因子仍在离线获得。静态步长省去运行时范围统计，但校准没有覆盖的极端值可能产生饱和误差。SmoothQuant v7 §2、表 2。

在行 token 的激活矩阵 $X\in\mathbb R^{T\times d}$ 中，per-token 指每行共享参数，per-input-channel 指每列共享参数。后者改善某些离群值的表示，却不能简单用普通整数 GEMM 后的外维缩放完成；推导见[量化矩阵乘法的缩放与执行路径](../../implementation/quantized-matmul-scaling-execution.md)。

GPTQ 的算法列块 $B$ 用于批量补偿，量化 group size $g$ 用于共享网格，Transformer block 用于网络分段加载。三者不是同一个“块”。GPTQ v2 §4–5。其后续代码的 `static-groups` 又指提前固定权重组网格，与静态激活量化不同，详见 [GPTQ 主页面](../../methods/gptq.md)。


## 7. 整数编码、固定点与执行精度

均匀网格不要求步长为 2 的幂。power-of-two scale 进一步限制步长，有助于用移位换算尺度，也减少了可选网格；一般步长则可以由整数固定点乘法和移位近似其比值，仍形成整数推理路径。（白皮书 §2.2.3；Jacob 等 arXiv v1 §2.2。）

存储位宽、乘法输入位宽、累加位宽和输出位宽应分别记录。例如 8-bit 输入可使用 int32 累加，然后再量化到 8-bit 输出；bias 与累加具有共同实数单位，不能直接把浮点 bias 当成整数相加。推导和小例子见 [整数累加、偏置与再量化](../../implementation/quantized-matmul-scaling-execution.md#6-从整数累加到下一层网格)。误差与模型质量的检查关系见 [量化误差诊断与验证](../../implementation/quantization-error-diagnosis.md)。

## 8. 非均匀码本与尺度的二次量化

以上仿射均匀网格不能覆盖所有 4-bit 表示。QLoRA v1 §3 的 NF4 根据标准正态分布的分位信息构造 16 个非等距重构值，并处理精确零值；权重按块缩放后映射到码本索引。其解释形式为 $\hat w_i=s_g c_{q_i}$：$q_i$ 是 4-bit 索引，$c$ 是码本，$s_g$ 是所在块的尺度。不能把索引直接当成等间隔整数并套用 $s(q-z)$，也不能将“正态分布下的设计动机”当作任意权重都最优的保证。

Double quantization 再压缩第一层尺度，而非把权重位宽再次减半。例如每 64 个权重配一个 FP32 尺度，额外占 $32/64=0.5$ bit/weight；将尺度存为 8 bit、每 256 个尺度再配一个 FP32 二级尺度，示意开销为 $8/64+32/(64\cdot256)\approx0.127$ bit/weight。实际格式还可能有其他元数据。（QLoRA v1 §3）

[Q-VLM](../../methods/q-vlm.md) 的公开路径使用 NF4 权重与浮点计算，说明“4-bit 存储”“4-bit 网格”和“原生整数乘法”必须分别确认；QLoRA 原文也明确使用低比特存储、通常 BF16 计算。本节来自其 §2–3 的局部研读，不代表完整 QLoRA 方法或微调实验已经 ingest。

## 来源身份

下表用于在没有本地资料库时辨识来源；具体论述的章节、公式、图表或代码位置见正文。

| 来源 | 版本或快照 | 说明 |
| --- | --- | --- |
| [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314v1) | `arXiv:2305.14314v1` | — |
| [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877v1) | `arXiv:1712.05877v1` | — |
| [A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295v1) | `arXiv:2106.08295v1` | — |
| [AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978v6) | `arXiv:2306.00978v6` | — |
| [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq/tree/d6e797a42b9ef7778de8ee2352116e0f48a78d61) | `d6e797a42b9ef7778de8ee2352116e0f48a78d61` | — |
| [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323v2) | `arXiv:2210.17323v2` | — |
| [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438v7) | `arXiv:2211.10438v7` | — |
