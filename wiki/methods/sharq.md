---
title: SharQ：动态稀疏 FP4 主干与量化残差
type: method
tags:
  - activation-quantization
  - sparsity
  - data-format
  - kernels
sources:
  - raw/papers/2026-09-21/sharq
  - raw/repositories/2026-09-21/sharq
updated: 2026-10-12
---

# SharQ：动态稀疏 FP4 主干与量化残差

SharQ将当前激活分成符合硬件N:M约束的FP4稀疏主干和FP4稠密残差。残差相对**量化后的主干**构造，同时接收被掩码去掉的值和主干量化误差；两路共用一份FP4权重，通过不同尺度布局执行稀疏/稠密GEMM。它以第二路低精度计算换取质量恢复，并非单纯删掉一半激活。（SharQ v1（`https://arxiv.org/abs/2606.26587v1`），§3、式6–13、图1–2。）

本页研读固定20页PDF的正文方法/理论/实验与必要格式附录，核表1–5及图2–4，未逐点重建全部曲线。所选固定代码的NVFP4/Llama、native准备与GEMM接口见[实现页](../implementation/sharq-sparse-residual-and-shared-scales.md)；没有编译、运行模型或GPU。免训练/校准不代表免在线归约、掩码选择或残差量化。

## 1. 稀疏规则决定选择单元

对行token $X\in\mathbb R^{T\times K}$、权重 $W\in\mathbb R^{K\times D}$，参考输出为 $Y=XW$。掩码 $M(X)$ 随当前输入生成，主干 $X_{sp}=M(X)\odot X$。NVFP4实例采用4:8 in-pairs：连续8值视为4个相邻pair，保留其中2个pair；HiF4实例使用标量2:4。不能把前者简化成每8值任意保留4个最大标量。（§3.1、3.3、附录A.1.4。）

图2与所选实现明确pair分数为两元素绝对值的**最大值**，选择分数最大的2个pair。例如 $(10,0,9,0,8,8,7,7)$ 保留前两pair，但按平方和最大的两pair会选 $(8,8)$ 和 $(10,0)$；这是硬件合法、便宜的选择规则，不是输出误差/Hessian最优掩码。并列分数的具体次序还须由实现确定。（教学反例；源码top2_pairs_8_maxabs/top2_pair_indices_8。）

与 [Atom](atom.md) 的校准后固定128通道和 [ResQ](resq.md) 的固定信号子空间不同，这里选择单元是每次输入的局部pair/标量，且两路都采用FP4。它没有学习一份永久稀疏权重，也没有将离群值送进额外INT8/FP16矩阵乘。

## 2. 残差为什么要在主干量化之后计算

令 $\widetilde X_{sp}=D_{sp}(Q_{sp}(X_{sp}))$ 为稀疏路径实际表示的解码值，定义

$$R=X-\widetilde X_{sp}
=(X-X_{sp})+(X_{sp}-\widetilde X_{sp}).$$

第一项是掩码外的信息，第二项是保留值的量化误差；若只取 $X-X_{sp}$，第二项不会被残差接住。稠密路径量化R，得到 $\widetilde R=D_{dn}(Q_{dn}(R))$，于是

$$X-(\widetilde X_{sp}+\widetilde R)=R-\widetilde R.$$

这是定义带来的恒等式，并不保证残差容易量化到零误差。教学上，$X=(3,1)$、主干 $(3,0)$、量化后主干 $(2.5,0)$ 时，正确残差为 $(0.5,1)$；只补 $(0,1)$ 会留下0.5的主干误差。（§3.3、式7–8，§3.5、式10–12；教学例子不是模型实验。）

两路若解码成同一权重 $\bar W$，理想输出为 $\widehat Y=(\widetilde X_{sp}+\widetilde R)\bar W$。论文式13在**固定低精度权重算子**下给出

$$\|X\bar W-\widehat Y\|_F
\le\|R-\widetilde R\|_F\|\bar W\|_2.$$

它只界定激活近似造成的输出扰动。相对原浮点W还存在 $X(W-\bar W)$；不同路径尺度若解码成不同W，恒等式也要增加路径差异项。有限精度GEMM、BF16中间输出/累加舍入不在该实数界内。完整分解见[公共残差解释](../implementation/quantized-matmul-scaling-execution.md#17.%20%E7%A8%80%E7%96%8F%E4%B8%BB%E5%B9%B2%E4%B8%8E%E9%87%8F%E5%8C%96%E6%AE%8B%E5%B7%AE%E6%80%8E%E6%A0%B7%E5%85%B1%E7%94%A8%E6%9D%83%E9%87%8D)。不能由这个上界推出必定优于dense NVFP4或所有模型质量单调提高。

## 3. 共用权重仍需要两种尺度视图

NVFP4使用E2M1四位载荷、UE4M3局部尺度和软件全局PTS。普通dense路径按16值一组，sparse路径按32值一组；SharQ将权重先按32组量化一次，再将每个组尺度复制给对应的两个dense16组，保证两路相同载荷和相同解码值。它没有对dense权重另做一次16组最优量化。（图2、§3.4；固定shared_weight_nvfp4实现。）

理想逻辑存储只计载荷和两份局部尺度视图为 $4+8/32+8/16=4.75$ bit/weight；单dense16尺度的4.5 bit不能直接套这里。激活主干压缩、pair metadata、主干尺度、稠密残差和残差尺度都需计费；不是只存2 bit/原元素。实际还包括对齐、尺度swizzle/复制、全局FP32尺度、BF16源/输出和workspace，详见实现页。

融合准备减少独立mask/残差张量的写回；残差GEMM可在epilogue加入稀疏输出。但两路GEMM仍需执行，单稀疏主干算术减半不表示整个线性层减半。原生路径的准备、RMSNorm与全局归约也不能由图2的一个框推断成只发一个CUDA kernel。

附录A.1.2称E2M1有NaN保留编码；所选匹配CUTLASS的E2M1编码为NanInfEncoding::NONE，numeric_limits无NaN/Inf，正负 $\{0,0.5,1,1.5,2,3,4,6\}$ 已占满16个编码。这里采用实际载荷语义，并把原文差异保留；E4M3**尺度**的特殊值语义属于另一层，不能与E2M1混用。HiF4附录则以64个S1P2元素、8+8+16位三级元数据组成4.5bit逻辑预算；相同平均位宽不等于相同编码或当前GPU原生支持。

## 4. 质量恢复率对应什么差距

§4.1以dense NVFP4 W/A为主要基线、FP16为参考，不使用额外校准/训练；表1报告5项分类准确率的非加权均值，另外列WikiText2 PPL和MMLU。对高分更好的均分，恢复率为 $(S_{SharQ}-S_{NV})/(S_{FP}-S_{NV})$；不能将该比例推广给PPL或池化全部任务。

| 模型/任务集合 | FP16均分 | NVFP4 | SharQ | 均分差距恢复 |
| --- | ---: | ---: | ---: | ---: |
| Llama3.1-8B，5项分类 | 72.50 | 70.32 | 71.55 | 约56.4% |
| Qwen2.5-7B，5项分类 | 70.94 | 69.43 | 70.38 | 约62.9% |
| Qwen3-30B-A3B，5项分类 | 70.07 | 68.40 | 69.17 | 约46.1% |
| Qwen3-VL-8B，表2的5项多模态任务 | 81.37 | 79.97 | 80.57 | 约42.9% |

表1 Llama的PPL为6.94→6.73、MMLU为61.93→63.76；整体恢复未达到FP16的6.25/65.24。也非每任务都改善：Qwen2.5 Lambada为70.37→70.00，Qwen3 ARC-C为54.52→53.92。Qwen3-30B-A3B是MoE质量报告，不代表本次所选Llama native链已覆盖其全部专家部署。

表2五项多模态任务均改善NVFP4，POPE为87.73→88.20（+0.47个百分点）；这支持所测任务质量恢复，不能仅凭该分数证明跨模态表示的几何对齐已保持。没有共同输入/seed、方差或所有任务细节时，也不能把微小分数差改写成统计显著。

表4在Qwen3-30B-A3B比较NVFP4/HiF4/MXFP4的所列任务，均有恢复，例如PPL分别9.12→8.96、9.08→8.92、9.70→9.19；这是格式组合的质量证据，不证明每种格式都有同样可用的native高性能kernel。表5 Llama的sparse-only为PPL82.95/MMLU25.78，说明残差重要；dense+dense为6.63/63.61，而SharQ为6.73/63.76，BoolQ为81.22/80.58，不能概括为每项都达到或超过dense+dense。

## 5. 性能收益与基线绑定

§4.3/图3使用RTX5090、vLLM、Llama3.1-8B、batch4，输入256–2048且decode步数与输入等长。正文报告相对FP16约2.2–2.4倍延迟速度比、对FP8约1.2–1.4倍total-token吞吐；没有对比NVFP4端到端吞吐的共同曲线，不能改写为比所有FP4方案都快。

图3与正文的部分延迟数字有差异：图中512/512为5.75→2.30秒（标2.50×），1024/1024为11.81→4.93（约2.40×）；正文举256/256的2.85→1.17和2048/2048的24.88→11.11。这里保留各自来源/长度，不把它们拼成一套经复现的完整曲线。

表3 Wan2.2-T2V-A14B使用4张RTX5090：SharQ单独使480P/720P的223.41/717.88秒降至182.21/648.04，约1.23/1.11倍；叠加SageAttention才是141.63/457.88秒，约1.58/1.57倍。后者不能全部归因于SharQ线性层。

图4的Llama prefill总时间为NVFP4的111.2%，即**增加11.2%**；GEMM部分从30.1变为48.4个基线时间百分点，预处理从量化12.3+RMSNorm21.8=34.1降至26.7，节省7.4个百分点。局部sparse GEMM为dense NVFP4的0.66/0.72/0.71倍延迟，对应延迟减少34/28/29%，速度比分别约1.52/1.39/1.41；不是整个两路线性层的速度比。质量恢复、prefill开销和decode收益应分别判断。

## 来源身份

| 来源 | 固定身份 | 本页范围 |
| --- | --- | --- |
| SharQ: Bridging Activation Sparsity and FP4 Quantization for LLM Inference（`https://arxiv.org/abs/2606.26587v1`） | arXiv:2606.26587v1；PDF SHA-256 7c367ae4b19a55ea60e883837440562568fc3fb3ee4dc7df24b77e0d9ab8207c | 正文机制/条件界、表1–5、图2–4和必要格式附录；未重建所有曲线，未运行模型/GPU |
| actypedef/SharQ（`https://github.com/actypedef/SharQ/tree/3cc584fe48e32e59c8b4e37323470dcdd606bb11`） | 3cc584fe48e32e59c8b4e37323470dcdd606bb11 | pair maxabs规则与所选准备；实现范围见实现页 |
| 匹配 CUTLASS（`https://github.com/NVIDIA/cutlass/tree/baea077e425056e2d1d22d2bf5ed928908878790`） | baea077e425056e2d1d22d2bf5ed928908878790 | E2M1编码/特殊值语义；未核全部vendor或GPU |
