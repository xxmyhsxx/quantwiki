---
title: SpQR：敏感权重保留与两级量化元数据
type: method
tags:
  - ptq
  - weight-quantization
  - second-order
  - sparsity
sources:
  - raw/papers/2026-09-21/spqr/paper.pdf
  - raw/papers/2026-09-21/spqr/source.eprint
updated: 2026-09-22
---

# SpQR：敏感权重保留与两级量化元数据

SpQR 将大部分权重用小组均匀网格编码，将少数难以量化的权重以高精度稀疏形式保留，并进一步量化每组的 scale 和 zero-point。它解决两个相连的问题：小组量化的元数据太贵，以及少量敏感权重可能贡献大量误差。这里的 sparse 指高精度例外分支稀疏，不是把整个模型剪枝成稀疏网络。

## 1. 敏感性取决于还能如何补偿

原文“Quantization sensitivity of LLM weights”把敏感性定义为：固定一个权重的量化值后，让其余权重连续调整，所能达到的最小层输出误差。因此“大权重”“大输入幅值”和“难以补偿”并非同一概念。

沿用 [二阶补偿](../theory/layer-reconstruction-second-order-compensation.md) 的一致记法，设当前尚未固定的输入集合为 $R$，$G_R=X_RX_R^\mathsf T$ 正定，量化当前坐标引起偏移 $a=q-w$，则单步连续问题为

$$
\min_{\delta:\delta_j=a}\delta^\mathsf TG_R\delta
=\frac{a^2}{(G_R^{-1})_{jj}}.
$$

这是整理者对论文敏感性定义的展开；若写成 $\frac12\delta^\mathsf TH_R\delta$、$H_R=2G_R$，同一值为 $a^2/[2(H_R^{-1})_{jj}]$。必须使用相应逆矩阵的对角元素，不能把 Hessian、Gram 或逆 Cholesky 因子直接互换。实际阻尼也会改变这一代理问题。

随着 GPTQ 逐列固定权重，剩余集合与待量化权重都会变化，所以敏感性是动态的。稀疏保留值还可能吸收前面权重的补偿，并不必然等于最初的 FP16 权重。

## 2. 怎样选择例外并量化主体

原文算法 1 与“Sensitivity-aware compressed representation”给出主线：

1. 收集校准输入，构造带阻尼的二阶统计与逆 Cholesky 因子。
2. 在小组内估计基础量化误差，再尝试排除某个权重、重新拟合该组量化参数；若保持它为高精度能减少足够多误差，则按阈值 $\tau$ 选为例外。阈值控制整体例外比例，而不是对所有层直接取相同幅值阈值。
3. 排除例外后拟合主体网格。将实际要保存的量化元数据恢复出来，使用这些恢复值进行 GPTQ 式逐列量化与补偿。
4. 汇集主体编码、两级元数据、最终高精度例外及其位置。避免把主体量化完毕后才压缩元数据，否则前面补偿面对的网格与部署网格不同。

排除一个极端值可能同时收紧整组范围，所以选择收益不只来自该坐标自己的舍入距离。保留未结构化例外也能覆盖部分行、部分头以及零散位置；固定整行或整列高精度会把额外预算花在一些并不敏感的位置上。

## 3. 两级元数据怎样降低小组开销

一级每 $\beta_1$ 个权重保存一个尺度和零点；二级再将连续 $\beta_2$ 个一级尺度、一级零点各自做 min-max 量化。权重解码仍可写为

$$\widehat w=\widehat s(q-\widehat z),$$

但 $\widehat s,\widehat z$ 是从二级统计和低比特元数据解码的实数，$\widehat z$ 不要求为整数。这不等于普通整数 zero-point 的硬件契约。

以主体与一级元数据均为 3 bit、$\beta_1=\beta_2=16$ 为例，256 个权重保存 256 个 3 bit 码，16 对 3 bit scale/zero 码，以及 4 个 FP16 二级参数：

$$b_{\rm dense}=3+\frac{2\times3}{16}+\frac{4\times16}{16\times16}=3.625.$$

若直接对每 16 个权重保存 FP16 scale/zero，则为 $3+32/16=5$ bit/weight。两级量化的作用是用少量元数据误差换取更细网格；这些误差必须纳入实际重构。该算例来自原文表示章节的格式，不能把它当成所有实验都使用的组大小。

## 4. 稀疏分支怎样正确相加

论文实现保留例外位置的低比特主体码，以便稠密访问，并调整稀疏值补偿该位置的主体贡献。若最终希望该位置为 $w^*_{ij}$、主体解码为 $d_{ij}$，则教学表达为

$$r_{ij}=w^*_{ij}-d_{ij},\qquad y=Dx+Rx.$$

直接把 $w^*_{ij}$ 再加到 $d_{ij}$ 上会重复计入；$R$ 只有例外位置非零。论文按行、列排序例外，保存每项 16 bit 值、16 bit 列索引及每行 32 bit 累计计数。若例外密度为 $p$，在保留全部主体码的格式下，额外成本约为 $32p+32/n$ bit/weight，忽略末尾指针和对齐。1% 例外的值与列索引就增加约 0.32 bit，而非只增加 $0.01\times16$。

原文推理实现分别处理稠密反量化乘法和稀疏乘法，没有宣称两者已融合成一个 kernel。未结构化稀疏需要索引访问和负载均衡；换成通用稀疏算子不保证更快。权重在存储上低比特，解码后仍配合 FP16 输入计算。

## 5. 实验支持什么

SpQR v1 主实验对 LLaMA 用 RedPajama 校准，测 WikiText2/C4/PTB 与五个零样本任务。其 LLaMA-7B 表中，FP16 WikiText2/C4 PPL 为 5.68/7.08；3.94 平均 bit 的 SpQR 为 5.87/7.28；4.63 bit 配置为 5.73/7.13。论文的“near-lossless”采用约 1% 相对指标差的口径，不是参数无损，也不是所有生成任务都无损。3 bit 主体加例外和元数据不等于整模型恰好 3 bit。

LLaMA-65B 的元数据消融在近似存储预算下，3 bit 元数据配更小组的配置为 3.63 bit、WikiText2 PPL 3.74；FP16 元数据配较大组为 3.67 bit、PPL 3.84。它支持“两级量化允许更小组”的组合设计，不能解释成固定同一组时，降低元数据精度本身提高精度。

A100、batch 1、从头生成 100 token 的 LLaMA-7B 实验中，FP16、SpQR 配 PyTorch 稀疏算子、优化稀疏实现分别为 $47\pm2.3$、$30\pm2.2$、$57\pm2.4$ token/s。精度与格式没有自动带来速度，例外分支实现是必要部分。（Experimental Validation，LLaMA、消融和 inference 表。）

与 [SqueezeLLM](squeezellm.md) 相比，SpQR 的主体是小组均匀网格、敏感性来自可补偿的层误差；后者是任务梯度加权的非均匀标量码本。两者共享稀疏保留思想，但不能互换选择依据与部署格式。本页未执行原算法或 GPU 复现。

## 来源身份

[SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression](https://arxiv.org/abs/2306.03078v1)，arXiv:2306.03078v1。使用敏感性定义、算法、表示、实验与限制章节；敏感性公式按明确的损失归一化重新推导。
