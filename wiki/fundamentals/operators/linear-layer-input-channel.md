---
title: 线性层与输入通道
type: concept
tags:
  - linear-algebra
  - matmul
  - weight-quantization
sources:
  - raw/papers/2026-09-21/quantization-white-paper/paper.pdf
  - raw/papers/2026-09-21/awq/paper.pdf
updated: 2026-09-22
---

# 线性层与输入通道

线性层把一组输入特征加权组合成输出特征。理解权重矩阵的行、列以及输入通道，才能分清量化粒度、通道重要性和缩放到底作用在哪里。本页从矩阵运算解释这些对象，再说明它们怎样支撑输入通道缩放与层输出误差分析。

## 1. 从一个输入向量到多个 token

设输入有 $C_{\mathrm{in}}$ 个特征、输出有 $C_{\mathrm{out}}$ 个特征。按列向量记法：

$$
x\in\mathbb{R}^{C_{\mathrm{in}}},\quad
W\in\mathbb{R}^{C_{\mathrm{out}}\times C_{\mathrm{in}}},\quad
b\in\mathbb{R}^{C_{\mathrm{out}}},\quad y=Wx+b.
$$

其中第 $o$ 个输出为

$$
y_o=\sum_{j=1}^{C_{\mathrm{in}}}W_{oj}x_j+b_o.
$$

在神经网络中通常把带偏置的仿射变换也称为“线性层”；严格的线性映射不包含偏置。乘加和累加器的硬件对应关系可参见量化白皮书 v1第 3 页、图 1 与式 (3)。本页以下维度展开和小例子为教学解释。

把 $T$ 个 token 的特征按列放在一起，得到

$$
X\in\mathbb{R}^{C_{\mathrm{in}}\times T},\qquad
Y=WX+b\mathbf{1}_T^{\mathsf T}\in\mathbb{R}^{C_{\mathrm{out}}\times T}.
$$

$T$ 是这里参与运算的 token 数，可以包含多个序列的 token，不应直接等同于 batch size。若程序把特征放在最后一维，例如输入形状为 $(B,L,C_{\mathrm{in}})$，则可将前两维展平为 $(BL,C_{\mathrm{in}})$，采用 $X_{\mathrm{row}}W^{\mathsf T}+b$ 的行向量写法。两种记法描述同一计算，矩阵转置只是约定变化。

## 2. 输入通道和输出通道

对上述 $W$：

| 对象 | 矩阵位置 | 含义 |
|---|---|---|
| 输入通道 $j$ | $X$ 的第 $j$ 行、$W$ 的第 $j$ 列 | 一个输入特征如何影响所有输出 |
| 输出通道 $o$ | $W$ 的第 $o$ 行、$Y$ 的第 $o$ 行 | 如何将所有输入组合成第 $o$ 个输出 |
| 单个权重 $W_{oj}$ | 第 $o$ 行、第 $j$ 列 | 输入 $j$ 对输出 $o$ 的系数 |

例如

$$
W=\begin{bmatrix}1&2&-1\\0&3&4\end{bmatrix},\qquad
x=\begin{bmatrix}2\\1\\-1\end{bmatrix},\qquad
b=0,
$$

则 $y=[5,-1]^{\mathsf T}$。第二个输入通道对应列 $[2,3]^{\mathsf T}$；它同时影响两个输出，并不是某一个输出神经元。

## 3. 权重误差怎样影响输出

若量化后权重为 $\widehat W=W+E$，输入和偏置暂时不变，则

$$
\widehat y-y=Ex=\sum_jE_{:,j}x_j.
$$

这说明同样大小的权重误差，乘上不同大小的输入特征，可能造成不同输出误差。它为 AWQ 观察激活分布提供了直接解释。例：若只有 $W_{1,2}$ 增加 0.1，在 $x_2=1$ 时第一个输出增加 0.1；在 $x_2=10$ 时增加 1。误差不只取决于权重本身的大小。

对多个 token，进一步有

$$
\|EX\|_F^2=\operatorname{tr}(EXX^{\mathsf T}E^{\mathsf T}).
$$

这是展开矩阵乘法与 Frobenius 范数得到的恒等式，说明完整输出误差还受输入通道间相关性的影响。平均绝对激活只是重要性的一个启发式信号，不等同于这个二次目标的全部信息。AWQ 的经验依据见原论文 v6§3.1、表 1；不要把本页代数解释写成作者证明了某种最优性。

输入统计怎样进入优化目标，可继续读 [层输出重构与二阶误差补偿](../../theory/layer-reconstruction-second-order-compensation.md)：它从这里的输出误差展开出发，说明固定一个权重后为何可以调整其他权重，以及输入相关性对补偿有什么影响。

## 4. 算子、计算图与 kernel

矩阵乘法、加法和归一化描述“计算什么”，属于算子层面的概念；计算图还描述张量在算子之间怎样流动、被哪些分支使用。kernel 是在特定设备上执行一段计算的程序：多个算子可以融合进一个 kernel，一个算子也可能由多个 kernel 实现。

因此，改写单个线性层公式与优化整个计算图是两件事。例如把输入除以缩放值、再把权重对应列乘回去，能保持未量化的线性结果，但若输入还供其他分支使用，就必须维护那些分支的结果；不能直接更改共享张量。具体条件见[对角缩放与等价变换](../../theory/diagonal-scaling-equivalent-transform.md)。AWQ 的实际融合案例见原论文 §4.2。

从公式走向实现时，还需将逻辑下标映射到实际地址，明确 stride、dtype、设备和输出别名；这些约定见 [张量布局与算子接口](tensor-layout-and-kernel-contracts.md)。逐元素、归约和矩阵乘怎样组织工作，见 [GPU 算子的计算模式](gpu-kernel-computation-patterns.md)。

## 5. 对后续量化研究的用途

“按输入通道缩放”和“按组量化”可以同时发生：前者为 $W$ 的每一列选择缩放值，后者为每一行内的一组连续元素共享量化参数。它们使用不同分组方式，不能因为都出现 channel／group 就视为同一个参数。下一步可读 [均匀量化与分组](../quantization/uniform-quantization-and-groups.md)，再看 [AWQ](../../methods/awq.md) 怎样把通道统计用于缩放和裁剪。

本页解释矩阵线性层。卷积可以在特定实现中转化为矩阵运算，但本页没有分析卷积的空间复用、分组卷积或相应硬件成本，不能直接把 LLM 的通道统计与性能结论迁移过去。

## 来源身份

下表用于在没有本地资料库时辨识来源；具体论述的章节、公式、图表或代码位置见正文。

| 来源 | 版本或快照 | 说明 |
| --- | --- | --- |
| [A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295v1) | `arXiv:2106.08295v1` | — |
| [AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978v6) | `arXiv:2306.00978v6` | — |
