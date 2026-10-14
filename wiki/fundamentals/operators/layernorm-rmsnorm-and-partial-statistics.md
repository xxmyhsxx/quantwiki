---
title: LayerNorm、RMSNorm 与部分统计：归一化和不变性的条件
type: concept
tags:
  - normalization
  - numerics
  - transformer
  - training
sources:
  - raw/papers/2026-09-21/rmsnorm
updated: 2026-10-14
---

# LayerNorm、RMSNorm 与部分统计：归一化和不变性的条件

LayerNorm 用行内均值和中心化方差归一化，RMSNorm 则只用未中心化的平方均值。省去中心化会改变平移性质，保留的尺度性质也有条件。本页依据《Root Mean Square Layer Normalization》v1 解释原式、梯度与部分统计，再连接现代带 ε 的实现；论文的模型训练证据与现成 GPU kernel 分别核对。

## 1. 统计的是一个样本的特征

设一层的列向量输入 $x\in\mathbb R^m$，经 $W\in\mathbb R^{n\times m}$ 得到 $a=Wx$。论文称 a 为送入神经元的加权和，归一化统计在这个样本的 n 个特征上计算；不跨 batch 中的其他样本。令

$$
\mu=\frac1n\sum_i a_i,\qquad
\sigma^2=\frac1n\sum_i(a_i-\mu)^2,\qquad
r^2=\frac1n\sum_i a_i^2.
$$

原论文的 LayerNorm 与 RMSNorm 分别使用

$$
z_{\rm LN}=\frac{a-\mu\mathbf1}{\sigma},\qquad
z_{\rm RMS}=\frac a r,\qquad
y=f\bigl(g\odot z+b\bigr).
$$

g 是逐特征 gain，b 是偏移，f 是后续逐元素激活；论文比较时改变的是 z 的定义，f 不属于归一化统计本身。原式均未显式写 ε，LayerNorm 需 $\sigma>0$，RMSNorm 需 $r>0$。带 gain/偏移后的输出不保证均值为零或方差为一；原论文完整层也不能按现代某个无 bias 的模型实现改写。（§3–4，式1–5。）

展开平方给出 $r^2=\sigma^2+\mu^2$。RMS 保留了均值所占的能量，所以 r 不是中心化标准差。$\mu=0$ 且分母非零时，两种 z 才相等。完整 RMS 归一化使 $\|z_{\rm RMS}\|_2=\sqrt n$，但不会强制其均值为零；共同平移 $a+c\mathbf1$ 一般会改变 RMS 输出。这里的矩关系与范数关系是原定义的代数展开。

这也解释了归一化的融合边界：当前输入决定统计量，不能像固定推理统计的 BN 那样把整层折入一份线性权重。逐特征 gain 的吸收与输入相关统计需分开，见 [等价缩放的归一化接口](../../theory/diagonal-scaling-equivalent-transform.md)。

## 2. 正的共同缩放与梯度

先限定无 ε、r>0。对一个正标量 α，

$$
r(\alpha a)=\alpha r(a),\qquad z_{\rm RMS}(\alpha a)=z_{\rm RMS}(a).
$$

因此，当整份 W 或当前输入 x 共同乘 α，且 g、b、后续计算保持相同时，该归一化层的输出不变。逐权重行或逐特征用不同尺度一般不成立。原式6写 r(αa)=αr(a)；严谨地说，任意实数 α 应写绝对值，负 α 会翻转标准化向量，不能包含在上述输出不变结论中。平移也没有这一性质。（§4.1，式6–7、表1；正尺度与非零条件为代数澄清。）

尺度还影响反向梯度。对 z_RMS=a/r，其 Jacobian 为

$$
J(a)=\frac{\partial z_{\rm RMS}}{\partial a}
=\frac1r\left(I-\frac{aa^\top}{nr^2}\right).
$$

I 是 $n\times n$ 单位矩阵，$J(a)a=0$ 表示归一化对径向幅度变化没有一阶响应。若 $c=\partial L/\partial(g\odot z_{\rm RMS}+b)$ 是后续损失传回的列向量，则

$$
\nabla_W L=\bigl[J(a)\operatorname{diag}(g)c\bigr]x^\top.
$$

gain与偏移的梯度为$\nabla_g L=c\odot z_{\rm RMS}$、$\nabla_b L=c$；它们也依赖同一次归一化输出与上游梯度。

这是原式8–9的等价矩阵写法。比较缩放前后时，还需固定 g、b、后续计算及相应上游梯度 c。正尺度使 J(αa)=J(a)/α：仅将 W 变为 αW 时，W 的梯度变为原来的1/α；仅将 x 变为 αx 时，Jacobian 的逆尺度与外侧输入因子抵消，W 梯度不变。作者据此把权重尺度与梯度尺度的反向关系称为隐式学习率适配，未证明任意优化器、步长或整模型训练都稳定。（§4.2，式8–10；Jacobian 和链式写法为等价展开。）

## 3. ε 改变的是精确尺度关系

现有实现页记录的常见稳定化形式可以写成归一化向量

$$
z_\epsilon(a)=\frac a{\sqrt{r(a)^2+\epsilon}},\qquad \epsilon>0.
$$

对 α>0，

$$
z_\epsilon(\alpha a)
=\frac a{\sqrt{r(a)^2+\epsilon/\alpha^2}}.
$$

固定 ε 不随输入同比例变换，因此一般不再有 $z_\epsilon(\alpha a)=z_\epsilon(a)$。当 $r^2$ 远大于 ε、缩放后也保持这一条件时，两者可以接近；接近的程度仍依赖数值与尺度。上节无 ε 的 Jacobian 和精确梯度缩放关系不能直接当作这一版本的恒等式。以上是整理者对带 ε 定义的代数分析，原论文没有展开 ε。（具体实现定义见 [RMSNorm 的数学输出与残差状态](../../implementation/rmsnorm-cuda-triton-kernels.md#1.%20%E5%85%88%E5%8C%BA%E5%88%86%E6%95%B0%E5%AD%A6%E8%BE%93%E5%87%BA%E4%B8%8E%E6%AE%8B%E5%B7%AE%E7%8A%B6%E6%80%81)。）

尺度不变性与旋转关系不同。正交变换保持完整平方和，所以无逐特征仿射的完整 RMS，在相同 ε 下仍有正交交换关系；逐特征 gain 要另外处理。见 [归一化与残差的旋转条件](../../theory/orthogonal-rotation-and-hadamard-quantization.md#3.%20%E5%BD%92%E4%B8%80%E5%8C%96%E5%92%8C%E6%AE%8B%E5%B7%AE%E4%B8%BA%E4%BB%80%E4%B9%88%E9%87%8D%E8%A6%81)。

## 4. 前缀统计为何有不同条件

pRMSNorm 只用前 k 个特征估计统计量，再归一化全部 n 个特征：

$$
k=\lceil np\rceil,\qquad
r_p(a)=\sqrt{\frac1k\sum_{i=1}^k a_i^2},\qquad
z_p(a)=\frac a{r_p(a)},\qquad 0<p\le1.
$$

p 在这里按比例取值，例如6.25%=0.0625；原文用“前 p%”描述。采样范围是固定前缀，不是每轮随机抽取，分母是 k，也不是只输出前 k 维。无 ε、r_p>0 时，对正共同尺度仍有 r_p(αa)=αr_p(a)。这一代数性质不要求特征独立；作者提出的 i.i.d. 结构假设用于说明子集能否代表整行，而非证明实际训练后的通道都独立同分布。（§5、表1。）

代表性不足会改变归一化幅度。$r_p$ 不保证等于完整 r，$\|z_p\|_2$ 也不再保证为 $\sqrt n$；若前缀全零而其他特征非零，完整 RMS 有定义，原式 pRMS 却没有。作者承认 RMS 估计有偏且可能不准，并观察小样本统计下梯度不稳定。6.25% 是其任务中成功使用的配置，CIFAR 小模型另取12.5%，不是通用阈值。（§5，§6.1、附录A.4；非零前缀条件和范数区别为整理者展开。）

部分统计也一般不保持任意正交变换后的前缀能量。因此上一节完整 RMS 的旋转交换关系不能直接搬给 pRMSNorm；能否交换必须按所选变换和统计子集重新判断。这是统计定义的关系推导，论文没有做量化旋转实验。

## 5. 训练证据怎样连接到实现

原论文测的是模型训练时间与任务质量。多数时间在 TITAN X Pascal 上测，平均三次运行；不同框架和模型分别保留。Transformer 的表4另用 Tesla V100、TensorFlow base模型，隐藏维512、FFN2048、8 heads、训练300K步；每1K训练步的时间为 LayerNorm248秒、RMSNorm231秒、pRMSNorm225秒，分别缩短6.9%和9.3%。任务为WMT14 EN-DE，newstest2013选模，Test14/Test17对应newstest2014/2017、case-sensitive detokenized sacrebleu（13a）；label smoothing为0.1、每batch约25000 source/target tokens，Adam与4000-step warmup。表中BLEU依次为26.6/27.7、26.8/27.7、26.5/27.8。它支持这组训练配置的质量接近和时间减少，不是现代 inference kernel 的延迟测量。（§6.1，表4，附录A.1。）

减少统计元素也不保证端到端更快。论文把部分 pRMS 比完整 RMS 慢归因于框架切片开销；前缀统计仍需生成全部 n 维输出。表8最高63.9%的时间减少来自图文检索模型的 pRMS，每0.1K训练步12.02→4.34秒，不能当作完整 RMS 的普遍收益。质量也有反例：CIFAR 表10在 RTX2080Ti 上比较时，完整 RMS 测试误差8.83%，pRMS 为10.37%，后者虽略快但质量更差；附录A.4在空间宽高维统计、通道维施加gain/bias，p取12.5%，不是当前decoder特征轴kernel的直接对照。（§6.1/6.3/6.4，表3/8/10，结论及附录A.4。）

原论文提供的是机制、条件和所列训练证据。当前框架如何选择 dtype、统计精度、归约分段、残差写回与舍入阶段，另由 [RMSNorm 的 CUDA/Triton 实现](../../implementation/rmsnorm-cuda-triton-kernels.md)承接；读过某个现代 kernel 不表示复现了2019年的作者实验。本页没有取得或运行作者训练代码，也没有编译 kernel、训练模型或测量设备性能。

## 来源身份

- Root Mean Square Layer Normalization，arXiv:1910.07467v1，`https://arxiv.org/abs/1910.07467v1`；§3–5与式1–10、表1，§6代表实验与附录A相关设置；关键公式及表4/8/10已回看原页，未逐格核查全部实验表或作者实现。LayerNorm定义采用同篇§3，不登记未读的独立LayerNorm论文。
