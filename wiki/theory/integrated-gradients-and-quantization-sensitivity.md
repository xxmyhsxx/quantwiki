---
title: 积分梯度与量化敏感性：基线、路径和归因边界
type: concept
tags:
  - attribution
  - sensitivity
  - quantization-error
sources:
  - raw/papers/2026-09-21/integrated-gradients/paper.pdf
  - raw/papers/2026-09-21/qig/paper.pdf
  - raw/repositories/2026-09-21/qig/source/qmllm/methods/qig/quantize/auto_scale.py
  - raw/repositories/2026-09-21/qig/source/qmllm/methods/qig/quantize/auto_scale_wa_distort.py
  - raw/repositories/2026-09-21/qig/source/qmllm/quantization/qlinear.py
updated: 2026-09-15
---

# 积分梯度与量化敏感性：基线、路径和归因边界

积分梯度（Integrated Gradients，IG）把一个标量输出相对参考输入的变化分配给输入坐标。它适合解释“沿指定路径，哪些输入维度贡献了输出变化”，但归因大小不自动等于保留高精度后的任务收益。[QIG](../methods/qig.md) 将这个思想用于量化误差，再把归因转成校准权重；两步需要分别理解。

## 1. 为什么单点梯度不够

设 $F:\mathbb R^d\to\mathbb R$ 是要解释的标量，$x$ 是输入，$x_0$ 是参考点。局部梯度只说明当前点附近的响应；如果当前点已饱和，它可以为零，而 $F(x)-F(x_0)$ 仍很大。原论文 §2.1 用 $F(x)=1-\mathrm{ReLU}(1-x)$、$x_0=0,x=2$ 说明这一点：输出变化为 1，终点梯度为 0。

IG 沿直线 $z(\alpha)=x_0+\alpha(x-x_0)$ 积分：

$$
\mathrm{IG}_j(F;x,x_0)=(x_j-x_{0,j})\int_0^1
\frac{\partial F(z(\alpha))}{\partial z_j}\,d\alpha.
$$

必须固定标量、路径和基线，才有确定的归因问题。网络若输出 token×channel 张量，应说明选哪个输出或怎样归约；不能直接把向量梯度写成一个没有定义的“重要性”。（Sundararajan 等，PMLR 70，2017，§3 式 1）

## 2. 完整性保证的究竟是什么

对沿路径满足链式法则与微积分基本定理的函数（例如常见连续、分段光滑网络），

$$
\sum_j\mathrm{IG}_j
=\int_0^1\nabla F(z(\alpha))^\mathsf T(x-x_0)d\alpha
=F(x)-F(x_0).
$$

这就是完整性。只有 $F(x_0)=0$ 时，才能把右侧简称为完整输出 $F(x)$。原论文 §3 Proposition 1 及脚注给出网络函数的正则条件；离散量化的跳变和人为代理梯度需要另行检查，不能只凭“几乎处处有梯度”套用。

**教学例子：**$F(x)=x^2,x_0=1,x=2$ 时，IG 为 $3$，不是 $F(2)=4$。把目标改成 $F(x)-c$，其中 $c$ 为固定常数，不改变梯度、IG 或端点差。因而只从目标中减去一个常数，不能解释归因结果改善。

归因也依赖路径。以 $F(x_1,x_2)=x_1x_2$、从 $(0,0)$ 到 $(1,1)$ 为例，直线路径分配为 $(1/2,1/2)$；先变 $x_1$ 再变 $x_2$ 的分段路径分配为 $(0,1)$，总和均为 1。原论文 §4 区分路径方法、直线路径与对称性；这里是据定义构造的教学展开，不是新的实验。

## 3. 有符号贡献不能直接当非负误差系数

即便 $F\ge0$，单个 IG 仍可为负。取 $F(x_1,x_2)=(x_1-x_2)^2$、基线 $(0,0)$、终点 $(2,1)$，沿直线可得

$$\mathrm{IG}_1=2,\quad\mathrm{IG}_2=-1,\quad\sum_j\mathrm{IG}_j=1.$$

负号表示沿所选路径抵消了目标的增加，不能直接翻译为“恢复该 token 高精度会有负收益”。取绝对值后得到 $(2,1)$，总和变为 3；再归一化为 $(2/3,1/3)$，得到的是非负分配规则，已经失去原来的有符号完整性。IQR 裁剪也会改变总和和相对贡献。

如果要构造 $\sum_t\lambda_t\|e_t\|^2$，非负 $\lambda_t$ 有明确意义；有负系数时损失甚至可能因误差增大而下降。[二阶重构](layer-reconstruction-second-order-compensation.md) 中相应 Gram 矩阵的半正定性也依赖非负性。这说明“原始 IG 可解释”与“变换后的权重适合优化”需要不同证据。

## 4. 向量反向、量化分支和数值近似

QIG 固定代码 `06fa8813` 的两个 `auto_scale` 文件先计算
$E_{bt}=\operatorname{mean}_h|f_{\rm fp}(X)-f_{\rm q}(X)|_{bth}$，再用全 1 的 `grad_outputs` 对输入求导。其实际标量是 $F=\sum_{b,t}E_{bt}$，结果按**输入 token** 聚合；不是只计算每个输出 token 对同位置输入的对角 Jacobian。attention 的跨 token 作用可以进入这个梯度，但不构成因果识别。

实现用 32 个含端点的等距采样、每个权重 $1/32$ 近似积分，并计算 $\operatorname{mean}_h|\mathrm{IG}_{bth}|$。有限采样、数值精度、绝对值及裁剪都需要与原始恒等式分开。可用 $|\sum_j\widehat{\mathrm{IG}}_j-[F(x)-F(x_0)]|$ 检查**后处理前**的完整性残差，但残差小不证明所得校准权重最优。

尤其在 QIG 的 W/A 路径，`WALinear.forward` 使用 `no_grad`，因此不能把自动微分结果当作量化分支真实函数的完整导数。即使使用 [STE 等代理梯度](../fundamentals/quantization/post-training-and-quantization-aware-training.md)，也仍需说明代理对应哪个函数；本快照没有在该线性层使用 STE。具体分支、基线和损失差别见 [QIG 的实现核对](../methods/qig.md)。

本页原始 IG 论文只按需阅读 §2–4；数学例子用于检验定义和解释边界，不是模型量化复现。

## 来源身份

下表用于在没有本地资料库时辨识来源；具体论述的章节、公式、图表或代码位置见正文。

| 来源 | 版本或快照 | 说明 |
| --- | --- | --- |
| [Axiomatic Attribution for Deep Networks](https://proceedings.mlr.press/v70/sundararajan17a/sundararajan17a.pdf) | `2017 PMLR 70, pp.3319–3328` | — |
| [Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients](https://arxiv.org/abs/2603.17809v1) | `arXiv:2603.17809v1` | — |
| [ucas-xiang/QIG](https://github.com/ucas-xiang/QIG/tree/06fa88134671647827dea08dc37c50c81e4eeac7) | `06fa88134671647827dea08dc37c50c81e4eeac7` | — |

## 教学计算材料

保留已有教学计算脚本及当时结果，供核对推导与反例；这些材料不代表模型复现或性能实验。

- [check_math.py](../assets/integrated-gradients-and-quantization-sensitivity/checks/check_math.py)
- [math-check.json](../assets/integrated-gradients-and-quantization-sensitivity/checks/math-check.json)
