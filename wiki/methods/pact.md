---
title: PACT：任务驱动的激活裁剪与代理梯度
type: method
tags:
  - qat
  - activation-quantization
  - clipping
sources:
  - raw/papers/2026-09-22/pact
  - raw/papers/2026-09-22/lsq
  - raw/papers/2026-09-22/lsq-plus
updated: 2026-10-13
---

# PACT：任务驱动的激活裁剪与代理梯度

PACT 用可学习上界替换 CNN 的 ReLU，再将非负、有界激活均匀量化。上界由任务损失和 L2 正则共同调整，在尾部裁剪与内部网格分辨率之间取舍。它不冻结模型只做范围校准；少量裁剪参数之外，原实验仍从头训练网络。

本页依据 PACT: Parameterized Clipping Activation for Quantized Neural Networks，arXiv:1805.06085v2，核读15页PDF的正文、附录A–C及关键公式/图表；比较回查LSQ v3的步长梯度。归档源包只有嵌入PDF的TeX包装，不含可逐式核读的正文TeX。数字均为作者报告，未核作者代码或运行模型。

## 1. 裁剪上界同时决定步长

对ReLU前输入$x$，给定正上界$\alpha>0$，论文§4式1–2可写成

$$
y=\operatorname{clip}(x,0,\alpha),\qquad K=2^b-1,\qquad
s=\frac{\alpha}{K},\qquad q=\operatorname{round}(y/s),\qquad y_q=sq.
$$

$b$为激活位宽，$q\in\{0,\ldots,K\}$是整数码，$y_q$才是用于训练前向的恢复值。网格包含0和$\alpha$，有$2^b$个值、$K$个间隔；下界和零点固定为0。负输入先归零，正输入超过上界后饱和；不是将有符号激活直接映射到同一个无符号网格。

缩小$\alpha$使$s$变小，提高区间内分辨率，同时裁剪更多正尾部；增大则反之。未饱和舍入误差与尾部误差的基础见[校准与范围选择](../theory/calibration-and-range-selection.md#2.%20%E4%B8%BA%E4%BB%80%E4%B9%88%E8%8C%83%E5%9B%B4%E8%B6%8A%E5%A4%A7%E4%B8%8D%E4%B8%80%E5%AE%9A%E8%B6%8A%E5%A5%BD)。PACT用整网任务目标选择这个取舍，不以张量MSE最小为默认准则。

作者最终选择每个被替换的激活层共享一个$\alpha$，不是每个输入、token或通道动态重估阈值。附录A.1图7的粒度比较只覆盖**无量化的CIFAR10-ResNet20**，逐层共享在该实验较好；其减少尺度数量和便于同层MAC后统一恢复的动机，不构成任何量化模型上逐层最优的证明。

## 2. 任务梯度只通过作者指定的裁剪代理

§4式3先取$\widetilde\partial y_q/\partial y=1$，再使用裁剪对上界的导数，规定

$$
\frac{\widetilde\partial y_q}{\partial\alpha}
=\frac{\widetilde\partial y_q}{\partial y}\frac{\partial y}{\partial\alpha}
=\begin{cases}0,&x<\alpha,\\1,&x\ge\alpha.\end{cases}
$$

这是论文选择的STE代理，不是离散前向的真实导数，也不是将式2中的每个运算直接交给普通自动微分所得的完整链式梯度。边界$x=\alpha$保留原文取1的约定；下界/输入导数的边界处理及实际反向仍需实现核对。基础的离散前向与代理反向见[PTQ、QAT与代理梯度](../fundamentals/quantization/post-training-and-quantization-aware-training.md#3.%20%E4%B8%BA%E4%BB%80%E4%B9%88%E9%9C%80%E8%A6%81%20STE)。

在同一$\widetilde\partial y_q/\partial y=1$约定下，离开两端的输入代理可由裁剪展开为$\widetilde\partial y_q/\partial x=\mathbf1\{0<x<\alpha\}$。因此上界以下的阈值梯度为零，不表示未饱和输入没有向更早层传播任务梯度的路径；这是整理者的链式说明。

以分类训练为例，记$\theta$为网络权重及其他训练参数，$\boldsymbol\alpha$为各层上界，目标可整理为

$$\min_{\theta,\boldsymbol\alpha}\ \mathcal L_{\rm CE}(f_Q(u;\theta,\boldsymbol\alpha),t)+R_W(\theta)+R_\alpha(\boldsymbol\alpha).$$

$u,t$是训练图像和标签，$f_Q$使用量化前向；$R_\alpha$是原文的L2项。对于共享上界$\alpha_\ell$，令$g_i=\partial\mathcal L_{\rm CE}/\partial y_{q,i}$，整理者展开为

$$\widetilde\nabla_{\alpha_\ell}\mathcal L
=\sum_{i\in\ell}g_i\mathbf1\{x_i\ge\alpha_\ell\}
+\frac{\partial R_\alpha}{\partial\alpha_\ell}.$$

求和覆盖使用该上界的激活元素，具体损失的batch/空间归约已包含在$g_i$中。贡献可能相互抵消，不能仅凭饱和点多就断言上界会增大。按梯度下降，负的合计任务梯度推动上界增大，正的合计推动缩小；如果当前没有元素到达上界，这条任务梯度为零，正则仍可收缩范围。

为展示正则方向，可采用$R_\alpha=\lambda_\alpha\sum_\ell\alpha_\ell^2/2$，于是其梯度为$\lambda_\alpha\alpha_\ell$。这里的$1/2$是整理者记号约定，原文未给统一的完整损失系数规范。网络参数也通过任务损失更新，不能把此公式写成“只训练裁剪阈值”的PTQ流程。

### 初始化和正则为什么配合使用

§4.2、附录A.2建议相对激活取适度大的初值，再用L2收缩：过小初值让大量饱和点进入阈值梯度，早期可能不稳；过大则任务梯度很少，可能停在低分辨率范围。图2的$\alpha_0=10$来自**全精度CIFAR10-ResNet20**轨迹，不能当作所有量化实验统一默认值。

作者观察$\lambda_\alpha$取权重L2的同类系数可用，较高位宽的最佳系数略小；这是实验经验。§4.1图3来自预训练SVHN网络的一维逐层阈值扫描，冻结其余权重、bias、BN及其他层上界，观察正则缓解部分损失平台。作者关于“不扰动全局最小点”的概括只支持这组观察；一般加入L2会改变驻点，不能据此保证任意正则强度都保留任务最优解。

公式需要$\alpha>0$，但v2没有披露投影、重参数化或步长下限等具体保障。复现时必须明确该处理，不能自行补写为作者使用了softplus，也不能允许$\alpha=0$后仍按式2除法执行。

## 3. 与LSQ：前向可换算，反向不同

对同一位宽和非负输入，令$\alpha=Ks$，PACT与[LSQ的无符号激活前向](lsq.md)可表示相同的网格。这个等价只描述给定参数时的输出，不意味着训练相同。

| 项目 | PACT v2 | LSQ v3 |
| --- | --- | --- |
| 直接学习的量化参数 | 激活上界$\alpha$ | 权重与激活各自的步长$s$ |
| $0<x<\alpha$时的代理 | $\widetilde\partial y_q/\partial\alpha=0$ | $\widetilde\partial y_q/\partial s=\operatorname{round}(x/s)-x/s$ |
| $x>\alpha$时的代理 | 对$\alpha$为1 | 对$s$为$K$ |
| 关键附加设计 | 阈值L2正则、初始化与共享范围 | 专用步长梯度缩放$1/\sqrt{NQ_P}$与初始化 |
| 原实验训练起点 | CNN从头训练 | 同结构浮点模型微调 |

若对round使用恒等STE、保留实际舍入值并对其他运算求导，再将LSQ的$s$改写为$\alpha/K$，则区间内部的上界代理成为

$$\frac{\widetilde\partial y_q}{\partial\alpha}
=\frac{\operatorname{round}(x/s)-x/s}{K}.$$

这是整理者的链式换算，区别于PACT式3的零。教学例：$b=2,\alpha=3,s=1$，未饱和的$x=1.49,1.51$在PACT中上界代理均为0；上述换算后的代理分别为$-0.49/3,+0.49/3$，仍要乘上游任务梯度才能决定更新。因此只在前向令$s=\alpha/K$，再套常规round STE，不足以说明复现了PACT规定的反向。（PACT §4式1–3；LSQ §2.1式3、图2。）

LSQ保留内部舍入位置的信号，并缩放共享步长梯度；PACT的L2则改变训练目标，不是同一种梯度尺度处理。两者都在所选任务损失下训练，不能按“裁剪”与“步长”的变量名称将一个归为数值MSE、另一个归为任务训练。LSQ全部推导与消融由[LSQ方法页](lsq.md)承接。

## 4. 原实验流程、参数预算与迁移条件

原论文§5、附录B使用TensorFlow/Tensorpack，从头训练CNN；PACT负责激活，联合低位权重采用DoReFa。其流程是确定量化层与位宽、将相应ReLU换成PACT、设置各层上界与正则，在量化前向下训练网络及上界，推理时固定上界生成网格。DoReFa权重规则不能自动算作PACT激活公式的一部分。

按逐层共享方案，新增参数数目等于被替换的激活模块数，不是所有模型共用一个$\alpha$；网络权重、BN、梯度和优化器状态仍需训练计账。推理每层可保存$s=\alpha/K$，整数码仍需由当前输入生成；不能只凭阈值固定就断言全图已有整数内核或零额外成本。

ImageNet的ResNet18/50采用full pre-activation结构，训练为SGD、momentum 0.9、batch 256、110 epochs，初始学习率0.1，在30/60/85/95 epochs乘0.1，权重L2系数$10^{-4}$（附录B）。这些设置和CNN质量结果不能直接移到LLM，也不能将量化前向解释成训练梯度及所有状态都是低比特。

原定义将负输入归零，适合该论文替换ReLU的场景。GELU/SiLU输出、有符号隐藏态或KV可能需要保留负值，直接套$[0,\alpha]$会改变激活语义；有符号扩展、粒度、目标和反向须单独定义与验证。LLM中的[OmniQuant裁剪](omniquant.md)学习随当前权重极值变化的相对强度、优化块重构；其替换PACT/LSQ的组件实验不等于复现二者原始任务训练或证明普遍优劣。

[LSQ+](lsq-plus.md)给出适应偏斜正负激活的另一种定义：固定整数端点，学习步长与实数网格偏移，而不是把激活截成非负。其Swish CNN结果和MSE初始化条件由方法页承接，不改写PACT原有的阈值代理或移植为任意有符号模型的效果保证。（LSQ+ v1 §3–5。）

## 5. 质量证据与首末层条件

表1报告下列top-1准确率，权重用DoReFa、激活用PACT；按§4.2，首末层未量化，W/A标签只代表所选中间层。差值是百分点。

| 数据/模型 | 浮点基线 / % | W4A4 / % | W2A2 / % |
| --- | ---: | ---: | ---: |
| CIFAR10-ResNet20 | 91.6 | 91.3 | 89.7 |
| SVHN CNN | 97.8 | 97.8 | 97.7 |
| ImageNet-AlexNet | 55.1 | 55.7 | 55.0 |
| ImageNet-ResNet18 | 70.2 | 69.2 | 64.4 |
| ImageNet-ResNet50 | 76.9 | 76.5 | 72.2 |

W4A4可接近这些浮点基线；W2A2的ResNet18/50却分别下降5.8/4.7个百分点，不能由某些浅网络低位表现很好推导任意模型的极低位无损。附录C.2中的PACT/DoReFa由作者运行，其他方法多数引用各自论文，基线和配置不同，不能把这些表写成统一协议排行榜。

**v2基线差异。** 表1的ResNet18浮点为70.2%，附录表5的标题及浮点行却为70.4%，两表W4A4同为69.2%；差值分别为1.0与1.2个百分点。因此保留表号与各自数值，不自行统一基线，也不照抄“所有模型都在1%以内”或“达到浮点精度”作为逐项无条件结论。

附录A.3表2进一步比较首末层：CIFAR10-ResNet20、中间层4位，首末层均取2/4/8/32位时，验证错误为15.9/9.7/9.2/8.9%。8位相对32位只增加0.3个百分点，2位增加7.0个百分点，支持在该模型保护端层；不构成任何模型统一采用端层8位的规则，也不改变主质量表首末层未量化的条件。

## 6. 硬件收益是特定系统建模

§6图6用14nm CMOS硬件实现估计MAC面积，报告W16A16降至W2A2约14倍密度改善；系统吞吐则来自DeepMatrix详细性能建模。等芯片面积下配置总片上内存16MB、2GHz、每核$8\times64$ MAC，16/8/4/2位对应4/8/16/32核，分别考虑无限、32、64 GB/s外部带宽。

在ResNet50、64 GB/s带宽的模型中，16→4位使峰值计算能力增加4倍、吞吐增加4.5倍；作者将额外收益归于低位数据更易留在固定容量片上内存，减少外部传输。低位系统配置的首末层仍为8位；再降端层所给1.24倍是条件估计。这些结果不证明任意GPU或部署框架的端到端加速，数学网格和固定阈值也不能替代真实整数格式、布局、恢复与后端读取的核验，见[量化矩阵乘执行](../implementation/quantized-matmul-scaling-execution.md)。

## 来源身份

- PACT: Parameterized Clipping Activation for Quantized Neural Networks（`https://arxiv.org/abs/1805.06085v2`），arXiv:1805.06085v2；正文§1–7、式1–3、图2–3/6–8、表1，附录A–C及表2/5的条件核对。源包仅为PDF包装；图像主要用于定性核查，未逐点重建曲线，未核作者代码或复现训练/硬件。
- Learned Step Size Quantization（`https://arxiv.org/abs/1902.08153v3`），arXiv:1902.08153v3；本轮定向回查§2.1–2.3、式3、图2与已有方法页的步长代理/缩放，完整既有范围见LSQ页。任务梯度聚合、损失记号及同前向的梯度换算为整理者解释。
- LSQ+: Improving low-bit quantization through learnable offsets and better initialization（`https://arxiv.org/abs/2004.09576v1`），arXiv:2004.09576v1；本页第4节仅连接正负激活的网格平移，原文§3–5，完整研读范围见LSQ+页。
