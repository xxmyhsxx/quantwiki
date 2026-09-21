---
title: OmniQuant、AffineQuant 与 FlatQuant：可学习变换路线比较
type: comparison
tags:
  - ptq
  - equivalent-transform
  - reconstruction
sources:
  - raw/papers/2026-09-21/quarot/paper.pdf
  - raw/papers/2026-09-21/spinquant/paper.pdf
  - raw/papers/2026-09-21/omniquant/paper.pdf
  - raw/papers/2026-09-21/affinequant/paper.pdf
  - raw/papers/2026-09-21/flatquant/paper.pdf
  - https://github.com/OpenGVLab/OmniQuant/blob/feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14/quantize/omniquant.py
  - raw/repositories/2026-09-21/affinequant/source/quantize/affinequant.py
  - raw/repositories/2026-09-21/flatquant/source/flatquant/train_utils.py
updated: 2026-09-15
---

# OmniQuant、AffineQuant 与 FlatQuant：可学习变换路线比较

这三篇共同研究：冻结基座权重，以少量校准数据学习量化前的表示与网格，减少 Transformer block 输出误差。它们的区别不仅是“矩阵越来越大”，还包括哪些位置能变换、怎样保持数值可用，以及运行时怎样支付变换成本。

本页比较 OmniQuant v3（2024-03-18）、AffineQuant v1（2024-03-19）和 FlatQuant v4（2025-08-10）。各篇深度解释分别保留在 [OmniQuant](../methods/omniquant.md)、[AffineQuant](../methods/affinequant.md)、[FlatQuant](../methods/flatquant.md) 主页面；下列判断区分论文机制、固定代码和跨论文比较边界。

## 1. 每篇具体增加了什么

先统一符号：按 PyTorch 布局 $Y=XW^\mathsf T$，把**激活实际右乘的总矩阵**统一记为 $R$。没有平移时，配对权重始终是 $WR^{-\mathsf T}$：

| 方法页中的记法 | 激活侧 $R$ | 对应权重侧 |
|---|---|---|
| OmniQuant 的对角 $D$ | $D^{-1}$ | $WD$ |
| AffineQuant 的 $A$ | $A^{-1}$ | $WA^\mathsf T$ |
| FlatQuant 的混合 $P$，含可选对角 $D$ | $D^{-1}P$ | $WDP^{-\mathsf T}$ |

忽略额外尺度和平移时，AffineQuant 的 $A$ 对应 FlatQuant 的 $P^{-1}$，不是 $P$。同一个变量名在不同论文中可能指相反方向；上述对齐只比较单个合法线性接口。平移需额外补偿偏置，Q/K 和 V/O 的成对位置各有自己的恢复关系。

| 维度 | OmniQuant | AffineQuant | FlatQuant |
|---|---|---|---|
| 主要变量 | 相对权重裁剪 LWC、逐通道缩放/平移 LET、注意力配对尺度 | 延续裁剪与平移，在可用位置引入非对角可逆矩阵 | 两因子 Kronecker 变换、head 内小矩阵、对角尺度，以及权重/激活/KV 裁剪 |
| 解决的问题 | 手工缩放/独立裁剪难以适配整个块的量化误差 | 对角尺度不能混合通道，可能限制低位宽网格适配 | 完整混合矩阵昂贵、可放置位置有限；需要兼顾分布与执行 |
| 约束与稳定化 | 尺度非零、sigmoid 裁剪、具体算子融合条件 | 渐进 mask，希望保持对角占优；求逆与稳定因子有条件 | Kronecker 结构、U/V 正交参数化与对角倒数；仍需监测病态 |
| 运行代价的处理 | 尽量将合法尺度/平移融合到已有参数 | 为融合而限制部分位置；Norm 全矩阵分支仍有在线乘法 | 接受必要在线混合，通过小矩阵和变换/量化 kernel 融合压低开销 |
| 单篇最有价值的深入问题 | 为什么 LWC 与 LET 应联合学；输入误差如何推进 | 更大空间何时真的有收益；稳定性保证缺哪些条件 | 结构约束换来多少开销下降；质量与内核路径如何对应 |

（OmniQuant §3/算法 1；AffineQuant §3、附录 A.2；FlatQuant §3、附录 B/C.9。）

表达空间并不构成一个无条件性能顺序：完整矩阵族包含对角形式，但实际训练有约束、初始化与局部最优；FlatQuant 又主动使用结构化子集。变换族的数学包含关系，不能代替整套算法或全模型部署的比较。

## 2. 相同块重构名称下的目标差别

令 $x_l^{fp}$ 来自浮点前缀，$x_l^q$ 来自量化前缀。定向阅读的固定实现显示：

| 方法与代码 | 校准时比较什么 | 意义 |
|---|---|---|
| OmniQuant `feffe8ea`，quantize/omniquant.py | $F_l(x_l^{fp})$ 与 $\widehat F_l(x_l^q)$；可选 aug_loss | 当前块目标带有前层量化误差，但不重新优化前面所有块 |
| AffineQuant `0f3ba939`，quantize/affinequant.py | 同样维护浮点与量化前缀；可选 aug_loss | 扩大变换空间时仍沿用这一误差推进路线 |
| FlatQuant `9d88ffcb`，flatquant/train_utils.py | $F_l(x_l^{fp})$ 与 $\widehat F_l(x_l^{fp})$ | 主要优化当前块在教师输入分布上的误差，不能直接称为与前两者相同的累积误差补偿 |

这些是当前快照行为，不反推所有论文表格都用了完全相同的代码。[重构目标](../theory/layer-reconstruction-second-order-compensation.md) 中的输入、输出边界、权重变量和损失归约方式，都需要随实现记录。

例如第二块，前两者的基本参考是 $F_2(F_1(x))$，学生计算 $\widehat F_2(\widehat F_1(x))$；FlatQuant 的所读校准实现则比较 $F_2(F_1(x))$ 与 $\widehat F_2(F_1(x))$。前者要求当前块在受扰输入上对齐原路径，后者隔离当前块自身的量化误差。两种目标各有侧重，不能只凭是否包含前层误差判定哪个最终任务更好。

OmniQuant/AffineQuant 开启 `aug_loss` 时再加参考 $F_l(x_l^q)$；等权相加在输出目标层面等价于拟合两个参考的均值，推导见 [OmniQuant 的双参考目标](../methods/omniquant.md#5-训练流程数据与参数设置)。FlatQuant 的 $L/\operatorname{stopgrad}(L)$ 则保留同一个参考、按当前损失改变梯度尺度，没有增加第二个教师。这两项不能统一概括成“增强重构”。

## 3. 计算图位置决定表达力能否用上

一般线性层里 $XP\cdot P^{-1}W^\mathsf T=XW^\mathsf T$，但这个恒等式没有替我们处理 Norm、RoPE、非线性、残差和多个消费者。[矩阵基础](../fundamentals/mathematics/invertible-transforms-and-kronecker-products.md) 解释逆与条件数，[等价变换](../theory/diagonal-scaling-equivalent-transform.md) 解释图边界。

OmniQuant 主要学习可融合的通道尺度；AffineQuant 在 W4A4 的 Norm 后也限制为对角，而全矩阵选项会保留运行时乘法。FlatQuant 用小矩阵支持这些位置的在线混合，并在门控非线性后给 down 投影提供变换。它还把 Q/K 配对放在 RoPE 后，避免让一般矩阵未经检验就跨越位置旋转。

因此判断“哪个方法更强”前，应先列出目标模型中实际允许变换的位置：只需要权重低比特、需要 W/A 联合量化、还要压 KV，所用变换与部署负担并不相同。

## 4. 哪些证据能比较，哪些不能

- **保留同篇反例。**AffineQuant 表 2 的 LLaMA-13B W4A4 六任务均值低于 OmniQuant；表 11 的 LLaMA-7B W2A16g128 也未改善。这直接限制“更大矩阵总是更好”的概括。
- **优先使用机制消融。**OmniQuant 的 LWC/LET 与相对范围比较，AffineQuant 的 mask/稳定因子消融，FlatQuant 的 LT/PS/LCT 与裁剪先后顺序，分别回答每个新增设计做了什么。它们的模型、位宽和其他组件不同，不能把增益数值直接相加。
- **不要直接拼三篇排行榜。**例如 FlatQuant 表 1 的 OmniQuant C4 数值与 AffineQuant 表 3 所引用的 OmniQuant 数值并不相同；量化对称性、KV 配置、校准和评测版本都可能影响结果。差异原因未完整闭合时，保留各自协议和出处，不选有利基线。
- **速度必须对应同一运行对象。**OmniQuant 的后端结果、AffineQuant 的融合主张、FlatQuant 的融合 kernel/论文计时/后续真实模型部署，是不同层次的证据。FlatQuant 的小 batch decode 低于 1× 尤其关系到交互式场景，不能只比较峰值。

## 5. 对研究选择有什么用

如果需要清楚理解“怎样学习量化参数”，从 OmniQuant 的具体裁剪和块前向入手；如果要检验“通道混合是否比对角尺度有用”，AffineQuant 提供机制与稳定性问题；如果还要覆盖更多线性位置并讨论部署，FlatQuant 提供结构化变换和融合设计。

三者都没有证明更平的通道包络必然提升所有任务，也没有在本轮材料中解决我们的多模态分布与端云场景问题。可以从这些文献提出后续待讨论的研究问题，但不能把“采用某篇方法”直接写成既定论文方案。单篇原理读深后，比较才有具体内容；继续扩展文献也应围绕尚未解释的差异。

另一个对照轴是 [QuaRot](../methods/quarot.md) → [SpinQuant](../methods/spinquant.md)：从固定正交旋转走向以最终 CE 学习可融合旋转。它保持 RMSNorm 所需的正交结构，和本页的局部块 MSE/一般可逆变换路线并非同一组变量逐级放宽；具体比较由 SpinQuant 方法页承接。这个差别有助于区分“变换族不足”“校准目标不合适”和“部署位置受限”三个研究问题。（QuaRot v2 §4；SpinQuant v4 §3–4。）

> 来源维护（2026-09-21）：上列固定 commit 的外部代码引用对应已移除的本地快照；保留原版本身份，本轮未重新审查相关代码结论。

## 来源身份

下表用于在没有本地资料库时辨识来源；具体论述的章节、公式、图表或代码位置见正文。

| 来源 | 版本或快照 | 说明 |
| --- | --- | --- |
| [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456v2) | `arXiv:2404.00456v2` | — |
| [SpinQuant: LLM quantization with learned rotations](https://arxiv.org/abs/2405.16406v4) | `arXiv:2405.16406v4` | — |
| [OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models](https://arxiv.org/abs/2308.13137v3) | `arXiv:2308.13137v3` | — |
| [AffineQuant: Affine Transformation Quantization for Large Language Models](https://arxiv.org/abs/2403.12544v1) | `arXiv:2403.12544v1` | — |
| [FlatQuant: Flatness Matters for LLM Quantization](https://arxiv.org/abs/2410.09426v4) | `arXiv:2410.09426v4` | — |
| [OpenGVLab/OmniQuant](https://github.com/OpenGVLab/OmniQuant/blob/feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14/quantize/omniquant.py) | `feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14` | 本地快照已移除 |
| [bytedance/AffineQuant](https://github.com/bytedance/AffineQuant/tree/0f3ba9392bb0037f30750c4b2dbdd38469a2d7da) | `0f3ba9392bb0037f30750c4b2dbdd38469a2d7da` | — |
| [ruikangliu/FlatQuant](https://github.com/ruikangliu/FlatQuant/tree/9d88ffcb7d2c6bda59fb5c44dad36adc101aadb1) | `9d88ffcb7d2c6bda59fb5c44dad36adc101aadb1` | — |
