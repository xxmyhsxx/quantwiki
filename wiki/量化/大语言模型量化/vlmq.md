---
title: VLMQ：token 重要性加权的二阶量化
slug: vlmq
sources:
  - raw/papers/quantization/vlm/2025-08-vlmq-token-saliency-driven-post-training/paper.pdf
  - raw/papers/quantization/vlm/2025-08-vlmq-token-saliency-driven-post-training/source.eprint
  - raw/papers/quantization/llm/2025-04-gptaq-asymmetric-calibration-v3/paper.pdf
paper_version: arXiv:2508.03351v2
verification: 全文与附录研读；GPTAQ 必要章节局部研读；作者实现未定位；未运行模型实验
updated: 2026-09-14
---

# VLMQ：token 重要性加权的二阶量化

VLMQ 用局部注意力模块重构损失的梯度估计各 token 的重要性，再把这些因子放进线性层的二阶重构目标。它选择性地改变校准误差的代价，最终导出低比特权重；不删除视觉 token，不为每个 token 选择不同权重位宽，也不要求推理时再次计算梯度。

本页依据 arXiv:2508.03351v2（2026-03-06），题名为 *Token Saliency-Driven Post-Training Quantization for Vision-language Models*。不要把这一版本的正文、实验与早期 “Hessian Augmentation” 题名下的内容混用。完整阅读覆盖 28 页及附录 A–E；没有定位到可验证的作者代码实现，以下流程来自论文，未标为代码复现。

## 1. 从视觉冗余到重构目标

视觉语言模型将视觉 embedding 与文本 embedding 送入共享语言骨干；视觉 token 数量、语义密度与文本不同，且两种模态的激活可能存在分布差异。结构和 mask 的必要前提见 [[vision-language-model-tokens-and-quantization|视觉语言模型中的 token 与量化对象]]。（§1、§3、附录 B）

图 2 用 Qwen2-VL-7B 第 20 层的 4096 维激活做 PCA，观察视觉/文本特征分离。这是投影下的分布现象，不是“所有视觉信息都冗余”的证明。表 1 的先导试验在 Qwen2-VL-7B INT3 的 DocVQA 上，纯文本校准为 86.86，图文 COCO 为 88.09；随机降低部分视觉 token 的重构权重可能进一步改善，但比例从 0.5 增至 0.75 又下降。不能从一个任务推出固定冗余比例。

与 [[mbq|MBQ]] 在模态组之间分配权重不同，VLMQ 为不同 token 分别估计重要性。但差异不止粒度：梯度损失、观察位置、输出误差范数和基础量化算法均改变，不能把两者看作只改一个开关。

## 2. 量化设置与符号

主实验是语言骨干的 weight-only INT2/INT3/INT4。`INT3g128` 表示 3-bit 权重、group size 128；无分组 INT3 与它不同，也不表示 W3A3。权重使用仿射量化，反量化值仍参与重构，网格与整数 zero-point 的定义见 [[uniform-quantization-and-groups|均匀量化与分组]]。（§5.1、附录 B/E）

本页采用以下符号：

| 符号 | 维度 | 含义 |
|---|---|---|
| $W$ | $o\times d$ | 当前线性层浮点权重 |
| $U$ | $d\times N$ | 量化路径传到当前层的输入，不必是整数张量 |
| $F$ | $d\times N$ | 对应浮点参考路径的输入 |
| $\widehat W$ | $o\times d$ | 量化后反量化的权重 |
| $G=\operatorname{diag}(g_1,\ldots,g_N)$ | $N\times N$ | token 重要性因子 |

“量化路径输入”可以只是受前层低比特权重影响的浮点激活，并不意味着当前方法量化了激活。token 轴 $N$ 与输入通道轴 $d$ 不可交换，详见 [[linear-layer-input-channel|线性层与输入通道]]。

## 3. 梯度从哪里来

论文把计算注意力、输出投影并加回残差的模块简写为

$$
\operatorname{Attn}(X)=X+\operatorname{MHSA}(X),\qquad
L_{\rm block}=\|\operatorname{Attn}(F)-\operatorname{Attn}(U)\|_F^2.
$$

这是带残差的注意力模块，不是包含全部 FFN 的整个 decoder block。式中省略的模型归一化等细节不能据此从实现删除。两条路径在本文符号下分别提供参考输出和受量化影响的输出。（§4.2、图 5–6、式 (5)）

对某个投影的输出 $Z\in\mathbb R^{o\times N}$，取得 $P=\partial L_{\rm block}/\partial Z$，按输出通道聚合：

$$
g_n=\frac1o\sum_{c=1}^{o}|P_{cn}|.
$$

Q/K/V/O 投影各有自己的输出梯度和因子，不是全模型共用一张 token 排名。局部反向降低了收集成本；它测的是局部重构损失的敏感性，不等于最终回答交叉熵的敏感性。对照 [[mbq|MBQ]] 的回答交叉熵梯度，必须先统一目标后才谈哪一种指标更好。（§4.1–4.2、式 (4)、算法 1）

定理 1 与附录 D 用链式法则和一阶展开连接参数扰动与激活扰动：

$$
\Delta L\approx\nabla_\theta L^\mathsf T\Delta\theta
\approx\nabla_Z L^\mathsf T\operatorname{vec}(\Delta Z).
$$

这能解释为何关注梯度，但不能证明取绝对值、通道平均、平方加权后的规则最优；尤其 INT2 的大扰动下，局部展开不提供端到端误差保证。

**实现边界：** 两条路径完全相同时，重构损失和梯度都为零，无法据此区分 token。算法 1 对启动或退化情况的处理尚不明确，需作者实现补足；本轮没有运行验证。

## 4. 为什么是 G²，为什么不能只替换 GPTQ 的 Hessian

[[gptq|GPTQ]] 的基本局部目标使用同一输入重构两侧输出。这里的**非对称校准来自 GPTAQ v3 §4.1**：区分量化路径输入与浮点参考输入；**token 因子来自 VLMQ v2 附录 C**。结合后，使重要位置的量化路径输出接近浮点参考输出：

$$
\min_{\widehat W}\|(\widehat W U-WF)G\|_F^2.
$$

对一行 $w$，令 $\delta=(\widehat w-w)^\mathsf T$、$r=w(F-U)\in\mathbb R^{1\times N}$，则残差恰好为 $\delta^\mathsf TU-r$。于是

$$
J(\delta)=\|(\delta^\mathsf TU-r)G\|_2^2
=\delta^\mathsf TA\delta-2c^\mathsf T\delta+rG^2r^\mathsf T,
$$
$$
A=UG^2U^\mathsf T,\qquad c=UG^2r^\mathsf T,\qquad \nabla^2J=2A.
$$

这里 $G$ 先乘残差，再平方，故各 token 的实际平方误差系数是 $g_n^2$。例如因子 1 与 2 对相同误差贡献的比例为 1:4，不是 1:2。若想让误差系数是 $\lambda_n$，应使用 $G_{nn}=\sqrt{\lambda_n}$；那是另一种因子定义，不能悄悄替换论文做法。（VLMQ 附录 C，式 (15)–(19)）

这也是 [[layer-reconstruction-second-order-compensation|层输出重构与二阶误差补偿]] 的一个加权实例：Hessian 对这个固定线性重构目标是精确二阶矩阵，但它不是全模型任务损失的 Hessian。非对称路径还引入 $r$ 和 $c$，仅把 $UU^\mathsf T$ 换成 $UG^2U^\mathsf T$ 却遗漏残差项，不能称为完整的 VLMQ/GPTAQ 路线。

### 固定一个量化坐标的自洽推导

在当前自由坐标内固定 $\delta_q=a$，假设 $A$ 正定，令 $B=A^{-1}$、$v=Bc$。拉格朗日条件为 $2A\delta-2c+\mu e_q=0$，代入约束可得

$$
\delta=v+\frac{a-v_q}{B_{qq}}Be_q
=\frac{a}{B_{qq}}Be_q+B^{(-q)}c,
$$
$$
B^{(-q)}=B-\frac{Be_qe_q^\mathsf TB}{B_{qq}}.
$$

$B^{(-q)}$ 在完整坐标表示中保留第 $q$ 行/列为零；剩余子块才是删去该变量后的逆矩阵。不能直接把维度为 $(d-1)\times(d-1)$ 的矩阵拿来与未裁切的 $d$ 维向量相乘。第一项满足量化约束，第二项补偿非对称残差而不改变已固定坐标。$F=U$ 时 $r=c=0$，退化为加权 GPTQ 型单步补偿。

上述约束解由本页按 GPTAQ v3 §4.1、附录 A.1 的推导整理，再代入 VLMQ 式 (19) 的加权变量；GPTAQ 的这一部分已按需研读，不代表其全文 ingest 完成。

此解只保证固定输入、因子、残差和一个坐标约束下的连续最小值，不保证全离散网络的最优量化。实际逐列量化会改变待优化状态；阻尼也会改变目标。当许多 $g_n=0$ 或有效样本秩不足时，$A$ 可能奇异。即使采用“平均对角值的 1%”阻尼，全零 $A$ 仍不会得到正定矩阵。

## 5. 算法、效率与实现所需信息

论文算法 1 的主流程是：

1. 保存同一校准序列的浮点路径和量化路径输入，按 decoder 层顺序处理。
2. 在量化当前线性模块前，对带残差注意力模块做前向与一次局部反向，保存 Q/K/V/O 输出梯度并转成 token 因子。
3. Q/K/V/O 使用加权二阶统计和相应残差项；Up/Gate/Down 使用原 GPTAQ 的普通统计。并不是给所有线性层一律施加相同 token 权重。
4. 按基础算法处理量化顺序、分组、阻尼与补偿，更新量化路径，继续下一层。
5. 导出兼容的量化权重表示；梯度和 $G$ 仅用于离线校准。

直接为每次单权重量化重新计算完整残差很昂贵。GPTAQ §4.2 将 $W(F-U)$ 分解到输入通道，并预计算与输入偏差有关的修正矩阵，配合共同列顺序、逆矩阵的 Cholesky 表达和延迟批更新。VLMQ 声称可沿用这些优化；加权时相应统计由 $UG$ 和 $(F-U)G$ 构造，不能只套一个未加权残差。高效通道分解算法与上节“固定完整残差的单步最优解”应分别理解，后者不是可直接逐元素照搬的高效实现。（GPTAQ v3 §4.2、算法 1；VLMQ 附录 C）

本轮 GPTAQ 只是解决关键依赖的局部研读，没有完整审计其余理论与实验，也没有检查 VLMQ 如何实际接入这些优化。论文的 INT2 个别实验改用 GPTQ 作为基础算法，必须在模型级标明；不能把默认 GPTAQ 视为所有实验的同一配置。

可核对的实验设置包括：512 个 ShareGPT4V 改进 COCO 图像描述样本；Qwen 系列序列长度 512，LLaVA-OneVision 为 986；丢弃视觉段被截断的样本；`act_order` 开启，`static_group` 关闭，默认阻尼比例 0.01。校准序列包含回答，需记录模板、padding、视觉段与 loss mask；这与真实生成时没有目标回答的输入不同。（§5.1、附录 E.1）

评测使用 LMMs-Eval；ScienceQA 有专门答案后处理。模型 revision、评测 commit、数据划分、提示及答案解析必须固定，不能把其他工具的默认值补作本文事实。缺少作者实现意味着零梯度启动、归一化细节、组顺序/逆置换及导出内核适配尚不能据本文直接保证。

## 6. 主结果、反例与可比范围

表 2 的平均值基于 ChartQA、DocVQA validation、MME-RealWorld 英文/中文、OCRBench、ScienceQA、SeedBench 2 Plus、TextVQA validation 八项；不同任务的平均分不是统一概率，也不与 MBQ 原论文的平均值等价。

| 模型与设置 | GPTQ | GPTAQ | VLMQ | 含义 |
|---|---:|---:|---:|---|
| Qwen2-VL-2B，INT3g128 | 61.78 | 63.44 | 62.90 | 低于 GPTAQ，存在负面结果 |
| Qwen2-VL-7B，INT3g128 | 73.22 | 73.68 | 74.40 | 相比 GPTAQ +0.72 |
| Qwen2.5-VL-7B，INT2g128 | 55.58 | 41.69 | 57.46 | 同样用 GPTQ；不能将所有差值归于 token 因子 |

INT2 的“16.45”来自表 3 的一个任务：Qwen2.5-VL-7B 在 MME-RealWorld 中文上 GPTQ 13.89、VLMQ 30.34，相差 16.45 个百分点。相同模型八项平均只提升 1.88；DocVQA 从 78.79 降到 73.20，TextVQA 从 74.36 降到 65.88。不能把单项提升宣传成整体提升，更不能省略任务间取舍。

高位宽下增益缩小：表 13 的 Qwen2-VL-7B INT4 从 GPTAQ 73.63 到 VLMQ 73.87，不能套用 INT2 的改善幅度。

### 消融能排除什么，不能排除什么

| 实验 | 作者报告 | 解释边界 |
|---|---|---|
| 表 5，随机视觉降权 | 低重要性比例 0.25/0.5/0.75，降权因子 0.01/0.05/0.1；完整方法 TextVQA/DocVQA 为 74.80/75.76 | 随机策略不是普遍有效；表中各随机设置 DocVQA 均低于 GPTQ 74.90 |
| 表 6/14，重要性指标 | 无分组 INT3：常数 69.03；FastV 注意力分数 68.29；PACT 注意力分数 67.07；梯度 69.70 | 这两种注意力策略在此条件下不如常数，不证明所有注意力指标均无效 |
| 表 7/15，梯度目标 | 层 MSE 67.77 / 0.29 GPUh；全网 CE 68.64 / 0.70；局部注意力块 MSE 69.70 / 0.21 | 同时改变目标与反向范围；“全网 CE 过拟合”是作者解释，未被此表单独证明 |
| 表 8/16，重构边界 | 无增强 69.03；O 输入处截断且只增强 QKV 为 55.54；attention 输出处但只增强 QKV 为 56.21；完整 QKVO 为 69.70 | 保留模块与残差完整性的经验支持；不构成任意块划分定理 |
| 表 9，基础算法 | Qwen2.5-7B INT2：GPTQ 55.58→57.46；GPTAQ 41.69→52.10 | 两种基础上均有改善，最终结果仍强烈受基础算法影响；选择规则需独立于测试集固定 |

论文没有为这些小幅平均改善普遍提供多种子置信区间。图、表和附录的跨任务明细支持的是指定条件下的报告，不是稳定性已验证的结论。如何检查代理误差与任务质量偏离，见 [[quantization-error-diagnosis|量化误差诊断与验证]]。

表 12 的部分平均值与任务列不一致，因此不使用其中 Qwen2.5-VL-32B 的排名结论；这不影响前文基于其他表格的机制与边界分析。

## 7. 校准成本与推理收益

表 10 在单张 H100 80GB 上报告，7B 模型的 GPTAQ/VLMQ 峰值显存为 24.76/29.05 GB，时间列为 0.27/0.29 小时。额外局部反向增加校准成本，但不成为推理时的梯度计算或显存负担。

表 11 在 RTX 5090 上采用 GPTQ W3 内核：四种线性层形状报告约 9.20–9.49 倍加速，单位为 μs。例如输入/输出维度 $(3584,4608)$ 的 Q/K/V 投影，FP16 为 38.97 μs，W3 为 4.17 μs。它是单算子结果，未给出完整图像到回答的时延；形状、batch、测量方法和基线实现仍需补齐。

最终权重表示不需要推理时的梯度计算，这是校准方法的性质；具体位宽、group、zero-point、布局及后端是否支持仍决定能否运行。不能把“GPTQ 兼容”理解为所有后端、所有低位宽都自动可用。执行条件见 [[quantized-matmul-scaling-execution|量化矩阵乘法的缩放与执行路径]]。

## 8. 可复用认识与边界

- **Strong：** 在 token 粒度表达重构代价，局部反向兼顾计算成本，并可接入二阶量化路线；重要性因子只在校准阶段使用。
- **Weak：** 梯度依赖局部代理目标，尚未证明跨任务稳定；额外校准显存、模块边界和基础算法选择都会影响可用性，整体平均改善也可能伴随个别任务退化。

本方法没有解决逐层混合位宽、部署后漂移或端云协同校准。论文已全文研读，GPTAQ 仅局部研读，作者实现与模型复现尚未核对。与 MBQ 的差异见 [[mbq-vlmq-comparison|MBQ 与 VLMQ 的条件化比较]]。
