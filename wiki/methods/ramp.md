---
title: RAMP：输出散度、聚类阈值与 CPU 计算图成本
type: method
tags:
  - mixed-precision
  - sensitivity
  - deployment
sources:
  - raw/papers/2026-09-26/ramp-edge-cpu-mixed-precision/paper.pdf
updated: 2026-09-28
---

# RAMP：输出散度、聚类阈值与 CPU 计算图成本

RAMP 为 CPU 视觉模型选择 FP32/INT8 混合配置：一次只量化一层，观察最终分类输出的变化；用一维 K-means 对敏感度产生少量阈值，再比较完整候选模型的质量与延迟。它最可复用的结论是：**敏感度、分配策略与执行代价是三个问题，单层延迟之和会遗漏精度边界导致的图融合变化。**

本页依据 v1 正文、算法和实验。对象为四个图像分类网络、SuSy 合成图像检测数据、ARM64 CPU，不能视为已验证的 LLM W3/W4 方法。通用预算背景见 [混合精度分配](../theory/mixed-precision-allocation.md)。

## 1. 单层干预测的是全网输出

浮点参考网络为 $f$。仅把第 $l$ 层变成 INT8、其他层保留 FP32，得到 $f^{(l)}$。对校准输入比较 $f(x)$ 与 $f^{(l)}(x)$ 的最终分类输出，而非该层中间激活。权重和激活精度在层内绑定，候选是 W8A8 与 FP32，不是单独选择权重位宽。

OAT（one at a time）需要一个浮点参考与 $L$ 个单层量化配置，避免遍历 $2^L$ 个组合，但忽略联合量化后的相互作用。以浮点状态测得的单层排名，不是联合最优分配的证明。与 [CASA](casa.md) 的跨层修正目标有联系，但 RAMP 不进行同样的方向加权或交换优化。

## 2. JSD 的对象与指标选择

对于归一化分类分布 $P,Q$，令 $M=(P+Q)/2$，

$$\operatorname{JSD}(P,Q)=\tfrac12\sum_iP_i\log\frac{P_i}{M_i}+\tfrac12\sum_iQ_i\log\frac{Q_i}{M_i}.$$

它对称，且自然对数下上界为 $\ln2$，以 2 为底时上界为 1。论文将其描述为输出 logits 上的散度，并写范围 $[0,1]$；严格实现必须先规定 logits→概率的映射、温度、对数底及跨样本归约。不能将任意有负值的 logits 直接当概率。本地仅有论文，未取得实现，这些参数尚不能从代码确认；softmax 是合理实现选择，不冒充作者已公开细节。

JSD 直接观察最终输出对扰动的响应，比权重局部误差更靠近分类决策；但输出分布变化小也不保证 argmax 不变，小间隔处尤其如此，见 [排序稳定性](../theory/quantization-ranking-stability.md)。

论文比较 13 种代理，包括参数量、标准差、通道范围、SNR/SQNR、HAWQ-V2 风格分数，以及最终输出的 MSE、MAE、cosine、KL、JSD。它先检查相关性，再用敏感层集合命中率：

$$\operatorname{Recall@Top-k}=|L_{\rm true}\cap L_{\rm proxy}|/k.$$

$L_{\rm true}$ 是按单层量化后的真实准确率降幅选出的 $k$ 层。两集合大小相等时 precision 与 recall 数值相同。标签参与**评价代理指标**，不能据“部署评分无需标签”称整个实验不使用标签。

作者的平均命中率 JSD/KL 同为 0.766，cosine 为 0.726，HAWQ-V2 风格代理为 0.509。KL 与 JSD 在所列阈值上的排名表现相同，选择 JSD 还依赖对称/有界等性质。这不是所有任务上 JSD 优于 KL 的证明，也不是对包含 QAT 的 [完整 HAWQ-V2](hawq-v2.md) 系统的等条件否定。

## 3. 从敏感度到候选策略

令层敏感度为 $\Omega_l$。一维 K-means 寻找 $K$ 组及均值 $\mu_k$，最小化

$$\sum_{k=1}^{K}\sum_{l\in C_k}(\Omega_l-\mu_k)^2.$$

与固定“最低 50% 层”不同，聚类允许大多数低分层落入一组，把少数高分层单独分组。论文主要用 $K=5$；层分数跨度大并不自动证明必须在 log 空间聚类，图中的对数纵轴也不等于算法输入取了对数。

**正文与算法需要区分：**正文描述把低敏感簇映射到 INT8，但 PDF 第 9 页算法实际按**簇中心作为阈值**，生成

$$p_l^{(k)}=\begin{cases}\mathrm{INT8},&\Omega_l\le\mu_{(k)},\\\mathrm{FP32},&\text{otherwise}.\end{cases}$$

这不是“前 $k$ 个完整簇都量化”。一个簇内高于中心的成员会被排除，最大中心也未必覆盖最大敏感度层。本页按明确算法记录，保留与正文叙述的差异；归档时官方仓库只有 README，不能用实现裁决。

候选配置需实际联合量化并测准确率/延迟，形成 Pareto 前沿，再选择 knee/elbow。这降低候选数，但没有解原始全部配置的约束优化，也没有保证任意内存或延迟预算可行。肘点还依赖坐标归一化、离散估计与质量要求；论文未完整给出可重现的选择细节。聚类 silhouette 高只支持分数组内相似，不证明任务精度或部署最优。

## 4. 为什么单层没有收益，也可能值得保持 INT8

常见局部成本表把全网时间写为 $\sum_lt_l(p_l)$。但相邻 INT8 算子可能融合，插入 FP32 层会切断整个整数子图。更接近问题的教学表达为

$$T(\boldsymbol p)=\sum_lt_l(p_l)+\sum_{(l,j)}c_{lj}(p_l,p_j)+F(\boldsymbol p),$$

其中 $c$ 包括精度边界转换，$F$ 表达融合、缓存和调度变化；这只是解释模型，并非论文实现的求解器。

作者测试了 JSD+Latency：先排除单层 INT8 加速不足的层，再对剩余层分配。该过滤在 8 组模型—硬件组合中让 4 组变慢、2 组改善、2 组不变。Pi 5 上 ConvNeXt-Tiny 从 94.6 ms 变为 115.5 ms，准确率同为 97.2%；ResNet-18 从 20.8 ms 变为 49.7 ms，换来 0.5 个百分点精度。不能只看单层 FP32 更快便断定回退它更优。

图中的 ConvNeXt 深度卷积聚合耗时从 7.7 ms 增至 16.5 ms，而 Q/DQ 部分没有明显增加。作者的机制解释重点是**破坏融合使算子区域变慢**，不是简单说多了转换节点就一定多出同量级转换耗时。所有数值都绑定 ONNX Runtime 默认 CPU provider 与目标 CPU；换后端后必须重测。（原文 graph fragmentation 实验。）

## 5. 质量与性能证据的口径

主实验是 Apple M1、Raspberry Pi 5 Cortex-A76，静态 PTQ、batch 1，模型为 ResNet-18、EfficientNet-B0、ConvNeXt-Tiny、TinyViT。报告绝对 top-1 与每图中位延迟。下面选两组保留量级：

| 平台/模型 | FP32 准确率 / 延迟 | RAMP 准确率 / 延迟 |
| --- | --- | --- |
| M1 / TinyViT | 97.5% / 39.0 ms | 97.3% / 27.4 ms |
| Pi 5 / ConvNeXt-Tiny | 97.2% / 167.9 ms | 97.2% / 94.6 ms |

作者定义 collapse 为相对 FP32 下降超过 15 个百分点。RAMP 在 8 组中零 collapse，**不等于零精度损失**：例如 M1 的 EfficientNet-B0 从 96.6% 到 95.1%。八组平均下降约 0.59 个百分点，平均加速约 1.81 倍。正文“completely avoids performance degradation”应理解为避免其定义的灾难性失败，不能照搬成无损。

论文一方面称最终策略可跨两个 ARM 平台迁移、不需要逐层延迟 profiling；另一方面选择部署点时明确使用候选的完整准确率—延迟 Pareto 前沿。因此可确认的是“敏感度与候选生成无需逐层硬件过滤”，不能扩大为整个部署决策无需目标设备测量。

SuSy 是特定图像分类数据集；论文没有充分给出阈值选择与最终测试的独立划分、校准样本数、线程与完整计时配置。对跨域、LLM、GPU 和其他运行后端的推广仍未验证。模型与硬件的结论限于作者测量，本项目未运行 ONNX 模型或性能实验。

## 来源身份

- [RAMP: Robust Adaptive Mixed-Precision Quantization for Edge CPU Vision Models](https://arxiv.org/abs/2609.28262v1)，arXiv:2609.28262v1；2026-09-26 归档。已读方法、算法、结果和限制；算法与主表回查 PDF 第 9 页。
