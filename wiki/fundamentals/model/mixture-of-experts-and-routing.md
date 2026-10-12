---
title: MoE 与专家路由：稀疏计算、容量与量化对象
type: concept
tags:
  - moe
  - routing
  - weight-quantization
  - calibration
aliases:
  - mixture-of-experts
  - DeepSeekMoE
sources:
  - raw/papers/2026-09-21/deepseek-v2/paper.pdf
updated: 2026-10-07
---

# MoE 与专家路由：稀疏计算、容量与量化对象

Mixture-of-Experts（MoE）用多个专家网络提供参数容量，让每个 token 只调用其中一部分。由此分开了总参数量与单 token 的参与计算参数量，同时引入专家选择、负载与通信问题。本页以 DeepSeek-V2 v5 的 DeepSeekMoE 为具体例子，解释其路由和量化关系；不同 MoE 的 gate、共享专家和路由策略须按各自定义核对。

在 [稠密 decoder](transformer-autoregressive-inference.md) 中，每个 token 经过同一个 FFN。这里替换的是 FFN 子层：token 经 router 分配到专家，专家输出汇合回隐藏状态，再沿残差计算继续。输入序列和 attention 的因果可见性仍然存在，按专家选择并不等于删除上下文 token。V2 的 attention 则采用 [MLA](multi-head-latent-attention.md)，它是另一项结构变化。

## 1. 共享专家、路由专家与 gate 分别做什么

采用论文列向量约定。设 FFN 输入 $u_t\in\mathbb R^D$，共享专家数为 $N_s$，路由专家数为 $N_r$，每 token 选 $K_r$ 个路由专家。每个专家 $f_i:\mathbb R^D\to\mathbb R^D$ 是一个 FFN；这里先把它视为完整函数，避免将内部实现与路由混在一起。

§2.2.1，式 (20)–(22) 的基本计算是

$$r_{i,t}=u_t^Te_i,\qquad
s_{i,t}=\operatorname{softmax}_i(r_{i,t}),\qquad
g_{i,t}=s_{i,t}\,\mathbf 1\{i\in\mathcal J_t\},$$

$$h'_t=u_t+\sum_{i=1}^{N_s}f_i^{(s)}(u_t)
+\sum_{i=1}^{N_r}g_{i,t}f_i^{(r)}(u_t).$$

$e_i\in\mathbb R^D$ 是 router 中对应专家的可学习向量，$\mathcal J_t$ 是选中的专家集合。基本 top-$K_r$ 路由按 affinity $s_{i,t}$ 选择；softmax 保持 logits 的排序。完整模型还会施加下一节的设备限制。

- **共享专家**对所有 token 执行，按上述公式直接加入输出。
- **路由专家**只对被分配给它的 token 执行，输出乘对应 gate。
- **router**既决定离散的专家集合，也提供连续的输出权重。

原文式 (21) 保留选中专家原来的 $s_{i,t}$，没有写出选中集合内再归一化。虽然所有 routed affinity 之和为 1，截断后的 gate 之和一般不为 1；不能自行补入重归一化并称其为本版本公式。实际实现是否存在额外缩放，应另读对应代码。

DeepSeekMoE 的两个设计意图是将专家切得更细，以组合更多专业化路径；将共同信息交给共享专家，减少路由专家之间的冗余。它们是作者的架构动机，不意味着专家已被证明具有固定、可命名的语义分工。

§3.1.2 给出的 V2 配置为 60 层、隐藏维 5120；第一层 FFN 保持稠密，其余 59 层使用 MoE。每个 MoE 层有 2 个共享专家、160 个路由专家，每 token 选 6 个路由专家；各专家中间宽度为 1536。全模型总参数为 236B、每 token 激活参数为 21B，其中还包括 attention 等部分，不能只按 $6/160$ 推算整模型比例。

## 2. 设备限制怎样改变选择过程

专家并行（EP）把不同专家部署到不同设备；一个 token 选到的专家越分散，隐藏状态要发往的设备就可能越多。这里的分工与 [张量并行](../../implementation/tensor-parallel-quantization.md)不同：EP 划分专家，TP 划分一个算子或参数张量的计算。

从路由公式到执行，需要保留 token 原位置、专家 ID 与 gate 的对应关系：按专家汇集输入行，各专家处理自己的 token 子批次，再按原 token 汇合加权输出。EP 下还涉及隐藏状态发送与专家结果返回；这些逻辑依赖由公式决定，具体 dispatch、汇合与通信布局仍由实现承担。

[MoE 专家并行执行](../../implementation/moe-expert-parallel-execution.md)进一步解释全局与本地 expert 编号、目的 rank 与专家配对计数、通信量化和输出回传。执行索引与放置需要匹配上面的专家函数，不自行改变模型路由定义。

§2.2.2 的 device-limited routing 按两步执行：先根据专家 affinity 选择 $M$ 个候选设备，再在这些设备上的专家中选 top-$K_r$。原文没有给出完整的设备评分聚合公式，本页不补猜成某种固定求和或最大值规则。

V2 使用 8 路专家并行，$M=3,K_r=6$，因此 6 个被选专家最多落在 3 个设备上。这限制了每 token 的目的设备数量；每个设备可能接收该 token 后计算多个专家。设备预选择也可能改变相对全局 top-$K_r$ 的结果，所以它是模型路由规则的一部分。

§3.1.3 报告训练使用 16 路流水并行、8 路专家并行和 ZeRO-1，不依赖 TP；共享专家计算与专家间 all-to-all 通信重叠。这些是该训练系统的选择，不是 MoE 在所有硬件上都无需 TP 或通信可完全隐藏的保证。

## 3. 平衡损失与 token dropping 解决什么问题

自动学习的 router 可能把大量 token 送往少数专家：冷门专家训练不足，热点设备拖慢整个步骤。§2.2.3 为三种不同对象设置辅助损失：

| 对象 | 关注的负载 | 作用 |
| --- | --- | --- |
| 专家 | 每个专家被选次数与平均 affinity | 缓解路由塌缩和专家利用不足 |
| 设备计算 | 同一设备上专家的合计工作 | 缓解不同设备计算量不均 |
| 接收通信 | 每个设备接收多少 token 的隐藏状态 | 缓解目的设备通信热点 |

以 expert-level loss 为例，一段序列有 $T$ 个 token，

$$f_i=\frac{N_r}{K_rT}\sum_{t=1}^T\mathbf 1\{i\in\mathcal J_t\},\qquad
P_i=\frac1T\sum_{t=1}^Ts_{i,t},\qquad
\mathcal L_{\rm ExpBal}=\alpha_1\sum_i f_iP_i.$$

$f_i$ 是经过均匀使用基准归一化的选择频率，$P_i$ 是平均路由概率，$\alpha_1$ 控制辅助目标的权重。均匀使用时 $f_i=1$、$P_i=1/N_r$。辅助损失鼓励平衡，**不保证每一步严格均匀**；§2.2.4 明确指出这一限制。

作者因此在训练中加入设备级 token-dropping：以设备平均计算预算作为容量，capacity factor 为 1.0，按 affinity 丢弃低分的专家处理任务，直到符合预算。约 10% 训练序列被保护、不做 dropping；§3.1.2 说明评测不做 token dropping。这里丢弃的是 MoE 设备上的处理，不能解释为从输入序列或历史 KV 中删掉这个 token，也不能把训练中的容量策略直接当作评测执行语义。

## 4. 总容量、单 token 计算与批次成本为何不同

下面是根据结构整理的记账关系。若共享专家和路由专家各有 $P_s,P_r$ 个参数，router 有 $P_{\mathrm{router}}$ 个参数，一个专家层的存储参数量为

$$P_{\mathrm{total}}=N_sP_s+N_rP_r+P_{\mathrm{router}}.$$

一个 token 的参与计算参数量约为

$$P_{\mathrm{active}}=N_sP_s+K_rP_r+P_{\mathrm{router}}.$$

这些是专家层的参数记账，尚未加入 attention、embedding 等；参数量也不是 FLOPs 或字节流量的完整模型。它们帮助区分三个问题：

- **权重是否需要保存？** 若模型完全常驻，各专家权重仍须保存，只是位置可由 EP 分散。少激活参数不自动减少到同等比例的常驻权重内存。
- **批次访问了多少专家？** 多个 token 的专家集合并集可能远大于 $K_r$，不能把单 token 的访问比例当作整个 batch 的访问比例。
- **专家 GEMM 的行数是多少？** 是实际送到该专家的 token 数 $M_i$，不是整个批次 token 数；热门与冷门专家的形状不同，还涉及 dispatch、汇合与通信。

因此 MoE 的量化执行既要核对专家权重的表示，也要核对实际 token 分布和 [矩阵乘执行路径](../../implementation/quantized-matmul-scaling-execution.md)。236B/21B 不是可直接套用的吞吐倍数。

## 5. 量化误差可以同时改变专家选择与输出

以下是整理者从结构得出的分析，原报告没有据此提出最优 MoE 量化配置。

**路由选择存在离散边界。** 量化隐藏状态或 router 参数可能改变 $r_{i,t}$。对于没有设备预选择的直接 top-$K_r$，若每个 logit 扰动的绝对值不超过 $\epsilon$，且原第 $K_r$ 与第 $K_r+1$ 名之间严格间隔大于 $2\epsilon$，所选集合保持不变；推导与 ties 条件见 [排序稳定性](../../theory/quantization-ranking-stability.md)。设备候选也会变化的路由，还须单独检查设备选择；参数的位宽本身并不提供这个扰动界。

**集合稳定仍不等于输出稳定。** 固定输入与集合 $\mathcal J_t$，记一个被选专家的输出为 $v_i=f_i^{(r)}(u_t)$，gate 与输出误差分别为 $\delta g_i,\delta v_i$，则 routed 输出误差为

$$\delta y_r=\sum_{i\in\mathcal J_t}
\left(g_i\delta v_i+\delta g_iv_i+\delta g_i\delta v_i\right).$$

还要加入共享专家的输出误差；若输入也改变，专家输入和残差也会改变。若集合改变，会出现新专家进入、旧专家退出的贡献，不能只分析原先选中的专家重构误差。router 参数虽少，其影响位置也不能由参数占比判断。

**专家校准取决于路由条件。** routed expert 看到的是被 router 分配来的输入分布，共享专家则处理全部 token。总校准 token 多，不代表每个专家都有足够覆盖；低频专家的重要性也不只由平均频率决定。这是结构带来的统计条件，可接到 [校准与范围选择](../../theory/calibration-and-range-selection.md)和 [混合精度分配](../../theory/mixed-precision-allocation.md)，不能据此直接宣布所有专家必须统一精度，或固定采用某种 router 位宽。

## 6. 报告的核心观点与本页边界

DeepSeek-V2 的效率思路同时作用于两种成本：DeepSeekMoE 减少单 token 使用的 FFN 参数，MLA 减少历史 attention 表示；共享专家、设备限制、平衡目标与系统优化处理稀疏计算附带的训练和通信问题。作者用 236B 总参数、21B 激活参数模型及相应训练、质量和服务结果说明这一组合的可行性，不能只用稀疏比例解释全部收益。

报告还包含 8.1T token 预训练、长上下文扩展、SFT 与基于 GRPO 的对齐。这里以 §2.2、§3.1.2/3.1.3 的架构和必要训练条件为主要依据，研读训练、推理效率与对齐说明以辨别整体贡献；未将对齐算法展开为本页的另一条主题，也未逐项审查全部任务表或归因每项质量变化。服务中的 FP8 参数与平均 6-bit 缓存是作者报告，完整格式、专家算子与路由路径不由该报告给出；本次未核读这些实现代码。

本页完成的是模型结构和量化对象的解释。[MoE 路由与量化专家执行](../../implementation/moe-routing-and-quantized-expert-execution.md)沿固定 SGLang 版本连接 top-k、token–专家配对、块尺度、两次专家 GEMM 与共享输出。该页选定的有偏 sigmoid 路由是实现类支持的另一配置，不能替换上面的 V2 论文公式；源码核读也不表示已运行或测量这些路径。

## 来源身份

- DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model（`https://arxiv.org/abs/2405.04434v5`），`arXiv:2405.04434v5`，2024-06-19 修订；采用已归档的同版本 PDF 与 TeX 源包。路由公式和训练条件定位见正文；容量记账、误差展开和校准联系是本页整理者分析。本页没有用 DeepSeek-V3 的路由规则替代 V2。
