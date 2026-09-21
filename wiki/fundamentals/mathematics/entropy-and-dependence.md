---
title: 熵、条件熵与依赖：量化代理指标的解释边界
type: concept
tags:
  - information-theory
  - sensitivity
sources:
  - raw/papers/2026-09-21/luq/paper.pdf
  - raw/articles/2026-09-21/mit-6441-lecture-1/lecture.pdf
  - raw/papers/2026-09-21/q-vlm/paper.pdf
updated: 2026-09-15
---

# 熵、条件熵与依赖：量化代理指标的解释边界

熵描述一个指定概率分布的不确定性。用它评价量化层之前，需要先说清随机变量、概率怎么估计、在哪个维度归约，以及该指标与任务误差有什么证据联系。本页用离散变量说明，不将连续激活、量化码和归一化幅度自动视作同一个分布。

## 1. 熵在测量什么

设离散随机变量 $U$ 的概率为 $p(u)$，则

$$
H(U)=-\sum_u p(u)\log_2p(u),\qquad 0\log_2 0:=0.
$$

其单位是 bit；自然对数对应 nat。以二元变量为教学例子，等概率两种结果的熵为 1 bit，完全确定的结果为 0。此处概率必须非负、总和为 1，不能直接把任意张量代入 $-x\log x$。（MIT 6.441，Spring 2010，Lecture 1，第 3 页）

对量化激活可以建立不同统计对象：码值频率描述各码出现多少次；对单个浮点数的软量化概率描述它在候选网格点间的分配；沿 token 归一化幅度描述数值集中在哪些位置。它们回答不同问题，计算出的“熵”不能不加区分地比较。

## 2. 条件熵与互信息的区别

给定联合分布 $p(u,v)$，

$$
H(V\mid U)=-\sum_{u,v}p(u,v)\log_2p(v\mid u),
\qquad I(U;V)=H(V)-H(V\mid U).
$$

条件熵表示知道 $U$ 后，$V$ 还剩多少不确定性；互信息表示知道 $U$ 消除了多少关于 $V$ 的不确定性。链式关系为 $H(U,V)=H(U)+H(V\mid U)$。（MIT 讲义第 4、6 页）

**教学反例：**令 $U$ 为公平的二元变量。若 $V=U$，则 $H(V)=1$、$H(V\mid U)=0$、$I(U;V)=1$；若 $V$ 是与 $U$ 独立的公平二元变量，则 $H(V)$ 仍为 1，但 $H(V\mid U)=1$、$I(U;V)=0$。因此在相同边缘分布下，较大的条件熵不能被直接解释为更强的统计依赖。

同理，仅知道两个边缘分布不能确定联合分布；逐元素相乘后归一化也不会自动获得真实的跨层联合概率。确定性网络若以完整层状态作为条件、且无额外随机性，下一状态的条件熵可以为零，但该层对扰动依然可能很敏感。概率依赖、函数敏感性与量化误差传播是不同对象。

## 3. 怎样理解量化中的熵代理

[Q-VLM](../../methods/q-vlm.md) 式 4 采用条件熵形式，把它作为联合校准相对逐层校准的收益代理；图 2/5 用有限模型、层和样本展示其与误差差异的关联。这是经验代理的依据，不是信息论直接保证“条件熵越大，量化误差依赖越强”。

复用这类指标时，首先复现概率构造与归约方式，再检查它是否预测指定的校准收益或任务误差。更换位宽、网格、输入长度或模型后，原来的阈值不一定可比。[量化误差诊断](../../implementation/quantization-error-diagnosis.md) 还要求区分代理目标改善、输出误差改善和任务质量改善。

本页例子是根据定义构造的数学检验，不是对 Q-VLM 实验的复现，也不否认一个缺少普遍定理的代理在特定条件下可能有效。

## 4. token 向量聚类的熵与量化敏感性

[LUQ](../../methods/luq.md) v3 §3.2 对完整 token 向量做 K-means，以每簇有效 token 数除总数形成 $p_k$，再计算 $H(C)=-\sum_kp_k\log p_k$。随机变量是簇标签 $C$，不是某个浮点坐标，也不是前文跨层条件熵。固定 $K$、使用底 2 时，$0\le H(C)\le\log_2K$；例如四个等概率簇为 2 bit，全部落在一个簇为 0。

该指标衡量所选聚类的占用分布，不直接给出任务误差。对于同时旋转的向量与中心，$\|(z-\mu)R\|_2=\|z-\mu\|_2$，对应的聚类解与熵保持不变；但有限次随机 K-means 不保证返回同一局部解。论文附录 B 通过层排序稳定性选 $K=100$，属于经验设置，需要保留数据和聚类条件。

**教学反例：**令向量 $(0.4,0.4)$ 用步长 1 的坐标网格舍入，平方误差为 0.32。旋转到 $(\sqrt{0.32},0)$ 后再量化，误差约为 0.1886；对应旋转不会改变欧氏聚类距离，却改变量化误差。因此，旋转不变的熵不能独自决定坐标量化的难度。[正交旋转与量化](../../theory/orthogonal-rotation-and-hadamard-quantization.md) 解释了这种表示差别，[混合精度分配](../../theory/mixed-precision-allocation.md) 则说明代理排序如何限制可选配置。

## 来源身份

下表用于在没有本地资料库时辨识来源；具体论述的章节、公式、图表或代码位置见正文。

| 来源 | 版本或快照 | 说明 |
| --- | --- | --- |
| [LUQ: Layerwise Ultra-Low Bit Quantization for Multimodal Large Language Models](https://arxiv.org/abs/2509.23729v3) | `arXiv:2509.23729v3` | — |
| [MIT 6.441 Information Theory, Lecture 1](https://ocw.mit.edu/courses/6-441-information-theory-spring-2010/resources/mit6_441s10_lec01/) | `Spring 2010, Lecture 1` | 获取：None；标识：2010-mit-6441-lecture-1-snapshot-2026-09-14 |
| [Q-VLM: Post-training Quantization for Large Vision-Language Models](https://arxiv.org/abs/2410.08119v3) | `arXiv:2410.08119v3` | — |
