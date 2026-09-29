---
title: CASA：方向敏感的位宽分配与跨层交换
type: method
tags:
  - mixed-precision
  - second-order
  - ptq
sources:
  - raw/papers/2026-09-26/casa-cross-layer-sensitivity/paper.pdf
updated: 2026-09-28
---

# CASA：方向敏感的位宽分配与跨层交换

CASA 把“一个模块有多敏感”细化为“该候选的误差落在模块的哪些方向”，先求独立模块预算分配，再用相邻层误差关系提出位宽交换，最终依据模型校准损失接受交换。它不学习量化器本身，也没有消除低比特下二阶近似的局限。

本页依据 arXiv:2609.25916v1，核对正文、Algorithm 1、附录 C–D 及 G 的相关证明。阅读前可先看 [HAWQ-V2](hawq-v2.md) 的平均 trace 代理和 [曲率加权误差](../theory/curvature-weighted-quantization-error.md) 的双侧度量。本次未运行分配或模型评测；本地材料中无作者代码。

## 1. 一个标量究竟丢失多少信息

对线性权重 $W\in\mathbb R^{d_o\times d_i}$，记 $E=\widehat W-W$、输入二阶矩 $A$、输出侧曲率因子 $B$。忽略一阶项、跨模块项，并以 Kronecker 因子近似任务 Hessian 后，模块代价为

$$Q(E)=\operatorname{tr}(B E A E^\top).$$

它已是代理：任务驻点近似、经验梯度外积、因子独立近似并不自动成立。CASA 不是直接获取完整任务 Hessian。

标量代理写成 $\alpha\|E\|_F^2$。若 $A,B\succ0$，令 $m=\lambda_{min}(A)\lambda_{min}(B)$、$M=\lambda_{max}(A)\lambda_{max}(B)$，则

$$m\le\frac{Q(E)}{\|E\|_F^2}\le M.$$

上下界由对应特征向量外积形成的秩一误差达到。若用双向乘性比衡量最坏失真，固定 $\alpha>0$ 时为 $\max(M/\alpha,\alpha/m)$，两端平衡给出

$$\alpha^*=\sqrt{mM},\qquad D^*=\sqrt{M/m}=\sqrt{\kappa(A)\kappa(B)}.$$

这是 Theorem 3.1 的内容：即使最优标量也不能同时吻合全部误差方向。若存在零特征值及非零正曲率方向，对所有方向的有限乘性近似失效；这不表示实际模型损失无穷。

**结论的量词很重要。** 定理对所有非零矩阵误差取最坏情形，而实际 RTN/GPTQ 在有限档位产生的是很小的候选集合。因此极大的条件数表示存在严重失真的可能方向，不证明实际候选必然命中、不证明每对模块都会排序颠倒。附录 B 的条件数热图也受阻尼和小特征值估计影响。

如果两个模块真实单位失真代价为 $\gamma_1>\gamma_2$，标量代理却反排，预算只允许一个模块用高位，且低高位失真为 $d_L>d_H$，错误分配的额外代价是 $(\gamma_1-\gamma_2)(d_L-d_H)$。这是排序错误怎样转成预算后果的构造，不是每次运行的损失下界。（§3.2、附录 G。）

## 2. 第一阶段：实际使用对角因子的候选表

完整理论保留 $A,B$，**Algorithm 1 的实际统计只收集对角**：

$$a_j=\mathbb E[x_j^2],\quad b_i=\mathbb E[g_i^2],\qquad
\ell_{l,m,q}=\sum_{i,j}b_i^{(l,m)}a_j^{(l,m)}(E_{ij}^{(l,m,q)})^2.$$

它保留坐标之间不同权重，但舍弃输入/输出相关项。不能把实现描述成完整协方差重构，也不能拿完整矩阵定理直接证明对角近似无损。一次前向与反向校准可为全部候选共享统计；候选误差仍要分别生成。

每个 Transformer 层 $l$ 的模块 $m$（Q/K/V/O、Gate/Up/Down）从配置集合选一项。以二元变量 $P_{l,m,q}$ 表示选择，求

$$\min_P\sum_{l,m,q}P_{l,m,q}\ell_{l,m,q},\quad
\sum_qP_{l,m,q}=1,\quad
\sum_{l,m,q}P_{l,m,q}c_{l,m,q}\le B_{avg}\sum_{l,m}n_{l,m}.$$

$c$ 是候选实际位预算，$n$ 是参数数；论文说明 BPW 计入 scale/zero-point。不能用“模块数相等”代替参数和元数据成本相等。§5 用 2–8 整数位宽、RTN 生成代理误差，SCIP 求 MCKP；最终主量化器是 group size 128 的 GPTQ。因此 RTN 排名到 GPTQ 结果还存在代理迁移。

## 3. 闭式位宽公式能解释什么

§4.1 另设连续高码率误差模型：$\mathbb E[\operatorname{vec}(E_i)\operatorname{vec}(E_i)^\top]=2^{-2\beta_i}\Sigma_i$。定义 $\Gamma_i=\operatorname{tr}((A_i\otimes B_i)\Sigma_i)>0$，得到

$$\min_\beta\sum_i\Gamma_i2^{-2\beta_i},\qquad \sum_i n_i\beta_i\le B_{total}.$$

KKT 条件使边际收益每单位存储相等，给出无上下限情形

$$\beta_i^*=\tfrac12\log_2(\Gamma_i/n_i)+c,\qquad
c=\frac{B_{total}-\tfrac12\sum_i n_i\log_2(\Gamma_i/n_i)}{\sum_i n_i}.$$

若有上下限则需裁剪并重求活动集合的预算常数。这个公式解释“同样敏感度下，大模块提一位更贵”，也显示 $\Gamma_i/n_i$ 翻倍会使该模块相对其他模块的位宽差增加半位；固定总预算下，公共常数 c 也会重新调整。它是连续指数失真模型的最优解，**不是实际离散 MCKP、RTN 残差或 2 bit 极低码率下的精确公式**。不能把它与二元选择变量的线性规划松弛混为一谈。（相关证明见附录 G。）

## 4. 第二阶段：跨层项用于提案，校准损失用于接受

完整二阶模型还含 $E_l$ 与 $E_{l'}$ 的有符号交叉项。CASA 只近似相邻层同类模块，并通过 Cauchy–Schwarz 得到可由已有对角统计计算的绝对值上界：

$$\overline C_{q,q'}^{(l,m)}=
\sum_{i,j}\sqrt{b_i^{(l,m)}b_i^{(l+1,m)}}\sqrt{a_j^{(l,m)}a_j^{(l+1,m)}}
|E_{ij}^{(l,m,q)}|\,|E_{ij}^{(l+1,m,q')}|.$$

这省去额外跨层校准统计，却丢掉误差符号与抵消；它不是测得的跨层 Hessian。并且只适用于相应形状能按坐标配对的模块，不能无说明推广到任意架构相邻张量。

按照式 16 和 Algorithm 1，筛选分数是

$$J_{cross}(q)=2\sum_{l,m}\overline C_{q_{l,m},q_{l+1,m}}^{(l,m)},\quad
\Delta_{cross}(s)=J_{cross}(q\oplus s)-J_{cross}(q).$$

它**没有把变化的 self-term 加回筛选分数**。正文同时称其改进“total objective”，容易误导；实际算法应以上述明确步骤为准：

1. 从第一阶段配置出发，枚举预算内一次升档配一次降档的交换。模块参数量不同，升降一档不必等成本，必须检查实际 BPW。
2. 只保留 $\Delta_{cross}<0$ 中最优的至多 $K=100$ 项。
3. 应用候选配置，用主量化流程评估校准集 next-token loss。
4. 只接受严格低于当前校准损失的候选；无改善即停止，最多 20 轮。

最多 20×100 个候选模型评估不是零成本；共享统计不代表无需重复量化和前向计算。论文在单张 B200 上做实验，本地未核验搜索耗时或缓存复用。

## 5. 三种“保证”必须分开

Proposition 4.2 中，全局求解带跨层二次目标的最优配置，不差于在相同目标下评价第一阶段配置，这是在同一可行集上的全局最优性质。它没有证明上面的候选截断与局部搜索能找到该全局最优。

Algorithm 1 能直接保证的是**被接受迭代的实测校准损失单调下降**，不能推出独立 WikiText-2 PPL 或任务准确率单调改善。原表 2 就有反例：Llama-2-7B / 2.25 BPW，self 的 PPL 11.01、完整 CASA 11.19，而平均准确率 42.1→42.7；Qwen3-8B / 3.0 BPW 的 PPL 9.41→9.46，准确率 60.2→61.5。

绝对值上界也不保证识别有益抵消。附录 G 的教学构造 $H=\begin{pmatrix}1&\rho\\\rho&1\end{pmatrix}$、$e=\sqrt R(s,t)$、$s,t\in\{-1,1\},0<\rho<1$，self 都为 $2R$，完整代价却是 $2R+2\rho Rst$。同号与反号可相差 $(1+\rho)/(1-\rho)$ 倍；只看绝对值无法区分二者。

## 6. 实验支持及仍需保留的边界

主实验覆盖 Llama-2-7B、Llama-3/3.1-8B、Qwen3-8B/14B，C4 校准、WikiText-2 PPL 和常识任务平均准确率。表 2 的代表结果如下，均为作者报告：

| 模型 / BPW | Q-Palette 的 PPL / 平均准确率 | CASA self | 完整 CASA |
| --- | ---: | ---: | ---: |
| Llama-3-8B / 2.25 | 47.20 / 34.0 | 35.75 / 35.6 | 22.41 / 39.2 |
| Llama-3-8B / 2.50 | 21.43 / 39.2 | 13.21 / 45.3 | 11.86 / 47.0 |
| Llama-2-7B / 2.25 | 11.59 / 42.4 | 11.01 / 42.1 | 11.19 / 42.7 |

Llama-3-8B 浮点参考为 5.49 / 65.0，低位下改善不等于恢复浮点质量。这里 Q-Palette 原 condensed quantizer 被作者替换为 GPTQ，因此是在共享量化器下比较分配指标，并非原方法完整配方的直接排名。2.25 BPW 行的 uniform 2 bit 带脚注，只作参考，不是严格同预算对照。

附录 D 的 self-only 局部搜索对照有助于区分“只是多做搜索”与跨层筛选的贡献，15 个设置中 CASA 在 13 个取得更低 PPL；仍不是普遍保证。表 3 的五模型条件数与收益相关性也不能升格为因果关系或通用质量预测器。附录 C 表 4 用四任务均值，不能与主表均值直接拼接比较。

附录 C 尝试把跨层项一起放入整数规划，并把部分异常归因于 McCormick 的 LP 松弛过松。这里需补充数学边界：二元变量配合完整 McCormick 约束时，乘积线性化在整数可行点上是精确的；松弛弱会影响求解难度，但若整数问题已被精确求到最优，不能仅以 LP 松弛弱解释最终质量。缺少求解 gap、状态或额外近似证据时，这不是“所有联合优化都不可行”的证明。

## 来源身份

| 来源 | 版本 | 范围 |
| --- | --- | --- |
| [Beyond Scalar Sensitivity: Activation-Aware Mixed-Precision LLM Quantization with Cross-Layer Refinement](https://arxiv.org/abs/2609.25916v1) | arXiv:2609.25916v1 | Theorem 3.1、§4–5、表 2–3、Algorithm 1、附录 C–D/G；公式、主结果及算法回查页图 |
