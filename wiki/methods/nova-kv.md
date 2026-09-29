---
title: NOVA-KV：查询加权变换与等体积分组向量量化
type: method
tags:
  - kv-cache
  - vector-quantization
  - attention
  - calibration
sources:
  - raw/papers/2026-09-21/nova-kv/paper.pdf
updated: 2026-09-29
---

# NOVA-KV：查询加权变换与等体积分组向量量化

NOVA-KV 用校准 query 的二阶矩把 K 的量化误差映射为 attention logit 误差，再通过混合高低方差坐标，让每个小组都使用固定宽度的码本索引。实际方案是 K 的 4 维 VQ 与 V 的旋转后 INT2 SQ，另保留 BF16 的开头和近期 token；“2-bit”只描述主 payload。

本页核对 v1 正文、算法与系统附录 B、理论 E–H、混合架构 J 和 chunked prefill 实验 K，并参照附录 I 的 LooGLE 对照；没有运行作者服务或独立核对其代码。前置见 [KV 对象与粒度](../theory/kv-cache-quantization-objects-and-granularity.md)、[码本与位宽预算](../theory/codebook-quantization-and-bit-budget.md)。

## 1. 先区分三个失真目标

采用行向量，$Q,K\in\mathbb R^{T\times d}$ 为 RoPE 后张量，$S=\operatorname{softmax}(QK^\top/\sqrt d)$，$O=SV$。记 $\Delta S=S-\widehat S$、$\Delta V=V-\widehat V$，则精确有

$$O-\widehat O=\Delta S V+S\Delta V-\Delta S\Delta V.$$

K 与 V 分别优化前两项，并不是共同最小化最终输出误差；忽略最后一项是小误差近似。softmax 的谱范数 Lipschitz 常数不超过 $1/2$，所以在相同可见 token/mask 下

$$\|S-\widehat S\|_F\leq\frac{1}{2\sqrt d}\|Q(K-\widehat K)^\top\|_F.$$

这一界允许用 logit 误差作为 K 的代理，但“更低 logit MSE”不保证每个问题答案更好，也不直接保证最高分 token 排名不变。

固定校准查询时，令 $M_q=Q^\top Q$，则

$$\|Q(K-\widehat K)^\top\|_F^2=\sum_j(k_j-\widehat k_j)M_q(k_j-\widehat k_j)^\top.$$

这是精确恒等式，$M_q$ 是未去均值的二阶矩，不是中心化 covariance。GQA 下应把共享同一 KV head 的 query heads 的二阶矩相加。它描述校准查询的加权方向，无法保证未来自回归查询分布不漂移。

## 2. K 的变换为什么通常不是正交矩阵

令 $\bar k$ 为校准 key 均值，$\widetilde S_k=\sum_j(k_j-\bar k)^\top(k_j-\bar k)$。构造

$$M_q^{1/2}\widetilde S_k M_q^{1/2}=E\Lambda E^\top,\qquad R_K=M_q^{1/2}E.$$

变换 $r=(k-\bar k)R_K$，反变换 $\widehat k=\widehat rR_K^{-1}+\bar k$。满秩时 $R_KR_K^\top=M_q$，于是

$$\|r-\widehat r\|_2^2=(k-\widehat k)M_q(k-\widehat k)^\top.$$

这条广义 Parseval 关系对任意重构误差精确成立，不依赖 Gaussian 或高分辨率假设；但前提是同一正定 $M_q$、满秩变换和准确算术。实际对 $M_q$ 的特征值下限设为 $10^{-30}$，服务时变换转成 BF16，不能把符号等式当作数值误差为零的证明。

$M_q^{1/2}$ 改变距离度量，$E$ 再把加权 K 的能量解相关。高分辨率 companding 分析给出 $RR^\top=cM_q$ 的最优形状条件；低秩分析选择 $E$ 的前 $p$ 列，最小残差是尾部特征值和。两者都是说明变换的理由，实际部署仍使用满秩 $p=d$ 后量化，不能称为直接删除低能量维度。$M_q\propto I$ 才退化到缩放后的普通 K-PCA；任意正交旋转一般不会实现查询加权距离。

共同减去 $\bar k$ 会让一个 query 的所有 token logits 减去同一常数，因此普通 softmax 下无需在读缓存时加回。但这一性质要求所有参与归一化的项同步处理。GPT-OSS 的 learned sink logit 是额外归一化项，作者专门对它施加同样的偏移，并在 split 合并后只加一次 sink；不能直接套用“不必处理均值”的简化到所有模型。

## 3. V 的理论与实际路径应分开

V 的误差是 $\|S(V-\widehat V)\|_F^2$。$M_s=S^\top S$ 在 token 轴上耦合误差，$M_o=V^\top M_sV=O^\top O$ 在 head 维度上汇总它。最优 rank-$p$ 线性投影取 $M_o$ 的前 $p$ 个特征向量；只需累计 $d\times d$ 输出二阶矩，无须显式建立 $T\times T$ 的 $M_s$。

这并不意味着任何逐 token 的 V-MSE 都等于输出误差。例如一个 query 对两 token 权重均为 $1/2$：误差 $(a,-a)$ 完全抵消，$(a,a)$ 完全叠加，二者逐 token 平方误差相同。一个 head 空间正交变换不会消除这种 token 间耦合。

**部署的 V 路径沿用 OSCAR 的旋转和 INT2 SQ，而非直接部署上述推导的输出 PCA+VQ。** 附录 B.5 比较 PCA 型 $U_S$ 和 $U_S$ 后再 Hadamard 的 VQ，两者相对部署 SQ 在 RULER 上只改善约 0–2 点，作者因此选择更简单的 SQ 读取；V 逆旋转吸收到输出投影中。

## 4. 为什么分组要混合大小方差坐标

若变换坐标独立 Gaussian，第 $\ell$ 组有 $g$ 维，体积参数 $v_\ell=\prod_{i\in G_\ell}\sigma_i^2$，每坐标分配 $b_\ell$ bit，高分辨率近似为

$$D_\ell\simeq C_g2^{-2b_\ell}v_\ell^{1/g}.$$

总预算 $\sum_\ell b_\ell=Lb$，$L=d/g$，把位宽暂视为任意实数，最优分配是

$$b_\ell^*=b+\frac{1}{2g}\log_2\frac{v_\ell}{(\prod_m v_m)^{1/L}}.$$

其总失真与如何分组无关；但固定宽度 $b_\ell=b$ 时，总失真正比于 $\sum_\ell v_\ell^{1/g}$，由 AM–GM，只有各组体积相同才能达到这个无约束模型的最优值。直观上，把大方差全部放同一组，会让这一组的 256 个中心不够用，同时浪费小方差组的中心。

部署做法是按方差降序排序，然后将坐标轮流发给 $L$ 个组，再把置换折入 $R_K$。它是近似均衡的确定性启发式，**不是任意谱都能精确等体积的保证**。$g=4,b=2$ 时每组 $2^{gb}=256$ 个四维中心，索引固定 8 bit，保留规则的 cache layout。

理论的三个边界必须保留：真实位宽不能任意为负或连续；独立 Gaussian 与高分辨率不自动适用于 2 bit；部署还把每 token 的 $r$ 除以 $\rho=\|r\|_2/\sqrt d$，存储 $\rho$，这个归一化不在定理 2 的模型中。附录 H 特意关闭 RMS 归一化检验理论，在 held-out 系数上拟合位宽指数约 1.82–1.88，低于渐近值 2；2-bit 方差排序分组的理论失真比实测高 81.5%。这支持分组排序趋势，不证明部署失真精确服从公式。附录 G.2 的 Gaussian 率失真讨论也不能变成有限码本达到下界的保证。

## 5. 校准、编码、读取与实际开销

每 layer/KV head 独立统计、变换和训练码本。原文采用 198 个 GPQA-Diamond prompt 的 prefill 数据，使用长序列覆盖 RoPE 位置；不使用正确答案或生成 token 来拟合。GPQA 也出现在评测中，故不能宣称完全跨域校准；MMLU 替代校准的实验提供了部分跨域证据。每组 k-means 最多 25 轮，空簇用高误差样本重新初始化；FP16 中心离线映射到 FP8，转换矩阵离线高精度保存、服务时 BF16。

写入时：减均值、乘 $R_K$、按 RMS 归一化、分组最近中心搜索、保存索引和 scale。近期 token 先进入 BF16 带，再以 8 步为块量化历史。读取时：每步先计算 $\widetilde q=qR_K^{-\top}$，attention kernel 根据索引取中心并恢复 scale，在变换域直接累加 logits；不对每个历史 K 做一次稠密逆变换。查表与量化仍有指令和写入代价，不等于零成本。

对 $d=128$，历史 K 是 $2+16/128=2.125$ BPE；V 是 $2+32/128=2.25$ BPE，平均 2.1875。首 64 与最近 256 token 保留 BF16，在 128K 时平均约 2.22 BPE。更短上下文保护带占比更大。模型级变换、均值、码本不计入这个 cache BPE，必须另算：原文 Qwen3-8B 给出约 0.30 Gbit，128K、batch 1 摊销约 0.03 BPE，批量增加后摊薄。GPT-OSS 的窗口层保持 BF16且 $d=64$，不能照抄 2.22 的账。

分块 prefill 会读取已量化的前序块，影响后续隐藏状态。保留整个 in-flight prompt 的精确副本可消除此路径误差，但使 prefill 期间缓存接近 BF16 大小；原文附录 K 单独对照了该取舍。这也是跨论文比较 TurboQuant、NOVA-KV 时必须核对的配置。

## 6. 结果支持什么

以下均为作者报告，并非本地复现。

| 对照 | 结果 | 应如何解读 |
| --- | --- | --- |
| Qwen3-8B，RULER 128K | BF16 83.4，OSCAR 25.3，NOVA-KV 75.4 | 改善很大，但仍比 BF16 低 8 点 |
| 同模型 64K/128K，NOVA 变换+SQ | 两项均 0；配 VQ 为 76.4/75.4 | 不能单独部署能量集中的变换并沿用统一 SQ |
| 128K 分组消融 | 方差排序 37.2，随机 74.4，均衡 75.4 | 混合坐标贡献主要收益，均衡相对随机还有小幅改善 |
| 五项生成任务平均，Qwen3-8B | BF16 74.7，NOVA 72.9，OSCAR 72.2 | 未检出显著差异不等于等效或无损 |
| GPT-OSS 五项平均 | BF16 76.5，NOVA 72.4，OSCAR 13.9 | 本配置有明显架构敏感性，不是所有 SQ 的普遍失败 |
| LooGLE，Qwen3-8B | 31.50 / 28.81 / 26.38，分别为 BF16/NOVA/OSCAR | NOVA 对 BF16 的 −2.69 差异仍显著，不能概括为全面无损 |

RULER 的三次、生成任务的五次采样只反映生成/评测波动，压缩参数固定，没有测重新校准或重训码本的方差。作者 McNemar 检验在每题多次采样多数投票后进行；$p>0.05$ 不构成性能相等证明。

吞吐主图在单 H100、30K–90K 输入、排除 prefill/TTFT 的 decode 窗口测量，Qwen 报告约 1.6–3.4× BF16，和 OSCAR 接近；较大批量还受 BF16 容量上限影响。不能泛化成任意请求的端到端加速。附录 8K profiling 中 Qwen3-8B、batch 128，NOVA CUDA 为 49.7 ms/步、OSCAR 39.3 ms/步，NOVA 慢约 26%，其中编码/准备占 5.3 ms。GPT-OSS 的短上下文也不保证缩短每步时间。主图的“相当吞吐”必须附上上下文和批量条件。

## 7. 与 TurboQuant / QJL 的关系及后续验证

[TurboQuant](turboquant.md)用随机旋转后的分布设计免数据校准量化器，[QJL](qjl.md)针对内积建立无偏估计；NOVA-KV 用校准查询的方向性选择非正交距离，再训练小码本。它们优化对象、统计来源、保护带、元数据和 prefill 路径不同，不能仅用名义 bit 数排优劣。

NOVA 的可迁移认识是：**K 重构应按读取它的查询方向衡量，规则布局可以通过分组设计改善码率利用。** 需要本地验证的则是校准外 logit/输出误差、含保护带及码本的总显存、长短请求的编码开销、混合 attention 的 sink/均值一致性，以及整段 prefill 到 decode 的真实收益。

## 来源身份

- [NOVA-KV，arXiv:2608.04074v1](https://arxiv.org/abs/2608.04074v1)：§3–4 与附录 E–H 支撑变换/分组；§5、附录 B/C/I/J/K 支撑配置、消融和系统边界。V 的误差抵消例子与上述精确性边界是整理者推导。
