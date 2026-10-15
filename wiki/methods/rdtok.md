---
title: RDTok：不依赖问题的视觉特征码率分配
type: method
tags:
  - vlm
  - token-compression
  - rate-distortion
  - optimization
sources:
  - raw/papers/2026-10-03/rdtok
updated: 2026-10-15
---

# RDTok：不依赖问题的视觉特征码率分配

RDTok保留视觉token的全部位置，给不同特征向量分配不同编码码率。它先测每个向量在不同码率下的重构误差，再在总系数预算内用整数动态规划分配码率。编码不读取用户问题，适合一份图像表示供多次提问复用；它没有通过减少序列长度来节省LLM attention或KV。

本页依据Query Independent Variable Rate Visual Token Coding v1，完整主文及5页原件图表。未归档作者代码或运行模型。连续注水与实际bit成本复用[码本与率失真预算](../theory/codebook-quantization-and-bit-budget.md)，这里解释有限候选表上的分配及其评价条件。

## 1. 固定变换codec，测每个token的失真曲线

冻结视觉塔输出 $N$个 $z_i\in\mathbb R^d$。用与评测不重合的32张训练图拟合均值 $\mu$和512维正交PCA基 $U$：

$$y_i=U(z_i-\mu),\qquad \widehat z_i=U^{\mathsf T}\widehat y_i+\mu.$$

固定基后，截断子空间的误差不随系数码率变化，可在同一codec的source-MSE分配比较中抵消；[截断PCA](../fundamentals/mathematics/invertible-transforms-and-kronecker-products.md)仍会丢信息，不能称可逆压缩。

token内部先按系数方差 $\sigma_j^2$、高分辨率代理
$D_j(b)\propto\sigma_j^22^{-2b}$作连续反向注水：
$b_j(r)=[\frac12\log_2(\sigma_j^2/\theta)]_+$，选 $\theta$令 $\sum_jb_j=r$。
再把 $2^{b_j}$取整为量化等级，在 $\pm4\sigma_j$范围量化。它是codec构造；与后面跨token的整数分配是不同步骤。（§2.1。）

对每个token实际测量17个重构点：

$$D_i(\ell)=\|z_i-\widehat z_i(64\ell)\|^2,\qquad \ell=0,\ldots,16.$$

零码率将系数置零，重建为 $\mu$；这个位置依然进入LLM。它是“丢弃信息”的编码端点，不是物理删行，也没有消除该位置的后续计算。

## 2. 动态规划精确解决有限网格的可加目标

给定可行整数预算 $B$（每单位64bit），求

$$\min_{\ell_i\in\{0,\ldots,16\}}\sum_iD_i(\ell_i),
\qquad\sum_i\ell_i=B.$$

令 $J_k(b)$为前 $k$个token恰花 $b$单位的最小失真，则

$$J_k(b)=\min_{0\le\ell\le\min(16,b)}
\{J_{k-1}(b-\ell)+D_k(\ell)\},\quad
J_0(0)=0,\ J_0(b>0)=+\infty.$$

回溯得到各token码率，时间复杂度 $O(NB\cdot16)$。精确性来自已测有限表、可加失真和单资源约束；不要求曲线凸，也不表示最终答案、KL或真实传输时延最优。论文只观察59.6%的边际增益差满足递减，因此跨token的简单凸注水不能替代此DP。（§2.2、式3–5。）

等预算“pruning”对照先把少数位置设满码率，其余设零；预算不整除16时还给一个位置余数。它是编码预算的端点分配，不是与真实删token、位置重编号或稀疏kernel的同成本比较。

## 3. 系数预算、物理字节与重复提问分别计量

表中256/384/512 bits/token只计系数预算，相对 $2560\times16$ bit的FP16特征称160×/约106×/80×。评测模型本身以float32执行；这些比例不是模型参数压缩、KV缩短或实测网络带宽收益。

rate-index还需保存。原文用 $\log_2(17)\approx4.09$ bit计信息量；独立固定宽字段至少5bit，平均接近信息量需要具体编码。PCA基/均值/方差、取整量化等级后的packing、headers与对齐也未给完整码流账，不能把名义系数率当实际文件大小。DP平均58ms只属于作者所测encoder侧求解，完整拟合、17点测量、打包/解码与LLM时延未计齐。

不读问题使同图的 $D_i$与分配不随提问改变。ChartQA的A问题生成attention排序、B问题测试对照说明了这种复用价值；它不保证任意未来问题的任务分数优于读问题的方法。若用每band输出KL给曲线加权，便已引入当前问题，需另评跨问题复用。（§2.3、§3.3。）

## 4. source-MSE、输出KL与任务答案不是同一目标

实验冻结Qwen3-VL-Instruct 4B/8B，在ChartQA与COCO captions每cell评288图，codec由各数据集训练图拟合。表1最强attention分配基线在256bit的任务准确率仍略高：4B为63.5对RDTok63.2，8B为70.8对70.5。因此精确source-MSE最优不能改写成所有任务准确率最佳。

论文另报告相同特征MSE、不同误差方向/集中形态的输出KL可差2.9–14倍；原目标只是可复用的源侧代理。KL、输出保留与任务准确率分别报告，不能据一个指标替代另一个。其KL完整token/归约及output-preservation判定未给齐，本页不补写协议。（§3.1–3.4、Table1/Figure4。）

## 来源身份

| 来源 | 固定版本 | 核读与使用范围 |
| --- | --- | --- |
| Query Independent Variable Rate Visual Token Coding（`https://arxiv.org/abs/2610.00204v1`） | `arXiv:2610.00204v1` | 完整正文/5页PDF、Figure1–4/Table1；两层分配、DP、系数预算与query-transfer回查。未核作者代码/物理码流、完整输出指标协议或模型/传输运行；不将source最优扩成任务或时延最优。 |
