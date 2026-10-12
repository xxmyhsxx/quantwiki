---
title: MLA：联合潜变量、位置分支与缓存表示
type: concept
tags:
  - attention
  - kv-cache
  - low-rank
aliases:
  - multi-head-latent-attention
  - MLA
sources:
  - raw/papers/2026-09-21/deepseek-v2/paper.pdf
  - raw/papers/2026-09-21/deepseek-v2/source.eprint
updated: 2026-10-07
---

# MLA：联合潜变量、位置分支与缓存表示

Multi-head Latent Attention（MLA）把各头的内容 Key 和 Value 联合表示为一个低维潜变量，再用独立分支承载位置。它减少需要跨解码轮次保存的元素，并让内容投影能够移到查询和输出一侧。本页依据 DeepSeek-V2 v5，解释这种结构怎样改变量化对象；它是训练得到的模型结构，不能直接当作任意已训练 MHA 的无损 PTQ 转换。

先了解 [标准 Q/K/V、GQA 与自回归缓存](transformer-autoregressive-inference.md)。MLA 保留因果注意力和逐层缓存复用，但缓存内容、头的表示与计算宽度发生变化。

## 1. 联合压缩的是各头的内容 K/V

采用论文的列向量约定：当前隐藏状态 $h_t\in\mathbb R^D$，query head 数为 $H$，每头内容维为 $d_h$，KV 潜变量维为 $d_c$，query 潜变量维为 $d_q$。与使用右乘矩阵的页面对照时，需要转置投影矩阵，见 [线性层与输入通道](../operators/linear-layer-input-channel.md)。

按 §2.1.2，式 (9)–(13) 的主干公式，

$$c_t=W^{DKV}h_t\in\mathbb R^{d_c},\qquad
k^C_{t,i}=W^{UK}_i c_t,\qquad v^C_{t,i}=W^{UV}_i c_t.$$

$W^{DKV}\in\mathbb R^{d_c\times D}$；$W^{UK}_i,W^{UV}_i\in\mathbb R^{d_h\times d_c}$ 是总上投影矩阵中第 $i$ 个头的块。**K 和 V 共用同一个 $c_t$**，并非各自保存一个 $d_c$ 维潜变量；不同头通过不同投影读取它。

query 也可经过低秩投影：

$$c^Q_t=W^{DQ}h_t\in\mathbb R^{d_q},\qquad
q^C_{t,i}=W^{UQ}_i c^Q_t\in\mathbb R^{d_h}.$$

历史 query 不被后续 token 直接查询，因此 query 压缩主要服务训练激活内存，不能再计为历史缓存压缩。Appendix B 的 V2-Lite 没有压缩 query，所以 query 压缩也不是所有 MLA 变体必须具有的特征。

上述公式遵循论文架构节的简写。§3.1.2 另说明实际模型在压缩潜变量后加入 RMSNorm，并在宽度瓶颈施加额外缩放。下面的吸收推导以**实际进入上投影的潜变量**为对象；这些归一化与缩放仍须保留在实际计算位置，不能把省略版公式当作完整实现，或把非线性 RMSNorm 吸收到一个固定矩阵中。

## 2. 为什么可以不展开全部历史内容 K/V

先忽略位置分支。以下是对原文矩阵吸收说明的等价展开，使用实数线性代数、固定权重和同一组注意力概率：

$$
(q^C_{t,i})^T k^C_{j,i}
=(q^C_{t,i})^T W^{UK}_i c_j
=\underbrace{\bigl((W^{UK}_i)^Tq^C_{t,i}\bigr)^T}_{\bar q_{t,i}^T}c_j.
$$

将查询变为 $\bar q_{t,i}\in\mathbb R^{d_c}$，就能直接与历史 $c_j$ 点积；$W^{UK}_i$ 还可以和 query 的上投影组合。计算 Value 加权和时，令 $a_{t,j,i}$ 为第 $i$ 头对位置 $j$ 的注意力概率，

$$z_{t,i}=\sum_{j\le t}a_{t,j,i}c_j,\qquad
o_{t,i}=W^{UV}_i z_{t,i}.$$

将输出矩阵按头分块为 $W^O_i\in\mathbb R^{D\times d_h}$，则

$$u_t=\sum_{i=1}^{H}W^O_iW^{UV}_i z_{t,i}.$$

这把 Value 上投影移到了历史加权求和之后，并允许与输出投影组合。**虽然缓存潜变量共享，$z_{t,i}$ 仍然逐头不同**，因为各头的注意力概率不同；不能先用一组公共概率求和，再复制给所有头。

这说明适当的解码路径可以直接读取潜变量，避免物化全部历史内容 K/V。它没有证明任意 prefill/decode 内核都采用同一计算方式或必然更快。矩阵结合次序在实数下等价，加入权重量化、中间舍入和不同累加顺序后通常不再逐位等价；要讨论量化，必须先说明使用哪些组合矩阵和中间对象。

## 3. RoPE 为什么需要分离

如果对内容 query 和 key 都施加 RoPE，记位置 $t$ 的旋转为 $R_t$，分数包含

$$ (q^C_{t,i})^T R_t^TR_jW^{UK}_i c_j.$$

$R_t^TR_j$ 依赖当前 query 和历史 key 的位置。一般不能把它与任意上投影交换，也不能将 $W^{UK}_i$ 一次性吸收到一个与历史位置无关的 query 矩阵中。这是 §2.1.3 针对该计算路径的障碍，不能扩大成“任何低秩表示都无法编码位置”。

DeepSeek-V2 将内容与位置拆开。位置分支宽度为 $d_R$，每头有自己的位置 query，但所有头共用一条位置 key：

$$q^R_{t,i}=R_tW^{QR}_i c^Q_t\in\mathbb R^{d_R},\qquad
k^R_t=R_tW^{KR}h_t\in\mathbb R^{d_R}.$$

query/key 分别拼接为 $[q^C_{t,i};q^R_{t,i}]$ 和 $[k^C_{j,i};k^R_j]$。结合上一节，基础注意力公式可以写为

$$\ell_{t,j,i}=
\frac{\bar q_{t,i}^Tc_j+(q^R_{t,i})^Tk^R_j}{\sqrt{d_h+d_R}},\qquad
a_{t,j,i}=\operatorname{softmax}_{j\le t}(\ell_{t,j,i}).$$

分母仍由原来的拼接 head 维度 $d_h+d_R$ 决定，不能因为点积在潜空间执行，就改成 $\sqrt{d_c+d_R}$。这是 §2.1.3，式 (14)–(19) 与 Appendix C，式 (37)–(47) 的基础结构；长上下文扩展的额外缩放需另按模型配置处理。

需要保存的是 **$c_j$ 和经过位置旋转的共享 $k^R_j$**。内容分支继续享有矩阵吸收；位置分支单独参加分数计算。因果 mask、位置规则和缓存前缀身份仍要一致，才能复用历史表示。

## 4. 缓存容量怎样计算

以下按每层、每 token 计元素，假设普通 K/V 每头等宽 $d_h$，不计精度、元数据与分配开销；对应论文表 1 的元素口径：

| 结构 | 历史保存对象 | 元素数 |
| --- | --- | --- |
| MHA | 每个 query head 对应的 K/V | $2Hd_h$ |
| GQA | $H_{kv}$ 组 K/V | $2H_{kv}d_h$ |
| MQA | 一组共享 K/V | $2d_h$ |
| MLA | 联合 KV 潜变量与共享位置 key | $d_c+d_R$ |

§3.1.2 的 V2 配置为 $D=5120,H=128,d_h=128,d_c=512,d_q=1536,d_R=64$。因此 MLA 每层、每 token 保存 576 个元素。注意 $Hd_h=16384$ 并不等于隐藏维 5120；不能套用某些稠密模型的 $D=Hd_h$ 配置关系。

若有 $N_l$ 层、$B$ 个等长请求、每个请求 $L$ 个有效 token，所有缓存元素均占 $b$ 字节，则逻辑容量是整理者推导的

$$M_{\rm MLA}=N_lBL(d_c+d_R)b.$$

若两类对象使用不同位宽 $p_c,p_R$，按理想位打包计，

$$M_{\rm MLA}=N_lBL\left(d_c\frac{p_c}{8}+d_R\frac{p_R}{8}\right)+M_{\rm metadata}.$$

实际分配还要处理尺度、对齐、页表、保护窗口、复制与临时张量，见 [缓存格式与生命周期](../../theory/kv-cache-quantization-objects-and-granularity.md#11-从位宽到可执行存储格式)。576 个元素本身既不是 576 字节，也不能直接解释作者相对另一个已部署模型报告的显存比例。

具体分布式框架还会按 token 分片和复制这份逻辑历史，预留页与临时张量需要另计；[vLLM 普通 MLA DCP](../../implementation/vllm-mla-context-parallel-execution.md#2-调度-block-与每-rank-的物理-page)给出每 rank、DCP 子组和 TP group 的条件化记账，不改变本节的联合缓存对象。

## 5. 潜变量量化怎样同时影响 K 和 V

下面是基于结构的整理者推导，不是原文提出的专用量化算法。固定当前 query、权重与可见位置，设缓存解码后的误差为

$$\hat c_j=c_j+e_j,\qquad \hat k^R_j=k^R_j+r_j.$$

则该头的 logit 扰动为

$$\delta\ell_{t,j,i}=
\frac{\bar q_{t,i}^Te_j+(q^R_{t,i})^Tr_j}{\sqrt{d_h+d_R}}.$$

同时，内容 Value 的误差为 $W^{UV}_ie_j$。所以一次潜变量误差既改变匹配分数，又改变读取内容，并通过不同上投影影响多个头。若暂时固定注意力概率，内容误差传到输出为

$$\delta u_t=\sum_iW^O_iW^{UV}_i\sum_{j\le t}a_{t,j,i}e_j.$$

实际概率也会受 logit 扰动影响，需结合 [Attention 的 softmax 与联合误差传播](../../theory/attention-quantization-error-propagation.md)，不能把最后这个条件公式当作完整输出误差。

这解释了为什么 [普通 K/V 的粒度选择](../../theory/kv-cache-quantization-objects-and-granularity.md)不能原样搬到 MLA：潜变量不是单独的 K 或 V，latent 通道也不等同于普通 head 通道。缓存量化需要明确潜变量、位置 key 的表示与位宽，以及读取端采用潜空间计算还是展开计算；本页不据此指定最优粒度或默认精度。

## 6. 作者依据与本页范围

MLA 的核心主张是以少量历史表示支持多头注意力。Appendix D.1、表 8 对 MHA/GQA/MQA 的比较使用约 7B 稠密模型、1.33T 训练 token，并调整层数以对齐参数量。Appendix D.2、表 9 的 MLA/MHA 比较使用约 16B MoE、1.33T token，以及约 250B MoE、420B token 两个规模；各对模型除 attention 外采用相同架构设置，实际总参数与激活参数仍有差别。作者报告 MLA 在多数所列任务上更好、缓存元素更少；小模型 C-Eval 的 MLA 为 50.9、MHA 为 51.6，因此不应写成所有任务都更好或无条件优于 MHA。

§3.2.3 的 5.76 倍是**已部署 V2 相对 DeepSeek 67B 的最大生成吞吐**：单节点 8 张 H800，V2 参数转为 FP8、缓存平均 6 bit，长度分布来自实际部署的基线服务，V2 可容纳更大 batch。这个结果体现结构、量化、系统优化与容量共同作用，不能归因于 MLA 单一因素，也不能解释为相同 batch 下的延迟倍数。该节未给出足以重建完整 MLA 缓存量化格式与内核的细节。

本页核读 §2.1、§3.1.2/3.2.3、Appendix B/C/D 的相关内容，按 PDF 页图核对关键公式与表号；没有逐项审查全部评测附表或运行模型。固定源码中的投影吸收、联合缓存、展开/潜空间读取和精度转换，已在[MLA 缓存与 Attention 执行](../../implementation/mla-cache-and-attention-execution.md)单独展开；它解释框架如何承接本页结构，不将具体后端路径当作所有 MLA 的规定。特殊混合精度表示与候选选择接续到[vLLM packed 缓存与 DSA](../../implementation/vllm-mla-packed-cache-and-dsa-execution.md)，其格式和 indexer 属于后续固定实现，不从本页 V2 模型报告推导。报告的另一项核心结构 [DeepSeekMoE 与专家路由](mixture-of-experts-and-routing.md)改变 FFN 计算和参数访问，与 MLA 对缓存表示的改变共同组成其效率设计。

## 来源身份

- DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model（`https://arxiv.org/abs/2405.04434v5`），`arXiv:2405.04434v5`，2024-06-19 修订；采用已归档的同版本 PDF 与 TeX 源包，正文定位见各节。矩阵吸收的展开、容量记账和缓存量化误差是本页整理者推导。
