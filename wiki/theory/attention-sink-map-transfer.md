---
title: Attention sink 选头图谱：量化迁移、集合边界与校准
type: concept
tags:
  - attention
  - calibration
  - reliability
sources:
  - raw/papers/2026-09-23/bos-sink-topology/paper.pdf
updated: 2026-09-29
---

# Attention sink 选头图谱：量化迁移、集合边界与校准

本页依据 BOS-Sink Topology v1，研究在浮点模型上测得的“哪些头关注首 token”能否直接迁移到 NF4 权重量化模型。核心区别是：全部头的排序高度相关，不保证 top-k 成员、层内排序或目标域图谱保持不变。它是校准有效性的诊断研究，没有证明图谱变化一定损害任务质量，也没有提出新的缓存淘汰算法。

## 1. “BOS”是位置定义，不是统一的特殊 token

对长度 $T>1$ 的输入 $x$，层 $l$、Query head $h$ 的分数为

$$s_{lh}(x)=\frac1{T-1}\sum_{t=1}^{T-1}\alpha_{lh}(t\to0),\qquad
\bar s_{lh}=\frac1{|D|}\sum_{x\in D}s_{lh}(x).$$

$\alpha$ 是 softmax 后的注意力概率。排除位置 0 的自注意，避免因果掩码下只有一个可见位置的平凡值 1。Llama-3.2 的位置 0 是专用 BOS，Qwen2.5 默认不插 BOS，位置 0 可能是正文或模板控制 token。因此本页的 sink 是首位置注意力集中，不能直接解释为同一语义对象。

GQA 的共享 K/V 不合并 Query head 的注意力分布。统计对象为 $N=LH_q$ 个“层—Query head”槽位，取 $k=\lceil0.1N\rceil$；两个 Qwen 模型都是 $N=336,k=34$，Llama-3.2-1B 为 $N=512,k=52$。（原文 §III-A–E。）

这里的注意力分数不是输出贡献：输出还取决于 Value、输出投影、残差和后续网络。首 token 高权重也不意味着它对所有任务都应占用高精度，相关压缩对象见 [KV cache 粒度](kv-cache-quantization-objects-and-granularity.md)。

## 2. 单输入稳定与校准图谱迁移是不同统计量

Sample-STC 先在每个输入上计算 bf16/NF4 的 top-k 集合、Spearman 排序相关和层内变化，再跨输入平均。Cal-STC 则先求 $\bar s$，再计算一次集合与排序。TopK 是非线性操作，二者不能交换；平均的个体稳定性不是离线图谱的稳定性。

等大小集合 $A,B$ 的 Jaccard 为

$$J=\frac{|A\cap B|}{|A\cup B|},\qquad
\text{成员保留率}=\frac{|A\cap B|}{k}=\frac{2J}{1+J}.$$

因此 $J=0.619$ 对应约 76.5% 的成员保留，不能把 61.9% 当作成员保留率。这个换算适用于每对等大小集合；对 Sample-STC 的均值作同一非线性换算，一般不等于逐样本保留率的均值。

随机选两组 $k$ 个头时，$E|A\cap B|=k^2/N$，期望交集与期望并集之比为 $k/(2N-k)$。原文明确用它作约 0.053 的 null 参考；它不是 $E[J]$ 的精确等式，后者需对交集的超几何分布取期望。不能将“比随机好很多”进一步读为选头策略已经满足任务要求。

## 3. 层内相对质量与绝对 sink mass

每层将头分数归一化为 $p_{lh}=s_{lh}/\sum_hs_{lh}$，再比较两精度的分布。原文 LayerShift 使用自然对数下 Jensen–Shannon **距离**：

$$d_{JS}(p,q)=\sqrt{\tfrac12\operatorname{KL}(p\|m)+\tfrac12\operatorname{KL}(q\|m)},\qquad m=(p+q)/2.$$

其范围为 $[0,\sqrt{\ln2}]$。这是 [RAMP](../methods/ramp.md) 所讨论 JSD 的平方根，不能把数值与以 2 为底的散度直接比较。Cal-LS 在校准均值形成的层分布上计算，再对层平均；Sample-LS 则逐输入计算。

归一化也丢失信息：一层所有头的 sink 分数同时缩小一半，$p_l$ 不变，LayerShift 为零。因此它衡量头间相对分配，不衡量该层 sink 总量的保持。层内 Spearman 又只看次序，微小近并列变化可能让它大幅下降，而 JS 距离仍小。三者应分别读取，不能找一个指标替代全部诊断。

原文表 IV 中 Qwen2.5-0.5B 的第 22 层 JS 距离 0.276、层内相关 0.033，而全模型 Cal-Rank 仍为 0.980。另一方面 Qwen2.5-1.5B 的第 26 层 LayerShift 0.014 低于模型均值，层内相关却只有 0.587。全局高相关可主要来自不同层之间的大尺度差别。

## 4. 选头边界为什么比全局排序脆弱

若所有头分数扰动绝对值不超过 $\epsilon$，浮点第 $k$ 和 $k+1$ 名间隔大于 $2\epsilon$，即可保证 top-k 集合不变；这是 [排序稳定性](quantization-ranking-stability.md) 的同一确定性论证。高 Spearman 本身不提供这个边界余量。

原文在 500 个 C4 输入、名义 4,096 token 下报告：

| 模型 | Cal-Rank | Cal-Set Jaccard | Cal-LS |
| --- | --- | --- | --- |
| Qwen2.5-0.5B | 0.980 | 0.619 | 0.045 |
| Qwen2.5-1.5B | 0.990 | 0.789 | 0.029 |
| Llama-3.2-1B | 0.983 | 0.793 | 0.019 |

Qwen 尾层变化大是观测标记，尚不是机制解释。层权重动态范围与 LayerShift 的相关性没有支持所提解释；三模型也不足以分离规模、模型族和 GQA 比例的作用。高漂移层与导致全局 top-k 换人的层不必相同。

论文以 $1/4096$ 对比头分数 0.65–0.80，得到数千倍集中度。**整理者口径修正：**对每个 Query 在其所有可见位置上均匀注意，首 token 的因果基线应为

$$\frac1{T-1}\sum_{t=1}^{T-1}\frac1{t+1}=\frac{H_T-1}{T-1},$$

其中 $H_T$ 是调和数。$1/T$ 只对应最后位置的均匀分布，不能直接充当跨全部 Query 平均的基线。高集中现象仍然成立，但不沿用该倍数作为严格匹配的对照。

## 5. 精度迁移与域迁移不能相减成因果份额

LongBench multifieldqa_en 的 75 个样本提供三种 Jaccard：A 为目标域 bf16 对目标域 NF4；B 为 C4 bf16 对目标域 NF4；C 为 C4 bf16 对目标域 bf16。$A-B$ 改参考域、$C-B$ 改目标精度；Jaccard 非加性，这些是描述性对照，不是可相加的独立损害分解。

Qwen2.5-1.5B 的 A/B/C 为 0.889/0.360/0.360。B=C 仅表示两个交集大小相同，不代表目标域两精度选了完全一样的头。Llama 的 A/B 为 0.733/0.793，跨域参考反而更接近 NF4 目标，故“域漂移总更严重”也不成立。（表 V。）

重校准试验从同一 500 条 C4 池抽取 $n\in\{8,16,32,64,128,500\}$，与完整池 NF4 图谱比较；用两个不交叠 250 样本图谱的重合度作为经验平台值，以其 90% 为门槛。两个 Qwen 在最小测试值 8 就达标，Llama 首次达标为 32，但其 $n=8$ 结果离门槛小于抽样标准差，不是精确四倍样本需求。

小样本和参考共享数据池，每个规模只有五次抽样，这不是独立目标域泛化保证。只更新预选高漂移层时，两 Qwen 最高分别为 0.619、0.838，仍未达到全图标准；部分层分数变化没有触及全局集合边界。实际选头迁移需要检查整个使用集合，而不是仅盯显著异常层。

## 6. 测量与部署证据的边界

作者用 RTX 3060 Ti、batch 1、teacher forcing、`use_cache=False`、eager attention；NF4 无 double quant，计算 FP16，参考 BF16。输入、checkpoint 和 attention 后端配对固定，但**同时改变了权重编码与计算 dtype**，不能把全部差异归因于纯权重舍入。

hook 只累计首列并丢弃完整注意力输出，减少跨层保留；eager 计算本身并未因此变为线性内存。128-token 测试与直接返回 attention 的偏差小于 $10^{-6}$，只核对所测 hook 统计。2K/4K 条件先用 Qwen tokenizer 拼窗再解码，各模型重新分词；512 条件则截断原文档且允许短样本。上下文长度对照含采样方式变化，不是单变量因果试验。

Jetson Orin NX 8GB、Transformers 4.57.6 的 16 样本工作量：Qwen0.5B 为 2.9 s，Llama1B 为 5.7 s；Qwen1.5B 的 6.8 s 只有耗时意义，因为 NF4+eager 返回 NaN，未恢复有效图谱。`max_memory_allocated` 是 GPU 分配统计，在统一内存设备上也不是整机峰值。没有实际下游 sink 策略、任务质量或 KV 淘汰收益验证。

## 来源身份

- [Global Ranks Survive, Selected Heads Shift: BOS-Sink Topology under 4-bit Weight-Only Quantization](https://arxiv.org/abs/2609.23585v1)，arXiv:2609.23585v1；正文六页完整研读，指标回查 PDF 第 2 页；本地无作者代码、未复现实验。成员保留换算、归一化反例和因果均匀基线为整理者展开。
