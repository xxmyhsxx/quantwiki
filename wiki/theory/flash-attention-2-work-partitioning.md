---
title: FlashAttention-2：累计状态、序列并行与warp分工
type: method
tags:
  - attention
  - kernels
  - performance
  - gpu
sources:
  - raw/repositories/2026-09-21/flash-attention
  - raw/papers/2026-09-21/flashattention-2
  - raw/papers/2026-09-21/flashattention
updated: 2026-10-13
---

# FlashAttention-2：累计状态、序列并行与warp分工

FlashAttention-2继续精确分块attention与反向概率重计算，重点改变每步归一化工作和GPU分工：片上保留未归一化输出，沿query块增加并行，再将warp分工由分K/V改为分Q，减少输出合并通信。它没有将稠密算术量降成线性，收益取决于工作块数量、片上资源、数值与硬件路径。

本页依据 FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning，arXiv:2307.08691v1，固定14页PDF与同版TeX，核读正文§1–5及算法1–2、图2–7、表1。v1有若干公式与形状不一致，下文以明确的整理者推导解释正确关系，保留原件定位；尚未用作者代码或GPU运行验证这些差异。基础的精确softmax合并、IO模型、mask/dropout与反向梯度见[FlashAttention原论文](flash-attention-tiling-and-recomputation.md)，本页只补FA2的变化。

## 1. 非矩阵乘工作也会限制吞吐

FA1已避免完整score/概率矩阵写回HBM，但在线softmax仍要指数、最大值/分母归约、重缩放和归一化。FA2 §3.1以A100的理论峰值为动机：FP16/BF16矩阵乘312 TFLOPs/s，非矩阵乘FP32为19.5 TFLOPs/s，峰值比16。它说明计数少的标量工作仍可能占用较多时间，不等于每条指数、除法或归约指令都固定慢16倍；不同执行单元、依赖与带宽要分别分析。

### 循环中保留累计，最后归一化一次

对一个query块，保留逐行$m,\ell$和未归一化累计$A$。新key块score为$S_j$，整理者从定义
$A=\sum_t e^{s_t-m}v_t$推导：

$$m'=\max(m,\operatorname{rowmax}(S_j)),\qquad
\alpha=e^{m-m'},\qquad E_j=e^{S_j-m'},$$
$$\ell'=\alpha\ell+\operatorname{rowsum}(E_j),\qquad
A'=\operatorname{diag}(\alpha)A+E_jV_j.$$

循环结束才做$O=\operatorname{diag}(\ell)^{-1}A$，并保存自然对数
$L=m+\log\ell$。初值$m=-\infty,\ell=0,A=0$，首个有效块建立状态；空tile与全行无有效key仍须专门处理。这里$E_j$未归一化，不能在每块除自己的分母后再直接相加。

[FA1页](flash-attention-tiling-and-recomputation.md)的$A$只是对保存归一化$O$的等价教学表达；FA2把未归一化累计作为片上循环状态，免去每块对完整向量的归一化。反向由最终全行$L$恢复$P_{ij}=e^{S^{\rm masked}_{ij}-L_i}$，不必另保存$m,\ell$两个向量。（§3.1，算法1–2。）

### v1原件的重缩放式不能直接照抄

PDF p6算法1第10行、同页两块推导和同版TeX，将旧累计前的$\operatorname{diag}(e^{m_{\rm old}-m_{\rm new}})$写成取逆。按累计定义，当最大值增大时，旧项必须缩小，正确关系是乘$\alpha$，不能乘$1/\alpha$。PDF p5的“un-scaled”式也把旧分母因子写成取逆并漏掉最大值重缩放；p4背景的输出/在线更新有归一化因子不一致。本页使用上述推导，不把原伪码改写成已核作者实现。

**教学反例。** 两个key的score为$0,\log3$、value为$1,0$。第一个块$m=0,\ell=A=1$，第二块使$\alpha=1/3$，故$A'=1/3,\ell'=4/3$，输出$1/4$。若按取逆旧缩放，$A'=3$，输出$9/4$，与整行softmax不符。这是代数核对，不证明发布代码也执行了该错误式。

## 2. 前向沿query块并行，反向保留共享归约

FA1论文算法以K/V块为外层，要对每个K/V块扫描Q和输出。FA2前向交换循环：query块为外层，K/V块为内层；一个thread block拥有一组输出行，将$m,\ell,A$保留片上，所有key块结束后写一次$O,L$。不同query块的输出行不重叠，前向不需要合并这些块的输出。

并行维由batch×head扩为batch×head×query块。若$B=1,H=4$，只有4个按头工作块；再分为32个query块就有128个逻辑工作块，可能更好利用多个SM。它增加可调度工作，但真实驻留与速度仍受每块寄存器、共享内存、启动和尾部浪费限制，不由这个乘积直接保证。（§3.2，图2。）

论文明确将交换循环和沿序列并行的先行建议与实现归给Phil Tillet的Triton版本，不能把所有调整都写成FA2独创。FA2没有给新的参数化IO定理；FA1针对K/V外层的流量证明也不能直接冒充FA2代码的逐点性能证明。

反向在单个attention head内按K/V列块并行：各工作块独占对应$dK,dV$，但不同列块都对同一$dQ_i$贡献
$dS_{ij}K_j$。§3.2明确使用atomic add汇总$dQ$，所以反向不是所有输出都独立，原子加法和求和顺序也涉及数值/确定性。算法2先在HBM预计算
$D_i=\operatorname{rowsum}(dO_i\odot O_i)$，再恢复局部概率并累加梯度；它与FA1伪码内层重复算$D$的组织不同。完整梯度与dropout同mask要求复用[反向重计算](flash-attention-tiling-and-recomputation.md)。

## 3. warp分Q减少哪一种通信

图3的FA1前向方案将K/V分给不同warp，各warp产生同一组输出行的部分贡献，需要shared-memory写回、同步和合并。论文称这为“split-K”，本节指它在attention块中的K/V分工，不将其直接等同于所有GEMM的归约维拆分。

FA2让不同warp拥有不同Q行，共享可访问的K/V。每个warp完成自己输出行的$QK^{\mathsf T}$和$PV$，避免这类跨warp的输出部分和归约。它减少相关shared-memory读写，不能称整个kernel没有共享内存、同步或warp内通信。反向的数据依赖更复杂，原文也保留同步。（§3.3，图3。）

原文通常用4或8 warps，块大小从$\{64,128\}\times\{64,128\}$按head维度和设备手调。更大块可减少部分加载/循环，也会增加寄存器与共享内存状态；spill或容量超限会降低速度甚至无法运行。资源与自动调优的共同机制见[Kernel配置选择](../implementation/kernel-configuration-and-autotuning.md)，这里不是所有workload的最优配置。

## 4. 因果块、共享KV与原件形状边界

对等长、同位置域的因果attention，key位置$c$只在$c\le r$时对query位置$r$可见。对一个tile：

- $\min c>\max r$：整个上三角块不可见，可跳过。
- $\max c\le\min r$：整个下三角块可见，无需逐元素因果mask。
- 其余跨对角块：按元素屏蔽$c>r$；“每行块只mask一个块”只适用于相应方形对齐几何。

PDF p6 causal第2点把无需mask的行/列不等式方向写反，与第1点跳过上三角冲突；上述条件由$c\le r$推导。非等长Q/K、位置偏移、分页与滑窗需要实际接口定义，不能套用这个简单边界。

MQA/GQA使多个query heads使用同一KV head。§3.1说明可通过head索引避免实际复制K/V，反向则把共享KV的$dK,dV$跨query heads相加。已有[模型形状与共享](../fundamentals/model/transformer-autoregressive-inference.md)承接基本映射；head整除/排列、dtype、head dim和现代库支持仍需固定源码核验。它减少逻辑KV头数，不等于移除了每个query head的归一化或全部读取。

其余v1差异也需保留：

| 原件位置 | 原写法与问题 | 本页采用的关系 |
| --- | --- | --- |
| p3背景反向 | $dK=QdS^{\mathsf T}$，一般形状不合 | $dK=dS^{\mathsf T}Q$，与算法2第16行一致 |
| p7算法2第4行 | 行归约$D$标为$\mathbb R^d$ | 每query行一个值，$D\in\mathbb R^N$ |
| p3缩放脚注 | 将常见缩放写为$1/d$，基本伪码省略缩放 | 常见模型取$1/\sqrt d$；若加入固定$\tau$，score及$dQ,dK$必须一致乘它 |

这些是对固定v1表达的核对。它不授权替换归档版本，也不证明作者代码、后来版本或全部dtype的前向/反向已验证。

## 5. 性能数字要与计数口径一起读

§4.1的attention基准使用单A100-80GB-SXM4，$N=512$至16K、$BN=16K$，hidden2048，$d=64,H=32$或$d=128,H=16$，分别测有/无causal。作者报告相对当时FA1为1.7–3.0倍、Triton FA1为1.3–2.5倍、标准实现为3–10倍；这些不是当前软件排行榜，正文未充分披露该组dtype、dropout和代码commit，不能补套FA1论文的配置。

原文前向matmul计数式$4N^2dH$是单序列两次矩阵乘，batch总量还要乘$B$；causal近似除2，反向乘2.5对应五次matmul，包含重计算。它忽略非matmul、尾部/无效tile和实际指令差异，不能直接作为全部设备运算。固定$BN$时改变$N$还改变batch，不能将图中吞吐变化归于序列长度一个因素。（图4–6。）

表1是8×A100-80GB-SXM的GPT式模型训练，报告TFLOPs/s/GPU：

| 模型/长度 | 无FA | FA1 | FA2 |
| --- | ---: | ---: | ---: |
| 1.3B，2K | 142 | 189 | 196 |
| 1.3B，8K | 72 | 170 | 220 |
| 2.7B，2K | 149 | 189 | 205 |
| 2.7B，8K | 80 | 175 | 225 |

2.7B/8K相对无FA约2.81倍、相对FA1约1.29倍；p11正文误写“compared to FA2”自己，比较对象以表的FA1列区分。完整batch、dtype和优化配置未充分披露，本轮也没有重跑训练。

模型FLOPs计数为$6NP_{\rm model}+12L_{\rm layers}hN^2$，其中$P_{\rm model}$为参数数，$h$为hidden维度；作者为与既有文献一致，没有将causal attention项除2。因此225/312约72%的model FLOPs utilization，与attention微基准按causal除2的利用率不是同一口径。

H100-80GB-SXM5结果使用未适配TMA等新特性的同一实现；正文报最高335 TFLOPs/s，图7(b)在无causal、$d=128,N=16K$的前向＋反向点标338，保留此差异。未来1.5–2倍及TMA/FP8/其他设备适配属于预期。§5由约2倍算子收益推及“16K与8K同成本”也不是一般结论：固定batch/head时dense attention算术随$N^2$增长，整模型成本还包含投影、MLP、调度和其他状态。

所选后续固定库的[普通CUDA前后向](../implementation/flash-attention-2-cuda-execution.md)已按独立commit核到分派、舍入、LSE及梯度归约；它不替代论文时期代码或当前框架外部pin的验证。

## 来源身份

- Dao-AILab/flash-attention（`https://github.com/Dao-AILab/flash-attention/tree/73c992c8ca746935548df620ef3c1b6238fe6e68`），固定commit `73c992c8ca746935548df620ef3c1b6238fe6e68`；后续普通CUDA实现连接，实际selected配置与源码定位见实现页，保留版本差异。
- FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning（`https://arxiv.org/abs/2307.08691v1`），arXiv:2307.08691v1；正文§1–5、算法1–2、图2–7和表1，PDF/同版TeX核对；重缩放/因果条件校正、形状与计数展开、教学反例为整理者分析。未逐点重建曲线，未核作者代码、全部接口或GPU/模型运行。
- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（`https://arxiv.org/abs/2205.14135v2`），arXiv:2205.14135v2；复用§3.1–3.2与附录B/C的归一化定义、反向数学和IO范围；不把原循环流量证明直接归给FA2实现。
