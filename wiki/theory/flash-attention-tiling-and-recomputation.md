---
title: FlashAttention：精确分块、IO复杂度与反向重计算
type: method
tags:
  - attention
  - memory
  - kernels
  - performance
sources:
  - raw/papers/2026-09-21/flashattention
updated: 2026-10-13
---

# FlashAttention：精确分块、IO复杂度与反向重计算

FlashAttention 将两次矩阵乘和行softmax一起分块，在片上恢复正确的全行归一化；训练反向再由保存的输出、逐行统计和随机状态重算局部概率。它避免将完整的score和概率矩阵写到HBM，减少峰值工作内存，并在片上容量相对head维度足够大时减少访存；稠密attention的算术量仍随序列长度二次增长。

本页依据 FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness，arXiv:2205.14135v2：正文§1–5、附录B/C及D/E的必要实现与实验条件；固定PDF共34文件页，同版TeX可读。原文算法、定理和所引表格编号均回查PDF。作者CUDA代码、当前库支持、设备执行与性能未在本轮验证。

## 1. 精确attention要保留什么

对单头、等长输入$Q,K,V\in\mathbb R^{N\times d}$，输出$O\in\mathbb R^{N\times d}$：

$$S=\tau QK^{\mathsf T},\qquad
P=\operatorname{softmax}(\operatorname{mask}(S)),\qquad O=PV.$$

$\tau$是固定缩放，通常为$1/\sqrt d$；mask在softmax前把不可见位置置$-\infty$。多头、batch另按各自实例处理。基本形状与因果语义见[Transformer](../fundamentals/model/transformer-autoregressive-inference.md)。

原文算法0将$S,P\in\mathbb R^{N\times N}$写回HBM，再读出做softmax和$PV$。仅融合mask/softmax仍可能保留二次规模的概率矩阵，训练还要保存或重算它。FlashAttention改变中间值的存放与执行顺序，计算同一个稠密或指定mask的attention；“精确”指实数域的数学等价，不保证浮点逐位相同，也没有把指数计算和低精度舍入误差消掉。（§2–3.1，算法0–1。）

它处理的是一次attention调用的临时矩阵。自回归推理跨轮保留的KV cache仍需要存储；避免保存$P$不能推出无需历史K/V或自动完成低比特KV部署。

## 2. 分块怎样保持全行softmax

softmax分母跨越整行key。若每个key块单独softmax再相加或平均，得到的归一化域就变了。对已处理的一组key，本页用每查询行的最大值$m$、指数和$\ell$、未归一化向量累计$A$表示状态：

$$\ell=\sum_j e^{s_j-m},\qquad A=\sum_j e^{s_j-m}v_j,\qquad O=A/\ell.$$

初值为$m=-\infty,\ell=0,A=0$；首个有效块建立统计与累计，只在$\ell>0$时解释归一化输出。尚未遇到有效key的空状态需要相应分支。

新块的局部最大值、分母和累计为$\widetilde m,\widetilde\ell,\widetilde A$，令

$$m'=\max(m,\widetilde m),\qquad
\alpha=e^{m-m'},\qquad\gamma=e^{\widetilde m-m'},$$
$$\ell'=\alpha\ell+\gamma\widetilde\ell,\qquad
A'=\alpha A+\gamma\widetilde A,\qquad O'=A'/\ell'.$$

新旧指数都转到$m'$这个基准，所以同时重缩放分母和向量累计。原文算法1保存归一化的$O$，等价更新为
$O'=(\alpha\ell O+\gamma\widetilde A)/\ell'$；上述$A$表示是整理者的等价改写，不是作者另存一份全局累计的实现声明。

**教学反例。** 两个块各有一个key，score为$0,\log3$，value为$0,10$，整行输出是7.5。两个块各自归一化后再平均会得5；保留局部最大值、分母和累计才能恢复各块的全局权重。

### 原论文的循环与片上资源

原文算法1以K/V块为外层、Q块为内层：先载入$K_j,V_j$；依次读取$Q_i,O_i,m_i,\ell_i$，在片上计算$B_r\times B_c$的score/概率块和乘积，更新后将输出与统计写回HBM。每个K/V元素读取一次，Q和输出随K/V块多次扫描；“一个融合kernel”不表示所有输入都只读一次。

设$M$为模型中片上可容纳的标量数，必要渐近约束为

$$B_cd=O(M),\qquad B_rd=O(M),\qquad B_rB_c=O(M).$$

附录C据此取$B_c=\Theta(M/d)$、$B_r=\Theta(\min(M/d,d))$。算法1具体写$B_c=\lceil M/(4d)\rceil$、$B_r=\min(\lceil M/(4d)\rceil,d)$；这是论文的算法容量表达，不能代替真实kernel的资源配置。实际预算需按dtype将字节换成元素，合计Q/K/V、score/概率、累加器和统计的同时存活状态，并核寄存器、共享内存、spill和驻留数量。相关基础见[GPU分块与融合](../fundamentals/operators/gpu-kernel-computation-patterns.md)。

有限有效score且每个完整查询行至少有一个有效key时，分母为正。某个中间tile可以全被mask，但直接代入$-\infty-(-\infty)$会产生未定义算术；原伪码未完整披露分支，实际实现须处理空块和尾部。全行无有效key的输出规则也需接口规定，不能从有限输入推导中补猜。

## 3. IO复杂度、内存和算术量分别计算

原文定理2的模型是输入/输出在HBM、片上容量满足$d\le M\le Nd$，计算HBM与片上之间的元素访问量。令$T_c=\lceil N/B_c\rceil=\Theta(Nd/M)$：

$$\operatorname{IO}_{FA}
=\Theta(Nd+NdT_c)
=\Theta(N^2d^2/M).$$

第一项包括K/V和基本输入输出；第二项来自每个K/V块对Q、O及统计的扫描。区间内$N^2d^2/M\ge Nd$，所以可吸收基项；把最后一式外推到任意大$M$、得到少于输入输出的流量是不成立的。（§3.2定理2、附录C证明。）

| 数量 | 物化中间矩阵的算法0 | FlashAttention稠密算法 |
| --- | --- | --- |
| HBM访问量，论文模型 | $\Theta(Nd+N^2)$ | $\Theta(N^2d^2/M)$，在规定$M$区间 |
| 算术量 | $O(N^2d)$ | 前向仍$O(N^2d)$，反向因重算增加常数工作 |
| attention中间状态 | 可能保存$N^2$的score/概率 | 每行$m,\ell$共$O(N)$；输入输出本身仍$O(Nd)$ |

当$N\gg d$时，与物化基线的主项之比按$d^2/M$缩放；原文在$d^2\ll M$的典型区间解释访存优势，$d\le M\le Nd$本身不保证访问更少，$M$接近$d$时模型主项还可能更高。这个量级关系不是实测加速比。

固定$d,M$时，IO主项对$N$仍为二次。原文“sub-quadratic HBM accesses”的概括不能解读成固定硬件下的次二次长序列算法。减少峰值内存、减少累计访存、减少FLOPs与更短延迟是不同结论；与实测DRAM/L2和启动开销的关系见[roofline分析](../fundamentals/hardware/arithmetic-intensity-and-roofline.md)。

### 下界没有证明每个M都最优

命题3排除的是在整个$M\in[d,Nd]$区间都达到$o(N^2d^2/M)$的算法。证明只在$M=\Theta(Nd)$时使用读取输入/写出输出的$\Omega(Nd)$下界。它没有给每个固定$M$的匹配下界，原文也把依赖$M$的参数化下界留为未来工作；不能将§5的“optimal within constants”概括扩大成逐点最优证明。

## 4. 反向为什么只重算局部概率

前向保存$Q,K,V$所需输入、输出$O$、最终逐行$m,\ell$；反向读取$dO$。对当前tile，用**整行最终统计**恢复

$$P_{ij}=\frac{\exp(S^{\rm masked}_{ij}-m_i)}{\ell_i}.$$

这里不是当前tile自己的分母。附录B.2式3–6给出无dropout的反向链：

$$dV=P^{\mathsf T}dO,\qquad dP=dOV^{\mathsf T},$$
$$D_i=\sum_jP_{ij}dP_{ij}=dO_i^{\mathsf T}\sum_jP_{ij}V_j=dO_i^{\mathsf T}O_i,$$
$$dS_{ij}=P_{ij}(dP_{ij}-D_i),\qquad dQ=\tau dSK,\qquad dK=\tau dS^{\mathsf T}Q.$$

$D_i$将长度$N$的概率/梯度归约换成长度$d$的输出点积。因此当前概率块一旦恢复，就能算$dV$、$dP$、$dS$并累加输入梯度，不必把完整$P,dP,dS$写回HBM。$\tau$为固定常数时$dQ,dK$仍要乘它；这里没有可学习$\tau$的梯度定义。

完整反向是**算法4**，完整含mask/dropout前向是算法2，不能混淆编号。算法4以K/V块为外层：片上累加该块$dK,dV$，扫过Q/O/dO和统计，同时读写$dQ$；外层块结束再写出$dK,dV$。伪码在内层重复计算$D_i$，本页没有把整理者可预计算的优化归为作者已实现。

### dropout必须重现同一次mask

令$Z$是inverted-dropout乘子，取0或$1/(1-p_{\rm drop})$，前向为$O=(P\odot Z)V$。反向相应改成

$$dV=(P\odot Z)^{\mathsf T}dO,\qquad dP=(dOV^{\mathsf T})\odot Z.$$

仍有$D_i=dO_i^{\mathsf T}O_i$，但$O$必须来自同一次带dropout的前向。分母统计仍由dropout之前的softmax计算；不能以保留下来的概率重新归一化。算法2保存RNG状态，算法4恢复并按对应tile顺序重现$Z$，避免保存$N^2$个随机mask；另抽同分布随机数会改变梯度。并行RNG的计数器、元素映射和具体状态结构需代码核查。（附录B.3–B.4。）

这是一种选择性重计算：重算score与概率块，不重跑全部模型或每块临时输出。定理5及附录C给反向同样的$\Theta(Nd+NdT_c)=\Theta(N^2d^2/M)$访存模型；$O(N)$辅助状态排除输入、输出、输出梯度和输入梯度，不是整个训练总内存。附录B反向说明中的$O(N^2)$ FLOPs按固定$d$理解；把$d$显式列为变量时仍是$O(N^2d)$。

## 5. 实验支持什么，不能推出什么

以下均为作者报告，未在本轮复现。

| 证据与条件 | 原文结果 | 可支持的判断 |
| --- | --- | --- |
| 图2，A100，batch64、16heads、$N=1024,d=64$，前向＋反向；caption未明确dtype/mask/dropout | 标准/FA：66.6/75.2 GFLOPs，HBM R/W 40.3/4.4 GB，41.7/7.3 ms | 此配置增加算术仍因减少访存而更快；不补套另一基准的设置 |
| 表11，单A100 40GB、batch16、8heads、$d=64$、FP16、dropout0.1、随机padding mask；随机QKV，不含投影，100次均值 | $N=1024$：PyTorch/FA 8.29/2.55 ms；$N=4096$：124.19/37.49 ms，均前向＋反向 | 对所比物化实现有明显收益；padding长度从$[N-20,N]$抽取 |
| 表21，单A100 40GB、batch16、8heads、$d=64$、FP16，前向＋反向内存，**无dropout/mask** | $N=4096$：PyTorch/FA 17024/836 MB | attention调用内存下降；与表11不是同配置、也不是全模型内存 |
| 表7，A100-SXM4-40GB、batch64、16heads、$d=64$、mask＋dropout，$N=128$ | Apex FMHA/FA：前向0.10/0.08、反向0.17/0.20、合计0.27/0.28 ms | 短序列下FA合计略慢，融合与重计算不保证对所有基线更快 |

端到端证据可看表1：同MLPerf初始化、BERT-large达到MLM准确率72%，8×A100-80GB、FP16 Apex O2、10次均值，20.0±1.5变为17.4±1.4分钟。按数值耗时减少13%，速度比约1.15倍；原文“15% faster”不等于耗时减少15%。（§4.1、附录E.1。）

表2的GPT-2 small在8×A100-40GB、有效batch512、400K steps、mixed precision下，HF/Megatron/FA为9.5/4.7/2.7天，PPL均18.2。表数约为HF的3.5倍速度，caption却概括“up to3×”，本页保留实数与差异。表4的4K context比1K得到更低PPL包含上下文改变，不能称同一attention算子的精度收益。

### 稀疏扩展与实现边界

§3.3的block-sparse扩展按预定义块mask跳过零块，IO为$\Theta(Nd+N^2d^2s/M)$，$s$是非零块比例。它对选定的稀疏attention精确，但通常已改变dense模型；Path-256的63.1%是该稀疏路径，不能据此给稠密算法的64K性能背书。

§5承认新attention变体与硬件需要CUDA工程适配。历史实现讨论Turing/Ampere及head dim 16/32/64/128，不代表当前FA2/FA3或任意dtype、异长Q/K/V、GQA、分页、分布式attention已覆盖。现代固定实现的接口和在线状态见[vLLM Attention](../implementation/vllm-attention-operator-design.md)与[MLA执行](../implementation/mla-cache-and-attention-execution.md)；这两页保留各自源码版本与所选分支，原论文不替代库内部核验。量化后的Q/K/V、概率舍入与尺度位置还会改变数值路径，不能由原浮点算法等价性推出无损低位实现。

## 来源身份

- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（`https://arxiv.org/abs/2205.14135v2`），arXiv:2205.14135v2；正文§1–5、算法0–1、定理1–2/命题3，附录B算法2/4与式3–6、定理5，附录C证明，D/E必要实现和实验条件、图2与表1/2/4/7/11/21。PDF与同版TeX核对；归一化累计改写、dropout梯度展开和教学例为整理者分析。未逐点重建全部曲线/对照，未核作者CUDA代码或运行模型、设备与性能。
