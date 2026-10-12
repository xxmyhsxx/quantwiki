---
title: FlexQ：INT6 分组量化与二值 Tensor Core 执行
type: method
tags:
  - activation-quantization
  - group-quantization
  - data-format
  - kernels
sources:
  - raw/papers/2026-09-21/flexq/paper.pdf
updated: 2026-10-12
---

# FlexQ：INT6 分组量化与二值 Tensor Core 执行

FlexQ选择6-bit权重和以6 bit为主的激活，用group128控制局部范围，在所测Llama的down_proj输入保留8 bit。系统将多位整数拆成位平面，用二值Tensor Core执行位对乘积，再按位权和组尺度恢复浮点输出。它没有把INT6改名成硬件原生格式，而是重新组织已有二值指令的计算和访存。（FlexQ v2（`https://arxiv.org/abs/2508.04405v2`），§4、图5–9。）

本页依据固定17页PDF及正文源文件，核方法、式1–6、表1–6与代表性能图；代码分析见[所选实现](../implementation/flexq-bitplanes-and-group-restoration.md)。没有运行作者模型、编译或GPU。与 [Atom](atom.md) 保护少量原通道并采用INT4/INT8混合不同，FlexQ统一使用6-bit线性层权重，通过激活精度和位平面布局处理质量与执行成本。

## 1. 免外部校准并不等于没有数据相关观察

§4.1/图4以Llama-7B和Llama2-7B逐种线性层量化的PPL观察敏感性，down_proj输入最敏感。实际配方据此固定：所有线性层权重6 bit，down_proj输入8 bit，其他线性层输入6 bit；非GLU的OPT按W6A6统一处理。它不是为每个请求或每个block在线求解敏感度，也不提供所有新架构的最佳精度分配。

权重可离线按自身范围量化和打包；激活在每次前向从当前group计算范围。calibration-free指这一量化配方不需要额外校准数据拟合尺度或变换，作者仍使用WikiText2做敏感性分析和质量评测，在线也仍需范围归约、舍入和packing。不能由“免校准”推出不会遇到分布/范围问题。

沿归约维每128元素共享一个尺度，缩小范围的同时增加元数据及组恢复次数。图5将QKV/gate/up输入量化打包融合到norm，将down输入量化融合到激活，o_proj前另有量化打包节点；attention本身及残差仍保留浮点。W6Ax不能解释成整个模型的所有计算和KV缓存都是INT6。

## 2. 位平面乘法需要位权与符号位

论文式1–5从非负整数二进制展开：令 $a$、$b$ 分别为q-bit、p-bit整数，$a=\sum_{r=0}^{q-1}2^ra_r$、$b=\sum_{s=0}^{p-1}2^sb_s$、每个bit为0或1，则 $ab=\sum_{r,s}2^{r+s}a_rb_s$。向量位对的内积可用AND后的popcount求出。W6A6/W6A8分别有36/48个逻辑位对；布局和指令可并行处理，不意味着启动36/48个独立kernel，也不等于一次原生INT6 MMA。

对有符号二补码，最高位的权重应为负。以B表示通用位宽，定义 $\lambda_B(r)=2^r$（$r<B-1$）、$\lambda_B(B-1)=-2^{B-1}$，则

$$ab=\sum_{r=0}^{q-1}\sum_{s=0}^{p-1}\lambda_q(r)\lambda_p(s)a_rb_s.$$

例如2-bit的−1为11，4-bit的−3为1101；按有符号位权乘积为3，若全部取正位权则变成3×13=39。对称量化不消除符号位；unsigned展开不能直接套给signed码。（本段为分配律教学推导，固定实现用位对是否恰好包含一个最高位决定符号。）

论文式6把分组尺度加入bmma求和，却省略式5的 $2^{r+s}$；式4将bmma定义为未加权的1-bit矩阵乘积，故按该定义，式6不能直接还原多位乘积。采用行token $A\in\mathbb R^{T\times d}$、$W\in\mathbb R^{d\times o}$，A、W的整数码位宽分别为q、p，完整signed分组关系应为

$$\widehat Y_{tj}=\sum_g s^A_{t,g}s^W_{g,j}
\sum_{r=0}^{q-1}\sum_{s=0}^{p-1}\lambda_q(r)\lambda_p(s)
\operatorname{BMMA}(A_g^{(r)},W_g^{(s)})_{tj}.$$

这里BMMA只计算0/1位平面沿组内K的内积，尺度恢复和位权是另外两层操作。固定代码确实有temp_mul的位权与符号处理；原式缺项不应据此升级成作者实际kernel缺少位权。公共解释见[位分解执行](../implementation/quantized-matmul-scaling-execution.md#16-多位整数怎样用二值点积执行)。

## 3. 紧凑存储需要适合指令的布局

§4.3把输入组织为 $[K/128,M/c,q,c,128]$ 位布局，$c=\min(M,8)$，每个位平面沿K连续打包；权重采用对应布局。所选代码用ballot收集32个bit、brev调整顺序，128个bit装入4个uint32。它保持6-bit信息量，却通过整字和chunk访问对齐接口，不能用“单元素6 bit不对齐”推断所有packed访问都只剩37.5%利用率。

M很小时，将位维度放进MMA的行空间并做warp内位权归约，减少单纯将M补到8造成的空计算；仍有尾部/padding及布局限制。group恢复融合减少额外全局读写，多stage和寄存器缓冲尝试重叠加载与计算，L2 evict_first只是缓存优先提示。图6–9及所测shape支持此设计，不能据布局图保证任意shape无bank conflict或在所有GPU上最快。

若只计6-bit载荷和每128元素一个FP16尺度，元素预算为 $6+16/128=6.125$ bit、相对FP16为约2.61倍；这不是整网显存比例。原生激活尺度有复制/对齐，仍有FP16输出、attention/KV与workspace。INT6与FP6也有不同的编码、范围和计算语义，不能互换。

## 4. 质量结果绑定混合激活配方

表2虽统标W6A6，§5.1明确Llama的down_proj实际A8，OPT才全W6A6。表2的WikiText2结果：

| 模型 | FP16 | FlexQ | 绝对PPL增量 |
| --- | ---: | ---: | ---: |
| Llama1-7B | 5.68 | 5.70 | 0.02 |
| Llama1-65B | 3.53 | 3.55 | 0.02 |
| Llama2-70B | 3.31 | 3.34 | 0.03 |
| Llama3-8B | 6.14 | 6.24 | 0.10 |

“至多0.1”覆盖该表所列Llama的WikiText2，不能推广为全部模型/数据集。C4的Llama3-8B为8.88→9.05，增0.17；表5的OPT-13B WikiText2为10.13→10.37，增0.24，且不优于该行所有基线。表1的4/6 bit对照也只覆盖所列SmoothQuant/OmniQuant/I-LLM，不能推出所有4-bit方法必然大幅损失。

表3–4采用6项常识任务：Llama1-7B FP16/FlexQ均分64.09/63.80，降0.29个百分点；Llama1-65B为71.04/70.84，降0.20个百分点。Llama2-70B的DuQuant/FlexQ为70.62/70.59，不能复述“所有模型均优于其他方法”。单次略高于FP16的分数没有方差/置信区间，不足以证明量化提升一般泛化能力。基线DuQuant采用DuQuant+LWC，缺失任务/模型保留空项，不视为共同协议完整排名。

## 5. 微基准和端到端证据分别说明

§5.3的kernel实验使用RTX3090Ti 24GB、CUDA12.1，batch1/4/8及多种Llama/OPT线性层shape，down采用W6A8。对cuBLAS W8A8的平均速度比为1.78/1.81/1.82，对ABQ-LLM对应位宽为1.24/1.24/1.27；这些是不同shape的报告均值，不是每个真实请求的速度。（图10。）

端到端使用A6000 48GB/FasterTransformer，比较FP16、CUTLASS W8A16和SmoothQuant W8A8；ABQ-LLM未纳入端到端比较。图11固定输入15 token、变化sequence长度；报告Llama13B最高2.38倍速度/2.28倍内存比FP16，跨所列模型对SmoothQuant速度比1.25–1.33、内存比1.19–1.24。图12固定sequence1024、batch1/2/4/8；batch8的7B/13B比FP16为1.89/2.01倍。不能混成相同batch、相同长度下统一的1.33倍结果，也不能直接与Atom的continuous batching吞吐比较。

表6只对 $(1,4096)\times(4096,4096)$ GEMV测递进优化：cuBLAS37.02μs、vanilla29.44、warp归约26.54、pipeline16.12、cache提示15.03；去掉融合恢复14.92。最终对cuBLAS速度比约2.46，恢复的额外延迟约0.74%（相对14.92）；这是局部shape的消融，不能泛化成所有batch恢复成本低于1%。默认代码计时把packing与GEMM分开，质量/性能代码及真实导出条件见实现页。

## 来源身份

| 来源 | 固定身份 | 本页范围 |
| --- | --- | --- |
| FlexQ: Efficient Post-training INT6 Quantization for LLM Serving via Algorithm-System Co-Design（`https://arxiv.org/abs/2508.04405v2`） | arXiv:2508.04405v2；PDF SHA-256 b0dc825fbc7f6e9552c3d26780949932a6d0c397858a518da1e23e323d45ca38 | 正文方法/实验、式1–6、表1–6、代表设计与性能图；未逐点重建所有曲线，未运行模型或GPU |
