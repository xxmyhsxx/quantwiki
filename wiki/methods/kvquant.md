---
title: KVQuant：RoPE 前缓存、敏感度码本与稀疏离群值
type: method
tags:
  - kv-cache
  - calibration
  - codebook
  - sparsity
sources:
  - raw/papers/2026-09-22/kvquant/paper.pdf
updated: 2026-09-28
---

# KVQuant：RoPE 前缓存、敏感度码本与稀疏离群值

KVQuant v6 对 Key 使用 RoPE 前逐通道量化，对 Value 使用逐 token 量化，再结合离线敏感度码本、逐向量稀疏离群值和首 token 浮点保留。它将低比特码本索引存入缓存，读取时查表恢复浮点计算；不是 INT3/INT4 attention GEMM。

本页读原文正文及 RoPE、敏感度、归一化、校准、位宽估算、kernel 和限制附录；本地归档无代码，未进行实现运行。基础对象与 K/V 误差路径见 [KV cache 粒度](../theory/kv-cache-quantization-objects-and-granularity.md)。

## 1. 为什么量化轴与 RoPE 位置要共同选择

Key 在 RoPE 前往往有跨 token 较稳定的高幅通道；按通道设尺度可以使这些通道不支配其他通道的网格。Value 的离群值更依赖 token，所以按 token 设尺度。这里的“稳定”来自论文所研究模型的观察，不是所有架构的必然分布。

RoPE 对成对通道作位置相关旋转。即使输入某通道始终很大，旋转后大值会以不同位置角度分散到另一通道，削弱逐通道静态尺度的适配性。令 $k_n$ 为位置 $n$ 的未旋转 Key，$\widetilde q_m=R_mq_m$，KVQuant 的读路径是

$$\widehat\ell_{mn}=\widetilde q_m^TR_n\widehat k_n/\sqrt d.$$

它缓存 $\widehat k_n=Q(k_n)$，在读取后恢复位置旋转。不能忘记 $R_n$，也不能用单个与历史位置无关的 Query 变换消除所有 $R_n$。误差为 $\widetilde q_m^TR_n(\widehat k_n-k_n)/\sqrt d$，RoPE 的正交性保持向量误差范数，却不会消除分数量化误差。

“先量化后旋转”与“先旋转后量化”数值不同，因为网格是坐标相关的。论文还报告逐 token Key 的基线可能更适合 post-RoPE，故 pre-RoPE 与 per-channel 的配合不能拆成无条件通则。（原文 Key 粒度、pre-RoPE 方法与 RoPE/粒度消融附录。）

## 2. 固定码本如何支持动态缓存

每层分别为 K/V 学一套 $2^b$ 个标量 signpost。令激活元素 $A_i$ 所属通道/向量的尺度与偏移为 $s_i,z_i$，归一化 $u_i=(A_i-z_i)/s_i$。存索引 $j(i)$ 后，恢复值

$$\widehat A_i=s_ic_{j(i)}+z_i.$$

同层共享码本 $c$，每个向量只改变仿射恢复参数；因此不需要每到一个 token 就运行 K-means。码本是非均匀标量表示，不是将多个通道联合编码的向量量化。相关预算区别见 [码本与位宽](../theory/codebook-quantization-and-bit-budget.md)。

权重为对角敏感度 $F_{ii}$ 时，原坐标平方误差目标变为

$$\sum_iF_{ii}(A_i-\widehat A_i)^2
=\sum_iF_{ii}s_i^2(u_i-c_{j(i)})^2.$$

所以在归一化数据上训练码本必须用 $F_{ii}s_i^2$，不能只用 $F_{ii}$。固定分配下的码字更新是带这些权重的均值；再重新分配到最近码字，迭代得到局部解。尺度平方来自坐标变换，是等式；聚类是否得到全局最优则没有保证。（原文归一化推导附录；加权均值为目标的整理者展开。）

## 3. “Fisher”权重的具体近似

论文从损失变化的一阶近似 $\Delta L\approx J^T\Delta A$ 分析其平方：

$$E[(\Delta L)^2]\approx J^TE[\Delta A\Delta A^T]J.$$

只有误差交叉二阶矩可忽略时，才得到 $\sum_iJ_i^2E[\Delta A_i^2]$。**零均值本身不足以消去交叉项**：还需不相关等条件。原文以零均值高斯量化噪声解释这个步骤，但真实确定性舍入不自动满足独立、零均值或局部线性。

作者用校准梯度平方形成敏感度权重，将其称为对角 Fisher。它是输出损失扰动的经验代理，不应写为激活 Hessian 的精确对角或真实任务误差保证。Fisher 与梯度平方的区别见 [曲率加权量化误差](../theory/curvature-weighted-quantization-error.md)。

## 4. 离群值按谁的范围决定

将每向量超过上下阈值的元素从稠密量化对象中排出，以浮点值和位置另存，稠密剩余区间缩到 $[-1,1]$ 学习/使用码本。Key 的“向量”是同一通道跨 token 的统计，Value 的“向量”是一 token 的通道集合。不能用一个全层阈值替代它们：普通低幅通道的异常值可能远低于高幅通道的常见值。

Key 的尺度、偏移、离群阈值用离线校准固定，新增 token 按固定参数编码，不需反复重编码历史缓存。因此“逐通道一定需要在线分组窗口”并不成立；代价转为对校准分布的依赖和可变稀疏占用。Value 每个新 token 在线求阈值、尺度与偏移。

“1% outliers”是目标设置，尤其对采用固定阈值的 Key，运行数据的实际比例可能变化。不能把它当作对所有输入的严格稀疏上界。作者给出 CPU top-k 与 GPU 后续投影重叠的量级：A6000 + Xeon Gold 6126 下 top-k 0.026 ms，QKV projection 0.172 ms，重叠约 0.173 ms；这不是所有服务调度下免费 CPU offload 的保证。

首 token 另保留 FP16，并从离线校准统计和码本训练中排除，减少 attention sink 的极端影响。它不是 [KIVI](kivi.md) 的最近 $R$ 个 token 滑动窗口；二者预算和保留对象不同。

## 5. 写入、读取与稀疏格式

按论文的矩阵朝向，Key 以通道×token 存储，稀疏部分用 CSC，新增 token 就追加一列；Value 以 token×通道存储，用 CSR，新增 token 追加一行。CSR/CSC 名称必须连同矩阵方向解释。

作者的 kernel 将四位索引查共享层码本，结合向量尺度恢复 FP16；Key 再在读出时应用 RoPE，然后与 Query 乘加。稀疏分支按非零项平衡工作量，避免离群分布不均，作者称稠密与稀疏路径在一次调用中完成。所有 arithmetic 以原文 FP16 描述为准，没有原生 INT4 attention 算术主张。

prefill 的任务设置先用浮点 K/V 完成当前 prompt 的 attention，再压缩缓存供后续生成；不能据此认定 prefill 也已获得低比特计算收益。原文限制还明确：当前稀疏缓存追加会因拼接而复制历史数据，未来计划分块分配。微基准收益并未消除这个端到端实现开销。

## 6. 真实位宽不是标题中的三位

主要结果用 16 条、每条 2K token 的 WikiText-2 训练样本及其梯度校准。预算包括稠密码、FP16 尺度/偏移、码本、稀疏值/索引、指针和首 token。

按原文估算，稀疏值 16 bit、元素位置 16 bit，每 token 的 CSR/CSC 指针 32 bit。仅 1% 稀疏元素的值和位置就额外贡献约 $0.01(16+16)=0.32$ bit/元素，尚未计其他元数据。所以 nuq3-1% 实际约 3.32–3.35 bit，对 FP16 的 KV 压缩约 4.8 倍，而非 $16/3$。这是原文指定长度和格式的估算，不是本地实测分配大小。

## 7. 质量、容量和速度分别支持什么

**短/常规上下文质量。** PPL 用 teacher forcing、模型原支持长度（LLaMA 2K、Llama-2 4K、Llama-3/Mistral 8K）。例如 LLaMA-7B WikiText-2 浮点 5.68，nuq3-1% 为 5.75。保留 sink 与离群值是该配置的一部分，不能将其写成纯三位全缓存结果。

**长上下文任务。** 32K 扩展模型的 passkey、LongBench、RULER 才检验利用上下文的能力。LLaMA-2-7B-32K 的 LongBench 平均为 FP16 31.96、nuq3-1% 31.21；RULER 为 56.40 对 53.65，nuq2-1% 只有 36.54。故简单 passkey 保持不能证明复杂长上下文能力无损。比较 KIVI 时也保留其分组、残差窗口与不同平均位宽条件。

**百万/千万容量。** 标题对应内存估算：LLaMA-7B 的 nuq2 缓存，1M token 约 64 GB，8 卡提供 10M 的容量设想。这里不是 nuq3-1% 的完整质量配置，也没有百万/千万任务质量或完整延迟实测。附录明确还需要训练支持超过 100K 的模型。能存下、能执行和能正确利用上下文是三个独立命题。

**kernel 速度。** A6000、batch 1、LLaMA-2-7B-32K、nuq4-1%：16K 历史下 Key 的总 packing+dense+sparse 为 126.3 μs，对 FP16 matvec 219.4 μs；Value 为 124.5 对 203.7 μs。作者按每层激活、1,000 次迭代平均测量。约 1.7 倍指这一层次，不是 nuq3 的全模型或多请求服务吞吐。不能把“KV 常受带宽限制”升级成查表、稀疏和 RoPE 在所有形状下都免费。

原文不同表格的 sink 设置和基线 group-size 文字存在差异，本页使用明确标出配置的主表与附录预算，不合并无条件数值。没有代码核验、模型复现或本地硬件计时。

## 来源身份

- [KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization](https://arxiv.org/abs/2401.18079v6)，arXiv:2401.18079v6；2026-09-22 归档。方法、长上下文结果与预算/实现/限制附录分别承接。
