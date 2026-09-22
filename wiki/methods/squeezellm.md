---
title: SqueezeLLM：敏感性加权的非均匀量化与稠密稀疏分解
type: method
tags:
  - ptq
  - weight-quantization
  - sensitivity
  - sparsity
sources:
  - raw/papers/2026-09-21/squeezellm/paper.pdf
  - raw/papers/2026-09-21/squeezellm/source.eprint
  - raw/repositories/2026-09-21/squeezellm/source/squeezellm/quant.py
  - raw/repositories/2026-09-21/squeezellm/source/squeezellm/outliers.py
  - raw/repositories/2026-09-21/squeezellm/source/quantization/nuq.py
updated: 2026-09-22
---

# SqueezeLLM：敏感性加权的非均匀量化与稠密稀疏分解

SqueezeLLM 用任务损失的梯度敏感性指导非均匀标量量化，再将少量极端和高敏感权重放到 FP16 稀疏分支。其出发点是单 batch 生成常受权重读取带宽限制：压缩存储后查表恢复浮点权重，可能比坚持使用等间距整数网格更合适。这个前提不自动覆盖大 batch、prefill 或所有硬件。

## 1. 敏感性来自任务梯度

令 $w$ 为展平模型参数，$e=\widehat w-w$，任务损失的 Taylor 近似为

$$\mathcal L(w+e)-\mathcal L(w)\approx g^\mathsf Te+\tfrac12e^\mathsf TH_{\rm task}e.$$

原文 Methodology 假设模型已收敛从而忽略一阶项，用样本梯度外积构造经验 Fisher，再忽略跨参数耦合：

$$F=\frac1{|\mathcal D|}\sum_{d\in\mathcal D}g_dg_d^\mathsf T,
\qquad f_i=F_{ii},\qquad
\min_{Q}\sum_i f_i(w_i-Q(w_i))^2.$$

这里是连续的几层近似：目标处梯度未必为零，经验 Fisher 不普遍等于真实 Hessian，对角化又舍弃了权重间相互抵消。不能把它写成 [GPTQ](gptq.md) 的输入 Gram 矩阵 $XX^\mathsf T$：后者精确描述固定输入下的线性层平方重构，前者试图反映最终任务的局部敏感性，并需要反向传播。

还应区分 $\mathbb E[g_i^2]$ 与 $(\mathbb E[g_i])^2$；前者在正负梯度抵消时仍可很大。以上是论文的敏感性定义，本页没有核对其完整梯度采集代码或实际运行。

## 2. 加权 K-means 如何决定非均匀网格

每个输出通道单独存一张包含 $2^b$ 个 FP16 代表值的表，权重保存 $b$ bit 表索引。代表值无需等间距；这是**标量非均匀量化**，与 [AQLM](aqlm.md) 或 [QuIP#](quip-sharp.md) 多维码字的表示不同。

对固定簇 $S_k$，最优代表值由目标求导得到

$$c_k=\frac{\sum_{i\in S_k}f_iw_i}{\sum_{i\in S_k}f_i}.$$

对固定代表值，若 $f_i>0$，单个权重仍选择欧氏距离最近的代表值，因为 $f_i$ 对该点所有候选相同。敏感性的主要作用是拉动共享代表值；它不是给每个点重新定义一套不同的最近邻边界。零总权重簇不由这个公式唯一确定，需要实现选择回退。

教学例子：同一簇内两个数为 0 和 2，敏感性为 9 和 1，普通均值为 1，加权均值为 0.2。加权误差从 10 降为 3.6，而未加权误差从 2 升为 3.28。它说明“权重 MSE 更小”与“敏感性目标更小”并非一回事；不是模型精度实验。

## 3. 为什么还需要稀疏保留

非均匀码本仍只有有限代表值。分布尾部会占用代表值，极少数高敏感点又会把代表值拉向自身，因此论文分两种依据选择例外：

- 按权重分布分位数提取范围外的尾部值，缩小主体需要表示的范围。
- 按 Fisher 敏感性提取少量高敏感值，直接用 FP16 保存，让剩余代表值服务于其他权重。

默认实验比较 dense-only 与 0.45% 稀疏配置，后者由 0.05% 敏感值和 0.4% 尾部值构成。二者不是同义分类，高幅值不必高敏感；支持集合的处理需避免重复计数。（Methodology，Dense-and-Sparse；Evaluations，Quantization Details。）

论文以 $W=D+S$ 解释分解。实际非均匀表不一定包含精确零，直接计算 $Q(D)x+Sx$ 会在选出位置多加 $Q(0)$。已有固定代码 `a5fd71f3` 的 `outliers.py:remove_outliers` 先选敏感值并将主体对应位置置零，再从剩余权重提取幅值例外；`quantization/nuq.py` 通常对这些零位置赋零聚类权重（整行样本权重和为零时回退为等权），仍生成其最近代表值索引。`quant.py:QuantLinearLUT.pack2` 随后计算 `zero_mapping`，在所选非零例外上保存 $S'=S-Q(0)$，从而在实数算术下让 $Q(D)+S'$ 的这些位置恢复目标值。它解决的是重复相加，并非要求学习出一个恰好为零的代表值。

同一代码片段将 LUT 和稀疏值注册为 FP32，CSR 行列索引为 INT32；这与论文按 FP16 讨论的紧凑表示需要分别计账。这里只核对提取、聚类输入和打包片段，没有验证梯度采集、完整 CUDA 路径或模型运行。[SpQR](spqr.md) 也采用主体加稀疏修正，但其主体网格、元数据与索引格式不同，checkpoint 不能互换。

若不含例外和其他参数，$m\times n$ 矩阵逐行 FP16 表的理想存储为

$$b_{\rm dense}=b+\frac{16\cdot2^b}{n}.$$

对 $b=3,n=4096$ 是 3.03125 bit/weight；整个模型按矩阵大小加权后不同。加入稀疏分支还要算 FP16 值、列索引、行指针与布局开销，而非只算高精度值本身。

## 4. LUT 和稀疏算子怎样配合

原文 Dense-and-Sparse Kernel Implementation 描述 3/4 bit CUDA matvec：读取压缩索引，查 FP16 表，逐片恢复权重后进行浮点乘加。输入激活保持浮点，不属于原生整数乘法。

稀疏矩阵用 CSR 表示。由于不同输出行例外数目很不均匀，一线程负责一行会失衡；论文按非零项数量分配工作，设每线程处理 10 个非零项，并处理跨线程的行归约。dense 与 balanced sparse 路径在一次调用中组织，减少单独累加输出的开销。这里说明论文设计，没有把它等同于本地已验证的 kernel 调度。

这解释了为什么“只有不到 1% 稀疏值”不代表开销也不到 1%：索引、访存、负载均衡与归约可能比算术量更重要。通用前提见 [roofline](../fundamentals/hardware/arithmetic-intensity-and-roofline.md) 与 [GPU 计算模式](../fundamentals/operators/gpu-kernel-computation-patterns.md)。

## 5. 实验与消融

v4 对 LLaMA 等基础模型用 C4 训练集的 100 个随机样本估计敏感性，PPL 以 2048 token 分块评测。LLaMA-7B、3 bit、C4 消融为：

| 配置 | PPL |
| --- | ---: |
| FP16 | 7.08 |
| 非均匀 K-means，无敏感性加权、无稀疏 | 18.08 |
| 敏感性加权，无稀疏 | 7.75 |
| 敏感性加权，0.05% 稀疏 | 7.67 |
| 敏感性加权，0.45% 稀疏 | 7.56 |

该消融说明非均匀表示本身不够，敏感性目标和例外保留提供不同增量。（附录 Sensitivity-Based Quantization 表；主文主结果表。）

主文在 A6000、batch 1、LLaMA-7B 生成 128 token 的测量中，FP16、dense-only 3.02 平均 bit、0.45% 稀疏的 3.24 平均 bit 配置分别用时 3.2、1.5、1.7 秒。这些是作者实现的整段生成时间；稀疏路径提高质量同时增加时间。论文中某个 grouped GPTQ 实现的重排开销不能推广为所有现代 GPTQ 后端都慢。

少量校准梯度与对角近似可能遗漏分布变化或跨权重相互作用。方法页提供原文报告和教学推导，本次没有运行模型、生成评测或 CUDA 复现。

## 来源身份

[SqueezeLLM: Dense-and-Sparse Quantization](https://arxiv.org/abs/2306.07629v4)，arXiv:2306.07629v4。使用 Memory Wall、Methodology、Evaluations，以及评测配置、敏感性消融和稀疏负载不均衡附录。

固定代码：[SqueezeAILab/SqueezeLLM](https://github.com/SqueezeAILab/SqueezeLLM/tree/a5fd71f353bf569feb7b55737c1c1493e78e8f31)，commit `a5fd71f353bf569feb7b55737c1c1493e78e8f31`；仅使用正文列明的提取、聚类与打包片段。
