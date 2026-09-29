---
title: HAWQ-V2：平均曲率、扰动能量与混合精度搜索
type: method
tags:
  - mixed-precision
  - second-order
  - qat
sources:
  - raw/papers/2026-09-22/hawq-v2/paper.pdf
updated: 2026-09-28
---

# HAWQ-V2：平均曲率、扰动能量与混合精度搜索

HAWQ-V2 将逐层敏感度从最大 Hessian 特征值改为平均特征值，再结合每档位宽的实际权重扰动建立配置代价，在受排序约束的候选中选择模型大小与误差代理的 Pareto 前沿。它的结果包含量化感知微调，主体实验是 CNN 分类与检测；不能把它直接归为无需训练的 LLM PTQ。

本页依据 arXiv:1911.03852v1，覆盖 §2–3、附录 A–C。共同的预算问题见 [混合精度分配](../theory/mixed-precision-allocation.md)，任务 Hessian 与 Fisher 的差别见 [曲率加权误差](../theory/curvature-weighted-quantization-error.md)。本次没有运行训练或 Hessian 估计。

## 1. 最大曲率为什么不足，平均曲率也有什么前提

在训练驻点附近，单独扰动第 $i$ 个块的参数 $w_i$，记误差为 $e_i$，参数个数为 $n_i$，损失的该对角块 Hessian 为 $H_i$。二阶近似为

$$\Delta L_i\approx\tfrac12 e_i^\top H_i e_i.$$

$\lambda_{max}(H_i)$ 控制最坏方向，不能反映其余方向的代价。论文的例子 $F_1(x,y)=100x^2+y^2$ 和 $F_2(x,y)=100x^2+99y^2$ 有相同最大 Hessian 特征值 200，但第二方向代价相差 99 倍。

论文 Assumption 1 不仅假设驻点和 $H_i\succeq0$，还要求微调后扰动在各 Hessian 特征方向具有相同系数：$e_i=\alpha_i\sum_{j=1}^{n_i}v_j$。于是

$$\|e_i\|^2=n_i\alpha_i^2,\qquad
\tfrac12 e_i^\top H_i e_i=\tfrac12\frac{\operatorname{tr}H_i}{n_i}\|e_i\|^2.$$

平均 trace $s_i=\operatorname{tr}(H_i)/n_i$ 因而在等扰动能量下用于排序。（§2.1，Lemma 1。）原证明把二阶 Taylor 写成等号，并在最后的差值中省掉共同 $1/2$；排序不受后者影响，但一般非二次损失仍有高阶余项，不能保留成全局精确不等式。

**另一种教学解释。** 若随机扰动满足 $\mathbb E[ee^\top]=\sigma^2I$，有 $\mathbb E[e^\top He]=\sigma^2\operatorname{tr}H$，同样连接平均曲率与期望误差。它是随机各向同性假设下的期望关系，与论文的确定性等系数假设不同；真实舍入误差一般不满足任意一种。

**方向反例。** $H=\operatorname{diag}(100,1)$ 下，$e_1=(1,0)^\top$、$e_2=(0,1)^\top$ 能量相同，二阶代价分别为 50、0.5；平均 trace 代理对两者都给出 25.25。因而精确估计 trace 也无法恢复已经丢失的误差方向。这不是计算精度问题，而是代理的信息限制。

## 2. 不显式构造 Hessian，怎样算 trace

取随机向量 $z$ 满足 $\mathbb E[zz^\top]=I$，可以是独立 Rademacher 或标准高斯，有

$$\operatorname{tr}H=\mathbb E[z^\top Hz],\qquad
\widehat{\operatorname{tr}H}=\frac1M\sum_{r=1}^M z_r^\top H z_r.$$

Hessian-vector product 可以通过 $Hz=\nabla_w[(\nabla_wL)^\top z]$ 得到，不存 $n_i\times n_i$ Hessian。这里 $z$ 在求导时固定。省掉二次存储不意味着无需反向图或二阶求导成本。（式 2.6–2.7。）

有两种不同近似：用有限训练样本构造子采样 Hessian，以及用有限随机 probe 估计该 Hessian 的 trace。增加 probe 不会自动修复数据分布偏差；比较层敏感度也需保持损失归约和样本口径一致。

§3.1 的 ResNet50 第 21 块实验支持约 512 以上样本、50 probe 时估计趋稳；作者报告 4 GPU 上全部 54 块 trace 约 30 分钟。这是该模型/实验的离线估计成本，不是任意 LLM 的通用配置，也不是推理速度。

## 3. 从层排序到具体位宽

只有 $s_i$ 尚不能选位宽，因为不同层参数量、值域和量化残差不同。对位宽 $b$ 计算 $e_i(b)=Q_b(W_i)-W_i$，式 2.11 的配置评分为

$$\Omega(\mathbf b)=\sum_i s_i\|e_i(b_i)\|_2^2.$$

原件用带横线的 $\overline{\operatorname{Tr}}$ 表示平均 trace，不能误读为总 trace 再重复计入参数量。$1/2$ 对相同设置的候选排序无影响，因此这里省略。

流程为：估计平均 trace → 按敏感度排序 → 只允许更敏感层使用不低于另一层的位宽 → 计算各配置的存储量和 $\Omega$ → 删除同时更大且代理误差更高的支配点 → 在给定大小预算内选择前沿配置 → 量化感知微调并评测。

这个排序是减少组合搜索的附加限制，不是从平均 trace 单独推出的全局最优性质。若一层的实际量化残差很小，强制它因高 trace 而使用更高位宽，可能排除更好的预算分配。

对 $L$ 层、$m$ 档位宽，无约束有 $m^L$ 个配置；固定排序后非降序分配等价于选择每档层数，数量为

$$\binom{L+m-1}{m-1}
=\sum_{j=1}^{\min(m,L)}\binom mj\binom{L-1}{j-1}.$$

这是对附录 B 的组合解释，$L=50,m=4$ 时为 23,426，与文中约 $2.3\times10^4$ 一致。但该减少来自舍弃非单调配置，不能当作对原搜索空间的无损压缩。论文附录组合符号的上下标印法不符合通常约定，本页按“选档位、切连续组”的文字含义写出。

跨层求和还忽略 $e_i^\top H_{ij}e_j$；QAT 后参数和曲率也可能变化。作者明确没有保证 $\Omega$ 最优就给出最高最终准确率。

## 4. 激活 Hessian 为什么可以按样本处理

对层激活 $a_j(x)$ 求任务损失 Hessian，可分析激活精度。在样本独立、总损失为逐样本之和的条件下，不同输入对应激活互不依赖，所以把全数据激活拼接后，Hessian 按样本成块对角。

一致的记法是先定义 $H_j^{(r)}=\nabla^2_{a_j(x_r)} f_r$，再写

$$H_{a_j}=\frac1N\operatorname{blockdiag}(H_j^{(1)},\ldots,H_j^{(N)}),\qquad
z^\top H_{a_j}z=\frac1N\sum_r z_r^\top H_j^{(r)}z_r.$$

这样可以逐样本生成相应大小的 probe，适应检测任务变分辨率输入，无需保存全数据拼接向量。原文 §2.2 对子块定义及式 2.10 的 $1/N$ 有重复计数的记号风险，上式只保留一次归约。

若有跨样本损失、训练态 batch 统计等耦合，这个块对角论证需重新检查。权重收敛到驻点也不意味着激活梯度为零，所以从权重的 Lemma 1 直接推广到激活，仍有一阶项与扰动分布假设，不能只换求导变量就视为完整证明。

## 5. 量化与微调的实际角色

附录 A 先裁剪到 $[q_0,q_{2^b-1}]$，以 $\Delta=(q_{2^b-1}-q_0)/(2^b-1)$ 舍入，前向使用恢复值，反向使用 STE。允许裁剪范围小于原 min/max，以减少离群值支配。

配置评分本身不必逐个候选做 QAT，这使候选选择便宜；最终结果却包含 QAT。原文 v1 没有完整列出所有训练轮数、优化器和各网络每层位宽，本文不补造复现配方。方法的对象与 [LSQ](lsq.md) 不同：HAWQ-V2 主要决定哪些层用几位，LSQ 主要学习指定配置下的量化步长；两者不是同一个优化变量。

## 6. 结果应怎样读

以下为原文表 1–4 的作者报告，保留原表“MP”表示混合精度，不把“2 MP”解释为所有层或精确平均均为 2 bit。

| 模型与数据 | 浮点参考 | HAWQ-V2 | 权重大小与设定 |
| --- | ---: | ---: | --- |
| Inception-V3 / ImageNet Top-1 | 77.45 | 75.68 | 7.57 MB，2 MP 权重 / 4 MP 激活 |
| ResNet50 / ImageNet Top-1 | 77.39 | 75.76 | 7.99 MB，2 MP / 4 MP |
| SqueezeNext / ImageNet Top-1 | 69.38 | 68.38 | 1.07 MB，3 MP / A8 |
| RetinaNet / COCO 2017 mAP 0.5:0.05:0.95 | 35.6 | 34.1 / 34.4 / 34.8 | 同为 17.90 MB 权重，激活分别 A4 / 4 MP / A6 |

RetinaNet 的 4 MP 激活比 A4 提高 0.3 mAP，但激活压缩率从 8.00 降到 7.62；是精度与存储的取舍，不能写成相同激活预算下无代价提升。不同方法的首尾层配置和训练条件未必一致，表内结果不能作为脱离协议的算法排名。

论文报告模型大小和理论压缩比，没有在这些表中给出真实混合位宽部署的端到端延迟；硬件是否支持每档算子、转换与融合成本需单独验证。

## 来源身份

| 来源 | 版本 | 范围 |
| --- | --- | --- |
| [HAWQ-V2: Hessian Aware trace-Weighted Quantization of Neural Networks](https://arxiv.org/abs/1911.03852v1) | arXiv:1911.03852v1 | §2–3、Lemma 1、式 2.6–2.11、表 1–4、附录 A–C；关键公式及平均 trace 横线回查页图 |
