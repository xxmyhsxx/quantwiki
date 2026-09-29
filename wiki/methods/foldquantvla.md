---
title: FoldQuantVLA：共享坐标一致性与原生整数动作推理
type: method
tags:
  - vlm
  - ptq
  - deployment
  - mixed-precision
sources:
  - raw/papers/2026-09-28/foldquantvla/paper.pdf
  - raw/repositories/2026-09-28/foldquantvla/source/foldquant/foldq.py
  - raw/repositories/2026-09-28/foldquantvla/source/foldquant/llm_gptq.py
  - raw/repositories/2026-09-28/foldquantvla/source/foldquant/kernels/tensorrt/int4_per_row/cuda/per_row_quant_int4_cuda.cu
  - raw/repositories/2026-09-28/foldquantvla/source/foldquant/kernels/tensorrt/int4_per_row/cuda/dit_int4_rowwise_gemm_fused_cuda.cu
  - raw/repositories/2026-09-28/foldquantvla/source/foldquant/kernels/tensorrt/int4_per_row/plugin/per_row_int4_linear_residual_plugin.cpp
updated: 2026-09-28
---

# FoldQuantVLA：共享坐标一致性与原生整数动作推理

FoldQuantVLA 把同一激活变换贯穿校准、权重舍入与 TensorRT 整数执行。对象是视觉—语言—动作策略的语言骨干与迭代动作专家的投影，不是整个机器人策略的所有算子。v1 的独特价值在于把等价变换的数学条件落实到共享输入和运行图，并同时报告动作数值、闭环结果与编译后延迟。

本页使用论文 v1 和官方代码 `4e76b60b4232a33a640e31639ce2a329ae0bebb3` 的定向路径阅读；没有构建 TensorRT 引擎、运行模型或机器人。

## 1. 动作推理为什么需要独立验证

一次策略查询以图像、语言、状态和初始噪声为条件，输出 $H\times d_a$ 的动作块；控制器执行前 $K\le H$ 个动作后重新观测。GR00T 的实验动作专家迭代 4 次，$H=16$；π0.5 迭代 10 次，$H=10$。因此一次观测到动作的延迟不等于单次专家迭代延迟，动作供给率也不等于闭环反馈频率。

量化误差会改变动作，进而改变下一次观测。固定观测下与浮点输出接近，只能筛查局部数值退化，不能证明轨迹或成功率不变。

目标 W4A4/W8A8 包括骨干和专家的 attention/FFN 投影。视觉编码器、非投影 attention 运算、归一化、残差运算和动作 decoder 保留浮点；GR00T 的 adaptive-LayerNorm modulation 在 W4A4 引擎中是 W4A16 类 weight-only 路径。不能把方法名称解释为全图四位。

## 2. 一份激活只能有一套共同坐标

列向量下 $y_i=W_ix+b_i$。选择可逆 $T$ 后，共享激活 $z=Tx$，每个消费者都必须使用 $\widetilde W_i=W_iT^{-1}$：

$$\widetilde W_i z+b_i=W_ix+b_i.$$

若消费者按另一变换 $T_i$ 融合，尚未量化就会产生

$$W_i(T_i^{-1}T-I)x.$$

Q/K/V 或 up/gate 可以复用一次变换和量化，但除了 $T$ 相同，还要激活位宽、裁剪比、尺度规则相同。只共享形状或变量名不足以共享量化缓冲。

对有限展开的计算图，节点变换满足

$$\widetilde f_v((z_u))=T_vf_v((T_u^{-1}z_u)),\quad
\widetilde W_{vu}=T_vW_{vu}T_u^{-1},\quad\widetilde b_v=T_vb_v.$$

外部输入输出使用单位变换，可由拓扑顺序归纳证明浮点函数保持。任意非线性都能形式上加上前后逆变换，但不代表这些算子能免费融合。实际方法只改变投影输入，输出保持原坐标，从而不移动一般旋转穿过非线性、RoPE 或残差汇合点。（论文 consistent folding 命题。）

## 3. 缩放在旋转前后不是同一个候选

运行时允许 $T=D_oRD_i$，$R^TR=I$，两侧 $D$ 为正对角矩阵：

| 方案 | 运行激活 | 离线权重 |
| --- | --- | --- |
| fold-before | $RS^{-1}x$ | $WSR^T$ |
| fold-after | $P^{-1}Rx$ | $WR^TP$ |

$RSR^T$ 通常是稠密矩阵，所以不能任意用一个对角 $P$ 表示旋转前缩放。校准必须在施加缩放的对应坐标里统计。共享输入的 SmoothQuant 型尺度使用跨消费者输入列最大权重 $w_c$ 与校准激活最大值 $a_c$，$s_c=a_c^\alpha/w_c^{1-\alpha}$，实际带数值下限与范围约束。

默认 $R$ 为宽 64 的归一化块 Hadamard，FWHT 成本 $O(d\log64)$，保范数但不保证每个输入峰值都减小。`foldq.rotation_block_for` 还会对不能整除的输入宽度缩小块；部分调用者通过 padding 保持块宽。因此论文默认 64 不是所有导出节点的硬编码保证。

若前驱 RMSNorm 输出为 $x=\Gamma\bar h$，可以将 $D_i\Gamma$ 合并为 gain；没有把尺度穿过 RMS 分母。o/down 输入缺少这样的 gain，但 down 前的缩放可放到 gated MLP 的线性 up 分支。共享消费者和边界必须同步，见 [等价变换](../theory/diagonal-scaling-equivalent-transform.md)。

## 4. 动态逐 token 量化能消除什么变化

令 $q=2^{b_a-1}-1$，裁剪比 $r$ 固定在构建阶段，逐 token

$$\delta_t=r\|z_t\|_\infty/q,\qquad
\widehat z_t=\delta_t\operatorname{clip}(\operatorname{round}(z_t/\delta_t),-q,q).$$

零向量单独对应零码。论文仅在语言 INT4 位置校准 $r$，其他位置为 1。尺度动态不等于变换或裁剪比随 timestep 重新学习。

理想算术下 $\widehat{\lambda z}=\lambda\widehat z$（$\lambda>0$），因为尺度同步乘 $\lambda$，舍入输入不变。固定 $T$ 和量化权重时，投影误差 $e_T(\lambda u)=\lambda e_T(u)$。若所有去噪步的归一化方向 $u_\tau/\|u_\tau\|$ 同分布，则

$$E\frac{\|e_T(u_\tau)\|^2}{\|u_\tau\|^2}
=E_{v\sim\nu}\|e_T(v)\|^2$$

不依赖 timestep。**方向稳定是额外假设**；动态尺度只消除幅度变化，不能保证方向分布、迭代累积误差或动作风险稳定。代码里小尺度阈值、浮点舍入还会破坏极小值下的精确齐次性。一个专家引擎可以接收所有 timestep，并不要求这个定理成立。（论文 amplitude invariance 命题；代码量化函数中的 `1e-12` 下限。）

## 5. GPTQ 必须使用变换后的二阶矩

校准 token 为 $U\in\mathbb R^{d\times N}$，$Z=TU$。固定表示后，四位权重用 GPTQ 近似优化

$$\min_{Q\in\mathcal Q}\|(\widetilde W-Q)Z\|_F^2,\qquad
G=ZZ^T=TUU^TT^T.$$

先对旧坐标求 Hessian 再随意旋转最终权重，不是在优化同一离散目标；阻尼、per-row 尺度和量化集合也必须绑定最终坐标。这个目标仍然只处理权重误差，没有自动纳入在线激活舍入误差。背景见 [二阶重构](../theory/layer-reconstruction-second-order-compensation.md)。

代码 `llm_gptq.compute_gptq_hessians_llm` 的 hook 在所支持的缩放位置先除尺度，再 `_rot_last`，随后累计二阶矩；有可选 token 加权分支，本页不将其默认启用或与论文配置等同。`foldq.fold_site` 接收共同 rotation，进行 fold；`_pack` 根据可用 GPTQ 状态或 RTN 生成编码及 per-row scale。快照包含更多可选分支，不代表全部用于报告实验。

## 6. 从导出到整数乘法的已核对路径

INT4 权重与激活各两个有符号码打包到一个字节。整数累加后

$$\widehat y_{jt}=s_j^W\delta_t\sum_cq^W_{jc}q^A_{ct}+b_j.$$

累加为 INT32，scale/bias 和输出为浮点；没有先展开整个浮点权重再 GEMM。这与 [weight-only 即时反量化](../implementation/quantized-matmul-scaling-execution.md) 是不同路径。

固定代码的具体核对：

- `foldq.py` 的 `fold_site/_pack`：fold-before/after 的字段、调用者提供的共同 rotation、INT4 nibble 或 INT8 编码与 FP32 尺度。
- `per_row_quant_int4_cuda.cu` 的 `per_row_fwht_quant_regs_bf16_to_int4_kernel`：读入 BF16、可选 pre-scale、归一化 64 点 FWHT、可选 post-scale、逐行最大值与裁剪、`rintf` 舍入到 $[-7,7]$，偶通道入低 nibble。对应快速分支要求块宽与输入宽度匹配。
- `dit_int4_rowwise_gemm_fused_cuda.cu`：`int4b_t` 两输入、INT32 累加、`Sm80` TensorOp 模板、`GemmShape<16,8,64>`，epilogue 乘激活/权重尺度并可融合 bias/residual，返回 BF16。
- `PerRowInt4LinearResidualPlugin::enqueue`：先量化，再调用整数 GEMM；FWHT 分支调用融合量化 prologue。**dense rotation 分支实际是 permutation→cuBLAS block rotation→量化的多次调用**，不能因为插件名或旧注释就说所有旋转均单 kernel 融合。

这些是代码静态证据。`Sm80` 模板声明本身不能证明在每张 GPU 上生成同样机器指令；论文明确 H100 路径把四位输入降低为 INT8 算术，故 H100 上的成功率不构成原生 INT4 速度验证。本次未检查最终 SASS 或编译产物，也未遍历所有模型导出与融合插件。

## 7. 选择性 INT8 的收益与混杂因素

保留语言 attention `o_proj` 和 FFN `down_proj` 为 W8A8，其余目标投影 W4A4，两种配置都保持整数 GEMM。这些层把结果写回残差流，且输入缺少可直接融合的 normalization gain；只是选择动机，不是证明它们在所有模型中最敏感。

N1.6/N1.7 的 o/d INT8 配置还改变 dense-rotation 校准 preset，直接对主 W4A4 表作差不能全归因于位宽。论文另给 preset-matched 控制；N1.5、π0.5 的主比较仅改目标位置精度。o 和 down 同时改变，二者单独贡献未分离。

P2 使用与 128 校准观测分离的 32 个中途观测、相同初始噪声。例如 N1.6 的最小动作 cosine 从 0.461 提至 0.850，最大坐标绝对误差的样本中位数从 0.074 降至 0.039。cosine 忽略幅度，均值还可能掩盖少量严重异常；不能只用平均 cosine 作为部署门槛。

## 8. 证据分为四层

| 层次 | 作者结果与必要限制 |
| --- | --- |
| 数值 | 四 checkpoint 留出动作保真度改善；只覆盖 32 观测，不能证明闭环成功率。 |
| 模拟器 | LIBERO P3 每格 800 episodes；preset-matched o/d 比较均未达到 0.05 显著水平，没有确立普遍模拟器增益。SimplerEnv 均分接近可掩盖个别任务巨大下降。 |
| 真实机器人 | N1.7 四任务、每配置 80 次，观测到 W4A4 64/80、o/d INT8 74/80；每任务仅 20 次，任务/会话依赖使合并 Fisher 检验不能当作完整因果证明。 |
| 延迟 | batch 1、RTX 4070 Ti SUPER / Jetson AGX Orin，W4A4 相对浮点 TensorRT 为 1.25–1.52× / 1.20–1.33×；相对 eager 的更大收益包含编译贡献。 |

N1.7 在 Orin 为浮点 TRT 146 ms、W4A4 119 ms、o/d INT8 120 ms；π0.5 却从 172 ms 增至 203 ms。不能将“额外 1 ms”推广到所有 checkpoint。桌面仅一次计时 run，10 次预热后 60 次迭代，没有跨 run 方差。

N1.6 权重存储由浮点 TRT 3802 MB 降为 W4A4 1011 MB，但完整 engine 为 5325→2525 MB；实际 serving 常驻内存 10609→8933 MiB，未物化被替换 PyTorch 权重时则 6349→4673 MiB。文件、权重与驻留不能用相同压缩比例替代。

失败边界也需要保留：探索性 SmolVLA W4A4 为 27.1% 对 BF16 71.1%，Evo-1 为 81.0% 对 91.6%；校准差异限制归因，尚无证据证明选择性 INT8 能恢复它们。不同硬件、emulated port 与作者原始系统的结果不可直接排名。

## 来源身份

- [FoldQuantVLA: Native Low-Bit Quantization of Vision-Language-Action Models via Consistent Folding](https://arxiv.org/abs/2609.24433v1)，arXiv:2609.24433v1；正文、公式、实验协议与限制。
- [cair-vinuni/FoldQuantVLA](https://github.com/cair-vinuni/FoldQuantVLA/tree/4e76b60b4232a33a640e31639ce2a329ae0bebb3)，commit `4e76b60b4232a33a640e31639ce2a329ae0bebb3`，2026-09-28 快照；上述五类路径的定向静态阅读，未执行模型或 GPU。
