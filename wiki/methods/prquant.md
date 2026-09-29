---
title: PRQuant：连续尾块与静态权重残差补偿
type: method
tags:
  - ptq
  - reconstruction
  - data-layout
  - microscaling
sources:
  - raw/papers/2026-09-23/prquant/paper.pdf
updated: 2026-09-28
---

# PRQuant：连续尾块与静态权重残差补偿

PRQuant 将缩放后难量化的 FFN 输入通道排到尾部，为这些通道保存另一份低比特权重残差，并复制对应激活，用更宽的单次 GEMM 计算主项与补偿项。目标是在 MXFP4 下改善精度，同时避免动态 gather 与在线激活残差构造。v1 的实验证据主要是任务准确率和逐层重构误差，尚不足以确认所宣称的低运行开销。

前置机制是 [等价缩放](../theory/diagonal-scaling-equivalent-transform.md) 与 [矩阵乘法执行路径](../implementation/quantized-matmul-scaling-execution.md)；MXFP4 的共享块尺度见 [microscaling 格式](../fundamentals/numeric-formats/fp8-and-mx-data-formats.md)。下文 $Q$ 表示量化后反量化的实数值，不能仅从公式推出实际低比特存储或硬件指令。

## 1. 为什么补偿缩放后的权重

当激活某通道幅度很大，等价缩放能压低它，同时放大对应权重列。W4A4 中权重也只有有限表示能力，误差可能从激活迁移到权重，尤其是少数列。PRQuant 用静态校准寻找这类列，而非逐 token 动态选择离群值。

这与 [激活引导补偿](activation-guided-compensation.md) 的全矩阵可达空间分析不同：PRQuant 限定了表示形式，只给少数原始坐标列增加第二个低比特分量；它也不是学习一个任意低秩分支。

## 2. 缩放和置换的等价关系

设输入 $X\in\mathbb R^{T\times d}$，up/gate 权重为 $C\times d$，down 权重为 $d_o\times C$。FFN 中间量为

$$Z=(XW_{up}^T)\odot\sigma(XW_{gate}^T),\qquad Y=ZW_{down}^T.$$

校准得到 $s_j=(\max_t|Z_{tj}|+\varepsilon)^\alpha$，$D=\operatorname{diag}(s)$。$P$ 为通道置换，令

$$\widetilde Z=ZD^{-1}P,\qquad\widetilde W=W_{down}DP,$$

则 $\widetilde Z\widetilde W^T=ZW_{down}^T$。论文将变换离线融合为

$$W'_{up}=P^TD^{-1}W_{up},\quad W'_{gate}=P^TW_{gate},\quad W'_{down}=W_{down}DP.$$

逐元素非线性与置换相容；缩放只放在 up 分支的线性输出，不能任意移到 gate 非线性内部。若有 bias，需同步变换相应 bias。融合还要求该中间通道由这些相邻模块共同拥有，没有未同步的旁路消费者。（论文方法式，PDF 第 5 页；bias 与消费者条件为整理者补充。）

在 $W_{down}D$ 上先计算列级 MXFP4 量化误差，选 top-$k$ 列，再由 $P$ 排到末尾。论文没有完整规定列误差归约范数、$k$ 与候选 $\alpha$ 的默认值，不能补造可复现配置。置换改变分组成员后量化误差会变，所以后续要重新量化变换后的权重。

## 3. 第二个低比特分量恢复什么

取尾部 $k$ 列为 $\widetilde W_t$，构造

$$R=\widetilde W_t-Q(\widetilde W_t),\qquad
\widehat Y=Q(\widetilde Z)Q(\widetilde W)^T+Q(\widetilde Z_t)Q(R)^T.$$

$R$ 在离线计算、再量化保存；复制的激活是 $\widetilde Z_t$ 本身，不是 $\widetilde Z_t-Q(\widetilde Z_t)$。所以这是权重误差的第二次编码，不能声称显式补回激活误差。

**整理者分解。** 对尾列单独记 $Z_q=Z_t+E_Z$，$W_q=W_t+E_W$，$R=-E_W$，$Q(R)=R+E_R$。组合得到

$$Z_q(W_q+Q(R))^T-Z_tW_t^T
=E_ZW_t^T+Z_qE_R^T.$$

这里 $W_q$ 必须等于实际主体尾列的量化结果；若独立量化尾切片改变了块尺度，则还会留下两种主体编码之差。满足这个前提时，原始权重误差 $E_W$ 被更小残差的量化误差替代，但激活误差 $E_Z$ 仍在。残差是否更好表示，依赖尺度、分组和编码，不是代数恒等式保证精度必然提升。

把两项合并可写为增广矩阵乘

$$\widehat Y\approx Q([\widetilde Z,\widetilde Z_t])\,Q([\widetilde W,R])^T.$$

论文保留约等号是有意义的：拼接后重新分块、选择共享尺度，未必与分别量化一致。仅当块边界、量化参数和舍入保持一致时，独立量化两项与拼接 GEMM 才有相同的数值。$C,k$ 的块对齐或 padding 都要在导出时核查。

## 4. 校准与导出顺序

对每个候选 $\alpha$：统计并缩放→选高误差列→构造置换→重新量化主体及尾列残差→组成增广激活和权重→计算相对原始浮点 down 输出的 NRMSE：

$$\operatorname{NRMSE}=\frac{\|\widehat Y-ZW_{down}^T\|_F}{\|ZW_{down}^T\|_F}.$$

选择 NRMSE 最小者，导出变换与低比特增广权重，再融合生产者。该目标只评价校准 down 输出，不是全模型任务损失，也没有优化整个网络的所有变换。（论文 calibration/export algorithm，PDF 第 6 页。）

MXFP4 的块尺度还可搜索；论文的 E8M0-W 消融改善了误差，但未完整公开其搜索细节。不能把这个变体的结果与默认 PRQuant 混为一个固定配置。

## 5. 连续尾块减少了什么，增加了什么

两种激活供给方式：

- **运行时尾切片**：变换融合后，最后 $k$ 通道天然连续。切片可为 view，但把 $[Z,Z_t]$ 变为 GEMM 所需连续缓冲区仍可能涉及复制、量化或专门读法；“零拷贝切片”不等于“零拷贝拼接”。
- **生产者扩展**：离线复制 up/gate 的最后 $k$ 个输出行，推理直接产生 $C+k$ 通道；down 归约维也扩到 $C+k$。消除了显式复制动作，但更多权重读取、乘加和中间激活仍是真实成本。

忽略 padding/元数据，down 权重与该 GEMM 的算术量增加约 $k/C$；若同时扩展 up/gate，这两层也各增加约 $k/C$。这是教学形状估算，不是整模型耗时比例。低比特残差不会自动免费；MoE 还依赖专家选择及分组执行方式。

排列的数值作用来自重新分组，使少数困难列聚在相同块中；如果逐元素独立量化且同步置换参数，单纯置换本身不会改善误差。它的实现作用则是将分散索引变成规则尾块。这两条作用应分开验证。

## 6. 作者实验证据

正文设置为 Qwen3-4B-Instruct-2507 与 Qwen3-30B-A3B-Instruct-2507，attention 使用 MXFP8，FFN/MoE 使用 MXFP4，只对 down-projection 加入方法。作者列 Ascend A2 NPU、vllm-ascend 0.20.2 RC1 与 OpenCompass；这不等于已经证明整个模型是统一 W4A4 或移植到其他硬件后同样有效。

各方法使用同一 `mix_calib.jsonl` 校准材料；正文未给完整样本数和候选参数，脚注文件名又写为 `mix_cailb.jsonl`，未取得实际配置。本页不据文件名假定具体数据组成。五任务为 ARC-Challenge、ARC-Easy、HellaSwag、PIQA、WinoGrande。

五任务零样本平均分：4B 从默认 MXFP4 的 80.96 提至 82.20，BF16 为 84.07；30B-A3B 从 89.35 提至 89.90，BF16 为 90.62。4B 的 WinoGrande 从 60.38 降至 59.91，平均改善不是所有任务改善。比较中的 QuaRot 只在 down 层使用随机 Hadamard，并非它原论文完整系统配置，不能据此排序算法整体优劣。（原文 end-to-end accuracy 表。）

专家 down 输出误差消融：默认 MXFP4 0.1861，完整方案 0.1678；去缩放 0.1791，去残差 0.1809，去置换 0.1687。它支持缩放/残差是主要数值收益，置换在该消融中只带来小增益；E8M0-W 为 0.1658。没有统计置信区间，不能对很小差异过度解读。

**证据缺口：**全文没有独立吞吐、端到端延迟、峰值内存或 online-copy/offline-copy 计时表，未公开残差宽度和完整配置，归档无可确认作者代码。因而“兼容规则 GEMM”是设计性质，“显著降低延迟”尚不能在本页作为已验证结论。代码阅读或模型复现均未进行。

## 来源身份

- [PRQuant: Permutation Residual Quantization for Low-Overhead Inference](https://arxiv.org/abs/2609.22106v1)，arXiv:2609.22106v1；使用正文、算法和适用层附录，公式回查原始页图。采用归档身份，不从 PDF 水印推断不同版本。
