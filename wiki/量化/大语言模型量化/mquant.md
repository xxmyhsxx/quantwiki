---
title: MQuant：模态静态量化、等价重排与旋转幅度抑制
slug: mquant
sources:
  - raw/papers/quantization/vlm/2025-02-mquant-unleashing-inference-potential/paper.pdf
  - raw/papers/quantization/ptq/2024-04-quarot-outlier-free-4-bit-inference-rotated-llms/paper.pdf
  - raw/papers/quantization/ptq/2024-01-slicegpt-compress-large-language-models/paper.pdf
  - raw/repositories/quantization/2026-07-stiphyjay-mquant-7b3e8196/source/fake_quant/quant_utils.py
  - raw/repositories/quantization/2026-07-stiphyjay-mquant-7b3e8196/source/fake_quant/gptq/qwen2vl_gptq_plus.py
  - raw/repositories/quantization/2026-07-stiphyjay-mquant-7b3e8196/source/exam/quant_qwen2vl.py
  - raw/repositories/quantization/2026-07-stiphyjay-mquant-7b3e8196/source/evaluation/eval.py
  - raw/repositories/quantization/2026-07-stiphyjay-mquant-7b3e8196/source/fake_quant/observer/minmax.py
  - raw/repositories/quantization/2026-07-stiphyjay-mquant-7b3e8196/source/fake_quant/quantizer/uniform.py
updated: 2026-09-14
---

# MQuant：模态静态量化、等价重排与旋转幅度抑制

MQuant 把视觉与文本的分布差异落实为两组静态激活网格，再用保持注意力语义的序列重排降低执行开销。它还处理 Hadamard 旋转在权重中制造的集中峰值。相较 [[mbq|MBQ]] 与 [[vlmq|VLMQ]] 对重构目标赋予重要性，MQuant 重点改变网格与执行组织；这不是“模态重要性”的另一种梯度估计。

本文依据 arXiv:2502.00425v2（2025-08-10，ACM MM 2025 格式），下称“论文”；完整 18 页含附录已研读。代码指官方快照 `7b3e8196e6930b5ce16ec3f058e46687b5d59a6d`，只定向核对 Qwen2-VL 相关路径。论文报告、代码状态和教学推导分别说明，未运行模型实验。

## 1. 对象与问题

对象是视觉编码器、连接器和语言骨干组成的 [[vision-language-model-tokens-and-quantization|视觉语言模型]]。主设置包括视觉侧 W8A8、语言侧 W4A8，以及两侧都 W4A8；不能用一个“W4A8”掩盖组件区别。算法 1 给编码器和连接器配置逐输出通道权重参数、逐张量静态激活参数；语言骨干的视觉与文本激活分别配置静态参数。权重使用 [[gptq|GPTQ]]，不是为每种模态各存一份权重。

图 1 展示的视觉激活范围约为 $[-20,10]$，文本主要在 $\pm0.5$ 内。这是所示统计，不能推广为所有层的固定范围。共用宽网格会让小幅文本值被粗略舍入；为了文本收窄网格又可能裁掉视觉大值。另一方面，大量视觉 token 增加 prefill 成本，逐 token 动态估计尺度也有归约、访存和调度开销。（论文 §2–3、图 1、附录 A.10–A.12）

## 2. 模态分别进行静态校准

采用行 token 矩阵 $X\in\mathbb R^{N\times d}$，$m(n)\in\{v,t\}$ 表示第 $n$ 个位置所属模态。对一个量化点，离线估计 $(s_v,z_v)$、$(s_t,z_t)$，运行时使用

$$
q_{nj}=\operatorname{clip}\!\left(\operatorname{round}(X_{nj}/s_{m(n)})+z_{m(n)},q_{min},q_{max}\right),
\qquad \widehat X_{nj}=s_{m(n)}(q_{nj}-z_{m(n)}).
$$

这就是 MSQ（Modality-Specific Static Quantization）。每组在其模态的 token 和特征维上共享网格，参数仍随层和量化点变化。具体编码定义见 [[uniform-quantization-and-groups|均匀量化与分组]]。**静态是推理时不重新估计范围，运行时仍须执行量化、反量化或相应融合。**（论文 §3.1、算法 1）

教学例子：采用 $[-127,127]$ 对称网格，最大绝对值 20 对应步长 $20/127$；仅最大绝对值 0.5 的组可用 $0.5/127$。两者相差 40 倍，说明为何同一网格可能损害小幅模态。这只说明数值分辨率，不证明哪种模态对任务更重要。

论文使用 256 个校准样本，表 8 在 Qwen2-VL-7B/W4A8 的 128、256、512 样本下变化较小。它支持这组实验对样本数不太敏感，不能保证跨任务、分辨率或数据分布仍适用。分别选网格也不能消除模态内部的长尾与离群值；数据和统计条件见 [[calibration-and-range-selection|校准数据与范围选择]]。

## 3. 重排为什么可以不改变注意力

AIFS（Attention-Invariant Flexible Switching）把交错图文 token 整理成视觉连续、文本连续的布局，以减少按模态执行时反复切片和拼接。关键是保存原来的位置语义，而不仅是移动向量。（论文 §3.1、图 3，附录 A.2/A.4）

令置换 $\pi(i)$ 表示新位置 $i$ 对应的旧位置，$P$ 为置换矩阵。设已经附带原位置编码的查询、键、值为 $Q,K,V$，加性注意力 mask 为 $M$，则应同时使用

$$
Q'=PQ,\quad K'=PK,\quad V'=PV,\qquad M'_{ij}=M_{\pi(i),\pi(j)}.
$$

逐行 softmax 与这种行列置换相容，所以

$$
\operatorname{softmax}\!\left(Q'K'^\top/\sqrt{d_h}+M'\right)V'
=P\operatorname{softmax}\!\left(QK^\top/\sqrt{d_h}+M\right)V.
$$

这里 $d_h$ 是每个 attention head 的维度，输出按 $P^{-1}$ 对应回原位置即可。这是对论文设计的代数展开，说明精确算术下的注意力等价条件，未包含量化误差。

例如原顺序为 `[t0,v1,v2,t3]`，重排为 `[v1,v2,t0,t3]`。原来的 `v1` 能看到 `t0`，即使 `t0` 现在排在后面；`t0` 仍不能看到视觉位置。重排后的因果关系是 $\pi(j)\le\pi(i)$，不是新下标的 $j\le i$。直接继续使用普通下三角 causal mask 会改变模型。

RoPE 必须携带原 position IDs；多维位置编码也须保留原坐标。padding 的无效性、KV cache 的顺序和生成阶段位置需一起维护。该等价关系没有减少 attention 的 token 数或复杂度。论文给出批处理和多轮结果，但任意布局仍需实现对应置换，不能仅凭单段视觉区间推导保证所有场景。

实现时应在本次输入预处理阶段建立排列与模态边界，并让后续层消费一致布局。普通 FlashAttention 的 `is_causal=true` 不能自动表示新 mask；论文 §4.3 通过修改后的 mask/区间元数据支持该路径。重排和适配的成本是否低于节省的操作，要以实际后端测量。外维尺度为什么便于整数 GEMM，见 [[quantized-matmul-scaling-execution|量化矩阵乘法的缩放与执行路径]]。

## 4. 旋转为什么还需要幅度抑制

MQuant 使用 QuaRot 的全局旋转与在线 Hadamard 机制。两者的区别、归一化与残差条件见 [[orthogonal-rotation-and-hadamard-quantization|正交旋转机制]]。设非线性后的激活为 $A\in\mathbb R^{N\times d}$，down projection 为 $B\in\mathbb R^{d\times o}$，归一化 Hadamard 为 $H$，则

$$
AB=(AH)(H^\top B).
$$

在线执行 $AH$，权重侧变换可预先计算。标准 Walsh–Hadamard 的全正首行将同号均值集中到首个变换通道：$(HB)_{0j}=\sqrt d\,\overline B_j$。当它超过其余元素幅度时，会撑大权重网格，导致其他通道被粗量化。正交变换保持整体能量，却不保证峰值下降。（论文 §3.2、式 6–9、附录 A.3；QuaRot §4 Stage 1a/1b）

RMS 在这里指 Rotation Magnitude Suppression，**不是 RMSNorm**。将 $A'=AH$、$B'=H^\top B$ 的首个输入通道拆开：

$$
A'B'=A'_{:,1:}B'_{1:,:}+A'_{:,0:1}B'_{0:1,:}.
$$

主支路的权重网格不再被首通道支配；集中通道单独处理，最后相加。拆分在量化前是恒等式，量化后误差取决于各支路自己的网格。论文图 5 描述主支路 W4A8 GEMM 与集中通道 W4A8 GEMV、浮点输出相加，而非简单删掉首通道或笼统保留所有离群值为高精度。

算法 2 先依据式 9 标记可能受影响的层，再旋转、拆分、量化。论文表 2 在所测模型中观察到视觉侧受影响更普遍，但不是所有架构都必然如此。额外支路只有一个输入通道不代表实际耗时恒为 $1/d$：启动、访存和相加成本也须计入。

## 5. 从算法到可执行模型

完整流程应覆盖以下依赖，而不是简单串联几个模块名称。（论文算法 1/2、附录 A.13/A.14）

1. 固定模型组件、量化点和校准输入，记录图文布局、分辨率、有效 token、位宽及粒度。
2. 处理归一化与旋转的等价变换。LayerNorm 的中心化、增益、偏置需按图移动；Pre-LN 与 Post-LN 的残差位置不同。SliceGPT §3.1–3.2 给出计算等价和 LN 转换依据，MQuant 图 10/11 展开其视觉结构，不能只替换归一化类名。
3. 在所需位置加入在线 Hadamard，为受影响权重配置 RMS 拆分，再确定权重和各模态激活网格。GPTQ 使用真实校准输入统计；参数对应变换后的模型。
4. 保存量化权重、scale/zero-point、旋转与拆分信息；对每次输入建立 AIFS 排列、位置和 mask，使内核使用与校准相同的网格与语义。
5. 分别核对浮点图等价、量化质量以及真实执行性能。假量化中的 FP32/FP16 计算不能作为低比特内核速度证据。

**固定代码的实际边界。**`fake_quant/quant_utils.py` 的 `ActQuantWrapper.split_weights/forward` 确实拆出首通道，但前向只对其余激活调用量化器，两个 `Linear` 均以浮点计算；`fake_quant/gptq/qwen2vl_gptq_plus.py` 的视觉拆分分支只将 `mlp.fc2.L2` 列入 GPTQ。该路径不等于图 5 的两支 W4A8 内核。

此外，`exam/quant_qwen2vl.py` 为各层配置单个静态 `ActQuantizer`，所读 observer 路径没有展示语言层双模态网格路由；`--aifs` 只找到参数定义，未找到消费该参数的执行链。因此当前快照支持部分旋转/静态假量化理解，尚不能据此确认 MSQ+AIFS 性能链完整可复现。不能用论文算法替代码补齐缺失功能，也不能据此否定作者另有实验实现。

校准协议也要单独恢复：论文 §4 称从相应 benchmark 训练集取样；固定 Qwen2-VL 入口把同一 `dataset` 对象传给 `calib_qwen2vl_plus` 与评测，前者按间隔选取其中输入，`evaluation/eval.py` 的 `eval_dataset` 遍历全部数据。若照该入口复现，校准样本会参与后续评测，不能作为独立留出测试；这不等于已经证明论文私有实验使用同一协议。

## 6. 哪些实验解释了组件作用

以下均为作者报告。表 7 使用 Qwen2-VL-7B、视觉与语言侧 W4A8，在 GPTQ+Hadamard 静态基线上逐步加入组件：

| 设置 | TextVQA | DocVQA | OCRBench | MME |
|---|---:|---:|---:|---:|
| BF16 | 84.43 | 93.87 | 842 | 2319 |
| 基础静态量化 | 71.44 | 83.96 | 670 | 1911 |
| 加 MSQ+AIFS | 78.95 | 87.55 | 721 | 2095 |
| 再加 RMS | 84.32 | 93.58 | 824 | 2255 |

模态处理与旋转峰值处理各恢复一部分精度。表 7 把 MSQ 与 AIFS 合并，不能从这一表单独估计重排贡献；其三个量化设置延迟接近，也不能把总加速归于最后加入的 RMS。

表 10 更直接分离网格和布局代价。其口径为 prefill 线性层，输入为三图交错文本、图像分辨率 $2240\times2240$、50 个文本 token，NVIDIA RTX 6000 Ada；不是所有模型或完整服务 TTFT：

| 激活设置；权重均逐输出通道 W4 | 延迟（秒） | TextVQA | 对机制的意义 |
|---|---:|---:|---|
| 逐 token 动态 A8 | 1.253 | 84.32 | 运行时估计细粒度尺度 |
| 单一逐张量静态 A8 | 1.016 | 40.20 | 速度低开销，但共用范围损害精度 |
| MSQ | 1.085 | 84.32 | 分模态网格恢复质量，布局处理仍有成本 |
| MSQ+AIFS | 1.017 | 84.32 | 维持该质量并接近单网格静态延迟 |

由原始时间计算，AIFS 相比 MSQ 缩短约 6.3%，整套相比动态基线缩短约 18.8%；后者也可表述为约 1.23 倍吞吐，不能把这两个百分比混为一谈。表 12 的 decode 报告从动态 16.4 到 MSQ+AIFS 13.06，再到加定制 GEMV 的 8.2，约两倍收益包含内核贡献；该表未清楚标出 decode 数值单位，不用于绝对时间预算。

## 7. 优势、局限与方法选择

**优势在于把数值方案与执行布局一起设计。**分模态静态网格在论文条件下缓解共用范围问题，AIFS 使这种网格更易高效执行；RMS 则给出旋转可能失败的具体机制与补救。对于视觉 token 多、希望降低激活动态处理开销的场景，这比只看权重压缩率更有参考价值。

**局限首先是条件依赖。**静态网格依赖校准分布，不能据少量样本数消融证明分布外稳健；重排要求 mask、位置与 cache 一致，适配成本由后端决定；RMS 针对的是特定集中通道，不是通用离群值消除保证。表 3 仍有明显质量损失，例如 InternVL2-8B 的 OCRBench 从 794 到 725（两侧 W4A8），不宜统称所有配置“低于 1% 损失”。

论文主对照对 RTN、SmoothQuant、QuaRot 也施加静态激活设置，比较结果不能推广为击败它们所有原生配置。速度测试集中在指定 GPU、形状和内核，未证明端云切分、网络通信或目标端设备收益。视频在附录 A.16 仍属未来方向讨论。公开快照的缺口进一步限制直接复现。

与 [[mbq-vlmq-comparison|MBQ、VLMQ 的重要性机制]] 对照，后续比较应区分“优化误差时重视谁”“推理时共用哪个网格”“怎样组织数据执行”。[[q-vlm|Q-VLM]] 还改变联合校准的层范围，其熵代理与视觉端优化关注搜索成本；其公开实现也说明，W4A4 标签不自动代表原生 INT4 GEMM。上述维度可能组合，但必须在同一对象、数据与后端下验证，不能从论文各自领先的分数推导组合必然更好。
