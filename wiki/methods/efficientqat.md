---
title: EfficientQAT：逐块全参数训练与整网尺度微调
type: method
tags:
  - qat
  - llm
  - weight-quantization
  - reconstruction
sources:
  - raw/papers/2026-09-22/efficientqat/paper.pdf
  - raw/repositories/2026-09-22/efficientqat/source/quantize/block_ap.py
  - raw/repositories/2026-09-22/efficientqat/source/quantize/quantizer.py
  - raw/repositories/2026-09-22/efficientqat/source/quantize/int_linear_fake.py
  - raw/repositories/2026-09-22/efficientqat/source/quantize/int_linear_real.py
  - raw/repositories/2026-09-22/efficientqat/source/quantize/utils.py
  - raw/repositories/2026-09-22/efficientqat/source/main_e2e_qp.py
  - raw/repositories/2026-09-22/efficientqat/source/datautils_e2e.py
updated: 2026-09-22
---

# EfficientQAT：逐块全参数训练与整网尺度微调

EfficientQAT 把量化训练拆成两阶段：Block-AP 每次训练一个 Transformer block 的权重和量化参数，以较大的局部自由度获得低比特初始化；E2E-QP 冻结整数权重与零点，只端到端更新分组尺度，让不同块共同适应最终目标。核心取舍是把大量可训练变量限制在局部，把整网优化限制在少量连续变量。

本页依据 arXiv:2407.11062v3，覆盖方法、实验与附录，并定向核对官方代码 `39175493b2d14617d342a0a7956875e6ac16221b`。论文主实验是 Transformer block 线性层的 **weight-only** 量化；Embedding、归一化与输出分类头保留高精度。没有复现模型训练或验证 GPU 性能。

## 1. 为什么需要两种不同的训练自由度

整网 QAT 若维护所有浮点潜在权重、梯度与优化器状态，内存成本很高。只调整量化尺度又可能不足以救回很差的初始整数编码。EfficientQAT 的处理顺序是先逐块改变权重与网格，再用最终模型目标联合调整网格尺度。

局部阶段仍存在误差累积与代理目标的限制，不能从块 MSE 小推出整网任务损失小；第二阶段因此有独立作用。它与 [LLM-QAT](llm-qat.md) 的生成数据、教师输出蒸馏路线不同，也没有将主实验扩展到低比特激活或 KV cache。

## 2. Block-AP：在量化前向下重构整个块

对一个共享量化参数的权重组，令潜在权重为 $w$、尺度为 $s>0$、整数零点为 $z$，位宽为 $b$。论文式 (1)–(2) 的标量写法为

$$
q=\operatorname{clip}\big(\operatorname{round}(w/s)+z,0,2^b-1\big),
\qquad \widehat w=(q-z)s.
$$

论文将潜在权重描述为 FP16/BF16；训练中的参数、算子和累加 dtype 仍须分别检查。固定代码在块训练前执行 `qlayer.float()`，前向使用 AMP，不能据论文的模型存储精度宣称训练全部为 FP16。

以第 $i$ 个块为例，官方 `quantize/block_ap.py:block_ap` 分别推进浮点前缀与已量化前缀。整理成目标为

$$
\min_{W_i,s_i,z_i}
\operatorname{mean}\left[
\left(\widehat F_i(x_i^q;W_i,s_i,z_i)-F_i(x_i^{fp};W_i^{fp})\right)^2
\right].
$$

$x_i^q$ 来自已经量化的前面各块，$x_i^{fp}$ 来自浮点前缀，目标输出在优化当前块前缓存；代码的 `MSELoss()` 对输出张量元素取均值。两条输入路径从第一块的相同输入开始，随后分别更新。当前块可以补偿先前量化带来的输入偏差，但不会回头修改已经完成的块。这是非对称输入重构，与量化网格是否有非零 zero-point 是两件事，机制见 [层输出重构](../theory/layer-reconstruction-second-order-compensation.md#8-为什么还需要考虑多个层的联合重构)。

“All Parameters”需要与具体实现对应：论文概括为 $W,s,z$；固定代码的 `weight_parameters` 选择块中名字含 `weight` 且不是尺度/零点的参数，因此也会包括归一化权重。归一化权重可以训练但并未被低比特量化，线性层 bias 在包装类中作为 buffer 保存。不能把名称解释成任意模型中所有参数无条件参与优化。

块内前向的数值规则也已核对到 `UniformAffineQuantizer.fake_quant`：沿输入维度分组，用每组 MinMax 初始化；对 $w/s$ 舍入使用 STE；潜在零点先做 STE 舍入，再用直通裁剪限制到编码范围；尺度用直通裁剪限制在 $[10^{-4},10^4]$；整数编码的最终裁剪使用普通 `clamp`。这些节点的反向规则并不相同。

### 步长梯度与零点梯度

忽略边界点、将有效尺度视作 $s$，在编码未饱和时，STE 给出

$$
\frac{\widetilde\partial\widehat w}{\partial w}=1,
\quad
\frac{\widetilde\partial\widehat w}{\partial s}
=\operatorname{round}(w/s)-w/s,
\quad
\frac{\widetilde\partial\widehat w}{\partial z}=0.
$$

若编码饱和到端点 $q_e\in\{0,2^b-1\}$，则对应结果为 $0$、$q_e-z$、$-s$。内部零点梯度为零，是编码中的加零点与反量化中的减零点相消；饱和时编码分支被截断，减零点的路径仍存在。这里 $z$ 以整数编码单位定义。

**原文差异：** 附录 B 将饱和区的零点导数写为 $-1$，但由式 (1)–(2) 与上述参数化应得到 $-s$；固定代码也通过最后的乘尺度产生 $-s$。本页明确区分原文写法与链式法则推导，不直接照抄附录作为实现规范。步长部分与 [LSQ](lsq.md) 有相同的代理项，但并不意味着使用了 LSQ 的专门梯度缩放。

## 3. E2E-QP：固定整数编码，仍然能够改变权重

完成逐块训练后，保存低比特 $q,z$，默认只训练每组尺度。此时每次前向只执行

$$
\widehat w_j=(q_j-z_g)s_g,\qquad j\in g,
$$

不再从浮点潜在权重重新舍入得到 $q_j$。所以

$$
\frac{\partial\widehat w_j}{\partial s_g}=q_j-z_g,
\qquad
\frac{\partial L}{\partial s_g}
=\sum_{j\in g}\frac{\partial L}{\partial\widehat w_j}(q_j-z_g).
$$

第一式是固定整数值下的精确导数，第二式是共享尺度的链式法则展开；不是 LSQ 的舍入代理。例：$q=(0,1,3)$、$z=1$ 时，把 $s$ 从 0.2 改成 0.25，会把权重从 $(-0.2,0,0.4)$ 改为 $(-0.25,0,0.5)$。编码没有变化，组内权重只能沿这一共同缩放方向调整，不能独立改变每个值或重新分配整数索引。

固定代码的 `main_e2e_qp.py` 先冻结模型，再为量化线性层启用 `scales.requires_grad`；`QuantLinear.forward` 解包整数权重与零点、乘尺度，然后调用浮点 `torch.matmul`。代码没有在这一路径重新放置 Block-AP 的尺度裁剪。**整数存储、少量可训练参数和实际低比特乘法是三种不同性质**，此训练前向并不证明运行了 INT2 GEMV。

论文 §3.3 只简要描述目标数据集。固定代码补足了常规预训练数据分支：`datautils_e2e.py:group_texts` 将 `input_ids` 复制为 `labels`，交给 `AutoModelForCausalLM` 与 Trainer 的语言模型损失接口，使用 next-token 监督；该路径没有传入教师 logits。指令数据分支则按配置屏蔽输入部分和 padding。这里核对的是输入/标签与训练入口，没有运行 Transformers 损失或依赖版本的完整调用链。

训练尺度只减少参数梯度及优化器状态，整网仍需反向传播中间激活。激活长度、batch、checkpointing、反量化临时量和未量化模块都影响训练内存，不能按可训练参数比例直接推算整网内存下降倍数。

## 4. 训练顺序与主实验设置

1. 确定位宽与组大小，准备浮点模型和训练/验证输入；用每组 MinMax 初始化尺度与零点。
2. 按模型顺序取得当前块的浮点目标，开启权重 fake quant，在量化前缀输入上更新当前块权重与量化参数。
3. 固化当前块量化结果，用它推进下一块的量化输入；浮点目标路径独立推进。
4. 导出低比特整数权重、零点和高精度尺度；加载整网并冻结整数部分。
5. 用整网语言模型或任务训练目标更新尺度，分别检查最终任务质量、存储格式与目标后端执行。

§4.1 主实验中，Block-AP 使用 4,096 条 RedPajama 样本，长度 2,048，batch 2，每块 2 epochs；量化参数学习率 $10^{-4}$，权重学习率在 2 bit 时为 $2\times10^{-5}$，3/4 bit 为 $10^{-5}$。E2E-QP 使用 4,096 条样本、长度 4,096、batch 32、1 epoch；2 bit 的尺度学习率 $2\times10^{-5}$，3 bit 为 $10^{-5}$。这些是作者的实验配置，不能仅凭同样的样本条数就判断 token 预算相同。

## 5. 消融分别支持了什么

表 4 的 Llama-2-7B、W2g64 消融中，Avg. PPL 为 WikiText2 与 C4 PPL 的均值；Avg. Accuracy 为五项零样本常识任务均值。

| Block-AP | E2E-QP | Avg. PPL ↓ | Avg. Accuracy ↑ |
|---|---|---:|---:|
| 无 | 无 | 453.49 | 40.69 |
| 有 | 无 | 8.53 | 58.99 |
| 无 | 有 | 9.33 | 55.71 |
| 有 | 有 | 7.68 | 60.14 |

该配置中两阶段结合最好，局部训练改善初始化，整网尺度训练进一步修复任务表现。不能由这一表格断言任意量化初始化都必须两阶段。

表 5 进一步固定为只有 Block-AP：只训练 $(s,z)$ 得到 10.26 PPL / 55.20 Accuracy，只训练 $W$ 为 14.32 / 46.50，联合 $(s,z,W)$ 为 8.53 / 58.99。它支持“权重与网格共同适应”这一解释，不能把全部收益归因于训练步长，也不能认为更多任意参数必然更好。

表 6 比较 E2E-QP 变量：只训 $s$、只训 $z$、同时训 $(s,z)$，对应平均准确率为 60.14、60.08、60.18，差异很小；但后两者要保留浮点零点，平均量化位数由约 2.28 增至 2.50。默认只训尺度，是质量与格式开销的取舍。

数据量也是比较条件。附录 H 表 15 在 Llama-2-7B、W2g128、128 条样本下报告 EfficientQAT 的 C4 PPL 为 **8.95**，OmniQuant 为 15.02；邻近正文把前者写成 8.02，与表不符，本页采用表值并保留差异。相同样本条数也未控制训练轮数、序列长度或优化变量，不能视为完全等算力比较。

## 6. 质量、训练成本、表示成本与执行性能

### 模型质量不等于无损

表 1 的 Llama-2-70B 五任务均值：FP16 为 72.41，EfficientQAT W3g128 为 71.76，W2g64 为 69.48。任务为 WinoGrande、PIQA、HellaSwag、ARC-Easy、ARC-Challenge，附录 I 指明 lm-evaluation-harness v0.4.2 的 `acc`，不是 `acc_norm`。2 bit 仍有明显质量差距；同表 QuIP# 的 2 bit 为 70.91，不能写成全面优于向量量化。不同表示的码本与元数据预算也需另算。

### 训练成本要按阶段解释

表 7 在单张 A100-80GB 上报告 Llama-2-70B：Block-AP 26.6 小时、29.9 GB；E2E-QP 约 14.3 小时，2 bit 内存 34.2 GB；总时间 40.9 小时。7B 对应总时间 4.8 小时。分阶段训练内存不是推理峰值，也不是整个模型始终以相同形式驻留 GPU 的证明。

表 9 的跨方法 GPU 小时来自多份工作，附录 D 对 LLM-QAT 的时间还引用了 BitDistiller。本页不把它当作在同一机器、数据与实现上重新实测的公平加速比。

### 每组元数据改变平均位宽

按附录 E 的紧凑表示，一个组的 $g$ 个权重各占 $b$ bit，尺度占 16 bit、整数零点占 $b$ bit，则量化线性层的理想平均位宽为

$$
b_{\mathrm{eff}}=b+\frac{16+b}{g}.
$$

W2g64 为 2.28125 bit/weight；W2g128 为 2.140625。表 11 与该计算一致，表 12 将后一项写成 2.10，不能据此改变公式。非量化层、对齐、打包空位与辅助索引尚未包含；表 11 的完整 70B 模型 W2g64 大小为 19.16 GiB，不能直接用名义 2 bit 乘参数总数代替。

固定代码还展示了理想格式与实际容器的区别：`QuantLinear.pack` 用每个 int32 存放 $\lfloor32/b\rfloor$ 个编码，3 bit 时每个字有 2 bit 空位；它还保存 `g_idx` buffer。故该代码 checkpoint 的字节数并不必然等于附录紧凑公式。按参数数量计算，$g=64$ 的尺度占比是 $1/64\approx1.56\%$，尺度与零点合计是 $2/64\approx3.13\%$；§3.2 将两者合称约 1.6% 的文字不够准确。

### 局部 GEMV 不代表整个服务

附录 C 表 10 比较单张 A100-80GB 上 FP16 PyTorch 与 INT2 BitBLAS 的 GEMV。矩阵 $(11008,4096)$ 一项为 61 μs 与 21 μs，约 2.90 倍；这是给定形状的矩阵向量计算，不是端到端生成速度、吞吐或训练加速。本页不将概括的 2.9–4.4 倍套到所有模型和 batch。

使用标准均匀量化也不自动获得任意后端兼容性。导出需核对零点编码、分组方向、布局与 pack/repack；固定代码 Block-AP 的导出零点路径采用 `.round()`，没有复用 fake quant 中完整的零点裁剪，而 E2E 前向解包后仍用浮点 matmul。部署一致性与快速内核应分别验证，参见 [量化部署后端](../implementation/quantized-llm-deployment-backends.md)。

## 7. 指令与多模态扩展改变了哪些条件

§4.2 在 Llama-1 上使用 Alpaca 与 5-shot MMLU；QLoRA 对照包含合并适配器后再用 GPTQ 量化的步骤。它比较的是最终压缩产物及特定微调协议，不能推广为 QLoRA 一般不能在量化基座上推理。

附录 G 将预先经 Block-AP 量化的 Vicuna 放入 LLaVA-1.5 流程：先冻结语言模型训练 projector，再联合训练 projector 与语言模型的 E2E-QP 尺度。它不证明视觉编码器也已低比特化。表 14 使用 MMBench、MME、MM-Vet、ScienceQA，并把 MME 感知分数归一到 100；2 bit 13B 的均值为 59.9，高于 QLoRA 后 Block-AP 的 58.0，但 4 bit 13B 的 62.0 低于对应 62.4。多模态结论必须保留这个反例，不能沿用正文“各配置都更好”的概括。

## 来源身份

- [EfficientQAT: Efficient Quantization-Aware Training for Large Language Models](https://arxiv.org/abs/2407.11062v3)，arXiv:2407.11062v3；§3–4、表 1/4–7、附录 B–I。
- [OpenGVLab/EfficientQAT 固定代码](https://github.com/OpenGVLab/EfficientQAT/tree/39175493b2d14617d342a0a7956875e6ac16221b)，commit `39175493b2d14617d342a0a7956875e6ac16221b`，2026-09-22 获取；本页所列训练、量化、标签与打包代码经过阅读，未执行模型或 GPU 内核。论文差异、代码观察、整理者推导与作者实测在正文分别标注。
