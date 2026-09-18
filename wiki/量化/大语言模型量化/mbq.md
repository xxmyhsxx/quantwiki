---
title: MBQ：模态平衡的视觉语言模型量化
slug: mbq
sources:
  - raw/papers/quantization/vlm/2024-12-mbq-modality-balanced-quantization-for-large-vision-language-models/paper.pdf
  - raw/papers/quantization/vlm/2024-12-mbq-modality-balanced-quantization-for-large-vision-language-models/source.eprint
  - raw/repositories/quantization/2026-07-thu-nics-mbq-a4d460df/source/qmllm/methods/mbq/quantize/pre_quant.py
  - raw/repositories/quantization/2026-07-thu-nics-mbq-a4d460df/source/qmllm/methods/mbq/quantize/auto_scale.py
  - raw/repositories/quantization/2026-07-thu-nics-mbq-a4d460df/source/qmllm/methods/mbq/quantize/auto_scale_wa.py
  - raw/repositories/quantization/2026-07-thu-nics-mbq-a4d460df/source/qmllm/methods/mbq/entry.py
  - raw/repositories/quantization/2026-07-thu-nics-mbq-a4d460df/source/qmllm/quantization/quant_funcs.py
  - raw/repositories/quantization/2026-07-thu-nics-mbq-a4d460df/source/qmllm/models/qwen2_vl/qwen2_vl.py
  - raw/repositories/quantization/2026-07-thu-nics-mbq-a4d460df/source/configs/llava_onevision/MBQ_search/7b_weight_only.yaml
paper_version: arXiv:2412.19509v2
code_commit: a4d460dfb4b1c07b5d1f3ddda6e86d1c90d6e7f1
verification: 全文与附录研读；定向官方代码核对；未运行模型实验
updated: 2026-09-14
---

# MBQ：模态平衡的视觉语言模型量化

MBQ 在校准阶段用梯度估计视觉和回答文本的敏感性，再用模态加权的输出误差选择通道缩放。它改变的是“哪一种缩放更值得保留”，最终仍可导出常规低比特权重。它没有按视觉/文本 token 分别存储两套权重，也没有为每个 token 动态分配位宽。

本页依据 MBQ v2（2025-03-21）的全部 21 页及上述固定官方代码快照。论文提出的机制、代码细节、作者报告和我们的推导分开说明。先了解 [[vision-language-model-tokens-and-quantization|视觉语言模型中的 token 与量化对象]]，有助于理解这里的“模态”究竟标记什么。

## 1. 对象、问题与实验范围

研究对象是 LLaVA-OneVision、InternVL2、Qwen2-VL 的视觉语言生成模型。主实验集中于语言骨干中的线性层；视觉编码器另有量化消融。视觉位置与文本位置进入共享语言层，但在校准样本中的数量、激活分布和任务梯度可能不同。对所有位置直接累积输出误差，可能让 token 多的部分主导缩放选择。（§1、§3、补充 §7）

图 1 的一个具体例子是 LLaVA-OneVision-7B 第 13 个块、COCO 图像描述样本：视觉与文本位置的平均绝对梯度约为 $1.7\times10^{-8}$ 和 $2.1\times10^{-7}$。它说明该样本、该层和该损失下存在差异，不能推出所有视觉 token 都不重要，也不能把平均梯度比直接读成实际任务损失的倍数。

主设置有两种，不能混写：

| 设置 | 权重 | 激活 | 需要保留的条件 |
|---|---|---|---|
| W3A16 | 3 bit，非对称，group size 128 | 16 bit | weight-only；分组轴及元数据影响存储与内核 |
| W4A8 | 4 bit，对称，per-output-channel | 8 bit，对称，per-token | 两侧都量化，不能等同于 W4A16 |

补充表 11 另有 W4A16、W8A8。校准使用 128 个 ShareGPT4V 改进的 COCO 图像描述样本，按模型对话模板处理。主比较中的 AWQ/SmoothQuant 默认 Pile 校准与 MBQ 的图文校准不是相同条件；表 4 才进一步区分校准数据、加权与损失类型。（§4.1、表 2–4、11）

## 2. 缩放怎样改变量化误差

采用列向量输入约定：$X\in\mathbb R^{d\times N}$，$W\in\mathbb R^{o\times d}$，$Y=WX$；$N$ 汇集 token 位置，$d$ 为输入通道。正对角矩阵 $E=\operatorname{diag}(e_1,\ldots,e_d)$ 满足

$$
WX=(WE)(E^{-1}X).
$$

量化后，weight-only 输出为 $\widehat Y=Q_W(WE)E^{-1}X$；W+A 输出为 $\widehat Y=Q_W(WE)Q_X(E^{-1}X)$。这里 $Q$ 返回反量化后的近似实数值，不是整数编码。量化打破等价，因而可以搜索 $E$ 来改变误差。输入通道的含义见 [[linear-layer-input-channel|线性层与输入通道]]；可融合位置与非线性限制见 [[diagonal-scaling-equivalent-transform|对角缩放与等价变换]]。

这与 [[awq|AWQ]] 的通道缩放、[[smoothquant|SmoothQuant]] 的两侧难度迁移共享代数基础。MBQ 的核心增量是用模态敏感性重新评价候选缩放，不是发现了另一条矩阵恒等式。（§3.1–3.2）

论文式 (1)–(2) 将 $Z=\min(W)$ 写作偏移，采用 $(W-Z)/S$ 的记法；它不是通常意义上的整数 zero-point。固定代码 `quant_funcs.py:pseudo_quantize_tensor` 实际计算 $z=\operatorname{clip}(-\operatorname{round}(\min(W)/s))$，随后对整数编码裁剪。不能把论文的实数偏移与代码的整数 $z$ 混用，具体区别见 [[uniform-quantization-and-groups|均匀量化与分组]]。

## 3. 梯度为什么进入目标，推导能支持到哪里

令 $L$ 是校准图文序列的回答预测交叉熵，$g=\partial L/\partial Y$，$\Delta Y=\widehat Y-Y$。一阶展开是

$$
L(Y+\Delta Y)-L(Y)\approx\langle g,\Delta Y\rangle.
$$

这提示同样大小的重构误差，在不同梯度方向上可能有不同影响。MBQ 按模态聚合绝对梯度，用其平均大小作为敏感性代理，再加权重构误差。使用反向传播来测敏感性不等于用优化器训练权重；这仍是 [[post-training-and-quantization-aware-training|PTQ]]。

但要保留两层限制：第一，小扰动泰勒展开不是低位宽下的精度保证；第二，平均绝对梯度不是最大绝对梯度。论文式 (10)–(12) 从三角不等式进一步替换成模态平均梯度的“上界”，没有给出足够的附加条件。

例如 $g=(2,0)$、$\Delta=(1,0)$ 时，$|g^\mathsf T\Delta|=2$，而 $\operatorname{mean}(|g|)\|\Delta\|_1=1$，不能作为上界。一般成立的是

$$
|g^\mathsf T\Delta|\le\sum_i|g_i\Delta_i|
\le\|g\|_\infty\|\Delta\|_1.
$$

因此模态均值加权在这里是有梯度动机、由实验支持的代理目标；不能称为已经证明的最优模态分配。

## 4. 实际权重、归一化与尺度搜索

以下是代码 commit a4d460df 的可核对实现，不能自动视作每张论文表格的精确运行版本。

**先分清 mask。** Qwen2-VL 接入的 `caption_mask` 来自 `labels != -100`，`vision_mask` 来自展开后的图像 token 标记；回答之前的标签被忽略，交叉熵另有因果移位。因此“文本组”在这个实现里主要指回答位置，不是全部提示文本，也不是所有非图像位置。mask 的位置还必须与插入视觉 embedding 后的序列对齐。（`qwen2_vl.py`，回答标签构建、`forward`、约 324–335 行）

**再聚合梯度。** `GradCacheHook` 对单一样本、单一模态的输出梯度取绝对值均值，再跨样本平均。`run_mbq` 用 attention 的输出投影与 MLP 的下投影梯度比，分别服务对应缩放组；不是每个线性层都独立学习任意模态权重。视觉/回答比 $r_l$ 还被下限化为

$$
\rho_l=\max\left(r_l,\operatorname{median}_{k\in\mathcal G}r_k\right),
\quad r_l=\bar g_{v,l}/\bar g_{a,l},
$$

其中 $\mathcal G$ 分 attention 和 MLP 两组。此中位数下限属于固定实现细节，不来自前述泰勒上界。（`pre_quant.py:GradCacheHook`、`run_mbq`，约 265–305、352–360 行）

**MAE 和 MSE 不只差一个平方。** 设视觉与回答 mask 选中的标量元素数是 $K_v,K_a$，误差张量为 $D=\widehat Y-Y$。默认 MAE 分支计算

$$
J_{\rm MAE}(E)=\frac{\sum_{i\in a}|D_i|+\rho_l\sum_{i\in v}|D_i|}{K_a+K_v}.
$$

所以视觉 token 多仍会增加总贡献。例如 $K_a=10,K_v=100,\rho=0.1$、各元素误差都为 1 时，两组加权分子都是 10；这不是两组各取均值后再乘 0.1。代码 MSE 分支则是 $\operatorname{mean}_a(D^2)+\rho_l\operatorname{mean}_v(D^2)$。论文的 MAE/MSE 消融在解释时必须留意这类归一化实现差别，不能未经运行确认就把全部收益归于范数变化。（`auto_scale.py` 约 145–179 行；WA 分支也采用对应形式）

**搜索是有限候选。** `get_act_scale` 返回输入每通道的平均绝对值 $m_c$。每组检查 $t\in\{0,0.05,\ldots,0.95\}$ 共 20 个候选：

$$
u_c=\max(m_c^t,10^{-4}),\qquad
e_c=u_c/\sqrt{\max_j u_j\min_j u_j}.
$$

每个候选模拟量化、计算输出与加权损失，再恢复参数；保留最低损失候选。weight-only 路径用 $Q(WE)E^{-1}$ 与原输入计算，WA 路径分别量化缩放后的权重与输入。它没有遍历所有正对角矩阵，也没有通过梯度下降学习 $E$。（`auto_scale.py` 约 110–196 行，`auto_scale_wa.py` 约 98–143 行）

## 5. 从校准到部署的流程

1. 固定模型、图像预处理、对话模板、位宽与分组；构造图像、提示和回答，生成有效位置、视觉位置与回答标签。
2. 在原模型上计算回答交叉熵并反传，收集分模态梯度统计，形成缩放组使用的比值。固定实现以单样本小批次处理并累积；它不是在线推理步骤。
3. 逐块缓存线性层输入，按上节的候选与目标搜索缩放。所读 LLaVA-OneVision W3 配置是 `reweight: true`、`loss_mode: mae`、`distort: false`。仓库另有使用受前层量化影响输入的分支，本轮未把所有分支行为都列为已验证。
4. 在合法的归一化/线性相邻位置应用缩放，保存尺度结果；重新加载时要与原权重版本匹配。
5. 应用量化器得到模拟量化模型，或交给匹配的打包与内核执行路径。`entry.py` 所读路径主要是尺度缓存与 pseudo quant；这条路径本身不是论文 W3 kernel 的端到端部署证明。

缩放需保持共享分支、归一化、残差和非线性边界的计算等价；模态 mask 则决定哪些位置参与统计与重构。

## 6. 主结果与消融怎样读

以下均为作者报告。MBQ 的优势在于将模态敏感性纳入已有缩放搜索，但收益依赖模型、位宽和基线。表 2 中 LLaVA-OneVision-7B W3A16 的平均分为 MBQ 65.3、GPTQ 64.1、FP 67.5；表 3 中 InternVL2-26B W4A8 的 MBQ 与 RTN 都是 72.7，说明加权并不总能带来额外收益。不同校准协议下的平均分不足以单独归因。

表 4 对 LLaVA-OneVision-7B 区分校准与目标，在同一表内更适合判断增量：

| 路径 | 校准及目标 | MMMU | SEED |
|---|---|---:|---:|
| W3A16 | AWQ，Pile | 36.6 | 51.5 |
| W3A16 | AWQ，COCO | 38.7 | 61.8 |
| W3A16 | MBQ，COCO，MSE | 40.8 | 64.8 |
| W3A16 | MBQ，COCO，MAE | 42.0 | 66.4 |
| W4A8 | SmoothQuant，Pile | 30.9 | 41.6 |
| W4A8 | SmoothQuant，COCO | 29.2 | 10.2 |
| W4A8 | MBQ，COCO，MSE | 41.9 | 63.5 |
| W4A8 | MBQ，COCO，MAE | 42.6 | 64.4 |

图文校准并不自动改善所有方法；数据覆盖、目标与数值策略需要一起检查，见 [[calibration-and-range-selection|校准数据与量化范围选择]]。此表的 MAE 与 MSE 差异仍要结合实现归一化与重跑条件解释。

进一步影响方法选择的证据有：

- **粒度不是越细越好。** §4.3.2 中，直接按 token 梯度加权比模态平衡的 OCRBench 分数低约 1.5。[[vlmq|VLMQ]] 使用不同梯度目标与重构算法，因此这不能推广成 token 级方法无效。
- **图文收益不能代替纯文本验证。** 表 7 的 LLaVA-7B MMLU：FP 65.9，MBQ W3 62.9，W4A8 61.8，仍有能力损失。
- **高位宽下优势可能消失。** 表 11 的 LLaVA-7B W4A16：AWQ 与 MBQ 都为 67.7；W8A8：MBQ 68.3、SmoothQuant 68.5、RTN 68.6。

表 2 与表 4 的 SEED 数值存在未解释的跨表差异；以上校准消融只使用表 4 内部比较，不拼接两表得出结论。

## 7. 实现加速与证据边界

论文 W3 内核将八个 3-bit 权重装入三个字节，计算时反量化到 FP16，并非硬件原生 INT3 乘法。量化参数、打包、访存与反量化成本如何进入执行，见 [[quantized-matmul-scaling-execution|量化矩阵乘法的缩放与执行路径]]。（§4.4、表 8–9）

表 9 在 RTX 4090、FlashAttention-2 下分别测量阶段成本：视觉编码器 729 tokens 的 FP16/W4A8 为 11.2/9.7 ms；语言 prefill 512 tokens 为 68.8/59.4 ms；decode FP16/W3A16/W4A8 为 29.6/21.1/26.3 ms，并按文中列出的输入长度平均。不同阶段的收益明显不同。

29.6/21.1≈1.40 是该 decode 口径下的收益，不是整次图像到回答的端到端加速，也不是任意边缘硬件的收益。视觉编码、prefill、decode、KV cache、传输应分开测量。论文没有验证云端选择样本、部署后漂移、参数下发或端云协同收益。

## 8. 当前可用结论与未解决项

- **Strong：** 将模态敏感性直接用于通道缩放选择，兼容 weight-only 与 W+A 两条路径；梯度只在校准阶段使用，推理时不需要重新估计重要性。
- **Weak：** 模态均值掩盖组内差异，重要性依赖校准任务与 mask；尺度搜索受候选族限制，收益随模型和位宽变化，且没有最优模态分配的理论保证。

核心公式、归一化和实验条件见前文。论文已全文研读，代码仅定向核对，未运行模型或内核复现。与 token 级二阶路线的比较见 [[mbq-vlmq-comparison|MBQ 与 VLMQ 的条件化比较]]。

后续工作已把 MBQ 用作多模态量化中「数据感知方法」的代表，与数据无关的 HQQ 在 int8／int4／int3 下对照，并同时测量准确率、校准误差与选择性预测表现；该研究得到的量级是 int4 MBQ 加外部置信度估计器可保留约 98% 的 bf16 精度、显存约减少 75%。这属于同一方法在另一组评测维度上的表现，不能反过来替代本页的精度与实现证据，详见 [[quantization-reliability-and-selective-prediction|量化与可靠性：选择性预测评测]]。
