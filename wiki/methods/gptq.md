---
title: GPTQ：基于二阶补偿的权重量化
type: method
tags:
  - ptq
  - weight-quantization
  - reconstruction
  - second-order
sources:
  - raw/papers/2026-09-21/quarot/paper.pdf
  - raw/papers/2026-09-21/spinquant/paper.pdf
  - raw/papers/2026-09-21/gptaq/paper.pdf
  - raw/papers/2026-09-21/vlmq/paper.pdf
  - raw/papers/2026-09-21/gptq/paper.pdf
  - raw/repositories/2026-09-21/gptq/source/gptq.py
  - raw/repositories/2026-09-21/gptq/source/quant.py
  - raw/repositories/2026-09-21/gptq/source/opt.py
  - raw/repositories/2026-09-21/gptq/source/datautils.py
  - raw/repositories/2026-09-21/gptq/source/README.md
paper_version: arXiv:2210.17323v2
code_commit: 2d65066eeb06a5c9ff5184d8cebdf33662c67faf
verification: 全文含附录研读与定向代码核对；未运行模型实验
updated: 2026-09-15
---

# GPTQ：基于二阶补偿的权重量化

GPTQ 对训练好的模型做逐层权重量化。每固定一列的量化值，就利用校准输入形成的二阶信息调整尚未量化的权重，补偿输出误差。它通过共享列顺序、延迟批更新和 Cholesky 重写，把这种补偿扩展到很大的 Transformer 模型。

本页依据 GPTQ v2（2023-03-22，ICLR 2023，以下简称 P），完整阅读 16 页、附录 A.1–A.4、4 图、22 表和算法 1。官方代码快照 `2d65066eeb06a5c9ff5184d8cebdf33662c67faf`（以下简称 C）仅定向静态阅读。所有模型质量、耗时与速度数字均为作者报告。

**实现层深入入口：** 量化器参数、权重位打包布局，以及 vLLM／SGLang 的加载与 repack 路径见 [GPTQ 的实现核对](../implementation/gptq-implementation.md)。

前置阅读：[线性层与输入通道](../fundamentals/operators/linear-layer-input-channel.md)、[均匀量化与分组](../fundamentals/quantization/uniform-quantization-and-groups.md)。[层输出重构与二阶误差补偿](../theory/layer-reconstruction-second-order-compensation.md)提供完整的约束优化推导与教学例子。

## 1. 对象、问题与研究动机

P 主要研究 OPT 与 BLOOM 的 weight-only PTQ，主实验为 3/4-bit 权重，激活不量化；嵌入与输出层保留 FP16。目标是减少巨大模型的权重存储和低 batch 生成时的权重读取量，同时把量化本身控制在可接受的时间和内存内。

RTN 在每个网格上独立选择最近值，成本低，却没有利用输入通道相关性来修复输出。OBQ 逐权重选择并补偿，质量较好，但按不同行独立进行逆 Hessian 更新代价很大。GPTQ 的贡献既包括量化策略，也包括把计算组织得足够高效。[P §1–4、图 1–2]。

“one-shot”表示不进行端到端再训练，不表示无校准数据、无参数更新或只做一次舍入。P 的小模型实验还包含 CNN 和 BERT，但不能据此把所有视觉模型上的效果当作已知普遍结论。

## 2. 重构目标、符号与二阶信息

采用 $W\in\mathbb R^{r\times d}$、$X\in\mathbb R^{d\times n}$、$Y=WX$。$r$ 是输出通道数，$d$ 是输入通道数，$n$ 是校准输入向量数，包含汇集后的 token。$\widehat W$ 是网格上的近似实数值，不一定已经打包成低比特整数。

P 式 (1) 的层输出平方重构目标可明确写成

$$
\min_{\widehat W\in\mathcal G}\|(W-\widehat W)X\|_F^2,
\qquad H=2XX^{\mathsf T}\in\mathbb R^{d\times d}.
$$

$\mathcal G$ 表示量化网格约束。对固定输入和单行权重，$H$ 是这个二次目标的精确 Hessian，各输出行共享它；它并非完整语言模型任务损失的 Hessian。重构目标、有限校准数据和逐步离散选择不保证最终任务损失最优。[P §3]。

主实验使用逐输出行的非对称均匀 min–max 网格。C 的 `Quantizer` 将范围扩展到包含零，以 $\Delta=(x_{\max}-x_{\min})/(2^b-1)$ 和 $z=\operatorname{round}(-x_{\min}/\Delta)$ 编码；舍入后裁剪到 $[0,2^b-1]$ 再反量化。全零范围有特殊处理，另有可选对称与范围搜索分支。C `quant.py:6`、`quant.py:36`。这些默认分支不能不加区分地混入 P 的实验。

## 3. 单步补偿如何成立

设 $F$ 是仍可调整的坐标，量化坐标 $q$ 带来的改变量是 $a=Q(w_q)-w_q$。在自由变量的局部连续最优点，求解带约束 $\delta_q=a$ 的二次问题得到

$$
\delta_F=\frac{Q(w_q)-w_q}{[H_F^{-1}]_{qq}}(H_F^{-1})_{:,q},
\qquad \Delta L=\frac{(Q(w_q)-w_q)^2}{2[H_F^{-1}]_{qq}}.
$$

因此补偿大小取决于输入相关性，而非只看该权重幅度。OBQ 根据增量代价贪心挑下一个权重；固定它后，用秩一消元更新剩余自由坐标的逆 Hessian。[P 式 (2)–(3)]。

这是单步连续受约束最优，不是整层离散问题的全局最优。只要后续也要舍入、校准分布有偏或数值计算有误，最终质量就仍需检查。逆矩阵的更新及完整拉格朗日推导见[机制页](../theory/layer-reconstruction-second-order-compensation.md)。

## 4. 三个改动怎样使算法可扩展

### 4.1 所有输出行共享列顺序

GPTQ 不再对每一行独立寻找贪心最优顺序，而按共同列顺序量化。这样各行的剩余输入坐标一致，只需维护同一个 Hessian。P 报告顺序简化在其测试中影响较小，这是一项经验依据，不是任意顺序质量相同的定理。

P 给出的复杂度由 OBQ 的 $O(rd^3)$ 降到 $O(\max\{rd^2,d^3\})$，可节省一个约 $\min(r,d)$ 的因子。Hessian 仍占 $O(d^2)$ 存储，不能称为线性内存算法。[P §4 Step 1、图 2]。

### 4.2 块内即时更新，块外延迟更新

决定当前列量化值，只需要此前对当前列的补偿。对较晚列的更新可以累积起来，用矩阵乘法一次应用。P 采用列块大小 $B=128$，在块内逐列更新，块完成后更新其右侧权重。这样主要改善 GPU 运算与访存组织，并不减少理论运算总量。[P Step 2、式 (4)–(5)]。

用统一的列向量记法，设 $A=H_F^{-1}$，将坐标集合 $S$ 同时固定所需的改变量为 $a_S$，多坐标补偿与剩余逆矩阵为

$$
\delta_F=A_{:,S}(A_{S,S})^{-1}a_S,
\qquad A_{\mathrm{remaining}}
=\left(A-A_{:,S}(A_{S,S})^{-1}A_{S,:}\right)_{-S,-S}.
$$

这是 P 式 (4)–(5) 的维度一致展开。实际算法仍在块内递归决定各列的量化值；不能把整个块一次 RTN 再做上述补偿，当作与算法 1 相同。Cholesky 写法把所需逆矩阵信息预先整理好，见下一节。

必须区分三个概念：

| 名称 | 职责 |
|---|---|
| Transformer block | 一组网络层；用于逐块加载与更新下一个块的校准输入 |
| 算法列块大小 $B$ | 将权重补偿合并执行的计算分块 |
| 量化 group size $g$ | 决定多少权重共享 scale/zero-point，影响精度与元数据 |

$B=128$ 不表示启用了 $g=128$。P 的基础表格采用逐行网格，带 `g` 标记的实验才表示更细量化分组。

### 4.3 阻尼与 Cholesky 形式

反复消元可能累积数值误差，使理论上正定的逆矩阵失去这一性质。P 在 $H$ 对角线上加入平均对角值的 1%，并预先计算上三角因子

$$
H_\lambda=H+\lambda I,\qquad H_\lambda^{-1}=R^{\mathsf T}R.
$$

算法 1 把 `H^{-1}` 变量覆盖成了这个三角因子，容易误读。本页用 $R$ 单独表示它。对列块 $[i,k)$，其中 $k=\min(i+B,d)$，执行：

1. 对每个 $j=i,\ldots,k-1$，计算 $Q_{:,j}=\operatorname{quant}(W_{:,j})$。
2. 令 $E_{:,j-i}=(W_{:,j}-Q_{:,j})/R_{jj}$。
3. 块内更新 $W_{:,j:k}\leftarrow W_{:,j:k}-E_{:,j-i}R_{j,j:k}$。
4. 块结束后更新 $W_{:,k:d}\leftarrow W_{:,k:d}-ER_{i:k,k:d}$。

当前列的更新使其恰落在 $Q$，后续列保留补偿。分母是 $R_{jj}$；若计算局部损失则涉及 $R_{jj}^2$。把原逆 Hessian 的对角元素直接替代这里的 $R_{jj}$ 会改变算法。P 算法 1；C `gptq.py:98`。

## 5. 校准与官方实现流程

P §5 的校准集为 C4 中随机抽取的 **128 段、每段 2048 tokens**。量化使用单张 A100 80GB，逐 Transformer block 加载，汇集内部线性层输入并建立 Hessian。当前块量化后再运行一次，产生下一块的校准输入。因此后面的块会看到前面块已经量化后的输入；不等于每个块内部的所有层也严格按真实执行依赖逐层重采样。

C 的 `opt.py:76`对应这一流程：先同时收集该块各线性层输入，依次量化，重新前向，再交换 `inps/outs`。

P 附录 A.2.1 描述 PPL 评价：用两个换行拼接文本，使用各模型默认 tokenizer，切成互不重叠的 2048-token 片段，聚合下一 token 预测的负对数似然后取指数。窗口长度、tokenizer、文本分隔与数据 split 都影响数值。zero-shot 任务采用 EleutherAI harness 的处理和打分方式，逐样本、不 padding。当前代码有后续协议更新，见 §8，不能只写数据集简称就认为结果可比。

C 中应保留的细节：

- `gptq.py:32`将 token 展平，按累计输入 batch 样本数归一化累加 $2XX^{\mathsf T}$。它是与论文同类的二阶统计，但计数不应直接写成 token 总数；整体归一化与成比例阻尼配合时，不改变精确算术中的补偿比值。
- `gptq.py:75`将校准中对角为零的通道权重置零，并把相应对角改为 1。校准未激活不证明部署时永不激活，这是一项需要保留的覆盖假设。
- 计算主要使用 FP32，关闭 TF32；完成后将量化网格值写回原权重 dtype。此时仍是浮点张量，不能宣称已得到压缩 checkpoint。`gptq.py:60`。
- 动态分组在量化过程中获取网格；但 C 的组参数从外层 `W` 切片读取，当前列块内部补偿在 `W1` 中进行。当组边界落在同一个算法列块内部时，不能笼统称为“始终基于全部最新补偿权重”。常见 $g=B=128$ 与任意 $g,B$ 的情况需要区分。`gptq.py:110`。

## 6. 质量、消融与反例

### 6.1 主实验并非所有模型都“几乎无损”

以下为 P 表 3–4 的 WikiText2 PPL，越低越好；统一在本表内部比较。基础设定为逐行非对称网格，未启用额外 group。

| 模型 | FP16 | RTN 4-bit | GPTQ 4-bit | GPTQ 3-bit |
|---|---:|---:|---:|---:|
| OPT-125M | 27.65 | 37.28 | 31.12 | 53.85 |
| OPT-13B | 10.13 | 11.32 | 10.31 | 11.61 |
| OPT-66B | 9.34 | 110 | 9.55 | 14.16 |
| OPT-175B | 8.34 | 10.54 | 8.37 | 8.68 |
| BLOOM-176B | 8.11 | 8.37 | 8.21 | 8.64 |

这些结果支持在所测大模型上相对 RTN 的改进，也展示了小模型和 OPT-66B 的明显退化。P 将 66B 的异常与早期层 dead units 联系起来，但这不是已经排除其他因素的因果证明。图 1 与表 3–4 展示的“更大模型通常更易量化”是该实验集合中的趋势。

P 表 5、附录表 9–12 还报告 PTB/C4：OPT-175B 的 FP16/GPTQ 4-bit 为 PTB 12.01/12.26、C4 10.13/10.28；BLOOM-176B 为 PTB 14.59/14.75、C4 11.71/11.81。更小模型损失可能明显更大。C4 使用同一语料的训练部分做校准，附录明确说明 C4 评价不完全属于跨任务 zero-shot；训练/验证分离本身也不等于跨域泛化。[P 表 11–12 图注]。

### 6.2 分组与极低比特

P 表 5 中 OPT-175B 的 3-bit WikiText2 PPL：逐行 8.68、g1024 8.45、g128 8.45；BLOOM-176B 为 8.64、8.35、8.26。更细分组可能改善质量，但并非每项指标都单调改善，例如 OPT 的 LAMBADA 在 g1024/g128 下为 77.39%/76.42%。图 4 展示中等 OPT 模型 4-bit 分组的改善。

P 表 7 的 2-bit 结果：

| 模型 | FP16 | 2-bit g128 | 2-bit g64 | 2-bit g32 | 3-bit 逐行 |
|---|---:|---:|---:|---:|---:|
| OPT-175B | 8.34 | 9.58 | 9.18 | 8.94 | 8.68 |
| BLOOM-176B | 8.11 | 9.55 | 9.17 | 8.83 | 8.64 |

这些 2-bit 值并非模型所有状态平均正好 2 bit。按 P 提到的每组 FP16 scale 和 2-bit zero-point，忽略其他成本的理想平均是 $2+18/g$ bit/weight；P 将 g128/g32 约写为 2.2/2.6 bit。另有 g8 三值量化，OPT-175B PPL=9.20；P 说明其平均压缩不如上述 2-bit 方案，FPGA 应用只是可能方向，未提供相应实测。

### 6.3 小模型、任务变化与耗时

P 表 1 中 ResNet18 的未量化基线 top-1=69.76%，4-bit GPTQ=69.37%、OBQ=69.56%；3-bit GPTQ=67.88%、OBQ=68.69%、BRECQ=68.47%。因此不能写成 GPTQ 在所有场景最准确。附录表 8 中 BERT-base 的 3-bit SQuAD F1 为 GPTQ 86.02/OBQ 85.29，但 4-bit 为 88.18/88.23；OPT-125M 3-bit PPL 为 53.85/69.32。这里保留作者表中的比较，未独立 ingest 各基线来重建完整公平性审查。

图 3、附录表 13–22 覆盖 LAMBADA、PIQA、ARC-easy/challenge 和 StoryCloze。3-bit GPTQ 通常明显优于崩溃的 RTN，4-bit RTN 在部分任务并不更差。例如 BLOOM-3B PIQA，RTN/GPTQ 为 69.86%/69.42%；OPT-175B ARC-challenge，FP16/GPTQ 4-bit 为 43.94%/42.75%。部分量化分数超过 FP16 也不能据单次结果宣布能力提升。原文未报告这些差值的置信区间。

P 表 2 的量化时间：OPT-13B 20.9 分钟、30B 44.9 分钟、66B 1.6 小时、175B 4.2 小时；BLOOM-176B 3.8 小时，均为单 A100 的量化过程。它们不是推理延迟。对 ZeroQuant-LKD 的数百小时比较包含线性外推，不是把该方法实际运行到 175B 的测量。

## 7. 存储与推理加速

P §5 报告 OPT-175B 3-bit 权重模型约 63GB，包含保持 FP16 的嵌入和输出层；2048 token 历史 K/V 另约 9GB，因此能放入单张 A100 80GB。该内存解释绑定此模型、位宽和上下文条件，不能直接推成任意 175B 部署保证。

P 的内核面向量化矩阵乘浮点向量：读取压缩权重后即时反量化，减少权重流量。它不把原乘法变成原生 INT3 乘法，也没有量化激活。对低 batch decode 有利的访存策略，不能自动推广到大 batch/prefill。[P §5、§6、附录 A.2.2]。

| OPT-175B，batch=1，生成长度 128 | FP16 | GPTQ 3-bit | 平均每 token 延迟比 |
|---|---|---|---:|
| A100 80GB | 5 GPU，230ms | 1 GPU，71ms | 3.24× |
| A6000 48GB | 8 GPU，589ms | 2 GPU，130ms | 4.53× |

这是 P 表 6 的历史结果。模型按连续层分布到 GPU，作者报告该调度的通信占比小于 5%，不是 tensor parallel 的同卡数对照。附录说明其他算子及框架开销保持相同，因此其内核解释有依据；仍须保留基线 GPU 数和分布方式。

大 batch 下，P 提议先完整反量化再做浮点矩阵乘，举出的 OPT-175B FC2、16×1024 token 案例是反量化开销较小，不是上述低 batch 加速率仍成立。进一步的执行原理见[量化矩阵乘法的缩放与执行路径](../implementation/quantized-matmul-scaling-execution.md)。

同一量化格式在批量场景下也可以换用别的内核：[Marlin](../implementation/marlin-batched-w4a16-gemm.md) 使用与原始 GPTQ 略有差别的权重布局，并对 GPTQ 补充了按组裁剪阈值搜索与可变长度校准序列，从而把 W4A16 的接近理论上限加速保持到 batch 约 16–32。格式改造与内核设计是两件事，复现时需要分别记录。

## 8. 论文算法与后来代码的边界

C `quant.py:137`的 `Quant3Linear` 用 3 个 32-bit 容器存放每 32 个 3-bit 权重，保存每输出行的 scale 和已乘 scale 的 zero，并保留 bias。`forward` 检查输入只能是一条向量，否则报错；普通路径转换到 FP32，`faster` 路径到 FP16。本轮没有读取或运行完整 CUDA 内核，不能保证所有形状都能正确执行。该类也没有通用 group 参数布局，不能把支持分组的数值算法等同于这个单独打包路径支持任意 group。

C 相对主论文还包含后续特性：README明确区分更新，核心实现可见 `gptq.py:81`。

- `act-order`：按 $\operatorname{diag}(H)$ 降序重排列，所有行仍共享同一顺序。这里是校准平方幅度统计，不是 AWQ 的平均绝对激活，也不是 OBQ 的逐权重贪心代价。
- `static-groups`：预先定义组网格；与 act-order 配合时按原始列归属选择组，最终还原列顺序。此“static”指组网格，不是 SmoothQuant 的静态激活量化。
- `true-sequential`：README 描述 LLaMA 路径可在同一个 Transformer block 内进一步按执行依赖处理；本轮没有完整读取该集成，不将其当作 P 中 OPT/BLOOM 默认实验。
- `new-eval`：后续 PTB/C4 数据预处理改变。C `datautils.py` 同时保留旧/新路径：PTB 的 split 与分隔符不同，C4 的随机片段与拼接截断不同。不能混用结果。`datautils.py:32`、`datautils.py:102`。

## 9. 证据问题与尚待验证

| 问题 | 本页处理与影响 |
|---|---|
| 式 (3) 最后的 $-p$ 未定义 | 按上下文“删除 $q$”解释，保留原文错误说明 |
| 表 5 与附录表 9 不一致 | OPT-175B 3-bit PTB 分别为 12.68 与 12.86；两者原件均如此，不擅自统一 |
| A.2.1 称 validation set，代码按数据集采用不同 split | 主文协议与当前代码分别记录；未证明某一路径复现全部表格 |
| P 的 PIQA 引用指向蛋白质数据库论文 | 所列任务数值保留为作者报告，不沿用该条错误书目来定义语言任务；本轮不展开任务独立 ingest |
| 历史方法/硬件范围 | “首次”“最大”“无原生混合精度支持”等按发表时期理解，不当作当前产业现状 |
| 泛化与全面质量 | 有限校准与有限任务不能证明普遍无损；§7 也指出偏差等指标尚未充分研究 |
| 实际执行 | 只核对所列代码文件；未运行权重压缩、模型评测、GPU 内核或速度基准；P 的完整复现代码声明不等于本项目完成复现 |

研究联系见[AWQ、GPTQ 与 SmoothQuant：对象、机制与证据比较](awq-gptq-smoothquant-comparison.md)。新的实测比较需要统一模型版本、量化配置、校准与评测协议，以及后端和硬件；不同论文的 PPL 与加速倍数不能拼接排名。

## 10. 非对称校准与 token 加权的后续扩展

GPTAQ v3 §4 将受前层量化影响的输入 $U$ 与浮点参考输入 $F$ 区分，重构 $\widehat WU$ 与 $WF$，额外处理输入偏差引入的残差；这与本页同输入的局部重构目标不同。[VLMQ v2](vlmq.md) 默认在该非对称路线中引入 token 因子，形成 $\|(\widehat WU-WF)G\|_F^2$，但部分 INT2 实验又选择 GPTQ 为基础算法。

因此“用到 GPTQ 内核”“基于 GPTQ 补偿”和“使用 GPTAQ 非对称校准”要分开记录。VLMQ 的带权重构与残差推导由其方法页承接；本次对 GPTAQ 仅按需要局部研读，不将其列为已独立完成的全文 ingest。

## 11. 旋转与权重量化的先后关系

[QuaRot](quarot.md) 在固定旋转坐标中重新收集输入并运行 GPTQ；[SpinQuant](spinquant.md) 的主配置则先用 W16 与模拟低比特激活学习旋转，再固定旋转执行 GPTQ。旋转改变待量化权重、激活和相应二阶统计，因此不能直接把原坐标系的量化码或校准矩阵原样当作新坐标的结果。两篇的流程也说明 GPTQ 是可组合的权重补偿阶段，不自行完成激活或缓存的低比特部署。（QuaRot v2 §4 Stage 2a；SpinQuant v4 §4.2、表 3。）

## 来源身份

下表用于在没有本地资料库时辨识来源；具体论述的章节、公式、图表或代码位置见正文。

| 来源 | 版本或快照 | 说明 |
| --- | --- | --- |
| [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456v2) | `arXiv:2404.00456v2` | — |
| [SpinQuant: LLM quantization with learned rotations](https://arxiv.org/abs/2405.16406v4) | `arXiv:2405.16406v4` | — |
| [GPTAQ: Efficient Finetuning-Free Quantization for Asymmetric Calibration](https://arxiv.org/abs/2504.02692v3) | `arXiv:2504.02692v3` | — |
| [VLMQ: Token Saliency-Driven Post-Training Quantization for Vision-language Models](https://arxiv.org/abs/2508.03351v2) | `arXiv:2508.03351v2` | — |
| [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323v2) | `arXiv:2210.17323v2` | — |
| [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq/tree/2d65066eeb06a5c9ff5184d8cebdf33662c67faf) | `2d65066eeb06a5c9ff5184d8cebdf33662c67faf` | — |
