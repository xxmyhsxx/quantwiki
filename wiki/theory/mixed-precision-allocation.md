---
title: 混合精度分配：选择变量、预算与部署口径
type: concept
tags:
  - mixed-precision
  - sensitivity
  - memory
  - deployment
sources:
  - raw/papers/2026-09-28/same-bit-width-different-outcomes/paper.pdf
  - raw/papers/2026-09-28/foldquantvla/paper.pdf
  - raw/papers/2026-09-26/ramp-edge-cpu-mixed-precision/paper.pdf
  - raw/papers/2026-09-23/quantization-retrieval-damage/paper.pdf
  - raw/papers/2026-09-22/loftq/paper.pdf
  - raw/papers/2026-09-21/spqr/source.eprint
  - raw/papers/2026-09-21/squeezellm/source.eprint
  - raw/papers/2026-09-21/luq/paper.pdf
  - raw/papers/2026-09-21/billm/paper.pdf
updated: 2026-09-29
---

# 混合精度分配：选择变量、预算与部署口径

混合精度分配决定不同层、模块或张量用什么量化配置。它与固定配置下的误差加权不同：[QIG](../methods/qig.md) 改变哪些 token 的误差更受重视，[LUQ](../methods/luq.md) 则决定哪些层使用更低的权重精度。本页以 LUQ 的层间二选一为具体依据，展开预算与搜索的共通问题，不把它概括为所有混合精度算法。

## 1. 先明确选择的是什么

设第 $l$ 层有 $n_l$ 个参与量化的权重，$b_l$ 是权重编码的平均参数位宽。一般问题可以写为

$$\min_{\boldsymbol b}\mathcal E(\boldsymbol b)\quad
\text{s.t. }C(\boldsymbol b)\le B,$$

其中 $\mathcal E$ 是明确的重构或任务误差，$C$ 是存储、延迟等代价，$B$ 为预算。这是问题表述，不是说论文求得了全局最优。配置实际上还包括量化器、粒度、scale/zero-point、残差与内核；相同 $b_l$ 不代表相同误差或代价。

层间、层内通道和 token 精度分配的执行代价不同。层间选择通常较容易映射到每层可用的内核；“存在对应内核”仍不保证选择代价可逐层简单相加。共享缓冲区、内存带宽、调度和输入形状会改变收益，见 [量化矩阵乘法的执行路径](../implementation/quantized-matmul-scaling-execution.md)。

[SpQR](../methods/spqr.md) 和 [SqueezeLLM](../methods/squeezellm.md) 将额外精度分配到零散权重，分别按可补偿误差与任务敏感性等依据选择。此时高精度值之外还需保存位置，并执行稀疏修正；细粒度分配的预算分析见 [低比特表示设计空间](../research/low-bit-representation-design.md)。

## 2. 平均位宽和完整内存必须分别计算

参数加权平均为

$$\bar b=\frac{\sum_l n_l b_l}{\sum_l n_l}.$$

只有各层参与计算的参数量相同时，才能用层数平均。完整模型文件还包括元数据及未量化参数；运行峰值还包括激活、KV cache、视觉端、工作区和临时副本。

[LoftQ](../methods/loftq.md) v4 表 5 将前 16／8／4 层用 4 bit、其余用 2 bit 的方案标为“3／2.5／2.25 bit”。这是固定前缀配置，不是已求得最优的敏感性分配。教学上，对 $L$ 个等参数量层、前 $k$ 层高位宽，有 $\bar b=2+2k/L$；同样前 4 层，$L=32$ 时为 2.25，$L=40$ 时为 2.20。例子说明标签不能脱离层数与参数量直接用于存储预算；LoRA 因子和量化元数据还需另计。

其中 KV cache 随批量与上下文增长，且有自己的粒度与布局选择，不能并入「权重平均位宽」一起换算，见 [KV cache 量化的对象与粒度](kv-cache-quantization-objects-and-granularity.md)。

LUQ 用 BiLLM 的约 1.08 与 GPTQ 的 4 作为两档参数位宽。但 **BiLLM 原文 §3.3 式 14** 将 $N_{\rm param}=1+r_{\rm salient}$ 与分组标记存储 $N_{\rm storing}=1+1/b_{\rm block}$ 分开，§4 表 6 的内存公式还包括压缩 bitmap 和 scale。因此 1.08 不是每个参数连同全部元数据只占 1.08 bit 的承诺，也不能再无条件把未压缩标记成本当实际最终文件大小。实际占用需要明确格式与压缩方式。

按 LUQ 名义参数口径、等参数量的 32 层中 16 层用 1.08 bit，$\bar b=2.54$；相对 4 bit 的下降为 $1-2.54/4=36.5\%$。这不等于整机内存下降 36.5%，也推不出同倍数加速。BiLLM 的二值残差与分组知识来自原论文局部研读，具体展开见 [LUQ 的底层量化器](../methods/luq.md)。

## 3. 固定排序把组合搜索缩成前缀搜索

两档位宽、$L$ 层原本有 $2^L$ 种组合。若固定排序 $\pi$，只允许前 $k$ 层取低精度，候选只剩 $L+1$ 个：

$$b_{\pi_i}(k)=\begin{cases}b_{\rm low},&i\le k,\\b_{\rm high},&i>k.\end{cases}$$

LUQ §3.2–3.3 以层输出的 K-means 簇频率熵升序构造 $\pi$。它用排序缩小搜索空间，但可能排除更好的非前缀组合；固定排序也不保证每个层的敏感性在其他层量化后保持不变。校准代理的定义和旋转条件见 [熵与依赖](../fundamentals/mathematics/entropy-and-dependence.md)。

质量门槛与容量门槛是两种停止条件：

- 若要求 $A(k)\ge\tau$，可以在候选中找满足质量的最大 $k$，争取更多压缩。
- 若要求 $C(k)\le B$，在存储随 $k$ 下降、并希望少降精度时，可先找达到容量要求的最小 $k$，再检查质量；若质量并不单调，仍需比较可行候选。

不能把二者都写成“满足内存上限后继续取最大 $k$”，那会偏向全低精度。LUQ 的 LLaVA 和 Qwen 配置分别用质量门槛与容量场景选择，不是同一个多任务优化准则。

另一类有序分配见 [HAWQ-V2](../methods/hawq-v2.md)，它使用多档有序分组：按平均 Hessian trace 排序，只允许敏感度高的层位宽不低，再以“平均曲率乘候选量化扰动能量”的总和选择存储—误差 Pareto 前沿。固定排序缩小搜索空间但仍可能排除更优配置；trace 估计准确不能补回它丢失的误差方向。其最终结果还包含 QAT，不能直接当作无需训练的 PTQ 结果。

同样的预算求解器不保证同样的分配质量：[CASA](../methods/casa.md) 用候选误差的方向加权代价替代标量敏感度，实际以对角因子建立 MCKP，再用跨层上界筛选交换、校准损失决定接受。连续高码率模型的闭式位宽公式与实际离散配置搜索是两个层次，不能以连续最优性证明局部搜索全局最优。

## 4. 二分搜索需要可检验的单调性

LUQ §3.3 提到二分以减少候选评价。对于名义存储，降低某层位宽通常给出明确的单调减少；对于任务分数，并无普遍的单调下降保证。量化误差抵消、生成答案变化和有限评测样本都可能造成非单调。

**教学反例：**$A(0..3)=(0.90,0.70,0.86,0.60)$，门槛 $0.80$。$k=1$ 失败，但 $k=2$ 仍可行。遇到第一次失败即停止，或未经检查就二分，都可能错过候选。可以采用完整前缀扫描，或先验证所用评价条件的近似单调性；这些是方法复用时的判断，不是已替 LUQ 跑完的实验。

搜索用过的任务分数不再自动是独立测试证据。[校准、验证与测试分工](calibration-and-range-selection.md) 要同时覆盖参数校准、排序/位宽选择和最终报告，不能只隔离量化器使用的输入。

## 5. 部署格式改变后要重新连接质量证据

LUQ 附录 D 将 BiLLM/GPTQ 所选的层配置映射到 IQ1_M/Q4_K_M 来测试 CPU 吞吐。它说明层间混合方案能够适配现有引擎，却不能直接证明替换格式后仍有主实验的精度。必须在实际导出格式上重新评测，并报告模型范围、输入长度、batch、硬件、内核版本与内存测量口径。

这也限定了“预算”的含义：理论权重容量、导出文件大小、加载后的常驻内存和端到端峰值应分开记录。对于端侧设备，视觉编码和 KV cache 等剩余项可能决定最终是否能运行。

不同部署框架对格式、粒度与 KV 位宽的支持并不一致，选定配置前需要核对目标后端的支持面与实际内核路径，见 [部署框架与后端支持](../implementation/quantized-llm-deployment-backends.md)；块结构层面的位宽口径与元数据组织见 [GGUF 块量化存储格式](../implementation/gguf-block-quantization-formats.md)。

## 6. 任务敏感度与图执行成本要分别连接

[检索间隔分配](quantization-ranking-stability.md) 用固定浮点 top-1/top-2 文档对的分数差变化，衡量单层 W3 对决策边界的影响，再按每参数收益提升到 W4。它与局部 MSE 或分类 logits 散度优化的对象不同；对较小层除以参数量只是预算启发式，不能证明跨层最优。

[RAMP](../methods/ramp.md) 在 CPU 图像分类上，用逐层 FP32→INT8 干预后的最终 logits 分布 JSD 聚类。原算法按敏感度不超过选中簇的质心量化，不等于把整簇全部量化；聚类质量也不是任务质量保证。它还展示 Q/DQ 边界和失去算子融合的图切分成本，故真实延迟通常不是独立层耗时的简单和，需要对候选完整执行图测量。

[FoldQuantVLA](../methods/foldquantvla.md) 的输出投影升到 INT8 又提供另一类选择：按共享输入站点及残差路径制定精度预设。部分 checkpoint 同时换了校准预设，必须先看匹配预设的对照，才能把变化归因于位宽。由此，选择配置至少需分清敏感度计算对象、选择单元、完整预算、图执行和受控质量对照，不能以“混合精度”统一解释不同方法。

[跨 TTS 系统的组件消融](../implementation/tts-quantization-evaluation.md) 又说明，敏感组件随模型及量化对象变化：OmniVoice 权重量化与 per-tensor 激活量化的瓶颈不同，两个组件一起量化的损失也可能大于独立损失之和。应先分开测量权重、激活和粒度，再复测联合配置；按模型类别或参数占比直接分配精度没有足够依据。保护组件或 GPTQ 校准后的均值改善，还须满足预先声明的质量区间，并在目标 runtime 上验证成本。

## 来源身份

下表用于在没有本地资料库时辨识来源；具体论述的章节、公式、图表或代码位置见正文。

| 来源 | 版本或快照 | 说明 |
| --- | --- | --- |
| [LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models](https://arxiv.org/abs/2310.08659v4) | `arXiv:2310.08659v4` | 前缀混合精度的名义标签与实际预算 |
| [LUQ: Layerwise Ultra-Low Bit Quantization for Multimodal Large Language Models](https://arxiv.org/abs/2509.23729v3) | `arXiv:2509.23729v3` | — |
| [BiLLM: Pushing the Limit of Post-Training Quantization for LLMs](https://arxiv.org/abs/2402.04291v2) | `arXiv:2402.04291v2` | — |
| [SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression](https://arxiv.org/abs/2306.03078v1) | `arXiv:2306.03078v1` | 细粒度例外预算 |
| [SqueezeLLM: Dense-and-Sparse Quantization](https://arxiv.org/abs/2306.07629v4) | `arXiv:2306.07629v4` | 敏感值与尾部保留 |

- [The Undetected Damage of Quantization on Retrieval and How to Fix It](https://arxiv.org/abs/2609.24322v1)，arXiv:2609.24322v1；排序稳定性、间隔分配与校准保证。
- [RAMP](https://arxiv.org/abs/2609.28262v1)，arXiv:2609.28262v1；CPU 混合精度、敏感度聚类与图切分成本。
- [FoldQuantVLA](https://arxiv.org/abs/2609.24433v1)，arXiv:2609.24433v1；共享输入折叠、目标投影位宽和编译基线。

- [Same Bit Width, Different Outcomes: Post-Training Quantization of Text-to-Speech Across Architectures](https://arxiv.org/abs/2609.28974v1)，v1；组件消融、质量判据与真实执行对照。
