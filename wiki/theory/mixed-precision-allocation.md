---
title: 混合精度分配：选择变量、预算与部署口径
type: concept
tags:
  - mixed-precision
  - sensitivity
  - memory
  - deployment
sources:
  - raw/papers/2026-09-26/casa-cross-layer-sensitivity
  - raw/papers/2026-09-22/hawq-v2
  - raw/papers/2026-09-21/coopq
  - raw/papers/2026-09-21/channel-wise-mixed-precision
  - raw/papers/2026-09-28/same-bit-width-different-outcomes
  - raw/papers/2026-09-28/foldquantvla
  - raw/papers/2026-09-26/ramp-edge-cpu-mixed-precision
  - raw/papers/2026-09-23/quantization-retrieval-damage
  - raw/papers/2026-09-22/loftq
  - raw/papers/2026-09-21/spqr
  - raw/papers/2026-09-21/squeezellm
  - raw/papers/2026-09-21/luq
  - raw/papers/2026-09-21/billm
  - raw/papers/2026-10-07/dbw
  - raw/repositories/2026-10-07/dbw
  - raw/papers/2026-10-07/stagq
updated: 2026-10-13
---

# 混合精度分配：选择变量、预算与部署口径

混合精度分配决定不同层、模块或张量用什么量化配置。[QIG](../methods/qig.md) 在固定配置下改变哪些 token 的误差更受重视，[LUQ](../methods/luq.md) 则决定哪些层使用更低的权重精度。本页结合层间、通道和因子分配，解释选择变量、敏感性代理、预算及搜索条件，并连接实际部署成本。

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

## 7. 改变因子宽度时，rank 与 bit 必须共同计费

[DBW](../methods/dbw.md)把 $W\in\mathbb R^{m\times n}$ 的主体改成两侧因子，对每对分量分配位宽 $b_c$；0 bit 删除该分量。每个分量含 $m+n$ 个因子元素，因此忽略元数据与对齐的原矩阵口径是

$$b_{\rm original}=\frac{(m+n)\sum_c b_c}{mn}.$$

对方阵，全秩两侧因子的元素数是原矩阵的两倍；因子平均 2 bit 并不等于原矩阵 2 bpw。减少因子宽度与增加保留分量位宽可以共享同一预算，而把不同形状矩阵的因子位宽直接平均会丢掉这个成本差异。DBW 固定代码先按原权重载荷在 block 内分配矩阵份额，再除以各矩阵因子元素数，最后分配分量整数位宽；其 Hamilton 投影有整数预算、正归一化与位宽上限条件，不保证重构目标全局最优。（DBW v1 §2.2、附录 C.2；固定 `a45f26e1` 的 `block_ap.py::block_ap`。）

这个预算还未包含组尺度/偏移、系数与 word padding；整数存储的因子在固定前向中恢复为完整浮点张量后，再执行两次矩阵乘，运行峰值和延迟不能只按上式估计。完整的选择变量、训练与保存/读取路径见方法页。

## 8. 多个可读精度与逐层位宽分配是不同选择

[StagQ](../methods/stagq.md)让同一份主体编码按前缀提供 p2–p6：每组共用一对元数据，读取更深 plane 时派生新的原点/步长，所有精度仍读取同一份保护记录。它的首要选择是“一个编码同时提供哪些 operating points”，不是为每一层分别训练独立量化器。共享前缀使各精度不能任意独立拟合；在相同 rate/失真/校准信息/decoder 类别下，其可行集合属于固定精度码的子集，不保证优于最佳独立单精度表示。（StagQ v1 §2.1–3。）

共享 pair 为 $8/g$ bpw，每新增主体 plane 加 1 bpw；codebook、稀疏值与索引/计数仍要计入。该论文的 Llama 两位 XXS 是 logical 2.35 bpw，而当前 real-weight kernel 因固定 row slots 读取 2.47 bpw；更紧凑的可变长布局仍是未来工程。保存流可截取，实际 kernel 在 load 时重打包成连续 $M$-bit code，也不能据此宣称已支持服务中即时切换。这个案例再次说明：payload 精度、逻辑字段、物理读取和运行峰值属于不同预算。（StagQ v1 §4.1、附录 E/G/A。）

## 9. 分位阈值不自动满足精确预算

对 $d$ 个等长度通道，每通道只取2/3/4 bit时，主体平均为 $\bar c=d^{-1}\sum_i c_i$，可达值是有限离散集合，间隔为 $1/d$；“平均可取分数”不表示任意实数都能精确表示。[CMPQ](../methods/cmpq.md) v4算法1用激活范数分位点与严格比较决定升级/降级，以名义预算表达分配比例。

若基线全为3、升级数量为 $k_4$、降级数量为 $k_2$，则

$$\bar c=3+\frac{k_4-k_2}{d}.$$

阈值只间接控制这两个整数。四个分数都等于1时，目标3.5的中位阈值也是1，严格大于者为0，算法字面仍为3；即使目标可由恰好两个升级实现，也不保证分位比较选出两个。分数各异时，端点2取最大分位点并严格小于，会至少留一个3-bit通道；端点4同理至少留一个3-bit通道。实现若用端点特判或排序名额处理，应另据代码说明，不将整理者修复方案冒充作者实现。

给定名额也需声明取整与并列处理：例如仅升级且 $b\ge3$ 时，严格上限可要求 $k_4\le\lfloor d(b-3)\rfloor$，而最近平均值可能用四舍五入，两者约束不同。不同通道长度时又必须按参数量加权，不能直接用通道数量比例。即使主体预算准确，中心、保护值/索引与对齐仍须计入第2节的完整代价；通道排序也没有保证任务质量全局最优。

## 10. 联合量化的边际贡献与交互代理

单层在其他层全浮点时测出的损失，未必等于多个层一起量化时的损失增量。[CoopQ](../methods/coopq.md) v2的SPQE从全4-bit状态出发，沿随机排列逐层降到2-bit，在不同联合状态下重复评价NLL。为解释成本，设 $q_i=1$ 为低精度、$g(q)$ 为相对全高的损失增加；一次排列中第 $i$ 层的成本边际为 $d_i=g(q+e_i)-g(q)$，其中 $e_i$ 是第 $i$ 个单位向量，降精度前 $q_i=0$。随机排列期望给出该低精度成本游戏的Shapley值；它平均分摊状态依赖影响，不保证每个具体组合都可由边际均值相加恢复。CoopQ原文用高精度集合与“前−后”的NLL边际，符号相反，不能直接混用为正损失。

**教学反例：**两层成本 $g(q)=q_1+q_2+2q_1q_2$。先量化1或先量化2，对齐层列后的边际向量分别为 $(1,3)$、$(3,1)$；均值 $(2,2)$，用分母2得到协方差

$$C=\begin{pmatrix}1&-1\\-1&1\end{pmatrix}.$$

真实联合附加损失为正的2，边际协方差非对角却为−1：每条路径总成本固定为4，一层分得较多时另一层分得较少。将协方差直接当作真实二阶交互会丢掉这个区别。对全部边际统一反号，协方差不变，也不能借此消除目标一阶项的符号问题。

对称矩阵 $K$ 的二进制代理 $F(q)=a^\top q+q^\top Kq$，其Shapley值为 $\psi_i=a_i+K_{ii}+\sum_{j\ne i}K_{ij}$：对角项随玩家自身加入，成对项的总系数 $2K_{ij}$ 平分到两位玩家。若只从均值中扣除非对角行和，仍留下 $K_{ii}$；不能未经额外对角约定就称为准确隔离一阶贡献。以上为整理者有限游戏推导，CoopQ式8–10是经验代理，其实验有效性与真实Hessian/精确游戏分解是不同证据。

求解器线性化可以精确求固定代理，不能补回代理未表达的真实损失。预算也须明确全低基线：升级增量约束应扣除基线占用；若右侧直接用总预算，则左侧要计入基线。实际方法页保留原文符号、形状、对角与预算衔接的未澄清处，未据此推断作者代码缺陷。

## 来源身份

| 来源 | 版本或快照 | 说明 |
| --- | --- | --- |
| CASA（`https://arxiv.org/abs/2609.25916v1`） | `arXiv:2609.25916v1` | 对角候选代价、跨层上界与局部接受条件；§4–5、Algorithm 1 |
| HAWQ-V2（`https://arxiv.org/abs/1911.03852v1`） | `arXiv:1911.03852v1` | 平均 Hessian trace、有序分组与 Pareto 选择；§2–3、附录 A–C |
| CoopQ: Cooperative Game Inspired Layerwise Mixed Precision Quantization for LLMs（`https://arxiv.org/abs/2509.15455v2`） | `arXiv:2509.15455v2` | SPQE、协方差代理与二进制求解；第10节有限游戏及Shapley代数为整理者展开 |
| Channel-Wise Mixed-Precision Quantization for Large Language Models（`https://arxiv.org/abs/2410.13056v4`） | `arXiv:2410.13056v4` | 算法1的分位分配；第9节有限预算、严格比较及计数例为整理者展开 |
| StagQ（`https://arxiv.org/abs/2610.05977v1`） | `arXiv:2610.05977v1` | 前缀可行集合、共享元数据、全精度保护与 logical/physical rate；§2–4、附录 E/G/A |
| Differentiable Bit-Widths（`https://arxiv.org/abs/2610.06026v1`）；MMAI-Laboratory/DBW（`https://github.com/MMAI-Laboratory/DBW/tree/a45f26e1d153ab6bc64dd27d056a3710a33f2e7a`） | `arXiv:2610.06026v1`；`a45f26e1d153ab6bc64dd27d056a3710a33f2e7a` | 成对分量的 rank/bit；`block_ap.py` 矩阵份额、`fake_dbw.py` 整数投影与 `real_dbw.py` 浮点恢复；细节见 DBW 方法页 |
| LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models（`https://arxiv.org/abs/2310.08659v4`） | `arXiv:2310.08659v4` | 前缀混合精度的名义标签与实际预算 |
| LUQ: Layerwise Ultra-Low Bit Quantization for Multimodal Large Language Models（`https://arxiv.org/abs/2509.23729v3`） | `arXiv:2509.23729v3` | — |
| BiLLM: Pushing the Limit of Post-Training Quantization for LLMs（`https://arxiv.org/abs/2402.04291v2`） | `arXiv:2402.04291v2` | — |
| SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression（`https://arxiv.org/abs/2306.03078v1`） | `arXiv:2306.03078v1` | 细粒度例外预算 |
| SqueezeLLM: Dense-and-Sparse Quantization（`https://arxiv.org/abs/2306.07629v4`） | `arXiv:2306.07629v4` | 敏感值与尾部保留 |

- The Undetected Damage of Quantization on Retrieval and How to Fix It（`https://arxiv.org/abs/2609.24322v1`），arXiv:2609.24322v1；排序稳定性、间隔分配与校准保证。
- RAMP（`https://arxiv.org/abs/2609.28262v1`），arXiv:2609.28262v1；CPU 混合精度、敏感度聚类与图切分成本。
- FoldQuantVLA（`https://arxiv.org/abs/2609.24433v1`），arXiv:2609.24433v1；共享输入折叠、目标投影位宽和编译基线。

- Same Bit Width, Different Outcomes: Post-Training Quantization of Text-to-Speech Across Architectures（`https://arxiv.org/abs/2609.28974v1`），v1；组件消融、质量判据与真实执行对照。
