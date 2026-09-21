---
title: SmoothQuant：迁移激活量化难度的 W8A8 方法
type: method
tags:
  - ptq
  - llm
  - activation-quantization
  - equivalent-transform
  - outliers
sources:
  - raw/papers/2026-09-21/mbq/paper.pdf
  - raw/papers/2026-09-21/smoothquant/paper.pdf
  - raw/repositories/2026-09-21/smoothquant/source/smoothquant/smooth.py
  - raw/repositories/2026-09-21/smoothquant/source/smoothquant/calibration.py
  - raw/repositories/2026-09-21/smoothquant/source/smoothquant/fake_quant.py
  - raw/repositories/2026-09-21/smoothquant/source/smoothquant/opt.py
  - raw/repositories/2026-09-21/smoothquant/source/smoothquant/ppl_eval.py
  - raw/repositories/2026-09-21/smoothquant/source/examples/export_int8_model.py
  - raw/repositories/2026-09-21/torch-int/source/torch_int/nn/linear.py
  - raw/repositories/2026-09-21/torch-int/source/torch_int/functional/quantization.py
paper_version: arXiv:2211.10438v7
code_commit: c61476d728e42ae0d8a35e7e78494edcac3237b5
verification: 全文含附录研读与定向代码核对；未运行模型实验
updated: 2026-09-22
---

# SmoothQuant：迁移激活量化难度的 W8A8 方法

SmoothQuant 面向同时量化权重与激活的 W8A8 推理。它利用输入通道缩放，把一部分激活范围差异转移到权重，再用适合整数矩阵乘的量化粒度执行。变换在量化前保持线性层函数，量化后是否准确取决于迁移强度、网格、校准分布和具体模型。

若继续追问“这些缩放能否直接由输出误差学习”，可读 [OmniQuant 的可学习等价变换](omniquant.md)：它以统计缩放初始化，再在 Transformer block 的重构目标下联合学习变换与裁剪。比较时需区分变换形式、优化变量和最终位宽/后端，不能把其 W4A4 模拟结果与本页 W8A8 实际推理结果直接比较。

本页依据 SmoothQuant v7（2024-03-29，论文标注 ICML 2023，以下简称 P），完整阅读 13 页、附录 A、10 图、11 表。v7 包含较晚加入的模型结果，不能将全部内容归为最初投稿时的实验。代码采用 `c61476d728e42ae0d8a35e7e78494edcac3237b5`（以下简称 C），以及按需查阅的 torch-int `65266db1eadba5ca78941b789803929e6e6c6856`（以下简称 T）。代码只做定向静态核对，未执行。

**实现层深入入口：** 平滑统计与融合的实际写法、torch-int 的 INT8 线性接口，以及动态激活量化与离线平滑的区别见 [SmoothQuant 的实现核对](../implementation/smoothquant-implementation.md)。

前置阅读：[均匀量化与分组](../fundamentals/quantization/uniform-quantization-and-groups.md)、[对角缩放与等价变换](../theory/diagonal-scaling-equivalent-transform.md)。本页保留理解方法所需的推导；更完整的内外维缩放解释见[执行路径页](../implementation/quantized-matmul-scaling-execution.md)。

## 1. 研究对象与真正的困难

P §1 的出发点是大模型存储与计算成本；图 1 描绘发表时期的模型大小和设备容量差距。与仅量化权重的 [GPTQ](gptq.md)、[AWQ](awq.md)相比，本方法要让矩阵乘的两个输入都适合 INT8。

P §3、图 2/4 观察到：在所测大模型的部分线性层，激活少数通道的幅度持续较大，而其他通道很小；对应权重相对平缓。统一 absmax 网格被大幅度值拉宽，普通通道可用的等级很少。论文给出的约 $2^8m_j/m$ 是通道最大幅度 $m_j$ 相对全张量最大幅度 $m$ 的有效等级估计，不是另一个实际编码位宽。

例如激活有一个值为 100 的通道，而另一通道约在 $[-1,1]$，INT8 对称步长约为 $100/127\approx0.787$，小通道只能落到少数等级。这一教学例子说明范围失衡，不证明任意离群值都应被裁掉。P 的方法尝试保留信息并改变表示范围。

P 表 1 的 OPT-175B 四任务平均准确率：FP16 71.6%，直接 per-tensor 激活量化 32.3%，per-token 31.7%，模拟 per-input-channel 71.4%。它说明这组设置下沿 token 分粒度未解决通道离群值；也说明不能把 per-token 一概称为一定更准确。

“离群值在固定通道持续出现”来自该论文的观察，不是任意模型、任意输入都成立的定理。后续模型使用不同 alpha/粒度，正说明不能照搬同一统计假设和超参数。

## 2. 为什么不直接按激活输入通道量化

沿用 P 的行 token 约定：$X\in\mathbb R^{T\times d}$、$W\in\mathbb R^{d\times r}$，$Y=XW$。它与本 Wiki 多数页面的 $WX$ 约定互为转置；PyTorch `Linear.weight` 存为 $r\times d$。

P 式 (1) 使用对称 INT8，$\Delta=\max|X|/127$，先除步长再最近舍入。实际编码范围和超出静态校准范围时的饱和规则须由实现明确；简化公式没有写出全部数值边界。

若激活每 token 一个步长 $\Delta_{X,t}$，权重每输出通道一个步长 $\Delta_{W,o}$，则

$$
Y_{to}\approx\Delta_{X,t}\Delta_{W,o}\sum_jq^X_{tj}q^W_{jo}.
$$

两个缩放都在求和维度之外，可在整数累加后施加。反过来，若激活有逐输入通道的 $\Delta_{X,j}$，它留在 $\sum_j$ 内，不能用一次普通整数 GEMM 后的行列缩放恢复。这是 P §3、式 (2)、图 3 所说的硬件映射困难，并非数学上绝对不能实现；更复杂内核或分解仍可能实现，但代价和接口不同。

## 3. 缩放如何在两侧分配难度

取正对角矩阵 $D=\operatorname{diag}(s)$，则精确算术下

$$
XW=(XD^{-1})(DW)=\widetilde X\widetilde W.
$$

定义输入通道统计 $a_j=\max_t|X_{tj}|$、$b_j=\max_o|W_{jo}|$，SmoothQuant 选择

$$
s_j=\frac{a_j^\alpha}{b_j^{1-\alpha}},\qquad 0\le\alpha\le1.
$$

当 $a_j,b_j>0$ 时，变换后的对应通道最大值为

$$
\widetilde a_j=a_j/s_j=(a_jb_j)^{1-\alpha},\qquad
\widetilde b_j=b_js_j=(a_jb_j)^\alpha.
$$

这给出 P 式 (4) 的直观解释：alpha=0 时把各权重输入通道最大值归一到 1；alpha=1 时把激活通道最大值归一到 1；alpha=0.5 时两侧对应通道最大值均为 $\sqrt{a_jb_j}$。这不意味着不同通道的范围全部相等，也不证明联合量化误差最优。零通道与数值下限按代码处理。[P §4、图 5]。

量化后的误差同时来自两侧。若 $E_X=Q(\widetilde X)-\widetilde X$、$E_W=Q(\widetilde W)-\widetilde W$，则

$$
Q(\widetilde X)Q(\widetilde W)-XW
=E_X\widetilde W+\widetilde XE_W+E_XE_W.
$$

这是对该方法的代数展开：改善激活网格并不保证总误差下降，因为权重误差也可能增大。P 图 10 对 OPT-175B/LAMBADA 的消融用 W8A16、W16A8、W8A8 分别观察两侧影响：约 0.4–0.6 是该实验的良好区域；alpha 太大时权重变难量化，太小时激活仍困难。这个区间不适用于所有模型。

**融合。** 若输入来自归一化输出 $X=\gamma\odot N(U)+\beta$，可把 $\gamma,\beta$ 除以 $s$，同时把后继权重的输入维度乘以 $s$。多个投影共享同一输入时必须使用相同缩放并同步修改；残差和非线性不能随意跨过。P §4 明确允许某些残差情形增加运行时缩放，所以“任何模型都零开销”超出证据。融合的详细条件见[对角缩放页](../theory/diagonal-scaling-equivalent-transform.md)。

## 4. 量化配置与算子范围

P 表 2 的早期配置如下，三者均使用 per-tensor 权重量化：

| 配置 | 激活粒度 | 激活步长何时确定 | 取舍 |
|---|---|---|---|
| O1 | per-token | 运行时动态 | 更细范围，需要动态统计 |
| O2 | per-tensor | 运行时动态 | 参数较少，仍需动态统计 |
| O3 | per-tensor | 校准时静态 | 运行时无需重新求范围，依赖校准代表性 |

平滑因子 $s$ 和 INT8 量化步长 $\Delta$ 是不同参数：O1/O2 的 $s$ 仍离线确定；动态的是激活量化步长。O3 的静态参数应在平滑后的模型上校准。

P 图 6 令线性层和 attention BMM 使用 INT8 输入，LayerNorm、Softmax、残差等保留浮点计算。W8A8 因而不是整张计算图、所有偏置和累加输出一律 INT8。P 表 7 后续模型使用 **per-token 激活 + per-output-channel 权重**，这与表 2 O1 的 per-tensor 权重不同，不能统一标成 O1。

## 5. 校准、实现与代码差异

P §5.1 使用 Pile 的 512 个随机句子校准平滑统计及静态量化步长；alpha 在 Pile validation 子集上做快速网格搜索。OPT/BLOOM 用 0.5，GLM-130B 用 0.75。GLM 静态步长校准还裁去最高 2% tokens 的相关极值；本地所读通用代码没有复原这一专门分支。每个下游任务共用校准后的模型，不能把用于选 alpha 的数据继续称为独立验证证据。

本地 C 的具体路径：

1. `calibration.py:13`收集每个线性层输入沿 token/sample 的逐通道最大绝对值。默认 512 样本、最大长度 512，shuffle seed=42；句子是截断上限，不能与 GPTQ 的固定 2048-token 段数直接等同。代码默认值不证明 P 全部实验采用同样长度。
2. `smooth.py:19`对共享 LayerNorm 的所有 FC 合并取输入通道权重 absmax，设 1e-5 下限，计算缩放，再除归一化参数、乘所有 FC 权重。RMSNorm 版本只除其 weight。这里与 AWQ 的平均绝对激活统计不同。
3. `smooth.py:75`按模型图结构处理：OPT 为 Q/K/V 和 FC1；Llama/Mistral 为 Q/K/V、gate/up；Mixtral 包括路由 gate 与各 expert 的 w1/w3。并非所有线性层都单独做一次平滑；共享消费者必须一起变换。
4. `export_int8_model.py:28`先平滑，再通过 `get_static_decoder_layer_scales` 获取输入/输出尺度，最后调用 `from_float` 导出真正 INT8 模型或导出给 FT 使用的材料。

**模拟与实际 INT8 是两条路径。** C `fake_quant.py:48`的 `W8A8Linear` 用浮点张量保存网格值，调用浮点 `F.linear`，只能说明模拟量化的取值，不是整数内核。其 absmax 工具在当前输入上计算范围，不能冒充 O3 静态执行。

C `opt.py:15`的真实路径调用 torch-int 的 INT8 线性层、BMM 和归一化融合接口。T `nn/linear.py:16`确实保存 `torch.int8` 权重并调用 `_CUDA`；`from_float` 用输入、权重、输出尺度构造重缩放系数。T 的某些投影使用 INT8 bias/output，FC2/out_proj 则保留 FP32 bias/output；P 图 6 的 FP16 示意不是这些代码张量 dtype 的逐项保证。T `linear.py:182`。

真实 OPT attention 将 INT8 K/V 投影输出拼接为 cache，QK BMM 产生浮点结果，Softmax 后将概率乘 127 舍入为 INT8，再做 PV BMM。C `opt.py:122`、`opt.py:188`。这是该路径的代码证据，不能推出全部后端或后来模型也采用相同 KV 格式。Llama-like fake quant 默认 `quantize_bmm_input=False`，PPL 入口则显式设为 True，并以 BF16 加载；默认演示和评测入口不可混用。C `fake_quant.py:183`、`ppl_eval.py:65`。

本轮仅确认这些接口和数据流，未审查完整 CUDA/CUTLASS/FT 内核，也未将 T 的当前 commit 认定为 P 的实验环境。

## 6. 模型质量与消融证据

### 6.1 OPT/BLOOM/GLM 与任务口径

P 表 3 的 OPT-175B 七任务平均准确率与 WikiText PPL：

| 方法 | 七任务平均准确率 ↑ | WikiText PPL ↓ |
|---|---:|---:|
| FP16 | 66.9% | 10.99 |
| 直接 W8A8 | 35.5% | 93080 |
| ZeroQuant | 35.8% | 84648 |
| LLM.int8() | 66.7% | 11.10 |
| Outlier Suppression | 36.0% | 96151 |
| SmoothQuant O1 | 66.5% | 11.11 |
| SmoothQuant O2 | 66.4% | 11.14 |
| SmoothQuant O3 | 66.8% | 11.17 |

七任务为 LAMBADA、HellaSwag、PIQA、WinoGrande、OpenBookQA、RTE、COPA。这里保留 P 自行实现/配置的基线结果，不代表这些方法其他版本或所有模型上的能力。比如 P 为 ZeroQuant 尝试保留 self-attention 输入 FP16，仍未解决本表退化。

这两行基线现在有可直接对照的原始来源。[LLM.int8()](llm-int8.md) 的向量级量化与混合精度分解在 175B 以内模型上报告恢复位宽基线，其代价是少数离群维度走 16 bit；[ZeroQuant](zeroquant.md) 用逐组权重、逐 token 动态激活与逐层蒸馏达到 INT8，并在 GPT-NeoX 20B 上把退化的来源定位到自注意力输入激活、因而保留该处为 FP16。P 在本表为 ZeroQuant 保留同一处高精度，与该文的归因一致，但这不表示两个实现的其余配置相同。两篇的零样本指标、模型族与评测协议都不同，因此「SQ 高于该行基线」只能在这张表内部解释。

P 表 4 则使用 **四任务平均**：OPT/BLOOM 为 WinoGrande、HellaSwag、PIQA、LAMBADA，GLM 为 LAMBADA、MMLU、MNLI、QNLI。GLM 改用这些数据集是为避开作者指出的部分训练集重叠。不同列不可直接比较模型能力。

| 模型与各自四任务集 | FP16 | O1 | O2 | O3 |
|---|---:|---:|---:|---:|
| OPT-175B | 71.6% | 71.2% | 71.1% | 71.1% |
| BLOOM-176B | 68.2% | 68.3% | 68.4% | 67.4% |
| GLM-130B | 73.8% | 73.7% | 72.5% | 72.8% |

静态 O3 有实际退化；P 将 BLOOM 的变化归因于静态统计与评测激活差异，这是作者解释，不是本项目验证。图 7 给出不同 OPT 规模的整体趋势，但不能把“保持准确率”理解为逐项分数完全相同。

### 6.2 指令模型与后续架构

P 表 5：OPT-IML-30B 的 FP16/O3 为 LAMBADA 69.12%/69.77%、WikiText 14.26/14.37。它支持该指令模型在这两个指标上的结果，不等于验证开放对话、指令遵循或安全能力。

P 表 6：LLaMA 使用序列长度 512、per-token 激活、alpha=0.8。7B/13B/30B/65B 的 FP16→W8A8 PPL 分别为 11.51→11.56、10.05→10.08、7.53→7.56、6.17→6.20。不能与长度 2048 的另一张表直接比较。

P 表 7 使用 WikiText-2、长度 2048、per-token 激活与 per-output-channel 权重：

| 模型 | FP16 PPL | W8A8 PPL | alpha |
|---|---:|---:|---:|
| Llama-2-7B | 5.474 | 5.515 | 0.85 |
| Llama-2-13B | 4.950 | 4.929 | 0.85 |
| Llama-2-70B | 3.320 | 3.359 | 0.90 |
| Falcon-7B | 6.590 | 6.629 | 0.60 |
| Falcon-40B | 5.228 | 5.255 | 0.70 |
| Mistral-7B | 5.253 | 5.277 | 0.80 |
| Mixtral-8x7B | 3.842 | 3.893 | 0.80 |

不同 alpha 和模型结构必须保留。表中没有这些模型的端到端速度，因此不能把 OPT 的加速数字移到 Llama/Mixtral。

P 表 9 的 MT-NLG 530B 四任务平均同为 73.1%，但 HellaSwag 为 62.1%→60.4%，LAMBADA 为 76.6%→77.2%。平均不变不表示所有能力无损；P 的“lossless”应理解为作者对有限指标小变化的表述，没有统计等价证明。

## 7. 实现加速与测量边界

### 7.1 Context/prefill 与后端

P §5.3 的 context 测量为 batch=4，一次生成整批输入的全部隐藏状态，记录峰值显存，硬件为 A100 80GB。PyTorch 路径主要展示单 GPU；FT 路径支持 tensor parallel。后端与 GPU 数分别记录。

| 原文位置与场景 | FP16 → SmoothQuant O3 | 支持的解释 |
|---|---|---|
| 图 8 / 表 11，PyTorch，OPT-30B，batch=4、长度 256、1 GPU | 343.0ms → 227.6ms | 约 1.51×；这是 context 延迟 |
| 图 9，FT，OPT-30B，batch=4、长度 512、1 GPU | 图中舍入显示 186ms → 119ms | 约 1.56×，基线同为 FT |
| 图 9，OPT-66B、长度 512 | 2 GPU 的 236ms → 1 GPU 的 229ms | 减卡后类似延迟，不是同卡数算子对照 |
| 图 9，OPT-175B、长度 512 | 8 GPU 的 432ms → 4 GPU 的 366ms | 减卡后延迟改善，保留并行条件 |

PDF 图 8 最终可见内容是 OPT-13B/30B、长度 128/256/512；抽取文本还包含绘图文件中不可见的其他数值，本页不把它们当成该图已展示的证据。图 9 的全局显存近似减半，但这依赖图示形状、模型与后端，不是所有长上下文条件下的保证。

P 表 11 的 OPT-30B、batch=4、长度 256：O1/O2/O3 为 246.7/240.2/227.6ms，展示更粗粒度、静态统计在该实现中的成本收益。这不取消第 6 节中的精度代价。

### 7.2 Decode 的报告与协议缺口

P 表 8 的代表结果为：

| 模型与 GPU 数 | batch / SeqLen | FP16 / SQ latency（ms，原表字段） | 原表速度比 |
|---|---|---|---:|
| OPT-30B，1 GPU | 1 / 512 | 422 / 314 | 1.35× |
| OPT-30B，1 GPU | 16 / 512 | 2488 / 1753 | 1.42× |
| OPT-175B，8 GPU | 1 / 512 | 426 / 359 | 1.19× |
| OPT-175B，8 GPU | 16 / 1024 | 4133 / 3231 | 1.28× |

原正文称 per-token decoding latency，但表格与配套 TeX 仅写 Latency，未充分说明 SeqLen 是何种长度、prompt/输出组合、累计或平均方式、该表后端以及多 GPU 显存的聚合口径。本页保留原数值和作者速度比，不把它们自行换算成 tokens/s，不与 GPTQ 表 6 的单 token 测量横比。

表 8 还显示 OPT-30B batch=16、SeqLen=1024 时 FP16 OOM 而 SQ 可运行；显存收益在部分大 batch 项只有约 1.5–1.7×，不是严格 2×。OOM 不应换算成无限加速。

### 7.3 530B 的主要收益是部署容量

P 表 10 的 MT-NLG 530B 从 16 张 A100 80GB FP16 改为 8 张 INT8。长度 128 时为 232→253ms，略慢；长度 1024 时为 1707→1689ms，接近。总显存分别为 1040→527GB 和 1095→570GB。结合表 9，证据支持在较小精度变化下减半设备数量与近似延迟，不支持“所有输入长度都加速”。

## 8. 局限、跨方法关系与未验证内容

- **等价不是量化无损。** 平滑改变两侧量化误差，且校准最大值不保证覆盖部署分布。静态参数、裁剪、alpha 的适配都影响结果。
- **对象与后端决定加速。** P 附录 A 明确说明 GPTQ 的当时内核主要面向单 token 生成，SQ 使用 FT；直接比较会混入后端优势。附录关于小 batch/batched 场景优劣主要是分析，未给出统一实现的直接对照。
- **同族缩放的后续发展。** [QServe](qserve.md) 的 SmoothAttention 把同一思想用到 KV cache 的 Key 上，并给出一条与本文不同的经验：块输出模块的迁移强度应接近 0，而输入模块仍需平滑。这说明缩放公式相同不代表不同图位置的超参数可以通用。
- **KV cache 有条件。** 附录指出长上下文/大 batch 中 KV 占比会变大；本页给出真实 OPT 路径的 INT8 cache 证据，仍不能把所有激活量化自动等同于 KV cache 量化。
- **W4A4 组合是未来设想。** 附录提出结合 GPTQ 改善权重量化，未在本篇实现或验证；不能写为 SmoothQuant 已完成的能力。
- **证据范围。** 基线、训练数据污染处理和支持的模型均按 P 的当时条件解释。LLM.int8() 与 ZeroQuant 已完成全文研读（见 [LLM.int8()](llm-int8.md)、[ZeroQuant](zeroquant.md)），但 P 所使用的仍是其当时实现与配置，本页不据此断言两者之间的精度排序；Outlier Suppression 等其余基线仍未全文研读，也未独立核验其实现。
- **代码与实验不同版本。** C 的 fake quant、真实 OPT 和 PPL 入口并不共享完全相同的格式/路径；T 的部分量化工具有不同的 clamp/零范围处理。当前接口可读不等于数值行为已验证。
- **尚未复现。** 未下载模型或校准数据，未运行 CUDA、FT、模型质量或性能测试；全部论文附录已读不表示代码仓库全部已审查。

三者的统一问题比较见[AWQ、GPTQ 与 SmoothQuant：对象、机制与证据比较](awq-gptq-smoothquant-comparison.md)。本轮不据这些历史结果预设后续研究的选题或实验结论。

MBQ v2 的 W4A8 实验沿用两侧量化与通道缩放思路，但用模态加权重构选择尺度。其表 4 中，单纯将 SmoothQuant 校准数据换为 COCO 并未自动提高 LLaVA-OneVision-7B 的结果。这提示需要检查校准目标、mask、归一化和实现设置，而不能仅据某组基线退化推断 SmoothQuant 对所有视觉语言模型无效；具体数据与条件见 [MBQ 的消融与实现](mbq.md)。

## 来源身份

下表用于在没有本地资料库时辨识来源；具体论述的章节、公式、图表或代码位置见正文。

| 来源 | 版本或快照 | 说明 |
| --- | --- | --- |
| [MBQ: Modality-Balanced Quantization for Large Vision-Language Models](https://arxiv.org/abs/2412.19509v2) | `arXiv:2412.19509v2` | — |
| [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438v7) | `arXiv:2211.10438v7` | — |
| [mit-han-lab/smoothquant.git](https://github.com/mit-han-lab/smoothquant/tree/c61476d728e42ae0d8a35e7e78494edcac3237b5) | `c61476d728e42ae0d8a35e7e78494edcac3237b5` | — |
| [Guangxuan-Xiao/torch-int](https://github.com/Guangxuan-Xiao/torch-int/tree/65266db1eadba5ca78941b789803929e6e6c6856) | `65266db1eadba5ca78941b789803929e6e6c6856` | — |
