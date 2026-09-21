---
title: AWQ：激活感知的权重量化
type: method
tags:
  - ptq
  - weight-quantization
  - equivalent-transform
  - calibration
sources:
  - raw/repositories/2026-09-21/llm-awq/source/awq/entry.py
  - raw/repositories/2026-09-21/llm-awq/source/awq/kernels/csrc/quantization_new/gemm/gemm_cuda.cu
  - raw/repositories/2026-09-21/llm-awq/source/awq/kernels/csrc/quantization_new/gemv/gemv_cuda.cu
  - raw/papers/2026-09-21/mbq/paper.pdf
  - raw/papers/2026-09-21/awq/paper.pdf
  - raw/papers/2026-09-21/quantization-white-paper/paper.pdf
  - raw/papers/2026-09-21/smoothquant/paper.pdf
  - raw/papers/2026-09-21/gptq/paper.pdf
  - raw/repositories/2026-09-21/llm-awq/source/awq/quantize/auto_scale.py
  - raw/repositories/2026-09-21/llm-awq/source/awq/quantize/auto_clip.py
  - raw/repositories/2026-09-21/llm-awq/source/awq/quantize/quantizer.py
  - raw/repositories/2026-09-21/llm-awq/source/awq/quantize/pre_quant.py
  - raw/repositories/2026-09-21/llm-awq/source/awq/quantize/qmodule.py
  - raw/repositories/2026-09-21/llm-awq/source/awq/utils/calib_data.py
paper_version: arXiv:2306.00978v6
code_commit: d6e797a42b9ef7778de8ee2352116e0f48a78d61
verification: 原文研读与定向代码核对；补充材料缺件；未运行模型实验
updated: 2026-09-22
---

# AWQ：激活感知的权重量化

AWQ 用校准激活判断哪些输入通道需要更多保护，再通过等价缩放与权重裁剪减轻低比特量化的损失。它主要面向 LLM 的 weight-only PTQ，通常使用 INT3／INT4 权重和 16-bit 激活。配套系统 TinyChat 通过反量化融合、打包和其他 kernel 优化，把压缩后的权重用于实际推理。

理解 AWQ 的关键不是记住“保护 1% 权重”，而是理解：输入如何放大权重误差、通道缩放为何可能改善重要通道、为何也会伤害同组其他通道，以及怎样用校准数据选择折中。

本页依据 AWQ 原论文 v6（2026-04-25，以下称 P0），及官方代码 commit `d6e797a42b9ef7778de8ee2352116e0f48a78d61`（以下称 C0）。P0 的全部 15 页和全部编号图表已研读；C0 只定向阅读相关实现，没有执行。源码中的潜在附录缺件和其他证据限制见末节。本文给出的质量与速度数值均为作者报告，不是本项目复现。

建议阅读顺序：[线性层与输入通道](../fundamentals/operators/linear-layer-input-channel.md) → [均匀量化与分组](../fundamentals/quantization/uniform-quantization-and-groups.md) → 本页；[对角缩放与等价变换](../theory/diagonal-scaling-equivalent-transform.md)进一步解释共性原理与计算图条件。

本页讲机制、论文数值与证据边界；**打包约定、内核入口、以及在 vLLM 与 SGLang 里如何加载同一份 AWQ 权重**见 [AWQ 的实现核对](../implementation/awq-implementation.md)。

## 1. 研究对象与动机

P0 关注端侧部署的两类困难：模型权重放不进设备内存，以及 batch=1 生成阶段读取大量权重带来的带宽限制。低比特权重能减少存储和读取量，但直接 round-to-nearest（RTN）可能损害模型质量；保留少数 FP16 权重虽可保护质量，却增加混合精度数据组织与运算复杂度。P0 §1—3.1、图 1—2、表 1。

AWQ 的对象需要限定清楚：

- **量化目标：**主要为线性层权重；activation-aware 指利用激活信息，并非把激活也量化到 4 bit，更不是 KV cache 量化。
- **量化范式：**从已训练模型出发进行 PTQ，使用校准数据而不做权重梯度训练。
- **典型设置：**INT3／INT4、group size=128；P0 的极低比特组合实验另用 INT2-g64。
- **推理场景：**系统加速分析以权重访问占主导的端侧生成场景为中心，不能自动外推到长上下文、大 batch 或所有 prefill 负载。

“把重要权重保留为 FP16”是提出方法前的对照实验。AWQ 最终通过缩放再量化来保护重要通道，不要求固定留下 1% FP16 权重；也不能据此推断实际模型中的嵌入、归一化参数和所有辅助张量一律 INT4。

## 2. 符号、通道和基本量化器

采用 P0 的列向量约定：

| 符号 | 形状／含义 |
|---|---|
| $W$ | $C_{\mathrm{out}}\times C_{\mathrm{in}}$，原权重 |
| $X$ | $C_{\mathrm{in}}\times T$，校准 token 的输入特征 |
| $Y=WX$ | $C_{\mathrm{out}}\times T$，暂略偏置的输出 |
| $s$、$D=\operatorname{diag}(s)$ | 长度 $C_{\mathrm{in}}$ 的正通道缩放及其对角矩阵 |
| $N$、$g$ | 权重位宽、量化 group size |
| $\Delta$、$z$ | 每组量化步长和 zero-point；不同于通道缩放 $s$ |
| $Q(W)$ | 落到量化网格后恢复到实数域的权重，用于描述数值计算 |

输入通道 $j$ 对应 $W$ 的第 $j$ 列；C0 的 group-wise 量化则在每个输出行内，沿输入维度将连续 $g$ 个权重分组。因此“按输入通道缩放”和“按组共享量化参数”是两个同时存在的维度。

P0 式 (1) 用如下简化式分析误差：

$$
Q(w)=\Delta\operatorname{Round}(w/\Delta),\qquad
\Delta=\max(|w|)/2^{N-1}.
$$

这里 $w$ 表示一组权重；式中没有完整给出有限整数范围和饱和处理。C0 的默认路径则使用仿射量化，整数范围 $[0,2^N-1]$：

$$
\Delta=\frac{\max(w_{\max}-w_{\min},10^{-5})}{2^N-1},\quad
z=\operatorname{clip}(-\operatorname{round}(w_{\min}/\Delta),0,2^N-1),
$$
$$
q=\operatorname{clip}(\operatorname{round}(w/\Delta)+z,0,2^N-1),\quad
\widehat w=\Delta(q-z).
$$

C0 `quantizer.py:61`。分析式和实现式用途不同，不应把一者无条件替代另一者。均匀量化的一般定义、舍入与裁剪的区别见[基础页](../fundamentals/quantization/uniform-quantization-and-groups.md)。

## 3. 为什么观察激活能帮助权重量化

设量化引入权重误差 $E=\widehat W-W$。单个输入 $x$ 的输出误差为

$$
\widehat y-y=Ex=\sum_j E_{:,j}x_j.
$$

即使权重误差幅度相同，较大的输入特征也可能放大它的输出影响。这是对 P0 §3.1 动机的代数解释，不是平均绝对激活最优性的证明；通道之间的相关性和误差抵消也会影响最终结果。

P0 表 1 使用 INT3-g128，在 OPT 模型中按激活、权重或随机方式选择部分通道保留 FP16。以 OPT-6.7B 为例：

| 设置 | WikiText PPL，越低越好 |
|---|---:|
| 原 FP16 | 10.86 |
| RTN | 23.54 |
| 按激活选 0.1% 通道保留 FP16 | 11.58 |
| 按激活选 1% 通道保留 FP16 | 11.39 |
| 按权重选 1% 通道保留 FP16 | 22.37 |
| 随机选 1% 通道保留 FP16 | 24.23 |

这支持该实验范围内的激活感知选择；没有证明所有模型、数据和位宽都只需要同一个比例。下一步的目标是在更规整的低比特格式中得到相近保护，而非直接部署这一混合精度对照。P0 第 4 页表 1。

## 4. 缩放怎样改变误差，为什么需要搜索

### 4.1 精确的等价变换

对正对角矩阵 $D$，精确算术下有

$$
WX=(WD)(D^{-1}X).
$$

有偏置时两边保留相同偏置。变换改变数值分布，未量化的函数不变；量化后 $Q(WD)D^{-1}X$ 一般不等于 $WX$。P0 式 (4) 正是要选择能减小这个差异的缩放。

### 4.2 单项误差的近似分析

设一个权重 $w$ 与输入 $x$ 相乘；把该权重乘以 $s>1$，输入除以 $s$。定义有符号舍入残差 $r(u)=\operatorname{Round}(u)-u$，则根据 P0 式 (1)—(3)：

$$
e=\Delta r(w/\Delta)x,\qquad
e'=\frac{\Delta'}{s}r(ws/\Delta')x.
$$

$\Delta'$ 是缩放后的组步长。如果缩放没有显著改变组最大值，且归一化舍入残差的统计幅度近似不变，则误差幅度比例可近似为

$$
\frac{\Delta'}{\Delta}\frac1s.
$$

这是统计近似，**不是每个元素都成立的精确误差比**。原文所述 0.25 对应均匀残差模型下的平均绝对误差；有符号残差位于 $[-0.5,0.5]$，其均值不能写为 0.25。此外，保留 FP16 只是不引入这里讨论的低比特量化，不等于浮点运算没有误差。

### 4.3 重要通道与其他通道的折中

若增大 $s$ 使组步长上升，未被放大的权重也会使用更稀疏的网格。P0 表 2 对 OPT-6.7B 的 1% 重要通道进行缩放，报告如下：

| $s$ | 1 | 1.25 | 1.5 | 2 | 4 |
|---|---:|---:|---:|---:|---:|
| 步长变化比例 | 0% | 2.8% | 4.4% | 8.2% | 21.2% |
| 平均 $\Delta'/\Delta$ | 1 | 1.005 | 1.013 | 1.038 | 1.213 |
| 重要通道的近似误差比例 | 1 | 0.804 | 0.676 | 0.519 | 0.303 |
| WikiText-2 PPL | 23.54 | 12.87 | 12.48 | **11.92** | 12.36 |

$s=4$ 时近似重要通道误差更小，但整体 PPL 变差。不能把“放大重要通道”写成无限增大缩放的理由。P0 第 4 页表 2。

### 4.4 从统计量到候选比较

P0 式 (4)—(5) 的目标和受限搜索族为

$$
\mathcal L(s)=\left\|Q(WD)D^{-1}X-WX\right\|,\quad
a_j=\operatorname{mean}_t|X_{jt}|,\quad
s_j=a_j^\alpha.
$$

原文未在该范数上指定下标；C0 使用所检查模块输出的均方误差。P0 描述在 $[0,1]$ 上用 20 点网格搜索 $\alpha$：0 表示不作通道差异缩放，较大的指数强化相对激活幅度的差异。搜索后再做权重裁剪。

**当前代码细节：**C0 `auto_scale.py:109` 实际枚举 $\alpha\in\{0,0.05,\ldots,0.95\}$，不包含 1；候选缩放先设 $10^{-4}$ 下限，再除以 $\sqrt{\max(s)\min(s)}$。它把候选权重缩放、模拟量化并除回缩放值，运行参考模块，比较输出 MSE，然后恢复原始参数。每一候选都必须从相同参考状态出发。

检查模块可为单个线性层、attention 或 MLP，取决于模型适配。因而不能把所有搜索都写成逐权重 MSE，也不能把这一受限网格的最优候选称为全局最优缩放。对未出现或幅度很小的通道，代码的数值下限承担避免退化的作用，其具体行为属于该版本。

## 5. 裁剪、融合和可执行流程

### 5.1 裁剪目标不是纯权重 MSE

P0 仅简述通过 weight clipping 减小 MSE，以下精确定义来自 C0 `auto_clip.py:11`。对于输出通道 $o$、输入分组 $g$、token $t$：

$$
y_{ogt}=\sum_{j\in g}w_{oj}x_{jt},\qquad
\ell_{og}=\operatorname{mean}_t(\widehat y_{ogt}-y_{ogt})^2.
$$

当前实现将该组权重裁到对称阈值区间，再执行实际配置的量化器，用输入计算该组点积并比较误差。每个输出通道、每个分组分别选阈值。因为没有先把各组贡献相加再求平方，它忽略不同组误差之间的交叉项，是完整层输出误差的代理目标。

默认 `n_grid=20,max_shrink=0.5` 实际尝试比例 $1.00,0.95,\ldots,0.55$ 共 10 个候选，不包含 0.50。代码按名称跳过 q/k/query/key/Wqkv 等层；输入 token 抽样、输出通道分批和分组整除也有形状约束。不能推断任意矩阵都可直接套用。这里未运行代码验证边界输入。

### 5.2 等价缩放如何落到计算图

有条件时，把输入逆缩放合入前驱：归一化输出的仿射参数除以 $s$，后继权重列乘以 $s$；线性前驱还需同步处理相关权重行和偏置。共享输入与残差分支需要一起维护，不能只修改某一个消费者看到的公式。

一般非线性并不满足缩放可交换性；C0 也使用 `ScaledActivation` 在激活后显式相除，并非所有缩放都能无开销融合。Llama 分支的 attention 输出缩放还检查相关投影权重形状是否相等。相关定位：C0 `auto_scale.py:34`、Llama 适配 `auto_scale.py:215`、`apply_scale` 函数 `auto_scale.py:449`。更一般的推导见 [等价变换页](../theory/diagonal-scaling-equivalent-transform.md)。

### 5.3 从输入到量化产物

以 C0 主流程为例，区分模型参数、缓存特征、缩放结果和最终打包权重：

1. **输入与设置。**提供已训练模型、tokenizer、校准数据、位宽、group size 和 zero-point 配置；先固定版本与目标模型。
2. **收集激活。**按 Transformer block 缓存相关线性层输入及必要的 attention 参数，保存未施加本轮缩放／裁剪时的 block 输出用于后续 block。
3. **搜索缩放。**计算通道平均绝对激活，逐候选比较参考模块的输出误差，保留最佳缩放；候选间恢复原参数。
4. **应用缩放。**按计算图规则调整前驱与后继参数，同时对用于裁剪的缓存输入作相应逆缩放。
5. **搜索并应用裁剪。**对允许裁剪的层，按输入分组的输出代理误差选择阈值，修改浮点权重范围。
6. **保存预处理结果。**`run_awq` 返回带模块名的 `scale` 和 `clip` 记录；`apply_awq` 可将记录应用到相应模型。它们本身不是 INT4 checkpoint。
7. **选择模拟或实际量化路径。**模拟路径仍用浮点张量承载量化值；实际路径生成整数权重与量化参数，替换为能够消费压缩布局的模块。

C0 `pre_quant.py:102`、quantizer.py:106。C0 在当前 block 的缩放／裁剪前就缓存下一 block 输入，因此不能把它写成自动用前面所有量化误差重新传播校准数据的流程。

当前实际量化路径要求 zero-point 配置，`WQLinear` 只实现 4 bit，并有输入分组、输出通道和打包对齐约束。模块保存 `qweight`、`scales`、`scaled_zeros` 和可选 bias；`qweight` 的 int16 是打包容器，不代表每个原始权重仍占 16 bit。当前 forward 依据输入 token 数是否小于 8 分派到相应 GEMV／GEMM CUDA 入口。这些都是版本相关实现，不是 AWQ 数学定义。quantizer.py:125、qmodule.py:78。本次进一步核对该调用：新版 GEMV 只分派 g128，GEMM 主机分支也固定 `G=128`；模拟量化支持其他分组不能推导出此真实路径同样支持。新旧内核的布局区别见实现页。底层完整循环、布局解包数值及运行正确性尚未验证。

### 5.4 校准数据不是一个默认数字

P0 §5.1 给出从 Pile 取小校准集、20 点搜索；图 8 的消融明确每条序列为 2048 tokens。C0 `run_awq` 默认 `n_samples=512,seqlen=512`，数据函数筛选文本后拼接再切块；输入文本条数不等于最终块数。CLI `entry.py` 又显式覆盖为 `n_samples=128,seqlen=512`，因此函数默认、实际调用与论文实验是三个层次。具体产物和加载顺序见 [实现页](../implementation/awq-implementation.md#2-原仓库从搜索到加载的状态变化)。这些代码设置不能填回原文所有实验。calib_data.py:5。

## 6. 与相关方法的关系

| 方法 | 相关优化机制 | 与 AWQ 的关系 |
|---|---|---|
| RTN | 直接按既定网格最近舍入 | 是重要基线；分组本身已可改善效果，不能只与很弱的无分组基线比较 |
| GPTQ | 层输出重构；利用输入二阶信息更新尚未量化的权重来补偿误差 | 与 AWQ 的缩放候选搜索不同；两者都使用校准数据，可以组合 |
| SmoothQuant | 可逆对角缩放，在 W8A8 中将激活量化困难部分转移到权重 | 共享变换思路，但量化对象、统计量和目标不同，不能直接复制缩放公式 |

GPTQ v2§3—4 定义 $\min_{\widehat W}\|WX-\widehat WX\|^2$，二阶信息来自 $2XX^{\mathsf T}$ 及数值稳定处理，并非整个语言任务损失的 Hessian。SmoothQuant v7第 4 页式 (4) 使用通道最大激活和权重构造缩放，而 P0 使用平均绝对激活。

因此 P0 所称“无 reconstruction”应结合其不采用 GPTQ 式权重补偿机制理解，不能进一步写成“不比较输出”“不优化任何校准目标”或“不会过拟合”。AWQ 首轮仅局部阅读这两篇依赖；2026-09-14 后续已分别完成全文研读，完整机制与证据见 [GPTQ](gptq.md)、[SmoothQuant](smoothquant.md)。跨方法的对象和比较边界见[三种方法比较](awq-gptq-smoothquant-comparison.md)。

在视觉语言模型中，相同缩放恒等式也可以使用不同的重构评价规则。[MBQ](mbq.md) 用回答/视觉位置的梯度统计加权缩放搜索；其表 4 还区分了图文校准与加权目标的作用。不能把 MBQ 特定实验中的 Pile 基线表现视为 AWQ 无法使用图文校准的固有限制。方法之间真正变化的维度见 [重要性加权路线的比较](../research/mbq-vlmq-comparison.md)。

## 7. 质量证据、消融与局限

### 7.1 缩放能否代替混合精度保护

P0 表 3 的 OPT、INT3-g128 消融如下，指标为 PPL，越低越好：

| 方法 | 1.3B | 2.7B | 6.7B | 13B | 30B |
|---|---:|---:|---:|---:|---:|
| FP16 | 14.62 | 12.47 | 10.86 | 10.13 | 9.56 |
| RTN | 119.47 | 298.00 | 23.54 | 46.04 | 18.80 |
| 1% FP16 | 16.91 | 13.69 | 11.39 | 10.43 | 9.85 |
| $s=2$ | 18.63 | 14.94 | 11.92 | 10.80 | 10.32 |
| AWQ | 16.32 | 13.58 | 11.39 | 10.56 | 9.77 |

AWQ 接近混合精度保护，但不是每一格都更好，也没有完全恢复 FP16。此表未充分隔离每个实现细节的独立贡献，不能用它精确估计裁剪、归一化等各自带来多少收益。

### 7.2 不同模型的语言建模质量

P0 表 4 比较 LLaMA／Llama-2 7—70B、INT3/4-g128 的 WikiText-2 PPL。以 Llama-2 为例：

| 设置 | 7B | 13B | 70B |
|---|---:|---:|---:|
| FP16 | 5.47 | 4.88 | 3.32 |
| INT3 RTN | 6.66 | 5.52 | 3.98 |
| INT3 GPTQ-R | 6.42 | 5.41 | 3.86 |
| INT3 AWQ | 6.24 | 5.32 | 3.74 |
| INT4 RTN | 5.73 | 4.98 | 3.46 |
| INT4 GPTQ-R | 5.63 | 4.99 | 3.43 |
| INT4 AWQ | 5.60 | 4.97 | 3.41 |

原表还列出未 reorder 的 GPTQ 和 LLaMA 系列，AWQ 在这些已报告设置下取得更低 PPL；这是作者结果，不是对所有模型或所有后端的排名。表中没有重复试验区间，小差距不能直接解释为稳定优势。

P0 表 5 的 Mistral-7B-Instruct-v0.2：FP16/INT4/INT3 PPL 分别为 4.14/4.30/4.83；Mixtral-8x7B-Instruct-v0.1 为 5.94/6.05/6.52。该表支持这些架构可以进行 AWQ 量化，但没有同表其他量化方法的竞争比较。

### 7.3 指令、多模态、代码与数学任务

| 证据位置 | 对象与条件 | 作者报告与解释 |
|---|---|---|
| 图 5、§5.2 | Vicuna 7/13B，INT3-g128；80 问题、两种展示顺序共 160 trials，GPT-4 评价 | AWQ 相比 RTN/GPTQ 有更多对 FP16 的获胜案例；仍有不少失败案例。顺序交换减轻偏差，不证明评审完全无偏 |
| 表 6 | OpenFlamingo-9B，COCO 5k 样本；0/4/8/16/32-shot；仅量化语言部分 | 32-shot FP16 CIDEr=81.70，INT4 AWQ=80.53，INT3 AWQ=74.47；仍分别下降 1.17 和 7.23 |
| 表 7 | VILA 7/13B，INT4-g128，11 个视觉语言基准 | 质量整体接近；例如 VILA-7B VizWiz 59.6→57.8，MM-Vet 35.1→35.9，既有下降也有上升，不能严格称为无损 |
| 表 8 | CodeLlama-7B-Instruct，MBPP，INT4-g128 | FP16→AWQ 的 pass@1 为 38.53→40.64，pass@10 为 49.77→49.25；缺少方差信息，不推断量化提升编程能力 |
| 表 8 | Llama-2 7/13/70B，GSM8K，INT4-g128 | FP16 为 13.87/26.16/56.41，AWQ 为 13.57/25.25/56.40，结果接近但不完全相同 |
| 图 6—7 | LLaVA-13B 视觉推理、OpenFlamingo INT4-g128 4-shot 图像描述 | 已核对图像与回答；这些是定性选例，不能当作完整任务准确率或充分推理能力证明 |

表 6 的 0-shot 情形同样存在差距：FP16 CIDEr=63.73，AWQ INT4=62.57，INT3=56.33。不同 few-shot 条件不能合并成一个“无损”标签。仅量化语言部分也不意味着整个 VLM 含视觉模块恰好压缩 4 倍。

### 7.4 数据效率、跨分布与极低位宽

P0 图 8／§5.3 在 OPT-6.7B、INT3-g128 上比较校准序列数量，序列长度为 2048 tokens；作者报告 AWQ 使用 16 条时即可优于 GPTQ 使用 192 条的对应 PPL。该观察只对应给定模型、方法配置和评价，不表示每次量化都只需 16 条。

跨分布实验使用 Pile 中 PubMed／Enron，校准与评价集不重叠、每种评价使用 1k 样本。AWQ 从同分布换到跨分布校准后，PPL 增量为 0.50／0.60，GPTQ 为 4.89／2.33；同分布通常仍更好。它支持这组实验中 AWQ 相对更稳健，不能证明对任意分布都不会过拟合。

P0 表 9 的 INT2-g64 结果是 **AWQ+GPTQ** 组合：OPT-6.7B 的 GPTQ PPL=16.65，组合=15.71，FP16=10.86；OPT-13B 为 16.74→13.25，FP16=10.13。RTN 在该表中严重退化，组合有改善但仍有明显损失。不能把组合结果写成独立 AWQ 的 INT2 能力。

## 8. TinyChat 怎样把压缩转化为加速

本节的压缩权重、反量化与浮点计算属于 weight-only 执行路径。它与模拟量化、W8A8 整数矩阵乘的区别见 [量化的三类执行路径](../implementation/quantized-matmul-scaling-execution.md#3-三类执行路径)；下面保留 TinyChat 特有的实现与测量条件。

TinyChat 是 AWQ 配套的原生内核；同一份 AWQ 权重也可以交给为批处理设计的通用内核执行，例如 [Marlin](../implementation/marlin-batched-w4a16-gemm.md)，以及 vLLM 中单独列出的 awq_marlin 方法（见 [部署框架与后端支持](../implementation/quantized-llm-deployment-backends.md)）。两者硬件与批量条件不同，加速比不能互换。

### 8.1 带宽限制与理想上限

P0 §4.1、图 3 讨论 RTX 4090 上 batch=1 的生成负载，采用峰值约 165 TFLOPS、带宽约 1 TB/s 的分析口径。以 $C_{\mathrm{out}}\times C_{\mathrm{in}}$ 的矩阵向量乘为例，约 $2C_{\mathrm{out}}C_{\mathrm{in}}$ FLOPs 对应 FP16 权重的 $2C_{\mathrm{out}}C_{\mathrm{in}}$ 字节，忽略其他流量时约 1 FLOP/Byte；4-bit 权重使这一理想算术强度约为 4 FLOPs/Byte。

Roofline 上限可写为 $P\leq\min(P_{\mathrm{peak}},B_{\mathrm{memory}}I)$，其中 $I$ 是运算量／实际搬运字节数。这是对论文分析的数量级展开，未测量本地设备。缩小权重流量有用的前提是这部分流量确实重要；激活、KV cache、元数据、对齐、解包以及其他算子开销都不能被忽略成零。prefill 和大 batch 的权重复用不同，长上下文也可能改变瓶颈。

### 8.2 反量化、打包和融合

W4A16 在 P0 的实现中先将压缩权重反量化为 FP16 再参与矩阵计算。TinyChat 把解包／反量化融合进矩阵运算，避免先生成一整份 FP16 权重并写回 DRAM；矩阵向量与矩阵矩阵两种路径都需要相应实现。

图 4 的 ARM 128-bit SIMD 例子把 32 个 4-bit 权重按 $w_0,w_{16},w_1,w_{17},\ldots$ 配对，使一次按位与和移位操作同时处理多个权重。P0 的 GPU 示例按 $w_0,w_2,w_4,w_6,w_1,w_3,w_5,w_7$ 排列。它们是平台相关布局，不是所有 AWQ 后端通用的存储格式。

论文还融合归一化中的算子、QKV 投影和注意力相关操作，并预分配 KV cache。减少中间读写和 kernel 启动次数本身就能加速，因此必须与低比特量化贡献分开讨论。P0 §4.2、图 4。

### 8.3 性能证据的正确比较口径

P0 §5.4 报告 LLM 吞吐测试采用 batch=1、固定 4-token prompt、生成 200 tokens、中位延迟。图 9 中 RTX 4090／Llama-2-7B：

| 路径 | tokens/s |
|---|---:|
| Hugging Face FP16 | 52 |
| TinyChat FP16 | 62 |
| TinyChat AWQ W4A16 | 194 |

194/62≈3.13 是相对更强 FP16 路径增加量化实现后的比值；194/52≈3.73 还包含其他系统优化。此实验不是仅改变位宽、其余实现完全相同的比较。论文还说明 Falcon 的原始实现存在 KV cache 问题，相关系统加速更不能全部归因于量化。

图 9 包含 RTX 4090、Jetson Orin、8GB 笔记本 RTX 4070 的结果；4070 上 Llama-2-13B 报告 33 tokens/s，部分 FP16 模型 OOM。图 10 比较 AutoGPTQ、llama.cpp、exllama，包含 Orin 64GB 上的 Llama 系列与其他模型；作者称对 llama.cpp 最高约 1.7 倍。树莓派 4B 的 7B 模型为约 0.7 tokens/s。OOM、未支持和可运行但较慢是不同状态，不能一律折算成倍数。

表 10 的 VILA-7B 在 A100/4090/Orin 上，FP16 吞吐为 81.6/58.5/11.5，AWQ 为 155.3/168.1/35.6 tokens/s；VILA-13B 的 Orin 为 6.1→17.5。不能把面向 LLM 的 4-token prompt 协议擅自当作 VLM 的完整图像输入协议，论文未充分交代的细节保持未报告。

这些是 P0 中的历史测量和支持范围，不代表上述框架当前版本的能力或排序。当前项目没有验证任何速度数字。

## 9. 原文问题、完整性与后续验证

本轮没有用其他材料悄悄修补作者未说明的条件。以下问题必须随方法保留：

| 问题 | 证据与影响 |
|---|---|
| 潜在附录缺件 | 原始 `source.eprint` 内的 `supp.tex` 引用不存在的 `text/appendix.tex`，编译元数据把 supp 标为 ignore；PDF 没有附录。所查官方页面未提供独立入口。现有 PDF 全文已读，但全部潜在补充材料未能认证 |
| 图 2 与表格数值不一致 | 图 2 标注 OPT-6.7B、INT3-g128，给出 PPL 43.2→13.0；表 1—3 对应 RTN 为 23.54，1% FP16／AWQ 为 11.39。原文没有解释，不能拼接为同一实验 |
| 错误交叉引用 | 第 5 页 OPT 消融段引用 Table 5，但最终 Table 5 是 Mistral／Mixtral；相应 OPT 消融应查 Table 3 |
| 图 3 模型名称不一致 | 图注称 Llama-2-7B，§4.1 正文称 LLaMA-7B；无法据此确定实际实验对象 |
| Vicuna 计数聚合不明 | 正文为 80 问题、160 trials，图中每条堆叠合计 80；具体聚合方式未充分说明 |
| “无损”“不会过拟合”过强 | 表 7 存在分数变化；图 8 仅验证特定数据和模型。没有统计等价或普遍泛化保证 |
| 论文与代码版本不绑定 | C0 的默认数据、网格端点和裁剪细节已经定位，但未证明该 commit 对应 P0 全部实验，不能反填论文设置 |

本页的事实、推导和执行状态应分别理解：等价变换是精确算术下的代数关系；误差缩小包含统计近似；质量与速度为作者报告；代码流程是 C0 的静态阅读；本项目没有模型复现、CUDA 验证或性能测量。

若后续进入复现，先确定目标模型／权重版本、位宽与 group、校准与独立评测协议、实际后端及硬件。再分别验证未量化变换等价性、模拟与实际量化数值、模型质量和端到端性能。原文未报告的实验条件不能用默认值假装补齐。

## 来源身份

下表用于在没有本地资料库时辨识来源；具体论述的章节、公式、图表或代码位置见正文。

| 来源 | 版本或快照 | 说明 |
| --- | --- | --- |
| [MBQ: Modality-Balanced Quantization for Large Vision-Language Models](https://arxiv.org/abs/2412.19509v2) | `arXiv:2412.19509v2` | — |
| [AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978v6) | `arXiv:2306.00978v6` | — |
| [A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295v1) | `arXiv:2106.08295v1` | — |
| [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438v7) | `arXiv:2211.10438v7` | — |
| [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323v2) | `arXiv:2210.17323v2` | — |
| [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq/tree/d6e797a42b9ef7778de8ee2352116e0f48a78d61) | `d6e797a42b9ef7778de8ee2352116e0f48a78d61` | — |
