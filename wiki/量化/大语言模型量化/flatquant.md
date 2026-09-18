---
title: FlatQuant：结构化可学习变换与低比特执行
slug: flatquant
sources:
  - raw/papers/quantization/vlm/2026-05-splitq-breaking-modality-heterogeneity-low-bit/paper.pdf
  - raw/papers/quantization/ptq/2024-05-spinquant-llm-quantization-learned-rotations/paper.pdf
  - raw/papers/quantization/ptq/2024-10-flatquant-flatness-matters-llm-quantization/paper.pdf
  - raw/papers/quantization/ptq/2024-03-affinequant-affine-transformation-quantization/paper.pdf
  - raw/references/pytorch/2026-09-orthogonal-parametrization/orthogonal.html
  - raw/repositories/quantization/2026-07-ruikangliu-flatquant-9d88ffcb/source/flatquant/trans_utils.py
  - raw/repositories/quantization/2026-07-ruikangliu-flatquant-9d88ffcb/source/flatquant/function_utils.py
  - raw/repositories/quantization/2026-07-ruikangliu-flatquant-9d88ffcb/source/flatquant/train_utils.py
  - raw/repositories/quantization/2026-07-ruikangliu-flatquant-9d88ffcb/source/flatquant/flat_utils.py
  - raw/repositories/quantization/2026-07-ruikangliu-flatquant-9d88ffcb/source/flatquant/flat_linear.py
  - raw/repositories/quantization/2026-07-ruikangliu-flatquant-9d88ffcb/source/flatquant/quant_utils.py
  - raw/repositories/quantization/2026-07-ruikangliu-flatquant-9d88ffcb/source/flatquant/model_tools/llama_utils.py
  - raw/repositories/quantization/2026-07-ruikangliu-flatquant-9d88ffcb/source/REALQUANT.md
  - raw/repositories/quantization/2026-07-ruikangliu-flatquant-9d88ffcb/source/benchmarks/layer_benchmark.py
updated: 2026-09-15
---

# FlatQuant：结构化可学习变换与低比特执行

FlatQuant 在量化前学习通道混合，让权重与激活更适合均匀网格；同时用两个小矩阵的 Kronecker 乘积表示大变换，并融合在线变换和量化，控制运行开销。它仍采用冻结基座、逐 Transformer block 校准的 PTQ 路线，进一步连接了 [[affinequant|AffineQuant]] 的一般矩阵动机与实际低比特执行。

本页依据 arXiv:2410.09426v4（2025-08-10，ICML 2025），27 页含附录 A–E 已全文研读。官方代码固定为 `9d88ffcb7d2c6bda59fb5c44dad36adc101aadb1`，下称“代码”；定向核对 LLaMA 校准、变换、量化和部署说明，未运行模型或内核。论文质量、论文计时、后续真实部署说明分别标明。

## 1. 什么叫 flatness，实际优化什么

论文关注通道幅度不均衡：[[diagonal-scaling-equivalent-transform|逐通道缩放]] 可以把激活难度转给权重，但不能直接把一个通道分散到其他通道；固定 [[orthogonal-rotation-and-hadamard-quantization|Hadamard 变换]] 能混合通道，却没有为每层的数据学习变换。FlatQuant 通过块输出目标联合调整两侧表示。（§2，图 1–2。）

这里要区分三件事：

- **通道包络变平**：取各通道的 Frobenius 范数得到 $d\in\mathbb R^N$，理想等幅包络为 $d'=\|d\|_2\mathbf1/\sqrt N$；图 7 使用到此包络的欧氏距离，并对块内权重/激活汇总，越小表示通道范数越均衡。
- **量化误差随层/token 的曲面更低、更平**：图 2 与附录 D 画的是输出 MSE 分布，不是参数扰动下的优化损失曲面。附录使用 C4 的 128 个样本做平均。
- **实际校准目标**：式 4 仍是整个 block 的输出重构 MSE，没有直接最小化上述 flatness 指标。图 7 中两个指标一起下降是经验观察，不是通用因果证明。

通道范数相近不意味着通道内没有极端元素，也不等价于低峰度。该距离还随整体幅度缩放而变化；两侧变换与逆变换共同作用，应结合 [[quantization-error-diagnosis|输出误差与任务验证]]，不能只追求好看的平坦包络。

## 2. 从一个线性层到可计算的变换

令 $X\in\mathbb R^{T\times n}$、$W\in\mathbb R^{m\times n}$、$Y=XW^\mathsf T$。对可逆 $P\in\mathbb R^{n\times n}$，

$$Y=(XP)(P^{-1}W^\mathsf T),\qquad
\min_P\|Y-Q(XP)Q(P^{-1}W^\mathsf T)\|_F^2.
$$

$Q$ 表示量化及反量化后的实数张量，非整数码值直接相乘。实际权重布局中存 $WP^{-\mathsf T}$；带偏置的输入侧配对保留原偏置。输出侧变换若同时改变线性层输出，则偏置也需变换。（§3.1 式 2；代码 `flat_linear.py`。）

完整 $n\times n$ 在线矩阵太昂贵，因此选

$$P=P_1\otimes P_2,\quad n=n_1n_2,\quad
P_1\in\mathbb R^{n_1\times n_1},\ P_2\in\mathbb R^{n_2\times n_2}.
$$

将每个 token 按行 reshape 为 $V\in\mathbb R^{n_1\times n_2}$，只需计算 $P_1^\mathsf T V P_2$ 后展开。逆侧用 $P_1^{-1}\otimes P_2^{-1}$；不形成完整大矩阵。索引推导、非对称数值例子和布局条件见 [[invertible-transforms-and-kronecker-products|可逆变换与 Kronecker 基础]]。

每个变换的存储从 $n^2$ 降到 $n_1^2+n_2^2$，每 token 的乘加从 $n^2$ 降到 $n(n_1+n_2)$；固定乘积时选择接近的因子。例如 4096 分成 $64\times64$，8192 分成 $64\times128$。这是用结构约束换开销，不是任意完整矩阵的无损表示。（§3.1；代码 `flat_utils.kronecker_matmul`。）

**尺度与裁剪仍有各自作用。**论文在混合前加入可学习对角尺度 $D=\operatorname{diag}(c)$，得到 $(XD^{-1}P)(P^{-1}DW^\mathsf T)$；尺度用于平衡两侧幅度，可合入前面的 Norm 或适当线性分支。代码的 `diag_scale` 对输入做乘法，对应此处 $D^{-1}$ 的约定，不能直接照抄论文 $c$ 的方向。变换后再学习权重、激活与 KV 的裁剪范围：重分配离群值后，裁剪才在新的表示中决定牺牲哪些尾部。（§3.1，图 3，表 18。）

## 3. 在 Transformer 中放在哪里

### 单个线性层怎样完成一次校准前向

先看没有输出侧变换的投影，用论文的 $D$ 方向。令 $R=D^{-1}P$，并用 $C_w$ 表示可学习权重裁剪；$Q_a$ 内含激活范围选择、裁剪和反量化，$Q_w$ 按裁剪后的权重估计网格：

$$\begin{aligned}
X'&=XD^{-1}P=XR,\\
W'&=WDP^{-\mathsf T}=WR^{-\mathsf T},\\
\widehat Y&=Q_a(X')\,[Q_w(C_w(W'))]^\mathsf T+b.
\end{aligned}$$

没有裁剪和量化时，$X'W'^\mathsf T=XW^\mathsf T$。学习时每步都从冻结的 $W$ 重新生成 $W'$，两侧通过同一个 $D,P$ 接收块损失梯度；不是先独立“压平”激活，再把另一侧当成常量。共享输入的 Q/K/V 或 up/gate 可以复用 $X'$，但各投影仍有自己的权重网格。V 投影的输出侧变换会改变权重处理顺序，见第 4 节的代码说明。

校准后可将 $D$ 融入合法的上游参数，将 $DP^{-\mathsf T}$ 融入权重；需要的 $P$ 混合仍在线执行。离线固化权重并不固定未来所有 token 的激活范围，per-token 网格仍由运行时输入决定。（§3.1、图 3；固定代码 `flat_linear.py:_train_forward/reparameterize`，`trans_utils.py`。）

### 各个变换的配对位置

下面是 LLaMA 风格结构。归一化、变换、RoPE、残差和注意力分数保留浮点；主要线性层做低比特权重/激活计算。W4A4 并不是所有算子都执行 INT4。

| 变换 | 位置与配对 | 为什么这样安排 |
|---|---|---|
| $P_a$ | Norm 后共享输入 → Q/K/V；逆侧进入三个投影权重 | 三个分支共用输入，不需要重复计算三次输入变换 |
| $P_h$ | RoPE 后，$Q'=QP_h^{-\mathsf T}$、$K'=KP_h$ | $Q'K'^\mathsf T=QK^\mathsf T$，不必把一般矩阵跨过位置旋转 |
| $P_v$ | $V'=VP_v$，对应逆侧最终进入 O 投影 | 可合入 V 权重；attention 在每个 head 内只混合 token，故 $S(VP_v)=(SV)P_v$ |
| $P_o$ | attention 输出的 head 混合，配对 O 权重 | 与 head 内 $P_v$ 构成结构化输出变换，避免再在线执行一遍完整 $P_v$ |
| $P_{ug}$ | Norm 后 → up/gate 两个共享输入分支 | 两个逆侧分别合入 up/gate 权重 |
| $P_d$ | SiLU(gate) 与 up 相乘之后 → down 投影 | 在非线性之后在线执行混合，逆侧合入 down 权重；不尝试将一般矩阵穿过 SiLU |

（§3.2、图 3；代码 `model_tools/llama_utils.py`。）

在代码的默认 LLaMA 路径中，$P_a,P_{ug},P_d$ 用两因子分解；$P_h,P_v$ 是 head dimension 的小方阵，并跨 head 复用；$P_o$ 是 attention head 数量维度的小方阵。attention 输出 reshape 成“head 数 × head dimension”，在线混合前一维，后一维的 $P_v$ 已由 V 分支提供；O 权重消费两者逆转置的 Kronecker 配对。论文先描述一般的输出变换，再将 $P_o/P_v$ 融合，不能照着未融合框架重复施加两遍。

### 注意力到 O 投影怎样完整恢复

以下先关闭量化，省略 batch、偏置和 dropout，以 $H$ 表示 Q head 数、$d_h$ 表示每头维度。对完成 RoPE 的第 $j$ 个 head，有

$$Q'_j=Q_jP_h^{-\mathsf T},\quad K'_j=K_jP_h,
\quad S_j=\operatorname{softmax}(Q_jK_j^\mathsf T/\sqrt{d_h}+M).
$$

配对变换保持分数及相同 mask $M$ 下的 $S_j$ 不变。设 $V'_j=V_jP_v$，则该 head 的输出是 $S_jV'_j=(S_jV_j)P_v$；即便不同 head 的 $S_j$ 完全不同，这一步仍各自成立。现在把一个输出 token 的原始各 head 结果排列为 $O_t\in\mathbb R^{H\times d_h}$，合并后的结果为

$$O'_t=P_o^\mathsf T O_tP_v,\qquad
o'_t=\operatorname{vec}_{row}(O'_t)
=o_t(P_o\otimes P_v).
$$

令 $R_o=P_o\otimes P_v$，原 O 投影权重为 $W_o\in\mathbb R^{d_{model}\times Hd_h}$，保存

$$W'_o=W_oR_o^{-\mathsf T},\qquad
o'_tW_o'^\mathsf T=o_tR_oR_o^{-1}W_o^\mathsf T=o_tW_o^\mathsf T.
$$

因此 $P_v$ 可先进入 V 投影，attention 之后只需在线左乘 $P_o^\mathsf T$，再由 O 权重补偿两者。若把 $P_o$ 提前到各 head 的 attention 之前，会让不同 $S_j$ 的加权相互混入，一般不能抵消。这里 $P_h$ 保留论文的变换名，下标 $j$ 才表示具体 head；该实现跨 head 共享 $P_h$。（§3.2；固定代码 `llama_utils.py:FlatQuantLlamaAttention.forward/reparameterize`。）

量化后实际得到的是 $\widehat S_j$ 和量化后的变换 V，通常不再严格等于原 $S_jV_jP_v$；O 侧逆变换只还原坐标，无法撤销注意力概率、裁剪与舍入误差。完整等价链说明量化前为何成立，最终误差仍须由块目标及任务评测衡量。

为什么 down 前的对角尺度能提前合入 up？令门控输出 $z=\operatorname{SiLU}(g)\odot u$，则 $zD^{-1}=\operatorname{SiLU}(g)\odot(uD^{-1})$；只需改变 up 的输出尺度。一般矩阵 $P_d$ 没有这种逐坐标可交换性，所以仍留在 $z$ 之后。它补充了 [[omniquant|OmniQuant]] 与 AffineQuant 未在 down 前引入同类一般混合的位置。

GQA 中 Q head 多于 KV head。代码在 `repeat_kv` 前应用共享 head 内变换，再扩展 KV；跨 head 的 $P_o$ 放在各 head 独立 attention 已完成之后。若改为每头不同的变换，需要重新核对 head 对应关系，不能把当前共享结构的等价性直接套过去。

## 4. 怎样学习矩阵与量化参数

块目标为

$$\min_{\Theta_l}\mathbb E_{X_l\sim\mathcal C}
\|F_l(X_l)-\widehat F_l(X_l;\Theta_l)\|_F^2,
\quad\Theta_l=\{P,c,\alpha_w,\alpha_a,\text{KV 裁剪参数}\}.
$$

它冻结原始权重，学习重参数化和网格，而非全模型 QAT。含反向的 PTQ 分类、舍入代理梯度见 [[post-training-and-quantization-aware-training|PTQ、QAT 与代理梯度]]。

**矩阵参数化。**附录 B.1 采用 $P_i=U_i\operatorname{diag}(s_i)V_i^\mathsf T$，用正交 $U_i,V_i$ 和对角倒数构造逆。代码的 `SVDSingleTransMatrix` / `SVDDecomposeTransMatrix` 直接学习这些因子，`inv_t=True` 返回 $P_i^{-\mathsf T}$，不是每步对已形成的矩阵做一次 SVD。U/V 使用 PyTorch Cayley 参数化；相应 API 的定义及用途已纳入 [[invertible-transforms-and-kronecker-products#3. SVD 形式怎样表示可逆矩阵|矩阵基础]]，参考文档只作这一解释。

代码将对角值初始化为 1 并直接学习，未对其施加正值或离零下界；因此数学可逆性需要 $s_i\ne0$，数值上还需远离病态。不能把“用了 SVD 形式”写成“永远不会奇异”。直接求逆分支会临时转 double 后求逆再转回，这也是代码事实，不证明任意混合精度环境都稳定。

**一次校准的中间状态：**

1. 捕获校准输入，冻结基座；当前块用浮点模式计算教师输出。
2. 初始化小矩阵、可选对角尺度和裁剪参数；构造在线激活变换与离线权重配对的临时前向。
3. 对变换后的权重裁剪、按输出通道量化；对激活按 token 定范围和量化；Q/K/V 按各自开关处理。
4. 计算整个块的输出 MSE，反向更新变换与裁剪。代码用 STE 处理舍入，不能把离散量化的精确导数当成该代理梯度。
5. 当前块结束后保存参数；所有块校准完成再进行重参数化与权重量化，导出需要的矩阵、网格和打包权重。

**与 OmniQuant 的重要区别：**当前代码 `train_utils.cali_flat_quant` 的教师和学生都接收 `fp_inps`，每层结束用教师输出推进下一层输入；没有 OmniQuant 默认的浮点/量化前缀双路径。优化中还采用 $L/\operatorname{stopgrad}(L)$，前向数值约 1、梯度按当前损失缩放；日志保存未归一化 MSE。相同的“块重构”标签并不表示相同误差补偿目标。

论文 §4.1 配方：WikiText-2 的 128 段、每段 2048 tokens，batch 4，15 epochs，AdamW；矩阵/尺度初始学习率 $5\times10^{-3}$，裁剪 $5\times10^{-2}$，cosine 衰减。默认 AMP+SVD 配方在 LLaMA-3-8B 报告约 0.9 小时、27,554 MiB（表 4）。该表未直接给出 GPU 型号，不用推理计时的 RTX 3090 补成校准硬件事实。

论文主配置为权重 per-channel、激活 per-token 对称 INT4，KV 按 head 大小 128 做非对称量化。代码激活裁剪按量化器学习共享的两端强度，权重裁剪是每个输出通道两端；不是每个 token 各学习一套静态阈值。KV 张量先按 head reshape，因此最后一维的范围归约实现 head 内量化；一般 activation group size 开关并非任意可用。

KV 量化在这里属于附带对象；KV 侧的粒度选择、流式约束与分页布局问题另有专门整理，见 [[kv-cache-quantization-objects-and-granularity|KV cache 量化的对象与粒度]]。

裁剪先后也应按算子核对：`flat_linear.py:_train_forward` 对权重先做输入侧逆变换、再裁剪；若 V 投影还有输出侧变换，随后才施加该变换并量化。因此“变换后裁剪”是总体设计，不能不看分支就认定每个权重都在所有变换完全结束后裁剪。

## 5. 怎样变成实际低比特执行

### 融合省去中间访存

在线计算 $P_1^\mathsf T V P_2$ 有两次小矩阵乘，后面还要取范围、量化与打包。若每一步都写回显存，会增加中间张量读写与 kernel 启动。论文 §3.3 将变换与量化放进一个 Triton kernel，尽量在片上保存中间结果，再交给 CUTLASS INT4 矩阵乘；KV 低比特执行使用 FlashInfer 路线。

**与 W4A8 路线的分歧。** [[qserve|QServe]] 论证按组 W4A4 在 Ampere 与 Hopper 上必须在主循环内反量化部分和、因而被 CUDA 核心限制，转向 W4A8；FlatQuant 则用结构化在线变换与融合内核承担变换成本，主张 W4A4 可以落地。两者的前提不同（变换放在哪里、反量化是否融合、内核实现方式），这类分歧需要同硬件、同批次、同测量口径的对照实验来判断。

这不是“所有模型操作融合成一个 kernel”，也不是消除了变换算术。需要区分 [[quantized-matmul-scaling-execution|存储、反量化与实际矩阵乘]]。

可以用单层成本看出为何额外 FLOPs 很少仍可能不加速。取输入和输出维度均为 4096，每 token 原投影约 16,777,216 次乘加，$64\times64$ 两因子变换增加 524,288 次，即原投影的 3.125%；这只算该输入变换，不含 Q/K、量化、内存访问或模型其他操作。实际延迟应满足

$$t_{\text{在线变换+量化}}+t_{\text{低比特矩阵乘}}+t_{\text{其他新增开销}}
<t_{\text{原浮点投影}}
$$

才算这一局部路径加速；融合项应按整体计时，不能再重复加一次量化时间。小 batch decode 的矩阵乘和访存形态不同，启动、范围归约或缓存处理可能吃掉算术节省。该不等式是解释部署取舍的成本分解，不是从 FLOPs 推算论文速度；真实边界见第 6 节。

附录 B.3 的默认设计要求中间结果装得进片上空间；尺寸过大时有两类退化：沿非归约维切片、变换后再单独量化，或把第一次乘法结果写回显存后完成第二次乘法。因而“一次融合”也有 shape/容量边界。表 6 的 kernel 加速比是相对未融合变换，不能充当全模型相对 FP16 加速比。

### 校准前向与部署前向不是同一份证据

定向阅读的 `FlatQuantizedLinear` 通过实数 `F.linear`/普通线性层执行 fake quant；KV 量化器也返回反量化后的浮点值。它可以模拟低比特误差，但仅运行这一路不能证明低比特缓存占用或 INT4 速度。打包函数 `save_quantized_weights_with_safetensors` 另将权重写成 INT4 对应的打包数据。

固定快照 `REALQUANT.md` 描述了后续真实权重部署：有独立 whole-model benchmark、软件/CUDA 版本限制、hidden dimension 为 2 的幂等限制，并说明 prefill attention 使用当时可取得的未量化 query/key；它也提示若要接近原论文计时应选更早版本。文档示例的 KV 非对称参数与其支持范围说明并不完全一致，需按目标后端核对，不能直接宣称任何 KV 配置均可导出。

本轮没有跑真实模型，未审计全部 kernel 分支。代码中的单层 benchmark 与后续 whole-model benchmark 也不是同一测量对象；论文所称端到端结果保留为作者报告，不据此宣称本地已复现整模型速度。

## 6. 能改变方法判断的证据

### 变换、裁剪与尺度的作用

论文表 3/16，LLaMA-3-8B 的 W4A4KV4，RTN 权重量化：

| 组件 | WikiText2 PPL | 六任务平均准确率 | 解释 |
|---|---:|---:|---|
| 无组件 | 1266.60 | 30.99 | 极低位宽下直接量化失效 |
| 仅可学习变换 LT | 8.50 | 66.82 | 变换承担主要改善 |
| LT + 逐通道尺度 PS | 7.95 | 67.08 | 额外尺度有小幅增益 |
| LT + 可学习裁剪 LCT | 7.11 | 70.72 | 变换后裁剪提供重要增量 |
| LT + PS + LCT | 6.98 | 71.23 | 组件联合在此配置最好，不表示增益可线性相加 |

表 16 还报告不含 LT 的 PS、LCT 或两者组合表现差乃至 NaN，支持“只学尺度/阈值不足以解决这个配置”。不能推广成所有模型和位宽必须使用 LT。

表 18 保持 FlatQuant 框架，LCT 放在变换前得到 WikiText2 7.37、QA 68.62；放在变换后为 6.98、71.23。它支持在改变表示后选择范围；逆变换保持未量化计算等价，却不能恢复已经裁掉的信息。

### 表达力与校准代价

图 5 展示矩阵因子接近时计算较快、PPL 变化较小。但附录 C.6 明确：精度试验只量化最后一个 Transformer block，速度试验用普通 PyTorch 矩阵乘且未使用融合 kernel。不能据此宣称所有层、模型和维度都不存在结构表达力损失。

表 4 的 FP32+直接求逆为 2.2h、35,384 MiB、WikiText2 6.95；AMP+SVD 为 0.9h、27,554 MiB、6.98。它支持后者在所测设置的成本取舍；不能说“时间和显存都减半”。附录也承认部分模型或更低位宽需回到高精度训练。

### 精度、迁移与失败边界

- **简单 RTN 已有竞争力，但 GPTQ 并非处处无用。**表 1 的 LLaMA-3-8B，FlatQuant RTN/GPTQ 的 WikiText2 为 6.98/6.90，C4 为 11.13/11.21；改变指标会改变胜负。两者还需比较校准时间。
- **“损失小于 1%”的范围有限。**表 2，LLaMA-3-70B 六任务平均 FP16 79.95、FlatQuant RTN 79.01，相差 0.94 个百分点；LLaMA-3-8B 为 73.23→71.23，相差 2.00 个百分点。不是所有任务、模型均小于 1%。
- **更低位宽仍有明显代价。**表 14，LLaMA-3-8B W3A3KV3 的 WikiText2 10.82、QA 58.45，W4A4KV4 为 6.98、71.23，FP16 为 6.14、73.23。可用程度应按任务判断。
- **K/V 的敏感性不同。**表 12 的 KV-only、其余保持高精度：LLaMA-3-8B K4V2 PPL 6.60，K2V4 7.70。它支持该配置中 key 更敏感，不等于所有模型固定采用同一精度分配。
- **校准来源实验支持有限的稳健性。**表 17，WikiText2/C4/Pile 三种校准来源的 WikiText2 PPL 6.98–7.04，QA 71.04–71.23；未检验所有领域、长上下文或视觉 token 分布。表 15 的同模型跨量化配置复用，也不等于跨模型直接复用矩阵。
- **混合精度仍有空间。**表 19 将 top-5 blocks 与 down 投影提升到 W8A8 后，QA 71.23→72.18，但该表没有给出对应混合方案速度，不能把质量增益与统一 INT4 的最高速度合并报告。

附录 C.1–C.2 还评估 Qwen、DeepSeek 与 MT-Bench，说明作者测试范围超出基础 LLaMA；其中 DeepSeek-R1 的 AIME 2024 从 FP8 基线 79.8 降到 73.3，不能用其他 QA 平均值掩盖推理任务损失。本轮未核验这些模型的完整代码与评测协议，不以其结果替代对目标场景的验证。

### 实际速度有场景限制

论文 §4.3、图 4/9/10 报告 RTX 3090 上 LLaMA-2-7B，prefill 2048 tokens、随后 decode 256 tokens，batch 64 时约 2.30× / 1.76× 相对 FP16。附录 C.8 同时指出 **batch 小于 16 时 decode 加速比低于 1**，量化开销超过 KV 访存节省。不能为了达到高加速比，把单请求需求擅自换成大 batch 吞吐场景。

附录 C.7 表 20 的显存数据来自 batch 1、单个 Transformer layer、单 token decode，不能当成整个 7B 模型显存。在线变换参数约 3.41 MB、额外 FLOPs 约 2.61% 是附录 B.2 对 LLaMA-2-7B 的分析口径，不表示真实运行零开销。后续 `REALQUANT.md` 中 batch 1 的真实整模型 decode 也未加速，与这一边界方向一致，具体数值不与论文版本混合。

## 7. Strong / weak 与下一步研究意义

**Strong：**用结构化矩阵把一般通道混合扩展到更多关键位置；学习的变换、尺度与裁剪相互适配。RoPE 后 Q/K 配对和 down 前在线变换明确处理了计算图位置问题。融合 kernel 则把方法的变换成本纳入低比特执行设计，而不仅展示 fake quant 精度。

**Weak：**Kronecker 结构有表达力限制，校准仍需反向与内存；对角参数无自动病态保护，AMP 也有例外。学习目标仍是有限校准数据上的块输出；更低位宽、推理任务、小 batch decode 和某些部署形状都有实质代价。

它与 OmniQuant/AffineQuant 的关系应按“变换空间—位置—优化—执行”比较，见 [[omniquant-affinequant-flatquant-comparison|三种可学习量化方法比较]]。对多模态研究，它提供可检验的变换模块和实现思路，但还需要回答视觉/文本分布、校准权重和端云运行场景是否改变最合适的约束；论文没有替我们证明这些结论。

[[spinquant|SpinQuant]] 提供另一个重要对照：它用最终 CE 学习可融合的全局残差/头内正交旋转，FlatQuant 用块 MSE 学习局部可逆变换及量化参数。前者的正交性允许跨 RMSNorm 传递残差坐标；后者允许拉伸，却不能直接替换该全局旋转。比较应同时控制变换位置、目标和在线算子成本，不把“可逆矩阵包含正交矩阵”当成整套方法必然更优的证明。（SpinQuant v4 §3；FlatQuant v4 §3。）

[[splitq|SplitQ]] v1 §4.2–4.3 将 FlatQuant 用于多模态通道分组：各组独立变换，主体再加权重平滑与文字激活补偿。分组后的变换相当于重排坐标中的分块对角矩阵，限制跨组混合以隔离模态冲突；辅助路径则带来额外位宽与计算。这是对结构、目标和执行图的共同修改，不能归纳成给原 FlatQuant 换一个图文校准集。该多模态扩展的收益与缺口见 SplitQ 的组件消融和部署说明。
