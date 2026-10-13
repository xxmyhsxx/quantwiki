---
title: StagQ：共享元数据的前缀权重流与稀疏保护
type: method
tags:
  - weight-only
  - mixed-precision
  - bit-planes
  - kernels
sources:
  - raw/papers/2026-10-07/stagq
updated: 2026-10-07
---

# StagQ：共享元数据的前缀权重流与稀疏保护

StagQ 解决的是同一份权重需要提供多种精度的问题：保存一个 2-bit base 与有序 refinement planes，加载较浅前缀就得到低精度模型。它将每组共享元数据、普通 affine 解码和少量稀疏保护一起设计，而不是分别保存五份 checkpoint。本页依据 `arXiv:2610.05977v1` 的正文与必要附录；没有取得作者代码或运行模型。[1][2]

理解分组网格先读[量化映射](../fundamentals/quantization/uniform-quantization-and-groups.md)，存储口径接到[混合精度预算](../theory/mixed-precision-allocation.md)，稀疏例外的作用接到[表示设计](../research/low-bit-representation-design.md)。这里的多精度指同一编码支持不同读取精度，不表示已实现服务过程中逐请求即时切换。

## 1. 前缀约束改变的是可行表示集合

普通 PTQ 可以分别为每个精度选择范围、尺度与码字。前缀表示要求较低精度由较高精度的同一流截取，不能独立重新拟合。因此在**相同 rate 定义、失真度量、校准信息与 decoder 类别**下，前缀码属于固定精度码的子集；最优固定精度量化器的失真不会更差。这是可行集合的关系，不意味着任何实际单精度工具都胜过 StagQ。多精度的收益来自减少多份模型的保存与维护，并接受表示约束。[1，§2.1]

论文区分按高精度 leading digits 截取、每阶段另加尺度的残差/位面、每精度查表重建三类设计。StagQ 的取舍是：每组只保存一对参数，由它推导所有精度的尺度与原点；每个权重的主体仅需一个普通整数的 affine 解码。它仍有**每组元数据 codebook 查表**，不能把“无逐权重查表”写成整个 kernel 没有查表。[1，§2–3；附录 G]

## 2. 二进制 refinement 怎样折叠成一个 affine 整数

设最大精度为 $N$，读取精度为 $M\in\{2,\ldots,N\}$。论文实例 $N=6$，暴露 p2 至 p6。每组有 base 原点 $lo$ 与步长 $s>0$；主体 base code $q\in\{0,1,2,3\}$，后续 $b_k\in\{0,1\}$ 表示当前残差的符号，零残差约定取正号。第 $k$ 层 refinement 添加的半步为 $t_k=s/2^{k-1}$：[1，式 (7)–(9)]

$$\widehat w_{\rm main}^{(M)}=lo+sq+\sum_{k=3}^{M}(2b_k-1)\frac{s}{2^{k-1}}.$$

第一个 refinement 是 $\pm s/4$，不是 $\pm s/2$；新网格间距为 $s_M=s/2^{M-2}=2t_M$。把 code 拼成

$$u_M=q\,2^{M-2}+\sum_{k=3}^{M}b_k2^{M-k},$$

则

$$lo_M=lo-s\frac{2^{M-2}-1}{2^{M-1}},\qquad
\widehat w_{\rm main}^{(M)}=lo_M+s_Mu_M.$$

推导来自把每个二值项的 $-t_k$ 合成原点位移；数据相关的正项则组成 $u_M$。因此加载更深前缀既追加低位，也改变派生原点/步长，不能只丢弃低位后沿用最高精度的 affine 常数。[1，式 (8)–(10)]

原文图 1 的 $lo=0,s=1,w=2.15$ 是一个直观例子：

| 读取 | 已读 code | $u_M$ | $lo_M$ | 重建 |
| --- | --- | --- | --- | --- |
| p2 | $q=2$ | 2 | 0 | 2 |
| p3 | $b_3=1$ | 5 | −0.25 | 2.25 |
| p4 | $b_4=0$ | 10 | −0.375 | 2.125 |
| p5 | $b_5=1$ | 21 | −0.4375 | 2.1875 |
| p6 | $b_6=0$ | 42 | −0.46875 | 2.15625 |

表中每个 bit 决定向左或向右半步，子网格相对父网格错开；单个权重的绝对误差不必每层严格下降。网格覆盖范围内最近点的误差界为 $s_M/2$，对超出可达范围的值则另有 clipping 与 side record 问题。

附录 C 的最优性只覆盖 refinement：父 cell 是区间，一 bit 分成两个区间，父区间条件密度均匀，每子区间以单个标量重建并最小化平方误差。这时阈值在中点，两子重建值在四分点，递归得到 dyadic schedule。实际 LLM 组内密度不完全均匀，作者测得条件质心朝组中心偏移；不能把这个条件定理推广成所有模型/网格的最优性。作者在 Llama-3.1-8B 的 28 个矩阵、两种 group size 上测试统一 refinement rescale，最好中位输出误差改善至多 0.6%，最终仍取系数 1。[1，§3.1–3.2；附录 C/D]

## 3. 一个元数据 pair 如何兼顾五种读取精度

Encoder 先在 200 个固定的 2048-token C4 窗口测量输入通道均方 $E_c=\mathbb E[x_c^2]$。AWQ 风格变换量化 $W\operatorname{diag}(a)$，其中 $a_c\propto E_c^{\alpha/2}$，普通投影 $\alpha=0.25$、down 为 0.5。相应输入需乘逆尺度；可折叠到前置 normalization，否则 timed decoder 对 Llama/Phi 在运行时施加，OLMo 的测速省略此乘法。[1，附录 B/E]

变换前后保持函数的实数关系是 $Wx=(W\operatorname{diag}(a))(\operatorname{diag}(a)^{-1}x)$。量化发生在放大部分列后的矩阵，逆尺度负责恢复其与输入的配合；不能只放大权重却省略这个配合，再把结果解释成等价变换。

变换后沿输入列分组，每组 $g=16$ 或 32。每个矩阵拟合两份各 16 项的 FP16 codebook，分别编码 $\log_2s$ 与 $lo/s$；每组只存两个 4-bit index，从 256 种 pair 中选择。查到两个值后还须恢复 $s$ 与 $lo$，不能将 codebook 数值直接当作已解码的原点与步长。pair 对所有精度共用。[1，§3.1]

选择目标平均五种精度的激活对角加权权重误差，并加入精细网格无法追回的 base 误差：

$$D(lo,s)=\frac1{N-1}\sum_{M=2}^{N}\sum_c E_c(w_c-\widehat w_c^{(M)})^2+\lambda\Phi(lo,s),$$

$$\Phi(lo,s)=\sum_c E_c\max(|w_c-\widehat w_c^{(2)}|-\rho,0)^2,
\qquad \rho=s\sum_{k=3}^{N}2^{-(k-1)},\quad \lambda=4.$$

$\rho$ 是所有后续半步朝同一方向的最大移动距离；$N=6$ 时为 $15s/32$。若某 base 误差比它还大，追加所有 refinement 也不能完全追回。这里的 $c$ 是组内对应输入通道，$E_c$ 是激活二阶矩，目标没有使用完整通道协方差，也没有 GPTQ 的逐列反馈。[1，式 (5)–(6)；附录 E]

拟合不是直接穷举任意连续 pair：先为 codebook 比较 uniform、Lloyd 与固定两端的 Lloyd 的九种组合，仅用目标第一项选择；若六位误差超过两位误差的十分之一，回退 uniform。随后每组的 256 候选选择才加入 floor term。预保护的值在网格拟合时向内截到下一个值，目标仍遍历整组；不能写成简单删除元素后对剩余元素独立拟合。[1，附录 E]

## 4. 两次稀疏保护为何不能合成一次最大误差筛选

**Pre-grid guard 在拟合前改变主体的分布。** 每组最多取一个值，按配置选择 magnitude ratio 或收益/bit：ratio 比较最大与次大绝对值，Llama 的相关分支取 $T=4$；gain 则把候选向内截断并重拟合矩阵 codebook/pair，用组内两位误差下降 $\Delta_j=D_2(V)-D_2(V^{j\to})$，保留 $\Delta_j/b>\tau$ 的候选。收益包含重拟合对其他值的影响，候选也相互影响，这是启发式，不是全局最优 knapsack。[1，§3.5]

**Post-grid guard 在固定网格后补残差。** ratio 分支或 Phi fused gate/up，每输出行取激活加权两位误差最大的一项；gain 分支每行先取分数最高三项，再保留超过同一阈值的项。它不能补回“拟合前缩窄范围”对整个主体的收益。作者在 Phi 的 80 个 attention 矩阵上报告：同类选中项拟合前保护可移去 26.9% 的两位误差，拟合后只移去其本身承担的 6.2%；这个结果限于所列 group/threshold 条件。[1，§3.5；附录 E]

完整重建为

$$\widehat W^{(M)}=(1-m)\odot\widehat W_{\rm main}^{(M)}+P.$$

$m$ 标记保护位置。pre 位置的 $P$ 是保护值（可能经过其记录格式的量化）；post 位置的 $P$ 是**两位 base 重建加记录残差**。这些值不随 $M$ 改变，后续 refinement 跳过这些位置。因此完整运行不能在更高精度主体上无条件再加“两位残差”，也不能读取更深前缀时丢掉 side record。[1，式 (4)；§3.6]

保护值与索引也花费 bits：gain-pre 使用 INT8 对称编码及每输出行 FP16 scale；ratio-pre 为 FP16 值及 flat index；post 为 FP16 残差及列位置，gain 分支每行另有两位计数。论文的 gain-pre 位置 rate 按接近枚举编码的 $\log_2(n/k)+\log_2e$ bits/项计费，但 timed layout 实际使用固定 slots，没有保存那种枚举码。这是逻辑 rate 与读取字节产生差异的来源之一。[1，附录 E/A]

## 5. 从可截取文件到 GEMV 读取布局

保存流按 base、plane 3、plane 4…排列，读取 p$M$ 可以跳过更深 planes。实际 CUDA GEMV 则在加载时一次重打包为连续 $M$-bit digits；“无需重新量化”并不等于无需布局转换，也不等于当前实现支持运行时不重新加载就切精度。[1，§3.3；附录 G]

附录 G 描述 CUDA C++ GEMV：不用 tensor cores/CUTLASS/cuBLAS，沿 AnyBCQ 的 tiling、half 累加和 atomic-add 结构；256-thread block 负责论文记法中的 1024-column output tile 与 64-row input tile。每组读取其 metadata pair 的 256-entry table 后做 affine 解码；端到端路径把该表载入 shared memory，GEMV 表的计时则从 constant memory 读。两阶段 guard 在同一 pass 应用。论文只报告真实权重通过该 decode 路径的两位设置，没有作者代码可供本轮核查，因此这些为**论文实现描述**。[1，附录 G]

两份 matrix-level codebook 共 $2\times16\times16=512$ bits，group pair 本身为 $8/g$ bpw；$g=32$ 时为 0.25。连同 side record、索引/计数、row scale 等得到完整 logical rate。每加一个主体 plane 增加 1 bpw，但它不消除固定保护记录的开销。

kernel 的 256-entry pair table 由两份 codebook 派生，属于运行时展开的读取表；512-bit 的文件字段计账不能直接当作这张运行表的 shared/constant-memory 占用。

| 口径 | 论文的 Llama-3.1-8B 两位例子 | 能支持的判断 |
| --- | --- | --- |
| 主体精度 | 2 bit | 只说明 base payload |
| XXS logical rate | 2.35 bpw | 格式字段的声明位宽记账 |
| 当前 real-weight kernel 读取 | 2.47 bpw | row guard slots 按最满行预留后的实际读取 |
| 可变长度 row record 的设想 | 2.39 bpw | 尚需另一读取 kernel |
| 再加 attention presence bit 的设想 | 2.354 bpw | 同样属于未来布局 |

后两项不是已经实现的性能数据；2.35 bpw 也不是完整模型文件或运行峰值（embedding/head 保持 BF16，激活、KV 与 workspace 另算）。[1，§4.1；附录 A]

## 6. 实验支持哪些结论，哪些取舍仍未隔离

作者在 Llama-3.1-8B、Phi-4、OLMo-2-7B 的 transformer-block linear 上量化，embedding 与 LM head 保持 BF16，统一 harness 报告 5-shot MMLU 与 token-normalized WikiText-2 PPL，Llama 另有五项 zero-shot CSR。XXS 主要用 $g=32$，但 Llama value 为 16；XS 在 Llama 用 16。guard criterion 与阈值按模型、配置及 block 选择；Llama/Phi 的候选还使用 WikiText-2/MMLU（Llama 另用 CSR）选择，不能将这些任务直接看成独立留出选型证据。[1，§4.1；附录 B/A]

两位 XXS 对各模型该精度最强多精度基线的 MMLU 改善分别约 3.1、4.0、7.0 点，logical rate 略低；Llama 的 38.38 对 35.33 对应 2.35 对 2.38 bpw。三位 Llama 的 61.89 对 58.13 对应 3.35 对 3.50 bpw；Phi 的约 1.2 点改善伴随约 0.33 bpw 增加；OLMo 的 0.9 点在作者 1-point tie band 内。四位以上 XXS 与 Any-Precision 的 MMLU 视为持平，而后者更省 bits。不能概括成所有精度、所有质量指标和匹配存储下都更优。[1，表 1–2]

这些是完整 encoder 配置的比较：细组、保护与 pair 拟合一起变化。固定 $g=32$ 的 guard 消融虽将两位 PPL 从 28.47 降到 20.86，两位 MMLU 却从 26.90 变为 25.09；不能把完整 XXS 的提升全归给 guard。作者没有分别消融 $\lambda$、AWQ 指数与 pair 拟合。单精度控制只在同一 encoder/Llama 内解除前缀约束，两位 dedicated 高 3.62 MMLU 点但 rate 略增；它也不是与最佳任意固定精度量化器的比较。[1，表 3–4；附录 E/A]

性能证据要分三层：[1，§4.3；附录 F/G/A]

- **微基准：** 单张 A100 40GB、batch-one、九形状×三精度的合成权重 GEMV，多数情况下快于两个基线；attention 形状预留 guard 空间，六个 FFN 形状没有，后者偏向 StagQ；该表还省略 activation scale。
- **Decoder 计时：** 同为合成权重、batch-one，StagQ 额外融合格式无关算子，18 个比较中赢 14 个；三位关闭这些融合和 per-channel multiply 后，在三模型上均比最快基线慢 0.3%/0.6%/2.9%。OLMo 省略 per-channel multiply 也偏向 StagQ。这不隔离格式本身的速度贡献。
- **真实权重数值检查：** Llama 两位 decode 在前四个 WikiText-2 blocks 的 PPL 相对相同权重 FP16 执行增加 0.0086，低于作者 0.10 容差；没有因此得到全模型/全部精度的真实 serving 速度证据。

## 7. 对现有知识的增量与后续边界

StagQ 将多精度可读性、元数据和 decoder 放在同一约束下：共享 pair 让细组便宜，dyadic digits 让多项重建折成 affine 整数，pre/post guard 保护难以追回的尾部，同时带来所有精度都必须承担的固定记录。它补足[表示设计](../research/low-bit-representation-design.md)中“例外释放主体容量”的多精度情况，并为[预算口径](../theory/mixed-precision-allocation.md)提供 logical rate 与实际固定 slots 读取不同的例子。

[DBW](dbw.md)在低秩分量上联合分配 rank/bit，StagQ 则对原权重主体建立多个前缀 operating points；两者的选择单元和 decoder 不同，名义 2 bit 不能直接对比压缩率或吞吐。对具体部署，应先核全部字段/布局、精度切换加载成本、保护位置重建、量化逆变换与完整质量，再测实际 decoder/服务成本。

当前证据限于论文给出的 dense 模型、配置和单卡 batch-one 描述；MoE、更大模型、生产服务中的运行时切换尚是后续方向。JPEG2000 附录 H 用于辨认 embedded quantization 的背景，不将其 Laplacian 重建常数直接套到 LLM。

## 来源身份

- StagQ: Constraint-Driven Multi-Precision Weight Quantization for LLMs（`https://arxiv.org/abs/2610.05977v1`），`arXiv:2610.05977v1`，2026-10-05 首次提交；2026-10-07 取得固定 PDF [1] 和同版 TeX 源包 [2]。使用正文和附录 A–G，核对关键公式、代表表和页图，未重算全部任务附表。章节、式号、表号按该版本。正文代数展开和预算解释为整理者说明，表中性能与质量均为作者报告。
