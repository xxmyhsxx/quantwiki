---
title: QJL：用二值随机投影直接估计注意力分数
type: method
tags:
  - kv-cache
  - random-projection
  - attention
sources:
  - raw/papers/2026-09-22/qjl/paper.pdf
updated: 2026-09-28
---

# QJL：用二值随机投影直接估计注意力分数

QJL v2 对 Key 保存随机投影的符号和向量范数，用未二值化的 Query 投影直接估计内积；Value 仍采用常规逐 token 标量量化。其目标是保留 attention 所需的分数，不要求先恢复一个逐坐标接近原向量的 Key。

本页研读完整正文，回看原 PDF 的估计器、概率界与算法。原文理论推导有符号及尺度依赖问题，下面将可直接验证的恒等式、数量级分析、实践修改和作者实验分开；本地未归档代码，也未运行模型。前置对象见 [KV cache 粒度](../theory/kv-cache-quantization-objects-and-granularity.md)。

## 1. 二值化哪一侧，保存什么

设单个 Key、Query 为 $k,q\in\mathbb R^d$，取共享矩阵 $S\in\mathbb R^{m\times d}$，每个元素独立服从标准高斯分布。Key 到达时保存

$$h(k)=\operatorname{sign}(Sk)\in\{-1,+1\}^m,\qquad \nu_k=\|k\|_2.$$

Query 到达时算浮点 $Sq$，再对每个历史 Key 计算

$$\widehat a(q,k)=\frac{\sqrt{\pi/2}}m\nu_k\langle Sq,h(k)\rangle.$$

这是原文定义 3.1 与算法 1。两侧都取 sign 会变成另一类角相似度估计，不能继续使用这个内积公式。这里的“1 bit”是**每个投影坐标一位**，每个 Key 有 $m$ 位符号码，并另存范数；不等于每个原通道一位，更不等于整个 K/V 缓存一位。

对 $k=0$，由范数因子定义估计为零，无需把零向量纳入后面的方向分解。进入标准 attention 时还要除以 $\sqrt d$；原文算法的简化 softmax 记法省略了这个因子，工程比较必须使用同一个分数尺度。

## 2. 为什么不恢复 Key 仍能估计内积

以下是对原文无偏性证明的等价展开。固定非零 $k$，令 $u=k/\|k\|$，将 $q=(q^Tu)u+q_\perp$。对一行高斯向量 $s$，$s^Tu$ 与 $s^Tq_\perp$ 独立，后者零均值，所以

$$E[(s^Tq)\operatorname{sign}(s^Tk)]
=(q^Tu)E|s^Tu|=(q^Tu)\sqrt{2/\pi}.$$

乘回 $\sqrt{\pi/2}\|k\|$ 即得 $E[\widehat a]=q^Tk$。期望是对随机矩阵 $S$ 而言，要求被估计的向量固定或独立于这次随机性；不是说某一次部署抽样的误差为零，也不是说 softmax 或生成答案无偏。

还可直接算出方差。令单行估计 $z=\sqrt{\pi/2}\|k\|(s^Tq)\operatorname{sign}(s^Tk)$，则

$$E[z^2]=\frac\pi2\|q\|^2\|k\|^2,\qquad
\operatorname{Var}(\widehat a)=\frac{(\pi/2)\|q\|^2\|k\|^2-(q^Tk)^2}{m}.$$

这是整理者由高斯二阶矩得到的精确式。它解释了增加投影数怎样降低方差，也解释了范数与离群通道为什么影响绝对误差。误差由 $\|q\|\|k\|$ 控制，并非对可能接近零的 $q^Tk$ 保证小相对误差。

## 3. 高概率保证应保留到什么程度

高斯投影的绝对值具有指数尾部，用集中不等式可以得到相对范数误差 $\epsilon\|q\|\|k\|$ 所需投影数的量级

$$m=O\!\left(\epsilon^{-2}\log(1/\delta)\right),\qquad 0<\epsilon\le1.$$

对一个固定 Query 和 $n$ 个固定 Key 做 union bound，将 $\delta$ 换成 $\delta/n$，得到 $O(\epsilon^{-2}\log(n/\delta))$。这解释了投影预算随缓存长度的对数关系，但**本页不采用原文给出的精确常数**。

原 PDF 第 6 页引理 3.5 证明中的两处尾概率指数写为正号，不能由它们推出随 $m$ 增加而收紧的概率界；第 7 页定理 3.6 又给出 $m\ge2r^2\epsilon^{-2}\log n$，但其证明把引理的相对容差换成 $\epsilon/r^2$。在 $\|q\|,\|k\|\le r$、追求未缩放内积的绝对误差 $\epsilon$ 时，这一步对应的主导依赖应为 $r^4/\epsilon^2$，并非 $r^2/\epsilon^2$。这些问题已经回看原 PDF，不是文字提取损坏；仅补一个负号不能视为完成整套常数证明。

**可精确复算的常数反例（整理者推导）。** 取 $q=(1,0)$、$k=(0,1)$。每行的 $s_1\operatorname{sign}(s_2)$ 仍是标准正态，所以 $\widehat a\sim N(0,\pi/(2m))$，真实内积为零。令 $\epsilon=0.1,\delta=0.05$，引理给出的下限向上取整为 $m=542$；但实际尾概率为

$$P(|\widehat a|>0.1)=\operatorname{erfc}\!\left(0.1\sqrt{542/\pi}\right)\approx0.06323>0.05.$$

其中 $\operatorname{erfc}$ 是互补误差函数。这个独立高斯特例直接反驳所印精确常数，不否定无偏性或大 $O$ 数量级。其[标准库计算脚本](../assets/qjl/verify_estimator.py)与[固定结果](../assets/qjl/estimator-check.json)可复算；不是模型实验，也不对实践正交化分支作相同分布断言。

对实际 attention，更稳妥的表述是直接给出归一化分数误差。设 $\ell_j=q^Tk_j/\sqrt d$，若对所有 $j$ 有 $|\widehat\ell_j-\ell_j|\le\eta$，则

$$e^{-2\eta}\le\frac{\widehat p_j}{p_j}\le e^{2\eta},\qquad p=\operatorname{softmax}(\ell).$$

分子指数比在 $[e^{-\eta},e^\eta]$，分母比也在同一区间，故得此式。这是确定性结论。原文进一步写成 $1\pm3\epsilon$ 需要限制小误差范围；不能对任意正 $\epsilon$ 使用线性界。例如 $\epsilon=0.5$ 时 $e^{2\epsilon}>1+3\epsilon$。

还要区分“固定 Query 下同时覆盖全部 Key”与“同一个 $S$ 对任意自适应 Query 都有效”。后续隐状态可能依赖先前的压缩结果，因此固定向量的概率证明不能自动成为整条自回归轨迹的统一保证。Value 量化误差也不在这个 Key 分数定理内。

## 4. 实践为何又加入离群值与正交化

**离群通道单独编码。** 作者观察到深层 Key 的高幅通道相对稳定，将这些通道分出，用独立 QJL 实例并分配更多位。对通道分区 $k=(k_o,k_b)$、$q=(q_o,q_b)$，内积是两部分之和；分别估计再相加，需分别保留范数、投影配置和通道位置。多存的位预算用于控制高范数部分的误差，不能只核算主体符号码。

**正交化投影。** 作者通过 QR 正交化 $S$ 的行，报告质量改善。但独立高斯行的证明不直接适用于相关的正交行，归一化和缩放也会影响无偏常数。若 $m>d$，不可能让全部 $m$ 行在 $\mathbb R^d$ 中两两正交，需要说明分块等构造；v2 正文没有足够实现细节让本页确定所有配置的处理方式。

这与 [Hadamard 等价旋转](../theory/orthogonal-rotation-and-hadamard-quantization.md) 不同：等价旋转利用可逆配对保持未量化内积；QJL 的投影、符号压缩与统计估计一般不可逆。不能因为实践里有“正交化”就认为其输出精确等价。

**Value 独立量化。** 按 token 求范围、归一化并舍入到少量位的整数；仍需要恢复用的尺度、偏移等元数据。Key 分数估计不能代替 $\widehat P\widehat V$ 中 Value 侧的误差分析。

## 5. 数据表示与运行成本

共享投影矩阵只需存一次，但写每个 Key 要做投影与 bit packing，读每个 Query 要做 $Sq$ 和对历史符号码的估计。作者实现了量化及内积估计两个 CUDA 核心 kernel，外层缓存管理由 PyTorch 承担；没有证据表明所有服务调度和分页开销都已融合。

若 Key 本体有 $m$ 个符号位、范数用 $b_\nu$ 位，每个原元素的存储至少为

$$b_K=\frac{m+b_\nu}{d},$$

还要加离群分支、通道索引、共享矩阵的摊销与对齐。Value 另计其码字、尺度和偏移；等大 K/V 的平均位宽才可写为 $(b_K+b_V)/2$。标题“Zero Overhead”不能解释成没有任何元数据或在线运算，算法 1 自身就明确保存 $\nu_k$。

相比 [KVQuant](kvquant.md) 以离线码本恢复 Key 再做乘加，QJL 改变了读取时算分数的接口；相比 [KIVI](kivi.md) 的组内标量网格，投影数和离群预算成为新的参数。三者需要按同一真实字节数、任务与执行路径比较，不能只看“2/3/4 bit”标签。

## 6. 作者实验支持的范围

v2 实验在单张 A100 80GB 上进行，模型质量包括 LongChat-7B-v1.5-32K 的六项 LongBench QA，以及 Llama-2-7B/Llama-3-8B 的短任务。论文分别写到 16,384 的上下文设置与 31,500 的最大输入长度，具体截断/执行关系未交代清楚；没有代码核对时不能合并成一个已确认协议。

在表 1 的标称 3 bit 设置下，QJL 的 Qasper 为 29.44，FP16 为 29.42；TriviaQA 为 41.52，FP16 为 42.83；这些数据支持所测配置大体接近，不能写成所有任务无损。只需很少生成步的短任务也不能充分检验压缩缓存累积误差。

图 3 的计时明确是**单层 attention 模型**：Llama-2 生成 128 token，Llama-3 生成 64 token，图中长度覆盖 1K–64K。正文另提到 128K，图本身没有展示这一端点。作者报告 Llama-2 解码可比浮点快、Llama-3 大体相当，同时减少缓存占用；这不是完整模型的端到端吞吐测试，更不是分页、多请求服务结果。

文中“KIVI 不支持 Llama-3”等说法指当时比较实现的数据类型与 GQA 支持，不是算法层面的不兼容结论。本页不将这些历史实现状态外推到当前框架。约 81% 缓存压缩对应作者的标称三位口径，不能不核元数据便作为整机显存下降比例。

## 来源身份

- [QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead](https://arxiv.org/abs/2406.03482v2)，arXiv:2406.03482v2；2026-09-22 归档。定义 3.1、引理 3.2/3.5、定理 3.6、算法 1、§4 与图 3；精确方差及 softmax 指数界为整理者展开，原文常数证明问题保留。
