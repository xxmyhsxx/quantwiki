---
title: 量化矩阵乘法的缩放与执行路径
type: implementation
tags:
  - matmul
  - kernels
  - data-format
  - performance
sources:
  - raw/papers/2026-09-22/loftq/paper.pdf
  - raw/papers/2026-09-22/qera/paper.pdf
  - raw/papers/2026-09-21/billm/paper.pdf
  - raw/papers/2026-09-21/luq/paper.pdf
  - raw/papers/2026-09-21/qig/paper.pdf
  - raw/papers/2026-09-21/splitq/paper.pdf
  - raw/papers/2026-09-21/masquant/paper.pdf
  - raw/papers/2026-09-21/spinquant/paper.pdf
  - raw/papers/2026-09-21/quarot/paper.pdf
  - raw/papers/2026-09-21/flatquant/paper.pdf
  - raw/papers/2026-09-21/omniquant/paper.pdf
  - https://github.com/OpenGVLab/OmniQuant/blob/feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14/models/int_llama_layer.py
  - raw/papers/2026-09-21/qlora/paper.pdf
  - raw/papers/2026-09-21/mquant/paper.pdf
  - raw/papers/2026-09-21/vlmq/paper.pdf
  - raw/papers/2026-09-21/mbq/paper.pdf
  - raw/papers/2026-09-21/quantization-white-paper/paper.pdf
  - raw/papers/2026-09-21/integer-only-quantization/paper.pdf
  - raw/papers/2026-09-21/smoothquant/paper.pdf
  - raw/papers/2026-09-21/gptq/paper.pdf
  - raw/repositories/2026-09-21/smoothquant/source/smoothquant/fake_quant.py
  - raw/repositories/2026-09-21/gptq/source/quant.py
  - raw/repositories/2026-09-21/torch-int/source/torch_int/nn/linear.py
updated: 2026-09-22
---

# 量化矩阵乘法的缩放与执行路径

同样写着低比特量化，实际可能是浮点模拟、压缩权重配合即时反量化，或整数输入参与矩阵乘法。理解性能前，需要先明确数据怎么存、乘法吃什么类型、在哪里缩放，以及运行时的主要开销。

本页依据 SmoothQuant v7 §2–4、式 (2)、附录 A及 GPTQ v2 §5、附录 A.2.2展开。公式为对原文的教学推导，代码范围和历史测量在各方法页限定；没有本项目性能实测。

## 1. 矩阵乘法的外维与归约维

使用行 token 约定 $X\in\mathbb R^{T\times d}$、$W\in\mathbb R^{d\times r}$，$Y=XW$。一个输出值

$$
Y_{to}=\sum_{j=1}^{d}X_{tj}W_{jo}
$$

把输入通道 $j$ 求和消掉，所以 $d$ 是归约维；token $t$ 和输出通道 $o$ 是保留的外维。PyTorch 权重存为 $r\times d$，对应计算转置，不改变这个区别。两种矩阵约定与通道含义见 [线性层与输入通道](../fundamentals/operators/linear-layer-input-channel.md)。

量化步长、编码与粒度的定义见 [均匀量化与分组](../fundamentals/quantization/uniform-quantization-and-groups.md)。对称量化时，若激活按 token、权重按输出通道共享步长，

$$
X_{tj}\approx a_tq^X_{tj},\quad W_{jo}\approx b_oq^W_{jo},
\quad Y_{to}\approx a_tb_o\sum_jq^X_{tj}q^W_{jo}.
$$

整数乘法与较宽整数累加完成后，再做外维缩放。较宽累加、结果转换、bias 和融合接口仍须查后端，W8A8 不等于输出或累加只有 8 bit。[SmoothQuant 式 (2)、图 3]。

若激活步长为逐输入通道 $a_j$，则

$$
Y_{to}\approx b_o\sum_j a_jq^X_{tj}q^W_{jo}.
$$

$a_j$ 留在归约内部，不能直接提到求和外面。例如某次整数乘积为 $(1,1)$，对应 $a=(1,2)$，真实加权和为 3；另一输出若乘积为 $(1,-1)$，整数和为 0、加权和却为 -1，单一后置缩放无法恢复。分组部分和或定制运算可处理这种情况，但不再是一次普通整数 GEMM 加行列缩放的同一执行形式。

SmoothQuant 先离线调整权重与前驱参数，改善激活输入通道范围，随后采用外维可分离的量化方式。其逐输入通道平滑因子 $s_j$ 不应误写成运行时 GEMM 的激活步长 $a_t$。离线缩放如何保持浮点函数、何时能融合到前驱参数，见 [对角缩放与等价变换](../theory/diagonal-scaling-equivalent-transform.md)；本节讨论的是变换之后的运行时量化与乘法。

模态静态尺度也可位于外维：令 $a_t=s_{m(t)}$，同一模态的 token 共享一个离线尺度，整数累加后仍可乘 $s_{m(t)}b_o$。数学上可分离，不代表交错布局下没有路由开销；[MQuant](../methods/mquant.md) 的 AIFS 将模态位置聚集，并同步置换注意力 mask 与原位置编码，试图减少切片和拼接。其表 10 分别测量 MSQ 与 MSQ+AIFS，说明网格、布局与内核应分开评价。（MQuant v2，§3.1、表 10）

## 2. zero-point 会引入哪些额外项

若 $X_{tj}\approx a_t(q^X_{tj}-z^X_t)$、$W_{jo}\approx b_o(q^W_{jo}-z^W_o)$，则

$$
Y_{to}\approx a_tb_o\left[
\sum_jq^X_{tj}q^W_{jo}
-z^W_o\sum_jq^X_{tj}
-z^X_t\sum_jq^W_{jo}
+d z^X_tz^W_o\right].
$$

这是从仿射量化定义得到的展开，说明 zero-point 会带来修正项；能否预计算或融合取决于参数是否静态、粒度和后端。本页不据此声称某个具体内核已经实现这些项。SmoothQuant 的简化推导主要采用对称量化；GPTQ 原始权重量化常使用非对称网格，其反量化路径包含 zero-point。

## 3. 三类执行路径

| 路径 | 数据和计算 | 可以支持什么判断 |
|---|---|---|
| 模拟量化 | 将数值舍入到低比特网格，仍用浮点存储与乘法 | 观察该模拟规则下的质量变化；不能证明压缩或加速 |
| Weight-only 压缩 + 即时反量化 | 打包权重，运行时解码为浮点，与浮点激活相乘 | 减少权重存储和访存；收益依赖反量化及权重复用 |
| W8A8 整数矩阵乘 | 权重与输入以 INT8 交给整数矩阵乘接口，随后缩放/转换 | 可利用对应整数算力；额外量化、非矩阵算子与显存仍影响端到端收益 |

可核对的固定版本例子：

- SmoothQuant C `c61476d…` 的 `fake_quant.py:103`使用浮点 `F.linear`，属于第一类。
- GPTQ C `2d65066e…` 的 `quant.py:137`保存 3-bit packed 权重，forward 调用单向量 CUDA 路径，属于第二类；本轮未运行内核。
- torch-int T `65266db1…` 的 `linear.py:16`保存 INT8 权重并调用整数扩展，属于第三类接口；本轮未验证底层 kernel 的性能和数值。

第二类路径的性能上限与批量强相关：[Marlin](marlin-batched-w4a16-gemm.md) 在 A10 上把接近理论上限的加速保持到 batch 约 16–32，之后随进入计算受限区间逐步下降，它用离线重排、异步拷贝与条带划分把权重读取的重叠做到接近峰值。第三类路径的困难在反量化发生的位置：[QServe](../methods/qserve.md) 论证按组 W4A4 在主循环内反量化部分和会被 CUDA 核心限制，因而改用渐进式分组量化，让全部计算落在 INT8 张量核心上。

[Q-VLM](../methods/q-vlm.md) 固定代码提供一种组合路径：NF4 权重压缩、激活量化后恢复浮点、多 token 时反量化权重再做浮点线性运算。QLoRA v1 §3 同样明确低比特存储与通常 BF16 计算分开。因而看到激活量化节点和 W4A4 标签，也不能直接归为原生整数 GEMM；[非均匀码本与尺度元数据](../fundamentals/quantization/uniform-quantization-and-groups.md) 决定解码需要哪些信息。

## 4. 为什么 token 数改变瓶颈

矩阵乘约需 $2Tdr$ 次浮点运算等价计数。仅看读取一次权重的理想流量：FP16 为 $2dr$ 字节，$b$-bit 权重为 $bdr/8$ 字节。于是仅由权重流量估算的算术强度是

$$
I_{\mathrm{weight,FP16}}\approx T,
\qquad I_{\mathrm{weight},b}\approx16T/b
\quad\text{（FLOP/Byte）}.
$$

这只是解释趋势的理想模型，忽略了激活、输出、元数据、重复读取和其他层。$T$ 很小时，减少权重读取尤其有价值；$T$ 增大，权重被多个 token 复用，计算负荷相对提高。因此 GPTQ 原始单 token kernel 与 SmoothQuant context/batched 测量的目标不同。[GPTQ A.2.2；SmoothQuant 附录 A]。

真实时间还包含解包、量化统计、缩放、缓存、设备通信和其他算子。用相同比例缩小权重，不意味着同倍数降低总显存或延迟。长上下文的 KV cache 也可能改变主要内存占用；是否压缩 cache 必须查实际路径。

上面的理想模型只预测瓶颈方向，能否维持由内核实现决定。Marlin 的测量给出一个具体刻度：A10 的算力带宽比约为 200 时，batch 小于约 50 仍受权重读取限制，但既有单 token 内核在 batch 增大后收益迅速消失，只有针对批处理重新设计的调度才能在该区间维持接近理论上限的加速（[Marlin](marlin-batched-w4a16-gemm.md) §3.1 与内核基准图）。因此“低比特权重加大 batch”不会自动得到低比特级别的加速。

再往下还差一层：内核要求的数据布局与配置匹配条件（反量化位技巧、元数据对齐、模板命中）属于实现契约，见 [权重量化反量化内核的契约](weight-only-dequant-kernels.md)。

## 5. 应怎样记录性能证据

性能比较至少绑定模型/权重版本、输入输出长度、batch、prefill/decode、位宽与粒度、GPU 型号和数量、后端、并行方式、延迟聚合与显存口径。只知道两个“加速倍数”，无法判断哪个方案在用户的场景更好。

格式名与实际内核的对应关系由部署框架决定：同一算法可能同时存在原生内核与批处理内核两套实现，框架列出的格式也不代表都走了高效路径，见 [部署框架与后端支持](quantized-llm-deployment-backends.md)。

GPTQ 的历史结果和单向量限制见 [GPTQ §7–8](../methods/gptq.md)；SmoothQuant 的后端、减卡实验和 decode 协议缺口见 [SmoothQuant §7](../methods/smoothquant.md)。本页没有为它们补造统一基准。

视觉语言模型还包含视觉编码器、连接器和图文 prefill，整体时间分解见 [视觉语言模型的阶段成本](../fundamentals/model/vision-language-model-tokens-and-quantization.md)。[MBQ](../methods/mbq.md) 分别报告 RTX 4090 上视觉编码、prefill 与 decode；[VLMQ](../methods/vlmq.md) 的 RTX 5090 结果则是线性层微基准。两者不能合并为一个“多模态端到端加速比”，也不能由格式兼容直接推出所有 GPTQ 后端都支持相同位宽与分组。（MBQ v2 表 8–9；VLMQ v2 表 11。）

## 6. 从整数累加到下一层网格

本节依据 Jacob 等整数推理论文 arXiv v1（J，2017）§2、图 1.1、附录 A–C，展开一种静态仿射整数执行方案。它补足前面的整数矩阵乘法，不表示所有 W8A8 后端都采用相同输出类型。

考虑单个输出 $y=\sum_{j=1}^d x_jw_j+b$。输入、权重步长分别为 $\Delta_x,\Delta_w$，零点为 $z_x,z_w$，先算整数累加值

$$
A=\sum_j(q^x_j-z_x)(q^w_j-z_w)+q_b,\qquad
q_b=\operatorname{round}\!\left(\frac{b}{\Delta_x\Delta_w}\right).
$$

$A$ 与 $q_b$ 的实数单位都是 $\Delta_x\Delta_w$；bias 用 int32，零点为 0。反量化累加值为 $\widehat y_{\mathrm{acc}}=\Delta_x\Delta_w A$。在无溢出条件下，整数乘加对编码后的数值是精确的，输入/权重/bias 的量化仍可能引入误差。J 图 1.1 的 uint32 标签与正文 §2.4 的 int32 不一致；这里采用正文的有符号定义。

如果下一层输入网格为 $(\Delta_y,z_y)$，理想再量化规则是

$$
q_y=\operatorname{clip}\left(z_y+\operatorname{round}(MA),q^y_{\min},q^y_{\max}\right),
\qquad M=\frac{\Delta_x\Delta_w}{\Delta_y}.
$$

再量化将一个网格的整数值转到另一网格，包含缩放、舍入及饱和，不能省略成“把 int32 转成 int8”。融合 ReLU 时实数下界 0 对应编码 $z_y$，不是一般的整数 0；ReLU6 上界也需要映射到输出网格。（J §2.4。）

**教学例子：**$q_x=(7,3),z_x=5,\Delta_x=0.5$ 表示 $(1,-1)$；$q_w=(6,2),z_w=4,\Delta_w=0.25$ 表示 $(0.5,-0.5)$。取 $b=0.25$，则 $q_b=2,A=2\times2+(-2)\times(-2)+2=10$，累加结果表示 1.25。若 $\Delta_y=1,z_y=8$，则 $M=1/8,q_y=9$，输出表示 1。最后的差来自输出网格舍入，而不是点积错误。

若权重按输出通道量化，bias 的单位和 $M$ 也逐输出通道变化；若激活步长随 token 动态变化，bias 单位可能随之变化，不能直接复用一个固定 $q_b$。J 的静态 per-tensor 方案不自动解决这种动态实现问题。

## 7. 固定点乘法、宽度与舍入

J §2.2 对其经验上 $0<M<1$ 的情形，写成 $M=2^{-n}M_0$，$M_0\in[0.5,1)$，用整数近似 $2^{31}M_0$ 实现固定点乘法，再右移 $n$ 位。普通量化步长可以不是 2 的幂；integer-only 不要求所有步长都是 power-of-two。乘法系数需要离线准备，运行时可使用整数运算。

一般量化配置也可能有 $M\ge1$，需要额外左移或其他表示，并重新分析溢出。系数舍入到表示边界、乘法与右移两次舍入，也需要实际后端定义；不能保证它与上节单次理想 $\operatorname{round}(MA)$ 逐位相同。

J 附录 B 特别说明舍入指令选择。以 -12 除以 8 为例，中点为 -1.5，向上打破平局得到 -1，按作者要求的中点远离零得到 -2。仅写“round-to-nearest”不足以定义所有中点行为；例如 +2.5 在 ties-to-even 下为 2，在 ties-away-from-zero 下为 3。负数偏差可能跨层累积，训练模拟与部署规则需匹配。

int32 也不是无条件安全。若中心化输入绝对值至多 $Q_x$、权重至多 $Q_w$，一个保守充分条件是

$$
dQ_xQ_w+|q_b|\le2^{31}-1.
$$

这比只写“使用宽累加器”更明确。输入从 uint8 转为 int8 时，编码与零点都减 128 才保持实数值不变。

J 附录 B 的特定 ARM NEON 优化让权重不取 -128，从而可将两个 int8 乘积先累加到 int16，再汇入 int32：$2\times128\times127=32512<32767$；若允许两次 $(-128)(-128)$，和为 32768，会溢出 int16。这解释的是该微内核为何使用窄权重范围，不是所有 INT8 量化器都必须放弃一个编码。本轮没有运行该历史 ARM 实现。

J §2.3 将零点修正组织为行/列和，对 $N\times N$ 矩阵辅助工作为 $O(N^2)$、乘法为 $O(N^3)$。这与白皮书 §2.3.4 所说非对称权重可能增加运行时成本并不矛盾：形状、权重复用、动态统计与融合方式决定占比，不能推广成向量乘或所有后端都免费。

## 8. 残差、拼接与模拟边界

若 $u=\Delta_1(q_1-z_1)$、$v=\Delta_2(q_2-z_2)$，输出网格为 $(\Delta_o,z_o)$，理想加法为

$$
q_o=\operatorname{clip}\left[z_o+\operatorname{round}\left(
\frac{\Delta_1}{\Delta_o}(q_1-z_1)+\frac{\Delta_2}{\Delta_o}(q_2-z_2)
\right),q^o_{\min},q^o_{\max}\right].
$$

三者共用同一网格时简化为裁剪后的 $q_1+q_2-z$。不同网格需要尺度对齐；J 附录 A.2 的实现叙述先对齐一支再转换输出，可能多一次舍入。拼接在各输入输出网格相同时无需数值再量化；J 附录 A.3 用绑定量化参数实现这一点，但“无算术转换”不等于没有内存访问成本。

J 附录 C.4 在残差相加前后区分量化节点，C.8 在 BN 融合后的权重上模拟量化。[QAT 的前向图](../fundamentals/quantization/post-training-and-quantization-aware-training.md)应反映部署的实际边界，而非任意逐算子插入量化。附录 A.1 还指出非线性函数可以采用固定点计算，但没有完整展开所有函数算法；这不能证明一个含 LayerNorm、GELU 等算子的模型已被本轮实现为全整数推理。

进一步验证要区分代数、舍入一致性和模型质量，见 [量化误差诊断与验证](quantization-error-diagnosis.md)。

## 9. 计算模拟、缓存存储和后端收益分别验证

[OmniQuant](../methods/omniquant.md) 提供一个具体边界案例：默认线性层和注意力输入被映射到低比特网格后以浮点数参与计算；官方 `real_quant` 另行打包纯权重，而论文表 3 的实际速度来自 MLC-LLM。不能将默认模拟路径、打包路径与论文的后端测试视为同一次验证。（OmniQuant v3 §4.5；各代码位置见方法页。）

KV 还需核对缓存写入位置。官方 commit `feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14` 的 `models/int_llama_layer.py:QuantLlamaAttention.forward` 先把 K/V 写入 `past_key_value`，再对参与矩阵乘的 K/V 做 fake quant；因此缓存仍是浮点张量。要获得低比特缓存占用，必须另有整数码与 scale/zero-point 的存储、读取和计算路径；仅对读取值做量化模拟不会缩小已存缓存。

论文表 3 在 A100-80G、MLC-LLM、生成 512 tokens 的 7B 测试中，W4A16g128 为 134.2 token/s，W3A16g128 为 83.4，FP16 为 69.2。该实例说明低位宽存储更小也可能运行更慢，结论依赖打包、访存和 kernel 支持；不代表当前所有后端的性能顺序。论文没有相应的 W4A4/W6A6 实际加速结果。

## 10. 在线变换的代价与融合

[FlatQuant](../methods/flatquant.md) v4 §3.3、附录 B.3 展示一种可复用的优化：将两次小矩阵变换与范围计算、量化融合，减少中间张量显存读写和 kernel 启动，再交给 INT4 矩阵乘。变换算术仍在；片上容量不足时需要切片或额外读写，并非所有形状都能完全融合。

其附录 C.8 在 RTX 3090、LLaMA-2-7B、2048-token prefill 后 decode 256 tokens 的测试中，batch 小于 16 时 decode 加速比低于 1，而 batch 64 才达到约 1.76×。压缩 KV 的访存收益需抵扣量化与变换开销；不能用大 batch 吞吐替代单请求延迟，也不能把 kernel 融合自身的加速比写成全模型加速比。论文计时与后续真实权重部署的差别在方法页分别记录。

## 11. 旋转路线怎样区分存储、算子和系统结果

[QuaRot](../methods/quarot.md) v2 §4 的 W4A4KV4 将线性权重/输入交给 INT4 GEMM、INT32 累加后返回 FP16；四位 K/V 是压缩缓存，attention 读取后仍做浮点计算。真实缓存初始化还能一边存入四位 cache，一边把当前浮点 K/V 用于 prefill。因此“cache 为四位”“QK 点积为四位”和“整个模型是整数图”是三个独立主张，应逐项核对实现。

在线变换必须计入所在执行阶段。QuaRot 表 15 的单次追加/attention（RTX 3090，32 个宽度 128 的头、已有 2047 token），batch 1 的 INT4 cache 加 FP32 Hadamard 比 FP16 慢，batch 32 则更快；表 16 的约 3.33 倍是 LLaMA-2-70B 形状、2048 token、batch 32 的单 Transformer block prefill，不能当作整模型解码延迟。

[SpinQuant](../methods/spinquant.md) 的 R1/R2 可离线融合，但 R3/R4 仍在线。其 v4 表 6 测量的是 M1 Pro CPU 上 W4A8，表 14 则是 H100 上 FP8 权重/激活的 Hadamard 附加成本，二者都不能直接配上 W4A4KV4 精度结果声称同一配置达到该速度。比较方法应把位宽、粒度、硬件、模型版本、batch、序列长度、阶段和测量单位一并固定。

## 12. 低秩辅助分支不是免费或统一位宽

一个 $d\times o$ 主体矩阵加秩 $r$ 分支，需要额外保存 $r(d+o)$ 个因子元素，并对适用 token 执行两次小矩阵乘。若主体为 $b$ bit、因子为 $p$ bit，额外权重字节相对主体约为 $r(d+o)p/(dob)$；方阵、$r/d=2\%$、$b=4,p=16$ 时约 16%。这是教学存储估算，不含尺度、缓存、对齐及其他模块，不能当作整个模型显存比例。

[MASQuant](../methods/masquant.md) v1 式 29 的旁路使用未量化激活；[SplitQ](../methods/splitq.md) v1 式 17、20 则分别量化权重低秩因子和激活残差。SplitQ 在主体低于四位时仍保留至少四位的辅助路径。两者即使主体都写 W4A4，存储、激活中间量和运算图也不相同。

SplitQ 表 12 的 RTX 4090、Qwen2.5-VL 7B、2048 token、W4A4、batch 1 prefill，从仅 MOCD 的 70.99 ms 到加 CWS 的 75.55 ms，再到全方法 81.78 ms，直接说明辅助分支有成本。此结果不包含 W3A2 内核或 decode。MASQuant 对文字 decode 不需补偿的结构动机，也仍要由真实条件跳过、融合和计时确认；零 mask 的矩阵乘法未必被实际省掉。

比较这些方法至少应同时给出主体与旁路位宽、实际秩、适用 token、变换成本以及 prefill/decode；不能只按论文的 W/A 标签匹配速度和质量。

[QERA](../methods/qera.md) v2 附录 A.7–A.8 进一步区分了离线分解与推理：exact 比 approx 多计算完整二阶矩与矩阵平方根，但在相同秩、dtype 和执行方式下，都可保存为 $x\widetilde W+(xA)B$。这只说明 exact 没有引入额外种类的在线分支，不说明两者相对无补偿模型没有成本。教学例：$4096\times4096$ 主体、秩 32、FP16 因子，会新增 262,144 个因子元素，即 0.5 MiB，摊到主体参数上增加 0.25 bit/weight；若主体含尺度为 4.25 bit/weight，二者合计 4.50，仍未包含其他模块与运行缓冲。把 $AB$ 合并回主体后再次量化，又会改变残差，不能继续沿用合并前的精度保证。

## 13. 校准成本与混合格式部署

[LoftQ](../methods/loftq.md) v4 §3.3 冻结量化基座、只更新低秩适配器，可以省去基座参数的梯度与优化器状态；这不免除反量化、激活保存或向前面可训练模块传播梯度。其附录 B、表 9 的 21 秒／43 秒分别是 Xeon E5-2650 v4 @ 2.20GHz CPU 对单个 $4096\times4096$／$5120\times5120$ 矩阵做 5 轮 NF4 初始化的时间，不能当作整模型初始化或推理延迟。附录 A 的压缩比例则把主体与适配器一起除以原预训练大小，属于存储保留比例，不能替代训练峰值显存。

[QIG](../methods/qig.md) v1 表 6 报告的是 A800 上收集激活、归因及缩放搜索的总时间，不能当作推理速度。其固定快照的 W/A 路径使用浮点缓冲区和 `F.linear`；推理不用重新计算 IG，不意味着当前代码已经实现了实际 W4A8 加速。

[LUQ](../methods/luq.md) v3 附录 D 提供另一种边界：主实验选层时使用 BiLLM 1.08/GPTQ 4 参数位宽，部署测试改用 IQ1_M（约 1.75 bpw）/Q4_K_M。保留相同层配置不等于保留相同量化权重与精度，主表质量不能直接与替换格式后的速度组成一个已验证结果。

其 Qwen2.5-VL-7B 表 5 在 i7-13620H 上从 4.8 到 9.0 tokens/s，在 Threadripper PRO 7965WX 上从 14.1 到 18.7；表列内存从 4.4GB 到 3.4GB。这是作者的 10 次生成吞吐与内存报告，缺少完整输入形状、线程/后端版本及视觉端计时条件，不概括为整段图文请求的时延。

[混合精度预算](../theory/mixed-precision-allocation.md) 还需把编码参数、scale、分组/索引、未量化模块和运行缓冲分别计入。BiLLM v2 §3.3、表 6 已明确区分参数位宽与标记/尺度存储，故“1.08 bit 参数”尤其不能直接代替完整模型占用。

> 来源维护（2026-09-21）：上列固定 commit 的外部代码引用对应已移除的本地快照；保留原版本身份，本轮未重新审查相关代码结论。

## 来源身份

下表用于在没有本地资料库时辨识来源；具体论述的章节、公式、图表或代码位置见正文。

| 来源 | 版本或快照 | 说明 |
| --- | --- | --- |
| [LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models](https://arxiv.org/abs/2310.08659v4) | `arXiv:2310.08659v4` | 冻结主体训练、单矩阵初始化计时与存储口径 |
| [QERA: an Analytical Framework for Quantization Error Reconstruction](https://arxiv.org/abs/2410.06040v2) | `arXiv:2410.06040v2` | 离线分解与在线低秩运算的边界 |
| [BiLLM: Pushing the Limit of Post-Training Quantization for LLMs](https://arxiv.org/abs/2402.04291v2) | `arXiv:2402.04291v2` | — |
| [LUQ: Layerwise Ultra-Low Bit Quantization for Multimodal Large Language Models](https://arxiv.org/abs/2509.23729v3) | `arXiv:2509.23729v3` | — |
| [Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients](https://arxiv.org/abs/2603.17809v1) | `arXiv:2603.17809v1` | — |
| [Breaking Modality Heterogeneity in Low-Bit Quantization for Large Vision-Language Models](https://arxiv.org/abs/2605.19929v1) | `arXiv:2605.19929v1` | — |
| [MASQuant: Modality-Aware Smoothing Quantization for Multimodal Large Language Models](https://arxiv.org/abs/2603.04800v1) | `arXiv:2603.04800v1` | — |
| [SpinQuant: LLM quantization with learned rotations](https://arxiv.org/abs/2405.16406v4) | `arXiv:2405.16406v4` | — |
| [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456v2) | `arXiv:2404.00456v2` | — |
| [FlatQuant: Flatness Matters for LLM Quantization](https://arxiv.org/abs/2410.09426v4) | `arXiv:2410.09426v4` | — |
| [OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models](https://arxiv.org/abs/2308.13137v3) | `arXiv:2308.13137v3` | — |
| [OpenGVLab/OmniQuant](https://github.com/OpenGVLab/OmniQuant/blob/feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14/models/int_llama_layer.py) | `feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14` | 本地快照已移除 |
| [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314v1) | `arXiv:2305.14314v1` | — |
| [MQuant: Unleashing the Inference Potential of Multimodal Large Language Models via Full Static Quantization](https://arxiv.org/abs/2502.00425v2) | `arXiv:2502.00425v2` | — |
| [VLMQ: Token Saliency-Driven Post-Training Quantization for Vision-language Models](https://arxiv.org/abs/2508.03351v2) | `arXiv:2508.03351v2` | — |
| [MBQ: Modality-Balanced Quantization for Large Vision-Language Models](https://arxiv.org/abs/2412.19509v2) | `arXiv:2412.19509v2` | — |
| [A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295v1) | `arXiv:2106.08295v1` | — |
| [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877v1) | `arXiv:1712.05877v1` | — |
| [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438v7) | `arXiv:2211.10438v7` | — |
| [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323v2) | `arXiv:2210.17323v2` | — |
| [mit-han-lab/smoothquant.git](https://github.com/mit-han-lab/smoothquant.git/tree/c61476d728e42ae0d8a35e7e78494edcac3237b5) | `c61476d728e42ae0d8a35e7e78494edcac3237b5` | — |
| [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq/tree/2d65066eeb06a5c9ff5184d8cebdf33662c67faf) | `2d65066eeb06a5c9ff5184d8cebdf33662c67faf` | — |
| [Guangxuan-Xiao/torch-int](https://github.com/Guangxuan-Xiao/torch-int/tree/65266db1eadba5ca78941b789803929e6e6c6856) | `65266db1eadba5ca78941b789803929e6e6c6856` | — |

## 教学计算材料

保留已有教学计算脚本及当时结果，供核对推导与反例；这些材料不代表模型复现或性能实验。

- [validate_foundations.py](../assets/quantized-matmul-scaling-execution/checks/validate_foundations.py)
- [validation.json](../assets/quantized-matmul-scaling-execution/checks/validation.json)
