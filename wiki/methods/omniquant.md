---
title: OmniQuant：可学习裁剪与等价变换的块级校准
type: method
tags:
  - ptq
  - llm
  - clipping
  - equivalent-transform
  - reconstruction
sources:
  - raw/papers/2026-09-21/affinequant/paper.pdf
  - raw/papers/2026-09-21/flatquant/paper.pdf
  - raw/papers/2026-09-21/omniquant/paper.pdf
  - raw/papers/2026-09-21/smoothquant/paper.pdf
  - raw/papers/2026-09-21/roformer/paper.pdf
  - https://github.com/OpenGVLab/OmniQuant/tree/feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14
  - https://github.com/OpenGVLab/OmniQuant/blob/feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14/main.py
  - https://github.com/OpenGVLab/OmniQuant/blob/feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14/datautils.py
  - https://github.com/OpenGVLab/OmniQuant/blob/feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14/generate_act_scale_shift.py
  - https://github.com/OpenGVLab/OmniQuant/blob/feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14/quantize/omniquant.py
  - https://github.com/OpenGVLab/OmniQuant/blob/feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14/quantize/quantizer.py
  - https://github.com/OpenGVLab/OmniQuant/blob/feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14/quantize/utils.py
  - https://github.com/OpenGVLab/OmniQuant/blob/feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14/models/transformation.py
  - https://github.com/OpenGVLab/OmniQuant/blob/feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14/models/int_llama_layer.py
  - https://github.com/OpenGVLab/OmniQuant/blob/feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14/quantize/int_linear.py
  - https://github.com/OpenGVLab/OmniQuant/blob/feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14/quantize/int_matmul.py
  - https://github.com/OpenGVLab/OmniQuant/blob/feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14/scripts/llama/llama-7b/w4a4.sh
  - https://github.com/OpenGVLab/OmniQuant/blob/feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14/scripts/llama/llama-7b/w3a16.sh
updated: 2026-09-15
---

# OmniQuant：可学习裁剪与等价变换的块级校准

OmniQuant 冻结原始浮点权重，学习决定量化结果的裁剪强度和等价变换参数。它每次优化一个 Transformer block 的输出误差：等价变换调整激活与权重的量化难度，裁剪再调整变换后权重的网格，两者可以在同一目标下联合学习。

这补充了 [AWQ](awq.md) 的缩放/裁剪搜索、[SmoothQuant](smoothquant.md) 的统计缩放和 [GPTQ](gptq.md) 的逐权重补偿路线。它需要反向传播，但属于论文所定义的可学习 PTQ，不等于全模型权重微调。[PTQ、QAT 与代理梯度](../fundamentals/quantization/post-training-and-quantization-aware-training.md) 解释这种分类。

本页依据 arXiv:2308.13137v3（2024-03-18，ICLR 2024），下称“论文”；25 页含附录 A1–A8 已全文研读。官方代码固定为 `feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14`，定向核对 LLaMA 校准、变换与执行路径；未运行模型。代码与论文的设置差别单独说明。

## 1. 量化对象与优化变量

论文评估 OPT、LLaMA-1/2、Falcon 和 LLaMA-2-chat。纯权重量化为 W2/W3/W4A16，默认每输出通道一组，也提供沿输入维划分的 group-wise 方案；权重与激活联合量化主要为 W4A4/W6A6，权重 per-channel、激活 per-token。Q/K/V 也参与模拟量化，Softmax 输出默认保持 16-bit。（§4.1、表 A7。）

位宽标签不表示整个模型的所有张量都是该精度：默认代码替换 Transformer block 内的线性层及注意力乘法输入，不将 embedding、最终 lm_head、残差和归一化全变成低比特整数算子。更不意味着已经实现低比特 KV 缓存存储，见第 6 节。

| 变量 | 学习内容 | 不应混淆的对象 |
|---|---|---|
| LWC：上下界强度 $\gamma,\beta$ | 改变权重范围和网格 | 不是每个权重各自的舍入开关，也不是被裁掉权重的比例 |
| LET：通道缩放 $s$、平移 $\delta$ | 改变线性层输入与权重的表示 | 不是推理时的量化步长 |
| LET：注意力缩放 $s_a$ | 调整 Q/K 对应坐标的范围 | 不是 attention 分数或 token 重要性权重 |

原权重不作为优化变量，但量化过程中使用的变换后权重会随 LET 改变；最终导出的权重也与原权重不同。“冻结原权重”不能解释成整个过程权重数值不变。

## 2. 块级目标怎样带入前层误差

这里要同时区分三个尺度：量化器在一个通道或 group 内建立网格；LET 在相连算子之间改变表示；优化器在整个 Transformer block 的输出上评价这些改变。前两项都是局部操作，评价它们的目标却包含块内后续计算，所以不能各自挑选最小张量误差后就认为已经完成联合优化。

令 $F_i$ 为第 $i$ 个完整 Transformer block，包含注意力、MLP、非线性与残差；$x_i^{fp}$ 为浮点前缀产生的输入，$x_i^q$ 为前面已量化块产生的输入。将论文式 1 与算法 1 合并解释为

$$
\min_{\theta_i}\;\mathbb E_{x\in\mathcal C}
\left\|F_i(W_i,x_i^{fp})-\widehat F_i(W_i,x_i^q;\theta_i)\right\|_F^2,
\qquad \theta_i=\{\gamma,\beta,s,\delta,s_a\}.
$$

$\mathcal C$ 是校准集，不需要回答标签；参考目标来自浮点模型。代码用元素平均 MSE。两条输入路径从同一个初始输入出发，每完成一个块，分别用浮点块和已量化块推进。因此当前块会尝试补偿前层已经造成的偏差，但不会反向重新优化前面的块，也不包含所有后续块的任务损失。

这与 [层输出重构](../theory/layer-reconstruction-second-order-compensation.md) 中固定输入的单线性层二次问题不同，不能直接套用 $2XX^\mathsf T$ 作为整个 Transformer block 的精确 Hessian。与 [Q-VLM](q-vlm.md) 的区别是，这里的边界按已有 Transformer block 固定；Q-VLM 尝试选择哪些连续层放在一起校准。GPTQ 的算法列块又是计算组织上的分块。

局部优化降低了同时保留梯度的范围，但仍需当前块的前向、反向、优化器状态和校准激活；“只学习少量参数”不等于校准成本可以忽略。

## 3. 可学习裁剪：学习相对强度

对一个权重量化组，记当前待量化权重为 $\widetilde W$，极值为 $a=\min\widetilde W$、$b=\max\widetilde W$。通常考虑 $a<0<b$ 的非退化范围。令 $K=2^N-1$，则论文式 2 为

$$
l=\beta a,\quad u=\gamma b,\quad
h=\frac{u-l}{K},\quad z=-\operatorname{round}(l/h),
$$
$$
q=\operatorname{clip}\big(\operatorname{round}(\widetilde W/h)+z,0,K\big),
\qquad \widehat W=h(q-z).
$$

$q$ 是整数码值，$\widehat W$ 才是重构前向使用的实数权重。论文将 $W_q$ 用作码值的写法不能直接放进浮点矩阵乘而省略 $h,z$。量化组、zero-point 和实际端点见 [量化网格与分组](../fundamentals/quantization/uniform-quantization-and-groups.md)。

取 $\gamma=\sigma(g)$、$\beta=\sigma(b_0)$，优化无约束潜变量 $g,b_0$，将强度限制在 $(0,1)$；两者趋近 1 时接近 min–max。收缩范围让主体网格更密，同时牺牲尾部，是否值得由块输出误差决定。零点取整后实际可表示端点为 $-hz$ 和 $h(K-z)$，不必恰好等于 $l,u$。退化范围需实现保护，当前代码将步长下限设为 $10^{-5}$。

**为什么用相对强度。**LET 每次更新都会改变 $\widetilde W$ 的极值；LWC 相对这些当前极值重新计算端点，使范围能够随权重变换调整。论文表 A14 和图 A5 在同一校准框架中替换裁剪组件，支持这一参数化在所测设置中的价值；不证明学习绝对阈值或步长的方法普遍无效。

例如，某组范围从 $[-10,8]$ 变为 $[-20,12]$，同一对强度 $(\beta,\gamma)=(0.8,0.75)$ 会使目标端点从 $[-8,6]$ 自动变成 $[-16,9]$。若直接学习绝对端点而暂时保持其值，端点仍为 $[-8,6]$，对新分布的裁剪程度已不同。相对参数化提供了随当前极值调整的路径，但也限制搜索范围：通常不能把端点扩到极值以外，sigmoid 靠近端点时梯度还会变小。因此它是优化设计的取舍，不是扩大到所有可能网格的搜索。

### 一个裁剪与输出误差的数值例子

下面是教学构造，使用式 2 的 unsigned 3-bit 仿射量化，舍入中点取偶数。设一行权重 $w=(-7,1,1,7)$，输入 $x=(0.01,1,1,0.01)^\mathsf T$，浮点输出为 2。

| 范围设置 | 步长与零点 | 反量化权重 $\widehat w$ | 权重平方误差 | 输出 $\widehat wx$ |
|---|---|---|---:|---:|
| $\beta=\gamma=1$ | $h=2,z=4$ | $(-8,0,0,6)$ | 4 | -0.02 |
| $\beta=\gamma=0.5$ | $h=1,z=4$ | $(-4,1,1,3)$ | 25 | 1.99 |

裁剪牺牲两端的大权重，却恢复了输入更强的中间两个通道。权重误差增加，当前输出误差反而明显下降；这解释为什么 LWC 由输出目标学习，而非只拟合权重直方图。换成只激活最后通道的 $x=(0,0,0,1)^\mathsf T$，两种输出分别为 6、3，裁剪反而更差。哪个范围合适取决于校准输入及后续计算，不能由“离群值大”单独决定。

### 一次反向更新经过哪些节点

代码 `UniformAffineQuantizer` 为每输出通道、或每行内的每个 group 建立一对强度，潜变量初值为 4，即 sigmoid 后约 0.982。若权重矩阵有 $C_{out}$ 行、每行 $C_{in}$ 个元素、组长 $G$，裁剪变量数量约为 $2C_{out}\lceil C_{in}/G\rceil$，不是全模型只有两个参数。图 A1 的强度分布反映范围收缩，不能据此说“裁掉一半权重”。

**梯度怎样通过量化。**`quantize/quantizer.py:round_ste` 前向做真实舍入，反向把舍入的局部导数近似为 1；裁剪仍影响梯度。固定当前权重极值时，$\partial h/\partial g=b\gamma(1-\gamma)/K$、$\partial h/\partial b_0=-a\beta(1-\beta)/K$，于是块损失可经步长回传到强度。LET 的梯度还经过变换权重和范围计算。代码零点使用普通 `.round()`，没有对零点套同一个 STE。代理规则见 [代理梯度](../fundamentals/quantization/post-training-and-quantization-aware-training.md#3-为什么需要-ste)；它不是离散目标真实可导或全局最优的保证。

进一步固定反向中的整数零点 $z$，令 $v=w/h$、$r=\operatorname{round}(v)$。不在裁剪边界时，该代码计算图给出的代理导数为

$$
\frac{\widetilde\partial\widehat w}{\partial h}
=\begin{cases}r-v,&0<r+z<K,\\-z,&r+z<0,\\K-z,&r+z>K.\end{cases}
$$

内区的 $r-v$ 来自外侧乘步长与内侧除步长两条路径；饱和区里整数码固定，只剩表示端点随步长移动。边界点按实现的 clamp 规则处理。对一组权重，裁剪潜变量的梯度可以写成

$$
\frac{\widetilde\partial L}{\partial g}
=\sum_j\frac{\partial L}{\partial\widehat w_j}
\frac{\widetilde\partial\widehat w_j}{\partial h}
\frac{b\gamma(1-\gamma)}{K}.
$$

$\partial L/\partial\widehat w_j$ 来自完整块的反向传播，已经包含输入、注意力、非线性与残差对输出的影响。LET 的缩放参数还有经过 $\widetilde X$、$\widetilde W$ 和极值计算的路径，不能仅把上述裁剪梯度换个变量名当作 LET 的完整梯度。这是固定代码代理链式法则的教学展开，非论文新增的收敛定理。

## 4. 可学习等价变换及融合条件

### 线性层：缩放、平移与偏置一起改变

缩放主要调整通道间的幅度差异，平移还可以调整通道中心。举例，两个 token 的通道值为 $X=[(9,1);(11,-1)]$，逐通道平移 $\delta=(10,0)$ 后成为 $[(-1,1);(1,-1)]$。两个 token 的范围宽度由 8、12 都变为 2，因此共享一个 token 网格时可减少跨通道中心差异的影响。这是动机示例；同一标量平移所有通道并不会缩小范围，任意逐通道平移也不保证降低最终量化误差。

采用论文的行 token 约定，$X\in\mathbb R^{T\times C_{in}}$、$W\in\mathbb R^{C_{in}\times C_{out}}$、$b\in\mathbb R^{1\times C_{out}}$，$D=\operatorname{diag}(s)$ 且 $s_j\ne0$：

$$
XW+b=\underbrace{(X-\mathbf1\delta)D^{-1}}_{\widetilde X}
\underbrace{DW}_{\widetilde W}
+\mathbf1\underbrace{(b+\delta W)}_{\widetilde b}.
$$

展开后 $-\mathbf1\delta W$ 与偏置补偿相消，量化前保持线性函数；量化时使用 $Q_a(\widetilde X)Q_w(\widetilde W)+\mathbf1\widetilde b$，其中 $Q_a$ 为 min–max、$Q_w$ 含 LWC。这使激活平滑与权重裁剪在同一块目标中相互影响，而非两个互不相关的预处理。（式 3–4。）

缩放初始化复用 SmoothQuant v7 式 4 的 $s_j=a_j^\alpha/w_j^{1-\alpha}$，$a_j,w_j$ 分别为该输入通道的最大绝对激活、权重；OmniQuant 随后自由学习各通道缩放，不再被一个共享 $\alpha$ 的曲线约束。平移初始化在论文中归于 Outlier Suppression+；当前代码实际对每段输入的通道区间中点做 EMA。具体统计定义不能混写成激活均值。（§4.1；`generate_act_scale_shift.py:get_act_scales/get_act_shifts`。）

图 3 和表 A5 使用四处 LET：归一化到 Q/K/V 投影、V 投影到输出投影、Q/K 相互缩放、第二个归一化到 FFN 第一层。未对 FFN 非线性之后到第二层的接口使用 LET；作者认为该处稀疏激活导致梯度不稳定，不应把这当成所有 FFN 的一般结论。LLaMA 的 gated MLP 中，共享归一化输出的 up/gate 两个投影需一起处理。

以下以 LLaMA block 的主路径对应这些位置；$H$ 的形状为 $B\times T\times d$，head 拆分后 Q 为 $B\times n_h\times T\times d_h$。实际 GQA 的 K/V head 数可能不同，不能直接把 MHA 的参数共享照搬过去。

```text
H ── RMSNorm1 ── Q/K/V 投影 ── Q/K 的 RoPE ── 注意力 ── O 投影 ── +H
       [s_qkv]        [Q/K 成对缩放]           [V/O 变换]           │
                                                               H'
H' ─ RMSNorm2 ── gate/up 投影 ── SiLU(gate) × up ── down 投影 ── +H'
       [s_ffn：gate/up 共用]                    [不放 LET]
```

| 位置 | 参数怎样作用与抵消 | 不能漏掉的条件 |
|---|---|---|
| Norm1 → Q/K/V | Norm 输出各通道除以 $s_{qkv}$，三组投影输入列同时乘回 | 三个消费者共用；使用平移时同步补偿 bias |
| Norm2 → gate/up | Norm 输出缩放，gate/up 的输入列一起乘回 | 门控两条支路都依赖相同输入 |
| V → O | V 的 head 内坐标逆变换，O 的对应输入列变换 | head reshape、偏置及注意力加权和都必须对应 |
| Q/K | 同一 head 点积的对应坐标，一侧除、一侧乘 | 注意 RoPE 的位置和交换条件，见下文 |

量化发生在变换后：线性层输入用激活量化器，临时权重用 LWC；Q/K/V 在注意力乘法入口另作激活量化。FFN 的 down 投影仍量化其输入和权重，“不放 LET”不等于不量化。图与表用于解释计算链，不表示所有历史模型实现拥有相同 Norm、bias 或 head 结构。

### 注意力：恒等式成立的位置很重要

对已经进入注意力点积的 $Q,K\in\mathbb R^{T\times d_h}$，取非零对角 $D_a$，有

$$
(QD_a^{-1})(KD_a)^\mathsf T=QK^\mathsf T.
$$

保留原有 $1/\sqrt{d_h}$ 和 mask 后，未量化 Softmax 概率也相同。这是式 5 的基础，量化后两侧误差仍可能改变注意力。

V 的对应变换可通过输出投影抵消：当每行注意力概率之和为 1 时，$P[(V-\mathbf1\delta_v)D_v^{-1}]$ 等于 $(PV-\mathbf1\delta_v)D_v^{-1}$，后继线性层配合改权重、偏置即可。若概率被量化而行和不再为 1，平移抵消还会引入额外误差；默认 Softmax 不量化不能自动覆盖所有改动后的设置。

**不能省略 RoPE。**RoFormer v5 §3.1–3.2 用位置相关的二维旋转对 Q/K 成对编码。若把点积处的任意逐通道缩放直接移到 RoPE 前，一般需要缩放与相对旋转交换才能保持等价。每个旋转对使用相同缩放是一个充分条件。推导与反例见 [平移与位置编码的融合边界](../theory/diagonal-scaling-equivalent-transform.md#7-平移与注意力位置编码的融合边界)。

当前 `models/transformation.py:smooth_q_k_temporary/inplace` 将自由缩放写入 Q/K 投影，`QuantLlamaAttention.forward` 在投影后执行 RoPE，所读路径没有施加旋转对共享约束。因此不能仅用式 5 证明该路径对任意学得参数都严格保持未量化函数。这是代数与代码定位得到的限制；没有测量其在具体 checkpoint 上造成的误差，也不由此否定论文所有实验。

LET 能否不增加独立运行时操作，取决于完整计算图：归一化的仿射参数、所有消费者、偏置、位置编码和后端均需兼容。数学上的等价变换参数可消去，不表示量化 scale/zero-point、可能新增的偏置或部署元数据也消失。

## 5. 训练流程、数据与参数设置

算法 1 的可复述流程为：

1. 固定模型与量化粒度，准备校准 token 和初始化统计，冻结原模型参数。
2. 为当前 block 计算浮点前缀的参考输出，保留量化前缀产生的输入。
3. 初始化并按配置启用 LWC/LET。每次前向从当前块原权重构造临时变换，再模拟量化，以块输出 MSE 联合更新参数。
4. 完成该块优化后固化变换和量化权重，推进量化输入到下一块。浮点参考路径独立推进。
5. 根据目标后端另行导出、打包并评测。参数训练完成不等于真实低比特执行完成。

把第 3 步展开，一次迭代的状态为：当前块原权重 $W_i$、浮点参考输出和量化前缀输入固定；可训练状态是当前块的 LWC/LET 参数。根据当前 LET 从 $W_i$ 构造 $\widetilde W_i$，重算其极值和 LWC 网格；模拟量化得到输出，计算 MSE，再经代理梯度更新两组参数。下一次必须重新从 $W_i$ 构造临时权重，不能把同一缩放反复乘到上一次变换结果上，否则实际得到的是累积变换。

```text
浮点输入 = 量化输入 = 首块输入
对每个 block：
    参考输出 = 原浮点 block(浮点输入)          # 无梯度，缓存
    初始化当前 block 的 LWC/LET
    重复校准轮数：
        临时 block = 从原权重构造 LET，再执行 LWC 假量化
        损失 = MSE(临时 block(量化输入), 参考输出)
        反向并只更新当前 block 的 LWC/LET
    固化当前 block；量化输入 = 固化 block(量化输入)
    浮点输入 = 参考输出
```

这是论文算法 1 的解释性伪代码，不含官方可选 `aug_loss`。例如校准第二块时，参考是 $F_2(F_1(x))$，量化输出是 $\widehat F_2(\widehat F_1(x))$；若两侧都改用 $\widehat F_1(x)$，就换成了另一个目标，失去对原浮点路径的直接对齐。联合优化表示当前块的 LWC 与 LET 在同一次损失反向中一起更新，不表示整个网络的所有块同时训练。

**`aug_loss` 增加的到底是什么。**固定当前块，记学生输出 $z=\widehat F_l(x_l^q)$，两个不参与梯度的参考为 $a=F_l(x_l^{fp})$、$b=F_l(x_l^q)$。代码 `quantize/omniquant.py` 直接相加两个采用相同归约的 MSE；把 $\|\cdot\|_{\mathrm{MSE}}^2$ 定义为元素平方的平均，有

$$L_{aug}=\|z-a\|_{\mathrm{MSE}}^2+\|z-b\|_{\mathrm{MSE}}^2
=2\left\|z-\frac{a+b}{2}\right\|_{\mathrm{MSE}}^2
+\frac12\|a-b\|_{\mathrm{MSE}}^2.
$$

后一项对当前块的可训练参数是常量。因此在输出目标层面，它要求学生兼顾“回到原浮点路径”和“在已经扰动的输入上保持原块行为”，等价于对两个参考的均值做两倍 MSE；不是新增任务标签或两次不同学生前向。受参数表达力限制时，学生未必能达到该均值。系数 2 也会改变梯度尺度，不据此断言具体优化器轨迹相同。[AffineQuant](affinequant.md) 的所读实现沿用此项，[路线比较](../research/omniquant-affinequant-flatquant-comparison.md) 将其与 FlatQuant 的单参考目标区分。

论文 §4.1 使用 WikiText2 中 128 个随机的 2048-token 片段、batch 1、AdamW、零 weight decay；通常 20 epochs，W2A16 为成本折中取 40。论文的 LWC/LET 学习率分别为 $5\times10^{-3}$、$10^{-2}$。表 A12 的时间属于单张 A100-80G：LLaMA-7B 的纯权重/权重激活校准分别为 1.1/1.6 h，65B 为 8.9/14.4 h，20 epochs；不能与摘要的 A100-40G 概述无条件混用。

分模型策略也重要：论文对 LLaMA 纯权重量化只启用 LWC；OPT 纯权重量化仍用 LWC+LET；权重激活量化启用两者。具体超参数与数据应随实验记录，不能把上述设置当作通用默认。

当前官方代码的 WikiText2 loader 从 train 抽校准片段，test 用于困惑度评测；不是因为同名数据集就认定数据泄漏。已缓存的 token 和初始化统计也应绑定模型、分词器、seed、序列长度及来源；代码缓存名未包含所有这些条件，复现不能只依赖缓存文件存在。（`datautils.py:get_wikitext2`、`main.py`。）

## 6. 论文与固定实现怎样对应

| 环节 | 核对结果 | 使用时的边界 |
|---|---|---|
| 原权重与学习变量 | `main.py` 冻结模型；`quantize/omniquant.py` 的优化器仅接收 smooth 与 bound 参数；每次使用临时变换权重 | 支持冻结基座、学习量化参数的解释；并非全权重 QAT |
| 块目标 | `fp_inps` 与 `quant_inps` 分开推进；W4A4 脚本额外启用 `aug_loss`，增加浮点块在量化输入上的重构项 | 论文算法 1 的目标与该脚本目标不完全相同；README 示例又未启用这一项 |
| 学习率与平移 | main 的默认 LWC/LET 学习率为 $10^{-2}/5\times10^{-3}$，与正文对应关系不同；LLaMA 或 A16 时不学习 shift | 不能用代码默认值填作论文设置；论文平移消融也不能直接由当前 LLaMA 默认配置复现 |
| Q/K 融合 | 缩放写入投影后才进行 RoPE，无旋转对绑定 | 需检查第 4 节的交换条件，不能泛称所有 LET 都已严格等价 |
| 量化前向 | `UniformAffineQuantizer` 返回反量化浮点值，`QuantLinear` 调 `F.linear`，`QuantMatMul` 调浮点矩阵乘；激活沿最后一维动态估计范围 | 学得变换是离线参数，激活网格仍可随输入变化；这是精度模拟路径 |
| KV cache | LLaMA attention 先构造 `past_key_value`，随后才对用于乘法的 K/V 做 fake quant | 返回缓存仍是浮点张量；低比特计算模拟不等于压缩缓存存储 |
| 真实打包 | `real_quant` 只接受 W2/W3/W4 且 A16，通过 AutoGPTQ 模块 pack；README 明确提示该路径可能减内存却变慢 | 不能把它与论文 MLC-LLM 的加速路径混为一谈；本轮未读取外部 kernel 或执行导出 |

上表范围是所读快照与 LLaMA 主路径，没有完成 OPT/Falcon/Mixtral 所有分支的验证。实际部署区别见 [量化矩阵乘法的执行路径](../implementation/quantized-matmul-scaling-execution.md)。

## 7. 哪些实验值得保留

**组件是互补的，但收益依赖配置。**表 A2 在 LLaMA-7B/W4A4 上报告 WikiText2 与 C4 平均 PPL，以及六项 zero-shot 平均准确率：

| 配置 | 平均 PPL ↓ | 平均准确率 ↑ |
|---|---:|---:|
| LET | 16.97 | 48.83 |
| LET + 网格搜索裁剪 | 15.82 | 49.59 |
| SmoothQuant + LWC | 15.80 | 50.15 |
| LET + LWC | 12.87 | 52.65 |

它支持在共同目标下同时调整变换与权重范围的价值。表 A8 中轮流训练并加倍 epochs 得到 PPL 12.80、准确率 52.50，与联合训练各有优劣；应理解为联合训练在该预算下的效率取舍，不能声称所有指标绝对最优。

表 A3 的 LLaMA-13B/W3A16，LWC+LET 的 WikiText2 PPL 为 5.65，去掉 LET 为 5.68，去掉 LWC 为 7.65。W4A4 则从 10.87 在去掉 LET 后升到约 $5.4\times10^3$。这解释了为什么作者在 LLaMA weight-only 路线省去 LET，也说明不能把激活量化困难与纯权重问题混为一谈。

**位置、初始化和误差目标有实际作用。**表 A5 去掉归一化到 Q/K/V 的 LET，平均 PPL 从 12.87 升至 19.87；其他位置的影响较小。表 A6 用全 1 缩放初始化为 13.64，说明学习也受初始化影响。表 A13 的 W3A16g128 权重平均 L1 距离仅从 0.0042 降至 0.0040，而最后 block 输出 L1 从 1.37 降至 0.79，提示只看权重误差不足以评价方法。上述均是作者在指定设置中的观察。

**低位宽仍会损失质量，且不是所有设置领先。**表 2 的 LLaMA-7B 六任务均分，FP16 为 64.09，W6A6 为 63.17，W4A4 为 52.65。表 A20 的 OPT-66B/W2A16g128，OmniQuant WikiText2 PPL 为 30.84，AWQ 为 14.54；因此不能照搬“所有模型配置均优于基线”的概述。表 A25 中原 RPTQ 与统一量化条件后的 RPTQ* 也分列，比较前必须核对 LN/Softmax 位宽。表 A17 的 W4A8 与 W8A4 差异可用于识别所测模型的激活量化困难，不能直接推出层间最优位宽分配。

**校准分布稳定性是局部证据。**表 A10 用 WikiText2、C4、Pile 校准 LLaMA-7B，W4A4 在 WikiText2 的 PPL 为 11.23、12.17、12.04，在 C4 为 14.61、14.24、14.22。表 A11 增加到 256 个样本也未持续优于 128。它们支持这几种文本设置下的变化相对有限，不是多模态、长上下文、部署漂移或多随机种子的鲁棒性保证。见 [校准数据与范围选择](../theory/calibration-and-range-selection.md)。

**实际速度由后端决定，位宽越低不一定越快。**表 3 使用 MLC-LLM、A100-80G、生成 512 tokens；7B 的报告如下。WM 是权重内存，RM 是运行内存，不能互换。

| 配置 | WM | RM | token/s |
|---|---:|---:|---:|
| FP16 | 12.6 G | 14.4 G | 69.2 |
| W4A16g128 | 3.8 G | 5.7 G | 134.2 |
| W3A16g128 | 3.2 G | 5.1 G | 83.4 |
| W2A16g128 | 2.2 G | 4.1 G | 83.9 |

4-bit 在这张表中反而更快；§4.5 明确此次只部署 weight-only，W4A4/W6A6 没有对应的实际加速实验。输入长度、batch、后端 commit 与 prefill/decode 计时边界未在该表充分交代，本轮没有补猜或复测。图 A4/A6 的模型大小与位宽取舍同样只反映所测模型、指标和表示开销，不是普遍的“3-bit 最优定律”。

后端侧的量化模式、校准流程与编译步骤如何组织，见 [部署框架与后端支持](../implementation/quantized-llm-deployment-backends.md)：MLC-LLM 把校准做成独立运行阶段并把统计写回权重文件，这与本页离线学习量化参数后再导出的流程不是同一个界面。

## 8. 优势、局限与研究用途

| 与已有路线比较 | OmniQuant 增加的选择 | 需要承担的代价或限制 |
|---|---|---|
| SmoothQuant 的统计缩放 | 从共享迁移指数的缩放族出发，进一步学习通道参数和裁剪 | 需要块前向/反向，仍受初始化和校准分布影响 |
| AWQ 的候选搜索 | 连续优化更多参数，并评价完整 block 的输出 | 搜索空间更灵活不等于全局更优；训练成本更高 |
| GPTQ 的逐权重补偿 | 通过裁剪与表示变换共同调整量化结果 | 没有对每个权重提供同一种二阶补偿机制，极低位宽仍可能失败 |

这一比较定位变量、目标和求解方式，不把不同量化对象与后端的精度/速度直接排名。OmniQuant 自身的裁剪替换实验更能解释设计：表 A14 在 LLaMA-7B/W4A4 上，替换为 PACT、LSQ 的 PPL 为 18.25、15.03，LWC 为 11.26；W3A16 的差距则较小，分别为 6.95、6.63、6.47。结合图 A5 的范围轨迹，作者将其解释为相对端点更适应 LET 引起的范围变化。这里比较的是作者在相同框架中的组件改装，不是 PACT/LSQ 原方法的完整复现或普遍优劣定论。

**优势在于把可调整变量和输出目标连起来。**它在不训练全部基座权重的前提下，联合适配激活表示与权重网格，并保留均匀量化的实现形式。对低位宽下仅靠初始统计或有限搜索不够的情况，这是值得比较的路线。

**限制来自代理目标、校准成本和实际计算图。**块 MSE 不保证任务质量或分布外表现；极低位宽仍有明显损失；每块训练与激活缓存需要资源；融合必须满足位置编码、偏置和分支条件；模拟、权重打包与实际加速之间仍有独立的实现工作。

它为“哪些参数可以更新”提供了实例，但没有验证部署后漂移下的在线适应或端云通信收益。变换参数改变时，邻接权重、偏置和量化网格可能都要重新生成；已融合部署模型未必能只下发几个 scale 就完成更新。用于后续研究时，应先分清可训练变量、导出产物和需要替换的数据，再讨论更新成本。

## 9. 从通道尺度继续到通道混合

[AffineQuant](affinequant.md) 在可用位置将逐通道尺度扩展为非对角可逆矩阵，并以 gradual mask 稳定优化；这回应 LET 不能混合通道的限制，但引入求逆与部署位置的约束。[FlatQuant](flatquant.md) 进一步用 Kronecker 结构和在线变换/量化融合，在 Norm 后及 down 投影前使用可学习混合。（AffineQuant v1 §3；FlatQuant v4 §3。）

三者都学习量化前的表示，实际变换空间、裁剪对象和输入误差路径并不相同。先读本页的完整机制，再沿 [三种方法比较](../research/omniquant-affinequant-flatquant-comparison.md) 判断哪些差异有助于研究；不能用“后续论文”代替单篇理解，也不能以更大的参数空间推导普遍更好的结果。

> 来源维护（2026-09-21）：上列固定 commit 的外部代码引用对应已移除的本地快照；保留原版本身份，本轮未重新审查相关代码结论。

## 来源身份

下表用于在没有本地资料库时辨识来源；具体论述的章节、公式、图表或代码位置见正文。

| 来源 | 版本或快照 | 说明 |
| --- | --- | --- |
| [AffineQuant: Affine Transformation Quantization for Large Language Models](https://arxiv.org/abs/2403.12544v1) | `arXiv:2403.12544v1` | — |
| [FlatQuant: Flatness Matters for LLM Quantization](https://arxiv.org/abs/2410.09426v4) | `arXiv:2410.09426v4` | — |
| [OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models](https://arxiv.org/abs/2308.13137v3) | `arXiv:2308.13137v3` | — |
| [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438v7) | `arXiv:2211.10438v7` | — |
| [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864v5) | `arXiv:2104.09864v5` | — |
| [OpenGVLab/OmniQuant](https://github.com/OpenGVLab/OmniQuant/tree/feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14) | `feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14` | 本地快照已移除 |

## 教学计算材料

保留已有教学计算脚本及当时结果，供核对推导与反例；这些材料不代表模型复现或性能实验。

- [check_math.py](../assets/omniquant/checks/check_math.py)
- [math-validation.json](../assets/omniquant/checks/math-validation.json)
