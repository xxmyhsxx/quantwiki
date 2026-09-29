---
title: 码本量化：标量、向量、加性表示与位宽预算
type: concept
tags:
  - vector-quantization
  - weight-quantization
  - data-format
  - memory
sources:
  - raw/papers/2026-09-21/turboquant/paper.pdf
  - raw/papers/2026-09-21/nova-kv/paper.pdf
  - raw/papers/2026-09-21/quip-sharp/source.eprint
  - raw/papers/2026-09-22/qtip/paper.pdf
  - raw/papers/2026-09-23/sphquant/paper.pdf
  - raw/papers/2026-09-21/aqlm/source.eprint
  - raw/papers/2026-09-21/squeezellm/source.eprint
  - raw/papers/2026-09-21/spqr/source.eprint
  - raw/repositories/2026-09-21/squeezellm/source/squeezellm/quant.py
updated: 2026-09-29
---

# 码本量化：标量、向量、加性表示与位宽预算

码本定义“哪些值可以被表示”，索引定义“这一组权重选哪个值”。低比特并不要求可表示值等间距，也不要求每个标量独立编码。理解 QuIP#、AQLM 和 SqueezeLLM，需要同时说明重构空间、选择目标、码本存储和推理解码。

## 1. 从均匀标量走到向量码本

[均匀量化](../fundamentals/quantization/uniform-quantization-and-groups.md) 的码字可以通过 scale 与 zero-point 生成，不必逐项存表。非均匀标量码本保存 $K$ 个数，$\widehat w=C[z]$；向量码本则保存 $K$ 个 $g$ 维向量，$\widehat v=C[z]\in\mathbb R^g$。若 $K=2^B$，主体码率为 $B/g$ bit/weight。

| 表示 | 一次索引表示什么 | 代表实例 |
| --- | --- | --- |
| 均匀标量 | 一个等间距数值 | GPTQ 等方法的常见网格 |
| 非均匀标量 | 一个可学习代表值 | SqueezeLLM 每输出行 LUT |
| 固定结构向量 | 一组有相关结构的数值 | QuIP# 的 8 维 E8P |
| 可学习加性向量 | 多个码本向量的和 | AQLM |

教学例子：预算为 1 bit/weight、两个数一组。标量方案各选两个值，其四种组合构成矩形；向量方案可把四个码字放在任意四个二维位置，更好覆盖斜向分布。这是更多表示自由度，不是任意任务上的必然优势；向量码本还要付出存储和搜索代价。

## 2. 选最近码字时，“距离”是什么

普通向量量化选择 $z=\arg\min_k\|v-C[k]\|_2^2$。若目标是层输出重构，输入统计会改变距离。对行误差 $e$，目标为 $eGe^\mathsf T$，$G=XX^\mathsf T$；$G$ 的非对角项使不同坐标误差能够抵消。

教学反例：$G=\begin{bmatrix}1&0.9\\0.9&1\end{bmatrix}$，两个候选误差为 $e_a=(1,0)$、$e_b=(1,-1)$。欧氏平方误差分别为 1 和 2，而输出代理误差分别为 1 和 0.2。选择欧氏最近候选与选择最小输出误差候选可能相反。

[QuIP#](../methods/quip-sharp.md) 用块间二阶反馈后再查固定向量码本；[AQLM](../methods/aqlm.md) 则在码本与索引优化中显式使用整个输入 Gram。它们都使用向量表示，却没有执行完全相同的距离搜索。[SqueezeLLM](../methods/squeezellm.md) 的标量加权聚类又采用任务梯度构造的对角敏感性，不能与上述输入 Gram 混同。

## 3. 残差量化与加性量化的区别

加性表示为 $\widehat v=\sum_{k=1}^M C_k[z_k]$。残差向量量化从 $r_0=v$ 出发，依次选择 $C_k[z_k]$ 并令 $r_k=r_{k-1}-C_k[z_k]$；前面的决定通常不再回改。它实现简单，但早期选择未必适合后续可用码字。

AQLM 使用残差 K-means 初始化，随后交替更新索引和码本。原文 §3 的目标展开含码本间、输入组间交叉项，因此最终过程超出了单次贪心残差编码。QuIP# 的 3/4 bit 实验则用带尺度的固定格码本顺序量化残差。两者都能写成向量和，优化算法仍然不同。

一个码本有 $2^{MB}$ 项时可以直接表达全部组合；使用 $M$ 个各 $2^B$ 项码本仅需线性数量的存储，但将表示限制为码字之和。它在表示自由度、码本内存和解码加法之间取舍，不是无代价保留任意大码本的全部能力。（AQLM §3；QuIP#“Scaling E8 to Higher Bitrates”。）

## 4. 为什么 E8P 能用很小的底表

格是由离散规则生成的点集。QuIP# 的 E8P 从 $E_8$ 的平移与符号对称性构造 65,536 个 8 维码字；只存 256 个绝对值模式并紧凑打包，用符号、偶校验和平移位恢复其余结构。其底表为 1 KiB，跨层共享；不是把 65,536 个任意向量压进 1 KiB。

[QuIP# 方法页](../methods/quip-sharp.md) 展开 8+7+1 bit 编码。这里的一般认识是：结构约束同时限制可学习自由度并降低解码成本。论文中“格的球堆积密度高”有助于解释设计，但本身不是任意有限分布、有限码本和神经网络任务损失上的最优证明。

## 5. 三类格式怎样计账

下表按单个 $m\times n$ 矩阵、整组无 padding、理想紧凑存储计账；浮点表或尺度按注明的 16 bit 计算。真实文件还需加未量化参数、对齐与容器，运行峰值另加缓存和工作区。

| 表示 | 平均 bit/weight | 含义 |
| --- | --- | --- |
| AQLM，组宽 $g$，$M$ 个 $2^B$ 项 FP16 码本 | $MB/g+16gM2^B/(mn)+16/n$ | 索引 + 层码本 + 行尺度 |
| SqueezeLLM，逐输出行 $2^b$ 项 FP16 表 | $b+16\cdot2^b/n$ | 主体索引 + 每行 LUT，不含稀疏 |
| SpQR，主体 $b_w$、一级统计 $b_q$、两级组宽 $\beta_1,\beta_2$ | $b_w+2b_q/\beta_1+64/(\beta_1\beta_2)$ | 主体 + 量化 scale/zero + 四个 FP16 二级参数，不含稀疏 |

AQLM 的 $1\times16$ 与 $2\times8$ 在 $g=8$ 时主体都为 2 bit/weight，但 FP16 码本分别为 1 MiB 与 8 KiB。码本开销对小矩阵占比更高，不能只引用大模型平均值。SpQR 主体码保留所有位置时，例外另外保存值和索引，不能用简单的 $(1-p)b+p\cdot16$ 代替整个格式。

这些公式分别依据 AQLM“Estimating model size”、SqueezeLLM 的逐行表设计与 SpQR 表示章节推导。部署核算必须换入真实 dtype；例如 [SqueezeLLM 的固定代码片段](../methods/squeezellm.md) 使用 FP32 表与稀疏值，不能原样套入 FP16 理想公式。[GGUF](../implementation/gguf-block-quantization-formats.md) 展示另一个按实际块字节核算的例子，具体格式不能与论文码本互换。

## 6. 从编码走向计算

压缩权重可在使用前分块解码，也可借助输入相关查表。对 AQLM 的一个输入组 $x_j$，定义 $T_{j,k,z}=C_k[:,z]^\mathsf Tx_j$，则

$$y_i=s_i\sum_{j,k}T_{j,k,z_{ijk}}.$$

这是实数算术中的代数改写；实际浮点求和顺序会改变末位误差。临时表需要 $({n}/{g})M2^B$ 项，码本太大时构表和访存会抵消收益。GPU 还可能选择查表恢复向量后乘加，具体路径由实现决定。

小 batch decode 的低权重复用使读取节省更有价值；batch 增大时，计算与解码组织的重要性上升。判断方法时应沿“编码大小 → 解码方式 → 访存与算术 → 质量和整段推理测量”走通，不能把压缩比直接当作速度比。

## 7. 状态路径可以隐式表示长向量

[QTIP](../methods/qtip.md) 将码字限制为 trellis 图上的合法路径，避免显式保存任意高维码本。$(L,k,V)$ 结构有 $2^L$ 个状态、每步输出 V 个数；长度 T 的序列主体为 kT 位，开放路径另需 $L-kV$ 初始位。tail-biting 用首尾重叠省去这部分，但要求闭环搜索；论文的两遍 Viterbi 是近似闭环选择。

因此“高维”应分清整条路径 T、单节点输出 V 与二阶反馈输入块宽。路径结构牺牲任意码字自由度，换取可加距离下动态规划与局部位窗口并行解码。有效码率还要计入 LUT、变换向量、对齐与未量化参数，不能只看 k。

## 8. 球面方向与幅值是另一种结构分配

[SPHQuant](../methods/sphquant.md) 将8D向量拆为8个符号位、正单位方向表索引和半径。固定方向 c 后，最佳幅值是投影 $|w|^\top c$，不是原范数；误差分为方向不可达部分与半径量化部分。增加半径位主要改善后者，所以码本方向覆盖仍然重要。

其主体位宽为 $(7B+V)/8$；B2/V4实际2.25位，再加组参数和表。符号分离使B2方向表仅1KiB，但B3按同规则已达128KiB。量化器的名义位宽、完整存储与GPU常驻内存是三个不同口径，不能以相同“W2”标签代替预算核对。

## 9. 在线 KV 码本：分布、度量与固定宽度分组

[TurboQuant](../methods/turboquant.md) 对 Haar 旋转后的单位向量坐标使用由理论分布生成的标量码本，不需用模型数据训练；[NOVA-KV](../methods/nova-kv.md) 根据校准查询二阶矩构造非正交 K 变换，再训练每组的向量码本。前者的“data-oblivious”不等于无预计算，后者的欧氏最近中心也不是忽略 attention：查询度量已通过变换进入坐标。

向量码本还涉及分组的码率利用。若坐标独立 Gaussian、高分辨率近似适用，组 $\ell$ 的失真正比于 $2^{-2b_\ell}(\prod_{i\in G_\ell}\sigma_i^2)^{1/g}$。固定宽度各组均用 $b$ bit/元素时，混合高低方差坐标以均衡组体积，可减轻某些组码点不足、另一些组预算浪费；这不是按单坐标位宽排序。NOVA 的轮流分组只是近似均衡，2-bit 实测也不能替代渐近假设。

$g=4$、每组 256 个中心意味着索引 8 bit，即主体 2 bit/元素；中心表、每 token RMS scale、保护带和共享变换另计。码本随模型固定而缓存随长度/批量增长，两种开销的摊销不同。TurboQuant 的混合位宽示例还存在 2.25 与标称 2.5 的算术差别，不能仅从表格标签推算存储。

## 10. 从失真目标推导标量量化器

设标量随机变量 X 的密度为 f，一共有 M 个重构中心 $c_i$，区间为 $[a_{i-1},a_i)$，用平方误差计费：

$$D=\sum_{i=1}^{M}\int_{a_{i-1}}^{a_i}(x-c_i)^2f(x)\,dx.$$

先固定中心。对任意输入，选择距离最近的中心使每个样本误差最小，因此有序中心的内部分界是 $a_i=(c_i+c_{i+1})/2$。再固定区间，对概率质量非零的区间，令关于 $c_i$ 的导数为零，得到

$$c_i=\frac{\int_{a_{i-1}}^{a_i}x f(x)\,dx}{\int_{a_{i-1}}^{a_i}f(x)\,dx}=\mathbb E[X\mid X\text{ 落入第 }i\text{ 区间}].$$

交替这两步就是 Lloyd–Max：分区和中心都不使目标增大，但局部驻点、空区间与初始化使“迭代收敛”不等于“找到全局最优”。经验分布把积分换成样本求和，就连接到标量 K-means。边界中点依赖欧氏平方距离；非对称或码字相关的代价需要重新求分界。

若允许区间外输入，需要明确过载区间怎样映射、是否 clipping。对称码本可减少存储或计算，但也是约束。以 $X\sim U[-1,1]$、一位两中心为例，对称解是边界 0、中心 $\pm1/2$，MSE 为 $1/12$；选端点 $\pm1$ 的 MSE 为 $1/3$。中心应代表区间内样本，而不一定覆盖原始端点。

## 11. 固定码率、熵与率失真下界

M 个可能索引的固定长度至少是 $\lceil\log_2 M\rceil$ bit。若索引分布不均匀，离散熵 $H(I)=-\sum_i p_i\log_2p_i$ 可以更低；无损熵编码利用这种不均匀性减少平均长度，但要另计码流边界、随机访问、并行解码与元数据。没有实际编码时，低熵不等于已经省下缓存字节。

率失真函数把问题进一步放宽：

$$R(D)=\inf_{p(\widehat x\mid x):\,\mathbb E d(X,\widehat X)\le D}I(X;\widehat X).$$

给定来源分布与失真函数，它最小化互信息；不是某个有限码本的搜索算法，也不是一条逐样本误差保证。一个使用 B 个固定 bit 的码，满足 $I(X;\widehat X)\le H(I)\le B$，所以 $R(D)\le B$ 是可实现失真的必要条件。共享随机性独立于 X 时，应条件于该随机性计算信息量，而非把免费共享随机数当成数据传输。

**Gaussian 平方误差的简单下界。** 对 $X\sim N(0,\sigma^2)$，用以 2 为底的微分熵，$h(X)=\tfrac12\log_2(2\pi e\sigma^2)$。固定二阶矩的 Gaussian 最大熵给出

$$I(X;\widehat X)=h(X)-h(X\mid\widehat X)\ge\tfrac12\log_2\frac{\sigma^2}{D},\qquad D=\mathbb E(X-\widehat X)^2.$$

于是 $D\ge\sigma^2 2^{-2B}$；互信息非负还需与零下界合并。这里用到残差的条件熵不超过其无条件熵，以及残差方差不超过其 MSE。这个信息论下界并不保证一个 B-bit 标量 Lloyd–Max 码本达到它；允许长块编码的极限与单向量、固定码本的问题不同。

一般连续 d 维来源的同类 Shannon 下界需要相应空间的密度与微分熵。单位球面分布在 $\mathbb R^d$ 中没有这种全维密度，不能把表面积对数直接代入；[TurboQuant 的有限维反例](../methods/turboquant.md#5-原文球面下界的适用性问题)说明忽略测度条件会得到错误下界。离散熵、连续微分熵和球面上的表面测度熵是不同对象。

## 12. 高分辨率近似为何给出不同码率规律

当标量量化区间足够小、密度在区间内变化缓慢，宽度 $\Delta$ 的一个 cell 约贡献 $f(c)\Delta^3/12$。令中心密度为 $\lambda(x)$、$\int\lambda=1$，M 个中心对应局部区间宽度约 $1/(M\lambda(x))$，便得到

$$D\approx\frac1{12M^2}\int\frac{f(x)}{\lambda(x)^2}\,dx.$$

在中心总量约束下求变分，$\lambda(x)\propto f(x)^{1/3}$，从而

$$D\approx\frac1{12M^2}\left(\int f(x)^{1/3}dx\right)^3.$$

$M=2^b$ 时出现 $2^{-2b}$。这依赖高分辨率与足够规则的密度及尾部控制；在两三个码点、重尾或 clipping 明显时，不应把近似等号直接替换成严格上界。固定率 SQ、高维 VQ 和熵约束量化的常数与最优 cell 设计也不相同。

对独立 Gaussian 坐标组成的 g 维组，NOVA-KV 使用高分辨率模型 $D_\ell\approx a_\ell 2^{-2b_\ell}$，$a_\ell=C_g v_\ell^{1/g}$，其中 $v_\ell=\prod_{i\in G_\ell}\sigma_i^2$、$C_g$ 为该高分辨率模型的维度常数，$b_\ell$ 按每坐标计费，各组大小相同。总预算 $\sum b_\ell=L\bar b$ 下，若暂允许任意实数位宽，拉格朗日条件令各组预测失真相等，得到

$$b_\ell=\bar b+\tfrac12\log_2\frac{a_\ell}{(\prod_m a_m)^{1/L}}.$$

加入 $b_\ell\ge0$ 后，代理问题的解应截断为 $b_\ell=\max(0,\tfrac12\log_2(a_\ell/\tau))$，再选择 $\tau$ 满足总预算。例：$a=(16,1)$、总预算 1，未约束解是 $(1.5,-0.5)$，非负解为 $(1,0)$。负 bit 没有实际编码意义；即使非负，这仍是连续代理，零位附近还超出高分辨率模型的可靠范围。

固定每组相同位宽时，$\sum a_\ell$ 决定预测失真。对固定坐标方差乘积，混合坐标使各组体积接近可减少 AM–GM 间隙，但一个具体离散分组未必能完全均衡。设计时应依次核对：来源分布和目标、近似适用区间、整数索引预算、元数据、编码/解码成本，最后才是任务效果。

## 来源身份

- [QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks](https://arxiv.org/abs/2402.04396v2)，arXiv:2402.04396v2；BlockLDLQ、E8P 与高位宽残差编码。
- [Extreme Compression of Large Language Models via Additive Quantization](https://arxiv.org/abs/2401.06118v4)，arXiv:2401.06118v4；§3、位宽核算与推理章节。
- [SqueezeLLM: Dense-and-Sparse Quantization](https://arxiv.org/abs/2306.07629v4)，arXiv:2306.07629v4；非均匀量化与逐行 LUT。
- [SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression](https://arxiv.org/abs/2306.03078v1)，arXiv:2306.03078v1；两级元数据与例外存储。
- [SqueezeAILab/SqueezeLLM](https://github.com/SqueezeAILab/SqueezeLLM/tree/a5fd71f353bf569feb7b55737c1c1493e78e8f31)，commit `a5fd71f353bf569feb7b55737c1c1493e78e8f31`；`quant.py` 中 LUT、稀疏值和索引的 dtype。

- [QTIP](https://arxiv.org/abs/2406.11235v4)，arXiv:2406.11235v4；§2.3–3.2，路径编码与尾咬合位预算。

- [SPHQuant](https://arxiv.org/abs/2609.24875v1)，arXiv:2609.24875v1；§3与附录§7，球面分解与实际位预算。
- [TurboQuant](https://arxiv.org/abs/2504.19874v1)，v1；§3–4，分布码本、残差分数估计与位宽边界。
- [NOVA-KV](https://arxiv.org/abs/2608.04074v1)，v1；§3–4、附录 B/G/H，查询度量、分组与实际码本成本。

- §10–12 依据 TurboQuant v1 §2–3 的 Lloyd–Max/高分辨率讨论与 NOVA-KV v1 附录 G 的 Gaussian 分组模型展开；条件均值、标量信息下界、中心密度及非负预算例子为整理者推导。不采用 TurboQuant 的球面 Shannon 下界，不把连续代理当成有限码本可达性证明。
