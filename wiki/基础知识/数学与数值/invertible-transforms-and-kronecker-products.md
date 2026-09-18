---
title: 可逆变换、数值条件与 Kronecker 乘积
slug: invertible-transforms-and-kronecker-products
sources:
  - raw/papers/quantization/vlm/2026-05-splitq-breaking-modality-heterogeneity-low-bit/paper.pdf
  - raw/papers/quantization/vlm/2026-03-masquant-modality-aware-smoothing-quantization/paper.pdf
  - raw/papers/optimization/2020-02-cayley-stiefel-efficient-riemannian-optimization/paper.pdf
  - raw/papers/quantization/ptq/2024-03-affinequant-affine-transformation-quantization/paper.pdf
  - raw/papers/quantization/ptq/2024-10-flatquant-flatness-matters-llm-quantization/paper.pdf
  - raw/references/pytorch/2026-09-orthogonal-parametrization/orthogonal.html
updated: 2026-09-15
---

# 可逆变换、数值条件与 Kronecker 乘积

量化前可以改变坐标，使数值更适合有限网格；另一侧施加逆变换，保持浮点计算。[[affinequant|AffineQuant]] 和 [[flatquant|FlatQuant]] 都依赖这件事，但“可逆”“稳定”“表达力强”和“算得快”是四个不同问题。本页解释它们需要的矩阵基础。基础恒等式来自两篇论文的方法部分，下列证明与小例子是我们的教学展开。

## 1. 可逆只保证可以还原

采用 [[linear-layer-input-channel|线性层]] 的约定：$X\in\mathbb R^{T\times n}$，$W\in\mathbb R^{m\times n}$，$Y=XW^\mathsf T$。对可逆 $P\in\mathbb R^{n\times n}$，

$$Y=(XP)(P^{-1}W^\mathsf T).
$$

在代码的权重布局中，应保存 $W'=WP^{-\mathsf T}$。逆与逆转置的差别来自存储方向，不能靠变量名判断。

[[diagonal-scaling-equivalent-transform|对角缩放]] 只逐坐标放大或缩小；非对角矩阵还可混合坐标。[[orthogonal-rotation-and-hadamard-quantization|正交变换]] 满足 $P^{-1}=P^\mathsf T$，保持长度与内积；一般可逆矩阵没有这些性质。

把每行归一化并不能把任意可逆矩阵变成正交矩阵。例如 $\begin{pmatrix}1&0\\1&1\end{pmatrix}$ 行归一化后，两行内积为 $1/\sqrt2$，仍不正交。AffineQuant §3.1 的旋转类比因此应理解为动机，不能作为一般矩阵分解定理。

量化后令 $E_X=Q(XP)-XP$、$E_W=Q(P^{-1}W^\mathsf T)-P^{-1}W^\mathsf T$，有

$$\widehat Y-Y=E_XP^{-1}W^\mathsf T+XP E_W+E_XE_W.
$$

因此一侧更平或其张量 MSE 更小，都不能独立保证输出误差更小。逆变换也不能恢复被裁剪或舍入丢失的信息；优化只能寻找两侧误差传播更有利的表示。

### 浮点恒等式为什么还能学出不同变换

为对应 AffineQuant，暂将权重记为 $B\in\mathbb R^{n\times m}$，令 $Z=XA^{-1}$、$T=AB$。从 $AA^{-1}=I$ 微分得到

$$d(A^{-1})=-A^{-1}(dA)A^{-1},\qquad
dZ=-Z(dA)A^{-1},\quad dT=(dA)B.
$$

在没有量化、裁剪且两侧严格配对时，$d(ZT)=-Z(dA)B+Z(dA)B=0$，改变 $A$ 不改变输出。加入量化后前向变成 $Q_a(Z)Q_w(T)$，两侧不再精确抵消；舍入、裁剪和范围选择使不同坐标系有不同输出误差，这才提供优化表示的空间。（依据 AffineQuant §3.1 恒等式的教学推导。）

若反向传播给出 $G_Z=\partial L/\partial Z$、$G_T=\partial L/\partial T$，链式法则为

$$\frac{\partial L}{\partial A}
=-Z^\mathsf T G_Z A^{-\mathsf T}+G_TB^\mathsf T.
$$

第一项来自激活侧求逆，第二项来自权重侧正向变换；固定输入平移时只需把 $X$ 换成 $X-\delta$。量化实现中的 $G_Z,G_T$ 包括裁剪、范围估计及舍入代理梯度，不能假定都是恒等传递。若有效矩阵为 $A^*=A\odot G$，应先对 $A^*$ 计算上述梯度，再逐元素乘 mask $G$。只优化变换后的权重、忽略激活侧逆矩阵，并非同一个目标。

这也解释了求逆稳定性为何影响校准：$A^{-1}$ 同时进入前向和反向，接近奇异时可能放大数值扰动。这里核对的是可微矩阵代数；离散量化使用 [[post-training-and-quantization-aware-training|代理梯度]]，不把该公式当成舍入函数的精确导数。

## 2. 可逆、对角占优与条件数

若每行满足 $|a_{ii}|>\sum_{j\ne i}|a_{ij}|$，矩阵严格行对角占优，因而可逆。证明很短：假设 $Az=0$ 且 $z\ne0$，选择 $|z_k|$ 最大的坐标，则

$$|a_{kk}||z_k|=\left|\sum_{j\ne k}a_{kj}z_j\right|
\le\sum_{j\ne k}|a_{kj}||z_k|<|a_{kk}||z_k|,
$$

产生矛盾。这解释了 AffineQuant §3.2 使用的充分条件；不意味着每个可逆矩阵都对角占优。

**优化过程还要守住条件。**设一行的占优余量 $m_i=|a_{ii}|-\sum_{j\ne i}|a_{ij}|>0$。对更新 $E$，三角不等式给出更新后余量至少为

$$m_i-|E_{ii}|-\sum_{j\ne i}|E_{ij}|.
$$

若该下界仍为正，就能保证这一行继续占优。只缩小非对角梯度而不约束对角更新，并不能自动满足它；例如 $A=I$、$E=\operatorname{diag}(-1,0)$ 就会变成奇异矩阵，与非对角稳定因子多小无关。AffineQuant 附录 A.2 的界依赖更新轨迹和对角项保持非零，应保留这些前提。

数值稳定性还取决于条件数 $\kappa_2(P)=\sigma_{\max}(P)/\sigma_{\min}(P)$。矩阵 $\operatorname{diag}(1,10^{-6})$ 严格对角占优且可逆，却有条件数 $10^6$；逆变换可放大某些方向上的微小扰动。正交矩阵的条件数为 1，但量化误差还受网格和分布影响，条件数好也不等于量化精度最高。

实际应区分浮点等价残差 $\|PP^{-1}-I\|$、最小奇异值/条件数与最终输出重构误差。前者很小不代表另两者合格；参见 [[quantization-error-diagnosis|量化误差诊断]]。

## 3. SVD 形式怎样表示可逆矩阵

令 $P=U\operatorname{diag}(s)V^\mathsf T$，其中 $U,V$ 正交且每个 $s_i\ne0$，则

$$P^{-1}=V\operatorname{diag}(1/s)U^\mathsf T,
\qquad P^{-\mathsf T}=U\operatorname{diag}(1/s)V^\mathsf T.
$$

真正 SVD 的奇异值非负；若代码直接学习可正可负的对角值，上式仍成立，但奇异值是 $|s_i|$。任何 $s_i$ 接近零都会使倒数变大。

FlatQuant 附录 B.1 用此结构组织可学习变换。它可以直接优化这些因子，不必每步先形成大矩阵再调用 SVD。其代码中的正交因子使用 Cayley 参数化：对 $B^\mathsf T=-B$，

$$U=(I+B/2)(I-B/2)^{-1}.
$$

PyTorch `orthogonal` API 的 Cayley 定义直接支持这一解释；它保证所参数化方阵正交，未约束另一个对角参数远离零。这里采用 2026-09-15 保存的 2.14 API 文档说明数学含义，不作为论文软件版本证据。

### Cayley 参数化与约束更新不是同一个步骤

上面是用自由参数生成正交因子；[[spinquant|SpinQuant]] 则直接更新正交旋转，并让每步尽量留在约束集合中。其基础来自 Li、Li 与 Todorovic 的 *Efficient Riemannian Optimization on the Stiefel Manifold via the Cayley Transform*（ICLR 2020，arXiv v1，§3.2–3.2.2、§4.1 算法 1）；本次只局部研读这些相关内容，不把整篇优化论文列为全文 ingest。

Stiefel 集合为 $\mathrm{St}(n,p)=\{R\in\mathbb R^{n\times p}:R^\mathsf TR=I_p\}$。对约束微分可得允许的一阶方向 $Z$ 满足 $R^\mathsf TZ+Z^\mathsf TR=0$，这就是切空间条件。以下仅展开 SpinQuant 所需的方阵 $n=p$：令 $D=-\nabla L(R)$ 是无动量的下降方向，

$$Y=\tfrac12(DR^\mathsf T-RD^\mathsf T),\qquad
YR=D-R\operatorname{sym}(R^\mathsf TD),$$

其中 $\operatorname{sym}(A)=(A+A^\mathsf T)/2$。右式是 $D$ 的切空间投影，$Y$ 反对称。令 $a=\eta/2$，用

$$C=(I-aY)^{-1}(I+aY),\qquad R_+=CR.$$

因为 $(I-aY)^\mathsf T=I+aY$，两个关于 $Y$ 的多项式彼此交换，所以 $C^\mathsf TC=I$，继而 $R_+^\mathsf TR_+=I$。实反对称矩阵的特征值为纯虚数或零，故实数 $\eta$ 下 $I-aY$ 可逆。小步展开为 $R_+=R+\eta YR+O(\eta^2)$；在光滑目标的非驻点，足够小的步长沿负投影梯度下降。这个局部结论不是对量化 STE 训练的全局收敛保证。（上述证明为依据该论文式 1–5 的教学展开。）

精确求解与快速近似也要区分。把上式改写为

$$Z=R+\tfrac\eta2Y(R+Z),$$

即可用 $Z_{k+1}=R+\frac\eta2Y(R+Z_k)$ 迭代，避免显式形成逆矩阵。在相容范数下，$\eta\|Y\|/2<1$ 是收缩的充分条件，迭代极限为精确 Cayley 解；有限次迭代与浮点舍入仍会带来正交残差。算法 1 通过步长限制和有限步迭代控制成本，含动量时还需把方向投影到当前切空间。不能把“每行归一化”当成完整正交化，也不能认为正交因子数值稳定就自动解决一般可逆矩阵中对角尺度接近零的问题。

## 4. Kronecker 乘积与 reshape

对 $P_1\in\mathbb R^{n_1\times n_1}$、$P_2\in\mathbb R^{n_2\times n_2}$，$n=n_1n_2$，定义 $P=P_1\otimes P_2$：其第 $(i,j)$ 个矩阵块为 $(P_1)_{ij}P_2$。

将行向量 $x\in\mathbb R^n$ **按行展开的逆操作** reshape 为 $V\in\mathbb R^{n_1\times n_2}$，则

$$x(P_1\otimes P_2)=\operatorname{vec}_{row}(P_1^\mathsf T V P_2).
$$

展开索引即可验证：输出坐标 $(j,b)$ 是 $\sum_{i,a}V_{ia}(P_1)_{ij}(P_2)_{ab}$。列优先 vec 约定的公式不同；实现需明确布局。（FlatQuant §3.1 式 3。）

一个非对称例子可以检验转置方向：

$$P_1=\begin{pmatrix}1&2\\0&1\end{pmatrix},\quad
P_2=\begin{pmatrix}2&0\\1&1\end{pmatrix},\quad
V=\begin{pmatrix}1&2\\3&4\end{pmatrix}.
$$

先算 $VP_2=\begin{pmatrix}4&2\\10&4\end{pmatrix}$，再算 $P_1^\mathsf T VP_2=\begin{pmatrix}4&2\\18&8\end{pmatrix}$；直接用 $(1,2,3,4)(P_1\otimes P_2)$ 也得到 $(4,2,18,8)$。反向还原使用两个小矩阵的逆，因为

$$(P_1\otimes P_2)^{-1}=P_1^{-1}\otimes P_2^{-1}.
$$

两因子都可逆时乘积可逆，其 2-范数条件数为两因子条件数的乘积；两个中等程度病态的小矩阵也可能合成较差的大矩阵。

## 5. 节约从哪里来，损失了什么

完整 $P$ 存 $n^2$ 个数，每 token 约 $n^2$ 次乘加。Kronecker 形式只存 $n_1^2+n_2^2$ 个数，两次小乘法约 $n(n_1+n_2)$ 次乘加。固定乘积时，两因子接近 $\sqrt n$ 最省：存储比最多约 $n/2$，乘加比最多约 $\sqrt n/2$。这比较的是变换本身，不是整个模型加速比。

例如 $n=4096$，取 $64\times64$，参数由 16,777,216 变为 8,192；每 token 乘加由 16,777,216 变为 524,288。不要实际构造大 Kronecker 矩阵，否则丢掉存储优势。内核还受布局、分块和片上容量影响，见 [[flatquant|FlatQuant 的融合实现]]。

单个 Kronecker 乘积是受限矩阵族，不能表示任意 $n\times n$ 矩阵。例如在 $2\times2$ 分块中，各块必须是同一个 $P_2$ 的标量倍数。FlatQuant 图 5 的精度试验只量化 LLaMA-2-7B 的最后一个 block（附录 C.6）；它支持该试验中分解影响较小，不能证明结构约束没有表达力损失。

**受限结构不等于低秩。**有 $\operatorname{rank}(P_1\otimes P_2)=\operatorname{rank}(P_1)\operatorname{rank}(P_2)$：两因子分别满秩时，整个 $n\times n$ 变换仍满秩，没有把通道压到低维。第 4 节的两个 $2\times2$ 因子均满秩，所得 $4\times4$ 变换秩为 4。节省来自坐标之间的耦合方式受限，而非丢弃某个子空间。

反过来，$\operatorname{diag}(1,1,1,2)$ 也可逆满秩，却不能在这一固定 $2\times2$ 分块下写成单个 $P_1\otimes P_2$：左上块是 $I_2$，右下块是 $\operatorname{diag}(1,2)$，二者不成比例。增加独立对角尺度、改变分解形状或复合多个变换都会改变可表示集合，所以该反例只限制单个固定布局的 Kronecker 族，不代表 FlatQuant 的全部可学习参数空间。

## 6. 激活加权的低秩近似

低秩补偿需要先指定“近似什么、在哪种误差下近似”。[[masquant|MASQuant]] v1 §3.2、定理 2 最小化给定激活 $A$ 下的输出残差 $\|A(D-L)\|_F^2$，其中 $D$ 是待补偿权重差、$\operatorname{rank}(L)\le r$。它与直接最小化 $\|D-L\|_F^2$ 的普通截断 SVD 不同。以下是依据该目标的教学推导。

设 $A\in\mathbb R^{n\times d}$ 满列秩，$D\in\mathbb R^{d\times o}$，取可逆 $T$ 满足 $T^\mathsf TT=A^\mathsf TA$。例如对二阶矩特征分解 $P\Lambda P^\mathsf T$，可取 $T=\Lambda^{1/2}P^\mathsf T$。则

$$\|A(D-L)\|_F^2=\operatorname{tr}((D-L)^\mathsf TA^\mathsf TA(D-L))=\|T(D-L)\|_F^2.$$

令 $Z=TL$，可逆性保证 $\operatorname{rank}(Z)=\operatorname{rank}(L)$。问题转成对 $TD$ 的普通最佳秩 $r$ 近似。若 $TD=U\Sigma V^\mathsf T$，取 $Z_r=U_r\Sigma_rV_r^\mathsf T$，再映回 $L=T^{-1}Z_r$，最小平方误差为舍弃奇异值平方和。为何尾和是下界：任一秩 $r$ 矩阵只能保留至多 $r$ 维列空间；将 $TD$ 正交投影到该空间后，其余能量无法由该矩阵表达。选择最大 $r$ 个左奇异方向保留的能量最多，正好达到下界。

**数值例子。**令 $A=\operatorname{diag}(0.1,10)$、$D=\operatorname{diag}(2,1)$、$r=1$。普通 SVD 保留 $D$ 的第一方向，输出残差平方为 100；加权 SVD 看到 $TD=\operatorname{diag}(0.2,10)$，保留第二方向，输出残差平方只有 0.04。哪一方向重要由激活如何激发它决定，不能只比较权重奇异值。

但可逆 $T$ 保持 $D$ 的代数秩。上述 $D$ 与 $TD$ 的秩都是 2；加权后更容易近似与“变成低秩矩阵”不同。若 $A=I$、$D=I_d$，任意 $r<d$ 的最小误差就是 $d-r$，没有普遍快速衰减保证。有效秩、能量集中和代数秩必须分开。

若 $A^\mathsf TA$ 奇异，普通 $T^{-1}$ 不存在；可在受观测子空间用伪逆处理，但未被校准激活覆盖的方向没有同样约束。加 $\epsilon I$ 则对应原目标额外加上 $\epsilon\|D-L\|_F^2$，已经改变优化问题。近奇异矩阵还会使映回因子放大，不能只检查 SVD 的尾和。

[[splitq|SplitQ]] v1 式 21 使用另一类约束：固定截断奇异子空间、学习对角门控，并配合可学习逆变换。它限制可用方向，不等价于上面的加权最佳低秩解；因子再次量化、加入激活误差后，也没有自动继承上述最优性。

