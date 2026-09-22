---
title: QERA：激活二阶矩加权的低秩量化误差修正
type: method
tags:
  - ptq
  - low-rank
  - reconstruction
  - calibration
sources:
  - raw/papers/2026-09-22/loftq/paper.pdf
  - raw/papers/2026-09-22/qera/paper.pdf
updated: 2026-09-22
---

# QERA：激活二阶矩加权的低秩量化误差修正

量化后增加一个低秩分支，可以补回部分权重误差；但有限的秩应该用于恢复哪些方向？QERA 的回答是：先用输入激活的二阶矩度量残差，再做低秩近似。它给出固定量化主体、固定输入分布下的层输出平方误差最优解，并提供只保留对角统计的便宜近似。这里的“最优”不覆盖整个模型、任务损失、联合 W/A 量化或部署速度。

依据 **QERA: an Analytical Framework for Quantization Error Reconstruction，arXiv:2410.06040v2，ICLR 2025**（下称 P）。研读正文 §1–6、附录 A.1–A.10，定向回查方法、结果及成本的 PDF 页图；A.11 热图只作定性抽查，没有逐图验证作者的覆盖比例。未读取官方实现或运行模型。

## 1. 低秩分支补什么

沿用 P 的行向量约定：$x\in\mathbb R^{1\times m}$，$W\in\mathbb R^{m\times n}$，$y=xW$。这与 [线性层基础](../fundamentals/operators/linear-layer-input-channel.md) 的权重存储方向互为转置。量化器先产生编码 $W_q=q(W)$，解码后的数值矩阵为 $\widetilde W=dq(W_q)$；公式中的 $\widetilde W$ 不是整数编码本身。

令残差 $E=W-\widetilde W$，加入高精度因子 $A\in\mathbb R^{m\times k}$、$B\in\mathbb R^{k\times n}$，得到

$$\widehat y=x\widetilde W+(xA)B,\qquad C=AB,\quad\operatorname{rank}(C)\le k.$$

**QERA 求解期间固定 $\widetilde W$，只求 $C$。**底层量化器可以替换，但更换后残差也随之改变，必须重算补偿。它没有同时寻找最佳量化网格、整数编码和低秩因子。（P §3.1，式 5–9；附录 A.3。）

两种看似接近的目标实际不同：

$$J_W(C)=\|E-C\|_F^2,\qquad
J_X(C)=\mathbb E_x\|x(E-C)\|_2^2.$$

普通截断 SVD 最小化 $J_W$，把各输入方向同等对待；QERA 最小化 $J_X$，关心误差怎样被实际输入激发。它与 [层输出重构](../theory/layer-reconstruction-second-order-compensation.md) 使用相同类型的输入相关度量，但优化变量不同：GPTQ 选择量化权重并补偿尚未确定的权重，QERA 在固定主体之外保留低秩旁路。

已有 [激活加权低秩近似](../fundamentals/mathematics/invertible-transforms-and-kronecker-products.md#6-激活加权的低秩近似) 给出教学反例：$E=\operatorname{diag}(2,1)$，输入加权因子为 $\operatorname{diag}(0.1,10)$，秩为 1。按权重大小保留第一方向时，输出残差平方为 100；按输入加权保留第二方向时为 0.04。低秩预算应保护输出中重要的方向，不必保护最大的权重残差。

## 2. Exact：把输出目标转成可解的截断 SVD

定义未中心化二阶矩

$$R=\mathbb E[x^{\mathsf T}x]\in\mathbb R^{m\times m}.$$

它通常不等于协方差：若 $\mu=\mathbb E[x]$，则 $R=\operatorname{Cov}(x)+\mu^{\mathsf T}\mu$。P 称其为 autocorrelation matrix，不能在实现时擅自减掉均值。

设 $D=E-C$。按 P §3.2 式 13–17 等价整理，

$$J_X(C)=\operatorname{tr}(D^{\mathsf T}RD)=\|R^{1/2}D\|_F^2.$$

当 $R$ 正定时，其对称平方根 $T=R^{1/2}$ 可逆。令 $Z=TC$，可逆变换保持秩，原问题变成 $\min_{\operatorname{rank}(Z)\le k}\|TE-Z\|_F^2$。对加权残差做 SVD：

$$TE=U\Sigma V^{\mathsf T},\qquad
C_\star=T^{-1}U_k\Sigma_kV_k^{\mathsf T}.$$

可保存 $A=T^{-1}U_k$、$B=\Sigma_kV_k^{\mathsf T}$，推理时旁路只执行两次小矩阵乘，不构造稠密 $C$。最小目标值是 $\sum_{i>k}\sigma_i^2(TE)$，不是未加权 $E$ 的奇异值尾和。（P 定理 1、式 10–20；尾和是截断 SVD 的教学展开。）

这也解释了如何选秩的局部意义：增加 $k$ 不会增加这一固定问题的最优残差，但它不证明逐层增加秩一定提高整网任务分数。低秩因子本身可能秩不足 $k$；“秩 $k$”一般表示预算上界。截断边界奇异值相等时，最优解也未必唯一。

### 可逆性、校准与数值条件

实践使用 $N$ 个有效输入向量组成的 $X$ 估计 $\widehat R=X^{\mathsf T}X/N$。这使 exact 成为**经验分布下的解析解**，不会使有限校准集自动代表部署分布；$N$ 是进入统计的向量数，不能直接与文本条数混用。

P Remark 1 报告其所测矩阵可逆，并建议奇异时加小对角扰动。一般情况下，$X$ 不满列秩或输入方向未被激发都会使逆不存在。可以在观测子空间用伪逆，但无法由这些数据确定未观测方向；近奇异时求逆也可能放大因子。若采用 $R+\epsilon I$，优化目标变成 $J_X(C)+\epsilon\|E-C\|_F^2$，已不是原目标；若扰动的是平方根 $T$，则又对应另一种度量。后两点是教学推导，不能统称为“不改变最优解的数值修补”。

P 附录 A.7 的数值经验是：外积用 FP32，外积累加用 FP64，矩阵平方根用 FP64。附录 A.4 使用 CPU 上 SciPy 的 blocked Schur 算法。这是作者实现报告，不是本轮代码核对，也不说明其他平方根或分解算法必然不适用。

## 3. Approx：什么时候可以只统计每个通道的 RMS

P Assumption 1 的精确条件是

$$\mathbb E[x_i x_j]=0\quad(i\ne j).$$

此时 $R$ 为对角矩阵，取

$$S=\operatorname{diag}\!\left(\sqrt{\mathbb E[x_1^2]},\ldots,\sqrt{\mathbb E[x_m^2]}\right),\qquad
C_{\rm approx}=S^{-1}(SE)_k.$$

其中 $(SE)_k$ 是秩 $k$ 截断；各 RMS 必须非零才能直接取逆。它沿用 exact 的形式，但把 $m\times m$ 统计压缩为 $m$ 个数，缩放与逆缩放也变成逐行操作。SVD 仍然存在，并没有消除全部分解成本。（P §3.3，式 21–24；附录 A.2。）

**“不相关”必须按式 21 理解。**一般有 $\mathbb E[x_ix_j]=\operatorname{Cov}(x_i,x_j)+\mu_i\mu_j$，所以协方差为零但均值非零时，条件仍可能失败。RMS 也不是标准差，除非相应均值为零。

若条件不成立，approx 最小化的是以 $R$ 的对角部分代替完整 $R$ 后的目标。P §5 和附录 A.11 用部分层、部分通道的二阶矩热图讨论适用性，并指出一些 attention 投影输入有明显交叉项；这些定性图不证明所有层或数据都满足条件，也不直接给出任务误差上界。

**忽略交叉项的教学反例。**取 $E=\operatorname{diag}(2,1)$、$k=1$，

$$R=\begin{pmatrix}1&0.9\\0.9&1\end{pmatrix}.$$

approx 看到 $S=I$，保留第一方向，真实 $J_X=1$。exact 利用交叉项，最优值等于 $E^{\mathsf T}RE$ 的较小特征值，即 $(5-\sqrt{21.96})/2\approx0.1569$。两者有相同通道 RMS，方向之间的关系却会改变最优补偿。

## 4. 从校准到推理，或从初始化到微调

依据 P §3–4，可将核心流程整理为：

1. 确定模型、目标线性层、量化器与秩预算，保留参考 $W$，生成固定 $\widetilde W$。
2. 在明确的数据及模型状态上收集该层输入。exact 累积 $x^{\mathsf T}x$，approx 累积逐通道平方；记录有效向量计数、mask 与统计精度。
3. 构造 $T$ 或 $S$，检查可逆性，对加权残差做截断 SVD，再逆缩放左因子。
4. **PTQ 路线**直接使用 $x\widetilde W+(xA)B$；**QPEFT 路线**把这些因子作为初始化，固定量化主体，通过任务训练继续更新适配器。训练后不再保证因子仍是同一固定残差问题的解析最优解。
5. 分别检查经验层输出误差、整网质量与实际执行成本。若传播上游量化后的输入、改变 mask、重新量化主体或量化因子，须重新说明实际优化的目标。

上述 mask、状态和数值检查是由推导整理出的使用条件，不冒充已核对的官方代码步骤。P 图 1 使用 128 个样本考察 RoBERTa 初始化误差，这不是所有 PTQ/QPEFT 实验统一采用 128 条数据的充分依据。本轮也没有从论文中确认全部模块排除、校准抽样及适配器库的缩放约定。

PTQ 的低秩项在离线求出后无需训练；QPEFT 则以误差重构为起点，再优化任务目标。两者不能仅因为使用同形状因子就统称为 QAT。若实现额外乘 $\alpha/k$，必须让实际旁路乘积与解析 $C$ 对齐；这是接入检查，不是 P 新增的超参数结论。相关范式见 [PTQ、QAT 与代理梯度](../fundamentals/quantization/post-training-and-quantization-aware-training.md)。

## 5. 与已有低秩方法的具体关系

| 路线 | 固定与优化的对象 | 与 QERA 的区别 |
| --- | --- | --- |
| 普通残差 SVD | 固定量化主体，对 $E$ 截断 | 优化权重误差；当输入二阶矩为正标量乘单位阵时，与输出目标等价 |
| [LoftQ 的交替初始化](loftq.md) | 交替量化 $W-AB$ 与截断当前残差 | 主体也改变；量化子步骤只是近似更新，不能将 QERA 的固定主体最优性套到全部迭代 |
| LQER 的尺度加权 | 按输入统计缩放残差后分解 | P 附录算法 2 写为平均绝对激活；QERA-approx 使用 RMS，并给出对应二阶目标的条件 |
| QERA-exact / approx | 固定量化主体，求高精度低秩补偿 | 完整二阶矩与对角二阶矩的区别，不是不同的推理运算图 |
| MASQuant | 用文字量化主体近似各模态需要的缩放权重 | 同属激活加权低秩近似，但残差还包括模态尺度差异 |
| SplitQ | 从主体拆出低秩成分，并另外补偿激活残差 | 改变主体与分支结构，因子还有量化、子空间与门控约束 |

普通残差 SVD 与 LQER 的概括依据 P §2、附录 A.1，不代表已独立核对这些方法的全部原文或实现。LoftQ 条目已进一步按其 v4 §3、算法 1 核验：$T=1$ 时主体就是同一量化器直接作用于原权重的结果，多轮则可能改变主体；推导、顺序与实验条件见 [LoftQ](loftq.md)。MASQuant 与 SplitQ 的完整对象和公式分别见其 [共享权重补偿](masquant.md) 与 [权重／激活双补偿](splitq.md) 页面。

**CALDERA 的关系与原文公式边界。**P 附录 A.3 将 CALDERA 描述为同时优化量化主体与低精度因子的更大问题，并声称其中一个局部引理与 QERA-exact 等价。本轮未独立核对 CALDERA 原文。P 自身式 36–37 存在不能直接照搬的写法：$\Sigma V^{\mathsf T}/\sqrt N$ 一般不是此前定义的对称平方根，式 36 又写了 $V\Sigma$ 而非推导所需的逆奇异值。

教学上可用一致的因子表述消除混淆：若 $X=U_X\Sigma_XV_X^{\mathsf T}$ 满列秩，则 $T=\Sigma_XV_X^{\mathsf T}/\sqrt N$ 满足 $T^{\mathsf T}T=\widehat R$，虽非对称平方根，却同样可用于加权低秩推导，得到 $C=V_X\Sigma_X^{-1}(U_X^{\mathsf T}XE)_k$。这一代数解释支持加权分解本身，不将原文有问题的等式原样作为证明，也不替代 CALDERA 的独立核验。

## 6. 实验支持了什么，哪些概括需要收紧

### PTQ：局部最优并不意味着所有模型的 PPL 最低

P 附录 A.4.2 对误差重构方法使用 MXINT、block size 32；4/3 位主体对应表中平均 W-bits 4.25/3.25。HQQ 使用其内置 INT、group size 64，4 位时同列为 4.25，但网格与表示不同。低秩因子另有成本，不应把 W-bits 当作整模型含旁路后的位宽。

以下只取 P 表 3 中无需解决主表／附录冲突的行。指标为作者报告的 WikiText2 **word perplexity**，不是跨项目统一口径；P 说明默认使用模型上下文长度，Phi-3.5 与 LLaMA-3.1 则使用 2048。

| 模型与配置 | 无补偿 | LQER | QERA-approx | QERA-exact |
| --- | ---: | ---: | ---: | ---: |
| LLaMA-2-7B，W-bits 4.25，补偿秩 32 | 9.45 | 9.22 | 9.17 | 9.12 |
| LLaMA-2-7B，W-bits 3.25，补偿秩 64 | 13.32 | 14.00 | 10.99 | 10.67 |
| TinyLlama-1.1B，W-bits 4.25，补偿秩 32 | 19.40 | 16.23 | 15.66 | 16.16 |

前两行说明在这些条件下输入加权补偿有效；第二行也表明增加一个启发式补偿项可能比不补更差。第三行 exact 的整网 PPL 高于 approx，因此不能照搬“exact 总是最好”的概括。不同层误差的传播、分布差异与最终指标均不在单层最优性保证内，当前材料也不足以把这一反例归因于其中某一个原因。

### QPEFT：更好的起点与更好的训练后结果需要分别观察

P 图 1 和附录图 6 在 RoBERTa-base 上展示：LoftQ 增加迭代后，各层权重残差可以下降，而整网输出误差并不单调下降。它支持重新审视优化目标，不证明任何权重误差方法都无效，也不证明 QERA 的层目标与任务损失恒等。

P 表 1 中，RoBERTa-base 的 3 位、秩 8、W-bits 3.25 设置下，LoftQ 的 GLUE 汇总分数为 73.31，QERA-approx 为 77.43，增加 4.12 个分数点。这个 Avg. 混合准确率、相关系数等任务指标，不宜全部称为准确率提升。附录 A.4.1 说明该组用模拟 MXINT、block size 32、batch 64、训练 5 epochs，对各方法／任务搜索学习率，结果取三个种子的平均。

### 必须保留的原文不一致

- §4.1 将 QPEFT 概括为使用 approx，表 1 的标题也这样描述；但表 1 和附录表 6 的 **2 位 RoBERTa、秩 64** 明确标为 **QERA-exact**。70.18 → 76.23 的 6.05 分增益只能按这一行记录，不能归给 approx。
- 表 3 的 Phi-3.5、W-bits 4.25、exact PPL 为 **12.30**，附录表 13 对应行为 **13.00**。本页不裁定哪一个正确，不用它支撑精确的收益比较。
- 表 3–4 最后一列写为 LLaMA-3.1 **80B**，摘要与附录表 17 则写 **70B**。这不是本地抽取造成的自动纠错机会；本页避开该列的型号—数字组合，保留版本内差异。

## 7. 初始化成本与推理成本要分别记账

exact 需要完整二阶矩、矩阵平方根和加权 SVD；approx 只需逐通道平方和与缩放 SVD。P 图 8b 的逐层串行离线流程显示，CPU 平方根是主要成本差异来源。这不是两种方法的推理速度，也不是所有实现的复杂度下界；绝对成本还依赖统计精度、矩阵规模、CPU/GPU 分工与层间并行。

附录 A.8 建议将 approx 用于 QPEFT，把节省的初始化时间用于增加秩或训练步数。表 8 的 LLaMA-2-7B／SlimPajama：exact 秩 16、2 epochs 总计 6.8 h、PPL 6.31；approx 秩 64、2 epochs 为 2.6 h、PPL 6.18；approx 秩 16、4 epochs 为 4.5 h、PPL 6.21。A.4.1 报告 QPEFT 平台为四张 A100 80GB、AMD EPYC 64 核 CPU 和 1024GB RAM，矩阵平方根在 CPU 执行。这组比较同时改变了秩或训练量，支持的是作者平台上的总预算分配取舍，不能解释为同配置初始化的纯消融或其他设备的固定时间比。

在相同形状、秩、dtype 和后端下，exact 与 approx 都可部署成 $x\widetilde W+(xA)B$，没有运行时重新估计二阶矩或求平方根的需要。但相对于只有主体的路径，两者都增加 $k(m+n)$ 个因子元素和两次低秩乘法。详细记账见 [低秩辅助分支](../implementation/quantized-matmul-scaling-execution.md#12-低秩辅助分支不是免费或统一位宽)。不能把“exact 不比 approx 多一条推理分支”改写成“低秩补偿免费”。

## 8. 适用边界与继续使用

QERA 适合回答“固定这一份量化主体后，怎样用有限秩恢复参考线性层输出”。它的主要增量是目标与解析解，而非新的量化编码格式。复用时应分别核对：统计是否来自目标分布、非对角二阶矩是否重要、逆是否稳定、因子是否仍保持所需精度、低秩预算是否值得其实际成本。

P 图 3 的样本量曲线是特定实验中的观察，不能证明增加数据必然使任务质量单调提高；附录 A.6 将 SST2 校准后的训练异常归因于 padding 时，正文用的是假设，不能写成已隔离验证的因果结论。相关数据条件见 [校准统计与任务目标](../theory/calibration-and-range-selection.md#8-校准统计稳定不等于任务目标一致)。

本轮没有验证具体 checkpoint 的量化模块范围、官方代码的输入统计与分解路径、适配器缩放或低比特 Kernel；没有模型微调、PTQ 精度或端到端速度复现。也没有证明该定理直接覆盖激活／KV 量化，或低精度因子联合优化。所用原文分歧及未确认配置不影响固定 $R$、$E$、秩预算下的主推导，但限制了对应实验的复现与外推。

## 来源身份

| 来源 | 版本 | 本轮范围 |
| --- | --- | --- |
| [QERA: an Analytical Framework for Quantization Error Reconstruction](https://arxiv.org/abs/2410.06040v2) | `arXiv:2410.06040v2`，2025-02-15 | 正文与必要附录；公式、主表／附录差异及成本页图核对，A.11 定性抽查；官方代码未读 |
| [LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models](https://arxiv.org/abs/2310.08659v4) | `arXiv:2310.08659v4` | §3、算法 1 的主体更新与残差替换关系；代码未读 |

## 教学计算材料

[check_math.py](../assets/qera/check_math.py) 与 [math-validation.json](../assets/qera/math-validation.json) 核对小矩阵上的加权目标、最优尾和、对角近似反例、非零均值条件、替代因子、奇异统计和存储记账。它们不调用官方 QERA，不构成模型或性能复现。
