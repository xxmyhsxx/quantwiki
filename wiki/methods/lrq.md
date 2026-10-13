---
title: LRQ：低秩对数缩放、块重构与参数预算
type: method
tags:
  - ptq
  - low-rank
  - reconstruction
  - weight-quantization
sources:
  - raw/papers/2026-09-22/lrq
updated: 2026-10-13
---

# LRQ：低秩对数缩放、块重构与参数预算

LRQ 固定预训练权重，把逐权重选码除数的对数参数写成低秩乘积与广播偏置，再以 Transformer block 输出重构联合学习这些参数和网格步长。它保留逐元素改变选码坐标的能力，同时约束参数共享方式；低秩作用于对数参数，不要求原权重或最终量化权重低秩，也不在推理时增加一个低秩残差分支。

本页依据 LRQ: Optimizing Post-Training Quantization for Large Language Models by Learning Low-Rank Weight-Scaling Matrices，arXiv:2407.11534v2。核心方法、预算、消融与成本条件回查固定36页PDF；数字为作者报告，未核作者实现或运行模型。前置的网格/零点定义见[均匀量化](../fundamentals/quantization/uniform-quantization-and-groups.md)，除数与恢复步长的区别见[FlexRound](flexround.md)。

## 1. 低秩究竟约束哪一个矩阵

令 $W\in\mathbb R^{m\times n}$，$m=C_{out},n=C_{in}$。按论文§2.3式2及附录M，用明确的广播记号写为

$$
Z=LU+a\mathbf1_n^{\mathsf T}+\mathbf1_m c,
\qquad
\widehat W=s_1\odot\operatorname{round}\left(W\oslash[s_1\odot\exp(Z)]\right).
$$

| 参数 | 形状与作用 |
| --- | --- |
| $L$ | $m\times r$，输出侧因子，对应原文$L_2$ |
| $U$ | $r\times n$，输入侧因子，对应原文$U_2$ |
| $a$ | $m\times1$，各输出行的对数偏置，对应$r_2$ |
| $c$ | $1\times n$，各输入列的对数偏置，对应$c_2$ |
| $s_1>0$ | 网格步长，按选定量化粒度广播；不是$\exp(Z)$ |

$\oslash$、$\odot$分别是逐元素除法和乘法，$\exp$也是逐元素指数，不是矩阵指数。于是$Z_{ij}=(LU)_{ij}+a_i+c_j$，每个权重可以有不同正除数，但它们通过因子与偏置共享参数。外侧$s_1$决定恢复网格，内部$\exp(Z)$决定送入舍入函数的坐标。

LRQ v2 §2.1把所用FlexRound的正除数写成$\exp(S_2)$，因此这里$S_2$是对数参数。[FlexRound原方法页](flexround.md)的同名$S_2$表示正的乘法因子；不能只按符号名称把两版参数化混成同一个变量，也不能倒推FlexRound原文必须使用指数参数化。

**秩界的整理者分析。** $\operatorname{rank}(LU)\le r$；两个广播项各至多秩1，故

$$\operatorname{rank}(Z)\le\min(m,n,r+2).$$

逐元素指数不保持这个秩界。教学反例为

$$
Z=\begin{pmatrix}0&0\\0&1\end{pmatrix},\qquad
\exp(Z)=\begin{pmatrix}1&1\\1&e\end{pmatrix}.
$$

前者秩1，后者行列式$e-1\ne0$、秩2。因此“低秩缩放”应定位到对数参数的结构约束，不能解释成低秩反量化矩阵或低秩主权重。式2还省略有限整数范围和非对称零点，实际格式的裁剪、零点消费及反向规则仍须代码核对。

## 2. 块输出目标与初始化

原文§2.1/2.3以$\|WX-\widehat W\widetilde X\|_2^2$表示重构：$X$来自全精度路径，$\widetilde X$来自前序已量化块，两者不默认相同。对含attention、非线性和残差的整块，应明确比较完整块函数。以平方误差形式整理为

$$
\min_{\theta_{block}}\ \|F_W(X)-F_{\widehat W(\theta_{block})}(\widetilde X)\|_F^2,
\qquad \theta_{block}=\{(s_{1,j},L_j,U_j,a_j,c_j):j\in\mathcal I\}.
$$

这里$\mathcal I$是当前块所选目标矩阵的集合，$F_W$中的$W$代表整个块的固定权重；量化分支还包含该方案在重构时已启用的激活量化，后加激活/KV的时序见第5节。论文称为MSE，具体归约、mask与缓存协议未核实现。当前块的多组量化参数联合优化，不是对原权重做截断SVD，也不等于纯权重距离目标或整网任务训练。层/块及输入路径的共同解释由[重构机制](../theory/layer-reconstruction-second-order-compensation.md#8.%20%E4%B8%BA%E4%BB%80%E4%B9%88%E8%BF%98%E9%9C%80%E8%A6%81%E8%80%83%E8%99%91%E5%A4%9A%E4%B8%AA%E5%B1%82%E7%9A%84%E8%81%94%E5%90%88%E9%87%8D%E6%9E%84)承接。

论文令$L=0$、$U$取正态随机值、$a=c=0$，因此$Z=0$、$\exp(Z)=1$，从最近舍入开始；$s_1$按背景FlexRound的权重量化误差初始化，再进入块重构联合学习。原文未给正态方差，不能补写成已核配置。

**给定对数参数梯度后的链式法则。** 设$G_Z=\widetilde\partial\mathcal L/\partial Z$是当前采用反向规则得到的梯度，则整理者展开为

$$
\nabla_L\mathcal L=G_ZU^{\mathsf T},\qquad
\nabla_U\mathcal L=L^{\mathsf T}G_Z,
\qquad
\nabla_{a_i}\mathcal L=\sum_j(G_Z)_{ij},\quad
\nabla_{c_j}\mathcal L=\sum_i(G_Z)_{ij}.
$$

初始$U$梯度为零，$L$仍可能获得非零更新；如果两个因子都为零，乘积分支的一阶梯度都为零，偏置分支仍可更新。相同的乘积初始化问题在[LoftQ的初始化说明](loftq.md#1.%20%E4%B8%BA%E4%BB%80%E4%B9%88%E9%9B%B6%E8%BE%93%E5%87%BA%E9%80%82%E9%85%8D%E5%99%A8%E4%B8%8D%E8%B6%B3%E4%BB%A5%E6%81%A2%E5%A4%8D%E9%87%8F%E5%8C%96%E5%89%8D%E7%9A%84%E8%B5%B7%E7%82%B9)中已有解释，但LRQ没有LoftQ的残差SVD步骤。

若固定$s_1$、没有裁剪且对round采用恒等STE，则还可得到条件性的$G_Z=-W\odot\exp(-Z)\odot G_{\widehat W}$。这是为理解指数参数化而作的代理展开，不是已核作者反向实现；硬round真实导数几乎处处为零，规则及边界见[PTQ、QAT与代理梯度](../fundamentals/quantization/post-training-and-quantization-aware-training.md)。

## 3. 参数预算：因子、完整辅助变量与内存

按式2形状计数，单个$m\times n$矩阵的实际可训练元素为

$$
N_{LU}=r(m+n),\qquad
N_{\rm LRQ}=r(m+n)+(m+n)+N_s,
$$

其中$m+n$是两个广播偏置，$N_s$是共同网格步长数；若每输出行一个步长，则$N_s=m$。这里计的是实存因子元素，不是秩$r$矩阵的内在自由度$r(m+n-r)$；参数化有冗余，优化器仍须保存对应元素。

仅比较$LU$与自由同形对数矩阵的$mn$个参数，因子更少要求

$$r<\frac{mn}{m+n}.$$

因此原文$r<\min(m,n)$不足以保证省参数。教学例：$m=n=4096,r=3072$仍小于矩阵维度，两个因子却有25,165,824个元素，是原矩阵16,777,216个元素的1.5倍。若计LRQ两个偏置、与$mn+N_s$的自由对数矩阵加同样步长比较，条件进一步变成

$$r<\frac{mn}{m+n}-1.$$

比较完整FlexRound时还要把其原有通道参数计入；LRQ背景式1明确为简洁而隐去了这些向量，不能据该简式将全部实际变量默认为零。

### 表29只计主因子

附录J表29按Transformer块的矩阵形状计算比例：

| 模型 | 使用的因子宽度$r$ | 表29因子元素/矩阵权重 |
| --- | ---: | ---: |
| Llama 7B | 1024 | 39.51% |
| Llama 13B | 1024 | 31.57% |
| Llama 33B | 2048 | 48.60% |
| Llama 65B | 2048 | 39.51% |

表中B列只列$L/U$，没有计$a,c$或$s_1$。以7B为例，分母$4\times4096^2+3\times4096\times11008=202,375,168$，主因子79,953,920；按同样七矩阵形状另有78,080个广播参数，再加$N_s$个步长。39.51%因而不是完整辅助参数比例、全模型压缩率或峰值显存比例。

离线计算仍涉及同形的$Z$、正除数、量化权重和相应梯度；总内存还受当前块、校准张量、梯度及优化状态的驻留方式影响。论文没有分项披露这些张量的dtype与驻留协议，不能从因子元素数推出已验证的字节预算。前向中构造$LU$也有计算成本，少参数不自动意味着更快校准。

## 4. 参数共享与泛化证据

作者§2.2的假设是：极少校准数据下，同形自由参数容易拟合校准样本，却不一定迁移到MMLU。图3及附录C/D在选取的校准、未见样本上观察到LRQ保持相近校准RMSE而未见RMSE更低，支持这个解释；没有给出普遍泛化保证。512条文本也不是512个标量观测，序列长度、输出维度、相关性和输入分布都会影响有效信息量。

附录B表10：Llama 7B、权重逐通道非对称W8、静态逐张量A8、KV FP16，采用正文默认C4校准512样本、长度1024。附录I为该静态方案给出每块5000次及完整LRQ/FlexRound的batch、学习率，但表26未列出表10，表10也未逐组披露全部优化配置。其five-shot MMLU为：

| 参数化 | MMLU / % |
| --- | ---: |
| 同形自由对数参数，FlexRound | 30.20 |
| 只替换为$LU$ | 33.86 |
| $LU$加两个广播偏置，LRQ | 34.47 |

这支持该设置中参数共享和广播项各有作用，不是严格控制所有优化条件后的唯一因果证明，也不说明增加任意参数都改善泛化。

§3.1图4扫描Llama 7B的因子宽度64–8192，W8/A8、KV FP16：MMLU从$r=64$时33.97%到1024时34.47%，继续扩大至2048–8192后下降、接近FlexRound。大宽度扫点可超过部分投影的较小维度，应称因子宽度扫描，不能把每个候选都称为严格低秩或省参数。校准样本超过1024后也趋于饱和，图中存在轻微回落。

任务与指标还可能分歧。表7的Llama 2 7B、逐通道W3、激活/KV FP16，正文默认C4校准样本数和长度同上：LRQ zero-shot CSR平均59.07%高于FlexRound的58.74%，WikiText2 PPL却为6.48、差于6.34。此配置LRQ学习率0.0006、FlexRound为0.0025（附录I表28）；不能由一个指标或一组不同超参数的结果推出全场景最优。

## 5. 实际流程与混合配置

论文默认在单张A100 80GB上使用C4训练集随机512条、长度1024的校准数据。较小模型用$r=1024$，超过30B用2048；原权重固定，按块顺序优化，再接续后块的量化输入。各量化方案要分别记录（§3，附录I）：

| 方案 | 重构和量化顺序 | 必须保留的例外 |
| --- | --- | --- |
| W8、静态逐张量A8 | 按QDrop设置，每块5000次，激活采用RTN；若增加逐token KV量化，在整套块重构后加入 | Llama 2 70B的K/V投影用FlexRound；LRQ并非所有矩阵统一替代 |
| W8/W4、逐token A8 | 先按BRECQ设置完成权重重构，每块5000次、batch 2，再依次用RTN加入逐token激活和KV量化 | 70B K/V仍用FlexRound；W4的7B为attention LRQ、FFN FlexRound，70B为FFN LRQ、attention FlexRound |
| 逐通道W3/W4、A/KV FP16 | 权重块重构，具体学习率见附录I表28 | 不与逐token W4A8或静态W8A8结果混成同一配置 |

附录I图8–9保留softmax和归一化的输入为FP16。上表中的低位宽标签不表示全图所有张量都同精度，也不表示低秩参数与后加入的激活/KV参数一起联合训练。

从表示看，LRQ区别于[DBW](dbw.md)的低秩主权重，也区别于[QERA](qera.md)和[LoftQ](loftq.md)的加法残差。后两者在量化主体之外保留或训练低秩补偿；LRQ的因子只用于离线选码，不形成额外输出支路，也没有成对的输入逆缩放来保证未量化函数不变。

## 6. 离线成本与推理保存分开计

附录F表13报告Llama 7B、逐通道W8/静态A8、512样本、batch 2的离线成本：FlexRound为5小时7分、25.4 GB峰值显存，LRQ为5小时22分、23.5 GB。表14的Llama 2 7B、逐通道W4-only、同样样本/batch则为2小时50分、23.6 GB与3小时3分、21.3 GB。原文将LRQ更长的时间归于因子乘法；参数减少与总内存、总时间的变化不成同一比例。

附录G先固化整数码，再丢弃$L,U,a,c$，推理无需重算$LU$。真实非对称格式仍要正确保存零点、步长、整数载荷与布局；不能把原文省略这些细节的式2当作完整导出规范。[实际执行路径](../implementation/quantized-matmul-scaling-execution.md)说明量化数值、压缩文件与后端消费需要分别核验。

表15的2.33倍/2.77倍仅对应Llama 2 70B的W4/W3逐通道FFN矩阵乘、单token、LUT-GEMM对cuBLAS FP16；不是整模型服务吞吐。本文未核作者导出、打包、kernel或GPU运行，不能从因子可消去直接认定任意后端都兼容或获得这些性能。

## 来源身份

LRQ: Optimizing Post-Training Quantization for Large Language Models by Learning Low-Rank Weight-Scaling Matrices（`https://arxiv.org/abs/2407.11534v2`），arXiv:2407.11534v2；正文§2–4、式1–2、图3–4、表7，附录B/F/G/I/J/M。秩界、参数条件、初始化链与条件性STE展开为整理者分析；全部附表、作者实现与模型/性能复现未逐项核验。
