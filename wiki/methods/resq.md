---
title: ResQ：高精度子空间、组内旋转与混合精度执行
type: method
tags:
  - mixed-precision
  - rotation
  - low-rank
  - kv-cache
sources:
  - raw/papers/2026-09-21/resq/paper.pdf
  - raw/repositories/2026-09-21/resq/source/fake_quant/get_basis.py
updated: 2026-10-12
---

# ResQ：高精度子空间、组内旋转与混合精度执行

ResQ 先选出承载较大激活能量的低秩子空间，将其系数保留为8 bit，其余系数量化为4 bit；然后在两个子空间内部各自旋转，以改善量化分布。它保护的是投影后的信号分量，并保留完整输入维度，不是先量化再用低秩矩阵补偿误差，也不是删掉低能量维度。依据为 ResQ v2（`https://arxiv.org/abs/2412.14363v2`） 正文、必要证明和实现附录；原文最优性证明存在下述代数问题，经验结果与证明有效性分别判断。

与 [CMPQ](cmpq.md) 的原输入通道保护不同，ResQ 先改变坐标再分配组内精度；与 [SpinQuant](spinquant.md) 的任务损失训练不同，它通过校准统计量选择基底。公共推导见[高精度子空间与能量代理](../theory/orthogonal-rotation-and-hadamard-quantization.md#10-先选高精度子空间再在组内旋转)，固定代码的实际配置和执行边界见[实现页](../implementation/resq-basis-and-fake-quantization.md)。

## 1. 两个精度分组怎样得到一个输出

采用行 token 约定，$X\in\mathbb R^{n\times d}$、$W\in\mathbb R^{d\times o}$。正交基 $U=[U_l\ U_h]$ 中，$U_l$ 为 $d\times(d-r)$，$U_h$ 为 $d\times r$，$U_lU_l^\top+U_hU_h^\top=I$。$Q_L,Q_H$ 表示包含恢复尺度的量化后实数近似。定义

$$
A_l=Q_L(XU_l),\quad A_h=Q_H(XU_h),\quad
B_l=Q_L(U_l^\top W),\quad B_h=Q_H(U_h^\top W).
$$

旋转坐标中的激活是横向拼接 $[A_l\ A_h]$，权重是纵向堆叠 $[B_l^\top\ B_h^\top]^\top$，所以输出为 $A_lB_l+A_hB_h$，可用两路同精度 GEMM 再相加。原文式2–3用加号连接不同形状的块，本页将其明确为拼接。若恢复至原坐标，$\widehat X=A_lU_l^\top+A_hU_h^\top$、$\widehat W=U_lB_l+U_hB_h$，乘积中的交叉项因 $U_l^\top U_h=0$ 消失。去掉量化才精确等于 $XW$；量化后通常只是近似。（§4.1、式2–4。）

通常取 $r=d/8$，两组总宽度仍为 $d$。相应张量的纯元素名义成本为 $(7/8)4+(1/8)8=4.5$ bit，而不是4 bit，也不是额外增加一个与原张量等大的残差。完整成本还需尺度、零点、基底、在线变换、对齐及未量化张量；`down_proj` 和 query 的例外使全模型平均位宽不能直接写成4.5。

## 2. PCA 能证明什么，原文证明缺在哪里

ResQ 构造 $U=P\operatorname{diag}(R_l,R_h)$，$P=[P_l\ P_h]$ 按特征值递增排列，末尾 $r$ 个方向为高精度组。$R_l$ 应为 $(d-r)\times(d-r)$，$R_h$ 为 $r\times r$；原文将它们写成与矩形 $P_l,P_h$ 同形状，不能直接代入块乘法。组内正交旋转保持所选子空间及其能量，跨组旋转则可能把高能量信号移回低精度组。（§4.2、式5。）

对当前 $X$，特征基底来自 $X^\top X\in\mathbb R^{d\times d}$。原文将其写成 $XX^\top$，后者是 token 维度的 Gram，不能直接提供 $d$ 维特征基底。固定实现累计未中心化的 $X^\top X$，并没有减去均值；因此其保护的是二阶能量，不能无条件称为中心化方差最大方向。PCA 的可验证结论是：$P_h^\top P_h=I_r$ 时，最大化 $\|XP_h\|_F^2$ 取最大的 $r$ 个特征值方向。

令 $E_l=XU_l-A_l$、$E_h=XU_h-A_h$，正确关系为

$$\|X-\widehat X\|_F^2=\|E_l\|_F^2+\|E_h\|_F^2.$$

原文式6及附录A把范数直接相加为等式；附录还把低组范数替换成总范数减高组范数，并把范数换为 trace 而未取平方根。两组范数为3和4时总范数为5，既不等于7，也不能令低组为 $5-4$。此外，$X$ 有 $n$ 行时两组元素数为 $n(d-r)$、$nr$，附录按 $d-r,r$ 计数需另有逐 token 定义；期望界也不能中途省掉期望而变为逐样本界。（式6、定理4.2、附录A、PDF第13页。）

因此本页不把定理4.2登记为已成立的实际量化最优性证明。可另行证明：在**明确设定**的平方能量代理 $a\|XP_l\|_F^2+b\|XP_h\|_F^2$ 中，若 $a>b\ge0$，保护最大能量子空间使该代理最小。这里的固定系数是假设，并非从实际动态范围、舍入、GPTQ或任务损失自动推出。随机旋转改善近高斯分布也是近似动机；有限维固定向量经 Haar 旋转后位于有界球面，不能直接说它严格服从高斯分布。

## 3. 四处变换有不同的融合条件

| 位置 | 作用与实施条件 |
| --- | --- |
| $U_A$，残差流/attention与FFN输入 | 共享隐藏特征基底；配套修改 embedding、head、输入与输出线性投影，并处理归一化增益。浮点等价条件见[旋转与残差](../theory/orthogonal-rotation-and-hadamard-quantization.md#3-归一化和残差为什么重要)。 |
| $U_B$，value头内 | 将V旋转并在输出投影抵消，可折入两侧权重；按每层/头统计选择基底，不能把它与全局 $U_A$ 混用。 |
| $U_C$，RoPE之后的Q/K | Q、K同乘正交矩阵才保持未量化点积；RoPE阻断随意向前融合，需在线投影。论文用8-bit投影与输入降低成本，量化后的变换不再保证精确正交或无误差。 |
| $U_D$，FFN中间激活至`down_proj` | 非线性阻断向前融合，激活在线做Hadamard，逆变换折入down权重。该层权重/激活统一低精度，未使用一般的高精度 $d_{FFN}/8$ 子空间。 |

这是推理布局的不同职责，不能因为两路 GEMM 可执行就省略在线变换或缓存处理。（§4.3、图4、附录C。）

## 4. 质量结果使用什么对象和配置

主配置标为 W/A/KV=4/4/4，但除 `down_proj` 外保留1/8通道为8 bit，query最终向量报告为16 bit。权重使用GPTQ与逐通道对称量化，激活为逐 token 非对称量化，KV为头内非对称量化。基底用512个WikiText校准样本，GPTQ用128个；附录报告单张A100 80GB生成基底与量化，70B评测使用4张卡。使用harness 0.4.5及LongBench；8项常识均分混合 `acc`/`acc_norm`，MMLU为0-shot，GSM8K为5-shot，不能把所有列当同一种准确率。（§5.1、附录C。）

| 表1模型 | FP16：PPL/常识均分/MMLU | SpinQuant | ResQ |
| --- | --- | --- | --- |
| Llama-3-8B | 6.1 / 67.1 / 63.1 | 7.4 / 63.8 / 56.2 | 7.1 / 63.9 / 57.2 |
| Llama-3-70B | 2.9 / 73.1 / 75.9 | 6.2 / 65.7 / 59.4 | 4.1 / 71.1 / 73.9 |
| Llama-3.2-3B | 7.8 / 62.7 / 54.8 | 9.2 / 57.9 / 44.2 | 8.8 / 59.0 / 49.8 |

例如8B相对SpinQuant的PPL下降约4.05%，但仍比FP16高约16.39%；70B的6.2→4.1下降约33.87%，MMLU提升14.5**个百分点**，并非14.5%相对提升。改善参照低精度基线不等于恢复浮点质量。

表2的8B GSM8K flexible/strict 分别33.6/33.2，对比SpinQuant29.8/29.6；但RepoBench62.3低于SpinQuant62.6，不能说所有生成任务都提升。表3的Qwen2-VL保留视觉编码器16 bit：4/4/4下2B/7B MMMU为29.7/47.0，FP16为39.6/51.6；4/8/4的34.0/48.8是另一预算，不能混算。表4将大幅离群保护与组内旋转组合为强基线，8B PPL7.2对ResQ7.1、3B9.0对8.8支持该条件下改善，不能替代最优性证明。

表5移除变换的8B PPL为：完整7.1，去 $U_D$ 1607、去 $U_A$ 37.4、去 $U_B$ 7.2、去 $U_C$ 7.3、同时去B/C 7.4。消融显示不同位置贡献不同，不能推广为B/C总是可省略。图6中8B校准样本效果在128之后趋稳，3B趋势不清且512最好，也不能统一采用“128足够”。

## 5. 性能证据止于单块和指定内核

§5.3、图5使用 CUDA 11.8、CUTLASS INT4/INT8 Tensor Core GEMM，在RTX3090、batch size 1、序列长度512与8192上测**单decoder block**。报告相对16-bit基线1.61–3.03倍加速，相对统一INT4平均慢14%。这些结果不等于整网自回归生成速度、在线KV服务吞吐或A100上的相同加速。

附录表6标题为A100 GPU hours，格中使用分钟/小时，注明SpinQuant使用4卡；8B列QuaRot17m、ResQ35m、SpinQuant1h41m。报告量化成本时应同时保留资源和单位说明，不能未经澄清就把表内时间另乘4并宣称墙钟或GPU总成本。

当前保存代码提供的是浮点模拟量化及模型评测路径，未包含上述CUTLASS部署实现。实现页说明已核的基底、量化器、缓存与配置接口；没有安装、运行模型、编译内核或测量性能。

## 来源身份

| 来源 | 固定身份 | 实际使用范围 |
| --- | --- | --- |
| ResQ: Mixed-Precision Quantization of Large Language Models with Low-Rank Residuals（`https://arxiv.org/abs/2412.14363v2`） | arXiv:2412.14363v2；PDF SHA-256 `e34e498a2868534e5dde3be2d699e1906ca91699fade45cdbc71a2f61b8d85c2` | 正文机制、表1–6/图5–6、必要附录A–C；附表未逐格重算 |
| utkarsh-dmx/project-resq（`https://github.com/utkarsh-dmx/project-resq/tree/cb87d0d059455858daaa66f7410da4afebee978a`） | `cb87d0d059455858daaa66f7410da4afebee978a` | 本页只用所选未中心化Gram；完整静态范围见实现页 |
