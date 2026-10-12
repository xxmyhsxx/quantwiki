---
title: VEQ：模态与专家亲和性加权的 MoE 量化
type: method
tags:
  - vlm
  - moe
  - post-training-quantization
  - weighted-reconstruction
sources:
  - raw/papers/2026-09-21/veq/paper.pdf
  - raw/papers/2026-09-21/veq/source/arxiv-main.tex
updated: 2026-10-12
---

# VEQ：模态与专家亲和性加权的 MoE 量化

VEQ在MoE视觉语言模型中，分别按模态路由次数加权专家重构误差，按token–expert亲和性加权二阶校准。论文把AWQ上的版本称为VEQ-ME、GPTQ上的版本称为VEQ-MA；实验分别列两种结果，没有登记将它们串成同一量化流程的组合结果。

本页依据arXiv:2602.01037v1的正文、式1–5、表1–3和图1–7，回查PDF1–8。论文提到的作者代码未进入本轮核查；梯度用于校准敏感性，PTQ免参数训练仍可需要反向传播。本轮没有运行模型、SGLang、量化或评测。

## 1. 专家频率和敏感性来自什么观察

同一MoE层的视觉与文本位置可能选择不同专家。图4在128个COCO样本上报告SFT损失梯度的文字/视觉平均比值22.4，图5的单样本约15；图6显示不同输入子集和模态的路由频率差异。原文未充分说明梯度对应张量、层间/样本归约及所有mask，因此不能把22.4固定为每层、每任务通用系数，或与[QIG](qig.md)的积分归因、[VLMQ](vlmq.md)的局部重构梯度混称。

作者在batch64、约3万token的校准观察中仍遇到零token专家。这提示有效样本应按专家统计；某次校准未路由到专家，不能证明部署时也不会使用它。图中高频或大梯度是该数据/目标的代理，尚未建立所有专家的因果贡献排序。路由和实际加权执行见[MoE执行](../implementation/moe-routing-and-quantized-expert-execution.md)。

## 2. VEQ-ME：专家分数要怎样进入搜索

对专家i，令N分别为路由到它的文本/视觉token数，T为校准集中的模态token总数。式1定义

$$S_i=\gamma N_i^{text}+\beta N_i^{vis},\qquad
\beta=\frac{T_{text}}{T_{vis}},\quad
\gamma=\frac{\|\nabla_{text}\|}{\|\nabla_{vis}\|}.$$

β补数量差、γ表征所选损失的梯度差。这些比值需要视觉token数与视觉梯度分母非零，原文未给缺模态退路。以Ttext=10、Tvis=100、γ=2为教学例：专家A收到8/20、B收到2/80个token，原始路由次数A=28、B=82，而S分别为18、12。加权可改变跨专家比较，但这不是新的位宽分配或推理router训练。

论文沿列token约定，Xi是当前专家收到的输入，式2–3为

$$L_0=\sum_i\|W_iX_i-\widehat W_iX_i\|_F^2,\qquad
L_{ME}=\sum_iS_i\|W_iX_i-\widehat W_iX_i\|_F^2.$$

L0本来已按有效token累计，不等于每位专家的均方误差先归一化后等权。Si是在该累计误差之外再次加权。**它改变搜索结果还需要参数存在跨专家耦合**：如果每位专家独立选择θi且Si>0，argmin SiLi(θi)=argmin Li(θi)；若共享尺度候选θ或有共同预算，权重才会改变取舍。教学例：共同候选U/V的专家误差为(1,4)/(4,0)，未加权选V，加上(18,12)后选U。原文没给足搜索变量与耦合粒度，匹配作者实现仍待核；不能仅凭式3承诺独立逐专家搜索会变化。

同乘正系数会整体缩放LME，理想精确搜索的排序不变，这解释图7a的相对比例观察；正则、阈值及数值实现需另核。零token专家的Xi为空、Si=0，目标没有给出它的量化质量约束。

## 3. VEQ-MA：逐token加权二阶矩

对某专家的输入 $X\in\mathbb R^{d\times N}$，式4–5给出

$$C=\operatorname{diag}(c_j),\quad c_j=p_j\alpha_j,\qquad
\alpha_j=\begin{cases}\gamma,&\text{text},\\1,&\text{vision},\end{cases}$$
$$\widetilde H=(X\sqrt C)(X\sqrt C)^\top=XCX^\top.$$

p是当前token对当前专家的亲和性，β没有进入这里。使用sqrt(c)缩放输入，外积才得到c；若直接乘c再作外积，实际权重变成c²。对加权平方重构 $L=\|(W-\widehat W)X\sqrt C\|_F^2$，整理者推导每输出行的Hessian为2XCXᵀ。论文在标准GPTQ段写2XXᵀ、增强式写XCXᵀ，按原式保留；实现中的归一化、damping和绝对阈值尚未核，不能直接宣称省略2在所有分支都无影响。

原文在贡献/消融段混用“logits”和“概率/信心”，没有给足softmax前后、top-k重归一化等定义。实数sqrt(C)和半正定曲率要求c≥0；若p实际是可负原始logit，必须先明确非负化规则。即使非负，少量/低权token仍可能使矩阵奇异；逆矩阵、未覆盖专家及正则策略需要匹配实现。这是公式适用条件，尚未观察到作者代码失败。

MoE输出通常含router加权和：若仅考虑固定router、单专家线性扰动，最终平方误差对应p²；多个专家还存在交叉项。因此论文使用p的矩阵是其校准代理，不能称为完整MoE最终任务损失的精确Hessian。有关误差目标与曲率的关系见[GPTQ](gptq.md)及[校准加权条件](../theory/calibration-and-range-selection.md#61-moe-校准须记录专家覆盖与权重位置)。

## 4. 质量报告、平均口径与消融

表1列Kimi-VL-Instruct、Qwen3-VL-30B-A3B-Instruct的W3/W4 weight-only结果，浮点参照为BF16；评测描述使用lmms-eval与SGLang。实际有9个任务列，正文却写“7个”，以下均分按表中9列保留，不把A16进一步指定为某一种实际激活dtype。

| 表1平均分 | BF16 | W3最佳基线 / VEQ-MA | W4最佳基线 / VEQ-MA |
| --- | ---: | ---: | ---: |
| Kimi-VL | 74.73 | 63.37（AWQ） / 65.41 | 72.96（GPTQ） / 73.57 |
| Qwen3-VL | 78.98 | 64.16（GPTQ） / 67.14 | 73.20（GPTQ） / 74.36 |

W3均分差分别为2.04、**2.98个百分点**；摘要与表内标注把Qwen增益写成3.09，与所列均分不一致，不能抹去该差异。W4差为0.61/1.16个百分点。表1的Qwen W4 RTN九项算术均分为71.01，Avg却列70.89；这一差异不改变所述最佳基线。VEQ-MA仍低于BF16；Qwen W3的InfoVQA 47.14低于GPTQ 47.27，MME-R 43.25低于RTN 45.73，均分优势不等于每任务优势。VEQ-ME的Qwen W3均分63.93也低于GPTQ 64.16，不能让框架名称替代具体版本。

Kimi W3 TextVQA中GPTQ64.30→MA78.30是14.00个百分点、约21.77%相对增幅。正文21.4%未明确对照与口径，不能直接当成百分点，也未从该对照精确恢复。W3的MA与BF16均分差为9.32/11.84个百分点；bit-width节省仍需另计元数据、非量化参数、KV及工作内存。

表2/3只取MMMU、InfoVQA、ScienceQA三项：ME完整63.27，去γ/β为62.02/62.30；MA完整63.96，去p/α为63.63/62.46。表3去p行的42.33/64.30/83.35算术均分为63.33，和表内Avg63.63不同，二者都低于完整配置。上述是两个版本各自的组件消融，三任务均分不能直接与表1九任务均分相比。图7使用MMMU validation随机64样本的PPL搜索：MA将亲和性改为 $(1-\lambda)\mathbf1+\lambda\mathbf p$，λ=0为均匀、1为原p，最低所报PPL为2.2526。它检验所选代理/网格；λ在[0,1]且p非负时保持非负，论文未完整登记PPL标签/归约及γ0数值，不能由图补齐可复现配置。

## 5. 复用前须补齐哪些协议

原文没有充分登记主量化校准来源/样本ID、mask、各专家覆盖、γ统计粒度、ME共同搜索变量、p变换、权重组/零点/damping、模型及软件pin、具体benchmark split/模板/生成与计分设置。COCO128是动机分析、MMMU64是超参数验证，不能自动当作主校准协议；作者所称固定校准/评测条件亦不补出具体配置。没有给量化耗时、内存、吞吐或尾延迟表，不能把SGLang使用记录转成部署加速证据。

当前可复用的是两个加权位置及其条件：跨专家目标需要有效耦合，专家内曲率需要正确的非负token权重。它与[MBQ](mbq.md)的模态目标、[MASQuant](masquant.md)的尺度/补偿分别解决不同维度；分布变化后应重新核专家覆盖和任务敏感性。

## 来源身份

VEQ: Modality-Adaptive Quantization for MoE Vision-Language Models，来源 `https://arxiv.org/abs/2602.01037v1`，arXiv:2602.01037v1；正文方法/实验、式1–5、表1–3和图1–7，PDF1–8。参考文献只用于定位关系，未审计全部引用或逐点重建热图；未取得匹配作者实现，未运行模型、量化、SGLang或质量/性能评测。
