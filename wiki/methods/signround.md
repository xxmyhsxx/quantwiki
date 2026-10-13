---
title: SignRound：舍入偏移、裁剪与符号梯度搜索
type: method
tags:
  - ptq
  - rounding
  - clipping
  - reconstruction
sources:
  - raw/papers/2026-09-22/signround
updated: 2026-10-13
---

# SignRound：舍入偏移、裁剪与符号梯度搜索

SignRound 固定预训练权重，逐 Transformer block 优化舍入前的加法偏移与裁剪端点系数，以块输出 MSE 为目标，用 SignSGD 的梯度符号更新参数。它利用同一整数码对应一段连续参数区间的特点，控制有限步搜索的移动量；这不保证离散目标单调下降，也不保证联合裁剪始终改善任务质量。

本页依据 Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs，arXiv:2309.05516v5。已核正文、算法1、主要消融及附录A的时间条件，编号回查固定19页PDF；未核作者仓库或运行模型。该版本给出的代码入口为 AutoRound，本页不把其后续版本能力并入论文证据。

## 1. 学习舍入偏移与裁剪，保留什么格式

原文式3在最近舍入的输入中加入与权重同形的参数 $V$，式4让两个端点系数 $\alpha,\beta\in[0,1]$ 改变量化步长。对当前待量化权重集合，记极值为 $w_{min},w_{max}$，位宽为 $b$，则

$$
s=\frac{\alpha w_{max}-\beta w_{min}}{2^b-1}.
$$

$V$ 决定权重跨过哪一个舍入阈值；$\alpha,\beta$ 改变范围和共享网格。初始化 $V=0,\alpha=\beta=1$，从最近舍入及未收缩的端点出发。算法1保存的是这些参数，不是直接把原文式6的通用优化变量 $W_t$ 当作基座权重进行任务微调。

**原式与仿射恢复的差异。** 固定v5的式1、式3都没有显式减回零点 $zp$；式1不含$V$，式3加入偏移后写成 $s\operatorname{clip}(\operatorname{round}(W/s+zp+V),n,m)$。沿[均匀量化的仿射格式](../fundamentals/quantization/uniform-quantization-and-groups.md)整理，一个自洽的 unsigned $b$-bit 表达是

$$
q=\operatorname{clip}\left(\operatorname{round}(W/s+z+V),0,2^b-1\right),
\qquad \widehat W=s(q-z).
$$

这里 $z$ 是整数零点，运算按对应范围广播。这是为解释而补全的标准恢复式，不是原文式3的逐字转写，也不表示作者实现已核。$V$在最终选码后可以消去，恢复时需要整数码、步长与零点；它不增加逐权重推理尺度。

若 $w_{min}\le0\le w_{max}$ 且收缩后上下端点仍不同，正系数可缩小范围；但 $\alpha,\beta\in[0,1]$ 单独不能保证任意全正/全负组的 $s>0$，两系数全零也会使范围退化。极值是否先包含零、零点怎样求取和更新、退化保护、裁剪系数的共享轴及尾组处理，当前论文没有给出足够实现细节。

## 2. 为什么按梯度符号跨越舍入阈值

先固定 $s,z$，考虑未饱和的坐标 $w/s+z=k+f$，$k\in\mathbb Z,0<f<1$。忽略中点tie规则，整数码从 $k$切换到$k+1$的阈值是

$$V=\frac12-f.$$

同一码对应一段 $V$ 区间，不要求连续参数精确收敛到唯一点。**教学例子：**若 $w/s+z=6.2$，则 $V=0.29$ 得到6，$V=0.31$ 得到7；跨阈值比区间内再缩小微小数值距离更直接地改变硬前向。论文用 $V\in[-0.5,0.5]$ 讨论相邻选码搜索，联合裁剪又会移动网格，不能把整个方法永久解释成固定网格的二选一。（§3，式3–6。）

用 $\theta$ 统一表示实际学习的 $V,\alpha,\beta$，式6的更新规则可改记为

$$\theta_{t+1}=\theta_t-\eta_t\operatorname{sign}(g_t).$$

$g_t$ 是对当前优化变量的梯度，$\eta_t$ 是步长；sign保留方向并舍弃幅度。默认每块200步、初始学习率0.005、线性学习率衰减，论文以

$$\sum_t\eta_t\approx\frac{200\times0.005}{2}=0.5$$

解释累计移动预算。它给出绝对位移的上界预算，不表示每个参数都移动0.5；符号反转会抵消净位移。原文据此讨论从0出发的舍入搜索范围，以及从1出发、截到上界1的裁剪系数约在$[0.5,1]$内移动。具体离散调度、投影和边界更新未在算法1中展开，近似预算也不能当作严格区间保证；改为1000步或不同学习率后须重新核算。

§3及§4.4两处措辞写作“linear weight decay”，而§4.1明确写线性学习率衰减。上面的预算解释依据§4.1配置，不额外推定存在某个权重衰减项。

## 3. 代理梯度与 SignSGD 是两层选择

硬round对$V$的真实导数几乎处处为零。原文算法1只有 `loss.backward()`，未给出STE、饱和区或零点反向的定义；因此不能把一个惯用代理规则称为本次已核作者实现。[PTQ、QAT与代理梯度](../fundamentals/quantization/post-training-and-quantization-aware-training.md)解释这种差别。

**条件性的整理者展开：**若固定正步长与零点、未触发裁剪，并对round使用恒等STE，则标准恢复式给出

$$
\frac{\widetilde\partial L}{\partial V_{ij}}
=s\frac{\partial L}{\partial\widehat W_{ij}}.
$$

正$s$不改变该元素梯度的符号。STE负责提供离散节点的代理反向，SignSGD负责如何用已得到的梯度更新；对梯度再取sign不会让硬round成为可微函数。裁剪系数还经过共享范围、尺度及可能的零点路径，不能把上式换名当作$\alpha,\beta$的梯度。

纯sign更新在精确算术下满足$\operatorname{sign}(cg)=\operatorname{sign}(g)$（同一正标量$c$），所以整项损失等比例缩放不会改变方向。但不同位置/分组加权、额外正则和有限精度可改变梯度，不能据此宣称所有重构损失尺度都无关。公共[重构损失尺度](../theory/layer-reconstruction-second-order-compensation.md#11.%20%E9%87%8D%E6%9E%84%E6%8D%9F%E5%A4%B1%E7%9A%84%E5%B0%BA%E5%BA%A6%E6%80%8E%E6%A0%B7%E5%BD%B1%E5%93%8D%E4%BC%98%E5%8C%96)解释目标归约与有限步优化的条件。

这与[AdaRound](adaround.md)的软舍入及显式正则、[FlexRound](flexround.md)的逐元素除法参数化不同；[GPTQ](gptq.md)则使用输入Gram和顺序解析补偿。共同使用校准输入不意味着同一种更新规则或二阶保证。

## 4. 块输出目标、输入路径与候选保留

原文式5用$\|WX-\widetilde WX\|_F^2$表示重构，算法1实际比较整个模块的输出。设原块为$m_W$、量化块为$m_{\widehat W}$，当前batch输入为$x$，应按模块说明

$$L=\operatorname{MSE}\big(m_{\widehat W}(x),m_W(x)\big).$$

Transformer block含attention、非线性和残差，不能用单个$WX$替代完整前向。算法1在两支使用同一个$x$：取样、得到块输入和参考输出、量化/反量化权重、计算量化块输出、反向；在更新前比较当前loss，保存较低loss对应的$V,\alpha,\beta$，再做sign更新。

每步抽取batch，因此 `best_loss` 是已观察的当前batch损失最小候选，不是在固定独立验证集上证明全局最优。实现还需正确快照参数、管理梯度与边界；这些不能由简短伪代码推定已运行无误。

作者的 `ours*` 搜索包括200/1000步、裁剪学习率$1/T$或$2/T$、是否使用前一量化块的输出，共八种选择；70B只测试少数项。量化输入选项可改变$x$的来源，算法1仍让两支接收同一当前$x$，不能直接改写成AdaRound独立浮点/量化两路输入。不同参考路径如何影响目标，见[非对称输入](../theory/layer-reconstruction-second-order-compensation.md#7.%20%E5%8A%A0%E6%9D%83%E9%87%8D%E6%9E%84%E4%B8%8E%E9%9D%9E%E5%AF%B9%E7%A7%B0%E8%BE%93%E5%85%A5)。

## 5. 实验证据及反例

### 默认配置的收益和代价

§4.1量化Transformer块内Linear，保留embedding和通常的lm-head，激活仍为浮点；采用非对称权重格式，测试W4G−1、W4G128、W3G128、W2G128。G128表示128权重一组，G−1为不进一步分组的标签；实际共享轴与布局未核代码。校准从pile-10k按相同seed随机取512样本、长度2048；SignRound默认每块200步、batch 8、AMP。

主指标是11项zero-shot任务准确率的平均：HellaSwag、WinoGrande、PIQA、LAMBADA、TruthfulQA、OpenBookQA、BoolQ、RTE、ARC-Easy、ARC-Challenge、MMLU；PPL另作补充。它是这组任务的汇总，不能表示任意生成质量均改善。

表1的Mistral-7B W2G128：默认SignRound为52.71%，GPTQ为39.61%，16位参考为63.30%；相对GPTQ提高13.10个百分点，仍比参考低10.59个百分点。GPTQ比较开启act-order、true-sequential，分组时开启static_group；OmniQuant对照关闭LET、运行20个epoch，70B因NaN/成本使用128样本而非512，不能当成所有方法完全同配置的排行榜。（§4.2。）

### 联合裁剪与更多迭代都可能退步

表5，LLaMA-2、W2G128、上述默认数据/步数，11任务平均准确率：

| 可学习组件 | 7B / % | 13B / % |
| --- | ---: | ---: |
| RTN | 29.94 | 33.51 |
| 仅裁剪 | 40.53 | 49.77 |
| 仅舍入偏移 | 49.14 | 54.41 |
| 舍入与裁剪联合 | 48.64 | 53.46 |

两类变量各能改善RTN，联合却分别比仅舍入低0.50、0.95个百分点。表6的LLaMA-2-7B W4G−1从默认200步57.48%到1000步57.19%，也说明增加优化自由度或预算不能直接推出更好的下游质量。

表4在8个Adam初始学习率（0.0025–0.02）的已测范围内，LLaMA-2-7B W4G−1的最好结果为57.45%，SignSGD为57.48%；小差距不证明优化器普遍优越。表3的同模型W4G−1，FlexRound5000步57.49%，也略高于默认SignRound200步57.48%。这些是该论文适配实现的结果，不改写各方法原论文的实验；不能把“200步胜过5000步”写成无例外结论。

### 泛化与耗时要保留额外条件

表7的七个后续模型使用W4G128、1000步及模型级超参选择，评测采用harness 0.4.0+和作者称为实际量化的模型。其Variation列是相对变化：Mixtral-8x7B从66.98%到66.33%，下降0.65个百分点，约为0.97%的相对下降。不能把这些结果归为默认200步协议，也不表示本项目核验了部署实现。

附录A表8报告量化/调参成本：单张A100 80GB、W4G−1，GPU独占但CPU等可能共享；SignRound关闭 `low_gpu_mem_usage` 以提高速度、增加内存占用。LLaMA-2-70B为9116秒，约2.53小时；HQQ为119秒，OmniQuant使用128校准样本而其余为512。这是离线量化成本，不是推理时延，也不支持“最快量化”的概括。

附录表9的舍入方法耗时caption写200步，§4.3引用该表的正文却描述FlexRound/AdaRound的5000步设置，且说明其适配未支持AMP等优化。当前材料无法把该表所有时间对应到同一明确协议；保留冲突，不把耗时差直接归因于sign更新。

## 6. 已核范围与实现缺口

本页讲清论文参数化、阈值搜索、sign更新、块目标和主要证据；作者实现中的零点/退化范围、代理反向、投影、参数共享、best候选快照、导出/packing/kernel尚未核查，全部附表未逐项核验，没有模型或性能复现。优化变量固化进整数码的解释不等于任意后端都能获得同一质量或吞吐。

## 来源身份

Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs（`https://arxiv.org/abs/2309.05516v5`），arXiv:2309.05516v5；正文§3–5、式1–6、算法1、表1/3–7，附录A及表8–9。教学阈值、标准恢复式与条件性梯度展开已在对应位置区分。
