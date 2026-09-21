---
title: SpinQuant：用最终模型损失学习正交旋转
type: method
tags:
  - ptq
  - llm
  - rotation
  - optimization
sources:
  - raw/repositories/2026-09-21/spinquant/source/scripts/10_optimize_rotation.sh
  - raw/repositories/2026-09-21/spinquant/source/scripts/2_eval_ptq.sh
  - raw/papers/2026-09-21/spinquant/paper.pdf
  - raw/papers/2026-09-21/quarot/paper.pdf
  - raw/papers/2026-09-21/flatquant/paper.pdf
  - raw/papers/2026-09-21/cayley-stiefel/paper.pdf
  - raw/repositories/2026-09-21/spinquant/source/README.md
  - raw/repositories/2026-09-21/spinquant/source/optimize_rotation.py
  - raw/repositories/2026-09-21/spinquant/source/train_utils/optimizer.py
  - raw/repositories/2026-09-21/spinquant/source/train_utils/main.py
  - raw/repositories/2026-09-21/spinquant/source/train_utils/quant_linear.py
  - raw/repositories/2026-09-21/spinquant/source/train_utils/modeling_llama_quant.py
  - raw/repositories/2026-09-21/spinquant/source/train_utils/apply_r3_r4.py
  - raw/repositories/2026-09-21/spinquant/source/eval_utils/rotation_utils.py
  - raw/repositories/2026-09-21/spinquant/source/utils/data_utils.py
  - raw/repositories/2026-09-21/spinquant/source/utils/quant_utils.py
  - raw/repositories/2026-09-21/spinquant/source/utils/process_args.py
updated: 2026-09-15
---

# SpinQuant：用最终模型损失学习正交旋转

SpinQuant 的出发点是：**不同正交旋转在浮点下等价，在量化后却可能相差很大。**它冻结预训练权重，用最终语言模型的交叉熵学习可离线融合的残差与 Value 旋转，通过 Cayley 更新保持正交。主配置先让权重保持 16 位、模拟低比特激活/cache 来学习旋转，最后才用 GPTQ 量化权重。在线 Hadamard 保持固定，并非所有旋转都训练。

依据 arXiv 2405.16406v4（2025-02-20，ICLR 2025，24 页含附录）；代码为 `8f47aa3f00e8662caf1a484153920a07e5281c3a`。代码是作者公开的 Hugging Face 实现，README 明确论文主要实验使用内部实现。本文区分论文设置、公开代码和教学推导；没有运行模型或性能实验。

## 1. 它要改变什么，保留什么

[QuaRot](quarot.md) 已说明全局残差、FFN 和注意力中的正交等价；SpinQuant 保留这些结构约束，重点改变其中可学习的坐标系选择。图 4 的多次随机旋转试验展示最高约 13 个百分点的零样本平均差异，说明“正交”不是量化质量充分条件；这不是对所有模型、所有位宽都存在同样差距的保证。

研究对象仍以 LLaMA 类模型为主，附录扩展至 Mistral、Qwen 和部分指令模型。$d$ 为残差宽度，$d_h$ 为头宽，$L$ 为层数。主要训练变量是一个全局 $R_1\in\mathbb R^{d\times d}$，以及每层一个在该层各头共享的 $R_{2,\ell}\in\mathbb R^{d_h\times d_h}$，均满足 $R^\mathsf TR=I$。这类论文中的 rotation 包含一般实正交变换，不额外要求行列式为正。

## 2. 四种旋转的位置与等价关系

采用行 token 和 PyTorch 权重 $W[out,in]$ 的约定；[旋转机制页](../theory/orthogonal-rotation-and-hadamard-quantization.md) 展开 RMSNorm、残差和快速 Hadamard 的前提。下表对应论文 §3、图 2–3。

| 旋转 | 位置与作用 | 是否学习、是否在线 |
|---|---|---|
| $R_1$ | 全局残差坐标；embedding、Norm 后输入投影、各块输出和最终 head 配套变化 | 学习；校准结束后折叠进权重，推理不需显式全宽矩阵乘法 |
| $R_2$ | 每层各头的 V 特征坐标，与 O 投影头内逆变换配对 | 学习；两端均可离线折叠 |
| $R_3$ | RoPE 后 Q/K 使用同一头内 Hadamard | 固定；had 配置需在线，不能一般地跨过 RoPE 融合 |
| $R_4$ | FFN 门控输出至 down projection 的 Hadamard | 固定；激活侧在线，逆变换折叠进 down 权重 |

对无仿射 RMS 归一化 $N$，$N(xR_1)=N(x)R_1$；Norm 增益先融合到权重。输入投影保存 $W_{in}R_1$，输出保存 $R_1^\mathsf TW_{out}$，使块输入输出都处于同一残差坐标。FFN 非线性输入保持不变，其输出再通过 down 投影回到旋转残差。这与“让任意旋转穿过 SiLU”不同。

对一层中的某个头，设概率矩阵为 $S$、value 为 $V$：

$$S(VR_2)=(SV)R_2.$$

若将所有头按相同布局拼接，$B_2=I_h\otimes R_2$，V 投影输出变为 $VB_2$，O 权重保存 $R_1^\mathsf TW_oB_2$，得到

$$ (AB_2)(R_1^\mathsf TW_oB_2)^\mathsf T=AW_o^\mathsf TR_1.$$

这里 $A$ 是各头 attention 输出的拼接。**SpinQuant 的这一步只旋转头内坐标**，不需要 QuaRot 为完整 O 输入 Hadamard 额外执行的跨头在线混合。（`eval_utils/rotation_utils.py:rotate_ov_proj`；GQA 的重复头映射仍需匹配。）

对 RoPE 后 $Q_p,K_p$，有 $(Q_pR_3)(K_pR_3)^\mathsf T=Q_pK_p^\mathsf T$。对门控结果 $z$，有 $zW_d^\mathsf TR_1=(zR_4)(R_1^\mathsf TW_dR_4)^\mathsf T$。R3/R4 保留在线成本，原因分别是位置编码与门控非线性的边界。

论文的 `no_had` 只保留可离线的 R1/R2；`had` 加入 R3/R4。表 1 中 W4A4 通常依赖后者控制激活误差，但 W4A8 下未必值得支付在线成本；不要把 `no_had` 理解成没有任何旋转。

## 3. 学习目标、梯度与分阶段量化

### 优化最终语言模型损失

令 $W$ 是冻结的预训练权重，$\mathcal R=\{R_1,R_{2,1},\ldots,R_{2,L}\}$，$f_q$ 是含模拟量化的整个网络。用校准文本的下一个 token 作为标签：

$$\min_{\mathcal R}\ \mathbb E_{x\sim\mathcal D}
\left[-\sum_{t\in\mathcal V(x)}\log p_{f_q(W,\mathcal R)}(x_{t+1}\mid x_{\le t})\right],
\qquad R_i^\mathsf TR_i=I.$$

$\mathcal V(x)$ 表示有效预测位置；实际实现按有效 token 归约损失。这里不需要浮点教师输出作 MSE 目标，不学习原始 $W$。`modeling_llama_quant.py:LlamaForCausalLM.forward` 将 logits 和 labels 移位后调用 `CrossEntropyLoss`；`optimize_rotation.py` 先冻结参数，再建立可学习 R1/R2。（§3.2、附录 B.1。）

如果没有量化且严格满足上述等价关系，改变旋转不会改变 logits，也就没有改善原模型任务损失的空间。量化让不同旋转对应不同舍入/裁剪误差，才使最终损失依赖 $\mathcal R$。这个事实也给出实现检查：关闭量化后仍有显著 loss 变化，应先排查等价图和数值精度。

离散舍入不能直接提供通常意义的可用梯度。公开 `utils/quant_utils.py:STEQuantize/AsymSTEQuantize` 前向执行舍入、裁剪和反量化，反向把输入梯度原样传回，scale/zero 不返回梯度。这是 [STE 代理梯度](../fundamentals/quantization/post-training-and-quantization-aware-training.md)，不是舍入的真实导数；因此训练曲线和最终验证仍必要，也不能用光滑优化理论直接保证这套离散目标的全局最优。

旋转参数量为 $d^2+Ld_h^2$。例如 $d=4096,L=32,d_h=128$ 时为 17,301,504 个参数；这是教学计数，**不意味着训练显存也只占模型参数的同等比例**。梯度仍须通过整个网络传播，激活保存、优化器和分布式策略都影响成本。

### 为什么先 W16A4，再 W4A4

论文主流程是在 W16A4（cache 按目标设置为 4/16 位）的模拟网络中学习旋转，再固定旋转、融合权重，最后运行 [GPTQ 的二阶权重量化](gptq.md)。学习阶段若使用 RTN 权重噪声，旋转可能适应这一噪声；之后换 GPTQ，最终扰动发生变化。这是作者给出的阶段设计动机，而非“权重误差不重要”。（§4.2、表 3。）

表 3 的 LLaMA-2-7B 最终均为 W4A4KV4：用 W4A4 学习的结果为平均 60.9、PPL 6.8；用 W16A4 学习后再 GPTQ 为 64.0、PPL 5.9。该证据支持在其流程中分开处理两类误差，不证明所有量化器都应永远分阶段，也不把两阶段写成联合优化 GPTQ 内部离散决策。

## 4. Cayley 更新怎样保持正交

直接 $R\leftarrow R-\eta G$ 通常破坏 $R^\mathsf TR=I$，继而破坏 RMS 与逆转置等价。对本方法的方阵情况，下面以**下降方向**统一符号，暂忽略动量：令 $G=\partial\mathcal L/\partial R$、$D=-G$，构造

$$Y=\tfrac12(DR^\mathsf T-RD^\mathsf T),\qquad Y^\mathsf T=-Y,$$
$$R_+=(I-\eta Y/2)^{-1}(I+\eta Y/2)R.$$

乘在 $R$ 左侧的 Cayley 矩阵是正交的，因此精确运算下 $R_+^\mathsf TR_+=I$。小步展开为 $R_+=R+\eta YR+O(\eta^2)$，其中 $YR$ 为负梯度在正交约束切空间中的投影；学习的是沿约束允许方向的变化。符号若采用正梯度定义，则需要相应反转更新方向，不能仅照抄一个正号。（SpinQuant §3.2；Li 等 2020 §3.2、§4.1 算法 1；方阵简化与展开为教学推导。）

显式求逆成本高，可以改写成固定点迭代：

$$Z_0=R+\eta YR,\qquad Z_{k+1}=R+\frac\eta2Y(R+Z_k).$$

在 $\eta\|Y\|/2<1$ 的相容范数条件下，它向上述 Cayley 解收敛。**有限次迭代只是近似，不能宣称每步严格机器精度正交。**通用切空间、保持正交的证明和与 Cayley 参数化的区别见 [矩阵基础中的约束更新](../fundamentals/mathematics/invertible-transforms-and-kronecker-products.md)。

公开 `train_utils/optimizer.py:SGDG.step/Cayley_loop` 使用负梯度进入动量，转置到其内部布局后构造反对称矩阵，限制步长，执行 5 次固定点迭代；另有单位化和概率性 QR 校正。单位化本身不保证列间正交，核对长期漂移仍应测 $\|R^\mathsf TR-I\|$。本轮没有运行该优化器的模型训练。

## 5. 完整流程与论文、公开实现的边界

1. 准备预训练模型及校准文本，先融合 Norm 增益并建立旋转计算图；用随机 Hadamard 初始化 R1/R2，按 `had` 或 `no_had` 确定 R3/R4。
2. 冻结原模型权重，按学习阶段位宽配置模拟量化。前向得到最终 next-token CE，反向只更新 R1/R2，并保持正交约束。
3. 达到预定步数后保存旋转参数；将它们融合到 embedding、线性权重与 head。旋转 checkpoint 本身不是已经打包的四位模型。
4. 使用单独的校准过程对旋转后的权重做 GPTQ，设置动态激活和 cache 网格，评测最终组合；随后才转换到目标部署后端支持的格式。

论文 §4/附录 A 的主设置使用 800 个校准样本、100 步，学习率从 1.5 线性降至 0；GPTQ 使用 128 段、每段 2048 token。作者报告 7B 约 25 分钟、8B 约 30 分钟、70B 约 3.5 小时；该处没有充分给出 GPU 型号与数量，不能用公开脚本的 GPU 数补作这些耗时的条件。

激活采用动态非对称网格，主设置不裁剪 A/K/V；权重采用 GPTQ 及其裁剪搜索。公开配置中普通线性输入为 per-token，O 投影输入与 KV 可按 head 量化，权重默认对称、每输出通道；分组和对称性必须由实际参数确认，不从一个 W4A4KV4 标签反推出全部细节。（附录表 12；`train_utils/main.py`、`utils/process_args.py`。）

公开实现有几个会影响复现的区别：README 的示例允许不同训练位宽，后文 Note 才指出论文采用 W16 学旋转再 W4 GPTQ；训练脚本使用 cosine 调度，而论文描述 linear；`CustomJsonDataset` 拼接文本并分块，不单凭该类保证正好抽出论文的 800 个样本。优化入口会保存独立的旋转矩阵，评测入口再加载、融合和量化。公开仓库的 fake quant、导出配置与论文内部低比特后端也不是同一产物，不能把脚本能启动当作论文复现完成。

## 6. 消融、质量与失败边界

除另行标注，下面是 SpinQuant 论文内的比较，PPL 为 WikiText-2，平均零样本准确率采用 **8 项任务**（BoolQ、PIQA、SIQA、HellaSwag、WinoGrande、ARC-e、ARC-c、OBQA）。QuaRot 原文六任务平均不能直接放在同列排序。

| 问题 | 证据 | 解释 |
|---|---|---|
| 学习旋转是否有必要 | 表 2，Mistral-7B W4A4KV4，随机 R1–R4 为 52.4，学习 R1/R2 后 68.6；LLaMA-3-8B 对应 63.9 → 65.5 | 学习收益依模型而异，不固定为十几个百分点 |
| 可否完全去掉在线 Hadamard | 表 1，LLaMA-2-7B W4A4KV4：no_had 56.0/PPL 9.2，had 64.0/5.9；W4A8KV8 两者平均都为 65.8 | W4A4 的在线代价有质量理由，A8 时未必划算 |
| 初始化的作用 | 表 4，W4A4KV4、RTN 流程中，未优化的随机正交与随机 Hadamard 平均为 48.3 与 58.7；优化后都约 61.5 | Hadamard 起点较好，但学习可缩小差距；不表示最终永远与初始化无关 |
| 样本和步数如何选择 | 表 11，在所测设置中 128 与 800 样本 PPL 均约 6.2；10/25/50/100/200 步约 6.6/6.4/6.3/6.2/6.2 | 有收益饱和迹象，不应把 800 或 100 机械当作所有模型的必要最优值 |
| GPTQ 是否所有指标必增 | 表 16，LLaMA-2-7B W4A4KV16 从 RTN 的 61.8 到 GPTQ 的 64.1；70B 对应 71.1 与 71.0 | 权重补偿通常有价值，但不存在每任务或每平均必增的定理 |
| 与固定旋转如何比较 | 表 5，LLaMA-3-8B W4A4KV4：QuaRot+GPTQ 63.3/PPL 8.0，SpinQuant 65.5/7.3，FP16 69.6/6.1 | 同篇同表支持学习旋转的增益，也保留相对 FP16 的剩余损失 |

表 5 的 LLaMA-3-70B QuaRot 基线 PPL 20.2，与 QuaRot v2 自报 6.66 差别很大。这里尚未完成版本、旋转、量化参数与评测协议的逐项对齐，因此不拿它推导跨论文统一领先幅度，也不能武断归因为某一个种子。优先使用各论文内部的受控消融解释机制。

附录 B.2 的 SNR 定义为 $10\log_{10}(\|X\|_F^2/\|X-\hat X\|_F^2)$，用于分析输出信号与误差的比例。表 19 在 LLaMA-2-7B、W4A4、WikiText-2 测试集上的端到端平均 SNR，由无旋转 −2.9 dB，经随机旋转 0.9 dB 到学习旋转 6.8 dB；但图 7 同时显示部分层的指标可能变差。**最终 CE 改善不要求每层局部 MSE 都下降**；SNR 也不是实际优化目标。不能把包络变平或局部误差下降与最终任务改善混成一个目标。（附录 B.1–B.2、C，图 7–11。）

校准数据 C4/WikiText 的比较、weight-only 和指令模型试验提供有限的迁移证据；它们没有验证视觉 token、模态损失平衡或本项目的当前研究设想。它们也不消除 [校准集与独立评测集分工](../theory/calibration-and-range-selection.md) 的要求。

## 7. 学到的旋转免费，整套推理并不免费

R1/R2 能在校准完成后融合，所以从固定旋转换成学习旋转，不必新增这两类稠密在线乘法；`had` 仍有 R3/R4 的开销。作者的两个后端测量说明这一成本受格式和场景影响，不能混为 W4A4KV4 的统一速度。（§4.4、附录 A.6。）

- 表 6：macOS 14.5、M1 Pro CPU、LLaMA-3-8B，FP16 为 177.15 ms/token，W4A8 no_had 为 58.88，had 为 63.90。加入在线 Hadamard 比 no_had 慢约 8.5%；该结果是移动端 W4A8，不是四位激活 GPU 推理。
- 表 14：H100、LLaMA-3-70B、**FP8 权重/激活**。序列长 4096、batch 1，no_had TTFT/TTIT 为 153.58/9.85 ms，had 为 158.25/10.15 ms，约增加 3%。该后端利用 Tensor Core Hadamard 与 FBGEMM 融合；表内没有 FP16 基线，不能只凭它计算低比特相对 FP16 的加速。

TTFT 是首 token 延迟，TTIT 是后续 token 间隔。低比特导出和真实内核的适配还取决于对称性、group size、cache 和布局；[执行页](../implementation/quantized-matmul-scaling-execution.md) 解释为什么模拟量化 PPL 与打包后的端到端速度需要分开验证。

## 8. Strong / weak，以及它和 FlatQuant 的关系

**Strong：**在保留正交数值条件与可融合图结构的同时，用整个网络的最终损失选择表示；能够处理“随机旋转很好，但选哪个差异很大”的实际问题；原权重冻结，R1/R2 的学习结果可离线消费；W16 学习后 GPTQ 的消融明确说明了阶段设计。

**Weak：**学习仍需整网反向与数据，参数少不等于内存小；STE 与近似 Cayley 更新要验证；R3/R4 没有随任务学习，W4A4 常仍依赖在线变换；正交限制不能自由拉伸通道；最终质量仍有损失，公开代码/内部实验和跨论文基线仍有协议差异。

| 比较维度 | QuaRot | SpinQuant | FlatQuant |
|---|---|---|---|
| 选择变换的目标 | 固定随机旋转；随后可用 GPTQ 量化权重 | 整个量化网络的 next-token CE | block 输出重构 MSE，并学习尺度/裁剪等参数 |
| 变换空间与位置 | 正交，全局残差与局部固定 Hadamard | 可学习正交 R1/R2，固定在线 R3/R4 | 局部可逆变换，含 Kronecker 结构与可学习拉伸 |
| 成本关键 | 剩余在线 Hadamard 与低比特后端 | 整网反向校准；推理时 R1/R2 融合 | 局部在线变换与量化的融合效率 |

这个比较不能简化成谁的矩阵空间更大谁就更好。正交条件让 SpinQuant 的全局 R1 能跨 RMSNorm 工作；[FlatQuant](flatquant.md) 的一般可逆变换作用于不同局部配对位置，不能直接拿来替换 R1。二者的任务目标、图位置与部署策略共同变化，研究选择需要将这些因素分别对照，而不是仅比较论文排名。（SpinQuant §3–4；QuaRot §4；FlatQuant §3、附录 B。）

## 来源身份

下表用于在没有本地资料库时辨识来源；具体论述的章节、公式、图表或代码位置见正文。

| 来源 | 版本或快照 | 说明 |
| --- | --- | --- |
| [facebookresearch/SpinQuant](https://github.com/facebookresearch/SpinQuant/tree/8f47aa3f00e8662caf1a484153920a07e5281c3a) | `8f47aa3f00e8662caf1a484153920a07e5281c3a` | — |
| [SpinQuant: LLM quantization with learned rotations](https://arxiv.org/abs/2405.16406v4) | `arXiv:2405.16406v4` | — |
| [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456v2) | `arXiv:2404.00456v2` | — |
| [FlatQuant: Flatness Matters for LLM Quantization](https://arxiv.org/abs/2410.09426v4) | `arXiv:2410.09426v4` | — |
| [Efficient Riemannian Optimization on the Stiefel Manifold via the Cayley Transform](https://arxiv.org/abs/2002.01113v1) | `arXiv:2002.01113v1` | — |
