---
title: DeepSeek-V3的FP8训练：GEMM、缓存与通信
type: implementation
tags:
  - fp8
  - mixed-precision
  - moe
  - training
  - tensor-core
sources:
  - raw/papers/2026-09-21/deepseek-v3
updated: 2026-10-14
---

# DeepSeek-V3的FP8训练：GEMM、缓存与通信

DeepSeek-V3报告v2把FP8用于训练的多数核心GEMM，并压缩反向缓存和MoE通信。训练输入不仅是激活与权重，还包含上游梯度；输出既有本层结果，也有输入/权重梯度，最终由高精度主权重和优化器更新。理解这套配方需要分别追踪乘法输入、局部累加、跨组恢复、缓存、通信及长期状态，不能用“FP8训练”代替全部张量的精度说明。

本页依据§2–3的架构/训练基础设施、§3.3完整FP8节和附录B，定向核超参、MTP消融与推测报告。只核论文及必要原图，没有匹配HAI-LLM/PTX实现或训练/设备运行。基础[FP8与尺度](../fundamentals/numeric-formats/fp8-and-mx-data-formats.md)、[MoE路由](../fundamentals/model/mixture-of-experts-and-routing.md)和[MTP训练目标](../fundamentals/model/transformer-autoregressive-inference.md#3.1%20MTP%E7%9A%84%E9%A2%9D%E5%A4%96%E8%AE%AD%E7%BB%83%E7%9B%AE%E6%A0%87%E4%B8%8E%E6%8E%A8%E7%90%86%E7%94%A8%E9%80%94)由对应页承接。

## 1. 三个GEMM分别产生输出、输入梯度和权重梯度

采用右乘矩阵约定，理想线性层为$Y=XW$，$X\in\mathbb R^{M\times K}$、$W\in\mathbb R^{K\times N}$；$M$是输入token行数，$K/N$是输入/输出宽度。记上游梯度$G=\partial\mathcal L/\partial Y$：

| 操作 | 理想计算对象 | 图6的乘法输入与输出 |
| --- | --- | --- |
| Fprop | $Y=XW$ | 由激活/权重转为FP8输入，输出BF16 |
| Dgrad | $\partial\mathcal L/\partial X=GW^\top$ | 上游梯度/权重转为FP8，输入梯度BF16 |
| Wgrad | $\partial\mathcal L/\partial W=X^\top G$ | 缓存激活/上游梯度转为FP8，权重梯度FP32 |

这些式子说明反向需要原激活和转置视角，实际量化计算有舍入误差。作者让三种GEMM的低精度输入统一用E4M3，以细粒度尺度弥补动态范围；不是将所有训练梯度或参数状态存成FP8。Embedding、output head、MoE gate、norm及attention算子保留原高精度。这里的attention算子保护不能推出所有attention投影参数均不量化，后文有attention后Linear输入缓存的特例。（§3.3.1–3.3.2、图6。）

## 2. 当前组amax与K方向局部累加共同确定恢复

前向激活按每token的1×128通道tile，权重按128输入×128输出block收集**当前**amax，导出尺度并转为E4M3；不采用以历史amax预测当前尺度的delayed策略。细粒度让一个离群值只影响较小的共享范围，但每个K组的尺度不同，标准单一tensor scale的GEMM无法直接替代。（§3.3.2 Fine-Grained/Online Quantization、图7(a)。）

为解释尺度位置，在$K,N$可被128整除、矩阵索引从0开始的前向例子中，令$q^X,q^W$为解码后的FP8值，$s^X_{m,g},s^W_{g,\lfloor n/128\rfloor}$为相应组的恢复乘子，可将数值关系整理为

$$
Y_{mn}\approx\mathcal R_{\rm out}\left[
\sum_{g=0}^{K/128-1}
s^X_{m,g}s^W_{g,\lfloor n/128\rfloor}
\operatorname{Dot}_{\rm local}
\left(q^X_{m,128g:128(g+1)},q^W_{128g:128(g+1),n}\right)
\right].
$$

$g$标识K组，$\mathcal R_{\rm out}$表示输出转换，Dot是该组局部点积；这是整理者改写，近似号保留量化和累加舍入，不是原BF16 GEMM逐位恒等式。尺度在每组局部和之后参与，不能移到完整K归约之后统一乘一次。

作者报告H800 Tensor Core的FP8累加约保留14bit；随机矩阵$K=4096$的初测最大相对误差近2%，这不是所有矩阵的误差界。其策略每$N_C=128$元素、即4次WGMMA，将局部结果搬到CUDA Core的FP32寄存器，乘本组恢复尺度，再累入高精度结果。两warpgroup交错MMA与promotion以降低影响；128是作者在精度/开销间选定的间隔，理论FP8算力2×不等于整训练实测2×。（§3.3.2 Increasing Accumulation Precision、图7(b)。）

## 3. 反向缓存与持久优化器状态使用不同精度

Wgrad需要前向激活，FP8缓存可减少跨前/反向的存储。转置改变尺度共享轴：论文描述前向1×128到反向128×1，需要读取、恢复、转置并重新量化；不能把原FP8字节转置当作量化等价。[Microscaling的轴序条件](../fundamentals/numeric-formats/fp8-and-mx-data-formats.md#4.%20Microscaling%EF%BC%9A%E5%9D%97%E5%86%85%E5%85%B1%E4%BA%AB%E5%B0%BA%E5%BA%A6)解释同类问题，但两种配方的块尺寸/尺度编码并不相同。（§3.3.3、§3.5 Transposed GEMM。）

| 对象 | 报告中的表示或处理 | 作用 |
| --- | --- | --- |
| 普通Linear反向所需激活 | FP8缓存 | 供Wgrad使用 |
| attention后Linear输入 | 自定义E5M6缓存、2的整数幂尺度，反向改变tile方向 | 同时用于敏感的attention反向；未给packed/round规则 |
| MoE SwiGLU输入 | FP8缓存，反向重算输出 | 减少保存激活 |
| master weights、梯度累积缓冲 | FP32 | 保留长期更新和累计精度 |
| AdamW一、二阶moments | BF16 | 压缩优化器状态；作者报告无可观察退化 |
| RMSNorm、MLA up-projection输出 | 反向重算 | 以额外计算减少激活存储 |

E5M6和round-scale是该特例的文档描述，不能将其补成标准FP8或已验证无误差转换。高精度主权重存在也不意味着所有优化器状态同为FP32。（§3.2.3、§3.3.3、图6。）

## 4. MoE dispatch与combine有不同精度责任

MoE up-projection之前的激活先量化为FP8再dispatch，尺度为2的整数幂，接收后可作为Fprop输入；与MoE down-projection相关的激活梯度dispatch采用相似策略。前向和反向combine均保留BF16，用于专家输出或梯度汇合。因此低精度通信不是把整个往返链全部改为FP8。路由选集与gate仍由[MoE规则](../fundamentals/model/mixture-of-experts-and-routing.md#1.1%20V3%E7%9A%84sigmoid%E3%80%81%E9%80%89%E6%8B%A9%E5%81%8F%E7%BD%AE%E4%B8%8E%E8%B4%9F%E8%BD%BD%E5%B9%B3%E8%A1%A1)确定，FP8 payload不能替代这些模型语义。（§3.3.3 Low-Precision Communication。）

同样的1×128/128×128和分组恢复可在[所选DeepGEMM推理链](deepgemm-moe-expert-execution.md)中看到，但其固定SGLang消费和设备源码不等于本论文HAI-LLM训练实现；Fprop相似不足以证明Dgrad/Wgrad、缓存和优化器也已核。FP8训练还没有给出本模型低位KV缓存的packed布局。

## 5. DualPipe与网络配置限定训练系统收益

报告的训练床为2048张H800，PP16、EP64跨8节点和ZeRO-1 DP；通过重算、低精度状态等控制内存而不使用TP。DualPipe从流水线两端输入microbatches，将attention/MLP、dispatch/combine、PP通信，以及反向的input-gradient/weight-gradient工作重排，以重叠计算和通信。图4/5的“通信可隐藏”绑定约1:1的计算/通信比与调度；表2保留两份参数，峰值激活以PP+1对比其他策略的PP，PP stages和microbatches需可被2整除，不是没有任何气泡或无内存代价。（§3.1–3.2.1。）

该拓扑先经IB传到目标节点相同in-node index的GPU，再以NVLink转发给实际专家；作者给IB50GB/s、NVLink160GB/s，并用20/132个SM、10个通信channels配合warp specialization。目的节点限制和通信拓扑共同参与收益，不能外推到任意卡数/网络或当前DeepEP分支。Embedding/head放在同一PP rank让MTP物理共享参数与梯度，也有调度前提。§3.4的推理部署另采用阶段分离和attention TP，不继承这套训练并行配置。（§3.2.2–3.2.3、§3.5 Communication Hardware。）

## 6. 联合精度配方的证据范围与失败反例

附录B/图10对照BF16与FP8训练的是V2-like MoE：约16B/1.33T tokens和230B/约0.9T tokens；loss曲线用EMA0.9平滑。作者报告相对loss差<0.25%，不是完整671B模型的BF16配对质量验证，也不是下游准确率绝对只损失0.25%。

为省去反向重新分组，作者把所有Dgrad相关张量改为128×128 block；约16B模型、300B训练tokens实验发散。作者假设token间激活梯度不均和token-correlated outliers是原因，说明缓存便利可能牺牲梯度精度，不是所有block-FP8或所有模型都失败。MTP和路由平衡另有各自消融，不能把模型最终能力全归因于FP8。（附录B.1–B.2，图10。）

Zero-group/非有限值、量化端点/舍入、E5M6物理存储、完整梯度/optimizer collective、匹配HAI-LLM/PTX、推测验证与真实训练/性能未核。论文报告的计算精度、理论收益、局部初测和训练loss对照分别保留，未运行实验。

## 来源身份

- DeepSeek-V3 Technical Report（`https://arxiv.org/abs/2412.19437v2`），`arXiv:2412.19437v2`；PDF53页，首页18Feb2025。架构/MTP、训练基础设施、§3.3全文、§3.4–3.5、必要超参/消融及附录B；图3–7/10与关键PDF页回查。未核全部数据/后训练/模型质量表或匹配实现/运行。
