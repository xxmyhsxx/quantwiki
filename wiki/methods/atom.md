---
title: Atom：离群通道重排、分组整数 GEMM 与服务成本
type: method
tags:
  - mixed-precision
  - activation-quantization
  - group-quantization
  - serving
sources:
  - raw/papers/2026-09-21/atom
updated: 2026-10-12
---

# Atom：离群通道重排、分组整数 GEMM 与服务成本

Atom 将少量原始输入离群通道放在连续尾部，用INT8处理；其余权重和激活采用分组INT4，同时将重排、动态量化、分组恢复和KV读取融合进算子。它连接低位表示与大batch服务，而不是仅压缩权重后调用浮点GEMM。依据为 Atom v3（`https://arxiv.org/abs/2310.19102v3`） 正文、表1–4及性能图；固定作者代码的模拟/整数/基准三条路径见[实现页](../implementation/atom-mixed-gemm-and-paged-kv.md)。没有运行作者模型或GPU。

与 [ResQ](resq.md) 保护PCA投影方向不同，Atom按原通道校准能量选择高精度尾部；与 [LLM-MQ](llm-mq.md) 的低位weight-only GEMV不同，它面向权重复用后的整数GEMM及低位KV读取。大batch是否进入计算瓶颈需按实际算子判断，基础趋势见[矩阵乘法的token数与瓶颈](../implementation/quantized-matmul-scaling-execution.md#8.%20%E4%B8%BA%E4%BB%80%E4%B9%88%20token%20%E6%95%B0%E6%94%B9%E5%8F%98%E7%93%B6%E9%A2%88)。

## 1. 离线确定顺序，在线应用固定重排

§5.1使用128个WikiText2随机校准样本，选择平方和最大的128个输入通道保留INT8。离线把权重对应输入维重排，推理时将激活同样重排，才能保持未量化乘积。令行token激活 $X\in\mathbb R^{T\times d}$、权重 $W\in\mathbb R^{d\times o}$、置换矩阵 $P$，则 $(XP)(P^\top W)=XW$。重排后的分组量化改变网格和舍入，因此不保证与“先量化再重排”相同。

图7中的 dynamic reorder 指在线应用离线确定的索引，并非每个请求重新寻找离群通道。权重重排为一次成本，激活每次前向仍需按索引读取；作者用融合减少中间读写，报告其所测重排开销低于0.5%。输入分布改变时，固定保护集合不一定仍对应最大的通道。

在Llama门控MLP中，gate/up输出可以采用同一置换：逐元素SiLU和乘法与一致的置换交换，down输入权重再配套重排。任意稠密旋转不能这样穿过非线性；Q/K的RoPE维度也不能任意置换。固定实现保留这些位置区别，详见实现页。

## 2. 每个归约组都要先恢复尺度再相加

普通部分沿归约维按128元素分组，权重与激活对称量化；激活在推理时动态计算范围，权重离线结合GPTQ。论文选激活/权重裁剪比例0.9/0.85。对输出位置 $(t,j)$，令 $C^{(g)}_{tj}=\sum_{k\in g}q^A_{tk}q^W_{kj}$ 为第 $g$ 组整数累加、$s^A_{t,g},s^W_{g,j}$ 为恢复步长，则

$$\widehat Y_{tj}=\sum_g s^A_{t,g}s^W_{g,j}C^{(g)}_{tj}+Y^{high}_{tj}.$$

高组也按自己的INT8尺度恢复。归约组的尺度不同时，不能把所有整数部分先加起来再乘一个外部尺度；这一点与仅按输出列/输入token缩放的GEMM不同。Atom把分组恢复和相加融合到MMA流水，减少中间全局写回，但仍有尺度读取、转换及累加成本。（§4.2、图8；公共[分组归约与融合](../implementation/quantized-matmul-scaling-execution.md#1.1%20%E5%BD%92%E7%BA%A6%E7%BB%B4%E5%88%86%E7%BB%84%E4%B8%BA%E4%BB%80%E4%B9%88%E8%A6%81%E5%9C%A8%E7%B4%AF%E5%8A%A0%E9%80%94%E4%B8%AD%E6%81%A2%E5%A4%8D%E5%B0%BA%E5%BA%A6)。）

§4.2对Llama-7B的 $d=4096$、高组128、普通4 bit/高组8 bit、每128元素一个FP16尺度，给出元素预算

$$\frac{(4096-128)4+128\cdot8}{4096}+\frac{16}{128}=4.25\text{ bit}.$$

这绑定具体维度与元数据假设，不能推广成全模型或所有运行buffer的实际位宽。固定实现有尺度复制/对齐、权重输出通道共用尺度等具体布局；真实存储需另按容器计数。

## 3. KV压缩加速的是读取，attention仍做浮点计算

§4.4按attention头对K/V作非对称低位量化，在融合读取时恢复浮点，然后执行attention。它不需要为了压缩KV而把QK/softmax/AV都改成INT4。当前选定实现使用128维头、每个token/头两个FP16参数；低位数据与参数都要读取，不能仅按4-bit载荷估算全部流量。

原文用Softmax归一化解释K误差影响较小，但归一化不能普遍保证滤掉量化误差：K改变logits，Softmax概率和AV都可改变。V离群较少也是所采模型/位置的经验观察。误差传导的条件见[attention误差链](../theory/attention-quantization-error-propagation.md)。固定模拟路径在RoPE前量化K，而整数缓存路径将未旋转K保存并在decode读取时旋转；顺序与参数需逐项匹配。

## 4. 质量结果和消融支持什么

表1的6项0-shot任务为PIQA、ARC-e、ARC-c、BoolQ、HellaSwag、WinoGrande。7B FP16均分64.04，Atom W4A4为61.78，相差2.26**个百分点**；W3A3为51.37，不能将较大模型的恢复结论推广到7B的3 bit。65B对应70.97→69.57，差1.40个百分点。（§5.2、表1。）

| 表2模型/位宽 | WikiText2 | PTB | C4 |
| --- | ---: | ---: | ---: |
| Llama-7B FP16 | 5.68 | 8.80 | 7.08 |
| Llama-7B Atom W4A4 | 6.16 | 9.62 | 7.70 |
| Llama-7B Atom W3A3 | 11.77 | 20.84 | 15.43 |
| Llama-65B FP16 | 3.53 | 6.91 | 5.62 |
| Llama-65B Atom W4A4 | 3.89 | 7.22 | 5.92 |

65B的W4A4 PPL增量为0.36/0.31/0.30，均小于0.4；这是绝对PPL差，不是0.4%相对下降或全模型统一结果。质量实验的基线也有范围差异：SmoothQuant为作者另实现的Llama低位版本，QLLM引用原论文W4A4数值且部分任务缺失，OmniQuant部分大模型W3A3未评测，不能把空格当失败结果或共同协议完整对照。

表3的7B递进消融为：RTN W4A4 2315.52；保留128个FP16离群通道11.34；将高组也量化到INT8 11.39；加group128后6.22；加裁剪6.13；加GPTQ 6.04；加INT4 KV后6.16。它显示协同配方的逐步收益与KV代价，但顺序消融不是各组件独立收益的可加分解。

表4扩展到Llama2与Mixtral：Llama2-7B FP16/INT4/FP4 PPL为5.47/6.03/6.14，Mixtral为3.84/4.41/4.50。FP4表示通过模拟评测，不等于所测RTX4090的原生FP4执行；文中对未来MX硬件的期待也不是该论文已测性能。Mixtral共享专家重排索引的条件不能升级为所有MoE的保证。（§6。）

## 5. 大batch收益必须绑定测量边界

质量预处理报告单RTX Ada6000、65B约4小时；性能使用24GB RTX4090、CUDA11.3和INT4路径，两者硬件与目标不同。图11的算子微基准按Llama-7B/sequence1024配置：GEMM在batch512相对FP16/INT8为3.4/1.9倍；attention在batch128为3.5/1.8倍。不能将不同batch和不同算子加速直接相加为系统速度。（§5.3.1。）

图10的Punica服务采用FCFS和continuous batching，按ShareGPT的输入/输出长度分布构造工作负载；固定显存下允许不同配方使用不同最大batch，报告最高7.73倍对FP16、2.53倍对INT8。该收益包含低位计算、KV/权重容量与batch复用，不是固定同batch的纯GEMM比值。图中实线为实测，超出显存的虚线通过复用较小batch缓存估计；延迟为平均每token decode时间，**不含排队**，不能解释为完整服务请求响应时间。（§5.3.2、图10、PDF第9页。）

效率消融在batch4096下报告纯INT4约980 TOPS、加入INT8尾部900、分组恢复770；分别约8.16%和14.44%的相邻吞吐下降，不能与整网延迟下降混用。重排融合对比的25–35%改善只覆盖所测norm+GEMM组合。

公开快照同时保存质量模拟、真实整数kernel与效率benchmark。效率模型使用随机权重和缓存复用等简化；本次静态分析不把它与真实权重的质量测试合为一个已复现系统。具体接口、舍入与基准差异见[实现页](../implementation/atom-mixed-gemm-and-paged-kv.md)。

## 来源身份

| 来源 | 固定身份 | 本页使用范围 |
| --- | --- | --- |
| Atom: Low-bit Quantization for Efficient and Accurate LLM Serving（`https://arxiv.org/abs/2310.19102v3`） | arXiv:2310.19102v3；PDF SHA-256 `d33aaa47a1e86f346c27ce4e53aaec412e41c656f5473c657f38740b3945700d` | 正文设计/分析、表1–4与图6–11；必要补充范围，未逐点重建全部性能曲线 |
