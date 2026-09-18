---
title: 算术强度与 roofline 分析
slug: arithmetic-intensity-and-roofline
sources:
  - raw/repositories/serving/2026-08-stanford-cs336-lectures-8b59b507/source/lecture_02.py
  - raw/papers/quantization/ptq/2024-08-marlin-mixed-precision-autoregressive-parallel-inference/paper.pdf
  - raw/papers/quantization/ptq/2024-05-qserve-w4a8kv4-quantization-system-codesign/paper.pdf
updated: 2026-09-17
---

# 算术强度与 roofline 分析

「这个算子受什么限制」是决定优化方向的第一问题。低比特权重在解码阶段有效、在大批量下收益消失，本质上都是同一条判断在不同参数下的结果。本页给出这条判断的记账方法：两个强度量的定义、常见算子的取值、以及由它导出的临界批量。

定义与算例以 Stanford CS336 讲义（本地快照 `lecture_02.py`，含算子算术强度的逐项计算与 roofline 图）为准；临界批量的换算在 MARLIN 论文 §3.1 中给出同形推导（A10、$b_{opt}\approx50$），QServe 论文 §3.1 用同一框架比较 W4A16／W8A8／W4A8。硬件层级结构见 [[gpu-execution-and-memory-hierarchy|GPU 的执行模型与存储层次]]，量化侧的执行分类见 [[quantized-matmul-scaling-execution|量化矩阵乘法的缩放与执行路径]]。

## 1. 先分清两个术语

讲义专门提示这两个缩写发音相同、含义不同：

- **FLOPs**：浮点运算量，衡量「做了多少计算」；
- **FLOP/s**（也写作 FLOPS）：每秒浮点运算次数，衡量「硬件有多快」。

把两者混淆会直接导致错误结论，例如把「权重少了 4 倍」读成「时间少了 4 倍」。

## 2. 两个强度量

讲义把一次计算抽象为三步：输入从内存送到加速器、执行计算、结果写回内存。假设计算与通信可以完全重叠，则耗时是两者的较大值：

$$t=\max\left(t_{\mathrm{compute}},\ t_{\mathrm{memory}}\right),\qquad t_{\mathrm{compute}}=\frac{\mathrm{FLOPs}}{P_{\mathrm{peak}}},\qquad t_{\mathrm{memory}}=\frac{\mathrm{Bytes}}{B}.$$

于是有两个可以比较的量：

- **加速器强度** $I_{\mathrm{acc}}=P_{\mathrm{peak}}/B$：硬件每搬一字节能算多少浮点运算；
- **算术强度** $I=\mathrm{FLOPs}/\mathrm{Bytes}$：这次计算每搬一字节实际做了多少。

判定规则是：$I<I_{\mathrm{acc}}$ 时受内存带宽限制（memory-bound），$I>I_{\mathrm{acc}}$ 时受算力限制（compute-bound）。讲义给出的 H100 数值是 $P_{\mathrm{peak}}=989.5$ TFLOP/s（BF16，不含稀疏）与 $B=3.35$ TB/s，对应 $I_{\mathrm{acc}}\approx295$ FLOP/Byte。

## 3. 常见算子的算术强度

以下按讲义的函数逐项列出（输入输出均为 bf16，即每元素 2 字节，$n$ 为元素数或矩阵维度）：

| 计算 | FLOPs | 搬运字节 | 算术强度 | 在 H100 上的瓶颈 |
|---|---:|---:|---:|---|
| ReLU | $n$ | $4n$ | $1/4$ | 内存 |
| GELU | $20n$ | $4n$ | $5$ | 内存 |
| 点积 | $2n-1$ | $4n+2$ | 约 $1/2$ | 内存 |
| 矩阵向量乘（$n\times n$） | $n(2n-1)$ | $2n+2n^2+2n$ | 约 $1$ | 内存 |
| 大矩阵乘（$n\times n$ 乘 $n\times n$） | $n^2(2n-1)$ | $6n^2$ | 约 $n/3$ | 计算（$n$ 大时） |

讲义据此给出两条结论：**只有矩阵足够大时才受算力限制**；**推理中的矩阵向量乘天然受带宽限制**。它还提醒，GELU 的算术强度高于 ReLU，但两者都受带宽限制，因此「单独看」时 ReLU 并不比 GELU 快——局部优化的收益取决于瓶颈是否移动。

## 4. roofline 与 MFU

把「算术强度—性能」画成图就是 roofline：横轴是算术强度，每个硬件的可用性能是一条先上升后水平的折线，**拐点就是该硬件的加速器强度**（讲义对 roofline 图的三点说明）。由它可以直接读出模型浮点利用率：

$$\mathrm{MFU}=\min\left(1,\ \frac{I}{I_{\mathrm{acc}}}\right),$$

即实际 FLOP/s 与规格峰值的比值，忽略通信与其它开销。讲义的经验判断是 MFU 达到 0.5 已经相当好——这提醒我们规格峰值是上限，而不是可达目标。

## 5. 由它导出量化研究的两个常用判断

**解码阶段的收益来自减少权重流量。** 解码是矩阵向量乘形态，受带宽限制，耗时近似为搬运字节数除以带宽。把权重从 $w$ 位降到 $w'$ 位，权重流量按 $w'/w$ 缩小；只要元数据没有把省下的字节吃掉，这部分时间就近似等比下降。

**权重读取主导的临界批量可以估算。** 设权重位宽为 $w$ 位、批量为 $b$（即每个权重被使用 $b$ 次）。每个权重占 $w/8$ 字节、贡献 $2b$ 次浮点运算，因此算术强度近似为

$$I\approx\frac{2b}{w/8}=\frac{16b}{w},\qquad \text{受内存限制当且仅当}\quad b<\frac{I_{\mathrm{acc}}\,w}{16}.$$

代入检验：A10 的加速器强度约 200 FLOP/Byte，4 位权重给出 $b<50$——与 MARLIN 论文按「读完一个 4 位权重的时间可执行约 100 次浮点运算」得到的 $b_{opt}\approx50$ 一致；同一算法在 H100（$I_{\mathrm{acc}}\approx295$）上给出 $b<74$。位宽越低的权重每个字节能承载更多权重，因此同样批量下更可能落在受带宽限制的一侧；批量增大后这个不等式反过来，转向受算力限制，此时低比特的价值从「省流量」变成「用更高吞吐的计算单元」。这正是 QServe 论文用 W4A16 与 W8A8 的交叉点（其估算中 $m\approx78$）讨论精度组合的原因。

以上换算是本页依据讲义定义与两篇论文同类推导整理，用于量级判断；真实内核还受元数据、启动延迟与访存模式影响，不能直接当作加速比预测。

## 6. 测量口径

- **峰值不等于实测。** 规格数值需与 MFU 一起报告，否则「加速比」没有可比基准。
- **必须绑定数据类型。** 峰值算力随 dtype 变化（BF16／INT8／INT4 不同），带宽通常不变，因此加速器强度也随精度改变；跨精度的 FLOP/s 不能直接比较。
- **必须绑定批量与阶段。** 同一模型在 prefill（大矩阵乘形态）与 decode（矩阵向量形态）下处于 roofline 的不同位置，批量改变算术强度。
- **roofline 是上界模型。** 它忽略启动延迟、内存局部性、调度与内核融合，因此能判断瓶颈方向，不能替代实测；量化工作里频频出现的「理论 4 倍、实测要打折扣」都发生在这些被忽略的项上。

## 7. 局限与未验证

- 本页的定义与算子算例来自讲义原文，数值为其给出的 H100 规格；本轮没有在本地运行任何基准或 profiler。
- 临界批量的公式是本页的换算，只在「权重读取主导、忽略激活与输出流量、忽略元数据」的前提下成立；前缀元数据会改变每字节可承载的权重数，见 [[fp8-and-mx-data-formats|FP8 与 Microscaling 数值格式]] 与 [[gguf-block-quantization-formats|GGUF 与块量化存储格式]] 的位宽口径。
- 讲义是教学材料，其算子代数强度按「单独执行该算子」计算；真实模型中的算子会融合执行，融合后的算术强度需按融合后的数据流重新计算。
