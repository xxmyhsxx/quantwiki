---
title: AWQ、GPTQ 与 SmoothQuant：对象、机制与证据比较
slug: awq-gptq-smoothquant-comparison
sources:
  - raw/papers/quantization/ptq/2023-06-awq-activation-aware-weight-quantization/paper.pdf
  - raw/papers/quantization/ptq/2022-10-gptq-accurate-post-training-quantization/paper.pdf
  - raw/papers/quantization/ptq/2022-11-smoothquant-accurate-efficient-ptq/paper.pdf
updated: 2026-09-14
---

# AWQ、GPTQ 与 SmoothQuant：对象、机制与证据比较

这三种方法都利用少量校准数据改善模型量化，但量化对象、可调整变量与实现路线并不相同。比较的目的，是理解方法为什么这样设计，以及哪些证据能够相互对照；本页不是质量或速度排名，也不是整个 LLM 量化领域的研究总览。

依据为 AWQ v6 §3–5、GPTQ v2 §3–5、附录 A、SmoothQuant v7 §3–5、附录 A。代码、数值、消融和原文问题留在各自唯一方法页。

## 1. 从对象和问题开始

| 比较维度 | AWQ | GPTQ | SmoothQuant |
|---|---|---|---|
| 主研究对象 | 低比特权重，常见 W3/W4A16 | 低比特权重，主实验 W3/W4，激活不量化 | 权重与激活，W8A8 |
| 核心问题 | 输入通道重要性不同，权重量化误差怎样分配 | 固定量化值后，怎样补偿层输出误差，并让过程可扩展 | 激活离群值导致范围失衡，怎样适配整数矩阵乘 |
| 利用输入的方式 | 平均绝对激活等通道统计，以及输出误差候选比较 | 输入的未中心化二阶矩阵 $XX^{\mathsf T}$ | 逐输入通道最大绝对值与权重最大值 |
| 主要操作 | 等价缩放、参数搜索与裁剪 | 逐列量化、补偿剩余权重 | 等价缩放，平衡两侧范围，再 W8A8 量化 |
| 量化时是否改变未量化权重 | 通过缩放/裁剪改变表示 | 通过二阶补偿直接调整 | 通过缩放改变表示 |
| 校准但不端到端训练 | 是 | 是 | 是 |

“activation-aware”不等于“activation quantization”。AWQ 和 GPTQ 都依赖激活数据，主方法仍仅量化权重。SmoothQuant 的激活量化还需区分线性层输入、attention BMM 输入和实际 KV 存储路径。

## 2. 三个容易混淆的机制

**AWQ 与 SmoothQuant 共享代数变换，但目标不同。** 在列 token 约定下，两者都可写成 $WX=(WD)(D^{-1}X)$。AWQ 选择缩放以改善 weight-only 输出误差；SmoothQuant 需要同时平衡两侧误差。最大绝对激活、平均绝对激活及其 alpha 公式不能互换。[[diagonal-scaling-equivalent-transform|对角缩放与等价变换]]。

**GPTQ 的重构与 AWQ 的搜索并不只是“是否使用 MSE”的区别。** 两者都可能评价局部输出变化。关键是 GPTQ 逐步改变剩余权重，AWQ 在选定的缩放/裁剪族中搜索。比较时要说明可调整变量、目标、数据以及计算成本。[[layer-reconstruction-second-order-compensation|层输出重构与二阶误差补偿]]。

**数学上的可组合需要新的证据。** AWQ 表 9 有 INT2-g64 的 AWQ+GPTQ 组合实测，是那个设定下的证据；SmoothQuant 附录 A 提出的结合 GPTQ 达成 W4A4 是未来工作，不能视为已经完成。同样不能把一个组合实验推广成任意顺序、任意格式都兼容。

## 3. “低比特”与“快”之间还有执行路径

GPTQ 的原始系统使用低比特权重与浮点向量的 kernel，主要减少单 token 生成中的权重读取；AWQ 配套 TinyChat 还包含打包、反量化融合及其他系统优化；SmoothQuant 则利用 INT8 GEMM/BMM，在其所测 context 和 batched 场景中获得收益。

这些历史系统的硬件、模型、batch 和基线不同。GPTQ 的 3.24×、SmoothQuant 的 1.56×、AWQ 的某个吞吐比，不能直接组成方法排序。执行路径的共性解释见[[quantized-matmul-scaling-execution|量化矩阵乘法的缩放与执行路径]]。

## 4. 哪些比较已经有证据，哪些还没有

| 问题 | 现有证据与边界 |
|---|---|
| AWQ 与 GPTQ 在某些共同量化配置下谁更准 | AWQ 原文有直接基线和组合比较；应绑定其模型、位宽、group 和评测，而非跨论文拿数字 |
| SmoothQuant 与 weight-only 谁更快 | SQ 附录讨论场景差异，并指出后端对照困难；本项目没有统一基准 |
| 是否都不需要校准 | 都使用校准；三者的统计与优化作用不同 |
| 是否都能避免过拟合 | 有限分布实验证据不等于普遍保证，任何校准驱动选择都需要考虑分布变化 |
| 是否都压缩 KV cache | 不能从方法名推出；GPTQ/AWQ 主算法不解决 KV，SQ 要查具体后端 |
| 模型质量是否无损 | 文献结果包含退化、反例和协议差异，没有对所有能力的无损证明 |

W8A8 一侧的两个常被引用基线现在有独立主页面：[[llm-int8|LLM.int8()]] 用混合精度分解保留离群维度，[[zeroquant|ZeroQuant]] 用细粒度量化加逐层蒸馏，两者的精度恢复机制与可训练变量不同。把它们与 weight-only 方法并列时，仍要保留对象差异：前两者要处理激活量化与整数 GEMM 的输出再量化，AWQ 与 GPTQ 的主方法只量化权重。

阅读顺序可按问题选择：[[awq|AWQ]]看通道误差分配，[[gptq|GPTQ]]看二阶补偿与计算扩展，[[smoothquant|SmoothQuant]]看 W8A8 和硬件映射。三篇均有独立完整解释，不需要靠本比较页拼出方法。
