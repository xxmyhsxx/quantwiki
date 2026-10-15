# 量化方法：先确认对象与改变的量

用于方法解释、校准与路线比较。先区分权重、激活或 KV，以及校准过程与推理执行；只读与问题相符的入口，再定位章节。

| 要回答的问题 | 正文入口 | 可补查的词 |
| --- | --- | --- |
| scale、零点、分组和舍入是什么 | [均匀量化与分组](/Users/xiao/Documents/Projects/quantwiki/wiki/fundamentals/quantization/uniform-quantization-and-groups.md) | zero-point、粒度、裁剪 |
| AWQ、GPTQ、SmoothQuant 改变什么 | [经典方法比较](/Users/xiao/Documents/Projects/quantwiki/wiki/methods/awq-gptq-smoothquant-comparison.md) | activation-aware、二阶补偿、平滑 |
| 校准数据、目标与量化范围怎样选择 | [校准与范围](/Users/xiao/Documents/Projects/quantwiki/wiki/theory/calibration-and-range-selection.md) | 前缀、统计、重构、生成状态 |
| 缩放或旋转为什么能改变误差 | [对角缩放](/Users/xiao/Documents/Projects/quantwiki/wiki/theory/diagonal-scaling-equivalent-transform.md)、[正交旋转](/Users/xiao/Documents/Projects/quantwiki/wiki/theory/orthogonal-rotation-and-hadamard-quantization.md) | 融合、可逆、Hadamard、RoPE |
| 学习网格与普通 PTQ 有何不同 | [PTQ 与 QAT](/Users/xiao/Documents/Projects/quantwiki/wiki/fundamentals/quantization/post-training-and-quantization-aware-training.md) | STE、可学习、选码、梯度 |
| 极低位宽、码本或混合精度如何计账 | [表示设计](/Users/xiao/Documents/Projects/quantwiki/wiki/research/low-bit-representation-design.md)、[精度分配](/Users/xiao/Documents/Projects/quantwiki/wiki/theory/mixed-precision-allocation.md) | 残差、码本、元数据、预算 |

具体方法已知时，先在绑定 Wiki 的 `methods/` 按英文名或中文别名查文件，不必先读比较页。机制补查 `theory/`，跨方法联系补查 `research/`。只有问题涉及保存格式或实际消费时，才转读部署 reference。
