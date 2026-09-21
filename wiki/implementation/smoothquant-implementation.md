---
title: SmoothQuant 的实现核对：平滑融合、INT8 接口与离线边界
type: implementation
tags:
  - llm
  - activation-quantization
  - equivalent-transform
  - kernels
  - deployment
sources:
  - raw/papers/2026-09-21/smoothquant/paper.pdf
  - raw/repositories/2026-09-21/smoothquant/source/smoothquant/smooth.py
  - raw/repositories/2026-09-21/smoothquant/source/smoothquant/calibration.py
  - raw/repositories/2026-09-21/torch-int/source/torch_int/nn/linear.py
  - raw/repositories/2026-09-21/sglang/source/python/sglang/srt/layers/quantization/w8a8_int8.py
updated: 2026-09-22
---

# SmoothQuant 的实现核对：平滑融合、INT8 接口与离线边界

SmoothQuant 的公式只有一行（`WX = (WD)(D⁻¹X)`），但工程上要回答的是：**平滑因子在哪里算、按什么统计算、融合到哪些参数上、以及服务框架拿到这份 checkpoint 时还要不要做别的事。** 这需要区分离线平滑因子、运行时激活量化尺度和内核重缩放系数。

缩放公式、适用条件与论文证据见 [SmoothQuant](../methods/smoothquant.md)；融合的代数条件（前驱必须能吸收逆缩放、多消费者与残差要一起处理）见 [对角缩放与等价变换](../theory/diagonal-scaling-equivalent-transform.md)。

## 1. 快照与读取范围

| 快照 | 固定版本 | 读取范围 |
|---|---|---|
| mit-han-lab/smoothquant | `c61476d728e42ae0d8a35e7e78494edcac3237b5` | `smoothquant/smooth.py`、`calibration.py`；`act_scales/` 目录内容（快照中仅有 README） |
| guangxuan-xiao/torch-int | `65266db1` | `torch_int/nn/linear.py` 的 INT8 线性接口 |
| sgl-project/sglang | `2f730e29` | `srt/layers/quantization/w8a8_int8.py` |

未读取：`fake_quant.py`、`export_int8_model.py`、`opt.py` 的逐行实现、量化模型的示例脚本；未运行任何环节。

## 2. 原仓库：三个组件

仓库把平滑流程拆成三步，与我们的方法页描述一致，但代码里的边界更清楚：

1. **校准（`calibration.py`）。** `get_act_scales(model, tokenizer, dataset_path, num_samples=512, seq_len=512)` 对目标线性层的输入挂 hook，`stat_tensor` 用 `torch.max` 逐通道累积最大值（跨样本取大）。**统计量是最大绝对值，不是均值**——这是与 [AWQ](../methods/awq.md)（平均绝对激活）最容易被混同的差异，也是两者缩放公式不可互换的原因。
2. **平滑（`smooth.py`）。** 把缩放同时写进归一化参数与后续线性权重。
3. **模拟与导出（`fake_quant.py`、`export_int8_model.py`、`opt.py`）。** 前者用于验证模拟精度，后者导出真实 INT8 模型并给出 INT8 注意力路径；本页未展开。

## 3. 融合的代码形态

`smooth_ln_fcs` 与 `smooth_ln_fcs_llama_like` 是同一件事的两个版本，差别只在归一化是否带偏置：

```python
# 断言：归一化输出维 == 每个消费者线性层的输入维 == 激活统计维
assert ln.weight.numel() == fc.in_features == act_scales.numel()

# 权重侧统计取「所有消费者里的最大」，并设下限
weight_scales = torch.cat([fc.weight.abs().max(dim=0, keepdim=True)[0]
                           for fc in fcs], dim=0).max(dim=0)[0].clamp(min=1e-5)

scales = (act_scales.pow(alpha) / weight_scales.pow(1 - alpha)).clamp(min=1e-5)

# LayerNorm 版本：权重与偏置都除；激活之后吸收
ln.weight.div_(scales); ln.bias.div_(scales)
for fc in fcs: fc.weight.mul_(scales.view(1, -1))
```

四点实现知识：

- **多消费者取最大值。** `fcs` 是列表（例如 Q/K/V 共享同一归一化输出），`weight_scales` 先各自求逐输入通道最大值，再逐通道取最大。这与 [等价变换页](../theory/diagonal-scaling-equivalent-transform.md) 记录的「共享归一化的多个 FC 必须用同一组缩放」是同一约束的代码形态。
- **RMSNorm 分支不处理偏置。** `smooth_ln_fcs_llama_like` 只做 `ln.weight.div_(scales)`，没有 `ln.bias`；对带偏置的 LayerNorm 版本则会一起除，否则平移项不会被正确吸收。
- **下限 `1e-5` 出现两次**：一次在权重的逐通道最大值上（避免整列为零），一次在缩放本身上。零激活通道与零权重通道都要保护。
- **缩放只写到「输入列」**：`fc.weight.mul_(scales.view(1, -1))` 的形状说明缩放沿输入维广播，与「逐输入通道缩放」的定义一致。

快照中的 `act_scales/` 目录只有 README，说明**预计算的激活统计没有随快照保存**；复现时需要自行跑校准。

## 4. 执行侧：INT8 线性接口

torch-int 的 `W8A8B8O8Linear` 是这条路径的执行端：

- `from_float` 设置 `alpha = input_scale * weight_scale / output_scale`、`beta = bias_scale / output_scale`。前者把整数矩阵乘累加映射到输出网格，后者把独立量化的偏置映射到同一网格；这里的 alpha 不是平滑迁移强度；
- `from_float(module, input_scale, output_scale)` 负责转换：权重与偏置分别用 `quantize_per_tensor_absmax` 量化（**逐张量**，不是逐通道）；
- 论文 v7 表 2 的 O1/O2/O3 本来就使用逐张量权重；表 7 的后续模型设置才使用逐输出通道权重。这条 torch-int 路径的逐张量尺度不能单独当作论文与实现冲突，比较时须选同一实验配置。

## 5. 跨引擎：SGLang 的 w8a8_int8

该快照 `W8A8Int8LinearMethod` 的普通 GPU 路径提供了比方法名更具体的契约：权重以 INT8 保存，weight_scale 的形状为 `(本分片输出通道数, 1)`；加载后转置权重。执行时先 `per_token_quant_int8(x)`，再把量化输入、逐 token 尺度、逐输出通道权重尺度交给 `int8_scaled_mm`，输出恢复为输入的 dtype。

因此这个路径使用**动态逐 token 激活和逐输出通道权重**，不能直接等同于 O3 的静态逐张量配置；CPU 与 MoE 分支也不能据此一并推断。此处核对到 Python 调用和参数形状，尚未运行或审查底层整数内核。

`get_scaled_act_names()` 返回空列表，只说明该配置没有通过这一接口声明额外激活缩放；不能仅凭空列表证明 checkpoint 已做 SmoothQuant、所有平滑均已融合或模型已正确转换。若加载的是经过平滑的模型，平滑因子与相应权重/归一化参数必须在上游保持一致；本方法名本身也不能认证它来自哪一种量化算法。

## 6. 跨引擎：TensorRT-LLM 的 int8_sq

TensorRT-LLM 的量化配置里把这件事说得更直接：`int8_sq` 的描述是「权重先被平滑（smoothed）再按通道量化为 INT8，激活范围按张量校准」。也就是说**平滑是量化工具链的一部分**，与引擎解耦；这属于 TensorRT-LLM 自身配置说明，不能由 SGLang 的空钩子推出，见 [部署框架与后端支持](quantized-llm-deployment-backends.md) 的格式表。

## 7. 论文-代码对齐（本页补充）

- 方法页记录的公式 $s_j=a_j^\alpha/b_j^{1-\alpha}$ 在代码里逐字出现，其中 `a_j` 是校准输入的逐通道激活 absmax；共享消费者之间再次取最大值的是权重侧 `b_j`，不要把两种归约写反；
- 实现额外引入两处 `clamp(min=1e-5)`，属于数值保护而非算法部分；
- `alpha` 在函数签名上默认 `0.5`，但论文的模型特定取值由调用方传入（`smooth_lm(model, scales, alpha)`），默认值不等于论文配置；
- 论文表 2 与表 7 本身使用不同权重粒度；torch-int 和 SGLang 的已读路径也不同，均须逐配置记录。

## 8. 可复用的实现要点

1. 平滑统计用最大绝对值，逐输入通道累积；不要与平均绝对激活互换；
2. 共享归一化的所有消费者必须用同一组缩放，权重侧取逐通道最大值；
3. 归一化带偏置时，权重与偏置一起除；RMSNorm 则只除权重；
4. 通道与缩放都要设数值下限，并记录下限值；
5. 沿导出、加载和算子调用追踪平滑是否融合；空接口不能证明前处理正确，动态激活量化也不等于在线重新学习平滑因子；
6. 权重尺度是逐通道还是逐张量，随执行接口变化，必须按后端核对。

## 9. 验证状态与待验证

- 全部结论为 code-read；未运行校准、平滑、导出或任何 INT8 内核，未复现论文精度。
- `fake_quant.py`、`export_int8_model.py` 与 `opt.py` 的 INT8 注意力路径未逐行审查；本页不评价导出模型的数值正确性。
- SGLang 已补读普通 GPU 线性路径的权重/尺度形状、加载后转置和逐 token 量化调用；底层 INT8 内核、完整张量并行加载及 CPU/MoE 路径未核验。
- 预计算的 `act_scales` 未随快照保存，复现校准需要自行准备数据；本页没有运行 `get_act_scales`。

## 来源身份

下表用于在没有本地资料库时辨识来源；具体论述的章节、公式、图表或代码位置见正文。

| 来源 | 版本或快照 | 说明 |
| --- | --- | --- |
| [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438v7) | `arXiv:2211.10438v7` | 表 2 与表 7 的权重粒度 |
| [mit-han-lab/smoothquant.git](https://github.com/mit-han-lab/smoothquant/tree/c61476d728e42ae0d8a35e7e78494edcac3237b5) | `c61476d728e42ae0d8a35e7e78494edcac3237b5` | — |
| [Guangxuan-Xiao/torch-int](https://github.com/Guangxuan-Xiao/torch-int/tree/65266db1eadba5ca78941b789803929e6e6c6856) | `65266db1eadba5ca78941b789803929e6e6c6856` | — |
| [sgl-project/sglang](https://github.com/sgl-project/sglang/tree/2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc) | `2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc` | — |
