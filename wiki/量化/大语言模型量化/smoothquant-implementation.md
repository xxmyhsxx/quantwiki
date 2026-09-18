---
title: SmoothQuant 的实现核对：平滑融合、INT8 接口与离线边界
slug: smoothquant-implementation
sources:
  - raw/repositories/quantization/2026-06-mit-han-lab-smoothquant-c61476d/source/smoothquant/smooth.py
  - raw/repositories/quantization/2026-06-mit-han-lab-smoothquant-c61476d/source/smoothquant/calibration.py
  - raw/repositories/quantization/2026-07-guangxuan-xiao-torch-int-65266db1/source/torch_int/nn/linear.py
  - raw/repositories/serving/2026-06-sgl-project-sglang-2f730e29/source/python/sglang/srt/layers/quantization/w8a8_int8.py
updated: 2026-09-17
---

# SmoothQuant 的实现核对：平滑融合、INT8 接口与离线边界

SmoothQuant 的公式只有一行（`WX = (WD)(D⁻¹X)`），但工程上要回答的是：**平滑因子在哪里算、按什么统计算、融合到哪些参数上、以及服务框架拿到这份 checkpoint 时还要不要做别的事。** 第三问的答案是本页最有用的部分——它是「离线平滑、在线只需要 INT8 内核」这一分工的直接证据。

缩放公式、适用条件与论文证据见 [[smoothquant|SmoothQuant]]；融合的代数条件（前驱必须能吸收逆缩放、多消费者与残差要一起处理）见 [[diagonal-scaling-equivalent-transform|对角缩放与等价变换]]。

## 1. 快照与读取范围

| 快照 | 固定版本 | 读取范围 |
|---|---|---|
| mit-han-lab/smoothquant | `c61476d728e42ae0d8a35e7e78494edcac3237b5` | `smoothquant/smooth.py`、`calibration.py`；`act_scales/` 目录内容（快照中仅有 README） |
| guangxuan-xiao/torch-int | `65266db1` | `torch_int/nn/linear.py` 的 INT8 线性接口 |
| sgl-project/sglang | `2f730e29` | `srt/layers/quantization/w8a8_int8.py` |

未读取：`fake_quant.py`、`export_int8_model.py`、`opt.py` 的逐行实现、量化模型的示例脚本；未运行任何环节。

## 2. 原仓库：三个组件

仓库把平滑流程拆成三步，与我们的方法页描述一致，但代码里的边界更清楚：

1. **校准（`calibration.py`）。** `get_act_scales(model, tokenizer, dataset_path, num_samples=512, seq_len=512)` 对目标线性层的输入挂 hook，`stat_tensor` 用 `torch.max` 逐通道累积最大值（跨样本取大）。**统计量是最大绝对值，不是均值**——这是与 [[awq|AWQ]]（平均绝对激活）最容易被混同的差异，也是两者缩放公式不可互换的原因。
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

- **多消费者取最大值。** `fcs` 是列表（例如 Q/K/V 共享同一归一化输出），`weight_scales` 先各自求逐输入通道最大值，再逐通道取最大。这与 [[diagonal-scaling-equivalent-transform|等价变换页]] 记录的「共享归一化的多个 FC 必须用同一组缩放」是同一约束的代码形态。
- **RMSNorm 分支不处理偏置。** `smooth_ln_fcs_llama_like` 只做 `ln.weight.div_(scales)`，没有 `ln.bias`；对带偏置的 LayerNorm 版本则会一起除，否则平移项不会被正确吸收。
- **下限 `1e-5` 出现两次**：一次在权重的逐通道最大值上（避免整列为零），一次在缩放本身上。零激活通道与零权重通道都要保护。
- **缩放只写到「输入列」**：`fc.weight.mul_(scales.view(1, -1))` 的形状说明缩放沿输入维广播，与「逐输入通道缩放」的定义一致。

快照中的 `act_scales/` 目录只有 README，说明**预计算的激活统计没有随快照保存**；复现时需要自行跑校准。

## 4. 执行侧：INT8 线性接口

torch-int 的 `W8A8B8O8Linear` 是这条路径的执行端：

- 构造参数为 `in_features, out_features, alpha=1.0, beta=1.0`，其中 `alpha/beta` 是权重与激活的尺度系数；
- `from_float(module, input_scale, output_scale)` 负责转换：权重与偏置分别用 `quantize_per_tensor_absmax` 量化（**逐张量**，不是逐通道）；
- 因此「逐通道权重」这一说法在 SmoothQuant 论文里成立，而这条特定实现的加载接口使用的是逐张量权重尺度——两者需要在具体后端上分别核对，不能互相替代。

## 5. 跨引擎：SGLang 的 w8a8_int8

SGLang 提供 `w8a8_int8` 方法：内核走 `sgl_kernel.int8_scaled_mm`（参数含 `scales_a`、`scales_b`），配置类实现 `get_config_filenames`、`get_quant_method`、`is_layer_skipped`。最值得记录的是一处**否定性证据**：

```python
def get_scaled_act_names(self) -> List[str]:
    return []
```

该接口在其它框架里用于声明「哪些激活需要额外缩放」（即平滑的运行时钩子）。本快照里它返回空列表，说明**这个后端不做在线平滑**：平滑必须已经离线完成、并被吸收进归一化参数与线性权重，引擎只需要做 INT8 计算。这与 [[quantized-matmul-scaling-execution|执行路径]] 的分类一致——引擎侧是「W8A8 整数矩阵乘 + 外维缩放」，平滑属于量化器侧。

## 6. 跨引擎：TensorRT-LLM 的 int8_sq

TensorRT-LLM 的量化配置里把这件事说得更直接：`int8_sq` 的描述是「权重先被平滑（smoothed）再按通道量化为 INT8，激活范围按张量校准」。也就是说**平滑是量化工具链的一部分**，与引擎解耦；这一点与我们上一节在 SGLang 看到的空钩子互相印证，见 [[quantized-llm-deployment-backends|部署框架与后端支持]] 的格式表。

## 7. 论文-代码对齐（本页补充）

- 方法页记录的公式 $s_j=a_j^\alpha/b_j^{1-\alpha}$ 在代码里逐字出现，但 `a_j` 的取法被实现固定为「所有共享消费者上的逐通道最大激活」——论文没有规定共享归一化时如何进行这个归约；
- 实现额外引入两处 `clamp(min=1e-5)`，属于数值保护而非算法部分；
- `alpha` 在函数签名上默认 `0.5`，但论文的模型特定取值由调用方传入（`smooth_lm(model, scales, alpha)`），默认值不等于论文配置；
- 「逐通道权重」是论文设置，而 torch-int 的加载接口用逐张量尺度；引用时应写明是哪一侧。

## 8. 可复用的实现要点

1. 平滑统计用最大绝对值，逐输入通道累积；不要与平均绝对激活互换；
2. 共享归一化的所有消费者必须用同一组缩放，权重侧取逐通道最大值；
3. 归一化带偏置时，权重与偏置一起除；RMSNorm 则只除权重；
4. 通道与缩放都要设数值下限，并记录下限值；
5. 明确平滑发生在离线还是在线：若目标引擎的「需要缩放的激活」清单为空，则必须先离线融合；
6. 权重尺度是逐通道还是逐张量，随执行接口变化，必须按后端核对。

## 9. 验证状态与待验证

- 全部结论为 code-read；未运行校准、平滑、导出或任何 INT8 内核，未复现论文精度。
- `fake_quant.py`、`export_int8_model.py` 与 `opt.py` 的 INT8 注意力路径未逐行审查；本页不评价导出模型的数值正确性。
- SGLang 侧只核对了方法名、内核入口与 `get_scaled_act_names` 的返回值；未追踪 `int8_scaled_mm` 的尺度语义（逐张量还是逐通道），也未核对张量并行下的切分方式。
- 预计算的 `act_scales` 未随快照保存，复现校准需要自行准备数据；本页没有运行 `get_act_scales`。
