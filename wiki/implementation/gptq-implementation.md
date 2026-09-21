---
title: GPTQ 的实现核对：量化器、打包与两套服务路径
type: implementation
tags:
  - llm
  - weight-quantization
  - data-format
  - kernels
  - serving
sources:
  - raw/repositories/2026-09-21/gptq/source/gptq.py
  - raw/repositories/2026-09-21/gptq/source/quant.py
  - raw/repositories/2026-09-21/gptq/source/quant_cuda.cpp
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/layers/quantization/utils/gptq_utils.py
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/layers/quantization/auto_gptq.py
  - raw/repositories/2026-09-21/sglang/source/python/sglang/jit_kernel/gptq_marlin_repack.py
  - raw/repositories/2026-09-21/sglang/source/python/sglang/srt/layers/quantization/gptq/schemes/gptq_marlin.py
updated: 2026-09-17
---

# GPTQ 的实现核对：量化器、打包与两套服务路径

GPTQ 的论文讲的是「固定一个权重后怎样补偿剩下的」，而工程上要回答的是另外三件事：**校准与量化参数在哪里算出来、权重被打成什么位布局、服务框架拿到这份 checkpoint 后认不认。** 后两件尤其容易出问题——和 [AWQ 的实现核对](awq-implementation.md) 一样，跨引擎能否直接加载取决于打包约定，而不是补偿公式。

算法目标、二阶推导与证据边界见 [GPTQ](../methods/gptq.md)；Marlin 使用改造后的 GPTQ 格式并把它变成批处理内核，见 [Marlin](marlin-batched-w4a16-gemm.md)。

## 1. 快照与读取范围

| 快照 | 固定版本 | 读取范围 |
|---|---|---|
| ist-daslab/gptq | `2d65066e`（2026-06 快照） | `gptq.py`、`quant.py`、`quant_cuda.cpp/.cu` 的存在与结构 |
| vllm-project/vllm | `568afb3a` | `layers/quantization/auto_gptq.py`、`utils/gptq_utils.py` |
| sgl-project/sglang | `2f730e29` | `srt/layers/quantization/gptq/`、`jit_kernel/gptq_marlin_repack.py` |

未读取：`quant_cuda_kernel.cu` 的内核实现、模型适配文件（llama/opt/bloom）逐行、`zeroShot/` 评测脚本。本页不评价内核性能。

## 2. 原仓库：两类组件

仓库由两层构成，职责边界清楚：

**量化参数估计（`quant.py` 的 `Quantizer`）。** `find_params(x, weight=...)` 决定 scale 与 zero：先取 `xmin/xmax`；对称模式置 `zero = (maxq+1)/2`，非对称模式用 `round(-xmin/scale)`；当 `mse=True` 时按 `maxshrink × grid` 枚举收缩比例 $p$，对每个候选重新算 scale/zero 并计算量化误差的 $L_\mathrm{norm}$ 次幂，取最小者。**注意：这一步与二阶补偿无关**，它只决定网格；补偿发生在 `gptq.py` 里。

**逐层量化与补偿（`gptq.py` 的 `GPTQ`）。** `fasterquant` 的签名本身就是实现要点清单：

```text
fasterquant(blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False)
```

- 阻尼按**相对量**取：`damp = percdamp * mean(diag(H))`，再 `H[diag,diag] += damp`——即阻尼随权重/Hessian 尺度缩放，不是一个绝对常数（与 [层输出重构](../theory/layer-reconstruction-second-order-compensation.md) 里「按平均对角值比例」的记录一致）；
- 主循环按 `blocksize` 切块，块内更新、块间延迟（方法页 §4.2 已解释这一组织的动机）；
- `actorder=True` 时按 Hessian 对角排序重排列，并在结束时把补偿结果**逆置换回原顺序**（`gptq.py` 末尾处理）——顺序是内部实现细节，导出权重必须回到原坐标；
- `static_groups=True` 时提前固定分组网格，而不是每次量化时重算。

## 3. 产物格式：打包是自定义的

`Quant3Linear.pack(linear, scales, zeros)` 展示了 GPTQ 自己的位布局：

1. 先把 `zeros * scales` 合成 `self.zeros`（与 AWQ 的实现同构，便于反量化时少一次运算）；
2. 整数化后**转置**（`intweight.t()`），使打包沿输入维进行；
3. 用一段显式的手写循环把 **10 个 3-bit 值装进一个 int32**，并把第 10 个值的剩余位借到下两个 int32 中（`<< 30`、`>> 2`、`<< 31` 等位移与掩码）。

第 3 步说明 GPTQ 的 3-bit 权重**不是逐元素可寻址的规整布局**，而是为某个内核定制的位流。这就是「为什么需要转换脚本」的根源：任何别的内核要读它，必须先解包再按自己的布局重排。

此外，`g_idx`（分组索引/激活重排相关）与 `desc_act` 是否出现在 checkpoint 里，取决于量化时的选项；服务框架据此决定加载路径（下一节）。

## 4. 跨引擎：vLLM

vLLM 的 `auto_gptq.py` 与 `gptq_utils.py` 做了两件实现层的事：

- **配置解析与覆盖。** 从 checkpoint 的配置文件名（`get_config_filenames`）读取；`gptq_utils.override_config` 用 `get_dynamic_override` 取 `desc_act` 与 `sym`，再以 `(weight_bits, is_sym)` 查 `TYPE_MAP` 得到量化类型——也就是说**位宽与对称性必须在加载时被识别**，不能假定默认值。
- **一个显式的退化规则。** `AutoGPTQConfig.__init__` 中：`if desc_act and group_size == -1` 则把 `desc_act` 关掉（该分支带警告）。逐通道量化与激活重排不能同时生效，属于加载期必须处理的组合。

加载之后 vLLM 仍有 GPTQ 原生内核与 `gptq_marlin` 两条路径（对应部署页里并列的方法名），后者需要一次 repack。

## 5. 跨引擎：SGLang

SGLang 的 `gptq/` 下同时有原生与 marlin 两套 scheme。marlin 路径的实现细节更能说明部署复杂度：

- `has_g_idx=self.quant_config.desc_act`——**是否带 g_idx 直接决定 repack 的行为**；
- `marlin_repeat_scales_on_all_ranks(...)` 与 `scales_and_zp_input_dim`／`scales_and_zp_size = input_size // group_size` 的计算：在张量并行下，尺度张量要按**分片后的输入维**切分或复制，`input_size_per_partition` 与全量 `input_size` 是两个分支；
- `jit_kernel/gptq_marlin_repack.py` 暴露 `gptq_marlin_repack(b_q_weight, perm, out, size_k, size_n, num_bits)`，其中 `perm` 是必需的输入——marlin 布局要求一个显式置换，它来自 GPTQ 的列顺序与 marlin 的分块规则。

repack 的几何（16 行 tile、pack_factor、awq 与 MoE 变体）与 marlin 内核的模板匹配条件见 [权重量化反量化内核的契约](weight-only-dequant-kernels.md)。

## 6. 论文-代码对齐（本页补充）

方法页已记录论文与后来代码的差别（如算法列块与分组网格不是同一个「块」）。代码核对再补三点：

- **网格搜索有两种**：`Quantizer` 的 MSE 收缩搜索是逐层的范围搜索，Marlin 的格式改造又加了一次分组裁剪阈值搜索（见 Marlin 页 §7）；两者都叫「搜索」，优化对象不同。
- **顺序置换有两处**：`actorder` 的列置换与 marlin 的 `perm` 是不同层次的置换，前者影响补偿顺序并要还原，后者是布局要求。
- **对称性的影响是连锁的**：`sym` 决定 zero 的取值（`(maxq+1)/2`），同时决定加载端选哪个量化类型，进而决定走不走 marlin。

## 7. 可复用的实现要点

1. 分清「网格参数」与「补偿」两类产物：scale/zero/g_idx 可以来自独立估计，补偿只改剩余权重；
2. 记录列置换与还原顺序，导出时必须回到原坐标；
3. 把位布局当成格式的一部分写进文档（GPTQ 的 10 连排布就是反例：论文里不会出现）；
4. 加载端先解析 `sym`、`desc_act`、位宽，再决定内核路径，并处理互斥组合；
5. 张量并行下确认尺度/零点是切分还是复制。

## 8. 验证状态与待验证

- 全部结论为 code-read；未构建、未量化、未加载任何模型，也未验证 `pack` 与任何内核的数值一致性。
- `quant_cuda_kernel.cu` 未逐行审查；本页只说明其存在与在仓库中的位置。
- vLLM 与 SGLang 的路径选择逻辑读取的是各自快照；版本不同时默认值、支持的位宽与是否要求 g_idx 都可能变化。
- 「同一份 GPTQ checkpoint 在 vLLM 与 SGLang 上是否给出相同输出」属于待验证项，本页只核对两侧都做 repack 与置换这一事实。

## 来源身份

下表用于在没有本地资料库时辨识来源；具体论述的章节、公式、图表或代码位置见正文。

| 来源 | 版本或快照 | 说明 |
| --- | --- | --- |
| [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq/tree/2d65066eeb06a5c9ff5184d8cebdf33662c67faf) | `2d65066eeb06a5c9ff5184d8cebdf33662c67faf` | — |
| [vllm-project/vllm](https://github.com/vllm-project/vllm/tree/568afb3a13806beb53bb2e6bd518269357b237c0) | `568afb3a13806beb53bb2e6bd518269357b237c0` | — |
| [sgl-project/sglang](https://github.com/sgl-project/sglang/tree/2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc) | `2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc` | — |
