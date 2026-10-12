---
title: AQLM 的原始校准、模型转换与 2×8 推理读取
type: implementation
tags:
  - aqlm
  - reconstruction
  - vector-quantization
  - kernels
sources:
  - raw/repositories/2026-09-21/aqlm/source/main.py
  - raw/repositories/2026-09-21/aqlm/source/aq_engine.py
  - raw/repositories/2026-09-21/aqlm/source/src/aq.py
  - raw/repositories/2026-09-21/aqlm/source/src/beam_search_xtx.py
  - raw/repositories/2026-09-21/aqlm/source/src/finetune.py
  - raw/repositories/2026-09-21/aqlm/source/src/modelutils.py
  - raw/repositories/2026-09-21/aqlm/source/finetune.py
  - raw/repositories/2026-09-21/aqlm/source/convert_legacy_model_format.py
  - raw/repositories/2026-09-21/aqlm/source/convert_to_hf.py
  - raw/repositories/2026-09-21/aqlm/source/inference_lib/src/aqlm/inference.py
  - raw/repositories/2026-09-21/aqlm/source/inference_lib/src/aqlm/utils.py
  - raw/repositories/2026-09-21/aqlm/source/inference_lib/src/aqlm/inference_kernels/kernel_selector.py
  - raw/repositories/2026-09-21/aqlm/source/inference_lib/src/aqlm/inference_kernels/dequantization.py
  - raw/repositories/2026-09-21/aqlm/source/inference_lib/src/aqlm/inference_kernels/cuda_kernel.cpp
  - raw/repositories/2026-09-21/aqlm/source/inference_lib/src/aqlm/inference_kernels/cuda_kernel.cu
updated: 2026-10-11
---

# AQLM 的原始校准、模型转换与 2×8 推理读取

本页接续 [AQLM 方法](../methods/aqlm.md) 与 [PV 训练和选码](aqlm-pv-training-and-code-updates.md)，核读固定 commit `e79a896ed6656fe4ed06193d42d004e7d0bbdbb2`。原始校准根据输入二阶矩交替优化码本、尺度和整数码；模型转换改变保存容器；推理读取则必须匹配码本形状、整数表示和所选内核。三者有共同的加性表示，但不是同一优化目标或执行过程。

范围是 `main.py → AQEngine → QuantizedWeight/beam_search_xtx`，所选逐块微调和保存接口，PV 产物的条件转换路径，以及 CUDA 2×8、输出组宽 1、输入组宽 8 的前向读取。源码静态研读没有安装、导入/JIT、编译、模型运行或性能测量；CPU/Numba、Triton、其他 CUDA 格式和全部模型适配器不属于完整核读范围。

## 1. 收集的是 token 加权的输入二阶矩

`main.py:init_aq_engines` 临时包裹待量化的 Linear。包装器在原层前向前把实际输入交给 `AQEngine.add_batch`，同时把当前 Transformer block 的输出写入缓存。三维输入被展平成 token×输入通道，因此 `nsamples` 统计实际输入向量数，不是文档或序列数；没有在 `add_batch` 内另按 attention mask 排除位置。

令每个输入向量为 $x_t\in\mathbb R^d$，目前已有 $n$ 个、新批有 $m$ 个，缓冲保存

$$
G_n=\frac1n\sum_{t=1}^n x_tx_t^\mathsf T,
\qquad
G_{n+m}=\frac{n}{n+m}G_n+\frac1{n+m}\sum_{t=n+1}^{n+m}x_tx_t^\mathsf T.
$$

这对应源码先缩放旧 `XTX`，再把新输入乘 $1/\sqrt{n+m}$ 后累加矩阵乘积。它是未中心化二阶矩，默认用 FP64 累积；没有减去均值，也不是整网任务损失的 Hessian。`init_aq_engines_parallel` 按各设备实际 token 数加权合并局部均值，不能直接等权平均不同大小的数据分片。公共推导与归约条件见[输入统计与二阶重构](../theory/layer-reconstruction-second-order-compensation.md#1.1%20%E4%BA%8C%E9%98%B6%E7%9F%A9%E7%9A%84%E5%BD%92%E7%BA%A6%E4%B8%8E%E5%88%86%E7%89%87%E5%90%88%E5%B9%B6)。

`AQEngine._compute_mse` 对 $W,\widehat W\in\mathbb R^{r\times d}$ 使用

$$
L=\frac1r\operatorname{tr}\bigl((\widehat W-W)G_n(\widehat W-W)^\mathsf T\bigr).
$$

由于 $G_n$ 已除以 token 数，这等于线性层输出误差对 token 和输出通道的平均。多设备输出分片仍除以整层 $r$，汇总时求和。连续优化直接读取 FP64 缓冲；选码入口把它转成码本 dtype，不能把“FP64 累积”写成整个搜索均用 FP64。

## 2. 初始化、连续更新和输入相关选码

`QuantizedWeight.__init__` 先按输出组×输入组×组内坐标重排权重。所选 `scale_nbits=0` 路径以每个输出组的整行组权重范数加 `EPS=1e-9` 初始化一个尺度，归一化后调用 `init_aq_kmeans`。

残差 K-means 逐本聚类当前权重残差，保存中心和各组索引，再减去该本恢复值。PyTorch 路径可抽部分点拟合中心，但仍对所有组分配最近中心。初始化没有使用 `XTX`；激活度量从后续交替优化进入。低位中心值、每输入组尺度及其二次聚类分支与此常规路径分别判断。

`AQEngine.quantize` 的一个 epoch 先用 Adam 更新所有 `requires_grad=True` 的量化参数，随后调用激活相关 beam search 更新整数码。Adam 的 betas 为 `(0, 0.95)` 且启用 AMSGrad；步数、学习率和早停阈值来自入口参数。早停在新 epoch 的第一个连续更新前比较此前 beam search 后的损失，不是每个 Adam 小步均要求严格下降。

`QuantizedWeight.beam_search_update_codes_` 显式传入 `XTX` 时选择 `beam_search_xtx`；[PV 的选码](aqlm-pv-training-and-code-updates.md#4.%20V%20%E6%AD%A5%E7%9A%84%E8%B7%9D%E7%A6%BB%E3%80%81%E5%80%99%E9%80%89%E4%B8%8E%E6%9B%B4%E6%96%B0%E5%8D%95%E5%85%83)没有传它，转入权重 L2 搜索。不能因共用函数名就合并两种度量。

### 一次替换怎样利用整个输入相关性

搜索外循环遍历输入组，内循环遍历码本；给定随机源时打乱这些顺序。不同输出组独立维护候选，同一行的输入组与多个码本则相互耦合。

设当前误差为 $E=W-\widehat W$，将一组中的某本旧码替换为新码，恢复权重的变化为 $\Delta$。$\Delta$ 包含尺度且只在所选输入组有非零项。以下是源码候选评分的等价推导，忽略各候选共用的 $1/r$：

$$
\Delta L=\operatorname{tr}(\Delta G_n\Delta^\mathsf T)
-2\operatorname{tr}(E G_n\Delta^\mathsf T).
$$

`_beam_search_squared_errors` 去掉旧贡献后，计算候选自身的组内二次项及它与完整残差的交叉项；后者读取 `XTX[:, input_group_slice]`，包含其他输入组的影响。只保留对角子块会变成另一个目标。随后在每个旧 beam 内取较优码，再跨旧 beam 保留有限候选；`_beam_search_select_best` 同步重排码，并重新解码候选权重，继续下一坐标。

这不是全部组合的全局穷举。即使候选含旧码、精确算术下局部最优候选可不增，也不能保证浮点评分、整段连续更新或模型 KL 单调下降。源码对循环顺序的解释明确是经验性选择，没有证明或该顺序的消融。

## 3. 块重构与后续 PV 是不同阶段

`main.py:quantize_aq` 为当前子层组收集统计及替换前 block 输出，然后替换 Linear，调用 `src/finetune.py:finetune_groupwise`。其优化变量是 block 内仍可训练的浮点参数，整数码不经梯度更新；`_compute_mse_on_batch` 用 `F.mse_loss` 比较当前 block 输出与缓存目标。

开启 `true_sequential` 时，后续子层组的输入统计和训练目标来自已经装入此前量化子层的当前 block；不是每组都回到全浮点 block。有验证集时则不同：`val_outs` 在子层组循环前计算一次，其目标保持为当前 block 开始量化前的验证输出，输入仍包含此前 block 的量化前缀。逐块微调用这份固定验证目标选择较好参数与判断早停，不能把训练目标的刷新规则扩到验证目标。（`main.py:quantize_aq`；`src/finetune.py:finetune_groupwise`。）

同一 block 完成后，再运行更新后的 block 产生下一块输入，以 `inps, outs = outs, inps` 传递量化前缀状态。这解释了为什么校准统计必须对应实际计算状态。

根目录 `finetune.py` 的整网教师 KL 与 PV 更新另由[已有实现页](aqlm-pv-training-and-code-updates.md)承接。block MSE、层线性输出 MSE 与整网 KL 的对象、归约和离散变量更新方式应分别保留。

## 4. 保存格式、PV 转换与导出条件

| 产物 | 所选保存内容 | 后续接口 |
| --- | --- | --- |
| 原始校准目录 | `0.pth` 等完整 block 对象、block 外 `not_quantized_weights.pt`、入口 `args.pt` | `convert_to_hf.py` 按 block 参数遍历生成 state dict |
| PV 的 `best_model` | 按完整权重名保存 `QuantizedWeight`，另存 `non_quantized_state_dict.pth` | 条件匹配后由 legacy 转换恢复模型，再按 block 保存 |
| HF 风格产物 | `pytorch_model.bin`、带 AQLM 参数及不量化名单的 `config.json`，可选 tokenizer/safetensors | 对应框架必须建立匹配的量化 Linear；配置存在不证明已加载运行 |

`convert_legacy_model_format.py` 的所选 PV 读取器先以 `strict=False` 加载非量化状态，再把具备量化文件的 Linear 替换为训练版 `QuantizedLinear`，要求这条路径无 bias。**在此固定入口中，`--pv_fsdp_dir` 进入该读取器还依赖 `--monkeypatch_old_pickle` 分支**；不能仅凭参数名称推断默认入口会读取 PV 目录。`save_quantized_model` 保存 block 与 block 外参数，却不写 `args.pt`；HF 转换仍需可匹配的原始量化参数元数据。缺少这份材料时，不能把转换目录称为完整可部署包。

`convert_to_hf.py` 从 `args.pt` 读取码本位数、数量及输入／输出组宽，遍历 block 的 `named_parameters`，把浮点参数转为 FP16，将整数参数改为所需的有符号容器，并移除参数名中的 `quantized_weight.`。这条元数据路径要求统一配置与实际层布局相符；不自动解决混合层配置或二次量化的训练表示。

其 `pack_int_data` 对 $b$ 位非负码执行

$$
z_s=\begin{cases}z,&z<2^{b-1},\\z-2^b,&z\ge2^{b-1},\end{cases}
\qquad z=z_s\bmod 2^b.
$$

随后按位数选 int8/int16/int32/int64。8 位码 200 会保存为 int8 的 −56，按模恢复仍为 200。这是有符号表示映射，不是跨元素按任意 $b$ 位紧密打包；例如 6 位码也占一个 int8 容器。训练码的 `IntCodes` 原始字节包装又是另一接口。

HF 风格推理 `QuantizedLinear` 的尺度形状为输出组×1×1×1。本文所选路径因此使用直接浮点码本及逐输出组尺度；训练类允许的每输入组尺度、低位中心值等不能自动视为同样受支持。导出时 FP16 转换还会引入舍入，不能声称恢复权重逐位不变。全部模型适配器、框架替换和 safetensors 重载未运行验证。

## 5. 推理分发与所选 2×8 读取

推理版 `QuantizedLinear` 保存码本形状 $M\times2^b\times g_o\times g_i$、整数码 $N_o\times N_i\times M$ 与逐输出组尺度。其中 $M$ 是码本数量，$b$ 是每本索引位数，$g_o,g_i$ 是输出／输入组宽；沿用 $r,d$ 表示输出／输入维度，$N_o=r/g_o,N_i=d/g_i$ 是组数。第一次 forward 准备两种算子，根据展平后的输入向量数选择：CUDA／CPU 的阈值为 6，XPU 为 256。这个阈值是代码分发条件，不是我们测得的最优 batch。

对 CUDA、两本 256 项、$g_o=1,g_i=8$，`kernel_selector.py` 在小输入时选择 `code2x8_matmat`，较大输入选择 `code2x8_matmat_dequant`。其他配置可进入不同 CUDA、Triton 或通用恢复路径，不能把以下说明扩到所有 AQLM 配置。

### 小输入：查码、乘加、归约，再缩放

`cuda_kernel.cpp:code2x8_matmat` 展平输入，对每个向量调用 matvec，之后执行行尺度乘法和 bias 加法。所选非 Turing 的 `Code2x8MatVec` 将两个小码本装入共享内存，将码存储的原始字节重解释为 `uint8_t`，连续读取每输入组的两个索引。因而 int8 的 −56 对应的字节仍可索引第 200 项，不做负下标访问。

每组先以 half2 或 bfloat162 加两个中心，再与该组输入进行相同低精度的乘加，分量转 FP32 加到线程局部累计值；warp 用 shuffle 归约，结果转回输入低精度 dtype，C++ 层再乘尺度并加 bias。**FP32 局部／warp 累计不表示所有加法与乘法都在 FP32 完成。** 16 字节向量化读取覆盖 8 个双码组；该非 Turing 路径按输入维度除以 64 计算码行步幅，本文限定输入维度为 64 的倍数、连续布局且输入与码本 dtype 匹配。Python 类仅检查组宽整除，不能由此声称任意 8 的倍数或任意步幅均安全支持。

### 较大输入：解码整矩阵，再调用 Linear

`code2x8_matmat_dequant` 先分配未缩放的稠密权重，`Code2x8Dequant` 查两个码本并以低精度加法写出各组；随后 `F::linear` 计算矩阵乘法，最后乘行尺度与加 bias。此路径会产生完整稠密权重临时张量，并非始终直接在压缩权重上计算。

通用 `dequantize_gemm` 则先用模运算恢复非负整数码，经 `embedding_bag` 累加中心并把尺度乘入权重，然后调用 `F.linear`。尺度在权重端或输出端在精确算术下可等价，低精度舍入顺序却不同。不能由共同的加性公式推断各后端逐位一致，也不能将论文的速度数字当作当前代码已测性能。

## 来源身份与剩余范围

- Vahe1994/AQLM（`https://github.com/Vahe1994/AQLM/tree/e79a896ed6656fe4ed06193d42d004e7d0bbdbb2`），固定 commit `e79a896ed6656fe4ed06193d42d004e7d0bbdbb2`。本页各节给出实际文件与函数；此版本与 AQLM v4、PV-Tuning v2 的论文实验分别登记。
- [AQLM 方法](../methods/aqlm.md)解释加性表示、激活相关联合编码和作者实验；[PV 实现](aqlm-pv-training-and-code-updates.md)解释后续整网梯度与 L2 选码。
