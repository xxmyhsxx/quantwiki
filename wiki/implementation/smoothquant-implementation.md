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
  - raw/repositories/2026-09-21/smoothquant/source/examples/export_int8_model.py
  - raw/repositories/2026-09-21/smoothquant/source/smoothquant/fake_quant.py
  - raw/repositories/2026-09-21/smoothquant/source/smoothquant/opt.py
  - raw/repositories/2026-09-21/torch-int/source/torch_int/nn/fused.py
  - raw/repositories/2026-09-21/torch-int/source/torch_int/nn/bmm.py
  - raw/repositories/2026-09-21/torch-int/source/torch_int/kernels/linear.cu
  - raw/repositories/2026-09-21/sglang/source/python/sglang/srt/layers/quantization/__init__.py
  - raw/repositories/2026-09-21/sglang/source/python/sglang/srt/layers/quantization/int8_kernel.py
  - raw/repositories/2026-09-21/sglang/source/sgl-kernel/csrc/gemm/int8_gemm_kernel.cu
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/layers/quantization/__init__.py
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/kernels/linear/__init__.py
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/kernels/linear/scaled_mm/cutlass.py
updated: 2026-09-22
---

# SmoothQuant 的实现核对：平滑融合、INT8 接口与离线边界

SmoothQuant 提供离线重参数化，不定义唯一的 checkpoint 格式或运行时。要把算法接到服务端，需要区分**用于平滑的通道统计、量化网格的尺度、整数累加器的重缩放系数**。同样写 W8A8 的两个模型，也可能采用不同粒度、激活策略和注意力路径。

方法与论文配置见 [SmoothQuant](../methods/smoothquant.md)，融合的代数约束见 [对角缩放与等价变换](../theory/diagonal-scaling-equivalent-transform.md)。这里核对官方 OPT 导出，以及 SGLang、vLLM 的 Dense INT8 消费路径。

## 1. 版本与证据范围

| 仓库 | 固定 commit | 本页核对 |
| --- | --- | --- |
| smoothquant | `c61476d728e42ae0d8a35e7e78494edcac3237b5` | 通道统计、平滑、fake quant、OPT INT8 导出与注意力 |
| torch-int | `65266db1eadba5ca78941b789803929e6e6c6856` | Linear、LayerNormQ、BMM 接口及 INT8 Linear CUTLASS 入口 |
| SGLang | `2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc` | w8a8_int8 的 Dense 加载、Triton 动态量化、CUTLASS host 分派 |
| vLLM | `568afb3a13806beb53bb2e6bd518269357b237c0` | compressed-tensors W8A8 scheme、后端选择、CUTLASS 包装 |

不是最新支持表，也未运行这些框架。CPU、MoE、完整 TP 加载和全部设备内核不在本页核验范围。

## 2. 离线链：两次统计服务不同目的

`calibration.py:get_act_scales` 对 Linear 输入挂 hook，将样本/token 维展平，跨 token 和样本累计**逐输入通道最大绝对值** $a_j$。这是平滑统计，不能与 AWQ 的平均绝对激活互换。

`smooth.py:smooth_ln_fcs` 对共享归一化输出的全部消费者取

$$
b_j=\max_{\text{消费者 }l,\ \text{输出 }i}|W^{(l)}_{ij}|,\qquad
s_j=\frac{a_j^\alpha}{b_j^{1-\alpha}}.
$$

代码对 $b_j$ 与最终 $s_j$ 各设 `1e-5` 下限，再将 LayerNorm 的 weight、bias 都除以 $s_j$，每个消费者权重的输入列乘以 $s_j$。RMSNorm 版本只改 weight。共享 Q/K/V 必须共用同一个 $s$；遗漏 LayerNorm bias 或只改一个消费者都会破坏等价关系。下限是实现的数值保护；默认 $\alpha=0.5$ 也不是所有论文实验的统一配置。

`smooth_lm` 按模型结构选择可融合位置。OPT 的前置 Norm→QKV、Norm→fc1 不意味着 out_proj/fc2 也用同样方式融合；Llama/Mistral 的 QKV 与 gate/up 消费关系同样需要按实际图处理。预计算 act_scales 未随这个快照保存，不能声称已具备无需数据的复现材料。

**第二次统计发生在平滑之后。** `examples/export_int8_model.py` 先读 act_scales、以 $\alpha=0.5$ 平滑，再运行 `get_static_decoder_layer_scales`。后者收集每个 Linear 输入/输出的全张量 absmax，除以 127，形成 attention 输入、Q/K/V 输出、out_proj 输入、fc1/fc2 输入的静态标量尺度。前一次决定通道迁移，后一次定义改变后的执行图上的 INT8 网格；不能拿未平滑模型的旧范围直接替代。

## 3. 三种产物必须分清

| 入口 | 实际产物 | 能说明什么 |
| --- | --- | --- |
| `fake_quant.py:W8A8Linear` | 权重和激活舍入到网格后乘回尺度，仍以浮点调用 F.linear | 模拟精度；不是 INT8 权重文件或 INT8 GEMM |
| `export_int8_model.py` 默认分支 | `Int8OPTForCausalLM.from_float`，替换模块并保存真实 INT8 buffers | 官方 OPT/torch-int 专用执行格式 |
| 同脚本 `--export-FT` | 平滑后的浮点模型及 raw_scales 文件 | 供后续转换的中间产物，不是已经完成的 INT8 engine |

fake-quant 的 per_tensor 激活范围从当前传入张量现算，per_token 按最后一维现算，两者都是动态模拟；不能把函数名中的 per_tensor 当成离线静态校准。权重参数可选 per_channel/per_tensor，输出量化也是独立选项。官方真实导出脚本明确构造 OPT；平滑函数支持更多模型不等于这份真实导出脚本支持所有模型。

## 4. 官方 OPT 的真实 INT8 图

### 4.1 Linear 的尺度在哪里用

torch-int 的 `W8A8B8O8Linear.from_float` 对权重和 bias **各做逐张量量化**，保存 INT8 weight/bias 和浮点系数

$$
a=\frac{s_xs_w}{s_y},\qquad b=\frac{s_{\rm bias}}{s_y}.
$$

若整数累加器 $C=q_xq_w^{\mathsf T}$，输出网格由 $aC+bq_{\rm bias}$ 舍入得到。这里 $a$ 不是平滑迁移强度 $\alpha$。论文 v7 表 2 的 O1/O2/O3 使用逐张量权重，后续模型表 7 使用逐通道；这条实现不能直接套到所有论文设置。

`torch_int/kernels/linear.cu:linear_a8_w8_b8_o8` 读取 row-major A、column-major B，以 INT32 累加；在编译 `CUDA_ARCH>=800` 分支用 Sm80 TensorOp、CTA $256\times128\times64$、MMA $16\times8\times32$，epilogue 用 FastLinearCombinationClamp 输出 INT8。Sm75 是另一个 TensorOp 分支，Sm70 是 SIMT。其选择基于编译宏，仍要通过 CUTLASS `can_implement`；不是仅凭算子名就保证所有输入形状支持。

### 4.2 注意力与残差不全是 INT8

`smoothquant/opt.py:Int8OPTAttention.from_float/forward` 展示完整衔接：

1. `LayerNormQ` 先把已平滑的 Norm 仿射参数除以静态输出尺度；forward 用浮点 layer_norm 后 round、clamp 到 [-128,127]、转 INT8。该 Python 类没有因为名字含 Q 就自动使用单个融合 CUDA Norm 内核。
2. Q/K/V 投影采用 W8A8B8O8。Q 的 $1/\sqrt{d}$ 系数同时融合进 Q 投影权重/bias 和 Q 输出尺度，不能只改其中一个。
3. QK BMM 的系数为 $s_qs_k$，输出 FP32；mask 与 softmax 在浮点域进行。
4. 概率乘 127 后 round、转 INT8，PV BMM 用 $(s_v/127)/s_{\rm out}$ 作为输出缩放，产出 out_proj 所需的 INT8 输入。
5. out_proj 和 fc2 用 W8A8BFP32OFP32 输出浮点，再转 residual dtype 相加；fc1 用带 ReLU 的 INT8 输出层。

该注意力缓存保存已经投影的 INT8 K/V，并通过拼接 past_key_value 扩展。它不是 vLLM/SGLang 的分页 KV 协议；Linear 格式兼容也不能推出此注意力与服务端 KV cache 兼容。INT8 主体、省内存和端到端吞吐需分别验证。

## 5. SGLang：动态逐 token W8A8 的具体契约

注册名 `w8a8_int8` 对应 `W8A8Int8Config`，不是执行 SmoothQuant 的校准入口。普通 Dense GPU 分支由 `W8A8Int8LinearMethod` 创建：

| 参数 | 加载前 | 加载后/运行时 |
| --- | --- | --- |
| weight | INT8 $(N_p,K_p)$ | 转置为 $(K_p,N_p)$，保留 column-major stride |
| weight_scale | FP32 $(N_p,1)$ | 逐输出通道，与该输出分片一起加载 |
| activation | FP16/BF16 输入 | 动态 INT8 $q_x$ 与 FP32 $(M,1)$ 尺度 |

`int8_kernel.py:per_token_quant_int8` 要求 contiguous 输入，每个 Triton program 处理一行：以 FP32 取 absmax、下限 `1e-10`、除 127 得尺度，用 CUDA libdevice round 后转 INT8。内部还可计算原始浮点输入和，普通 W8A8 路线不启用该选项，[QoQ](qserve-implementation.md) 会用它。

随后 `int8_scaled_mm` 计算

$$
Y_{m,n}\approx s_{x,m}s_{w,n}\sum_k q_{x,mk}q_{w,kn}+b_n,
$$

输出 dtype 跟随输入。尺度不沿 $K$ 改变，因此能在完整 INT32 累加后按输出行、列缩放。这里每次前向现算的是**量化尺度**，不是在线重新估计或学习平滑 $s_j$。

### 下层检查比配置类更严格

`sgl-kernel/csrc/gemm/int8_gemm_kernel.cu` 要求 A/B 都是 2D INT8，A 的末维 stride 为 1，B 的首维 stride 为 1，$K$ 是 16 的倍数，$N$ 是 8 的倍数；两组尺度为 contiguous FP32，数量分别为 $M,N$，bias dtype 必须与输出一致。

配置声明 FP16/BF16、min capability 75，但底层 **SM75 只接受 FP16 输出**。SM80 系列使用 INT8 Tensor Core 与 INT32 累加；SM86/89 走针对较小 shared memory 的 shape 表；SM90 在 CUDA 12+ 编译条件下走 CUTLASS 3，否则回到 CUTLASS 2 的分支。此文件只列 75–89 与恰为 90 的设备分派，其他 capability 报未实现；不能把配置的最低值解释成此算子支持所有更高架构。

例如 SM80 的 $M\le16,N\le4096$ 分支用 CTA $16\times64\times128$、6 stages，$N>4096$ 改 5 stages。小 $M$ 与大 $M$ 的 tile/pipeline 不同，W8A8 标签并不能说明实际占用与速度。设备主循环、量化舍入与输出 visitor 继续见 [W8A8 算子](w8a8-quantization-gemm-kernels.md)，未做性能实测。

`get_scaled_act_names()` 返回空列表，仅表示该接口没有额外声明缩放，不能证明模型已平滑或上游参数匹配。

## 6. vLLM：通过 compressed-tensors 表达 W8A8

该 vLLM 快照没有独立 `smoothquant` 注册项。已核对的消费入口是 `compressed-tensors`：配置选择器匹配静态 tensor 激活或动态 token 激活，权重为静态、对称 8-bit，策略可为 tensor/channel；激活可对称或非对称。相应 scheme 为 `CompressedTensorsW8A8Int8`。

scheme 创建 INT8 $(N_p,K_p)$ weight；channel 权重尺度为 FP32 $(N_p,1)$，tensor 策略则按融合前的逻辑投影保存尺度。静态输入额外保存 input_scale，非对称静态输入还有 zero point。动态输入不从 checkpoint 读取一个固定激活尺度。

`init_int8_linear_kernel` 根据配置选择 CUDA 候选 CUTLASS、Triton、Humming。选中 `CutlassInt8ScaledMMLinearKernel` 后：

- 转置权重；融合 QKV/MLP 的逐张量权重尺度展开成对应输出通道尺度。
- 静态对称输入以最大 input_scale 统一；静态非对称输入先合并范围再生成统一尺度/零点。
- 非对称输入预先求 $\sum_kq_{w,kn}$；静态情况再乘固定输入零点。前向调用 `scaled_int8_quant`，按是否存在零点选择 `cutlass_scaled_mm` 或 `cutlass_scaled_mm_azp`，补偿 $z_x\sum_kq_w$ 项。

这证明存在消费该 W8A8 表示的路径，**不证明官方 OPT/torch-int 导出可以直接加载**。二者参数名、尺度粒度、bias 表示、激活静态性及 attention 图均不同。可行接入需保留正确融合后的浮点图，并用目标格式重新导出/转换；本页未实现或运行该转换。vLLM 的激活量化设备函数与 CUTLASS 输出校正见 [W8A8 算子](w8a8-quantization-gemm-kernels.md)；其中明确区分两个框架的依赖版本及主循环核验范围。

## 7. 如何判断链路是否闭合

需要同时满足：平滑前通道统计与全部消费者一致；平滑后按目标策略校准或动态量化；导出张量、元数据、模型图符合框架 loader；实际设备和 shape 能进入预期内核。只有最后加载成功并与参考图比较输出，才算部署的数值验证。

TensorRT-LLM 的 `int8_sq` 是另一套显式平滑量化工具链，保留在 [部署框架与后端支持](quantized-llm-deployment-backends.md)，不能从 SGLang 的空接口倒推出它的前处理行为。

本页完成上述固定版本代码核对，未运行校准、模型导出、注意力 BMM device kernel、完整 TP 或 GPU benchmark。论文精度与本地代码阅读分别保留；不能把 fake-quant 的 F.linear 结果写成 INT8 加速，也不能把格式中有 weight_scale 当成 SmoothQuant 算法身份证明。

### CPU 教学核对

4 项检查覆盖共享消费者与 LayerNorm bias 的融合、INT8 输出网格系数、逐 token/逐输出尺度的外维乘法、静态 clipping 与动态范围。使用 NumPy float64/整数代数，不模拟 CUDA 的全部 round、饱和和浮点累加细节。

可复跑的[脚本](../assets/smoothquant-implementation/check_contracts.py)与[2026-09-22 结果](../assets/smoothquant-implementation/checks-2026-09-22.json)已保存；运行方式为 `python3 wiki/assets/smoothquant-implementation/check_contracts.py`，需 NumPy。默认只打印，指定 `--output` 才写结果。这些计算没有导入或执行上游模型/CUDA 代码，不构成 checkpoint 可加载、内核数值正确或性能复现的证据。

## 来源身份

下表用于在没有本地资料库时辨识来源；具体论述的章节、公式、图表或代码位置见正文。

| 来源 | 版本或快照 | 说明 |
| --- | --- | --- |
| [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438v7) | `arXiv:2211.10438v7` | 表 2 与表 7 的权重粒度 |
| [mit-han-lab/smoothquant.git](https://github.com/mit-han-lab/smoothquant/tree/c61476d728e42ae0d8a35e7e78494edcac3237b5) | `c61476d728e42ae0d8a35e7e78494edcac3237b5` | — |
| [Guangxuan-Xiao/torch-int](https://github.com/Guangxuan-Xiao/torch-int/tree/65266db1eadba5ca78941b789803929e6e6c6856) | `65266db1eadba5ca78941b789803929e6e6c6856` | — |
| [sgl-project/sglang](https://github.com/sgl-project/sglang/tree/2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc) | `2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc` | — |
| [vllm-project/vllm](https://github.com/vllm-project/vllm/tree/568afb3a13806beb53bb2e6bd518269357b237c0) | `568afb3a13806beb53bb2e6bd518269357b237c0` | compressed-tensors W8A8 与 CUTLASS 包装 |
