---
title: GPTQ 的实现核对：校准、导出协议与服务内核选择
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
  - raw/repositories/2026-09-21/gptq/source/llama.py
  - raw/repositories/2026-09-21/gptq/source/quant_cuda.cpp
  - raw/repositories/2026-09-21/gptq/source/quant_cuda_kernel.cu
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/layers/quantization/__init__.py
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/layers/quantization/auto_gptq.py
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/layers/quantization/utils/gptq_utils.py
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/kernels/linear/__init__.py
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/kernels/linear/mixed_precision/marlin.py
  - raw/repositories/2026-09-21/sglang/source/python/sglang/srt/layers/quantization/__init__.py
  - raw/repositories/2026-09-21/sglang/source/python/sglang/srt/layers/quantization/gptq/gptq.py
  - raw/repositories/2026-09-21/sglang/source/python/sglang/srt/layers/quantization/gptq/schemes/gptq_linear.py
  - raw/repositories/2026-09-21/sglang/source/python/sglang/srt/layers/quantization/gptq/schemes/gptq_marlin.py
  - raw/repositories/2026-09-21/sglang/source/python/sglang/srt/hardware_backend/gpu/quantization/gptq_kernels.py
  - raw/repositories/2026-09-21/sglang/source/python/sglang/srt/layers/quantization/marlin_utils.py
  - raw/repositories/2026-09-21/sglang/source/python/sglang/jit_kernel/gptq_marlin_repack.py
  - raw/repositories/2026-09-21/sglang/source/python/sglang/jit_kernel/csrc/gemm/marlin/gptq_marlin_repack.cuh
  - raw/repositories/2026-09-21/sglang/source/sgl-kernel/csrc/gemm/gptq/gptq_kernel.cu
updated: 2026-09-22
---

# GPTQ 的实现核对：校准、导出协议与服务内核选择

GPTQ 的二阶补偿决定量化后的权重，导出协议决定服务框架能否解释它，加载后的分派再决定实际内核。**原始 GPTQ 的量化评估成功，不代表其保存入口能产出 vLLM 或 SGLang 可直接加载的 checkpoint。**

算法目标与推导见 [GPTQ](../methods/gptq.md)。本页沿 Llama 校准、原始 3-bit 保存、服务格式及内核选择展开；与 [AWQ 实现](awq-implementation.md) 对照时，应比较完整格式和执行条件。

## 1. 固定版本与范围

| 代码 | 固定 commit | 本页核对范围 |
| --- | --- | --- |
| IST-DASLab/gptq | `2d65066eeb06a5c9ff5184d8cebdf33662c67faf` | Llama 校准、网格与补偿、3-bit 打包及两个 CUDA GEMV |
| vllm-project/vllm | `568afb3a13806beb53bb2e6bd518269357b237c0` | Dense GPTQ 配置、混合精度后端选择、Marlin 加载与调用 |
| sgl-project/sglang | `2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc` | Dense 原生/Marlin、TP 元数据、repack、原生 CUDA host 分派 |

这是固定快照的代码核对，不是最新版本支持表。未运行模型校准、checkpoint 加载或 GPU 内核；其他模型适配、MoE 和非 CUDA 后端不能据此外推。

## 2. 从校准数据到补偿后的浮点权重

### 2.1 Llama 的实际顺序

`llama.py:llama_sequential` 捕获首个 decoder block 的输入，把当前块放到 GPU，对线性层输入挂 hook，逐样本重放。开启 `true_sequential` 后，按 `[k,v,q] → [o] → [up,gate] → [down]` 四组处理：前组已经量化，后组收集到的输入随之改变。

完成全部子组后，用量化后的当前块重新计算输出，作为下一块输入。因此同时有块内子组顺序与块间量化输入传播。`fasterquant` 最终把**反量化后的浮点值**写回 `layer.weight`；此时仍可用浮点 Linear 评估，还没有压缩部署格式。

### 2.2 Hessian 归约的单位

`GPTQ.add_batch` 将 $(B,S,K)$ 输入展平、转置成 $(K,BS)$，累积外积。`nsamples` 按传入 batch 的 $B$ 增加，**不是按 token 数 $BS$ 增加**。代码归约可写为

$$
H=\frac{2}{\sum B}\sum_{\text{传入样本及其 token}}xx^{\mathsf T}.
$$

函数本身没有 valid-token mask，也不做均值中心化；上游样本构造决定 padding 是否参与。不能把它写成有效 token 平均的协方差。固定校准集的全局常数缩放通常在相对阻尼和补偿比值中抵消，但序列长度分布、mask 和样本权重的变化不只是常数变化。

### 2.3 网格、分组与补偿

`quant.py:Quantizer.find_params` 估计 scale/zero，逐输出通道取包含零的范围；对称模式用中点零点，非对称模式用 $\operatorname{round}(-x_{\min}/s)$。可选 MSE 模式枚举收缩比例并按误差的 `norm` 次幂选网格；**Llama 入口传入 `mse=False`**。类默认 `sym=True`，但 CLI 的 `--sym` 默认未开启，最终以调用参数为准。

`gptq.py:fasterquant` 的具体顺序为：

1. 对 $H$ 对角为零的输入列清零权重并修正对角。
2. 可选 act-order 按对角重要性重排权重列和 $H$。
3. 加 `percdamp × mean(diag(H))` 阻尼，Cholesky 求逆，再取逆矩阵的上三角 Cholesky 因子。循环中的 `Hinv` 指该因子，不能直接当作裸 $H^{-1}$。
4. 按默认 128 列的计算块处理；逐列量化后以误差除以因子对角更新块内剩余列，块结束统一更新后续块。
5. 将结果逆置换回原输入坐标并写回浮点权重。

`blocksize` 与 `groupsize` 不是同一个块。动态分组在当前处理顺序的组边界重新估计网格；`static_groups=True` 则预先为原始连续列组估计，act-order 后仍按原始列位置找组。动态分组导出可能需要记录原输入列属于哪个网格，这正是服务格式 `g_idx` 的职责。

## 3. 原始保存入口的限制

`llama.py` 的 CLI 支持多种量化位宽，但 `--save` 固定调用 `llama_pack3`，替换成 `Quant3Linear` 后保存 state_dict。它没有按位宽选通用打包器，也不生成完整服务配置。

`fasterquant` 没有返回全部分组尺度与索引，保存字典保留的是每层 Quantizer 对象；动态分组后只有当前/最后一组参数。`Quant3Linear` 每输出通道只存一组尺度/零点，没有通用分组 `g_idx`。因此“能评估分组量化”不等于“该入口能正确导出分组模型”。

### 3.1 3-bit 位流

设 PyTorch 权重 $W\in\mathbb R^{N\times K}$。`Quant3Linear.pack` 保存正的缩放零点 $zs=z\,s$，整数化后转置为 $(K,N)$：

$$
q_{k,n}=\operatorname{round}\left(\frac{W_{n,k}+zs_n}{s_n}\right),
\qquad
\widehat W_{n,k}=s_nq_{k,n}-zs_n.
$$

| buffer | 形状 | 含义 |
| --- | --- | --- |
| `qweight` | $(3K/32,N)$，int32 | 沿输入轴的位容器 |
| `scales/zeros` | $(N,1)$ | 浮点尺度、已乘尺度的正零点 |
| `bias` | $(N,)$ | 输出初始化值；无原始 bias 时为零 |

32 个 3-bit 编码占三个 32-bit 字。第一字装 q0–q9 与 q10 的低 2 bit；第二字装 q10 的高 1 bit、q11–q20 与 q21 的低 1 bit；第三字装 q21 的高 2 bit、q22–q31。q10、q21 跨字，不是每字十个的独立容器。

打包器依赖编码已在 $[0,7]$，没有对普通编码额外 clamp/mask；送入 4-bit 编码会污染邻近编码。更不能只改 metadata 把它变成服务框架的 4-bit GPTQ。

### 3.2 原始 CUDA 是专用 GEMV

`Quant3Linear.forward` 要求 `x.shape[-1] == x.numel()`，只接受一个展平 token。`quant_cuda.cpp` 的 device guard 不检查完整 shape/dtype 契约。

`quant_cuda_kernel.cu` 的块宽为 256、打包行块高为 24，即每块沿 $K$ 消费 256 个编码。每线程负责一个输出通道，shared memory 复用输入，拆码后累加；不同 $K$ tile 用 atomicAdd 汇入已由 bias 初始化的输出。这里是 CUDA core GEMV，没有 Tensor Core MMA。

grid 向上取整但内核没有完整尾部保护；从访问式可推出正常安全调用至少需要 $K,N$ 都按 256 对齐，打包器只满足 $K$ 的 32 对齐还不够。Python 中“1024 block”的注释不能替代实际常量。

普通分支将输入转 FP32；快速分支用 FP16 输入、half2 和共享反量化表，每 32 个权重的一段先 half2 累加，再转入 FP32 总和。二者舍入不同，不能宣称逐 bit 等价。快速分支按 FP32 指针访问尺度、零点与输出，因此也不能对整个模块调用 `.half()` 后假定契约不变。

## 4. 服务格式与 vLLM 加载

对于常见 Dense 4/8-bit 标准打包输入，令 $p=32/b$，$K_p,N_p$ 是 TP 局部维度，$G$ 是当前 rank 保留的尺度组数：

| 参数 | 加载前形状 | 语义 |
| --- | --- | --- |
| `qweight` | $(K_p/p,N_p)$，int32 | 沿输入维打包 |
| `scales` | $(G,N_p)$ | 组、输出通道尺度 |
| `qzeros` | $(G,N_p/p)$，int32 | 沿输出维打包，零点解释依赖格式版本 |
| `g_idx` | $(K_p,)$，int32 | 原输入列到尺度组的映射 |

这来自服务端 `create_weights`，不是原始 3-bit 类的产物。bits、group_size、desc_act、sym、参数名和格式版本也必须对齐。尤其 GPTQ v1/v2 零点约定不能仅凭 shape 判断：SGLang 配置与 scheme 虽记录 `checkpoint_format/use_v2_format`，本页所读普通 GPU kernel 包装没有据此转换零点，故不把字段存在当作 v2 兼容验证完成。

### 4.1 vLLM 的配置名不等于内核名

固定快照的 `gptq`、`gptq_marlin`、`auto_gptq` 均映射到 `AutoGPTQConfig`。其 TYPE_MAP 只列对称 4/8-bit；非对称和 3-bit 不因命名为 GPTQ 就被此入口接受。

当 `desc_act=True` 且 `group_size=-1` 时配置归一为 desc_act=False，因为一个输出通道只有一组尺度，推理不再需要跨组映射。这不否定量化处理顺序影响舍入/补偿。逐层动态覆盖还可改位宽、分组或跳过模块，应与全局配置一起看。

Dense 内核交由 `choose_mp_linear_kernel` 选择。CUDA 候选按顺序含 CutlassW4A8、Machete、AllSpark、Marlin、Conch、Exllama、TritonW4A16、Humming。选择器先按 `--linear-backend` 过滤，再检查禁用项、capability 和 `can_implement(config)`，取首个通过者，无候选则报错。候选存在不表示适用，**`gptq_marlin` 配置名在此快照也不保证最后调用 Marlin**。

### 4.2 选中 Marlin 后的处理

`mixed_precision/marlin.py` 整理参数轴、可选 padding、权重 repack、尺度及 bias 重排，再在 apply 中传入组索引、排序索引与 workspace。无 act-order 时允许部分 padding 情形，有 act-order 时要求更严格的原始形状；TP 分组跨界也限制 padding。因此不能把此快照描述成“原生与 Marlin 两条固定路线”。

## 5. SGLang：显式分支与运行时 fallback

SGLang 分别注册 GPTQConfig 与 GPTQMarlinConfig。对尚未标为 Marlin 格式的 checkpoint，兼容性检查通过且用户选择为空、marlin 或 gptq_marlin 时可以提升到 Marlin；显式 `gptq` 保留原生。

| 路线 | 配置声明 | 路径 |
| --- | --- | --- |
| 普通 CUDA GPTQ | 2/3/4/8-bit，FP16，配置 min capability 60 | scheme → GPTQLinearKernel → gptq_gemm |
| GPTQ Marlin | 对称 4/8-bit，FP16/BF16，配置 min capability 80，组 -1/32/64/128 | repack、尺度重排 → Marlin GEMM |

配置声明仍须满足实际布局、设备和内核形状条件。原生 scheme 在分组、row-parallel 且 desc-act 时关闭 `use_shuffle`；否则加载后可调用 gptq_shuffle，将 g_idx 转成排序索引或空张量。调用位置是 `process_weights_after_loading`，不是旧注释中的首次 forward。

原生 `gptq_kernel.cu` 的 host 分派继续按展平 token 数 $M$ 选择：

- use_shuffle=True：8-bit 在 $M>24$、其他位宽在 $M>50$ 时先重建 FP16 权重再调用 cublasHgemm；较小时走融合量化 matmul。
- use_shuffle=False：2/3-bit 总是重建；4/8-bit 在 $M>8$ 时重建，否则走替代融合路径。

这是代码阈值，未做本地性能测量。临时 FP16 权重意味着 checkpoint 低 bit 与运行中始终保持低 bit 权重流不能画等号。

## 6. act-order、组排序与物理重排

三种操作有关联但职责不同：

1. **量化顺序**改变补偿处理次序；结束后权重回原输入坐标。
2. **组排序**处理动态分组恢复原坐标后交错的组。Marlin 排序 g_idx，repack 用 g_idx_sort_indices 读取原列，执行接口同时接收排序信息，保持输入、权重与尺度的对应。
3. **物理 tile 重排**把编码改为内核的线程访问顺序。即使没有组排序也要做。Python repack 的 perm 必传但可为空，输出 int32 容器形状为 $(K/16,N\times16/p)$。

教学上，一致置换满足 $XW^{\mathsf T}=(XP^{\mathsf T})(WP^{\mathsf T})^{\mathsf T}$；只改权重或错配组尺度会改变结果。这一恒等式不代表 CUDA 已验证。tile 契约见 [权重量化反量化内核](weight-only-dequant-kernels.md)，批处理动机见 [Marlin](marlin-batched-w4a16-gemm.md)。

**TP 的尺度复制。** 两框架 Marlin 均区分全量 $K$ 与局部 $K_p$。act-order 使局部列可能使用任意全局组，故保留全局尺度；group_size=-1 且 row-parallel 时，唯一尺度由全量输入轴定义，也要复制。其他对齐分组才切组轴，不能无条件用 $K_p/g$ 重建元数据。`is_k_full` 则描述 act-order 与 row-parallel 组合下归约维是否完整，不是“是否加载了完整模型”。

## 7. 核对顺序与验证边界

接到 GPTQ 模型，应按以下问题逐级核对：

1. 是浮点反量化权重、原始专用 3-bit state_dict，还是带服务配置的 GPTQ checkpoint？
2. bits、sym、分组、desc_act、格式版本和逐层覆盖是否匹配目标快照？
3. 参数轴、零点约定和 TP 切分是否一致？
4. 实际选中哪个后端，发生了哪些 shuffle/repack/padding，token 数是否触发重建？
5. 再测模型精度、跨引擎输出、prefill/decode 延迟与峰值显存。

本页已核对原始 CUDA GEMV 与 SGLang 原生 host 分派；未逐行覆盖全部位宽 device kernel、所有 vLLM 候选后端或 MoE。网格范围搜索与 Marlin 分组裁剪同属离线工作但目标不同，不以名称相似判为等价。没有模型/GPU 运行证据，不能报告部署复现或性能收益。

### CPU 教学核对

4 项检查覆盖跨 int32 的 3-bit 位流与反量化点积、越界编码污染邻位、组排序的坐标一致性、按样本序列计数的 Hessian 归约。教学形状含 N=7，仅测试 CPU 位流，不满足原始 CUDA 的输出维对齐要求，未用于 GPU 调用。

可复跑的[脚本](../assets/gptq-implementation/check_contracts.py)与[2026-09-22 结果](../assets/gptq-implementation/checks-2026-09-22.json)已保存；运行方式为 `python3 wiki/assets/gptq-implementation/check_contracts.py`，需 NumPy。默认只打印，指定 `--output` 才写结果。这些计算没有导入或执行上游模型/CUDA 代码，不构成 checkpoint 可加载、内核数值正确或性能复现的证据。

设备端的 16×64 repack 坐标、U4B8 位转换、在线输入置换、异步流水和三类跨 CTA 合并，见 [现代 Marlin 的 GPTQ 主循环](weight-only-dequant-kernels.md#10-gptq--现代-marlinrepack-后怎样参与计算)。这也解释了为什么 FP32 MMA 不等于所有临时结果都保留 FP32。

## 来源身份

| 来源 | 固定版本 | 说明 |
| --- | --- | --- |
| [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq/tree/2d65066eeb06a5c9ff5184d8cebdf33662c67faf) | 完整 commit 同 §1 | 原始量化与 3-bit 执行 |
| [vllm-project/vllm](https://github.com/vllm-project/vllm/tree/568afb3a13806beb53bb2e6bd518269357b237c0) | 完整 commit 同 §1 | 配置、选择器与 Marlin |
| [sgl-project/sglang](https://github.com/sgl-project/sglang/tree/2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc) | 完整 commit 同 §1 | 原生、Marlin 与 repack |
