---
title: QServe/QoQ 的实现核对：两级编码、离线重排与 SGLang W4A8
type: implementation
tags:
  - weight-quantization
  - activation-quantization
  - data-format
  - kernels
  - serving
sources:
  - raw/repositories/2026-09-21/omniserve/source/scripts/ckpt_converter/checkpoint_converter.py
  - raw/repositories/2026-09-21/omniserve/source/omniserve/modeling/layers/quantized_linear/w4a8_linear.py
  - raw/repositories/2026-09-21/sglang/source/python/sglang/srt/layers/quantization/__init__.py
  - raw/repositories/2026-09-21/sglang/source/python/sglang/srt/layers/quantization/qoq.py
  - raw/repositories/2026-09-21/sglang/source/python/sglang/srt/layers/quantization/int8_kernel.py
  - raw/repositories/2026-09-21/sglang/source/sgl-kernel/csrc/gemm/qserve_w4a8_per_chn_gemm.cu
  - raw/repositories/2026-09-21/sglang/source/sgl-kernel/csrc/gemm/qserve_w4a8_per_group_gemm.cu
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/layers/quantization/__init__.py
updated: 2026-09-22
---

# QServe/QoQ 的实现核对：两级编码、离线重排与 SGLang W4A8

[QServe](../methods/qserve.md) 把 UINT4 权重在主循环内还原为可计算的 INT8，再用 INT8 Tensor Core 与 INT8 激活相乘。实现的关键是：**两级尺度与零点的数值条件、权重和元数据的物理排列、推理调用是否使用同一约定。**

SGLang 的 `qoq` 实现了这条 W4A8 **线性层**路径；这不等于同时接入 QServe 的 SmoothAttention、KV4、分页布局或整套调度系统。vLLM 固定快照的内置注册表没有 `qoq` 条目，不能把 W4A8 这个位宽标签当作格式兼容证明。

## 1. 版本与读取边界

| 仓库 | 固定 commit | 本页实际读取 |
| --- | --- | --- |
| mit-han-lab/omniserve | `02b2925aa6fa3b92b06316a1524b7f38922cd9c8` | Llama checkpoint converter、W4A8 线性类、权重和元数据重排 |
| sgl-project/sglang | `2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc` | QoQ 配置/加载/调用、Triton 激活量化、两类 CUDA GEMM 的解码与 host 分派 |
| vllm-project/vllm | `568afb3a13806beb53bb2e6bd518269357b237c0` | 内置量化注册表 |

这是固定快照的代码核对，未运行模型、converter 或 GPU。未读取上游完整校准器、OmniServe KV4 device kernel 与调度实现，也不把当前 OmniServe commit 当成论文全部实验的原始运行版本。

## 2. converter 消费的已经是量化结果

`scripts/ckpt_converter/checkpoint_converter.py` 明确限制为 Llama，读取上游 LMQuant 的 `model.pt` 与 `scale.pt`，而不是从原始浮点模型重新搜索平滑、旋转和裁剪。每个线性层需要：

- `weight.scale.0`：第一阶段/逐通道尺度；
- 分组时的 `weight.scale.1`：第二阶段组尺度；
- `weight.zero`：最终阶段零点；
- fake-quant 权重及与之匹配的其他模型参数。

脚本将负零点编码按条件加 8，再调用 `W4A8OF16LinearDynamicInputScale.from_linear`；它还复制非投影参数以保留已完成的图变换，最终保存权重并复制配置文件。它没有在此入口重新证明上游误差目标，也没有自动保证复制来的配置含 SGLang 所需的 `wbits/group_size`。

分组分支的 from_linear 先对 $W/s_1$ 取整，再除 $s_2$、加 $z$，检查落在 $[0,15]$ 后转整数容器；第二阶段这里没有重新做完整范围搜索或通用 nearest-round 量化。这与其输入应为**已经落在目标网格上的 fake-quant 权重**有关。把任意浮点 W 和猜测的尺度送入该打包器，不是完整 QoQ 量化算法。

## 3. 两种表示与零点符号

设逻辑权重为 $W\in\mathbb R^{N\times K}$，$u_{n,k}\in[0,15]$ 为无符号 4-bit 编码，$h=\lfloor k/g\rfloor$。

### 3.1 逐通道：零点放到 epilogue

`group_size=-1` 时，权重表示为

$$
\widehat W_{n,k}=s_{1,n}u_{n,k}-(s_{1,n}z_n).
$$

因此保存 `s1_scales=s_1`、`s1_szeros=s_1z`，后者是**正的、已经缩放的零点**。主循环只乘无符号编码与 INT8 激活，最后统一减零点校正。

### 3.2 分组：主循环内恢复 INT8

分组表示为

$$
v_{n,k}=u_{n,k}s_{2,n,h}+b_{2,n,h},
\qquad b_2=-z\,s_2,
\qquad \widehat W_{n,k}=s_{1,n}v_{n,k}.
$$

`s2_zeros` 保存 $b_2$ 的字节表示，不是裸 $z$，也不是 $+zs_2$。与逐通道的 `s1_szeros` 符号恰好不同：前者在主循环**加**，后者在 epilogue **减**。

第二级尺度和零点虽由 torch.int8 buffer 保存，内核把尺度字节用掩码读为无符号值，把零点按补码进行字节加法。必须分别检查无符号中间乘积 $u\,s_2$ 是否不超过 255、最终 $v$ 是否在 [-128,127]；“放进 int8 容器”本身不保证数学范围正确。

论文保护范围 [-119,119] 的动机、原文算术差异与条件推导见 [方法页](../methods/qserve.md#3-渐进式分组量化)。本 commit 的 from_linear 中该保护范围断言已被注释，执行的只是第一阶段 [-128,127] 检查。因此本 converter 不能独立证明最终 INT8 恢复绝不会溢出，上游校准结果仍是前提。

## 4. 权重 shape 相同，字节顺序也可能不同

OmniServe 与 SGLang 的权重容器都呈现为 INT8 $(N,K/2)$，但它不是“每行相邻两个逻辑权重直接拼成一字节”。`from_linear` 对逻辑编码先作：

~~~text
reshape(N/32, 2, 2, 8, K/32, 2, 4, 4)
permute(0, 4, 3, 6, 1, 5, 2, 7)
permute(0, 1, 2, 3, 5, 6, 7, 4)
byte = low_nibble + (high_nibble << 4)
reshape(N/32, K/32, 32, 16) → reshape(N, K/2)
~~~

两个 permute 是依次应用，后者的轴编号基于前者结果。这形成 $32\times32$ tile 中按线程消费次序排列的 nibble 对，不能只对二维权重做 transpose 替代。reshape 本身要求 $N,K$ 都按 32 对齐。

分组元数据从逻辑 $(N,K/g)$ 转为 $(K/g,N)$，每 32 个输出通道再按

~~~text
0, 8, 16, 24, 1, 9, 17, 25, …, 7, 15, 23, 31
~~~

重排。scale 与负缩放零点必须使用同一顺序，否则 shape 正确但每个线程取到的网格错误。相比 [AWQ/GPTQ 的 Marlin repack](weight-only-dequant-kernels.md)，这里的重排在 producer 已完成，consumer 不再补做。

### SGLang 加载契约

`QoQConfig` 接受 `wbits=4`、`group_size=-1/128`，FP16 输入、min capability 80；配置文件名为 `quant_config.json` 或 `quantize_config.json`。Dense 参数如下：

| 参数 | 当前分片的形状/dtype | 内容 |
| --- | --- | --- |
| qweight | $(N_p,K_p/2)$，INT8 | 已完成计算感知重排的 nibble 容器 |
| s1_scales | $(N_p,)$，FP16 | 逐输出尺度 |
| s1_szeros，逐通道 | $(N_p,)$，FP16 | $+zs_1$ |
| s2_scales，g128 | $(K_p/128,N_p)$，INT8 | 重排后的组尺度字节 |
| s2_zeros，g128 | 同上 | 重排后的 $-zs_2$ 补码字节 |

`process_weights_after_loading` 只是重新包装 Parameter，没有平滑、旋转、重新量化或 repack。Python 检查 $N_p\bmod32=0$、$K_p\bmod2=0$，分组时再检查 $K_p\bmod128=0$；这些只是已实现的前置检查，不是所有 tile 安全条件的完整证明。特别是 TP 切片还须保持物理 tile/组边界，本页没有验证多 rank 加载。

## 5. 逐通道前向中的近似在哪里

`QoQLinearMethod.apply` 调用 `per_token_quant_int8(x, scale_dtype=x.dtype, cal_sum=True)`。Triton 内以 FP32 计算 absmax、尺度、编码和原始 $x$ 的通道和；尺度及输入和最终以 FP16 保存。它返回的 input_sum **不是**整数编码求和。

令 $s_{x,m}$ 为激活尺度，$\widehat x_{mk}=s_{x,m}q_{x,mk}$。SGLang CUDA epilogue 计算

$$
Y^{\rm code}_{m,n}
=s_{x,m}s_{1,n}\sum_kq_{x,mk}u_{n,k}
-\left(\sum_k x_{mk}\right)(z_ns_{1,n}).
$$

严格把量化激活与量化权重相乘，则第二项应使用 $\sum_k\widehat x_{mk}$。忽略浮点舍入，两者之差为

$$
Y^{\rm code}_{m,n}-Y^{\rm quant}_{m,n}
=\left(\sum_k(\widehat x_{mk}-x_{mk})\right)z_ns_{1,n}.
$$

这与方法页区分的论文近似一致，本次已追到 `int8_kernel.py` 的求和与 `qserve_w4a8_per_chn_gemm.cu` 的减法，而不只按论文公式推断。输入量化误差的和越大、缩放零点越大，该差异可能越大；并不保证逐通道误差互相抵消。FP16 input_sum/scale 的舍入是另一个误差来源。

分组路径不调用 cal_sum：每个组使用自己的零点，必须在权重还原时处理，无法用一个全输入求和统一校正。bias 在两条 GPU GEMM 返回后由 Python 加上，输出 FP16。

## 6. 主循环怎样把 W4 变成 INT8 MMA

`qserve_w4a8_per_group_gemm.cu:share_to_reg_one_stage_B` 每线程以 uint4 读 128 bit、即 32 个 nibble。掩码 `0x0F0F0F0F` 与移位把低/高 nibble 拆到独立字节；加载四个组尺度和对应零点字节，进行：

1. 将同一尺度乘到包含四个编码的 32-bit 寄存器。它依赖每字节乘积不向相邻字节进位，而不是四条独立 INT8 乘法。
2. `__byte_perm` 广播负缩放零点，`__vadd4` 做四路字节加法，得到可按有符号 INT8 解释的权重。
3. 使用 `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`，INT8×INT8→INT32；整归约结束后再乘 $s_xs_1$，写 FP16。

尺度 $s_2$ 在这里是**整数乘法的一部分**，不同于普通沿 $K$ 分组的浮点尺度。正是这个约束让组内恢复能融入主循环，而不需要每组把 INT32 部分和转浮点。若 $u\,s_2$ 越过字节边界或最终有符号值越界，寄存器算术不能替上游数值约束兜底。

激活侧仍使用 ldmatrix 从 shared memory 分发，W4 权重侧依赖预排好的 uint4 访问。论文“不能直接使用 ldmatrix”针对的是存储 4-bit、计算 8-bit 的权重分发，不能扩大成整个 kernel 完全没有 ldmatrix。

### host 分派与 shape 边界

两条 CUDA 包装要求 2D contiguous INT8 输入/权重、FP16 输出及相关浮点元数据，设备分支要求 SM80+；分组包装进一步验证组大小必须为 128。输出 $M$ 是当前 token 行数，影响模板：

| 路线 | 条件 | CTA $(M,N,K)$ / stages |
| --- | --- | --- |
| 逐通道 | $M>256$ | $(128,128,64)$ / 3 |
| 逐通道 | $128\le M\le256$ | $(64,64,64)$ / 4 |
| 逐通道 | $M<128$ | $(32,64,128)$ / 3 |
| g128 | $M>128$ | $(128,64,64)$ / 4 |
| g128 | $M=128,K\le4096$ | $(64,64,64)$ / 4 |
| g128 | $M=128,K>4096$ | $(64,64,128)$ / 3 |
| g128 | $M<128$ | $(32,64,128)$ / 3 |

这些是固定代码条件，不是本地调优结果。写回处检查行 $<M$，不能由此推断输出列与 $K$ 的所有尾块也都安全；任意非标准维度需继续核对 tile 加载/写回和完整测试。这里没有“检查失败则自动改为 FP16 GEMM”的通用 fallback 承诺。

## 7. 设备流水、线程分工与写回

以下直接沿 SGLang `qserve_w4a8_per_group_gemm.cu:dense_kernel0` 的 g128 主循环核对，并与 `per_chn` 的输出段对照。两者都用多级 global→shared 环形缓冲和两套寄存器片段；它们不是普通 INT8 GEMM 换一个权重指针。

### 数据移动与寄存器布局

一个 CTA 计算 $B_M\times B_N$ 输出，每轮处理 $B_K$ 个归约元素。线程块的 `threadIdx.x` 是 warp lane，`threadIdx.y` 区分 warp；`warp_mn` 分配输出子块，`slice_id` 分配 CTA 内的 $K$ 子块。`SLICES=B_K/WARP_K`，源码 `SPLITK=1`；这里多个 warp 的 $K$ 分片合并发生在 CTA 内，不是跨 CTA 的全局 split-K。

共享内存每个 stage 的主要字节数是

$$
B_MB_K+\frac{B_NB_K}{2}+2B_N
$$

（g128 的 A、打包 B、INT8 zeros 和 INT8 scales；本模板 A/B padding 为零）。权重一直以 W4 留在 global/shared，扩大到 W8 发生在**寄存器**；因此不会先生成完整 INT8 权重矩阵。

A 的每线程拷贝单位是 16 bytes，shared 的向量列坐标经过 `col ^ ((row / 2) & 3)` 重排；`share_to_reg_one_stage_A` 按同一重排计算地址，以 `ldmatrix.m8n8.x4` 装载 A fragment。B 使用导出器已形成的 tile 顺序，`share_to_reg_one_stage_B` 每 lane 读取 `uint4`，将其中低/高 nibble 分离；loaded 的 x/y/z/w 按 0/2/1/3 等片段位置交给后续恢复。这说明前面的 nibble 重排是为了让设备端少做一次通用转置。

g128 元数据的组索引是 `global_iter_k * B_K / 128`，并与 B 所在 stage 一起加载。$B_K=64$ 时相邻两个 stage 可以读取同一组；不能把源码中 `scales_load_interval` 的声明当成已经消除了这些重复加载。字节恢复随后应用 $u\,s_2+(-z\,s_2)$，满足保护范围时按有符号 INT8 进入 MMA。

### 预热、稳态和清空

1. 预热加载 `STAGES-1` 份 A/B，提交 copy groups；等待 `STAGES-2`，CTA 同步，再预装第一个寄存器片段。
2. 内层每次处理 $K=32$ 的指令片段。先加载下一片到 `(iter_k+1)%2`，再用当前 `iter_k%2` 片段做两条 $16\times8\times32$ INT8 MMA，覆盖一个 $16\times16$ 输出单元。
3. global→shared 的下一 stage 加载穿插在计算中；倒数第二个内部 K 步完成本轮余下搬运、元数据加载、commit/wait 和 ring 切换，供最后一步预读下一 stage 的首片。
4. 结束时 commit、wait 0、CTA barrier，之后才把同一 shared 区域复用成 INT32 部分和缓冲。

`cp_async_cg_A` 用 PTX predicate 禁止无效拷贝，**没有给未写 shared 位置自动补零**。这和 [W8A8 的 CUTLASS prologue](w8a8-quantization-gemm-kernels.md) 中 `cp_async_zfill` 的含义不同；不能把两个实现的尾部安全性互相借用。

### CTA 内合并与输出坐标

若 `SLICES>1`，各 $K$ slice 按循环次序将 INT32 accumulator 写入同一 shared 输出区域，后续 slice 读取前一部分和并相加，每次有 CTA barrier；最后只有 `slice_id==0` 读取完成值并负责写回。这里没有先将每个组的部分和转换成浮点，也没有输出原子加法。

对一个 $16\times16$ 输出单元，设 lane 为 $\ell$、该 lane 的 accumulator 元素序号为 $j\in[0,7]$，忽略 CTA/warp 基址，其坐标为

$$
r=\lfloor\ell/4\rfloor+8\lfloor(j\bmod4)/2\rfloor,\qquad
c=2(\ell\bmod4)+8\lfloor j/4\rfloor+(j\bmod2).
$$

32 lanes × 8 元素恰好覆盖 256 个位置。代码每次以 `j=0,2,4,6` 组成相邻两列的 half2 存储。g128 路径把 INT32 转 FP32、乘 $s_1s_x$ 后转 FP16；逐通道路径则算 `float(C) * wscale * ascale - w_sz * a_ssum`，再写 FP16。二者的尺度结合顺序也不能用同一个逐位参考代替。

### 尾块与短 K：host 检查仍不充分

输出只检查行 $r<M$，没有对应的列 $c<N$ guard；B、列尺度与元数据的加载也依赖完整输出 tile。因此 $N$ 至少应满足所选 `CTA_N` 对齐，而不是仅为偶数就安全。

预热的 A/B 拷贝调用传 `true`，没有按 `k_0_0_ld < gemm_iters` 屏蔽；A 的已有 predicate 只检查行。因此就读到的设备路径而言，要按完整 tile 审查 $K\bmod B_K=0$，并保证 $K\ge(STAGES-1)B_K$，才能避免预热读取超出归约范围。这是从指针和 predicate 推出的必要使用条件，不是框架已强制验证的条件，也不是对全部内存访问的充分证明。g128 的组大小检查并不保证短 $K$ 安全，例如 $K=128$ 仍不足以填充小 M 路径的两个 $B_K=128$ 预热 tile。

[共享 CPU 教学脚本](../assets/w8a8-quantization-gemm-kernels/check_contracts.py)独立枚举上述输出坐标：完整 tile 恰好覆盖一次；人为取 $N=63$ 时可指出未被行 guard 消除的越界列。它只检查索引契约，没有运行 CUDA 或触发实际越界。任意新模型维度接入仍需按 kernel 配置做 GPU 内存检查及参考比较。

## 8. 接入结论与仍缺的证据

已经建立的是：上游 fake-quant/尺度 → OmniServe nibble 与元数据重排 → SGLang QoQ 参数消费 → 动态激活量化 → 两种 W4A8 GEMM。它回答为什么需要特定产物、两种零点为何不同、计算为什么能使用 INT8 Tensor Core。

仍需实际部署验证的是：上游量化器是否保证保护范围、converter 与目标配置/参数名是否完全匹配、TP 是否正确、目标维度是否安全、输出误差以及 prefill/decode 性能。**SGLang 注册 qoq 不证明它实现了论文完整 W4A8KV4 系统**；其配置只为 LinearBase 返回方法，未在此实现 KV4/SmoothAttention。vLLM 此快照也没有内置 qoq 名称；其他 W4A8 方案不能通过改名字直接消费本布局。

### CPU 教学核对

5 项检查覆盖 64×256 编码与独立坐标映射的 tile 打包一致性/往返、组尺度通道重排、字节乘法的无进位条件与溢出反例、负缩放零点相加、原始输入和校正的误差恒等式。教学反例的校正差最大约 0.0153，不是模型质量测量。

可复跑的[脚本](../assets/qserve-implementation/check_contracts.py)与[2026-09-22 结果](../assets/qserve-implementation/checks-2026-09-22.json)已保存；运行方式为 `python3 wiki/assets/qserve-implementation/check_contracts.py`，需 NumPy。默认只打印，指定 `--output` 才写结果。这些计算没有导入或执行上游模型/CUDA 代码，不构成 checkpoint 可加载、内核数值正确或性能复现的证据。

## 来源身份

| 来源 | 版本 | 说明 |
| --- | --- | --- |
| [mit-han-lab/omniserve](https://github.com/mit-han-lab/omniserve/tree/02b2925aa6fa3b92b06316a1524b7f38922cd9c8) | `02b2925aa6fa3b92b06316a1524b7f38922cd9c8` | producer、打包和元数据 |
| [sgl-project/sglang](https://github.com/sgl-project/sglang/tree/2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc) | `2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc` | loader、激活量化与 GEMM |
| [vllm-project/vllm](https://github.com/vllm-project/vllm/tree/568afb3a13806beb53bb2e6bd518269357b237c0) | `568afb3a13806beb53bb2e6bd518269357b237c0` | 注册表边界 |
