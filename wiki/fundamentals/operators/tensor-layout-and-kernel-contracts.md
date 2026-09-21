---
title: 张量布局与算子接口：从逻辑下标到内存地址
type: concept
tags:
  - kernels
  - data-layout
  - gpu
sources:
  - raw/repositories/2026-09-21/vllm/source/csrc/libtorch_stable/activation_kernels.cu
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/layers/activation.py
  - raw/repositories/2026-09-21/sglang/source/python/sglang/jit_kernel/csrc/elementwise/activation.cuh
  - raw/repositories/2026-09-21/triton/source/python/tutorials/01-vector-add.py
  - raw/repositories/2026-09-21/triton/source/python/tutorials/03-matrix-multiplication.py
  - raw/repositories/2026-09-22/kernels/source/kernel-builder/src/init/templates/kernel_cuda/kernel.cu
  - raw/repositories/2026-09-22/kernels/source/kernel-builder/src/init/templates/torch-ext/torch_binding.cpp
updated: 2026-09-22
---

# 张量布局与算子接口：从逻辑下标到内存地址

实现算子的第一步是确定每个输出应从哪些输入地址计算，以及哪些输入属于支持范围。shape 只说明逻辑尺寸；stride、dtype、设备、别名和执行流共同决定 kernel 能否正确读写。相同公式和 shape，不保证两个实现可以互换。

[线性层与输入通道](linear-layer-input-channel.md) 解释矩阵的数学含义；本页解释数学下标怎样落到存储和调用接口。依据 Triton 固定版本的向量加法、矩阵乘教程，以及 Hugging Face kernel-builder 的 CUDA 模板；小尺寸地址表为整理者构造的教学例子。

## 1. 先把“支持这个算子”说具体

以 $C=AB$、$A\in\mathbb R^{M\times K}$、$B\in\mathbb R^{K\times N}$ 为例，数学维度一致还不够：

| 约定 | 需要明确的内容 | 容易遗漏的情形 |
| --- | --- | --- |
| 数学语义 | 是否转置、加 bias、应用激活；沿哪一轴归约 | 把 $AB$ 写成 $AB^T$，或归约错轴 |
| 存储 | shape、各维 stride、元素 dtype、对齐和打包方式 | 转置视图、切片和 INT4 容器 |
| 数值 | 输入、乘法、累加与输出各用什么精度 | 低精度局部和与 FP32 累加并不等价 |
| 写入 | 分配新输出还是修改给定输出；是否允许与输入重叠 | 原地转置覆盖尚未读取的值 |
| 执行 | 设备、stream、支持的尺寸与空输入处理 | 在错误设备或流启动，或零元素仍启动非法配置 |

这张表是接口分析方式，不表示以下教程已支持全部组合。Triton `03-matrix-multiplication.py:matmul` 要求 A 连续，输出固定为 FP16；设备函数虽然接收 stride，包装层仍限制了公开支持范围，而且没有穷尽检查所有 dtype、设备与异常尺寸。

## 2. stride 把逻辑坐标变成地址

设 $p$ 指向张量的第一个逻辑元素，二维张量的元素步长为 $(s_0,s_1)$，则元素偏移为

$$o(i,j)=i s_0+j s_1.$$

若每元素 $e$ 字节，字节地址为 $p_{\rm byte}+e\,o(i,j)$。C/C++ 的带类型指针加法已经按元素大小缩放，不能再乘一次 $e$。若从底层 storage 起点计数，还要加 storage offset；若接口已传入视图首元素指针，则不要重复加它。Triton 矩阵乘教程的 `Pointer Arithmetic` 部分正是按两个 stride 构造地址块。

取连续矩阵 $X$，shape 为 $(2,3)$，底层依次存放 `0,1,2,3,4,5`：

| 逻辑对象 | shape | 元素 stride | 某一位置的地址与值 |
| --- | --- | --- | --- |
| $X$ | $(2,3)$ | $(3,1)$ | $X[1,2]$ 的偏移为 $1\cdot3+2=5$ |
| 转置视图 $X^T$ | $(3,2)$ | $(1,3)$ | $X^T[2,1]$ 仍取偏移 $2+3=5$ |
| 每隔一列的视图 | $(2,2)$ | $(3,2)$ | 第二行第二个元素偏移为 $3+2=5$ |

转置视图改变坐标解释，未必搬动数据。若 kernel 把转置视图一律按 `i*列数+j` 读取，$X^T[1,0]$ 会错误地读到偏移 2，正确偏移是 1。反之，显式重排为连续存储会产生搬运成本，这个成本是否放进计时必须说明。

广播也可以用地址解释：按行加向量 $b_j$ 时，地址不依赖行号，相当于被广播维的 stride 为零。多个逻辑输入位置可以读同一地址；但若把这种映射用作输出，多个线程就会写同一位置。支持广播读取不能推出支持重叠输出。

## 3. shape、线程数量与逻辑块大小是三个量

Triton `01-vector-add.py:add_kernel` 用一个 program 处理 B 个逻辑元素：

$$i=pB+r,\qquad 0\le r<B,\qquad p<\lceil n/B\rceil.$$

`tl.arange(0,B)` 创建这组逻辑下标，`i<n` 保护 load/store。B 表示每个 program 的工作量，并不表示 B 个 CUDA 线程；`num_warps` 是另一个执行参数，实际元素如何分给线程由编译器布局决定。CUDA 的逐线程索引则常写作 `blockIdx.x*blockDim.x+threadIdx.x`。

教学例：n=10、B=8 时启动两个 program，覆盖逻辑候选 `0…7` 与 `8…15`。最后一个只有 8、9 有效。只给 store 加 mask 会防止越界写，却不能修复已经发生的越界读，因此读取侧也要保护。

对归约，失效位置不仅不能访问内存，还必须有合适的填充值；求和填 0、最大值填 $-\infty$ 的原因见 [归约与分块](gpu-kernel-computation-patterns.md)。

### 同一张量可以采用不同工作划分

以连续的逐元素输出 $Y\in\mathbb R^{M\times D}$ 为例，先定义输出拥有者，再选择执行参数：

| 分工 | 拥有者如何定位输出 | 适用条件 |
| --- | --- | --- |
| 每行一个 CUDA block | block 选 m，线程 t 以 $j=t+kT$ 遍历列 | 行内每列独立，或 block 内能完成该行所需协作 |
| 全局展平向量任务 | 一线程负责 V 个元素，用行内向量数 Q=D/V 拆出 $m=t/Q$、$q=t\bmod Q$ | D 可整除 V，输出片段不重叠；尾部全局任务另作保护 |
| Triton 二维 program 网格 | program ID 选 m 和列块 p，逻辑列为 $j=pB+\mathrm{arange}(0,B)$ | 列块能够独立计算，读写均按有效列 mask |

表中 t 在第一行是 block 内线程号，在第二行是全局线程号；整数除法向下取整。真实例子见[两库的门控激活](../../implementation/gated-activation-cuda-triton-kernels.md)：vLLM 的 CUDA 门控、SGLang 的 JIT CUDA 门控与 vLLM 的 Triton 裁剪变体分别采用这些划分。三种映射相近不代表数学变体相同。

B 决定一个 program 的逻辑工作范围，V 决定一次向量任务的元素数，T 决定线程数，三者不能互换。向量化也不同于合并访存：前者描述一个线程处理的一组元素，后者取决于同一 warp 的线程访问地址如何组成内存事务。需要行统计时，不能把同样的列块拆分直接套上独立归一化；[RMSNorm 的分段与归约](../../implementation/rmsnorm-cuda-triton-kernels.md)说明如何先得到整行统计，再写各列。

## 4. 布局支持必须沿包装层和设备函数一起看

Triton 的矩阵乘设备函数使用 A/B/C 的各维 stride，但同时对其作正值假设，包装层还要求 A 连续。不能据“签名里有 stride”就宣布支持零步长广播、负步长或任意视图。

Hugging Face `kernel_cuda/kernel.cu` 初始化模板进一步展示了一个完整调用的两层职责：主机包装检查输入是 CUDA、连续、FP32，并检查输入输出的 shape、dtype、device 一致；设置 device guard，取当前 CUDA stream，再启动执行 `input+1` 的设备函数。`torch-ext/torch_binding.cpp` 则注册算子及设备实现，并将 out 声明为会被修改的参数。

但这个模板没有检查输出是否连续，元素数还存入 int，也没有专门处理 n=0。因此它适合说明设备选择、stream 和注册的关系，不能照抄后宣称支持任意输出视图、超大张量或空输入。这些是本页源码核对发现的支持边界，未运行故障测试。

```mermaid
flowchart LR
    A[张量与数学约定] --> B[包装层检查和设备选择]
    B --> C[分配输出及选择执行配置]
    C --> D[在指定 stream 启动 kernel]
    D --> E[按 stride 和 mask 访问数据]
    E --> F[按数值与写入约定返回结果]
```

在 PyTorch 中注册可调用算子，也不自动得到正确的 backward、编译图元信息或所有图捕获能力；这些是额外接口责任，本页没有核对其完整实现。

## 5. 布局怎样接到量化内核

普通 FP16 矩阵的一个元素通常对应一个可直接寻址的浮点值；INT4 存储可能让多个逻辑值共享一个整数容器，还叠加位序、交织与 scale/zero 分组。此时必须先定义“逻辑坐标 → 容器与位偏移 → 量化参数”，不能只给二维 stride。

[AWQ 实现页](../../implementation/awq-implementation.md) 的 TinyChat int16、AutoAWQ int32 和 Marlin tile 布局是这套思路的具体实例。先弄清布局，再讨论 [GPU 访存与同步](../hardware/gpu-execution-and-memory-hierarchy.md) 以及 [正确性与计时](../../implementation/kernel-correctness-and-benchmarking.md)。本页只核对来源代码和地址关系，未运行所述 GPU 算子。

## 来源身份

| 来源 | 版本或快照 | 核对范围 |
| --- | --- | --- |
| [Triton 官方教程与工具](https://github.com/triton-lang/triton/tree/81a46fa0c04526e5df55a018ecfab72ff922f592) | `81a46fa0c04526e5df55a018ecfab72ff922f592` | 正文标明具体文件与函数；未运行 GPU 教程。 |
| [Hugging Face kernels](https://github.com/huggingface/kernels/tree/5c2cf07f7625e1b8c5fb60bcfb073cd055581cbc) | `5c2cf07f7625e1b8c5fb60bcfb073cd055581cbc` | kernel-builder 初始化模板；模板不是完整输入验证实现。 |
| [vLLM](https://github.com/vllm-project/vllm/tree/568afb3a13806beb53bb2e6bd518269357b237c0) | `568afb3a13806beb53bb2e6bd518269357b237c0` | CUDA 门控与 Triton 列块的工作映射。 |
| [SGLang](https://github.com/sgl-project/sglang/tree/2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc) | `2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc` | JIT 门控的全局向量任务映射。未运行 GPU 实现。 |
