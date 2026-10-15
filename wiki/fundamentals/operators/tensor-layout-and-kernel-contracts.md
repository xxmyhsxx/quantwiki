---
title: 张量布局与算子接口：从逻辑下标到内存地址
type: concept
tags:
  - kernels
  - data-layout
  - gpu
sources:
  - raw/articles/2026-09-21/cutlass-tile-iterators
  - raw/repositories/2026-09-21/vllm
  - raw/repositories/2026-09-21/sglang
  - raw/repositories/2026-09-21/triton
  - raw/repositories/2026-09-22/kernels
  - raw/articles/2026-09-21/cuda-programming-model
  - raw/articles/2026-09-21/cuda-tile-kernels
  - raw/articles/2026-09-21/nvidia-cuda-programming-model
  - raw/articles/2026-09-21/cutlass-layouts-tensors
updated: 2026-10-15
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

### CUTLASS 的布局、张量引用与容量

CUTLASS C++ 的 `Layout` 把逻辑坐标映射成元素偏移，并保存所需的动态 stride。例如 `RowMajor` 与 `ColumnMajor` 都接收二维 `(row,column)`，逻辑 rank 为2；连续轴的步长1隐含在类型中，所以 stride 向量的 rank 只有1。算法仍用相同逻辑坐标，具体 layout 决定其物理位置。（Layouts and Tensors，“CUTLASS Layout Concept”及附录。）

访问数据时，映射、数据指针和范围逐层组合：

| 对象 | 保存什么 | 在访问中承担什么职责 |
| --- | --- | --- |
| `Layout` | 布局类型及动态 stride | 将逻辑坐标变成元素偏移 |
| `TensorRef<T,Layout>` | 指针与 layout | 按映射读取或写入已有数据 |
| `TensorView<T,Layout>` | 在 TensorRef 上增加 extent | 描述有限的逻辑范围，可用 `contains(coord)` 判断坐标是否合法 |

构造 ref/view 只是建立访问描述，数据由调用方提供。view 保存 extent 后，来源示例仍先显式调用 `contains` 再访问；范围元数据本身不等于每次 load/store 都会自动检查或 mask。运算接口也可以只传一份 problem size，再用 TensorRef 表示各操作数，因为它们的 extent 已由运算前提确定。（“Accessing elements within a tensor”。）

逻辑元素数 **size** 与布局所需的元素容量 **capacity** 分别用于描述内容和预算存储。来源的 RowMajor 约定是 `capacity=rows*pitch`，pitch 包含行尾 padding。整理者沿用前面的二维地址公式：shape 为 `(2,3)`、pitch 为5时，size为6，`(1,2)` 的偏移为7，而该接口的 capacity为10；FP32数据对应40字节。只从首元素到最后有效元素算出的8个位置，是最小触及跨度，和该API的容量约定不同。

`Layout::packed(extent)` 返回描述紧密排列的 layout 对象，后续可据此分配、访问相应缓冲。它没有重排已有数据：把pitch为5的原缓冲换成pitch为3的描述，不会自动移除padding。需要把数据变成紧密排列时，还须实际搬运，这与上文“转置视图”和“物理重排”的区别相连。

### CUTLASS 2.x迭代器怎样推进和访问tile

旧CUTLASS 2.x还用tile iterator把固定大小的tile组织成可遍历序列。Layout回答坐标如何映射到存储，TensorRef绑定指针与布局；iterator进一步提供当前tile的推进及读写能力。Tile Iterator Concepts所说的concept是对类型成员的要求集合，并非C++20的 `concept` 声明；算法只需要求自己使用的能力。该快照明确说明这些tile access iterators在CUTLASS 3.0中已弃用，由以`cute::Layout`参数化的`cute::Tensor`统一表达；以下用于阅读2.x旧接口。（开头版本说明及“Definitions”。）

基础iterator声明`Element`和`Shape`：前者是组成tile的元素类型，可为数值类型或Array，后者描述tile范围，具体约定随实现。可读/可写iterator另定义`Fragment`，保存**各线程负责的那部分tile数据**。`load(frag)`把内存数据装入该线程的Fragment，`store(frag)`将其写回；改变iterator位置与执行数据读写是不同操作。访存Fragment与MMA寄存器片段的对应需按具体实现建立，后者见[Tensor Core的片段与布局](../../implementation/tensor-core-quantized-gemm.md)。（Base、Readable与Writeable概念。）

| 操作 | 移动或偏移的单位 | 阅读代码时要确认什么 |
| --- | --- | --- |
| Forward的`++`，Bidirectional另有`--` | 预定义遍历序列中的一个tile | 序列由iterator用途定义；GEMM中可能沿K，不能一律解释为指针加1 |
| RandomAccess的`add_tile_offset`、`+=`、`-=` | 逻辑坐标各维的整tile | 结合`Layout`、`TensorCoord`及tile Shape解释位置变化 |
| `add_pointer_offset` | 线性`Element`个数 | 它增加内部指针偏移，须先确定Element代表的实际类型，再换算标量数或字节数 |

支持相应能力的load/store还可接tile offset、pointer offset或两者，将偏移与该次访问合在一起。这里“contiguous memory”指tensor位于一片连续存储中，tile在其中仍按布局分布；tile内部的排列及[低比特容器的位序](../../implementation/block-scaled-matmul-execution.md#2.%20uint8%20%E5%AE%B9%E5%99%A8%E4%B8%8D%E7%AD%89%E4%BA%8E%E6%95%B0%E5%80%BC%20uint8)仍需按具体实现核对。（Contiguous Memory、Forward与Random Access概念及load/store overload。）

矩阵范围不能整除tile时，Masked概念提供`Mask`保护内存访问。`clear_mask`禁用所有受mask控制的访问，`enable_mask`开启它们，`get_mask/set_mask`读取或设置mask。具体谓词和被屏蔽load后的Fragment值由实现定义：开启全部访问不能代替构造正确尾部条件，关闭访问也不等于将载荷清零。已有Triton或CUDA Tile的补零/丢弃规则，各自沿对应接口理解。（“Masked Tile Iterator Concept”。）

## 3. shape、线程数量与逻辑块大小是三个量

Triton `01-vector-add.py:add_kernel` 用一个 program 处理 B 个逻辑元素：

$$i=pB+r,\qquad 0\le r<B,\qquad p<\lceil n/B\rceil.$$

`tl.arange(0,B)` 创建这组逻辑下标，`i<n` 保护 load/store。B 表示每个 program 的工作量，并不表示 B 个 CUDA 线程；`num_warps` 是另一个执行参数，实际元素如何分给线程由编译器布局决定。CUDA 的逐线程索引则常写作 `blockIdx.x*blockDim.x+threadIdx.x`。

逻辑张量的广播、类型提升、控制流与编译特化另见 [Triton 语言与编译语义](triton-language-and-compilation.md)；它们决定这段索引和算术代码的实际含义。

教学例：n=10、B=8 时启动两个 program，覆盖逻辑候选 `0…7` 与 `8…15`。最后一个只有 8、9 有效。只给 store 加 mask 会防止越界写，却不能修复已经发生的越界读，因此读取侧也要保护。

对归约，失效位置不仅不能访问内存，还必须有合适的填充值；求和填 0、最大值填 $-\infty$ 的原因见 [归约与分块](gpu-kernel-computation-patterns.md)。

CUDA Refresher的MatAdd用$i=\mathrm{blockIdx.x}\,\mathrm{blockDim.x}+\mathrm{threadIdx.x}$、$j$沿y维，访问C行优先`MatA[i][j]`，以`i<N && j<N`同时保护读写。按地址$iN+j$展开，在其`(16,16)`block的固定y、16-lane x段内，相邻有效lane相差N个float；一个warp跨两段，段间跳转不同。二维下标方便表达不等于地址已合并，仍须核线程线性化与stride。这是整理者地址观察，不是加法错误或性能测量；N定义、分配与host包装在保存示例中省略。

### 同一张量可以采用不同工作划分

以连续的逐元素输出 $Y\in\mathbb R^{M\times D}$ 为例，先定义输出拥有者，再选择执行参数：

| 分工 | 拥有者如何定位输出 | 适用条件 |
| --- | --- | --- |
| 每行一个 CUDA block | block 选 m，线程 t 以 $j=t+kT$ 遍历列 | 行内每列独立，或 block 内能完成该行所需协作 |
| 全局展平向量任务 | 一线程负责 V 个元素，用行内向量数 Q=D/V 拆出 $m=t/Q$、$q=t\bmod Q$ | D 可整除 V，输出片段不重叠；尾部全局任务另作保护 |
| Triton 二维 program 网格 | program ID 选 m 和列块 p，逻辑列为 $j=pB+\mathrm{arange}(0,B)$ | 列块能够独立计算，读写均按有效列 mask |

表中 t 在第一行是 block 内线程号，在第二行是全局线程号；整数除法向下取整。真实例子见[两库的门控激活](../../implementation/gated-activation-cuda-triton-kernels.md)：vLLM 的 CUDA 门控、SGLang 的 JIT CUDA 门控与 vLLM 的 Triton 裁剪变体分别采用这些划分。三种映射相近不代表数学变体相同。

B 决定一个 program 的逻辑工作范围，V 决定一次向量任务的元素数，T 决定线程数，三者不能互换。向量化也不同于合并访存：前者描述一个线程处理的一组元素，后者取决于同一 warp 的线程访问地址如何组成内存事务。需要行统计时，不能把同样的列块拆分直接套上独立归一化；[RMSNorm 的分段与归约](../../implementation/rmsnorm-cuda-triton-kernels.md)说明如何先得到整行统计，再写各列。

### CUDA Tile由block描述数据操作，编译器分配线程

Programming Model快照§1.2.2.3把CUDA Tile与SIMT区分：SIMT源码描述每个线程怎样工作，CUDA Tile源码描述每个block的控制流和整块数据操作，编译器决定block内线程数及工作映射。block是执行单位，tile是数据对象；一个block可产生多个不同shape/dtype的tile，tile元素数不能当线程数。（图8。）

该模型的array是device memory中的可变数据，tile则是仅在block内部产生和消费的不可变值，每次操作得到新tile。tile不一定对应一整块物化内存，编译器可用寄存器、shared或其他SM资源承接。本章要求tile各维在编译期已知且为2的幂，并禁止把tile直接作为kernel参数；这不代表所有CUDA分块算法都有同一形状限制。（§1.2.2.3.1。）

load/store在array与tile之间搬数据。tile-space坐标选一块区域，具体尾部处理须按所用API和边界策略核对；编程模型图中的补零不应替代接口检查。SIMT与CUDA Tile可在同一应用中按kernel分别使用、操作同一device数据；编译抽象相似也不使CUDA Tile与上文Triton的API、类型或调参规则互换。（§1.2.2.3.2–1.2.2.3.4，图9；具体Tile接口见下节。）

### Tile kernel启动、编译形状与array尺寸

Writing Tile Kernels快照§2.4.1–2.4.5沿上一节分工给出两套入口：Python用`@ct.kernel`及`ct.launch(stream,grid,kernel,args)`，分别传执行流、tile-block网格、kernel对象和参数；C++用`__tile_global__`及`<<<grid,1>>>`。C++第二项1是启动约定，实际线程数由编译器决定。Python的`ct.Constant`与C++模板/`integral_constant`把结构参数嵌入编译，tile shape/dtype因此可特化；array的实际长度仍可在运行时传入。

保存正文声明Tile C++从Toolkit13.3起提供，并限制tile函数与SIMT device函数直接互调；这不是本机安装或其他版本的核验。序言还把Python模块写作`cuda.tiles`，实际示例为`cuda.tile`；本页沿可辨示例解释，不推定两个名字可互换，未编译或导入该包。

### tile-space坐标与部分尾块访问

同快照§2.4.6、图19对array `(10,16)`、tile `(2,4)`得到tile grid `(5,4)`；tile坐标`(1,2)`定位元素行2–3、列8–11，不是元素`[1,2]`。C++先用`tensor_span`给指针附加多维范围，再用`partition_view`定义规整分块；Python `tiled_view`或`ct.load(array,index,shape)`表达同样的tile-space关系。gather/scatter才按逐元素index或pointer tile选地址。

| 访问 | Python保存稿规则 | C++保存稿规则 |
| --- | --- | --- |
| 部分越界的tile-space load | 默认`UNDETERMINED`；需`ZERO`才保证填零 | unmasked要求整tile有效；`load_masked`默认填零 |
| 部分越界的tile-space store | 丢弃越界位置写入 | unmasked要求整tile有效；`store_masked`丢弃越界写入 |
| 非规则gather/scatter | 默认检查元素边界，读填零/写丢弃；关闭检查后越界未定义 | pointer tile无自动边界检查，需逐元素mask |

**两种语言的tile-space保护都只针对部分尾块；整个tile在array之外仍为未定义行为。** grid应覆盖真实tile区间，空尺寸另由包装层定义处理，不能任意多启动。gather的默认零填充不能搬给`ct.load`；K尾补零能消去无效乘积，softmax的失效score却需按数学语义处理为$-\infty$，见[归约与mask](gpu-kernel-computation-patterns.md)。这些条件来自§2.4.6.1.3–2.4.6.2.1，未代替完整API reference。

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

从这组职责走到完整代码、编译、库加载与调用，见 [CUDA 算子编写与接入](../../implementation/cuda-operator-build-and-integration.md)。该教学例子分配独立输出，并明确处理空输入、当前设备/stream、FakeTensor 与推理接口范围。

## 5. 布局怎样接到量化内核

普通 FP16 矩阵的一个元素通常对应一个可直接寻址的浮点值；INT4 存储可能让多个逻辑值共享一个整数容器，还叠加位序、交织与 scale/zero 分组。此时必须先定义“逻辑坐标 → 容器与位偏移 → 量化参数”，不能只给二维 stride。

[AWQ 实现页](../../implementation/awq-implementation.md) 的 TinyChat int16、AutoAWQ int32 和 Marlin tile 布局是这套思路的具体实例。先弄清布局，再讨论 [GPU 访存与同步](../hardware/gpu-execution-and-memory-hierarchy.md) 以及 [正确性与计时](../../implementation/kernel-correctness-and-benchmarking.md)。本页只核对来源代码和地址关系，未运行所述 GPU 算子。

## 来源身份

| 来源 | 版本或快照 | 核对范围 |
| --- | --- | --- |
| Tile Iterator Concepts（`https://docs.nvidia.com/cutlass/latest/media/docs/cpp/tile_iterator_concept.html`） | `snapshot-2026-07-02`，取得于 `2026-07-02T02:41:08+08:00`；通用层次图于2026-09-21补采 | 完整471行/README与同hash通用图；使用legacy 2.x的tile/Fragment、推进与偏移单位、显式读写及实现定义的Mask。仅为类型要求，具体iterator实现、CuTe布局代数、packing/指令映射与编译运行未核。 |
| Layouts and Tensors（`https://docs.nvidia.com/cutlass/latest/media/docs/cpp/layout.html`） | `snapshot-2026-07-02`，取得于`2026-07-02T02:41:08+08:00`；通用层次图补采于2026-09-21 | 完整保存280行、README与配图；使用C++ Layout/TensorRef/View及capacity约定，地址/字节例为教学展开。CuTe布局代数、int4底层打包、附录仅列名称的layout实现与编译运行不在本次范围。 |
| CUDA Refresher: The CUDA Programming Model（`https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/`） | `snapshot-2026-07-02` | MatAdd二维映射/guard及保存全文；每16-lane段地址步长为整理者按C行优先推导，未运行，省略host示例不作完整程序。 |
| CUDA Programming Guide：Writing Tile Kernels（`https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-tile-kernels.html`） | `snapshot-2026-07-02`，图19补采于2026-09-21 | §2.4.1–2.4.6及图19；保存正文全读，启动/特化、坐标与不同边界规则；模块名差异保留，完整API/编译/运行未核。 |
| CUDA Programming Guide：Programming Model（`https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html`） | `snapshot-2026-07-02`，正文与2026-09-21补采图分开登记 | §1.2.2.3与图8–9；block/tile分工与array/tile语义，完整保存正文已读；未核API reference、编译布局或执行。 |
| Triton 官方教程与工具（`https://github.com/triton-lang/triton/tree/81a46fa0c04526e5df55a018ecfab72ff922f592`） | `81a46fa0c04526e5df55a018ecfab72ff922f592` | 正文标明具体文件与函数；未运行 GPU 教程。 |
| Hugging Face kernels（`https://github.com/huggingface/kernels/tree/5c2cf07f7625e1b8c5fb60bcfb073cd055581cbc`） | `5c2cf07f7625e1b8c5fb60bcfb073cd055581cbc` | kernel-builder 初始化模板；模板不是完整输入验证实现。 |
| vLLM（`https://github.com/vllm-project/vllm/tree/568afb3a13806beb53bb2e6bd518269357b237c0`） | `568afb3a13806beb53bb2e6bd518269357b237c0` | CUDA 门控与 Triton 列块的工作映射。 |
| SGLang（`https://github.com/sgl-project/sglang/tree/2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc`） | `2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc` | JIT 门控的全局向量任务映射。未运行 GPU 实现。 |
