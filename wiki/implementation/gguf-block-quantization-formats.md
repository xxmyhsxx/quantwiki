---
title: GGUF 与块量化存储格式
type: implementation
tags:
  - weight-quantization
  - data-format
  - granularity
  - mixed-precision
sources:
  - raw/repositories/2026-09-21/llama-cpp/source/gguf-py/gguf/gguf_reader.py
  - raw/repositories/2026-09-21/llama-cpp/source/ggml/src/ggml-common.h
  - raw/repositories/2026-09-21/llama-cpp/source/ggml/src/ggml-quants.c
  - raw/repositories/2026-09-21/llama-cpp/source/ggml/src/ggml.c
  - raw/repositories/2026-09-21/llama-cpp/source/tools/quantize/README.md
  - raw/repositories/2026-09-21/llama-cpp/source/tools/imatrix/README.md
  - raw/repositories/2026-09-21/llama-cpp/source/tools/imatrix/imatrix.cpp
updated: 2026-09-16
---

# GGUF 与块量化存储格式

论文里的量化方案要落到文件，还需要回答一组工程问题：权重怎么分块、每块保存哪些元数据、文件如何描述张量与模型、量化工具怎样按张量选择格式、以及校准统计在哪里参与。GGUF 与它的 k-quant／i-quant 家族是这些问题的一套完整答案，也是 llama.cpp 生态的存储与执行入口。

本页依据 llama.cpp 固定快照 `b820cc8e6f94843d32f92c8ebd7db837dae2bd8b` 的容器读取代码、块结构定义、量化与反量化实现、量化工具与 imatrix 工具文档和实现。以下区分代码事实、按代码推导的位宽计算与作者文档报告；本轮没有构建或运行 llama.cpp，也没有产出任何 gguf 文件。

## 1. GGUF 容器

GGUF 是「元数据 + 张量表 + 张量数据」的单文件容器。读取顺序（`gguf-py/gguf/gguf_reader.py`）为：

1. 魔数 `GGUF`（`0x46554747`）；
2. 版本号，当前为 3，读取器兼容 2 与 3；字节序可由版本字段判断，默认小端并支持大端文件；
3. 张量数量（uint64）与元数据键值对数量（uint64）；
4. 元数据键值对，带类型标记，支持字符串、数组与各类标量；`general.alignment` 属于其中之一；
5. 每个张量的信息：名称、维度数、各维大小、量化类型编号、相对数据段偏移；
6. 张量数据段，起始偏移按 `general.alignment` 对齐。

对齐值默认为 32，且必须是 2 的幂，否则读取器报错。单个张量的字节数不由元素数乘类型宽度得出，而按块计数：

$$n_{bytes}=n_{elems}\times \frac{type\_size}{block\_size}.$$

这两个常量来自类型表（`ggml/src/ggml.c`），例如 `q4_0` 的 `blck_size = QK4_0 = 32`、`type_size = sizeof(block_q4_0)`；`q4_K` 的 `blck_size = QK_K = 256`、`type_size = sizeof(block_q4_K)`。同一张表还登记了对应的反量化与参考量化函数。

这意味着 GGUF 的存储粒度是块，任何格式的位宽都必须按「块字节数除以块元素数」计算，不能只看名义位宽。张量数据的布局、量化的数学形式与容器本身相互独立：容器负责定位与描述，格式负责数值。

## 2. 32 元素块：早期格式

最早的 ggml 量化格式以 32 个权重为一块，块内共享一个或两个浮点参数（`ggml-common.h`）：

| 类型 | 块结构 | 字节数 | 每权重位宽 |
|---|---|---:|---:|
| q4_0 | fp16 尺度 + 16 字节半字节 | $2+16=18$ | 4.5 |
| q4_1 | fp16 尺度 + fp16 最小值 + 16 字节 | $4+16=20$ | 5.0 |
| q5_0 | fp16 尺度 + 4 字节高位 + 16 字节 | $2+4+16=22$ | 5.5 |
| q5_1 | 两个 fp16 + 4 字节高位 + 16 字节 | $4+4+16=24$ | 6.0 |
| q8_0 | fp16 尺度 + 32 个 int8 | $2+32=34$ | 8.5 |
| q8_1 | 两个 fp16 + 32 个 int8 | $4+32=36$ | 9.0 |

q4_0 的 4.5 位中，有 0.5 位用于 fp16 尺度：$16/32=0.5$。这是块量化的一般代价结构，也是 k-quant 要解决的问题——块越小越能适配局部分布，但元数据占比越高。

## 3. 二级尺度：K-quant 超块

k-quant 把块扩大为超块（super-block，$QK\_K = 256$），并在超块内再分 8 或 16 个子块，形成两级尺度：超块级用两个 fp16 保存尺度与最小值，子块级用 6 bit（或 8 bit）保存各自的尺度与最小值。以 4 bit 为例（`ggml-common.h`）：

```c
#define QK_K 256
#define K_SCALE_SIZE 12

typedef struct {
    ggml_half d;                  // super-block scale for quantized scales
    ggml_half dmin;               // super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // 4-bit quants
} block_q4_K;
```

共 $4+12+128=144$ 字节对应 256 个权重，即 4.5 bit/weight。反量化实现给出确切的数值关系（`ggml-quants.c` 的 `dequantize_row_q4_K`）：

$$\widehat w = d_1 q - m_1,\qquad d_1 = d\cdot sc,\quad m_1 = d_{\min}\cdot m,\quad q\in\{0,\dots,15\},$$

其中 $sc$ 与 $m$ 是 6 bit 的子块尺度与最小值，按 `get_scale_min_k4` 的打包规则从 12 字节中解出，每 32 个权重一组。也就是「超块级浮点尺度乘以子块级整数尺度」再乘整数码，减掉同构的偏置项。子块数量 8、每子块 32 个权重，是 256 与 4 bit 半字节打包共同决定的布局。

同一文件里的其他超块格式：

| 类型 | 结构要点 | 字节数 | 每权重位宽 |
|---|---|---:|---:|
| q2_K | 16 个子块，尺度与最小值各 4 bit | $4+16+64=84$ | 2.625 |
| q3_K | 高位掩码 + 低 2 bit + 12 字节 6 bit 尺度 | $2+64+32+12=110$ | 3.4375 |
| q4_K | 二级尺度 + 128 字节半字节 | $4+12+128=144$ | 4.5 |
| q5_K | 在 q4_K 基础上加高位 | $4+12+128+32=176$ | 5.5 |
| q6_K | 低 4 bit + 高 2 bit + 每子块 int8 尺度 | $2+128+64+16=210$ | 6.5625 |

这些位宽与源码注释一致（例如 q6_K 注释写明 6.5625 bpw）。q8_K 只有 $4+256+32=292$ 字节，源码注明它只用于中间量化与点积，不是存储格式。

**为什么用两级尺度。** 若子块各自使用独立 fp16 尺度，每 32 个权重需 $4+16=20$ 字节元数据，位宽代价远高于 0.5。把子块尺度再量化到 6 bit、再用超块级 fp16 兜底，就把元数据压到每 256 个权重 16 字节，同时保留子块级自适应。代价是多一次整数乘与偏置修正，以及打包解包的控制流——这正是 [执行路径](quantized-matmul-scaling-execution.md) 里「低比特存储不等于低比特计算」的具体来源。

## 4. 码本路线：i-quant

i-quant 走的是另一条路：不再对每个权重独立均匀取整，而是用网格／码本表示一组权重，并在超块级保留尺度或位移。源码注释给出了各界面的有效位宽：

| 类型 | 注释与结构要点 | 每权重位宽 |
|---|---|---:|
| iq1_s | fp16 尺度 + 索引 + 高位 | 1.5625 |
| iq1_m | 网格索引低 8 位 + 高 3 位与移位位 + 3 bit 子块尺度，**不含 fp16 超块尺度** | 1.75 |
| iq2_xxs | 注释称「几乎真正的 2 bit」，因每 256 权重一个 16 bit 尺度而到 2.0625 | 2.0625 |
| iq2_xs | 同一结构加每 32 权重的尺度 | 2.3125 |
| iq2_s | 索引 + 高位 + 尺度 | 2.5625 |
| iq3_xxs | 注释称「几乎真正的 3 bit」，同样含 16 bit 超块尺度 | 3.0625 |
| iq3_s | 索引 + 高位 + 符号 + 尺度 | 3.4375 |
| iq4_nl | 32 元素块，非线性码本 | 4.5 |
| iq4_xs | fp16 尺度 + 打包的高位尺度 + 128 字节 | 4.25 |

两个可以直接从源码读出的认识：

第一，iq1_m 把尺度也编码进 3 bit 子块尺度，因此结构里没有独立的 fp16 超块尺度，块内也没有单独的量级基准，位宽降到 1.75。位宽的下降不是「把 2 bit 索引写成 1.75 bit」，而是元数据组织方式的改变。

第二，iq2_xxs 与 iq3_xxs 的位宽高于名义位数，原因是「按 ggml 的块设计要求」每 256 个权重必须配一个 16 bit 尺度。这说明有效位宽由数据结构决定，名义位宽与实际存储之间可以有稳定差额；引用某格式的 bpw 时应注明是名义值还是该实现的实际块尺寸。

码本类格式的解码依赖网格表（源码中的查找表与索引运算），比 q4_K 的仿射形式复杂，因此对内核的友好程度更低；它们的收益主要体现在同样位宽下的重建质量。i-quant 的质量依赖重要性矩阵（下一节）。

## 5. 混合格式与量化工具

`Q4_K_M` 这类名称中的 M／S／L 表示层间混合：对敏感张量使用更高位宽，其余使用基础格式，从而在总大小与质量之间取舍。工具支持显式控制（`tools/quantize/README.md`）：

```bash
# 纯格式：所有张量同一类型
./llama-quantize --pure input-f32.gguf output-Q4_K.gguf q4_k_m 8
# 按张量指定，或对输出层/词嵌入单独指定
./llama-quantize --output-tensor-type q5_k --token-embedding-type q3_k input.gguf out.gguf q4_k_m 8
# 用正则按层号分别指定位宽
./llama-quantize --tensor-type "\.(\d*[13579])\.attn_k=q5_k" input.gguf out.gguf q4_k_m 8
```

相关参数与语义：`--allow-requantize` 允许对已量化的张量再次量化，文档警告这会显著劣于从 16/32 bit 量化；`--leave-output-tensor` 保持输出层不量化以提升质量；`--keep-split` 保持分片；`--prune-layers` 直接剪层。文档还给出默认实践：多模态组件（mmproj）通常保留 bf16 或 q8，因为其质量直接影响生成结果，而压缩带来的速度与内存收益可以忽略。

这一步是「格式选择」与「模型结构」的交界：哪些张量重要、哪些层可承受更低精度，是模型相关的工程判断；工具的作用是把判断表达为配置。与逐层混合精度的系统化方法对照见 [混合精度分配](../theory/mixed-precision-allocation.md)。

## 6. imatrix：把校准统计带进量化

量化工具有一个可选输入：重要性矩阵（imatrix）。生成方式为 `llama-imatrix -m model.gguf -f calib.txt -o imatrix.gguf`（`tools/imatrix/README.md`），其统计量在 `imatrix.cpp` 中为

$$\mathrm{values}[j]\mathrel{+}=x[j]^{2},$$

即按输入通道累积激活平方（对校准数据求和，使用时按计数取平均）。这与 [层输出重构](../theory/layer-reconstruction-second-order-compensation.md) 中 $G=XX^{\mathsf T}$ 的对角元素 $G_{jj}=\sum_t X_{jt}^2$ 是同一个量，只是此处只保留对角、不保留通道间相关。

量化侧消费这些统计的方式（`quantize_row_q4_K_impl`）是把每个元素的拟合权重改为

$$w_{l}=qw_{l}\cdot\sqrt{\sigma^{2}+x_{l}^{2}},\qquad \sigma^{2}=\frac{2}{256}\sum_{l}x_{l}^{2},$$

其中 $qw_l$ 是 imatrix 提供的通道权重。也就是在块内做加权拟合时，同时考虑该元素自身幅度与所在通道的激活统计。文档说明 `--process-output` 默认关闭，即通常不使用 imatrix 量化 `output.weight`。

imatrix 的定位是「校准统计改善量化拟合」，与 GPTQ 的二阶补偿有共同的信息来源（激活二阶统计），但机制不同：这里只改变参考量化器的加权最小二乘目标，不做逐列误差补偿，也不更新未量化权重。两者的差别见 [GPTQ](../methods/gptq.md) 与 [层重构页](../theory/layer-reconstruction-second-order-compensation.md)。

## 7. 作者报告的位宽与吞吐

量化工具文档给出 Llama-3.1-8B 的一组测量（原文表格，未注明所用硬件、线程与后端版本）：

| 格式 | 位宽 | 文件大小 (GiB) | 生成速度 (t/s @128) |
|---|---:|---:|---:|
| IQ1_S | 2.0042 | 1.87 | 79.73 |
| IQ1_M | 2.1460 | 2.01 | 72.92 |
| IQ2_XXS | 2.3824 | 2.23 | 79.86 |
| Q2_K | 3.1593 | 2.95 | 79.85 |
| IQ3_XXS | 3.2548 | 3.04 | 73.95 |
| Q3_K_M | 3.9960 | 3.74 | 71.68 |
| IQ4_XS | 4.4597 | 4.17 | 77.51 |
| Q4_K_M | 4.8944 | 4.58 | 71.93 |
| Q5_K_M | 5.7036 | 5.33 | 67.23 |
| Q6_K | 6.5633 | 6.14 | 58.67 |
| Q8_0 | 8.5008 | 7.95 | 50.93 |
| F16 | 16.0005 | 14.96 | 29.17 |

两点解释边界：这是单一 CPU／后端条件下的生成吞吐，不是跨硬件结论；位宽更高的格式在这里更慢，与「生成阶段受权重读取带宽限制」的一般判断方向一致，但缺少数值证明与硬件说明，不能拿来推算其他设备或批量场景。文档同时给出内存与磁盘需求表（例如 8B 模型 F32 为 32.1 GB、Q4_K_M 为 4.9 GB），并说明量化过程需要把模型载入内存。

## 8. 与论文格式的差别和适用条件

- **目的不同。** 这些格式服务于本地推理运行时的存储与内核，设计时同时考虑解码成本与内存带宽；研究论文提出的量化方案常以精度为第一目标，导出时需要映射或重新实现。例如 BiLLM 的 1.08 bit 参数位宽与 IQ1_M 的 1.75 bpw 不是同一口径，前者是参数位宽、后者是包含尺度的实际块尺寸（对照见 [LUQ](../methods/luq.md) 的部署讨论与 [混合精度分配](../theory/mixed-precision-allocation.md)）。
- **口径必须写清。** 名义位宽、块尺寸算出的位宽、包含元数据的文件大小、运行时常驻内存是四个不同量。
- **质量依赖配置。** imatrix、张量级别覆盖、输出层是否量化都影响结果；同一格式名在不同配置下不是一个质量点。
- **跨模型不通用。** 混合格式的层选择与 imatrix 都依赖具体模型与校准数据，不能直接套用另一模型的最优配置。

## 9. 局限与未验证

- 本页全部内容来自单项代码快照与工具文档，未构建、未运行 llama.cpp，未生成或读取任何 `.gguf` 文件；位宽为按块结构计算的推导值，不是磁盘实测。
- i-quant 的网格表与索引解码细节、CUDA／Metal／Vulkan 各后端的对应内核分支未逐一核对；本页不评价其性能表现。
- 作者表格缺少硬件、线程数、后端与测量方法说明，其数值只用于说明格式间的相对关系。
- llama.cpp 版本迭代频繁，格式编号与默认行为可能随版本变化；引用时应固定 commit。

## 来源身份

下表用于在没有本地资料库时辨识来源；具体论述的章节、公式、图表或代码位置见正文。

| 来源 | 版本或快照 | 说明 |
| --- | --- | --- |
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/tree/b820cc8e6f94843d32f92c8ebd7db837dae2bd8b) | `b820cc8e6f94843d32f92c8ebd7db837dae2bd8b` | — |
