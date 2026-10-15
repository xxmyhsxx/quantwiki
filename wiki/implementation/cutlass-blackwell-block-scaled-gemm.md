---
title: CUTLASS SM100 块缩放 GEMM：指令选择、协作分块与尺度布局
type: implementation
tags:
  - kernels
  - matmul
  - data-format
  - tensor-core
sources:
  - raw/articles/2026-09-21/cutlass-blackwell-gemm
updated: 2026-10-15
---

# CUTLASS SM100 块缩放 GEMM：指令选择、协作分块与尺度布局

SM100 的块缩放矩阵乘把低精度元素与两份块尺度交给 `tcgen05.mma`，以 FP32 累加，再由 epilogue 形成输出。配置一条合法路径，需要同时确定元素/尺度类型、A/B布局、对齐、1或2SM的MMA tile与输出接口。相同的MXFP4元素可以进入不同指令族，布局会改变可选指令和tile。

本页依据固定NVIDIA CUTLASS文档，重点解释dense路径的接口与物理布局。数值格式见[FP8与Microscaling](../fundamentals/numeric-formats/fp8-and-mx-data-formats.md)，一般K组恢复与Triton调用见[块缩放执行](block-scaled-matmul-execution.md)。下文的配置例表示文档所列合法组合；它接收已经准备好的低精度数据和尺度，不负责从模型张量选择量化范围。

## 1. 一次块缩放点积消费哪些数据

令A的逻辑轴为 $[M,K]$，把B的每个输出列按 $[N,K]$ 记为 $B_{n,k}$。$q^A,q^B$ 表示低精度元素解码值，$V$ 是每个尺度在K方向覆盖的元素数。指令计算的值关系可写成

$$
D_{mn}=C_{mn}+\sum_k
q^A_{mk}S^A_{m,\lfloor k/V\rfloor}\,
q^B_{nk}S^B_{n,\lfloor k/V\rfloor}.
$$

两份尺度的逻辑shape分别是 $[M,\lceil K/V\rceil]$ 和 $[N,\lceil K/V\rceil]$。每个输出把K方向不同组的恢复后乘积累加；尺度随K组变化时，不能先完成整段无尺度点积再乘一个总因子。这是文档“Block Scaled GEMMs”的值语义，下节把这组输入映射到具体的布局、指令和tile。

CUTLASS传给collective builder的类型把元素与尺度配置绑定起来：

| 输入类型族 | 元素格式 | 块尺度类型 | dense的V |
| --- | --- | --- | ---: |
| `mx_float8_t` | FP8 E4M3或E5M2 | `float_ue8m0_t` | 32 |
| `mx_float6_t` | FP6 E2M3或E3M2 | `float_ue8m0_t` | 32 |
| `mx_float4_t` | FP4 E2M1 | `float_ue8m0_t` | 32 |
| `nv_float4_t` | FP4 E2M1 | `float_ue4m3_t` | 16 |

UE8M0与UE4M3在这里承担尺度角色；“都是FP4”仍不足以定义同一个输入表示。block-scaled `tcgen05.mma` 的accumulator类型为float。输出的元素类型、epilogue计算类型与是否另产出输出尺度则由epilogue配置决定。（文档“Blackwell Narrow Precision Data Types”“Building a Block Scaled Kernel”。）

这一接口也与[所选SM90 DeepGEMM](deepgemm-moe-expert-execution.md)不同：后者的激活/权重尺度为FP32，每128个K元素先做四次K32 WGMMA，再软件恢复局部点积并累入FP32。本页的原生块缩放指令直接消费规定粒度的窄尺度，尺度表示与恢复位置都随接口改变。

## 2. 从布局与对齐选择指令族

本文档的AB布局记号中，T表示row-major，N表示column-major。例如TN是A行主序、B列主序。它们描述数学GEMM的A/B物理布局；上节的 `B[n,k]` 只是把数学B的列n写在前面，坐标对应关系见[CUTLASS逻辑坐标](cutlass-gemm-composition-and-dispatch.md)。兼容表中的alignment以**元素数**计，不是字节数。（文档“Layouts, Tensor Alignment Requirements”。）

下列dense组合展示了类型与布局对指令选择的共同约束：

| A/B类型与布局 | 两侧对齐要求 | `tcgen05.mma.kind`选择 |
| --- | --- | --- |
| NVFP4×NVFP4，TN | 各32元素 | `mxf4nvf4` |
| MXFP4×MXFP4，TN | 各32元素 | `mxf4`或`mxf4nvf4` |
| MXFP4×MXFP4，NT | 各128元素 | `mxf8f6f4` |

例如MXFP4的128元素相当于64字节FP4载荷；alignment参数仍填128。低于1字节的tensor copy还有额外约束，地址、步长和tile需与所选路径相容。同一kind的合法tile也继续受布局限制。（表3、表9–13。）

文档以MXFP4×MXFP4、A列主序/B行主序的NT组合贯通选择过程。这个组合选 `mxf8f6f4`，两侧alignment为128元素；表11为NT列出的tile恰为：

- 1SM：$128\times128\times128$、$128\times256\times128$；
- 2SM：$256\times256\times128$。

这里三个数依次为M、N、K。TN下的专门MXFP4指令族可以有K=256的另一组tile，不能把它挪到这个NT例子。选择合法族之后，才有意义比较不同tile的资源与耗时。

## 3. 整体MMA tile、每SM输出与cluster

SM100 mainloop builder的 `TileShape_MNK` 表示1或2SM `tcgen05.mma` 所需的**整体MMA tile**。文档NT例选

```cpp
using MmaTileShape_MNK = cute::Shape<_256, _256, _128>;
using KernelMainloopPolicy =
    cutlass::gemm::KernelTmaWarpSpecialized2SmMxf8f6f4Sm100;
using ClusterShape_MNK = cute::Shape<_4, _4, _1>;
```

2SM策略共同承担整体 $256\times256$ 输出tile；每CTA的输出部分为 $128\times256$，K tile仍是128。它沿M分担独立输出行，各输出都完成相应K计算，因此不属于split-K或sliced-K的部分和归约。每SM输出的对应关系来自表15。（“Building a Block Scaled Kernel”“Epilogue PerSmTileShape”。）

cluster是另一层协作形状。所选2SM MMA要求 `ClusterShape_MNK` 的第一维为2的倍数；文档允许第一维取2或4，第二维取1、2或4，示例选4×4×1。因此，**一次2SM MMA只涉及cluster内的一对CTA**，cluster形状描述更大一层的CTA组织。

mainloop与epilogue builder的示例均接收整体 `MmaTileShape_MNK`、`ClusterShape_MNK`；每SM tile解释各CTA的输出责任，builder参数仍保持整体tile。epilogue schedule应与1SM/2SM路径相容，可显式选择 `TmaWarpSpecialized2Sm`，或交给 `EpilogueScheduleAuto`。（表14–15与builder示例。）

`KernelScheduleAuto`按给定tile、类型和布局选择指令kind及1/2SM路径；它在该文档中属于配置派发，不是对候选逐一运行计时的autotune。文档更推荐使用兼容表列出的显式policy；例中选择256×256×128是性能假设，最优配置需要按实际workload测量。资源与实际测量的职责见[配置选择](kernel-configuration-and-autotuning.md)。

## 4. 尺度的逻辑矩阵怎样排成512B块

尺度逻辑上按“行号、K组号”索引，GMEM中却采用适配该接口的块布局。基本块覆盖128个M行或N列，以及K方向4个尺度，覆盖 $4V$ 个逻辑K元素（MX为128、NV为64）；一个尺度占1字节，所以基本块共 $128\times4=512$ 字节。（“Scale Factor Layouts”，来源两张尺度图。）

把基本块内行号写成 $m=32u+r$，其中 $0\le r<32$、$0\le u<4$，尺度组内编号为 $0\le v<4$。第一张图显示连续字节顺序为

```text
M0SF0–3, M32SF0–3, M64SF0–3, M96SF0–3,
M1SF0–3, M33SF0–3, M65SF0–3, M97SF0–3, ...
```

因此，整理者根据图中的地址推得基本块内字节偏移

$$\operatorname{offset}_{512B}(m,v)=16r+4u+v.$$

普通row-major的 $4m+v$ 不满足这幅图的次序：例如M32/SF0在字节4，M1/SF0在字节16。这个映射也对应[已有Triton packed-scale解释](block-scaled-matmul-execution.md)中的 `P[...,r,u,v]` 到逻辑行 $32u+r$ 的关系；两处的共同点是尺度块内顺序，具体kernel消费接口仍各自定义。

超过一个基本块后，块在K组方向优先排列。第二张图取M=512、尺度K组数SFK=16：共有4×4个基本块，同一128行块内的四个K组块起点为0、512、1024、1536；下一128行块起点为2048。令 $i=\lfloor m/128\rfloor$、$j=\lfloor s/4\rfloor$，$s$ 为K组号，且 $Q=\lceil\mathrm{SFK}/4\rceil$，则图示布局的块起点可概括为 $512(iQ+j)$。不满的边缘仍按基本块组织。

实际构造layout可使用 `Sm1xxBlockScaledConfig<SFVecSize>` 的 `tile_atom_to_shape_SFA(problem_shape)` 与 `tile_atom_to_shape_SFB(problem_shape)`。前者按M组织，后者按N组织；`problem_shape=(M,N,K,L)`还保留batch轴L。它把逻辑K元素映射到共享尺度，并构造所需的物理块布局。

这份标准尺度布局如何进入设备流水，见 [fast.cu 的 GB300 NVFP4 执行](blackwell-nvfp4-tmem-execution.md)：该页按SM103a固定源码说明A/B与尺度双环、尺度TMEM、部分重叠的累加区和FP16输出；其K96指令形式单独限定架构。

## 5. Epilogue产生什么输出

mainloop的FP32累加状态进入epilogue后，C/D的dtype、布局、alignment和计算类型决定怎样形成最终输出。文档样例以half C、float compute和E2M1 D说明这些配置彼此独立；单独把D设成FP4元素类型，并没有同时定义一份带新尺度的MX/NV输出。

若下一算子需要的是“低精度元素+块尺度”，epilogue还需输出SFD。文档用 `LinCombBlockScaleFactor` fusion指定：

- `SFDVectorSize`：多少个输出元素共享一个尺度；
- `ElementSFD`：输出尺度编码；
- `GmemLayoutSFD`：沿输出哪个方向生成尺度，通常与D布局对应。

这样，epilogue完成输出元素与新尺度的共同生成，后续consumer按这份表示读取。仅在末尾cast到FP4只生成元素值。（“Building a Block Scaled Kernel”，fusion示例。）

epilogue先定义还有资源原因：mainloop的 `StageCountAutoCarveout<sizeof(CollectiveEpilogue::SharedStorage)>` 要先从总shared预算中扣掉epilogue占用，才能决定stage配置。因此设计上可以先讨论mainloop，但C++类型依赖中要先建立epilogue。这使mainloop的stage预算与epilogue共享同一份SMEM容量。

同一保存文档还介绍SM120：它使用扩展的warp级 `mma.sync`，只列TN布局，cluster固定1×1×1且无multicast，并要求 `EpilogueScheduleAuto`。这些配置条件与SM100的1/2SM路径分别定义。（“Blackwell SM120 GEMMs”。）

## 来源身份

- Blackwell SM100 GEMMs — NVIDIA CUTLASS Documentation（`https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html`），`snapshot-2026-07-02`，正文取得于 `2026-07-02T02:41:08+08:00`；两张尺度图于2026-09-21补采。使用完整保存正文中的SM100类型/布局/对齐、tile/policy、epilogue与512B尺度布局，并核读同页SM120的独立条件；外链PTX/示例源码、TMEM与设备屏障、编译/GPU数值及性能不属本次范围。
