---
title: 块缩放矩阵乘：FP4/FP8 编码、Scale 布局与 Triton 调用
type: implementation
tags:
  - kernels
  - matmul
  - triton
  - data-format
  - floating-point
sources:
  - raw/articles/2026-09-21/triton-block-scaled-matmul
  - raw/repositories/2026-09-21/triton
updated: 2026-10-04
---

# 块缩放矩阵乘：FP4/FP8 编码、Scale 布局与 Triton 调用

块缩放点积把低精度浮点元素与共享尺度一起交给运算接口。理解它需要同时追踪逻辑数值、打包字节、scale 的存储顺序和实际点积调用；普通 FP32 输出不能说明乘法输入已恢复成 FP32。本页深读固定 Triton 教程的 **NVIDIA descriptor 路径**，不展开 AMD preshuffle 分支，也不把教程当作完整模型量化工具。

格式原理见[FP8 与 Microscaling](../fundamentals/numeric-formats/fp8-and-mx-data-formats.md)，尺度为何不能任意移出 K 归约见[量化矩阵乘的缩放](quantized-matmul-scaling-execution.md)，逻辑块与物理线程的区别见[Triton 语言语义](../fundamentals/operators/triton-language-and-compilation.md)。以下索引例子为整理者推导；本次没有编译、运行教程或测量吞吐。

## 1. 块尺度在 K 归约中的作用

设逻辑矩阵为 $A\in\mathbb R^{M\times K}$、$B\in\mathbb R^{K\times N}$。低精度位模式分别解码成 $q^A_{mk}$ 与 $q^B_{kn}$，每 V 个 K 元素共享一个尺度，尺度的**逻辑** shape 为

$$S^A\in\mathbb R^{M\times(K/V)},\qquad S^B\in\mathbb R^{N\times(K/V)}.$$

在本页选择的可整除尺寸下，教程参考运算等价于

$$C_{mn}=\sum_{g=0}^{K/V-1}S^A_{mg}S^B_{ng}\sum_{k=gV}^{(g+1)V-1}q^A_{mk}q^B_{kn}.$$

这是按教程 `initialize_block_scaled` 的 FP32 参考路径改写的值语义，不是对硬件每一步舍入的定义。K 方向各组的 scale 不同，通常不能把它们统一移到完整点积之后。B 的 scale 按输出列 n 与 K 组 g 索引，所以即使 B 作为 `[K,N]` 参加乘法，`tl.dot_scaled` 仍接收 `[N,K/V]` 的 B scale。

固定教程使用以下配置：

| 模式 | A/B 元素编码 | V | scale 的存储与解释 |
| --- | --- | --- | --- |
| `mxfp4` | 两侧 E2M1，两个元素打包为一个 uint8 | 32 | E8M0 位模式存于 uint8，代表 2 的幂 |
| `mxfp8` | 两侧 `torch.float8_e4m3fn` | 32 | 同上 E8M0 |
| `nvfp4` | 两侧 E2M1，两个元素打包为一个 uint8 | 16 | `torch.float8_e4m3fn` scale，按 E4M3 浮点值解释 |
| `mixed` | A 为 E4M3 FP8，B 为打包 E2M1 FP4 | 32 | E8M0 |

这些是该教程的输入/尺度组合，不是所有名为 NVFP4 的模型文件或量化器的完整格式定义。教程生成低精度随机值和尺度来研究 matmul，没有演示从模型张量选择最优 scale，也没有在此链中加入额外全局尺度。（`10-block-scaled-matmul.py:initialize_block_scaled`。）

## 2. uint8 容器不等于数值 uint8

`tools/mxfp.py:MXFP4Tensor` 将 E2M1 编码为四位 `S E E M`。非负幅值集合为 $0,0.5,1,1.5,2,3,4,6$，符号位另决定正负，包含带符号零。`to_packed_tensor(dim=1)` 沿 K 轴打包：第一个元素放低四位，第二个放高四位。

例如字节 `0x72` 的低四位 `0x2` 解码为 1，高四位 `0x7` 解码为 6；将整个字节数值转换成 FP32 会得到 114，含义完全不同。即使 storage dtype 是 uint8，送入 `tl.dot_scaled(..., "e2m1", ...)` 后也按打包 FP4 位模式解释，而不是整数乘法。

`MXScaleTensor.to(torch.float32)` 将有限 E8M0 编码 b 解为 $2^{b-127}$，255 为 NaN；因此字节 129 的尺度是 4，不是 129。该 helper 的 `_from_float` 对正有限数使用 `floor(log2(value))` 后夹住指数，对非正值/NaN/Inf 输出 NaN 编码。这里只说明固定 helper 的行为，不能将其当作所有 MX 量化器的尺度选择策略。

NVFP4 分支把 scale 转为 E4M3 浮点，不能再按 E8M0 解码。教程的尺度布局也不同于[ggml 的紧凑块结构](gguf-block-quantization-formats.md)；相同元素格式和共享粒度不保证两个接口可以直接交换字节。

## 3. A、B 数据怎样进入 descriptor

选取教程支持的一组 MXFP4 尺寸：$M=128,N=256,K=256$。教程默认 `BLOCK_M=128`、`BLOCK_N=256`、`BLOCK_K=256`、`VEC_SIZE=32`。

| 对象 | 全局存储 shape | 一个 K tile 的 load shape | 逻辑含义 |
| --- | --- | --- | --- |
| A 字节 | `[128,128]` uint8 | `[128,128]` | 每行 256 个 FP4，沿 K 打包 |
| B 字节 | `[256,128]` uint8 | `[256,128]` | 每个输出列保存 256 个 FP4，沿 K 打包 |
| 点积用 B | load 后逻辑转置为 `[128,256]` | 同左 | 物理 K 字节轴成为第一维，逻辑 K 仍是 256 |
| 输出 | `[128,256]` | `[128,256]` | 包装选择 FP16、FP32 或 E4M3 输出 |

教程先生成 B 为 `[N,K]`，使打包方向与 K 连续，再在设备中使用 `b.T` 参与 `[M,K] @ [K,N]` 的语义。不能把打包后的 `K/2` 当作逻辑 K，或对 B 进行普通整数类型转换后再乘。`semantic.py:dot_scaled` 根据 `e2m1` 与 K packing 将字节维展开为逻辑维，并检查两侧归约长度及 scale shape。

`TensorDescriptor.from_tensor` 还保留 shape、元素 stride 与每次 load/store 的 `block_shape`。固定版本要求基址 16 字节对齐、非末维的字节 stride 为 16 的倍数、末维 stride 为 1、rank 不超过 5、shape 为正，并检查 block shape。descriptor 不是对任意视图的通行证；零尺寸、尾部或其他尺寸必须同时满足包装的 reshape 与 descriptor 条件，不能只看 kernel 中 `cdiv` 就宣称全部支持。

## 4. Scale 的 packed 布局为什么需要再解一次

朴素的 scale 矩阵 `[M,K/V]` 在数学上直接，却不等于 Tensor Core/TMA 所用的批量搬运顺序。教程先按 128 行、四个 K 组组织 scale，以连续载入计算需要的片段；设备 load 后再恢复 `tl.dot_scaled` 的逻辑 scale shape。

将 packed 的概念 shape 写为

$$P:[M/128,\ K/(4V),\ 32,\ 4,\ 4].$$

源码实际先创建 `[M/128,K/(4V),32,16]`，再 reshape 为 descriptor 使用的 `[1,M/128,K/(4V),2,256]`；它们有相同的元素数与线性内容，但维度服务于不同搬运接口。

恢复逻辑矩阵采用

```python
# 相同元素的逻辑关系示意，省略 batch 与 tile 起点。
S = P.permute(0, 3, 2, 1, 4).reshape(M, K // V)
```

整理者从这次 permute 推导出的对应关系为

$$S[128i+32u+r,\ 4j+v]=P[i,j,r,u,v],$$

其中 $0\le r<32$、$0\le u,v<4$。这解释了为什么不能将 `[32,4,4]` 不经转置直接展平为行和组：行的低七位是 $32u+r$，不是 $4r+u$。

在上述 $M=128,K=256,V=32$ 的例子中，A 的 P shape 为 `[1,2,32,4,4]`；逻辑 `S[35,5]` 来自 `P[0,1,3,1,1]`，packed 线性偏移为 565，缩放 A 第 35 行的 K 元素 `[160,192)`。B 的行维代表输出列 n，应用同一转换；它最终仍是 `[N,K/V]`，不跟随 `b.T` 再转置。

该例的 `rep_m=1`、`rep_n=2`、`rep_k=2`，A/B 的 descriptor load block 分别为 `[1,1,2,2,256]`、`[1,2,2,2,256]`。所以每轮取回 1024/2048 个尺度，恢复为 `[128,8]` 与 `[256,8]`，正好覆盖两侧的当前 tile。（源码 `initialize_block_scaled` 的 scale reshape、`unpack_scale` 与设备函数的 `reshape/trans`。）

## 5. 点积调用与累加如何衔接

固定设备函数的每个 program 负责一个输出 tile；沿 K 循环时依次 load A/B 字节及其 packed scales，恢复逻辑 scale，执行以下 MXFP4 分支：

```python
# 来自固定教程设备主循环的核心调用；不是独立可运行 kernel。
accumulator = tl.dot_scaled(
    a, scale_a, "e2m1", b.T, scale_b, "e2m1", accumulator
)
```

MXFP8 分支两侧格式为 `"e4m3"`，mixed 分支分别为 `"e4m3"`、`"e2m1"`。NVFP4 和 MXFP4 虽然这里都传 `"e2m1"`，其 scale dtype、逻辑 shape 与共享粒度不同；格式字符串本身并不完整描述全部输入语义。

accumulator 先创建为 `[BLOCK_M,BLOCK_N]` FP32，每轮点积加入该状态。当前 `core.py:dot_scaled` 只允许 FP32 `out_dtype`；结束后才转为包装选择的目标 dtype 并 store。因此要分别辨识低精度乘法输入、尺度和累加、最终输出。选择 FP8 输出在本教程中只是末尾转换，并没有演示带输出 scale 的模型量化流程。

每轮数据 K 偏移增加 `BLOCK_K/ELEM_PER_BYTE`，scale 的 K 偏移增加 `rep_k=BLOCK_K/(4V)`。`NUM_STAGES=4` 是教程循环的流水配置，不意味着仅从 Python 代码就能确认特定指令、驻留或速度；实际 IR/PTX/SASS 检查见[Tensor Core 页](tensor-core-quantized-gemm.md)第 6 节，等待与缓冲复用见[异步流水](gpu-async-copy-pipelines.md)。

## 6. 教程参考结果说明什么，A100 能用哪部分

`initialize_block_scaled(..., compute_reference=True)` 将输入位模式和 scale 解为 FP32，按同一 packed-to-logical 转换恢复尺度，每个 scale 沿 K 重复 V 次，再做缩放后 matmul。它用于核对布局与值语义，不能把对原始未量化随机张量的误差当作同一检查，也不能把 FP32 参考当作硬件逐步 bit-exact 算术。

`validate_block_scaled` 将该参考与教程 FP16 输出比较；源码的 `atol=rtol=1e-3` 是该输入生成与尺寸下的上游检查配置，本次没有执行，不能直接推广为任意模型/数值范围的误差门槛。扩展尺寸时，先核对整除、scale tile 与 descriptor 条件，再明确输出舍入契约，按[正确性与测量](kernel-correctness-and-benchmarking.md)选择验证范围。

归档教程的 NVIDIA `supports_block_scaling` 检查 compute capability major 为 10 或 11，正文针对 Blackwell 一类原生块缩放 Tensor Core 的搬运和执行。**A100（SM80）不满足这条教程路径的硬件入口**；不能因为源码存在就把此 descriptor 实例当作 A100 上可直接运行的原生 FP4/FP8 kernel。

同时，固定 `core.py:dot_scaled` 文档说明了无原生 microscaling 硬件时的实验性软件模拟：通常先上转 BF16 后做 dot，部分后端另有规则。这是语言接口的另一种 lowering，不代表本教程的 descriptor/原生路径已经在 A100 可用，更不表示获得原生低精度算力。A100 上学习本页的编码、K 尺度与布局关系后，应结合实际环境选择解码/转换及普通 FP16/BF16 GEMM，或已有[Marlin W4A16](marlin-batched-w4a16-gemm.md)路径；不同存储格式须先核对转换，不能直接互换。

## 来源身份

- Triton 官方 Block Scaled Matrix Multiplication 教程（`https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html`），2026-07-02T02:41:08+08:00 快照，`block-scaled-matrix-multiplication-triton-documentation-snapshot-2026-07-02`；使用格式、scale 布局与执行解释，未沿用性能图作本地结果。
- Triton 固定源码（`https://github.com/triton-lang/triton/tree/81a46fa0c04526e5df55a018ecfab72ff922f592`），`81a46fa0c04526e5df55a018ecfab72ff922f592`；`python/tutorials/10-block-scaled-matmul.py` 的 NVIDIA 初始化、descriptor、设备循环、包装与参考比较；`python/triton/tools/mxfp.py` 的编码/解码/打包；`tools/tensor_descriptor.py` 的接口约束；`language/core.py/semantic.py` 的 `dot_scaled` 定义与形状检查。AMD 与底层后端具体指令分派未完整核读。
