---
title: 张量并行与量化：分片、通信和元数据
type: implementation
tags:
  - tensor-parallelism
  - inference
  - weight-quantization
sources:
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/layers/linear.py
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/models/llama.py
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/layers/quantization/auto_awq.py
  - raw/repositories/2026-09-21/sglang/source/python/sglang/srt/layers/quantization/awq/schemes/awq_linear.py
  - raw/repositories/2026-09-21/sglang/source/python/sglang/srt/layers/linear.py
  - raw/repositories/2026-09-21/sglang/source/python/sglang/srt/models/llama.py
updated: 2026-09-22
---

# 张量并行与量化：分片、通信和元数据

张量并行（TP）让多个 rank 共同计算同一批 token 的一个层。它拆的是一次运算的张量轴；数据并行把不同请求或样本分给不同副本，流水线并行则拆层。本文以 vLLM/SGLang 的普通稠密线性层为例，解释 TP 如何与量化布局相遇。

## 1. 列并行保留输出分片，行并行相加输入分片的贡献

使用数学记法 $Y=XW$，$X\in\mathbb R^{M\times K}$、$W\in\mathbb R^{K\times N}$。这里的“行、列”按右乘 $W$ 命名；PyTorch 保存的 `weight` 常为 $[N,K]$，不能直接按文件维度名称猜轴。

**列并行**把输出维分成 $p$ 份：

$$W=[W_0,\ldots,W_{p-1}],\quad Y_r=XW_r,\quad Y=[Y_0,\ldots,Y_{p-1}].$$

每个 rank 拿完整 $X$，产生 $[M,N/p]$。下一步能消费分片时无需马上通信；若需要完整 $Y$，用 all-gather 拼接。vLLM `ColumnParallelLinear.forward` 只有在 `gather_output` 且 TP 大于 1 时调用 all-gather。

**行并行**把归约维 $K$ 分成 $p$ 份：

$$X=[X_0,\ldots,X_{p-1}],\quad W=\begin{bmatrix}W_0\\\vdots\\W_{p-1}\end{bmatrix},\quad
Y=\sum_{r=0}^{p-1}X_rW_r.$$

每个 rank 算出完整形状 $[M,N]$ 的部分和。all-reduce 把部分和相加，使各 rank 得到完整结果；它不是拼接。若每个 rank 都在归约前加同一个 bias，结果会变成 $XW+pb$。两库所选 `RowParallelLinear.forward` 都处理了这个问题，普通路径仅 rank 0 将 bias 融入局部 GEMM。

教学例：$X=[1,2,3,4]$，右乘权重的两行分片为 $W_0=[1,1]^\top,W_1=[2,2]^\top$，局部结果分别为 3 和 14，相加得 17。拼接 `[3,14]` 不是原来的线性层输出。

## 2. 为什么 MLP 经常采用列并行接行并行

Llama 风格 MLP 为 $[\operatorname{SiLU}(XW_g)\odot XW_u]W_d$。按中间维 $F$ 同步切 gate/up，rank $r$ 可以独立计算匹配的两段，再把 $W_d$ 沿输入维切片：

$$U_r=\operatorname{SiLU}(XW_{g,r})\odot XW_{u,r},\qquad Y=\sum_rU_rW_{d,r}.$$

因此 gate/up 后不用先 all-gather；在 down 后归约即可。合并 checkpoint 的 gate/up 不能只按整个拼接数组平均切一刀，否则某个 rank 可能拿到 gate 而缺少配对的 up。两库的 `MergedColumnParallelLinear` 和 `LlamaMLP` 对应这一组织。

普通 attention 同样可在 rank 内处理各自 query heads，再由 o_proj 汇总输出贡献。这里描述的是常规 TP 链路；融合通信、序列并行或延后归约会改变通信位置，不能由类名断言每层一定启动几次独立 collective。

## 3. GQA 的 KV head 可能需要复制

vLLM `LlamaAttention` 和两库 `QKVParallelLinear` 区分两种情况：KV heads 不少于 TP rank 时按头分片；KV heads 更少时，在满足整除约束的条件下复制 KV heads。若全模型 $H_q=32,H_{kv}=8,p=16$，每 rank 有 2 个 query heads、1 个 KV head，每个全局 KV head 被两个 rank 使用。

这解释了为什么增加 TP 后，KV 内存不保证严格按 $1/p$ 缩小。权重 Q/K/V 的装载切片也必须匹配复制关系，不能把 Q 的 shard rank 原样用于 K/V。模型形状的定义见 [Transformer 与自回归推理](../fundamentals/model/transformer-autoregressive-inference.md)。

## 4. 量化时必须一起切码值、尺度与零点

以已经恢复为标准逻辑顺序的 AWQ 表示为例，$G$ 沿 $K$ 分组，权重码为 $[K,N]$，scales/zeros 为 $[K/G,N]$。实际 int32 每字保存多个码，装载还要按打包轴换算偏移。

| TP 分法 | 逻辑码值切片 | 元数据切片 | 关键前提 |
| --- | --- | --- | --- |
| 列并行 | 输出列 $N_r$ | 同一组的相同输出列 $N_r$ | 打包单元与后端输出 tile 对齐 |
| 行并行 | 输入行 $K_r$ | 对应 group 行 | shard 边界与 group 的关系已定义 |

教学例：$K=256,N=128,G=128,p=2$。列并行每 rank 得 $[256,64]$，scale 为 $[2,64]$；行并行每 rank 得 $[128,128]$，scale 为 $[1,128]$。若改为 $p=4$ 的行并行，$K_r=64<G$，一个全局 group 跨两个 shard。数学上可以复制全局 scale，但具体格式/后端必须支持这种契约；不能自动变成两个重新校准的 group，也不能假设加载器支持。

SGLang 所选 `AWQLinearScheme.create_weights` 检查分片输入维与 group size、输出维与 pack factor 的整除关系。vLLM `AutoAWQMarlinLinearMethod.create_weights` 则将全局/分片形状和量化配置交给 kernel 选择器，并按输入分片准备 group 元数据。GPTQ 的 `g_idx`、act-order、全局尺度与 Marlin 重排又有额外约束，见 [GPTQ 实现](gptq-implementation.md)及 [AWQ 实现](awq-implementation.md)。本页例子不替代这些后端检查。

## 5. 省掉权重流量，不代表通信同步缩小

权重从 FP16 变成 INT4，不会自动把行并行的 $[M,N]$ 部分和也改为 INT4。普通 all-reduce 的通信量由输出 shape、通信 dtype、rank 数与 collective 算法决定。算子越快，通信和调度可能越显眼；增加 rank 还会缩小本地 GEMM 的 $K/N$，降低局部复用或不再满足 tile 条件。

SGLang `RowParallelLinear.forward` 可通过 `skip_all_reduce` 或 `reduce_results` 延后/关闭此处归约，还存在 attention TP group 与受条件控制的量化通信分支。因此评估时应记录实际分派，不能把“有量化通信函数”写成“AWQ 一定量化通信”。

定位顺序是：核对各 rank 的数学切片和 group 身份 → 检查本地 kernel 支持形状 → 确认归约和 bias 次数 → 分开测局部算子、通信与服务延迟。后两项的测量口径见 [服务性能评测](serving-performance-evaluation.md)。本次为固定源码阅读与代数教学验证，没有运行多 GPU collective。

## 来源身份

- [vLLM](https://github.com/vllm-project/vllm/tree/568afb3a13806beb53bb2e6bd518269357b237c0)，`568afb3a13806beb53bb2e6bd518269357b237c0`；`layers/linear.py` 的三类并行线性层、`models/llama.py`、`layers/quantization/auto_awq.py:AutoAWQMarlinLinearMethod.create_weights`。
- [SGLang](https://github.com/sgl-project/sglang/tree/2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc)，`2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc`；`python/sglang/srt/layers/linear.py`、`models/llama.py` 与 `layers/quantization/awq/schemes/awq_linear.py`。
