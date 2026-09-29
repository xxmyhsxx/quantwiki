---
title: Kitty：动态 K 通道提精度与双层 2-bit 缓存页
type: method
tags:
  - kv-cache
  - mixed-precision
  - packing
  - triton
sources:
  - raw/papers/2026-09-21/kitty/paper.pdf
  - raw/repositories/2026-09-21/kitty/source/README.md
  - raw/repositories/2026-09-21/kitty/source/src/kitty/kvcache/kitty.py
  - raw/repositories/2026-09-21/kitty/source/src/kitty/kvcache/utils_kv_per_layer.py
  - raw/repositories/2026-09-21/kitty/source/src/kitty/kvcache/kernels/kitty_quant_pack.py
  - raw/repositories/2026-09-21/kitty/source/src/kitty/kvcache/kernels/kitty_attention.py
  - raw/repositories/2026-09-21/kitty/source/src/kitty/models/qwen3/modeling_qwen3.py
  - raw/repositories/2026-09-21/kitty/source/src/kitty_sim/kitty_simulate.py
  - raw/repositories/2026-09-21/kitty/source/src/kitty_sim/utils_quant.py
  - raw/repositories/2026-09-21/kitty/source/latency_benchmarking/benchmark_kitty.py
updated: 2026-09-29
---

# Kitty：动态 K 通道提精度与双层 2-bit 缓存页

Kitty 保留开头 sink 与近期缓冲，将历史 K 按 token 组逐通道量化、V 逐 token 量化，并把一小部分 K 通道从 2 bit 提高到 4 bit。系统创新是用“所有通道的低两位 + 被提升通道的高两位”表示一页，使变化的通道选择仍能进入规则的解包路径。

本页研读 v1 全部正文、结果表及伪代码，并定向追踪固定官方快照的 Qwen3 接入、缓存生命周期、选通道、打包、attention、模拟和计时入口。没有编译 Triton、运行模型或复现质量/性能。前置见 [KV 对象与粒度](../theory/kv-cache-quantization-objects-and-granularity.md)、[量化执行路径](../implementation/quantized-matmul-scaling-execution.md)。

## 1. 真实敏感度与在线代理不是同一个量

论文先做离线诊断：每次仅把 K 的一个通道量化到 INT2，其他通道保持浮点，然后测整个 softmax attention matrix 相对浮点的 MSE。GQA 中多个 query head 共享 K，图 2 分别显示各 query head 的误差。这是受 query、softmax 和其他通道共同影响的敏感度。

真正在线选择没有重算上述 MSE，而使用当前待量化 token 组内的平均幅值：

$$s_i=\frac1G\sum_{t=1}^{G}|K_{t,i}|,\qquad B=\operatorname{TopK}_i(s_i,D_{\rm boost}).$$

固定实现 `quantize_pack_k` 对每 batch、KV head、page 独立求 `abs().mean` 与 `topk`；一个页通常为 $G=128$ 个 token。这里的 dynamic 是页间可改变选择，不是每步重新量化全部历史页，也不是给整个模型学习永久 channel mask。

幅值是经验代理，不保证等于量化难度或 attention 重要性。恒定的大幅值通道可用 min–max 仿射重构得很准；而较小幅值通道若对应较大的 query 分量，仍可能显著改变 logits。对于只扰动第 $i$ 个 K 通道，logit 误差为 $q_i\Delta k_i/\sqrt d$，还要经过 softmax Jacobian；只看 $|k_i|$ 没有涵盖全部项。论文仅比较这个启发式与随机选择，未证明它是最佳敏感度估计器。

## 2. K / V 不同的分组与保护策略

默认 sink 数 $S=32$，页/分组大小 $G=128$，V local window $R=128$。K 在 RoPE 后进入缓存，不使用旋转或 pre-RoPE 存储。K 按同一通道跨 $G$ 个 token 求 min/max，boost 通道用 16 个码点，其余用 4 个码点；V 按每 token 的 head 维度求范围并用 2 bit。

| 部分 | K | V |
| --- | --- | --- |
| 开头 sink | 32 个 token FP16 | 同左 |
| 待量化缓冲 | Q-Buffer，积累至 128 个后形成一页 | Q-Buffer，收集从 local 淘汰的 token |
| 固定近期窗口 | 没有额外固定 local；Q-Buffer 长度周期变化 | 最近 128 个 token FP16 |
| 历史页 | 大部分 INT2，选中通道 INT4 | INT2 |

K 的 Q-Buffer 在页边界可变为空，不等价于始终保护最近 128 个 K。V 系统实现还可有不足一页的淘汰缓冲，因此其浮点历史不只有 local window。论文的算法模拟与实际系统在这一点上不同，见 §6。

对一组范围 $[m,M]$，实现使用 $s=(M-m)/(2^b-1)$、$q=\operatorname{round}((x-m)/s)$、$\widehat x=sq+m$。保存的是 FP16 scale 和实数域最小值；代码称 `zero_point` 的字段实际是 $m$，不是整数零点。K kernel 的一处注释写成 per-token，但实际 `x` 为 `[G,D]`，沿 axis 0 归约，确实是逐通道。

## 3. 双层 2-bit 页怎样恢复 INT4

对整数码 $q\in\{0,\ldots,15\}$：

$$q_{\rm low}=q\mathbin{\&}3,\qquad q_{\rm high}=(q\gg2)\mathbin{\&}3,\qquad
q=q_{\rm low}\mathbin{|}(q_{\rm high}\ll2).$$

低两位存所有 $D$ 个通道，高两位只存 $D_{\rm boost}$ 个选中通道。每个字节放同一通道连续四个 token 的两位码，位移为 0、2、4、6。保存一个长度 $D$ 的 uint8 映射，将逻辑通道对应到紧凑高位数组；未提升通道不读取高位，默认高位为零。

例如码 13 的低位为 1、高位为 3，合成 $1|(3\ll2)=13$。它是**一个 INT4 码的位平面拆分**，不是两个具有独立 scale 的 INT2 量化结果，也不是残差量化。scale/min 仍按原通道及其实际位宽选择。

论文伪代码采用 1-based 索引、哨兵 $D_{\rm boost}+1$ 和 `<=`；固定代码采用 0-based 索引、哨兵 $D_{\rm boost}$ 和 `<`。两套约定各自一致，不能混用。所谓稀疏部分按“被提升通道”紧凑保存，并非逐个删除数值为零的高位元素。

`qk_kernel` 查页表、读取低位/映射/选中高位，在片上反量化到 FP16 再做 `tl.dot`。这保留了 token 轴连续访问，但仍有数据依赖的 channel 映射、附加读取和解包指令；“没有 scattered reads”不能解释成所有地址与 mask 都固定或没有间接索引。

## 4. 接入与执行路径

固定快照中可追踪到如下路径：

1. 自定义 Qwen3 attention 做 Q/K normalization 与 RoPE，然后 `KittyCache.update` 写入新 KV。
2. prefill 使用原 attention 接口和完整浮点输入，之后 `quantize_prefill` 才分离 sink、尾部缓冲和历史页。
3. decode 仅支持单个新 token，`kitty_attention_forward` 依次执行 `qk_kernel`、PyTorch FP32 softmax、`sv_kernel`。
4. attention 完成后 `quantize_decode` 检查满 Q-Buffer，量化并打包一页，重置计数；约每 128 步摊销一次页量化。

这里没有先在 HBM 中还原整个浮点 KV，但会物化形状 `[B,H_Q,T]` 的 attention 分数，也不是 FlashAttention 式一个融合 kernel。低比特主要节省存储/读取，不意味着直接在 INT2 Tensor Core 上完成 attention。

`KVCache_Layer` 预分配 `MAX_BS × ceil(MAX_LEN/G)` 页并固定填写页表。因此“分页布局”已经实现，却不等于拥有动态页回收、连续批处理或跨请求前缀共享。代码没有把通用 decode mask 传给该自定义接口，计数也由整批共享；可确认的是当前固定批量、同步长度的 Qwen3 原型，不能据此宣称任意 padding、变长调度和所有 HF 模型均已兼容。

## 5. 从代码计算真实存储

设 boost 比例 $r=D_{\rm boost}/D$。只算历史 payload，K 为 $2+2r$ bit/元素，K/V 平均为 $2+r$。Kitty 的 $r=12.5\%$ 与 Kitty-Pro 的 $r=25\%$ 分别对应 2.125 和 2.25，均不是纯 2 BPE。

固定实现每 K page/head 还存 $D$ 个 uint8 映射及 $2D$ 个 FP16 元数据，每 V page/head 存 $2G$ 个 FP16 元数据。完整填满的量化页平均为

$$b_{\rm page}=2+r+\frac{20}{G}+\frac{16}{D}.$$

当 $G=D=128$，Kitty 为 2.40625、Pro 为 2.53125 BPE；Pro 的纯页数据相对 FP16 约压缩 6.32 倍。此处仍未计 int64 页表、FP16 sink/Q-Buffer/local、预分配空页、attention 中间张量与模型权重。因此论文“接近 8×”只能按其测量口径理解，不能用于精确显存预算。

当前 `KittyCache` 把 `d_boosted=head_dim//4` 写在构造中，即默认 **Kitty-Pro**。把 `Kitty` 函数名直接对应论文 12.5% 行会选错配置。增大 boost 既改变 payload，也改变高位访问和解包开销。

## 6. 质量模拟与打包系统存在可见差别

论文也明确将质量模拟与系统测速分开。固定代码进一步显示：

| 核对项 | 质量模拟 | 打包系统 |
| --- | --- | --- |
| 张量存储 | fake quant 后仍是浮点 | uint8 payload + FP16 元数据 |
| 舍入 | `torch.round`，半整数取偶数 | 非负码使用 `floor(x+0.5)` |
| 微小范围 | 先把范围限制到 $10^{-4}$（半精度）再除码点数 | 把最终 scale 限制到 $10^{-6}$ |
| V 老化 | local 外的 token 逐步 fake quant | local 淘汰后仍先留 Q-Buffer，满页才打包 |
| K 页边界 | decode 在越过整缓冲边界后一 token 触发，返回量化前副本 | 填满页当步 attention 后打包，下一步已读量化页 |

因此模拟的任务质量和实际系统速度不能自动拼成“同一 bit-exact 路径已同时验证”的结论。特别是舍入、元数据精度、页边界/浮点驻留时间，需用相同 token 流逐层对齐才能证明差异不影响结果。这里是静态代码发现，并未测得这些差异的实际质量影响。

质量 helper 内虽然有 variance selection 分支，配置校验只允许 0/1（随机/幅值），论文也没有报告方差选择实验，不能把存在分支当成已评估能力。

## 7. 质量与速度的证据范围

| 作者实验 | 主要结果 | 限定 |
| --- | --- | --- |
| Qwen3-8B，四项任务平均 | FP16 77.15，KIVI-K2V2 61.39，保护 sink 后 69.80，Kitty 74.97，Pro 76.18 | sink 与通道提精度各有贡献；Pro 仍低 0.97 点 |
| LLaMA3.1-8B，四项平均 | FP16 53.70，Kitty 51.84，Pro 52.59 | 并非所有模型恢复到零差距 |
| LLaMA3.3-70B，四项平均 | 保护 sink 的 KIVI 73.56，Kitty 73.53，Pro 73.86，FP16 73.89 | 此例通道提精度收益很小，不能称每个模型都明显优于 sink-only |
| Qwen3-8B，AIME24/25，生成上限 32K | FP16 平均 68.84，Kitty 65.17；AIME25 单项 66.00→59.67 | 32K 是允许生成上限，不是每题实际使用 32K；仍有明显质量差距 |

质量实验使用 CoT、随机采样（temperature 0.6、top-p 0.95、top-k 20），一般生成上限 4096，重复 3–10 次。表中 ± 是作者所述最大观测偏差，不是标准差或 95% 置信区间；“均值略高于 FP16”不能直接解释为显著改善。表 3 的四项平均比正文概括更适合精确引用。

系统实验是单 A100 80GB、Qwen3-8B、短 prompt、总序列目标 8192，按不同 batch 测量。Kitty-Pro 通过更大的可容纳 batch，报告同显存预算下最多 8× batch 和 2.1–4.1×吞吐。它不是所有相同 batch 请求的延迟收益，也不代表长输入 prefill 同样加速。

固定 `benchmark_kitty.py` 的计时覆盖构造 Kitty cache 与 `generate` 循环，包含 prefill，使用 `max_length`，不是排除 prefill 的 NOVA decode-window 口径。其 HF quantized 分支打印 HQQ，但实际传入 `backend=quanto`；依执行参数判定，不能依打印文本判定，也不能把这份脚本当作论文表格确切运行配置的证明。

## 8. 在 KV 研究中的位置

[KVQuant](kvquant.md)用离群值旁路与敏感度码本，[AKVQ-VL](akvq-vl.md)按模态和 pivot token 分配精度，Kitty 按页选择 K 通道并解决混合码宽布局；[NOVA-KV](nova-kv.md)则通过查询加权变换与 VQ 改变误差度量和码本结构。它们的“重要性”所在轴不同，不能把保留 token、保留通道、优化重构目标看成同一种策略。

Kitty 最值得复用的是混合码宽的可执行表示，以及诊断敏感度与廉价在线代理的分离。后续实验先验证打包路径与模拟的误差/边界一致性，再比较相同有效 BPE、相同质量目标下的吞吐；不能只复现一个 `topk` 就认定重现了论文系统。

## 来源身份

- [Kitty，arXiv:2511.18643v1](https://arxiv.org/abs/2511.18643v1)：§3 诊断与在线代理，§4 图 3/算法 1/流水线，§5 表 2–4 与图 4–5。
- [Summer-Summer/Kitty 固定 commit dfd2c07b](https://github.com/Summer-Summer/Kitty/tree/dfd2c07b407d6b407179359207c612ab631f3ed1)：上述登记文件中的 `KittyCache`、`KVCache_Layer`、`quantize_pack_k/v`、`qk_kernel/sv_kernel`、Qwen3 attention、`KittyKVCache`、`fake_quant_groupwise_lastdim` 与 benchmark 分支。存储公式及静态差异是整理者按此版本核对，不代表上游后续版本或本地运行结果。
