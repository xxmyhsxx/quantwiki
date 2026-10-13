---
title: 推理服务的内存管理与批处理
type: implementation
tags:
  - llm
  - serving
  - memory
  - kv-cache
  - performance
sources:
  - raw/papers/2026-09-21/sarathi-serve
  - raw/papers/2026-09-21/distserve
  - raw/papers/2026-09-21/qserve
  - raw/papers/2026-09-23/disaggregated-quantization
  - raw/papers/2026-09-21/saw-int4
  - raw/papers/2026-09-21/pagedattention
updated: 2026-10-14
---

# 推理服务的内存管理与批处理

量化把每个权重、每个缓存元素压小，但能同时服务多少请求，取决于服务系统怎么用这些省下来的空间。同一个量化模型接进不同的引擎，有效批量不同，吞吐就不同——这也是为什么「量化之后快了多少」不能只由模型和内核决定。本页整理部署侧第一层知识：KV cache 的内存管理、批处理与调度。

本页的分页机制与相应实验依据 Efficient Memory Management for Large Language Model Serving with PagedAttention（arXiv:2309.06180，SOSP 2023，下称论文，vLLM 的原始论文，全文研读；抽取文本缺表格，数值按正文与图注记录）；第 5.1 节另接续 DistServe v3 的阶段分离。本 Wiki 中 KV cache 作为量化对象的结构见 [KV cache 量化的对象与粒度](../theory/kv-cache-quantization-objects-and-granularity.md)，后端与格式支持面见 [量化模型的部署框架与后端支持](quantized-llm-deployment-backends.md)。本页保留各论文的机制与实验口径。固定源码中的 token 调度、chunked prefill、前缀复用和批次准备另见 [vLLM 推理执行](vllm-inference-execution.md)；Radix 前缀树的保护、淘汰与批次映射见 [SGLang 推理执行](sglang-inference-execution.md)。这些源码分析不代表已研读对应服务论文。

## 1. KV 容量怎样限制可用批量

论文 §1 的动机分析：自回归生成逐 token 进行，每生成一个 token 都要读一遍权重与历史缓存，因此论文关注的小批量生成场景常受内存带宽限制，GPU 算力被闲置；提高吞吐要靠同时批处理多个请求。而批量能开多大，受到显存里 KV cache 占用的限制——论文给出的 A100 40GB 例子中，约 65% 显存被静态权重占用，接近 30% 用于请求的动态状态（即 KV cache），剩余少部分是临时激活。**在这一设置下，权重驻留固定、临时激活占比较小，KV 管理决定了可用批量。**

论文还给出两条量级：OPT-13B 每个 token 的 KV cache 需 800 KB（$2\times5120\times40\times2$ 字节，即键值两份、隐藏维、层数、FP16 每元素两字节），若序列长 2048，单请求缓存可达 1.6 GB；硬件趋势上，从 A100 到 H100 算力翻倍以上而显存上限仍停在 80GB，用于说明该论文关注的容量压力；这不是对所有硬件型号、batch 和长度的普遍判断。

显存容量决定能容纳多少状态，内存带宽决定搬运这些状态的时间，算力则约束计算速度，三者不能互相替代。大批量、长上下文和多卡执行会改变主导瓶颈；比较 TTFT、TPOT、吞吐及延迟目标下的有效吞吐，见 [服务性能评测](serving-performance-evaluation.md)。

## 2. 既有实现的三类浪费

论文讨论的既有服务实现为每个请求按**最大可能长度**预分配连续空间。论文把浪费分成三类：为未来 token 预留的槽位、因按最大长度过度分配造成的内部碎片、以及分配器带来的外部碎片。作者实测这些系统的有效 KV 内存占比低至 **20.4%**（区间 20.4%–38.2%）。压缩（compaction）看似是解决办法，但在 KV cache 规模下代价过高，而且预分配的整块空间也阻碍了共享。

## 3. 分页：PagedAttention 与块表

论文借用操作系统的虚拟内存与分页，把每个序列的 KV cache 切成固定大小的 **KV 块**，块内放到不同物理位置；注意力按块计算：

$$A_{ij}=\frac{\exp(q_i^{\mathsf T}K_j/\sqrt d)}{\sum_{t=1}^{\lceil i/B\rceil}\exp(q_i^{\mathsf T}K_t/\sqrt d)\mathbf 1},\qquad o_i=\sum_{j=1}^{\lceil i/B\rceil}V_jA_{ij}^{\mathsf T},$$

其中 $K_j,V_j\in\mathbb R^{d\times B}$ 是第 $j$ 个键块与值块，$B$ 是块大小，指数逐元素计算，$\mathbf1$ 将块内权重求和。未填满位置与未来 token 在指数前屏蔽为 $-\infty$；归一化分母覆盖全部有效块，不能分别对每块做 softmax 后直接相加。

这里是按普通注意力定义整理的公式修正：PagedAttention v1 §4.1 的块公式把 $\mathbf1$ 写在指数内部，与同文逐 token 定义不一致。正确顺序是先逐 token 取指数，再跨 token 求和；分页只改变存储与读取，不改变归一化对象。内核按块表逐个取块，允许块在物理显存中不连续。

内存管理器维护**逻辑块到物理块的映射（块表）**，每条记录还保存该块已填充的位置数：请求的 KV cache 表示为一串从左到右填充的逻辑块，新物理块只在需要时才分配。这样每个请求的浪费被限制在最后一个未填满的块内，从而几乎消除前述三类浪费。请求结束后其块被释放给其它请求使用。

**块大小是可调取舍。** 块越大，注意力内核一次能并行处理的 KV 位置越多、利用率越高；块越大也会增加内部碎片、降低可共享的概率。论文的消融显示：在 ShareGPT 轨迹上 16 到 128 都表现良好，而在序列较短的 Alpaca 轨迹上较大块显著变差；于是默认取 16。

## 4. 共享与写时复制

分页让「同一个物理块被多个逻辑块引用」变得自然，论文用**引用计数 + 块粒度的写时复制**实现共享，并给出三种场景：

| 场景 | 可共享的部分 | 论文报告的内存节省 |
|---|---|---|
| 并行采样（一个提示多个输出） | 提示部分的 KV | Alpaca 6.1%–9.8%，ShareGPT 16.2%–30.5% |
| 束搜索（beam search） | 提示与候选之间的共同前缀，且随解码演化 | Alpaca 37.6%–55.2%，ShareGPT 44.3%–66.3% |
| 共享前缀（系统提示、few-shot 示例） | 预置前缀的物理块 | 吞吐相对 Orca（Oracle）：单示例前缀 1.67×，5 示例前缀 3.58× |

机制细节值得注意：被多个序列共同引用的块在写入时才复制（与操作系统的 fork 相同），因此只复制一个块；束搜索中失效候选的块在引用计数归零后释放。论文指出，既有系统在这种场景下需要频繁整段拷贝 KV cache，这是共享带来的主要节省来源。

## 5. 批处理与调度

论文 §2.3 说明批量能摊销权重搬运，但请求到达时间与长度都不同；按请求级批处理会让先到的请求等待或让后到的请求排队，还会为对齐长度而填充计算。此前提出的**迭代级调度**（亦称连续批处理）改为每轮迭代后移除已完成请求、加入新请求，因此可在迭代边界接纳新请求，并减少按最长请求填充的浪费；实际接纳仍受资源与调度约束。这里讨论逻辑批处理，不排除具体实现为 CUDA Graph 捕获规格补齐 token，后者见 [vLLM 算子设计](vllm-attention-operator-design.md)。

在长度未知与显存有限的前提下，论文实现的是先来先服务调度，并在显存不足时抢占：驱逐以**序列为整体**（all-or-nothing），同一请求内的多个序列作为一个**序列组**整体抢占或恢复。被驱逐的缓存有两条恢复路径——**交换**（把块拷到 CPU 内存，交换空间上界由 GPU 上用于 KV 的显存决定）与**重算**（重新执行提示阶段生成缓存，由于解码出的 token 可以与原提示拼接成新提示，重算延迟通常低于原始延迟）。两者的相对优劣取决于 CPU–GPU 带宽与 GPU 算力，论文以实验比较。

分布式方面，论文采用张量并行的 SPMD 执行：注意力按 head 切分，各 worker 处理同一批 token 的不同 head，因此共享同一个中心化的 KV cache 管理器与块表映射；worker 只在每轮开始时接收控制消息，运行中通过 all-reduce 同步，不在内存管理上互相协调。

### 5.1 DistServe：两阶段实例与 KV 转移

DistServe v3 把 prefill 和 decode 放在不同实例上；每个实例持有同一模型的一份完整权重，实例内部仍可用 TP 拆算子、PP 拆层。它解除的是两阶段的执行与资源耦合，不要求为两阶段训练不同权重。本页第 6 节的阶段独立权重/SSD 流式方案另有生命周期。（DistServe §2.3、§4，图6；并行轴见[张量并行](tensor-parallel-quantization.md)。）

#### 阶段干扰与各自的批处理目标

混合批次中的 decode 要等待长 prefill 所在轮次结束，生成间隔随之变长；decode 行也占用本来用于 prompt 的资源，可能增加 TTFT。分轮执行或改变优先级仍会让另一阶段排队。Chunked prefill 可以缩短单次占用，但小 chunk 可能利用率不足，并须反复读取前面 chunk 的 KV；这些是作者在 v3 设置下讨论的代价，不能据此判断所有现代混合调度都较慢。（§2.3，图2。）

分离后，prefill 按目标模型/GPU profile 饱和长度 $L_m$：短 prompt 可以拼成接近该长度的批次，已经饱和的长 prompt 再加请求通常只延长批时间。Decode 则聚合请求摊销权重读取；但有效 batch 仍受 KV 容量与 TPOT 限制，batch 很大时也可能接近计算受限。这解释了两阶段为什么可能需要不同的 TP/PP 与实例数量。（§3.1–3.2、§4.3，图3–5。）

#### 执行时间与排队决定 TP/PP 的取舍

作者用固定 prompt 长度、泊松到达、FCFS、单请求已饱和且不批处理的简化模型说明取舍。设 R 为 requests/s，D 为串行服务时间，稳定条件 $RD<1$ 下的 M/D/1 平均 TTFT 为

$$\overline T=D+\frac{RD^2}{2(1-RD)}.$$

第一项是执行时间，第二项是排队。在理想两段 PP 中，总执行时间近似 D、最慢阶段近似 D/2；两路 TP 的加速系数为 $1<K<2$，执行时间为 D/K。作者式2–3改写为

$$\overline T_{PP}\approx D+\frac{RD^2}{4(2-RD)},\qquad
\overline T_{TP}=\frac D K+\frac{RD^2}{2K(K-RD)}.$$

按这两式的分母，分别需 $RD<2$、$RD<K$ 才处于对应稳定区间。低到达率时 TP 缩短执行时间有利；到达率较高时，PP 的流水服务容量可能更有利于减少排队。这是论文的固定服务时间/理想流水近似，变长请求会产生气泡，通信又影响 K，不能用它直接选择真实部署配置。（§3.1，式1–3与图4。）

#### 带宽约束怎样进入配置搜索

算法1面向高跨节点带宽：先枚举 TP/PP，按权重容量筛选候选，用历史请求长度与到达分布驱动模拟器，二分搜索各阶段满足 SLO 的最大 rate；分别选 goodput/GPU 最佳配置，再按总目标 rate 向上取整所需副本数。它依赖有限候选集与延迟估计，不是任意负载下的最优性证明；权重能装入也不代替 KV/工作区预算。（§4.1，算法1。）

跨节点带宽有限时，算法2利用“KV 只需交给对应层”这一结构：为两阶段枚举共同 PP 阶段，把对应层段放在同一节点，让 KV 走节点内 NVLink；再枚举各段的节点内资源组合，限制配对资源不超过节点 GPU 数，联合模拟并复制配置。两阶段的 TP 与资源量仍可不同。作者的 OPT-66B、512-token 例子中 KV 约1.13GB，10 requests/s 对应约11.3GB/s（约90Gbps）；这是平均流量量级，不能单凭它证明某个请求的传输已隐藏。（§3.3、§4.2，算法2。）

附录A的模拟器延迟模型以 profile/interpolation 拟合系数：prefill 区分总 token 数 $\sum l_i$ 与长度平方和 $\sum l_i^2$，decode 区分权重读取项与历史 KV 项。这些是该模型/硬件与内核下的近似，量化、GQA 或新的 kernel 改变流量和计算时需要重新核模型，不能沿用系数当通用规律。

#### 首 token、KV 缓冲与 decode 的接纳节奏

控制器按 FCFS 将请求送到最短 prefill 队列，再交给负载最低的 decode 实例。Prefill 产生首 token 和 prompt KV；decode 消费该 token 及已传入的 KV，继续生成。被采样的首 token 尚未形成自己的 KV，阶段交接仍遵循[自回归步骤](../fundamentals/model/transformer-autoregressive-inference.md#3.%20Prefill%E3%80%81decode%20%E4%B8%8E%20teacher%20forcing)。

遇到突发流量，decode 按自身接纳节奏 **pull** KV；等待转移的 KV 暂留在 prefill GPU，作为队列缓冲。这避免立即把突发状态压入 decode 显存，却仍占用 prefill 的容量。论文报告节点内异步 CudaMemcpy、跨节点 NCCL，但没有在本次阅读中核到真实释放、同步、格式重分片或失败恢复协议。负载模式显著变化时重新搜索 placement；抢占与容错仍是论文未来工作。（§4.3–5，图6。）

这些机制针对有延迟目标且有资源分配余地的服务。离线吞吐优先、单卡/资源受限部署未必适合；百万上下文收益只属 §7 讨论。该论文的实验口径和版本内差异见[DistServe 的评测条件](serving-performance-evaluation.md#4.1%20DistServe%20%E7%9A%84%E5%AE%B9%E9%87%8F%E7%9B%AE%E6%A0%87%E4%B8%8E%E5%AE%9E%E9%AA%8C%E6%9D%A1%E4%BB%B6)，不以当前 vLLM/SGLang 源码代替其历史基线。

### 5.2 Sarathi-Serve：剩余 token 预算与混合轮次

Sarathi-Serve v3 在同一实例内用有界混合批次减轻阶段干扰。迭代级调度改变每轮的请求成员；hybrid batching 把 prefill/decode 放入同一轮；chunked prefill 把一个 prompt 分多轮。三者各有作用：论文当时 Orca 已有 hybrid batching，完整长 prefill 仍会拉长混合轮次；当时 vLLM 的分轮基线也会暂停已有 decode。这些历史基线不代表[当前固定 vLLM](vllm-inference-execution.md)的全部路径。（Sarathi §2.5、§3.2，图7。）

#### 用 decode 轮次的算力余量处理 prefill

不同请求的 token 可以合为线性层的行数 M，摊销权重读取。在权重流量主导的区间，增加一些 prefill 行只小幅增加线性层时间；越过拐点后，计算时间随 M 上升。因此需要限制本轮 prefill 量，而非把任意完整长 prompt 并入 decode。图6在 LLaMA2-70B/A100 的不同 TP 下展示了先平坦后上升的趋势，拐点依赖硬件/并行与固定开销，不能把论文的 token 数当通用阈值。（§3.1；[roofline 的强度关系](../fundamentals/hardware/arithmetic-intensity-and-roofline.md#2.%20%E4%B8%A4%E4%B8%AA%E5%BC%BA%E5%BA%A6%E9%87%8F)。）

给定 TBT 目标，profile/模拟选定总 token 预算 $\tau$。每轮先放入运行中的 decode，各贡献一个 token；再接续未完成 prefill；最后在剩余预算与 KV 接纳条件允许时加入新请求。Chunk 长度按剩余预算与剩余 prompt 长度决定，不是每个请求固定得到 $\tau$ 个 token。（§4.1–4.2，算法3。）

在运行 decode 数不超过预算、KV 接纳可行的教学情形中，令 D 为本轮 decode 请求数、$c_i$ 为各 prefill chunk 长度，则

$$M=D+\sum_i c_i\le\tau.$$

例如 $\tau=512,D=64$，prefill 最多贡献448行。这是对调度正文的整理，不是论文原式或完整实现。算法3第18行写 `B←R_new`，内层也没有显式取下一个新请求；图注称可选一个 ongoing prefill，正文/循环则遍历未完成 prefill，不能逐行照抄成已核可运行调度器。

“Stall-free”指持续推进已有 decode，避免等待一个完整长 prompt 的专用 prefill；混合轮次仍可能比纯 decode 慢。图9的 LLaMA2-70B/4×A100、context length4096、decode batch1、预算512例子中，chunk 混合轮次约为纯 decode 的3.7×，完整 prefill 混合轮次为28.3×。它说明干扰减轻，不证明延迟不变或 TTFT 最优。（§4.2，图9。）

#### Chunk 的历史依赖与流水线气泡

当前 chunk 的 attention 仍需读取同请求前面各 chunk 的 KV，并保留当前段内的因果范围；切分没有删除历史。N 个等长 chunk 中，第一段作为历史被后续读取 N−1 次，第二段 N−2 次，依此类推。教学上共有 $N(N-1)/2$ 个“历史 chunk 读取单位”；这是依赖记账，不是作者实测 HBM 字节或与完整 prefill 的流量差。小 chunk 另增加启动/准备成本和利用率损失。（§4.3。）

PP 气泡有三类来源：相邻 prefill 的长度不同、prefill 与 decode 工作量不同、不同 decode 的历史长度不同。Sarathi 用有界且较均匀的混合 token 批次减少前两项；第三项仍使相同 M 的轮次耗时不同，所以不能写成消除所有气泡。较小预算有利于降低 TBT，却可能增加 prefill/TTFT 开销；较大预算提高效率，又可能扩大轮次差异。Tile 对齐也影响利用率，作者用 Vidur 辅助 profile/模拟选择预算，动态预算留作未来工作。（§3.3、§4.3，图8。）

量化改变权重流量、矩阵乘接口及拐点后，应重核预算。与第5.1节的阶段分离相比，本机制在同一实例组合工作；DistServe 另需转移 KV、分别配置阶段实例。Sarathi §6只作定性讨论，将两条路线的定量性能比较留作未来工作，不能把不同论文的容量倍数直接排名。指标与消融条件见[Sarathi 的评测](serving-performance-evaluation.md#4.2%20Sarathi%20%E7%9A%84%20token%20%E7%BA%A7%20TBT%20%E7%9B%AE%E6%A0%87%E4%B8%8E%E5%8F%96%E8%88%8D)；本文未核作者实现或运行服务。

## 6. 与量化研究的接口

这四点决定了量化收益如何被服务系统放大或抵消：

**KV 位宽改变每 token 的存储字节数，从而改变有效批量。** 块内元素位宽减半，同样的显存能容纳更多块。论文的 OPT-13B 例子给出换算起点：每 token 800 KB 是 FP16 口径，位宽下降会按比例减少这一项，但残差窗口、逐 head 尺度等元数据会占回一部分——[KIVI](../methods/kivi.md) 的全精度残差需要额外保存和管理。[SAW-INT4](../methods/saw-int4.md) v1 §2 针对其统一分页池分析了集成成本，不能据此断言所有分页系统都无法支持残差窗口。

**量化释放出的显存不一定转化为吞吐。** 请求供给低于系统容量时，额外显存不自动增加完成请求数；量化仍可能降低单请求延迟。另一种情形是显存已足以容纳所需批量：论文 §6.2 的 OPT-175B + Alpaca 配置中，短序列和较大可用 KV 空间使 Orca 也能聚合大批量，作者据此判断系统转为计算受限，vLLM 的优势减小。评估量化收益时需报告并发、请求速率与实际瓶颈。

**逐请求速度与系统吞吐会背离。** 缓存占用更小的配置能开更大批量、跑更多请求，但单请求的解码速度可能因批量变大而下降；只看逐请求速度会得出反向结论，这一现象在 [SAW-INT4](../methods/saw-int4.md) 的服务级实验中给出定量例子。

**量化元数据必须与分页访问保持一致。** [QServe](../methods/qserve.md) v3 §5.1 将逐 head 动态尺度与零点随页保存；这是一种布局选择，不构成“只能随页保存或放弃动态估计”的二选一。其他设计也需定义元数据索引和更新规则，并计入访问成本。对于按固定块回收的池，token 淘汰若没有腾空块或执行压缩搬移，就不会释放整个物理块；算法压缩与实际内存回收需分别测量。

### 分阶段权重也有生命周期

[Disaggregated Quantization](../methods/disaggregated-quantization.md) 为prefill和decode训练独立权重，prompt状态由前者产生、生成由后者消费。ODP借用prefill暂不用的decode权重空间，用双缓冲从SSD逐块装入prefill权重，结束前恢复被覆盖内容。其零额外设备**权重**驻留不包含新增SSD副本、host缓冲、KV和工作区；短提示可能无法隐藏装入时间。

与同权重模型的缓存重算还存在语义区别：已经缓存的assistant状态由decode权重产生，若把相同token历史重跑prefill，会换成另一套权重生成的状态。形状兼容不足以证明数值/任务等价，原论文尚未验证多轮cache策略。不能直接继承分页系统的重算假设。（DQ §2.5、§5；其训练、原型实现与计时限制见方法页。）

## 7. 证据与口径

论文的端到端结果：在相同延迟水平下，vLLM 相对 FasterTransformer 与 Orca 提升吞吐 **2–4×**（摘要与结论口径）；在 ShareGPT 合成负载上可支撑的请求速率相对 Orca（Oracle，假设已知输出长度）为 1.7–2.7×、相对 Orca（Max，按最大长度预留）为 2.7–8×、相对 FasterTransformer 最高 22×；OPT-13B 同时处理的请求数比 Orca(Oracle) 多 2.2×、比 Orca(Max) 多 4.3×；聊天场景相对三个 Orca 基线约 2× 请求速率。

微基准方面，论文报告 PagedAttention 的注意力内核相对优化过的 FasterTransformer 实现有 **20–26%** 的额外延迟（来自块表访问、额外分支与变长序列处理），但只影响注意力算子，端到端仍有大幅收益——这是一处必须同时报告的代价，不能只引用端到端倍数。

口径条件：硬件为 GCP A100 实例，模型 OPT-13B/66B/175B 与 LLaMA-13B，负载基于 ShareGPT 与 Alpaca 的输入输出长度合成、请求到达按泊松过程生成，主要指标是「端到端延迟 ÷ 输出长度」的归一化延迟对请求速率的曲线，多数实验用 1 小时轨迹、OPT-175B 用 15 分钟。Orca 是作者自行实现的基线，分别使用 Oracle／Pow2／Max 预留假设；Oracle 预先知道输出长度，在实践中不可达。引用倍数需带上这些条件。

## 8. 局限与未验证

SGLang 的服务论文尚未接续；DistServe与Sarathi-Serve v3的选定论文范围已补，作者实现与运行仍未核。本页数值均为论文报告，未运行服务或基准；现代实现的行为以各执行页的固定源码为准。

## 来源身份

| 来源 | 版本或快照 | 说明 |
| --- | --- | --- |
| Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve（`https://arxiv.org/abs/2403.02310v3`） | `arXiv:2403.02310v3` | §2–4、算法1–3与必要实验，token预算/混合轮次/历史读取/PP气泡，实验条件由评测页承接；未核作者实现或运行 |
| DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving（`https://arxiv.org/abs/2401.09670v3`） | `arXiv:2401.09670v3` | §2–5与附录A，两阶段/队列/并行/placement及KV pull；实验和版本差异由评测页承接，未核实现或运行 |
| Efficient Memory Management for Large Language Model Serving with PagedAttention（`https://arxiv.org/abs/2309.06180v1`） | `arXiv:2309.06180v1` | — |
| QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving（`https://arxiv.org/abs/2405.04532v3`） | `arXiv:2405.04532v3` | 分页量化元数据布局 |
| SAW-INT4: System-Aware 4-Bit KV-Cache Quantization for Real-World LLM Serving（`https://arxiv.org/abs/2604.19157v1`） | `arXiv:2604.19157v1` | — |
| Disaggregated Quantization（`https://arxiv.org/abs/2609.26333v1`） | `arXiv:2609.26333v1` | §2.5与§5，阶段权重驻留与多轮cache边界 |
