---
title: QServe：W4A8KV4 量化与系统协同
slug: qserve
sources:
  - raw/papers/quantization/ptq/2024-05-qserve-w4a8kv4-quantization-system-codesign/paper.pdf
  - raw/repositories/quantization/2026-07-mit-han-lab-omniserve-02b2925a/source/omniserve/modeling/layers/quantized_linear/w4a8_linear.py
updated: 2026-09-16
---

# QServe：W4A8KV4 量化与系统协同

权重量化在单用户场景常能带来接近理论上限的加速，但云侧服务需要同时处理多请求：计算强度上升，瓶颈从权重读取转向计算与反量化。QServe 的出发点是，4 bit 激活方案的理论峰值在这种场景下无法兑现，因为按组量化的反量化落到了吞吐低得多的 CUDA 核心上。它给出一个精度组合与内核实现都围绕这条判断设计的方案。

本页依据 QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving（arXiv:2405.04532，下称论文，全文含 artifact 附录研读），并定向核对官方实现固定快照 02b2925a 的 W4A8 线性层与内核目录。论文中的所有速度与精度数字均为作者在特定硬件与版本下的测量，本轮没有运行任何实验。

## 1. 研究对象与精度选择问题

论文 §1 把既有整数量化分为三类：W8A8、W4A16、W4A4。前两类被视为几乎无损，W4A4 精度损失明显，但预期能通过 4 bit 张量核心获得更高吞吐。作者指出这一预期在当时的 GPU 上没有兑现：当时最好的 W4A4 服务系统 Atom 在 A100 上跑 Llama-2-7B 时，性能反而比 TensorRT-LLM 的 W4A16／W8A8 低 20%–25%。

作者给出的原因不是算法，而是运行时的反量化开销（§1）：现有 4 bit 方法在反量化权重或部分和时的开销达 20%–90%。W4A16 在 FP16 张量核心上计算，权重是 INT4，因此需要内核内反量化；W4A4 为保证精度必须对权重与激活都按组量化，而 INT4 张量核心产生 INT32 部分和，按组缩放要求在主循环内把部分和转成浮点，这些运算落在 CUDA 核心上。在 A100 上，一次 CUDA 核心运算的代价约等于 50 次 INT4 张量核心运算，因此降低位宽不一定加速。

这就是本方法要解决的问题：在批量服务场景下，同时保住 4 bit 权重的存储收益与 8 bit 张量核心的计算吞吐，并把反量化开销压到主循环之外或压到可接受的量级。

## 2. Roofline：为什么是 W4A8KV4

论文 §3.1 给出判断依据。对 $m\times n\times k$ 的 GEMM，当 $n,k$ 远大于 $m$ 时计算强度约为 $m$（$m$ 为序列数），这正是解码阶段的形态。A100 的峰值算力为 FP16/INT8/INT4 分别 312/624/1248 TOPS，显存带宽 2 TB/s，于是：

- W4A16 在 $m<78$ 时理论吞吐更高（内存受限，权重流量主导）；
- W8A8 在 $m>78$ 时更好（计算受限，INT8 张量核心更快）；
- W4A8 若能全部在 INT8 张量核心上计算，就在两端都接近最优。

注意力侧的计算强度约为 1 MAC/element，与批量无关，访存由 KV cache 主导，因此 KV4 相对 KV8 提供约 2 倍峰值。论文指出 batch=64 时注意力已占总运行时间的 50% 以上，KV 位宽因此直接影响端到端收益。

**为什么不选 W4A4。** 4 bit 张量核心的峰值是 8 bit 的两倍，理论交叉点在 $m>78$，但论文 §3.2 说明该收益无法在 Ampere 与 Hopper 上实现。按组量化的 W4A4 必须在主循环内做 INT32 到 FP32 的部分和转换：一方面 CUDA 核心峰值仅为 INT4 张量核心的 2%，使主循环被慢速运算支配；另一方面同时保存 FP32 与 INT32 两组部分和寄存器，而输出驻留式数据流下大 GEMM 本就受寄存器限制，寄存器压力压低同时驻留的 warp 数，进一步削弱隐藏延迟的能力。

**与后续同类工作的对照。** 上述判断针对的是 Ampere 与 Hopper 上按组量化的 W4A4 主循环开销，它在 QuaRot 的在线 Hadamard、以及 [[flatquant|FlatQuant]] 一类融合实现下会被部分抵消或转移；论文没有覆盖后者的实现方式，因此不宜把「W4A4 不划算」当作与架构无关的结论。

## 3. 渐进式分组量化

为了让 W4A8 的全部计算落在 INT8 张量核心上，需要让 4 bit 权重的反量化结果恰好落在 INT8 范围内，而不是先反量化成浮点。论文 §4.1 用两级量化实现这一点。

设权重 $W\in\mathbb R^{k\times n}$。第一级为逐输出通道对称 INT8：

$$ \widehat{\mathbf W}={\mathbf Q_{\mathbf W}}^{(0)}_{\mathrm{s8}}\cdot \mathbf s^{(0)}_{\mathrm{fp16}},$$

其中 $ {\mathbf Q_{\mathbf W}}^{(0)}_{\mathrm{s8}}$ 是中间 8 bit 张量，$\mathbf s^{(0)}_{\mathrm{fp16}}$ 是逐通道 fp16 尺度。第二级对这个中间张量做按组非对称 INT4：

$$ {\mathbf Q_{\mathbf W}}^{(0)}_{\mathrm{s8}}=\left({\mathbf Q_{\mathbf W}}_{\mathrm{u4}}-\mathbf z_{\mathrm{u4}}\right)\cdot \mathbf s^{(1)}_{\mathrm{u8}},$$

$\mathbf z_{\mathrm{u4}}$ 与 $\mathbf s^{(1)}_{\mathrm{u8}}$ 分别是按组的无符号 4 bit 零点与无符号 8 bit 尺度。计算时先把 $ {\mathbf Q_{\mathbf W}}_{\mathrm{u4}}$ 反量化回中间 8 bit 张量，再按 W8A8 的方式做 INT8 矩阵乘。

**保护范围 [-119, 119] 的由来。** 朴素地做这两级量化不保证中间值仍落在 $[-128,127]$。论文给了一个反例：某组 8 bit 权重位于 $[-113,120]$，那么 4 bit 非对称量化的尺度为 $(120-(-113))/(15-0)=16$、零点为 7，值 120 编码为 15，反量化得 $(15-7)\times16=128$，越界。作者指出打开算术指令的饱和选项会严重损害吞吐（最多降 67%），于是改为从数学上留出余量。由

$$ \widehat q_{\mathrm{s8}}=\left\lfloor \frac{q_{\mathrm{s8}}}{s_{\mathrm{u8}}}\right\rceil\cdot s_{\mathrm{u8}}\le q_{\mathrm{s8}}+\frac{1}{2}s_{\mathrm{u8}}$$

且 $s_{\mathrm{u8}}$ 最大为 17（由 $[0,15]$ 与 $[-128,127]$ 的端点组合决定），要求 $\widehat q_{\mathrm{s8}}\le127$ 就得到 $q_{\mathrm{s8}}\le119.5$。因此把第一级对称范围从 $[-127,127]$ 收窄到 $[-119,119]$，用可控的精度余量换取无反量化溢出。

固定代码与该设计一致：`w4a8_linear.py` 中保存着对第一阶段权重范围的断言，注释直接把 119 称作「那个魔法数字」，说明这个常数来自上述推导而非经验搜索。

**与既有两级量化的区别。** QLoRA 的 Double Quantization 与 VSQuant 也引入两级尺度，但它们的第二级是对分组浮点尺度再量化，目的是减小元数据体积：先按目标位宽分组量化，再压缩尺度。QServe 的两级顺序相反，第二级量化的是中间位宽张量，目的是让反量化输出落在 INT8 可计算范围。DGQ 也限制尺度以满足 INT8 计算，但它把反量化内核与 GEMM 内核分开，导致端到端比 cuBLAS 的 W8A8 还慢；QServe 靠保护范围把反量化融合进 GEMM 内核并做寄存器级并行，作者报告其按组 W4A8 GEMM 相对 cuBLAS W8A8 有 1.5 倍加速。

## 4. SmoothAttention：缓解 KV4 的精度损失

论文 §4.2 的观察是：Value 矩阵没有显著的离群模式，而 Key 矩阵的每个 head 中存在固定的离群通道，幅值约为其余值的 10 倍。KV8 尚能容忍，KV4 的量化级数不足。做法沿用 [[smoothquant|SmoothQuant]] 的通道缩放，把难度从 Key 转移出去：

$$ \mathbf Z=(\mathbf Q\mathbf\Lambda)\cdot(\mathbf K\mathbf\Lambda^{-1})^{\mathsf T},\qquad \mathbf\Lambda=\mathrm{diag}(\lambda).$$

由于查询不量化，无需像 SmoothQuant 那样在激活与权重之间搜索迁移强度，作者直接取

$$ \lambda_i=\max(|\mathbf K_i|)^{\alpha},$$

实践中 $\alpha=0.5$ 足够。**位置编码带来的额外约束：** 把缩放融合进前置线性层权重（$\mathbf W_Q\lambda$ 与 $\lambda^{-1}\mathbf W_K$）可以省去额外的内核调用，但 RoPE 在同一 head 内把通道 $i$ 与 $i+D/2$ 配对旋转，因此必须加上 $\lambda_i=\lambda_{i+D/2}$ 的硬约束，取两者幅值的较大者作为公共值。这个条件与 [[diagonal-scaling-equivalent-transform#7. 平移与注意力位置编码的融合边界|平移与位置编码的融合边界]] 中推导的「缩放需与相对旋转交换」一致，只是这里直接按配对结构构造满足条件的缩放。

这一观察与 [[kivi|KIVI]] 的独立结论一致（键有固定通道离群值、值没有），但两者在粒度与布局上给出不同方案：KIVI 对键用逐通道量化并保留全精度残差窗口，QServe 用逐 head 动态量化并把尺度与零点存进分页；[[saw-int4|SAW-INT4]] 则指出混合精度残差与分页布局冲突。共同结构见 [[kv-cache-quantization-objects-and-granularity|KV cache 量化的对象与粒度]]。

## 5. 逐层的表示与范围调整

论文 §4.3 针对不同线性层使用不同处理，四类手段都保持浮点计算等价：

**块输入模块旋转。** 对 QKV 投影、FFN 第一层这类消费块输入的模块，用缩放后的 Hadamard 矩阵旋转激活，权重侧反向旋转。旋转阵是酉矩阵，可吸收进上一块的输出权重，不增加运行时算子。这一思路与 [[quarot|QuaRot]] 相同。

**块输出模块平滑。** 对输出投影与 FFN 第二层，用逐通道因子除激活、乘权重。作者报告了一处与 SmoothQuant 不同的经验：如果对这些模块直接套用与输入模块相同的迁移强度，Llama-2-7B 的 WikiText-2 困惑度会恶化 0.05；实践中迁移强度应接近 0，即平滑因子主要由权重而非激活决定。同一套缩放公式在不同计算图位置给出不同的最优参数，这一点不能从公式本身推出。

**激活感知的通道重排。** 用 $\mathrm{mean}(|X|)$ 衡量通道重要性，把显著程度相近的通道排进同一个量化组，避免离群通道把自己的范围强加给同组其他通道。作者强调这不同于 Atom 保留部分权重为 FP16 的做法：这里不引入混合精度，只改变分组构成。

**权重裁剪。** 对绝大多数线性层最小化层输出误差，对 `q_proj` 与 `k_proj` 改用块输出均方误差：

$$ \arg\min_{\alpha}\left\|\mathrm{Block}(\mathbf X;\mathbf W)-\mathrm{Block}\left(\mathbf X;Q(\mathbf W;\alpha)\right)\right\| .$$

目标函数按层选择，而不是全局统一，这一点与其后 [[flatquant|FlatQuant]] 用统一的块级目标形成对照。

## 6. 系统侧：W4A8 GEMM 与 KV4 注意力

论文 §5 的执行映射是：所有 GEMM 取 W4A8 输入、在 INT8 张量核心上计算、输出 FP16；注意力在 CUDA 核心上以 FP16 计算；整个块输入输出均为 FP16。激活量化被融合进前置 LayerNorm 或激活内核，输出投影前另插入一个量化节点。

**KV cache 管理。** QServe 沿用 vLLM 与 TensorRT-LLM 的分页 KV 布局，但量化粒度不同：那两者对 KV 使用逐张量静态量化，QServe 因位宽更低而要求逐 head 动态量化，于是把每个 head 的 fp16 尺度与零点紧跟在分页中量化后的 KV 特征之后存放，支持运行时更新，并支持 in-flight batching。这解释了 KV4 意味着多少额外元数据这一实现问题：元数据随页存放，而不是集中在一处。

**主循环里的指针算术。** 张量核心的 GEMM 内建函数要求每个线程按跨步方式取数，朴素实现在每 4 个通道就要做一次地址计算，而地址计算在 CUDA 核心上执行，A100 上其吞吐比 INT8 张量核心低 32 倍；跨步访问也无法用 128 bit 打包加载打满带宽。数据与计算类型相同时，`ldmatrix` 可以自动完成分发；但 W4A8 下存储是 4 bit、计算是 8 bit，`ldmatrix` 保证的是每个线程拿到相同的字节数而不是相同的元素数，造成错配，因此无法使用。

作者的解法是**计算感知的权重重排**：把整个 GEMM 划成若干 $32\times32$ tile，按计算时实际使用顺序存储权重，即某线程需要的 32 个通道被拼成一个 128 bit 字，下一个线程的 32 个通道紧随其后。权重静态，重排无运行时开销，效果是地址计算降到与 `ldmatrix` 相当，同时保证每线程 128 bit 的高带宽事务。零点与尺度采用同样重排。

**快速反量化。** UINT4 到 UINT8 的拆包通过把每 32 个权重按 $w_0,w_{16},w_1,w_{17},\dots$ 重排，利用寄存器级并行用三条逻辑操作完成。UINT8 到 SINT8 的零点减法则被移出主循环：对逐通道量化，

$$ \mathbf O=(\mathbf Q_{\mathbf X}\mathbf Q_{\mathbf W})\odot(\mathbf s_{\mathbf W}\times\mathbf s_{\mathbf X})-(\mathbf Q_{\mathbf X}\odot\mathbf S_{\mathbf X})\mathbf{ZS}_{\mathbf W},\qquad \mathbf X(\mathbf{ZS}_{\mathbf W})=\mathbf t_{\mathbf X}\times(\mathbf z_{\mathbf W}\odot\mathbf s_{\mathbf W}),$$

其中 $\mathbf t_{\mathbf X}=\mathbf X\mathbf 1_k$，即每个 token 的输入通道求和。两项都是外维缩放形式，可以放进 GEMM 的 epilogue，而 $\mathbf t_{\mathbf X}$ 能在前一个访存受限内核里顺带算出（每个 W4A8 内核之前总是有一个访存受限内核），附加延迟可忽略。这就是先乘后减的次序。

按组量化时零点也是按组的，无法合并进 epilogue，且每个权重多一次 INT8 乘法。作者仍选择先乘后减，原因是它允许寄存器级并行：GPU 有 `vadd4`，一条 INT32 ALU 指令完成四次 INT8 加法，但没有对应的四次 INT8 乘法指令，只能用在高位补 24 个零来模拟。这种模拟要求每次 INT8 乘法的结果不超出 INT8 范围，而这正是渐进式分组量化的保护范围所保证的。先减后乘的次序不满足该条件，只能逐个相乘，效率极低。两级设计在这里同时服务于数值正确性与内核调度，这是本方法算法与系统协同的核心。

其余为常规优化：多级软件流水与异步拷贝、共享内存 swizzle 消除 bank conflict、重排线程块划分以复用权重、在输入 token 数较少时沿 $K$ 维切分并用共享内存做 warp 间归约。

**KV4 注意力。** 用 TensorRT-LLM 的 KV8 内核作基线，把静态逐张量访问替换为动态逐 head 的 4 bit 访问后，L40S 上快 1.7 倍，但 A100 上反而慢 1.1 到 1.2 倍。原因是 A100 FP32 CUDA 核心的 roofline 拐点只有 9.8 Ops/Byte，而从缓存中反量化一个 INT4 需要 5 次 ALU 操作（掩码、移位、整型转浮点、浮点乘、浮点减），反量化本身就打满了这个上限，使融合内核变成计算受限。作者的应对是双向的：把内核中的 FP32 运算换成 FP16 以抬高计算上限；用位技巧把每个元素的反量化降到 2 次操作；简化控制流、预取尺度与零点、简化地址计算。最终在 A100 上相对 KV8 基线快 1.5 倍。分项贡献为 0.48 ms 到 0.44（位技巧）、0.39（控制流）、0.36（QK 与 SV 转 FP16，各 0.03），再到 0.28 ms（异步预取），端到端约 1.7 倍。

这段分析回答了 [[quantized-matmul-scaling-execution|执行路径页]] 提出的问题：KV 位宽减半为什么不等于延迟减半。答案在反量化算术强度与 CUDA 核心拐点上，而不在带宽。

## 7. 精度与性能证据

**设置（论文 §6.1）。** 精度评估覆盖 Llama-1／2／3、Mistral-7B、Mixtral-8x7B、Yi-34B；激活为逐 token 对称 INT8，KV cache 为逐 head 非对称 INT4；W4A8KV4 g128 指权重使用渐进式分组量化、组大小 128，不带 g128 则表示逐通道。基线为 SmoothQuant、GPTQ-R、AWQ，以及 Atom（W4A4）与 QuaRot（W4A4）。两篇 4 bit 基线因不支持渐进式分组量化，用的是普通按组量化。

**精度。** WikiText-2 困惑度相对 W8A8 的 SmoothQuant 与 W4A16 的 AWQ 最多升高 0.16；一致优于 Atom；相对 QuaRot 的 W4A4 最多低 0.49。零样本五任务中相对 FP16 的损失为 7B／13B／70B 分别为 1.03%、0.89%、0.40%；WinoGrande 上相对 QuaRot 高 4.82 个百分点。LongBench 长上下文结果相对 BF16 基线退化很小。

**消融（L40S，64 请求，1024 输入与 512 输出；论文在效率评估一节的算法消融图）。** 从 RTN W8A8 出发：权重降到 4 bit 使困惑度明显恶化，但速度提高到 1.12 倍、省 3.5 GB；块输入旋转改善 0.18；用块输出 MSE 做裁剪再改善 0.16，此时 W4A8 的困惑度已与 W4A16 相当；KV 量化到 4 bit 又恶化 0.14，但同时带来 1.47 倍加速并把显存减半；SmoothAttention 改善 0.05 且无系统开销；渐进式分组量化再改善 0.04，反量化开销增加可忽略；通道重排改善 0.03。这组数字说明精度与加速的每一步都可单独归因，且 KV4 是换来速度的精度支出。

**吞吐。** 论文在 A100-80G 与 L40S-48G 上以相同显存预算测最大可达吞吐，输入 1024、输出 512。相对 TensorRT-LLM 的最佳精度配置：A100 上 Llama-1-30B 约 2 倍、Llama-2 系列 1.2 到 1.4 倍、Mistral 与 Yi 1.2 倍、Qwen1.5 2.4 倍；L40S 上为 1.47 到 3.47 倍。同批次拆分显示 Llama-2-7B 的 1.88 倍等于 1.45 倍（同批加速）乘 1.3 倍（批量增大）。作者还指出，L40S 上运行七个模型中的五个可达到高于 A100 上 TensorRT-LLM 的吞吐。

**必须保留的条件：** A100 实验使用逐通道权重、L40S 使用按组权重，作者明确原因是 L40S 的 CUDA 核心更强、能承担反量化开销。也就是说同一论文在两块 GPU 上用的是不同量化配置，跨 GPU 直接比较吞吐会把配置差异混入。此外 Atom 只支持 Llama-2-7B、QuaRot 不支持 GQA，对应模型被跳过；QuaRot 没有分页注意力支持。

## 8. Strong / Weak

**Strong：**把精度组合的选择建立在 roofline 与 CUDA 核心代价上，而不是只比较平均分；给出让 W4A8 全部落在 INT8 张量核心的机制（渐进式分组量化与保护范围），并说明该机制同时解决数值溢出与寄存器级并行两个不同问题；KV4 的收益与代价用同一套 roofline 解释；消融把每个组件的精度贡献与系统开销分开报告。

**Weak：**结论依赖 A100／L40S 与 TensorRT-LLM v0.9.0 等当时版本，硬件换代后拐点与配置取舍会变，论文自身就因 CUDA 核心强弱在 A100 与 L40S 上用不同权重粒度；对 W4A4 的否定针对当时的按组实现，未覆盖后来融合在线变换与量化的实现；KV cache 为逐 head 动态量化，意味着运行时仍需估计范围，静态方案的开销优势不适用；未提供与其他 4 bit 格式在同一导出格式下的端到端对照。

## 9. 与其他页面的关系

与 [[smoothquant|SmoothQuant]] 共享通道缩放思想：SmoothAttention 是它的注意力变体，块输出平滑则给出「迁移强度应接近 0」的反例，说明同一公式在不同图位置的最优参数不同。与 [[quarot|QuaRot]] 共享旋转抑制离群值的做法，但 QServe 把旋转与平滑作为精度组合的一部分、并把讨论重点放在主循环开销上。与 [[flatquant|FlatQuant]] 的关系最需要区分：两者都做 W4A4KV4 级别的低比特推理，FlatQuant 主张用结构化在线变换加融合内核控制 W4A4 的开销，QServe 则论证当时的 W4A4 主循环反量化不可行、转向 W4A8。这类分歧应通过同一硬件、同一批次与同一测量口径的对照实验判断，而不是引用各自论文的排名。

部署侧的格式与后端映射见 [[quantized-llm-deployment-backends|部署框架与后端支持]]，W4A16 批量内核的对照见 [[marlin-batched-w4a16-gemm|Marlin]]，存储格式与位宽口径见 [[gguf-block-quantization-formats|GGUF 块量化存储格式]]。

## 10. 局限与未验证

- 未运行 QServe 或任何基线系统；本页速度、精度与显存数字全部来自论文报告。
- 代码核对限于 W4A8 线性层的模块结构与内核目录划分，`kernels/csrc/qgemm` 下的 CUDA 与 PTX 实现未逐行审查。
- 论文的 4 bit 基线对照使用普通按组量化，且部分模型因不支持而被跳过，因此优于 Atom 或 QuaRot 的范围受限。
- artifact 附录给出的复现要求（A100 或 L40S、Docker、约 512 GB 磁盘）说明该验证门槛较高；本项目没有满足该条件的实验环境。
