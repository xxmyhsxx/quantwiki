---
title: FlashAttention-2 CUDA：普通前向、精度与反向归约
type: implementation
tags:
  - attention
  - kernels
  - memory
  - gpu
sources:
  - raw/repositories/2026-09-21/flash-attention
  - raw/papers/2026-09-21/flashattention-2
updated: 2026-10-13
---

# FlashAttention-2 CUDA：普通前向、精度与反向归约

这条 CUDA 路径接收 Q、K、V，分块计算 attention 输出 O。前向用每行的归一化量 LSE 代替完整概率矩阵的存储；训练时再用 Q、K 和 LSE 恢复概率块，计算 dQ、dK、dV。执行过程为：**query 行块流式累加输出 → 保存 O 和 LSE → 反向重算概率 → 合并各 KV 列块的梯度贡献**。

本页核对包声明为 2.8.4 的固定源码，主例采用 SM80、FP16、head 宽度 64、等长 causal attention、dropout=0，以及标准前向和默认反向。它有构建与分派前提，见第 5 节。分块数学与工作分配的来由可接着读 [FA1](../theory/flash-attention-tiling-and-recomputation.md) 和 [FA2](../theory/flash-attention-2-work-partitioning.md)；本页重点解释实际状态、精度和梯度归约。核对来自静态源码，没有编译或 GPU 运行。

## 1. 输入、输出与训练时保存什么

令 B 为 batch 数，Nq、Nk 为 query/key 长度，Hq、Hkv 为 query/KV head 数，d 为 head 宽度。主例取 $B=1,N_q=N_k=1024,H_q=H_{kv}=32,d=64$，暂不涉及跨 head 共享 KV。

| 对象 | 主例形状与 dtype | 在计算中的作用 |
| --- | --- | --- |
| Q | $[B,N_q,H_q,d]$，FP16 | 每个 query 产生一行 attention 输出 |
| K、V | $[B,N_k,H_{kv},d]$，FP16 | K 决定匹配分数，V 提供加权求和的内容；本入口两者宽度相同 |
| O | $[B,N_q,H_q,d]$，FP16 | 前向结果，交给后续网络层；训练反向也会使用 |
| LSE | $[B,H_q,N_q]$，FP32 | 每行所有可见 key 的 log-sum-exp；有效行用自然对数，轴顺序与 O 不同 |

对一个 batch/head，记缩放后的分数为 $S_{ij}=\tau q_i^\top k_j$，默认 $\tau=d^{-1/2}$。有效行的 LSE 与可见项概率满足

$$
LSE_i=\log\sum_{j\ {\rm visible}}\exp S_{ij},
\qquad P_{ij}=\exp(S_{ij}-LSE_i).
$$

不可见项的 P 为 0。因此保存 LSE 就能在反向重算任意概率块，不必保存完整 $N_q\times N_k$ 的 P。这里的关系先说明算法；后文会说明实际的低精度舍入和无有效 key 时的特殊值。

Python 的 autograd 在启用梯度且 Q/K/V 至少一个 requires_grad 时保存 Q、K、V、padded O、LSE 和两元素 int64 的 rng_state；反向接收后续网络传来的 dO，并分配与输入同 dtype 的 dQ/dK/dV。本例 d64 无需 padding；其他宽度可能由 Python 补到 8 对齐，反向再裁回原宽度。（`flash_attn_interface.py::FlashAttnFunc.forward/backward:845–911`；输出与 LSE 分配见 `flash_api.cpp:430–449`。）

## 2. 前向：一个 query 行块怎样得到 O 和 LSE

标准 d64 特化让一个 CTA（协作线程块）负责最多 128 个 query 行和一个 head。它保留自己的 query 块，扫描对应的可见 K/V 块，每次处理最多 128 个 key；这些 key 块按逆序扫描。causal mask 只让 query 使用允许的 key：一般边界为 $j\le i+N_k-N_q$，主例等长时就是 $j\le i$。因果上界还用于跳过整块不可见的 K/V。（`flash_fwd_kernel.h::compute_attn_1rowblock:83–155`；`mask.h::Mask::apply_mask:171–190`。）

### 每读一块，更新三种行状态

每行维护原始 QK 分数的最大值 m、指数和 $\ell$、未归一化输出向量 A。这里的 **m 尚未乘 $\tau$**，与论文中直接在 S 坐标下定义的最大值不同。

设新读入块的原始分数为 $r_{ij}=q_i^\top k_j$，$J_i$ 是当前块中对 query i 可见的 key 集合。在线累计的数学初态为 $m=-\infty,\ell=0,A=0$；对有可见项的块，更新可解释为：

$$
\begin{aligned}
m'&=\max\left(m,\max_{j\in J_i} r_{ij}\right),\\
\alpha&=\exp\bigl(\tau(m-m')\bigr),\\
E_{ij}&=\exp\bigl(\tau(r_{ij}-m')\bigr),\quad j\in J_i,\\
\ell'&=\alpha\ell+\sum_{j\in J_i} E_{ij},\\
A'&=\alpha A+\sum_{j\in J_i} E_{ij}v_j.
\end{aligned}
$$

不可见项的 E 为 0，无可见项的块不加入新贡献。实现的首个有效块直接建立 m、$\ell$、A，后续块执行重缩放更新。当新块抬高最大值时，旧的 $\ell$ 和 A 必须一起乘 $\alpha$，才能与新块使用同一归一化基准。所有可见块处理完后，才得到 $O_i=A_i/\ell_i$，并写出 $LSE_i=\tau m_i+\log\ell_i$。这些式子是对所选源码状态的数学解释；实际执行有以下精度转换。

| 阶段 | 实际状态与精度 | 为什么要分清 |
| --- | --- | --- |
| QK | FP16 Q/K 做 MMA（矩阵乘累加），分数在 FP32 累加 | FP32 分数来自低精度输入 |
| 在线 softmax | m、指数和、旧 A 的重缩放用 FP32 | 保持各块共同的数值基准 |
| 加入当前块的 V | FP32 指数块 E 先转 FP16，再与 FP16 V 做 MMA；A 在 FP32 累加 | PV 的输入并非全程 FP32 |
| 完成一行 | FP32 A 除以 $\ell$，O 转 FP16；LSE 写 FP32 | 保存 O 已经过舍入，影响后续 D 的计算 |

该路径不保证与物化 PyTorch 基线逐位相同。

代码用 exp2 执行指数，`scale_softmax_log2=τ log2(e)`，所以重缩放实际写成 $2^{(m-m')\tau\log_2e}$；返回 LSE 仍是自然对数单位。正确的 $\alpha$、指数块转 FP16、最终归一化分别可见 `softmax.h:128–186`、`flash_fwd_kernel.h:342–367,407–477`，MMA 精度由 `kernel_traits.h:18–34`确定。此固定实现与 FA2 论文 v1 的伪码分别核查，不能由这里追认论文时期全部代码。

### 没有有效归一化量的行要单独处理

标准路径在 row_sum 为 0 或 NaN 的分支写 LSE=**+inf sentinel**；无可见 key 的早退写 O=0、LSE=+inf。这个特殊值用于实现边界，不是数学上的 $\log0=-\infty$，也不能作为有效行的归一化量直接交给别的库合并 attention 状态。本次未验证任意 NaN/Inf 输入或所有 mask 组合。（`softmax.h:169–185`；`flash_fwd_kernel.h:92–127`；`flash_api.cpp:498–504`。）

## 3. 反向梯度与三阶段计算

反向已知 dO，要把损失对 O 的变化传回 Q、K、V。入口要求 dO/O 与 Q/K/V 同 dtype，本例均为 FP16。对一个 batch/head，下面是 dropout=0 时的数学关系；dS 指**缩放后分数 S 的梯度**：

$$
\begin{aligned}
dV&=P^\top dO, & dP&=dOV^\top,\\
D_i&=\sum_j P_{ij}\,dP_{ij}=dO_i\cdot O_i,\\
dS_{ij}&=P_{ij}(dP_{ij}-D_i),\\
dQ&=\tau\,dS K, & dK&=\tau\,dS^\top Q.
\end{aligned}
$$

dP 表示损失对每个概率的梯度；D 是一行内按概率加权的 dP，用于 softmax 反向的行修正。因为 $O_i=\sum_jP_{ij}v_j$，这个修正可以从 dO 与 O 的点积得到。最后的 $\tau$ 来自 $S=\tau QK^\top$ 的链式法则。上述 D 的等式解释了预处理的设计；代码使用的是已经转为 FP16 的保存 O，不能把实际 D 称为未舍入前向状态的精确结果。

所选反向把工作分成三次 launch：

| 阶段 | 读取什么 | 产生什么，以及下一阶段如何使用 |
| --- | --- | --- |
| `dot_do_o` | 保存的 FP16 O、输入 FP16 dO | 转 FP32 后计算每行 D；默认同时清零 dQ 工作区 |
| `seqK-parallel` | Q/K/V、dO、LSE、D | 每 CTA 负责一个 KV 列块，恢复 P、沿可见 query 块累计该块的 dK/dV；把对 dQ 的贡献加到工作区 |
| `convert_dQ` | FP32 dQ 工作区 | 默认读取已累计的 dQ，乘 $\tau$并转 FP16；deterministic 分支先合并 split，再缩放与转换 |

KV 列块各自可以完成自己的 dK/dV，但多个列块会影响同一个 query 的 dQ；这就是 dQ 需要额外工作区和最后一次转换的原因。（`flash_bwd_launch_template.h::run_flash_bwd_seqk_parallel:73–125`；D 的计算见 `flash_bwd_preprocess_kernel.h::dot_do_o:24–49`。）

### 重算概率并不消除低精度转换

`seqK-parallel`重算 FP32 累加的 QK，并应用同一 causal mask；可见项用保存的整行自然 LSE 恢复 $P=\exp(\tau QK-LSE)$，不可见项置 0，内部同样用 exp2 换底。恢复后的 P 有两种用途：

- P 先转 FP16，参与 dV 的 MMA。
- dP 的 MMA 用 FP32 累加，dS 用**尚未转 FP16 的重算 P**计算 $P(dP-D)$；随后 dS 转 FP16，参与 dQ/dK 的 MMA。

dQ/dK 的 MMA 在 FP32 累加。dK 在 epilogue 乘 $\tau$，dQ 在最后的 `convert_dQ`乘 $\tau$，最终 dQ/dK/dV 都回到 FP16。dropout=0 时 keep probability 为 1，无需额外 dropout 因子。（`flash_bwd_kernel.h:474–550,577–594,622–690,728–792`；`flash_bwd_preprocess_kernel.h:243–252`；`flash_api.cpp:123–131,803–935`。）

本分支的 dK/dV 使用 CTA 内部局部累计后写出，没有使用全局 FP32 dKaccum/dVaccum；C++ 的对应分配被注释、指针设空。源码中另有 `convert_dKV`，不能仅凭函数存在断定这次调用执行了它。（`flash_api.cpp:896–897,922–925`。）

## 4. 两个影响反向归约与内存的分支

### deterministic 改变 dQ 的跨 CTA 合并方式

默认的 FP32 dQ 工作区形状为

$$
[B,\operatorname{round128}(N_q),H_q,\operatorname{round32}(d)],
$$

其中 round128/round32 表示向上补到相应倍数。不同 KV 列块的 CTA 向同一位置 atomicAdd，提交次序可以改变浮点舍入。

deterministic=True 时，多分配一个 split 轴：

$$
[n_{\rm split},B,\operatorname{round128}(N_q),H_q,\operatorname{round32}(d)],
\qquad n_{\rm split}=\left\lceil\frac{{\rm numSM}}{BH_q}\right\rceil.
$$

每 CTA 写自己的 split，按列块编号 $x,x+{\rm gridDim.x},\ldots$处理工作；最后 `convert_dQ`按 split 递增顺序合并。kernel 内仍使用 atomicAdd，但跨 CTA 的写入目标已经分离。因此这里的 deterministic 来自合并组织，不能概括为完全没有原子操作，也不承诺跨设备或版本逐位相同。（`flash_api.cpp:885–897,935`；`flash_bwd_launch_template.h:76–82`；`flash_bwd_kernel.h:120–125,666–678,827–837`；`flash_bwd_preprocess_kernel.h:243–248`。）

假定设备有 108 个 SM，主例得到 4 个 split；8 个 KV 块分别由 split 处理为 [0,4]、[1,5]、[2,6]、[3,7]。默认 dQ 工作区为 8 MiB，deterministic 版本为 32 MiB。存 D 的 `softmax_d`另为 FP32 $[B,H_q,\operatorname{round128}(N_q)]$。这些数字是形状乘 dtype 的教学推算，排除了保存输入、输出、其他工作区和 allocator，不是训练峰值测量。

### GQA 前向共享 KV，反向仍需合并 query head 的贡献

当 $H_q/H_{kv}=r$ 为整数时，query head h 使用 KV head $\lfloor h/r\rfloor$，前向不先复制完整 K/V。同一个 KV head 的梯度则包含 r 个 query head 的贡献。

本入口反向先分配 FP16 的 `dk_expanded,dv_expanded`，形状 $[B,N_k,H_q,d]$。每个 query head 的 FP32 局部累计先转 FP16 写入展开缓冲，再 reshape 为 $[B,N_k,H_{kv},r,d]$，用 `sum_out`沿 r 归约到最终 FP16 dK/dV。因此，前向共享 KV 不代表反向无需额外展开缓冲，也不能称为跨 head 全程 FP32、最后只舍入一次。本次未读 PyTorch `sum_out`的设备内核，其内部 accumulator dtype 保持未核。（`flash_fwd_kernel.h:148–155`；`flash_bwd_kernel.h:110–113,728–792`；`flash_api.cpp:900–907,967–970`。）

## 5. CUDA 入口、标准路径分派与布局约束

前面的状态流成立于所选标准路径。教学配置还取无限左窗口、softcap=0、无 ALiBi、return_attn_probs=False、默认非 deterministic，并假定构建启用所需前后向支持。设备属性以运行时查询为准，本次没有查询 GPU。

入口沿 `flash_attn_func` → `FlashAttnFunc.apply` → `_wrapped_flash_attn_forward` → 扩展 `flash_attn_2_cuda.fwd` → pybind 绑定的 `mha_fwd`。Python 根据 `torch.__version__ >= "2.4.0"`的**字符串比较**选择 custom-op wrapper 或直接调用 wrapper，两者目标仍是同一 CUDA 扩展。它不是语义版本解析，例如 `"2.10.0"`按字串会进入另一分支，目标环境需核实际判断。ROCm/Triton 不在本页路径。（`flash_attn_interface.py:12–28,61–114,147–150,845–878,1156–1230`；`flash_api.cpp:1481–1487`。）

### dropout=0 时的标准前向与 split-KV 分派

`mha_fwd`的 `set_params_splitkv`会运行 split 启发式。它用 $\lceil N_q/64\rceil$估计 query 块数；若

$$
BH_q\lceil N_q/64\rceil\ge0.8(2\,{\rm numSM}),
$$

直接取 1 split。假定 108 SM，主例为 $512\ge172.8$，满足条件。只有 `params.num_splits<=1`且未强制 split 时，`run_mha_fwd`才进标准特化；否则进 split-KV。这个 64 是启发式的估计粒度，标准主例的 query tile 是 128。（`flash_api.cpp::num_splits_heuristic/set_params_splitkv/run_mha_fwd:243–325,474–476`。）

标准 d64 前向经 `flash_fwd_hdim64_fp16_causal_sm80.cu`选择
`Flash_fwd_kernel_traits<64,128,128,4,false,false,half_t>`：query/key tile 各 128，4 warps、128 线程，grid 为 $[\lceil N_q/128\rceil,B,H_q]=[8,1,32]$。traits 的 MMA warp 布局沿 M 轴分工，对应 query 行拥有者。Q/K/V 共享缓冲按形状合计 $3\times128\times64\times2=48$ KiB，launcher 对 ≥48 KiB 请求 dynamic shared-memory opt-in；O epilogue 复用片上空间需遵守同步。这是布局推算，不能给出寄存器占用、驻留或性能结论。（`flash_fwd_launch_template.h:63–91,203–219`；`kernel_traits.h:74–109`。）

反向另按运行时 opt-in 共享内存容量选择 tile：≥144 KiB 用 query/key tile 128/128、8 warps；否则用 query tile64/key tile128 和 V-in-registers 分支。该阈值保留为代码条件，不替代实际设备查询。（`flash_bwd_launch_template.h:160–185`。）

### 接口接受什么布局，测试对象又是什么

C++ 检查 CUDA、Q/K/V 同为 FP16 或 BF16、最后维 stride=1、$d\le256$且 8 整除、Hq 能被 Hkv 整除。`maybe_contiguous`只在最后维不连续时复制，其他 batch/row/head stride 会传给 kernel，不要求四维张量全部连续。主例全连续输入的元素 stride 为 $(NHd,Hd,d,1)$，参数单位是**元素**而非字节。这些入口检查不能证明所有配置都有已构建且可运行的设备支持。（`flash_attn_interface.py:27–28,98`；`flash_api.cpp:60–83,351–449`。）

`return_attn_probs`是测试接口。即使设为 True，dropout=0 时 Python 仍关闭实际的 return_softmax，返回 LSE 和空 S_dmask；非零 dropout 的返回对象也不能无条件当作通常的归一化概率。本例不保存完整 attention 矩阵。（`flash_attn_interface.py:866,1205–1215`；`flash_api.cpp:435–446`。）

dropout=0 关闭 dropout 运算和 generator 推进，但仍分配 rng_state 接口对象，并按梯度条件保存。前后向 dropout 运算有编译期 Is_dropout guard，部分构造和元数据读取在 guard 外；没有检查编译产物，不能再断言不存在任何 RNG 读取。非零 dropout 的完整 Philox 元素映射和 sign 编码仍未核。（`flash_api.cpp:483–492`；`flash_fwd_kernel.h:69–78,359–361`；`flash_bwd_kernel.h:448–449,537–546`。）

## 6. 核对范围与后续使用

本次核 Python/C++ 普通接口、上述 SM80 FP16/d64/causal/dropout0 的标准前向与 seqK 反向，以及必要的 GQA/deterministic 分支。15 个相关源码文件已与固定 commit 的 Git 对象作字节比对，不能据此宣称重验条目所列 25846 个全部来源文件。正文文件路径均相对于该来源源码根目录。

未覆盖 split-KV 设备计算/combine、varlen、分页/append/rotary KV、其他 head 尺寸或架构、非零 dropout、local/ALiBi/softcap、ROCm/Triton、FA3/FA4 或 cute、完整构建与 CUDA Graph。[vLLM Attention](vllm-attention-operator-design.md) 和 [MLA](mla-cache-and-attention-execution.md)各有外部 pin 与调用边界，本页独立 commit 的普通路径不能直接作为它们的库内部执行证据。要判断配置能否运行及实际收益，需按[运行验证](kernel-correctness-and-benchmarking.md)开展正确性与性能测量；本轮没有运行模型或测 GPU 性能。

## 来源身份

- Dao-AILab/flash-attention（`https://github.com/Dao-AILab/flash-attention/tree/73c992c8ca746935548df620ef3c1b6238fe6e68`），固定 commit `73c992c8ca746935548df620ef3c1b6238fe6e68`、包声明 2.8.4。
- FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning（`https://arxiv.org/abs/2307.08691v1`），arXiv:2307.08691v1；§3 累计与分工用于对照，论文伪码和固定实现分别核查。
