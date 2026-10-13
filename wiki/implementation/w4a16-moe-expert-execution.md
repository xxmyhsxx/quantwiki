---
title: vLLM W4A16 MoE：AWQ 专家格式、配对读取与低精度加权
type: implementation
tags:
  - moe
  - weight-quantization
  - routing
  - kernels
  - data-format
sources:
  - raw/repositories/2026-09-21/vllm
updated: 2026-10-07
---

# vLLM W4A16 MoE：AWQ 专家格式、配对读取与低精度加权

Dense W4A16 已解释一块低比特矩阵怎样恢复；MoE 还需要确定**当前 tile 属于哪个专家、输入来自哪个 token、输出写回哪个配对，以及 gate 在哪种精度下乘**。本页连接 AWQ 加载、Marlin 专家设备计算与 token 汇合；路由规则和模型侧 shared/rho 归属由[既有 MoE 执行页](moe-routing-and-quantized-expert-execution.md#8.%20vLLM%20%E5%AF%B9%E7%85%A7%EF%BC%9A%E5%90%8C%E4%B8%80%20FP8%20%E8%AE%A1%E7%AE%97%E7%94%B1%20factory%20%E4%B8%8E%20modular%20kernel%20%E5%88%86%E5%B7%A5)承接。

固定来源为 vLLM commit 568afb3a13806beb53bb2e6bd518269357b237c0（`https://github.com/vllm-project/vllm/tree/568afb3a13806beb53bb2e6bd518269357b237c0`）。本文是静态源码核对，教学例子为独立索引与舍入计算，没有加载模型、编译或运行 GPU。

## 1. 固定可连通的分支与接口

选择 `AutoAWQMoEMethod → MARLIN/MarlinExperts` 的 Standard 格式，非对称 `kU4`/W4A16、整数 packed zeros、组长128、CUDA SM80+、BF16 输入/参数尺度/输出，记这一数据类型为 T。记输入为 $X\in\mathbb R^{M\times D}$，本机专家数为 $E$，每 token 路由数为 $R$，每专家 intermediate size 为 $I$。取 $D,I\ge256$ 且均被128整除，使两次 GEMM 都使用多组恢复。[1–5]

本分支 TP/EP/DP/PCP 均1，无 sequence parallel、all-to-all、LoRA、EPLB、DBO、batch-invariant 或 latent transform；采用普通 gated SiLU，无 clamp/bias 或 fused shared expert 槽位。`apply_router_weight_on_input=False`，不设置 `VLLM_MARLIN_INPUT_DTYPE`，`VLLM_MOE_SKIP_PADDING=0`。后者保留普通配对 padding，不混入 modular 的 graph padding 分支。FP16 是设备支持的另一种 T 类型，但本页模型侧延伸只沿 BF16；SM80+ 是核读范围，不是所有 Marlin MoE 的最低能力声明。[1][3][4][8][17][28]

进入专家接口的是合法 `topk_ids[M,R]` 和 FP32 `topk_weights[M,R]`，每个 ID 满足 $0\le e<E$。EP=1 时全局与本地专家数相同、`expert_map=None`。NoDPEP prepare 不物理排序 X；无激活量化 dtype 时返回原隐藏状态和空 activation scale，后续两次 GEMM 都读 BF16 激活。[6][7][9][11]

`AutoAWQConfig` 先排除 ignored 层，再检查 Marlin MoE 的形状/组长条件，失败可转 MoeWNA16；`use_marlin=True` 这个成员不能代替 oracle 的 backend 判断。此处明确选取并通过 `moe_backend="marlin"` 的配置检查：device、量化 scheme、activation、parallel、shape、Standard format 和 LoRA 等均由候选接口核定。auto 候选顺序和 Batched/Humming/emulation 分支没有并入本页。[1][2][3][8]

## 2. AWQ 专家参数怎样变成 Marlin 布局

`w13` 合并 gate/up 两块权重，逻辑输出前 I 列为 gate、后 I 列为 up；`w2` 是 down。下表采用 GEMM 的逻辑 `[reduction, output]` 方向，每个专家单独保存量化组。`qweight/qzeros` 是 INT32、每 word 八个4-bit码；尺度是 BF16。[1][10]

| 参数 | 加载时的 AWQ 形状 | 准备后的 Marlin 形状 |
| --- | --- | --- |
| `w13_qweight` | `[E,D,2I/8]` | `[E,D/16,4I]` |
| `w2_qweight` | `[E,I,D/8]` | `[E,I/16,2D]` |
| `w13_scales` / `w13_qzeros` | `[E,D/128,2I]` / `[E,D/128,2I/8]` | 形状相同，分别重排列序 |
| `w2_scales` / `w2_qzeros` | `[E,I/128,D]` / `[E,I/128,D/8]` | 形状相同，分别重排列序 |

`process_weights_after_loading` 委托 WNA16 oracle 的 `_process_awq_weights_marlin`。它为每个专家直接调用 `awq_marlin_repack`；这与 [Dense AWQ](awq-implementation.md#MPLinear%EF%BC%9A%E6%A0%87%E5%87%86%E5%8C%96%E5%90%8E%E5%86%8D%E9%80%89%E6%8B%A9%E5%86%85%E6%A0%B8%E5%B8%83%E5%B1%80) 的“先标准化、再 `gptq_marlin_repack`”不同。MoE 包装循环传入每个 `[k,n/8]` AWQ 矩阵，CUDA repack 要求 k 被16、n被64整除，输出 `[k/16,2n]`。本页 D/I 条件满足这些要求，无需额外 intermediate padding。[1][2][15][16]

尺度沿每专家做 Marlin 64元素置换。零点先解包并撤销 AWQ nibble 次序，再做相同64列置换、nibble interleave 和 INT32 packing；保持原整数 z，不加1、不减8、不预乘 scale。最终 qweight、scales、qzeros 必须作为配套格式使用，不能只换权重。[2][4]

这里有一个需要保留的调用条件：oracle 对 `w13_scales` 置换传入的 `size_k` 是 I，实际第一 GEMM 的 reduction 是 D。所选 D 和 I 都大于128，两种值均进入多组64元素置换，故本页链能对应；**D=128 或 I=128 的单组边界没有由本次核读证明**，不能默默将源码形参改写为 D 再推广结论。[2][4]

准备完成后 `g_idx=None`、sort indices 为 `[E,0]`，没有 act-order。`int4_w4a16_moe_quant_config` 以 `block_shape=[0,128]` 携带权重尺度和零点，A1/A2 均无量化 dtype；原 AWQ 参数经替换交给 `MarlinExperts`。[1][2][5]

## 3. 配对索引怎样连接 token、专家与输出

把第 t 个 token 的第 j 路定义为配对 $p=tR+j$，合法范围为 $0\le p<MR$。`moe_align_block_size` 按专家收集 p，每个专家尾部补到 $B_M$ 的倍数；sentinel 是 MR。`expert_ids[b]` 标明第 b 个专家 block，`num_tokens_post_padded` 给出实际消费长度 P。Marlin 总是调用 align，不采用既有 FP8/Triton 的小批量 bypass。[3][12][20]

$B_M$ 从 `[8,16,32,48,64]` 中选择首个满足 $MR/(E B_M)<0.9$ 的值，没有满足者取64。设专家 e 的有效配对数为 $m_e$，则 $P=\sum_e\lceil m_e/B_M\rceil B_M$。`sorted_token_ids` 的分配容量通常为 $MR+E(B_M-1)$，小输入还可收紧；容量与 P 不同，设备只按 P 的 block 调度。[3][12]

**教学索引例。** 取 M=3、R=2、E=2，ID 矩阵为 `[[1,0],[0,1],[1,0]]`，则 $B_M=8$、P=16，而排序容量为20。一种合法的专家内次序是：

| expert block | 有效配对 p | 末尾 padding | 第一 GEMM 的 X 行 | 第二 GEMM 的 activation 行 / 两次 GEMM 的 C 行 |
| --- | --- | --- | --- | --- |
| expert 0 | `1,2,5` | 五个 `6` | `0,1,2` | `1,2,5` |
| expert 1 | `0,3,4` | 五个 `6` | `0,1,2` | `0,3,4` |

这张表解释地址关系，不声称 GPU 的原子收集保证专家内次序稳定。设备第一 GEMM 用 `p//R` 读原 token，第二 GEMM 的 `top_k=1` 使其用 p 读激活；两者均 scatter 到 p。第二次传1是读取 stride 的协议，没有把模型的路由数 R 改成1。[3][18][20]

每个 block 的有效配对在前、padding 在后。设备统计 `p<MR` 的 valid prefix，异步 A 载入与最终写回按此前缀限界；这不等于每个 padding lane 都跳过 MMA。普通 predicated `cp.async` 不执行拷贝时也不能当成零填充。padding 槽的结果不写成有效配对；已分配容量中、P 所覆盖 block 之外的 expert sentinel 不参与设备调度。[18][24]

本页只接合法原始路由。align 的 `ignore_invalid_experts=True` 能排除无效 ID，但无 map 的最终 sum 不接收 `topk_ids`，不能据此承诺任意 `-1` 路由都可靠跳过；有 expert map 的 pad-aware sum 是另一条条件分支。[3][12][20]

## 4. 当前 block 怎样选择专家的低比特数据

对任一次 GEMM，记 reduction 为 k、输出宽度为 n、组数为 $L=k/128$。设备读 `expert_ids[b]`，切换 B 的专家基址，同时以专家 ID 差值移动 scales/zeros 指针。本分支用16字节 `int4` 向量作地址单位；它包含四个 INT32 word，不是“一个4-bit权重”的 C++ 类型。[18]

| 数据 | 每专家 stride，以16字节向量计 | 每量化组 stride |
| --- | --- | --- |
| packed B | $kn/32$ | B 自身随 K tile 推进 |
| BF16 scales | $Ln/8$ | $n/8$ |
| packed integer zeros | $Ln/32$ | $n/32$ |

例如第一 GEMM 的 D=256、I=256，即 k=256、n=512、L=2。专家1的 B/scales/zeros 基址分别为4096/128/32个16字节向量；它的组1 scales/zeros 基址为192/48个向量，后者等于192个 INT32 word。这个独立地址例同时满足两次 GEMM 的主线形状条件。[18]

进入 shared 后，组元数据按128个 reduction 元素更新；thread-K 为64时可复用两段，thread-K 为128时一段对应一组。整数 z 和 q 使用相同 lane→输出列关系以及 FP16/BF16 的1024/128共同浮点基底，先相减抵消基底，再在 T 类型的 B fragment 上乘本组尺度。64列置换、lane/word/fragment 对应和 shared ring 细节沿用[Dense Marlin 零点消费](weight-only-dequant-kernels.md#11.%20%E9%9D%9E%E5%AF%B9%E7%A7%B0%20AWQ%20%E2%86%92%20vLLM%20Marlin%EF%BC%9A%E6%95%B4%E6%95%B0%E9%9B%B6%E7%82%B9%E6%80%8E%E6%A0%B7%E5%8F%82%E4%B8%8E%E8%AE%A1%E7%AE%97)；MoE 新增的专家轴由上述 stride 承接。MoE 模板直接包含共同 dequant/dtype/MMA 代码，本页没有把对称 GPTQ 固定减8或 float-zp 混入整数 z 路径。[4][18][25–27]

转换与累加精度的前置解释见[浮点表示与累加](../fundamentals/numeric-formats/floating-point-and-accumulation.md)。若用 $\operatorname{RN}_T$ 表示转换/运算结果落到 T 的舍入，一个恢复元素可表示为

$$
\widetilde W_{e,k,n}=\operatorname{RN}_T\!\left((q_{e,k,n}-z_{e,\lfloor k/128\rfloor,n})s_{e,\lfloor k/128\rfloor,n}\right).
$$

尺度本身已是 T。SM80+ 将恢复后的 T fragment 送入 FP32 MMA accumulator；按 CTA 切片需要做共享/跨 CTA 归约。两次 Python GEMM 调用都显式传 `use_atomic_add=False`、`use_fp32_reduce=True`、`is_zp_float=False`，不使用 Dense 的 atomic 环境启发式。只有所需跨 CTA 的部分和使用 FP32 `C_tmp`；真实中间和输出 cache 仍是 T。[3][17][18][25–27]

## 5. 两次 GEMM、SiLU 与 gate 的真实舍入顺序

| 阶段 | 逻辑输入 / 输出 | 配对协议与加权 |
| --- | --- | --- |
| gate/up GEMM | X `[M,D]` → C13 `[MR,2I]` | `size_m=M, top_k=R`；读 `p//R`，写 p；不乘 router weight |
| SiLU×up | C13 前后两半 → H `[MR,I]` | 普通 gated SiLU，结果仍是 T；没有 A8 量化 |
| down GEMM | H `[MR,I]` → C2 `[MR,D]` | `size_m=MR, top_k=1`；读 p，写 p；在输出阶段乘第 p 路 gate |
| routed sum | C2 的 `[M,R,D]` view → Y `[M,D]` | 按原 j 次序把 T 配对结果提升 FP32 求和，再转 T |

两次 GEMM 使用相同 `sorted_token_ids/expert_ids/P`，因此 H 没有被重新排成专家连续矩阵。内核计算按专家 block 组织，cache 则按原配对 p 存放。`_custom_ops.moe_wna16_marlin_gemm` 经 `_moe_C` 注册进入 host；host 检查 A/B/scales 的设备与连续性、k/n 和 packed shape，再由组数/量化类型/架构选择模板。[3][15][17][18][21]

**普通 SiLU 有两个 T 舍入位置。** 设第一 GEMM 存出的 gate/up 为 $u_p,v_p\in T$。`silu_kernel` 在 float 中用 `expf` 计算，再返回 T；随后把这个 T 和 up 提升 float 相乘，再返回 T：[13][22][23]

$$
a_p=\operatorname{RN}_T\!\left(\phi_{32}(\operatorname{float}(u_p))\right),\qquad
h_p=\operatorname{RN}_T\!\left(\operatorname{mul}_{32}(\operatorname{float}(a_p),\operatorname{float}(v_p))\right).
$$

$\phi_{32}$ 指源码的 float SiLU 计算，不是无限精度的数学函数；u/v 来自 gate/up 权重，与下面的 router gate g 是不同变量。不能把这条普通激活链改写成只在 `SiLU(u)*v` 最后舍入一次；它也不同于既有 FP8 页的融合 SiLU/量化分支。

**Marlin 的 router gate 在低精度中乘。** 设无 bias 的 down FP32 归约结果为 $F_p$，原 gate 为 $g_p^{32}$。设备先把 gate 转 T，把 F 转 T，再用 `__hmul2` 做 T 乘法并写 C2：[18][26]

$$
C2_p=\operatorname{RN}_T\!\left(\operatorname{RN}_T(F_p)\,\operatorname{RN}_T(g_p^{32})\right),\qquad
Y_{t,d}=\operatorname{RN}_T\!\left(\sum_{j=0}^{R-1}{}_{\!\text{FP32}}\operatorname{float}(C2_{tR+j,d})\right).
$$

求和记号的 FP32 表示按 j=0,…,R−1 顺序加到 float accumulator。它不等于既有 FP8/Triton 的“在 FP32 accumulator 上乘 gate 后转 T”。教学例取 BF16、$F=1.00390625$、$g=0.99609375$：F 在 BF16 中以 ties-to-even 舍入为1，Marlin 顺序得到0.99609375；直接把 FP32 的 $Fg=0.9999847412109375$ 舍入为 BF16 则得到1。例子说明舍入顺序可改变数值。

这里的 sum 来自实际 `MarlinExperts.apply` 传入的 `self.moe_sum` callback，继而调用 CUDA `ops.moe_sum`；不是 standalone `fused_marlin_moe` 无 callback 分支的 `torch.sum`。有 map 的 sum 可按原 ID 跳过无效配对，本页无 map 主线直接按合法 R 路求和。[3][20]

## 6. 缓冲复用与输出归属

需要区分三类 workspace：[3][4][17]

- modular 的 T 激活缓冲：`workspace1=[MR,max(I,D)]`；`workspace2` 容量为 `MR*max(2I,D)`。apply 把 workspace2 交给 cache13，先作为 `[MR,2I]` gate/up 输出，再复用为 `[MR,D]` down 输出；workspace1 承接 `[MR,I]` 激活，消费后可复用为 routed `[M,D]` 输出。
- Marlin 的 INT32 归约锁：`marlin_make_workspace_new` 分配4倍 SM 数目的零数组，`_fused_marlin_moe` 无传入 workspace 时创建并供两次 GEMM 使用。虽然 AutoAWQ 层创建过 `layer.workspace`，所选 `MarlinExperts.apply` 没有把它传入，不能称实际复用了该成员。[1][3][4]
- 每次 GEMM host 的 FP32 `C_tmp`：用于所需跨 CTA 部分和，按 fragment 布局存储，不是完整恢复后的权重或一个持久专家输出池。

cache 互换避免输出归约与仍被消费的数据别名。专家结果已在 down 阶段乘 gate、在 `moe_sum` 阶段归约；`finalize_weight_and_reduce_impl` 返回 `TopKWeightAndReduceNoOP`，所选 CUDA modular 的 NoDPEP finalize 将 routed workspace 复制到调用者输出，没有再乘一次 gate 或再做一次专家 sum。[3][7][8][14]

模型继续怎样用 routed 结果取决于 runner：BF16 DeepSeek 所选分支的 routed rho 缩放、原始 X 的 shared MLP、两者相加及外层 residual，见[既有模型汇合链](moe-routing-and-quantized-expert-execution.md#8.5.%20cache%20%E6%98%AF%E4%B8%B4%E6%97%B6%20workspace%EF%BC%8Cfinalize%20%E4%B8%8D%E5%86%8D%E5%8A%A0%E6%9D%83%E4%B8%80%E6%AC%A1)。本页替换的是该链中的 routed experts 执行；不据此推断 FP16 的 shared/rho 补偿、其他模型或多设备归约都相同。

## 来源身份

本页使用统一归档于 2026-09-21 的 vLLM 官方 v0.26.0 固定源码，source_id 为 `vllm-project-vllm-568afb3a`，resolved commit 为 568afb3a13806beb53bb2e6bd518269357b237c0（`https://github.com/vllm-project/vllm/tree/568afb3a13806beb53bb2e6bd518269357b237c0`）。

### 核读定位与验证边界

正文的方括号编号对应下表；文件路径相对 vLLM 固定仓库根目录。

| 编号 | 固定源码文件 |
| --- | --- |
| [1] | `vllm/model_executor/layers/quantization/auto_awq.py` |
| [2] | `vllm/model_executor/layers/fused_moe/oracle/int_wna16.py` |
| [3] | `vllm/model_executor/layers/fused_moe/experts/marlin_moe.py` |
| [4] | `vllm/model_executor/layers/quantization/utils/marlin_utils.py` |
| [5] | `vllm/model_executor/layers/fused_moe/config.py` |
| [6] | `vllm/model_executor/layers/fused_moe/all2all_utils.py` |
| [7] | `vllm/model_executor/layers/fused_moe/prepare_finalize/no_dp_ep.py` |
| [8] | `vllm/model_executor/layers/fused_moe/modular_kernel.py` |
| [9] | `vllm/model_executor/layers/fused_moe/utils.py` |
| [10] | `vllm/model_executor/layers/fused_moe/routed_experts.py` |
| [11] | `vllm/model_executor/layers/fused_moe/expert_map_manager.py` |
| [12] | `vllm/model_executor/layers/fused_moe/moe_align_block_size.py` |
| [13] | `vllm/model_executor/layers/fused_moe/activation.py` |
| [14] | `vllm/model_executor/layers/fused_moe/topk_weight_and_reduce.py` |
| [15] | `vllm/_custom_ops.py` |
| [16] | `csrc/libtorch_stable/quantization/marlin/awq_marlin_repack.cu` |
| [17] | `csrc/libtorch_stable/moe/marlin_moe_wna16/ops.cu` |
| [18] | `csrc/libtorch_stable/moe/marlin_moe_wna16/marlin_template.h` |
| [19] | `csrc/libtorch_stable/moe/marlin_moe_wna16/generate_kernels.py` |
| [20] | `csrc/libtorch_stable/moe/moe_align_sum_kernels.cu` |
| [21] | `csrc/libtorch_stable/moe/torch_bindings.cpp` |
| [22] | `csrc/libtorch_stable/activation_kernels.cu` |
| [23] | `csrc/libtorch_stable/cuda_vec_utils.cuh` |
| [24] | `csrc/libtorch_stable/quantization/marlin/marlin.cuh` |
| [25] | `csrc/libtorch_stable/quantization/marlin/dequant.h` |
| [26] | `csrc/libtorch_stable/quantization/marlin/marlin_dtypes.cuh` |
| [27] | `csrc/libtorch_stable/quantization/marlin/marlin_mma.h` |
| [28] | `vllm/envs.py` |

下表的核读位置使用文件名或局部路径简写，完整路径以上方编号表为准。

| 核对对象 | 定位 |
| --- | --- |
| AWQ 选择/创建/转换；WNA16 oracle | [1] `auto_awq.py:335–358,547–735`；[2] `oracle/int_wna16.py:136–246,308–365,675–835,1335–1368` |
| repack / 元数据置换 | [15] `_custom_ops.py:1170–1189`；[16] `awq_marlin_repack.cu:228–263`；[4] `marlin_utils.py:471–555` |
| Standard experts / 配对与两次 GEMM | [3] `experts/marlin_moe.py:97–218,302–390,609–662,710–812,937–947`；[12] `moe_align_block_size.py`；[20] `moe_align_sum_kernels.cu:85–323,626–721` |
| host / 模板 / 生成规则 | [17] `marlin_moe_wna16/ops.cu:626–891`；[18] `marlin_template.h:310–562,684–791,974–983,1223–1426,1535–1764,1795–1900,2162–2186`；[19] `generate_kernels.py:65–72,177–220` |
| SiLU / CUDA sum / finalize | [13] `fused_moe/activation.py:145–161`；[22] `activation_kernels.cu:27–75,156–169,290–294`；[20] `moe_align_sum_kernels.cu:373–493,757–920`；[7] `no_dp_ep.py:57–97`；[14] `topk_weight_and_reduce.py:44–77` |

生成脚本登记所选同 T、kU4、group-blocks=8 的模板规则。本文核通所选 BF16/SM80+/G128 多组、单设备合法路由的准备、专家读取、恢复与输出接口。单组形参边界、SM75/A8、其他组长/格式、Batched/其他 backend、act-order、任意无效 ID、TP/EP、LoRA/DBO/graph/多 stream 和其他 tile/容量组合仍需沿具体条件核查。
