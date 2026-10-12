---
title: DBW：在 SVD 分量上联合学习秩与位宽
type: method
tags:
  - qat
  - mixed-precision
  - low-rank
  - weight-quantization
  - kernels
sources:
  - raw/papers/2026-10-07/dbw/paper.pdf
  - raw/repositories/2026-10-07/dbw/source/dbw/utils/decompose.py
  - raw/repositories/2026-10-07/dbw/source/dbw/fake_dbw.py
  - raw/repositories/2026-10-07/dbw/source/dbw/block_ap.py
  - raw/repositories/2026-10-07/dbw/source/dbw/utils/quantize.py
  - raw/repositories/2026-10-07/dbw/source/dbw/real_dbw.py
  - raw/repositories/2026-10-07/dbw/source/dbw/triton_kernel/gemm.py
  - raw/repositories/2026-10-07/dbw/source/train_e2e_qp.py
  - raw/repositories/2026-10-07/dbw/source/dbw/utils/data_e2e.py
  - raw/repositories/2026-10-07/dbw/source/script/train/llama-2-7b.sh
  - raw/repositories/2026-10-07/dbw/source/script/train/common/block.sh
  - raw/repositories/2026-10-07/dbw/source/script/train/common/e2e.sh
updated: 2026-10-07
---

# DBW：在 SVD 分量上联合学习秩与位宽

DBW 把一个线性权重矩阵改成两侧因子与中间系数，给每对分量学习整数位宽。正位宽决定该分量两侧向量的量化网格，0 bit 则删除整对分量，使有效秩与精度共享预算。它的增量是把“先截断到固定秩、再以统一位宽量化”改成可训练的联合分配，并以 Transformer block 输出重构选择这些变量。

本页依据 **Differentiable Bit-Widths: Co-optimizing Pruning and Quantization via SVD for Ultra-Efficient LLM Compression，arXiv:2610.06026v1**（P），核读正文方法、实验及附录 A、C、D.3、E、F；附录 B 为相关工作背景，未逐项核验所有引用；指令模型及 MMLU/GSM8K 扩展表未逐单元复核。定向查看文件页 4、6、9、18、19、24 的公式、主表和实现示意，并核对作者代码 **MMAI-Laboratory/DBW，commit `a45f26e1d153ab6bc64dd27d056a3710a33f2e7a`**（C）的初始化、位宽分配、块训练、打包、恢复和整网微调入口。以下分别标明论文报告、固定源码事实与教学推导；本轮没有编译、训练、加载模型或测性能。

## 1. 0 bit 删除的是成对分量，改变的是中间维度

采用 PyTorch 线性层存储方向：$W\in\mathbb R^{m\times n}$，行输入 $X\in\mathbb R^{T\times n}$，输出 $Y=XW^{\mathsf T}$。对原权重作薄 SVD：

$$W=U\operatorname{diag}(\sigma)V^{\mathsf T},\qquad
U\in\mathbb R^{m\times C},\quad V\in\mathbb R^{n\times C},\quad C=\min(m,n).$$

第 $c$ 个分量是 $\sigma_c u_cv_c^{\mathsf T}$。DBW 给 $u_c$ 和 $v_c$ 使用同一个位宽 $b_c$；若 $b_c=0$，丢弃这两个向量及对应系数。保留集合 $S=\{c:b_c>0\}$，$r=\lvert S\rvert$，则

$$\widehat W=\sum_{c\in S}\widehat\sigma_c\widehat u_c\widehat v_c^{\mathsf T},\qquad
\widehat Y=\big(X\widehat V_S\operatorname{diag}(\widehat\sigma_S)\big)\widehat U_S^{\mathsf T}.$$

删除分量使两次稠密矩阵乘的公共维度从 $C$ 变为 $r$，不需要逐权重稀疏索引。这里“秩 $r$”更严格地说是因子宽度，上式只保证 $\operatorname{rank}(\widehat W)\le r$；训练或量化后的向量不必线性独立。（P §2.2、图 1；C `utils/decompose.py:build_svd_cache`、`fake_dbw.py:get_quv/get_w`。）

**SVD 是起点，不是每步保持的约束。**C 在 FP32 下分解原始权重，将因子保存为 FP16；训练直接更新因子和系数，并逐行归一化恢复后的两侧向量。逐行单位范数不等于所有列相互正交，因此训练后的系数也不能无条件继续解释为有效矩阵的真实奇异值。原始薄 SVD、一般可训练因子和最佳截断 SVD 的解析保证应分开，见[低秩近似基础](../fundamentals/mathematics/invertible-transforms-and-kronecker-products.md#6.%20%E6%BF%80%E6%B4%BB%E5%8A%A0%E6%9D%83%E7%9A%84%E4%BD%8E%E7%A7%A9%E8%BF%91%E4%BC%BC)。

它也不同于 [LoftQ](loftq.md) 或 [QERA](qera.md) 的低秩残差分支：DBW 用两个因子替换整个主权重，不再额外执行一个完整量化主体加旁路；后两者则在量化主体之外保留补偿容量。

## 2. 必须先把原权重 bpw 换成因子预算

每个分量的两侧向量合计 $m+n$ 个元素。忽略系数、尺度、偏移及 word 对齐时，理想整数载荷为

$$B_{\rm payload}=(m+n)\sum_{c=1}^C b_c,\qquad
b_{\rm original}=\frac{B_{\rm payload}}{mn}.$$

这是依据表示形状的**教学计账**；$b_{\rm original}$ 以原矩阵元素数为分母，不等于因子中保留元素的平均位宽。若目标为 $b_\star$ original-weight bpw，理想的分量位宽总和为

$$A=\frac{mn}{m+n}b_\star,\qquad
\overline b_{\rm all\ components}=\frac{A}{C}=\frac{mn}{C(m+n)}b_\star.$$

C `block_ap.py:block_ap` 的 `new_bit = org_size / svd_size * args.wbits` 正是这一步。`fake_dbw.py:apportion_softmax` 随后按 `rank * bit` 分配总和；这里的 `bit` 是因子分量均值，并非直接把原权重目标 1.61 分配给每个分量。

**教学例。**对 $d\times d$ 方阵，完整两侧因子有 $2d^2$ 个元素，若各分量均取 2 bit，原权重口径是 4 bpw。要达到 1.61 bpw，包含零分量在内的因子均值应为 0.805 bit。若所有保留分量都是 3 bit，则 $r/d\approx1.61/6\approx26.8\%$；它不表示原矩阵约 73.2% 的逐元素权重被置零。

同样，源码的“truncation ratio”定义是

$$\rho_{\rm factor}=1-\frac{r(m+n)}{mn},$$

而分量删除比例是 $\rho_{\rm component}=1-r/C$。C `get_effective_avg_bit` 和块统计累计使用前者；P 附录 E.1 将 42.91%–48.90% 叙述为删除的 SVD components，但表 17 的 `Ratio` 与代码的参数减少口径相连，不能据此把它直接代入 $1-r/C$。本页保留这处措辞与统计口径差异，避免用表中的 `Ratio` 反推未经确认的 rank。

完整预算还要加入每组两侧尺度/偏移、系数、位宽组计数、尾部空位和未量化模块；运行峰值另含恢复后的浮点因子及激活。[混合精度预算](../theory/mixed-precision-allocation.md#2.%20%E5%B9%B3%E5%9D%87%E4%BD%8D%E5%AE%BD%E5%92%8C%E5%AE%8C%E6%95%B4%E5%86%85%E5%AD%98%E5%BF%85%E9%A1%BB%E5%88%86%E5%88%AB%E8%AE%A1%E7%AE%97)解释了这几种口径不能互换。

## 3. 连续分配怎样成为预算内的整数位宽

### 3.1 分量级分配与 Hamilton 修正

P §2.2 式 1 对连续变量 $\beta_c$ 使用非负归一化。本页把连续配额写成 $a_c$，避免把它与最终整数位宽混为一物：

$$p_c=\frac{\operatorname{ReLU}(\beta_c)}{\sum_i\operatorname{ReLU}(\beta_i)},\qquad a_c=A p_c,\qquad n_c=\lfloor a_c\rfloor.$$

在分母为正、$A$ 为非负整数且没有位宽上限干预时，令

$$R=A-\sum_c n_c,\qquad f_c=a_c-n_c.$$

选择余数最大的 $R$ 个分量，各加 1，得到 $b_c=n_c+h_c$、$h_c\in\{0,1\}$，于是 $\sum_c b_c=A$。这是 P 附录 C.2 的 Hamilton 最大余数法。舍弃小数后的亏损由第二步补回，而不需要提高全部分量位宽；这只解释离散预算投影，没有证明重构或任务目标的全局最优。

C 实际还执行 `clamp(0, BIT_MAX)`，其中 `BIT_MAX=5`，然后只给尚未到 5 bit 的分量补位，必要时循环；属于带容量限制的实现。它用 `int(rank * bit - n.sum())` 取得剩余额度：连续矩阵预算一般不为整数，且计算受 dtype 舍入影响，因此不能对任意输入声称严格满足原始浮点预算。理想精确算术下，无上限分支分配的是 $\lfloor A\rfloor$；若预算超过 $5C$，所有分量到上限后循环停止。归一化还要求至少一个正的分量参数。上述条件是现有流程的数学前提，不是本轮新增的保护逻辑。（C `fake_dbw.py:apportion_softmax`。）

### 3.2 矩阵级分配共享 block 预算

P §2.2.1 式 3 再学习每个矩阵的预算份额 $\alpha_m$，以 $\widehat A_m=T\alpha_m/\sum_j\alpha_j$ 归一化。矩阵形状不同，统一的 $T$ 必须先选好计费单位；源码以原权重载荷为共同单位：

$$B_{\rm block}=b_\star\sum_m m_mn_m,\qquad
B_m=B_{\rm block}\frac{\alpha_m}{\sum_j\alpha_j},\qquad
\overline b_m=\frac{B_m}{C_m(m_m+n_m)}.$$

C `block_ap.py:block_ap` 初始化 `n_bits_param` 为各矩阵原始载荷占比，每次前向从 `total_memory` 计算份额，再除以 `svd_size_list`，传给分量级分配。因此矩阵较大或因子扩张比例不同，不能直接平均它们的位宽。连续分配阶段守恒的是 block 的理想载荷；各矩阵整数化与最终 packing 后的字节数仍需另算。源码未对矩阵份额执行 ReLU，故其归一化只有在参数和分母保持适当正值时才能作为非负预算分配解释。P 附录 C.4 观察矩阵预算没有坍缩到零，明确说这是经验现象，不是保证。

### 3.3 STE 是代理梯度，不是对整数搜索求精确导数

C `floor_ste/round_ste` 使用 `x + (round_or_floor(x) - x).detach()`：前向为整数值，反向把舍入节点近似成恒等函数；ReLU 和归一化的导数仍由链式法则传播。Hamilton 的排序、TopK、整数剩余数和加 1 选择没有可微导数，不能把“每个分量最多加 1 bit”解释为梯度误差已被同一个 1-bit 上界控制。P 附录 C.2 的稳定性论据是作者消融结果，不是优化收敛或梯度正确性的证明。代理梯度背景见 [PTQ、QAT 与代理梯度](../fundamentals/quantization/post-training-and-quantization-aware-training.md)。

## 4. LBQ 把位宽放进量化比例，0 bit 单独定义

对一个非恒定向量组，令 $\ell=\min w$、$d=\max w-\min w>0$，归一化 $t=(w-\ell)/d$。位宽 $b>0$ 的 LBQ 为

$$q_b=\operatorname{round}\big(t(2^b-1)\big),\qquad
\widehat w=\ell+d\frac{q_b}{2^b-1};\qquad
b=0\Rightarrow\widehat w=0.$$

这是 P 附录 C.3 式 8 的展开。零位宽会让分母消失，必须由删除分量来定义，不能按普通恢复公式除以零。$\ell$ 是浮点恢复偏移，文中名为 $z$，与 [EfficientQAT](efficientqat.md) 的整数域 zero-point 不同。

固定 $t,d,\ell$，忽略裁剪并对舍入使用 STE，可得如下**教学导数**：

$$\frac{\widetilde\partial\widehat w}{\partial b}
=d\frac{t(2^b-1)-q_b}{(2^b-1)^2}(\ln2)2^b.$$

所以即使没有发生范围裁剪，非零舍入残差也可把信号传给位宽；这与只把位宽用作裁剪上端点不同。它仍是正位宽、固定当前组统计下的代理导数，不包含硬删除集合改变时的真实导数。

C `fake_dbw.py:quant/dequant/get_quv` 沿每个向量按 128 元素分组，以 MinMax 加 $10^{-6}$ 的跨度恢复；该路径把 128 写在 reshape 与 repeat 中，不能据构造函数中的 `group_size` 就宣称任意组大小都已支持。代码先用硬 mask 删除 $b=0$ 行，再逐行归一化两侧向量，把非负系数乘到一侧。因而源码输出还包含论文简式之外的方向归一化；实现级数值复现须保留这一节点。

## 5. 训练顺序：论文联合目标与固定脚本要分别读

P §2.2.2 以非负归一化系数定义熵：

$$\pi_c=\frac{\operatorname{ReLU}(\sigma_c)}{\sum_i\operatorname{ReLU}(\sigma_i)},\qquad
L_{\rm ent}=-\sum_c\pi_c\log\pi_c.$$

减小熵鼓励谱能量集中，让更多尾分量更容易删除；它不保证输出质量不下降，也不是秩的精确连续替代。P 算法 1 描述联合优化 $L_{\rm rec}+L_{\rm ent}$，其中 $L_{\rm rec}$ 是当前 Transformer block 与浮点参考输出的元素 MSE。

C 默认训练流程则更具体：

1. 对块内线性权重作薄 SVD，建立浮点目标前缀和当前已量化前缀。当前块的目标来自浮点路径，训练输入来自已处理的量化路径，MSE 包含先前块传下来的输入偏差。
2. `progressive` 预调阶段关闭量化，在浮点因子前向下以 MSE、归一化系数熵及非负系数均值联合更新；两个正则项的权重都来自 `lambda_sv`，默认 $10^{-6}$。
3. 开启 LBQ，按需要以中间响应统计初始化分量份额；局部主训练更新两侧因子、系数、分量份额、矩阵份额及选定归一化权重。**这一主循环只计算 MSE，没有继续加熵项**；优化器是 AdamW。它与论文算法 1 的简化目标不能写成逐节点相同。
4. 按位宽重排并删除 0-bit 行，导出整数因子与元数据；处理后的块推进下一块的量化输入。SVD 预调、局部量化训练和之后整网调参是三个不同计算阶段。
5. `train_e2e_qp.py` 冻结模型后，先把 FP16/BF16 parameters 转为 FP32，再只开放两侧 `scales/zeros` 和 `sigma`，通过常规语言模型标签做整网微调；BF16 前向配置不等于训练参数也全部为 BF16。整数 qweight 与位宽组计数固定，没有再次学习 rank/bit。预训练分支将 token 块复制为 `labels`，交给因果语言模型损失入口，没有教师 logits。这里只核读数据/标签与 Trainer 接口，未运行依赖中的实际损失计算或保存流程，不据此断言整网微调后 checkpoint 的保存 dtype。

（C `block_ap.py:block_ap` 的 `progressive`、主循环与 `fake_to_real`；`utils/quantize.py` 参数选择；`train_e2e_qp.py:get_accelerate_model/train`；`utils/data_e2e.py:group_texts/_build_pt_blocks_from_examples`。）

整网阶段借鉴 EfficientQAT 的少参数训练安排，但 **DBW 同时训练浮点偏移与中间系数**，不是 EfficientQAT 默认的仅分组尺度路径。打包整数固定仍可改变恢复后的有效权重；整网反向的激活及恢复缓冲也仍存在，不能按可训练参数量直接换算训练显存。

### 论文配置与脚本配置的区别

P §3.1、附录 A 使用 RedPajama-1T 4,096 条、长度 2,048，batch 2，每块 2 epochs，权重/分量位宽学习率 $2\times10^{-5}$、矩阵位宽 $10^{-4}$；主要配置为 1.61 original-weight bpw、group size 128。作者用 RTX 3090 24GB 与 256GB 主机内存，较大模型需要多张 GPU；不能把它写成所有模型单卡 24GB 可训练。

C 的 `script/train/llama-2-7b.sh` 保留两次顺序训练：第一阶段 4,096 条、2 epochs，但分量位宽和系数参数组的学习率为 $10^{-5}$；第二阶段经 `common/e2e.sh` 使用混合 RedPajama/WikiText 5,120 条、长度 4,096、1 epoch，单卡 microbatch 4、梯度累积 8、学习率 $2\times10^{-5}$、BF16 前向。不能用论文的局部样本数概括完整训练预算，也不能把所有公开脚本都视为表 8 的同一学习率配置。

## 6. 导出文件保存什么，固定源码怎样读取

C `fake_dbw.py:get_int` 按位宽降序重排两侧因子，删除 0-bit 行，把整数编码、每组跨度与浮点最小值、非负系数及初始系数总和交给 `block_ap.py:fake_to_real`。每块另存位宽到行数的 JSON，以便重建 bit groups；不需要为每个原权重元素保存稀疏 mask，但仍有组计数和恢复参数。

`real_dbw.py` 的表示与调用是：

| 对象或步骤 | 固定代码的实际含义 |
| --- | --- |
| 两侧因子 | 以行表示 SVD 分量；U 侧大小 $r\times m$，V 侧大小 $r\times n$ |
| `Unpacker.qweight` | 将每行因子的 feature 维转到 dim 0 后打包，形状为 $\lceil d/\lfloor32/b\rfloor\rceil\times r_b$ 的 INT32，$r_b$ 为该 bit 组的行数 |
| `scales/zeros` | FP16 组参数；恢复为 $q\,s+\ell$，其中 `scales` 已除以 $2^b-1$，`zeros` 为浮点偏移 |
| `sigma/sigma_total` | FP16 中间系数与初始总和；将 $\operatorname{ReLU}(\mathrm{sigma})\mathrm{sigma\_total}$ 乘到 V 侧 |
| Triton 解包 | `dequant_dim0` 移位、按位 AND，把整数值写到完整 FP16 张量；此 kernel 不执行 GEMM，也未乘组尺度 |
| 浮点恢复 | 拼接各位宽行，再以浮点算子恢复尺度/偏移、逐行 norm 和系数 |
| 主计算 | `F.linear(x, v)` 后 `F.linear(out, u.mT)`，即两次稠密浮点矩阵乘；真实计算 dtype 随输入转换，不是 INT3 Tensor Core 乘法 |

每次 `realDBW.forward` 都调用 `get_uv`，没有在该入口看到“预先缓存恢复后的因子”的支路。因此常驻整数压缩与前向临时完整恢复可以同时成立，恢复成本、峰值和带宽不能仅由载荷 bpw 推出。实际物理表示至少包含两侧 packed word、组参数、中间系数、bias、未压缩层及容器开销。

### INT3/INT5 的 word 空位不能忽略

C `Unpacker.pack` 每个 INT32 装 $\lfloor32/b\rfloor$ 个编码，`dequant_kernel_dim0` 用同一规则读取。3 bit 一 word 装 10 个元素，用 30 bit；5 bit 装 6 个，也用 30 bit，此外还有尾部不满 word 的开销。若某个 bit 组的因子宽度为 $d$、行数为 $r_b$，其一侧整数载荷是

$$B_{\rm words}=32\,r_b\left\lceil\frac d{\lfloor32/b\rfloor}\right\rceil\ \text{bit}.$$

在足够长且整除的维度下，3/5-bit 的实际 word 成本分别为 3.2 和约 5.333 bit/element，而不是严格 3/5。对齐尾部还可能进一步增加。该核对只针对固定快照的保存路径，没有测得任何模型文件大小或显存数字。

## 7. 论文 mixed-precision GEMM 与公开 commit 不能混为同一执行

P §2.3、图 3、附录 C.5 描述：把同位宽分量归组，各组执行 $XV_b\Sigma_bU_b^{\mathsf T}$ 并累加，使用 CUDA streams 并行运行多个小 GEMM；对不整除 32 的位宽分拆到可整除的编码平面，例如把 INT3 拆成 2+1 bit，再在 kernel 中拼回以减少 word 浪费。论文指出这并非完整 fused kernel，短序列可能受到多组 launch 开销限制。

C 的已核对入口则是上一节的普通 word packing、Triton 完整解包、浮点恢复和两次 `F.linear`。在这些调用及 `dbw/triton_kernel/gemm.py` 中没有对应的 bit-plane 分拆、CUDA streams 并行 bit-group GEMM 或融合恢复乘法。**因此论文的 packing/GEMM 测速是作者报告，不能自动归为这个 commit 已提供并运行验证的性能。**数学上按组求和与拼接后的两次乘法可表达同一个因子模型，执行代价却不同；还需独立核对舍入、归一化与系数节点才能保证完整数值一致。

这不是根据文件名判断 kernel：文件名虽为 `gemm.py`，核对到的 `dequant_dim0` 实际只恢复整数值。低比特存储、低比特读取和乘法的 dtype 分工可接[量化矩阵乘执行](../implementation/quantized-matmul-scaling-execution.md)。

## 8. 作者结果支持哪些结论

P 使用 WikiText-2/C4 长度 2,048 的 PPL，以及 lm-eval 0.4.2 的八项零样本 `acc`：PIQA、ARC-e、ARC-c、HellaSwag、WinoGrande、RACE、LAMBADA-OpenAI/Standard。主表为 seed 46 单次结果；附录 A.4 在 LLaMA-2-7B 上补充三个校准种子 44/45/46，不能把该稳定性范围扩写到所有模型与基线。

| 条件与定位 | 作者报告 | 可以支持的结论与边界 |
| --- | --- | --- |
| LLaMA-2-7B，1.61 bpw、g128，表 1–2 | WikiText2/C4 PPL 8.78/12.30，八任务平均 49.06%；FP16 为 5.47/6.97、63.19% | 极端压缩仍保留部分能力，但没有消除质量损失；PTQ1.61 对应为 12.70/17.73、39.20%，位宽标签相同仍需保留训练/粒度差别 |
| LLaMA-3-8B，相同 DBW 名义配置，表 1–2 | PPL 15.96/22.79、八任务 44.38%；FP16 为 6.14/8.88、65.98% | 不能用较大模型的 favorable comparison 概括为极低位宽近似无损 |
| LLaMA-2-7B，SVD-LLM 后 OmniQuant，表 5 | 顺序方案 Avg. PPL 77.10、八任务 29.35%；DBW 10.54、49.06%，表中 Mem 都为 1.56GB | 该匹配载荷附近的联合训练明显更好；不是所有顺序分解算法不可能成功的证明 |
| LLaMA-2-7B，固定协议的分配消融，表 7a | Heuristic 11.69 PPL/45.10%；Learnable 10.54/49.06% | 学习分配比这个 SVD-aware 静态基线有增量；不应只拿 Uniform 650.81/25.05% 来把全部收益归给 STE |
| LLaMA-1-7B，可训练变量消融，表 7b | $\beta,\alpha$ 为 10.54/47.55%；再加 $\sigma$ 为 10.65/48.98% | 增加系数自由度提高了此平均准确率，却没有逐指标同时改善 |
| LLaMA-2-7B，Hamilton 消融，表 11 | Floor-only 11.83/46.13%；Random 11.86/46.18%；Hamilton 10.54/49.06% | 比较支持该离散投影；Floor-only 同时少用了预算，不能据此隔离排序操作自身的全部收益 |
| LLaMA-2-7B，0.5 bpw 扩展，表 16 | DBW Avg. PPL 27.68；SVD-LLM+GPTQ 为 21.62，但后者 0.77 bpw | 不构成相同预算的顺序方案胜出，也不能把 DBW 写成任何存储预算下全面领先 |

附录表 17 报告 LLaMA-2-7B 的完整 weight memory 为 12.31GB→1.56GB（7.89 倍），LLaMA-2-70B 为 145.49GB→16.16GB（9.00 倍）；按论文说明包含高精度 embedding/norm 与量化恢复参数。它们是权重内存口径，不是端到端峰值、KV cache 或保存文件测量；本轮也没有在固定源码中复现它们。

### 性能证据的范围

P 图 5 报告 LLaMA-65B、RTX 3090、FP16 cuBLAS 与 OmniQuant INT2 的延迟对照，10 次 warmup、30 次平均；作者称为 end-to-end latency，并说明 64 tokens 时 DBW 略慢于 OmniQuant，长 token 范围的收益更好。但附录 A.2 又以 CUDA GEMM runtime 描述测量，未给出完整服务、prefill/decode 分解和多 GPU 调度协议；65B 压缩后的权重仍有 13.66GB，FP16 121.11GB。不能将这一图直接转成“单卡 RTX 3090 实际装载 FP16 65B 的整段生成加速”。

附录表 12 则明确是一个 LLaMA-65B block 的七个矩阵，token 范围 1–128：FP16 1,544MB/3.08ms，INT3 290MB/2.42ms，INT2 193MB/1.83ms。它支持作者优化 kernel 下的局部存储与算子结果，表内没有逐长度的完整分布，不能当成任意 decode/prefill 都有相同 speedup。对应代码差异见第 7 节，服务测量口径见[服务性能评测](../implementation/serving-performance-evaluation.md)。

## 9. 未解差异与使用边界

- 论文算法 1 将重构和系数熵写成同一联合目标；固定快照把熵与额外系数均值正则放在无量化的 `progressive` 阶段，主 LBQ 训练仅 MSE；学习率及完整整网微调预算也不完全由表 8 描述。
- 论文的 INT3/INT5 分 bit-plane 无填充 packing 与并行 bit-group CUDA GEMM，尚未在本轮固定路径里找到；不据此宣称论文性能被复现，也不据文件名推断实际运算。
- 附录 E.1 的 component pruning 措辞与代码的 factor-parameter reduction 不同。表 19 还把各矩阵 `Bit` 标成 original-weight bpw，但这些数按表中 FP16 大小加权约为 1.11，而表尾为 1.61；`Mem` 总和约 38.86MB 则与均值 5.55MB 接近一致。本页以形状与源码推导预算，不将这张表当成精确数值换算依据；未修改原件或断言哪个字段应被替换。
- 训练后的因子不保持 SVD 正交性；低熵、位宽预算不坍缩和 STE/Hamilton 的经验稳定性，没有提供整个 block 优化的全局保证。
- 方法和主实验针对权重量化，未量化激活或 KV；低位宽仍有明显任务损失。动态 rank/bit 的部署收益依赖目标矩阵、token 数、恢复与 kernel 调度，不由 nominal bpw 自动推出。

DBW 承接[低比特表示设计](../research/low-bit-representation-design.md)中的另一个变量：除了变换坐标、码本和例外，还可以改变主体矩阵的因子宽度；承接[混合精度分配](../theory/mixed-precision-allocation.md)时，选择单元是成对分量，成本是因子元素数而非原权重元素数；承接 [EfficientQAT](efficientqat.md) 时，需逐阶段核对可训练变量和真实恢复路径。上述机制、固定源码与作者结果已经核读，运行正确性、模型质量及性能复现仍是独立验证层级。

## 来源身份

| 来源 | 固定版本 | 本页使用范围 |
| --- | --- | --- |
| Differentiable Bit-Widths: Co-optimizing Pruning and Quantization via SVD for Ultra-Efficient LLM Compression（`https://arxiv.org/abs/2610.06026v1`） | `arXiv:2610.06026v1` | §2–3 方法与实验；附录 A、C、D.3、E、F；算法 1、表 1–2、5、7–8、11–12、16–17、19；扩展模型与任务表未逐单元核验 |
| MMAI-Laboratory/DBW（`https://github.com/MMAI-Laboratory/DBW/tree/a45f26e1d153ab6bc64dd27d056a3710a33f2e7a`） | `a45f26e1d153ab6bc64dd27d056a3710a33f2e7a`，2026-10-07 获取 | `decompose.py`；`fake_dbw.py`；`block_ap.py`；`quantize.py`；`real_dbw.py`；`triton_kernel/gemm.py`；`train_e2e_qp.py`、预训练标签分支与 LLaMA-2-7B 两阶段脚本。未安装依赖、编译 kernel、下载模型或运行训练/推理 |
