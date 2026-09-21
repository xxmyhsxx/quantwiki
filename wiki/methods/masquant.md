---
title: MASQuant：分模态平滑与共享权重补偿
type: method
tags:
  - ptq
  - vlm
  - equivalent-transform
  - low-rank
sources:
  - raw/papers/2026-09-21/masquant/paper.pdf
  - raw/repositories/2026-09-21/efficientai/source/masquant/README.md
  - raw/repositories/2026-09-21/efficientai/source/masquant/main.py
  - raw/repositories/2026-09-21/efficientai/source/masquant/quantize/masquant.py
  - raw/repositories/2026-09-21/efficientai/source/masquant/quantize/quantizer.py
  - raw/repositories/2026-09-21/efficientai/source/masquant/quantize/int_linear.py
  - raw/repositories/2026-09-21/efficientai/source/masquant/quantize/svd_utils.py
  - raw/repositories/2026-09-21/efficientai/source/masquant/infer_mas.py
  - raw/repositories/2026-09-21/efficientai/source/masquant/custom_dataset.py
updated: 2026-09-15
---

# MASQuant：分模态平滑与共享权重补偿

MASQuant 解决一个具体冲突：视觉、文字、音频希望使用不同的逐通道尺度，但共享语言层通常只保存一套主体权重。它先分别学习模态尺度，再以文字尺度对应的量化权重为基准，用低秩分支近似其他模态需要的权重差异。核心价值是把“适合各模态的表示”和“共享主体存储”联系起来；核心代价是补偿矩阵、额外运算与校准依赖。

依据 arXiv:2603.04800v1（2026-03-05，CVPR 2026），本地 10 页含全部图表已全文研读，无独立附录。公开代码定向核对至 EfficientAI commit `3d32ae427eec57166ea67f3018cd4568be84496f`，下称“代码”。论文、代码、下文教学推导分别标明，未运行模型或硬件实验。

## 1. 对象与问题：同一尺度为什么发生冲突

主要实验量化 Qwen2.5-VL 3B/7B 与 Qwen2.5-Omni 3B/7B 的语言部分；Omni 限于 Thinker，不覆盖 Talker、语音生成器与模态编码器。不能将其称为完整多模态模型所有模块的低比特化，也不是 KV cache 方法。对象边界见 [图文 token 与量化对象](../fundamentals/model/vision-language-model-tokens-and-quantization.md)。（§5.1。）

本页采用论文的右乘权重布局：$X^{(m)}\in\mathbb R^{T_m\times d}$，$W\in\mathbb R^{d\times o}$，$Y^{(m)}=X^{(m)}W$；$m$ 表示 token 所属模态。PyTorch Linear 的存储权重是这里的 $W^\mathsf T$，参见 [线性层与输入通道](../fundamentals/operators/linear-layer-input-channel.md)。

[SmoothQuant](smoothquant.md) 使用激活与权重通道范围确定尺度。若混合所有 token 的最大值由某个模态主导，其他模态在不同通道上可能被不均匀地缩小，单个 token 内的有效动态范围反而更差。图 1、4 表明主导模态随层和通道变化，不能简化为“视觉永远比文字大”。该问题涉及通道之间的比例，不是单纯整体幅度大或小。（§4.1。）

对角恒等式本身没有失效：任意可逆对角 $S_m$ 都有

$$X^{(m)}W=(X^{(m)}S_m^{-1})(S_mW).$$

真正的约束是：不同 $S_m$ 通常对应不同 $S_mW$；若都保存，会增加权重副本。只保留一份变换后权重，便不能一般地同时满足所有模态的浮点等价式。参见 [对角缩放的配对条件](../theory/diagonal-scaling-equivalent-transform.md)。

## 2. MAS：分别学习模态尺度

用模态内第 $i$ 通道的激活最大绝对值 $a_i^{(m)}$ 与权重第 $i$ 行的最大绝对值 $w_i$ 初始化：

$$s_i^{(m)}=\sqrt{a_i^{(m)}/w_i},\qquad S_m=\operatorname{diag}(s^{(m)}).$$

这是指数 $\beta=0.5$ 的尺度初始化，随后学习尺度，不把一次范围计算当作最终算法。零值需数值保护。论文式 12–14 使用分模态加权 MAE 重构，可概括为

$$\min_{\{S_m\}}\sum_m\lambda_m\left\|Q_a(X^{(m)}S_m^{-1})Q_w(S_mW)-X^{(m)}W\right\|_1.$$

上式保留线性层机制；实际代码在 Transformer block 输出上计算损失，还存在共享尺度和归一化选项。损失权重、token 数量、每模态均值或总体均值共同决定优化目标，不能只记录 $\lambda_m$。详见 [校准目标与数据](../theory/calibration-and-range-selection.md)。

### SQNR 分析究竟说明什么

论文定理 1 用均匀量化噪声近似解释平滑错位。下面统一记号，给出其成立条件下的教学展开：若逐 token 对称量化步长约为 $\|z\|_\infty/q_{\max}$，并假设高分辨率量化噪声方差为 $\Delta^2/12$，则

$$\operatorname{SQNR}(z)\approx10\log_{10}\left(\frac{12q_{\max}^2}{d}\frac{\|z\|_2^2}{\|z\|_\infty^2}\right).$$

假设各模态独立平滑后该 token 的各坐标恰好等幅，令共享尺度与该模态尺度之比为 $\rho_i=s_i^{\rm shared}/s_i^{(m)}$，共享平滑与独立平滑的信噪比在线性域中的近似比值是

$$\frac{\sum_i\rho_i^{-2}}{d\max_i\rho_i^{-2}}\le1.$$

所有 $\rho_i$ 相同则等号成立：整体缩小十倍并不必然损害这种动态量化的 SQNR。二维 $\rho=(1,10)$ 时比值是 $0.505$，约损失 $2.97$ dB。这里比较的是不均匀通道缩放；实际低比特、裁剪、离散舍入或非等幅输入不满足上述理想条件，不能把近似当作普遍误差界。论文用模态范围比表达时还依赖范围到尺度的映射，不能忽略初始化指数或学习后尺度。（§4.2，式 15–20；此处公式为条件化教学重写。）

## 3. CMC：一套主体权重怎样服务不同尺度

以文字为基准保存 $B=Q_w(S_tW)$。对非文字模态，记

$$A=X^{(m)}S_m^{-1},\qquad D=S_mW-B.$$

于是目标输出是 $A(B+D)$。注意 $D$ 不仅含 $(S_m-S_t)W$，还含文字主体权重的量化误差 $S_tW-Q_w(S_tW)$。CMC 寻找秩不超过 $r$ 的 $L=L_1L_2$，使 $\|A(D-L)\|_F^2$ 尽量小。（§4.3，式 22–28。）

### 为什么先白化，再截断

将激活二阶矩写为 $A^\mathsf TA=P\Lambda P^\mathsf T$，取 $T=\Lambda^{1/2}P^\mathsf T$。在 $A$ 满列秩时，$T$ 可逆且

$$T^\mathsf TT=A^\mathsf TA,\qquad(AT^{-1})^\mathsf T(AT^{-1})=I.$$

因此 $\|A(D-L)\|_F^2=\|T(D-L)\|_F^2$。对 $TD=U\Sigma V^\mathsf T$ 做秩 $r$ 截断，得到

$$L_1=T^{-1}U_r\in\mathbb R^{d\times r},\qquad
L_2=\Sigma_rV_r^\mathsf T\in\mathbb R^{r\times o},$$

$$\min_{\operatorname{rank}(L)\le r}\|A(D-L)\|_F^2=\sum_{i>r}\sigma_i^2(TD).$$

完整的数学解释与反例见 [可逆变换与激活加权低秩近似](../fundamentals/mathematics/invertible-transforms-and-kronecker-products.md)。这个结论是**给定校准激活、残差和秩的最优近似**，没有证明任意模态差异天然低秩。可逆左乘保持代数秩；白化能改变谱能量分布与有效秩。图 5 的有效秩下降是特定模型的观察，不是上述定理的必然结果。奇异二阶矩需要伪逆或阻尼，不能继续无条件使用普通逆和原最优性表述。

### 激活量化以后还有什么误差

论文式 29 的非文字路径为

$$\widehat Y=Q_a(A)B+AL_1L_2,$$

文字路径没有该补偿项。由此严格得到

$$Y-\widehat Y=A(D-L)+(A-Q_a(A))B.$$

第一项是低秩截断误差，第二项是未被这一定理直接最小化的激活量化误差。两项还可能同向或抵消，所以“加权 SVD 最优”不能升级为“W/A 联合量化总误差最优”。式 29 的旁路使用未量化的 $A$，也不能据主体 W4A4 推断旁路是 INT4。

## 4. 流程、实现与剩余缺口

1. 在对应模型和校准数据上构建文字、视觉、音频 mask，统计各模态通道范围；分别初始化并学习尺度。
2. 固定学习后的尺度，用文字尺度生成主体量化权重；对非文字模态收集缩放激活的二阶矩。
3. 为每个目标线性层构建权重残差，白化后截断 SVD，保存低秩因子及各模态尺度。
4. 推理按模态缩放和量化输入，共用主体权重，向非文字输出加入相应低秩分支；检查 mask、矩阵方向和量化网格。

代码补齐了论文未充分展开的若干条件，但不是论文主表配置的自动证明：

| 核对位置（固定代码版本） | 实际含义与边界 |
|---|---|
| `README.md` Quick Start、`main.py` 参数与量化配置 | 示例 128 样本、2 epochs；代码默认 LET 学习率 0.05、AdamW；权重 per-channel、激活 per-token，示例关闭 group。默认 epochs 10 与示例 2 必须区分 |
| `quantize/masquant.py` block 优化与 `loss_multi_modal_mae_alpha` 分支 | masked 绝对误差按总 mask 数归一化；默认视觉系数 0.5，有梯度信息时又设为 0.126。它不是论文所述统一等权设置，不可静默混合结果 |
| `svd_utils.py::get_white_matrix/_svd/_low_rank` | 用缩放后模态激活构造二阶矩，对 $TD$ 做 SVD；比例秩按最小维度取整，设最小 rank 32，并直接求逆、裁剪因子。极小比例未必对应同样小的实际秩，裁剪后也不再严格保留截断最优解 |
| `svd_utils.py::modality_err_low_rank_decomposition` | `quant_cmc=0` 对应本页 $D=S_mW-Q_w(S_tW)$；开启后目标变为两套量化权重之差。排除 visual、audio、lm_head 及名称含 down 的线性层，不能说每个线性层都有 CMC |
| `svd_utils.py::get_effectivate_rank` | 使用归一化奇异值的熵指数 $\exp(-\sum p_i\log p_i)$，再除以最小维度；不是代数秩，也不是保留 99% 能量所需的秩 |
| `int_linear.py::forward_mas_infer`、`quantizer.py::fake_quant` | 量化后反量化为浮点，再调用线性运算；补偿输入不经过该激活量化器。所查路径不能作为真实 INT4 内核的证明 |

还需保留两项会影响实际研究的缺口：

- **校准与测试独立性**：README 将一个 CMC 数据选项写成 COCO + libritest；`custom_dataset.py::prepare_dataset` 的音频分支读取名为 `libri_test_other.jsonl` 的本地文件。当前没有该文件内容与主表样本清单，不能断定与评测重合，也不能宣称已经排除重合。音频收益必须伴随这项协议待核对状态。
- **decode 开销**：论文称文字 decode 无 CMC 额外成本，是其文字基准设计的结构动机；公开 forward 对单 token 改用文字 mask，但仍保留零掩码音频/视觉计算及可能的旁路乘法，没有据此验证无开销。真实融合、条件跳过和计时需另核对。

## 5. 哪些实验改变对方法的判断

以下均为作者报告，不是本项目复现。质量主表与 W4A4 速度表不能拼成同一已验证部署配置。

- **图文收益并非全面领先**：表 1，Qwen2.5-VL 3B W4A8 平均分 MBQ 64.6、MASQuant 64.7；TextVQA 从 MBQ 的 73.4 降至 69.2，而 MMMU 从 41.2 升至 46.7。方法价值需要结合任务，不能只凭平均分说普遍改善。
- **音频对错误平滑很敏感**：表 2，Qwen2.5-Omni 3B W4A8 的 LibriSpeech WER，FP16 3.9、SmoothQuant 77.4、MBQ 9.5、MASQuant 3.6；但 WenetSpeech MASQuant 8.7，弱于 MBQ 8.5，亦弱于 FP16 7.5。保留前述数据独立性缺口，不把单个大幅收益扩大到所有音频任务。
- **损失与最终任务有取舍**：表 4，模态权重 $(1,1)$ 的 PPL/平均准确率为 17.2/56.9，$(1,0.1)$ 为 33.7/58.1。等权更保语言困惑度，却不是该表准确率最高；表 5 增加 epoch 也没有让 PPL 与任务分同步单调改善。
- **rank 是条件化的质量—代价选择**：图 5–6 支持特定层、模型的补偿能量集中与 SQNR 随 rank 改善，不证明固定小秩适用于所有分布。表 6 的 $d+2rd$ 等项是简化附加成本，不是全模型总量。

表 7 使用 RTX 4090、Qwen2.5-VL 7B、序列长度 2048、W4A4 prefill。batch 1 时 FP16 为 191.82 ms/13.73 GB，MBQ 为 68.65 ms/4.85 GB，CMC rank ratio 0.01 为 71.62 ms/4.97 GB，0.05 为 77.10 ms/5.37 GB。后者比 MBQ 慢约 12.3%，不能把开销统一写成 5–10%。batch 8、ratio 0.05 为 696.44 ms/9.42 GB；不同 batch 不能混用。（质量主表主要列 W4A16、W8A8、W4A8、W4A6。）

存储也应按位宽核算：每个非文字模态额外保存 $r(d+o)$ 个因子元素，相对 $do$ 个 $b$ bit 主体权重，若因子为 $p$ bit，附加比例约为 $r(d+o)p/(dob)$。教学例：方阵、$r/d=2\%$、主体 4 bit、因子 16 bit 时附加约 16%，而非 4%。尚未计尺度、缓存和其他模块；一般成本见 [量化矩阵乘法的执行路径](../implementation/quantized-matmul-scaling-execution.md)。

## 6. Strong / weak 与方法位置

**Strong**：明确区分模态各自需要的尺度与共享权重存储约束；补偿目标可写成有条件的解析低秩问题；同一框架覆盖图文与音频输入，使模态冲突从动机变成可分析的残差。

**Weak**：CMC 的保证不覆盖全部激活量化误差；白化与低秩依赖校准分布和数值条件；旁路的位宽、实际秩与执行影响收益；部分公开损失设置与论文默认不一致，音频数据清单和真实低比特内核仍需核对。

与 [MBQ](mbq.md) 的差别是从损失加权走向分别学习表示；与 [MQuant](mquant.md) 的差别是处理变换后权重共享，而分模态量化网格本身未必要求多份权重。[SplitQ](splitq.md) 再把问题拆成通道结构、权重误差和激活误差：它并不是把本方法的模态旁路简单增大。

## 来源身份

下表用于在没有本地资料库时辨识来源；具体论述的章节、公式、图表或代码位置见正文。

| 来源 | 版本或快照 | 说明 |
| --- | --- | --- |
| [MASQuant: Modality-Aware Smoothing Quantization for Multimodal Large Language Models](https://arxiv.org/abs/2603.04800v1) | `arXiv:2603.04800v1` | — |
| [alibaba/EfficientAI](https://github.com/alibaba/EfficientAI/tree/3d32ae427eec57166ea67f3018cd4568be84496f) | `3d32ae427eec57166ea67f3018cd4568be84496f` | — |

## 教学计算材料

保留已有教学计算脚本及当时结果，供核对推导与反例；这些材料不代表模型复现或性能实验。

- [algebra-check.json](../assets/masquant/checks/algebra-check.json)
- [check_algebra.py](../assets/masquant/checks/check_algebra.py)
