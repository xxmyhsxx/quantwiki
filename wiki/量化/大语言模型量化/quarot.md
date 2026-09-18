---
title: QuaRot：固定旋转、全图等价与四位推理
slug: quarot
sources:
  - raw/papers/quantization/ptq/2024-04-quarot-outlier-free-4-bit-inference-rotated-llms/paper.pdf
  - raw/repositories/quantization/2026-07-spcl-quarot-5008669b/source/fake_quant/rotation_utils.py
  - raw/repositories/quantization/2026-07-spcl-quarot-5008669b/source/fake_quant/main.py
  - raw/repositories/quantization/2026-07-spcl-quarot-5008669b/source/quarot/nn/linear.py
  - raw/repositories/quantization/2026-07-spcl-quarot-5008669b/source/quarot/transformers/kv_cache.py
  - raw/repositories/quantization/2026-07-spcl-quarot-5008669b/source/e2e/quantized_llama/modeling_llama.py
updated: 2026-09-15
---

# QuaRot：固定旋转、全图等价与四位推理

QuaRot 先把模型改写到适合量化的坐标系，再量化权重、激活和 KV cache。关键贡献不只是“用 Hadamard 打散离群值”，而是找出残差、归一化、门控 FFN、注意力和缓存中可以配对抵消的变换，并把大部分变换折叠到权重，留下必要的快速在线运算。**固定旋转不需要学习，但默认 GPTQ 权重量化仍需要校准数据。**

本文依据 QuaRot arXiv 2404.00456v2（2024-10-29，NeurIPS 2024，21 页含附录），代码固定为 `5008669b08c1f11f9b64d52d16fddd47ca754c5a`。正文中的实验均为作者报告；代码阅读与下述教学推导不代表模型或硬件复现。后续 [[spinquant|SpinQuant]] 针对旋转选择的差异进一步学习旋转，本页先解释固定路线本身。

## 1. 对象、符号与为什么旋转有用

主要对象是采用 RMSNorm、门控 FFN 和 RoPE 的 LLaMA 类模型。残差宽度为 $d$，FFN 中间宽度为 $f$，注意力有 $h$ 个头，每头宽度 $d_h$。行 token 矩阵 $X\in\mathbb R^{T\times d}$；沿用 [[linear-layer-input-channel|线性层的权重布局]]，$W\in\mathbb R^{m\times d}$，输出 $XW^\mathsf T$。

对正交矩阵 $R\in\mathbb R^{d\times d}$，

$$XW^\mathsf T=(XR)(WR)^\mathsf T,\qquad R^\mathsf TR=I.$$

旋转保持二范数，却改变每个坐标的幅值。当少数坐标决定整个 token 的量化范围时，分散峰值可能减小步长，让普通坐标获得更多有效等级。QuaRot 使用随机符号与归一化 Hadamard 的组合处理全局残差，在线部分采用快速 Hadamard；随机化、维度限制和峰值反例见 [[orthogonal-rotation-and-hadamard-quantization|正交旋转机制]]。（§3.1、§3.4、图 1。）

这不是误差必降定理。若 $E_X=Q_a(XR)-XR$、$E_W=Q_w(WR)-WR$，量化输出误差为

$$E_X(WR)^\mathsf T+(XR)E_W^\mathsf T+E_XE_W^\mathsf T.$$

两侧误差及传播方向共同决定输出；张量更平、权重 MSE 更小或浮点等价都不能单独保证任务质量。QuaRot 的固定旋转在整个校准与部署过程中保持一致，不是在每个 token 上重抽随机矩阵。

## 2. 残差和 FFN 怎样保持等价

### 全局残差坐标只换一次，整张图配套改变

定义无增益 RMS 归一化

$$N(x)=x/\sqrt{\|x\|_2^2/d+\epsilon}.$$

有 $N(xR)=N(x)R$。若原 RMSNorm 的增益为 $D=\operatorname{diag}(\gamma)$，先把它融合进后续线性层；输入权重改为 $W'_{in}=W_{in}DR$，便得到

$$N(xR)(W_{in}DR)^\mathsf T=N(x)D W_{in}^\mathsf T.$$

因此 Q/K/V 投影、FFN 的 up/gate 投影在浮点下得到原来的中间结果。输出投影改为 $W'_{out}=R^\mathsf T W_{out}$，其结果成为原输出右乘 $R$；输出偏置也应变为 $bR$。残差加法于是满足 $(x+F(x))R=xR+F(x)R$。（§3.4、§4 Stage 1a；上式为统一权重方向的教学展开。）

embedding 改为 $ER$，最终语言头改为 $W_{head}R$，中间各块使用同一个全局残差坐标系。不能只旋转线性权重而遗漏 Norm 增益、残差、head 或偏置；一般可逆矩阵也不能直接替代 $R$ 穿过 RMSNorm，因为它不保持分母。

### 门控产生新的中间分布，仍需在线 Hadamard

FFN 的中间激活为 $z=\operatorname{SiLU}(g)\odot u\in\mathbb R^{T\times f}$。全局残差变换让 $g,u$ 保持原值，并不自动改善 $z$ 的量化分布。对归一化正交 Hadamard $H_f$，

$$zW_d^\mathsf T R=(zH_f)(R^\mathsf TW_dH_f)^\mathsf T.$$

右侧权重可以离线保存；$zH_f$ 必须在门控乘法之后、激活量化之前在线执行。不能把 $H_f$ 跨过 SiLU 或逐元素乘法搬到输入权重中。非二次幂 FFN 宽度需按受支持的 Hadamard 分解实现，而不是任意补零后丢弃维度。（§4 Stage 1b；`fake_quant/main.py` 的 `down_proj` 设置，`rotation_utils.rotate_mlp_output`。）

## 3. 注意力和 KV cache 的配对变换

### Value 与输出投影：头内和跨头分开处理

设第 $j$ 个头的注意力概率为 $S_j$、value 为 $V_j$。因为 $S_j(V_jH_{d_h})=(S_jV_j)H_{d_h}$，头内 Hadamard 可直接折叠进 V 投影权重。它不改变注意力概率，只改变被加权的特征坐标。

将各头结果拼为 $A\in\mathbb R^{T\times hd_h}$。QuaRot 对 O 投影输入所需的完整变换可分解为

$$H_o=H_h\otimes H_{d_h}=(I_h\otimes H_{d_h})(H_h\otimes I_{d_h}).$$

第一部分已经由 V 权重吸收，注意力输出处只需在线完成第二部分，即沿头轴混合；O 权重离线改为 $R^\mathsf T W_oH_o$。**跨头混合位于各头注意力完成之后**，不能提前混合具有不同 $S_j$ 的 V，再假定仍与原 attention 等价。（§4 Stage 1c、图 2–3；`rotate_ov_proj`、`main.py` 的 `online_partial_had`；此式对应等宽头的行优先拼接，GQA 需按实际头映射处理。）

### Query 与 Key：必须明确 RoPE 前后

令 $Q_p,K_p$ 已经过 RoPE。在每个头中使用相同正交 $H_{d_h}$：

$$(Q_pH_{d_h})(K_pH_{d_h})^\mathsf T=Q_pK_p^\mathsf T.$$

因此 logits、mask 和 Softmax 概率在未量化时不变。一般旋转不与不同位置的 RoPE 交换，不能直接将这个 post-RoPE 变换无条件折叠到 Q/K 投影。新 K 在 RoPE 后旋转，再量化写入 cache；历史缓存已经在同一坐标系，不需每步重新旋转。查询 Q 在线旋转，保持 FP16，与反量化后的 K 计算。（§4 Stage 1d、Stage 2c；`QKRotationWrapper.forward`。）

V 的头内旋转已在投影权重内，量化缓存存的是该坐标下的 V。要同时记住“缓存保存什么坐标”和“O 投影如何抵消它”，否则局部 QK 恒等也不能保证最终输出正确。

## 4. 从旋转浮点模型到实际低比特执行

论文主设置是 W4A4KV4。位宽只概括被量化对象，不表示所有算子都用 INT4。（§4 Stage 2、§5 Setup。）

| 对象 | 主要设置 | 执行含义 |
|---|---|---|
| Transformer 线性权重 | 对称 INT4，每输出通道；GPTQ，配合权重 MSE 裁剪搜索 | 旋转后重新校准并量化；不能直接复用原坐标中的整数权重 |
| 线性层输入激活 | 对称 INT4，动态 per-token，裁剪比例 0.9 | 先完成必要在线变换，再确定该 token 的 scale |
| K/V cache | 非对称 INT4，group size 128，论文主配置裁剪比例 0.95 | 存整数与元数据；读取时反量化，不等于 INT4 attention 点积 |
| 线性 GEMM | INT4 输入与权重、INT32 累加，再应用 scale 输出 FP16 | scale 的外维关系见 [[quantized-matmul-scaling-execution\|量化矩阵乘法执行]] |
| attention 与其它算子 | 查询及 attention 浮点计算；RMSNorm 用 FP32，残差用 FP16 | Softmax、SiLU、RoPE 等也不是纯整数全图 |

执行流程为：融合 Norm 并配套改写浮点权重 → 固定旋转 → 收集旋转后线性输入做 GPTQ → 保存量化权重/参数 → 部署时在线 Hadamard、动态量化与低比特线性。GPTQ 主配置使用 128 段校准数据，每段 2048 token；其补偿机制由 [[gptq|GPTQ]] 解释。RTN 变体不需激活校准数据，但更低位宽下质量差异明显。

KV 一侧的粒度与布局选择另有专门讨论：本页的旋转作用在 RoPE 之后的 Q/K 上并按 head 施加，而 [[saw-int4|SAW-INT4]] 在服务约束下改用块对角 Hadamard 且只旋转键，并报告融合后的旋转开销只占总运行时 0.1%–0.3%；两者的共同框架见 [[kv-cache-quantization-objects-and-granularity|KV cache 量化的对象与粒度]]。

固定代码把模拟与真实打包分开：`fake_quant/main.py` 添加量化/旋转 wrapper，用浮点张量模拟误差；`quarot/nn/linear.py:Linear4bit` 用 `uint8` 存两个四位权重，接收 `PackedQuantizedTensor` 并调用整数 matmul，再反量化。模拟脚本明确保留 `lm_head` 输入高精度，不能由 W4A4 标签推断所有边界层都量化。

缓存的真实实现 `MultiLayerPagedKVCache4Bit.update` 在 prefill 初始化时写入量化 cache，但返回当前浮点 K/V 给 attention；decode 每次追加一个 token，随后调用缓存解码内核。该实现的 `asym_quantize_and_pack_i4` 按最后一维取范围，保存实数偏移 $o=-\min(x)$，解码为 $q s-o$；它没有论文 0.95 裁剪步骤。这与常见整数 zero-point 网格也应分别核对，不能把模拟质量和部署代码默认行为直接当成同一配置。

本轮只核对上述 Python 入口、打包和调用关系，未逐行验证 CUDA/CUTLASS 内核，也未运行模型。固定版本代码与论文的差别保留为复现条件，不用仓库 README 的早期摘要数字覆盖 v2 结果。

## 5. 哪些精度与消融证据有研究价值

以下 PPL 均为论文的 WikiText-2 口径，越低越好；零样本平均为六项任务的百分比，不能和另一篇八任务平均直接排名。（表 1–13。）

| 要判断的问题 | 有条件的证据 | 可以得到的认识 |
|---|---|---|
| 四位是否近似无损 | LLaMA-2-7B：FP16 PPL 5.47 → QuaRot+GPTQ 6.10，六任务 69.82 → 65.64；70B：3.32 → 3.79，77.07 → 75.98（表 1–2） | 大模型损失较小，但 7B 损失仍明显；“保留约 99%”是 70B 平均分的相对比例，不是每任务无损 |
| 是否有旋转就不需 GPTQ | 同为旋转后 INT4，7B RTN PPL 8.37，对比 GPTQ 6.10（表 1、3） | 改变表示与权重误差补偿互补，固定变换不替代第二阶段 |
| 更细粒度是否有益 | 7B 权重与激活 group 128 时 5.93，group 64 时 5.88，相比主配置 6.10（表 4） | 精度可改善，但 scale 数量、存储与 kernel 要一起评估 |
| 裁剪的组件作用 | 只量化激活的 7B 试验，比例 1.0 为 5.938，0.9 为 5.828，0.85 又为 5.850（表 5） | 适度裁剪有用，越小越好不成立；此数不能当作 W4A4KV4 结果 |
| K 与 V 是否同样敏感 | 只量化 cache 的 7B：K4V2 为 5.75，K2V4 为 8.06，K2V2 为 9.23；FP16 为 5.47（表 6） | 该模型/粒度下 K 极低位宽代价更大；不直接推广为所有模型统一精度分配规则 |
| 任意正交矩阵是否一样 | 7B 使用随机稠密正交残差变换时 PPL 7.45，随机 Hadamard 为 6.10，其余在线 Hadamard 保持相同（表 8） | 正交性只约束等价和条件数，所选基底仍影响量化 |
| 极低位宽和模型迁移边界 | W2A16 的 70B GPTQ 从 25.30 改善到旋转后 5.60，但 7B 仍为 22.07（表 7）；LLaMA-3-70B W4A4KV4 为 2.86 → 6.66，六任务 79.94 → 69.21（表 11–12） | 显著改善不等于损失已可接受；LLaMA-2 的结论不能原封不动迁移 |

6/8 位 RTN、FP16/FP32 Hadamard、Phi-3 扩展也在附录中核对。它们支持存在更宽松的工作区间，不证明所有模型、每项指标和所有执行精度都严格不变。重复模型排名不逐项转录。

## 6. 加速来自哪里，边界在哪里

论文 §5.2 的整块性能主要在 RTX 3090 上测量**单个 Transformer block**，避免模型容量限制；这与完整模型的请求延迟、吞吐和显存不是同一指标。

- Prefill 序列长 2048，表 16 的 LLaMA-2-70B 单块 batch 1 为 3.16 倍，batch 32 为 3.33 倍；7B 各所列 batch 约 1.97–2.16 倍。收益来自低比特线性与整块配合，不能只归因于矩阵旋转。
- 表 15 测量已有 2047 个 token 后的追加/attention，32 个宽度 128 的头：batch 1，FP16 为 0.713 ms，INT4 cache 加 FP32 Hadamard 为 1.163 ms，反而更慢；batch 32 对应 2.098 与 1.247 ms。缓存压缩节省带宽，但旋转、打包和启动开销可能主导小 batch。
- 表 17 的单 token 解码内存测量，7B block、batch 16、cache 长 4096，从 1.416 GB 降至 0.378 GB，约 3.75 倍；这里包含缓存、权重等该测试的内存，不能推出完整模型恰好减少四倍。

在线变换与低比特反量化的代价在其他系统中可以对照 [[qserve|QServe]]：它给出 A100 上 FP32 CUDA 核心的 roofline 拐点、以及从缓存反量化一个 INT4 所需的 ALU 次数，说明这两类操作都可能把注意力内核推向计算受限。具体数值依赖硬件与内核版本，不宜直接套用到本页的 GPU 与配置。
- 表 14 的单线性层速度需绑定矩阵形状。在线 Hadamard 的相对开销随形状变化，不能将某个约 7% 的结果写为全模型常数。

这些结果解释为什么 [[quantized-matmul-scaling-execution|执行路径]] 必须分别看低比特存储、线性整数计算、attention 反量化、prefill/decode 和端到端测量。额外融合与专用内核是性能条件的一部分。

## 7. Strong / weak 与后续路线

**Strong：**在适配的 RMSNorm/attention 图中，固定旋转不需要反向学习；正交逆为转置，数值条件好；全局残差与 V 变换能离线吸收，剩余在线 Hadamard 有快速算法；同一图改写把权重、激活和 cache 纳入一条可部署路线，并可叠加 GPTQ。这使它适合作为研究“变换是否有必要”的清晰基线。

**Weak：**固定基底未针对最终任务误差优化；非线性和 RoPE 使部分变换仍须在线；高精度算子、cache 元数据和小 batch 开销不会因四位标签消失；更低位宽和不同模型的质量损失仍大。数学上的任意正交等价不等于任意旋转都有相同质量，也不替代对实现中融合、精度和缓存坐标的验证。

因此下一步 [[spinquant|SpinQuant]] 的问题很具体：保持可融合的正交结构，能否通过最终语言模型损失找到更好的残差/Value 旋转？与 [[flatquant|FlatQuant]] 比较时还要同时区分变换位置、是否允许拉伸、优化目标及在线成本，不能把这些工作压成单一“矩阵越来越复杂”的顺序。
