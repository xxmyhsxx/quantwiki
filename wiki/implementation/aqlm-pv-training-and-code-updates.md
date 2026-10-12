---
title: AQLM 的 PV 微调：梯度、选码目标与权重写回
type: implementation
tags:
  - aqlm
  - pv-tuning
  - vector-quantization
  - training
sources:
  - raw/repositories/2026-09-21/aqlm/source/README.md.txt
  - raw/repositories/2026-09-21/aqlm/source/finetune.py
  - raw/repositories/2026-09-21/aqlm/source/src/pv_optimizer.py
  - raw/repositories/2026-09-21/aqlm/source/src/pv_utils.py
  - raw/repositories/2026-09-21/aqlm/source/src/configurable_adam.py
  - raw/repositories/2026-09-21/aqlm/source/src/aq.py
  - raw/repositories/2026-09-21/aqlm/source/src/beam_search_l2.py
  - raw/repositories/2026-09-21/aqlm/source/src/utils.py
  - raw/repositories/2026-09-21/aqlm/source/src/datautils.py
  - raw/repositories/2026-09-21/aqlm/source/src/memory_efficient_loss.py
  - raw/papers/2026-09-22/pv-tuning/paper.pdf
updated: 2026-10-12
---

# AQLM 的 PV 微调：梯度、选码目标与权重写回

本页核读 **Vahe1994/AQLM `e79a896ed6656fe4ed06193d42d004e7d0bbdbb2`** 的根目录 `finetune.py` 到 PV 优化器、L2 选码及权重写回。它说明 [PV-Tuning](../methods/pv-tuning.md) 如何使用一份恢复权重梯度，同时训练连续表示参数并构造离散选码目标。源码 README 已记录 PV 分支合并；本快照与论文 arXiv:2405.14852v2 分别保留身份，不视为论文每张表的精确运行版本。

范围是已加载 AQLM 权重后的单码本/两码本 PV 主链，以及数据分块、KL、优化器和保存接口。本页依据静态代码与教学算例，未运行模型、分布式通信或性能测试。原始激活感知校准、条件转换与所选 CUDA 2×8 前向读取另由[校准与推理实现页](aqlm-calibration-export-and-inference.md)承接；全部加载适配器、其他后端与分布式组合仍未完整核读。

## 1. 真正进入 PV 的条件与三份状态

`main` 实际以 `update_codes and not force_dequantize` 选择 PV 分支。常规用法打开 `update_codes`，并按需打开中心/尺度和未量化参数训练。两个开关 `update_codes` 与 `force_dequantize` 同时打开会进入 P-only 优化器，而后者明确拒绝 `update_codes`；不能照后者的 help 文字推断该组合会正常做 PV。

`load_student_model` 先把权重载入为 `QuantizedWeight`。PV 分支经 `create_dequantized_model` 把相应层替换为普通 `nn.Linear`，同时保留名字到量化对象的映射。这形成三种不同职责的状态：

| 状态 | 内容与作用 | 谁会更新 |
| --- | --- | --- |
| 前向恢复权重 $W^k$ | 当前码本、尺度和整数码恢复出的稠密张量，供学生前后向 | 每步结束由量化对象重新写回 |
| 量化表示 $C^k,s^k,z^k$ | 中心、尺度及整数组索引 | 中心/尺度用连续梯度；索引用搜索 |
| 可选连续缓冲 $B^k$ | 与完整权重同形状，用于跨步积累未能被量化表示吸收的位移 | Adam类更新，再按 `delta_decay` 向恢复权重收缩 |

`ParameterRole` 分别管理普通未量化参数、量化表示的连续参数、整张反量化权重或STE缓冲。整数码不参与普通梯度下降。未量化参数可以不加入优化器，却仍需传播梯度；没有更新某类参数，不等于没有对应前后向计算。

## 2. 格式、尺度与训练存储不是同一个位宽

设一块权重覆盖 $g_o\times g_i$ 个元素，码本数为 $M$，每本索引占 $b_c$ bit、共有 $K=2^{b_c}$ 个中心。$N_o$、$N_i$ 分别是输出、输入组数，即 `out_features / g_o`、`in_features / g_i`；块索引 $a,b$ 分别指向输出组、输入组。代码布局为

$$
C\in\mathbb R^{M\times K\times g_o\times g_i},
\qquad z\in\{0,\ldots,K-1\}^{N_o\times N_i\times M}.
$$

`_dequantize_weight` 通过带码本偏移的 `embedding_bag(mode="sum")` 累加中心，再乘块尺度、换轴还原为输出行×输入列：

$$
W_{ab}=s_{ab}\sum_{m=1}^{M}C_{m,z_{abm}}.
$$

`scale_nbits=0` 仍有每输出组一个尺度，沿输入组广播，并非没有尺度。其他配置可有每块尺度或尺度码本。所选直接中心/尺度路径的具体浮点dtype来自加载与训练配置；`codebook_value_nbits>=16` 只是返回原中心参数，不自动将其转为FP16。低位中心值与低位尺度还有重新聚类和STE分支，本页只核其接口，不把它们当成所选常规路径的默认行为。（`aq.py:QuantizedWeight`、`get_codebooks/get_scales/forward`。）

逻辑索引率为 $Mb_c/(g_og_i)$，训练中整数张量可以用 `int32` 或指定dtype保存。`IntCodes` 为FSDP把整数存储的**原始字节**装进默认float64容器，必要时在末尾填充，读回时按原dtype重解释；它不是将码值做数值浮点转换，也没有把任意 $b_c$ 位码重新紧密打包。`estimate_nbits_per_parameter` 是表示预算估计，不等于优化器、稠密缓冲或序列化文件的实际字节数。（`utils.py:IntCodes`、`aq.py:get_codes/wrap_codes_for_fsdp_`。）

## 3. 一次反向如何连接到 P 与 V

`StraightThroughAdamW.step` 的顺序固定为传播梯度、连续优化、离散选码、写回恢复权重。以非分片路径或完整STE缓冲路径说明：

1. 学生在 $W^k$ 上得到梯度 $G^k$，梯度累积完成后才调用 `step`。
2. `_propagate_grads_to_optimized_parameters` 把 $G^k$ 传给可选STE缓冲，并对当前 `QuantizedWeight.forward()` 调用 `backward(G^k)`，得到中心/尺度的梯度。两类梯度来自同一次学生计算。
3. `ConfigurableAdamW.step` 更新所启用的普通参数、中心/尺度及连续目标。即使变量分组依次处理，也没有在新中心上重新运行模型或重算损失梯度。
4. `_optimize_quantized_weights` 将已更新的整张权重或STE缓冲作为 `reference_weight`，以更新后的中心/尺度搜索新整数码。
5. `_update_dequantized_weights` 将新表示的解码结果写回学生权重，供下一次前向。

因此实际选码目标是**绝对权重**，不是“Adam输出减旧权重”的位移；论文算法3的这一接口歧义，在本快照所选链上得到澄清。选码时的码本已被连续更新，不能把整步描述为固定旧码本的投影，也不能直接套用“先精确P、再以新梯度精确V”的损失单调性证明。

`delta_decay=d` 控制离散更新后保留多少连续目标：

| 值 | 实际路径 |
| --- | --- |
| $d=1$ | 不建STE缓冲；直接更新前向恢复权重作为临时目标，选码后以新解码权重覆盖 |
| $d=0$ | 保留独立STE缓冲；其未被量化吸收的差异可跨步积累 |
| $0<d<1$ | 选码后令 $B\leftarrow(1-d)B+dW^{k+1}$，部分保留差异 |

前向最终仍用 $W^{k+1}$，而非混合后的 $B$。`delta_decay` 只控制这份全权重缓冲，不替代各层内部可能使用的STE规则。

优化器的 LAMB 比例、Adam矩状态和 `code_trust_ratio` 也应分开：前两者产生连续目标，最后一个按选码变化阈值决定保留哪些组，并非变化量的硬上界，具体见 §5。该快照可分别指定更新计算dtype、动量存储dtype与CPU卸载，不能因量化码低位就推断训练状态都低位。（`configurable_adam.py:step/_inner_adam_step_and_update_statistics`。）

## 4. V 步的距离、候选与更新单元

PV 调用 `beam_search_update_codes_` 时没有传 `XTX`，因此进入 `beam_search_l2`，优化权重欧氏误差，而非原AQLM校准的激活二次型。整网任务信息经 $G^k$ 进入目标权重，不能说代码直接对KL做穷举，也不能误写成又积累一次GPTQ Hessian。

对固定且非零的块尺度，搜索将参考块除以 $s_{ab}$，比较未缩放的加性中心和；固定正因子 $s_{ab}^2$ 不改变这个块的候选排序。不同块的优先级与实际变化范数则在恢复权重尺度上计算。（`aq.py:beam_search_update_codes_`、`beam_search_l2.py:beam_search_optimal_codes`。）

单码本、beam size 1、无随机舍入且不强制改变时，`_greedy_find_best_codes` 对未缩放目标 $\widetilde B_{ab}=B_{ab}/s_{ab}$ 分块计算 $\|C_j\|^2-2\widetilde B_{ab}^\top C_j$，省略对候选相同的 $\|\widetilde B_{ab}\|^2$，在整张表中选最近中心。多码本路径从旧组合出发，依次替换某一本的码、保留有限个候选；优化器用 `random.Random(None)` 打乱码本顺序。它不是所有 $K^M$ 组合的全局搜索，该随机源也没有接入入口的 `args.seed`。

对所选单码本/两码本路径，在精确算术、关闭强制和随机舍入时，每次扩展含当前候选自己的旧码，所以至少可保留一个不更差的候选；有限beam因此可有局部L2不增性质，但不保证全局最优，更不保证整网KL下降。浮点算分、中心先更新及完整训练步需要分别判断。三本以上且beam大于1时，中间候选重排后的 `residue` 没有同样按来源索引重排，本页不将上述判断扩到这类配置。

`force_update` 在最优候选索引与旧码相同时改用第二候选。代码创建的 `direction` 没有参与后续评分；不能仅凭help文字保证沿梯度方向下降。索引改变也未必使解码值改变，例如重合中心或不同加性组合可得到同一个向量。随机舍入选项按候选残差距离计算概率，也不保证无偏；当前根入口没有传入其温度，默认关闭。

## 5. 更新比例与 trust ratio 的实际边界

令一张权重矩阵共有 $N=N_oN_i$ 个码组，`max_code_change_per_step=f` 先给出候选组数

$$
k=\lceil fN\rceil.
$$

它限制可以搜索的码组，不是保证这么多组都会改变；一个组的任一码本索引改变即计为该组改变。向上取整也意味着小矩阵的实际候选比例可略高于 $f$。默认按参考权重与当前解码值之间的块误差降序选组。若 $f=1$ 而只设trust ratio，所有组仍先完成搜索，随后才截断，不能据trust ratio推断搜索工作量已经降低。

这里“当前解码值”的时点很关键。记连续更新后但未选新码的权重为

$$
\overline W=D(C^{k+1},s^{k+1},z^k),
\qquad W'=D(C^{k+1},s^{k+1},z').
$$

排序比较 $\|B-\overline W\|$，变化量比较 $\|W'-\overline W\|$，分母也是 $\|\overline W\|$。因此这项限制不包括中心/尺度从 $W^k$ 到 $\overline W$ 的变化；在STE模式下，排序差异还包含跨步积累的缓冲残差，不只是当步Adam位移。

代码把各候选组实际变化的平方范数按既定顺序累加，开方得前缀范数 $c_j$；阈值为 $T=\rho\|\overline W\|$。随后以 `1 + searchsorted(c, T, side="left")` 截取前缀。**这会保留第一个达到或超过阈值的组**，不是仅保留所有严格满足阈值的组。

整理者算例：三个互不重叠组的变化范数依次为0.03、0.04、0.10，前缀范数为0.03、0.05、约0.1118；阈值0.08时，前两个已满足限制，代码仍保留第三个，使结果超过阈值。因此越界不只限于“至少让第一个组变化”的情形。若阈值超过所有前缀，切片自然保留全部；等于某前缀时则保留到该前缀。

本快照保留跨阈值的组；论文附录 K 则在末块中回滚至满足约束，但允许至少一个权重变化。要限制**整步**权重变化，还须核对连续参数与选码两部分，见[公共训练页的分阶段变化](../fundamentals/quantization/post-training-and-quantization-aware-training.md#3.2%20%E5%87%8F%E5%B0%91%E6%9B%B4%E6%96%B0%E5%9D%90%E6%A0%87%E4%B8%BA%E4%BB%80%E4%B9%88%E5%8F%AF%E8%83%BD%E5%85%81%E8%AE%B8%E6%9B%B4%E5%A4%A7%E6%AD%A5%E9%95%BF)。

## 6. FSDP、缓冲与保存接口

FSDP 分片学生的稠密参数；量化表示则由 `split_quantized_weights_between_ranks` 按估计表示大小做贪心分配，把一整张 `QuantizedWeight` 交给一个owner，其余rank持占位对象。两种分配粒度不同。优化器按一致名字顺序收集梯度片段到owner，在那里更新表示，再将新解码矩阵按原分片大小发送回各rank。（`pv_utils.py`；`pv_optimizer.py:_aggregate_tensors_by_name/_update_dequantized_weights`。）

有STE缓冲时，owner持完整目标与聚合梯度。`delta_decay=1` 的多rank分支则不同：只有owner把自己的反量化参数片段加入该类优化器组，非STE分支没有把聚合梯度装成完整可优化目标，而后再聚合参数片段。静态读取不足以声称它与完整矩阵的单rank更新等价。

训练快照保存学生的rank-local state、优化器状态和metadata；优化器额外保存量化对象与STE缓冲。`save_pv_model` 则按名字导出量化对象，rank0另存未量化参数，并排除对应的稠密恢复权重。这两类产物分别服务继续训练与后续格式转换，不是同一个部署包；[条件转换接口](aqlm-calibration-export-and-inference.md#4.%20%E4%BF%9D%E5%AD%98%E6%A0%BC%E5%BC%8F%E3%80%81PV%20%E8%BD%AC%E6%8D%A2%E4%B8%8E%E5%AF%BC%E5%87%BA%E6%9D%A1%E4%BB%B6)说明 legacy 入口的分支与元数据要求，未验证跨world size恢复或转换后推理。

## 7. 训练数据与 KL 的实际接口

`prepare_training_dataset` 对已含 `input_ids` 的材料直接使用，入口只检查首条长度。原始文本先按较长文本中的换行处分割、逐文档分词，再用 `group_texts` 拼接当前 `Dataset.map` 批次并截成定长块；**每个map批次**的不足整块尾部被丢弃，不是全语料仅丢一次尾部。该辅助函数没有另外插入EOS，是否已有特殊token取决于分词器。训练设置 `add_labels=False`，因为监督来自教师分布。

`compute_loss_on_batch` 使用相同输入上的教师与学生 log-softmax，实际目标是

$$
\frac1{BL}\sum_{b=1}^{B}\sum_{t=1}^{L}
D_{\mathrm{KL}}\bigl(p_T(\cdot\mid x_{b,\leq t})\,\|\,p_S(\cdot\mid x_{b,\leq t})\bigr).
$$

此式中的 $B$ 是当前损失 microbatch 的序列数，$L$ 是每条序列的长度；$p_T$、$p_S$ 分别是教师、学生的词表概率。§3 的 $B$ 表示 STE 缓冲张量，入口的全局 `batch_size` 则统计所有 rank 合计的序列数。

普通分支把序列位置展平成batch再做 `batchmean`；分块分支对各位置KL求均值。这里没有用next-token硬标签做shift，也没有在损失处额外按mask筛位置；模型接收attention mask不等于损失mask。定长完整块与带padding的外部预分词数据应分别判断。评测PPL另在 `evaluate_perplexity` 里对logits/标签移位，以真实后继token的交叉熵计算。

`loss_tokens_per_chunk` 路径分块构造logits，但 `memory_efficient_loss.py:maybe_checkpoint` 实际在 `checkpoint_enabled=True` 时直接计算、False时调用checkpoint，与参数名相反。根入口传 `checkpoint_last_chunk=False`，通常造成前面块直接计算、最后块checkpoint。

入口的 `batch_size` 是所有rank合计的**序列数**，累积步数为 `batch_size / (world_size * microbatch_size)`。batch128、序列8192时才对应名义上的 $2^{20}$ token；序列4096时为 $2^{19}$。实际训练token量还取决于数据尾部、采样及循环条件。

## 8. 默认值、README 示例与论文分别登记

| 项目 | 此快照根入口默认 | 此快照 README 的 PV 示例 | 论文 v2 附录 K 所述 |
| --- | --- | --- | --- |
| 连续学习率 / 选码目标学习率 | $10^{-5}$ / $10^{-2}$ | $3\times10^{-4}$ / $10^{-2}$ | 子空间PV：$3\times10^{-4}$ / $3\times10^{-3}$ |
| 全权重STE缓冲 | `delta_decay=0` | `delta_decay=0` | 分别比较有无STE及混合效果 |
| 候选组比例 / 选码trust ratio | 0.01 / None | 0.01 / 未设置 | 按约0.01相对变化选择更新数量 |
| beam size | 1 | 5 | AQLM消融为8 |

README 示例显式开启三类参数训练、LAMB和debias；三个训练开关默认均未开启，默认参数并不是可以直接重现论文的完整配方。`code_selection_temperature` 虽被CLI解析，`create_pv_optimizer` 没将它转交构造器，实际仍用默认0；不能因存在参数说明就认为该入口启用了随机组选择。

源码还保留与执行协议有关的边界：训练循环 `batch_index <= microbatches_since_epoch_start` 在初始计数0时跳过首批；PPL分块条件使用严格的 `start + seqlen < total_length`，整除时也排除末个完整块。保存接口没有承诺未完成累积梯度与随机状态的逐位恢复。

## 来源身份与后续阅读

- Vahe1994/AQLM（`https://github.com/Vahe1994/AQLM/tree/e79a896ed6656fe4ed06193d42d004e7d0bbdbb2`），固定commit `e79a896ed6656fe4ed06193d42d004e7d0bbdbb2`；本页各节列出文件与函数，仓库根入口和 `src/finetune.py` 的原始逐块微调不是同一条流程。
- PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression（`https://arxiv.org/abs/2405.14852v2`），arXiv:2405.14852v2；理论、论文配置和作者报告由[方法页](../methods/pv-tuning.md)承接。
- [AQLM](../methods/aqlm.md)：加性表示、原始激活感知联合编码及质量/推理证据。
- [校准采样分布](../theory/calibration-and-range-selection.md#5.1%20%E5%90%8C%E4%B8%80%E8%AF%AD%E6%96%99%E4%B8%8D%E7%AD%89%E4%BA%8E%E5%90%8C%E4%B8%80%E9%87%87%E6%A0%B7%E5%88%86%E5%B8%83)：分词、map批次与尾部规则怎样改变实际数据覆盖。
