---
title: OnPTQ：量化策略轨迹、决策风险与未来分支校准
type: method
tags:
  - ptq
  - vlm
  - calibration
  - autoregressive-generation
sources:
  - raw/papers/2026-10-05/onptq/paper.pdf
updated: 2026-10-07
---

# OnPTQ：量化策略轨迹、决策风险与未来分支校准

OnPTQ 把自回归多模态 PTQ 看成一个闭环：量化模型改变 token 选择，选择又改变后续输入状态；校准参数更新后，需要重新收集当前模型的轨迹。它冻结浮点参考与预训练主体，只更新基础 PTQ 的校准参数，用“边界受损程度、同状态分布差异、短分支后果”选择并加权关键状态，同时保留原多模态上下文的重构约束。

依据 arXiv:2609.36828v1 的 §3–5、算法附录、理论推导、实验协议与必要补充分析，关键公式和表 1–2、4–5 结合原页核对。当前归档没有 OnPTQ 作者实现，本文未运行模型或训练。论文把参数更新交给 `BasePTQUpdate`，没有给出校准参数 θ 的完整逐项参数化及优化器、学习率和更新步数；不能仅凭算法框图声称已经具备可直接复现的训练配置。

## 1. 校准样本相同，前缀状态仍可能不同

设冻结的浮点模型为 D，量化模型为 $Q_\theta$，图像与指令为 $(I,x)$。量化模型生成的状态是

$$s_t=(I,x,y_{<t}),\qquad y_t\sim\pi_{Q_\theta}(\cdot\mid s_t).$$

静态校准在固定的 $d_{static}$ 下优化重构误差，部署却访问随 θ 变化的 $d_{Q_\theta}$。即使 image/instruction 样本不变，某一步选了不同 token，后续 hidden states 与条件输出都可能变化。dynamic per-token 激活量化只改变网格如何估计，并没有解决校准前缀的覆盖问题。（§3、式 (1)–(2)。）

OnPTQ 每轮使用当前 Q 的轨迹，并让 D/Q 在**同一个量化前缀**上比较。不能先分别运行两条自由生成轨迹，再把第 t 步的不同状态当成同一局部量化误差。[Decision Flips 的同前缀分析](../theory/quantization-ranking-stability.md#8.%20%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E7%9A%84%E5%88%86%E5%B8%83%E8%B7%9D%E7%A6%BB%E6%80%8E%E6%A0%B7%E5%8F%98%E6%88%90%20token%20%E7%BF%BB%E8%BD%AC)主要使用浮点或自然文本参考；这里选择的是当前量化策略访问的状态。

论文 §3 的诊断还说明，只增加 on-policy 覆盖不够：均匀重构所有量化轨迹让首次分叉延后 3.54 token，却降低约 2.5 个准确率百分点；局部 JS 与未来分支影响的 Pearson 相关仅 0.232。这些是所测 Qwen/W4A8 诊断，支持保留 context anchor 与选择性校准，不能外推所有生成模型。

## 2. Follow 与 Compare：找已翻转或边界被削弱的决策

第 k 轮从 $Q_{\theta^{(k)}}$ 的 greedy rollout 建立 trajectory bank。保存精确 token IDs，后续重放使用同一 prefix，不经文本解码再分词。D/Q 在同一 $s_t$ 输出 logits $z_D^t,z_Q^t$，定义

$$
a_t^R=\arg\max_a z_D^t(a),\qquad
\hat a_t=\arg\max_a z_Q^t(a),\qquad
a_t^C=\arg\max_{a\ne a_t^R}z_Q^t(a).
$$

aR 是浮点参考的首选，aC 是量化模型中排除 aR 后最强的竞争者，不一定是浮点模型的第二名。两者都不是任务真值。始终比较同一对 token：

$$
m_D^t=z_D^t(a_t^R)-z_D^t(a_t^C),\quad
m_Q^t=z_Q^t(a_t^R)-z_Q^t(a_t^C),\quad
e_t=[m_D^t-m_Q^t]_+.
$$

若量化首选不同，便是 mismatch，竞争者等于量化首选；若首选相同但 $e_t>0$，便是 near miss，当前还没有翻转，但领先边界被削弱。e 是方向性损伤，不是对称的全 logit MSE；增大原参考领先间隔不会贡献 e。（§4.2、式 (3)–(6)。）

同状态分布差异为 $j_t=\operatorname{JS}(\pi_D,\pi_Q)$。对概率 p/q，令 $m=(p+q)/2$，有 $\operatorname{JS}(p,q)=\tfrac12\mathrm{KL}(p\|m)+\tfrac12\mathrm{KL}(q\|m)$；它比较全词表分布，与仅检查一对 token 互补。候选集是 mismatch 与 near miss 的并集，预算选择发生在后面。

## 3. Branch：估计这次选择会把未来带到哪里

对候选状态分别强制 aR 与 aC，然后**两条分支均由当前 Q greedy 继续**，最长 H 步。D 评估两个不同未来状态上的分布分离：

$$
A_t=\frac1{H_t}\sum_{h=1}^{H_t}
\operatorname{JS}\left(\pi_D(\cdot\mid s_{t+h}^{R}),
\pi_D(\cdot\mid s_{t+h}^{C})\right).
$$

早停时采用两边共同有效的 $H_t\le H$。这一次比较故意采用不同 prefix：它回答两个行动的未来是否不同；前一节同状态 JS 才用于隔离当前位置的量化差异。D 没有生成两条未来，也没有为它们打任务奖励。（式 (8)。）

风险、选择与归一化权重为

$$
r_t=e_t(1+\lambda_bA_t)+\eta j_t,\qquad
\mathcal S=\operatorname{TopK}_{K_c}(r_t),\qquad
\tilde w_t=\frac{r_t}{\sum_{u\in\mathcal S}r_u}.
$$

TopK 在候选集内按风险选状态。A 放大已有边界损伤；j 检查当前全分布差异。未来分布不同仍不等于某一分支错误，例如两种表达可以给出同一个正确答案；A 是 label-free 后果代理，不是实际效用差。（式 (9)–(11)。）

## 4. Calibrate：保留上下文，再修复关键状态

总目标是

$$
\mathcal L=\mathcal L_{anchor}
+\gamma\sum_{t\in\mathcal S}\tilde w_t
\left(\mathcal L_{hidden}(s_t)+\mathcal L_{pair}(s_t)
+\mathcal L_{response}(s_t)\right).
$$

每轮先固定所选前缀、token pair 与风险权重，再更新 θ。各项角色与归约不能混同：

| 损失 | 比较什么 | 归约与作用 |
| --- | --- | --- |
| anchor | 原 image/instruction/prompt-prefix anchor 位置的 D/Q block hidden vectors | 向量 MSE 除 hidden width，再按位置与校准 blocks 平均；保留较广上下文 |
| hidden | 关键状态最后一个 prefix token 在各校准 block 的 hidden vector | 每 block 的 width-normalized MSE，再跨 blocks 平均；稳定该状态的表示 |
| pair | 同一 aR/aC 的 $m_Q-m_D$ | 单位阈值 Huber；恢复参考与竞争者的相对边界 |
| response | 同一状态下完整 next-token policy | forward KL：$\mathrm{KL}(\pi_D\|\pi_Q)$；保留其余 token 的概率结构 |

Huber 对 $|u|\le1$ 使用 $u^2/2$，否则使用 $|u|-1/2$。pair 损失同时惩罚边界缩小与扩大；风险 e 只用于挑选方向受损的状态，两者目的不同。j/A 的风险计算采用 JS，但 response loss 采用 KL，不能把全部 OnPTQ 目标简称“JS 蒸馏”。（式 (12)–(17)。）

这些项仍是浮点参考的无标签校准目标，无需任务标签或 reward model。它们没有消除[损失尺度与归约](../theory/layer-reconstruction-second-order-compensation.md#11.%20%E9%87%8D%E6%9E%84%E6%8D%9F%E5%A4%B1%E7%9A%84%E5%B0%BA%E5%BA%A6%E6%80%8E%E6%A0%B7%E5%BD%B1%E5%93%8D%E4%BC%98%E5%8C%96)的影响；论文没有验证将本页 MSE 一律换为分组 RMSE。也不能把“校准参数可学习”自动升级为预训练主体全参数微调。

## 5. 参数更新后必须重新访问状态

θ 更新改变生成策略，旧 bank 相对新模型已经 off-policy。算法每轮重新执行 Follow→Compare→Branch→Calibrate，D 始终冻结；使用当前 Q 产生新的前缀和分支。默认 R=2，greedy response 最多 64 token，H=4，$K_c=256$，$\lambda_b=\eta=1$，$\gamma=0.2$。（§4.3、算法 1、附录 C.1。）

刷新改变的是校准状态，并非部署时不断调用教师。到推理时只保留最终量化模型，轨迹采集、分支、教师比较和风险筛选全部是离线校准过程。

“多刷新几轮更好”没有保证。表 4 在 Qwen3-VL-8B-Instruct W4A6 上：一轮 MMMU/ScienceQA 为 45.9/87.4，两轮不刷新为 46.8/88.3，两轮刷新为 48.8/89.7，五轮刷新反而为 44.2/86.1。它支持该协议下刷新有用，也说明选择和校准可能越过合适的适配程度。

## 6. Decision–Consequence 理论解释了结构，没有认证风险分数

固定共同状态 s、共同未来 continuation policy μ 以及有界效用 G，令 $V_\mu(s,a)$ 是先取 a、后续用 μ 时的期望效用。只改变当前一步的动作分布，得到 hybrid return $J_{M\to\mu}(s)=\sum_a\pi_M(a\mid s)V_\mu(s,a)$。定理 1 为

$$
|J_{Q\to\mu}(s)-J_{D\to\mu}(s)|
\le \operatorname{TV}(\pi_Q,\pi_D)
\left(\max_aV_\mu(s,a)-\min_aV_\mu(s,a)\right).
$$

证明利用两边概率差的总和为零，先把 V 减去最大最小值的中点，再以 TV×半跨度界住各项。只有两个动作时，pair-normalized 概率差乘两动作的未来价值差就是精确分解。softmax pair margin 是这两个动作归一化后的 log odds，所以 e 可以表示参考动作概率向竞争者转移的方向。（§4.4、附录 B。）

这个界对共同 μ 下的单步策略差成立，不是 D/Q 分别自由生成完整轨迹时的累计任务损害界。实际 r 用 e/j 代理 TV，用短分支的 dense-policy JS 代理价值跨度；它们单位、范围和目的不同，不能把 r 当作定理右侧的严格上界。默认 greedy 校准与真实任务正确率也没有因该期望界就获得保证。

## 7. 原文质量结果和口径

只量化语言组件，视觉/音频 encoders 保留原精度；权重 per-channel、激活 dynamic per-token。W4A16/W8A8/W4A8/W4A6 是语言组件 W/A，不表示整个多模态图和 KV 都采用该精度。VLM matched runs 使用固定 128 个 ShareGPT4V-improved COCO Caption 样本及各模型 chat template；matched Omni-7B 使用 OmniBench 0–127 校准，1,014 个独立样本评测，不把校准 ID 纳入测试。（§5.1、附录 C.1。）

表 1 的均分为 MMMU、OCRBench、VizWiz、ScienceQA、TextVQA 五项非加权平均。代表结果如下：

| 模型与位宽 | 浮点均分 | MASQuant 均分 | OnPTQ 均分 | 解释条件 |
| --- | --- | --- | --- | --- |
| Qwen2.5-VL-7B，W4A6 | 74.5 | 63.0 | 65.8 | 仍低于浮点；Dense/可用基线转引 MASQuant，OnPTQ 另行评测 |
| Qwen3-VL-8B-Instruct，W4A6 | 76.6 | 71.6 | 73.4 | 作者用公开实现与共同本地评测流程复现比较方法 |
| Qwen3-VL-8B-Instruct，W4A8 | 76.6 | 75.7 | 75.9 | 平均改进较小；OnPTQ 的 MMMU 49.1 低于 MASQuant 51.2 |
| Qwen3-VL-8B-Instruct，W8A8 | 76.6 | 76.5 | 76.3 | 没有在每种设置上领先 |

“共同本地评测”指论文作者的实验，不是本项目运行。Qwen2.5-VL 表中的历史来源与单独评测不能当成完全配对复现；未报告任务均分的多 seed 不确定性。原文在 8 个 VLM 位宽/模型组合的 6 个平均分上领先，不能改写成所有模型、所有任务更好。

表 2 的 Omni-7B W4A8：MASQuant/OnPTQ 的 LibriSpeech WER 为 2.9/2.8、WenetSpeech 为 8.0/7.8，OmniBench 为 43.8/44.0；MMMU 却为 48.8/43.3。WER 越低越好，不宜将它与准确率混成一个未经定义的均值。

论文的 sample-wise **Flip Rate 是正确性指示量变化**：

$$\frac1N\sum_i\mathbf1[c_i^D\ne c_i^Q],\qquad c_i^D,c_i^Q\in\{0,1\}.$$

它同时包含 correct→incorrect 与 incorrect→correct，净 accuracy 变化是后者减前者。两条都错但答案 ID 不同不计为此 flip；与[Decision Flips 的 token 首选不一致](../theory/quantization-ranking-stability.md#8.1%20%E6%AF%94%E8%BE%83%E5%AF%B9%E8%B1%A1%E4%B8%8E%E4%B8%89%E4%B8%AA%E6%8C%87%E6%A0%87)不同。图 4 使用 ScienceQA-IMG N=2,017 和 MMMU N=900 的单次配对评测，不能直接拿其百分比与 teacher-forced token flip 相比。（附录 C.3。）

## 8. 消融支持与仍需保留的边界

表 4 的 Qwen3-VL W4A6 消融支持这组协议中的组件价值：完整方法 MMMU/ScienceQA 为 48.8/89.7，去 anchor 为 44.3/85.9，去 consequence 为 45.7/87.3，全部 on-policy 状态均匀校准为 45.4/86.8。它说明原 context anchor、后果筛选与刷新相互配合；不能只抽出“用生成数据就能改善”这个结论。

补充分析的模型与预算不同，需分别读：Qwen2.5-VL W4A8 的 32 个 source-disjoint prompts 产生 767 个有效状态，独立 H=16 后果评估选出 39 个 joint-critical states；选择对照用相同 128 校准 prompts，每个 prompt 的 mismatch/near-miss 中选一个，权重固定为 1，仅改变选择规则。它不是默认全局 K=256、H=4 风险加权训练的同一实验。

该对照中 OnPTQ 的 joint-critical erosion 为 1.609，低于 mismatch-only 的 2.446；但 consequence-only 的 pair-sign flip 为 43.59%，低于 OnPTQ 的 53.85%。不同代理目标会有取舍，不能说完整风险在所有固定状态指标上最优。风险迁移分析另规定每 prompt 保存两个状态；2.7% exact-prefix retention 与 9.4% prompt/timestep retention 只描述所存关键状态，未存全部风险排序。（附录 D。）

表 5 在单张 B200、Qwen2.5-VL-7B、序列长 2,048 的 prefill：batch 1 的 FP16/W4A8 延迟为 36.73/20.48 ms、峰值显存 16.22/10.15 GB；batch 8 为 271.39/192.76 ms、21.13/15.06 GB。这说明作者指定 W4A8 推理配置的收益，不是 OnPTQ 相对同格式其它校准方法的额外加速，也没有验证 decode、服务并发或其它硬件。

OnPTQ 额外成本在离线 dense/quantized 比较、两条短 greedy branch 与参数更新；论文未给出完整校准墙钟与峰值成本。短 H 可能漏掉长程或随机后果，模型范围限于 Qwen 多模态，θ 与底层更新细节也需作者实现进一步核对。

## 9. 与现有量化路线的连接

[MBQ/VLMQ/QIG 的目标加权](../research/mbq-vlmq-comparison.md)先解决模态或 token 怎样影响局部目标；OnPTQ 进一步把“哪些生成状态参与校准”和“哪些边界可能影响未来”作为选择变量。[LLM-QAT](llm-qat.md)则用生成数据训练量化学生权重，训练自由度与 OnPTQ 校准参数不同；[ConQuR](conqur.md)的在线批次前向适配激活分布，也没有使用生成决策的反事实分支。比较时需同时控制状态来源、参考、选择指标、可训练变量和最终执行图。

## 来源身份

| 来源 | 固定版本 | 本页用途 |
| --- | --- | --- |
| Calibrate the Decisions That Change the Future: On-Policy Post-Training Quantization for Multimodal Large Language Models（`https://arxiv.org/abs/2609.36828v1`） | arXiv:2609.36828v1 | §3–5、算法 1、附录 B–D：闭环目标、理论、来源与评测协议；表 1–2、4–5：质量、消融及 prefill；Limitations：短程与模型范围 |
