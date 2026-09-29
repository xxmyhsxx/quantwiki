---
title: 激活引导补偿与正交残差：从误差分解到 L2-SmoothRot
type: method
tags:
  - ptq
  - activation-quantization
  - reconstruction
  - rotation
sources:
  - raw/papers/2026-09-23/activation-guided-compensation/paper.pdf
updated: 2026-09-28
---

# 激活引导补偿与正交残差：从误差分解到 L2-SmoothRot

权重量化可以通过修改其他权重补偿误差，但激活量化改变了输入：允许权重任意变化，也未必能恢复原输出。这篇论文以线性层最小二乘投影区分可补偿误差与正交残差，再用残差上界解释随机符号 Hadamard、符号采样和通道缩放，形成无需反向传播的 L2-SmoothRot 配置。

本页依据 arXiv:2609.21450v1，覆盖正文、附录 A/B 的推导与附录 C 的结果。理论是固定校准输入上的局部分析；实验是 W4A4KV4 模拟量化，不能由此推出真实整数内核加速。已有 [二阶重构](../theory/layer-reconstruction-second-order-compensation.md) 解释共享输入与非对称输入的补偿，本页进一步回答补偿的不可达部分是什么。

## 1. 固定输入、变换和量化器后，权重还有哪些自由度

采用行 token 约定，$X\in\mathbb R^{T\times d}$，$W\in\mathbb R^{d'\times d}$。可逆变换 $P$ 产生

$$Z=XP,\qquad V=WP^{-\top},\qquad XW^\top=ZV^\top.$$

定义反量化后的激活 $\widetilde Z=Z+A$，其中 $A$ 为激活量化误差；$\widetilde V$ 是可行低比特权重。目标为

$$J(P,\widetilde V)=\frac1T\|\widetilde Z\widetilde V^\top-ZV^\top\|_F^2.$$

$T$ 是校准 token 数，并非推理 batch 大小。改变 $P$ 会同时改变输入、权重、激活误差以及可用量化网格，不能把它当作只影响激活最大值的旋钮。（§3.1，式 1–2。）

理论的激活量化采用无裁剪、动态逐 token 对称均匀网格：$q_A=2^{b_A-1}-1$，$\Delta_t=\|Z_{t,:}\|_\infty/q_A$，$\widetilde Z_{tj}=\Delta_t\operatorname{round}(Z_{tj}/\Delta_t)$。权重量化不限于最近舍入，可以使用 GPTQ 类优化。全精度图的等价变换及融合条件见 [对角缩放](../theory/diagonal-scaling-equivalent-transform.md) 和 [正交旋转](../theory/orthogonal-rotation-and-hadamard-quantization.md)。

## 2. 精确投影分解：不能补偿的是哪一部分

令 $\Pi=\widetilde Z\widetilde Z^\dagger\in\mathbb R^{T\times T}$，其中 $\dagger$ 是 Moore–Penrose 伪逆。$\Pi$ 投影到量化激活的列空间；这是校准样本轴上的空间，不是输入通道轴上的旋转。

先展开，再把激活误差输出投影到两个正交子空间：

$$\widetilde Z\widetilde V^\top-ZV^\top
=\widetilde Z(\widetilde V-V)^\top+AV^\top,$$

$$AV^\top=\Pi AV^\top+(I-\Pi)AV^\top.$$

取连续目标

$$(V^\star)^\top=V^\top-\widetilde Z^\dagger AV^\top,$$

则论文定理 1、附录 B.1 给出

$$J=\underbrace{\frac1T\|\widetilde Z(\widetilde V-V^\star)^\top\|_F^2}_{J_{\mathrm{AGWC}}}
+\underbrace{\frac1T\|(I-\Pi)AV^\top\|_F^2}_{J_\perp}.$$

交叉项为零，因为第一项输出属于 $\operatorname{col}(\widetilde Z)$，第二项与其正交。伪逆使恒等式不要求满列秩；但低秩时连续最优权重可能不唯一。

固定 $X,P,Q_A$ 后，调整权重只能改变 $J_{\mathrm{AGWC}}$。若允许无约束实数权重，取 $\widetilde V=V^\star$ 即达到误差下限 $J_\perp$；有限量化集合未必包含它，还会留下第一项。改变变换会改变两项，故“旋转只改残差、补偿只改权重误差”不是一般的独立优化结论。

**教学例子。** 设单输入通道、两个校准位置，$Z=(1,1)^\top$、$V=1$、$\widetilde Z=(1,0)^\top$。这是用于展示输入失配的代数例子，不声称由本文无裁剪单通道量化器产生。任意标量权重只能输出 $(v,0)^\top$，第二个位置的目标 1 无法恢复；最优 $v=1$ 时平均误差为 $1/2$。增大权重自由度的求解精度无法消除这个正交残差。

当 $A=0$，$V^\star=V$，恢复共享输入重构。$A\ne0$ 时，仅把 GPTQ 的共同输入换成量化输入，却仍以该输入乘原权重作为教师，缺少针对浮点目标的失配补偿。论文 §4.2 将 GPTAQ 放在非对称重构框架中，但目标只差常数不代表不同顺序算法产生相同整数权重。QEP/CoreQ 的关系在本文属于作者定位，本页未独立研读其原始实现。

## 3. 从恒等式到上界：持续离群通道只是分析模型

论文 §3.2 把 $X=X^{co}+X^{reg}$：少数持续离群通道在所有 token 上取共享水平 $L_k$，其余变化全部归入 $X^{reg}$。这不要求真实离群通道严格常数；被减去的水平与实际 token 的差也留在 regular 部分。

取 $P=\Lambda^{-1}DH^\top$，$\Lambda=\operatorname{diag}(\lambda_k)>0$，$D$ 为符号矩阵，$H$ 为归一化 Hadamard。对应 $V=W\Lambda DH^\top$。定义

$$J_{co}=\sum_t\|(X^{co})_{t,:}\Lambda^{-1}DH^\top\|_\infty^2,$$

$$J_{reg}=\|W\Lambda\|_2^2\sum_t\|(X^{reg})_{t,:}\Lambda^{-1}DH^\top\|_\infty^2.$$

这里矩阵 $\|\cdot\|_2$ 是谱范数。无裁剪舍入使 $|A_{tj}|\le\Delta_t/2$；再依次用投影不增范数、矩阵乘法范数界和三角不等式，得到式 5：

$$J_\perp\le\frac{d}{4q_A^2T}
\left(\sqrt{\|W\Lambda\|_2^2J_{co}}+\sqrt{J_{reg}}\right)^2.$$

因此 $J_{co}$、$J_{reg}$ **不是精确分解中的额外误差项**，二者存在交叉上界项。优化上界也不保证实际残差或整网任务损失单调下降。

## 4. 随机符号解决多个离群通道的相干叠加

记离群通道数为 $N_{co}$，缩放后最大水平为 $L_{\Lambda,max}=\max_k|L_k|/\lambda_k$。固定 Hadamard 下，各离群通道都摊到全部坐标，但在某个输出坐标可能同号相加：幅度可到 $N_{co}L_{\Lambda,max}/\sqrt d$，平方因此依赖 $N_{co}^2$。

随机独立 $\pm1$ 符号让这些贡献成为 Rademacher 和。对固定坐标，交叉项期望为零；控制最大坐标还需要 Hoeffding 尾界及对 $d$ 个坐标的 union bound，不能把期望直接当最大值保证。（定理 2、附录 A/B.2。）

| 选择 | 对 $J_{co}$ 的界 |
| --- | --- |
| 固定 $D=I$ | $T N_{co}^2 L_{\Lambda,max}^2/d$ |
| 随机一个 $D$，概率至少 $1-\delta$ | $2T N_{co}L_{\Lambda,max}^2[\log(2d)+\log(1/\delta)]/d$ |
| 独立采样 $N_s$ 个，取集合中的最优 $J_{co}$，概率至少 $1-\delta$ | $2T N_{co}L_{\Lambda,max}^2[\log(2d)+\log(1/\delta)/N_s]/d$ |

采样数只缩小置信项，不把整个上界除以 $N_s$；它保证候选集合含有较好 $J_{co}$ 的概率，不保证按验证 PPL 选出的候选最小化 $J_{co}$。实践中的 PPL 筛选因此仍需要实验支持。

## 5. L2 缩放来自哪个目标，放松了什么

命题 1 对所有 token 和坐标取联合界，得到

$$J_{reg}\le\frac{2\log(2dT/\delta)}d
\|W\Lambda\|_2^2\|X^{reg}\Lambda^{-1}\|_F^2.$$

谱范数保留权重列之间的耦合，难以给出逐通道闭式尺度。作者以 Frobenius 范数替代谱范数，令 $a_k=\|(X^{reg})_{:,k}\|_2^2$、$b_k=\|W_{:,k}\|_2^2$，得到代理

$$R_2(\Lambda)=\left(\sum_k\lambda_k^2b_k\right)
\left(\sum_k a_k/\lambda_k^2\right)
\ge\left(\sum_k\sqrt{a_kb_k}\right)^2.$$

Cauchy–Schwarz 的等号条件为 $\lambda_k^2b_k\propto a_k/\lambda_k^2$，故非零通道上

$$\lambda_k\propto(a_k/b_k)^{1/4}
=\sqrt{\frac{\|(X^{reg})_{:,k}\|_2}{\|W_{:,k}\|_2}}.$$

公共正比例因子不改变该代理的乘积。零范数通道不适用直接相除的公式；论文的闭式推导没有因此给出实际实现的全部数值处理。这是 **Frobenius 代理的最优解**，并非原始量化目标或谱范数乘积的全局最优解。（§4.4、附录 B.3。）

进一步用通道最大值上界二阶矩，得到 $\lambda_k\propto\sqrt{\|X^{reg}_{:,k}\|_\infty/\|W_{:,k}\|_\infty}$。它对应 SmoothQuant 风格、平衡指数为 $1/2$ 的形式，不能推导任意可调指数都最优。实践未显式识别 $X^{reg}$，使用全精度完整激活 $X$ 的范数替代，这是另一层近似。（§4.5–4.6，式 15–17。）

## 6. L2-SmoothRot 的实际顺序与量化配置

论文 §4.6–5 的操作可还原为：

1. 在未变换的浮点模型上，用 512 个 WikiText-2 train 样本估计 FFN down 输入通道 L2 范数，与 down 权重列范数组合得到尺度。
2. 只在 FFN down 输入使用该缩放，按 SmoothRot 的放置方式融合到对应 up/down 权重。不要把 down 输入处的逆尺度擅自移动过门控非线性。
3. 在 QuaRot 框架中给 attention-output、FFN-down 在线 Hadamard 加符号；RoPE 后 Q/K 在每个头内同乘 $R=DH^\top$，因 $(QR)(KR)^\top=QK^\top$ 保持未量化点积。
4. 采样 10 个候选种子，每个种子联合生成 offline、attention-output、FFN-down、QK 四类符号。同类符号跨层共享，QK 还跨全部 Q/K 头共享。
5. 先以 RTN 在验证集筛前 3 个，再分别执行 GPTAQ，用验证 PPL 选最终候选。GPTAQ 使用 128 个 WikiText-2 train 样本，校准时已经开启激活量化，并与浮点路径目标比较，纳入传播误差。

实验 decoder Linear 为 W4A4：权重逐输出通道对称、激活逐 token 对称；K/V 为每 token、每 head 的非对称 4 bit。激活和 K/V 裁剪比例分别 0.9、0.95。**裁剪后的超界误差不再满足前述半步长界**；投影恒等式仍成立，但无裁剪残差上界不能原封不动覆盖实验量化器。局部理论也没有完整描述上游误差和整网非线性。

尺度可融合不代表在线 Hadamard 没成本；不做反向传播也不代表无需多次校准或验证前向。论文没有提供足以确立端到端加速的硬件计时证据。

## 7. 实验支持了什么，也没有支持什么

以下均为作者报告，统一使用 GPTAQ 后端比较变换，不是原论文各自最佳配置的跨论文排名。表 1 覆盖 Llama 1/2/3/3.2 与 Mistral，共八个 1B–13B 模型；PPL 使用 WikiText-2 test 与 C4 validation 子集各 64 样本。零样本平均为 PIQA、ARC-E/C、HellaSwag、WinoGrande、LAMBADA 六任务。（§5、表 1/3。）

| Llama3-8B，W4A4KV4 + GPTAQ | WT2 PPL | 六任务平均准确率 |
| --- | ---: | ---: |
| 浮点参考 | 5.94 | 73.30 |
| QuaRot | 7.45 | 65.71 |
| SpinQuant | 7.33 | 68.36 |
| L2-SmoothRot | 7.20 | 68.10 |

L2-SmoothRot 在八个模型上取得表内最低 WT2/C4 PPL，但只在五个模型上取得最高平均准确率；Llama3.2-3B 为 60.94，低于 SpinQuant 的 62.03。更好的 PPL 不能替代任务表现。

表 2 在 Llama3-8B 上从 QuaRot 7.45 出发，单独加入 signed online rotation、sign sampling、L2 scaling 后 PPL 分别为 7.30、7.40、7.32，组合为 7.20。这支持组合有用，不能证明误差贡献可加或已经实测分离 $J_{co}$ 与 $J_{reg}$。

图 1 对 100 个预计算种子，分别取 $N\in\{4,10,16,32\}$ 个候选、重复 1000 次子集抽样，比较 RTN top-1 与 top-3 后 GPTAQ 重排。它支持便宜筛选加少量精筛的取舍，不是 1000 次独立模型训练。SpinQuant 对照训练旋转 100 步、冻结 16 bit 权重而使用 4 bit 激活/KV，之后同样 GPTAQ 量化权重。

## 8. 如何复用这个分析

先固定输入、量化器和允许调整的变量，再问误差是否落在可补偿输出空间；随后区分恒等分解、上界、代理最优和实际配置四层证据。变换与权重优化应配合，不能把无裁剪局部界包装为整网性能保证。

与 [SpinQuant](spinquant.md) 的区别是如何选择变换，而非是否具有量化校准；与 [GPTQ](gptq.md) 的区别涉及输入失配及重构目标；与 [SmoothQuant](smoothquant.md) 的共同点是成对尺度，差异在统计量、放置位置、位宽及后续旋转。先核对这些条件，再比较效果。

## 来源身份

| 来源 | 固定版本 | 本页使用范围 |
| --- | --- | --- |
| [Understanding LLM Quantization through Activation-Guided Compensation and Orthogonal Residuals](https://arxiv.org/abs/2609.21450v1) | arXiv:2609.21450v1 | §3–5、定理 1/2、命题 1、附录 A–C；原件页图核对关键公式及消融表 |
