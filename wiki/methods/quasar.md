---
title: QUASAR：训练中的损失感知重构
type: method
tags:
  - qat
  - optimization
  - sensitivity
  - weight-quantization
sources:
  - raw/papers/2026-09-21/quasar
updated: 2026-10-11
---

# QUASAR：训练中的损失感知重构

QUASAR 改变 QAT 每次前向中的“潜在浮点权重 → 离散码 → 恢复权重”映射。它保持原来的 STE 反向和训练目标，但在少量裁剪候选中寻找编码，再按在线敏感度拟合恢复尺度与偏移。核心区别是编码范围不必等于最终恢复范围，且随着潜在权重变化持续重做重构；不是只在训练前执行一次更好的 PTQ。

本页依据 `arXiv:2608.13966v1` 的正文、算法1、理论与必要附录 A–D。已核读主要公式、证明链及格式/实验条件，未取得匹配作者实现，未运行训练、模型或 GPU。理解其增量可沿[代理梯度](../fundamentals/quantization/post-training-and-quantization-aware-training.md)、[曲率代理](../theory/curvature-weighted-quantization-error.md)、[编码与恢复参数](../theory/calibration-and-range-selection.md#2.1%20%E9%80%89%E7%A0%81%E8%8C%83%E5%9B%B4%E5%92%8C%E6%81%A2%E5%A4%8D%E5%8F%82%E6%95%B0%E5%8F%AF%E4%BB%A5%E5%88%86%E5%BC%80%E4%BC%98%E5%8C%96)阅读。

## 1. 前向和更新位于不同权重点

设潜在参数为 $w_t$，恢复参数为 $r_t$。恒等 STE 将在 $r_t$ 计算的梯度施加到 $w_t$，与浮点目标的真实梯度存在差异

$$e_t=\nabla\mathcal L(r_t)-\nabla\mathcal L(w_t),\qquad \Delta_t=r_t-w_t.$$

任务损失局部展开中，只有一阶项可忽略时，才有 $\mathcal L(r_t)-\mathcal L(w_t)\approx\tfrac12\Delta_t^\mathsf TH_t\Delta_t$。训练中并不总在驻点；实际任务 Hessian 也不必半正定。QUASAR 使用非负在线代理

$$\widehat S_t(r)=\sum_i h_{t,i}(r_i-w_{t,i})^2,$$

其中 $h$ 来自平方梯度的指数移动平均，AdamW 下复用二阶状态 $v$，SGD 对照则另外维护。它是实用敏感度代理，不是精确任务 Hessian；mini-batch 聚合梯度的平方与逐样本 Fisher 也不是一般恒等式，见[统计对象](../theory/curvature-weighted-quantization-error.md#1.%20%E4%BB%BB%E5%8A%A1%20Hessian%E3%80%81%E6%A8%A1%E5%9E%8B%20Fisher%20%E4%B8%8E%E7%BB%8F%E9%AA%8C%E6%A2%AF%E5%BA%A6%E5%A4%96%E7%A7%AF)。

## 2. 每组、每步如何选择恢复权重

整数实验沿权重输入维每128个数分组，一组潜在权重 $w\in\mathbb R^g$ 共用仿射恢复参数。论文使用 $r_i=sq_i+z$；本页记偏移为 $o$，避免与整数 zero-point 混同。

1. 由当前权重极值计算中心 $c=(\min w+\max w)/2$ 与半宽 $\rho=(\max w-\min w)/2$。
2. 遍历范围因子 $f\in\mathcal F$，在 $[c-f\rho,c+f\rho]$ 上舍入、裁剪得到合法整数 $q_f\in\{0,\ldots,2^b-1\}^g$。主实验 $f=0.30,0.35,\ldots,1.00$，共15个候选。
3. **固定这些码**，最小化 $\sum_i h_i(sq_{f,i}+o-w_i)^2$，按[加权仿射拟合](../theory/calibration-and-range-selection.md#2.1%20%E9%80%89%E7%A0%81%E8%8C%83%E5%9B%B4%E5%92%8C%E6%81%A2%E5%A4%8D%E5%8F%82%E6%95%B0%E5%8F%AF%E4%BB%A5%E5%88%86%E5%BC%80%E4%BC%98%E5%8C%96)取得 $s_f,o_f$。
4. 用拟合后的恢复值计算 $\widehat S_t(f)$，选择最小者；前向执行 $r=s_{f^\star}q_{f^\star}+o_{f^\star}$。
5. 原训练目标反传，通过同一 STE 更新潜在权重，后续前向重算候选与拟合。

这里的最优性仅在**给定码的仿射参数**和**有限裁剪候选集合**内成立，不是对所有离散编码的全局求解。两个权重若共享码，拟合尺度不能单独拆开它们；搜索范围改变谁共享码，拟合改变每个码恢复成什么值。拟合完成后不要求再用新尺度对潜在权重重新选码，否则会改变被比较的候选。论文未完整给出零总权重、常量码组等退化数值分支，需作者代码补核。

只取 $f=1$ 且均匀加权时，算法注释称其退化为仅优化恢复参数的 Denoising QAT；论文自身的 Standard QAT 基线则直接复用 min–max 编码参数。这是所列基线定义，不能扩写成所有 QAT 都只用极值。与[LC-QAT](lc-qat.md)相比，QUASAR 不引入仿射向量网格或 DGE，而是在既定标量部署表示中优化重构。

## 3. 理论连接为何需要额外条件

正文定理1分析的是恒等 STE–SGD：$w_{t+1}=w_t-\eta G_t$，其中 $G_t=\nabla\mathcal L(r_t)+\xi_t$。目标须二次连续可微、$L$-smooth且有有限下界；重构在抽取当前 minibatch 之前由历史确定；噪声条件为条件均值0及 $\mathbb E_t\|\xi_t\|^2\le M\|\nabla\mathcal L(r_t)\|^2+\sigma^2$。另外要求对所有所考虑候选及连线点 $u\in[0,1]$，同一个有限常数 $C$ 满足

$$\left\|\nabla^2\mathcal L(w_t+u\Delta_t)\Delta_t\right\|^2\le C\widehat S_t.$$

梯度差的线段积分与 Jensen 不等式于是给出 $\|e_t\|^2\le C\widehat S_t$，具体推导见[公共代理梯度页](../fundamentals/quantization/post-training-and-quantization-aware-training.md#3.4%20%E9%87%8D%E6%9E%84%E8%AF%AF%E5%B7%AE%E6%80%8E%E6%A0%B7%E8%BF%9B%E5%85%A5%20STE%20%E6%9B%B4%E6%96%B0%E7%9A%84%E6%9D%A1%E4%BB%B6%E7%95%8C)。Adam 的二阶状态本身不保证此条件；若所有敏感度有一致正下界 $h_{\min}$，可取 $C=L^2/h_{\min}$，但源算法未证明该下界始终成立。

当 $0<\eta<1/[L(1+M)]$，定理的联合驻点界为

$$
\frac1T\sum_{t<T}\mathbb E\left[\|\nabla\mathcal L(w_t)\|^2+
(1-L\eta(1+M))\|\nabla\mathcal L(r_t)\|^2\right]
\le\frac{2(\mathcal L(w_0)-\mathcal L^\star)}{\eta T}
+L\eta\sigma^2+\frac CT\sum_{t<T}\mathbb E\widehat S_t.
$$

重构误差是这里显式受当前重构映射控制的一项；更小的代理误差收紧该条件界，不保证不同训练轨迹的真实损失逐步排序。它没有证明直接最小化代理等于全局最小化低比特任务损失。

要约束最终部署参数 $r_T$ 的函数值，推论1还加入**全局 PL 条件** $\|\nabla\mathcal L(x)\|^2\ge2\mu(\mathcal L(x)-\mathcal L^\star)$ 与 $\eta<1/\mu$。附录 A 先迭代潜在损失的递推，再用 smoothness 与终点梯度差得到

$$\mathcal L(r_T)-\mathcal L^\star\le\frac{2L}{\mu}(\mathcal L(w_T)-\mathcal L^\star)+\frac C\mu\widehat S_T.$$

因此终点重构也需要控制，不能只看平均训练误差。主实验采用 AdamW，该 SGD/PL 定理不自动覆盖真实 LLM 训练；附录补 plain SGD 的1024步、batch32对照，是支持机制的实验，不是对所有假设的认证。

损失差的局部二阶项是 $\Delta^\mathsf TH\Delta$，线性化梯度差平方则是 $\Delta^\mathsf TH^2\Delta$。论文用上述常数条件连接两者；不能直接把平方梯度 EMA 再平方就称为更准确 Hessian。附录 A.3 的 $h$ 与 $h^2$ 对照报告前者更稳定，但这仍属于所列训练设置。

## 4. 相同训练思想怎样适配不同格式

| 格式 | 选码和恢复参数 | 需要保留的条件 |
| --- | --- | --- |
| 仿射整数 | 裁剪候选生成整数码；固定码拟合 $s,o$ | $o$ 是实数加性偏移，若后端只允许整数零点 $z_{\rm int}$，须另满足 $o=-s z_{\rm int}$；“同格式”不证明任意加载器能直接消费 |
| NVFP4 | E2M1码；每16权重组FP8恢复尺度；张量FP32尺度由absmax确定并固定 | 固定码下也需在可表示FP8尺度上选择；浮点闭式解不能直接当作已编码尺度。论文报告真实检查点评测走vLLM W4A4，训练使用模拟量化 |
| Rubin INT3 LUT（附录B末） | 每512权重存3-bit索引与8个E4M3中心；固定索引按敏感度加权均值并按目标格式舍入 | 已使用中心需正权重和；未用项/零权重及打包接口需实现另核。本文只整理作者格式描述，不认证硬件支持或实际执行 |

LUT主体预算为 $(512\cdot3+8\cdot8)/512=3.125$ bit/weight，未计其他元数据与对齐。训练后舍弃敏感度估计与候选搜索，保留码及对应恢复参数；这解释作者“无推理新增步骤”的范围。未取得匹配内核/导出器时，不能据此宣称本项目已验证兼容性或推理速度。

## 5. 证据与实际限制

整数实验为权重量化，激活、norm、embedding、LM head保留BF16；QAD冻结未量化模块，任务适配则训练全部权重。QAD使用Open-PerfectBlend的教师响应，4096步、序列长4096、约246M token与forward KL；两个模型为Qwen3-4B-Thinking-2507和Llama-3.1-8B-Instruct。同模型/位宽各QAT方法共享数据、目标、日程与预算，学习率取所有方法均稳定的最大候选。对齐指标使用128个留出样本，见[同前缀分布与任务评测](../implementation/model-quality-evaluation.md)。以下均为作者报告。

| 核查结果 | 能说明什么 |
| --- | --- |
| Qwen INT2：QUASAR KL0.126、八任务均分53.2；最低KL基线Denoising0.179，最高任务基线BitDistiller48.9 | KL下降约29.6%，任务均分增4.3个百分点；“最强基线”需按指标分别选择 |
| Llama INT2：QUASAR KL0.105/均分59.5；BitDistiller0.151/56.0 | KL下降约30.5%、均分增3.5个百分点，仍低于浮点教师69.4 |
| Qwen INT3：QUASAR KL0.054/均分62.7；LSQ KL0.070/均分63.4 | 最低KL不保证最高任务均分；不能用对齐排序替代任务排序 |
| Qwen INT2消融：每步完整重构0.112，Standard0.175，初始化only0.172，仅范围搜索0.114，仅Adam加权0.164 | 每步重算和范围搜索在该设置有作用；消融数值不同于主表0.126/0.186，不擅自视为同一运行 |
| Qwen3-4B-Base数学适配：INT2 QUASAR PPL1.71/五任务均分29.6，最佳QAT均分18.7，FP-SFT1.59/46.1 | 目标位宽训练比所列“先浮点SFT再PTQ”更能保住新增能力；仍未恢复浮点质量 |
| NVFP4 Qwen3-8B：KL0.016→0.011、均分67.1→67.6；Qwen3.5-9B：0.009→0.006、69.4→70.2 | 表内均分提升0.5与0.8个百分点；正文“0.5–2.0”未由这两行支持，保留差异 |

数学适配用OpenMathReasoning、序列长18432、交叉熵，评价为温度0.6的avg@32；它与冻结未量化模块的QAD协议不同。附录沿1000条FP-SFT轨迹喂相同token计算位置KL，是同前缀诊断，不能等同于量化模型自由生成的状态分布。128 WildChat prompt的LLM-judge偏好也不是人工认证或广泛安全性评估。

训练成本表在8张H100、Qwen INT3同配置下为2.930→2.972秒/步，约1.4%增加；重构自身0.294→0.335秒增加约13.9%。不能把总步时占比低写成搜索本身没有成本，更不能由这张训练表推断部署吞吐。

仍需实现材料核对EMA读取时点、退化拟合分支、尺度/偏移的存储精度、格式投影和导出/加载。

## 来源身份

- QUASAR: Lowering the Loss Floor of Quantization-Aware Training with Loss-Aware Reconstruction（`https://arxiv.org/abs/2608.13966v1`），`arXiv:2608.13966v1`；§3–4、算法1、§5定理1/推论1及附录A证明，§6和必要附录B–D的格式、数据、消融与诊断。
