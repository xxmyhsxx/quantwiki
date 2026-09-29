---
title: TTS 量化评测：组件敏感性、质量判据与真实执行
type: comparison
tags:
  - ptq
  - mixed-precision
  - evaluation
  - deployment
sources:
  - raw/papers/2026-09-28/same-bit-width-different-outcomes/paper.pdf
  - raw/repositories/2026-09-28/tts-ptq-map/source/experiments/lib/rtn.py
  - raw/repositories/2026-09-28/tts-ptq-map/source/experiments/lib/torch_actq.py
  - raw/repositories/2026-09-28/tts-ptq-map/source/experiments/lib/calib_ptq.py
  - raw/repositories/2026-09-28/tts-ptq-map/source/experiments/lib/onnx_nbits.py
  - raw/repositories/2026-09-28/tts-ptq-map/source/experiments/results/bootstrap_ci.py
updated: 2026-09-29
---

# TTS 量化评测：组件敏感性、质量判据与真实执行

*Same Bit Width, Different Outcomes* 比较 13 个 TTS 系统，核心问题是：同样位宽为什么产生完全不同的语音质量和部署结果。它提供的是组件消融与评测程序，不是一个新的通用量化器。结论应落在**模型、组件、量化对象、粒度、采样步数、runtime 和质量标准**的组合上，不能将“W4 可用”作为架构属性。

本页研读完整实验正文与限制，静态核对官方快照的 RTN、激活模拟、校准 PTQ、ONNX 改写和 bootstrap 实现；未生成语音、运行真实低比特算子或复现模型成绩。通用前置见 [混合精度分配](../theory/mixed-precision-allocation.md)、[模型质量评测](model-quality-evaluation.md)和 [缩放到执行](quantized-matmul-scaling-execution.md)。

## 1. 比较单位是匹配基线的配置，不是模型名与位宽

三个核心系统为 Supertonic V3（99M，flow matching＋vocoder，默认 NFE=8）、OmniVoice（0.6B，masked-diffusion LM＋token head＋codec，默认 NFE=32）和 Kokoro（82M，feedforward）。另有八个复查模型及程序固定后加入的 Chatterbox、VoxCPM 两个 held-out 模型。

每个支持的语言使用固定 200 条 FLORES-200 devtest 句子和逐句种子；量化条件与同平台、同原生浮点精度、**同采样步数 NFE** 的原模型配对。不同系统原生精度包含 FP32、FP16、BF16，不能将所有比值统称为“相对 FP16”。校准用与评测不重叠的 64 条英文 dev 句子；OmniVoice 另用 64 条韩文。

改变 NFE 会同时改变未量化模型质量和计算量。若所有量化条件都与默认步数的浮点结果比较，就会混入采样预算效应。论文 §4.3 的配对对照显示：更多步数可以减轻 WER 惩罚，却不一定缩小自然度代理差距；不能把增加步骤后的改善全部归因于量化误差被恢复。

## 2. 位宽之外必须记录 scale 的轴和执行方式

论文的对称 RTN 使用

$$q_{\max}=2^{b-1}-1,\quad s=\frac{\max|W|}{q_{\max}},\quad
\widehat W=s\,\operatorname{clip}\left(\operatorname{round}(W/s),-q_{\max},q_{\max}\right).$$

最大值的归约范围由粒度决定：整个 tensor、一个输出通道，或输出通道内部连续 128 个权重。这里的 channel 是特征通道，不是音频声道。Linear/普通 Conv 的权重输出轴为 0，快照对 ConvTranspose 使用轴 1；盲目对所有张量沿同一轴缩放会改变实验。

主扫描量化 Linear 与卷积权重，bias、embedding、normalization 保持浮点。`rtn.py` 的模拟路径先用 FP32 计算 scale、round/clip，再把反量化权重转回原 dtype，交给浮点算子；论文对低比特表示采用 16-bit scale 的描述，不能据此把这条 fake-quant 路径认作真正打包权重和 FP16 scale 的部署实现。

激活 per-channel scale 为每个输入特征在当前调用的时间等维度共享；per-token 则通常在该时刻的特征维共享。`torch_actq.py` 是 forward pre-hook 模拟：Linear 的 per-channel 归约除末维之外的维度，卷积保留通道维；卷积 per-token 还归约 batch 和 channel。因此协议依赖张量布局与 batch，不能只根据同名 scheme 推定与目标 runtime 完全相同。该实现显式跳过 LSTM，所谓全模型 A8 也不代表所有算子都执行 INT8。

## 3. 敏感组件不是固定的架构标签

W4 per-channel 对 Supertonic 的 UTMOS 降低约 2.8，对 Kokoro 仅约 0.07。更细 group:128 明显改善若干模型，但不能自动恢复 Supertonic；8-bit per-tensor 权重也能令其 UTMOS 降低 1.642，而 WER 接近浮点。位宽更高和文本可辨识都不足以证明音质保存。

组件消融也不能简单相加。例如 OmniVoice 的 LM 与 token head 单独 W4 分别约损失 0.28、0.17 UTMOS，联合却损失 0.93；head＋codec 联合损失 0.99。局部误差经过共同生成路径会交互，组件参数量占比或独立损失不是联合收益的精确预测。

敏感性还会随量化对象变：OmniVoice 的权重量化主要损害 codec，但 per-tensor A8 最敏感的是 token head（单独损失 2.31，对比 codec 0.14）。因此不能拿权重消融表直接决定激活位宽。更一般的分配条件见 [混合精度分配](../theory/mixed-precision-allocation.md)。

两个 held-out 试验中，作者事前按已有地图预测的敏感组件都不正确；执行消融后才找到 Chatterbox 的 flow decoder 和 VoxCPM 的 local DiT。这支持“测量能找到模型特有瓶颈”，没有证明一种模型类别可以预测瓶颈，也没有证明该测试顺序优于其他顺序。（§4.7、§5。）

## 4. GPTQ 能恢复部分组件，不是统一解药

表 1 在 W4 group:128 下报告：F5-TTS vocoder 的相对 UTMOS 从 RTN 的 −0.48 改善到 GPTQ 的 −0.07，Kyutai depth transformer 从 −2.98 到 −0.08；VoxCPM local DiT 却仍为 −0.41，未接近同一质量标准。不能将摘要式“恢复到 0.1 内”外推到所有敏感组件。

快照 `calib_ptq.py` 对选定组件中的 `nn.Linear` 收集输入统计并执行逐层 GPTQ（对称量化，group=-1 或 128、block=128、damp=0.01、关闭 act-order）。同组件其他可量化模块，以及未收集到输入的 Linear，仍用同位宽与粒度的 RTN。统计在量化各层前收集，不是量化上一层后逐层重新采样整条生成链。因此“vocoder 使用 GPTQ”是组件配置名，不能理解成内部每个卷积也做了同一种 Hessian 补偿。

激活感知缩放对部分设置有效，但作者扫描的最佳强度仍可能弱于直接细化粒度；已扫描设置的最优值也带有选择偏差。比较时需固定校准集、组件覆盖和可用内存，不只比较 GPTQ/AWQ 名称。

## 5. 自然度、可辨识度和统计等价是三层判断

UTMOS 是**预测的英文自然度 MOS**；WER/CER 是 ASR 转录误差代理，分别用于英文/韩文。一个不能替代另一个。这里 WER/CER 用逐句误差率均值；不同于全语料编辑数除以参考词/字总数。插入过多时 WER 可以超过 1，不能把 1.11 自动当成录入错误。

设同一句量化与浮点评分差为 $d_i=u_i^{q}-u_i^{fp}$，对匹配句子重采样 10,000 次，得到均值差的 paired 95% bootstrap 区间。论文采用的质量保留判据为

$$[L_U,U_U]\subseteq[-0.05,0.05],\qquad U_{\Delta\mathrm{WER}}\le0.01.$$

UTMOS 用双侧等价带，WER 只限制变差的上界；按未舍入端点判断。**区间包含零并不等于已证明等价**：很宽的区间同样包含零。均值差小于 0.1 也不是通过上述更严格标准。

论文表 3 的几组例子（端点按表显示，结论按作者未舍入判定）：

| 配置 | ΔUTMOS 的 95% 区间 | ΔWER 上界 | 判据结果 |
| --- | --- | --- | --- |
| Kokoro W4 group:128 | [−0.039, −0.035] | 0.002 | 通过；模拟配置 |
| Chatterbox T3 真实 INT4 | [−0.012, 0.020] | 0.003 | 通过 |
| Supertonic INT8，排除 vocoder | [−0.056, −0.018] | 0.005 | 未通过 UTMOS 条件 |
| F5-TTS vocoder W4 group:128 GPTQ | [−0.088, −0.046] | 0.004 | 未通过 UTMOS 条件 |
| Orpheus LM 真实 INT4 | [−0.022, 0.034] | 0.012 | 未通过 WER 条件 |

总计九个配置通过作者的英文代理判据，其中三个为 4-bit；这不是九个模型都在所有场景达到感知等价。bootstrap 量化的是固定设置下句子抽样不确定性，不校正在同一评测集上挑选最优强度或粒度的偏差，也不覆盖设备差异或所有生成种子。

12 人、96 clips 的非正式听测与 UTMOS 的总体相关较高，但主要支持严重退化的判断；中等差距并不稳定，VoxCPM 的两个配置还出现听者偏好与 UTMOS 排名相反。故不能称为人工确认了所有质量保留配置。韩文证据主要是 CER，不证明韩文自然度相同。（§5、脚注 3。）

## 6. 从 fake quant 到真实低比特图，还差算子覆盖

Supertonic 的 ONNX Runtime 全 dynamic INT8 使用 per-tensor 激活 scale，在 Mac mini M4 Pro 上令 WER 从 0.028 升至 1.11，且比 FP32 慢。另一个 **x86** MatMul-only 控制把卷积留在 FP32，报告 UTMOS 4.467、WER 0.029 并通过质量判据。这说明覆盖与 scale 策略必须实测；不同平台控制不能直接当成 M4 上只改一个变量的因果实验。

真实 INT4 使用 ONNX `MatMulNBits`，block=128。为扩大 flow estimator 的覆盖，作者将满足约束的 $1\times1$ Conv 改写成 MatMul：输入 $[N,C,L]$ 转为 $[N,L,C]$，乘转置后的权重，加 bias，再转回原布局。

`onnx_nbits.py` 检查卷积 kernel=1、group=1、stride=1、padding=0、dilation=1 等条件，不是所有卷积都可无条件改写。改写后作者报告相关 flow estimator 权重覆盖由 7.6% 增至 99.5%；这是覆盖率，不是 99.5% 的模型延迟已加速，更不能等同于整个 TTS 系统全 INT4。排除 vocoder 的配置仍保留该组件浮点。

低比特格式、实际算子以及保留的浮点组件需单独记录。torchao 的 INT4 weight-only 路径还使用 BF16 operands，并不是 W4A4；见 [缩放到执行](quantized-matmul-scaling-execution.md)。

## 7. 质量达标、加速、节能必须分别回答

论文表 2：Mac mini M4 Pro，4 线程、20 句、5 次重复，安静环境下重复差异在 2% 内。RTF 是合成时间/音频时长，越低越快；生成音频时长变化也会改变分母。

| 系统与配置 | RTF | 相对本模型 FP32 时间 | 峰值 RSS（MB） | J/音频秒 |
| --- | --- | --- | --- | --- |
| Supertonic FP32，NFE=8 | 0.148 | 1.00 | 607 | 3.25 |
| Supertonic 全 dynamic INT8 | 0.310 | 2.09 | 329 | 3.21 |
| Supertonic dynamic INT8，排除 vocoder | 0.218 | 1.47 | 436 | 4.64 |
| Supertonic INT4，排除 vocoder | 0.089 | 0.60 | 360 | 未测 |
| Kokoro FP32 | 0.069 | 1.00 | 2712 | 0.99 |
| Kokoro dynamic INT8 | 0.077 | 1.12 | 2722 | 1.25 |

Supertonic INT4 的时间约少 40%、峰值 RSS 少 41%，但其 ΔUTMOS 区间 [−0.094, −0.042] 未通过质量保留判据；Kokoro 真实 INT8 质量通过却慢约 12%。这些结果不能各取一项拼成“低比特同时保质、提速、节能”。

GPU 也有反例：RTX PRO 6000、OmniVoice NFE=8、compiled、200 句，W8A8 相对 compiled FP16 的时间/能耗为 1.11/1.27，而未量化 BF16 为 0.96/0.98。真实 INT4 的 OmniVoice 约为 2.1/2.2；Orpheus 却约为 0.50/0.45，且后者仍需面对表 3 的 WER 失败。收益取决于模型与后端，不能由位宽推断。

CPU 能耗统计为整次 20 句进程的全芯片超空闲能量，包含加载；RTF 只算合成，因此不能用表中功率乘 RTF 复算能耗。GPU NVML 是设备能量、不含主机功耗，并排除加载与编译；两者不宜直接用于跨硬件能效排名。

## 8. 可迁移的评测顺序与边界

沿论文的 staged procedure，先固定同平台、同 NFE 的基线并同时看自然度与转录；再比较粒度，做组件消融，选择保护或校准敏感组件；随后单独考察激活量化，组合配置后重新评测；最后在真实 runtime 核验算子覆盖、质量、延迟、内存和能耗。

这个顺序是有证据支持的排查程序，不是已证明最优的搜索算法。对于新模型，尤其不要跳过联合配置复测、动态激活策略和真实执行环节。本文支持复用的是控制变量、判据和失败模式，不是把任何已测系统的敏感组件或阈值直接升级成所有语音任务的标准。

## 来源身份

- [Same Bit Width, Different Outcomes: Post-Training Quantization of Text-to-Speech Across Architectures](https://arxiv.org/abs/2609.28974v1)，arXiv:2609.28974v1；§3–5、图 1–2、表 1–3。指标阈值、模型实验与系统数字均为作者报告。
- [tts-ptq-map 官方代码固定版本](https://github.com/uxfacdev/tts-ptq-map/tree/ea6f62da57ac86c1fe8f81818d689228c62b5d66)，静态核对量化、组件校准、算子改写及配对 bootstrap 路径；没有本地模型复现。等价判据解释和跨平台证据边界为整理者分析。
