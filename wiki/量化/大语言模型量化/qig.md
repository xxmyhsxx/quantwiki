---
title: QIG：以量化误差积分梯度指导 token 加权校准
slug: qig
sources:
  - raw/papers/quantization/vlm/2026-03-fine-grained-post-training-quantization-lvlm-qig/paper.pdf
  - raw/papers/foundations/2017-08-axiomatic-attribution-integrated-gradients/paper.pdf
  - raw/repositories/quantization/2026-07-ucas-xiang-qig-06fa8813/manifest.md
  - raw/repositories/quantization/2026-07-ucas-xiang-qig-06fa8813/source/qmllm/methods/qig/entry.py
  - raw/repositories/quantization/2026-07-ucas-xiang-qig-06fa8813/source/qmllm/methods/qig/quantize/pre_quant.py
  - raw/repositories/quantization/2026-07-ucas-xiang-qig-06fa8813/source/qmllm/methods/qig/quantize/auto_scale.py
  - raw/repositories/quantization/2026-07-ucas-xiang-qig-06fa8813/source/qmllm/methods/qig/quantize/auto_scale_wa_distort.py
  - raw/repositories/quantization/2026-07-ucas-xiang-qig-06fa8813/source/qmllm/quantization/qlinear.py
  - raw/repositories/quantization/2026-07-ucas-xiang-qig-06fa8813/source/qmllm/quantization/quant_funcs.py
  - raw/repositories/quantization/2026-07-ucas-xiang-qig-06fa8813/source/configs/llava_onevision/MBQ_search/7b_weight_activation.yaml
updated: 2026-09-15
---

# QIG：以量化误差积分梯度指导 token 加权校准

QIG 的问题是：图文 token 在共享语言层中不断交互，只按“视觉/文字”给一个平均重要性，可能掩盖模态内部的差别；直接照搬任务梯度或注意力，也未必测量了量化造成的误差。它沿参考激活到浮点激活的路径，对浮点与量化输出的差异计算积分梯度，再用 token 权重指导通道缩放搜索；另给出把权重用于 GPTQ 二阶统计的扩展。

它与 [[mbq|MBQ]]、[[vlmq|VLMQ]] 共享“改变重构误差的重要性分配”这一思路，特点是选择**量化相关输出差异和路径梯度**。它没有直接分配每个 token 的运行位宽，也不是 [[luq|LUQ]] 式层间混合精度。

依据为 *Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients*，arXiv:2603.17809v1（下称论文），18 页含附录 A–D 已全文研读。实现核对固定在官方快照 `06fa88134671647827dea08dc37c50c81e4eeac7`（简称 `06fa8813`），只覆盖下述关键路径；未运行模型或硬件实验。

## 1. 研究对象与动机证据

主要对象是视觉语言模型语言骨干中的线性权重与激活，主实验包括 LLaVA-onevision、InternVL2、Qwen2-VL。正文的 weight-only 设置采用非对称 g128 权重量化；W4A8 采用对称 per-channel 权重、per-token 激活。附录 B 对 g128 的总括与主文 W/A 描述不完全一致，不能把整个方法统一标成 W4g128A8；公开 W/A 代码确实调用 per-channel/per-token 路径。

主校准使用 ShareGPT4V 改进的 COCO Caption 数据，随机取 128 个图像—描述对，按目标模型的对话格式处理；评测采用论文所述 LMMs-Eval 协议。没有完整样本 ID、权重 revision 与执行环境记录，不能把所读代码默认值视为全部主表配置。（§4.1）

“粒度细”本身不是充分条件。表 1 的 InternVL2-8B W4A8 VizWiz 中，SFT 梯度从模态级 57.36 改成 token 级 55.78，再纳入特殊 token 为 55.65；注意力分数的三项则为 56.43、57.12、57.52。它说明**目标、粒度与特殊 token 的处理共同影响结果**，不支持把任何粗分数拆细就会改善。token 与共享参数的轴关系见 [[vision-language-model-tokens-and-quantization|图文 token 与量化对象]]。

## 2. 归因对象、基线和标量化

论文 §3.3 写出 $G(z)=f(z,w)-f(z,w_q)$，在 $z(\alpha)=x_q+\alpha(x-x_q)$ 上积分。这里 $x$ 是浮点输入，$x_q$ 是参考量化输入，$w,w_q$ 为浮点/量化权重。weight-only 若直接令 $x_q=x$，路径长度为零；§3.3 因而使用零输入作为该分支基线。

块输出通常是张量，必须补上标量定义才能求 IG。附录 B 实际采用输出差异的绝对值均值；代码进一步确定为

$$
E_{bt}(Z)=\frac1{d_o}\sum_h|f_{\rm fp}(Z)_{bth}-f_{\rm q}(Z)_{bth}|,
\qquad F(Z)=\sum_{b,t}E_{bt}(Z),
$$

$$
I_{btc}=(X-X_0)_{btc}\int_0^1
\frac{\partial F(X_0+\alpha(X-X_0))}{\partial Z_{btc}}\,d\alpha.
$$

$B,T,d_o$ 分别为 batch、token 数和输出通道数，$c$ 是输入通道。`autograd.grad` 的 `grad_outputs=ones` 对所有输出 token 求和后反向，再按**输入 token** 聚合；它不是每个输出只归因给相同位置输入。跨 token 依赖可以进入梯度，但这种路径贡献并不识别真实语义因果关系。

对于适当连续可微的标量 $F$，原始有符号 IG 满足

$$\sum_{b,t,c}I_{btc}=F(X)-F(X_0).$$

这只分解**相对基线的目标变化**。基线误差不为零时，不等于终点全部量化误差；主文同输入的权重差异也不同于联合差异 $f(X,w)-f(X_q,w_q)$。后者还含输入变化项。完整证明、负归因例子和量化非光滑条件见 [[integrated-gradients-and-quantization-sensitivity|积分梯度与量化敏感性]]。（论文 §3.3、附录 B；IG 原论文 §3）

## 3. 从归因变成校准权重

主文用 IQR 抑制极大 token 归因：计算第一/第三四分位数 $Q_1,Q_3$，裁剪到 $[Q_1-1.5\mathrm{IQR},Q_3+1.5\mathrm{IQR}]$，再按样本归一化。核心意图是防止少数特殊或异常 token 垄断搜索目标。（§3.3、附录 A 表 A1）

主文有符号公式并未充分说明怎样得到非负系数；公开代码用

$$u_{bt}=\frac1{d_i}\sum_c|I_{btc}|,$$

再归一化。这样避免负误差系数，但**取绝对值、裁剪、归一化后的 $\lambda$ 不再满足原始 IG 的有符号完整性**。非负性与有解释的优化收益不是同一保证。归因全部为零时，分母下限只能避免除零，不会自动产生有信息的分配；实现也不能因此被解释为默认退回均匀权重。

还有一个实质实现差异：`auto_scale.py` 的 WO 分支确实做 IQR 后再次归一化；`auto_scale_wa_distort.py` 的 W/A 分支虽然算了四分位边界，却直接返回未裁剪权重。不能以变量名 `weights_clipped` 断言裁剪已生效。

## 4. CWE 如何改变通道缩放搜索

统一用 token 为行的布局：$X\in\mathbb R^{T\times d}$，$W\in\mathbb R^{o\times d}$，$D=\operatorname{diag}(s)$。浮点恒等式为

$$XW^\mathsf T=(XD^{-1})(WD)^\mathsf T.$$

论文的逐通道均衡（CWE）以 token 系数评价缩放：

$$
\min_s\sum_t\lambda_t
\left\|[Q_a(XD^{-1})Q_w(WD)^\mathsf T-XW^\mathsf T]_{t,:}\right\|_2^2.
$$

$Q_w,Q_a$ 表示量化后反量化的张量；WO 省去 $Q_a$。$s$ 仍是一组**共享输入通道尺度**，token 分数改变搜索代价，没有为每个 token 保存一份不同权重。等价图与量化后误差的区别见 [[diagonal-scaling-equivalent-transform|对角缩放与等价变换]]。

固定代码的 WO 搜索用输入平均绝对幅度 $a_c$ 生成 $s_c=a_c^r$，$r\in\{0,0.05,\ldots,0.95\}$；做下限限制和几何范围归一化后，对每个候选临时量化、评价、恢复参数，选择最佳尺度，最后融合。token 权重在这次候选搜索前生成，不随每个候选重新积分。代码允许 MAE/MSE，而 `qig_entry` 默认 MAE；不能把论文的平方目标当作所有运行的实际损失。（`auto_scale.py::_search_module_scale`、`entry.py::qig_entry`）

W/A 还从当前量化/缩放路径收集 `input_feat_q`，以其作为参考输入；搜索目标 `org_out=block(x_q)`，而归因终点使用浮点特征 `x`。因此它不是把上面单层、同输入公式逐字实现。重构输入的选择与前层误差见 [[layer-reconstruction-second-order-compensation|非对称输入重构]]；不能无条件称这里为原始完整浮点网络监督。

## 5. 怎样进入 GPTQ 二阶统计

对固定非负 $\lambda_t$，定义 $\Lambda=\operatorname{diag}(\lambda)$。在线性同输入的平方重构中，

$$J(\delta)=\sum_t\lambda_t(x_t^\mathsf T\delta)^2
=\delta^\mathsf TX^\mathsf T\Lambda X\delta.$$

于是 Gram 矩阵为 $G=X^\mathsf T\Lambda X$，Hessian 为 $2G$；也可先令 $\widetilde X=\Lambda^{1/2}X$，再沿用 [[gptq|GPTQ]] 的补偿框架。论文 §4.3 的扩展采用这个加权思路。

注意 $\lambda$ 是平方误差的系数，若直接把 $\lambda X$ 喂给统计过程，得到的将是 $\lambda^2$ 权重。它与 [[vlmq|VLMQ]] 中位于范数内部的因子要按实际目标比较，不能只按“都用 token 权重”视为相同。零权重/有效秩不足仍可能要求阻尼。此推导不适用于直接把 MAE 当二次目标，也不证明全部离散权重选择最优。

## 6. 公开实现的流程与边界

可复述的论文流程是：收集图文校准激活 → 建立浮点与量化路径及参考点 → 32 步归因 → 形成 token 系数 → 搜索和融合通道尺度 → 最终量化与评测。实际落地还要核对以下关键差别：

| 环节 | 固定快照 `06fa8813` 的行为 | 影响 |
|---|---|---|
| WO 启用归因 | `_search_module_scale` 用函数属性 `_layer_idx` 门控，初值 0 且每次调用加两次；前五次搜索调用用均匀权重，第六次起满足 `>=9` | 计数单位是模块搜索调用，不是 Transformer 层；不是所有层/模块一开始都用 IG，跨模型复用进程也需考虑状态 |
| W/A 量化分支 | `WALinear.forward` 为 `no_grad`，使用浮点存储与浮点 `F.linear` | 量化线性路径被截断梯度；不能直接套用完整差异函数的 IG 定理，也不是已验证的低比特内核 |
| W/A 裁剪 | 算 IQR 边界但返回原始归一化权重 | 与论文裁剪消融不能直接对应 |
| 标量与搜索 | 归因为 MAE 差异，搜索可选 MAE/MSE | 归因目标与优化目标要分别记录 |
| 输入与分组 | W/A 收集 `q_input` 派生特征，量化为 per-channel/per-token | 参考点不等于简单对当前 $X$ 做一次标量量化；不能直接套 WO g128 |
| 启动配置 | 所读 `configs/llava_onevision/MBQ_search/7b_weight_activation.yaml` 仍设置 `method: mbq` | 文件位于 QIG 仓库不代表它调用 QIG；必须核对实际入口和缓存 |

量化层的梯度截断不意味着所有量化块输出都完全无梯度：残差等旁路可能仍保留输入依赖。能确定的是**穿过该量化线性层的导数未进入归因**，不是完成了一份所有模型计算图审计。上述是静态代码阅读结论，不是模型运行结果。

## 7. 哪些实验真正帮助判断方法

下列均为作者报告，具体范围以原表为准；不同任务的分数和平均值不与其他论文拼接。

| 证据 | 条件与结果 | 可以支持什么 |
|---|---|---|
| 主比较，表 2 | LLaVA-onevision-7B：WO 平均 MBQ 70.44 → QIG 72.04；W4A8 为 70.16 → 70.23 | 同一模型中两种量化对象的收益差别很大，不能用 WO 改善代表 W/A |
| 负面边界，表 2 | 同模型 W4A8 ChartQA 74.92 → 74.52、ScienceQA 94.70 → 94.25；Qwen2-VL-7B VizWiz 60.17 → 58.85 | 综合均值改善不保证每项任务改善 |
| GPTQ 扩展，表 5 | LLaVA-onevision-7B WO：ChartQA 73.72 → 74.12、VizWiz 54.87 → 56.95，但 AI2D 76.81 → 76.65 | token 加权可用于另一基础量化器，仍有任务取舍 |
| 裁剪消融，附录 A 表 A1 | LLaVA-onevision-7B W4A8，无 IQR 时 VizWiz/MMMU 为 54.32/41.37，IQR 为 59.10/45.00 | 后处理本身影响较大，不能把增益全归因于 IG；需对齐公开代码分支 |
| OCR 校准，附录 C 表 A3 | Qwen2-VL-7B W4A8、128 个 InfoVQA 校准样本，DocVQA/ChartQA/OCRBench 平均 MBQ 77.45 → QIG 80.97 | 支持特定 OCR 校准与评测组合；不是部署后漂移或跨域再校准验证 |
| 校准成本，表 6 | A800 80GB：8B 为 MBQ 0.55 h、QIG 0.58 h；26B 为 0.95 h、0.99 h | 在该实现/模型条件下总校准与搜索开销增量有限；不是推理时延 |

表 4 的目标消融需要保留解释限制：若其中 $f(x)-f(0)$ 或 $f(x)-f(x_q)$ 真是同一标量函数减固定常数，其梯度应与 $f(x)$ 相同。附录 B 的绝对误差和两条网络路径提供了可能不同的实际目标，但表头不足以唯一恢复实验；不能用这个表直接证明“减去基线常数使 IG 更好”。其末行与主表相应设置也未完全对齐，故不拿该数字作单变量因果证据。

附录 D 四组问答展示个案行为；更长、更接近浮点答案不等于更正确，不能以少量描述例子证明普遍减少幻觉。这里只保留证据边界，不转录案例全集。

## 8. Strong、Weak 与适用判断

**Strong：**它将重要性目标靠近待优化的量化差异，并用路径信息补充单点梯度；保留 token 内部差别但仍产出可融合的通道尺度。表 1、裁剪消融和 GPTQ 扩展使我们能分别观察分数、后处理与基础算法的作用，不只是一个排行榜结果。

**Weak：**完整性属于明确标量和真实路径梯度，无法直接保证绝对化、裁剪后的分配最优；公开 W/A 分支进一步存在梯度与裁剪差别。任务收益并不一致，校准分布、特殊 token、归因基线、函数门控和重构目标均影响结果。现有实验没有证明实际低比特端到端加速，也没有证明云端重新校准后下发参数的系统收益。

因此可把 QIG 用作“量化误差导向的重要性估计”的深入参照；复用时优先固定目标、基线、权重后处理和代码分支，再与均匀/token/模态权重比较。归因质量、模型任务质量和 [[quantized-matmul-scaling-execution|部署执行收益]] 应分别验证。本轮完成全文研读与指定代码路径核对，尚无模型复现结论。
