---
title: LayerPos：剪枝后的位置重编号与逐层取舍
type: method
tags:
  - vlm
  - sparsity
  - attention
  - positional-encoding
sources:
  - raw/papers/2026-09-23/layer-aware-position-embeddings/paper.pdf
  - raw/repositories/2026-09-24/layerpos/source/scripts/llava15/pos_emb.sh
  - raw/repositories/2026-09-24/layerpos/source/scripts/llava15-13b/pos_emb.sh
  - raw/repositories/2026-09-24/layerpos/source/scripts/internvl35/pos_emb.sh
  - raw/repositories/2026-09-24/layerpos/source/single_layer_rate_llava.py
  - raw/repositories/2026-09-24/layerpos/source/vlmeval/vlm/llava/llava_pkg/model/language_model/modeling_llama_self.py
updated: 2026-09-29
---

# LayerPos：剪枝后的位置重编号与逐层取舍

LayerPos 不负责选择保留哪些视觉 token，而是决定**选完以后，各语言层用什么位置编号**。它在经过校准的定位敏感层保留原编号，在其他层使用连续压缩编号，尝试同时保留感知和定位能力。它没有新增可训练参数，但需要带定位标签的逐层校准；无训练不等于无校准。

本页研读论文正文、附录及固定版本的启动脚本与 LLaVA 位置处理路径；静态代码核对未替代模型运行。视觉 token 的产生与合并见 [视觉 token 与量化对象](../fundamentals/model/vision-language-model-tokens-and-quantization.md)，集合删除机制见 [CoRePrune](coreprune.md)。

## 1. gather 后的数组下标不一定是位置编号

例如原序列位置为 $[0,1,2,3,4,5]$，保留 $[0,1,4,5]$。物理存储都可以连续排成四个向量，但 RoPE 可使用：

- **Sparse**：$[0,1,4,5]$，保留原先相对距离；文本位置也沿原坐标延续。
- **Continuous**：$[0,1,2,3]$，将全部幸存视觉和文本 token 重新连续编号。

RoPE 写作 $q_m=R(m)W_Qx_m$、$k_n=R(n)W_Kx_n$。在固定旋转频率下，未归一化 attention logit 为

$$z_{mn}=\frac{(W_Qx_m)^\top R(n-m)(W_Kx_n)}{\sqrt{d_h}}.$$

所有位置加同一常数不改变 $n-m$；剪枝后非均匀压缩却改变不同 token 对的距离，因此即使向量集合和顺序相同，logits 仍会变。视觉编码器内的空间位置信息不等同于语言解码器的序列位置；本方法改变后者，不能解释成重新生成二维图像坐标。

[CoRePrune 的删除恒等式](coreprune.md#1-删除不仅去掉一项也重归一化剩余注意力)固定幸存 logits。若删除时同时做连续重编号，就额外引入位置效应，那个公式不能单独描述全部变化。

## 2. 距离变化为何有作用，以及不能推出什么

作者在 LLaVA-1.5-7B、CDPruner 保留 64 token、1,000 条 TextVQA 上观察：连续编号在多数层给视觉 token 更大的来自文本的 attention 总量；感知任务倾向受益，RefCOCO 定位任务却明显更依赖稀疏编号。连续编号缩小空隙，也改写了模型原先利用的视觉位置关系。

这是特定模型与任务下的经验结果。论文式 4 用 RoPE 的振荡和式及上界讨论长距离衰减；**上界的衰减不能证明每一个 logit、每个 head 或 softmax 后视觉质量都随距离单调下降**。整理者反例：二维旋转、两个未旋转向量均为 $(1,0)$ 时，内积是 $\cos\Delta$，距离从 $\pi$ 增至 $2\pi$，内积反而从 $-1$ 增至 1。softmax 分母还取决于其他 Key。

因此“Sparse 必然降低注意力、Continuous 必然改善所有感知”不是定理；这里的机制解释必须与实际层统计和任务干预一起使用，也不能仅凭 attention 增大判定信息利用更正确。

## 3. 逐层校准与最终执行

作者在 1,000 条 RefCOCO 样本上，以所有层 Continuous 为基线。每次只将一层 $l$ 改成 Sparse，其他层保持基线，计算

$$\Delta_l=\frac{\operatorname{Acc}^{\mathrm{sparse}}_l-\operatorname{Acc}^{\mathrm{cont}}}{\operatorname{Acc}^{\mathrm{cont}}}\times100\%.$$

图中的 400% 是相对准确率变化，不是增加 400 个百分点；基线很低会放大这个比例。该扫描约需一套基线加 $L$ 套单层评测，不能称为零成本设置。

根据单层增益首次明显出现、随后趋近零的位置，选择定位敏感区间 $[L_{gs},L_{ge}]$，最终在整段使用 Sparse，区间外 Continuous。正文没有给出可机械复现的统计显著性阈值；“significant”不能直接当成通过假设检验。多个层一起切换也不等同于单层效应相加，最终性能须另外验证。

论文附录表 4 的层号为：

| 骨干 | 总层数 | 论文定位敏感区间 |
| --- | --- | --- |
| LLaVA-1.5-7B | 32 | 7–21 |
| LLaVA-1.5-13B | 40 | 8–13 |
| InternVL3.5-8B | 36 | 4–26 |

这些区间不是跨骨干常数。扫描脚本读取 `subRefCOCO_val` 的定位评测结果，但论文没有充分交代校准与各报告定位子集的完整隔离关系；不能自行补写成严格独立测试，也不能仅据文件名认定发生数据泄漏。

## 4. prefill 与 decode 必须采用同一套层内坐标

固定快照的 `modeling_llama_self.py` 路径同时维护原始位置与连续位置。在 prefill 剪枝时，原始位置跟随 `keep_indexs` gather，另生成长度为当前序列长度的连续编号；每一层按 `SPARSE_LAYERS` 选择其一。position IDs 进入当前层的 attention/RoPE。pre-LLM 剪枝使后续各层处理缩短序列；inside-LLM 剪枝则只从剪枝位置起缩短后续层的 KV，剪枝前的层仍可保留完整前缀，不能把整个缓存都按最终 token 数计算。

decode 时，连续层的新 token 位置从**该层** KV 长度延伸，稀疏层则沿原始序列坐标延伸。代码为 pre-LLM 剪枝保存原始最大位置与 prefill 长度；inside-LLM 路径维护解码阶段的双版本编号。两种路径处理状态的方式不同，不能只改 prefill 的一个数组，就假设生成阶段自然正确。

例如原 prefill 长度 6、剪后长度 4，首个生成 token 在连续层可用位置 4，在稀疏层应沿原坐标用 6；把两者都设成 KV 长度 4，会改变它与缓存 Key 的相对旋转。每层拥有自己的 KV，层与层采用不同策略本身不要求互相重旋转；真正需要一致的是同一层历史 Key 与新 Query 的坐标约定。

代码显式限制所核对路径 batch size 为 1。这里说明的是静态执行意图与数据流，尚未通过多步生成、不同剪枝位置、缓存实现和并发请求验证正确性；不能外推为现有服务框架可直接使用。

## 5. 固定代码与论文设置的可复现性差异

官方快照 `d6a5bcc` 的扫描程序遍历零起始层 0–31，画图时标成 1–32，提供了层号转换的明确依据。按这个约定核对启动脚本：

| 脚本 | 实际选择的零起始 Sparse 层 | 与论文表 4 的关系 |
| --- | --- | --- |
| LLaVA-7B | 6–21 | 对应第 7–22 层，比论文上界多一层 |
| LLaVA-13B | 7–12 | 对应第 8–13 层，与表一致 |
| InternVL3.5 | 3–25，另加 29 | 对应第 4–26 层及第 30 层，有区间外附加层 |

另外，13B 脚本标为全 Sparse 的选项只列 0–31，而模型有 40 层；按当前逐层成员判断，最后八层不在该集合中。它不能直接作为“全部 40 层 Sparse”的干净对照。

这些是**公开快照设置差异**，并不证明论文表格使用了这些脚本或结果错误。复现时应分别固定论文区间和当前脚本区间，并保留真正全 Sparse/全 Continuous 基线，不能默默混称为同一配置。本文未修改上游代码。

## 6. 综合恢复率不能替代定位成绩

LLaVA-1.5-7B，原 576 个视觉 token，CDPruner 保留 128 个，论文表 1：

| 位置策略 | 综合 Acc. | 相对综合成绩 Rel. | RefCOCO |
| --- | --- | --- | --- |
| Dense | 55.6 | 100.0 | 55.0 |
| Continuous | 47.2 | 84.9 | 5.4 |
| Sparse | 50.2 | 90.3 | 44.4 |
| LayerPos | 54.6 | 98.2 | 45.1 |

98.2% 是综合成绩保留，不是定位恢复到原来的 98.2%：RefCOCO 仍损失 9.9 个百分点。相对 Sparse/Continuous 的 Rel. 分别增加 7.9/13.3 个百分点，不能再解释成综合准确率增加同样百分点。

表中 Rel. 是该表综合 Acc. 相对 Dense 的比例；MME 原始分数与百分比指标需要不同尺度处理，不能直接平均原始列，也不能与 CoRePrune 的逐任务相对比值均分互换。表 1 的 RefCOCO 汇总口径与正文单独展示的 testA 案例不同，不能混比。13B 主表只列五个非定位指标，不能凭该表证明 13B 定位能力也被同等恢复。

论文报告 Nvidia A6000、PyTorch 2.1.2/CUDA 11.8 的实验设置，但没有给出足以验证真实延迟收益的完整系统表。没有额外训练参数不证明没有位置构造、切换或缓存处理成本；剪枝减少 token 的收益也不能全部归因于 LayerPos。结论限于所测图像模型与剪枝配置，尚不是视频时序位置或所有多轴 RoPE 的通用方案。

## 来源身份

- [Layer-Aware Position Embeddings for Visual Token Pruning in Multimodal Large Language Models](https://arxiv.org/abs/2609.23715v1)，正文 §3–4、表 1–3、附录 A/B、表 4。
- [LayerPos 官方代码固定版本](https://github.com/YahongWang1/LayerPos/tree/d6a5bccfad1c6a8b548fd4cd699678c8b7571c2e)，静态核对位置策略脚本、单层结果汇总和 LLaVA forward；未运行模型。RoPE 反例、坐标示例及证据边界为整理者分析。
