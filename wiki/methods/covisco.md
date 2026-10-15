---
title: CoVisco：原生摘要token、分段通信与两层视觉选择
type: method
tags:
  - vlm
  - token-compression
  - attention
  - deployment
sources:
  - raw/papers/2026-10-05/covisco
  - raw/repositories/2026-10-05/covisco
updated: 2026-10-15
---

# CoVisco：原生摘要token、分段通信与两层视觉选择

CoVisco在视觉encoder训练时加入可学习abstract tokens，让短摘要接口直接承接监督；编码后的patch流仍保留，需要细节时由selector补入语言输入。视频还可在进入encoder前按codec分数选patch。两层选择分别控制视觉计算和LLM序列，不能合成一个“token压缩率”。

本页依据CoVisco v1与作者代码 `604cd70d53afc39506682d423c07604a38d37b40`的codec接口、视觉encoder、selector和LLM接入。代码中的query tokens是视觉摘要学参，不是用户提问。未运行训练/评测；视觉token、输入长度和位置基础见[公共VLM页](../fundamentals/model/vision-language-model-tokens-and-quantization.md)。

## 1. abstract负责短接口，patch保留局部证据

视频分成 $S$段，每段有 $m$个可见patch，加入共享的 $Q=100$个abstract参数，形成 $S\times(m+Q)\times d$输入。各段扩展同一组学参，不是每段保存独立参数；图像取 $S=1$。ViT-L有24层、宽1024，交替执行：

| 层模式 | 每段query可见的KV | 信息怎样流动 |
| --- | --- | --- |
| 段内attention | 本段patch与abstract | 汇集局部细节 |
| abstract通信attention | 本段patch与所有段abstract | 异段信息通过摘要通道交换，无直接跨段patch交互 |

只有最终abstract送attention pooling与图像—图像、图像—文字、视频—文字SigLIP监督头；patch仍参与每层并通过交互获得间接梯度。原生压缩指训练出来的短输出接口，不是从ViT开头删除全部patch。（§3.1–3.5、附录A。）

两类attention成本分别为
$O(S(m+Q)^2)$与 $O(S(m+Q)(m+SQ))$。
固定原patch总数 $N=Sm$时，主要patch二次项从 $N^2$变成 $N^2/S$，但摘要通信项仍在；固定每段长度增长段数也不能称严格线性长视频成本。

## 2. codec选输入，selector选输出

本地主线先均匀取frame IDs，调用外部codec reader取得motion vectors/residual，各自归一化后融合，按patch区域求分数。每段以比例预算选输入，默认保留首帧全部patch，预算不足时也抬到一整帧，再从后续帧补top-K；输出是按原序排列的全局patch索引visidx及frame信息。（`precompute_video_visidx.py`。）

原始像素patch与4D RoPE都先依据完整时空网格建立，再用同一visidx gather，选中patch才进入embedding/encoder。视觉坐标分为segment/time/height/width，abstract使用预留时间0和10×10布局。它保留视觉网格几何，与语言decoder的[RoPE位置策略](layerpos.md)处在不同层。（`_onevision_encoder_src.py:get_segments`。）

encoder输出后，默认两层selector依次做段内patch self-attention、patch读本段abstract的cross-attention及FFN，再打分选patch。上下文化特征只用于评分，实际gather的是原encoder patch向量，并恢复原位置顺序。它没有读取用户问题，也不能找回codec前已经删掉的输入patch。（§3.6、`_token_selector_src.py`。）

| 语言模型输入策略 | 每段输出 | 总视觉长度 |
| --- | --- | ---: |
| abstract-only | 全部摘要 | $SQ$ |
| abstract+top-K | 摘要与选中patch | $S(Q+K)$ |
| full-patch | 全部编码patch | $Sm$ |

在abstract+top-K模式中改变 $K$，仍需先编码相同输入patch，另执行selector；不能按LLM长度降幅推定encoder同比省算。Projector把选定向量映到LLM宽度，按每段[q,v]组织，与等量image占位符逐样本核对后替换 `inputs_embeds`，再由语言模型生成。（`llava_covisco.py:_arrange_tokens/forward`；所读eval adapter的 `_encode_frames/_generate`。）

## 3. 短接口依靠分阶段训练，hard预算与score梯度分开

三阶段依次是：

1. 约565M图文与6.4M视频流预训练ViT，通过三监督头训练abstract接口。
2. 冻结ViT，用Qwen3-1.7B及约1M图像/1M视频训练selector、projector与LLM。
3. 冻结ViT与selector，用740K图像/800K视频instruction训练Qwen3-4B-Instruct-2507及projector。

因此不能给任意现成ViT直接加100个未训练token，就期待同样摘要能力。（§3.5–3.7。）

hard top-K索引不可微，训练时将选中的原patch乘连续score，使打分器经选中特征获得梯度；它也改变前向幅值，不是只改反向。SFT代码在no_grad或selector冻结时关闭这条gate，还有MMR去重、首帧限制等可选路径；不能把两个同名selector副本或所有配置视为论文简单top-K。（`_token_selector_src.py:forward`。）

hard K按当前ratio计算，固定SFT代码的soft sparsity目标却写死为40% patch数；它不是每个ratio上都精确匹配K的约束。报告预算时以实际输出K为准，soft正则与hard计数分列。（`llava_covisco.py:forward`。）

## 4. 固定代码仍有接续条件，短摘要也有质量代价

原件称使用HEVC原码流、不重编码；本快照能核到外部提取接口及后续融合/top-K，codec reader本体未归档。Eval visidx缺失/不匹配会回退full-patch，输入预算随之改变。图像位置平移在论文/预训练copy属于训练augmentation，SFT copy却在eval图像路径也采样shift；这些差异需在复现时固定，不能由归档完整性推出运行一致。（`_onevision_encoder_src.py:offset_positions`及eval adapter。）

视频四段的abstract-only为400 token。图像DocVQA表中abstract-only仅20.5，补入80% patch后75.1、full为75.5，说明细字/局部证据不能由短摘要完全替代。更多patch也不保证所有任务单调改善，预算需结合任务评价。（Table3–4。）

长视频表把64/128/256帧对应为4/8/16段，abstract-only长度随之为400/800/1600；无需追加SFT不等于固定400token处理任意长度。论文没有完整codec、视觉encoder到LLM的端到端时延或峰值内存表，理论attention成本与LLM长度只支持相应资源分析，不直接给出实测加速。（Table5、附录B。）

## 来源身份

| 来源 | 固定版本 | 核读与使用范围 |
| --- | --- | --- |
| CoVisco: Codec-Native Vision Encoder with Native Token Compression for Unified Image-Video Understanding（`https://arxiv.org/abs/2609.39924v1`） | `arXiv:2609.39924v1` | 完整active主文/附录A–D；framework/selector原图与PDF12–14表3–5/图2回查；预算、训练和质量分别使用，未核全部外部结果或运行。 |
| ernie-research/CoVisco（`https://github.com/ernie-research/CoVisco/tree/604cd70d53afc39506682d423c07604a38d37b40`） | `604cd70d53afc39506682d423c07604a38d37b40` | codec预选接口、4D位置/分段encoder、两selector副本、动态策略/LLM placeholder与所选eval adapter；外部codec、全部训练/任务/分布式循环及权重未核。静态主线不代表训练或模型验证。 |
