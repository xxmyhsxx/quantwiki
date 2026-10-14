---
title: Qwen2.5-Omni模型执行：音画输入、Talker状态与完整波形返回
type: implementation
tags:
  - vlm
  - inference
  - kv-cache
  - deployment
sources:
  - raw/repositories/2026-09-21/transformers
updated: 2026-10-14
---

# Qwen2.5-Omni模型执行：音画输入、Talker状态与完整波形返回

本页核对Summer-Summer/transformers的`hf-4.53.2`固定源码：输入媒体先形成mel特征、视觉网格与展开的语言序列，Thinker生成文字和隐藏状态，Talker生成语音code，Token2Wav恢复波形。公开包装器依次完成这三次调用，再一次返回文字序列与完整波形；论文的流式结构不能直接用作这个接口的返回时序。

模型原理与训练证据由[Thinker-Talker基础](../fundamentals/model/multimodal-perception-and-streaming-speech-generation.md)承接。这里核对processor、模型forward、生成包装器与Token2Wav的所选路径，没有取得发布checkpoint的实际配置、chat template、speaker参数或权重，也没有运行模型、attention后端或语音评测。

## 1. 模态占位长度由有效mel和视觉网格决定

`Qwen2_5OmniProcessor.__call__`要求提供`text`。对话先由chat template整理出文字和媒体标记，音频数组、图像与视频另交相应processor；最终输出的`input_ids[B,S]`包含普通文字、控制标记和展开后的模态占位。这里$B$是语言batch，$S$是展开后的语言序列长度，媒体条目数可以与$B$不同。通用`ProcessorMixin.apply_chat_template`在没有模板时直接报错；Omni覆盖方法还检查特定system prompt并给出语音模式警告，警告本身不验证其他提示能否工作。（`processing_qwen2_5_omni.py`的`__call__`、`replace_multimodal_special_tokens`、`apply_chat_template`；`processing_utils.py`的`apply_chat_template`。）

音频形成`input_features[N_a,C_m,T_pad]`与`feature_attention_mask[N_a,T_pad]`：$N_a$是音频条目数，$C_m$是输入mel通道数，$T_{pad}$含填充。processor把音频特征提取器的`attention_mask`改名，避免与语言序列mask混淆。对第$i$条音频，令有效mel帧数为$L_i=\sum_t M_{i,t}$，实际代码的两次长度缩减可写为

$$
L_i^{\rm conv}=\left\lfloor\frac{L_i-1}{2}\right\rfloor+1,
\qquad
N_i^{\rm audio}=\left\lfloor\frac{L_i^{\rm conv}-2}{2}\right\rfloor+1.
$$

前式对应stride-2卷积，后式对应kernel/stride均为2的平均池化；有效长度需足以执行池化，最终音频标记重复$N_i^{\rm audio}$次，以便每个位置接收一个编码表示。按这一公式，200个有效mel帧得到100个卷积位置，再得到50个语言位置；这是按代码构造的长度示例，不是模型实验。填充长度不能替代$L_i$。（`processing_qwen2_5_omni.py:161–170`；`Qwen2_5OmniAudioEncoder._get_feat_extract_output_lengths`。）

processor默认传`sampling_rate=16000`，但`WhisperFeatureExtractor.__call__`只检查传入采样率是否与自己的配置一致，随后提取特征，**不负责重采样**。调用者须先提供匹配的波形。其通用构造默认`feature_size=80`，Omni音频编码器构造默认`num_mel_bins=128`；实际processor配置应与编码器匹配，不能把两个默认构造对象直接拼接，也不能由本源码默认值断言某个发布checkpoint实际使用的mel配置。（`feature_extraction_whisper.py`的构造方法和`__call__`；`configuration_qwen2_5_omni.py`的`Qwen2_5OmniAudioEncoderConfig`。）

视觉processor提供`image_grid_thw`或`video_grid_thw[N_v,3]`，一行$(T,H,W)$记录时间patch数和空间patch网格。若空间merge边长为$m$，图像或不带音轨的视频标记数是$THW/m^2$；这里的$T$不是直接输入的原始帧数。视觉编码器将特征按窗口重排，经过窗口/所选全注意力层与merger，再恢复位置顺序，形成与展开占位数相配的语言宽度特征。网格与merge机制见[视觉token基础](../fundamentals/model/vision-language-model-tokens-and-quantization.md)。（`replace_multimodal_special_tokens`；`Qwen2_5OmniVisionEncoder.forward`。）

## 2. 音画交错同时约束序列顺序和三轴位置

`use_audio_in_video=False`时，视频标记按视觉位置数展开；设为`True`时，processor同时消耗该视频的音频长度，把两种占位按时间区间交错排列，每个区间先视觉、后音频。默认`position_id_per_seconds=25`、`seconds_per_chunk=2`，因此区间阈值是50个**时间ID单位**；一帧的全部空间位置可能共享时间ID，50不表示固定50个视觉token。此参数描述输入音画组织，与Talker输出code块或首包时间没有直接等式。（`replace_multimodal_special_tokens`、`get_chunked_index`。）

processor用`temporal_patch_size/fps`生成`video_second_per_grid`。当前代码把取得的`fps`复制到各视频后做除法，默认值为2.0；这个路径支持的是共同标量，不能从类型注解推定任意逐视频FPS列表都受支持。实际采帧与这个值需一致，否则序列中的时间坐标会与媒体时间错位。

Thinker的`get_rope_index`再根据这些标记和网格产生`position_ids[3,B,S]`，三个分量依次表示时间、高度、宽度。文字和音频位置的三个ID相同；视觉位置按合并后的空间网格展开。对视频时间patch序号$j$，代码先以浮点计算$j\cdot\Delta t\cdot p$，再转整数，其中$\Delta t$是`video_second_per_grid`，$p$是每秒位置ID数；它没有在乘$p$之前先把$\Delta t$截成整数。各段从已有位置的最大值加一开始，序列行号因此不等于时间ID。（`modeling_qwen2_5_omni.py`的`get_llm_pos_ids_for_vision`、`get_rope_index`，尤其视频时间计算。）

音画交错由processor展开占位，再由模型重建相同顺序的位置。媒体抽取工具、processor与`generate`的`use_audio_in_video`必须一致，processor的时间/分块参数也须与模型配置对应。包装器不会自动把processor参数写回模型配置；调用方的一处开关不能保证其他环节同步。

## 3. Thinker在prefill注入特征，decode延续位置与KV

音频编码首先去掉padding：`get_audio_features`依照`feature_attention_mask`将有效帧拼成`[C_m,\sum_i L_i]`，再交给音频塔。塔按`2*n_window`个mel帧分块；构造默认`n_window=100`，即200帧。每块经过两层Conv1d、位置表示和块内双向attention，之后按原音频长度重新组合，逐条平均池化、归一化并投影到Thinker宽度$D_T$，形成`[\sum_i N_i^{audio},D_T]`。块内编码和后续语言层的因果attention是不同可见范围。（`Qwen2_5OmniAudioEncoder.forward`、`padded_and_mask_function`。）

`get_audio_features`有一项实际接口边界：没有`feature_attention_mask`时，`else`会把传入的`audio_feature_lengths`清为`None`，随后仍尝试对mask求和。因而该固定路径不能用“只传lengths”代替mask；正常processor提供的mask需保留。音频输出行数另与缩减后的长度总和核对。（`modeling_qwen2_5_omni.py:1766–1805`。）

Thinker先取得文字embedding`[B,S,D_T]`。当`input_ids`存在且长度不为1时，forward进入模态注入分支，将音频、图像和视频特征转为该embedding的设备与dtype，按对应token ID的mask写入各占位。语言序列长度保持$S$，替换的是占位位置上的向量；这些连续特征不是Talker输出的离散code。该prefill判断依赖输入长度，不能视为对所有自定义调用都适用的阶段检测。（`Qwen2_5OmniThinkerForConditionalGeneration.forward`。）

共享语言decoder按配置进行因果attention。Q/K应用三轴RoPE，V不旋转；KV按key/value head数保存，attention前用当前K/V更新各层cache。`use_cache=True`且未传cache的普通路径创建`DynamicCache`。prefill根据多模态最大位置计算`rope_deltas`，并扣除左padding数量；decode用`cache_position[0]+rope_deltas`延续位置，使缓存序列长度与RoPE坐标的差异得到保留。生成准备函数在首轮后清空图像和视频pixel输入；音频参数可继续传递，但长度为1的forward不重复执行模态注入。（`Qwen2_5OmniThinkerTextModel.forward`、`Qwen2_5OmniAttention.forward`及Thinker的`prepare_inputs_for_generation`。）

这里的cache保存语言层已经计算的K/V，并不保存原始mel或视频，也不是Talker的cache；其基本作用见[自回归推理](../fundamentals/model/transformer-autoregressive-inference.md)。选中attention接口不证明相应设备kernel已经运行。

## 4. Talker逐code消费已生成的回答表示

音频输出开启后，包装器让Thinker返回各轮隐藏状态和完整文字序列。它先等待`self.thinker.generate`结束，再从结果构造Talker输入。prompt的底层embedding副本中，已注入的音频/图像/视频向量被置零，而高层Thinker隐藏状态仍保留；两者相加后作为prompt语义条件。回答部分同样组合各轮顶层隐藏状态与底层token表示，并加入speaker文字BOS。这里传递的是隐藏表示，包装器没有把Thinker的`past_key_values`交给Talker。（`Qwen2_5OmniForConditionalGeneration.generate:3898–3991`。）

Talker有两组角色不同的ID：`input_text_ids`用于重建prompt与回答的三轴位置；`input_ids`是codec mask/pad/BOS及后续语音code。prefill的`inputs_embeds`已由Thinker表示构造，末两个位置再加codec pad/BOS embedding，然后经`thinker_to_talker_proj`从$D_T$投影到Talker宽度$D_K$；配置的codec embedding宽度须与$D_T$匹配，才能在投影前相加。其输出head给出codec词表概率，而不是再次采样文字词表。（Talker的`forward`及包装器输入构造。）

后续每生成一个code，Talker将当前code embedding与`thinker_reply_part[:, :1, :]`相加；只要剩余状态多于一个，就移去刚消费的第一项，并把剩余张量随生成状态传到下一轮。包装器在回答状态末尾附加文字EOS与pad embedding，最终只剩pad时持续复用它。因此codec步数与文字token数不必相同；这一实现是按codec轮次消费已生成的回答状态，不是每步重新运行Thinker生成新文字。（`Qwen2_5OmniTalkerForConditionalGeneration.forward`、`_update_model_kwargs_for_generation`。）

Talker语言模型在自己的普通缓存路径创建`DynamicCache`，逐层保存语音生成的K/V；Thinker和Talker有各自的隐藏宽度、位置增量与cache。二者的语义条件相连，不代表物理共享KV。（`Qwen2_5OmniTalkerModel.forward`。）

生成后包装器以`talker_result[:, prompt_length:-1]`截取code，无条件移除最后一项。常见EOS停止与达到`max_new_tokens`上限的停止方式不能仅凭该切片视为同一种处理；本页没有运行这些边界，也未验证被移除项在所有停止条件下都是EOS。

## 5. Token2Wav的局部窗口作用于整段mel求解

Token2Wav接收完整code序列`[1,N_c]`、speaker条件向量和参考mel，DiT先产生mel，再由BigVGAN产生波形。默认`repeats=2`使每个code embedding沿序列方向重复两次，DiT目标长度为$2N_c$个mel位置；这是输入code到输出mel的比例，不是输入音频的两次降采样，也不能独立推出code对应的秒数。（`DiTCodecEmbedding.forward`、`Qwen2_5OmniToken2WavDiTModel.sample`。）

DiT默认`block_size=24`，按**mel序列位置**计算块编号。每层mask允许的key块与query块之差满足`-look_backward_block <= block_diff <= look_ahead_block`。构造默认的零起始层10允许前看一块，层0和20允许回看一块，其余层只看当前块。跨层传播能扩大整体依赖，但不能把论文的“过去两块、当前块、未来一块”直接写成该源码每层的四块窗口。局部attention也不使外层生成包装器自动产生音频块。（`DiTDecoderLayer.forward`、`_create_block_diff`；`Qwen2_5OmniDiTConfig`。）

这一固定版本还有明确的RoPE边界：`DiTAttention.forward`只对第一个attention head的Q/K应用RoPE，源码将其归于训练过程并注明后续版本计划修正。本页保留实际行为，不把语言层全head的三轴位置规则移植到DiT。（`modeling_qwen2_5_omni.py:DiTAttention.forward:3005–3008`。）

`sample`先为整段目标mel建立噪声状态，默认`num_steps=10`用于建立0到1的10个时间点，再经过sway变换交给内部四阶Runge-Kutta求解器。求解器遍历相邻时间点，共9个积分区间，每区间四次调用DiT预测；这按代码计算为36次预测调用，不能把参数名解释为10次网络评估。默认`guidance_scale=0.5`时，预测内部同时构造有条件和无条件分支，再做classifier-free guidance组合；增加时间点与改变guidance会改变生成预算与结果，质量对照需固定它们。（`sample`、`RungeKutta4ODESolver`、`DiTInputEmbedding.forward`；调用次数为整理者静态计数。）

包装器在调用前将Token2Wav及speaker条件/参考mel转成FP32。Token2Wav构造时，`flash_attention_2`因FP32限制改用SDPA，`eager`也改用SDPA；不能由Thinker或Talker的后端设置断言整条链都使用FlashAttention。DiT求解器返回最后时间点的整段mel`[1,C_{out},2N_c]`，BigVGAN一次处理它，最终clamp到`[-1,1]`、squeeze并移到CPU；包装器返回FP32波形。$C_{out}$是输出mel通道数，与输入mel的$C_m$是不同配置对象。（`Qwen2_5OmniToken2WavModel`、`Qwen2_5OmniToken2WavBigVGANModel.forward`、包装器末段。）

## 6. 返回时序与部署条件

包装器的实际顺序是

```text
processor输出 → 完整Thinker.generate
             → 构造全部回答表示 → 完整Talker.generate
             → 截取code → 整段DiT求解 → 一次BigVGAN
             → 返回(text_sequences, waveform)
```

返回文字含原prompt与回答的ID，不是已经decode的字符串；音频开启时限制`B=1`，DiT采样也另检查batch为1。`return_audio=None`根据Talker是否启用决定，关闭音频则直接返回Thinker结果。Talker/Token2Wav需初始化，speaker还依赖`from_pretrained`加载的`spk_dict.pt`；源码归档不能替代这些实际模型文件。（包装器的`enable_talker`、`from_pretrained`、`generate`。）

本路径没有对外yield音频块的循环。内部局部窗口、模型具备的流式设计以及调用者自行播放返回波形，是三个不同层次。首包、持续输出速率、峰值内存与RTF需要匹配runtime测量；对量化而言，还应分别记录Thinker、Talker、DiT与vocoder的覆盖、dtype、采样预算和speaker，并按[TTS量化评测](tts-quantization-evaluation.md)同时核对内容一致性、自然度和真实算子执行。

## 来源身份与核读范围

- Summer-Summer/transformers，分支`hf-4.53.2`，commit `37f8b0b53512e6aae0cfd15746c133c101783178`，包内版本`4.53.2`；`https://github.com/Summer-Summer/transformers/tree/37f8b0b53512e6aae0cfd15746c133c101783178`。静态核读Omni的processor、相关configuration，以及`modeling_qwen2_5_omni.py`中的音频/视觉编码衔接、三轴位置、Thinker/Talker所选forward与生成状态、DiT窗口/条件/求解、BigVGAN输出和整体包装器；另核Whisper采样率/特征提取与通用chat template选择。
- 未逐模块审核全部网络、训练与损失、通用加载/生成器及attention设备实现；未取得发布checkpoint配置、模板、speaker实体和权重，未验证模型加载、语音生成、停止边界、实际低比特、流式首包、质量或设备性能。
