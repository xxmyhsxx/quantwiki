---
title: Qwen2.5-Omni调用链：媒体准备、返回协议与低显存执行
type: implementation
tags:
  - vlm
  - inference
  - ptq
  - deployment
sources:
  - raw/repositories/2026-09-21/qwen2-5-omni
updated: 2026-10-14
---

# Qwen2.5-Omni调用链：媒体准备、返回协议与低显存执行

Qwen2.5-Omni官方仓库的所选调用链把对话拆成模板文本与媒体载荷，交给外部processor和模型，再由demo解码文字、保存波形。仓库还提供一个专用低显存模型：按阶段搬移模块，把整段语音code分块恢复为mel和wave，最终仍一次返回完整波形。内部按块计算与用户提前收到音频是不同的执行条件。

本页核对官方仓库固定commit的工具、demo消费与专用低显存路径。模型结构由[Thinker-Talker基础](../fundamentals/model/multimodal-perception-and-streaming-speech-generation.md)承接；另存的[transformers模型执行](qwen25-omni-model-generation-execution.md)解释所选processor、Thinker/Talker状态与Token2Wav内部，但其`hf-4.53.2`版本不能替代本仓库要求的`4.52.3`。这里没有取得checkpoint配置、模板、speaker实体或权重，也没有运行模型。

## 1. 模板文本与媒体载荷共同构成模型输入

Web demo先把用户上传的文件按MIME类型转成`image`、`video`或`audio`消息，文字消息保留原role和内容，并在历史前加入system消息。随后`apply_chat_template(..., add_generation_prompt=True, tokenize=False)`生成含控制标记的文本；`process_mm_info`按会话、消息和content顺序分别读取音频、图片与视频，返回`(audios, images, videos)`。这三个集合是媒体载荷，还不是展开后的语言token、mel特征或位置ID。（`web_demo.py:format_history/predict`；`qwen-omni-utils/src/qwen_omni_utils/v2_5/__init__.py:process_mm_info`及两个媒体处理文件。）

外部`Qwen2_5OmniProcessor`接收模板文本与三个媒体集合，生成模型输入；普通demo用`return_tensors="pt"`和`padding=True`，再将结果移到`model.device`及`model.dtype`。processor决定媒体特征、占位长度和时间坐标怎样匹配模型，工具负责读取与采样。核到工具输出，不能跳过processor、checkpoint模板与配置便宣称已经核定最终token布局。（`web_demo.py:108–115`；README “Transformers Usage”。）

`use_audio_in_video`连接三处责任：工具决定是否提取视频音轨，processor决定是否组织音画输入，生成入口接收同一条件。README明确要求三处设置一致，尤其多轮会话应维持同一约定；普通Web demo三处都传`True`。其语音模式还使用README指定的Qwen虚拟人system prompt与Chelsie/Ethan声音选择。该提示是作者给出的使用条件，源码调用本身不证明任意改写后的提示都能保留语音行为。（README “Use audio in video”“Prompt for audio output”；`web_demo.py:38–41、108–115`。）

两个低显存demo却只在工具和`generate`显式传`use_audio_in_video=True`，processor调用没有传这一参数。外部`4.52.3`的默认与处理行为未在本条目归档或核对，因此不能认定这些示例已保持端到端一致，也不能依据另存`4.53.2`的默认值宣称已经复现旧版错误。（`low_VRAM_demo_awq.py:158–165`；`low_VRAM_demo_gptq.py:116–123`。）

## 2. 音频数组的采样率前提与视频的采样元数据

工具把文件、URL或base64音频交给`librosa.load(sr=16000)`，读取区间由`audio_start/end`指定；启用视频音轨时则使用`video_start/end`。视频先经`av.open`检查是否存在audio stream，`use_audio_in_video=True`且没有音轨会触发断言，因此这一开关不能无条件套给静音视频。（`audio_process.py:process_audio_info`。）

直接传NumPy数组走另一条分支：只拒绝多于一维的数组，按`int(16000*start/end)`截取，没有采样率参数、读取文件头或重采样。调用者需先提供符合16kHz含义的单声道数组。作为静态教学反例，48000个样本若原本表示48kHz的一秒音频，直接传入不会被转换成16000个样本；后续按16kHz解释会对应三秒。这与“文件会被重采样到16kHz”是不同前提，不能由数组类型推定正确时长。（`audio_process.py:48–54、84–90`；时长换算为整理者计算。）

视觉工具把图像尺寸调整到28的倍数并约束像素范围。视频文件路径按选定reader取得`[F,C,H,W]`帧张量，默认目标采样率为2 FPS；`fps`与`nframes`互斥。显式`nframes`分支只round到2的倍数；FPS分支才按区间时长与目标FPS计算数量、施加最小/最大帧数和总帧数限制，再floor到2的倍数。两条分支最后都检查`2 <= nframes <= total_frames`，因此显式数量不能被概括成同样经过min/max夹限。reader用均匀索引取帧后，返回`sample_fps = nframes / total_frames * video_fps`。`fetch_video`再结合帧数、逐帧与总像素预算resize，输出float帧张量。若输入已提取的帧列表，工具按图片处理，帧数为奇数时复制末帧补齐为偶数。（`vision_process.py:smart_resize/smart_nframes`、三个`_read_video_*`与`fetch_video`。）

`process_vision_info(..., return_video_kwargs=True)`会额外返回`{'fps': video_sample_fps_list}`；经`process_mm_info`则得到音频、图片、视频和kwargs四项。默认三项调用不返回该字典，普通Omni demo也没有向processor传入有效FPS。因而输入原视频时长相同，不代表调整FPS、夹限帧数或裁剪后仍有相同序列和时间坐标；需要继续核对采样元数据如何进入匹配processor。kwargs含FPS列表也不证明所有版本processor都接受这种列表接口，本页未核`4.52.3`的接收实现。（`vision_process.py:473–498`；`v2_5/__init__.py:11–14`；`web_demo.py:110–112`。）

动态帧数和resize怎样影响语言序列预算，由[视觉token与时间坐标](../fundamentals/model/vision-language-model-tokens-and-quantization.md)承接；工具采样密度、视觉网格和三轴位置ID应在具体调用中分别核对。

## 3. 普通demo消费二元组，再保存24kHz WAV

普通`predict`先等待一次`model.generate(**inputs, speaker=voice, use_audio_in_video=True)`返回`text_ids, audio`。它随后解码文字、取解码结果的最后一行，才第一次yield文字更新；再将波形乘32767并转int16，写成24kHz WAV缓存，第二次yield音频文件路径。UI得到的是已经生成的文字与WAV文件，未在这里逐块消费生成中的wave。（`web_demo.py:115–128`。）

这个消费层说明两种可见更新，但不提供文字首token或语音首包的提前返回证据。`voice_chatting.ipynb`的inference同样取`output[0]`解码文字、`output[1]`作为波形，完成后以24kHz播放；screen/video information的所选inference则同时关闭视频音轨和语音输出，返回文字。音轨输入与语音输出是两个开关，视频理解不必产生语音。（对应cookbook的inference代码cell。）

专用低显存demo使用另一个返回协议：`output[0]`是文字序列，`output[1]`是离散speech codes，`output[2]`才是wave，之后flatten、移到CPU并保存24kHz WAV。它们来自本地`modeling_qwen2_5_omni_low_VRAM_mode.py`，不能沿用普通二元组示例，把codes当作PCM波形保存。（两个`low_VRAM_demo_*.py:video_inference`与末段；本地模型`Qwen2_5OmniForConditionalGeneration.generate`末段。）

## 4. 低显存路径轮换模块，但仍先完成两段自回归生成

所选低显存CUDA音频输出路径的主要顺序是：

```text
Talker/Token2Wav移到CPU，Thinker移到CUDA
→ 完整Thinker.generate，保留文字和隐藏状态
→ Thinker移到CPU，保留所需embedding在CUDA，Talker移到CUDA
→ 完整Talker.generate，截取speech codes，Talker移到CPU
→ Token2Wav移到CUDA，按块恢复mel/wave并累计
→ Thinker移回CUDA，返回(text_sequences, speech_codes, waveform)
```

阶段搬移改变模块同时驻留的范围，但Thinker结果、回答表示和完整codes仍是后续条件；它没有在Talker每生成一个code时立即向用户返回声音。末尾还把Thinker搬回CUDA，不能把实现概括成所有时刻都只有一个模块占GPU。`torch.cuda.empty_cache()`调用也不能代替峰值内存实测。（本地模型`generate:4695–4698、4753–4770、4838–4864`。）

音频输出限制语言batch为1，并依赖已经初始化的Talker/Token2Wav与speaker字典。该专用入口在检查条件前无条件调用`self.talker.to("cpu")`与`self.token2wav.to("cpu")`；因此不能把普通README的`disable_talker()`用法直接套到它，假定删除模块后仍可顺利走文字分支。这里核的是上述有语音模块的CUDA路径，未验证CPU-only、其他device map或停止边界。（本地模型`enable_talker/disable_talker/load_speakers/generate`。）

## 5. code、mel与波形块使用不同单位

本地Token2Wav收到完整codes后，总`generate`明确传`return_audio_in_chunk=True`，进入内部`dit_streaming`。它依次把已有codes放入buffer，首块不带左上下文，积累48个当前code和12个右上下文后求解mel；后续保留24个左上下文和12个右上下文，再加入下一段当前code。求解后裁去左右重叠的mel，把保留部分yield给vocoder；尾段不再等待不存在的右code。令$R$为本地模型从外部configuration取得的`repeats`，48个当前code对应保留的$48R$个mel位置；本归档未取得checkpoint实际$R$，不能直接代入其他版本的默认值。（本地模型`Qwen2_5OmniToken2WavModel.forward:4438–4486`及总`generate`末段。）

| 参数 | 所选代码的单位与作用 |
| --- | --- |
| DiT当前48、左24、右12 | speech code位置；决定buffer何时求解和保留多少上下文 |
| DiT裁切长度乘`repeats` | mel序列位置；把code重叠区换成需裁掉的mel长度 |
| vocoder左20、右20 | mel位置；为波形边界保留上下文并裁掉重复区 |
| `vocoder_upsample_rate=240` | 裁切的硬编码换算：240个波形样本/mel位置；将mel裁切长度换成wave样本数 |
| WAV `samplerate=24000` | 波形样本/秒；保存或播放的采样率 |

vocoder处理mel buffer后，去掉对应左右波形边界，把保留部分拼接到CPU上的`waveform_output`。最终Token2Wav **return累计波形**，总包装器再一次返回三元组。内部yield的对象是mel块，外部return的对象是整段wave；前者不能证明首包已经送到播放器。右上下文还意味着块计算需要后续条件，不能称为零lookahead。（本地模型`Token2Wav.forward:4488–4514`；BigVGAN的`forward`末段；总`generate:4854–4864`。）

分块分支仍预先建立30000个mel位置的noise和重复3000个位置的speaker条件，并保留完整codes与累计wave。因此它改变所选求解窗口，却没有由源码证明端到端内存与输出长度无关；固定buffer的长序列边界和实际峰值均未运行验证。（本地模型`Token2Wav.forward:4440–4444、4498–4514`。）

输入视频的2秒音画分段、这里的code buffer、mel上下文和波形样本是不同坐标。即使WAV采样率已知，也不能把48个code直接称为48个样本或48毫秒；码率、`repeats`和匹配模型配置仍需核定。首包与RTF另受前面两段自回归、搬移、分块求解及播放调度影响。

## 6. Thinker低比特与half/Euler形成联合配置

README把GPTQ-Int4/AWQ描述为Thinker权重量化。适配器进一步定位到`thinker.model.layers`：GPTQ列出各层attention的Q/K/V/O与MLP up/gate/down投影，AWQ返回Thinker decoder层并排除visual转换。外部GPTQModel/AutoAWQ负责加载与实际低比特算子，本次没有取得checkpoint的量化配置与权重，也没有核定这些依赖的packing和设备kernel；该适配范围不能扩写成整个Omni、Talker、codec或KV都为4bit。（README “GPTQ-Int4 and AWQ Usage”；两个低显存demo的适配类。）

专用模型还在`enable_talker`对Token2Wav调用`.half()`，生成时把speaker条件与参考mel转half，最后返回`wav.float()`。源码保留的“Token2Wav必须fp32”诊断文字不覆盖这些实际转换，输出float也不表示中间计算全程FP32。与另存transformers的FP32整段Token2Wav相比，这是不同数值路径，未做运行或音质验证。（本地模型`enable_talker`、Token2Wav构造及`generate:4854–4864`。）

`sample_chunk`采用Euler更新：在0到1之间建立`num_steps`个时间点，经sway变换后，对相邻点逐次做`y ← y + v(t,y)Δt`。默认10个点只有9个积分区间，因此是9次更新；默认guidance又组合有条件/无条件预测。时间点数、solver、guidance与dtype都改变生成预算和数值结果，不能将它们的变化归为纯权重量化效果。（本地模型`sample_chunk:4325–4378`；更新次数为整理者静态计数。）

README的7B对照中，Seed-TTS test-hard、Chelsie的WER原表数值为原模型8.7、GPTQ-Int4 10.3、AWQ 8.88（表头未另标单位）；这项内容一致性指标不证明自然度或统计等价。其浮点基线也不能与技术报告同名任务的分数默认为同checkpoint和采样配置。README另称变体因量化和CPU卸载略慢；“显存降低超过50%”的表格同时伴随卸载、分块、half与Euler等变化，缺少完整硬件和共同运行协议，不能据此证明纯4bit端到端加速。（README “GPTQ-Int4 and AWQ Usage”，评价与显存表。）

配对比较需固定媒体准备、提示、speaker、采样/solver预算与runtime，分别测量自然度、转录、首包、持续输出、峰值内存和真实算子覆盖；这套判断由[TTS量化评测](tts-quantization-evaluation.md)承接，不能把README的单项WER和显存结论合成同时保质、提速与节能。

## 7. 版本对应关系与外部实现边界

| 材料 | 已核身份与适用范围 |
| --- | --- |
| 普通Quickstart/Web demo | 指定`transformers==4.52.3`；模型与processor由外部包导入，模型文件从外部checkpoint加载 |
| `requirements_web_demo.txt` | 固定`qwen-omni-utils==0.0.4`，也固定上述transformers版本 |
| 同一归档的工具源码 | `qwen-omni-utils/pyproject.toml`登记`0.0.8`；本页媒体读取机制依据这份保存源码，未证明与demo安装的`0.0.4`同版 |
| 专用低显存示例 | 使用本地模型文件；README要求GPTQModel `2.0.0`或AutoAWQ `0.2.9`，仍依赖外部processor、configuration与加载/attention实现 |
| 另存transformers执行页 | `hf-4.53.2` fork的独立源码核对；可比较已核行为，不能证明官方`4.52.3`与发布模板的匹配 |

README另列DashScope流式API、Qwen Chat实时交互和vLLM路径。它们是不同服务/后端的入口声明，本次没有取得对应服务链，不能用它们替代普通Web demo或低显存包装器的实际返回时序。

## 来源身份与核读范围

- QwenLM/Qwen2.5-Omni，commit `d8a31ca56c0456b6edfcbcbf4bdbb6ae2200ef42`；`https://github.com/QwenLM/Qwen2.5-Omni/tree/d8a31ca56c0456b6edfcbcbf4bdbb6ae2200ef42`。固定ZIP与24个展开文件按封存名称逐字节对应，CRC无错；静态核读README的使用/量化/服务条件、Web demo输入与返回链、音频/视觉工具及版本、voice/screen/video/multi-round cookbook的代表调用、两个低显存demo与本地模型的总生成、Token2Wav分块、Euler、dtype和卸载路径。
- 未核普通transformers `4.52.3`模型/processor、utils `0.0.4`实现、checkpoint配置/模板/speaker实体/权重、外部量化loader/packing/kernel、全部本地网络和训练分支；未验证模型加载、长code/边界块、停止方式、语音质量、首包、RTF、峰值内存、CPU-only或设备执行。原件对应、静态调用核对与实际运行分别保留。
