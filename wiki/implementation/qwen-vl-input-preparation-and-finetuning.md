---
title: Qwen-VL 的视觉输入准备与微调样本
type: implementation
tags:
  - vlm
  - attention
  - inference
sources:
  - raw/repositories/2026-09-21/qwen3-vl
updated: 2026-10-14
---

# Qwen-VL 的视觉输入准备与微调样本

Qwen3-VL 官方仓库提供媒体加载工具、推理示例和微调脚本。它们把图像、采样视频与文字组织成模型输入，并在训练时构造回答标签、三轴位置和独立样本的 attention 边界。沿这条路径，需要同时保持媒体顺序、采样时间、视觉占位、监督位置与样本边界一致；只核模型名称或最后的 token 数不足以确认这些关系。

本页核对 QwenLM/Qwen3-VL 的固定快照。仓库中的 `qwen-vl-utils` 声明版本为 0.0.14；主 README 要求 Transformers ≥4.57.0，微调 README 声明 4.57.0.dev0，web requirements 另固定外部 HF commit `7aa888b7fa477d13153ffbfe107dfbd6c696014a`。这些依赖没有在本轮取得或运行。已核的 [Qwen2.5-VL 内部路径](qwen25-vl-visual-input-and-position-execution.md)属于另一份 4.53.2 快照，该版本没有 Qwen3-VL；不能用它补齐这里外部 processor、DeepStack 或交错 MRoPE 的实现。

下文工具文件 `vision_process.py` 位于 `qwen-vl-utils/src/qwen_vl_utils/`；微调文件路径相对于 `qwen-vl-finetune/`。简写 `data_processor.py/rope2d.py` 指 `qwenvl/data/` 下的文件，`train_qwen.py/trainer.py` 指 `qwenvl/train/` 下的文件。模型多层视觉输入与文字时间戳的结构作用见 [VLM 的视觉输入与量化对象](../fundamentals/model/vision-language-model-tokens-and-quantization.md)，本页继续解释这些输入怎样送到所选调用接口。

## 1. 媒体工具先决定空间尺寸和采样集合

`process_vision_info` 按会话、消息、content 的出现顺序收集媒体，再分别生成 image 和 video 列表；同类媒体的先后关系因此需要与模板中的占位顺序一致。工具输出的是 PIL 图像、视频张量及采样 metadata，还没有执行 ViT 或语言模型。（`vision_process.py::extract_vision_info/process_vision_info:483–534`。）

图像支持 PIL、文件路径、HTTP URL 和 base64。`fetch_image` 将其转换为 RGB；RGBA 先在白色背景上合成。`smart_resize` 把高、宽调整到 `image_patch_size × SPATIAL_MERGE_SIZE` 的倍数，常规路径再按像素上下限调整面积。代码中空间合并大小为2，Qwen3-VL 示例显式传 patch16，因而对齐因子为32；工具函数的默认 patch 仍是14，调用者不能省略这项后便假定使用了 Qwen3 的网格。（`vision_process.py:24–29,56–141`；主 README 的 New qwen-vl-utils Usage。）

这里的像素预算只约束所选预处理路径。明确提供 `resized_height/resized_width` 时，图像和视频调用 `smart_resize` 的分支没有传入该媒体计算出的 min/max 像素值。视频常规上限还兼顾每帧大小、采样帧数和总预算，最低分辨率可以抬高预算。因此不能把 `total_pixels` 当作所有配置都严格满足的完整 context 上限；文字、时间戳和模板边界也要另计。（`vision_process.py:123–139,448–475`。）

### 文件视频与预采样列表有不同的时间依据

文件视频先由解码后端取得原 FPS 和帧数，再按 `fps` 或 `nframes` 二选一决定采样数。显式帧数按因子2取最近倍数；FPS 路径默认2 FPS，施加最小/最大帧数参数并按可用帧数夹限、向下取偶数；默认min/max参数为4/768，最终统一检查结果至少为2且不超过可用帧数。解码器在所选范围内用 linspace 取帧索引，输出 `[F,C,H,W]`。后端按强制配置、torchcodec、decord、torchvision 的顺序选择；选定后端失败时可以退回 torchvision。（`vision_process.py::smart_nframes:144–181`；各 reader 与 `get_video_reader_backend:184–400`。）

原视频 FPS、实际采样 FPS 和帧索引是三个对象。reader 返回的有效采样率按“采样帧数/所选范围帧数 × 原 FPS”计算，不一定等于请求的 FPS。开启裁剪时，decord/torchcodec 的索引仍相对于原视频，而 torchvision 的索引相对于已读出的片段；metadata 中的 `total_num_frames` 则是片段帧数。外部 processor 怎样处理这些参照和裁剪偏移，本轮未核，不能宣称后端切换后时间戳必然等价。（`vision_process.py:205–225,234–289,310–329,358–377`。）

预采样列表走另一条路径：工具逐帧加载，不再按 `max_frames` 抽样或截断；奇数长度会复制末帧补成偶数。它用 `sample_fps` 和顺序索引 `0,…,F−1` 生成合成 metadata。该接口不能直接表示任意非均匀采样时刻；设置 `raw_fps` 也不会自动把顺序索引换成原视频索引。`sample_fps` 需要数值，主 README 中的字符串 `'1'` 示例与此处直接除法运算有差异。视频最后统一 resize，并转换为 float 张量；这一步不证明已经完成模型所需的归一化。（`vision_process.py:415–475`；主 README 的帧列表例。）

## 2. 模板和 processor 接过媒体，生成接口返回文字

README 的两阶段 Qwen3 调用先用 `processor.apply_chat_template(tokenize=False, add_generation_prompt=True)` 生成模板文字，再调用媒体工具，传入 patch16、`return_video_kwargs=True` 和 `return_video_metadata=True`。工具把每个视频返回为 `(tensor, metadata)`，示例将二者拆开交给 processor，并传 `do_resize=False` 与工具回传的 `do_sample_frames=False`，避免已经 resize 或采样的媒体再走一次同类处理。（主 README:639–652；`vision_process.py:477–534`。）

metadata 为外部 processor 提供时间依据；格式化时间戳文字、视觉占位展开、pixel/grid 的组织则由 processor 完成。本库没有其完整实现，不能只从上面的 metadata 字段补出所有三轴 ID、时间取整或模板规则。微调路径也调用外部 processor，但通过属性配置媒体参数，没有先调用这条工具路径；两条入口的默认值不能相互替代。

HF 示例从同一 checkpoint 名称加载 AutoModel 与 AutoProcessor，将处理结果送到模型 device，再调用 `generate`。示例把返回 IDs 按每行输入 ID 长度裁去前部，再 batch_decode，以得到回答文字。示例没有固定 checkpoint revision，实际权重、词表、模板和 dtype 仍需分别确认。（主 README:145–193,614–662。）

web demo 的 HF 分支用线程启动 `generate`，由 `TextIteratorStreamer` 逐次交付文字。vLLM 分支则把模板文字、媒体和处理参数组成 `prompt/multi_modal_data/mm_processor_kwargs`，交给同步 `LLM.generate` 后遍历结果；UI 的 yield 本身不证明逐 token 流式返回。demo 的消息转换还只保留 image/video/text，媒体上的额外 FPS、帧数和像素配置不会沿该转换保留。（`web_demo_mm.py:157–239`。）

视频 cookbook 的本地 inference 采用上述两阶段路径；另一个 API 例手工交错 `<0.0 seconds>` 等文字与图像 URL。后者说明了一种请求格式，不是本地 processor 自动时间戳算法的证据。加载 cell 使用占位 `model_path="***"`，并注明保存的输出来自 tiny test model；这些输出不算本项目的模型运行。（`cookbooks/video_understanding.ipynb`，cell 2、6、20–21。）

## 3. 微调记录先展开，再选择回答标签与三轴位置

dataset 配置给出 annotation 和媒体根目录；名称后缀 `%X` 可选择样本比例。数据读取器为普通记录及 packed 子记录补上媒体根路径，再按对话中 `<image>/<video>` 的顺序消费对应文件。缺少媒体或存在未消费媒体会报错，助手消息只作为文本。processor 的图像像素和视频帧数/FPS属性随后按训练参数配置；真实媒体处理仍由外部 processor 承担。（`qwenvl/data/__init__.py:38–56`；`qwenvl/data/data_processor.py:44–199,247–307`。）

`preprocess_qwen_visual` 对完整消息调用 `processor.apply_chat_template(tokenize=True, return_dict=True, return_tensors="pt")`，得到展开后的 IDs 和媒体张量；本调用没有显式传 truncation/max_length。标签先全部置为 `-100`，再扫描硬编码 ID `77091`，以命中位置后两位为起点。若当前序列内找到 `151645`，就以该终止位置加2作为切片上限，复制到终止 ID 及其后一个存在的位置；未找到时该段保持 `-100`。它依赖具体词表和模板，不能只凭消息的 assistant 角色保证标注正确；代码也未检查至少一个回答标签有效。本函数不做 next-token 移位，外部模型 loss 的移位与归约未核。（`qwenvl/data/data_processor.py:202–240`。）

输入位置与监督位置因此不同。模板与硬编码 ID 匹配时，图像和用户提示作为条件参与前向，回答及其终止边界则按上述切片纳入监督；三轴位置又描述这些位置的空间组织。共同区别见 [回答标签与 mask](../fundamentals/model/vision-language-model-tokens-and-quantization.md#4.%20%E6%A0%A1%E5%87%86%E5%BA%8F%E5%88%97%E3%80%81%E5%9B%9E%E7%AD%94%E6%A0%87%E7%AD%BE%E4%B8%8E%20mask)。记录读取失败时会重试，并可能改取下一条记录，因此 dataset 的登记长度也不证明每条原记录都被实际消费。（`qwenvl/data/data_processor.py:345–387`。）

Qwen3 分支用 `input_ids`、`image_grid_thw/video_grid_thw` 和空间 merge size 构造 `position_ids:[3,1,S]`，S是当前记录的实际展开长度。grid的t/h/w是processor给出的时间与空间patch网格数，区别于原帧数和像素尺寸。每个视频grid按t重复成t项，再将各项t设为1；代码注释假设外部processor已用时间戳文字分隔这些temporal patches。随后按vision-start和硬编码媒体ID找各视觉段，空间网格取整数除法 `h//merge_size,w//merge_size`。文字三个轴同值，视觉段遍历空间坐标；扫描起点按该段的 `t*(h//m)*(w//m)` 个视觉位置前进，m为merge_size，视频拆分后t为1。（`qwenvl/data/data_processor.py:389–427`；`qwenvl/data/rope2d.py::get_rope_index_3:5–91`。）

一个 temporal patch 内的本地时间坐标是0，但最终三轴 ID 还统一加上此前起点与文字段长度；下一段从前段三轴最大值加1接续。因此“局部t为0”不等于“所有最终 time IDs 都是0”，语义坐标也不等于序列行号。Qwen3 的这份函数没有使用传入的 `second_per_grid_ts`，不负责生成时间戳、旋转频段或 DeepStack 注入；调用方忽略返回的 delta，本页也不据此补写 decode 接续。（`rope2d.py:83–122`；`data_processor.py:408–427`。）

## 4. 拼接样本后，累计边界保持段内因果关系

普通 collator 把各记录组成 batch，右 padding IDs 和 labels，再按自身 tokenizer 的最大长度截 IDs、labels 和三轴位置；padding mask为 `input_ids != pad_id`。同批媒体的 pixel/grid 仍完整拼接，没有随 token 截断同步裁剪。因此可能留下“视觉占位已被截掉、媒体仍完整”的输入组合，不能把普通截断理解为安全删除一段多模态输入。（`data_processor.py::DataCollatorForSupervisedDataset:534–601`。）

开启 `data_packing` 时，dataset先独立处理一个 packed item里的每条原记录，再连接IDs、labels、位置和媒体，保留原记录长度。开启 `data_flatten` 或 `data_packing` 任一项，collator就进一步把 batch 拼为 `[1,ΣS_i]`，位置为 `[3,1,ΣS_i]`，不 padding，也不截断总长；只有 packing改变dataset item的组织方式。（`data_processor.py:442–517,604–688`。）

flattened collator 将原记录长度转为 int32 累计边界

$$c=[0,S_1,S_1+S_2,\ldots,\sum_iS_i]$$

并放进名为 `attention_mask` 的字段。这个字段此时不是“每位置0/1”的 padding mask。例如两条长度5和3的记录得到 `[0,5,8]`；它表示两个独立段，不是8个位置的可见性向量。这是依源码构造的教学例。

训练入口在上述 flag 开启时安装专用 patch，让 causal-mask 构造直接透传边界。Qwen3 attention patch对隐藏状态做Q/K/V投影及Q/K norm，消费外部给出的cos/sin旋转Q/K，再把张量换成序列优先并挤掉B=1。helper以相邻边界差求最大段长，将同一累计边界传给FA varlen的Q/K，设置 `causal=True`。这条接口把各原记录各自限定成段，段内保持因果attention；样本位置也保留各自计算的结果，没有因物理拼接统一重新编号。（`qwenvl/train/train_qwen.py:141–143`；`qwenvl/train/trainer.py:33–108,163–252`。）

边界的使用还受版本和dtype条件限制。patch依赖外部HF的具体类及接口，FA设备实现未核；helper接收却没有向FA调用传入dropout、scaling、sliding_window或softcap，虽计算target_dtype也没有转换Q/K/V。因此不能沿注释声称会自动恢复BF16，或把这条训练路径推广为任意cache/inference配置。

训练入口另外加载一份AutoTokenizer并设置CLI的 `model_max_length`，dataset/collator却使用 `processor.tokenizer`，代码未显式同步两者。因此CLI参数不是已经证明的实际输入上限。离线 `tools/pack_data.py` 又用Qwen2VLImageProcessor、手工模板和固定21起始计数估长后按4096装箱，没有Qwen3真实时间戳展开；图像/视频同存时还只计图像分支。4096是估算装箱目标，不能替代实际展开长度核对。（`train_qwen.py:155–188`；`data_processor.py:298–301,678–688`；`tools/pack_data.py:34–139,153–191`。）

## 5. 参数开关决定更新对象，保存调用保留其条件

训练脚本按模型路径名字选择外部dense/MoE类，并把BF16与FA2参数交给加载接口。Qwen3路径且basename含任意字母a时选择MoE，因此这是命名启发式，不能代替真实模型配置。README声明该快照的MoE配方不支持ZeRO-3，所选MoE脚本使用ZeRO-2；HF内部的路由辅助损失与分布式训练仍未核。（`train_qwen.py:92–139`；微调README:317；`scripts/sft_30a3b.sh:8–14`。）

非LoRA路径先按 `tune_mm_vision` 开关整个visual参数，再按 `tune_mm_mlp` 覆盖 `visual.merger`，最后开关language_model参数。脚本只明确访问这一个merger，不能将该开关泛化成所有DeepStack专用merger。代码直接给 `model.lm_head.requires_grad` 赋值，没有遍历head参数；若head是常规模块，这不能替代参数级冻结。外部HF对象及权重绑定未核。（`train_qwen.py::set_model:67–89`。）

LoRA路径先冻结已有参数，再向名为 `q_proj/k_proj/v_proj/o_proj` 的模块交PEFT配置；实际命中的语言或视觉模块需要匹配外部模型树。新增任务因子与固定基座的关系见 [LoRA](../methods/lora.md)。这条脚本的 `lora_dropout or 0.05` 会把显式0改为0.05，不能把参数默认0直接报告为实际dropout。优化器只收集可训练参数，再按weight decay及名称中的merger/visual分组；独立vision学习率仅在非零mm_projector_lr分支内生效。（`train_qwen.py:163–178`；`trainer.py::create_optimizer:316–495`。）

Trainer在发现 `checkpoint-*` 时请求恢复训练，否则开始新训练；结束后保存state并恢复 `use_cache=True`。非DeepSpeed路径在 `should_save` 为真时把state移到CPU并保存，DeepSpeed路径同步后交 `trainer.save_model`；末尾的 `processor.save_pretrained` 没有同样的rank guard。这里核的是保存调用与分支，没有验证实际产物、并发写入或重载。上述训练流程也没有由此证明权重、激活或KV已经量化。（`train_qwen.py:52–64,186–202`。）

本页实际核读媒体工具、所选HF/vLLM与web调用、视频cookbook必要cells，以及微调数据、位置、collator、attention patch、参数和保存主线。没有审计全部评测、cookbook或数据路径；发布checkpoint、模板、完整HF processor/model、FA设备、训练/生成结果及量化质量与性能均未验证。

## 来源身份

- QwenLM/Qwen3-VL，commit `96588727e44c78b25ba03ea03b8e12f7e64fd0da`，`https://github.com/QwenLM/Qwen3-VL/tree/96588727e44c78b25ba03ea03b8e12f7e64fd0da`；所选工具、示例和微调调用的静态核读范围见正文，外部模型依赖与运行保持未核。
