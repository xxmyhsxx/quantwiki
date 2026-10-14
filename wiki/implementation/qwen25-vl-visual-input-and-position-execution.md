---
title: Qwen2.5-VL：视觉输入、三轴位置与生成缓存
type: implementation
tags:
  - vlm
  - attention
  - kv-cache
sources:
  - raw/repositories/2026-09-21/transformers
updated: 2026-10-14
---

# Qwen2.5-VL：视觉输入、三轴位置与生成缓存

Qwen2.5-VL把图像或视频编码成与语言隐藏维一致的向量，替换文字序列中的视觉占位位置，再用三轴MRoPE执行语言prefill。生成后续文字时，视觉输入的影响留在各语言层的KV中，位置编码通过prefill保存的偏移量接续。沿这条路径，需要保持一致的是**媒体顺序、视觉位置数、空间网格、时间间隔与缓存位置**。

本页核对Summer-Summer/transformers的`hf-4.53.2`固定commit，说明`Qwen2_5_VLProcessor`和`Qwen2_5_VLForConditionalGeneration`的静态执行路径。文中源码路径均相对于`src/transformers/`；参数默认值来自类定义，发布checkpoint的配置、tokenizer和聊天模板未核验。这版源码没有Qwen3-VL模型，不能据此补写其DeepStack或交错MRoPE实现。视觉token与量化对象的基础关系见[视觉语言模型中的token](../fundamentals/model/vision-language-model-tokens-and-quantization.md)。

Qwen3-VL 官方仓库的工具、推理示例和微调数据接口另见[视觉输入准备与微调样本](qwen-vl-input-preparation-and-finetuning.md)；它补充调用侧关系，不替换本页的固定 Qwen2.5-VL 内部实现。

## 1. 像素网格与文字占位共同确定输入长度

processor接收已组织的文字`text`以及按出现顺序提供的`images`、`videos`。它先处理媒体，得到像素patch和每份媒体的`grid_thw=(T,H_g,W_g)`，再把文字中的一个`<|image_pad|>`或`<|video_pad|>`展开为

$$N_{\rm visual}=\frac{T H_g W_g}{m^2}$$

个同类占位token，最后tokenize文字。$T$是时间patch数，$H_g,W_g$是**空间合并前**的patch网格，$m$是每轴的空间合并大小；因此grid不能直接理解为最后语言序列的二维大小。每类媒体各有一个索引，随batch文字中该类占位的出现顺序前进；单靠总数相等，不能保证两张图没有交换。聊天模板负责媒体段及边界token的组织，本轮没有核对任何发布模板。（`models/qwen2_5_vl/processing_qwen2_5_vl.py::__call__:95–206`。）

图像processor默认`patch_size=14,temporal_patch_size=2,merge_size=2`。resize把高、宽调到`patch_size*merge_size=28`的倍数，并受像素上下限约束；随后rescale、normalize和重排。单张图为凑齐时间patch复制一帧，得到$T=1$。像素输出不是通常的`[B,3,H,W]`：一行是一个时空patch，宽度为$C\tau p^2$，C是通道数、$\tau$是每个时间patch的帧数、p是空间patch边长；默认RGB下为$3\times2\times14^2=1176$。一份媒体有$P=T H_gW_g$行，排列顺序让空间相邻的$m\times m$行连续，为后续merger保留四邻关系。（`models/qwen2_vl/image_processing_qwen2_vl.py::_preprocess:230–307`。）

例如，一张**预处理后**为$224\times224$的RGB图得到`grid_thw=(1,16,16)`、`pixel_values`形状`[256,1176]`，文字中则需要64个视觉占位。这里的256是ViT输入位置数，64是语言输入位置数；这是依照默认形状的整理者算例，不是模型运行结果。

视频端另有采样条件。`Qwen2VLVideoProcessor`类默认`do_sample_frames=False`，因而外部已经采好的帧不会默认再采样。启用内部采样时，`num_frames`与`fps`互斥。显式`num_frames`分支只将目标数round到时间patch的倍数；FPS分支必须有原视频metadata，才按FPS估计并施加最小/最大帧限制，再向下取倍数。两条路径随后都检查目标数不超过原有帧数。默认时间patch为2，奇数帧输入在重排前复制末帧；其他时间patch配置的补帧行为本页不作推广。（`models/qwen2_vl/video_processing_qwen2_vl.py:99–119,122–191,218–307`。）

与此同时，Qwen2.5-VL processor用调用中的FPS计算`second_per_grid_ts=temporal_patch_size/fps`，未提供FPS时取2.0；标量应用于所有视频，也接受与视频数一致的列表。这个数表示**相邻时间网格的秒间隔**，不是原视频FPS。代码没有从最终采样索引重新推算每段实际间隔：外部采样、内部帧数限制与传入FPS不一致时，不能仅凭该字段宣称时间已经准确对齐。（`processing_qwen2_5_vl.py:157–170`。）

## 2. ViT按窗口重排，merger恢复媒体内顺序

视觉入口将每行1176个默认像素值view为`[3,2,14,14]`，用kernel与stride均为`[2,14,14]`的Conv3d映射到视觉宽度$D_v$，得到`[P,D_v]`。输入先转换为投影权重的dtype；这条路径没有由此实施整数或低比特量化。（`modeling_qwen2_5_vl.py::Qwen2_5_VisionPatchEmbed:65–88`。）

ViT的空间RoPE按合并前的height/width网格建立位置，并按同样的四邻顺序排列。`get_window_index`在**合并后的网格**上生成索引：默认112像素窗口对应$4\times4$个合并组、即$8\times8$个原patch。forward将特征暂时view为`[P/4,4,D_v]`，以`window_index`重排组，再展开回`[P,D_v]`；RoPE也同步重排。因此改的是计算顺序，四邻组仍完整，空间坐标也没有被改成窗口内的新坐标。（`modeling_qwen2_5_vl.py::rot_pos_emb/get_window_index/forward:359–477`。）

窗口边界通过两种累积长度传给attention：

| ViT层所用边界 | 一段包含的patch | 可见范围 |
| --- | --- | --- |
| `cu_window_seqlens` | 一个空间窗口中的有效四邻组，长度乘4 | 窗口内部双向可见 |
| `cu_seqlens` | 一份媒体的**一个时间网格**中的$H_gW_g$个patch | 该时间网格内空间全可见 |

类默认32层，零起始的7/15/23/31层选第二种边界，其余选窗口边界。`cu_seqlens`将每份媒体的$H_gW_g$重复T次再累积，所以视频的full层没有把全部T个时间网格合成一段；不同图像也不相互attention。时间patch本身已在Conv3d中融合两个帧，语言层随后再混合各时间网格。索引padding使用−100补齐窗口，实际特征会过滤这些位置；重复的空段边界通过`unique_consecutive`删除，不为ViT新增伪patch。（`configuration_qwen2_5_vl.py:36–67`；`modeling_qwen2_5_vl.py:386–433,444–495`。）

这组可见范围并不保证所有后端获得相同成本。`flash_attention_2`分支不构造4D mask，传入分段累积长度；其他分支先分配`[1,1,P,P]`的dense mask，只开放段内位置，eager还会计算整个打分张量。源码注释说明这种mask路径并非与FA2 varlen逐项完全匹配；本页未运行底层后端，因此不能把论文的窗口复杂度直接当作所有分支的显存或时延。（`modeling_qwen2_5_vl.py::_prepare_attention_mask:434–453`；视觉attention接口见`212–279`。）

各ViT层完成后，merger先对**单patch的$D_v$维**做RMSNorm，再view为`[P/4,4D_v]`；两次Linear中间用GELU，输出`[P/4,D]`，$D$为语言隐藏维。这里是拼接后学习投影，未按四邻平均。输出还处在窗口组顺序，最后以`argsort(window_index)`恢复processor的媒体内顺序。这一步使后续占位scatter可以按原网格消费特征。（`modeling_qwen2_5_vl.py::Qwen2_5_VLPatchMerger:121–136`；`forward:499–503`。）

## 3. scatter把视觉向量放回图文序列

语言输入最初是`embed_tokens(input_ids)`产生的`[B,S,D]`，B为batch数，S已经包含展开后的视觉占位。视觉特征按每份媒体的`grid_thw.prod()/4`分段，随后依次拼接；`input_ids==image_token_id`或`video_token_id`确定待替换位置。mask沿D维展开，`masked_scatter`把特征按顺序写入这些位置。因此这里不再增加S，普通文字和视觉边界token继续使用各自embedding。（`modeling_qwen2_5_vl.py::get_image_features/get_video_features:1171–1201`；`Qwen2_5_VLModel.forward:1260–1295`。）

非torchdynamo编译路径先比较占位总数与特征行数，不一致会报错；编译时跳过这项显式检查。检查能发现数量问题，不能发现同类媒体交换或错误配对。视觉特征还会转到语言embedding所在device及dtype，视觉端dtype因此不必与语言端相同。

如果调用者直接提供`inputs_embeds`，上述视觉提取和scatter整个分支被绕过：此时图文向量由调用者负责，不能期待同时传入的pixel_values自动再注入。位置编码仍有自己的输入要求；直接调用时需提供可用于生成三轴ID的token/grid，或明确提供匹配的`position_ids`，不能只传一份无位置信息的混合embedding。（`modeling_qwen2_5_vl.py:1260–1338`。）

## 4. 三轴ID区分网格坐标与缓存序号

`get_rope_index`在prefill阶段依据token ID和grid建立`position_ids[3,B,S]`，轴顺序为time、height、width。它先移除`attention_mask==0`的padding，按`vision_start`后紧接image/video token识别媒体段，再按各类媒体的出现顺序消费grid。这要求模板边界、占位展开和媒体列表共同匹配，S的总数一致不是充分条件。（`modeling_qwen2_5_vl.py::get_rope_index:1046–1119`。）

各段位置连续接续，但不等于序列行号：

- 文字段的三个ID相同，按当前起点加0、1、2……。
- 图像段的本地time ID为0，height/width遍历`[H_g/m,W_g/m]`。
- 视频段的空间ID一样，time ID由时间网格序号和秒间隔计算。
- 下一段的起点取前面三轴ID的**全局最大值加1**，不按视觉段的token数量平移。

视觉坐标还统一加上此前的起点及文字段长度，因而图像time的“固定0”指本地坐标；最终送入MRoPE的值通常不是0。空间合并使三轴最大值可以远小于语言序列长度，prefill因此保存

$$\delta_b=\max(\mathrm{position\_ids}_b)+1-L,$$

max在该样本的有效三轴ID上取值；L是batch中`input_ids`的**padding后宽度**，不是该样本有效位置数。返回`rope_deltas`实际形状为`[B,1]`；这个差值连接多模态坐标与之后的缓存序号，也承担padding宽度的影响。（`modeling_qwen2_5_vl.py:1114–1170`。）

### 视频秒间隔先转整数的版本边界

这份固定源码先用`torch.arange(T)`得到整数时间网格序号r，再将`second_per_grid_t`转成该arange的dtype，之后才乘`tokens_per_second`并转long。因此对正的秒间隔$\Delta t$，可将本地time ID解释为

$$p_t(r)=\operatorname{long}\!\left(r\,\operatorname{int}(\Delta t)\,k\right),$$

$k$是`vision_config.tokens_per_second`，类默认4；这是对源码转换顺序的整理者表达，不是论文的通用定义。（`modeling_qwen2_5_vl.py:1122–1136`；`configuration_qwen2_5_vl.py:43–46`。）

例如，默认时间patch为2、调用FPS为4时，processor给出$\Delta t=0.5$秒；进入此处先变成整数0，所有时间网格的本地time ID都成为0。其空间ID、序列顺序与因果可见关系仍存在，但这条time轴不能表达网格之间的真实时间差。默认FPS为2对应$\Delta t=1$，不会触发这个0.5秒条件。这个静态反例说明，核对FPS时还必须核对转换顺序；本轮没有运行视频生成或衡量其质量影响。

## 5. prefill保存KV与偏移，decode只接入新文字

语言RoPE将三轴ID各自与频率相乘，在关闭autocast的FP32路径计算cos/sin，再转回hidden的dtype。`mrope_section`决定不同通道段取哪一轴的cos/sin；`apply_multimodal_rotary_pos_emb`只旋转Q和K，V不旋转。这是语言端三轴MRoPE，区别于ViT在第2节使用的二维空间RoPE；具体频段长度取实际配置，本轮未核发布checkpoint的分配。（`modeling_qwen2_5_vl.py::Qwen2_5_VLRotaryEmbedding/apply_multimodal_rotary_pos_emb:531–626`。）

prefill的图文`[B,S,D]`进入共享语言层，各层投影为Q/K/V，再将旋转后的K与未旋转V更新到cache。普通`use_cache=True`且未给cache对象时，文本模型建立`DynamicCache`；其每层保存`[B,H_kv,L_cache,d]`，$H_{kv}$是KV head数、$L_{cache}$是已保存的序列长度、d是head宽度。cache沿序列轴拼接新K/V，再交给attention读取。KV保留KV-head粒度，eager需要时才按GQA关系扩展到query heads。缓存里保存的是各层上下文状态，并非一份可以在下一轮重用的原始图像embedding。（`modeling_qwen2_5_vl.py:669–715,858–945`；`cache_utils.py::DynamicCache.update:516–561`。）

语言mask按配置建立full causal或sliding causal映射；它与ViT的窗口/full双向mask不同。新文字能读取哪些历史位置由这份语言mask决定，不能用视觉full层边界推断decode可见范围。低比特KV的对象与存储粒度见[KV cache量化](../theory/kv-cache-quantization-objects-and-granularity.md)；本路径的浮点cache操作不证明已启用KV量化。

进入常规缓存decode后，通用generation准备函数裁出尚未处理的token；Qwen2.5-VL的override在`cache_position[0]!=0`时将image/video的pixel_values置空，同时令`position_ids=None`，由模型forward接续三轴位置。若新输入长度为q、该轮cache起点为c，则三个轴使用

$$p_a(b,j)=c+\delta_b+j,\qquad a\in\{t,h,w\},\quad 0\le j<q.$$

常规逐token生成时q=1，c是已缓存的序列位置数；这时三个轴再次相同，并衔接prefill的最大坐标。视觉结果通过已有语言KV影响新文字，无需每轮重复跑ViT。（`generation/utils.py::_cache_dependant_input_preparation/prepare_inputs_for_generation:456–600`；`modeling_qwen2_5_vl.py:1318–1329,1539–1586`。）

这里的$\delta_b$存放在模型实例的`self.rope_deltas`中，并随输出返回；它不是DynamicCache自动推导出的字段。该forward虽接收`rope_deltas`参数，阶段判断与接续仍使用实例字段；仅把返回的偏移作为同名参数传入，不能认为已恢复实例状态。直接接续cache或复用实例时，position IDs、cache_position和对应prefill偏移须一起匹配。非编译forward用cache起点为0或空cache判断prefill；编译路径则依输入长度不为1判断。手动输入、编译与其他生成策略不能仅凭“当前只处理一个token”套用全部阶段判断。（`modeling_qwen2_5_vl.py:1216–1330`。）

## 6. 核读范围与执行证据

本页沿processor、图像/视频patch重排、ViT分段、merger、视觉scatter、三轴ID、语言Q/K与DynamicCache及generation输入准备作静态核对。已核源码与固定归档的对应关系；没有安装依赖、导入模型、加载checkpoint、编译attention、执行GPU生成或量化评测。图像64位置的算例与0.5秒反例均是由形状/转换顺序推出的教学判断。

尚未核对发布processor/tokenizer/chat template、所有图像或视频解码路径、底层SDPA/FA2设备内核、静态缓存及所有生成策略。因此本页支持沿所选普通路径检查输入对象和状态接续，不提供已跑通的部署配方、质量结论或性能收益。

## 来源身份

| 来源 | 固定版本 | URL |
| --- | --- | --- |
| Summer-Summer/transformers | `hf-4.53.2`，commit `37f8b0b53512e6aae0cfd15746c133c101783178` | `https://github.com/Summer-Summer/transformers/tree/37f8b0b53512e6aae0cfd15746c133c101783178` |
