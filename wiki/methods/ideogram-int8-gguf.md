---
title: Ideogram 4.0 的 INT8/GGUF PTQ：层保护与生成评测协议
type: method
tags:
  - diffusion
  - activation-quantization
  - mixed-precision
  - evaluation
  - data-format
sources:
  - raw/papers/2026-09-21/int8-gg/paper.pdf
  - raw/repositories/2026-09-21/llama-cpp/source/ggml/src/ggml-common.h
  - raw/repositories/2026-09-21/llama-cpp/source/ggml/src/ggml-quants.c
  - raw/repositories/2026-09-21/llama-cpp/source/ggml/src/ggml.c
updated: 2026-10-12
---

# Ideogram 4.0 的 INT8/GGUF PTQ：层保护与生成评测协议

这项研究在RTX3090上比较Ideogram 4.0的FP8、NF4、混合保护INT8 W8A8和GGUF Q4_K。方法复用SmoothQuant、动态激活网格和少量BF16层保护；更有用的知识增量是解释：提示格式、整体图像评分、文字正确率和相对参考模型的保真度，分别改变我们怎样判断量化质量。（论文 v2（`https://arxiv.org/abs/2606.12280v2`），§3–7、表1–5。）

本页核固定11页PDF的正文、方法、五张主表和代表图，并回查源文件；GGUF格式及两项一般性说法另与既有llama.cpp固定快照核对。没有取得作者量化/评测代码或权重，也没有生成图像、编译或运行GPU。公开记录标题与PDF首页标题不同，具体身份见页末。

## 1. 两个DiT分支都需要计费

论文描述9.3B的single-stream flow-matching DiT，以条件/无条件两套独立权重实现classifier-free guidance；每个分支有211个linear，FP8权重约9.3GB。Qwen3-VL文本编码器与VAE保持发布精度，不属于这里的量化主体。两分支的18.6GB磁盘权重既不包括所有模型组件，也不等于峰值VRAM。

参考是已发布FP8 checkpoint在该平台反量化为BF16后的输出。作者称没有公开BF16 checkpoint，因此“接近FP8”不能改写成接近原始BF16模型；BF16计算也不会恢复FP8保存之前丢失的信息。

## 2. 动态网格与离线缩放是两个步骤

未保护linear使用逐输出行INT8权重、逐token动态INT8激活。设 $W\in\mathbb R^{o\times d}$、$X\in\mathbb R^{T\times d}$，离线profiling得到输入通道范围 $S_j$，论文采用SmoothQuant强度 $\alpha=0.5$：

$$\lambda_j=\frac{S_j^{\alpha}}{(\max_i|W_{ij}|)^{1-\alpha}},\qquad
D=\operatorname{diag}(\lambda_j),\quad X'=XD^{-1},\quad W'=WD.$$

对非零有限 $\lambda_j$，未量化时 $X'(W')^\top=XW^\top$；量化后通过改变输入/权重范围改善网格利用。动态激活网格再从当前token计算范围，处理每步实际变化。这个配方有128条离线校准prompt，不属于“完全免校准”；动态激活尺度和离线通道缩放也不能混成同一参数。零范围保护、具体舍入/饱和及尺度字节格式，论文未充分展开，不能补写成作者实现已验证。（§3；一般机制见[对角缩放](../theory/diagonal-scaling-equivalent-transform.md)。）

作者在conditional DiT的linear上记录各去噪步的max-abs、标准差与峰度，早/晚半程fragility排序的Spearman为0.930。FFN down-projection占主要脆弱位置；top-N集合还含少数边界/embedding层。论文选择N=17，以211层为分母约8.06%，保护层保持BF16并取消激活量化，报告约1.5GB额外成本。因此W8A8是主体配方，不能理解成所有linear、组件与中间张量均为8 bit。两分支完整保护名单和映射未公开展开。

选择机制也不同于[FlexQ](flexq.md)固定保留down输入A8、[Atom](atom.md)保留少量原通道INT8，或[CentriQ](centriq.md)移除逐token均值再回注；它们分别保护模块、指定输入精度、坐标和输入相关旁路，没有共同协议时不能合成性能排名。

## 3. 保护消融与最终质量表使用不同提示

§5报告naive W8A8失效、SmoothQuant和保护各能恢复主要差距，保护贡献约78–92%，两者作用非简单相加；N=17达到作者目标，N=8明显退化。但这套2×2消融、保护数量选择与激活profiling使用raw自然语言prompt，未在最终JSON电池上重做。

所以它支持raw-prompt阶段的归因；最终表1–3只验证完整保护配方在JSON协议下的表现。0.930的跨时间排序相关不是跨提示格式稳定性的证明。作者预期模块名单对格式稳健，是待验证判断，不能写成在JSON协议下已证明“保护贡献78–92%”。

## 4. JSON扩写属于评测输入

三套PartiPrompts派生集合分别为128条校准、200条无文字质量prompt和100条文字prompt；后者63条具有机器可检查OCR目标。作者用LLM按公开caption规范重建expander，再用CaptionVerifier检查，300条评测caption均通过。它不是原厂生产API。每种量化版本共享相同JSON caption、seed1000、48步、1024×1024与V4_QUALITY_48预设。

论文报告raw提示下OCR exact-match约为零，JSON下约54–57%。它说明该模型的输入协议影响文字能力能否被合理测量，不支持任意模型通过JSON包装都提高约55个百分点。扩写过程也可能增加细节；只有各版本共享同一扩写结果，量化对照才没有混入版本各自不同的输入。（§3–4、表4；公共解释见[生成质量与输入协议](../implementation/model-quality-evaluation.md#12.%20%E7%94%9F%E6%88%90%E8%B4%A8%E9%87%8F%E3%80%81%E4%BF%9D%E7%9C%9F%E5%BA%A6%E5%92%8C%E6%8F%90%E7%A4%BA%E5%8D%8F%E8%AE%AE%E6%80%8E%E6%A0%B7%E5%88%86%E5%BC%80%E8%AF%84%E6%B5%8B)。）

## 5. 三类质量指标分别保留

| 版本 | HPSv2 | CLIP | PickScore | 对FP8的LPIPS | OCR exact（n=63） |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP8参考 | 0.2807 | 29.48 | 22.85 | — | 0.540 |
| NF4 | 0.2789 | 29.63 | 22.80 | 0.306 | 0.556 |
| INT8保护配方 | 0.2763 | 29.48 | 22.84 | 0.243 | 0.571 |
| Q4_K | 0.2783 | 29.64 | 22.79 | 0.277 | 0.540 |

表1–4中，HPSv2/CLIP/Pick是学习得到的偏好或图文评分，不能当作独立人工评审；PSNR/SSIM/LPIPS以同seed FP8图像为参考，文字使用PP-OCR。最低LPIPS表示最接近参考，并不等于评分最高或文字完全正确。INT8的PSNR/SSIM为18.53/0.712；比Q4_K的LPIPS差−0.034、95% CI [−0.041,−0.027]。INT8比NF4更接近只给点估计，未报告这对LPIPS的显著性检验。

表2按prompt配对、bootstrap10,000次计算95% CI，固定一个seed，未校正多重比较。INT8−FP8的CLIP为−0.005、CI [−0.146,+0.133]，Pick为−0.007、CI [−0.040,+0.027]；Q4_K−NF4两项CI也含零。它们表示此协议/样本未分辨出差异；没有预先声明等价界限，不能证明正式等价。表2并非所有区间含零：Q4_K−FP8的CLIP为+0.154 [0.014,0.298]、Pick为−0.055 [−0.093,−0.017]。作者关于“无实质分离”的解释，不能写成全指标统计上无差异。

HPSv2的最大点估计差实际为0.0044，未给逐图CI。OCR NED：FP8/NF4/INT8/Q4_K为0.133/0.116/0.099/0.123；INT8−NF4的NED CI据正文含零，但没有刊出端点。CI只反映固定seed下prompt抽样变化，不覆盖跨seed、所有文字任务、别的judge或硬件变化。与已有[质量等价判据](../implementation/model-quality-evaluation.md#8.%20%E6%97%A0%E6%98%BE%E8%91%97%E5%B7%AE%E5%BC%82%E4%B8%8D%E7%AD%89%E4%BA%8E%E6%BB%A1%E8%B6%B3%E8%B4%A8%E9%87%8F%E7%AD%89%E4%BB%B7)保持同一边界。

## 6. GGUF格式与作者编码器贡献需区分

Q8_0和Q4_K是weight-only存储格式，名义码宽不包括全部元数据。[GGUF格式](../implementation/gguf-block-quantization-formats.md#3.%20%E4%BA%8C%E7%BA%A7%E5%B0%BA%E5%BA%A6%EF%BC%9AK-quant%20%E8%B6%85%E5%9D%97)中，Q4_K的256元素超块有128字节INT4码、12字节子块参数和4字节FP16超块参数，共144字节，即4.5 bit/weight；Q8_0为32个INT8加一个FP16尺度，共34字节，即8.5 bit/weight。仅比4与8这两个整数会漏掉参数成本。

论文称自行实现NumPy Q4_K superblock quantizer和Torch反量化、与参考decoder验证。这个作者报告不能替代本项目取得代码、检验bit序或真实checkpoint。同段“没有GGUF量化器提供Q4_K encoder”不能作为GGUF生态的一般事实：既有llama.cpp固定快照的ggml-quants.c已有quantize_row_q4_K_ref，ggml.c的Q4_K类型表登记from_float_ref，实际量化入口也调用quantize_q4_K。该快照获取于2026-07-01，未再追溯论文6月版本所指工具环境；它能说明这一固定版本有通用编码器，不能说明该工具直接支持Ideogram的完整导出/执行。

§5又称Q8_0的round-to-nearest无损；这不能作为格式的一般性质。即使忽略FP16尺度舍入，32元素块取max=1、含元素0.1时，尺度1/127，码13恢复为13/127≈0.10236，已经不是原值。此为均匀网格教学反例，不是对作者特定权重或模型质量的实测；论文也没有给出Q8_0的逐元素无损证明。（llama.cpp的quantize_row_q8_0_ref保存FP16尺度并舍入int8码。）

## 7. 磁盘大小、显存与延迟不能互换

| 版本 | 两分支DiT磁盘权重GB | 已在显存内运行的秒/图 |
| --- | ---: | ---: |
| FP8参考 | 18.6 | 172.9 |
| NF4 | 10.4 | 164.5 |
| INT8保护配方 | 18.6 | 约184 |
| Q4_K | 10.44 | 203.3 |

表5在RTX3090 24GB集群上，两个DiT放同一张卡计算，VAE放另一张卡；“single-GPU-compute”不等于完整管线仅占一张卡。表中GB是磁盘DiT权重，不是报告的峰值VRAM，也不含冷加载/CPU offload成本。Q4_K较FP8约省43.9%磁盘权重，却比NF4慢约23.6%；INT8约比NF4慢11.9%（按表5数值复算）。

作者尚无融合INT8 GEMM，Q4_K用Torch反量化。硬件支持INT8并没有在该软件实现里自动变成速度收益。Q4_K是与NF4相近大小/评分的另一选项，10.44GB还略大于10.4GB；不能由作者“Pareto”措辞推出它在磁盘、质量、延迟上严格支配NF4。该论文未提供全配置搜索前沿。

## 来源身份

| 来源 | 固定身份 | 本页范围 |
| --- | --- | --- |
| Ideogram INT8/GGUF研究（`https://arxiv.org/abs/2606.12280v2`） | arXiv:2606.12280v2；PDF SHA-256 0c1c983163eb2d97fcb650ff13fedbc1aed95ad67caab4c977e4f8559037a549 | PDF首页题为Quantizing a Dual-Branch Flow-Matching Diffusion Transformer for Consumer GPUs: INT8 and GGUF Post-Training Quantization of Ideogram 4.0；归档公开记录题为Holding the FP8 Quality Ceiling at 8-Bit Weights and Activations；核正文、表1–5、图1及限制，未独立审计全部引用或获得作者实现 |
| ggml-org/llama.cpp（`https://github.com/ggml-org/llama.cpp/tree/b820cc8e6f94843d32f92c8ebd7db837dae2bd8b`） | b820cc8e6f94843d32f92c8ebd7db837dae2bd8b | Q4_K/Q8_0结构、参考编码/恢复和类型/量化入口；用于核格式的一般性说法，不是Ideogram作者代码 |
