---
title: ModelOpt PTQ：配置、校准与部署 checkpoint
type: implementation
tags:
  - llm
  - calibration
  - deployment
  - data-format
sources:
  - raw/articles/2026-09-24/modelopt-ptq-guides
  - raw/repositories/2026-09-21/vllm
updated: 2026-10-14
---

# ModelOpt PTQ：配置、校准与部署 checkpoint

ModelOpt 的 PTQ 输入是已训练模型、量化配置和需要时执行的校准数据。配置指定哪些模块降精度、使用什么表示；校准或搜索确定尺度与逐层选择；导出器将结果写成后端可消费的 checkpoint。这个过程需要分别理解模块范围、校准参数、搜索恢复状态和部署表示，不能仅由 `nvfp4` 或 `modelopt` 名称判断最终执行路径。

本页完整研读固定 `f2f0d695` 的 HF PTQ 与 recipes 两份文档，并用固定 vLLM `568afb3a` 的静态 FP8 Linear 入口说明消费端的要求。ModelOpt 的 YAML、量化器、搜索器及导出器代码未在该条目保存，本文按文档契约解释，未运行校准、导出或模型。量化数学复用[范围选择](../theory/calibration-and-range-selection.md)和[混合精度预算](../theory/mixed-precision-allocation.md)，后端选择接续[部署链路](quantized-llm-deployment-backends.md)。

## 1. Recipe 分开指定对象、格式与算法

HF 文档的基本调用为 `mtq.quantize(model, config, forward_loop)`：原位替换量化模块，以需要的前向统计确定参数。Recipe 将 model-body 格式/范围、KV 模式和可选校准算法组合起来；`quant_cfg` 描述对象与表示，`quantize.algorithm` 改变参数搜索。Weight MSE、GPTQ、max 是这里的校准选择，不是 checkpoint 格式或消费内核名。（HF：Getting Started、Recipe-based Quantization；Recipes：General recipes、Calibration variants。）

body 的范围决定压缩对象：`nvfp4_experts_only` 针对 routed experts，`mlp_only` 扩展到 MLP/FFN，`omlp_only` 再加入 `o_proj`，q/k/v 仍保留较高精度；general 默认规则还受 router、head、vision 等共同排除约束。NVFP4 W4A4 同时量化权重和输入，weight-only 则保留 BF16 激活。Recipe 的 wildcard 必须匹配实际模块名：Step-3.7 的 `moe`/`share_expert` 与通用 expert/MLP 模式不同，文档报告套用通用配置会导出 `quant_algo:null`；Megatron `_mcore` 规则套到 HF 路径则报告不匹配并报错。这些是不同配置的例子，不能合成统一失败行为。（Recipes：Model-body schemes、Model-specific recipes。）

命令行优先关系也影响实际对象。指定 `--recipe` 后，`--qformat` 被忽略；PTQ recipe 内含 KV 配置，同时忽略 `--kv_cache_qformat`。AutoQuantize recipe 若未给显式 `kv_cache` 字段，才回退到该 CLI 选项。选用 body 与 KV 两个轴不意味着任意组合均可导出或部署，保存稿以 shipped recipe 和消费端条件限定组合。（HF：Recipe-based Quantization、KV Cache Quantization。）

## 2. 尺度层级与前向覆盖决定校准需求

Recipes 的 NVFP4 default 声明 E2M1、block-16 与 FP8 block scales。其 MSE 变体搜索**静态权重尺度**并做 E4M3 scale sweep，激活仍使用 max 校准；local-Hessian 变体以局部输出重构近似选择权重尺度，文档限定当前 single-rank。改变权重尺度目标没有自动改变激活统计或 KV 配置。（Recipes：Full-model schemes、Calibration variants。）

`input_scale1` 的专家激活量化器用 `constant_amax=2688=6×448` 固定全局 `input_scale=1`，不收集这些量化器的激活前向统计；**per-block E4M3 激活尺度仍动态**。在对应 recipe 中，权重 max 直接来自权重张量，KV cast 也使用常数，因此文档说明该组合无需校准前向。这个条件不能外推到普通 W4A4，或把固定全局尺度理解为全部尺度静态。（Recipes：input_scale1；多级尺度的静态/动态关系见[校准分工](../theory/calibration-and-range-selection.md#5.%20%E6%A0%A1%E5%87%86%E3%80%81%E8%AE%AD%E7%BB%83%E3%80%81%E9%AA%8C%E8%AF%81%E4%B8%8E%E6%B5%8B%E8%AF%95%E6%80%8E%E6%A0%B7%E5%88%86%E5%B7%A5)。）

另一个变体 `nvfp4_act_headroom` 用校准中 per-block amax 分布的两个百分位构造全局余量：

$$a_{\rm global}=\max(\rho\,a_{\rm anchor},a_{\rm upper}).$$

文档默认 anchor/upper 为1/99.99百分位、ρ=16384；它让已观察块占用较低的 FP8 block-scale 范围，为未见离群值留空间，同时可能损失校准范围内的分辨率。作用对象是 NVFP4 **input quantizers**，FP8 和 weight-only 配置为 no-op。保存稿的 GLM/SciCode 行为改善只是所述研究报告，未给出通用质量保证，ρ仍需结合独立数据验证。（Recipes：nvfp4_act_headroom；上式按原文关系记号化。）

校准统计还取决于实际前向覆盖。文档的 diffusion_gemma self-conditioning 分支不在标准 PTQ 数据中执行，量化器未收集 amax 会造成导出问题，local exclusion 保留该分支 BF16。VLM 的默认量化范围虽可仅为语言分支，`--calib_with_images` 仍让图文输入走完整 VLM，将视觉特征送入被选中的量化器；Qwen3-VL/dense Qwen3.5 的 opt-in vision recipe 只声明 Linear/merger 的 FP8，patch embedding 与 vision-attention BMM 保留高精度。通用 multimodal quickstart 和 text-only preview不能据此证明 vision-quantized checkpoint 已完成服务验证。（Recipes：Extra exclusions；HF：VLM quantization、VLM calibration with image-text pairs。）

## 3. AutoQuantize 的搜索空间、预算与恢复状态

Weight AutoQuantize 在给定 `effective_bits` 约束下选择逐层格式。保存稿支持两种互斥配置：顶层 `candidate_formats` 为未匹配模块给出候选，允许局部 overrides；或顶层 `quantize` 固定 PTQ baseline，只用 `module_search_spaces` 指定真正搜索的模块。顶层候选隐含 BF16/no-quant 选择，匹配规则可用 `allow_no_quant:false` 排除它；固定模块应由 baseline表达。`disabled_layers` 与 `cost_excluded_layers` 分别控制搜索排除和预算计费，不能把二者都理解为已压缩对象。（HF：AutoQuantize for Hugging Face models。）

文档将5.4 effective bits用作 NVFP4/FP8 混合例子，但其含义依赖 cost model、计费对象及排除规则；它不是完整 checkpoint 文件的 bits/parameter，也不直接给出 GPU 峰值或速度。`gradient` scoring 需要反传，`kl_div` 不需要；实际评分、solver 和完整成本计算代码未在本项核读。预算的一般作用复用[混合精度分配](../theory/mixed-precision-allocation.md)。（HF：AutoQuantize。）

KV AutoQuantize 是另一成本对象：候选是完整 K/V 配置，config-level effective bits包含 packed scale 开销，以 eligible layers 的宽度加权计费。保存例子在 FP8-cast K/V（8.0 bits/scalar）和 NVFP4-cast K/V（4.5）间以5.4为目标；BF16只用于孤立 KL 参考，不是solver候选，FP8-K/NVFP4-V因配套 consumer 条件被排除。评分每步执行一次参考前向及各 eligible-layer/candidate 前向，FP32 reference/candidate log probabilities 的内存还随选中token数和词表增长。（HF：KV-cache AutoQuantize。）

`search_state_dict` 保存敏感性、选中 recipe 等搜索结果；`--auto_quantize_checkpoint` 存储可恢复的搜索状态，KV 搜索还保存 replay 所需的 candidate quantizer tensors。这是跳过重复评分、恢复搜索的产物。部署导出另将逐层 KV 格式记入 `kv_cache_quantized_layers`，需要配套 consumer；保存稿明确 pinned vLLM 0.26.0不消费该字段，stock 环境中此类 checkpoint 为 **export-only**。已有权重 `quantized_layers` 支持不能代替这个 KV 字段。（HF：KV-cache AutoQuantize、save/restore；所选 vLLM `weight_utils.py:get_quant_config` 处理的是权重混合配置。）

## 4. 导出表示与后端加载后的准备

校准后的模型交给 `export_hf_checkpoint` 产生 unified HF checkpoint，文档声明保留 HF 层结构和张量名，用于配套 TensorRT-LLM、vLLM、SGLang。Legacy `export_tensorrt_llm_checkpoint` 则接收 inference TP/PP 等配置，产物用于 TensorRT-LLM C++ backend，之后经 `trtllm-build` 构建引擎。统一命名不消除格式、模型、设备与 consumer 版本条件；文档的支持矩阵外链及 test_deploy release用例未在本项核验，短文本 load/generate smoke 也不能代替图文或完整质量验证。（HF：Exporting Checkpoints、Deployment、Unified HF Checkpoint Deployment Model Support Matrix。）

消费端可用一个已核的实例说明：vLLM `568afb3a` 的 `get_quant_config` 先按 `model_config.quantization` 取得量化类，再读取内嵌或外部量化配置交给该类解析；`ModelOptQuantConfigBase.from_config` 分别读传统 `quantization` 和新式配置中的算法、全局 KV scheme、group size、排除模块等字段。这里的校准算法已经固化为表示与参数，loader不会按 ModelOpt recipe 重跑校准。（`model_loader/weight_utils.py:get_quant_config`；`quantization/modelopt.py:from_config`。）

选定的 `ModelOptFp8LinearMethod` 消费 E4M3权重以及 FP32 `weight_scale/input_scale`。融合的多个逻辑输出分片有不同权重尺度时，加载后按最大尺度重标权重，再将权重转置成执行表示，将尺度收成标量，并交所选 FP8 kernel进一步准备和执行。该配置声明BF16/FP16激活与最低SM89；这是静态 per-tensor FP8 Linear 的源码范围，不等于已核NVFP4、MoE、其他FP8粒度或GPU运行。（`modelopt.py:ModelOptFp8Config/ModelOptFp8LinearMethod`。）

## 5. 源格式转换与作用范围的条件

HF 文档的 GPT-OSS MXFP4→NVFP4 cast从源E8M0尺度推导NVFP4参数，以源指数 $k_j$ 和转换使用的 $k_{\max}$ 表示，全局因子为 $s_2=2^{k_{\max}-8}$，块 amax 为 $6\,2^{k_j}$。原文只对 $k_{\max}-k_j\le17$ 的 E4M3可表示范围报告反量化bit-exact，超范围块回退到数据相关amax；不是任意MXFP4权重的无损格式互换。它要求NVFP4-family qformat，并排除AutoQuantize多格式搜索；cast/导出代码未核。（HF：MXFP4 → NVFP4 cast。）

共享粒度也随具体recipe/接口变化：Recipes 将 `nvfp4_mlp_weight_only` 声明为block32，不能仅凭NVFP4名称套用其他消费接口的粒度。GGML IQ1_S/IQ2_XS声明最终维可被256整除，HF导出packed GGML blocks；Megatron另限TP=PP=1及非fused-MoE。Checkpoint mirror表示某个指定checkpoint的配置映射，alias则继承既有recipe、仅改metadata，它们都不提供任意后端的互操作证明。物理数据、scale准备与消费的关系见[块缩放矩阵乘](block-scaled-matmul-execution.md)。（Recipes：Weight-only schemes、Checkpoint mirrors/aliases。）

## 6. 校准内存、分布式接入与产物追溯

Real Quant 示例先在 `init_quantized_weights` 上下文中加载压缩权重，再执行max校准；文档限定 low-memory mode用于FP8/NVFP4 max路径。普通layerwise按decoder layer推进校准以降低峰值；layerwise export变体还逐层落盘，可恢复已完成层且无需单独export pass。“Same numerics”是文档声明，未核offload、断点恢复及导出一致性实现。（HF：Real Quant；Recipes：layerwise/layerwise export。）

`--use_fsdp2` 接入多节点模型分片与校准，不等于推理时使用同样的TP/PP。保存稿明确KV AutoQuantize与该flag组合在加载前被拒绝，weight AutoQuantize仍带experimental warning；分布式KV评分、选择和checkpoint写入尚需同步。通信和导出可以增加耗时，不能用GPU数直接推导校准加速。（HF：Multi-Node Post-Training Quantization with FSDP2。）

运行记录与部署产物也有不同生命周期。文档的MLflow记录包含展开imports后的recipe、版本/调用、日志、quantizer summary及产生时的专家token计数；主rank上传。成功导出后才写本地 `.experiment.json` 指针，较早失败会保留既有checkpoint及旧指针，未tracked导出则清除继承指针。因此记录可用于追溯配置，判断本次产物是否完成还需核导出结果、consumer及[质量/执行验证](quantization-error-diagnosis.md)。上述追踪与恢复行为仅核文档，未访问MLflow服务。（HF：Tracking runs with MLflow。）

## 来源身份

| 来源 | 固定版本 | 使用范围 |
| --- | --- | --- |
| ModelOpt Post-Training Quantization Guides（`https://github.com/NVIDIA/Model-Optimizer/tree/f2f0d6958eb95374d816f3d6256a0be3738231a3`） | `f2f0d6958eb95374d816f3d6256a0be3738231a3`；获取于2026-09-24T02:12:20+08:00 | HF PTQ 811行与Recipes556行全文；API/对象/尺度、搜索预算、导出/转换、VLM/分布式与追踪条件。未核匹配YAML、量化器/导出器、外链和运行。 |
| vllm-project/vllm（`https://github.com/vllm-project/vllm/tree/568afb3a13806beb53bb2e6bd518269357b237c0`） | `568afb3a13806beb53bb2e6bd518269357b237c0` | 配置选择、ModelOpt配置解析、所选静态FP8 Linear的参数和加载后准备；设备kernel及实际加载/运行未核。 |
