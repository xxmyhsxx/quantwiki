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
updated: 2026-10-15
---

# ModelOpt PTQ：配置、校准与部署 checkpoint

ModelOpt 的 PTQ 输入是已训练模型、量化配置和需要时执行的校准数据。配置指定哪些模块降精度、使用什么表示；校准或搜索确定尺度与逐层选择；导出器将结果写成后端可消费的 checkpoint。这个过程需要分别理解模块范围、校准参数、搜索恢复状态和部署表示，不能仅由 `nvfp4` 或 `modelopt` 名称判断最终执行路径。

可以沿着四个问题理解这条流程：**量化谁、怎样确定参数、保存什么、谁来执行**。普通 PTQ 先固定表示，再确定参数；AutoQuantize 还会在预算内选择逐层表示。两条路径最后都需要导出，而推理框架只消费导出的表示和参数，不会重新执行 recipe 中的校准或搜索。

本页依据固定 `f2f0d695` 的 HF PTQ 与 recipes 文档，并用 vLLM `568afb3a` 的静态 FP8 Linear 说明消费端要求。ModelOpt 的 YAML、量化器、搜索器及导出器代码未在该条目保存，相关行为按文档契约解释；未运行校准、导出或模型。量化数学见[范围选择](../theory/calibration-and-range-selection.md)和[混合精度预算](../theory/mixed-precision-allocation.md)，不同后端的接入方式见[部署链路](quantized-llm-deployment-backends.md)。

## 1. Recipe 分开指定对象、格式与算法

Recipe 是量化配置的组合：它选择模型主体中的模块、这些模块采用的权重/激活表示、KV cache 模式，以及可选的校准算法。`quant_cfg` 描述对象与表示，`quantize.algorithm` 决定怎样求参数。例如 Weight MSE、GPTQ、max 是校准选择；它们不直接指定 checkpoint 容器或推理内核。（HF：Getting Started、Recipe-based Quantization；Recipes：General recipes、Calibration variants。）

### 先确定哪些模块会变化

模型主体的几个范围逐步扩大：`nvfp4_experts_only` 只选 routed experts；`mlp_only` 选 MLP/FFN；`omlp_only` 再加入 `o_proj`，q/k/v 仍保留较高精度。general 默认规则还受 router、head、vision 等共同排除约束。选择模块后，还要区分 W4A4 与 weight-only：前者同时量化权重和输入，后者保留 BF16 激活。

这些规则靠模块名匹配，因此同一 recipe 不一定能直接用于另一种模型。Step-3.7 使用 `moe`/`share_expert` 命名，文档报告通用 expert/MLP 规则会导出 `quant_algo:null`；Megatron 的 `_mcore` 规则用于 HF 路径则报告不匹配并报错。两例分别说明未正确选中模块和规则不兼容，并非同一种失败行为。（Recipes：Model-body schemes、Model-specific recipes。）

格式名也不能代替完整配置。例如 `nvfp4_mlp_weight_only` 在 recipes 中声明 block32，其他 NVFP4 接口可能要求不同粒度。Checkpoint mirror 是某个指定 checkpoint 的配置映射；alias 继承已有 recipe，仅改变 metadata。它们帮助复用配置，没有改变后端接受表示的条件。（Recipes：Weight-only schemes、Checkpoint mirrors/aliases。）

### 再确定最终生效的配置

指定 `--recipe` 后，`--qformat` 被忽略。PTQ recipe 自带 KV 配置，也会忽略 `--kv_cache_qformat`；AutoQuantize recipe 只有在未给显式 `kv_cache` 字段时，才回退到该 CLI 选项。因而应从最终 recipe 判断模型主体与 KV 的选择，不能只看命令行中出现了哪些选项。可用组合还受已提供的 recipe 和消费端限制。（HF：Recipe-based Quantization、KV Cache Quantization。）

## 2. 尺度层级与前向覆盖决定校准需求

固定表示后，基本调用 `mtq.quantize(model, config, forward_loop)` 原位替换量化模块，并在需要时运行前向以确定参数。权重范围可直接从权重取得；静态激活参数需要校准样本；动态激活参数则在推理时由当前输入计算。因此“是否需要校准”要逐个量化器、逐级尺度判断。

### 同一种格式可以采用不同的尺度策略

Recipes 的 NVFP4 default 声明 E2M1、block-16 与 FP8 block scales。MSE 变体搜索静态权重尺度并进行 E4M3 scale sweep，激活仍用 max 校准；local-Hessian 变体以局部输出重构选择权重尺度，当前文档限定 single-rank。这些选择改变的是权重参数的求法，激活统计与 KV 配置仍由各自规则决定。（Recipes：Full-model schemes、Calibration variants。）

`input_scale1` 给专家激活量化器设 `constant_amax=2688=6×448`，将全局 `input_scale` 固定为 1，因而不再收集这些量化器的激活前向统计。每块 E4M3 激活尺度仍动态计算。在对应 recipe 中，权重 max 来自权重张量，KV cast 也采用常数，文档据此说明整套组合无需校准前向。普通 W4A4 若仍有待估计的静态激活尺度，就不具备这一条件。多级尺度的关系见[校准分工](../theory/calibration-and-range-selection.md#5.%20%E6%A0%A1%E5%87%86%E3%80%81%E8%AE%AD%E7%BB%83%E3%80%81%E9%AA%8C%E8%AF%81%E4%B8%8E%E6%B5%8B%E8%AF%95%E6%80%8E%E6%A0%B7%E5%88%86%E5%B7%A5)。（Recipes：input_scale1。）

`nvfp4_act_headroom` 则从校准时每块 amax 的分布留出全局余量。将两个百分位值记为 $a_{\rm anchor}$、$a_{\rm upper}$，文档关系可写成

$$a_{\rm global}=\max(\rho\,a_{\rm anchor},a_{\rm upper}).$$

默认百分位为 1 和 99.99，$\rho=16384$。较大的全局范围让已观察块占用较低的 FP8 block-scale 区域，为未见离群值留空间，代价是校准范围内的分辨率可能下降。该变体只作用于 NVFP4 输入量化器，FP8 和 weight-only 配置下不生效。文档报告了 GLM/SciCode 的行为改善，但没有通用质量保证，余量仍需在独立数据上验证。（Recipes：nvfp4_act_headroom。）

### 被量化的路径必须实际收到输入

校准数据不仅要有代表性，还要经过被选中的模块。文档中的 diffusion_gemma self-conditioning 分支不在标准 PTQ 前向中执行，因而未收集到 amax；对应 local exclusion 将其保留为 BF16，避免导出缺少参数。

VLM 同样需要区分“量化哪些模块”和“给它们什么输入”。即使只量化语言分支，`--calib_with_images` 仍让图文输入经过完整 VLM，把视觉特征送入语言侧量化器。Qwen3-VL/dense Qwen3.5 的可选 vision recipe 则进一步选择 Linear/merger 的 FP8，patch embedding 与 vision-attention BMM 保留高精度。文本 preview 不能检验这些视觉路径的完整行为。（Recipes：Extra exclusions；HF：VLM quantization、VLM calibration with image-text pairs。）

### 大模型怎样降低校准内存

内存安排改变参数何时驻留、何时保存，不应与数值配置混为一谈。Real Quant 示例先在 `init_quantized_weights` 上下文中加载压缩权重，再做 max 校准；文档把 low-memory mode 限定为 FP8/NVFP4 max 路径。普通 layerwise 按 decoder layer 推进校准；layerwise export 还逐层落盘，可恢复已完成层，无需再执行独立 export pass。文档声明数值相同，但这里未核 offload、恢复及导出一致性的实现。（HF：Real Quant；Recipes：layerwise/layerwise export。）

`--use_fsdp2` 把模型分片到多个节点以完成加载和校准。它不决定推理时的 TP/PP：KV AutoQuantize 与该 flag 的组合在加载前被拒绝，weight AutoQuantize 仍标为 experimental，分布式 KV 评分、选择及 checkpoint 写入的同步尚待支持。通信和导出也会占用时间，增加 GPU 数不等于同比加速。分片参数的基础见[参数生命周期与 checkpoint](fsdp2-parameter-lifecycle-and-checkpoint.md)。（HF：Multi-Node Post-Training Quantization with FSDP2。）

## 3. AutoQuantize 在预算内选择逐层表示

普通 PTQ 已经指定表示，AutoQuantize 还要回答“各层分配多少精度”。权重搜索使用 `effective_bits` 约束，配置有两种互斥写法：

- 顶层 `candidate_formats` 给未匹配模块提供候选，局部 overrides 改变指定模块的候选。顶层候选隐含 BF16/no-quant，匹配规则可用 `allow_no_quant:false` 排除。
- 顶层 `quantize` 固定 PTQ baseline，只有 `module_search_spaces` 中的模块参与搜索。无需搜索的模块由 baseline 表达。

`disabled_layers` 决定哪些层不参与搜索，`cost_excluded_layers` 决定哪些层不纳入成本。搜索对象和计费对象因此可能不同。文档中的 NVFP4/FP8 混合例子使用 5.4 effective bits，其含义取决于成本模型和排除规则，不能直接换算完整文件或 GPU 峰值。`gradient` 评分需要反传，`kl_div` 不需要；这里未核评分器、solver 和完整成本计算代码。（HF：AutoQuantize for Hugging Face models。）

### KV 搜索有独立的成本与消费条件

KV AutoQuantize 把完整 K/V 配置作为候选，按可搜索层的宽度加权，成本包含 packed scale。保存例子在 FP8-cast K/V 的 8.0 bits/scalar 与 NVFP4-cast K/V 的 4.5 之间，以 5.4 为目标。BF16 只用于孤立 KL 参考，不是 solver 候选；FP8-K/NVFP4-V 因配套消费端条件被排除。每步评分运行一次参考前向，再运行各 eligible-layer/candidate 前向，FP32 概率缓冲的内存还随选中 token 数和词表增长。（HF：KV-cache AutoQuantize。）

### 搜索恢复文件与部署文件分开保存

`search_state_dict` 保存敏感性、选中的 recipe 等结果，`--auto_quantize_checkpoint` 保存可恢复状态；KV 搜索还保存 replay 所需的 candidate quantizer tensors。这些内容用于恢复搜索、避免重复评分。

部署导出另将逐层 KV 格式写入 `kv_cache_quantized_layers`。推理端必须识别这个字段才知道各层怎样读取缓存。保存稿明确指出，所配的 vLLM 0.26.0 不消费该字段，因此 stock 环境下这种产物仅能导出，尚不能直接部署。权重侧已经支持 `quantized_layers` 并不能补足 KV 侧接口。（HF：KV-cache AutoQuantize、save/restore；vLLM `weight_utils.py:get_quant_config`。）

## 4. 导出把量化结果变成部署产物

导出器将量化模块中的权重与恢复参数写入目标后端约定的表示。两种入口服务于不同流程：

| 导出入口 | 产物及后续消费者 |
| --- | --- |
| `export_hf_checkpoint` | unified HF checkpoint，文档声明保留 HF 层结构和张量名，交配套 TensorRT-LLM、vLLM、SGLang 加载 |
| `export_tensorrt_llm_checkpoint` | Legacy 产物，导出时指定 inference TP/PP 等配置，之后经 `trtllm-build` 构建 TensorRT-LLM C++ backend 引擎 |

统一张量名方便框架识别模型，但粒度、恢复参数、模型和设备仍需匹配。支持矩阵外链及 test_deploy release 用例未在本项核验；短文本 load/generate 也不能替代图文或完整质量验证。（HF：Exporting Checkpoints、Deployment、Unified HF Checkpoint Deployment Model Support Matrix。）

### 格式转换还要保留数值与形状条件

GPT-OSS 的 MXFP4→NVFP4 cast 利用源 E8M0 尺度推导参数。以源指数 $k_j$ 和转换使用的 $k_{\max}$ 表示，全局因子为 $s_2=2^{k_{\max}-8}$，块 amax 为 $6\,2^{k_j}$。原文只在 $k_{\max}-k_j\le17$ 的 E4M3 可表示范围内报告反量化 bit-exact，超范围块回退到数据相关 amax。该入口要求 NVFP4-family qformat，并排除 AutoQuantize 多格式搜索，不能用作任意 MXFP4 权重的无损互换规则。（HF：MXFP4 → NVFP4 cast；转换和导出代码未核。）

GGML IQ1_S/IQ2_XS 则要求最终维可被 256 整除，HF 路径导出 packed GGML blocks；Megatron 还限制 TP=PP=1、非 fused-MoE。它们说明导出要满足目标块布局，而不只是给张量改一个 dtype 标签。数据、尺度与消费接口的关系见[块缩放矩阵乘](block-scaled-matmul-execution.md)。（Recipes：Weight-only schemes。）

## 5. 后端加载后还要准备执行布局

vLLM 的固定源码提供了一个具体消费例子。`get_quant_config` 先按 `model_config.quantization` 取得量化类，再把内嵌或外部配置交给该类解析。`ModelOptQuantConfigBase.from_config` 分别从传统和新式配置读取算法、全局 KV scheme、group size、排除模块等字段。此时 recipe 中的校准过程已经结束，loader 读取的是结果。（`model_loader/weight_utils.py:get_quant_config`；`quantization/modelopt.py:from_config`。）

所选 `ModelOptFp8LinearMethod` 读取 E4M3 权重及 FP32 `weight_scale/input_scale`。若融合的逻辑输出分片具有不同权重尺度，它先按最大尺度重标权重，再转置权重、将尺度收成标量，交给所选 FP8 kernel 继续准备和执行。这解释了为什么“文件已导出”之后仍可能有重排和数值转换，也解释了为什么格式支持不能直接推出某个 kernel 的速度。

这一配置声明 BF16/FP16 激活和最低 SM89。上述证据仅覆盖静态 per-tensor FP8 Linear，不代表 NVFP4、MoE 或其他 FP8 粒度的加载行为，也没有设备运行结果。（`ModelOptFp8Config/ModelOptFp8LinearMethod`。）

## 6. 用运行记录追溯产物

搜索状态帮助继续校准，部署 checkpoint 交给推理框架，运行记录则帮助回答“这份产物由什么配置生成”。文档中的 MLflow 记录包含展开 imports 后的 recipe、版本/调用、日志、quantizer summary，以及产生时的专家 token 计数，由主 rank 上传。

成功导出后才写本地 `.experiment.json` 指针；较早失败会留下既有 checkpoint 和旧指针，未 tracked 的导出则清除继承指针。因此看到记录或指针仍在，不足以判断本次导出成功。实际交付需对应导出结果、消费端配置及[质量/执行验证](quantization-error-diagnosis.md)。（HF：Tracking runs with MLflow；这里仅核文档，未访问服务。）

## 来源身份

| 来源 | 固定版本 | 使用范围 |
| --- | --- | --- |
| ModelOpt Post-Training Quantization Guides（`https://github.com/NVIDIA/Model-Optimizer/tree/f2f0d6958eb95374d816f3d6256a0be3738231a3`） | `f2f0d6958eb95374d816f3d6256a0be3738231a3`；获取于2026-09-24T02:12:20+08:00 | HF PTQ 811行与Recipes556行全文；API/对象/尺度、搜索预算、导出/转换、VLM/分布式与追踪条件。未核匹配YAML、量化器/导出器、外链和运行。 |
| vllm-project/vllm（`https://github.com/vllm-project/vllm/tree/568afb3a13806beb53bb2e6bd518269357b237c0`） | `568afb3a13806beb53bb2e6bd518269357b237c0` | 配置选择、ModelOpt配置解析、所选静态FP8 Linear的参数和加载后准备；设备kernel及实际加载/运行未核。 |
