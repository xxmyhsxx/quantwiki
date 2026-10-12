---
title: QSLAW 的尺度参数、混合数据与浮点恢复路径
type: implementation
tags:
  - vlm
  - qat
  - fake-quantization
  - model-export
sources:
  - raw/repositories/2026-09-21/qslaw/source/README.md.txt
  - raw/repositories/2026-09-21/qslaw/source/llava/quantize/quantizer.py
  - raw/repositories/2026-09-21/qslaw/source/llava/quantize/int_linear.py
  - raw/repositories/2026-09-21/qslaw/source/llava/quantize/int_llama_layer.py
  - raw/repositories/2026-09-21/qslaw/source/llava/quantize/utils.py
  - raw/repositories/2026-09-21/qslaw/source/llava/quantize/replace.py
  - raw/repositories/2026-09-21/qslaw/source/llava/train/train.py
  - raw/repositories/2026-09-21/qslaw/source/llava/train/train_mem.py
  - raw/repositories/2026-09-21/qslaw/source/llava/train/llava_trainer.py
  - raw/repositories/2026-09-21/qslaw/source/llava/model/builder.py
  - raw/repositories/2026-09-21/qslaw/source/modify.py
  - raw/repositories/2026-09-21/qslaw/source/scripts/pretrain/pretrain.sh
  - raw/repositories/2026-09-21/qslaw/source/scripts/sft/chat_for_13b_v1_5.sh
  - raw/repositories/2026-09-21/qslaw/source/scripts/sqa/sqa_for_13b.sh
updated: 2026-10-12
---

# QSLAW 的尺度参数、混合数据与浮点恢复路径

本页核固定264dfb081c210a8bab6a3fb5707d827ade7b6802的LLaVA/Vicuna uniform、group128、lwc初始化路径：脚本→分层替换→尺度前向/训练对象→混合Dataset→保存/后续适配。方法与原实验见[QSLAW](../methods/qslaw.md)。int_llama_layer只核必要Linear构造，builder只核LoRA基座加载/OmniQuant入口，train只核所选参数、Dataset及主训练/保存调用；未全面核attention、tokenizer/图像预处理、所有模型/数据/依赖版本、NF4或bnb分支，未执行作者函数、训练、模型或GPU。

## 1. 从哪个入口开始

train_mem调用train(attn_implementation="eager")，另一个attention入口被注释。所选脚本均指定bits4、lwc、resume、scaling和start_layer5；TrainingArguments本身bits默认16、wbits4、abits16、group_size128、start_layer38。只设置bits不保证走作者尺度路径：bits4/8且lwc才调用replace_quantization，另一分支配置bitsandbytes，必须分别核。

replace_quantization要求resume初始化文件，逐层载入omni_parameters[i]且strict=False。所选普通Llama/Vicuna将Q/K/V/o及MLP gate/up/down换成QuantLinear；未量化视觉塔/projector/embedding/norm/lm_head的低位存储。首次模型、初始化权重、CLIP、数据集和外部运行依赖都不在本轮验证中，strict=False也不证明所有必要键匹配。

## 2. 先冻结基础网格，再加入每组尺度

UniformAffineQuantizer在所选构造中quant_method写为uniform；n_bits为2–16，开启group时按输入列连续reshape，每输出行独立分组。group_num为 $m\lceil n/g\rceil$，m输出、n输入、g组宽。不整除时padding分支只允许symmetric；本页解释常用n整除128，不承诺任意尾组。类里虽有NF4函数，不表示这些脚本自动进入NF4实验。

mode初为calibration，per_channel入口按组取xmin/xmax，加载的upbound/lowbound经sigmoid控制范围；非对称Δ=(xmax−xmin)/(2^b−1)，Δ裁到[1e−5,1e4]，零点由−xmin/Δ计算后裁范围并取整数。基础网格不是每步再用当前尺度重新估计。

smooth_and_quant_fake先量化一次建立网格，再恢复原浮点weight.data，设置weight_quant=True/act_quant=False、mode=training，删除uniform路径的learnable clipping参数，并把scales/zeros注册为buffer。add_scaling随后增加dynamic_scale，shape为[group_num,1]，初始全1。i>start_layer才增加尺度：脚本5表示从层索引6起（第7个block）；较早层仍量化，但没有组尺度。

## 3. 前向与反向的量不同

QuantLinear的原weight/bias按buffer登记；use_weight_quant时先用weight_quantizer恢复浮点权重，再调用F.linear。所选fake_quant的顺序是

$$u=\frac{w}{\Delta a},\quad q=\operatorname{clip}(\operatorname{round}_{STE}(u)+z,0,2^b-1),\quad d=\Delta(q-z).$$

temp=base scale×dynamic_scale只用于除法，最后乘base scale；没有反乘dynamic_scale、没有同位置X变换、也没有固定码的恢复尺度更新。scales/zeros固定后，每次前向仍重新舍入w/(Δa)。round_ste前向为round，反向为恒等；clamp非饱和内部的尺度代理梯度为−w/a²，饱和外为0，边界按PyTorch实现。真正硬前向在码不变时的导数为0，两者不能混为解析导数。

dynamic_scale是直接Parameter，所选uniform路径没有exp、正值投影或防零clamp；参数初始化为1不保证训练后始终正或远离0。上述公式要求a非零，原梯度说明通常考虑正尺度的非边界区间；没有据此认定作者实验出现零除或负尺度。

基座weight作为buffer，不在该模块named_parameters或所选优化器的参数列表；这不等于所有buffer的requires_grad标志必然为False。train在tune_mm_mlp_adapter时冻结模型参数，再启用projector，scaling条件下重新启用dynamic参数；prepare_model_for_kbit_training及精度/反向细节依外部PEFT版本，未审查库内实现。LLaVATrainer只把requires_grad=True的named parameters加入优化器，projector可有独立学习率。

参数少仍需要整网前向/反向及保留浮点基座。qlayer.half是初始化转换；训练bf16/autocast、外部prepare处理及输出dtype必须以实际环境为准，不能将配置名当成所有权重、激活和累加都为某一位宽。

## 4. warmup在所选代码中是阶段安排

| 所选脚本阶段 | 数据/配置 | 权重与训练对象 |
| --- | --- | --- |
| pretrain/pretrain.sh | 558k图文，未开mix，1 epoch，LR1e−3 | fake quant + projector/组尺度 |
| sft聊天或sqa的尺度阶段 | 开mix，2 epoch，LR2e−3，context1024 | fake quant + projector/组尺度 |
| sft聊天指令阶段 | 665k，1 epoch，LoRA r128/alpha256，context2048 | 恢复浮点Linear后接LoRA，加载projector；已加载尺度不继续作为组尺度模块训练 |
| sqa指令阶段 | ScienceQA，12 epoch，未显式覆盖LoRA r/alpha | 默认r64/alpha16，恢复浮点Linear后接LoRA |

脚本注释称pretrain已经完成modality-aware warmup，但聊天和sqa的mix尺度阶段只显式加载pretrain_mm_mlp_adapter，没有传scaling_resume；该默认新训练路径将dynamic_scale重新初始化为全1，没有接续预训练尺度。阶段顺序因此不等于同一组尺度先warmup再混合训练。mix在Dataset构造时决定，Trainer未含按η步切换数据的逻辑；warmup_ratio0.03控制学习率。论文的2 epoch/1:1/η机制与这些脚本分开记录。

LazySupervisedMixDataset先从硬写占位路径加载文本，将其连续token流分成T=floor(token数/context)块；只保留有图像的下游样本，最多mix_ratio×T条，实际数为M。__len__返回M+T，mix_ratio默认1但只有图文数足够时才M=T；文本尾部不足一块被舍去。文本样本传零图像，不等于完全绕过视觉/projector；后续模型如何处理零图像未在本页审查。

语言支路的切片起点实际写为(i−T)×context；正确承接其多模态前缀的起点应是(i−M)×context。只有M=T才一致。例如M=2、T=4时，语言首索引i=2得到−2×context，而不是0；按Python负切片可能读取末段或得到空片段，取决于总长度/边界。这个标准库算例说明比例/数据量前提，未执行真实Dataset或推断历史训练失败。lengths/modality_lengths只返回M条，下游sampler组合也需核实际配置，本页未据此宣称所有mix/sampler可运行。

## 5. 保存、后续LoRA与浮点消费

scale阶段的safe_save_model_for_hf_trainer按named parameter名字分别保存projector与dynamic参数：最终输出为output_dir/exp/mm_projector.bin和scaling_train.bin。modify.py把平坦model.layers.i.*键改成按i分层的dict，并**覆盖输入文件**；它只改字典结构，不把scaling_train.bin重命名成scaling.bin，也不打包权重。

聊天脚本随后调用modify.py的scaling.bin，再把同名文件交给scaling_resume；所选保存函数产生scaling_train.bin，脚本没有给出重命名/复制步骤。sqa脚本也加载scaling.bin但未显式调用modify。因此从这些入口到产物消费仍有文件名/字典结构前提，未核作者已发布checkpoint是否另经过整理。tune_mm_mlp_adapter的中间_save_checkpoint只保存projector相关键，未保存dynamic尺度；不能把中间checkpoint直接当成完整尺度恢复点。

scaling_resume存在且tune_mm_mlp_adapter=False时，replace给后部qlayer加入/加载尺度，smooth_and_quant_inplace恢复浮点权重，copy_weights递归将其写回原普通Linear，再丢掉临时qlayer。此分支评论虽叫“real quant”，实际没有在这条路径保存packed整数或接专用低位kernel；此后LoRA包装普通Linear。若含LoRA，forward还包含适配器贡献，不能说全部有效权重始终是最初4bit格点。

builder的所选LLaVA+LoRA+model_base入口可再调用omniquant_from_checkpoint，后者在scaling开启时要求scaling_resume，并同样恢复/写回普通Linear，然后加载projector/non_lora_trainables和PEFT适配器。bnb加载参数、其他模型名及外部库支持另有分支；本页未验证加载命令、完整产物往返或所有dtype组合。

低位整数、基础Δ/z、组尺度、projector和LoRA的存储责任必须分别计费。当前所选路径足以解释质量模拟和参数学习，不足以证明4bit权重驻留、低位GEMM或论文的GPU成本；没有对作者报告作运行复现。

## 来源身份

| 来源 | 固定身份 | 核查范围 |
| --- | --- | --- |
| xjjxmu/QSLAW（`https://github.com/xjjxmu/QSLAW/tree/264dfb081c210a8bab6a3fb5707d827ade7b6802`） | 264dfb081c210a8bab6a3fb5707d827ade7b6802 | 所选LLaVA/Vicuna uniform/group128，分层初始化、fake尺度前向、优化器、mix数据、脚本阶段、保存与浮点恢复/LoRA消费入口 |

ZIP原件和本页14个文件与归档逐字节核对；历史174文件完整性依归档登记，本轮未重新审计整仓库或新取Git对象。没有导入上游/下载权重或数据/安装/训练/运行模型/GPU。NF4、bnb、其他模型/尾组、完整attention/tokenizer/image/依赖接口、实际warmup采样、保存重载及论文质量/成本继续保留未验证范围。
