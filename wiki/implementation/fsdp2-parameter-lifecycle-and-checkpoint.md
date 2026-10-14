---
title: FSDP2：参数分片、计算期聚合与checkpoint
type: implementation
tags:
  - llm
  - tensor-parallelism
  - mixed-precision
  - memory
sources:
  - raw/articles/2026-09-24/pytorch-distributed-tutorials
updated: 2026-10-14
---

# FSDP2：参数分片、计算期聚合与checkpoint

FSDP2降低训练时每个rank持有的参数、梯度和优化器状态，但层计算需要的完整参数仍会临时出现。理解其内存和通信，需要追踪“持久分片 → 计算前all-gather → 梯度reduce-scatter → 本地优化器更新”，以及checkpoint究竟保存分片还是聚合后的完整张量。本页依据保存的FSDP2教程与三张图解释这条生命周期，未运行训练或保存恢复。

[张量并行](tensor-parallel-quantization.md)拆一次层运算的张量轴；本页的FSDP是训练状态分片，临时聚合后各rank仍对自己的数据执行计算。收录的Distributed目录只给其他教程入口，未展开DDP、TP、RPC、DeviceMesh或Monarch的匹配实现。教程站点版本与页面核验日期也不等于已核PyTorch库commit。

## 1. 参数组连接持久分片与临时完整参数

DDP在各rank保留模型副本、处理各自样本，最后用all-reduce同步梯度。FSDP2把持久模型状态分片：参数在前/反向需要完整表示时由all-gather聚合，反向的完整局部梯度经reduce-scatter归约并切回各rank，优化器只更新所属参数片段。它分解了DDP梯度同步的职责，没有取消同步。（How FSDP2 works及workflow/sharding图。）

| 状态或操作 | 本rank持有的对象 | 下一阶段需要的动作 |
| --- | --- | --- |
| 持久参数 | 每参数的DTensor分片，教程默认`Shard(0)` | 当前参数组all-gather出完整参数 |
| 层前向/反向 | 当前组临时unsharded参数与本rank的数据/激活 | 按配置重新分片；反向计算局部梯度 |
| 梯度同步 | 局部梯度参与reduce-scatter，得到归约后的所属片段 | 分片优化器更新持久参数 |

教程先对每个Transformer layer调用`fully_shard`，再处理root；root排除已经由子层管理的参数，把剩余参数形成自己的group。Group同时决定一次聚合哪些权重及通信范围，不能把“每个参数按dim0分片”理解成“每次通信只聚合一个参数”。先处理子层还能避免只按root管理时的一次大范围完整参数存活。（Model Initialization。）

`model.parameters()`在分片后表示DTensor，优化器必须在`fully_shard`之后构造，才能持有这些参数引用。FSDP2保留逐参数身份而不采用FSDP1 flat parameter；`FSDPModule`提供手动unshard和预取等接口。教程的`clip_grad_norm_`由DTensor操作通信部分结果以保留单设备语义，不能改为每rank独立裁剪本地分片范数。（Model Initialization、Gradient Clipping and Optimizer；migration的use_orig_params。）

## 2. 重新分片、预取与dtype共同决定峰值

持久状态分片不保证GPU峰值按rank数等比例下降：当前和预取group的完整参数、激活及通信缓冲仍占内存。`reshard_after_forward=True`对应教程的FULL_SHARD迁移项；False对应SHARD_GRAD_OP，前向后保留完整参数，不能对所有配置都写“前向后立即回片”。2D mesh对应的hybrid策略又有独立分片/复制范围。（Model Initialization、FSDP1-to-FSDP2 migration；峰值构成由生命周期解释。）

隐式预取在层i计算前把all-gather排入独立CUDA stream；CPU能及时提交、工作负载非CPU-bound时，下一组聚合才有机会与当前层计算重叠，反向按post-forward逆序。显式预取可提前排2个或更多group，或提前`unshard()`首组，代价是更多完整参数同时存活。更早入队并不保证通信已被隐藏，也不能仅凭预取数推导加速。（Forward/Backward with Prefetching及implicit图。）

MixedPrecisionPolicy的对象也应分别记。教程示例为：

| 对象 | 示例dtype | 对内存与数值的含义 |
| --- | --- | --- |
| 持久sharded参数 | FP32 | `param_dtype=BF16`没有把此副本变成BF16 |
| 临时unsharded计算参数 | BF16 | 降低计算/聚合后参数表示的成本 |
| 梯度reduce-scatter | FP32 | 用`reduce_dtype`单独控制跨rank归约精度 |
| 示例Adam状态 | FP32 | 优化器状态成本仍需单独计入 |

策略可按层设置；教程迁移项还说明buffers不被`fully_shard`分片，不能把模型全部张量归入同一分片预算。这种训练混合精度不同于导出packed低位权重，float8 all-gather和NF4扩展只是未展开外链，本文未据此补实现。（Enabling Mixed Precision、migration。）

## 3. Full state转换需要collective与文件写入分别协调

保存分片state dict可以保留DTensor表示，无需为了构造state dict先聚合全部参数；这不证明整个分布式checkpoint协议或metadata零通信。需要单文件完整权重时，`DTensor.full_tensor()`会all-gather，各rank须参与；教程让rank0逐张量移到CPU以降低GPU聚合峰值。最终写共享文件则需确定单writer，不能把collective调用也一起只放在rank0。（State Dicts with DTensor APIs。）

加载教程先在meta设备初始化并`fully_shard`，再用目标state dict相同的device mesh/placements将完整张量`distribute_tensor`成DTensor。最后`load_state_dict(assign=True)`直接赋值，因为meta张量不能`copy_`。这里的恢复还依赖参数名、shape和分片布局，不是任意文件交给loader都会兼容。（State Dicts with DTensor APIs。）

DCP在本教程指 **Distributed Checkpoint**，与[MLA的Decode Context Parallel](vllm-mla-context-parallel-execution.md)不同。`set_model_state_dict`的`broadcast_from_rank0=True`允许只有rank0读取完整CPU state，之后由分布式API分片/广播，各rank仍参与同步；`get_model_state_dict(full_state_dict=True, cpu_offload=True)`则聚合并移到CPU，不是纯本地分片保存。（State Dict with DCP APIs。）

保存稿两段full-state示例的最终`torch.save`都未在rank0 guard内：第一段仅rank0填充CPU字典，其他rank仍执行同路径save；第二段也未显示单writer保护。因此多rank共享路径存在竞争或覆盖空结果的静态风险，实际写入时应在完成所有rank的collective后只让选定writer落盘。匹配example代码和运行未核，不能把这个摘录问题扩大为已确认的当前库故障。（保存正文269–281及311–323行。）

教程把optimizer state保存指向外部example，未收录其实现；完整训练恢复、跨world-size转换、故障恢复和数值连续性没有在本项验证。分片state、完整模型state和部署checkpoint的用途需分别判断。

## 4. 量化校准和离散表示沿各自接口接入

[ModelOpt](modelopt-ptq-calibration-and-checkpoint.md)用`--use_fsdp2`接入分布式校准，但其搜索组合与导出限制由ModelOpt文档决定，FSDP2教程不能证明那些组合已实现。同样，[AQLM的PV训练](aqlm-pv-training-and-code-updates.md#6.%20FSDP%E3%80%81%E7%BC%93%E5%86%B2%E4%B8%8E%E4%BF%9D%E5%AD%98%E6%8E%A5%E5%8F%A3)另有整张QuantizedWeight的owner分配，不能由本教程推定其版本使用FSDP2。使用这些链路时，需核清计算参数、量化对象、优化器及保存产物分别由谁持有。

## 来源身份

| 来源 | 版本与时间 | 核读范围 |
| --- | --- | --- |
| PyTorch Distributed Tutorials and FSDP2（`https://docs.pytorch.org/tutorials/distributed.html`） | 登记`PyTorch Tutorials 2.14.0+cu130; FSDP tutorial last verified 2024-11-05`；获取于2026-09-24T02:12:20+08:00；页面自述Updated2025-09-02/Verified2024-11-05 | 目录217行、FSDP2教程411行全文及workflow/sharding/implicit三图；图于10-05补录，未证明9-24图片字节相同。匹配PyTorch/example实现、外链、训练和checkpoint运行未核。 |
