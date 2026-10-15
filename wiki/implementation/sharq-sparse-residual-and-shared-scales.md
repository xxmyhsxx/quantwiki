---
title: SharQ 的稀疏残差、共享尺度与所选接口
type: implementation
tags:
  - sparsity
  - data-format
  - kernels
  - activation-quantization
sources:
  - raw/repositories/2026-09-21/sharq
  - raw/papers/2026-09-21/sharq
updated: 2026-10-15
---

# SharQ 的稀疏残差、共享尺度与所选接口

SharQ 的原生路径把模型中的部分浮点线性层替换为两路 FP4 矩阵乘：稀疏主干利用硬件支持的稀疏计算，稠密残差补回被掩码去掉的信息和主干量化误差。两路结果相加，有偏置时再加一次偏置，作为该线性层的近似输出。它改变的是投影层的计算方式；attention、激活函数和模型残差连接仍有浮点计算。为什么采用这种分解、质量与速度怎样取舍，见[SharQ 方法页](../methods/sharq.md)。

执行分为三个相接的阶段：创建量化层时，权重量化为一份 FP4 载荷及两路所需的尺度视图；每次前向根据当前激活生成稀疏掩码，准备主干、残差和各自尺度；最后两路矩阵乘共用权重并合并输出。共享权重使两路补偿对应同一个近似线性映射，准备和合并则决定额外计算与中间读写成本。后文沿这条执行过程区分原生 FP4 接口、浮点模拟和可选融合。（`QLinearLayer`、`quantize_w32_shared`；固定版本见文末。）

本页依据所选 NVFP4/Llama 调用及准备、矩阵乘 host 接口的静态阅读，未编译或运行 GPU；完整框架加载和设备主循环未核验。导出与计时脚本的边界见第 6 节，来源版本与覆盖范围见文末。

## 1. 同名方法有native与模拟两类执行

QLinearLayer的SHARQ在创建时将权重转BF16，计算 $s_W=\max|W|/2688$（下界1e-9），除以尺度再转BF16，调用quantize_w32_shared。输入也用BF16，在线PTS同样从当前输入计算；prepared tuple携带压缩主干、pair metadata、两路激活尺度、稠密残差和全局尺度。两路GEMM输出全局因子为 $s_Xs_W$，bias只在合并后加一次。

SHARQ_SIM则由PyTorch做pair选择、32组主干/16组残差浮点fake quant，权重同样以32组解码值保存在BF16中，调用两次F.linear。它不是打包FP4、更不是稀疏Tensor Core；HIF4_SIM/SHARQ_HIF4_SIM也是模拟标签，不能据CLI名称宣称当前设备原生HiF4支持。本页未逐段核HiF4编码器。

main.py的实际量化入口明确传 **extra_fusion=False**，因而其SHARQ模式不走额外RMSNorm准备及GEMM累加融合。类默认值True不能覆盖调用者False；性能论文的融合报告与这一质量入口应分别记录。model_utils逐层包装并置use_cache=False，没有将该入口计为带生成缓存的服务复现。

所选QLlamaAttention让Q/K/V复用一份prepared输入，MLP让gate/up复用一份prepared输入；down和o_proj另准备。普通attention/SiLU/残差仍为浮点。可选kv_cache参数做4-bit浮点模拟：K在RoPE/cache更新前处理，V在更新/重复后处理，既不压缩物理缓存，也不能把主干FP4量化外推给整个attention。

## 2. 主干和残差怎样一起准备

fused_sparse_prepare以每线程32元素、每8值4pair处理BF16输入。pair评分取max(abs)，先选2pair再按原pair索引排序，保证压缩值和metadata次序一致。每32值压缩为16个四位保留值；每8值两个pair索引各占2 bit，写入metadata nibble。保留位置可以含数值0，N:M约束不能解释成恰好4个非零数。

选中值的max/6先限制在 [1/512,448]，转UE4M3再解码为实际尺度；FP4归一化值限制±6后按所选CUTLASS round_to_nearest转换。残差以**解码后的FP4值乘已编码尺度**从输入相减；掩码外直接保留原输入。再按16组对残差选择/编码尺度和FP4值，两个nibble打包成一个uint8。它接住主干量化误差，不是只把未选中值搬进第二路。

bindings要求CUDA、contiguous、BF16二维输入且K%128=0，并用torch.zeros分配压缩A、E、swizzled SFA、Q_res及SF_res，以保留padding零。稀疏CUTE布局对M/K有tile对齐，实际buffer字节不能从逻辑元素数直接推出；这里只核所选准备/host路径，没有把任意空输入、尾shape或同步正确性计为已验证。

按一组32个原激活的理想逻辑计账：压缩FP4为8B、metadata2B、主干尺度1B、残差FP4为16B、残差尺度2B，共29B/32=7.25 bit/原元素；不含PTS、对齐/尺度复制和临时BF16/输出。稀疏压缩只覆盖主干，不能宣称两路总激活仅2bit。

## 3. 共享权重需要保持相同解码值

shared_weight_nvfp4将每32个权重用同一UE4M3尺度量化为一份E2M1载荷，sparse view存一份组尺度；dense view将这个尺度复制给对应两个16组。两路仍有独立swizzled尺度buffer，但载荷不复制；其理想局部预算为4.75bit/weight，实际buffer另计padding/全局尺度。

不能用“dense按16组”推断它重新拟合了每16组权重范围。若另拟合，权重解码值会改变，共同W的残差恒等式不再直接适用。quantize_w32_shared还限制K%32=0及K/32≤1024；这些是所选接口条件，不是所有Blackwell格式的通用规范。

稀疏/稠密host分别使用对应CUTE scale布局，输入载荷uint8通过E2M1解释；scale uint8通过UE4M3解释。uint8只是字节容器，不能将其当作整数乘法。类型/尺度基础见[块缩放执行](block-scaled-matmul-execution.md)。

## 4. 融合累加仍读写稀疏中间输出

所选稀疏类型以float为累加器、BF16为C/D，sparse_matmul_tensor_scale先产生BF16 y_sparse。extra_fusion=True时，matmul_accum_tensor_scale读C_in=y_sparse、beta=1，并用**新分配的BF16 D**输出dense残差乘积与C_in之和；不是把稀疏结果一直留在寄存器，也不是原地复用同一输出buffer。CUTLASS epilogue乘全局FP32alpha指针并加入beta*C。

这避免另存完整y_res并再发PyTorch加法；仍有y_sparse的全局写回/读取和BF16舍入。extra_fusion=False则单独形成BF16 y_res再相加。与实数域 $(X_{sp}+R)\bar W$ 的等价关系应分开，不能宣称两次低精度GEMM逐bit等于一次原精度GEMM。

RMSNorm额外融合入口也不是一个CUDA kernel：bindings先发row-stats计算FP32 inv_rms/row_absmax，再torch.amax得到全局PTS，随后发第二个prepare kernel。后者重算norm、转BF16，除PTS再转BF16，才做pair选择和残差。接口把准备整合为一调用，并没有消除全局归约或两次kernel发射；其额外条件K/32≤1024也须保留。（fused_rmsnorm_sparse_residual_quantize_x。）

## 5. 模拟与native网格不能只凭名字相同

SHARQ_SIM的quantize_e2m1遍历升序可表示值，并只在diff严格变小时更新：正中点0.75会保留0.5。native所选CUTLASS转换使用round_to_nearest，exmy_base的guard/round/sticky规则落实ties-to-even，0.75对应1；这是静态教学中点差异，未运行GPU。负中点、范围限制和scale格式也须一起核对，不能宣称两入口逐bit一致。

模拟UE4M3函数是log2/尾数近似，先限制下界0.002；native使用实际UE4M3转换，下界1/512=0.001953125。SHARQ_SIM将主干解码值转BF16再构造残差，native准备在float中用编码FP4/UE4M3恢复并相减；即便整体思路相同，舍入链也不同。质量表未因本次源码核读成为native部署的数值复现。

E2M1载荷无NaN/Inf的编码定义已与原附录差异分开，见方法页；scale转换和非有限输入的完整运行行为未测。CMake明确选择sm120a或sm100a，默认sm120a；这不表示A100能执行所选Blackwell原生路径，也不表示仅凭编译选项已验证两种设备。

## 6. 导出文件和计时脚本尚不构成服务闭环

所选sharq_quant.py将Q/K/V及gate/up权重按输出行合并后量化，保存QW、SFW_sparse、SFW_dense、scale_b及保留浮点权重到safetensors，并在config写quant_method=sharq。相同name的质量wrapper将各projection的权重分别量化，不自动与合并后的PTS/weight网格相同；共享激活prepared不等于共享权重量化范围。

run_exp.py再调用外部vLLM bench latency/throughput，默认batch8、input1024、output128、500 prompts，与论文batch4/等长decode不同。脚本把config.json存在视为可跳过量化的条件，不能据此确认全部权重键、框架注册、加载消费或模型结果正确。本次未核外部vLLM改版、完整加载/设备主循环和服务曲线，未执行脚本。

## 来源身份

| 来源 | 固定身份 | 本页范围 |
| --- | --- | --- |
| actypedef/SharQ（`https://github.com/actypedef/SharQ/tree/3cc584fe48e32e59c8b4e37323470dcdd606bb11`） | 3cc584fe48e32e59c8b4e37323470dcdd606bb11 | 所选NVFP4/Llama调用、准备/共享权重、GEMM host及导出/默认计时；未核其他模型、专家、VLM/视频分支、完整框架加载或设备主循环同步 |
| 匹配 CUTLASS（`https://github.com/NVIDIA/cutlass/tree/baea077e425056e2d1d22d2bf5ed928908878790`） | baea077e425056e2d1d22d2bf5ed928908878790 | E2M1特殊值/必要转换；未核其余vendor实现或全部设备主循环 |
| SharQ（`https://arxiv.org/abs/2606.26587v1`） | arXiv:2606.26587v1 | 图2的两路表示与融合；论文报告与当前调用分别记录 |
