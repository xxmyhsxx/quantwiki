---
title: QuIP# 的 E8P 打包、解码与所选 Llama 路径
type: implementation
tags:
  - weight-quantization
  - vector-quantization
  - data-format
  - gpu-kernel
sources:
  - raw/repositories/2026-09-21/quip-sharp/source/README.md.txt
  - raw/repositories/2026-09-21/quip-sharp/source/quantize_llama/hessian_offline_llama.py
  - raw/repositories/2026-09-21/quip-sharp/source/quantize_llama/quantize_finetune_llama.py
  - raw/repositories/2026-09-21/quip-sharp/source/quantize_llama/hfize_llama.py
  - raw/repositories/2026-09-21/quip-sharp/source/lib/algo/quip.py
  - raw/repositories/2026-09-21/quip-sharp/source/lib/algo/finetune.py
  - raw/repositories/2026-09-21/quip-sharp/source/lib/codebook/latticee8_padded12.py
  - raw/repositories/2026-09-21/quip-sharp/source/lib/codebook/__init__.py
  - raw/repositories/2026-09-21/quip-sharp/source/lib/linear/quantized_linear.py
  - raw/repositories/2026-09-21/quip-sharp/source/lib/linear/fused_quantized_linear.py
  - raw/repositories/2026-09-21/quip-sharp/source/lib/utils/data_utils.py
  - raw/repositories/2026-09-21/quip-sharp/source/lib/utils/math_utils.py
  - raw/repositories/2026-09-21/quip-sharp/source/lib/utils/matmul_had.py
  - raw/repositories/2026-09-21/quip-sharp/source/lib/utils/finetune.py
  - raw/repositories/2026-09-21/quip-sharp/source/lib/utils/unsafe_import.py
  - raw/repositories/2026-09-21/quip-sharp/source/lib/utils/graph_wrapper.py
  - raw/repositories/2026-09-21/quip-sharp/source/model/llama.py
  - raw/repositories/2026-09-21/quip-sharp/source/quiptools/quiptools_e8p_gemv.cu
  - raw/repositories/2026-09-21/quip-sharp/source/quiptools/quiptools_wrapper.cpp
  - raw/repositories/2026-09-21/quip-sharp/source/quiptools/setup.py
  - raw/repositories/2026-09-21/quip-sharp/source/quiptools/test_e8p.py
  - raw/repositories/2026-09-21/quip-sharp/source/quiptools/benchmark_e8p.py
  - raw/repositories/2026-09-21/quip-sharp/source/eval/eval_speed.py
updated: 2026-10-12
---

# QuIP# 的 E8P 打包、解码与所选 Llama 路径

本页核固定1d8f873e9a2a8b86b12bb1064c312c5689b77d98的Llama、E8P12二位、had、rank0主链：离线统计→BlockLDLQ/块内微调→分层产物/HF容器→所选单行CUDA消费与多行恢复。读取必要Hadamard入口与形状条件，未逐项审计大型常量矩阵、整仓attention/多GPU协议或3/4bit残差码本，未导入上游、安装、编译或运行模型/GPU。论文机制和原实验由[QuIP#](../methods/quip-sharp.md)承接。

这条路径保存E8P索引及变换向量，加载自定义QuantizedLinear；单行输入在kernel内解码，多行输入先展开FP16权重再矩阵乘。与[原QuIP所选OPT的浮点保存](quip-ldlq-and-coordinate-restoration.md)相比，它已具备所选packed消费代码，但这仍不是本轮运行验收。

## 1. 离线保存的协方差怎样回到重构二阶矩

hessian_offline_llama在原始浮点模型逐block前传，对q/o/up/down输入挂hook；q与up分别代表具有共同输入的QKV和up/gate。register_H_hook将token展平，以FP64累加 $S=\sum_t x_tx_t^\top$、$u=\sum_t x_t$ 和token数T，跨worker加和后保存FP32的下三角协方差 $C=S/T-\mu\mu^\top$ 与 $\mu=u/T$、n/ct。不是在每块量化后重新采样的量化前缀统计。

quantize_linear将flatH还原对称矩阵，并加回 $\mu\mu^\top$，得到未中心化二阶矩 $H=S/T$；然后除以平均对角并加sigma_reg I，默认1e-2。直接把保存的flatH当作最终重构H会漏掉均值方向；将其称为任务损失Hessian也不准确。归约及尺度意义见[二阶矩的合并](../theory/layer-reconstruction-second-order-compensation.md#11-二阶矩的归约与分片合并)。

离线统计CLI默认256段、4096 token；量化/微调入口默认384段、4096 token，验证128段。论文的6144段与这些默认值属于不同配置，文件同名不能证明本轮复现了论文协议。sample_rp1t只选token长度达到ctx_size的文档前段，未核数据集版本或实际样本分布。

## 2. 变换、逆序块反馈和局部微调

quantize_llama_layer将Q/K/V沿输出行融合、up/gate沿输出行融合，其余o/down单独处理。每个原投影先按全矩阵RMS归一化，再垂直拼接；融合分支的恢复尺度仍按各投影保留。可选rescale_WH先修改列尺度及H，默认不开启；had以输入/输出随机符号和归一化Hadamard处理W/H。

block_LDL用Cholesky构成单位块下三角L和8×8块对角D。以下m/n分别为输出行/输入列，公式中的W已在变换坐标并除以Wscale，量化结果也在同一归一化坐标。LDLQ从最右输入块向左处理，对当前8列块G选

$$\widehat W_G=Q_{E8P}\big(W_G+(W_{>G}-\widehat W_{>G})L_{>G,G}\big).$$

它与方法页上三角/正序记法是相配的顺序约定。E8P的round通过 $2x^\top c-\|c\|^2$ 最大化实现欧氏最近码字，fast_quantize_part利用符号/奇偶对称缩小搜索，比较两种四分之一平移。没有把任意D或 $H_{GG}$ 纳入码字距离，不能说精确求解任意块内H加权离散问题。

Wscale由除尺度前的处理权重RMS再除scale_override或码本opt_scale（默认1.03）得到；权重除Wscale后选码、恢复时乘回。默认buffered每128输入列组织逆序递推，要求n被128整除；use_fp64/lowmem_ldlq走lowmem分支，no_use_buffered走逐块版本。quip_tune_iters默认10，以H残差和块逆得到坐标目标，再做欧氏E8P选码；这是离散码的附加更新，不等于块输出微调，也不由欧氏选码保证一般H度量下每步下降。有限精度/分块版本是否逐bit一致未运行验证。

块内微调按qkv→o→up→down替换模块：离散Qidxs为buffer，SU/SV变成可训练实数Parameter，尚未量化权重和其他requires_grad参数继续参与Adam，SU/SV与其他参数用不同学习率。前三次替换后以原浮点block的输入/输出做MSE微调、验证选择best_sd；最后down替换后，本函数没有再调用一次块内微调。最终SU/SV以FP16写回分层文件，norm另存；全模型CE恢复属于另外的e2e阶段，不由本入口自动完成。

这些参考缓存由原浮点前缀产生；父入口manual_seed(args.seed)，子块量化又manual_seed(idx)。它们不是原QuIP的逐块量化前缀H，也不能仅凭同一seed旗标认定所有历史配置一致。SU/SV微调后不必为±1，初始正交理论不能原样套到最终实数向量。

## 3. 逻辑16位索引如何变成kernel载荷

E8P12有codesz8、packsz4、idx_dtype int64、version1。逻辑上每8个权重选一个16bit码字；int64是**四个码字重排后的容器**，不是每个权重或每个码字都占64bit。

底表256个int32，共1024字节；每个word放8个4bit分量，先按[0,2,4,6,1,3,5,7]重排，利用模式奇偶性修正末分量，再将半整数a编码为2a+8。逻辑码c的高8位选模式，低8位b表示符号；p为这8位的XOR，实际符号用b XOR p（翻转最低位），p还选择−1/4或+1/4。因此不能把代码低8位直接解释为8个独立符号，再假设有一个另外存放的平移位。（latticee8_padded12.py的get_packed_abs_grid/get_full_grid。）

maybe_pack_idxs一次取相邻两输出行、相邻两8D输入块的四个逻辑码。64bit的低32位按顺序放四个模式字节；高32位将四个码的符号位交错，第j个符号位位于4j+i。随后以16输出行×64输入列tile重排，最终Qidxs为m×(n/32)个int64，传kernel时view为[m/16,n/64,8,4]。理想主体位数仍为 $64m(n/32)=2mn$；不能用连续uint16原顺序数组替代同容量的物理格式。共享解释见[逻辑码与载荷](../theory/codebook-quantization-and-bit-budget.md#42-逻辑码字与物理打包为什么需要分别核对)。

如m=16、n=64，32个int64占256字节，表示1024个权重；展开FP16为2048字节。这个8倍只针对理想主体，不包含SU/SV、可选scaleWH、融合尺度、码本、未量化层、容器或运行临时空间。模块导入仍生成完整65,536×8的FP32参考网格；1KiB描述的是紧凑底表，不能扩大为整个Python进程的码本相关内存。

## 4. 保存、加载与变换消费的闭环

quantize返回用于离线参考的稠密hatW，同时attr保存packed Qidxs、FP16 SU、已融合Wscale的FP16 SV及可选scaleWH；单投影RMS也融合进SV，融合投影则保留各分支scales。分层文件还有shapes/fused；配置保存codesz/packsz/idx_dtype、codebook/version和其余量化参数。普通加载不能只拿Qidxs而丢掉这些恢复信息。

hfize_llama构建自定义Llama，拷入Qidxs、SU/SV、可选scaleWH及fuse_scales、norm/可选head，保存HF容器；model_from_hf_path见quip_params选择该Llama类。首次QuantizedLinear forward按codebook_id构建对应消费类并检查version，将索引注册成设备buffer，legacy Wscale融合进SV。非持久Hadamard矩阵由维度构造，不从checkpoint读回。

flatten后的输入行数r控制E8P12路径，以下先按输入列n、输出行m记形状：

| 条件 | 实际消费 | 代价与精度边界 |
| --- | --- | --- |
| train_mode=False，r=1 | 输入FP32乘SU/可选逆列尺度，归一化输入Hadamard后除32、转FP16，packed E8P融合解码/MMA | kernel输出FP32；输出Hadamard后乘SV×32，再转回调用输入dtype；没有完整m×n解码权重临时矩阵 |
| train_mode=False，r>1 | 同一输入变换，decompress_packed_e8p展开FP16权重，再FP16 matmul | 每次forward创建m×n展开矩阵，再输出变换；r是全部前导维展平行数，不能只按请求batch决定是否走GEMV |
| train_mode=True | cache_WH预先展开并做两侧Hadamard/除32，后续普通FP16 matmul | 缓存稠密FP16矩阵，SU/SV在两侧消费；它不是仅保持二位权重的执行方式 |

除/乘32在理想表达中抵消，但位置和FP16转换影响数值，不能删除后声称逐bit等价。融合输出按Q/K/V或up/gate拆开再乘各分支尺度；所选Llama MLP随后做激活/门控、down，要求pretraining_tp≤1。

get_hadK只支持2幂及指定K×2幂维度，K表包含28、52、172等；实现没有通用补零接口。当前train_mode首次构缓存无条件访问had_left_T.T，而纯2幂输入时get_hadK返回None：这是该可选分支的静态前提缺口，未执行以确认报错或修改raw；默认ft_train_mode关闭。模块还assert rank==0，不能仅因CLI仍有lora_rank就宣称当前加载支持低秩修正。

## 5. CUDA恢复和乘加用什么类型

所选host检查CUDA/contiguous，FP16输入、int64索引、int32底表，n%64=0、m%16=0、m/n<65536；这比量化入口的m%2/n%4更严格，buffered又另需n%128。文件存在不证明任意shape或设备满足接口，未核所有外部调用/设备组合。

quiptools_e8p_gemv.cu将每int64作为uint2：一半是四模式、一半是交错符号；XOR折叠得到四个奇偶bit，用nibble XOR恢复符号。lop3、移位和FP16位模板配合add.f16x2恢复四分之一网格；随后mma.sync.aligned.m16n8k16用FP16权重/输入、FP32累加。32×32线程按输入64列块分工，对16行输出用FP32 atomicAdd汇合；这不是INT2整数乘法，原子汇合也不保证逐bit确定的归约顺序。

注释掉的异步拷贝/共享权重流水不是实际执行证据。多行分支同样按packed布局解码，但将half2写成完整FP16矩阵再调用PyTorch matmul，不能将单行融合kernel的临时存储或速度概括为所有prefill/decode。

eval_speed默认BS1、seqlen1、100次，重复同一token张量，use_cache=False；预热后在循环前后同步设备，将samples次model调用的总时长除以samples，得到平均调用时间。默认可用CUDA graph重放。loader固定attn_implementation='sdpa'，所读use_flash_attn参数没有参与选择。该脚本测量范围不是含真实历史KV的完整generate，也不自动等同论文4090或README RTX6000 ADA数据。test_e8p/benchmark_e8p使用另一组旧接口/输入，且正确性断言或计时被注释，不能用“有测试脚本”代替所选packed GEMV验收。

## 来源身份

| 来源 | 固定身份 | 本页范围 |
| --- | --- | --- |
| Cornell-RelaxML/quip-sharp（`https://github.com/Cornell-RelaxML/quip-sharp/tree/1d8f873e9a2a8b86b12bb1064c312c5689b77d98`） | 1d8f873e9a2a8b86b12bb1064c312c5689b77d98 | 所选Llama/E8P12/rank0/had统计、编码/微调、保存/加载与packed单行CUDA及多行恢复；其余分支/运行未核 |

固定身份依归档登记；本轮未取得新的独立Git对象或归档，不能把历史60文件完整性记录称为本轮重新比对。教学位恢复/格式计数及静态代码审查只支持所述机制，未复现模型质量、CUDA数值、同步/越界或性能。
