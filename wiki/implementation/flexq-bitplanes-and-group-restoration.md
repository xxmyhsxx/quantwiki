---
title: FlexQ 的位平面、组恢复与所选调用接口
type: implementation
tags:
  - data-format
  - kernels
  - group-quantization
  - performance
sources:
  - raw/repositories/2026-09-21/flexq/source/algorithm/main.py
  - raw/repositories/2026-09-21/flexq/source/algorithm/models/int_llama_layer.py
  - raw/repositories/2026-09-21/flexq/source/algorithm/flexq_quantize/flexqllm.py
  - raw/repositories/2026-09-21/flexq/source/algorithm/flexq_quantize/utils.py
  - raw/repositories/2026-09-21/flexq/source/algorithm/flexq_quantize/int_linear.py
  - raw/repositories/2026-09-21/flexq/source/algorithm/flexq_quantize/quantizer.py
  - raw/repositories/2026-09-21/flexq/source/engine/common/base.h
  - raw/repositories/2026-09-21/flexq/source/engine/CMakeLists.txt
  - raw/repositories/2026-09-21/flexq/source/engine/src/pack/bit_packing.cu
  - raw/repositories/2026-09-21/flexq/source/engine/src/bgemm/bgemm.cuh
  - raw/repositories/2026-09-21/flexq/source/engine/src/bgemm/flexq_bmma_kernel.h
  - raw/repositories/2026-09-21/flexq/source/engine/src/bgemm/flexq_bmma_op.h
  - raw/repositories/2026-09-21/flexq/source/engine/src/bgemm/flexq_bmma_library.h
  - raw/repositories/2026-09-21/flexq/source/engine/test/test_kernel.h
  - raw/repositories/2026-09-21/flexq/source/engine/test/test_w6a6_kernel.cu
  - raw/repositories/2026-09-21/flexq/source/engine/test_bgemm_kernel.cu
  - raw/repositories/2026-09-21/flexq/source/e2e/src/fastertransformer/kernels/flexqgemm/flexq_gemm_wrapper.cu
  - raw/repositories/2026-09-21/flexq/source/e2e/src/fastertransformer/kernels/flexqgemm/src/pack/bit_packing.cu
  - raw/repositories/2026-09-21/flexq/source/e2e/src/fastertransformer/layers/FfnLayer.cc
  - raw/repositories/2026-09-21/flexq/source/e2e/src/fastertransformer/models/llama/LlamaDecoderLayerWeight.cc
  - raw/papers/2026-09-21/flexq/paper.pdf
updated: 2026-10-12
---

# FlexQ 的位平面、组恢复与所选调用接口

本页核固定a42424d843adffebe4b28ddb62a760cb52b0d2f0：Llama W6/group128、down A8的质量入口；engine的signed W6A6/W6A8、M=1/4/8、K为128整数倍的packing/BMMA/恢复路径；所选FasterTransformer wrapper和FFN调用。方法与原公式边界见[FlexQ](../methods/flexq.md)。没有执行上游、编译或运行GPU，也未将整仓、全模型/全部shape、分布式及完整导出/加载链计为已核查。

## 1. 质量路径是浮点模拟

main.py的flex_linear_quant使普通线性层输入6 bit、down输入8 bit；group参数选择per_group，symmetric开启时会将disable_zero_point设为True。weight_bits由wbits指定为6。Q/K/V/attention概率量化参数设为16 bit，不应将W6Ax扩展成INT6 attention或压缩KV。

flexqllm逐block创建QuantLlamaDecoderLayer，打开量化标志、weight_quant_inplace写回模拟权重，保存量化参数并将层移回CPU。QuantLinear在前向仍通过量化器得到浮点weight/input，调用F.linear；既无bit-plane保存，也无原生BMMA调用。这个入口不需要额外校准输入拟合参数，但动态激活范围仍在前向计算。

UniformAffineQuantizer在group128上取min/max，对称时 $s=\max|x|/(2^{b-1}-1)$，以1e-5/1e4限制尺度；所选disable_zero_point=True对应signed范围 $[-2^{b-1},2^{b-1}-1]$。其他CLI配置可进入非对称分支，不能用所选设置代表全部行为。8-bit down是固定模块规则，不是运行时敏感度搜索。

## 2. 整数packing与在线量化是不同入口

engine的flexq_packing_kernel吃已量化的int矩阵，提取低q位，ballot收集各lane的bit，再用brev按顺序写4个uint32。所选 $K\bmod128=0$、M=1/4/8时，可解释为 $[K/128,M/c,q,c,4]$ 的uint32载荷，$c=\min(M,8)$；每个标量共q个bit。权重按N个输出行存储，读取时实现与激活的逻辑转置点积。

FasterTransformer的另一个flexq_bit_packing_kernel吃FP16，先按128组取absmax/上界，舍入/clip为signed码，再打包。激活尺度写为FP16并复制两份；SCALE_SIZE_X将M对齐到4，故小M的尺度容量不等于一个参数/组的理想元素预算。普通FP16输出、对齐与workspace还需计费。

所读native在线量化段没有模拟量化器的1e-5/1e4限制；范围为零、极小/极大值及所有尾部形状未运行。metadata恢复与在线编码还受FP16舍入影响，不能从共同的“max/31”名称直接推出与模拟逐bit一致。

## 3. AND-popcount不直接给出多位signed点积

bgemm.cuh的三参数bmmaSync采用mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc，INT32结果从0开始，计算位平面沿128个K元素的AND-popcount。它不是XOR的Hamming距离，也不是一次6-bit乘积。选定库实例为signed W6A6/W6A8。

flexq_bmma_kernel.h对每个位对累积结果。最终temp_mul从 $2^{r+s}$ 开始；若恰好一侧为最高位就取负，两侧最高位则取正，落实二补码的 $\lambda_q(r)\lambda_p(s)$。超出位数的MMA行列清零。论文式6漏掉的位权在此存在；这里的数学位对数量不能直接作为发出的kernel数量或全部tensor指令数量。

每128组的INT32位对结果与两侧FP16尺度经__hmul2得到的FP16乘积相乘，在float cfrag_float中跨组累加；随后再做float位权/warp归约、转half2存储，输出FP16。若先把不同组的整数结果无尺度合并，通常不等价；若先在FP32乘尺度，也与当前舍入链不同。代码恢复顺序是“每组按尺度累加每个位对，再按位权合并”，数学上可交换的线性求和在有限精度下仍有次序差异。

## 4. 小M布局、流水与固定条件

位维度进入MMA行列布局，将有用位对填入固定8×8×128二值指令空间。对c=1/4/8，小M chunk归约循环分别追加3/1/0轮shuffle；它补全同一输出跨线程的位贡献，没有减少36/48个逻辑位对本身。不能把“减少空行计算”写成完全没有padding。

模板硬编码GROUP_SIZE=128、MMA_K=128，要求stage>1及warp/block K相配；所选代码通过GPU_ARCH>=80的编译分支。bias初始化当前直接assert不支持，接口上有bias参数不代表该路径实现bias。公开库实例有限，不证明任意shape均有合适配置或可正常执行。

mainLoop把位载荷和尺度分stage放进共享内存，cp.async预取、同步后读取寄存器与BMMA；不能仅凭异步指令存在就声称所有访存已隐藏。W读取在GPU_ARCH>=89或==80时不加evict_first，其他所选分支调用带L2提示的cp.async；该策略需按硬件/workload测量，非所有架构通用保证。

## 5. 所选框架wrapper怎样调用

FLEXQGEMMWrapper的packed重载先检查K≥128且K%128=0，按W6A6或W6A8及M=1/2/4/其他选择固定kernel配置，并检查initSuccess再执行。代码注释将默认配置绑定Llama/A6000；不能把它当作调用时对全部硬件自动搜索。

FP16重载先在线量化/packing到workspace，再调用packed重载；x_scale/w_scale虽以float指针经过接口，wrapper按half读取，参数的名字/指针类型不决定真实字节格式。FfnLayer的int8_mode=5分支调用W6A6处理gate/up；更广泛norm/激活融合、attention和所有backend尚未逐段核查，论文图5的完整融合不能计为这里全部验证通过。

LlamaDecoderLayerWeight的mode5分支读取缩小形状的weight bin，但本次没有贯通质量模型到这个文件的导出、bit-plane排列、尺度初始化及全部加载消费。engine源码和框架wrapper路径同时存在，不证明模拟质量配方已完整导出到同一真实模型运行。

## 6. 基准计时和检查的实际含义

engine/test的benchmark分别计GEMM执行与activation packing，weight packing在测量前完成；不能把exec时间当作含全部在线量化、packing和整网运行的时间。报告普通TOPS的操作数为 $2MNK$，B-TOPS另乘 $p q$，二者口径不同。

test_kernel.h对FP16结果的通过门限是绝对误差 $0.0001\times65504=6.5504$，并非每个结果的相对0.01%；随机矩阵/随机尺度及这一宽容差不能替代模型质量评测。该上游检查器未在本项目执行，不能因文件含PASSED字符串或返回0就宣称结果已验证。

已确认的是所选表示、符号位、组尺度、FP16/FP32舍入、接口及基准口径；仍未验证真实权重导出闭环、编译/同步正确性、全部shape和硬件、模型/GPU质量或性能。论文报告保留原测量条件，不能扩展成当前环境实测。

## 来源身份

| 来源 | 固定身份 | 本页范围 |
| --- | --- | --- |
| [FlyFoxPlayer/FlexQ](https://github.com/FlyFoxPlayer/FlexQ/tree/a42424d843adffebe4b28ddb62a760cb52b0d2f0) | a42424d843adffebe4b28ddb62a760cb52b0d2f0 | 所选Llama模拟、engine signed位平面/BMMA/恢复、默认测试与框架wrapper；未核全仓、完整导出/加载及全部融合分支 |
| [FlexQ](https://arxiv.org/abs/2508.04405v2) | arXiv:2508.04405v2 | 位展开、group与runtime设计；式6缺位权与代码temp_mul分别说明 |
