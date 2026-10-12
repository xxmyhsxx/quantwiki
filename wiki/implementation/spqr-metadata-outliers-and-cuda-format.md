---
title: SpQR 的两级尺度、稀疏保护值与所选 CUDA 格式
type: implementation
tags:
  - weight-quantization
  - second-order
  - sparsity
  - data-format
  - gpu-kernel
sources:
  - raw/repositories/2026-09-21/spqr/source/README.md
  - raw/repositories/2026-09-21/spqr/source/main.py
  - raw/repositories/2026-09-21/spqr/source/spqr_engine.py
  - raw/repositories/2026-09-21/spqr/source/quant_groups.py
  - raw/repositories/2026-09-21/spqr/source/weight_permutation.py
  - raw/repositories/2026-09-21/spqr/source/modelutils.py
  - raw/repositories/2026-09-21/spqr/source/convert_legacy_model_format.py
  - raw/repositories/2026-09-21/spqr/source/convert_to_hf.py
  - raw/repositories/2026-09-21/spqr/source/inference_lib/src/spqr_quant/inference.py
  - raw/repositories/2026-09-21/spqr/source/inference_lib/src/spqr_quant/sparse_util.py
  - raw/repositories/2026-09-21/spqr/source/inference_lib/src/spqr_quant/inference_kernels/cuda_kernel.py
  - raw/repositories/2026-09-21/spqr/source/inference_lib/src/spqr_quant/inference_kernels/kernel_selector.py
  - raw/repositories/2026-09-21/spqr/source/inference_lib/src/spqr_quant/inference_kernels/common.cuh
  - raw/repositories/2026-09-21/spqr/source/inference_lib/src/spqr_quant/inference_kernels/spqr_cuda.cpp
  - raw/repositories/2026-09-21/spqr/source/inference_lib/src/spqr_quant/inference_kernels/spqr_cuda_kernel.cu
  - raw/repositories/2026-09-21/spqr/source/inference_demo.py
  - raw/repositories/2026-09-21/spqr/source/bench_matvec_kernel.py
  - raw/repositories/2026-09-21/spqr/source/tests/test.py
updated: 2026-10-12
---

# SpQR 的两级尺度、稀疏保护值与所选 CUDA 格式

本页核固定543d10e56f921e5eeb10d5f0aaaccd30b6662d0e的所选Linear校准、逐列补偿/保护、两级元数据、旧保存/浮点恢复、3bit与16×16的转换、CSR/PTCSR及packed matvec。方法与论文结果由[SpQR](../methods/spqr.md)承接；未核全部模型、lm-evaluation-harness、任意组宽/尾块、完整HF加载或设备同步/越界正确性。未导入上游、安装、编译、运行模型或GPU；cuda_kernel.py在导入时会JIT编译，文件阅读没有执行它。

主链有两种消费：旧统计文件能恢复普通浮点Linear；优化格式把码和元数据重排成64bit字，并在CUDA内恢复FP16权重、FP32累加稠密与稀疏贡献。旧保护值的覆盖语义和新接口的相加语义必须核对，不能把“有CUDA代码”当作保存产物已无误转换。

## 1. 校准前缀、顺序和误差代理

main.quantize_spqr捕获首block输入并关闭cache。每组投影先挂hook跑当前block，移除hook后量化/覆盖权重；block量化后输出成为下一block输入。true_sequential时Llama组为Q/K/V→o→up/gate→down，后组校准包含前组修改；不开启时所有投影先收集H。CLI默认wbits16、perchannel关闭、groupsize/QQ位宽未设、outlier_threshold为∞；这不构成低位宽或稀疏配方，需明确配置。

SPQRUtil.add_batch将token展平，以FP32流式加权保持 $H=(2/N_s)\sum_t x_tx_t^\top$，$N_s$为调用的batch序列数；既非中心化协方差，也非按全部token数平均。get_permutation_order可用identity、H对角降序act_order，或按输出权重列秩向量的Spearman相关做贪心邻接；后者不是按任务Hessian敏感性排序。

权重及H同置换，先记录零对角通道，加percdamp×mean(abs(diag H))阻尼，再将原零对角设1、对应权重列设0。对SPD H求逆及上三角Cholesky U，$H^{-1}=U^\top U$；默认blocksize128，按输入列从左向右，以 $e_j=(w_j-d_j)/U_{jj}$ 向后反馈，块外用addmm批量更新。有限精度、分组重拟合和保护会影响结果，未执行分块等价检查。

最终逐元素保护依据 $e_{ij}^2>\theta$；代码的相对阈值为

$$\theta=\tau\,\operatorname{mean}_j\!\left(\frac{\operatorname{var}_i W_{ij}}{U_{jj}^2}\right).$$

这里τ为outlier_threshold，W是置换/死通道处理后的浮点初值，var跨输出行且使用torch.var默认修正；逐列待量化w则已吸收先前反馈。U对角不能换成原H对角或完整逆矩阵同一列的对角；该代码代理没有额外1/2，与方法页采用的损失归一化需分别记录。

## 2. 先估计保护收益，再使用已恢复元数据

有限阈值且未启用simplified_outliers时，每个输入列组先做leave-one-out：对每个候选排除它、重拟合基础网格，比较全部保留元素的 $\sum_j(w_j-d_j)^2/U_{jj}^2$ 与全组基础误差，预测哪些排除能降低足够多误差。它不是只测候选自己的幅值；这里试拟合不启用QQ元数据量化。

预测例外以其余元素均值替换后，才拟合真正的网格；分母有clamp_min(1)。实际逐列补偿后再按当前误差选最终mask，两种mask不必相同。simplified_outliers只省掉组内重拟合，仍可执行最终误差保护。

每个输出行、每groupsize个连续输入权重，使用 $q=\operatorname{clip}(\operatorname{round}(w/s+z),0,2^b-1)$、$d=s(q-z)$。非对称默认round_zero为False/CLI None，z为实数；xmin/xmax直接来自组值，不自动扩展到包含0，相同端点另设[-1,1]。不要将它替换为整数零点契约。

QQ将当前输入组的每输出行s/z向量reshape为[-1,qq_groupsize]：**二级分组跨输出行**，不沿同一输出行跨多个输入组。先量化s/z并恢复，再用恢复的值量化权重和计算误差；压缩元数据不是最后才做。qq_zero是否执行还取决于round_zero与所设位宽，转换接口不能只检查wbits。

| 对象 | CLI/代码字段 | 本页记法 |
| --- | --- | --- |
| 一级权重组的输入列数 | groupsize；ModelArgs.beta2 | $g_w$ |
| 同一输入组的二级输出行数 | qq_groupsize；ModelArgs.beta1 | $g_q$ |

方法页沿论文把一级/二级组宽称β1/β2；代码配置中名称次序相反，beta1/beta2的docstring也不能替代reshape和消费索引。g_w=g_q=16时预算不变，其他组宽需按实际轴和设备支持单独核查。

## 3. 保护值在旧恢复中覆盖，在优化接口中相加

最终mask位置将输入0再量化，保存 $q_0=Q(0)$；浮点返回权重直接取当前w，保护位置反馈误差为0。保存的outliers_matrix是**当前绝对保护值w**，不是w减主体解码的残差；quant_weights/元数据/outliers/perm仍在置换坐标，返回weight才按逆置换恢复原列。

modelutils.layer_weight_dequantization恢复主体后，以outliers_matrix非零位置覆盖，再逆置换。普通加载覆盖Linear.weight，未交付低位packed驻留；以非零值作为mask还与“有独立bool保护mask”不同。保存元数据硬写keep_last_columns0/blocksize128，本页只按该默认范围核查，不扩展为任意保留列配置。

convert_legacy_model_format.load_legacy_tensor将同一outliers_matrix转CSR，值half、列short、累计计数int32；from_legacy只重排码和打包索引，没有减去主体的 $d_0=s(q_0-z)$。所选CPU dequantize及CUDA matvec均把稀疏值**相加**，所以这一接口直接相加时，保护位置成为 $w+d_0$；只有d0为0或此前已另行校正，才与覆盖w一致。代码没有据z实数自动保证d0=0。

教学例：s=0.5、z=1.25、3bit网格给q0=1、d0=−0.125；保护目标w=3，覆盖恢复为3，直接相加为2.875。正确相加残差应为3.125。这里是源码数据流与独立算术，没有运行作者函数或断言论文历史实验采用这一转换。

分层文件、args.pt及not_quantized_weights共同保存；转换脚本只替换存在文件的无bias Linear，并输出含QuantizedLinear的PT容器。非量化参数覆盖在所选main里受save_per_layer条件控制，完整保存/加载需核实际命令。convert_to_hf的所选main直接save_model保存模块并记录参数shape；本页只核容器入口，不证明外部Transformers版本支持加载。

## 4. 逻辑3.625位如何成为实际4位稠密载荷

所选converter固定位宽BITS=3；CUDA宏也实例化3/16/16，Python可接受bits/beta1/beta2参数不表示设备支持其他组合。优化格式每16个输入权重、每输出行保存一个uint64：

| 字段 | 位数 | 位置/作用 |
| --- | --- | --- |
| 一级s码/z码 | 3+3 | 最低6位 |
| 16个权重码 | 16×3=48 | 接下来48位 |
| 二级FP16参数的片段 | 8 | 位54–61 |
| 保留空位 | 2 | 最高2位 |

四个FP16二级值共64bit，片段按floor(row_in_tile/2)取8bit，每个字节在相邻两行重复；设备warp shuffle按线程间2/4/8恢复。这使16行中元数据片段共占128bit，比逻辑64bit多一份；另有每行2bit对齐。16×16主体/元数据tile为128字节，理想3.625位只有116字节，因此此格式稠密部分为 **4 bit/weight**，不能引用论文3.625作为物理读取。

以m为输出行数、n为输入列数，buffer按 $16\lceil m/16\rceil\lceil n/16\rceil$ 个int64分配；m/n整除时共mn/16字。余行/列的分配或清零不证明尾块kernel安全。本页不对未对齐形状作运行承诺。第二级tile参数按输入块优先索引，载荷按输出块/输入块/输出行排列，不能仅按flatten原码推断物理地址。

在CSR下，p为真实保护值密度，额外列/值约32p位，每行累计指针为 $32(m+1)/(mn)$ 位，另有每输入列16bit置换，即16/m位；若perm未保存则去掉。主体4位仍覆盖全部位置，稀疏项另加。容器、未量化参数、边界/对齐和运行缓存另计。

PTCSR每16输出行取最大行长度k，把16行的条目交错并补到16k，row_offsets按行块存；col_vals包含零填充。此时nnz/density计入填充，不能当真实保护比例，存储与负载平衡之间要结合实际分布判断。[格式预算与保护协议](../theory/codebook-quantization-and-bit-budget.md#52-保护值应覆盖还是相加由保存协议决定)区分这些口径。

## 5. packed matvec怎样恢复并消费

两级尺度先由half2减法/乘法恢复FP16 s/z，再以同样操作恢复FP16 d；不是将所有QQ参数提升FP32再计算。输入为FP16，转FP32后使用fmaf累加主体，稀疏half值乘输入也并入FP32，跨线程归约后转FP16输出。这与校准时的FP32量化近似、或旧浮点恢复路径不保证逐bit一致，保存元数据的dtype也会改变网格。

in_perm把两相邻原输入索引存成一个int32，kernel按ushort2读取，在线gather到共享输入再计算；置换要与W的列、保护列同坐标。稀疏ColVal的低16位列/高16位half值，在所选接口还读为signed short并由Python short转换；非负列小于32768是本页的范围，不能仅从16bit字段宣称支持全部65536列。perm的unsigned16容量、稀疏列的signed前提及n为偶数是不同条件。

CSR由row_offsets长度m+1判定，否则按PTCSR行块读取；分支都在同一matvec中恢复主体并加入稀疏。这是所选固定代码的融合实现，论文原始数据不能直接作为本版本的运行证据。同步调用的feature flag未设IS_ASYNC，spqr_matvec末尾cudaDeviceSynchronize；flag名字包含FP32不是对所有外部选项已验证的证据，host错误返回也未向Python上层传播。

QuantizedLinear.forward按x.shape[1]循环逐向量调用，最后固定reshape[1,seq,m]，不是原生矩阵乘或一般批量接口。本页按[1,seq,n]且FP16/连续输入解释；prefill多token会执行多次matvec，不能把单向量低位读取直接概括为大batch吞吐。

原测试以人工构造码/元数据/稀疏值对照CPU dequantize，所读fused辅助传identity排列；它没有核真实legacy绝对保护值转换的语义，也不证明任意置换。bench_matvec_kernel的计时段设置2000次、丢前10次再取最小值，主要计局部无perm入口，CSR/PTCSR一致断言被注释；ptcsr_path虽为可选参数，文件循环仍直接用它拼路径，省略时的静态前提没有闭合。计时表达不保证整个入口已可运行，也与带模型、真实cache的端到端时间不同。inference_demo虽构StaticCache逐token生成，其所选计时没有显式设备同步，CUDA内部同步和可选compile又会改变边界，不能作为本轮性能实测。

## 来源身份

| 来源 | 固定身份 | 本页范围 |
| --- | --- | --- |
| [Vahe1994/SpQR](https://github.com/Vahe1994/SpQR/tree/543d10e56f921e5eeb10d5f0aaaccd30b6662d0e) | 543d10e56f921e5eeb10d5f0aaaccd30b6662d0e | 所选Linear校准/保护、两级分组、旧保存/恢复、3/16/16优化转换及CSR/PTCSR融合matvec，必要容器/测试/计时入口 |

固定身份和历史885文件完整性依归档登记，本轮没有新的独立Git对象/归档可重新比字节；只读所选文件并核本地指纹。未取得新材料、执行作者函数/模型/设备或复现论文质量/性能，未完成全部模型/形状/配置、HF外部加载和设备生命周期正确性审查。
