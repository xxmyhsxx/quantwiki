---
title: QuIP 的逆序 LDLQ、坐标恢复与浮点保存路径
type: implementation
tags:
  - weight-quantization
  - second-order
  - rotation
  - data-format
sources:
  - raw/repositories/2026-09-21/quip/source/README.md.txt
  - raw/repositories/2026-09-21/quip/source/opt.py
  - raw/repositories/2026-09-21/quip/source/method.py
  - raw/repositories/2026-09-21/quip/source/bal.py
  - raw/repositories/2026-09-21/quip/source/quant.py
  - raw/repositories/2026-09-21/quip/source/vector_balance.py
  - raw/repositories/2026-09-21/quip/source/datautils.py
  - raw/repositories/2026-09-21/quip/source/modelutils.py
  - raw/repositories/2026-09-21/quip/source/near.py
  - raw/repositories/2026-09-21/quip/source/optq_ldlq_equiv.py
updated: 2026-10-12
---

# QuIP 的逆序 LDLQ、坐标恢复与浮点保存路径

本页核固定ac92cfc7a22f6100009e2caf53bb72257d3f3184的OPT校准、预处理、Balance/LDLQ与LDLQ_RG、lazy分块、默认保存/加载及必要网格/评测入口；等价检查脚本只读其配置。未核完整Llama/zeroShot、allbal/ADMM、GPTQ设备运行或legacy Quant3Linear/CUDA消费，未导入上游、安装依赖或运行模型/GPU。论文理论与原结果由[QuIP](../methods/quip.md)承接。

所选OPT路径将变换坐标内的低位宽近似**离线恢复成原坐标稠密浮点权重**，普通Linear继续消费它，默认state_dict没有保存二位packed主体或推理变换。它可表达某种量化近似的模型质量；不直接交付W2物理压缩或相同速度/内存收益。

## 1. 校准来自逐块量化前缀

opt_sequential先关闭use_cache，用Catcher记录首block输入；每个block对所选Linear挂hook，用当前inps跑校准，统计各投影的H。所有投影的H在该block量化前收集；随后预处理/量化各权重，用修改后的block重算输出，作为下一个block的输入。它是逐block推进，不是同block内每量化一个投影就重新收集下游投影H。

QuantMethod.add_batch将输入token展平成列，FP64累加Gram，nsamples增加调用的batch样本数；post_batch除以nsamples再转FP32。因此固定长度校准下是按序列数归一化的token平方和，既没有中心化，也没有除以全部token数或乘2。不要仅凭变量H就把它称为任务损失Hessian、协方差或统一尺度的 $2XX^\top$。

默认nsamples128、seed0，seq长度取model.config.max_position_embeddings；datautils的C4校准从足够长文档抽窗口，targets只保留末token但本路径统计H不消费这些标签。`--seed`传到数据抽样的Python random；所读OPT路径没有将它用于torch/SciPy随机正交矩阵的全局设种，不能凭一个flag确认整个随机变换可复现。

find_layers按**精确type**选择nn.Linear/Conv2d，Balance拒绝Conv2d及Transformers Conv1D。该OPT范围不自动覆盖自定义Linear子类、MoE、KV压缩或全部HF架构。正常结束恢复use_cache，不是带真实生成缓存的性能验收。

## 2. 坐标和H的修改有具体顺序

preproc先可选rescale，再proj，最后gptqH处理。rescale将H除以max(abs(H))，对H对角和 $W^\top W$ 对角设1e-8下界，以四次方根比例得到列尺度A；W右乘A，H两侧除A。退化全零H在最初max归一化的运行行为未验证，后面的下界不能自动证明所有输入安全。

proj生成两个输出/输入维正交矩阵U/V，将W变为 $UWA V^\top$；先将H按trace归一化并加1e-2 I，再做 $VHV^\top$。源码实际把butterfly作用于eye构成稠密U/V，随后普通矩阵乘；不是执行一个经核验的在线FWHT。extra0使用分块随机因子/置换，extra1为不分块的两因子/置换，extra2无置换。

最后gptqH把当时零对角通道设1并置相应权重列0，再加percdamp×平均对角的阻尼；该坐标与顺序影响实际算子，不能把它合并成只对原始Gram加一次阻尼。

`--incoh_processing`确实开启pre_gptqH/pre_rescale/pre_proj并设qfn=b，但它写 **args.proj_extra=1**，下游实际读取 **args.pre_proj_extra**。后者CLI默认0，所以仅加meta flag仍走extra0；显式 `--pre_proj_extra 1` 才走extra1。这是所选固定脚本的字段连接差异，未运行默认命令，也未修改raw。CLI默认quant=nearest、wbits16，不能把不带量化参数的启动称为LDLQ/QuIP校准。

## 3. 网格、逆序反馈与可选贪心分别是什么

qfn=b在已处理权重上用全矩阵 $s=2.4\sqrt{\operatorname{mean}W^2}+10^{-16}$，令maxq=2^b−1，将W/s从[−1,1]映射到[0,maxq]并先裁剪，随后LDLQ选整数码，恢复 $s(2q/\mathrm{maxq}-1)$。2bit时四个值为±s、±s/3，没有精确零；不能换成常见对称INT2的整数零点网格。Balance在此路径局部计算s，并不把它写回quantizer.scale；默认没有用这组元数据进行packed导出。

round_ldl以Cholesky得到下三角L，将各列除以对角，减去I，随后从最后输入列向第一列量化：

$$q_i=\operatorname{clip}\left(\left\lfloor w_i+(w_{\ge i}-q_{\ge i})L_{\ge i,i}+\eta_i\right\rfloor,0,2^b-1\right).$$

这里w/q是映射到码空间的行权重；L为严格下三角，所以只消费已量化的更大列索引。与方法页统一的上三角/正序记法是不同顺序约定，不可直接照搬同一个反馈矩阵。H SPD等条件是Cholesky所需，文件存在不表示全部校准满足。

默认η=0.5，整数正中点向上；unbiased使用[0,1)随机数。nearest的torch.round与此中点规则不同。输入预裁剪、输出裁剪及有限网格也限制无偏/理论最优的适用范围。round_ldl_gptqequiv则翻转H后做上三角/正序反馈；等价脚本专门用GPTQ qfn=c、Balance qfn=a、无预处理来对齐条件，没有验证普通qfn=b、所有顺序或全部低位宽模型等价。

不启用lazy_batch时，ldlqRG先对diag(H)升序排列，再逆序处理，所以高对角列先量化；结果按同一排列写回原列。普通ldlq并不自动排序。npasses默认0；开启贪心后用存储残差s计算H反馈并更新坐标，再在每轮末裁剪。round_ldl/round_ldl_block在裁剪wr后没有同步s；若裁剪改变数值且继续下一轮，s便不再等于当前wr-w，不能称为始终使用真实残差的精确坐标下降。无界网格理论也不保证有限网格迭代损失单调。普通LDL与非lazy的LDLQ_RG入口均assert禁止unbiased与非零greedy passes同时启用，CLI的提示不取消此条件。

教学例：2bit码空间w=[3,0.5]、$H=\begin{bmatrix}1&1\\1&2\end{bmatrix}$，逆序LDL得[3,1]；第一贪心轮裁剪前为[4,0]、裁剪后为[3,0]，存储s仍[1,−0.5]而真实残差为[0,−0.5]。该结果与初值不同，若配置多轮便会带着失配s继续；这是静态数据流配合独立算术的例子，未执行作者函数或验证历史实验采用了多轮。

普通ldlq的lazy_batch用关键字传递greedy/unbiased，默认选择128列块，从右到左推进，块内反馈和已完成右侧块反馈分开计算。实数域它是同一逆序递推的重新组织，但分块矩阵乘与舍入链不能保证逐bit等于非lazy版本。

固定代码的 **LDLQ_RG + lazy_batch** 有实参错位：round_sorted_ldlqRG_block以位置参数传n_greedy_passes/unbiased，而round_ldl_block的对应位置实际为blocksize/n_greedy_passes，真正unbiased留默认False。CLI默认npasses=0时形成blocksize=0，后续range的步长为零；非零npasses成为块宽，原unbiased成为贪心次数。前述正常128列和无偏条件不能直接套到此组合。这是签名、调用与循环的静态核对，未执行上游或确认论文历史实验使用该组合。（vector_balance.py的round_sorted_ldlqRG_block、round_ldl_block与round_vecbal_Hsort。）

check_nbits只检查不同值数量不超过2^b；它不是packing检查或完整质量/速度验收。

## 4. 还原原坐标与保存决定了真实执行格式

所选qfn=b主链的quantize_weight_vecbal返回恢复数值单位的 **FP16** 权重（wr.half），而非整数索引；原模型加载参数torch_dtype=auto不改变这一返回转换。专用qfn=a/ldl_gptqequiv提前返回的分支不执行此half转换。postproc对权重逆投影 $U^\top\widehat{\widetilde W}V$，再除A，覆盖普通layer.weight。恢复后的元素通常已不属于原四点网格；W2近似来自变换空间的码选择，普通Linear消费的是稠密浮点矩阵。

OPT的 `--save`直接torch.save(model.state_dict())，opt_pack3调用被注释；load_quant也将make_quant3注释后加载普通模型。legacy Quant3Linear/pack3/load_quant3是另一组定义，不能据这些名字宣称本路径已压缩、保存了U/V/A或有原生W2 CUDA读取。QuantMethod.free清掉投影/尺度/Gram；本轮未核legacy 3bit接口的完整打包、加载和执行。

数学上，可以保留变换坐标的低位权重，推理在线做输入/输出变换；也可以像所选路径那样先还原成稠密矩阵，再做普通浮点Linear。两者在实数域表达同一近似，但物理存储、舍入及运行成本不同，见[两种恢复位置](../theory/orthogonal-rotation-and-hadamard-quantization.md#41-先恢复浮点权重与保留量化坐标分别意味着什么)。实际压缩接口需要独立证据。

## 5. 错误统计与评测还要核坐标

Balance.fasterquant保留变换坐标内的w/quant_w，调用postproc还原layer和H之后，再以这些局部变量调用error_compute。后者用当前self.H算 $\operatorname{tr}(EHE^\top)$；存在非平凡投影/缩放时，残差与H已处在不同坐标，error字段不能直接当作方法页原重构损失。应把残差和H都留在处理坐标，或将两者一致还原后比较。这是调用和对象生命周期的静态发现，未执行上游或确认历史论文实验使用同一错误统计。

opt_eval是关闭cache的固定不重叠窗口PPL：截到完整seq块、每块首token没有预测项、丢尾部，再汇总CrossEntropy。它与带历史上下文的sliding window或生成质量不同；结果仍需绑定模型、窗口与dataset具体版本。默认后评测为wikitext2/ptb-new/c4-new，不能将C4校准来源与C4新验证协议混同。benchmark调用在主入口被注释，所选脚本也不提供本次已经测过的推理时间。

已核到的是校准/预处理、所选逆序LDLQ及lazy/排序条件、网格/恢复/保存和局部诊断边界。未执行OPT、等价检查脚本、生成或CUDA，未验证全部shape/退化输入/其他方法/模型，也未复现论文PPL或性能。

## 来源身份

| 来源 | 固定身份 | 本页范围 |
| --- | --- | --- |
| Cornell-RelaxML/QuIP（`https://github.com/Cornell-RelaxML/QuIP/tree/ac92cfc7a22f6100009e2caf53bb72257d3f3184`） | ac92cfc7a22f6100009e2caf53bb72257d3f3184 | 所选OPT、Balance/LDLQ/LDLQ_RG与lazy、预处理/网格及默认浮点保存/加载；其余分支与运行未核 |
