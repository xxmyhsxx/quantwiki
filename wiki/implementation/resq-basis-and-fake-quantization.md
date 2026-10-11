---
title: ResQ 实现：未中心化基底、权重融合与浮点模拟量化
type: implementation
tags:
  - mixed-precision
  - rotation
  - fake-quantization
  - kv-cache
sources:
  - raw/repositories/2026-09-21/resq/source/fake_quant/0_get_basis.sh
  - raw/repositories/2026-09-21/resq/source/fake_quant/2_eval_ptq.sh
  - raw/repositories/2026-09-21/resq/source/fake_quant/get_basis.py
  - raw/repositories/2026-09-21/resq/source/fake_quant/ptq.py
  - raw/repositories/2026-09-21/resq/source/fake_quant/utils/data_utils.py
  - raw/repositories/2026-09-21/resq/source/fake_quant/utils/process_args.py
  - raw/repositories/2026-09-21/resq/source/fake_quant/utils/fuse_norm_utils.py
  - raw/repositories/2026-09-21/resq/source/fake_quant/utils/quant_utils.py
  - raw/repositories/2026-09-21/resq/source/fake_quant/eval_utils/main.py
  - raw/repositories/2026-09-21/resq/source/fake_quant/eval_utils/rotation_utils.py
  - raw/repositories/2026-09-21/resq/source/fake_quant/eval_utils/gptq_utils.py
  - raw/repositories/2026-09-21/resq/source/fake_quant/eval_utils/modeling_llama_2.py
  - raw/papers/2026-09-21/resq/paper.pdf
updated: 2026-10-12
---

# ResQ 实现：未中心化基底、权重融合与浮点模拟量化

固定 [project-resq](https://github.com/utkarsh-dmx/project-resq/tree/cb87d0d059455858daaa66f7410da4afebee978a) 提供基底生成、旋转融合、浮点模拟量化和模型评测。这里核读 Llama、`rotate_mode=resq`、`rotation_granularity=full_shared`、高精度比例0.125、低精度附加比例0、非训练旋转、`config.pretraining_tp=1` 的主链，并追踪按头K量化及选定eager attention缓存调用；不代表全部模型、attention后端、TP切片或训练分支已核验。没有导入、安装或执行上游。

[方法页](../methods/resq.md)解释子空间混合精度与原文证明问题。本页回答：统计量来自什么前向，所选基底怎样进入权重，量化后的对象怎样保存和被消费，以及论文性能实现是否在快照中。

## 1. 示例命令与论文配置需要分开

`0_get_basis.sh` 和 `2_eval_ptq.sh` 的实际模型为 **Llama-3.2-1B**。评测脚本启用A/K/V=4、高组8、`high_fraction=0.125`、`full_shared`、头内K/V group size 64，**`w_bits=16`**；因此该命令不会进入权重量化分支，不能直接当成论文W4A4KV4复现命令。README所述3B示例与脚本身份也不同。脚本还设BF16模型参数，量化器的16-bit旁路只表示保持传入浮点dtype，并不把它强制改成FP16。

欲解释代码里的W4路径，本页另外核读 `w_bits<16` 且 `w_rtn=False` 的GPTQ分支，保留 `act_order=False` 默认条件。没有修改脚本、运行模型或确认这些参数组合的实际质量。K group size需匹配头维度；默认 `k_groupsize=-1` 虽通过wrapper构造断言，前向会明确 `raise NotImplementedError`，脚本使用的64头内分组才是所选已实现分支。（`process_args.py`；`QKRotationWrapper`。）

## 2. 基底来自未量化前向的二阶能量

`get_basis.py::get_basis` 以FP32加载模型并融合归一化增益；WikiText加载器将训练文本以换行拼接、随机采连续token片段，基底计算使用输入而非最后一位目标标签。hook采attention/MLP输入、V输出、RoPE后的K和down输入。当前块的浮点输出成为下一块输入：这里不是逐块GPTQ之后重新统计量化前缀。

每批将特征按token展平，累计 $X^\top X$。乘法先转double，累计缓冲由默认浮点 `torch.zeros` 建立，再作double特征分解；不能写成全过程FP64累计。代码没有减均值，故统计量为未中心化Gram。共享隐藏基底的归约为

$$H_A=\frac{\sum_{j=1}^{L}(H_{attn,j}+H_{mlp,j})}{2LNS},$$

其中 $L$ 为层数，$N$ 为校准批数，$S$ 为每批所选固定序列长度；各层两条输入统计使用相同采样长度，才对应此平均。V按每层KV头分别选基底；RoPE后K按KV头归约成每层共享头内基底，使GQA的Q/K可以使用同一个头维变换。（`full_shared` 分支。）

`perform_eigen_decomp` 添加 $0.01\operatorname{mean}(\operatorname{diag}H)I$ 后 `torch.linalg.eigh`，按特征值递增保存基向量，高组取最后 `int(high_fraction*dimension)` 维。标量I在精确算术中只平移特征值，不改变特征子空间；有限精度及重根的具体基向量仍可能不同。两组各自保存方阵随机旋转，低组0.0在本范围没有第三精度分组。

保存产物含共享 `attn_mlp`、每层value/key_pos/down基底、对应特征值及组内旋转。已有基底文件的缓存分支会跳过重新统计，路径存在不等于已核验模型、数据、配置和校准版本全部匹配。代码虽然计算down的PCA基底，所选推理融合仍使用Hadamard处理down，不能由“保存了down基底”推断它已被该路径消费。

## 3. 融合方向和头内重排怎样配套

PyTorch Linear权重形状为输出×输入，前向 $xW_t^\top$。`rotation_utils.py::fuse_basis_to_model` 调度到 `fuse_basis_shared`，构造 $U_A=P\operatorname{diag}(R_l,R_h)$：输入投影权重右乘 $U_A$，输出投影权重左乘 $U_A^\top$，embedding/head和偏置按相应方向处理。范数增益先吸收入侧线性层。`fuse_norm_utils` 还对embedding逐行减均值，该额外预处理不能仅凭正交恒等式就宣称整个准备流程对任意模型严格等价。

value头内变换通过 `rotate_ov_proj` 在V与o两侧成对应用。`rearrange_o_proj` 把各attention头末尾的高组列集中到整体末尾，同时重排o权重输入列并记录 `new_column_order`。`ActQuantWrapper` 先在原来的逐头分组上模拟量化，再按同一顺序重排激活，然后执行Linear；成对列重排保持所选未量化线性乘积，高精度名额不会凭空增加。

down权重预先变换，wrapper在线执行完整Hadamard后再量化；由于前面有gate/SiLU，不能把这个在线激活变换随意融合进up/gate。`main.py::ptq_model` 对down清除高/低附加组长度，默认采用统一低位宽；`int8_down_proj` 则是另一个配置。Hadamard的FP32选项和输出转回原dtype也影响实际舍入顺序。（`rotation_utils.py`、`quant_utils.py::ActQuantWrapper`。）

## 4. GPTQ 和量化器写回的是浮点近似

`ptq_model` 仅在 `w_bits<16` 时量化权重：选择加载已有模型、GPTQ或RTN。所选GPTQ依次设置输入列中部的低位量化器与尾部高位量化器，按块逐列量化，并借逆Hessian因子把误差传播到尚未量化列；down使用统一位宽。高精度列同样参与反馈，不是独立保留一个误差残差矩阵。这里保留未启用act-order的范围，不推广到重新排序后所有混合精度配置。

`WeightQuantizer.quantize`、`ActQuantizer.forward` 和STE前向返回已反量化的浮点值。GPTQ最终将 `Q` 写回 `layer.weight.data` 的原浮点dtype，wrapper将浮点近似送入普通 `nn.Linear`。非对称代码使用整数零点与恢复尺度；原文式1中的 $z_X=\min X$ 是实数偏移，两种记法不能直接当同一数值接口。

若启用保存，`save_dict` 中包含量化器与模型 `state_dict`，而非本文已核的INT4/INT8 packed权重协议。激活尺度/零点按所选token/组临时求取，wrapper之后释放；这些模拟网格能用于质量评估，不直接体现低位存储节省或整数Tensor Core执行。（`gptq_utils.py::fasterquant`、`quant_utils.py`、`main.py`。）

## 5. RoPE之后的Q/K与浮点缓存

`ptq_model` 对 `k_bits<16` 在RoPE函数调用后加wrapper，pre-RoPE路径明确未实现。所选ResQ加载每层key_pos基底与头内分组旋转，先用8-bit权重量化器模拟量化 $U_C$。wrapper执行原RoPE，再对Q和K分别做8-bit输入模拟量化、乘该浮点恢复矩阵，最后仅K做头内4/8-bit混合模拟量化并返回浮点tensor。

因此，Q返回的dtype保持模型浮点dtype，并不意味着Q数值未经量化；量化后的 $U_C$ 也不再严格正交，点积恒等式只描述移除这些量化时的参考计算。头维须为二次幂，所选K分组须等于头维；全token跨头分组和其他模型/后端的wrapper消费未全部核查。

在所选 `modeling_llama_2.py::LlamaAttention.forward` 的 `pretraining_tp=1` 模块调用链，Q/K完成RoPE wrapper后，K与V交给 `past_key_value.update`，再做GQA的 `repeat_kv`、浮点attention和o输入重排。V已在V投影的输出wrapper里模拟量化。`pretraining_tp>1` 的权重切片/`F.linear` 分支绕过q/k/v与o wrapper，MLP也有切片旁路，不能照用本页的wrapper消费结论；该分支未完整核查。这里未建立packed缓存writer、字节布局、尺度持久化与读取内核链；不能将“KV=4”质量模拟当成已实现4-bit物理KV cache。该文件也提供FlashAttention2/SDPA分派，本页不把所选eager链当成所有默认执行后端。

## 6. 与论文性能结果的断点

固定归档共有43个源码文件，保存范围中未包含论文所述CUTLASS INT4/INT8部署kernel；`fake_quant` 的浮点Linear也没有转入该实现。因此论文RTX3090单block的1.61–3.03倍结果仍是原文报告，不能从本快照重建并确认。完整部署还需要实际packed权重/KV协议、在线Q/K与Hadamard成本、两个GEMM输出归约和测量边界。后续获取或执行这些材料需另定范围，本次保留断点。

## 来源身份

| 来源 | 固定身份 | 核读范围 |
| --- | --- | --- |
| [utkarsh-dmx/project-resq](https://github.com/utkarsh-dmx/project-resq/tree/cb87d0d059455858daaa66f7410da4afebee978a) | `cb87d0d059455858daaa66f7410da4afebee978a`；ZIP SHA-256 `df2d405a31be96692ca58b9bd55476f2988b8a7e83aa9f482a07713dadb39e6f` | 本页列出的基底、融合、配置、量化器、所选GPTQ/eager缓存调用；不含全仓训练、全部模型与后端 |
| [ResQ](https://arxiv.org/abs/2412.14363v2) | arXiv:2412.14363v2 | §4–5与附录C；分开论文配置、代码示例、浮点模拟和实际部署 |
