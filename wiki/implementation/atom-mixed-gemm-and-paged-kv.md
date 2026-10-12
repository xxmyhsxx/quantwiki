---
title: Atom 的混合整数 GEMM 与分页 KV 读取
type: implementation
tags:
  - kernels
  - data-format
  - kv-cache
  - serving
sources:
  - raw/repositories/2026-09-21/atom/source/scripts/run_atom_ppl.sh
  - raw/repositories/2026-09-21/atom/source/model/main.py
  - raw/repositories/2026-09-21/atom/source/model/outlier.py
  - raw/repositories/2026-09-21/atom/source/model/modelutils_llama.py
  - raw/repositories/2026-09-21/atom/source/model/gptq.py
  - raw/repositories/2026-09-21/atom/source/model/qLinearLayer.py
  - raw/repositories/2026-09-21/atom/source/model/quant.py
  - raw/repositories/2026-09-21/atom/source/model/qLlamaLayer.py
  - raw/repositories/2026-09-21/atom/source/e2e/README.md.txt
  - raw/repositories/2026-09-21/atom/source/e2e/punica-atom/setup.py
  - raw/repositories/2026-09-21/atom/source/e2e/punica-atom/punica/models/llama.py
  - raw/repositories/2026-09-21/atom/source/e2e/punica-atom/punica/utils/kvcache.py
  - raw/repositories/2026-09-21/atom/source/e2e/punica-atom/punica/ops/__init__.py
  - raw/repositories/2026-09-21/atom/source/e2e/punica-atom/punica/ops/csrc/punica_ops.cc
  - raw/repositories/2026-09-21/atom/source/e2e/punica-atom/punica/ops/csrc/Reorder/Reorder.cu
  - raw/repositories/2026-09-21/atom/source/e2e/punica-atom/punica/ops/csrc/Norm/RMSNorm.cu
  - raw/repositories/2026-09-21/atom/source/e2e/punica-atom/punica/ops/csrc/GEMM/DenseLayerGEMM_i4.cu
  - raw/repositories/2026-09-21/atom/source/e2e/punica-atom/punica/ops/csrc/GEMM/DenseLayerGEMM_i4_o4.cu
  - raw/repositories/2026-09-21/atom/source/e2e/punica-atom/punica/ops/csrc/flashinfer_adapter/flashinfer_impl.cuh
  - raw/repositories/2026-09-21/atom/source/e2e/punica-atom/punica/ops/csrc/flashinfer/page.cuh
  - raw/repositories/2026-09-21/atom/source/e2e/punica-atom/punica/ops/csrc/flashinfer/quantization.cuh
  - raw/repositories/2026-09-21/atom/source/e2e/punica-atom/punica/ops/csrc/flashinfer/decode.cuh
  - raw/repositories/2026-09-21/atom/source/e2e/punica-atom/benchmarks/bench_textgen.py
  - raw/repositories/2026-09-21/atom/source/e2e/punica-atom/benchmarks/bench_textgen_all.py
  - raw/papers/2026-09-21/atom/paper.pdf
updated: 2026-10-12
---

# Atom 的混合整数 GEMM 与分页 KV 读取

固定作者快照 7e3618b1a7a7c86e1c93cc909b1510c046d76ac6 包含质量模拟、原生整数扩展和服务基准。三者有共同设计，但并未在所选路径中形成“同一真实权重、同一网格、同一缓存”的完整闭环。本页静态追踪Llama质量入口与native 7B形状：输入4096、FFN 11008、头维128、group/keeper均128。没有导入上游、编译或运行GPU；其他模型/格式、全部内存同步及边界形状未全面核查。论文机制与测量条件见 [Atom](../methods/atom.md)。

## 1. 校准选择原通道，重排仍保持浮点乘积

run_atom_ppl.sh 指定W4A4、对称group128、weight_channel_group=2、keeper128/INT8、激活裁剪0.9、权重0.85、GPTQ及KV量化。main.py先采浮点激活统计并生成/保存索引，重排模型，再加激活包装和逐block GPTQ，最后评测PPL。

outlier.py 的get_act_stats_llama将token展平，累积经缩放的输入Gram对角；其共同系数是 $2/N_{\rm seq}$，因此排序依据为校准token平方能量，不能仅凭hessian选项名解释成任务损失Hessian。get_reorder_index把大值排到最后，供连续128通道尾部使用。输出逐头排序的辅助索引虽被生成，所选Llama重排并未把全部输出索引应用到Q/K/V。

modelutils_llama.py 配对修改输入列与norm输出索引，gate/up采用相同输出置换，down采用对应输入置换；Q/K/V输出保留原顺序，避免直接破坏RoPE配对。变换的未量化恒等式见方法页；索引不是每token重新选择。

## 2. 质量脚本返回浮点值，没有打包缓存

gptq.py先处理普通列并将其量化误差反馈到剩余列，最后量化已受反馈的keeper尾部；高组不是直接从原权重无条件拷贝。Q最终写回原浮点weight，qLinearLayer.py调用F.linear。quant.py的普通权重按128输入列、2个相邻输出行共享参数；激活逐token/group恢复成浮点。尾部先隔离，普通部分量化后再接回INT8模拟值。

qLlamaLayer.py先对投影后的K模拟量化，再施加RoPE。past_key_value保存旋转后的浮点K及尚未经过该次V量化的浮点V；完整V在缓存拼接之后、AV之前模拟量化。QK、softmax及AV仍是浮点运算。这可以核对所选网格的质量，不能证明物理KV已缩为4 bit。

quant.py的激活/RTN对称INT4采用 $s=\max|x|/7$（另含裁剪与下限），码范围 $[-8,7]$；所选GPTQ普通权重则使用gptq.py的平移码0–15、zero=8、$s=(x_{\max}-x_{\min})c/15$。含正负的对称组对应 $2\max|x|c/15$，仅非负等组的边界另由find_params处理；keeper仍走quant.py的INT8模拟。模拟非对称KV采用**整数**零点 $\operatorname{round}(-\min/s)$。论文§2的对称公式为 $2\max|x|/15$；不能将这些分支或下面native编码视为逐bit相同。

## 3. native GEMM的表示、尺度和累加

setup.py将punica_ops.cc、RMSNorm/Reorder、整数GEMM和FlashInfer适配编译为扩展。LinearInt4保存普通uint8载荷（每字节两码）、INT8 keeper、两套FP16尺度；这里weight是torch.empty创建的参数，所选效率模型没有加载质量脚本生成的量化权重。

Python返回的激活tuple包含尾部INT8、普通packed INT4及各自尺度。C++的RMSNorm/Reorder入口选择group128/hidden4096，Activate入口选择FFN 11008。Reorder.cu按索引搬运，普通/尾部步长分别为max/7和max/127；尺度通过scale_index的复制/对齐布局供MMA读取。所读native Reorder段没有模拟脚本的0.9裁剪，不能把接口名相同当作网格一致。RMSNorm主体用FP32平方和与rsqrt，再衔接重排/量化。

DenseLayerGEMM_i4.cu采用group128；普通路径的MMA为m16n8k64.s32.s4.s4，keeper为m16n8k32.s32.s8.s8。每个归约组得到INT32部分和后，dequant读取两侧FP16尺度，通过__hmul2得到**FP16舍入后的尺度乘积**，转FP32乘整数部分和并加入float accumulator，然后清空INT32部分和；最终输出FP16。论文图8描述FP16恢复/相加，所选代码的跨组累加器实际为FP32，二者需分别记录。（dequant及group循环、keeper分支。）

尺度复制、padding、普通/尾部buffer和输出都占空间。论文4.25 bit元素预算不等于LinearInt4全部容器的实际占用；weight_channel_group=2的模拟布局也不能无需转换就交给native尺度读取。分组恢复的数学位置见[归约维分组](quantized-matmul-scaling-execution.md#15.%20%E5%BD%92%E7%BA%A6%E7%BB%B4%E5%88%86%E7%BB%84%E4%B8%BA%E4%BB%80%E4%B9%88%E8%A6%81%E5%9C%A8%E7%B4%AF%E5%8A%A0%E9%80%94%E4%B8%AD%E6%81%A2%E5%A4%8D%E5%B0%BA%E5%BA%A6)。

## 4. 输出INT4与KV的zero-point不是整数零点

K/V投影调用dense_layer_gemm_i4_o4，输出为uint8及每128输出元素一对FP16参数。需要区分非对称网格的意图与这个快照的实际统计：local_max_min对abs(y)取极值，得到 $M=\max|y|$、$m=\min|y|$；输出段据此令 $s=(M-m)/15$、$z=-m$，却对**原有符号**的y编码 $\operatorname{round}((y+z)/s)$，转int8后用&0xf装入nibble。&0xf是低位截取，不是超范围值的饱和clamp；这并非通常先统计原值min/max的非对称量化。（helper及输出段732/749–771行。）

写入half2的是 $(s,z)$；$z$仍是**实数偏移**，读取公式为 $\widehat y=qs-z$，并非 $s(q-z_{\rm int})$。读取语义不证明写入网格与浮点值范围匹配。

模拟KV先舍入整数零点，native则先用浮点s/z编码、再将参数写为FP16，两条路径有统计、网格及参数舍入差异。所读native输出量化与Reorder段没有显式零范围下限处理；模拟含1e-5下限。abs范围与原值编码不匹配，以及常量组/负值边界均未运行；保留静态差异，不能据此断言作者运行失败或声称这些接口完全等价。

量化标签__precision__s4只决定packed访问；quantization.cuh的KV解码实际取无符号nibble 0–15，再乘尺度、减实数偏移。它与GEMM的signed s4码含义不同，不能复用同一种zero-point或符号解释。

## 5. 分页writer到decode reader

KvPoolInt4数据形状为 $(C,L,2,H,P,D/2)$、uint8；参数为 $(C,L,2,H,P,2)$、FP16。C为物理页容量，2分别是K/V，P为页内token，D=128。请求通过indptr、物理页indices和last_page_offset定位有效历史，不能将逻辑序号直接当物理地址。

init/append在page.cuh将投影产生的packed K/V及half2参数复制进对应页，不在writer中施加RoPE。对每个token/头，仅载荷及这两个参数的元素预算为 $4+32/128=4.25$ bit；另有页表、空余页/尾页及其他buffer，不能据此声称总缓存恰为4.25 bit。

flashinfer_impl.cuh选择head128、Llama RoPE、theta=10000、scale=1、FP16 Q和输出。decode.cuh先读packed K/V及FP16参数，在float向量中解码；K按自身位置做float RoPE后参与QK，Q按当前seq_len−1旋转，V解码后合入浮点attention状态。这里没有先把完整KV恢复到全局FP16缓存。结果输出FP16；参数和输出的舍入仍在。该接口不自动支持别的RoPE配置或任意头维/GQA布局。

## 6. 效率基准有哪些简化

e2e/README明确效率模型不生成有意义的文本。llama.py的reorder_index使用randperm，预填充attention使用randn_like产生的K/V；所有decoder层的layer_idx为0。bench_textgen.py将KvPoolInt4设为1层，以共享缓存来减内存；decode确实经过INT4 writer/reader，但缓存不代表真实多层模型的独立状态。

bench_textgen.py以固定随机种子生成lognormal输入长度、随机输出长度与随机token。bench_textgen_all.py默认batch1–16，throughput为**输入+输出token总数/总时间**；论文服务图描述ShareGPT长度分布、batch8–256及generated tokens/s。公开默认脚本因此不能直接当作论文全部图10协议的逐项复现，也不能与真实模型质量结果合成“已测同一部署”。

本页确认所选原生指令、packed表示、尺度与分页调用链存在，并解释它们和模拟的差异；没有验证编译、同步正确性、全部形状、真实权重导出、端到端质量或性能。子模块只是原始材料的保存范围，未将未读vendor实现计为本页核查结果。

## 来源身份

| 来源 | 固定身份 | 本页范围 |
| --- | --- | --- |
| efeslab/Atom（`https://github.com/efeslab/Atom/tree/7e3618b1a7a7c86e1c93cc909b1510c046d76ac6`） | 7e3618b1a7a7c86e1c93cc909b1510c046d76ac6；保存的shallow source tree | 所选Llama校准/GPTQ/模拟；e2e原生GEMM、K/V输出量化、分页writer/reader与默认效率脚本 |
| Atom（`https://arxiv.org/abs/2310.19102v3`） | arXiv:2310.19102v3 | §2/4及图8的格式与恢复描述、服务测量协议；逐bit与代码一致性分别核对 |
