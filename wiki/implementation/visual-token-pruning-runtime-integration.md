---
title: 视觉token剪枝的运行接入：共同子序列、位置与缓存
type: implementation
tags:
  - vlm
  - token-compression
  - inference
  - deployment
sources:
  - raw/articles/2026-10-05/tensorrt-edge-llm-dart
  - raw/repositories/2026-10-05/tensorrt-edge-llm
updated: 2026-10-15
---

# 视觉token剪枝的运行接入：共同子序列、位置与缓存

TensorRT Edge-LLM的DART在语言engine执行前，从已组装的输入embedding选择视觉行。输出是按原顺序保留的子序列及配套长度/位置状态，随后由base engine和可选draft消费。选择集合只是第一步；所有消费者还必须看到同一份剪后prompt。

本页依据0.11.0官方文档和同版固定commit `e649dadcbbf049fa4caf2f09f3a3275ca56751db`，静态核读所列选择、压缩与调用主线，未构建/运行模型。位置机制复用[LayerPos](../methods/layerpos.md)，这里解释具体buffer怎样一起改变。该实现用embedding评分；本地说明明确它与原DART的decoder-layer-2投影key变体不同，不将两者当同一实现。

## 1. 每个视觉span有自己的额度

`llmRankRuntime.cpp`的prefill路径先把文字与视觉feature组装成 `[batch,N,H]` embedding并准备DeepStack，之后才调用 `pruneBatchForPrefill`。选择器按host token id识别视觉行；其余非视觉行全部保留，不从原始pixel选patch，也不读取语言attention权重。

连续视觉位置组成span。对长度 $n_s$、删除比例 $r$，保留额度为
$k_s=\max(1,\min(n_s,\lceil n_s(1-r)\rceil))$。
默认 $r=0.25$表示删除比例，实际数量还受逐span取整影响；默认视觉行不足16时跳过。独立额度避免一幅图/帧块挤占其他span的全部预算。（`visualTokenPruner.cpp:selectForSlot`。）

每span按embedding L1范数选最多4个image pivots，整请求另选最多4个text pivots供各span共用。Image pivots先占视觉额度；之后按固定pivot顺序，从尚未选择的span内候选中补入cosine最小者，每次分配剩余额度的一部分。新选token不会成为新pivot；这是一套固定参照下缩小候选池的贪心，不是动态更新代表集合的最远点算法。（`dartPruner.cpp:selectRetainedImageTokens/growSpanGreedily`。）

CUDA先算行L1/L2与pivot点积，FP16输入、FP32累加；norms/dots拷回主机并同步后，CPU完成精确选择。最多64个pivot rows一批，主导点积工作随pivot数、序列长和hidden宽度增加。无需attention scores仍有设备计算、传输、同步和host选择成本，不能把“training-free”读成没有运行开销。

## 2. 同一keep-list作用于所有base输入

基础compactor将保留视觉位置与全部非视觉位置合并、按原顺序排序。每slot先独立选择，再按剪后最大长度重排batch；这里支持子集选择，不包含token合并产生的新特征。

| 对象 | 如何变化 | 保持的对应关系 |
| --- | --- | --- |
| 输入embedding | 按keep-list gather到scratch再回写 | 每个幸存位置保留原向量，避免原地覆盖未读数据 |
| 每个已组装DeepStack平面 | 使用同一keep-list和batch row pitch | 语言层附加视觉输入与主embedding对齐 |
| mRoPE cos/sin | 幸存行取原坐标；续行从原prompt末尾搬到紧凑prompt之后 | 物理长度缩短，位置距离仍沿原序列 |
| host ids与ragged元数据 | ids按keep-list重排，更新query/attention长度、有效与物理token数 | engine、logits选择与缓存长度消费同一子序列 |

对旧长度 $N$、新长度 $K$，mRoPE后续第 $q$行从原 $N+q$搬到新 $K+q$。所以首个生成token仍使用未剪prompt之后的位置；ragged里的紧凑KV槽位不等于原RoPE坐标。若只缩短embedding而让续行按 $K$重编号，就改变了新Query与缓存Key的相对旋转。（`visualTokenPruner.cpp:executeBatchCompaction`；`exec/raggedBatchBuilder.cpp:finalizeSubsetSelection`。）

batch row pitch由旧最大长度变为新最大长度，连没有删除token的slot也可能需要搬移。实现按slot升序使用scratch搬运，并为每slot保留独立host/device索引区，避免后续写入覆盖尚待异步上传的keep-list。这些是维护共同布局的具体步骤，不是仅改tensor shape。

## 3. draft重嵌入还需要原始feature与ids同步

推测模式的draft会从host ids与raw视觉feature重新嵌入，base则已经消费压缩后的embedding。`compactAuxiliaryInputs`因此还按同一keep-list压缩ids，并按原视觉ordinal gather raw feature，再交给draft。当前draft不消费raw DeepStack，故不对它再做相同压缩；base已组装的DeepStack平面此前已处理。（`llmRankRuntime.cpp`的prefill调用、`visualTokenPruner.cpp`对应函数。）

profile把pruned、computed和reused分别记录。删掉的位置没有被复用，不能把剪枝数算成KV cache命中；新的decode步也不再重新做一次视觉选择，但仍消费剪后长度和上述位置状态。

## 4. 支持条件和性能口径随实现保留

本分支面向单设备mRoPE VLM、fresh KV；配置与执行还排除vision bidirectional attention、action runner、所列Omni/调试和diffusion路径。文档概括为不支持时静默no-op，源码实际存在不同结果：multi-device明确报错，managed context reuse拒绝请求，其他配置可能关闭剪枝；不能据一段文档给所有组合承诺。（`setVisualPrunerConfig`、`runtimeCoordinator.cpp`及prefill guards。）

官方报告限于Qwen3-VL-4B、FP16、单图、1995-token prefill：$r=.25$在Thor/x86 Blackwell为1.32×/1.18×，selector成本已计入；质量是指定数据集上的近无损观察。它不是所有输入的固定收益；支持矩阵的Orin FP16/INT8/INT4模型约束也不能扩成已测DART性能，embedding dtype与模型权重格式分别核对。

`run_dart_accuracy.py:evaluate_dataset`只在**所有配置都成功回答的共同问题集**上比较accuracy，并另报错误数；这不同于错误计零的全任务成功率。其estimated speedup是原/剪后ISL之比，没有计时；出现错误时还会隐藏这个估计，以免profile集合与评分集合不一致。实际速度仍需按[共同输入与计时边界](kernel-correctness-and-benchmarking.md)评价。

## 来源身份

| 来源 | 固定版本 | 核读与使用范围 |
| --- | --- | --- |
| TensorRT Edge-LLM DART与支持矩阵（`https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/features/visual-token-pruning.html`） | `0.11.0; docs-build-e649dadcbbf049fa4caf2f09f3a3275ca56751db; snapshot-2026-10-05`，取得于2026-10-05T17:40:36+08:00 | 两页完整正文、代码/表格；仅采用本实现与平台条件及所报prefill配置，不将文档概括覆盖全部源码分支。 |
| NVIDIA/TensorRT-Edge-LLM（`https://github.com/NVIDIA/TensorRT-Edge-LLM/tree/e649dadcbbf049fa4caf2f09f3a3275ca56751db`） | `e649dadcbbf049fa4caf2f09f3a3275ca56751db` | DART selector/CUDA norms-dots/gather、shared compactor、ragged finalizer及prefill/spec接入、accuracy评分。未核全部runtime/backend、子模块、draft算法或设备运行。 |
