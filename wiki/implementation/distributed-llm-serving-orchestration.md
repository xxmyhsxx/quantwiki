---
title: 分布式 LLM 服务编排：路由、副本与 KV 交接
type: implementation
tags:
  - llm
  - serving
  - kv-cache
  - deployment
  - performance
sources:
  - raw/articles/2026-09-24/nvidia-dynamo-architecture
updated: 2026-10-14
---

# 分布式 LLM 服务编排：路由、副本与 KV 交接

单个引擎决定本轮计算哪些 token；多引擎服务还要决定请求送给谁、哪些实例可接纳、历史 KV 怎样复用或交接，以及负载变化时怎样调整容量。一次路由的输出是目标服务实例及必要的协调信息，接收引擎仍需完成自己的 KV 接纳和模型执行。缓存位置、可用容量和实际传输完成是三个不同状态。

引擎内的进度、块池与批次见[vLLM 执行](vllm-inference-execution.md)和[SGLang 执行](sglang-inference-execution.md)；阶段分离为何可能减轻干扰，以及混合轮次的取舍，复用[DistServe 与 Sarathi](serving-memory-and-batching.md#5.%20%E6%89%B9%E5%A4%84%E7%90%86%E4%B8%8E%E8%B0%83%E5%BA%A6)。本页依据已保存的 Dynamo 架构文档解释引擎外的责任与交接，未核其匹配实现或运行服务。来源条目登记为 v1.4.2，但兼容稿含 v1.5.0/dev 内容；后文按明确版本行与快照声明分别解释。

## 1. 请求执行、容量控制与缓存状态

Dynamo 总体稿把系统分为三条相互配合的路径。Frontend 接纳和规范化请求，Router 根据负载及 KV 重叠选择 worker，prefill/decode worker 执行对应模型阶段并流式返回；这是请求路径。Planner 从运行指标计算两阶段容量目标，Operator 将目标协调为 Kubernetes 资源，Grove 等组件组织多节点放置与分组伸缩；这是容量控制路径。KV Events 传播缓存生命周期，KVBM 管理块复用、淘汰与跨层 offload/recall，NIXL 搬运实际 KV/数据；这是状态与传输路径。（Overall：System Model、Control Loops 及总体原图。）

三条路径的作用不同：Router 可以根据缓存事件减少潜在重算，目标引擎仍须确认缓存身份和可用状态；Planner 给出扩容目标，部署和发现完成后才形成可路由容量；拿到传输元数据后，还需使接收方需要的 KV 可用，才能执行依赖它的 Attention。文档没有展开事件一致性、完成同步、缓冲释放与失败恢复协议，架构箭头不能代替这些实现条件。

文档还区分两种接入。Integrated backend 将 Dynamo worker 和引擎放在同一进程；experimental sidecar 保留引擎原生服务接口，请求路径直接调用原生 gRPC，旁侧 CPU 服务承接发现和事件。保存稿明确 sidecar 的功能覆盖尚未达到 integrated 路径；不能把后者的完整功能清单直接用于旁侧模式。（Overall：Backend Execution Modes。）

## 2. Dynamo 两阶段请求与 KV 交接

在保存的分离路径中，Frontend 将预处理请求交给 `PrefillRouter`，后者按 KV 重叠/负载或简单负载均衡选择 prefill worker。Prefill 侧返回 backend-specific `disaggregated_params`；Router 将它加入 decode 请求，再选择 decode worker。Decode 侧据此协调 KV 传输，随后生成并通过 Frontend 返回 token 流。（Disaggregated Serving：Router Orchestration。）

协调信息的内容决定接收方如何找到提供 KV 的端点与状态，三个 backend 的例子分别为：

| backend | 文档中的传输元数据 | 文档说明的 prefill/decode 衔接 |
| --- | --- | --- |
| SGLang | `bootstrap_info`：host、port、room_id；初始化时登记 bootstrap 端点 | prefill 可在后台执行，decode 侧协调流程可提前进入 |
| vLLM | `kv_transfer_params`：块 ID 与远端 worker 连接信息 | 等待同步 prefill 完成后推进 decode |
| TensorRT-LLM | `opaque_state`：序列化的引擎内部元数据 | 等待同步 prefill 完成后推进 decode |

SGLang 例子的提前进入不表示 Attention 可以读取尚未可用的 KV；它描述阶段协调的重叠，不能将总体顺序图解释成所有 backend 都等完整 prefill 结束才首次路由 decode。保存稿称 NIXL 可在两侧 VRAM 间非阻塞传输，并与**其他请求**的 GPU 前向重叠；实际可用 transport 与完成语义仍取决于部署及 backend。上述字段属于 Dynamo 集成文档，未与本 Wiki 固定 vLLM/SGLang commit 的 connector 实现逐字段对应，也不证明不同 dtype、scale、块大小或布局可直接交换。（Disaggregated Serving：Efficient KV Transfer、Backend-Specific Transfer Metadata。）

## 3. 阶段容量调整与滚动更新

`xPyD` 分别计 x 个 prefill worker、y 个 decode worker。增加 worker 时，它向 discovery 注册并发布含 KV 容量的 `RuntimeConfig`；移除时先排空在途请求再退出发现。Router 在当前可用池内选 worker，Planner 改变池的目标规模。这些阶段服务 worker 的数量还需结合每个实例的 TP/PP 与资源配置才能换算 GPU 数，不能直接当成底层 GPU rank 数。（Disaggregated Serving：Runtime-Reconfigurable xPyD；Overall：Control Plane。）

独立阶段池可隔离长 prefill 的计算占用，也会占用专用 GPU 并引入 KV 传输。低并发时，聚合服务可能避免传输与池间资源碎片；持续负载下，两阶段争用更明显；decode 的 KV 容量已成为瓶颈时，从 decode 挪 GPU 增加 prefill 副本还可能降低总服务容量。最佳比例及 TP 选择取决于模型、量化、设备、backend 和长度分布。文档的“decode 用较大 TP、prefill 用较小 TP”是配置例子，不能从阶段名称推出统一选型。（Disaggregated Serving：Performance Characteristics。）

保存兼容稿的滚动更新声明采用 generation-specific **Dynamo runtime namespace** 隔离发现范围。它不是 Kubernetes namespace；同一 generation 的 prefill/decode 共用范围，不同 generation 的直接 worker 交互被隔离。Frontend 只向完整且 ready 的 WorkerSet 路由：先按 worker 数加权随机选择集合，再在集合内应用路由策略。Worker 数只是容量近似，这一步也不提供请求粘性。（Compatibility：Rolling Update Behavior。）

同稿的 N−2 窗口指当前及前两个 release line 的 frontend/worker 混用，保证范围是 Dynamo 自己拥有的发现元数据和线协议；不覆盖直接 worker-to-worker、特别是 prefill-to-decode 协议。显式启用的特性若无法安全表达，还可拒绝混合版本。因此 frontend 可发现不同版本实例，不等于跨版本 P/D 可以任意配对；该声明按保存快照归属，未核对应历史实现。（Compatibility：Mixed-Version Compatibility。）

## 4. 版本配套与能力范围

兼容稿明确列出的 **Dynamo v1.4.2 行**为：

| backend | engine pin | NIXL | CUDA toolkit | 最低 driver |
| --- | --- | --- | --- | --- |
| SGLang | 0.5.16 | 1.3.0 | 13.0 | 580.xx+ |
| TensorRT-LLM | 1.3.0rc22 | 1.3.1 | 13.1 | 580.xx+ |
| vLLM | 0.26.0 | 1.3.2 | 13.0 | 580.xx+ |

三者该行的 UCX 均为 1.21.x。它记录文档声明的配套关系，未验证容器、安装、构建或设备运行；引擎 pin 与量化格式、加载和内核条件仍要共同核对，见[部署链路](quantized-llm-deployment-backends.md#1.%20%E9%83%A8%E7%BD%B2%E9%93%BE%E8%B7%AF%E7%9A%84%E7%8E%AF%E8%8A%82)。（Compatibility：Backend engine pins、CUDA toolkit and minimum driver 的 1.4.2 行。）

保存稿同时写 current stable=v1.5.0，后半功能表明确为 v1.5.0，SGLang 组合说明又使用 dev pin 0.5.19；SGLang 的 LoRA 在前段写“不支持”，后段写 experimental，并限制卸载、分离与路由组合的端到端验证。这里保留这些版本和状态差异，不把全部功能表登记成 v1.4.2 的完整能力，也不据“某功能支持”推出任意两功能组合已验证。（Compatibility：Per-Backend Detail、Feature support by backend、Feature interactions。）

## 5. 文档性能结果的测量条件

总体稿的三张结果图支持不同的结论，应分别理解：

| 比较 | 原文/原图条件与结果 | 解释范围 |
| --- | --- | --- |
| 分离配置与 baseline | H100、FP8、3K ISL/150 OSL；单/双节点图在所标 Pareto 点写 1.3×/2× tokens/s/GPU | 图题为 Llama3.1 70B，正文图注为 R1 Distilled Llama 70B，模型身份差异未解决；Y 轴无数值刻度，无法重建绝对吞吐，倍数不是全部负载的保证 |
| KV-aware 与随机路由 | 100K 请求，R1 Distilled Llama 70B FP8、2 个 H100 节点、平均 4K ISL/800 OSL；报告 TTFT 约降至随机路由的 1/3，平均请求延迟约降至 1/2 | 说明该负载下的路由收益；保存稿未给评分公式、数据生成和完整运行配置 |
| KVBM host offloading 与 vLLM prefix cache | 聚合服务、Qwen3-8B、H100、20K ISL/100 OSL、QPS 0.25–2；QPS=2 的 mean TTFT 分别为 0.269s 与 3.223s，约 12× | 这是该到达率的首 token 延迟比，不是 decode 吞吐或量化收益；host 容量、前缀复用模式与完整协议未保存 |

以上为文档报告与原图核读，本次未复现。比较部署收益还应保留全部阶段/副本的 GPU 分母、到达过程、命中比例、延迟分布和 SLO，口径见[服务评测](serving-performance-evaluation.md)。Planner、KVBM、NIXL 与故障恢复等外链详细设计及匹配代码未在本项核读，本文不补造评分、传输同步或取消后的资源回收机制。

## 来源身份

| 来源 | 固定版本 | 使用范围 |
| --- | --- | --- |
| NVIDIA Dynamo Architecture and Compatibility Documentation（`https://docs.nvidia.com/dynamo/v1.4.2/knowledge-base/overview`） | 条目登记 v1.4.2；获取于2026-09-24T02:12:20+08:00；兼容稿含 v1.5.0/dev 内容 | 完整保存的 Overall Architecture、Disaggregated Serving、Compatibility 与四张原图；版本混杂和图文差异按正文保留。匹配实现、详细外链和运行未核。 |
