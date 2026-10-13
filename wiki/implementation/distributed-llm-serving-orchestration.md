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
  - raw/articles/2026-09-24/ray-serve-llm-architecture
updated: 2026-10-14
---

# 分布式 LLM 服务编排：路由、副本与 KV 交接

单个引擎决定本轮计算哪些 token；多引擎服务还要决定请求送给谁、哪些实例可接纳、历史 KV 怎样复用或交接，以及负载变化时怎样调整容量。一次路由的输出是目标服务实例及必要的协调信息，接收引擎仍需完成自己的 KV 接纳和模型执行。缓存位置、可用容量和实际传输完成是三个不同状态。

引擎内的进度、块池与批次见[vLLM 执行](vllm-inference-execution.md)和[SGLang 执行](sglang-inference-execution.md)；阶段分离为何可能减轻干扰，以及混合轮次的取舍，复用[DistServe 与 Sarathi](serving-memory-and-batching.md#5.%20%E6%89%B9%E5%A4%84%E7%90%86%E4%B8%8E%E8%B0%83%E5%BA%A6)。本页依据已保存的 Dynamo 与 Ray 架构文档解释引擎外的责任与交接，未核其匹配实现或运行服务。Dynamo 条目登记为 v1.4.2，但兼容稿含 v1.5.0/dev 内容；Ray 文档固定为 2.58.0/642b00f8。两者及本 Wiki 引擎源码各自保留版本。

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

## 4. Ray 的模型选择、副本路由与实例放置

Ray Serve 的 deployment 定义服务与伸缩单位，replica 是该 deployment 的 Ray actor 实例，deployment handle 用来调用其副本。`OpenAiIngress` 先按 `model_id` 选择 deployment；handle 的 request router 再选择一个 replica；该副本中的 `LLMServer` 管理一个 engine instance，将请求交给引擎执行并返回响应或 token 流。Router 运行在持有 handle 的 caller 进程中，因此一个集群可以有多个 router 实例。（Ray Overview：Ray Serve primitives、Core components；Request routing：Routing versus ingress、Request routing architecture。）

文档的默认 vLLM 放置为：server actor 申请 1 个 CPU、不申请 GPU；一个 engine 的 TP×PP 个 GPU worker bundles 由 placement group 组织。PACK 尽量在同节点放置，必要时可跨节点，TP ranks 优先靠近。server actor、engine instance 和 GPU worker 分别计数，replica 数不直接等于 GPU 数。（Overview：Physical placement、Engine management。）

`LLMConfig` 将 API 使用的 `model_id` 与权重位置 `model_source` 分开；`engine_kwargs` 传引擎参数，`deployment_config` 管副本/伸缩，placement 配置组织资源。Builder 取得组件默认部署选项，再应用配置、装饰 deployment 并绑定构造输入。文档的异步构造先等待引擎启动，再进入可服务状态；测试用 `sync_init` 返回的对象仍需 `start`。（Core：LLMConfig、Deployment protocols、Builder pattern、Async constructor pattern。）

Ray 文档中的 `LLMEngine` 是 OpenAI request/response 层的接口，tokenization、sampling 等细节封装在接口之后；不能与 vLLM/nano-vllm 的同名类直接等同。保存稿列 vLLM 为深度集成、SGLangServer 为 community integration，TensorRT-LLM 为未来候选，接口可扩展不表示现有后端能力相同。（Core：LLMEngine protocol、Engine implementations。）

副本路由的三个策略使用不同负载信息：

| 策略 | 保存文档中的选择依据 | 适用条件与边界 |
| --- | --- | --- |
| Power of Two Choices（Po2） | 随机抽两个副本，选 ongoing requests 较少者，并考虑 locality | 请求数是负载代理，不直接表示 token 工作量 |
| `PrefixCacheAffinityRouter` | 负载较均衡时，前缀匹配率≥10%选匹配最高者；低于10%选缓存利用率最低者 | 负载不均衡时退回 Po2；10%是该文档的策略阈值，未核实现默认值 |
| `KVAwareRouter` | 从 KV Events 估计未缓存 prefill 工作，与活跃 prefill/decode 工作合成 token load，选估计最低者 | 事件提供潜在缓存位置，接收引擎仍需验证实际身份与可用状态 |

这些依据来自 Request routing 的 Available routing policies。自定义 router 可以逐请求查询集中 metrics actor，代价是额外 RPC 与潜在瓶颈；也可以使用 Serve controller 轮询后广播的统计，减少逐请求查询但存在时滞。文档称集中方案有 strong consistency，所示原子 `get/set` 本身未证明“选择→更新”的复合调度操作可线性化；广播数据也不能直接当作当前真实容量。（Request routing：Design patterns for custom routing policies。）

## 5. Ray 的 DP 组与 P/D 编排

### DP 请求分配、集体通信与组级伸缩

DP attention 将不同请求交给不同 engine，组员的 MoE 专家 dispatch/combine 等 collective 仍须协同。Controller 建立 `data_parallel_size` 个 rank 的 gang；`DPServer` 从 gang context 取得 rank/组身份，rank 0 向 GCS-backed `GangMasterInfoRegistry` 登记 master 地址/端口，其他 rank 查询后把 DP 信息交给引擎。这份注册表保存控制地址，Attention KV 仍由引擎管理。（Ray Data parallel attention：Architecture overview、Components；专家通信的作用见[EP 执行](moe-expert-parallel-execution.md)。）

保存稿将一名组员失败时整组重启作为 gang 的生命周期；autoscaling 的 `min_replicas/max_replicas/initial_replicas` 计 **DP groups**，每组含 `dp_size` 个 engine instances。例如组数2、DP大小4表示8个 engine，GPU 数仍需结合每个 engine 的 TP/PP 与资源配置。普通独立副本数、DP 组数和组内 engine 数不能混用。（Data parallel attention：Autoscaling、Fault tolerance；例子为计数教学。）

该文档推荐 DP 的条件是高并发、稀疏 MoE 专家能被有效利用及 KV 容量受限，低负载下的协调成本未必值得。它对 GQA/MLA 的比较针对 **head 轴 TP 分片**：GQA 可按 KV heads 分片至相应 TP 范围，MLA 的共享 latent 不按同一 head 轴拆分。历史 token 轴的[MLA DCP 分片](vllm-mla-context-parallel-execution.md#1.%20%E6%89%80%E9%80%89%E8%B7%AF%E5%BE%84%E4%B8%8E%E4%B8%A4%E7%A7%8D%E5%B9%B6%E8%A1%8C%E8%BD%B4)是另一机制，不能把该比较推广为 MLA 缓存完全不可分片。（Data parallel attention：When to use/When not to use。）

### Decode 侧发起远程 prefill 与 connector 预热

Ray 的 P/D 请求先到 `PDDecodeServer`。它持有 prefill deployment handle，远程调用 `PDPrefillServer`，取得 KV metadata 后由自己的 engine 接续 decode 并流式返回。协调者在 decode 侧；两阶段各有 deployment/负载队列，可独立调整实例与 engine 配置。保存稿展示 `NixlConnector`，但“storage backend”只描述交接抽象，没有给出磁盘持久化、零拷贝、量化 scale/布局转换或完成释放协议。（Ray Prefill-decode disaggregation：Components、Request flow。）

文档还给出 experimental connector 预热：decode 的 `LLMConfig` 设置 `experimental_configs={"_prewarm_prefill_decode": True}`，用极短 dummy 请求走完整 P/D 路径，为 prefill 副本提前建立连接，并在标记健康前完成 handshake。它处理首次交接的冷连接成本，与前述等待引擎启动的生命周期分开；不能据配置示例声称匹配格式、传输同步或实际部署已验证。（Prefill-decode：Pre-warming the connector。）

收益由两阶段排队、网络/connector backend 与 KV 传输共同决定；短输出、网络带宽不足或小模型可使分离开销大于收益。DP+PD 的文档组合让两个阶段各有自己的 gang 和 DP 大小，具体配置与恢复/同步实现未核。本项完整读了七份保存正文及 ASCII 图，十处原图未保存，仅核正文与图注；外链指南/API、literalinclude 配置、匹配 Ray/engine/NIXL 源码与运行未核。（Prefill-decode：Performance characteristics、Design considerations；Data parallel：Combining with other patterns。）

## 6. Dynamo 版本配套与能力范围

兼容稿明确列出的 **Dynamo v1.4.2 行**为：

| backend | engine pin | NIXL | CUDA toolkit | 最低 driver |
| --- | --- | --- | --- | --- |
| SGLang | 0.5.16 | 1.3.0 | 13.0 | 580.xx+ |
| TensorRT-LLM | 1.3.0rc22 | 1.3.1 | 13.1 | 580.xx+ |
| vLLM | 0.26.0 | 1.3.2 | 13.0 | 580.xx+ |

三者该行的 UCX 均为 1.21.x。它记录文档声明的配套关系，未验证容器、安装、构建或设备运行；引擎 pin 与量化格式、加载和内核条件仍要共同核对，见[部署链路](quantized-llm-deployment-backends.md#1.%20%E9%83%A8%E7%BD%B2%E9%93%BE%E8%B7%AF%E7%9A%84%E7%8E%AF%E8%8A%82)。（Compatibility：Backend engine pins、CUDA toolkit and minimum driver 的 1.4.2 行。）

保存稿同时写 current stable=v1.5.0，后半功能表明确为 v1.5.0，SGLang 组合说明又使用 dev pin 0.5.19；SGLang 的 LoRA 在前段写“不支持”，后段写 experimental，并限制卸载、分离与路由组合的端到端验证。这里保留这些版本和状态差异，不把全部功能表登记成 v1.4.2 的完整能力，也不据“某功能支持”推出任意两功能组合已验证。（Compatibility：Per-Backend Detail、Feature support by backend、Feature interactions。）

## 7. Dynamo 文档性能结果的测量条件

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
| Ray Serve LLM Architecture（`https://docs.ray.io/en/v2.58.0/serve/llm/architecture/index.html`） | Ray 2.58.0；commit `642b00f8c3c5a01cab7c2e2bff8725fa027b23b4`；获取于2026-09-24T02:12:20+08:00 | 完整七份保存正文与ASCII图，模型/副本路由、资源单位、DP gang及P/D协调；十处原图仅核图注，匹配实现和运行未核。 |
