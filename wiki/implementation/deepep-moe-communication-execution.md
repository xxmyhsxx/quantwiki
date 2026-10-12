---
title: DeepEP 的 MoE 通信执行：去重、FP8 布局与输出回传
type: implementation
tags:
  - moe
  - expert-parallelism
  - fp8
  - kernels
sources:
  - raw/repositories/2026-09-21/sglang/source/docker/Dockerfile
  - raw/repositories/2026-09-21/sglang/source/scripts/ci/cuda/ci_install_deepep.sh
  - raw/repositories/2026-09-21/sglang/source/python/sglang/srt/layers/moe/token_dispatcher/deepep.py
  - raw/repositories/2026-09-21/vllm/source/docker/Dockerfile
  - raw/repositories/2026-09-21/vllm/source/tools/ep_kernels/install_python_libraries.sh
  - raw/repositories/2026-09-21/vllm/source/vllm/distributed/device_communicators/all2all.py
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py
  - raw/repositories/2026-09-21/vllm/source/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ll.py
  - raw/repositories/2026-10-07/deepep-sglang/source/deep_ep/buffer.py
  - raw/repositories/2026-10-07/deepep-sglang/source/deep_ep/utils.py
  - raw/repositories/2026-10-07/deepep-sglang/source/csrc/deep_ep.cpp
  - raw/repositories/2026-10-07/deepep-sglang/source/csrc/deep_ep.hpp
  - raw/repositories/2026-10-07/deepep-sglang/source/csrc/config.hpp
  - raw/repositories/2026-10-07/deepep-sglang/source/csrc/event.hpp
  - raw/repositories/2026-10-07/deepep-sglang/source/csrc/kernels/layout.cu
  - raw/repositories/2026-10-07/deepep-sglang/source/csrc/kernels/intranode.cu
  - raw/repositories/2026-10-07/deepep-sglang/source/csrc/kernels/internode.cu
  - raw/repositories/2026-10-07/deepep-sglang/source/csrc/kernels/internode_ll.cu
  - raw/repositories/2026-10-07/deepep-sglang/source/csrc/kernels/runtime.cu
  - raw/repositories/2026-10-07/deepep-sglang/source/csrc/kernels/utils.cuh
  - raw/repositories/2026-10-07/deepep-sglang/source/csrc/kernels/launch.cuh
  - raw/repositories/2026-10-07/deepep-sglang/source/csrc/kernels/ibgda_device.cuh
  - raw/repositories/2026-10-07/deepep-vllm/source/deep_ep/buffer.py
  - raw/repositories/2026-10-07/deepep-vllm/source/deep_ep/utils.py
  - raw/repositories/2026-10-07/deepep-vllm/source/csrc/deep_ep.cpp
  - raw/repositories/2026-10-07/deepep-vllm/source/csrc/deep_ep.hpp
  - raw/repositories/2026-10-07/deepep-vllm/source/csrc/config.hpp
  - raw/repositories/2026-10-07/deepep-vllm/source/csrc/event.hpp
  - raw/repositories/2026-10-07/deepep-vllm/source/csrc/kernels/layout.cu
  - raw/repositories/2026-10-07/deepep-vllm/source/csrc/kernels/intranode.cu
  - raw/repositories/2026-10-07/deepep-vllm/source/csrc/kernels/internode.cu
  - raw/repositories/2026-10-07/deepep-vllm/source/csrc/kernels/internode_ll.cu
  - raw/repositories/2026-10-07/deepep-vllm/source/csrc/kernels/runtime.cu
  - raw/repositories/2026-10-07/deepep-vllm/source/csrc/kernels/utils.cuh
  - raw/repositories/2026-10-07/deepep-vllm/source/csrc/kernels/launch.cuh
  - raw/repositories/2026-10-07/deepep-vllm/source/csrc/kernels/ibgda_device.cuh
  - raw/repositories/2026-10-07/deepep-vllm/source/csrc/kernels/configs.cuh
  - raw/repositories/2026-10-07/deepep-vllm/source/setup.py
updated: 2026-10-07
---

# DeepEP 的 MoE 通信执行：去重、FP8 布局与输出回传

DeepEP 接收模型路由给出的专家 ID，把隐藏状态送到对应 rank，再把专家输出还给原 token。**normal/HT 按目的 rank 去重隐藏状态，接收侧完成专家配对和加权；LL 按选中的专家发送，返程 combine 才施加 gate。** 复制单元不同，接收行、scale、handle 和返回归约的含义也不同。

本页接续 [MoE EP 框架链](moe-expert-parallel-execution.md)，解释固定外部库的 Python → C++ → CUDA 通信路径。模型路由、框架 group、专家 GEMM、routed scaling $\rho$ 和 shared MLP 仍由该页承接。前置是 [MoE 路由](../fundamentals/model/mixture-of-experts-and-routing.md)、[张量布局](../fundamentals/operators/tensor-layout-and-kernel-contracts.md)与[低精度累加](../fundamentals/numeric-formats/floating-point-and-accumulation.md)。

## 1. 版本对应与所选条件

框架固定为 SGLang `2f730e299f3b574e3bee2c6ef9669fa2a5b26dbc`、vLLM `568afb3a13806beb53bb2e6bd518269357b237c0`，外部来源分别保存：

| 路线 | DeepEP 固定源码 | 对应依据与边界 |
| --- | --- | --- |
| SGLang 默认 SM90 CUDA normal/LL | `9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee`，下称 S | 默认 Docker 的 `DEEPEP_COMMIT` 与普通 CUDA CI checkout 相同；Docker 延长两个 timeout，默认 CI 不作这两个修改 |
| vLLM 默认 CUDA Docker HT/LL | `73b6ea4a439ba03a695563f9fd242c8e4b02b37c`，下称 V | Docker `DEEPEP_COMMIT_HASH=73b6ea4` 经官方 commit API 展开，并传 `--deepep-ref` 覆盖安装脚本 |

vLLM 独立安装脚本的默认回退是 `d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb`，不是本页所读 Docker pin。环境变量/参数可覆盖，已有 workspace 可跳过 checkout；SGLang CI 也可跳过已安装包。两个框架的 CUDA 13 配方另补 CCCL include。本地 raw 保存未打配方补丁的上游原件；配方对应关系不能证明任意实际部署使用同一版本。Grace Blackwell/SBO、ROCm、hybrid-ep、DeepEP V2/ElasticBuffer 不从本页默认源码推出。

沿用均分连续 expert placement：$E$ 个 routed 专家、$P$ 个 EP rank、$L=E/P$ 个本地专家。每源 rank $q$ 有 $M_q$ 行，hidden 宽度为 $D$，每 token 选 $K$ 个不同专家；ID 为 int64、gate 为 FP32。模型输入与专家输出为 BF16，dispatch 为普通 E4M3 和正 FP32 per-128 scale。关闭 UE8M0、LogFMT、zero-copy、shrink/fault tolerance、EPLB、graph 与 SBO/TBO/DBO。

设备主线取 NVLink intranode normal/HT 的 $P\in\{2,4,8\}$，另核 hierarchical internode 的分组、载荷与两段回传；LL 沿 `low_latency_dispatch/combine`。具体支持集合与拓扑仍需满足后文约束。静态 API 对齐不表示已联合安装或运行。

## 2. Rank 行数与专家配对数不同

源 rank $q$ 的 token $t$ 对目的 rank $r$ 的布尔标记为

$$b_{q,t,r}=\mathbf{1}\{\exists j:\lfloor e_{q,t,j}/L\rfloor=r\}.$$

CUDA `get_dispatch_layout` 对专家计数统计每个有效路由槽，但先将同 token 的 rank presence 合成布尔值，故

$$n_{q,r}=\sum_t b_{q,t,r},\qquad R_r=\sum_q n_{q,r},\qquad
m_e=\sum_{q,t,j}\mathbf{1}\{e_{q,t,j}=e\}.$$

$R_r$ 是 normal/HT 收到的唯一 token 行数，$m_e$ 是专家配对数。例如一个 token 选 rank 1 的两个专家、rank 2 的一个专家，normal 向两个目的 rank 各送一行，rank 1 的 expert scatter 再生成两项配对；LL 发送三项专家副本。两者都保留这三项模型贡献。（S/V `layout.cu::get_dispatch_layout`、`intranode.cu::dispatch`、`internode_ll.cu::dispatch`。）

SGLang normal 传 `expert_alignment=128`，使本地专家计数为 $a_e=128\lceil m_e/128\rceil$；vLLM HT 此接口传 1。**alignment 只扩专家计算所需的配对容量，不将 `recv_x` 的 $R_r$ 行扩成 $\sum_e a_e$。** 接收 token 与 GEMM 配对/padding 是不同布局。（S `deep_ep.cpp:403–491`、`intranode.cu::notify_dispatch`；框架页 normal/HT。）

## 3. Intranode normal/HT：前缀、队列与本地 ID

`Buffer.dispatch` 在 RDMA 组数为 1 时走 `intranode_dispatch`。C++ 在通信流启动 notify，交换 $n_{q,r}$ 与专家计数，再按通信 channel 分源 token 区间。每 channel 使用发送/接收两个 CUDA blocks，channel 数 $C_h$ 为 `config.num_sms/2`。

`rank_prefix_matrix:[P,P]` 在目的 rank $r$ 的存储中保证当前目的列的 inclusive prefix：$F_{q,r}=\sum_{u\le q}n_{u,r}$。不要从张量 shape 推断每个 rank 都持有完整复制的全局矩阵。接收行按源 rank/channel 区间写入 `recv_x:[R_r,D]`，`recv_src_idx:[R_r]` 保存源 rank 内原 token 行号，源 rank 由 prefix 区间确定。（S `intranode.cu:26–65`；V `35–78`。）

发送端只在 $b_{q,t,r}=1$ 时推进目的队列位置，并记录 `send_head[t,r]`。各 channel 使用有界循环队列，slot 取逻辑 head/tail 对容量的模；sender 等可用空间，receiver 等已发布数据，处理后更新 head。队列复用容量不是整批行数上限，也不是自动取消模型路由的 dropping。（S/V `intranode.cu::dispatch`。）

接收 top-k 保留 $K$ 槽，只留下当前 rank 的局部专家 ID：

$$e'_{t,j}=\begin{cases}e_{t,j}-rL,&rL\le e_{t,j}<(r+1)L,\\-1,&\text{其余},\end{cases}
\qquad g'_{t,j}=\begin{cases}g_{t,j},&e'_{t,j}\ge0,\\0,&\text{其余}.\end{cases}$$

同一接收行可有多个非负 ID。SGLang `ep_scatter` 按有效槽复制到专家连续区间，vLLM 在自己的 ID 转换后进入 Triton 配对阶段；rank 去重没有删掉同 rank 的第二个专家。（S `intranode.cu:285–325`；V `344–386`；框架页接收排列。）

## 4. Normal FP8 平面、六项 handle 与返程

normal/HT 接收框架预量化的 `(x_fp8,x_scale)`，本页 $G=D/128$。C++ 读取输入 scale 的 token/group strides，CUDA 原样复制 FP32 数值，不在 dispatch 中重新量化。返回 `recv_scale:[R_r,G]` 行连续，strides 为 $(G,1)$，没有再给每个 scale 行补 128 元素。

NVLink arena 分别保存 hidden、int32 源行号、top-k ID、gate 与 scale 队列。所选 int64 ID 下，每 token/目的 rank 的数据平面槽位合计

$$B_{\mathrm{normal}}=D+4+8K+4K+4G.$$

这是 arena 元素计费，尚未计 prefix、head/tail、barrier 或真实链路流量；不是一个按该长度打包的报文。S 的 intranode ID 为 int64，V 默认 `TOPK_IDX_BITS=64` 时相同；internode 的 int32 wire ID 另见下一节。（S `deep_ep.cpp:375–386,489–514`；V `intranode_dispatch`、`setup.py` 与 `configs.cuh`。）

| normal handle 元素 | 含义 |
| --- | --- |
| `rank_prefix_matrix[P,P]` | 当前目的列按源 rank 的 inclusive 接收前缀 |
| `channel_prefix_matrix[P,C_h]` | 源端按目的 rank/channel 的 inclusive 发送前缀 |
| `recv_channel_prefix_matrix[P,C_h]` | 接收端各源 rank/channel 的起始偏移 |
| `recv_src_idx[R_r]` | 接收行对应源 rank 内的 token 行号 |
| `is_token_in_rank[M_q,P]` | 本次 token 是否需送到各目的 rank |
| `send_head[M_q,P]` | 源 token 对目的队列的逻辑位置，未发送项为负 |

`combine` 使用同次 handle 的第三项接收 channel prefix；不能与第二项发送 prefix 交换。目标 rank 按 rank/channel prefix 组织返回区间，并附带原 `recv_src_idx`；原 owner 实际按 send head/channel 顺序匹配返程，不能简化成只靠一个 source ID 数组。（S `buffer.py:378–384,424–432`；S/V `intranode.cu::combine`。）

框架已经完成 gate 加权与目的 rank 内归约：SGLang `ep_gather`、vLLM HT expert finalize 交付 BF16 `[R_r,D]`。所选 `combine` 不传 top-k weights，设备把各目的 rank 的 BF16 行提升为 float 相加，再转回 BF16 `[M_q,D]`，没有对 hidden 再乘 gate。API 可单独归约/返回 optional weights，这是元数据通道，不能理解为对 `x` 加权。所选返程 arena 每行 hidden $2D$ bytes 加 int32 source index，不带 FP8 scales。（S/V `buffer.py::combine`、`intranode.cu::combine`。）

## 5. Hierarchical internode：组去重、fanout 与两段舍入

normal/HT 的 $P>8$ 路线按连续八 GPU NVLink 组解释 rank：`rdma_rank=rank/8`、`nvl_rank=rank%8`。这要求实际拓扑与 rank 放置相符。$P/8$ 的 launcher 集合见第 9 节，不是所有八的倍数都支持。

每 token 的目的组八个 GPU presence 合成 bitmap；每个非空目的组只占一个 RDMA tail slot，接收组 forwarder 再按 bitmap 送到需要的组内 GPU。所以目的 GPU 行数与目的组 RDMA 行数不同，不能直接用前述每 rank 公式计算全链字节。（S `internode.cu:263–268,499–540,781–799`；V `295–302,620–667,950–967`。）

`SourceMeta` 为两个 int32：源 RDMA rank 与八个目的 GPU 的 presence bits，后者物理字段仍是 4 bytes。消息按 hidden、FP32 scales、SourceMeta、int32 top-k IDs、FP32 gates 排列，slot stride 为

$$B_{\mathrm{inter}}=16\left\lceil\frac{Db_x+4G+8+4K+4K}{16}\right\rceil,$$

本页 FP8 dispatch 的 $b_x=1$。Python 输入/返回 ID 为 int64，CUDA 打包时转为 int32，接收时恢复为 int64、本地化 ID 并将非本地 gate 置 0；不同于 intranode 的 int64 平面。（S `internode.cu:17–46,545–583,969–980`；V `17–45,673–711,1158–1169`。）

internode handle 顺序为 `(is_token_in_rank,rdma_channel_prefix_matrix,gbl_channel_prefix_matrix,recv_rdma_channel_prefix_matrix,recv_rdma_rank_prefix_sum,recv_gbl_channel_prefix_matrix,recv_gbl_rank_prefix_sum,recv_src_meta,send_rdma_head,send_nvl_head)`。SourceMeta 没有直接可读的原 token 行号，回源还依赖两段 prefix/head。（S `buffer.py:479–510`；V `494–527`。）

返程先在目的组内对 NVLink rank 贡献做 FP32 sum，转成 BF16 作为 RDMA 载荷；原 rank 再对各目的 RDMA 组贡献做 FP32 sum 并输出 BF16。两段 `combine_token` 不对 hidden 乘 gate，可选 weights 另行求和。因此多了一次中间 BF16 舍入，不能承诺与单阶段 FP32 归约逐位一致。（S local `1669–1696` / final `1751–1763`；V local `2046–2096` / final `2163–2188`；helper S `1274–1288,1319–1343`，V `1589–1606,1637–1661`。）

## 6. LL dispatch：发送量化与专家容量

LL 输入是 BF16 `x:[M_q,D]` 与源端 int64 `topk_idx:[M_q,K]`，不传 gate。$M_q\le M_{\max}$，各 rank 的 $M_{\max}$ 相同。发送 kernel 将 BF16 提升为 float，每 token 每 128 通道求

$$a_{t,g}=\max\!\left(10^{-4},\max_{d\in g}|x_{t,d}|\right),\quad
s_{t,g}=a_{t,g}/448,\quad
\widetilde x_{t,d}=\operatorname{cast}_{\mathrm{E4M3,satfinite}}\!\left(x_{t,d}\,448/a_{t,g}\right).$$

`round_scale=False,use_ue8m0=False` 保存正 FP32 restoring scale，既不是 inverse multiplier，也不是 UE8M0 exponent。同 token 在源端量化一次，多个专家发送引用该消息，不因目的专家不同重算 scale。（S `utils.cuh:450–475`、`internode_ll.cu:110–174`；V `utils.cuh:467–492`、`internode_ll.cu:213–278`。）

每个有效 top-k 槽用 $e/L$、$e\bmod L$ 确定目的 rank/本地专家，并对 global expert counter 原子分配发送 slot。FP8 消息为 16-byte 控制头（int32 源 token 行号、三个 reserved int）+ $D$ 个 E4M3 bytes + $4G$ 个 FP32 scale bytes：

$$B_{\mathrm{LL,dispatch}}=16+D+4G.$$

reserved 字段没有有效路由语义或清零承诺；ID 由目的专家分区决定，gate 留在原 owner。`LowLatencyLayout` 为同时支持 BF16/FP8 按 `16+max(2D,D+4G)` 计费分配容量；所选 FP8 kernel 的实际发送/接收消息步长就是本次 $16+D+4G$，不能把分配容量的较大计费长度当成消息之间的物理步长。（S/V `config.hpp::LowLatencyLayout`、`internode_ll.cu::dispatch`。）

接收容量 $C=P M_{\max}$。每源 rank 向一个专家最多贡献 $M_{\max}$ 行，这依赖 top-k 专家不重复。各源 rank 的 segment 通过原子计数拼接，顺序不保证按源 rank 或原 token 排序：

| 返回对象 | dtype / shape / 元素 strides |
| --- | --- |
| `packed_recv_x` | E4M3 `[L,C,D]`，连续 $(CD,D,1)$ |
| `packed_recv_x_scales` | FP32 `[L,C,G]`，先分配 `[L,G,C]` 再 transpose，$(GC,1,C)$ |
| `packed_recv_src_info` | int32 `[L,C]`，有效行保存源 token 行号 |
| `packed_recv_layout_range` | int64 `[L,P]`，每源 rank 的 `(begin<<32)|count` |
| `packed_recv_count` | int32 `[L]`，各 segment 实际计数之和 |

有效行是 `[0,recv_count[expert])` 前缀。src_info 本身不含源 rank，rank 由 range 所属列确定。返回容量用 `empty` 分配，尾部不保证清零，专家计算/回传由 count/range 限制。scale 的 row stride 为 1、group stride 为 $C$；不能凭三维 shape 推断连续或用 normal 的 scale 代替。（S `deep_ep.cpp:1140–1163`、`internode_ll.cu:255–330`；V `deep_ep.cpp:1464–1489`、`internode_ll.cu:362–461`。）

## 7. LL combine：回到原 token 后施加 gate

LL handle 为 `(src_info,layout_range,Mmax,D,E)`，这是两份固定实现的 schema，需与同次原 ID/weight 配合。专家输出保持未加权 BF16 `[L,C,D]`。目标 rank 按 range 遍历有效行、用 src_info 取源行号，将结果送回 owner 的 global expert/token slot。

`zero_copy=False,use_logfmt=False` 的实际返回载荷为 $2D$ bytes BF16，没有重新量化为 FP8；底层 combine slot 额外预留 $4G$ bytes 给未选 LogFMT，预留不表示普通 BF16 也发送这些 metadata。原 owner 按原 top-k 槽序 $j=0,\ldots,K-1$，跳过负 ID：

$$y_{q,t,d}=\operatorname{BF16}\!\left(\sum_{j:e_{q,t,j}\ge0}
g_{q,t,j}\operatorname{FP32}(z_{e_{q,t,j},q,t,d})\right).$$

$z$ 已是专家 BF16 输出；FP32 gate 乘法与 float accumulator 在回源 combine kernel 中完成，最后转 BF16。这里是源码 dtype/次序，没有编译确认乘加是否融合，也不保证与 normal 的“各 rank 先加权/归约并舍入，再跨 rank 相加”逐位一致。$\rho$ 与 shared 仍在框架返回后处理。（S `internode_ll.cu:544–550,616–630,663–781,830–909`；V `705–711,790–803,835–976,1040–1133`。）

vLLM 传 `out=output` 时 C++ 校验并复用连续 BF16 `[M_q,D]`，SGLang 默认由库分配。`Delegate` 表示 gate/归约交给 LL combine，不能在进入前再乘一次 gate；normal 已加权，不能直接套此返程。（S/V `deep_ep.cpp::low_latency_combine`、`buffer.py::low_latency_combine`；框架页 LL。）

## 8. 传输、event、hook 与 buffer 生命周期

normal intranode 用 CUDA IPC 共享 NVLink arena；C++ 汇集 IPC handles、建立 peer pointers。hierarchical normal 为同 NVLink index 的跨组 peers 建 NVSHMEM 路径；LL 则将所有 EP rank 纳入 NVSHMEM。禁用 NVSHMEM 的构建只允许所支持的 intranode normal，import 成功不能证明 LL 可运行。（S/V `Buffer.__init__/sync`、`runtime.cu::init`。）

两个版本的 `allow_nvlink_for_low_latency_mode` 默认均为 True，设置 `NVSHMEM_DISABLE_P2P=0` 并启用 IBGDA，vLLM LL manager 也显式传 True。LL 检查 `nvshmemi_get_p2p_ptr`：有 peer pointer 时直接 copy 并以 system release 发布 count/flag，否则使用 IBGDA put/atomic 通知。因此“LL 总是绕过 NVLink”或“调度报告 SM 占用为 0 就没有 CUDA 工作”均不由本链支持。量化、请求提交、peer copy、接收 packing 和归约仍由 CUDA kernel 执行；实际 peer 可达性与网络拓扑未运行确认。（S/V `ibgda_device.cuh`、`internode_ll.cu`、`buffer.py`。）

| 接口 | 实际同步含义 |
| --- | --- |
| normal `previous_event` | 通信流等待给定 CUDA event；未传时等待当前 compute stream |
| normal `async_finish=True` | 返回通信完成 event，消费者建立 stream wait；C++ 为 tensors 记录 stream |
| LL 普通 event 模式 | compute → comm launch；可返回 event，或插入 compute 等待 comm |
| LL `return_recv_hook=True` | 在调用时捕获的 current stream 上只 launch SEND；hook 再 launch RECV |

所选 non-cached normal dispatch 在 notify 后仍由 CPU 轮询可见的接收/专家计数，再分配返回张量并发起 payload dispatch；hierarchical normal 也有动态计数等待。`async_finish=True` 控制后续通信完成的 stream/event 衔接，不保证整个 Python 调用没有 CPU 等待。intranode cached 或 worst-case allocation 是其他路径，本页 caller 没有选用。（S `deep_ep.cpp:429–472,812–843`；V `487–541,1020–1048`。）

`EventHandle.current_stream_wait` 向当前 CUDA stream 插入依赖，不是 CPU 阻塞到 GPU 完成。LL async 与 hook 互斥；hook 返回不证明结果可在无依赖的另一 stream 使用。LL event wrapper 还保存 extra tensor 引用维持异步生命周期。（S/V `event.hpp`、`utils.py::EventOverlap`；S C++ `1131–1199,1252–1302`，V `1456–1541,1601–1665`。）

`LowLatencyLayout` 的 send/recv/signaling 各有 odd/even 两套区域，dispatch/combine 共用并切换索引；SEND phase 清下一套 signaling 元数据。显式 `clean_low_latency_buffer` 在前后执行全组 block barrier 并清两套 signal 区，用于此前 normal/chunked 使用后切回 LL；它不清全部 payload 或返回张量的容量尾部。（S `config.hpp:100–181`、`deep_ep.cpp:1067–1084`、`internode_ll.cu:10–26,183–192,604–613`；V 同名布局/函数。）

Buffer destroy 在同步和 barrier 后关闭 IPC、释放 NVSHMEM/工作存储；vLLM manager 显式 destroy，SGLang 使用默认析构路径。每轮 handle、长生命周期 Buffer 与完成 event 是不同对象，不能以 Python handle 清空替代 GPU completion 或 Buffer 回收。（S/V `deep_ep.cpp::destroy`；框架 manager/dispatcher。）

## 9. 支持边界与剩余工作

库条件比框架层“按 128 分组”更窄：

- intranode normal launcher 只列 $P=2,4,8$，notify 要求 $L\le128$，dispatch $K\le32$。旧 S 的 SM90 返程接收按 32 个 int4（512 bytes）的 warp 块组织 TMA 写；V 增加独立 int4 尾部 store。本页取 hidden bytes 按 512 对齐的行，不外推未运行的任意尾块安全性。
- hierarchical normal 的 $R=P/8$ 特化为 S $\{2,4,8,16,18,20\}$、V $\{2,3,4,6,8,12,16,18,20\}$，并要求八 GPU 组内拓扑匹配；$P>8$ 且为八的倍数还不充分。
- LL host 要求 $D\bmod512=0$、$C\bmod4=0$，launcher $K\le11$。S hidden 集合为 $\{2048,2560,4096,5120,6144,7168,8192\}$，V 另支持 3072。仅 $D\bmod128=0$ 不能证明可用。
- V 的 `TOPK_IDX_BITS` 默认 64，覆盖为 32 会改变 ID dtype 契约，本页 int64 caller 不自动兼容任意构建选项。
- V LL wrapper 要求 `NVSHMEM_QP_DEPTH >= 2*(Mmax+1)`，默认 depth 1024。$M_{\max}=256$ 满足，512 则需至少 1026；不外推为 S 的同一条件。V `enable_shrink=False` 时 rank mask pointer 为空，等待超时走 trap 路径，不是正常模型 dropping；S 旧 LL 接收等待没有同一套 shrink/timeout 处理。

上述定位为 S/V `launch.cuh`、`intranode.cu::notify_dispatch/dispatch/combine`、C++ LL checks；V `buffer.py:111–112,601,655`、`configs.cuh:64–70`、`setup.py:85–88`、`internode_ll.cu:380–405`。

本页已核 rank/组去重、局部 ID、FP8 布局与逆向归约，LL 动态量化、容量/scale strides、回源加权与双缓冲同步。没有安装、编译、GPU/模型/网络/故障或性能实测，也未审查 NVSHMEM 外部源码、完整 IBGDA WQE/完成协议、全部 ring 竞争或任意 overlap/graph、LogFMT/UE8M0/MNNVL/shrink、V2 与其他后端。

专家 GEMM 是独立设备边界，现由 [SGLang 匹配 DeepGEMM 0.1.3 的 SM90 接续页](deepgemm-moe-expert-execution.md)承接 contiguous/masked 行选择、TMA/scale、WGMMA、FP32 分组恢复与 BF16 输出。通信有效 range 与 GEMM 的 tile padding 分别解释；[DSA logits](dsa-sparse-device-execution.md)使用另一固定版本和对象，不能代替专家 GEMM。

## 来源身份与定位

DeepEP 完整来源为 S：9af0e0d（`https://github.com/deepseek-ai/DeepEP/tree/9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee`） 与 V：73b6ea4（`https://github.com/deepseek-ai/DeepEP/tree/73b6ea4a439ba03a695563f9fd242c8e4b02b37c`）。两个固定父仓库均保留上游归档与官方递归 Git tree，逐 blob 核对字节、类型与执行位；来源副本未应用框架配方补丁。下表路径相对各固定 DeepEP 快照。

| 文件 / 函数 | 核读职责 |
| --- | --- |
| `deep_ep/buffer.py` | 初始化；layout/dispatch/combine；三种 handle；LL kwargs、QP 条件与 hook/event |
| `deep_ep/utils.py`、`csrc/event.hpp` | tensor references、current stream wait 与 CUDA event 记录 |
| `csrc/deep_ep.cpp/.hpp` | rank 分组、IPC/NVSHMEM 生命周期、计数/分配、scale strides、LL phases/输出复用 |
| `csrc/config.hpp::LowLatencyLayout` | odd/even 容量计费、combine slot stride 与 clean metadata |
| `csrc/kernels/layout.cu` | 专家配对计数和 rank/RDMA 组 presence 去重 |
| `csrc/kernels/intranode.cu` | notify prefix/alignment、分平面队列、局部 ID、BF16/FP32 返回 |
| `csrc/kernels/internode.cu` | SourceMeta、组去重/fanout、int32 wire ID、两段 combine_token 舍入 |
| `csrc/kernels/internode_ll.cu`、`utils.cuh` | per-128 cast、消息/segment、BF16 返回与 FP32 gate 归约、pack/acquire/release |
| `csrc/kernels/runtime.cu`、`ibgda_device.cuh` | NVSHMEM team/alloc/barrier 与 peer pointer/put/atomic 调用边界 |
| `csrc/kernels/launch.cuh` | rank/hidden/type 特化与 launch 条件 |
| V `setup.py`、`kernels/configs.cuh` | TOPK_IDX_BITS 编译选项与默认 dtype |

框架版本与 caller 见 [EP 页来源表](moe-expert-parallel-execution.md#来源身份与定位)。配方定位为 SGLang `docker/Dockerfile:7–11,294–324`、`scripts/ci/cuda/ci_install_deepep.sh:34–39,101–112,146–151`，vLLM `docker/Dockerfile:479–489`、`tools/ep_kernels/install_python_libraries.sh:11–13,37–43,127–154,168–189`；这些脚本只作静态分析。
