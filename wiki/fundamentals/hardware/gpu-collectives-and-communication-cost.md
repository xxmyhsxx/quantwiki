---
title: 多 GPU 集合通信：归约、分片与 ring 成本
type: concept
tags:
  - tensor-parallelism
  - performance
  - memory
sources:
  - raw/articles/2026-09-24/iit-bombay-systems-for-ml-2026
updated: 2026-10-15
---

# 多 GPU 集合通信：归约、分片与 ring 成本

集合通信（collective）规定一组参与者如何交换、归约或重组张量。All-reduce 与 all-gather 处理不同的数学对象；相同的 all-reduce 又可以采用 ring 或 tree 等算法。因此判断多卡成本时，需要先说明各 rank 要得到什么，再计算所选算法搬运多少数据、经过哪些路径。

本页依据 IIT Bombay CS 794 Spring 2026 保存讲义第6讲“Distributed Training”和第8讲“Network Optimizations”解释公共机制。固定框架中的调用与分片见[张量并行与量化](../../implementation/tensor-parallel-quantization.md)、[FSDP2生命周期](../../implementation/fsdp2-parameter-lifecycle-and-checkpoint.md)；MoE的变长路由与回传另见[专家并行](../../implementation/moe-expert-parallel-execution.md)。

## 1. 归约相加，收集拼接，交换改变归属

设通信组有 $p$ 个 rank，以 $r=0,\ldots,p-1$ 标识参与者；rank是组内编号，不直接等同于主机号。以SUM归约为例，各rank持有同形状向量 $x_r$，将其分成 $p$ 段 $x_r^{(j)}$，定义

$$y^{(j)}=\sum_{r=0}^{p-1}x_r^{(j)},\qquad y=[y^{(0)},\ldots,y^{(p-1)}].$$

| 操作 | 输入怎样分布 | 操作后每个rank持有什么 |
| --- | --- | --- |
| All-reduce | 各rank有自己的完整 $x_r$ | 所有rank都得到归约后的完整 $y$ |
| Reduce-scatter | 各rank有自己的完整 $x_r$ | rank $j$只得到归约结果的第 $j$ 段 $y^{(j)}$ |
| All-gather | rank $j$有一段 $z_j$ | 每个rank得到按约定顺序拼接的 $[z_0,\ldots,z_{p-1}]$，不相加 |
| All-to-all | 每个rank $r$按目标切好 $x_r^{(0)},\ldots,x_r^{(p-1)}$ | rank $j$收到所有来源发给它的 $[x_0^{(j)},\ldots,x_{p-1}^{(j)}]$，不自动归约 |
| Broadcast | 指定root持有输入 $x$ | 将root的 $x$复制给组内所有rank |

Reduce只让指定root取得归约结果，再broadcast可实现all-reduce；另一种分解是先reduce-scatter，再all-gather这些已归约的段。后者要求通信组、段顺序和归约定义相容。它是操作语义的分解，不要求实现真的先生成一张完整 $y$ 再切开，也不保证不同浮点归约顺序逐位一致。（第6讲PDF第15–22页。）

教学例：两个rank分别有 $[1,2]$ 和 $[3,4]$，SUM reduce-scatter可让rank0得到 $[4]$、rank1得到 $[6]$；随后all-gather使两者都得到 $[4,6]$。直接收集原输入则得到 $[1,2,3,4]$，做的是另一种运算。

[FSDP2](../../implementation/fsdp2-parameter-lifecycle-and-checkpoint.md)的梯度reduce-scatter让参数owner只保留已归约的所属梯度，参数all-gather则临时重组计算所需权重。两次通信之间还有优化器更新与参数生命周期，不能把它们当作对同一张量直接相接的all-reduce。[TP行并行](../../implementation/tensor-parallel-quantization.md)需要相加各输入分片的输出贡献；列并行输出需要恢复完整轴时才做拼接。

## 2. Ring 用两阶段转发分摊归约与复制

考虑一个单向逻辑环，每rank的输入大小为 $S$ bytes，等分为 $p$ 块，每块 $S/p$ bytes。所有rank同时向后继发送、从前驱接收。（第8讲PDF第33–36页；以下按图中的数据流整理。）

1. **Reduce-scatter阶段**：每轮传一块部分和，接收者加上自己对应块的贡献再继续转发。经过 $p-1$ 轮，每个rank拥有一块已累积全部rank贡献的结果。最终块归属可随环的编号约定旋转。
2. **All-gather阶段**：沿环继续传送这些已完成的块，接收者保存并转发，不再重复加自己的输入。再经过 $p-1$ 轮，各rank集齐完整结果。

第8讲第36页把初始send、中间recv/reduce/copy/send及最后recv列为 $2p-1$ 个原语步骤，同时写明 $2(p-1)$ 次数据传输。前后两者计数对象不同：估算环上通信轮数时，使用两个阶段各 $p-1$ 轮，不能将图中七个步骤直接算成四卡的七轮传输。

由每轮发送一块可得，每rank的发送量为

$$V_{\rm send}=2(p-1)\frac{S}{p}.$$

接收量相同；如果指标把发送与接收相加，则是 $4(p-1)S/p$。这里 $S$是**通信张量**的字节数：例如 $n$ 个元素、每元素 $b$ bytes时 $S=nb$，不能用量化权重文件的大小代替行并行输出或梯度的大小。整个组的发送总量则是 $2(p-1)S$，与每rank口径也不同。

进一步作一个简化延迟模型：每轮固定开销为 $\alpha$ 秒、各环边有效单向带宽均为 $B_{\rm link}$ bytes/s，发送与接收可重叠，忽略本地归约/拷贝与额外争用，则

$$T_{\rm ring}\approx2(p-1)\alpha+
 \frac{2(p-1)S}{pB_{\rm link}}.$$

这是整理者从分块和轮数得到的教学模型。大消息能摊薄固定延迟；小消息或rank很多时，随 $p$ 增长的轮开销可能显著。它也说明增加rank不会使all-reduce字节量按 $1/p$ 下降。真实实现还会分channel、切更小片、使用树或硬件归约，应按实际算法和有效链路评估，不能从这一模型直接承诺扩卡收益。

## 3. 逻辑环、物理网络和传输方式分别决定成本

Ring规定谁向谁转发；物理上两者可能经过同机NVLink/PCIe，也可能跨NIC和交换网络。RDMA描述网络传输机制，GPUDirect RDMA进一步使适配条件下的NIC直接访问GPU内存，减少host中转；这些名称不决定通信的归约语义或逻辑拓扑。（第8讲PDF第9–14、31–39页；讲义具体协议示例不作为当前NCCL版本的实现保证。）

物理带宽不均时，应把高频流量放到相应路径上。讲义的rail例子将不同主机上相同局部GPU位置的NIC组织在同一rail，利用节点内快速连接先做局部归约，再通过rail进行节点间通信；完整all-reduce还需将跨节点结果分发给本地参与者。其收益依赖任务分组与硬件连接的匹配，不能仅凭“使用ring”或“使用NVLink”判断。（第8讲PDF第17–19页；最后一步按all-reduce输出语义补明。）

同一逻辑消息还可拆成多个channel并行搬运，各channel再按小片形成流水：一片在网络上传输时，另一片可在GPU上归约或转发。这样能重叠不同阶段，但更多channel也会消耗SM、缓冲和链路资源，分片过小还会放大逐片开销。（第8讲PDF第40–44页。）

计算与通信的重叠同样受依赖限制：数据并行可在某层梯度产生后开始同步，与更早层的反向计算重叠；使用该梯度的更新仍须等待归约完成。先给出张量的生产者、消费者和就绪时间，才能判断通信能隐藏多少，不能只看有没有异步API。（第6讲PDF第10页；FSDP2的具体预取条件见[参数聚合与峰值](../../implementation/fsdp2-parameter-lifecycle-and-checkpoint.md)。）

## 来源身份

| 来源 | 版本与时间 | 使用范围 |
| --- | --- | --- |
| Systems for Machine Learning — IIT Bombay CS 794（`https://www.cse.iitb.ac.in/~mythili/sysml/`） | `Spring 2026 course archive; slides exported 2026-09-24`；获取于2026-09-24T02:12:20+08:00 | 第6讲第10、15–24页与第8讲第9–44页的collective语义、ring图和网络分层；已核相关原页。通信量/延迟模型为整理者推导，未核课程引用的完整论文、NCCL实现或运行性能。 |
