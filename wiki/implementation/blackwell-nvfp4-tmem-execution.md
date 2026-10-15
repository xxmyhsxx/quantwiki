---
title: GB300 NVFP4 GEMM：输入双环、TMEM 累加与输出复用
type: implementation
tags:
  - kernels
  - matmul
  - fp4
  - tensor-core
sources:
  - raw/repositories/2026-09-24/fast-cu
updated: 2026-10-15
---

# GB300 NVFP4 GEMM：输入双环、TMEM 累加与输出复用

fast.cu 的 GB300 GEMM 接收已打包的 FP4 数据和块尺度，通过两条 TMA 环分别供给载荷与尺度，再由 `tcgen05` 在 TMEM 中累加，转为 FP16 输出。最终 r9 的两份累加区共享36列；输出必须先读走这段区域，下一输出才可覆盖它。输入 slot、尺度 TMEM 与输出累加区由不同协议复用。

本页核对固定源码的 r0 基线与 r9 所选执行主线，目标为 **GB300/B300、SM103a、CUDA 13.1 或支持所需指令形式的兼容工具链**。源码的 K96 形式限定这条路径；SM100 文档的类型、合法配置和尺度块格式仍由 [CUTLASS 块缩放 GEMM](cutlass-blackwell-block-scaled-gemm.md)解释。H100 的寄存器累加与 shared 输出见 [Hopper BF16 执行](hopper-gemm-tma-wgmma-pipeline.md)。

## 1. 量化产物是 GEMM 的输入

把 A 记为 `[M,K]`，B 的数学列记为 `[N,K]`，两者沿 K 连续保存，输出为 row-major FP16 `[M,N]`。载荷是两个 E2M1 元素合一字节，每16个K元素共用一份 UE4M3 尺度；FP32 累加后直接转为 FP16，alpha=1、beta=0。

host 的 `cast_to_fp4_with_ue4m3` 对每16个值求 amax（下限1e-4），将 amax/6 编码成尺度，再用**解码后的尺度**归一化原值、选择FP4码。这样实际恢复值由保存的尺度与码共同确定。host reference 计算这些恢复值的乘积，检验的是量化后 GEMM；它没有拿原始float乘积来衡量量化保真度。（`main.cu:36–145`、`small_gate_ours`。）

`pack_vec16_scales` 将逻辑尺度排成标准 `VEC16_UE4M3` 布局，块内顺序对应[已有512B尺度块](cutlass-blackwell-block-scaled-gemm.md#4.%20%E5%B0%BA%E5%BA%A6%E7%9A%84%E9%80%BB%E8%BE%91%E7%9F%A9%E9%98%B5%E6%80%8E%E6%A0%B7%E6%8E%92%E6%88%90512B%E5%9D%97)。自写 kernel 与 cuBLASLt 从同一组 host packed 数据和尺度分别复制设备输入，不需要另一套专有 repack。公开驱动要求 `K%32==0`，此时 A/B 的16B行stride对齐不增加额外载荷；内部K%16的小gate允许行尾对齐空间。（`main.cu:155–230,1481–1518`。）

## 2. 一对 CTA 协作一个输出，七个 warp 分开承担职责

每个2-CTA cluster 共同计算256×256输出，每CTA负责128行。每CTA有224线程，角色如下：

| 角色 | 参与者 | 使用的状态 |
| --- | --- | --- |
| A/B搬运 | 每CTA warp5的一个lane | 六槽A/B shared环、ab_ready/ab_free |
| 尺度搬运 | 每CTA warp6的一个lane | 七槽尺度 shared环、sf_ready/sf_free |
| 尺度转入TMEM及MMA | 偶CTA warp4的一个lane | 复用的尺度TMEM、输入descriptors、累加TMEM |
| 输出 | 两CTA各warp0–3 | 每warp32行，读TMEM并转FP16写global |

两个CTA分别加载自己的 A 行和 B 列分片，供一次 `cta_group::2` MMA 使用；SFA对应各自M半区，SFB搬运用multicast让所需尺度出现在两份shared中。这与两个CTA各自计算完整独立 GEMM 的分工不同。（`gemm0.cuh:696–930`；`gemm9.cuh:1230–1508`。）

shared 结构总计230400B，包含 A/B六槽、尺度七槽、barrier header及对齐。warp0分配512列TMEM；使用该地址的warp0–4先汇合，两个 TMA producer warp继续执行自己的环。shared阶段数与TMEM列数属于不同资源，不能把增加某一环的stage当成免费预取。

## 3. 载荷与尺度按各自的最后读取点归还

每条 shared 环都有 ready/free barrier。producer 等free后重填slot，再登记预期传输字节并发TMA；MMA角色等ready后消费。r9的A/B一槽在整个CTA对上期待65536B，尺度一槽期待9216B；ready/free各以一次相应通知推进。字节量由实际搬运与协作拓扑决定，和等待线程数分别计数。（`SmemCD`、`AB_TX/SF_TX`、producer loops。）

尺度先经 `tcgen05.cp.cta_group::2.32x128b.warpx4` 从shared转到TMEM。源码将尺度copy与后续MMA交给同一个issuer，依次复用一块尺度TMEM；shared尺度slot在其最后copy后，通过有序 `tcgen05.commit` 关联的通知归还。A/B slot则在最后读取它的MMA后归还。这里的commit是异步计算流的完成通知，不能将发出通知指令等同于立即允许producer覆盖。（`cp_sf_buffer/commit_consumed/mainloop_tile`。）

### r9 加宽输入窗口后，释放点随读取方式变化

r0每A/B槽搬96字节FP4载荷，即K192；每个K768组用四槽，每槽供应两次K96 MMA。它将K尾补到完整768组，由TMA越界补零。

r9每槽填满128B窗口，即K256，一个完整K768组只需三槽，但仍发八次K96。按从0开始编号，第2、5次MMA跨越相邻窗口：descriptor的地址字段与absolute-LBO字段分别取自前后slot。三槽的最后reader分别是MMA2、5、7，释放位置因此不能继续套用“每两条MMA归还一槽”。尺度仍按四批供给这八次MMA，独立推进它的七槽环。（`gemm0.cuh:644–687`；`gemm9.cuh:589–829`。）

r9另有K64组合与部分K组的活跃指令筛选。producer和consumer必须同时跳过相应槽位操作、保持slot/parity一致；本页展开完整K768主线，具体尾分支的本次核读范围集中登记于文末。

## 4. 两个累加区共享36列，输出顺序提供复用空间

r0只有一个256列累加区；r9把相邻输出轮流写到两个起点不同的256列区域。以分配到的TMEM基址为0表示：

| 区域 | 占用列，右端不含 | 生命周期 |
| --- | --- | --- |
| 累加区0 | `[0,256)` | 本轮FP32结果，由高列向低列读出 |
| 累加区1 | `[220,476)` | 相邻轮FP32结果，由低列向高列读出 |
| 尺度区 | `[476,512)` | 九个4列尺度copy tile，主算期间反复更新 |

前两区共享 `[220,256)` 的36列。完整两份累加加尺度需要548列；源码借用时间复用，把它们放进512列预算。尺度区没有与累加区重叠。（`D_STRIDE/D_DB/SF_TMEM_BASE`。）

每个epilogue warp按32列band发 `tcgen05.ld`。区0先取高端两个band，区1先取低端两个band，均先覆盖共享的36列。第二个band后，`tcgen05_wait_ld_then_arrive_release` 把 `wait::ld` 与free通知放在同一asm中；两CTA的四个epilogue warp各由一个lane通知，共8次到达组成acc_free。MMA角色在下一输出首条指令前等待该状态，才覆盖相邻累加区；此时旧输出剩余的非重叠列仍可继续读出。（`gemm9.cuh:1535–1551`。）

acc_free在这里表示“下一累加区已可使用”，不是整个旧结果都已写到global。r0则在第八个band才归还其唯一累加区；末轮还有MMA drain与两CTA的TMEM释放握手。（`gemm0.cuh:965–969`；`gemm9.cuh:1507–1508,1626–1642`。）

TMEM读出的寄存器值随后被打包为FP16，直接写global，不经过H100式shared输出区。r9完整16元素且满足32B对齐时走256-bit store，尾部用128-bit或标量路径。不过未满足32B条件的分支仍会发128-bit store，源码未另行检查每行16B对齐；正M/N的CLI条件不足以保证任意N行stride都适用。（`gemm9.cuh:1553–1616`。）

还需区分源区域释放与寄存器值的完成等待：所选 `tcgen05.ld` helper只发load，r0仅在末band、r9仅在第二band显式 `wait::ld`，其他band紧接转换与store。上述边缘释放顺序有实际指令依据，完整drain的值可用性不能据这些局部等待推定已经证明。

## 5. 调度表改变 A 的重复访问归属

cluster持续领取 `t=cluster_id+q*num_clusters` 的逻辑工作编号，再查host准备的route表取得真实输出tile。r9的owned调度按运行时SM侧分类，将M方向的tile行按两侧工作份额分配，边界至多共用一行；重复读取同一A行的任务尽量归同一侧，B列仍可被两侧使用。它分配的是**读取请求的复用归属**，没有把A的物理buffer安放到单侧。（`sched::make_owned_partition/verify_owned`。）

raster、pocket、Hilbert决定各池内访问次序，auto按形状/工作集选择其中一种。128MiB阈值和32×32网格的8×8 pocket是该实现的经验策略。实际cluster数由设备信息及驻留查询建立；源码每次launch还核对观测SM侧与计划侧，发生placement差异则使运行失败。（`main.cu:595–807`；`pick_schedule`。）

route表工作容量为4096个输出tile。超限时，即使某路径打印natural-raster fallback，最终上传仍abort；部分census失败也直接abort。只能按实际异常路径理解降级行为，不能由一条fallback消息推导任意大形状仍可执行。

## 6. 正确性门槛与计时各覆盖什么

共享驱动先做两层检查：小shape用量化后host FP32 reference、输出毒化、非有限拒绝、误差阈值 `0.5+0.002|ref|` 和两次逐bit确定性；用户shape再比较自写kernel与cuBLASLt首个heuristic，并检查重复结果。小gate的M=768、N=512都整除256 tile，实际列出的1136/1152，以及r8/9的1280/1024/1120检验K尾类；注释所称partial M/N不对应这些数值。用户shape没有另算独立host reference。检查失败即退出，通过后才进入计时；公开入口的K%32约束也作用于 `--check-only/--ours-only`。（`main.cu:987–1147,1481–1554`。）

cuBLASLt使用相同E2M1与VEC16尺度、FP32 compute、FP16输出、FAST_ACCUM=0，workspace上限64MiB，只请求第一个heuristic。它是明确的比较对象，不代表遍历全部vendor算法后的最快者。

| 模式 | 设备GEMM调用如何计时 |
| --- | --- |
| current / clock-pinned | 按至少256MiB或两倍L2的复用距离预算轮换地址，先跑牺牲slot，再event计时；配对轮次交换两实现先后，报告各自median及胜出轮数 |
| triton-style | 每次launch前在event外清理256MiB缓存区，按25ms warmup/100ms测量预算取逐launch median与IQR；顺序固定自写后vendor |
| sustained | 交替长时间运行两实现，以60秒block中20秒后的tail平均观察持续负载 |

clock-pinned shell请求1305MHz并在退出时重置；采样时钟符合条件与整个计时间隔均稳定是不同证据。地址轮换和清理区表达缓存控制方案，静态代码本身不证明实测L2全冷。小shape还可能因所需轮换slot超过40而被拒绝。（`measure_rotating/measure_triton_style/measure_sustained_block`、`run_bench_modes.sh`。）

以上计时均在量化、尺度打包、分配/H2D、tensor maps、schedule/probe及正确性检查之后；吞吐以逻辑 `2MNK` 计，即使早期版本额外执行补零MMA。它们度量不同负载协议下的设备GEMM调用，不包含从float输入形成量化产物的成本。

## 来源身份

pranjalssh/fast.cu（`https://github.com/pranjalssh/fast.cu/tree/2dfe5e26aecfd9e5f27bf9d5837deea01acda24b`），固定 `2dfe5e26aecfd9e5f27bf9d5837deea01acda24b`，取得于 `2026-09-24T19:52:11+08:00`；commit信息为“Add GB300 NVFP4 GEMM”，晚于H100文章。

本条完整核读raw/根README、Makefile、GB300 README与shell、`main.cu:1–1895`、`gemm0.cuh:1–1148`；r9定向核几何/结构、helper、完整K组、尾分派入口/普通收尾、producer/MMA/epilogue/teardown及所述调度承接，联合范围为 `1–340,349–890,1160–1663,2183–2301,2342–2384,2631–2769`。r1–8本体、r9深层尾变体、L2探测算法及其余表构造未全核；它们按已读调用与README说明范围使用，未将十级路线表写成十份实现均已审定。34文件的归档字节/执行位证明复用原登记；未编译、运行GPU、取得本地正确性或性能结果。
