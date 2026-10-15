---
title: CUDA softmax：线程分工、局部统计与两级归约
type: implementation
tags:
  - kernels
  - gpu
  - reduction
  - softmax
sources:
  - raw/repositories/2026-09-24/cudacodes
updated: 2026-10-15
---

# CUDA softmax：线程分工、局部统计与两级归约

固定 cudacodes 的教学实现对连续 FP32 矩阵 $X[M,N]$ 沿每行计算 softmax，将同形状结果写入独立输出。0–4版依次改变线程负责的工作、统计量的合并方式与每次访存宽度，主体仍是建立整行最大值和指数和，再输出概率。本页追踪输入到输出的静态计算设计，运行验证范围见末节。

范围为 `Maharshi-Pandya/cudacodes` commit `9a459b1007c68430c801389f4eaf31cf31b906a6` 的所选 softmax 主线，不覆盖仓库其他算子。稳定 softmax、在线指数基准与填充值见[计算模式](../fundamentals/operators/gpu-kernel-computation-patterns.md)，地址与公开接口的区别见[张量布局与接口](../fundamentals/operators/tensor-layout-and-kernel-contracts.md)。

## 1. 默认入口只运行第4版

`softmax/Makefile` 的默认目标将 `kernels/*.cu` 全部编入 `softmax`，主程序是 `bench.cu`，包括未完成的第5版源码；`test.cu` 不在这个默认目标内。`bench.cu:main` 没有命令行选版参数，固定 M=1024，让 N 从2048翻倍到131072，每个shape只调用一次 `run_kernel_4`。其他 runner 存在，不表示默认命令会逐版运行或验证它们。（Makefile、`bench.cu:benchmark_kernel_for_sizes/main`。）

主机路径先生成输入，分别分配设备输入 `matd` 与输出 `resd`，复制输入后调用runner，再把输出复制回host数组。设备kernel的输出由这次D2H继续消费；源码没有接入PyTorch tensor、autograd或模型调用。下面的异地输出范围来自这条实际路径，不能据指针签名推定任意输入/输出重叠都安全。

各版使用 `float` 数据和统计，按 `row*N+col` 寻址，没有stride、dtype、mask或显式stream参数。可直接解释的范围是非空有限FP32行、连续row-major布局、合法尺寸/设备资源及独立输出；包装没有完整检查或拒绝其他组合。尺寸、地址乘法与主机字节数使用int，也不能无条件推广到超大张量。

下文 T 表示 `blockDim.x`，t 表示 `threadIdx.x`。

| 版本与入口 | 默认分工 | 输入逻辑扫描与归约 |
| --- | --- | --- |
| 0 `run_kernel_0` | 1024线程/block，grid为ceil(M/1024)，每线程一行 | 三遍输入：max、指数和、输出 |
| 1 `run_kernel_1` | 同0 | 在线合并max/指数和，再读输入输出 |
| 2 `run_kernel_2` | grid=M，每行一个1024线程block | 线程局部在线统计，shared max与校正sum，再读输入输出 |
| 3 `run_kernel_3` | 同2，32个完整warp | warp内shuffle→shared汇集→第一warp归约，两遍输入 |
| 4 `run_kernel_4` | 同2，每任务使用float4并处理scalar tail | 相同统计链经公共helper实现，两遍输入 |

M=1024时，0/1默认仅有一个线程块（CTA），2–4则有1024个CTA。这是launch与行映射的差异，不能把各版性能变化单独归因于shared或shuffle。表中的逻辑扫描次数也不等于实测HBM流量，缓存和编译结果没有测量。

## 2. 一线程一行：先减少重复输入扫描

`naive_0.cu:softmax_kernel_0` 将 `row=blockIdx.x*T+threadIdx.x` 分给线程。它先完整扫描该行求最大值m，再扫描求 $\ell=\sum_j e^{x_j-m}$，第三次读取输入写 $y_j=e^{x_j-m}/\ell$。线程自己的读地址连续，但同warp的相邻线程负责不同row，在同一col访问相隔N个float；不能由单线程连续性直接推出warp访存合并。

`online_1.cu:softmax_kernel_1` 在第一遍保留m与对应指数和：遇到更大输入时，先将旧和转到新最大值的基准，再加入当前项。它合并前两遍，仍须完整行统计后第二遍读输入，写独立 `resd`。同仓库README中的online片段写回 `input[i]`，与该固定 `.cu` 的输出路径不同；本页以实际kernel为准。0–3的指数使用 `expf`，FP32归约顺序与其他实现仍可能不同。

## 3. 一行一block：先统一局部统计的基准

`sharedmem_2.cu:softmax_kernel_2` 令 `row=blockIdx.x`，线程t依次读列 `t,t+T,t+2T…`。同一轮相邻线程读取相邻列，每线程在寄存器中得到局部最大值 $m_t$ 和按该最大值定义的指数和 $\ell_t$。

归约先把 $m_t$ 写入shared并做max树，得到整行m。线程读取m后，把自己的 $e^{m_t-m}\ell_t$ 写入同一shared区，再做sum树，得到整行指数和。各局部分母先转到共同基准，不能直接相加；完整数学机制由前述[在线统计](../fundamentals/operators/gpu-kernel-computation-patterns.md)承接。最后每线程第二次读取自己负责的列，写入概率。（`sharedmem_2.cu` 的局部循环、`row_max`、校正shared写入、`row_norm`与输出循环。）

这里 `smem[1024]` 保存的是统计，不是整行输入。块屏障分别保证shared写入可读、下一轮归约依赖成立，以及所有线程读完最大值后才能复用shared存sum。默认T=1024是2的幂，原二分树按此覆盖；改成非2幂不能只改launch。有限非空行中，没有输入列的线程保留 $(-\infty,0)$，相对有限整行m的校正贡献为0，但仍参加collective。全空、全负无穷或NaN等需要另外定义语义，不能由这个初始化推出已支持。

## 4. Warp内用寄存器交换，跨warp仍用shared

`shfl_3.cu:softmax_kernel_3` 把max与校正后的sum分别拆成两级：每个完整warp用offset16/8/4/2/1的down-shuffle归约；各warp的lane0把结果写进shared；块屏障后第一warp读取这些统计，再做一次warp归约，写入 `smem[0]`，由整个block同步后读取。

默认1024线程组成32个完整warp，`FULL_MASK`覆盖实际参与者；无有效输入的线程没有提前返回，第一warp的无用统计槽则分别填负无穷或0。这样的中性填充与实际参与线程需要同时成立，不能凭固定树或mask参数宣称支持任意部分warp/稀疏成员。成员与内存依赖见[Warp collective的解释](../fundamentals/hardware/gpu-execution-and-memory-hierarchy.md#Warp%20collective%20%E7%9A%84%E6%88%90%E5%91%98%E4%B8%8E%E5%86%85%E5%AD%98%E4%BE%9D%E8%B5%96)。

这一版减少warp内部逐轮shared访问和块屏障，跨warp的shared与同步仍然存在。源码注释称XOR shuffle、甚至以持续锁步解释“无需同步”，实际调用是 `__shfl_down_sync` 并保留上述块级交换；本页按实际操作解释，不采用注释作为通用正确性保证。

## 5. Float4的标量尾部不修复行首对齐

`vectorized_4.cu:softmax_kernel_4` 保留每行一block，将行指针转换成 `float4*`。线程t处理编号 `t,t+T…` 的四元素向量，更新局部m和指数和；前N%4个线程另读scalar tail。整行统计通过 `cuda_utils.cuh:blockReduceMax/Sum` 完成，输出阶段再次读取float4及尾部，计算四个概率并写回。

默认动态shared只请求ceil(1024/32)个float，即128字节，保存warp统计。helper在第一warp写完最终 `smem[0]` 后没有统一尾屏障，caller明确在返回后同步才读取；max结果读取后的屏障又保护shared转作sum。独立复用helper时不能漏掉调用方协议。（`cuda_utils.cuh:blockReduce`；`vectorized_4.cu` 两次helper及其后屏障。）

scalar tail只保护列尾。行首字节偏移是 `row*N*sizeof(float)`：即使base满足16字节对齐，N=5时第二行仍偏移20字节，不能继续据base推定该行float4访问对齐。N<4时不执行向量循环，而任意非倍4长行仍须另核或拒绝；现有包装没有这项检查。默认bench的N均为至少2048的2幂，与这项行步长条件相容。这个反例来自行地址与对齐条件的静态分析。

该版局部统计和输出用 `__expf`，局部分母转到整行基准时仍用 `expf`。指数路径、归约顺序都变了，输入/输出是FP32不能保证与0–3或PyTorch逐位相同；应按[数值参考与容限](kernel-correctness-and-benchmarking.md)验证。

## 6. 第5版保留未完成状态

`blocktiling_5.cu` 宣称每block处理4行，但固定代码没有完成该分工：`bx`取的是 `blockDim.x`，row写成 `bx*4+threadIdx.y`，没有使用blockIdx；默认block=(4,32)时，M≥48下所有CTA重复行16–47，扩大grid也不增加行覆盖。x最快线性化又让一个warp跨8个ty/行，整warpshuffle没有建立每行四线程的独立归约。

其 `warpReduceMax(float)` 没有return却被赋值使用；四元素最大值数组只将第一项显式初始化为负无穷，其余为0；内部j/tile循环没有进入行列地址，反复读写同一位置。这些静态缺口使本页将该版保留为未完成候选，不报告本地编译或运行故障。按ty变化的提前return与后续block屏障也不是本页已核通的尺寸路径。它未被默认bench/test选中，但仍属于Makefile的wildcard构建输入。

## 7. 默认test和保存图表没有提供统一的通过证据

`test.cu` 以4096×4096调用第4版，复制回host后释放，没有参考实现、输出误差、行和检查或assert。`CUDA_CHECK`只检查CUDA API状态。它的文件名、调用或D2H均不能证明softmax数值通过。

默认CUDA bench每shape只运行一次第4版，runner的event包围设备kernel并等待stop；分配、H2D与D2H另列，不含在返回的kernel毫秒数中，也没有预热/重复均值。`torch_benchmark.py` 对同系列shape先做5次warmup，再取5次同步墙钟平均，每次F.softmax创建输出。两边计时范围与样本不同，不能直接作为严格同口径速度排名。

两个 `exec_time_ms_*.txt` 各7行，只写M与ms，第一列全部是1024，没有N、日期或设备/软件身份。按当前脚本顺序可以条件性对应N，但日志本身不能唯一确认。`plot.py`不读这些文件，直接硬编码另一组N和时间；图像视觉上与硬编码趋势相符，不证明图、脚本和日志属于同次运行。例：第五项若按默认顺序对应N=32768，日志CUDA/PyTorch为1.112768/2.2152901ms，绘图数组却为4.109728/6.2500477ms。两组不能合并；按顺序解释日志时，其第一项CUDA还慢于PyTorch，不能用图宣称日志所有shape都更快。

源码README的设备称呼也不能补成这些日志缺失的运行元数据。这里的全部结果只涉及独立前向row softmax，不提供attention全链、backward或模型收益。本次完成固定源码与图表核读及静态推导，没有编译、执行现存二进制或运行GPU。如何建立统一参考、检查实际调用路径和计时范围，继续读[正确性与性能测量](kernel-correctness-and-benchmarking.md)。

## 来源身份

- Maharshi-Pandya/cudacodes（`https://github.com/Maharshi-Pandya/cudacodes/tree/9a459b1007c68430c801389f4eaf31cf31b906a6`），固定完整commit `9a459b1007c68430c801389f4eaf31cf31b906a6`，获取于2026-09-24T19:52:11+08:00；所选softmax 6个kernel及声明header、helper、Makefile、bench/test、PyTorch脚本、仓库根/softmax README和日志/图表。78普通文件与原归档字节/执行位一致属于档案核对；未审仓库其他算子、未执行现存二进制、未安装/编译/运行GPU或模型。
