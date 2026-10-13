---
title: OSCAR 的混合 KV 缓存：INT2 格式、窗口更新与分层 Attention
type: implementation
tags:
  - kv-cache
  - rotation
  - data-format
  - kernels
  - serving
sources:
  - raw/repositories/2026-10-08/oscar
updated: 2026-10-08
---

# OSCAR 的混合 KV 缓存：INT2 格式、窗口更新与分层 Attention

低位宽缓存需要把三个接口接起来：**怎样编码历史 K/V、何时把保护窗口里的值移入量化区，以及 Attention 怎样同时读两种精度并恢复输出**。本页核对 OSCAR 公开仓库中 vendored SGLang 的实际路径，连接[KV 格式与生命周期](../theory/kv-cache-quantization-objects-and-granularity.md#11.%20%E4%BB%8E%E4%BD%8D%E5%AE%BD%E5%88%B0%E5%8F%AF%E6%89%A7%E8%A1%8C%E5%AD%98%E5%82%A8%E6%A0%BC%E5%BC%8F)。

固定来源为 `FutureMLS-Lab/OSCAR` commit `41ebcdba3db5f0ce1339c3727caea80df575d437`。这是 [WUSH-KV](../methods/wush-kv.md#5.%20%E6%B5%81%E5%BC%8F%E7%BC%93%E5%AD%98%E3%80%81%E7%AA%97%E5%8F%A3%E4%B8%8E%E7%9C%9F%E5%AE%9E%E6%88%90%E6%9C%AC) 所用量化器的公开前置实现；本页没有取得 WUSH-KV 作者修改后的 SGLang，不能用 OSCAR 的正交旋转代码证明 WUSH 的一般可逆变换已接入。本文是静态源码核对及独立教学算术，没有安装、导入上游程序、编译、运行模型/GPU 或测质量、显存与性能。

## 1. 固定一条可连通的执行路径

选择 BF16、单设备 TP1、普通全注意力 GQA，K/V head dimension 均128，组长128，即每个token/KV head的K与V各保存一对scale/zero。量化区用 affine INT2；sink 与 recent 区用 BF16；元数据保存 FP32。预计算的 K/V 旋转矩阵已提供，V 旋转折入投影，采用旋转暂存与裁剪/量化/打包kernel分开的普通路径，关闭 Lloyd-Max 及 fused rotate/clip/quant。[1–4][9][10][15]

`rotation/eval_oscar_gpqa.sh` 的 recipe 选择 FA3 prefill、Triton decode、sink64/recent256、K clip0.96/V clip0.92、FP32 元数据和 V absorption。这里沿其量化/Attention 接口，收窄到 TP1、单步每请求一个新 token、fresh 非 chunked prefill；不延伸到 CUDA graph、共享 Radix 前缀、MLA/SWA、推测执行、CP 或分离服务。作为读取形状例，取32个 Query head、8个 KV head，GQA 比率4。[2][6–8][12][17][18]

这些数值并非统一源码默认值。`srt/environ.py:214–239` 默认关闭 mixed windows，prefix/recent 为32/128，裁剪为0，V absorption 与 fused 也关闭；HP BF16、metadata FP32 是该处默认。model runner 显式解析环境并传给 pool，因此不能把 pool 构造函数的 BF16 scale 默认形参当成此路径的实际配置。K/V clip 来自运行配置，不由 rotation checkpoint 学出或导出。[5][8]

## 2. 旋转写入的坐标与精度

离线脚本以 FP64 累积统计、求特征分解，保存 FP32 rotation；Qwen3-8B 脚本选 `r_h_pbr`，由特征向量、Hadamard 与置换等正交因子组成。运行 loader 读取 `layers[layer_id].rotation[D,D]`，转成 HP dtype，组成 `[layer,D,D]`，**每层一块矩阵供该层 heads 共用**；只检查字段、层与形状，不检查正交性。[3:75–133,245–295][4:22–26,88–93][9:136–181]

采用行 token 约定，理想实数运算中的配对为

$$Q'=QR_K,\qquad K'=KR_K,\qquad V'=VR_V,$$
$$Q'(K')^\top=QK^\top\quad\text{若 }R_KR_K^\top=I,\qquad O=O'R_V^\top.$$

正交性的作用和 [WUSH 的逆转置配对](../methods/wush-kv.md#3.%20%E5%8F%98%E6%8D%A2%E6%94%BE%E5%9C%A8%E5%93%AA%E9%87%8C%EF%BC%8C%E5%A6%82%E4%BD%95%E4%BF%9D%E6%8C%81%E6%9C%AA%E9%87%8F%E5%8C%96%E8%AE%A1%E7%AE%97)不同；运行时矩阵已经转成 BF16，裁剪、量化及矩阵乘舍入也改变结果，不能把上述实数等式称为执行中逐 bit 等价。浮点舍入的背景见[浮点格式与舍入](../fundamentals/numeric-formats/floating-point-and-accumulation.md)。

普通未预旋转接口中，pool的 `_rotate_kv_inplace` 得到BF16的 `K @ Rk_BF16`；所选FA3 prefill则先由 `prepare_quantized_extend_qkv` 做相同BF16旋转，传入已变换标记让pool跳过再次旋转。V absorption 在模型加载后以 FP32 `Rv.T @ Wv` 折入 V 权重，再转回权重 dtype；池收到的 V 已处于旋转坐标，仅转成 BF16。该折叠不能被当成“先算原 V，再做 BF16 V@Rv”逐 bit 等价，也没有把输出逆旋转折入 O 投影。[10:565–600][19:107–151][21:500–509,685–726][22:665–667]

prefill 的量化 writer 将已旋转 BF16 K/V 提升到 FP32再裁剪与编码；decode 当前值写入 HP BF16，**不裁剪、不量化**。之后 flush 读取这些 HP 行，提升到 FP32再编码，不重新旋转。因而量化中段和保护区不仅存储精度不同，还经过不同的裁剪路径。[10:624–714,749–811][15:92–107][16:185–204]

## 3. affine INT2 的码、元数据与裁剪

记一个经过裁剪的 FP32 head 行为 $x\in\mathbb R^{128}$。单组 writer 的实际关系为

$$m=\min_i x_i,\quad M=\max_i x_i,\quad s=\frac{\max(M-m,10^{-8})}{3},\quad z=-\frac m s,$$
$$q_i=\operatorname{uint8}(x_i/s+z+0.5),\qquad \widehat x_i=(q_i-z)s.$$

这里 z 是**浮点 affine zero**，不是舍入后的整数 zero point。缓存保存 `[s,z]` 两个 FP32 数；没有额外 bias 字段。写成 $sq+b$ 时，$b=-sz$ 只是数学改写。`uint8(...+0.5)` 也不是显式 round-to-nearest-even；源码在此没有另加 `clamp(0,3)`。以上定位于 `oscar_rotation_clip_int2_kv.py:162–198`；常量行由范围下限处理，不另造异常分支。[10:410–437][15]

D128 时每个字节包含四个**通道四分区**中的码：

$$u_j=q_j\;|\;(q_{j+32}\ll2)\;|\;(q_{j+64}\ll4)\;|\;(q_{j+96}\ll6),\qquad 0\le j<32.$$

它不打包四个相邻通道，也不沿 token 轴打包；读者必须用同一通道映射解码。每个 token/KV head 的 K 或 V 是32字节码加8字节元数据。一组元数据属于整条 head 行，不能泛化为任意多组时“一个字节只有一个 scale”。[15:162–198,274–299]

**教学还原例。** 构造四个32通道分区，值分别恒为 `[-3,-1,1,3]`，取不改变这些值的裁剪阈值3。则 $s=2,z=1.5$，四个码为 `[0,1,2,3]`，32个字节都为 `0xe4`。按位移0/2/4/6解包，再计算 $(q-1.5)\times2$，恢复对应四分区。这是独立整数/标量算术例，不是 GPU producer 测试。

**同一“百分位裁剪”并不保证同一码。** 若比例 $\rho>0$，源码取 $c=\min(\lfloor\rho D\rfloor,D-1)$；普通 prefill 将 `abs(x)` 排序，使用0-based索引c的值作为对称阈值。它是离散索引，不是插值分位数。D128 的 decode flush 则在 `[0,max(abs(x))]` 上做7次二分：目标超阈值数量为 $D-c$，以严格 `abs>mid` 计数，最后取上界。[15:97–107,333–346][16:204–226,527–539,770–784]

例如独立教学行 `abs(x)=0..127`、$\rho=0.96$ 给出 $c=122$，排序阈值122；上述7次二分得到121.046875。这说明**默认非融合 prefill 与 flush 已有阈值差异**。两者的除法接口也有差别：普通单组 writer 使用 `/`，flush 的 z 与 x/s 显式使用 `tl.math.div_rn`；静态阅读不能保证编译后的 bit equality。[15:162–178][16:232,275–278]

fused 选项另有精度变化：K/R 的 BF16 输入经 `tl.dot` 得 FP32，直接裁剪，没有普通路径的旋转结果 BF16 暂存；D128 同样采用7次二分。FA3已预旋转路径不进入这一选项。该选项要求未预旋转、V 已吸收、至少一个裁剪比例为正、K/V 都为单组，本页只核其与主线的差异，不延伸全部配置。[10:649–673][15:640–709,797–913]

## 4. 分开的物理池与统一 slot 身份

实际 `_create_arenas` 每层分配六个独立张量，记 KV head 数为 H、量化 slot 容量为 C、高精度 slot 容量为 A：

| 池 | 形状与 dtype | 一条记录的意义 |
| --- | --- | --- |
| QP K、QP V | 各 `[C,H,32]` uint8 | 每 head 的128个2-bit码 |
| K、V scale/zero | 各 `[C,H,2]` FP32 | 按 `[s,z]` 交错保存 |
| HP K、HP V | 各 `[A,H,128]` BF16 | 尚未量化的旋转坐标行 |

它们不是同一底层 byte arena 的别名。`unified_kv_pool.py:396–493` 的真实分配比旧 shared-arena 注释更能确定布局。普通 `get_key/value_buffer` 返回 QP byte buffer，scale/zero 和 HP 有独立 getter；不能将普通 getter 当成全部历史的 dense K/V。[10:528–559]

页几何函数 `compute_page_geometry` 返回 $N_H=1,N_Q=4\times\text{HP itemsize}$，故 BF16 下 **一页有8个 QP token slot，flush interval也是8**；这不表示一个字节打包8个 token。若配置量化容量为 T，QP页数为 $Q=\lceil T/8\rceil+1$，$C=8Q$，QP page0保留作 dummy。recent ring 每请求长度为 $256+8-1=263$，容纳256条最近值及最多7条待批量移出值。[8:547–563][10:104–120,185–225]

请求表 `req_to_token[request,logical_position]` 保存统一整数身份。设 HP 的 ID 起点 $O=C$、共享 HP prefix 容量为 P，则：

- `slot<O` 是 QP 行，直接以 slot 读码/元数据。
- `O≤slot<O+P` 是共享 prefix HP 行，本地行号 `slot−O`；`slot=O` 合法。
- recent请求槽编号r的环位cursor 使用 `O+P+r*263+cursor`，同样减 O 后读 HP。

QP 和 prefix HP 页由 allocator 管理，按页去重释放；recent ring 由请求槽独占，`free` 对其 ID 是 no-op，请求结束才重置 cursor/counter。HP prefix 容量默认由请求槽数、sink和预留系数导出，不等于每请求实际保留的64条。allocator 的 `available_size` 只数共享 QP/prefix 空位，不把 recent ring 加进去。[8:547–563][10:219–241,357–371][11:31–127,163–351][12:1003–1010]

**物理预算的教学计账。** 每 token/KV head 的 K+V 量化载荷是64字节码与16字节 FP32元数据，共80字节；同样 BF16 K+V 为512字节，故这一完整量化记录的比例是 **6.4倍**，裸码比例才是8倍。以128×2个元素平均为2.5 bit/元素。[10:410–437]

六种主缓存张量的预分配总字节可写为

$$B_{\rm KV}=LH\left[80C+512(P+263R_{\max})\right],$$

L 是层数，$R_{\max}$ 是请求槽容量。这是依据分配形状的整理者计账，包含 dummy/空QP槽与预分配 HP prefix/ring，仍未计请求索引、旋转矩阵、split scratch 等，也不是运行峰值显存。应按实际分配判断容量，不能把80/512外推成整机收益；共同区别见[服务内存管理](serving-memory-and-batching.md)。

## 5. Extend 与 Decode 怎样移动窗口

fresh 最后一个 prefill chunk 的逻辑布局按最终长度 S 分为：prefix长度 $\min(S,64)$、recent长度 $\min(\max(S-64,0),256)$、剩余中段QP。非最终 chunk 的 allocator 不建立同样的 recent tail，因此以下教学时间线只采用一次非 chunked prefill，不推广到已复用或分块请求。[12:340–383,414–639]

prefill 的 QP 位置进入第3节 writer，HP位置只保存 BF16。所选 FA3接口虽先更新缓存，本次当前 chunk 的 Attention仍使用旋转后的 **dense BF16 K/V**；仅历史前缀按请求映射 gather QP/HP、恢复成 BF16后与当前 chunk 拼接。fresh 无历史的整段 prefill不对本次 chunk 做INT2来回转换。[10:624–714][19:107–151,291–347][20:687–857]

decode 按下面的真实先后顺序闭合；“分配当前槽”还没有覆盖当前 K/V：

1. 根据每请求旧counter判断是否flush，旧counter为0则flush后置7，否则减1；先按最坏情况为每请求预留一页8个QP目标slot，并分配本次recent HP槽。
2. plan读取**旧**seq_len/请求映射，以flush mask选择待移出的旧HP行；不flush或无效项返回unused QP目标，valid项返回旧HP global ID。
3. 统一allocator归还返回ID，recent HP的free为no-op；等待上一forward event，先量化有效旧HP，再由单独remap kernel把历史位置指向QP。
4. 写当前新token的HP映射，schedule_batch随后增加seq_len；forward真正保存当前HP K/V，再进入本次decode Attention。

依据是 `common.py:642–820`、`gpu_flush_int2.py:64–140,790–816`、`schedule_batch.py:2195–2214`、`scheduler.py:1473–1490` 与 `triton_backend.py:1694–1721`。即使不flush，host仍预留/plan/归还，设备项以valid mask跳过，不能照抄“整条pipeline不调用”的注释。[11–14][16][17]

**512→520的教学时间线。** 使用0-based逻辑位置、fresh/无共享前缀、`protected_len=0`，prefill结束counter为7。表中区间两端均包含：

| 时刻 | prefix HP | QP中段 | recent HP |
| --- | --- | --- | --- |
| prefill，S=512 | 0..63：64条 | 64..255：192条 | 256..511：256条 |
| 前7步加入512..518，S=519 | 0..63 | 64..255 | 256..518：263条 |
| 第8步先flush，再加入519，S=520 | 0..63 | 64..263：200条 | 264..519：256条 |

plan每个lane $j\in[0,7]$ 选择

$$f_j=S_{\rm old}-256-(8-1)+j,$$

并检查 $f_j\ge\text{protected\_len}$、$f_j\ge0$ 与旧映射为HP。本例旧S=519，恰选256..263。ring此时可绕回旧token256所在物理槽，**先量化旧行，再由forward写新行**保持未移出值完整；QP/HP的逻辑身份由remap与新映射更新。最后 $520=64+200+256$，数量守恒与位置守恒都成立。这些是按源码次序构造的教学推演，不是实际请求运行记录。[10:315–341][12:400–413,660–820][16:110–134]

## 6. Decode 分层读取与尺度精度

decode metadata扫描完整逻辑历史，按 `slot>=O` 区分HP，分别生成两套 `indptr/indices`；各tier保持自己的逻辑次序，QP ID不变，HP ID减O。HP长度取实际映射数量，不假定恒为64+256。例如逻辑IDs `[O+0,5,7,O+3]` 变成HP本地 `[0,3]` 与QP `[5,7]`。[17:40–151,442–523]

当前 Q通过独立 BF16 `Q@Rk` 进入相同坐标。mixed wrapper将QP byte/scale buffers、HP BF16 buffers、两套索引、split caps送入device；这里明确选择无额外logit cap/temperature/sinks bias的分支。Attention API中的 `sinks`额外logit要求None，**与仍开启的缓存prefix sink64不同**。HP split cap默认8也不能解释为flush8，两者管理不同对象。[5][17:1750–1800][18:2370–2522]

以H=8、D=128的QP物理地址为例，slot t、KV head h、四分区内通道 $\delta\in[0,31]$：

$$\text{byte offset}=256t+32h+\delta,$$
$$\text{metadata scalar offsets}=16t+2h+[0,1].$$

解出四个2-bit码后，K按与Q相同的四分区重排进入dot；V的四路PV累积按原四分区写回。这里不把源码join后的临时排列当成自然通道序。GQA比率4时，每个Q head使用其共享KV head的同一记录；具体producer映射与这里的地址必须配套。[18:1647–1703,1858–1909]

**FP32存储不等于FP32消费。** 选定GQA reader将FP32的 s、z **先转为Q dtype BF16**，再按 $(q-z)s$恢复用于dot的值。HP直接读已旋转BF16行。这个源代码类型关系已经能说明潜在舍入位置，但没有Triton编译输出，不声明每个中间指令都在哪一步舍入或两个路径逐bit一致。[18:1654–1682,1858–1880]

FA3历史前缀的框架恢复helper则把码与s/z都转成FP32，先做 $(q-z)s$，再存为BF16前缀张量；它与decode先转换s/z的消费路径不同。这项差别发生在FA3调用前，不能因底层码/元数据相同就假设恢复值相同。[19:199–239]

## 7. 两种精度的局部结果怎样成为一次 Attention

backend分配FP32局部输出 `[batch,Hq,HPcap+QPcap,128]` 与FP32局部LSE，wrapper接收这些scratch；每次先把全部LSE重置为 $-\infty$，空split虽可不写输出，却不会让旧scratch参与合并。实际Attention kernel调用为 **HP stage1、QP stage1、共同stage2三次**，另有Q旋转等操作；“unified”不表示三者同一次launch。[17:693–702][18:2370–2522]

每tier内按索引分段；真实长度为n、split数k时，每段跨度向32对齐：$32\lceil\lceil n/k\rceil/32\rceil$，最后按n截断。GQA的HP与QP stage1都用FP32 dot accumulator与online softmax状态，但**softmax概率先转BF16再做PV dot**。因此“softmax FP32”不意味着PV的两边输入保持FP32。局部输出为归一化 $O_s=a_s/l_s$，局部LSE为 $\lambda_s=m_s+\log l_s$，两者写FP32scratch。[18:294–554,1363–1940]

共同stage2跳过LSE为 $-\infty$ 的split，按稳定FP32递推合并两tier所有局部结果：

$$\Lambda=\operatorname{logsumexp}_s\lambda_s,\qquad O'=\sum_s e^{\lambda_s-\Lambda}O_s.$$

它需要用局部分母恢复全历史的softmax质量，**不能将HP与QP各自归一化输出直接相加**。这个公式的共同推导见[在线 softmax 与分段 Attention](sglang-attention-operator-design.md#4.%20Decode%EF%BC%9A%E6%8A%8A%E5%8E%86%E5%8F%B2%E7%BB%B4%E6%8B%86%E5%BC%80%EF%BC%8C%E5%86%8D%E6%8C%89%20LSE%20%E5%90%88%E5%B9%B6)。此处wrapper不要求最终LSE输出；stage2先将结果存成BF16，再由backend执行独立BF16 `O'@Rv.T` 恢复Value坐标并返回模型输出dtype。V absorption仍保留这一输出逆旋转。[18:2265–2367][17:1848–1856]

## 8. 来源定位与未覆盖范围

正文的 `[n]` 对应frontmatter第n个文件；runtime代码路径以下均相对 `sglang-research/python/sglang/`：

| 主张 | 固定源码落点 |
| --- | --- |
| 实验recipe/默认值/接入gate | `rotation/eval_oscar_gpqa.sh:83–118`；`srt/environ.py:214–239`；`srt/server_args.py:2586–2666`；`pool_configurator.py:33–59`；`model_runner_kv_cache_mixin.py:497–605,703–725` |
| 旋转身份与V折叠 | `memory_pool.py:136–181`；`unified_kv_pool.py:247–270,565–600`；`srt/models/utils.py:500–509,670–731`；`qwen3.py:665–667` |
| 码/裁剪/metadata | `QuantKernel/oscar_rotation_clip_int2_kv.py:92–198,333–361,640–774`；`QuantKernel/gpu_flush_int2.py:185–300` |
| arena/slot/window | `unified_kv_pool.py:104–120,185–241,396–559`；`unified_kv_allocator.py:163–351`；`common.py:340–383,414–820` |
| prefill与decode消费者 | `quantized_kv_prefill.py:107–151,291–347`；`flashattention_backend.py:687–857`；`triton_backend.py:1694–1859`；`triton_ops/decode_attention.py:294–554,1363–1940,2265–2522` |

本轮沿选定affine INT2、BF16/GQA/D128/G128、普通prefill writer、recent flush与mixed Triton decode核读。相邻的Lloyd-Max、多组/其他D、FP8输入、无mixed、其他Attention backend、Radix前缀复用、chunked prefill完整生命周期、graph与多设备没有纳入完整核查；FA3的框架拼接已核，库内device代码没有沿读。RotationZoo数值张量与实际模型、运行误差/容量/性能也未验证。

WUSH-KV仍需匹配其作者实现，核对每KV head一般可逆T、Query的逆转置补偿、V/O折叠及其存储更新；本页只能补共同量化器/窗口/读取接口的公开前置证据，不能关闭该缺口。

## 来源身份

| 来源 | 固定版本 | 使用范围 |
| --- | --- | --- |
| OSCAR: Offline Spectral Covariance-Aware Rotation for 2-bit KV Cache Quantization — 官方代码（`https://github.com/FutureMLS-Lab/OSCAR/tree/41ebcdba3db5f0ce1339c3727caea80df575d437`） | `41ebcdba3db5f0ce1339c3727caea80df575d437`，提交日期2026-06-28，归档日期2026-10-08 | README、离线旋转/评测配置及vendored SGLang中的固定writer/window/Attention；使用文件与位置见正文及frontmatter。未以整仓归档替代整仓研读，也未据其作者测速报告本地性能 |
