---
title: VLA的CPU动作块执行：供给窗口、指令缓存与W8A8
type: implementation
tags:
  - vlm
  - inference
  - deployment
  - mixed-precision
sources:
  - raw/papers/2026-09-28/vla-simd
  - raw/repositories/2026-09-28/vla-simd
updated: 2026-10-15
---

# VLA的CPU动作块执行：供给窗口、指令缓存与W8A8

vla.simd将相机帧、机器人状态与语言指令变成机器人单位的动作块，由外部控制客户端逐步消费。动作块让一次CPU查询供给多个控制周期；它同时带来观测陈旧、窗口对齐与下一块何时到达的问题。IMPACT则把不随观测变化的语言计算缓存下来，减少重复查询成本。

本页依据vla.simd v1论文与固定代码 `7354b6d73b3686395d709202dbb519fd2e101aa0`，重点核读IMPACT、转换/加载、所选linear量化与服务主线。没有编译或运行策略。通用整数缩放复用[量化矩阵乘执行](quantized-matmul-scaling-execution.md)，动作压缩与闭环评测对照见[FoldQuantVLA](../methods/foldquantvla.md)。

## 1. 动作窗口覆盖查询延迟，仍不等于高频反馈

令控制器频率为 $f_c$，查询耗时 $T_q$，延迟折算为
$d=\lceil f_cT_q\rceil$个控制步。策略预测 $N$步，但可执行窗口是 $H\le N$；例如论文Diffusion预测64步，只开放32步窗口。

在固定延迟、稳态、顺序查询模型中，两次查询启动相隔 $s\ge d$步：

| 执行方式 | 下一块到达前如何不断供 | 存在可行查询间隔的条件 |
| --- | --- | --- |
| 到达后从块首执行，lagged | 当前块要覆盖间隔：$s\le H$ | $d\le H$ |
| 按原观测时刻对齐并丢弃过期动作 | 到达时只剩 $H-d$步：$s+d\le H$ | $2d\le H$ |

两者均可取 $s=d$。$H/T_q$是动作供给率，不是新观测反馈频率；第二行忽略控制步取整后，才近似要求供给率至少 $2f_c$。这些条件保证该模型内有动作可执行，不保证动作平滑或任务成功；实际捕获/传输与查询尾延迟仍会消耗窗口。（论文§III、式1。）

## 2. 缓存指令，重新计算当前观测

IMPACT有三个计算寿命：权重按进程加载/pack；指令不变时复用语言结果；每次query重新处理图像和状态。

指令改变时，T5-small编码最多32个有效文本token。投影后的512宽文本向量进入后续encoder；非padding隐状态的均值还生成各ResNet stage的FiLM系数
$[\gamma_i;\beta_i]=G_i\bar e+b_i$。
两者按指令缓存。每次query仍对当前两幅RGB归一化、重新跑共享ResNet-18，在stage残差相加和ReLU之后施加
$F_i\leftarrow(1+\gamma_i)\odot F_i+\beta_i$。
图像依赖的特征没有跨观测缓存。（`impact_model.cpp:set_instruction/encode_view/predict`、`t5_text.cpp`。）

随后将视觉特征、归一化状态、latent和有效文本组成encoder序列。推理的latent设零，其投影在转换时折成常量token；六层encoder、四层decoder以50个learned action queries一次产生50步动作，再按训练动作均值/标准差恢复单位。视觉位置为二维sinusoidal，state/latent/text位置另外提供给Q/K。（论文§V；`act_transformer.cpp:build_tokens/encode/decode`。）

SO-101用两幅480×640图，产生600视觉token、六维动作；LIBERO用两幅256×256图，产生128视觉token、八维state与七维动作。相机顺序、训练normalizer和配置要匹配，不能只凭同为IMPACT互换输入。

训练沿ACT的CVAE：额外编码器从示范动作、状态和指令产生32维latent分布，以L1动作重构加权重10的KL先验惩罚训练；这条训练编码分支与推理 $z=0$分开。论文记每query策略60M参数，冻结文本编码器另35.3M；缓存减少重复计算，没有删除后者的权重或初始化成本。

## 3. 导出、加载打包与查询量化各做什么

`convert_impact.py`把训练checkpoint转换为后端布局：BN折入卷积，卷积权重改成 `[Cout,k,k,Cin]`，拆分QKV，并保存配置、tokenizer、normalizer与各权重arena。该转换器限制ResNet-18、post-norm/ReLU、匹配的T5与单线性FiLM；输出不能当任意ACT变体通用格式。

自有GGUF把配置和typed byte arenas封装起来，loader再交给各模块。该IMPACT导出路径默认浮点权重，完整identity词表映射也没有进行词表剪枝；文件扩展名本身不能证明已存INT8或缩减词表。（`tools/_gguf.py`、`src/io/gguf_vla_simd.cpp`。）

选定linear在**加载期**按输出通道absmax/127量化权重、K补齐到4，并pack成 `[N/16,Kp/4,16,4]` panel。需要相应ISA且输出宽度 $N$整除16。每次查询再对每个输入token动态absmax/127量化，零行scale取1；INT32点积后乘输入/权重尺度并加bias，恢复FP32。AVX-VNNI把有符号激活字节变换为unsigned并减去 $128\sum_kq_{w,k}$修正；这是机器指令适配，不是学习了另一套量化零点。（`Linear::init_int8/forward`、`quant_ops.h`与`hal/avx2/gemm_i8.cpp`。）

`IMPACT_INT8`用六组开关选择encoder、projection、decoder、卷积等算子组；T5、norm、非线性和不支持宽度继续浮点。论文卷积另用每输入tensor的动态尺度，不能把linear逐token约定套到所有算子。

浮点主链让linear、卷积面板与attention投影复用packed-panel SIMD微核，输出tile留寄存器累计；tile受寄存器、FMA延迟与load吞吐共同约束。QK/概率乘Value另走attention内核，Apple默认又可用Accelerate。因此量化收益要对目标CPU的完整query测量，不能只根据位宽判断。

## 4. 动作交给谁，以及性能证据覆盖到哪

`policy_server.py`按checkpoint相机顺序适配HWC uint8帧与float32状态，C API返回 `[chunk,action_dim]` float32动作。RPC等待队列容量为1，旧等待观测被替换；返回动作按双方约定窗口截取，时间戳锚定观测时刻，再逐步加控制间隔。外部LeRobot客户端消费这些动作；当前静态服务器主线不能替代客户端怎样丢弃过期动作的运行证明。

作者在30Hz、被动散热Pi5连续90秒后，报告IMPACT浮点33.5 actions/s、INT8 81.2 actions/s，分别满足前述C1/C2名义预算。查询计时含预处理到动作反归一化，排除相机捕获、传输和episode初始化。四CPU量化对照也有反例：Pi5 IMPACT约2.39×，M4的Octo/SmolVLA则为0.66×/0.71×；IMPACT在M4只量卷积比全量化更快。（论文§VI、TableII/IV。）

数值parity、模拟任务和机器人试验各有不同范围：FP32少量观测的参考对照不验证INT8等价；LIBERO 76.4%是GPU host的独立任务评估；SO-101每行20次且IMPACT合并CPU/GPU服务器，不据小样本接近宣称INT8成功率相同。CPU动作供给、完整服务往返与闭环质量分别评价。

## 来源身份

| 来源 | 固定版本 | 核读与使用范围 |
| --- | --- | --- |
| vla.simd: Efficient CPU Inference for Language-Conditioned Manipulation（`https://arxiv.org/abs/2609.24274v1`） | `arXiv:2609.24274v1` | 完整8页主文/限制、图TeX、算法1及表I–VII，关键PDF1–7视觉核读；预算、IMPACT、CPU协议分别使用。未核外部训练实现、模型或机器人运行。 |
| cair-vinuni/vla.simd（`https://github.com/cair-vinuni/vla.simd/tree/7354b6d73b3686395d709202dbb519fd2e101aa0`） | `7354b6d73b3686395d709202dbb519fd2e101aa0` | IMPACT/T5调用、ACT transformer/ResNet-FiLM主线、导出/GGUF、自选linear量化及common/AVX-VNNI、服务输入/动作时间戳。未核其他策略全链、卷积INT8设备分支、全部微核或机器人客户端。 |
