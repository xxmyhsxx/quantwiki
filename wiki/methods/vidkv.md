---
title: VidKV：视频 KV 的混合位宽、FFT 与三值保护
type: method
tags:
  - kv-cache
  - vlm
  - mixed-precision
sources:
  - raw/papers/2026-09-21/plug-and-play-kv-quantization/paper.pdf
updated: 2026-10-12
---

# VidKV：视频 KV 的混合位宽、FFT 与三值保护

VidKV v2对视频LLM的K按通道分配一/两位精度，对V采用三值码并选择部分视觉token升为两位；指定K配置还在普通通道上使用FFT。它保留token，免训练表示不微调模型；选择阈值仍使用任务反馈。论文提供质量表及缓存伪代码，尚不足以恢复唯一的物理格式与执行协议。

## 1. 保护轴与量化粒度是两件事

§3中输入是系统、视觉、文字token的拼接，K=XW_K、V=XW_V；算法1量化完整prefix的整组部分，剩余不足一组留浮点。不能沿用[CalibQuant](calibquant.md)“只压视觉KV”的对象。作者宣称K/V均按通道量化；**K混合精度沿通道选择，V保护沿token选择**，后者不等于V按token取量化范围。

以下将单head缓存记为 $K,V\in\mathbb R^{l\times d}$：l是完整prefix的token数，l_v/l_t是其中视觉/文字token数，d是head通道数；STP另用n表示保护token数量。

式4的普通分组整数码为 $q=\operatorname{clamp}(\operatorname{round}((x-z)/s),0,2^b-1)$，$z=\min x$、$s=(\max x-\min x)/(2^b-1)$，恢复sq+z。实验G=32、浮点残差长度R=128，常量范围、round规则及head/group展开须由实现补足。

粒度结论有原件冲突：表1在LLaVA-OV两位下，K/V均按通道的VideoDC/MovieChat为3.03/47.68，K通道/V按token为3.00/43.63；但K1.5/V1.58行反为C/T=2.79/47.08、C/C=2.21/13.76。表2把宣称C/C的主方案列为2.79/47.08，与前一C/T行相同。不能替作者修成笔误，也不能据此断言低于两位时V按通道普遍更优；[对象与粒度](../theory/kv-cache-quantization-objects-and-granularity.md)给出比较框架。

## 2. K：先选通道，再确定普通通道的编码

§4.2对每通道跨token统计range=max−min，选择比例k的较大范围通道用两位，其余用一位，名义平均 $b_K=2k+(1-k)=1+k$。k=.5/.75对应1.5/1.75。表1固定V两位时，range策略为VideoDC2.95/MovieChat48.28，variance为2.71/45.11、outlier计数为2.51/32.87；这是所列配置的对照，不是range的普遍最优证明。原outlier定义统计 $K_i>M\bar K$（M=3），不等于绝对幅度阈值。算法3的flatten、选择轴、并列及mask保存尚未完整说明。

普通通道的FFT只用于k=.5/.75；其他配置按实验段和算法3使用普通一位量化。作者图5展示一例量化误差2.39→1.14，并称FFT额外开销小于5%，未给足负载/计时协议，不能直接外推实时吞吐。

式5印作 $Q=\operatorname{sign}(\operatorname{FFT}(X))+1\in\{0,1\}$，$\widehat X=\operatorname{IFFT}(Qs_{fft}+z_{fft})$，$s_{fft}=\operatorname{mean}|\operatorname{FFT}(X)|$、$z_{fft}=0$。按通常实数sign，正负值加1得到0/2而非0/1，零得到1；复数sign的码义也未定义。若教学中把符号码改成c∈{0,1}，对称恢复应是(2c−1)s，这**不是**原式的字面实现。

算法3却写FFT后reshape为[l,2d]再GQuant(bit=1)，没有说明实/虚拆分、FFT轴/归一化、共轭约束与逆变换，也没有与式5衔接。因此“频域一位”尚不能直接推出每原始实数仅存一位。若每普通复数系数的实部/虚部各用一位，且系数数量不变，普通路径便需两位，混合K主体为 $2k+2(1-k)=2$；这是**条件预算**，紧凑实FFT或共轭布局可能改变结果，须有实际格式。沿token轴的FFT追加还涉及组边界/重变换，沿通道轴则涉及坐标恢复；不能从图示补出可直接供attention读取的布局。

## 3. V：三值阈值与语义保护怎样配合

式6为 $t=\operatorname{sgn}(V)\mathbf1_{|V|>\alpha}$，$\alpha=\gamma\operatorname{mean}|V|$，实验γ=.7；等于阈值时归零。它先定义{-1,0,1}码，不足以定义恢复幅值。算法2在[channel]轴取mean，对V[l,d]是每token统计，与正文所称按通道不一致；s只用于阈值且未返回，反量化/尺度协议须补。图4的attention加减对应三值码计算，真实恢复还须处理尺度、保护分支及位置，不能据此认定已实现更快的kernel。

语义token保护STP用式7 $\mathcal I_{ij}=X_v(i)X_t(j)^\mathsf T$ 选择前n=p·l_v个视觉token改用两位，其余保留三值。图3把筛选放在视觉特征/文字查询关联处；原件未明确多文字token怎样聚合、使用哪层特征、投影/归一化、n取整及并列处理。不能把这一个点积直接等同于完整QK softmax。算法2接收外部E，返回保护/未保护分区；恢复原token顺序还须E或索引，不能将分区列表当作完整原序缓存。

“1.58”来自 $\log_2 3\approx1.585$，是三值字母表容量；逐元素固定二进制记录需要两位。五个三值打包为一个字节可到1.6位/值，但这是整理者示例，作者未登记此格式。按论文四舍五入1.58，保护比例.2的名义平均为.8×1.58+.2×2=1.664，表称1.66；用精确log₂3为1.668。若p只占视觉token，完整prefix平均还依赖视觉占比。元数据、mask、FFT分量、残差和工作缓冲都须另计，公共[非整数位宽预算](../theory/kv-cache-quantization-objects-and-granularity.md#111-非整数位宽要先确定字母表和坐标表示)承接物理边界。

## 4. 缓存更新会重新影响旧K

附录A.1算法1的字面流程如下；它解释量化时刻，不能代替可执行实现。

1. Prefill令r=l mod G，前l−r个K/V量化，尾r个保留浮点；V按p选择STP或三值。返回给本次prefill的是原始K/V，压缩状态供后续decode。
2. Decode追加当前浮点K/V到残差，恢复旧K并拼接残差，本步attention先读取完整历史，再检查残差长度。
3. V残差满R时，新的R个V直接作1.58Quant并追加到旧码，清空浮点V；没有在这个分支调用STP，不能声称新生成token均有语义保护。
4. 同时K调用MixQuant(X_K)，其中X_K包含恢复旧K和新残差；按字面会重新量化**全部K**，不只是追加编码新组。选择/范围/FFT可能随之更新，旧误差也可能累积；实际是否如此须核作者实现。

算法还有Q_x/x_Q等变量、K_g/K_q笔误；输出用完整A_w同时乘已量化V和残差V，未给分段权重/反量化；attention缩放在式3出现而伪代码略去。保留这些接口缺口，不能将其当成已经运行失败的代码。[缓存生命周期](../theory/kv-cache-quantization-objects-and-granularity.md#12-缓存的生命周期会改变数值路径)解释为什么相同最终位宽不足以保证相同数值路径。

## 5. 质量证据须带任务与配置

§5/A.2报告LLaVA-OneVision-7B用8张RTX4090、至多32帧，Qwen2.5-VL-7B用8张A6000、至多16帧，基于Transformers和lmms-eval但缺少匹配版本、完整生成/样本/随机种子。VideoDC、VideoChat-GPT与WorldQA使用judge分数；MovieChat列judge和accuracy；TempCompass仅caption分支、WorldQA仅文字生成；VATEX使用BLEU-4/METEOR/ROUGE-L/CIDEr。它们不是同一尺度的“准确率”，也不代表完整任务变体。

| v2所列配置 | 16bit → K1.5/V1.58 → STP p=.2、K1.5/V1.66 | 条件 |
| --- | --- | --- |
| 表2 LLaVA-OV VideoDC / VideoChat-GPT平均 | 3.01→2.79→2.89；3.27→3.06→3.10 | 平均与各子任务分开读 |
| 表2 Qwen TempCompass | 56.53→49.10→49.20 | 损失7.43/7.33个百分点，不能笼统称近无损 |
| 表4 LLaVA-OV VATEX CIDEr | 27.42→20.09→23.06 | 相对下降26.73%/15.90% |
| 表4 Qwen VATEX CIDEr | 41.87→29.76→31.72 | 相对下降28.92%/24.24%；保护有恢复但未回到参考 |

表3固定LLaVA K1.5/V2时，FFT让TempCompass48.98→50.45、VideoChat平均3.18→3.23；固定保护比例的STP/RTP对照有提升也有VideoDC/MovieChat judge持平。表间也有差异：LLaVA STP的CU/平均在表2是3.60/3.10、表3是3.65/3.12；Qwen FFT的VideoChat平均在表2是3.15、表5是3.21。未给配置差异解释，不把各表拼成同一次单变量实验。

附录D按benchmark表现选择γ=.7，未给独立留出协议；不微调不意味着没有评测反馈。表6纯一位方案的LLaVA MovieChat accuracy31.15低于FP16的47.87，TempCompass12.8低于49.05，优于另一一位对照也不代表质量保持。附录C的周期性token筛选、重排和剪枝是后续想法，未变成本方案。

图2报告缓存占比19%（平均名义位宽1.54），不是简单用1.54/16就能还原的完整物理账。论文没有足够打包/元数据/残差实测记录，也没有可复核的端到端延迟/吞吐协议；质量表不能证明其格式已在实际缓存中省下同等字节或更快。

## 来源身份

Plug-and-Play 1.x-Bit KV Cache Quantization for Video Large Language Models，`arXiv:2503.16257v2`，来源 `https://arxiv.org/abs/2503.16257v2`。本次研读§3–5、附录A–G，回查PDF1–10、14–18的式1–7、表1–6、图1–7和算法1–3；参考文献未逐项核审，曲线未逐点重建。未取得匹配作者实现、获取现行代码或运行模型/FFT内核/GPU；公式、伪代码与主附表差异保留为来源边界。
