---
title: CalibQuant：视觉 KV 的逐通道一位量化与分数校准
type: method
tags:
  - kv-cache
  - vlm
  - ptq
sources:
  - raw/papers/2026-09-21/calibquant/paper.pdf
updated: 2026-10-12
---

# CalibQuant：视觉 KV 的逐通道一位量化与分数校准

CalibQuant v2压缩视觉token的K/V，保留token数量；K和V都按通道作均匀整数编码，把恢复尺度移入点积，再校准量化后的attention logits。其质量证据来自指定多模态任务，吞吐证据来自H100固定内存下的最大批量；“1bit”描述视觉码的主体，不能表示整个缓存或模型只有1bit。

## 1. 两个端点怎样表示一个通道

对一个head的视觉缓存 $K,V\in\mathbb R^{n_v\times d}$，式3–5对每通道取 $\alpha_i=\min_jK_{ji}$、$\beta_i=\max_jK_{ji}$，令 $s_i=(\beta_i-\alpha_i)/(2^b-1)$，保存整数 $Z_{ji}=\operatorname{round}((K_{ji}-\alpha_i)/s_i)$，恢复 $\widehat K=Z\operatorname{diag}(s)+\mathbf1\alpha^\mathsf T$。V另有自己的范围与码。b=1时，每元素只恢复为该通道两个端点之一，端点包含极端值可能让多数中间值产生大误差；它不等于对称sign或三值码本。

这是跨所选视觉token的通道范围，不是每token一个范围。它与[KIVI](kivi.md)的K按通道、V按token不同，论文§5.5比较的V消融却是**通道范围与全局范围**，不能当成所有V按token方案的普遍否定。常量通道 $\beta=\alpha$ 的除零处理、组大小、范围更新/重新编码与量化时刻没有足够实现登记，见[KV对象、粒度和生命周期](../theory/kv-cache-quantization-objects-and-granularity.md)。

## 2. 后缩放是恢复值的代数重排

式6给出单query恒等式：

$$q\widehat K^\mathsf T=(q\odot s)Z^\mathsf T+(q\cdot\alpha)\mathbf1^\mathsf T.$$

将query缩放一次，再与融合解包的码相乘，避免先写出完整浮点视觉K；元数据仍须保存和读取。其中$q\cdot\alpha$只对视觉keys共同；混合文字归一化时不能直接略去。恒等式保持的是**同一量化K的点积**，没有恢复量化前K，也不意味着采用一位整数Tensor Core。

对V，整理者补足混合模态条件：设视觉attention权重为 $p_v$、V码为 $Z_V$，则

$$p_v\widehat V=(p_vZ_V)\odot s_V+\left(\sum_jp_{v,j}\right)\alpha_V.$$

纯视觉一行权重和为1；与保留精度的文字KV共同softmax时，视觉概率质量通常小于1，不能省掉此质量因子。文字输出要与视觉输出在同一归一化下合并，分段attention另须带归一化状态；[误差传播](../theory/attention-quantization-error-propagation.md)解释K/V两条路径。

## 3. 校准改的是 logits，不是码本

式7–8将量化logits区间 $[\gamma,\delta]$ 线性映射为 $[\gamma-\tau_1,\delta-\tau_2]$：

$$g(x)=a(x-\gamma)+\gamma-\tau_1=ax+c,\quad a=1+\frac{\tau_1-\tau_2}{\delta-\gamma},\quad c=(1-a)\gamma-\tau_1.$$

它让两个端点都向下移动。$\tau_2>\tau_1$且 $\delta-\gamma>\tau_2-\tau_1$ 时，0<a<1压缩差值；$\tau_1=\tau_2$ 时只平移；若a≤0则会抹平或反转排序，不能笼统称为“总能收缩极值”。$\delta=\gamma$亦须另有策略；极值统计须限定有效分数，不能将mask的负无穷代入。论文先写τ>0，实际搜索集合却是{0,1,2,3}，未登记退化区间与斜率约束。

若**整行**都应用同一个g，共同c被softmax消掉，只剩 $\operatorname{softmax}(az)$；τ相等时概率完全不变。若只校准视觉分数，c相对文字仍会改变模态质量。教学例：视觉/文字logits原为(0,0)，只给视觉减1，视觉概率从1/2变为 $1/(1+e)\approx0.269$。因此必须区分全行、视觉子集及极值统计范围；原件的纯视觉公式不能补出混合路径的完整mask、拼接与归约实现。公共[子集偏移条件](../theory/attention-quantization-error-propagation.md#3.1%20%E5%85%B1%E5%90%8C%E5%81%8F%E7%A7%BB%E5%8F%AA%E5%9C%A8%E5%90%8C%E4%B8%80%E5%BD%92%E4%B8%80%E5%8C%96%E5%9F%9F%E5%86%85%E6%B6%88%E5%8E%BB)承接此区别。

作者用网格选τ并跨输入、层和head固定，图1/2在InternVL2.5-8B、COCO展示logits分布与softmax MSE改善。主搜索数据划分、目标、最终τ值、样本ID及留出协议没有给足，不能把直方图对齐视作全部任务质量保证。

## 4. 融合解包与完整缓存预算

§4.3报告Triton融合解包：qK在通道轴归约，pV在token轴归约，保留query/权重条带以减少访存。附录A.1用M位容器承载M/b个码，要求M可被b整除；第i个码占 $M-b(i+1)$ 位，即高位优先。整数恢复应取 $(u\gg(M-b(i+1)))\mathbin{\&}(2^b-1)$。附录指数乘法式需整数截断才能与其文字右移一致，不能直接用实数乘法/模替代。

算法1印作start=$b_h\,pid$、end=$(b_h+1)pid$，pid=0便得到空区间；其sum axis=1也不能直接等同于所写shape的通道归约。算法2的任务边界、整数除法与尾块mask同样不完整。它们提供融合思路，不能当作已验证可执行kernel；没有取得匹配作者代码来判断实际实现是否有这些问题。

整理者预算：每head两个视觉码主体约 $2n_vdb/8$ 字节，K/V各保存s/α共4d个浮点元数据，另计文字/新生成token、padding、页表及工作缓冲。假设n_v=1024、d=128、b=1、元数据FP16，视觉K+V主体32768字节，加1024字节元数据为33792字节，相对FP16两份缓存524288字节约15.52倍；若另有128个FP16文字token，总体约5.94倍。这是所设数据格式的教学预算，论文没有登记完整物理记录，不能替代实测显存比。

## 5. 质量表与吞吐怎样解读

§5报告H100 80GB上的LLaVA-1.5和InternVL2.5，COCO Caption、DocVQA ANLS、MMBench-Video；主表未完整登记split、样本、生成/计分、模型与软件pin。VLCache称按相同总内存调配压缩率，但具体配置和误差统计仍需补齐。

| v2所列结果 | 16bit / CalibQuant 1bit | 条件与限制 |
| --- | --- | --- |
| 表1 COCO CIDEr，LLaVA-1.5-7B / InternVL2.5-8B | 1.105/1.109；1.257/1.231 | 后者低于VLCache1.236，不能称各模型均最佳 |
| 表2 DocVQA ANLS，InternVL2.5-8B / 26B | 0.9135/0.8455；0.9242/0.8894 | 损失6.80/3.48个百分点；8B低于VLCache0.8558 |
| 表3 视频Overall，InternVL2.5-8B / 26B | 1.50/1.45；1.68/1.62 | 均低于VLCache1.48/1.64；正文误用DocVQA的0.8894等数字 |
| 表4 COCO消融，InternVL2.5-8B 1bit | CIDEr完整1.231、无校准1.194、V全局范围1.200 | 通道范围并非每指标最佳：SPICE0.231<0.235、METEOR0.287<0.288 |

吞吐协议是5–30GB预算下各自取能容纳的**最大batch**，视觉长度3328/8192，decode吞吐与单请求步延迟分开。表6的26B、3328、5GB：FP16 batch1/0.086秒/11.63 tokens/s，1bit batch10/0.079秒/126.58 tokens/s，约10.88倍吞吐；30GB batch6→56为40.82→459.02，约11.24倍。正文把前组归为8B、n=3329并写9.88倍，与表/比值不符；不据此改raw或给统一加速率。

图3(a)的30GB一位曲线标batch64，附录表5对应配置却列batch80；图中部分批量注释与表不一致。附录表7/8若干行吞吐无法由Max BS/Decode Speed恢复，例如表7 FP16 5GB为1/0.078≈12.82，而列38.46。表6 FP16 5GB的“Throughput Overall (500 tokens)”0.0232可由 $1/(0.067+500\times0.086)$ 恢复，单位更像**序列/秒**；若表示tokens/s须再乘500。这是对所列数字的推断，实际计时/单位仍需作者实现核对。内存预算是否包含完整模型/临时缓冲、warmup、同步、重复次数与服务负载未说明，不把不同batch的吞吐比转成同batch或单请求收益。

## 来源身份

CalibQuant: 1-Bit KV Cache Quantization for Multimodal LLMs，`arXiv:2502.14882v2`，来源 `https://arxiv.org/abs/2502.14882v2`。本次研读正文§3–5与附录A/A.1，回查PDF1–9、11–12的式1–8、表1–8、图1–3及算法1–2；参考文献未逐项审核，曲线未逐点重建。来源中的注释/iffalse草稿不作为发表正文；没有取得匹配实现、运行模型、Triton或GPU。
