# 模型与缓存：把问题放回计算图

用于模型差异、输入构造和跨轮状态。先确认模型结构与当前阶段，再判断所说的权重、激活或缓存究竟是哪一个对象。

| 要回答的问题 | 正文入口 | 可补查的词 |
| --- | --- | --- |
| token 怎样进入前向并生成下一个 token | [自回归主干与模型差异](/Users/xiao/Documents/Projects/quantwiki/wiki/fundamentals/model/transformer-autoregressive-inference.md) | prefill、decode、GQA、QK-Norm |
| MoE 的选择、专家和汇合如何分工 | [专家路由](/Users/xiao/Documents/Projects/quantwiki/wiki/fundamentals/model/mixture-of-experts-and-routing.md) | router、top-k、共享专家、激活参数 |
| MLA 实际保存什么 | [联合潜变量与位置分支](/Users/xiao/Documents/Projects/quantwiki/wiki/fundamentals/model/multi-head-latent-attention.md) | latent、投影吸收、位置 key |
| 图像怎样进入语言层，量化范围在哪里 | [视觉输入与量化对象](/Users/xiao/Documents/Projects/quantwiki/wiki/fundamentals/model/vision-language-model-tokens-and-quantization.md) | projector、merger、视觉 token、模态 |
| KV 的粒度、生命周期与误差怎样连接 | [KV 对象与粒度](/Users/xiao/Documents/Projects/quantwiki/wiki/theory/kv-cache-quantization-objects-and-granularity.md)、[Attention 误差传播](/Users/xiao/Documents/Projects/quantwiki/wiki/theory/attention-quantization-error-propagation.md) | K/V、量化窗口、scale、query 加权 |
| reasoning、蒸馏与数值量化有什么区别 | [推理模型后训练](/Users/xiao/Documents/Projects/quantwiki/wiki/fundamentals/model/reasoning-post-training-and-distillation.md) | GRPO、Distill、模式、输出预算 |

结构先查 `fundamentals/model/`，具体模型的调用、缓存更新和输入准备查 `implementation/`。VLA、DiT、TTS 或视觉剪枝等未列专题，可用模型名、任务名及机制词在 `methods/`、`fundamentals/model/` 定位，不把这张表当作支持范围。分页分配或调度问题再转部署 reference。
