# 部署与推理系统：区分产物、调度与执行

用于格式接入、框架职责及运行路径。先确定问题发生在校准/导出、加载/编译、请求调度还是算子执行，再选择对应正文。

| 要回答的问题 | 正文入口 | 可补查的词 |
| --- | --- | --- |
| 后端怎样接收量化产物，MLC 编译有什么作用 | [部署框架与后端](/Users/xiao/Documents/Projects/quantwiki/wiki/implementation/quantized-llm-deployment-backends.md) | MLC、model_lib、JIT、TensorRT、GGUF |
| AWQ 参数怎样打包、加载并被内核消费 | [AWQ 实现](/Users/xiao/Documents/Projects/quantwiki/wiki/implementation/awq-implementation.md) | checkpoint、repack、Marlin、TinyChat |
| 为什么分页与批处理会影响量化方案 | [服务内存与批处理](/Users/xiao/Documents/Projects/quantwiki/wiki/implementation/serving-memory-and-batching.md) | page、抢占、前缀共享、连续批处理 |
| 想先跟清一条紧凑的推理执行链 | [nano-vllm](/Users/xiao/Documents/Projects/quantwiki/wiki/implementation/nano-vllm-inference-execution.md) | Sequence、BlockManager、runner、KV 进度 |
| 当前问题涉及 vLLM 或 SGLang 的执行组织 | [vLLM](/Users/xiao/Documents/Projects/quantwiki/wiki/implementation/vllm-inference-execution.md)、[SGLang](/Users/xiao/Documents/Projects/quantwiki/wiki/implementation/sglang-inference-execution.md) | token 调度、Radix、slot、批次 |
| 多卡时参数和量化元数据怎样对应 | [张量并行与量化](/Users/xiao/Documents/Projects/quantwiki/wiki/implementation/tensor-parallel-quantization.md) | TP、分片、scale、zero、g_idx |

具体实现优先在 `implementation/` 用框架名、格式或组件名查找。MoE 通信补查 `expert-parallel`、`DeepEP`；MLA 分片补查 `context-parallel`、`DCP`。这是查找线索，不保证任一配置被支持；从正文保留版本和未核分支。只有问题追到线程、内存或指令执行时，才补 GPU 算子 reference。
