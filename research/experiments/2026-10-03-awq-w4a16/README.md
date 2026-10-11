# Llama-2-7B AWQ W4A16 / g128 验证

本任务执行已确认的首个 Dense LLM 量化闭环：浮点基线、AWQ 搜索、模拟量化、TinyChat v2 打包与重载、独立语料质量及 GPU 性能。实验在 A00 的 `/data/xiao/wiki-quant-lab/awq-w4a16` 执行；本目录用于保存实验代码、配置和最终确认的结果。当前记录的进度与验证范围见[状态](#状态)。模型权重与编译产物保留在服务器。

## 配置与比较范围

- 模型：已有本地 Llama-2-7B，FP16；模型文件哈希随结果保存。
- 来源实现：官方 `mit-han-lab/llm-awq` commit `d6e797a42b9ef7778de8ee2352116e0f48a78d61`，从本库的原始快照复制。仅接入 Transformers 5 的 decoder 返回值接口；数学搜索、量化规则、打包与 CUDA 内核保留原实现。
- 量化：W4A16、非对称 zero-point、group 128；保留 embedding、norm 与 lm_head 的 FP16。
- 校准：本地 Pile validation，seed 42、流式 shuffle buffer 10,000，取 128 条非空且不超过 512 tokens 的文本，拼成完整 512-token blocks，丢弃尾部。文档身份和最终 token blocks 随结果固定；这不等于原仓库全数据集 shuffle。
- 质量：独立 WikiText-2 raw test，全量完整 2048-token blocks；每块重置上下文，next-token NLL / PPL。这是语言建模质量检查，不能替代下游任务与可靠性评测。
- 后端：Transformers 5.17.0 的同一 Llama/SDPA 外围计算，INT4 线性层由官方 WQLinear 的 GEMV/GEMM CUDA 路径执行。不是论文中的完整 TinyChat 服务，也未验证 vLLM/SGLang。
- 性能：单 A100 80GB，batch 1 / 4，输入 512 / 2048 tokens，固定 64 个 decode steps，teacher-forced 相同后续输入；预热后同步计时并保存重复结果。分别报告 prefill、decode、PyTorch 活跃及保留显存。

## 依赖选择

保持服务器原有项目 Conda、PyTorch 2.11.0+cu128、Transformers 5.17.0 和 Triton 3.6.0。官方 PyPI 元数据核对发现：LLM Compressor 0.14.0 要求 compressed-tensors 0.19.0，而匹配 Torch 2.11 的 vLLM 0.26.0 要求 0.17.0；较早 Compressor 又要求更早的 Transformers/Accelerate。因此本次编译并直接加载所需的官方 AWQ 两个内核，不安装冲突依赖或降级环境。

读取已有 Pile `.zst` 数据时发现缺少解码库，仅补装 `zstandard==0.25.0`；`pip check` 通过。安装前后的完整清单随结果保留。

构建入口只绑定 WQLinear 实际调用的两个官方函数，不编译未使用的 attention、W8A8 和 norm 内核。构建与接口适配都保留在本任务代码中，不修改来源快照。

数值检查的预先界限：整数码打包往返及 checkpoint 重载必须逐元素相等；代表性线性层相对 RMSE 小于 0.005；FP16 仅缩放变换的模型 logits 相对 RMSE 小于 0.005；模拟与真实后端的模型 logits 相对 RMSE 小于 0.02。保留实际误差及 top-1 一致率，不把近似相等写成 bit-exact。质量与性能直接报告同协议差异，本次没有用户指定的 PPL 或加速率门槛。

## 执行入口

在服务器实验目录依次执行以下阶段。首个阶段固定输入，其后的阶段复用该文件；输出分阶段保存。重跑应复制任务代码和上游快照到新记录目录，创建 `logs/`、`results/` 后运行，不覆盖已被 Wiki 引用的固定记录。

```bash
bash run.sh prepare
bash run.sh kernel_smoke
bash run.sh baseline
bash run.sh calibrate
bash run.sh simulate
bash run.sh export
bash run.sh packed
bash run.sh performance_fp16
```

`run.sh` 限定现有项目环境、GPU 0 和 SM80。`awq_runtime.py` 是编译及 HF5 返回值适配入口；`binding.cpp` 只绑定两个官方 CUDA 函数；`experiment.py` 执行模型与数据检查、校准、打包往返、重载、质量及性能验证。

## 状态

正在建立基线与编译。运行结果完成后在本节固定实际指标、失败与验证边界；目前不能据此声称模型或性能已经通过。
