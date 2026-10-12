---
title: lm-evaluation-harness：请求、答案过滤与聚合
type: concept
tags:
  - evaluation
  - implementation
  - reproducibility
sources:
  - raw/repositories/2026-09-21/lm-evaluation-harness/source/lm_eval/tasks/mmlu_pro/_default_template_yaml
  - raw/repositories/2026-09-21/lm-evaluation-harness/source/lm_eval/tasks/mmlu_pro/_mmlu_pro.yaml
  - raw/repositories/2026-09-21/lm-evaluation-harness/source/lm_eval/tasks/mmlu_pro/utils.py
  - raw/repositories/2026-09-21/lm-evaluation-harness/source/lm_eval/evaluator.py
  - raw/repositories/2026-09-21/lm-evaluation-harness/source/lm_eval/evaluator_utils.py
  - raw/repositories/2026-09-21/lm-evaluation-harness/source/lm_eval/api/task.py
  - raw/repositories/2026-09-21/lm-evaluation-harness/source/lm_eval/api/model.py
  - raw/repositories/2026-09-21/lm-evaluation-harness/source/lm_eval/api/instance.py
  - raw/repositories/2026-09-21/lm-evaluation-harness/source/lm_eval/api/filter.py
  - raw/repositories/2026-09-21/lm-evaluation-harness/source/lm_eval/api/samplers.py
  - raw/repositories/2026-09-21/lm-evaluation-harness/source/lm_eval/api/metrics.py
  - raw/repositories/2026-09-21/lm-evaluation-harness/source/lm_eval/api/registry.py
  - raw/repositories/2026-09-21/lm-evaluation-harness/source/lm_eval/api/group.py
  - raw/repositories/2026-09-21/lm-evaluation-harness/source/lm_eval/filters/extraction.py
  - raw/repositories/2026-09-21/lm-evaluation-harness/source/lm_eval/filters/selection.py
updated: 2026-10-12
---

# lm-evaluation-harness：请求、答案过滤与聚合

本页跟踪Summer-Summer fork的固定commit `62617a8cbd7f3beddc2a9c029536e0e387976beb`，选MMLU-Pro配置和共用执行入口。任务名相同不保证分数与论文同口径；生成、抽取、逐例打分和汇总是四个不同步骤。

## 1. 从任务配置到模型请求

`simple_evaluate`解析模型/任务，设置随机种子与覆盖项；`Task.build_all_requests`构造few-shot上下文和`Instance`；`evaluate`按request_type分发给LM方法，把响应按doc_id及idx归回任务。Instance保留arguments、repeats、原始resps与filtered_resps；`TaskConfig`默认repeats=1。（`evaluator.py`、`api/task.py`、`api/instance.py`。）

| output_type | 实际请求与输出 | 下游判断 |
| --- | --- | --- |
| generate_until | context与generation_kwargs→生成字符串 | 先过滤/解析，再按任务指标计分 |
| multiple_choice | 每候选构造loglikelihood请求→(logprob,isgreedy) | 比较候选likelihood，不是生成字母再抽取 |
| loglikelihood | context/continuation→同一二元结果 | 指标由任务指定，isgreedy不是概率 |
| loglikelihood_rolling | 文档→累计loglikelihood | 词/字节计数参与PPL/BPB聚合；接口约定不代表每个backend已核 |

通用multiple_choice的acc取最大loglikelihood，acc_norm除以**Python字符串len(choice)**后再选，不是按token数归一；`acc_mutual_info`另减无上下文的候选loglikelihood。三者不能混用，也不能直接称已校准答案概率。MMLU-Pro在本快照走generate_until分支。（`ConfigurableTask.construct_requests/process_results`。）

## 2. MMLU-Pro输入与论文保持哪些差别

十四学科YAML分别按category过滤test与validation；后者用`first_n`取前五例。`format_cot_example`保留实际选项A–J，并直接拼入完整cot_content；示范的doc_to_target为空，不另拼doc.answer。测试题不附解答，只留下“Answer: Let's think step by step.”。不能把输出标签当成整段示范，也不能把修改validation顺序后的提示称同一输入。（`tasks/mmlu_pro`、`api/samplers.py`。）

默认配置为5-shot、max_gen_toks=2048、do_sample=false、temperature=0，遇“Question:”停止；CLI的gen_kwargs与few-shot参数可能覆盖，须保存**最终合并配置**。`apply_chat_template`默认false；启用时又区分整段few-shot放一个user消息与fewshot_as_multiturn的交替消息，输入token预算应由实际backend另核。（`simple_evaluate`、`ConfigurableTask.fewshot_context`。）

配置只写`TIGER-Lab/MMLU-Pro`与split，没有固定dataset revision；本仓库commit和任务metadata版本2.1/组版本2.0也不等于数据版本。需要另外固定数据内容、示范顺序、模型/tokenizer/chat模板和执行参数。[MMLU-Pro论文](mmlu-pro-options-and-answer-extraction.md)主表还有Gemini的0-shot例外，本快照不会仅凭模型名自动切为该协议。

## 3. 两个“取第一个”发生在不同层次

`custom-extract`只有`RegexFilter`的 `answer is \(?([ABCDEFGHIJ])\)?`，然后`TakeFirstFilter`。Regex用默认大小写敏感的`re.compile`、findall第0个匹配；没有匹配返回`[invalid]`。TakeFirst再选**重复生成中的第一个响应**，不是在一段文字内找最后答案。secondary Answer: regex仅为注释，没有论文的随机选择兜底。（`filters/extraction.py`、`filters/selection.py`。）

| 合成响应 | 固定代码过滤结果 |
| --- | --- |
| The answer is (B) | B |
| Answer is (B)／answer is (b)／Answer: B | [invalid]，大写前缀、小写字母或secondary格式未匹配 |
| answer is (A). Final: answer is (B) | A，一段内选首个匹配 |
| 两次响应分别给A、B | A，过滤后再选首个响应 |
| 四选项题生成answer is (J) | J；regex未核该题实际选项范围，随后与gold比较为错 |

generate_until的process_results把过滤后的字母与doc.answer交给exact_match，指标才执行ignore_case/ignore_punctuation。它不能补救更早的正则漏匹配；`[invalid]`与合法gold不等，仍以零分进入均值。[公共失败分母](model-quality-evaluation.md#3.1%20%E5%A4%B1%E8%B4%A5%E8%AE%A1%E9%9B%B6%E3%80%81%E9%9A%8F%E6%9C%BA%E8%A1%A5%E5%81%BF%E5%92%8C%E6%9C%89%E6%95%88%E6%A0%B7%E6%9C%AC%E8%AE%A1%E5%88%86%E6%9C%89%E4%B8%8D%E5%90%8C%E5%88%86%E6%AF%8D)解释为何不能与论文随机兜底或有效样本准确率混比。这些行是从固定原函数抽取的合成夹具结果，不是模型输出试验。

## 4. 学科均值与总分怎样聚合

`TaskOutput.calculate_aggregate_metric`按(metric,filter)聚合每例结果，键如`exact_match,custom-extract`。MMLU-Pro学科内是mean；组配置`weight_by_size=true`，`consolidate_group_results`按各学科的**实际指标样本数**加权，调用aggregate_subtask_metrics。教学例：100例准确率.9、10例准确率0，总分90/110≈81.8%；学科等权平均可能得到不同结果（此两科例为45%）。

按样本加权能复原可加和的逐例均值，不自动适用于PPL、F1等指标。改变limit/samples或缺失学科也会改变分母；必须保存各科样本数和过滤器键，不能把小样本debug结果冒充全量评测。stderr或bootstrap结果也不是另一次模型运行，更不能只凭误差条宣布质量等价，统计判据见[配对评测](model-quality-evaluation.md#8.%20%E6%97%A0%E6%98%BE%E8%91%97%E5%B7%AE%E5%BC%82%E4%B8%8D%E7%AD%89%E4%BA%8E%E6%BB%A1%E8%B6%B3%E8%B4%A8%E9%87%8F%E7%AD%89%E4%BB%B7)。

## 5. 两层缓存要随对照条件隔离

请求缓存保存已构造的Instances，键含task、shot、rank/world、chat/multiturn标志、system hash与tokenizer。它没有全面编码数据修订、模板正文、few-shot seed或generation_kwargs；改变协议后应刷新或重建，防止沿用旧prompt/生成参数。（`Task.build_all_requests`。）

响应缓存CachingLM以`hash_args(method, req.args)`索引指定数据库；key不含模型权重/量化配置身份。相同请求若让浮点与量化模型共用DB，可直接读到先前输出。隔离夹具用dict和stub LM核到第二模型未被调用；这是缓存主链验证，不是实际数据库/模型评测。`do_sample=true`读取时绕过缓存，源码随后仍可写入该请求结果；不能仅凭采样设置推断数据库不存它。（`api/model.py`。）

量化对照宜隔离缓存，并保留参考/压缩版本、最终任务配置、数据/提示hash、实际args、resps、filtered_resps、逐例分数和有效分母。`log_samples`打开时evaluate会返回doc/target、arguments、原始/过滤响应、filter/指标及doc/prompt/target hash；模型配置身份和有效分母还须结合完整结果记录。运行任务需保存这些返回结果，供复核输入、计分和缓存是否匹配。

## 来源身份与验证范围

固定仓库：`https://github.com/Summer-Summer/lm-evaluation-harness/tree/62617a8cbd7f3beddc2a9c029536e0e387976beb`，kitty分支收录快照。静态研读MMLU-Pro默认/组配置、十四学科YAML、prompt/category工具、simple_evaluate/evaluate、TaskConfig及所选构造/结果/少样本路径、LM接口/CachingLM、Instance、过滤器、metric注册/exact_match与组聚合入口。对归档SHA及上述29个选定文件核字节；13个Python文件AST语法解析，原函数隔离夹具验证提示/请求类型、八组抽取/计分、加权聚合与响应缓存。没有导入/运行完整框架、下载数据/模型或运行真实backend/API/GPU/量化评测；其余任务、全部后端、滚动分词、分布式故障与CI未核，不能代表当前EleutherAI主线。
