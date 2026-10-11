# raw 完整性与保存规范审查

审查日期：2026-10-05。前部保留首次只读审查的快照与建议，当时尚未修改资料；后续按用户授权修订规范、修复确定问题和重录公式的实际结果见[规范修订与修复复核](#规范修订与修复复核)（更新于 2026-10-06）。前部的缺口、计数和待办不能当成修复后的现状。报告中的“通过”均限定于相应检查，不能扩展为论文结论、译文、公式或程序行为全部正确。

## 首次审查判断（2026-10-05）

raw 已建立较好的来源留存基础，尤其是论文 PDF、固定版本信息和逐页图像。问题不在于缺少一套更复杂的目录架构，而在于部分“完整性”表述覆盖了尚未验证的内容，以及可读副本、网页附件和展开源码树的实际质量存在差异。

目前不能给出“所有资料完整且准确”的总认证。已确认的问题包括代码条目的版本标识错误、网页关键图片缺失或失效、PDF 文字层中的符号损坏、源码树中的额外文件与版本混杂。部分源码仍是 iCloud 占位文件；它们没有被认定为丢失，但本轮无法对尚未取得的字节作出内容一致的结论。

建议保留现有三类目录、条目 README 和 LOG，先修影响引用的具体问题，再对规范作少量调整。无需新增一套 manifest、评分体系或归档管理框架。

## 范围与证据

审查以开始时的 265 个正式条目为基线，并补查期间新增的 CoVisco、OnPTQ 两篇论文，TensorRT Edge LLM DART 文档，以及 CoVisco、TensorRT Edge LLM 两个代码仓库，共 270 个正式条目：142 篇论文、69 个文章/文档条目、59 个代码仓库。另有一个已经登记移除、但留下空目录骨架的旧 vLLM 目录，未计为正式条目。

新增资料完成保存后才被纳入核查，没有将其他任务的写入中间态判为缺陷。工作区已有的 README、AGENTS、SELF、Wiki 和研究代码改动均被保留。检查对象是此次所见的本地资料，未逐一重新下载全部上游文件，也未证明历史快照与当时网页逐字一致。

| 检查 | 结果 | 证明范围 |
| --- | --- | --- |
| 顶层条目元数据 | 270 份 YAML 可解析，必需字段、分类、日期目录和标签形式满足当前规定；fast.cu 的 commit 标识例外 | 字段存在及内部一致性，未逐项认证所有作者、发表日期和标签的语义 |
| 身份与去重 | 正式条目 source_id 无重复；代码仓库规范化地址无重复 | 当前目录内重复检查，未遍历全部上游迁移关系 |
| 登记文件 SHA-256 | 顶层 README 对应 630 个文件核对，加上 GPTQ 评议 README 的 3 个附件；632 个一致，1 个尚未取得本地字节，没有发现已核对文件与登记值不符 | 文件与已登记校验值一致，不证明登记值取得时已忠实于上游 |
| PDF | 164 份 PDF 共 4,084 页均可解析；其中 142 份论文原件共 3,240 页 | 本地 PDF 结构、页数和读取；不是逐字、逐公式或科研结论认证 |
| 页图与主读本 | 142 篇论文及 MIT 6.441 讲义的 3,248 张页图可解码；页数、顺序、主读本资源引用一致，并做重渲染对照 | 排查明显漏页、错页和视觉内容差异，未逐个符号人工核对 |
| 其他图片 | 98 张非页图 PNG/JPEG 均可解码 | 格式和像素数据可读，不认证标注、内容或源图裁切范围 |
| TeX 原包 | 92 个 source.eprint 包可解包，合计 2,884 个文件、1,106 个 TeX 文件；补查的两篇展开副本与包内字节一致 | 包可读且有源码；未重新编译全部源码，也未逐式验证源码与 PDF 对应 |
| 父仓库来源 | 59 个正式仓库均有原始归档或保留的 Git 对象作为对照依据；对已读取的文件检查字节、类型、链接、执行位及适用的 LFS 实体 | 仍有云端占位文件未读；有 5 个已披露的跟踪文件排除项 |
| 子模块版本 | 14 个仓库登记的 57 处子模块均与父 Git tree/gitlink 的 commit 一致；54 个不同子模块基线的远端 Git tree 均未截断 | 版本关系得到重新核实；内容检查仍受文件本地可用性限制 |
| 子模块文件 | 98,864 个文件比较通过；34,743 个文件在该次比较时仍未取得本地字节 | 包含重复展开位置；不是与父归档检查相加后的独立文件数 |
| 公开评议 | GPTQ 保存的 19 条 OpenReview 记录及全部字符串正文均在 reviews.md 中找到；JSON 的 count 也是 19；3 个附件校验值一致 | 保存快照内部完整，不证明论坛现在没有新记录 |
| Wiki 引用衔接 | 现有只读检查器检查 763 个本地来源引用，未报告缺失 | 只检查引用目标存在等规则；未开展 Wiki 内容审查，不能代替版本和科学性判断 |

PDF 重渲染以统一分辨率生成缩略图进行粗略像素与比例比较，12 页超过初始差异阈值。排除 PDF 链接注释后，8 页通过阈值，余下 4 页人工检查显示同一内容的字体/渲染差异，没有确认错页。这不是所有 3,248 页的逐字人工验收。

全文文字层检查确认：142 篇论文都按页保留非空检索文字。但非空不等于准确；长度差异也不能直接判为删节，因为不同 PDF 提取器可能重复提取叠加字形、图中标签或改变阅读顺序。7 份课程 PDF 有 94 页无可提取文字，本轮未对这些页做 OCR，不能宣称课程内容全部可检索。

中文摘要抽查了 AWQ、MWOP、LLM-MQ、BiLLM、重构损失尺度/RMSE、WUSH-KV，与相应原文摘要未发现实质性的含义或作者结果归属偏移。142 篇均有中文摘要，但本轮没有逐篇认证全部译文。

## 已确认的问题

### 版本标识

[fast.cu README](../../raw/repositories/2026-09-24/fast-cu/README.md) 的 version、source_id，以及 [raw LOG](../../raw/LOG.md) 使用了 41 位字符串：

```text
记录值：2dfe5e26aecfd9e5f27bf9d5837deea01acda24b6
实际值：2dfe5e26aecfd9e5f27bf9d5837deea01acda24b
```

归档 PAX 信息指向正确的 40 位 commit；容器目录前缀也带有错误标识。已通过上游 commit 和 Git tree 重新确认实际版本，归档的 34 个文件与该版本的 Git blob 内容一致。因此这是身份登记与容器标签问题，本轮没有发现其源码丢失。应修正身份字段及说明，并记录更正依据；不必为了改一个前缀重打原包或改变原包校验值。依据为 [fast.cu 的实际 commit](https://github.com/pranjalssh/fast.cu/commit/2dfe5e26aecfd9e5f27bf9d5837deea01acda24b)。

修正 source_id/version 时还应核对依赖这些身份信息的引用，不把字段更正静默视为原记录一直正确。

### 网页图像

有 9 张声明正文中的本地路径图片实际不存在，另有 3 张 Dynamo 图片的外链已失效：

| 条目 | 确认的缺口 | 影响 |
| --- | --- | --- |
| [vLLM V1 架构](../../raw/articles/2026-09-24/vllm-v1-architecture/README.md) | 5 张图未保存：Entrypoints、两张 V1 进程架构图、LLMEngine、Class Hierarchy | 架构关系是正文的一部分，离线阅读不能重建这些关系；README 所称“保留上游 URL”实际是相对路径 |
| [PyTorch 分布式教程](../../raw/articles/2026-09-24/pytorch-distributed-tutorials/README.md) | fsdp2-tutorial.rst 的 3 张根路径图片未保存：fsdp_workflow、fsdp_sharding、fsdp_implicit | FSDP 的工作流与分片说明缺少图像支撑 |
| [NVIDIA Dynamo 架构](../../raw/articles/2026-09-24/nvidia-dynamo-architecture/README.md) | 总览 SVG 使用 file:assets/img/dynamo-architecture.svg，本地没有目标文件 | 总体架构图无法从本地资料显示 |
| 同一 Dynamo 条目 | disagg-perf-benefit、kv-routing、kvbm-agg-performance 三张图使用带签名外链；本轮三次请求均返回 HTTP 403、Request has expired | 保存的地址已不能取得图像；返回的过期时间为 2026-09-30 18:04:54 UTC |

另外，8 个条目的 Markdown 明确保留了外部图像：compute-sanitizer、floating-point、nsight-compute、nsight-systems、perplexity、ptx-mma-layout、h100-gemm-worklog、nvidia-dynamo-architecture。本轮没有验证这些条目的每一张远端图片仍可访问。上表含 RST 图像与 vLLM 相对图像，范围比 Markdown 外链扫描更广。

部分 README 已诚实披露“未下载图片”，因此不能称为隐瞒；但披露并不能使正文图像完整。尤其是架构图、性能图和位布局图，丢失后改变了资料可用于理解和核对的范围。带签名 URL 有明确有效期，不能作为长期保存图片的唯一办法。参见 [AWS 签名链接有效期说明](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html)。

### 提取文字与公式

| 信号 | 全量扫描结果 | 判断 |
| --- | --- | --- |
| U+FFFD 替换字符 | 3 个提取稿共 61 个 | 已有字形不能通过当前文字层还原，需回查原件 |
| U+FFFE/U+FFFF 非字符 | 98 个提取稿共 10,416 个 | 大量出现在行末断词处，影响搜索和复制；不能把每个都计作一个公式错误 |
| C0/DEL 控制字符，排除制表、换行和回车 | 93 个论文提取稿共 3,450 个，其中 19 份含 274 个 NUL | 字形映射或排版控制信息进入文字层，具体含义需回查；UTF-8 合法不能证明数学文字准确 |

这些集合有重叠，不能直接相加成受影响条目数。每篇的具体计数见论文附表。

U+FFFD 的位置已经定位到文件页：

| 提取稿 | 文件页及数量 | 回查情况 |
| --- | --- | --- |
| [MIT 6.441 Lecture 1](../../raw/articles/2026-09-21/mit-6441-lecture-1/article.md) | 页 3：4；页 4：8；页 5：3；页 6：5；页 7：3 | 人工回查页 3，熵公式的求和符号在原件/页图中完整，文字层损坏；共 23 个 |
| [BiLLM](../../raw/papers/2026-09-21/billm/paper.md) | 页 1：1；页 2：5；页 3：1；页 4：9；页 5：1 | 人工回查代表页的图表与公式，原件可读；共 17 个 |
| [SplitQ](../../raw/papers/2026-09-21/splitq/paper.md) | 页 7：21 | 已定位，未在本轮逐个恢复这 21 个字形 |

多数主读本已经写明“文字层用于检索，公式和表格以页图为准”，这是正确的边界。问题是已知损坏只被总括说明覆盖，具体页没有警示，而主读本名称容易让人误认为文字可以直接用于数学推导。建议在受影响页附近标明已知提取缺陷及原件位置；不要盲目删除非字符、把控制字符替换为空白，或根据上下文猜写公式。

[perplexity 提取稿](../../raw/articles/2026-09-22/perplexity/article.md) 还存在类似 `p*\theta(x_t|x*{<t})` 的公式写法，不能直接当作正确 LaTeX 使用。该条目的历史说明把异常归于源网页显示，但原始网页载体未保存；本轮无法独立确定历史源文异常与转换错误各占多少，不能凭现行网页倒推当时状态。现行来源为 [Hugging Face 困惑度文档](https://huggingface.co/docs/transformers/en/perplexity)。

### 源码树的额外文件与混杂版本

父源码树与子模块都检查文件集合后，共发现 8 个仓库有 2,553 个上游固定树之外的额外文件。许多名称带“ 2”或“ 3”；截至本次副本比较，526 个与其对应的固定版本文件一致，22 个不同，2,005 个尚未取得本地字节。文件名只能提供线索，不能作为删除依据。

| 仓库 | 额外文件 | 与固定版本一致 | 与固定版本不同 | 本次尚未取得字节 |
| --- | ---: | ---: | ---: | ---: |
| EfficientAI | 260 | 147 | 0 | 113 |
| FlexQ | 1,802 | 0 | 0 | 1,802 |
| lm-evaluation-harness | 259 | 169 | 0 | 90 |
| MQuant | 175 | 175 | 0 | 0 |
| OmniServe | 18 | 18 | 0 | 0 |
| QuaRot，包括三个子模块 | 29 | 7 | 22 | 0 |
| SGLang | 3 | 3 | 0 | 0 |
| Transformers | 7 | 7 | 0 | 0 |

[QuaRot 的 CUTLASS 子模块](../../raw/repositories/2026-09-21/quarot/README.md) 固定于 ffa34e70756b0bc744e1dfcc115b5a991a68f132，正常 README 对应 CUTLASS 3.4。额外的 README 2.md 对应 CUTLASS 4.6.0。CUTLASS 的 14 个不同额外文件还逐一与当前独立 [CUTLASS 条目](../../raw/repositories/2026-09-21/cutlass/README.md) 的 e8ecfad75b44d1ad56264f5001d877e9e47fe080 比对，全部一致。可以确认同一个子模块目录中混入了另一份选定版本的文件；不能据此断言这些文件由哪次操作或哪个同步软件产生。

这 22 个不同文件涉及 README、CHANGELOG、LICENSE、CMake、Python 项目配置和格式配置等，完整路径见附表。固定版本对应的正常路径仍通过对照；当前确认的混杂发生在额外路径，不能说固定源码已整体被新版覆盖。额外文件仍会干扰全文检索、来源定位和自动遍历，违背“source/ 不混拼不同 commit”的要求。

建议先按来源归位或保存必要的独有信息，再清理已确认副本。526 个与对应固定版本相同的文件可以进入清理候选；22 个不同文件要先处理版本归属；2,005 个未读文件不能仅按后缀删除。

### 本地可用性与核验边界

末轮全目录扫描记录 230,779 个普通文件，逻辑大小约 10.87 GB，不含内部 .git 目录。论文和文章文件均已取得本地字节；代码目录仍有 50,489 个 iCloud dataless 文件，涉及 20 个条目。这是扫描时的可用性记录，下载继续进行时数量会变化；逐文件比较表中的未读数来自各次比较时刻，不能要求它们与末轮扫描严格相加一致。

本轮已请求下载已有文件，并对取得的内容继续检查。占位文件有路径和逻辑大小，不等于已可离线读取，也不等于云端原件丢失。最后一个尚未验证的登记校验值对应 [cs336-lectures 的 lecture_10.py](../../raw/repositories/2026-09-21/cs336-lectures/README.md)。

不应把文件存在、历史 README 写过“完整”、或 raw 之外的 Git 工作区状态，当作此次字节检查已经通过。raw 不随 Wiki Git 分发是明确的项目边界；因此 Wiki 仓库的 clone/push 结果本来就不能证明 raw 已获得另一份可恢复副本。建议确保原件与固定归档在实际归档位置可读取；大规模展开树可以再按用途决定是否常驻，不在本轮自动迁移。

### 局部说明与目录残留

有 21 个文章条目、24 份 Markdown 文件存在 101 处疑似不能在本地解析的链接；含重复链接，以及 PyTorch C++ 扩展文档的一处嵌套 URL 写法异常。15 个 CUDA/CUTLASS 条目的 README 留有 nvcc.html 或 dsl_dynamic_layout.html 路径；ModelOpt、Ray、TensorRT-LLM、vLLM、Dynamo 文档保留了上游相对路径或站点根路径。完整目标清单见附表。部分条目明确披露这一限制，不应把每个断链都等同于正文缺失。

建议保存原始路径的来源语义，把导航目标解析为原站点或固定 commit 的完整地址。无需为修一个导航链接复制整个网站。涉及图像时则另按正文资源要求处理。PyTorch 嵌套 URL 的源文可能本来就有该写法，未在本轮归因于本地转换。

有 5 个上游跟踪文件在展开树中被排除：QSLaw 的 llava/.DS_Store、QuIP# 的 .quiptools.cu.swp、SqueezeLLM 的 .DS_Store，以及 TensorRT-LLM 的两个 .tmp。条目已有披露，当前没有据此认定算法实现缺失；但“清理过的源码副本”与“固定树逐项完全一致”需要区分。不要以这些例外为理由引入全局零字节删除规则。

[LLM-MQ](../../raw/papers/2026-09-21/llm-mq/README.md) 已披露来源 PDF 的交叉引用结构异常，并另存 paper-readable.pdf 作为修复后的阅读副本。原件仍被保留、作用分别说明。本轮“可解析”包含读取器可恢复读取的情况，不等于原件 PDF 内部结构全无瑕疵；这不是新发现的未披露损坏。

Disaggregated Quantization 与 FoldQuantVLA 的条目说明把子模块 commit pin 归于 .gitmodules，属于机制表述不准确。实际版本关系已在本轮通过 Git tree/gitlink 核实一致，可以局部修正说明，不应把措辞错误误判为子模块版本错误。

OmniServe 的两个 models 符号链接目标不存在，但链接本身与来源一致，目标属于另行取得的模型相关内容，不宜直接判为归档错误。已移除的 vllm-364ee36a 目录只有空骨架、0 个普通文件；LOG 已说明其旧快照移除，是可清理的残留，不是又一份有效版本。

GPTQ 的 reviews/README.md 沿用独立 article 身份，但没有 tags。公开评议当前规则要求另有来源与角色等信息，却未明确它作为附件是否继承父条目标签。建议明确附件与独立条目的元数据适用范围，再做小修；不要把上游 source/ 中自带的 README 当作本项目元数据条目。

## 来源本身的准确性

“忠实保存”与“内容正确”需要分别判断。raw 不应该把来源的论断自动升级为本地已验证事实，也不应该把上游明显错误悄悄改成整理者版本。

[Floating Point and IEEE 754 本地材料](../../raw/articles/2026-09-22/floating-point/article.md) 的 FMA 十进制例子写 x=1.0008、x²=1.00160064，却把 x²−1 写成 1.60064×10⁻⁴。直接计算为 0.00160064，即 1.60064×10⁻³。相关表格中的 rn(x²) 也与正文 1.0016 不一致。本轮核对 [NVIDIA 现行官方页面](https://docs.nvidia.com/cuda/floating-point/index.html)，仍能看到相同的矛盾，不能将其简单归因于我们的提取程序。

建议保留来源正文，在条目说明中标记“源文疑点”及具体位置、版本、依据，后续 Wiki 解释采用核实后的推导并说明归属。这样的局部算术核查属于阅读质量审查，不要求复现每篇论文的实验或证明全部结论。

## 对现有规范的判断

现有规范值得保留的部分：raw 与 Wiki 分工清楚；原件不能被提取稿替代；使用固定版本与校验值；区分作者报告与本地验证；正常上游空文件保留；子模块和 LFS 单独处理；日期未知不猜测；避免重复 manifest。它们已帮助此次定位问题，无需整体推翻。

主要建议如下。

| 现行规定或实践 | 问题 | 推荐调整 |
| --- | --- | --- |
| 每仓库只保留一个条目、一份版本，优先较新且完整者 | 工作副本管理与证据保存发生冲突；新版不能替代旧版实现证据。现有引用检查未确认旧引用已失效，但规则有丢失必要证据的风险 | 按规范化仓库地址加 commit 去重；允许保留仍被引用或有实质研究价值的版本，只有无用且确认重复的版本才移除 |
| 修复后仍不能完整保存源码主体或必要依赖，应移除条目 | 将有价值的部分来源一并删除；“源码完整”“构建依赖齐全”“可运行”被混在一起 | 有实际内容且有价值时允许保留明确范围的部分资料；列出缺失对象、所影响用途及获取条件。仅链接占位或无实际代码仍不能叫完成代码收录 |
| 网页不要求 HTML，仅保留外链时说明未本地保存 | 对有稳定上游 Markdown 的文档合理；对可变网页或失效附件，缺少独立回查依据。当前记录异常原因时就遇到了该限制 | 优先固定上游正文源码；必要时保留最小原始 HTML/PDF。关键图片本地保存，装饰图片可外链并说明，不要求整站镜像 |
| 逐页图像完整即可辅助呈现全文 | 视觉副本完整不能推出公式文本、表格结构或文字检索准确 | 分别说明原件保存、视觉呈现、文字提取的结果及已核对范围，不使用一个“完整”包办全部意义 |
| 历史已核验完整的概括 | 文件后续可能被添加、同步成占位文件或发生版本混杂 | 写明实际对照版本、对象和日期；文件集合同时检查缺少与多出项。历史通过结果保留，但不能代替本次状态 |
| “主读本”作为统一文件角色 | 页图加文字层与经核对的数学正文质量不同 | 明示其用途及限制：逐页呈现/检索副本可以完成归档，未经核对的文字层不能宣称公式可直接复用 |

每仓库一份版本并不是当前查出的坏文件一定由此产生；它是从证据保存目标出发的设计判断。建议先改规则的意义，等有明确多版本需求再决定目录如何表达，不在这轮为所有仓库建立新版本架构。

也不建议将完整性要求反向放宽为“有文件就算完成”。部分保存要有真实材料、清楚范围和具体缺失；整个 PDF 已保存的论文，与仅有几段摘录的网页，不应使用同一种无条件完成表述。

## 建议的条目标准

### 共通信息

保留现有 source_id、kind、title、url、acquired_at、added_on、version、tags 八个字段。已知且有用的作者/机构、发表或修订信息补充即可，不要求猜填，不强制为每条来源增加大量状态字段。

正文仍回答“哪份资料、保存了什么、核验到哪里”：

1. 版本与身份能定位到确定资料，获取时间与发表/修订时间分别说明；本地整理时间不冒充来源时间。
2. 收录范围明确到全文、选定章节、具体页面、代码树或附件；明确不包含的实质部分。
3. 文件清单区分来源原件、忠实导出、文字提取和展开副本；原件/归档有校验值，已有转换或清理记录有依据。
4. 验证说明写明对照对象、范围、结果及已知缺陷；公开评议保留 forum/note 标识、时间、角色及对应投稿版本。

原件保存、阅读呈现与可复用文字分别写一句即可，不必为此建立新的多层状态系统。历史统一归档日与首次入库日不同的迁移记录，应继续保留实际说明，不能为了形式一致补一个未知日期。

### 不同材料的最低要求

| 材料 | 完成归档所需 | 不应据此宣称的结果 |
| --- | --- | --- |
| 论文 | 固定版本原件、README、声明范围的阅读/检索入口；正文、附录及来源实际取得的补充材料边界清楚 | 全部公式文字准确、译文逐字核对、实验复现 |
| 网页/文档 | 确定版本或捕获时间、声明范围的正文、必要公式/表格/代码与关键图像；保留可回查的来源依据 | 未收录的链接目标或远端附件也已离线完整保存 |
| 代码 | 固定 commit、实际主体源码、归档/Git 对照依据、子模块 pin 和已保存范围、LFS/链接/排除项说明 | 完整环境、权重、数据、构建成功或性能复现 |
| 公开评议 | 原始响应或逐条来源依据、记录标识和角色、投稿版本、导出入口和数量核对 | 论坛永久不变，或投稿版本等同于另一份 arXiv PDF |

### 公式、表格、图片和代码

**公式：** 原件应能看到完整表达，包括上下标、求和/积分范围、分式、矩阵布局、运算符、括号和式号。阅读副本能定位到文件页或原文式号。需要可复制 LaTeX 时优先利用同版本 TeX，并对实际要复用的公式回查；数学能渲染、文本非空或 UTF-8 合法均不足以证明表达一致。已知损坏在局部标记，保留原件，不猜写、不批量删除可疑字形。

**表格：** 原件中的行列关系、表头、单位、实验条件、脚注和强调信息要完整可见。转成 Markdown 表格时必须核对这些关系，尤其是跨行/跨列和多组配置。复杂表格可以保留清晰原图及定位信息，不强制错误地展平为纯文本。

**图片：** 正文依赖的结构图、结果图、位布局与算法流程图应本地保存，保持完整坐标轴、图例、标注和图注。来源本来没有独立图像文件时，页图或完整 PDF 足以作为视觉原件；不必人为拆成每张 figure。关键网页图像不能只保留临时签名链接。装饰图、图标是否本地保存按实际阅读需要决定。

**代码：** 保留来源版本的内容与格式，尤其是缩进、换行、接口名和配置；处理相对链接时不改代码语义。源码树检查既看缺少项，也看额外项；字节、类型、链接与执行位分开核对。gitlink 存在于父仓库 Git tree，.gitmodules 提供路径与 URL，不能仅靠 .gitmodules 证明 commit。参见 [Git 子模块文档](https://git-scm.com/docs/git-submodule)；LFS 指针与实体也要区分，参见 [GitHub LFS 指针说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)。

**版本：** 对 arXiv 使用包含 vN 的版本标识；不含版本的地址会随修订指向更新资料。参见 [arXiv 版本说明](https://info.arxiv.org/help/versions.html)。代码、网页附件和评议各自保留版本对应关系，不因为同属一个项目就视为同一时间状态。

我不建议为入库强制手工重录全部论文公式、拆图或重新编译全部 TeX。这样既昂贵，也会引入新的人工错误。raw 的最低责任是来源可追溯、原件可恢复、声明范围忠实，随后按实际研读与复用需求提升提取质量。

## 推荐执行顺序

1. 修正 fast.cu 的身份记录，补齐已确认缺少或失效的关键网页图像，标记已定位的提取损坏与源文疑点。这批工作直接改善引用依据。
2. 处理 8 个源码树的额外文件：先核对、归属和保留必要的不同版本信息，再清理确认副本；未取得字节的文件保持未验证，不按文件名批量删除。保留固定版本正常路径与原归档。
3. 对 raw/README.md 小幅修订版本保存、部分资料、关键网页资源和完整性表述；明确评议附件元数据规则。现有目录与八个字段基本保留。
4. 后续 ingest 实际引用某个公式、表格或代码路径时，再针对该对象核对准确性。若需要进一步认证全部译文或全部数学文字，应作为明确的新审查范围，不能把本轮机械检查当作已经完成。

本报告未执行上述修复。下一步最有价值的是完成第一批确定问题，再讨论代码多版本保存的具体需求；没有必要先重构整个 raw。

## 全量核查附表

附表逐条列出本轮覆盖范围。它们是此次审查记录，不是新增的常驻条目管理机制。计数依赖各次检查时的可用性，不能据此宣称未读取文件现在仍不可下载。

### 论文

142 篇论文的 PDF 可解析，页图可解码，页数、页序和主读本引用均一致。下表“字符信号”依次为替换字符、非字符、控制字符数量；0 不表示公式语义已获认证。TeX 文件数来自所保存源码包，未编译认证。校验列仅统计本条目顶层 README 对应的文件。

| 条目 | 版本 | PDF 页数 | TeX 原包 | 文件校验一致/登记 | 字符信号 |
| --- | --- | ---: | --- | --- | --- |
| [2026-09-21/adaround](../../raw/papers/2026-09-21/adaround/README.md) | arXiv:2004.10568v2 | 12 | 19 个 TeX 文件 | 3/3 | 0 / 125 / 30 |
| [2026-09-21/affinequant](../../raw/papers/2026-09-21/affinequant/README.md) | arXiv:2403.12544v1 | 19 | 未保存（允许） | 2/2 | 0 / 87 / 7 |
| [2026-09-21/akvq-vl](../../raw/papers/2026-09-21/akvq-vl/README.md) | arXiv:2501.15021v1 | 7 | 4 个 TeX 文件 | 3/3 | 0 / 38 / 8 |
| [2026-09-21/aqlm](../../raw/papers/2026-09-21/aqlm/README.md) | arXiv:2401.06118v4 | 20 | 24 个 TeX 文件 | 3/3 | 0 / 135 / 2 |
| [2026-09-21/atom](../../raw/papers/2026-09-21/atom/README.md) | arXiv:2310.19102v3 | 14 | 15 个 TeX 文件 | 3/3 | 0 / 180 / 0 |
| [2026-09-21/attention-is-all-you-need](../../raw/papers/2026-09-21/attention-is-all-you-need/README.md) | arXiv:1706.03762v7 | 15 | 10 个 TeX 文件 | 3/3 | 0 / 16 / 0 |
| [2026-09-21/awq](../../raw/papers/2026-09-21/awq/README.md) | arXiv:2306.00978v6 | 15 | 31 个 TeX 文件 | 3/3 | 0 / 162 / 0 |
| [2026-09-21/billm](../../raw/papers/2026-09-21/billm/README.md) | arXiv:2402.04291v2 | 20 | 未保存（允许） | 2/2 | 17 / 152 / 0 |
| [2026-09-21/brecq](../../raw/papers/2026-09-21/brecq/README.md) | arXiv:2102.05426v2 | 16 | 3 个 TeX 文件 | 3/3 | 0 / 81 / 15 |
| [2026-09-21/calibquant](../../raw/papers/2026-09-21/calibquant/README.md) | arXiv:2502.14882v2 | 12 | 2 个 TeX 文件 | 3/3 | 0 / 134 / 15 |
| [2026-09-21/cayley-stiefel](../../raw/papers/2026-09-21/cayley-stiefel/README.md) | arXiv:2002.01113v1 | 16 | 未保存（允许） | 2/2 | 0 / 42 / 8 |
| [2026-09-21/channel-wise-mixed-precision](../../raw/papers/2026-09-21/channel-wise-mixed-precision/README.md) | arXiv:2410.13056v4 | 23 | 2 个 TeX 文件 | 3/3 | 0 / 69 / 2 |
| [2026-09-21/coopq](../../raw/papers/2026-09-21/coopq/README.md) | arXiv:2509.15455v2 | 13 | 1 个 TeX 文件 | 3/3 | 0 / 235 / 8 |
| [2026-09-21/cudaforge](../../raw/papers/2026-09-21/cudaforge/README.md) | arXiv:2511.01884v2 | 33 | 9 个 TeX 文件 | 3/3 | 0 / 14 / 0 |
| [2026-09-21/deepseek-r1](../../raw/papers/2026-09-21/deepseek-r1/README.md) | arXiv:2501.12948v2 | 86 | 15 个 TeX 文件 | 3/3 | 0 / 112 / 18 |
| [2026-09-21/deepseek-v2](../../raw/papers/2026-09-21/deepseek-v2/README.md) | arXiv:2405.04434v5 | 52 | 7 个 TeX 文件 | 3/3 | 0 / 41 / 9 |
| [2026-09-21/deepseek-v3](../../raw/papers/2026-09-21/deepseek-v3/README.md) | arXiv:2412.19437v2 | 53 | 5 个 TeX 文件 | 3/3 | 0 / 93 / 10 |
| [2026-09-21/distserve](../../raw/papers/2026-09-21/distserve/README.md) | arXiv:2401.09670v3 | 18 | 12 个 TeX 文件 | 3/3 | 0 / 279 / 0 |
| [2026-09-21/fastkernels](../../raw/papers/2026-09-21/fastkernels/README.md) | arXiv:2605.23215v1 | 67 | 12 个 TeX 文件 | 3/3 | 0 / 146 / 2 |
| [2026-09-21/flashattention](../../raw/papers/2026-09-21/flashattention/README.md) | arXiv:2205.14135v2 | 34 | 32 个 TeX 文件 | 3/3 | 0 / 19 / 68 |
| [2026-09-21/flashattention-2](../../raw/papers/2026-09-21/flashattention-2/README.md) | arXiv:2307.08691v1 | 14 | 8 个 TeX 文件 | 3/3 | 0 / 4 / 4 |
| [2026-09-21/flashinfer-bench](../../raw/papers/2026-09-21/flashinfer-bench/README.md) | arXiv:2601.00227v1 | 39 | 14 个 TeX 文件 | 3/3 | 0 / 178 / 0 |
| [2026-09-21/flatquant](../../raw/papers/2026-09-21/flatquant/README.md) | arXiv:2410.09426v4 | 27 | 47 个 TeX 文件 | 3/3 | 0 / 139 / 2 |
| [2026-09-21/flexq](../../raw/papers/2026-09-21/flexq/README.md) | arXiv:2508.04405v2 | 17 | 7 个 TeX 文件 | 3/3 | 0 / 424 / 4 |
| [2026-09-21/fluxbin](../../raw/papers/2026-09-21/fluxbin/README.md) | arXiv:2608.15602v1 | 24 | 19 个 TeX 文件 | 3/3 | 0 / 327 / 46 |
| [2026-09-21/formal-transformer-algorithms](../../raw/papers/2026-09-21/formal-transformer-algorithms/README.md) | arXiv:2207.09238v1 | 16 | 1 个 TeX 文件 | 3/3 | 0 / 171 / 100 |
| [2026-09-21/fp8-formats](../../raw/papers/2026-09-21/fp8-formats/README.md) | arXiv:2209.05433v2 | 9 | 2 个 TeX 文件 | 3/3 | 0 / 6 / 0 |
| [2026-09-21/gptaq](../../raw/papers/2026-09-21/gptaq/README.md) | arXiv:2504.02692v3 | 17 | 未保存（允许） | 2/2 | 0 / 143 / 232 |
| [2026-09-21/gptq](../../raw/papers/2026-09-21/gptq/README.md) | arXiv:2210.17323v2 | 16 | 3 个 TeX 文件 | 3/3 | 0 / 52 / 4 |
| [2026-09-21/gqa](../../raw/papers/2026-09-21/gqa/README.md) | arXiv:2305.13245v3 | 7 | 9 个 TeX 文件 | 3/3 | 0 / 100 / 0 |
| [2026-09-21/helm](../../raw/papers/2026-09-21/helm/README.md) | arXiv:2211.09110v2 | 162 | 72 个 TeX 文件 | 3/3 | 0 / 360 / 151 |
| [2026-09-21/int8-gg](../../raw/papers/2026-09-21/int8-gg/README.md) | arXiv:2606.12280v2 | 11 | 1 个 TeX 文件 | 3/3 | 0 / 9 / 0 |
| [2026-09-21/integer-only-quantization](../../raw/papers/2026-09-21/integer-only-quantization/README.md) | arXiv:1712.05877v1 | 14 | 2 个 TeX 文件 | 3/3 | 0 / 202 / 2 |
| [2026-09-21/integrated-gradients](../../raw/papers/2026-09-21/integrated-gradients/README.md) | 2017 PMLR 70, pp.3319–3328 | 10 | 未保存（允许） | 2/2 | 0 / 190 / 0 |
| [2026-09-21/kernelbench](../../raw/papers/2026-09-21/kernelbench/README.md) | arXiv:2502.10517v1 | 37 | 28 个 TeX 文件 | 3/3 | 0 / 32 / 0 |
| [2026-09-21/kernelbenchx](../../raw/papers/2026-09-21/kernelbenchx/README.md) | arXiv:2605.04956v2 | 17 | 9 个 TeX 文件 | 3/3 | 0 / 19 / 0 |
| [2026-09-21/kitty](../../raw/papers/2026-09-21/kitty/README.md) | arXiv:2511.18643v1 | 12 | 8 个 TeX 文件 | 3/3 | 0 / 144 / 2 |
| [2026-09-21/kivi](../../raw/papers/2026-09-21/kivi/README.md) | arXiv:2402.02750v2 | 15 | 10 个 TeX 文件 | 3/3 | 0 / 101 / 0 |
| [2026-09-21/lc-qat](../../raw/papers/2026-09-21/lc-qat/README.md) | arXiv:2606.10531v1 | 14 | 1 个 TeX 文件 | 3/3 | 0 / 125 / 19 |
| [2026-09-21/llama-3](../../raw/papers/2026-09-21/llama-3/README.md) | arXiv:2407.21783v3 | 92 | 80 个 TeX 文件 | 3/3 | 0 / 59 / 0 |
| [2026-09-21/llm-int8](../../raw/papers/2026-09-21/llm-int8/README.md) | arXiv:2208.07339v2 | 20 | 1 个 TeX 文件 | 3/3 | 0 / 30 / 4 |
| [2026-09-21/llm-mq](../../raw/papers/2026-09-21/llm-mq/README.md) | snapshot-sha256-9a2101ae6befaff5566fc1ed1bd78cbf10ccb4a7e2c3ae2b985e9c3845562dc0 | 5 | 未保存（允许） | 3/3 | 0 / 6 / 0 |
| [2026-09-21/llm-qat](../../raw/papers/2026-09-21/llm-qat/README.md) | arXiv:2305.17888v1 | 15 | 13 个 TeX 文件 | 3/3 | 0 / 11 / 0 |
| [2026-09-21/longbench-v2](../../raw/papers/2026-09-21/longbench-v2/README.md) | arXiv:2412.15204v2 | 26 | 10 个 TeX 文件 | 3/3 | 0 / 229 / 0 |
| [2026-09-21/lora](../../raw/papers/2026-09-21/lora/README.md) | arXiv:2106.09685v2 | 26 | 3 个 TeX 文件 | 3/3 | 0 / 86 / 6 |
| [2026-09-21/luq](../../raw/papers/2026-09-21/luq/README.md) | arXiv:2509.23729v3 | 20 | 7 个 TeX 文件 | 3/3 | 0 / 43 / 0 |
| [2026-09-21/lvlm-compress-bench](../../raw/papers/2026-09-21/lvlm-compress-bench/README.md) | arXiv:2503.04982v1 | 17 | 2 个 TeX 文件 | 3/3 | 0 / 263 / 0 |
| [2026-09-21/marlin](../../raw/papers/2026-09-21/marlin/README.md) | arXiv:2408.11743v1 | 14 | 1 个 TeX 文件 | 3/3 | 0 / 174 / 5 |
| [2026-09-21/masquant](../../raw/papers/2026-09-21/masquant/README.md) | arXiv:2603.04800v1 | 10 | 9 个 TeX 文件 | 3/3 | 0 / 146 / 20 |
| [2026-09-21/mbq](../../raw/papers/2026-09-21/mbq/README.md) | arXiv:2412.19509v2 | 21 | 50 个 TeX 文件 | 3/3 | 0 / 229 / 4 |
| [2026-09-21/medical-mllm-compression](../../raw/papers/2026-09-21/medical-mllm-compression/README.md) | arXiv:2507.21976v3 | 16 | 1 个 TeX 文件 | 3/3 | 0 / 44 / 7 |
| [2026-09-21/microscaling-formats](../../raw/papers/2026-09-21/microscaling-formats/README.md) | arXiv:2310.10537v3 | 9 | 8 个 TeX 文件 | 3/3 | 0 / 6 / 0 |
| [2026-09-21/mmlu-pro](../../raw/papers/2026-09-21/mmlu-pro/README.md) | arXiv:2406.01574v6 | 24 | 2 个 TeX 文件 | 3/3 | 0 / 34 / 31 |
| [2026-09-21/mquant](../../raw/papers/2026-09-21/mquant/README.md) | arXiv:2502.00425v2 | 18 | 8 个 TeX 文件 | 3/3 | 0 / 197 / 20 |
| [2026-09-21/multi-query-attention](../../raw/papers/2026-09-21/multi-query-attention/README.md) | arXiv:1911.02150v1 | 9 | 1 个 TeX 文件 | 3/3 | 0 / 10 / 0 |
| [2026-09-21/multimodal-qat](../../raw/papers/2026-09-21/multimodal-qat/README.md) | arXiv:2408.03735v1 | 10 | 1 个 TeX 文件 | 3/3 | 0 / 130 / 0 |
| [2026-09-21/norm-tweaking](../../raw/papers/2026-09-21/norm-tweaking/README.md) | arXiv:2309.02784v2 | 10 | 1 个 TeX 文件 | 3/3 | 0 / 182 / 0 |
| [2026-09-21/nova-kv](../../raw/papers/2026-09-21/nova-kv/README.md) | arXiv:2608.04074v1 | 42 | 2 个 TeX 文件 | 3/3 | 0 / 7 / 59 |
| [2026-09-21/omniquant](../../raw/papers/2026-09-21/omniquant/README.md) | arXiv:2308.13137v3 | 25 | 2 个 TeX 文件 | 3/3 | 0 / 124 / 1 |
| [2026-09-21/pagedattention](../../raw/papers/2026-09-21/pagedattention/README.md) | arXiv:2309.06180v1 | 16 | 6 个 TeX 文件 | 3/3 | 0 / 215 / 0 |
| [2026-09-21/plug-and-play-kv-quantization](../../raw/papers/2026-09-21/plug-and-play-kv-quantization/README.md) | arXiv:2503.16257v2 | 18 | 13 个 TeX 文件 | 3/3 | 0 / 48 / 6 |
| [2026-09-21/q-vlm](../../raw/papers/2026-09-21/q-vlm/README.md) | arXiv:2410.08119v3 | 15 | 8 个 TeX 文件 | 3/3 | 0 / 24 / 0 |
| [2026-09-21/qapruner](../../raw/papers/2026-09-21/qapruner/README.md) | arXiv:2604.02816v1 | 12 | 4 个 TeX 文件 | 3/3 | 0 / 80 / 4 |
| [2026-09-21/qig](../../raw/papers/2026-09-21/qig/README.md) | arXiv:2603.17809v1 | 18 | 2 个 TeX 文件 | 3/3 | 0 / 268 / 11 |
| [2026-09-21/qlora](../../raw/papers/2026-09-21/qlora/README.md) | arXiv:2305.14314v1 | 26 | 1 个 TeX 文件 | 3/3 | 0 / 43 / 8 |
| [2026-09-21/qserve](../../raw/papers/2026-09-21/qserve/README.md) | arXiv:2405.04532v3 | 19 | 35 个 TeX 文件 | 3/3 | 0 / 203 / 12 |
| [2026-09-21/quantization-reliable-vqa](../../raw/papers/2026-09-21/quantization-reliable-vqa/README.md) | arXiv:2602.13289v1 | 6 | 1 个 TeX 文件 | 3/3 | 0 / 44 / 0 |
| [2026-09-21/quantization-white-paper](../../raw/papers/2026-09-21/quantization-white-paper/README.md) | arXiv:2106.08295v1 | 27 | 22 个 TeX 文件 | 3/3 | 0 / 29 / 52 |
| [2026-09-21/quarot](../../raw/papers/2026-09-21/quarot/README.md) | arXiv:2404.00456v2 | 21 | 18 个 TeX 文件 | 3/3 | 0 / 12 / 3 |
| [2026-09-21/quasar](../../raw/papers/2026-09-21/quasar/README.md) | arXiv:2608.13966v1 | 39 | 16 个 TeX 文件 | 3/3 | 0 / 78 / 18 |
| [2026-09-21/quip](../../raw/papers/2026-09-21/quip/README.md) | arXiv:2307.13304v2 | 34 | 14 个 TeX 文件 | 3/3 | 0 / 34 / 314 |
| [2026-09-21/quip-sharp](../../raw/papers/2026-09-21/quip-sharp/README.md) | arXiv:2402.04396v2 | 27 | 23 个 TeX 文件 | 3/3 | 0 / 160 / 141 |
| [2026-09-21/qwen2-5-omni](../../raw/papers/2026-09-21/qwen2-5-omni/README.md) | arXiv:2503.20215v1 | 20 | 10 个 TeX 文件 | 3/3 | 0 / 21 / 4 |
| [2026-09-21/qwen2-5-vl](../../raw/papers/2026-09-21/qwen2-5-vl/README.md) | arXiv:2502.13923v1 | 23 | 8 个 TeX 文件 | 3/3 | 0 / 35 / 0 |
| [2026-09-21/qwen3](../../raw/papers/2026-09-21/qwen3/README.md) | arXiv:2505.09388v1 | 35 | 10 个 TeX 文件 | 3/3 | 0 / 41 / 0 |
| [2026-09-21/qwen3-vl](../../raw/papers/2026-09-21/qwen3-vl/README.md) | arXiv:2511.21631v2 | 42 | 15 个 TeX 文件 | 3/3 | 0 / 84 / 0 |
| [2026-09-21/resq](../../raw/papers/2026-09-21/resq/README.md) | arXiv:2412.14363v2 | 18 | 14 个 TeX 文件 | 3/3 | 0 / 159 / 2 |
| [2026-09-21/rmsnorm](../../raw/papers/2026-09-21/rmsnorm/README.md) | arXiv:1910.07467v1 | 14 | 1 个 TeX 文件 | 3/3 | 0 / 40 / 22 |
| [2026-09-21/roformer](../../raw/papers/2026-09-21/roformer/README.md) | arXiv:2104.09864v5 | 14 | 1 个 TeX 文件 | 3/3 | 0 / 3 / 24 |
| [2026-09-21/ruler](../../raw/papers/2026-09-21/ruler/README.md) | arXiv:2404.06654v3 | 27 | 41 个 TeX 文件 | 3/3 | 0 / 34 / 0 |
| [2026-09-21/sarathi-serve](../../raw/papers/2026-09-21/sarathi-serve/README.md) | arXiv:2403.02310v3 | 18 | 32 个 TeX 文件 | 3/3 | 0 / 260 / 0 |
| [2026-09-21/saw-int4](../../raw/papers/2026-09-21/saw-int4/README.md) | arXiv:2604.19157v1 | 24 | 7 个 TeX 文件 | 3/3 | 0 / 60 / 6 |
| [2026-09-21/sglang](../../raw/papers/2026-09-21/sglang/README.md) | arXiv:2312.07104v2 | 20 | 10 个 TeX 文件 | 3/3 | 0 / 22 / 0 |
| [2026-09-21/sharq](../../raw/papers/2026-09-21/sharq/README.md) | arXiv:2606.26587v1 | 20 | 2 个 TeX 文件 | 3/3 | 0 / 25 / 2 |
| [2026-09-21/slicegpt](../../raw/papers/2026-09-21/slicegpt/README.md) | arXiv:2401.15024v2 | 22 | 未保存（允许） | 2/2 | 0 / 36 / 2 |
| [2026-09-21/smoothquant](../../raw/papers/2026-09-21/smoothquant/README.md) | arXiv:2211.10438v7 | 13 | 30 个 TeX 文件 | 3/3 | 0 / 150 / 0 |
| [2026-09-21/smoothquant-plus](../../raw/papers/2026-09-21/smoothquant-plus/README.md) | arXiv:2312.03788v1 | 12 | 1 个 TeX 文件 | 3/3 | 0 / 185 / 27 |
| [2026-09-21/spikemllm](../../raw/papers/2026-09-21/spikemllm/README.md) | arXiv:2604.18610v1 | 20 | 1 个 TeX 文件 | 3/3 | 0 / 217 / 14 |
| [2026-09-21/spinquant](../../raw/papers/2026-09-21/spinquant/README.md) | arXiv:2405.16406v4 | 24 | 37 个 TeX 文件 | 3/3 | 0 / 21 / 5 |
| [2026-09-21/splitq](../../raw/papers/2026-09-21/splitq/README.md) | arXiv:2605.19929v1 | 24 | 1 个 TeX 文件 | 3/3 | 21 / 45 / 13 |
| [2026-09-21/spqr](../../raw/papers/2026-09-21/spqr/README.md) | arXiv:2306.03078v1 | 26 | 4 个 TeX 文件 | 3/3 | 0 / 29 / 0 |
| [2026-09-21/squeezellm](../../raw/papers/2026-09-21/squeezellm/README.md) | arXiv:2306.07629v4 | 23 | 10 个 TeX 文件 | 3/3 | 0 / 160 / 1 |
| [2026-09-21/tritonbench](../../raw/papers/2026-09-21/tritonbench/README.md) | arXiv:2502.14752v1 | 13 | 22 个 TeX 文件 | 3/3 | 0 / 234 / 0 |
| [2026-09-21/turboquant](../../raw/papers/2026-09-21/turboquant/README.md) | arXiv:2504.19874v1 | 25 | 3 个 TeX 文件 | 3/3 | 0 / 45 / 83 |
| [2026-09-21/veq](../../raw/papers/2026-09-21/veq/README.md) | arXiv:2602.01037v1 | 10 | 1 个 TeX 文件 | 3/3 | 0 / 148 / 0 |
| [2026-09-21/vlmq](../../raw/papers/2026-09-21/vlmq/README.md) | arXiv:2508.03351v2 | 28 | 19 个 TeX 文件 | 3/3 | 0 / 208 / 2 |
| [2026-09-21/zeroquant](../../raw/papers/2026-09-21/zeroquant/README.md) | arXiv:2206.01861v1 | 24 | 14 个 TeX 文件 | 3/3 | 0 / 10 / 4 |
| [2026-09-22/efficientqat](../../raw/papers/2026-09-22/efficientqat/README.md) | arXiv:2407.11062v3 | 20 | 未保存（允许） | 2/2 | 0 / 0 / 0 |
| [2026-09-22/flexround](../../raw/papers/2026-09-22/flexround/README.md) | arXiv:2306.00317v2 | 27 | 未保存（允许） | 2/2 | 0 / 0 / 46 |
| [2026-09-22/gptvq](../../raw/papers/2026-09-22/gptvq/README.md) | arXiv:2402.15319v2 | 21 | 未保存（允许） | 2/2 | 0 / 0 / 32 |
| [2026-09-22/hawq-v2](../../raw/papers/2026-09-22/hawq-v2/README.md) | arXiv:1911.03852v1 | 13 | 未保存（允许） | 2/2 | 0 / 0 / 0 |
| [2026-09-22/kvquant](../../raw/papers/2026-09-22/kvquant/README.md) | arXiv:2401.18079v6 | 27 | 未保存（允许） | 2/2 | 0 / 0 / 18 |
| [2026-09-22/loftq](../../raw/papers/2026-09-22/loftq/README.md) | arXiv:2310.08659v4 | 23 | 未保存（允许） | 2/2 | 0 / 0 / 6 |
| [2026-09-22/lq-nets](../../raw/papers/2026-09-22/lq-nets/README.md) | arXiv:1807.10029v1 | 21 | 未保存（允许） | 2/2 | 0 / 0 / 0 |
| [2026-09-22/lrq](../../raw/papers/2026-09-22/lrq/README.md) | arXiv:2407.11534v2 | 36 | 未保存（允许） | 2/2 | 0 / 0 / 2 |
| [2026-09-22/lsq](../../raw/papers/2026-09-22/lsq/README.md) | arXiv:1902.08153v3 | 12 | 未保存（允许） | 2/2 | 0 / 0 / 0 |
| [2026-09-22/lsq-plus](../../raw/papers/2026-09-22/lsq-plus/README.md) | arXiv:2004.09576v1 | 8 | 未保存（允许） | 2/2 | 0 / 0 / 0 |
| [2026-09-22/ostquant](../../raw/papers/2026-09-22/ostquant/README.md) | arXiv:2501.13987v1 | 27 | 未保存（允许） | 2/2 | 0 / 0 / 36 |
| [2026-09-22/pact](../../raw/papers/2026-09-22/pact/README.md) | arXiv:1805.06085v2 | 15 | 未保存（允许） | 2/2 | 0 / 0 / 0 |
| [2026-09-22/pv-tuning](../../raw/papers/2026-09-22/pv-tuning/README.md) | arXiv:2405.14852v2 | 42 | 未保存（允许） | 2/2 | 0 / 0 / 54 |
| [2026-09-22/qera](../../raw/papers/2026-09-22/qera/README.md) | arXiv:2410.06040v2 | 30 | 未保存（允许） | 2/2 | 0 / 0 / 29 |
| [2026-09-22/qjl](../../raw/papers/2026-09-22/qjl/README.md) | arXiv:2406.03482v2 | 13 | 未保存（允许） | 2/2 | 0 / 0 / 36 |
| [2026-09-22/qtip](../../raw/papers/2026-09-22/qtip/README.md) | arXiv:2406.11235v4 | 25 | 未保存（允许） | 2/2 | 0 / 0 / 9 |
| [2026-09-22/signround](../../raw/papers/2026-09-22/signround/README.md) | arXiv:2309.05516v5 | 19 | 未保存（允许） | 2/2 | 0 / 0 / 6 |
| [2026-09-22/vptq](../../raw/papers/2026-09-22/vptq/README.md) | arXiv:2409.17066v2 | 16 | 未保存（允许） | 2/2 | 0 / 0 / 4 |
| [2026-09-23/activation-guided-compensation](../../raw/papers/2026-09-23/activation-guided-compensation/README.md) | arXiv:2609.21450v1 | 19 | 未保存（允许） | 2/2 | 0 / 0 / 70 |
| [2026-09-23/bos-sink-topology](../../raw/papers/2026-09-23/bos-sink-topology/README.md) | arXiv:2609.23585v1 | 6 | 未保存（允许） | 2/2 | 0 / 0 / 8 |
| [2026-09-23/coreprune](../../raw/papers/2026-09-23/coreprune/README.md) | arXiv:2609.26484v1 | 24 | 未保存（允许） | 2/2 | 0 / 0 / 25 |
| [2026-09-23/disaggregated-quantization](../../raw/papers/2026-09-23/disaggregated-quantization/README.md) | arXiv:2609.26333v1 | 29 | 未保存（允许） | 2/2 | 0 / 0 / 0 |
| [2026-09-23/layer-aware-position-embeddings](../../raw/papers/2026-09-23/layer-aware-position-embeddings/README.md) | arXiv:2609.23715v1 | 10 | 未保存（允许） | 2/2 | 0 / 0 / 3 |
| [2026-09-23/prquant](../../raw/papers/2026-09-23/prquant/README.md) | arXiv:2609.22106v1 | 12 | 未保存（允许） | 2/2 | 0 / 0 / 13 |
| [2026-09-23/quantization-retrieval-damage](../../raw/papers/2026-09-23/quantization-retrieval-damage/README.md) | arXiv:2609.24322v1 | 33 | 未保存（允许） | 2/2 | 0 / 0 / 8 |
| [2026-09-23/rgsq](../../raw/papers/2026-09-23/rgsq/README.md) | arXiv:2609.25492v1 | 14 | 未保存（允许） | 2/2 | 0 / 0 / 32 |
| [2026-09-23/sphquant](../../raw/papers/2026-09-23/sphquant/README.md) | arXiv:2609.24875v1 | 16 | 未保存（允许） | 2/2 | 0 / 0 / 10 |
| [2026-09-26/casa-cross-layer-sensitivity](../../raw/papers/2026-09-26/casa-cross-layer-sensitivity/README.md) | arXiv:2609.25916v1 | 33 | 未保存（允许） | 2/2 | 0 / 0 / 192 |
| [2026-09-26/quantization-price-prediction](../../raw/papers/2026-09-26/quantization-price-prediction/README.md) | arXiv:2609.28270v1 | 21 | 未保存（允许） | 2/2 | 0 / 0 / 36 |
| [2026-09-26/ramp-edge-cpu-mixed-precision](../../raw/papers/2026-09-26/ramp-edge-cpu-mixed-precision/README.md) | arXiv:2609.28262v1 | 13 | 未保存（允许） | 2/2 | 0 / 0 / 0 |
| [2026-09-26/the-alignment-illusion](../../raw/papers/2026-09-26/the-alignment-illusion/README.md) | arXiv:2609.30210v1 | 26 | 未保存（允许） | 2/2 | 0 / 0 / 14 |
| [2026-09-28/foldquantvla](../../raw/papers/2026-09-28/foldquantvla/README.md) | arXiv:2609.24433v1 | 8 | 未保存（允许） | 2/2 | 0 / 0 / 5 |
| [2026-09-28/same-bit-width-different-outcomes](../../raw/papers/2026-09-28/same-bit-width-different-outcomes/README.md) | arXiv:2609.28974v1 | 5 | 未保存（允许） | 2/2 | 0 / 0 / 0 |
| [2026-09-28/vla-simd](../../raw/papers/2026-09-28/vla-simd/README.md) | arXiv:2609.24274v1 | 8 | 未保存（允许） | 2/2 | 0 / 0 / 0 |
| [2026-10-03/conqur](../../raw/papers/2026-10-03/conqur/README.md) | arXiv:2605.10793v2 | 21 | 未保存（允许） | 2/2 | 0 / 0 / 8 |
| [2026-10-03/decision-flips](../../raw/papers/2026-10-03/decision-flips/README.md) | arXiv:2610.00694v1 | 20 | 未保存（允许） | 2/2 | 0 / 0 / 0 |
| [2026-10-03/mwop](../../raw/papers/2026-10-03/mwop/README.md) | arXiv:2610.01434v1 | 37 | 未保存（允许） | 2/2 | 0 / 0 / 0 |
| [2026-10-03/quantforge](../../raw/papers/2026-10-03/quantforge/README.md) | arXiv:2609.34680v1 | 41 | 未保存（允许） | 2/2 | 0 / 0 / 16 |
| [2026-10-03/rdtok](../../raw/papers/2026-10-03/rdtok/README.md) | arXiv:2610.00204v1 | 5 | 未保存（允许） | 2/2 | 0 / 0 / 4 |
| [2026-10-03/reconstruction-loss-scale-rmse](../../raw/papers/2026-10-03/reconstruction-loss-scale-rmse/README.md) | arXiv:2610.00983v1 | 20 | 未保存（允许） | 2/2 | 0 / 0 / 2 |
| [2026-10-03/scale-sensitivity](../../raw/papers/2026-10-03/scale-sensitivity/README.md) | arXiv:2609.37416v1 | 56 | 未保存（允许） | 2/2 | 0 / 0 / 755 |
| [2026-10-03/wush-kv](../../raw/papers/2026-10-03/wush-kv/README.md) | arXiv:2609.38121v1 | 30 | 未保存（允许） | 2/2 | 0 / 0 / 199 |
| [2026-10-03/xor-trellis](../../raw/papers/2026-10-03/xor-trellis/README.md) | arXiv:2610.00432v1 | 23 | 未保存（允许） | 2/2 | 0 / 0 / 16 |
| [2026-10-05/covisco](../../raw/papers/2026-10-05/covisco/README.md) | arXiv:2609.39924v1 | 26 | 1 个 TeX 文件 | 3/3 | 0 / 0 / 18 |
| [2026-10-05/onptq](../../raw/papers/2026-10-05/onptq/README.md) | arXiv:2609.36828v1 | 20 | 2 个 TeX 文件 | 3/3 | 0 / 0 / 33 |

GPTQ 的公开评议另核对 3 个附件，均一致；19 条快照记录的 ID 和字符串正文均完整导出。公开评议不被等同于 arXiv v2 PDF 的逐字版本。

### 文章与文档

69 个条目的 README 与入口存在。正文份数包括 article.md、pages/ 下的 Markdown/RST 等可直接阅读正文，不包含本项目 README。限制来自本轮检查与条目声明，不表示未列限制的网页已逐字回查。

| 条目 | 版本 | 正文份数 | PDF 原件/页数 | 文件校验一致/登记 | 已确认限制或需回查项 |
| --- | --- | ---: | --- | --- | --- |
| [2026-09-21/cuda-async-copies](../../raw/articles/2026-09-21/cuda-async-copies/README.md) | snapshot-2026-07-02 | 1 | — | 4/4 | 本地链接候选 1 |
| [2026-09-21/cuda-programming-guide](../../raw/articles/2026-09-21/cuda-programming-guide/README.md) | snapshot-2026-07-02 | 1 | — | 1/1 | 本地链接候选 1 |
| [2026-09-21/cuda-programming-model](../../raw/articles/2026-09-21/cuda-programming-model/README.md) | snapshot-2026-07-02 | 1 | — | 9/9 | 本地链接候选 1 |
| [2026-09-21/cuda-simt-kernels](../../raw/articles/2026-09-21/cuda-simt-kernels/README.md) | snapshot-2026-07-02 | 1 | — | 9/9 | 本地链接候选 1 |
| [2026-09-21/cuda-tile-kernels](../../raw/articles/2026-09-21/cuda-tile-kernels/README.md) | snapshot-2026-07-02 | 1 | — | 2/2 | 本地链接候选 1 |
| [2026-09-21/cutlass-blackwell-gemm](../../raw/articles/2026-09-21/cutlass-blackwell-gemm/README.md) | snapshot-2026-07-02 | 1 | — | 3/3 | 本地链接候选 1 |
| [2026-09-21/cutlass-cute-dsl](../../raw/articles/2026-09-21/cutlass-cute-dsl/README.md) | snapshot-2026-07-02 | 1 | — | 1/1 | 本地链接候选 1 |
| [2026-09-21/cutlass-efficient-gemm](../../raw/articles/2026-09-21/cutlass-efficient-gemm/README.md) | snapshot-2026-07-02 | 1 | — | 4/4 | 本地链接候选 1 |
| [2026-09-21/cutlass-gemm-api](../../raw/articles/2026-09-21/cutlass-gemm-api/README.md) | snapshot-2026-07-02 | 1 | — | 2/2 | 本地链接候选 1 |
| [2026-09-21/cutlass-grouped-schedulers](../../raw/articles/2026-09-21/cutlass-grouped-schedulers/README.md) | snapshot-2026-07-02 | 1 | — | 8/8 | 本地链接候选 1 |
| [2026-09-21/cutlass-layouts-tensors](../../raw/articles/2026-09-21/cutlass-layouts-tensors/README.md) | snapshot-2026-07-02 | 1 | — | 2/2 | 本地链接候选 1 |
| [2026-09-21/cutlass-profiler](../../raw/articles/2026-09-21/cutlass-profiler/README.md) | snapshot-2026-07-02 | 1 | — | 2/2 | 本地链接候选 1 |
| [2026-09-21/cutlass-quickstart](../../raw/articles/2026-09-21/cutlass-quickstart/README.md) | snapshot-2026-07-02 | 1 | — | 2/2 | 本地链接候选 1 |
| [2026-09-21/cutlass-synchronization](../../raw/articles/2026-09-21/cutlass-synchronization/README.md) | snapshot-2026-07-02 | 1 | — | 1/1 | 本地链接候选 1 |
| [2026-09-21/cutlass-tile-iterators](../../raw/articles/2026-09-21/cutlass-tile-iterators/README.md) | snapshot-2026-07-02 | 1 | — | 2/2 | 本地链接候选 1 |
| [2026-09-21/flashinfer-append-paged-kv-cache](../../raw/articles/2026-09-21/flashinfer-append-paged-kv-cache/README.md) | snapshot-2026-07-02 | 1 | — | 1/1 | 本轮文件与结构检查未确认缺陷；未逐字认证来源 |
| [2026-09-21/flashinfer-bench-gpu-time](../../raw/articles/2026-09-21/flashinfer-bench-gpu-time/README.md) | snapshot-2026-07-02 | 1 | — | 1/1 | 本轮文件与结构检查未确认缺陷；未逐字认证来源 |
| [2026-09-21/flashinfer-fp4-quantize](../../raw/articles/2026-09-21/flashinfer-fp4-quantize/README.md) | snapshot-2026-07-02 | 1 | — | 1/1 | 本轮文件与结构检查未确认缺陷；未逐字认证来源 |
| [2026-09-21/flashinfer-gemm-fp8-blockscaled](../../raw/articles/2026-09-21/flashinfer-gemm-fp8-blockscaled/README.md) | snapshot-2026-07-02 | 1 | — | 1/1 | 本轮文件与结构检查未确认缺陷；未逐字认证来源 |
| [2026-09-21/flashinfer-kv-cache-layout](../../raw/articles/2026-09-21/flashinfer-kv-cache-layout/README.md) | snapshot-2026-07-02 | 1 | — | 5/5 | 本轮文件与结构检查未确认缺陷；未逐字认证来源 |
| [2026-09-21/flashinfer-mm-bf16-fp4](../../raw/articles/2026-09-21/flashinfer-mm-bf16-fp4/README.md) | snapshot-2026-07-02 | 1 | — | 1/1 | 本轮文件与结构检查未确认缺陷；未逐字认证来源 |
| [2026-09-21/flashinfer-mm-fp4](../../raw/articles/2026-09-21/flashinfer-mm-fp4/README.md) | snapshot-2026-07-02 | 1 | — | 1/1 | 本轮文件与结构检查未确认缺陷；未逐字认证来源 |
| [2026-09-21/flashinfer-nvfp4-kv-cache](../../raw/articles/2026-09-21/flashinfer-nvfp4-kv-cache/README.md) | snapshot-2026-07-02 | 1 | — | 1/1 | 本轮文件与结构检查未确认缺陷；未逐字认证来源 |
| [2026-09-21/flashinfer-recursive-attention](../../raw/articles/2026-09-21/flashinfer-recursive-attention/README.md) | snapshot-2026-07-02 | 1 | — | 2/2 | 本轮文件与结构检查未确认缺陷；未逐字认证来源 |
| [2026-09-21/flashinfer-release-notes](../../raw/articles/2026-09-21/flashinfer-release-notes/README.md) | snapshot-2026-08-20 | 1 | — | 1/1 | 本轮文件与结构检查未确认缺陷；未逐字认证来源 |
| [2026-09-21/flashinfer-single-decode](../../raw/articles/2026-09-21/flashinfer-single-decode/README.md) | snapshot-2026-07-02 | 1 | — | 1/1 | 本轮文件与结构检查未确认缺陷；未逐字认证来源 |
| [2026-09-21/flashinfer-single-prefill](../../raw/articles/2026-09-21/flashinfer-single-prefill/README.md) | snapshot-2026-07-02 | 1 | — | 1/1 | 本轮文件与结构检查未确认缺陷；未逐字认证来源 |
| [2026-09-21/luq-project](../../raw/articles/2026-09-21/luq-project/README.md) | snapshot-2026-09-15 | 1 | — | 10/10 | 本轮文件与结构检查未确认缺陷；未逐字认证来源 |
| [2026-09-21/mit-6441-lecture-1](../../raw/articles/2026-09-21/mit-6441-lecture-1/README.md) | Spring 2010, Lecture 1 | 1 | 1/8 | 2/2 | 提取字符异常 23 / 10 / 0 |
| [2026-09-21/nvidia-cuda-programming-model](../../raw/articles/2026-09-21/nvidia-cuda-programming-model/README.md) | snapshot-2026-07-02 | 1 | — | 5/5 | 本轮文件与结构检查未确认缺陷；未逐字认证来源 |
| [2026-09-21/nvidia-cutlass-introduction](../../raw/articles/2026-09-21/nvidia-cutlass-introduction/README.md) | snapshot-2026-07-02 | 1 | — | 11/11 | 本轮文件与结构检查未确认缺陷；未逐字认证来源 |
| [2026-09-21/nvidia-shared-memory](../../raw/articles/2026-09-21/nvidia-shared-memory/README.md) | snapshot-2026-07-02 | 1 | — | 1/1 | 本轮文件与结构检查未确认缺陷；未逐字认证来源 |
| [2026-09-21/nvidia-tensor-cores](../../raw/articles/2026-09-21/nvidia-tensor-cores/README.md) | snapshot-2026-07-02 | 1 | — | 6/6 | 本轮文件与结构检查未确认缺陷；未逐字认证来源 |
| [2026-09-21/nvidia-warp-primitives](../../raw/articles/2026-09-21/nvidia-warp-primitives/README.md) | snapshot-2026-07-02 | 1 | — | 4/4 | 本轮文件与结构检查未确认缺陷；未逐字认证来源 |
| [2026-09-21/pytorch-orthogonal](../../raw/articles/2026-09-21/pytorch-orthogonal/README.md) | snapshot-2026-09-15 | 1 | — | 1/1 | 本轮文件与结构检查未确认缺陷；未逐字认证来源 |
| [2026-09-21/triton-block-scaled-matmul](../../raw/articles/2026-09-21/triton-block-scaled-matmul/README.md) | snapshot-2026-07-02 | 1 | — | 1/1 | 本轮文件与结构检查未确认缺陷；未逐字认证来源 |
| [2026-09-21/triton-fused-attention](../../raw/articles/2026-09-21/triton-fused-attention/README.md) | snapshot-2026-07-02 | 1 | — | 1/1 | 本轮文件与结构检查未确认缺陷；未逐字认证来源 |
| [2026-09-21/triton-group-gemm](../../raw/articles/2026-09-21/triton-group-gemm/README.md) | snapshot-2026-07-02 | 1 | — | 1/1 | 本轮文件与结构检查未确认缺陷；未逐字认证来源 |
| [2026-09-21/triton-introduction](../../raw/articles/2026-09-21/triton-introduction/README.md) | snapshot-2026-07-02 | 1 | — | 3/3 | 本轮文件与结构检查未确认缺陷；未逐字认证来源 |
| [2026-09-21/triton-matmul](../../raw/articles/2026-09-21/triton-matmul/README.md) | snapshot-2026-07-02 | 1 | — | 2/2 | 本轮文件与结构检查未确认缺陷；未逐字认证来源 |
| [2026-09-21/triton-persistent-matmul](../../raw/articles/2026-09-21/triton-persistent-matmul/README.md) | snapshot-2026-07-02 | 1 | — | 1/1 | 本轮文件与结构检查未确认缺陷；未逐字认证来源 |
| [2026-09-21/vllm-release-notes](../../raw/articles/2026-09-21/vllm-release-notes/README.md) | snapshot-2026-08-20 | 1 | — | 1/1 | 本轮文件与结构检查未确认缺陷；未逐字认证来源 |
| [2026-09-21/wechat-masquant](../../raw/articles/2026-09-21/wechat-masquant/README.md) | snapshot-2026-06-30 | 1 | — | 7/7 | 本轮文件与结构检查未确认缺陷；未逐字认证来源 |
| [2026-09-21/wechat-qig](../../raw/articles/2026-09-21/wechat-qig/README.md) | snapshot-2026-06-30 | 1 | — | 8/8 | 本轮文件与结构检查未确认缺陷；未逐字认证来源 |
| [2026-09-22/compute-sanitizer](../../raw/articles/2026-09-22/compute-sanitizer/README.md) | snapshot-2026-09-22 | 1 | — | 1/1 | 保留远端图片，未全量检查远端可用性 |
| [2026-09-22/cuda-libdevice-round](../../raw/articles/2026-09-22/cuda-libdevice-round/README.md) | CUDA 12.9.1 | 1 | — | 0/0 | 本轮文件与结构检查未确认缺陷；未逐字认证来源 |
| [2026-09-22/floating-point](../../raw/articles/2026-09-22/floating-point/README.md) | v13.4 snapshot 2026-09-22 | 1 | — | 1/1 | 保留远端图片，未全量检查远端可用性；已确认源文 FMA 算术矛盾 |
| [2026-09-22/nsight-compute](../../raw/articles/2026-09-22/nsight-compute/README.md) | snapshot-2026-09-22 | 3 | — | 0/0 | 保留远端图片，未全量检查远端可用性 |
| [2026-09-22/nsight-systems](../../raw/articles/2026-09-22/nsight-systems/README.md) | snapshot-2026-09-22 | 1 | — | 1/1 | 保留远端图片，未全量检查远端可用性 |
| [2026-09-22/perplexity](../../raw/articles/2026-09-22/perplexity/README.md) | Transformers main snapshot 2026-09-22 | 1 | — | 1/1 | 保留远端图片，未全量检查远端可用性；公式写法异常，历史原因未认证 |
| [2026-09-22/ptx-mma-layout](../../raw/articles/2026-09-22/ptx-mma-layout/README.md) | CUDA 11.8 archive | 1 | — | 1/1 | 保留远端图片，未全量检查远端可用性 |
| [2026-09-22/pytorch-performance-tools](../../raw/articles/2026-09-22/pytorch-performance-tools/README.md) | snapshot-2026-09-22 | 5 | — | 0/0 | 本轮文件与结构检查未确认缺陷；未逐字认证来源 |
| [2026-09-24/bilibili-llm-system-software](../../raw/articles/2026-09-24/bilibili-llm-system-software/README.md) | page-description-snapshot-2026-09-24; BV1R7jc6qEmn | 1 | — | 1/1 | 本轮文件与结构检查未确认缺陷；未逐字认证来源 |
| [2026-09-24/cuda-softmax-worklog](../../raw/articles/2026-09-24/cuda-softmax-worklog/README.md) | indexed-text-snapshot-2026-09-24 | 1 | — | 1/1 | 本轮文件与结构检查未确认缺陷；未逐字认证来源 |
| [2026-09-24/gpu-mode-lecture-materials](../../raw/articles/2026-09-24/gpu-mode-lecture-materials/README.md) | 77a8df418834e5789c12da23e7d2719e0efabef1 | 1 | 3/185 | 4/4 | 本轮文件与结构检查未确认缺陷；未逐字认证来源 |
| [2026-09-24/h100-gemm-worklog](../../raw/articles/2026-09-24/h100-gemm-worklog/README.md) | html-body-snapshot-2026-09-24 | 1 | — | 1/1 | 保留远端图片，未全量检查远端可用性 |
| [2026-09-24/iit-bombay-systems-for-ml-2026](../../raw/articles/2026-09-24/iit-bombay-systems-for-ml-2026/README.md) | Spring 2026 course archive; slides exported 2026-09-24 | 1 | 7/369 | 7/7 | 部分 PDF 页无可提取文字，未 OCR |
| [2026-09-24/leetgpu-gpt2-transformer-challenge](../../raw/articles/2026-09-24/leetgpu-gpt2-transformer-challenge/README.md) | public-index-post-snapshot-2026-09-24 | 1 | — | 1/1 | 本轮文件与结构检查未确认缺陷；未逐字认证来源 |
| [2026-09-24/microsoft-system-for-ai](../../raw/articles/2026-09-24/microsoft-system-for-ai/README.md) | repository-commit-2a974e1a483eb5c2f714ba5001ac33d8c58998d5 | 1 | 6/273 | 6/6 | 部分 PDF 页无可提取文字，未 OCR |
| [2026-09-24/modelopt-ptq-guides](../../raw/articles/2026-09-24/modelopt-ptq-guides/README.md) | commit-f2f0d6958eb95374d816f3d6256a0be3738231a3 | 3 | — | 2/2 | 本地链接候选 29 |
| [2026-09-24/nvidia-dynamo-architecture](../../raw/articles/2026-09-24/nvidia-dynamo-architecture/README.md) | v1.4.2 | 4 | — | 3/3 | 本地链接候选 25；保留远端图片，未全量检查远端可用性；缺 1 张本地图；3 张签名外链过期 |
| [2026-09-24/pytorch-distributed-tutorials](../../raw/articles/2026-09-24/pytorch-distributed-tutorials/README.md) | PyTorch Tutorials 2.14.0+cu130; FSDP tutorial last verified 2024-11-05 | 3 | — | 2/2 | RST 缺 3 张本地图 |
| [2026-09-24/ray-serve-llm-architecture](../../raw/articles/2026-09-24/ray-serve-llm-architecture/README.md) | ray-2.58.0-commit-642b00f8c3c5a01cab7c2e2bff8725fa027b23b4 | 8 | — | 7/7 | 本地链接候选 8 |
| [2026-09-24/tensorrt-llm-quantization](../../raw/articles/2026-09-24/tensorrt-llm-quantization/README.md) | source-commit-a6f4151497372f14a5261113a12be2b406e9b5fc | 2 | — | 1/1 | 本地链接候选 2 |
| [2026-09-24/vllm-v1-architecture](../../raw/articles/2026-09-24/vllm-v1-architecture/README.md) | vllm-main-f9ad9dd6b495f7492fa975e0ac48164b090d99ad | 2 | — | 1/1 | 本地链接候选 21；缺 5 张本地图 |
| [2026-09-24/vllm-v1-zhihu-source-reading](../../raw/articles/2026-09-24/vllm-v1-zhihu-source-reading/README.md) | snapshot-2026-09-24; article states vLLM ba22152096b2484faa3579624a253d54804d876d (2026-07-06) | 1 | — | 7/7 | 本轮文件与结构检查未确认缺陷；未逐字认证来源 |
| [2026-09-28/vla-simd-model-bundle](../../raw/articles/2026-09-28/vla-simd-model-bundle/README.md) | 2026-09-28 page snapshot | 1 | — | 1/1 | 本轮文件与结构检查未确认缺陷；未逐字认证来源 |
| [2026-10-04/pytorch-cpp-cuda-operators](../../raw/articles/2026-10-04/pytorch-cpp-cuda-operators/README.md) | cpp-extension-2.14; custom-ops-snapshot-2026-10-04 | 3 | — | 3/3 | 本地链接候选 1 |
| [2026-10-05/tensorrt-edge-llm-dart](../../raw/articles/2026-10-05/tensorrt-edge-llm-dart/README.md) | 0.11.0; docs-build-e649dadcbbf049fa4caf2f09f3a3275ca56751db; snapshot-2026-10-05 | 3 | — | 3/3 | 本轮文件与结构检查未确认缺陷；未逐字认证来源 |

### 代码仓库

59 个正式仓库均有父归档或 Git 对象基线。版本列只显示前 8 位便于阅读，完整标识以条目 README 为准，fast.cu 的错误另已说明。父主体计数不包含子模块；“缺少”列列出的 5 个文件已有排除记录。

| 条目 | 版本前缀 | 父基线 | 父主体已比较 | 比较时未取得字节 | 已披露缺少项 | 全树额外文件 | 子模块 pin |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| [2026-09-21/affinequant](../../raw/repositories/2026-09-21/affinequant/README.md) | 0f3ba939 | 归档 zip | 125 | 81 | 0 | 0 | 无登记子模块 |
| [2026-09-21/aqlm](../../raw/repositories/2026-09-21/aqlm/README.md) | e79a896e | 归档 zip | 32 | 19 | 0 | 0 | 无登记子模块 |
| [2026-09-21/atom](../../raw/repositories/2026-09-21/atom/README.md) | 7e3618b1 | Git 对象 | 192 | 141 | 0 | 0 | 4/4 一致 |
| [2026-09-21/cs336-lectures](../../raw/repositories/2026-09-21/cs336-lectures/README.md) | 8b59b507 | 归档 zip | 254 | 166 | 0 | 0 | 无登记子模块 |
| [2026-09-21/cuda-samples](../../raw/repositories/2026-09-21/cuda-samples/README.md) | b7c5481c | Git 对象 | 1254 | 768 | 0 | 0 | 无登记子模块 |
| [2026-09-21/cutlass](../../raw/repositories/2026-09-21/cutlass/README.md) | e8ecfad7 | Git 对象 | 4168 | 2785 | 0 | 0 | 无登记子模块 |
| [2026-09-21/efficientai](../../raw/repositories/2026-09-21/efficientai/README.md) | 3d32ae42 | 归档 zip | 683 | 466 | 0 | 260 | 无登记子模块 |
| [2026-09-21/flash-attention](../../raw/repositories/2026-09-21/flash-attention/README.md) | 73c992c8 | Git 对象 | 689 | 380 | 0 | 0 | 4/4 一致 |
| [2026-09-21/flashinfer](../../raw/repositories/2026-09-21/flashinfer/README.md) | a0a6b019 | 归档 zip | 1217 | 900 | 0 | 0 | 4/4 一致 |
| [2026-09-21/flatquant](../../raw/repositories/2026-09-21/flatquant/README.md) | 9d88ffcb | 归档 zip | 57 | 42 | 0 | 0 | 3/3 一致 |
| [2026-09-21/flexq](../../raw/repositories/2026-09-21/flexq/README.md) | a42424d8 | 归档 zip | 3682 | 3447 | 0 | 1802 | 无登记子模块 |
| [2026-09-21/gptq](../../raw/repositories/2026-09-21/gptq/README.md) | 2d65066e | 归档 zip | 22 | 19 | 0 | 0 | 无登记子模块 |
| [2026-09-21/kitty](../../raw/repositories/2026-09-21/kitty/README.md) | dfd2c07b | 归档 zip | 24 | 4 | 0 | 0 | 2/2 一致 |
| [2026-09-21/kivi](../../raw/repositories/2026-09-21/kivi/README.md) | 876b4d2d | 归档 zip | 25 | 15 | 0 | 0 | 无登记子模块 |
| [2026-09-21/llama-cpp](../../raw/repositories/2026-09-21/llama-cpp/README.md) | b820cc8e | 归档 zip | 1726 | 1290 | 0 | 0 | 无登记子模块 |
| [2026-09-21/llm-awq](../../raw/repositories/2026-09-21/llm-awq/README.md) | d6e797a4 | 归档 zip | 121 | 59 | 0 | 0 | 无登记子模块 |
| [2026-09-21/lm-evaluation-harness](../../raw/repositories/2026-09-21/lm-evaluation-harness/README.md) | 62617a8c | 归档 zip | 8506 | 5988 | 0 | 259 | 无登记子模块 |
| [2026-09-21/marlin](../../raw/repositories/2026-09-21/marlin/README.md) | 1f25790b | Git 对象 | 11 | 7 | 0 | 0 | 无登记子模块 |
| [2026-09-21/mbq](../../raw/repositories/2026-09-21/mbq/README.md) | a4d460df | 归档 zip | 38 | 48 | 0 | 0 | 2/2 一致 |
| [2026-09-21/mlc-llm](../../raw/repositories/2026-09-21/mlc-llm/README.md) | 6474f7db | Git 对象 | 415 | 278 | 0 | 0 | 18/18 一致 |
| [2026-09-21/mquant](../../raw/repositories/2026-09-21/mquant/README.md) | 7b3e8196 | 归档 zip | 124 | 117 | 0 | 175 | 无登记子模块 |
| [2026-09-21/nano-vllm](../../raw/repositories/2026-09-21/nano-vllm/README.md) | bb823b3e | 归档 zip | 18 | 8 | 0 | 0 | 无登记子模块 |
| [2026-09-21/omniserve](../../raw/repositories/2026-09-21/omniserve/README.md) | 02b2925a | 归档 zip | 156 | 98 | 0 | 18 | 无登记子模块 |
| [2026-09-21/qig](../../raw/repositories/2026-09-21/qig/README.md) | 06fa8813 | 归档 zip | 57 | 44 | 0 | 0 | 无登记子模块 |
| [2026-09-21/qslaw](../../raw/repositories/2026-09-21/qslaw/README.md) | 264dfb08 | 归档 zip | 75 | 99 | 1 | 0 | 无登记子模块 |
| [2026-09-21/quarot](../../raw/repositories/2026-09-21/quarot/README.md) | 5008669b | 归档 zip | 42 | 25 | 0 | 29 | 3/3 一致 |
| [2026-09-21/quip](../../raw/repositories/2026-09-21/quip/README.md) | ac92cfc7 | 归档 zip | 26 | 17 | 0 | 0 | 无登记子模块 |
| [2026-09-21/quip-sharp](../../raw/repositories/2026-09-21/quip-sharp/README.md) | 1d8f873e | Git 对象 | 40 | 20 | 1 | 0 | 无登记子模块 |
| [2026-09-21/qvlm](../../raw/repositories/2026-09-21/qvlm/README.md) | 191bdc9a | 归档 zip | 146 | 70 | 0 | 0 | 无登记子模块 |
| [2026-09-21/qwen2-5-omni](../../raw/repositories/2026-09-21/qwen2-5-omni/README.md) | d8a31ca5 | 归档 zip | 17 | 7 | 0 | 0 | 无登记子模块 |
| [2026-09-21/qwen3-vl](../../raw/repositories/2026-09-21/qwen3-vl/README.md) | 96588727 | 归档 zip | 97 | 57 | 0 | 0 | 无登记子模块 |
| [2026-09-21/resq](../../raw/repositories/2026-09-21/resq/README.md) | cb87d0d0 | 归档 zip | 28 | 15 | 0 | 0 | 无登记子模块 |
| [2026-09-21/saw-int4](../../raw/repositories/2026-09-21/saw-int4/README.md) | e51bfa72 | Git 对象 | 15 | 3 | 0 | 0 | 3/3 一致 |
| [2026-09-21/sglang](../../raw/repositories/2026-09-21/sglang/README.md) | 2f730e29 | 归档 zip | 6831 | 0 | 0 | 3 | 无登记子模块 |
| [2026-09-21/sharq](../../raw/repositories/2026-09-21/sharq/README.md) | 3cc584fe | 归档 zip | 66 | 0 | 0 | 0 | 1/1 一致 |
| [2026-09-21/smoothquant](../../raw/repositories/2026-09-21/smoothquant/README.md) | c61476d7 | 归档 zip | 25 | 0 | 0 | 0 | 无登记子模块 |
| [2026-09-21/spinquant](../../raw/repositories/2026-09-21/spinquant/README.md) | 8f47aa3f | Git 对象 | 34 | 0 | 0 | 0 | 无登记子模块 |
| [2026-09-21/spqr](../../raw/repositories/2026-09-21/spqr/README.md) | 543d10e5 | Git 对象 | 885 | 0 | 0 | 0 | 无登记子模块 |
| [2026-09-21/squeezellm](../../raw/repositories/2026-09-21/squeezellm/README.md) | a5fd71f3 | Git 对象 | 92 | 0 | 1 | 0 | 无登记子模块 |
| [2026-09-21/tensorrt-llm](../../raw/repositories/2026-09-21/tensorrt-llm/README.md) | d567f924 | Git 对象 | 10690 | 0 | 2 | 0 | 无登记子模块 |
| [2026-09-21/torch-int](../../raw/repositories/2026-09-21/torch-int/README.md) | 65266db1 | Git 对象 | 68 | 0 | 0 | 0 | 1/1 一致 |
| [2026-09-21/transformers](../../raw/repositories/2026-09-21/transformers/README.md) | 37f8b0b5 | 归档 zip | 4775 | 0 | 0 | 7 | 无登记子模块 |
| [2026-09-21/triton](../../raw/repositories/2026-09-21/triton/README.md) | 81a46fa0 | Git 对象 | 1684 | 0 | 0 | 0 | 无登记子模块 |
| [2026-09-21/triton-puzzles](../../raw/repositories/2026-09-21/triton-puzzles/README.md) | 4d794abe | 归档 zip | 4 | 0 | 0 | 0 | 无登记子模块 |
| [2026-09-21/vllm](../../raw/repositories/2026-09-21/vllm/README.md) | 568afb3a | 归档 zip | 6078 | 0 | 0 | 0 | 无登记子模块 |
| [2026-09-22/efficientqat](../../raw/repositories/2026-09-22/efficientqat/README.md) | 39175493 | 归档 tar | 48 | 0 | 0 | 0 | 无登记子模块 |
| [2026-09-22/kernel-skills](../../raw/repositories/2026-09-22/kernel-skills/README.md) | 7b7337a1 | 归档 zip | 134 | 0 | 0 | 0 | 无登记子模块 |
| [2026-09-22/kernels](../../raw/repositories/2026-09-22/kernels/README.md) | 5c2cf07f | 归档 zip | 778 | 0 | 0 | 0 | 无登记子模块 |
| [2026-09-23/rgsq](../../raw/repositories/2026-09-23/rgsq/README.md) | a69e2d59 | 归档 tar | 86 | 0 | 0 | 0 | 无登记子模块 |
| [2026-09-24/cudacodes](../../raw/repositories/2026-09-24/cudacodes/README.md) | 9a459b10 | 归档 tar | 78 | 0 | 0 | 0 | 无登记子模块 |
| [2026-09-24/disaggregated-llama-cpp](../../raw/repositories/2026-09-24/disaggregated-llama-cpp/README.md) | a0557fcf | 归档 tar | 3619 | 0 | 0 | 0 | 无登记子模块 |
| [2026-09-24/disaggregated-quantization](../../raw/repositories/2026-09-24/disaggregated-quantization/README.md) | 4a99d6e0 | 归档 tar | 4872 | 0 | 0 | 0 | 1/1 一致 |
| [2026-09-24/fast-cu](../../raw/repositories/2026-09-24/fast-cu/README.md) | 2dfe5e26（登记错误） | 归档 tar | 34 | 0 | 0 | 0 | 无登记子模块 |
| [2026-09-24/layerpos](../../raw/repositories/2026-09-24/layerpos/README.md) | d6a5bccf | 归档 tar | 715 | 0 | 0 | 0 | 无登记子模块 |
| [2026-09-28/foldquantvla](../../raw/repositories/2026-09-28/foldquantvla/README.md) | 4e76b60b | 归档 tar | 914 | 0 | 0 | 0 | 4/4 一致 |
| [2026-09-28/tts-ptq-map](../../raw/repositories/2026-09-28/tts-ptq-map/README.md) | ea6f62da | 归档 tar | 1017 | 0 | 0 | 0 | 无登记子模块 |
| [2026-09-28/vla-simd](../../raw/repositories/2026-09-28/vla-simd/README.md) | 7354b6d7 | 归档 tar | 154 | 0 | 0 | 0 | 无登记子模块 |
| [2026-10-05/covisco](../../raw/repositories/2026-10-05/covisco/README.md) | 604cd70d | 归档 tar | 2562 | 0 | 0 | 0 | 无登记子模块 |
| [2026-10-05/tensorrt-edge-llm](../../raw/repositories/2026-10-05/tensorrt-edge-llm/README.md) | e649dadc | 归档 tar | 1775 | 0 | 0 | 0 | 7/7 一致 |

上述已取得内容未发现未披露的字节、类型、链接或执行位差异。这个结论不包括未读文件，也不把额外文件排除在来源纯净性问题之外。

旧 vllm-364ee36a 只有空目录骨架，没有条目 README 或普通文件；LOG 已记录快照移除，未列入 59 个正式仓库。

### 子模块

以下 57 处版本均重新对照父 Git tree/gitlink，登记版本一致。未读数量来自子模块比较时刻；同一版本在多个位置展开时分别检查。

| 父条目 | source/ 内路径 | 子模块版本前缀 | pin | 已比较文件 | 当次未取得字节 | 额外文件 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| [atom](../../raw/repositories/2026-09-21/atom/README.md) | kernels/3rdparty/llm-awq | 19a5a2c9 | 一致 | 49 | 21 | 0 |
| [atom](../../raw/repositories/2026-09-21/atom/README.md) | kernels/3rdparty/nvbench | 75212298 | 一致 | 108 | 58 | 0 |
| [atom](../../raw/repositories/2026-09-21/atom/README.md) | kernels/3rdparty/torch-int | 65266db1 | 一致 | 34 | 34 | 0 |
| [atom](../../raw/repositories/2026-09-21/atom/README.md) | kernels/3rdparty/torch-int/submodules/cutlass | c975e2cc | 一致 | 3024 | 2098 | 0 |
| [flash-attention](../../raw/repositories/2026-09-21/flash-attention/README.md) | csrc/composable_kernel | 791afc64 | 一致 | 3920 | 2654 | 0 |
| [flash-attention](../../raw/repositories/2026-09-21/flash-attention/README.md) | csrc/cutlass | 71275920 | 一致 | 3943 | 2785 | 0 |
| [flash-attention](../../raw/repositories/2026-09-21/flash-attention/README.md) | third_party/aiter | 9bab8388 | 一致 | 2832 | 1877 | 0 |
| [flash-attention](../../raw/repositories/2026-09-21/flash-attention/README.md) | third_party/aiter/3rdparty/composable_kernel | af7118e3 | 一致 | 4031 | 2735 | 0 |
| [flashinfer](../../raw/repositories/2026-09-21/flashinfer/README.md) | 3rdparty/cccl | 87686768 | 一致 | 4710 | 3463 | 0 |
| [flashinfer](../../raw/repositories/2026-09-21/flashinfer/README.md) | 3rdparty/cutlass | b46b16d0 | 一致 | 3523 | 3376 | 0 |
| [flashinfer](../../raw/repositories/2026-09-21/flashinfer/README.md) | 3rdparty/nixl | 05e4243f | 一致 | 321 | 266 | 0 |
| [flashinfer](../../raw/repositories/2026-09-21/flashinfer/README.md) | 3rdparty/spdlog | c3aed4b6 | 一致 | 105 | 67 | 0 |
| [flatquant](../../raw/repositories/2026-09-21/flatquant/README.md) | third-party/cutlass | ad7b2f5e | 一致 | 3280 | 3019 | 0 |
| [flatquant](../../raw/repositories/2026-09-21/flatquant/README.md) | third-party/fast-hadamard-transform | d1a56eee | 一致 | 12 | 4 | 0 |
| [flatquant](../../raw/repositories/2026-09-21/flatquant/README.md) | third-party/nvbench | a1715140 | 一致 | 141 | 108 | 0 |
| [kitty](../../raw/repositories/2026-09-21/kitty/README.md) | third_party/lm-evaluation-harness | 62617a8c | 一致 | 8638 | 5856 | 0 |
| [kitty](../../raw/repositories/2026-09-21/kitty/README.md) | third_party/transformers | 37f8b0b5 | 一致 | 2658 | 2117 | 0 |
| [mbq](../../raw/repositories/2026-09-21/mbq/README.md) | 3rdparty/LLaVA-NeXT | 1a7e8b2b | 一致 | 119 | 98 | 0 |
| [mbq](../../raw/repositories/2026-09-21/mbq/README.md) | 3rdparty/lmms-eval | 9c030a59 | 一致 | 660 | 551 | 0 |
| [mlc-llm](../../raw/repositories/2026-09-21/mlc-llm/README.md) | 3rdparty/argparse | 557948f1 | 一致 | 45 | 22 | 0 |
| [mlc-llm](../../raw/repositories/2026-09-21/mlc-llm/README.md) | 3rdparty/googletest | 45804691 | 一致 | 162 | 83 | 0 |
| [mlc-llm](../../raw/repositories/2026-09-21/mlc-llm/README.md) | 3rdparty/stb | ae721c50 | 一致 | 263 | 165 | 0 |
| [mlc-llm](../../raw/repositories/2026-09-21/mlc-llm/README.md) | 3rdparty/tokenizers-cpp | 34885cfd | 一致 | 23 | 14 | 0 |
| [mlc-llm](../../raw/repositories/2026-09-21/mlc-llm/README.md) | 3rdparty/tokenizers-cpp/msgpack | 092bc69b | 一致 | 535 | 317 | 0 |
| [mlc-llm](../../raw/repositories/2026-09-21/mlc-llm/README.md) | 3rdparty/tokenizers-cpp/sentencepiece | 11051e3b | 一致 | 164 | 102 | 0 |
| [mlc-llm](../../raw/repositories/2026-09-21/mlc-llm/README.md) | 3rdparty/tvm | 06db68d4 | 一致 | 2283 | 903 | 0 |
| [mlc-llm](../../raw/repositories/2026-09-21/mlc-llm/README.md) | 3rdparty/tvm/3rdparty/OpenCL-Headers | b590a6bf | 一致 | 53 | 0 | 0 |
| [mlc-llm](../../raw/repositories/2026-09-21/mlc-llm/README.md) | 3rdparty/tvm/3rdparty/cutlass | b2dd65dc | 一致 | 6597 | 0 | 0 |
| [mlc-llm](../../raw/repositories/2026-09-21/mlc-llm/README.md) | 3rdparty/tvm/3rdparty/cutlass_fpA_intB_gemm | 96b48221 | 一致 | 80 | 0 | 0 |
| [mlc-llm](../../raw/repositories/2026-09-21/mlc-llm/README.md) | 3rdparty/tvm/3rdparty/cutlass_fpA_intB_gemm/cutlass | cc85b64c | 一致 | 5035 | 0 | 0 |
| [mlc-llm](../../raw/repositories/2026-09-21/mlc-llm/README.md) | 3rdparty/tvm/3rdparty/libflash_attn | 07ba35ba | 一致 | 21 | 1 | 0 |
| [mlc-llm](../../raw/repositories/2026-09-21/mlc-llm/README.md) | 3rdparty/tvm/3rdparty/libflash_attn/cutlass | e0aaa3c3 | 一致 | 3680 | 1904 | 0 |
| [mlc-llm](../../raw/repositories/2026-09-21/mlc-llm/README.md) | 3rdparty/tvm/3rdparty/tvm-ffi | 84ee1a07 | 一致 | 488 | 0 | 0 |
| [mlc-llm](../../raw/repositories/2026-09-21/mlc-llm/README.md) | 3rdparty/tvm/3rdparty/tvm-ffi/3rdparty/dlpack | 84d107bf | 一致 | 37 | 0 | 0 |
| [mlc-llm](../../raw/repositories/2026-09-21/mlc-llm/README.md) | 3rdparty/tvm/3rdparty/tvm-ffi/3rdparty/libbacktrace | 79392187 | 一致 | 70 | 0 | 0 |
| [mlc-llm](../../raw/repositories/2026-09-21/mlc-llm/README.md) | 3rdparty/xgrammar | d4f57c44 | 一致 | 44 | 34 | 0 |
| [mlc-llm](../../raw/repositories/2026-09-21/mlc-llm/README.md) | 3rdparty/xgrammar/3rdparty/dlpack | bbd2f4d3 | 一致 | 26 | 11 | 0 |
| [quarot](../../raw/repositories/2026-09-21/quarot/README.md) | third-party/cutlass | ffa34e70 | 一致 | 5661 | 0 | 16 |
| [quarot](../../raw/repositories/2026-09-21/quarot/README.md) | third-party/fast-hadamard-transform | 4ea722e4 | 一致 | 16 | 0 | 4 |
| [quarot](../../raw/repositories/2026-09-21/quarot/README.md) | third-party/nvbench | d8dced8a | 一致 | 181 | 0 | 9 |
| [saw-int4](../../raw/repositories/2026-09-21/saw-int4/README.md) | third_party/sglang-fast-rotation | 0fcc2419 | 一致 | 3684 | 0 | 0 |
| [saw-int4](../../raw/repositories/2026-09-21/saw-int4/README.md) | third_party/sglang-kmeans | 43925c00 | 一致 | 3760 | 0 | 0 |
| [saw-int4](../../raw/repositories/2026-09-21/saw-int4/README.md) | third_party/simple-evals | ee3b0318 | 一致 | 25 | 0 | 0 |
| [sharq](../../raw/repositories/2026-09-21/sharq/README.md) | third-party/cutlass | baea077e | 一致 | 6890 | 0 | 0 |
| [torch-int](../../raw/repositories/2026-09-21/torch-int/README.md) | submodules/cutlass | c975e2cc | 一致 | 5122 | 0 | 0 |
| [tensorrt-edge-llm](../../raw/repositories/2026-10-05/tensorrt-edge-llm/README.md) | 3rdParty/NVTX | 2fb879e5 | 一致 | 41 | 0 | 0 |
| [tensorrt-edge-llm](../../raw/repositories/2026-10-05/tensorrt-edge-llm/README.md) | 3rdParty/googletest | f132c893 | 一致 | 252 | 0 | 0 |
| [tensorrt-edge-llm](../../raw/repositories/2026-10-05/tensorrt-edge-llm/README.md) | 3rdParty/nlohmannJson | 55f93686 | 一致 | 1163 | 0 | 0 |
| [tensorrt-edge-llm](../../raw/repositories/2026-10-05/tensorrt-edge-llm/README.md) | 3rdParty/xgrammar | 5b4e9ce9 | 一致 | 238 | 0 | 0 |
| [tensorrt-edge-llm](../../raw/repositories/2026-10-05/tensorrt-edge-llm/README.md) | 3rdParty/xgrammar/3rdparty/cpptrace | 6689d14c | 一致 | 152 | 0 | 0 |
| [tensorrt-edge-llm](../../raw/repositories/2026-10-05/tensorrt-edge-llm/README.md) | 3rdParty/xgrammar/3rdparty/dlpack | bbd2f4d3 | 一致 | 37 | 0 | 0 |
| [tensorrt-edge-llm](../../raw/repositories/2026-10-05/tensorrt-edge-llm/README.md) | 3rdParty/xgrammar/3rdparty/googletest | df1544bc | 一致 | 247 | 0 | 0 |
| [disaggregated-quantization](../../raw/repositories/2026-09-24/disaggregated-quantization/README.md) | third_party/Liger-Kernel | bb715c93 | 一致 | 445 | 0 | 0 |
| [foldquantvla](../../raw/repositories/2026-09-28/foldquantvla/README.md) | third_party/cutlass | bf9da7b7 | 一致 | 5888 | 0 | 0 |
| [foldquantvla](../../raw/repositories/2026-09-28/foldquantvla/README.md) | models/groot_n1_6/external_dependencies/LIBERO | 8f1084e3 | 一致 | 1116 | 0 | 0 |
| [foldquantvla](../../raw/repositories/2026-09-28/foldquantvla/README.md) | models/groot_n1_7/external_dependencies/LIBERO | 8f1084e3 | 一致 | 1116 | 0 | 0 |
| [foldquantvla](../../raw/repositories/2026-09-28/foldquantvla/README.md) | models/pi05/third_party/libero | f78abd68 | 一致 | 1116 | 0 | 0 |

### 与固定版本不同的额外文件

下面 22 个文件均位于 QuaRot 的三个子模块中。差异是相对于该子模块固定版本的对应正常路径，未据此推断所有文件的来源操作或精确另一版本身份。

| source/ 内额外路径 | 对应固定版本路径 | 进一步核对 |
| --- | --- | --- |
| third-party/cutlass/README 2.md | third-party/cutlass/README.md | 与独立 CUTLASS e8ecfad7 对应文件一致 |
| third-party/cutlass/cuDNN 2.cmake | third-party/cutlass/cuDNN.cmake | 与独立 CUTLASS e8ecfad7 对应文件一致 |
| third-party/cutlass/LICENSE 2.txt | third-party/cutlass/LICENSE.txt | 与独立 CUTLASS e8ecfad7 对应文件一致 |
| third-party/cutlass/CHANGELOG 2.md | third-party/cutlass/CHANGELOG.md | 与独立 CUTLASS e8ecfad7 对应文件一致 |
| third-party/cutlass/setup 2.cfg | third-party/cutlass/setup.cfg | 与独立 CUTLASS e8ecfad7 对应文件一致 |
| third-party/cutlass/CMakeLists 2.txt | third-party/cutlass/CMakeLists.txt | 与独立 CUTLASS e8ecfad7 对应文件一致 |
| third-party/cutlass/CUDA 2.cmake | third-party/cutlass/CUDA.cmake | 与独立 CUTLASS e8ecfad7 对应文件一致 |
| third-party/cutlass/PUBLICATIONS 2.md | third-party/cutlass/PUBLICATIONS.md | 与独立 CUTLASS e8ecfad7 对应文件一致 |
| third-party/cutlass/cuBLAS 2.cmake | third-party/cutlass/cuBLAS.cmake | 与独立 CUTLASS e8ecfad7 对应文件一致 |
| third-party/cutlass/CITATION 2.cff | third-party/cutlass/CITATION.cff | 与独立 CUTLASS e8ecfad7 对应文件一致 |
| third-party/cutlass/.gitignore 2 | third-party/cutlass/.gitignore | 与独立 CUTLASS e8ecfad7 对应文件一致 |
| third-party/cutlass/CONTRIBUTORS 2.md | third-party/cutlass/CONTRIBUTORS.md | 与独立 CUTLASS e8ecfad7 对应文件一致 |
| third-party/cutlass/bin2hex 2.cmake | third-party/cutlass/bin2hex.cmake | 与独立 CUTLASS e8ecfad7 对应文件一致 |
| third-party/cutlass/pyproject 2.toml | third-party/cutlass/pyproject.toml | 与独立 CUTLASS e8ecfad7 对应文件一致 |
| third-party/fast-hadamard-transform/setup 2.py | third-party/fast-hadamard-transform/setup.py | 与固定版本不同，另一版本身份尚未完全确定 |
| third-party/fast-hadamard-transform/README 2.md | third-party/fast-hadamard-transform/README.md | 与固定版本不同，另一版本身份尚未完全确定 |
| third-party/nvbench/README 2.md | third-party/nvbench/README.md | 与固定版本不同，另一版本身份尚未完全确定 |
| third-party/nvbench/.clangd 2 | third-party/nvbench/.clangd | 与固定版本不同，另一版本身份尚未完全确定 |
| third-party/nvbench/.clang-format 2 | third-party/nvbench/.clang-format | 与固定版本不同，另一版本身份尚未完全确定 |
| third-party/nvbench/CMakeLists 2.txt | third-party/nvbench/CMakeLists.txt | 与固定版本不同，另一版本身份尚未完全确定 |
| third-party/nvbench/.gitignore 2 | third-party/nvbench/.gitignore | 与固定版本不同，另一版本身份尚未完全确定 |
| third-party/nvbench/.git-blame-ignore-revs 2 | third-party/nvbench/.git-blame-ignore-revs | 与固定版本不同，另一版本身份尚未完全确定 |

### 本地链接候选

101 处是引用出现次数，含重复，并不是 101 个缺失正文文件。路径存在性扫描覆盖本项目 README 与 Markdown，另检查两份 RST 的 image/figure 指令；不等同于运行全部网站链接、API 或源码内部引用。

| 条目 | 文件 | 出现次数 | 未在本地解析的目标 |
| --- | --- | ---: | --- |
| [2026-09-21/cuda-async-copies](../../raw/articles/2026-09-21/cuda-async-copies/README.md) | README.md | 1 | `../02-basics/nvcc.html` |
| [2026-09-21/cuda-programming-guide](../../raw/articles/2026-09-21/cuda-programming-guide/README.md) | README.md | 1 | `02-basics/nvcc.html` |
| [2026-09-21/cuda-programming-model](../../raw/articles/2026-09-21/cuda-programming-model/README.md) | README.md | 1 | `../02-basics/nvcc.html` |
| [2026-09-21/cuda-simt-kernels](../../raw/articles/2026-09-21/cuda-simt-kernels/README.md) | README.md | 1 | `nvcc.html` |
| [2026-09-21/cuda-tile-kernels](../../raw/articles/2026-09-21/cuda-tile-kernels/README.md) | README.md | 1 | `nvcc.html` |
| [2026-09-21/cutlass-blackwell-gemm](../../raw/articles/2026-09-21/cutlass-blackwell-gemm/README.md) | README.md | 1 | `../pythonDSL/cute_dsl_general/dsl_dynamic_layout.html` |
| [2026-09-21/cutlass-cute-dsl](../../raw/articles/2026-09-21/cutlass-cute-dsl/README.md) | README.md | 1 | `cute_dsl_general/dsl_dynamic_layout.html` |
| [2026-09-21/cutlass-efficient-gemm](../../raw/articles/2026-09-21/cutlass-efficient-gemm/README.md) | README.md | 1 | `../pythonDSL/cute_dsl_general/dsl_dynamic_layout.html` |
| [2026-09-21/cutlass-gemm-api](../../raw/articles/2026-09-21/cutlass-gemm-api/README.md) | README.md | 1 | `../pythonDSL/cute_dsl_general/dsl_dynamic_layout.html` |
| [2026-09-21/cutlass-grouped-schedulers](../../raw/articles/2026-09-21/cutlass-grouped-schedulers/README.md) | README.md | 1 | `../pythonDSL/cute_dsl_general/dsl_dynamic_layout.html` |
| [2026-09-21/cutlass-layouts-tensors](../../raw/articles/2026-09-21/cutlass-layouts-tensors/README.md) | README.md | 1 | `../pythonDSL/cute_dsl_general/dsl_dynamic_layout.html` |
| [2026-09-21/cutlass-profiler](../../raw/articles/2026-09-21/cutlass-profiler/README.md) | README.md | 1 | `../pythonDSL/cute_dsl_general/dsl_dynamic_layout.html` |
| [2026-09-21/cutlass-quickstart](../../raw/articles/2026-09-21/cutlass-quickstart/README.md) | README.md | 1 | `../pythonDSL/cute_dsl_general/dsl_dynamic_layout.html` |
| [2026-09-21/cutlass-synchronization](../../raw/articles/2026-09-21/cutlass-synchronization/README.md) | README.md | 1 | `../pythonDSL/cute_dsl_general/dsl_dynamic_layout.html` |
| [2026-09-21/cutlass-tile-iterators](../../raw/articles/2026-09-21/cutlass-tile-iterators/README.md) | README.md | 1 | `../pythonDSL/cute_dsl_general/dsl_dynamic_layout.html` |
| [2026-09-24/modelopt-ptq-guides](../../raw/articles/2026-09-24/modelopt-ptq-guides/README.md) | pages/hf-ptq.md | 23 | `../deepseek/README.md`<br>`../kimi/README.md`<br>`./hf_ptq.py`<br>`./../llm_qat/README.md`<br>`./scripts/huggingface_example.sh`<br>`../../modelopt_recipes/configs/ptq/presets/model`<br>`./example_utils.py`<br>`../../modelopt_recipes/model_type/README.md`<br>`../megatron_bridge/README.md`<br>`../../modelopt/recipe/config.py`<br>`../../modelopt_recipes/general/auto_quantize`<br>`./scripts/parser.sh`<br>`./slurm/multinode_fsdp2_ptq.slurm`<br>`../llm_eval/README.md`<br>`../vllm_serve/README.md`<br>`../../tests/examples/hf_ptq/test_deploy.py`<br>`../../modelopt/torch/utils/mlflow.py`<br>`../benchmark.md` |
| [2026-09-24/modelopt-ptq-guides](../../raw/articles/2026-09-24/modelopt-ptq-guides/README.md) | pages/ptq-recipes.md | 6 | `general/ptq/`<br>`general/qad/`<br>`models/README.md`<br>`../examples/megatron_bridge/tutorials/Qwen3.6-35B-A3B/README.md`<br>`README.md` |
| [2026-09-24/nvidia-dynamo-architecture](../../raw/articles/2026-09-24/nvidia-dynamo-architecture/README.md) | pages/compatibility.md | 8 | `/dynamo/reference/release-artifacts`<br>`/dynamo/dev/cli/installation/install-dynamo`<br>`/dynamo/reference/model-early-access-builds`<br>`/dynamo/dev/knowledge-base/modular-components/backends/tensor-rt-llm/known-issues`<br>`/dynamo/dev/reference/api/kubernetes/full-api-reference#componentcheckpointconfig`<br>`/dynamo/reference/releases-data` |
| [2026-09-24/nvidia-dynamo-architecture](../../raw/articles/2026-09-24/nvidia-dynamo-architecture/README.md) | pages/disaggregated-serving.md | 1 | `/dynamo/kubernetes/operations/performance-tuning` |
| [2026-09-24/nvidia-dynamo-architecture](../../raw/articles/2026-09-24/nvidia-dynamo-architecture/README.md) | pages/overall-architecture.md | 16 | `/dynamo/knowledge-base/modular-components/kvbm/overview`<br>`file:assets/img/dynamo-architecture.svg`<br>`/dynamo/knowledge-base/concepts/system-architecture/sidecar-backends-experimental`<br>`/dynamo/knowledge-base/concepts/system-architecture/architecture-flow`<br>`/dynamo/knowledge-base/concepts/communication-planes/request-plane`<br>`/dynamo/knowledge-base/modular-components/planner/planner-design`<br>`/dynamo/kubernetes/fault-tolerance/overview`<br>`/dynamo/knowledge-base/kubernetes/kubernetes-operator/gateway-api-routing`<br>`/dynamo/knowledge-base/modular-components/router/router-design`<br>`/dynamo/knowledge-base/concepts/communication-planes/discovery-plane`<br>`/dynamo/knowledge-base/concepts/communication-planes/event-plane`<br>`/dynamo/knowledge-base/kubernetes/multinode/grove` |
| [2026-09-24/ray-serve-llm-architecture](../../raw/articles/2026-09-24/ray-serve-llm-architecture/README.md) | pages/request-routing.md | 8 | `../../api/doc/ray.serve.request_router.RequestRouter.rst`<br>`../../api/doc/ray.serve.request_router.RequestRouter.choose_replicas.rst`<br>`../../api/doc/ray.serve.request_router.RequestRouter.on_request_routed.rst`<br>`../../api/doc/ray.serve.request_router.RequestRouter.on_replica_actor_died.rst`<br>`../../api/doc/ray.serve.request_router.LocalityMixin.rst`<br>`../../api/doc/ray.serve.request_router.MultiplexMixin.rst`<br>`../../api/doc/ray.serve.request_router.FIFOMixin.rst` |
| [2026-09-24/tensorrt-llm-quantization](../../raw/articles/2026-09-24/tensorrt-llm-quantization/README.md) | pages/quantization.md | 2 | `kv-cache-compression.md` |
| [2026-09-24/vllm-v1-architecture](../../raw/articles/2026-09-24/vllm-v1-architecture/README.md) | pages/architecture-overview.md | 21 | `../assets/design/arch_overview/entrypoints.excalidraw.png`<br>`../api/README.md#offline-inference`<br>`../../vllm/entrypoints/llm.py`<br>`../../vllm/entrypoints/cli/main.py`<br>`../serving/online_serving/README.md`<br>`../../vllm/entrypoints/launchers/api_server`<br>`../../vllm/v1/utils.py`<br>`../../vllm/v1/engine/core.py`<br>`../../vllm/v1/engine/utils.py`<br>`../../vllm/v1/executor/multiproc_executor.py`<br>`../../vllm/v1/worker/gpu_worker.py`<br>`../../vllm/v1/engine/coordinator.py`<br>`../assets/design/arch_overview/v1_process_architecture_tp4.png`<br>`../assets/design/arch_overview/v1_process_architecture_tp2_dp4.png`<br>`../configuration/optimization.md#cpu-resources-for-gpu-deployments`<br>`../assets/design/arch_overview/llm_engine.excalidraw.png`<br>`../../vllm/engine/llm_engine.py`<br>`../../examples/applications/api_server/server.py`<br>`../../vllm/engine/async_llm_engine.py`<br>`huggingface_integration.md`<br>`../assets/design/hierarchy.png` |
| [2026-10-04/pytorch-cpp-cuda-operators](../../raw/articles/2026-10-04/pytorch-cpp-cuda-operators/README.md) | pages/cpp-extension.md | 1 | `[https://developer.nvidia.com/nvshmem](https://developer.nvidia.com/nvshmem)` |

### 检查方法与复用边界

- 用 YAML 实际解析器核对元数据；逐个读取登记文件计算 SHA-256。登记值与当前文件一致只能证明这两者一致。
- 用 pypdf/pdfinfo 读取 PDF，Pillow 解码本地图像，并以 Poppler 重渲染页图进行粗略内容对照。
- 对归档逐成员解压，对已取得展开文件比较内容、类型、符号链接及执行位；对保留 Git 对象读取固定 tree/blob，不执行上游代码、hook 或内容过滤器。
- 通过固定版本远端 Git tree 重核子模块 pin 和可读取的展开文件；对 LFS 实体按指针适用时核对摘要与大小。
- 检查本地 Markdown/RST 图像与链接；只对已指出的 Dynamo 签名图片做远端失效确认，没有将未检查外链标为可用。
- 用原文与代表 PDF 页做摘要、公式和提取质量抽查，保留全量机械检查与抽查语义检查的区别。
- GPTQ 评议重新检查全部保存记录；LUQ 演示视频校验值一致、文件识别为 MP4，本轮未进行完整播放或逐帧检查。

一次性检查脚本和详细中间结果存于 /private/tmp/quant-raw-review-20261005/，没有将它们变成项目新的维护体系。本文保存可长期讨论的结论与逐条覆盖范围；临时目录本身不作为永久交付保证。

## 规范修订与修复复核

更新日期：2026-10-06。按用户“先修订文档、再修复确定问题，公式必须重新录，图片可只拆关键图”的授权执行。同日按用户补充明确验收边界：raw 保证忠实转录，原文疑点标注即可，不重做作者证明。以下是修复后的状态；前面的只读快照继续用于说明问题曾经存在，不作为当前待办清单。

### 现在可以确认什么

270 个正式条目仍为 142 篇论文、69 个文章/文档和 59 个代码仓库。已修复确认的身份登记、缺图、阅读链接、部分文字与公式提取、源码额外副本及混杂版本问题；142 篇论文均有可复制的公式阅读入口。公式不再仅依靠 PDF 文字层中损坏的符号。

本次 raw 验收以来源保存、转录忠实和读本显示准确为目标。原文数学有疑点时，记录位置与简短理由；已确认的原文错误保留原式并单列更正。标注疑点即可，不要求替作者重新证明定理，也不将作者证明是否成立作为转录完成的条件。源码保存核查与运行或性能验证仍分别记录。

### 规范为什么这样修订

正式规则已写入 [raw/README.md](../../raw/README.md)。现有三类目录、单条目 README 和 LOG 足以承接此次修复，没有增加 manifest、评分系统或另一套管理目录。

| 修订 | 理由与影响 |
| --- | --- |
| 条目说明分别写原件保存、阅读呈现、公式对照与原文疑点 | 一个“完整”状态会把文件存在与科研正确混为一谈。按实际证据说明范围，可直接判断一份资料现在能用于什么。 |
| 正文、行内、附录及图内公式须可复制；保留固定来源位置、原标签与宏上下文 | 单纯抽取 PDF 文本会破坏上下标、根号、矩阵和阅读顺序。恢复源码或依据原件转录，才能追溯具体表达；读本编号不能猜成作者式号。 |
| 来源原式、通用 Markdown 显示和整理者注记分开 | 上游也有笔误或推导疑点。原式保留，疑点标明位置和简短理由即可；已有依据的更正单列，不要求 raw 重做作者证明。 |
| 关键图独立保存，其余视觉内容由完整原件覆盖 | 独立结构图、位布局图与算法图便于理解和引用；拆出全部装饰图片只增加副本，不能提高公式正确性。图内数学仍须核对。 |
| 按上游地址与完整 commit 去重、核对子模块与额外文件 | 同名文件和短版本号不足以区分固定证据。父仓库归档也不能替代子模块源码。已知不同版本先保留归属证据，再清理混杂副本。 |
| 正文相对链接按固定来源解析，临时签名图片不能成为唯一依据 | 离线资料应能打开已保存内容；未保存的上游页面应保留真实外链，不能伪装成本地文件。 |

研读目录放在 [wiki/READING.md](../../wiki/READING.md)，由 [Wiki 规范](../../wiki/README.md) 维护。它记录确定版本、已研读范围、Wiki 承接及未验证内容，当前覆盖 130 份来源，其中 70 条有范围依据、60 条范围待核实。没有用目录存在或公式提取完成代替“全文已研读”，也没有在 raw 复制反向列表。

### 确认的保存与阅读问题已怎样修复

| 对象 | 修复与复核结果 |
| --- | --- |
| [fast.cu](../../raw/repositories/2026-09-24/fast-cu/README.md) | 将 41 位错误标识修正为 `2dfe5e26aecfd9e5f27bf9d5837deea01acda24b`；保留原归档及其 PAX/目录标签差异，没有重打原包。 |
| DQ、FoldQuantVLA 与 GPTQ 评议的条目说明 | 补清父仓库 gitlink、子模块版本和评议继承元数据的关系；没有把父包误当子模块文件验收。 |
| 8 个代码条目的 2,553 个额外文件 | 实际读取后确认 2,531 个与固定版本副本相同；22 个不同副本先保存到 QuaRot 的 `mixed-files-2026-10-05.tar.gz`，再清理额外路径。14 个 CUTLASS 副本与独立收录的 e8ecfad7 版本相符，另 8 个来源未定，未猜定其归属；正常固定版本文件不改写。 |
| 网页关键图 | 补齐 vLLM 的 5 张固定 commit 架构图、FSDP 的 3 张图及 Dynamo 的 4 张 SVG，共 12 个已确认缺口；另保存正文依赖的 285 个图片资源，合计 297 个文档图像资源。FSDP 补图来自当前官方资料，与旧正文快照的历史一致性尚未建立，已在条目注明。 |
| 本地阅读导航 | 修复 28 个条目、31 个文件的 376 处导航引用；代码块未改写。未收录的上游文件转为适当的来源链接，已保存附件使用本地入口。 |
| TeX 来源 | 补取 48 个确定版本源包，当前 140 篇有源包，其中 138 篇是可恢复的 TeX，2 篇包内仍为 PDF；没有 TeX 的 PACT、LQ-Nets 依据 PDF 转录。 |
| [GPTQ](../../raw/papers/2026-09-21/gptq/README.md) | 同包包含旧 `arxiv.tex` 与最终 `gptq.tex`，读本原先选错入口。依据固定 v2 PDF 的会议版式、摘要数值和算法图改为最终稿，全部 107 处数学重新恢复，关键图及定位同步修正。旧稿没有被误判成作者数学错误。 |
| [Qwen3 训练流程图](../../raw/papers/2026-09-21/qwen3/README.md) | 原 PDF 的 Type0 字体在 Poppler 渲染中丢失标签，改用本机 CoreGraphics 重渲染流程图及论文第 9 页预览；原 PDF 保留。 |
| [困惑度文档](../../raw/articles/2026-09-22/perplexity/README.md) | 依据可识别的上游快照恢复两处丢失表达；同时更正示例的标签移位：所印 token 示例的可评测目标数是 2，而原代码把其中 1 个屏蔽。这里只核算 toy labels，没有执行 GPT-2。 |
| [MBQ](../../raw/papers/2026-09-21/mbq/README.md)、[QIG](../../raw/papers/2026-09-21/qig/README.md)、[XOR-Trellis](../../raw/papers/2026-10-03/xor-trellis/README.md) | 核对参数化输入宏及图路径宏。MBQ 实际 22 个案例输入、8 张案例图片未增加数学式；QIG 图名 `.pdf` 与实际 `.pdf.pdf` 不一致，原源码不改写；XOR 的三张附录图按宏的实际目录取得，未把宏路径误报为本地缺失。 |
| 关键图的图注关联 | 改正 24 处不相属或未取到的图注定位，覆盖 16 个条目；正确识别 wrapfigure 与 table/minipage 中的 captionof。先前整篇或邻近图注被归给单图属于本次生成读本的问题，已修复，原作者源文件不改写。 |

### 公式与图片的实际覆盖

| 范围 | 本次产物与核对 |
| --- | --- |
| 138 篇 TeX 论文 | 恢复 41,902 个显式来源位置，包含正文、实际包含文件、附录、表格/图注中的独立与行内数学。每处保留源码范围和 SHA-256、原标签、宏上下文；其中 41,896 处为数学表达，6 处为绘图坐标或插图容器，明确排除，未虚报公式错误。 |
| 4 篇 PDF 论文与 MIT 讲义 | LLM-MQ、Integrated Gradients、PACT、LQ-Nets 的 266 个表达依据固定 PDF 转录；MIT 6.441 的 36 个表达另录。页、原式号或标注、必要条件和原件入口保留。 |
| 文章与多页讲义 | 10 个含显式数学的 Markdown/网页条目恢复 95 个表达；3 个课程条目的 16 份 PDF 共 827 页完成视觉粗筛，299 个数学表达转录到有页定位的读本，保留 108 张公式/关键图整页预览。普通程序代码与数学表达分别处理。 |
| 论文源图 | 解析 1,335 处图像引用，对应 1,308 个确定源文件，包含已解析的动态案例宏。所有已识别源图完成粗筛；图内公式和必要布局另作定向放大核对。当前保存 472 个固定源图文件及 9 个有 PDF 页与裁切范围的关键图，未要求复制全部装饰图。多页源图的第二页也核对并补了必要预览。 |
| 图内读本 | 恢复 187 个读本表达块，包含原标注、明确说明的流程节点重构及更正；有些与正文或不同版本的同图重复，不能解释为 187 个新的作者独立公式。数学图和必要条件在各条目 formulas.md、figures.md 中逐项相联。 |

41,902 是出现位置数量，包含重复表达、变量和行内条件，不是 41,902 条独立定理；不能用数量替代核查质量。自动编号只用于本地定位。来源的原式与兼容显示通过 MathML 对照，字形与科研条件另外检查；无法仅由图确定的求和范围、转置或完整机制没有猜填。

### 原文数学发现与处理

以下保留有确定依据的实质问题。表内“更正”限定于相应公式或条件，不自动否定作者实现、整篇结论或实验。

| 资料与定位 | 核实问题、处理与限制 |
| --- | --- |
| [LLM-MQ](../../raw/papers/2026-09-21/llm-mq/formulas.md)、[Integrated Gradients](../../raw/papers/2026-09-21/integrated-gradients/formulas.md)、[LQ-Nets](../../raw/papers/2026-09-22/lq-nets/formulas.md)、[PACT](../../raw/papers/2026-09-22/pact/formulas.md) | Taylor 符号和余项、IG 的非零/抵消与绝对连续前提、矩阵逆的秩条件及 PACT 的 STE 条件分别写清。来源原式保留，不能把代理梯度当 round 的普通导数。 |
| [PagedAttention](../../raw/papers/2026-09-21/pagedattention/formulas.md) M00050 | 块内 exp 与求和次序错误；零 logits 的两 token 反例中，原式权重各为 1，正确为各 1/2。讲义中同一错误另作标记。 |
| [QuIP](../../raw/papers/2026-09-21/quip/formulas.md) M00303 | 附录式 (9) 的量化损失缺少减 W；一维精确重构本应为 0，原式产生非零损失。 |
| [PV-Tuning](../../raw/papers/2026-09-22/pv-tuning/formulas.md) M00249 | 展开损失与 smoothness 二次代理一般是上界不等式，不能写恒等式；以二维例子核对。正确的单调改进条件另保留，没有据此否定下降主张。 |
| [LLM.int8()](../../raw/papers/2026-09-21/llm-int8/formulas.md) M00018/M00019 | 仿射零点与整数范围不相合；给出明确范围的修正及 [1,2] 的数值反例，避免无偏移码溢出。 |
| [Integer-only Quantization](../../raw/papers/2026-09-21/integer-only-quantization/formulas.md) M00084 | clamp 左侧输入 r，右侧却用自由变量 x，统一为 r；量化间距要求 b>a、整数 n>1。 |
| [BRECQ](../../raw/papers/2026-09-21/brecq/formulas.md) M00108 | 饱和步长梯度混用层数 n 和激活上界，且缺整数上界因子；针对明确的 ReLU 无符号量化器重推。Qp=3 的饱和有限差分为 3，原式一般不能取 1。 |
| [AQLM](../../raw/papers/2026-09-21/aqlm/formulas.md) M00056/M00067 | 两行连接导致交叉项符号错误；应为 −2 交叉项。一维精确重构损失 0，错误双负号式给 4。 |
| [LLM-QAT](../../raw/papers/2026-09-21/llm-qat/formulas.md)、[QSLAW](../../raw/papers/2026-09-21/multimodal-qat/formulas.md)、[SignRound](../../raw/papers/2026-09-22/signround/formulas.md) | 非对称整数码、实数偏移与整数零点未一致使用，反量化漏偏移或漏乘步长；给出显式定义与零输入反例。零点为 0 的特殊配置及实际代码另行判断。 |
| [EfficientQAT](../../raw/papers/2026-09-22/efficientqat/formulas.md) | 附录把前向的 2^N−1 写成 2^(N−1)，饱和零点梯度漏步长 s；按指定 STE 推导并保留边界约定。只对饱和普通导数使用有限差分，未把 STE 内部当真实微分。 |
| [LSQ+](../../raw/papers/2026-09-22/lsq-plus/formulas.md) | PDF 的微分变量与同版 TeX 不一致；宏的星号签名造成读本括号错误已修复。偏移梯度全为零的推论还缺上饱和条件，有明确反例。 |
| [ResQ](../../raw/papers/2026-09-21/resq/formulas.md) | 块矩阵相加的维度歧义、正交残差范数应为平方和、trace 缺平方根、期望错误线性相减、协方差方向与元素计数分别纠正。Theorem 4.2 标注“原文证明存在疑点”，相关步骤问题与原式一并保留；未据此断言定理结论必为假，不要求继续重推才能完成 raw 核查。 |
| [QJL](../../raw/papers/2026-09-22/qjl/formulas.md) | 自由下标、尾界正指数、常数与范数半径的量纲、算法索引以及图内缺 sqrt(π/2) 分别核对。换成负号仍不足以修好原尾界：提供高斯固定向量下的独立 MGF 推导与安全界，并给出原常数的反例；attention 界限定固定 q/k、独立随机投影、范数与误差范围。 |
| [检索损害与量化](../../raw/papers/2026-09-23/quantization-retrieval-damage/formulas.md) | Conformal 上尾阈值应为第 k 小值，不能写第 k 大值；k=n+1 采用无穷阈值。n=5 的排序反例及实际 flip 构造显示错误阈值可超过 α；保证是交换性下的边际联合风险，不能误作条件风险。 |
| [Kitty](../../raw/papers/2026-09-21/kitty/formulas.md) | 图内 attention 分母 d 与同文正文 sqrt(d) 不符；保留原图，以 d=4 的概率差核实更正。 |
| [SmoothQuant](../../raw/papers/2026-09-21/smoothquant/formulas.md) | 图中变换后 X 的 AbsMax 被误标为缩放因子 (1,4,1,3)；逐项计算应为 (2,4,2,3)，同时验证变换前后矩阵乘法保持一致。 |
| [QServe](../../raw/papers/2026-09-21/qserve/formulas.md) | 保护范围推导方向及 127−17/2 的算术错误单列；以非零最近舍入尺度 s≤16、整数零点与 clip 范围重新给出 [-119,119] 的安全依据。图内 −5、−16 和 −2 的字节常量错误另修正，核对 32 bit 跨 byte 进位；未运行 PTX。 |
| [OSTQuant](../../raw/papers/2026-09-22/ostquant/formulas.md) | 附录右侧 S₂ 应为 X₂，图内非线性分支混用原激活/缩放激活名称；以明确的 A/B 定义和逐元素非零尺度修正等价式，给出 σ(0.5) 与 σ(1) 的反例。 |
| [floating-point](../../raw/articles/2026-09-22/floating-point/formulas.md) | 示例 1.0008×x²−1 的残差为 1.60064×10^-3，原指数误写 −4。 |
| [GPU Mode](../../raw/articles/2026-09-24/gpu-mode-lecture-materials/formulas.md)、[IIT Bombay](../../raw/articles/2026-09-24/iit-bombay-systems-for-ml-2026/formulas.md)、[Microsoft](../../raw/articles/2026-09-24/microsoft-system-for-ai/formulas.md) | 打包图漏 F/重复 G；算术强度 100 与 1000；GHz 到 TFLOP/s 的换算；链式导数 2.805 与 1.972602；两次 GEMM 输出；0x72 与 144；FP32 最小有限值、FP16 范围；线程/块索引次序等分别用整数、矩阵或有限差分核查。合法 MMA 形状仍依 dtype、布局和 ISA。 |

公式复核还澄清了未必构成原文错误的范围，例如：HELM 可见像素与隐藏重叠文字、QERA 的对数轴标签、SpikeLLM 的类别标签、VEQ 的两种梯度比率口径、SqueezeLLM 的有符号分解与 magnitude、RMSE 在零残差处的不可微性、QServe 的缩放对非线性分支的条件、排序间隔在 ties 下的边界。未把正常简写或不同统计口径机械判为错误。

### 修复后的验证

- 270 份正式条目元数据及 2,224 个 README 登记校验值通过检查；805 个本地阅读 Markdown 的资源路径扫描未报告断链或外部图片依赖。未把上游源码内部链接和全部网络 URL 的可用性算入这一结果。
- 41,902 个来源位置逐一核对源码字符范围、摘要及展开副本与源包字节；41,896 个数学显示式与来源展开的 MathML 相合，6 个非公式容器明确排除，没有未解决的数学显示解析错误。该检查不等于作者所有推导成立。
- PDF 转录、文章/讲义重录、关键图内表达及新更正式均完成本地 KaTeX 渲染和相应可见页检查；矩阵乘法、码范围、位掩码、十六进制、误差反例及适用的有限差分独立核算。QServe 的保护范围另遍历 61,184 个整数/尺度/零点组合。
- 886 个论文关键图、文档资源、课程页图或人工裁切对象通过解码/适用的源包字节核对；其中 472 个论文源图文件与固定源包原字节相同，裁切图保留原 PDF 摘要、页和范围。原有完整 PDF/全页图继续保留。
- 父仓库 76,775 次归档比较、24,619 次 Git 对象比较，以及 57 处子模块 133,607 次文件比较均已取得实际字节。没有未读文件、额外文件或子模块 pin 差异。仍有原 README 已披露的 5 个历史排除项（2 个 DS_Store、1 个 swp、2 个 tmp），不静默把它们登记成完整树一致；固定归档/Git 对象保留。不同比较有重叠，235,001 次不能当独立文件数。
- 对首次审查已读取并登记的 313 个 PDF、源包或归档再次核对，原字节未改变；新取得源包和读本另外登记。没有执行来源 TeX、脚本、hook 或作者代码。
- Wiki 检查器通过 118 页、784 处来源登记、763 个本地来源、21 个固定外链及 2,123 个本地链接检查，无错误、警告或跳过。它仅验证结构与引用，不能证明 Wiki 科学性全部正确。

### 原文疑点与剩余事项

**原文疑点标注后即可完成相应的 raw 核查。** ResQ Theorem 4.2 标注“原文证明存在疑点”，保留具体位置与已有依据。QJL 与检索损害中的更正继续保留已写明的假设。原文疑点与转录一致性分别记录，不要求进一步重推定理，也不把这些疑点列为 raw 的阻塞事项。

**后续研读按实际用途开展。** 本轮未逐项校对全部 142 份中文摘要、所有实验表格和源码行为；课程的非公式图表也未全部转为结构化文本，原 PDF 保留。后续 Wiki 或研究任务可按实际需要核对相应公式、表格与实现条件，并更新同版本研读记录；这不构成 raw 必须重新证明论文的要求。

**P1：解决历史来源和可恢复副本的证据。** FSDP 补图与旧文档快照的一致性仍待历史依据；raw 本身被 Git 忽略，此次本地修复不随 Wiki 提交分发。若要确认异地可恢复，应实际检查保存的 PDF、源包与代码归档副本，不能用一次 Git push 替代。此次未迁移 raw，也未建立新备份系统。

我不建议再为每个条目增加多个状态字段、目录或常驻管理脚本。当前更有价值的是维护已有原件、可用读本、明确的更正和可追溯研读范围：这次发现的错误多数来自版本、提取、符号与条件，没有证据表明另一套目录架构能解决它们。临时检查实现留在 `/private/tmp/quant-raw-repair-20261005/`，长期依据和实际限制已写入条目、规范、LOG 和本报告。
