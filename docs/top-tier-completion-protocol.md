# 顶会级完整性协议与自主执行门槛

冻结日期：2026-08-28。该协议在查看 NoLiMa-Hard 和 LongBench 迁移结果前冻结，用于避免看到结果后改变主要假设、指标或样本选择。第一 Qwen seed 明确标为探索性 pilot；确认性结论来自后续预注册 seed 和第二模型。

**实现审计修正（2026-08-29 14:35 CST）。** 在已经看到 Qwen NoLiMa 和部分 LongBench 结果后、尚未执行规则确认评测和任何 Mistral 训练时，逐条复现 `Transformers 5.15.1 + Accelerate 1.14.0` 的实际 sampler，发现冻结的 100-step 运行只消费 1,024 行候选集中的 100 行。六条件仍使用完全相同的有序 fact/replica 身份，task/filler/length/replica 计数相同，且 independent/paired 的真实 prompt-token 总量逐 seed 完全相等；但是随机子集只有 85--89 个唯一事实，paired 中实际得到跨位置重复暴露的事实只有 11--15 个，没有任何事实覆盖完整四位置，且两 pairing 条件的位置直方图不完全相等。因此原 fixed-100 结果保留为“随机部分剂量/探索性”证据，不能支撑严格 block-complete pairing 的主因果表述。该问题来自与结果分数无关的数据加载审计，但修正是在部分 Qwen 结果可见后作出，论文必须明确标为 post-hoc implementation correction，不能伪称盲态预注册。

修正版主设计固定为每 seed 24 个事实 × 4 个 replica = 96 行：在 `task × filler × length × independent assigned-position` 的每个 stratum 用 seed 化 SHA-256 顺序选 3 个完整事实块。这样 independent 每个事实固定在一个位置、paired 每个事实恰好覆盖四位置；六条件均为 p000/p025/p050/p100 各 24 行、24 个事实各曝光 4 次，pairing 间原始输入 Token 总量差为 0。Qwen 三 seed 的输入 Token 总量分别为 777,397、777,462、777,413。batch=1、gradient accumulation=1、固定 96 optimizer steps 正好消费整个子集一轮；新 adapter、全部主要评测和第二模型都从 `checkpoint-96` 重跑。原 100-step 输出只用于透明敏感性对照，不进入 strict primary mean。

**修正版前瞻预算（2026-08-29 14:58 CST）。** 按当时旧 LongBench 剩余 1,796 条、已完成 Qwen seed-1 套件实测吞吐和 ¥2.78/GPU-hour 重新计量，strict block-96 完整顺序队列约需 135.52 GPU 小时：GPU 期望费用 ¥376.74，加 ¥15 存储/恢复预留后为 ¥391.74；25% contingency 上限 ¥489.67，向上取整建议准备 ¥500。单卡期望完成约 2026-09-04 06:29 CST，25% 墙钟上界约 2026-09-05 16:21 CST。该预算已包含 Qwen 与 Mistral 各三 seed 六条件、两家族 rule/NoLiMa/LongBench、两家族代表 seed 的 MMLU/IFEval/机制诊断，也包含安全切换前正在运行的旧 LongBench 尾段；它是储备上限而非消费目标。机器可读基线为 `configs/autodl_strict_block96_budget.json`：

```bash
python3 scripts/estimate_autodl_budget.py \
  --config configs/autodl_strict_block96_budget.json \
  --start-time 2026-08-29T14:58:00+08:00
```

以下 14:17:11 fixed-100 预算命令仅保留为审计历史，不再用于 strict 主队列的资金承诺。从 2026-08-28 19:48 CST 已完成状态起，原始保守预算按长上下文 2.135 秒/条、100-step 训练约 2.9 秒/step、5090 价格 ¥2.78/小时估算。到 2026-08-29 14:17:11，Qwen 探索性 seed 全部工作包、两个确认性 seed 的 12 个训练单元和确认性 NoLiMa 均已完成；确认性 LongBench 为 3,724/7,200 treatment rows（51.72%）。使用 `estimate_autodl_budget.py` 的 `--skip`、`--remaining-units` 和逐 workload `--seconds` 进行仅前瞻校准：已完成工作全部剔除；当前 LongBench 只计剩余 3,476 行并采用最新实测 1.489 秒/条；尚未执行的 Mistral LongBench 保留 1.300 秒/条；规则评测和尚未实测的 Mistral 长检索保留 2.135 秒/条。余下约 103.97 GPU 小时，GPU 小计 ¥289.05，加 ¥25 存储/恢复预留后的期望余额为 ¥314.05，含 25% 预备金的上限为 ¥392.56，估算器当前向上取整建议为 ¥400；已准备 ¥500 仍有约 ¥107.44 的预算上限余量。按当前假设的期望结束时刻约为 2026-09-02 22:16 CST，额外放宽 25% 的墙钟上界约为 2026-09-04 00:15 CST。机器可读历史基线在 `configs/autodl_top_tier_completion_budget.json`。

```bash
python3 scripts/estimate_autodl_budget.py \
  --config configs/autodl_top_tier_completion_budget.json \
  --skip qwen_seed1_rule_eval_remaining_at_1948 \
  --skip qwen_seed1_nolima_gate \
  --skip qwen_seed1_longbench_transfer \
  --skip qwen_seed1_mmlu_regression \
  --skip qwen_seed1_ifeval_regression \
  --skip qwen_seed1_nolima_mechanisms \
  --skip qwen_confirmatory_seed2_seed3_training \
  --skip qwen_confirmatory_nolima_eval \
  --remaining-units qwen_confirmatory_longbench_eval=3476 \
  --seconds qwen_confirmatory_longbench_eval=1.489 \
  --seconds mistral_longbench_eval=1.300 \
  --start-time 2026-08-29T14:17:11+08:00
```

## 一、论文最终只主张什么

核心处理设计是严格 matched 的 `pairing × supervision = 2 × 3`：

- pairing：`independent` 在相同总曝光与 Token 预算下独立抽取位置；`paired` 让同一事实显式跨位置成组出现；
- supervision：`answer`、`answer + evidence ID`、`answer + exact evidence quote`；
- 主张对象：答案正确性、证据可验证性、最弱位置、位置 gap，以及向未见长度、干扰、无词面重合和自然多文档 QA 的迁移。

不把“再次观察到 U 型曲线”写成主要贡献，也不把注意力权重直接写成因果机制。若外部分布上方法没有差异，结论降级为“低成本长上下文 SFT 可修复规则任务，但这种修复不自动迁移”，不选择性隐藏负结果。

## 二、确认性实验矩阵

### A. Qwen2.5-7B-Instruct

1. 固定三个训练/数据 seed：`20260825`、`20260826`、`20260827`；
2. 原 fixed-100 三 seed 作为随机部分剂量敏感性分析保留；主结果每个 seed 完成六个 block-complete 96-step、等事实、等曝光、等位置计数、等输入 Token 预算 QLoRA 条件；
3. 96-step 修正是在部分 Qwen OOD 结果可见后由实现审计触发，因此 Qwen 标为 corrective replication；尚未训练的 Mistral 按同一 96-step 设计作前瞻第二家族验证；
4. 六条件全部保留，不因 pilot 排名删掉 evidence-ID 中间条件；
5. 保存逐 step Loss、学习率、gradient norm、token accuracy、checkpoint 与输入/环境 hash。

100 steps 的原选择依据是 seed-1 训练在该点已出现近零 completion Loss，且多个 evidence-supervised 单元的总体答案分数接近天花板；继续优化更可能增加记忆风险，不能保证修复仍可见的 worst-position 失败。严格修正版不是按测试分数重选 checkpoint，而是把步数机械调整为能同时整除四位置和完整事实块的 96，使 batch-one 训练恰好消费 24 个完整块一轮；2,000-step scheduler horizon、60-step warmup、优化器和所有其他超参数保持不变。论文同时报告原 fixed-100 敏感性结果和修正版，禁止在两者中按分数挑赢家。

### B. 第二模型家族

默认使用无需 gated 账号的 `Mistral-7B-Instruct-v0.3`，保持约 7B 参数量和 32K 可用窗口，重新用其 tokenizer 生成并审计数据。至少运行三个 seed 的完整 2×3 矩阵。若模型或许可证不可用，替代品必须满足：公开权重、不同于 Qwen 的模型家族、至少 32K、约 7B–9B、可固定 revision；替换原因写入论文而不静默更换。

第二家族在生成数据前还必须通过机器可读的 chat-protocol 审计：原生模板保留 system/user/assistant 三个角色，训练全文必须以推理 prompt Token 为严格前缀，从而可安全构造 completion-only mask。审计记录模板哈希、渲染哈希、Token 数和固定 model revision；若原生模板不支持该协议，先冻结并测试显式兼容变换，再重新生成该家族的全部数据，不能混用两套渲染规则。

实际门禁结果已冻结：Mistral v0.3 的原生推理模板保留 system 指令，但带 assistant 的训练全文会丢弃该 system turn，因此原生协议不能安全构造 completion-only mask。项目选择 `merge-system-into-first-user-v1`，在训练和推理两条路径中都把首个 system 文本与首个 user 文本用两个换行符合并后再调用原生模板。真实 tokenizer 审计得到 prompt 28 tokens、全文 41 tokens、completion 13 tokens，全文严格以前 28 个 prompt tokens 开头，三个 sentinel 均保留；选择结果、模板/渲染哈希和固定 revision 写入 `chat_protocol_audit.json`，其 SHA-256 又被固定在 `configs/mistral7b_v03_model.json`。Qwen 数据保持原生协议，不做该变换。

### C. 评测层

| 层 | 数据 | 目的 | 论文角色 |
|---|---|---|---|
| 规则 matched | 4,200 条/模型条件、七位置 | 处理内有效性与位置因果控制 | 主消融；同时报告总体近天花板与剩余 worst-position 失败 |
| NoLiMa-Hard gate | 1,050 条/模型条件；10 cases × 5 books × 3 lengths × 7 positions | 无词面重合、位置严格可控、1K/8K/32K | 主要 OOD 位置结果 |
| LongBench v1 multi-doc | HotpotQA、2WikiMQA、MuSiQue 各 200 | 自然多文档和多跳迁移 | 自然任务结果；不声称位置因果 |
| 机制诊断 | locate-only、oracle-long、oracle-short | 检索、长干扰利用、纯推理解耦 | 支撑“为什么”而非只报分数 |
| 通用能力回归 | MMLU 全量 test + IFEval 官方 541 prompts | 排除知识/推理和指令遵循灾难性遗忘 | 2 pp 非劣门槛、安全性/外部效度 |

NoLiMa 派生数据遵循 Adobe Research License，仅用于非商业研究；发布时保留归属与许可，不重新分发超出许可范围的内容。LongBench 仓库为 MIT，但组成数据集仍保留各自条款，因此优先发布生成脚本、哈希和官方来源指针，而不是重新托管全部原文。

## 三、预注册指标与统计

主指标按以下顺序解释：

1. `mean worst-position answer accuracy`；
2. `mean answer position gap`，越低越好；
3. 总体 answer accuracy；
4. evidence-ID accuracy 与 exact-quote accuracy；
5. LongBench 官方英文 QA token-F1；
6. MMLU 格式鲁棒 option accuracy（JSON 有效性与截断率单列）与 IFEval 官方 strict/loose prompt/instruction accuracy；
7. JSON 有效率、输出截断率和不受支持引用率作为生成诊断。

规则位置实验的统计单位是同一问题、事实和 filler 的位置等价组，使用按任务/长度/filler 分层的配对 group bootstrap。NoLiMa-Hard 的 1,050 行只来自 10 个语义 case；其 factorial 与机制分析均按 `metadata.case_id` 聚类、按 one-hop/two-hop 任务分层重采样，并在每个被抽中的 case 内保持 5 本书、3 个长度、7 个位置和全部模型条件配对，不能把 150 个 case×book×length 组当成独立语义样本。自然 LongBench 使用按任务 × 长度档分层、模型条件间配对的问题 bootstrap。正式图表保存 5,000 次抽样索引、95% 区间、预注册主效应和交互效应，并按每个统计量使用 Holm 校正。多 seed 最终采用分层模型或 seed-level paired effect 汇总，不能把同一 seed 的数千条预测当成独立训练重复。

## 四、继续、停止与降级规则

- 规则测试饱和不会停止项目；它触发更难 OOD，而不是继续在同一规则测试上堆训练步数。
- NoLiMa 若能区分方法：完成全部 Qwen seed、第二家族和机制诊断。
- NoLiMa 若所有训练条件都接近 base：仍完成 LongBench 与机制诊断；论文重点转为“分布内修复不迁移”。
- NoLiMa 若所有训练条件都接近 100%：扩展官方 NoLiMa easy+hard 全集或更长长度，不把 gate 当最终主表。
- 自然 QA 下降超过 2 个百分点且置信区间排除 0：必须报告通用性代价，并检查混入少量一般 SFT 的保持基线；不删除失败条件。
- 第二家族方向与 Qwen 相反：报告模型 × 处理交互，不用只挑 Qwen 得出普遍结论。
- 只有数据、模型、脚本、逐样本结果、统计、图表和训练审计全部有 hash，才允许把状态标为 arXiv-ready。

## 五、arXiv-ready 硬门槛

- 完整英文稿：摘要、引言、相关工作、方法、实验、结果、机制、局限性、伦理/许可证、复现说明和参考文献；
- 主表和图中所有数字由机器可读结果自动生成，禁止手工抄写后失去追溯；
- 代码与数据卡说明随机种子、模型 revision、GPU、量化、LoRA、学习率、训练步数、解码和评分；
- 最终算力账本从显式正式目录的 canary/run metadata 去重生成，报告有效 GPU 时间与租价折算下界，并与平台实际计费范围明确区分；
- 可编译的自包含 LaTeX 源包，清除密码、绝对路径、内部日志和大模型权重；
- 论文结论通过 claim-to-artifact 审计，每一条定量主张能回溯到 JSON/CSV 与 SHA-256；
- 最终 PDF 与源码包通过独立编译和匿名/非匿名两种检查；
- arXiv 打包器必须同时验证 submission audit 零 pending 和 full-evidence manifest 的 `final_release_ready=true`，不能只凭可编译 PDF 打包；
- 作者姓名、顺序、单位、联系邮箱、ORCID、主分类和发布许可只在最终提交动作前确认。缺少这些身份信息不会阻塞实验、写作和 arXiv 源包准备，但会阻塞不可逆的最终提交。

## 六、当前执行状态

| 工作包 | 状态 | 完成定义 |
|---|---|---|
| Qwen seed-1 六条件训练 | 已完成 checkpoint-100 | 六个 adapter 与训练状态 hash 存在 |
| Qwen seed-1 规则正式评测 | 已于 2026-08-28 21:46:59 完成冻结门禁：七个 run 各 4,200 条，5,000 次分层配对 bootstrap、精确提示长度和 post-hoc 谱系审计全部 validated | 六条件各 4,200 条并通过聚合、精确提示长度与谱系审计 |
| NoLiMa-Hard 数据与 runner | fixed-100 seed-1 于 2026-08-29 00:09:41 完成；两个 post-pilot Qwen seed 于 2026-08-29 12:50 完成并 validated。历史阶段共 12 个 treatment runs × 1,050，逐 seed 按 10 个 `case_id`、任务 2/6/2 分层做 5,000 次聚类 bootstrap，并生成两个许可证安全失败目录；completion、归档及相邻 artifact SHA-256 全部通过，但 realized-subset 修正后均只作部分剂量诊断 | strict 阶段使用三-seed runner，逐 seed 做 5,000 bootstrap |
| LongBench 自然迁移数据与 runner | seed-1 已于 2026-08-29 01:44:21 完成并 validated：7 runs × 600、官方最大参考答案 token-F1、5,000 次任务×长度分层配对 bootstrap、图表、失败目录及事务性归档 SHA-256 均通过 | 7 runs × 600，官方 F1、5,000 bootstrap 与论文图 |
| MMLU 通用能力回归 | seed-1 已于 2026-08-29 02:58:41 完成 14,042 × 7；发现并修复 JSON/32-token 截断混入知识分数的评测混杂，冻结生成已按格式鲁棒选项提取重算 5,000 次 bootstrap，旧分析保留为诊断，新 completion/归档 SHA-256 已 validated | 全量 14,042 × 7 runs、格式鲁棒选项提取、配对 bootstrap 与 2 pp 非劣检验 |
| IFEval 指令遵循回归 | seed-1 已于 2026-08-29 05:47:49 完成并 validated：541 × 7 原始生成、官方 revision scorer、5,000 次 prompt 分层配对 bootstrap 与归档 SHA-256 全部通过；没有 Holm 显著差异，但 strict-prompt 的 2 pp 非劣均未被区间证明 | 官方 541 prompts × 7 runs、官方 strict/loose scorer 与 2 pp 非劣检验 |
| 历史 Qwen fixed-100 seed-2/3 | 两 seed 的 12 个训练单元已于 2026-08-29 08:49:11 全部 validated；两个 seed 各有 6 个 checkpoint-100/canary，1,200 个逐步 metric rows 无缺步或重复。历史 NoLiMa 于 12:50 validated；LongBench 于 12:50 开始，14:17:11 为 3,724/7,200 treatment rows（51.72%）。这些产物保留作透明诊断，不进入 strict 主均值 | 完成已付费 LongBench 尾段并冻结归档；禁止继续当作严格 pairing 证据 |
| 实际训练子集审计与 block-96 修正 | 2026-08-29 14:35 已精确复现三 seed 的 sampler：原 100-step 每 seed 仅 11--15 个事实得到 paired 跨位置重复、0 个完整四位置块，故 strict claim 不成立；已生成并本地 validated 三 seed 修正版，六条件各 96 行、24 个完整事实块、四位置各 24 行、pairing 输入 Token 差 0 | 原 fixed-100 标为探索性；`qwen_fixed100_realized_subset.json` 必须明确失败 strict gate，`qwen_block96_realized_subset.json` 必须三 seed 全部通过后才能进入主表 |
| Mistral 第二家族 | 数据准备及补充源谱系门禁已于 2026-08-28 22:31:53 validated：29GB 固定 revision 快照、三个 seed 的六条件独立分词、1,050 条模型专属 NoLiMa 和 1,350 条机制诊断均通过最终门禁。恢复过程固定仓库 commit `cb14780…6430` 与 dataset revision `378115b1…3ddd`，重建出与 Qwen 冻结 manifest 完全相同的 needle/book SHA-256，并把下载清单的 SHA-256 纳入 family completion | 三 seed × 六条件及同一 OOD 套件 |
| NoLiMa 机制诊断 | seed-1 已于 2026-08-29 07:39:13 完成并 validated：Base 与四个 factorial corners 共 6,750 条生成，free/locate/oracle-long/oracle-short 分解、oracle 去重、按 10 semantic cases 聚类 bootstrap 与归档 SHA-256 全部通过 | free/locate/oracle-long/oracle-short，oracle 去重，按 10 semantic cases 聚类 bootstrap |
| 通用能力回归 | Qwen seed-1 的 MMLU 与 IFEval 已完成；MMLU 格式鲁棒分数全部通过 2 pp 非劣，IFEval 未见显著差异但区间无法证明 strict-prompt 2 pp 非劣 | MMLU + IFEval 两类回归表与 2 pp 非劣检验完整 |
| 失败案例审计 | 已生成 Qwen 探索性 rule、NoLiMa、LongBench 及两个确认性 NoLiMa seed 的五份目录；最终加固发现其 CSV/Markdown 将少数组指纹写成 Python list 表示，安全 JSON、计数、比例、源哈希和论文分数不受影响。canonical byte 重算门禁现按设计拒绝这五份旧视图；重绘工具、单/多目录原子重打包与篡改单测已通过，待当前 LongBench 阶段结束后依次替换四个已完成包并重跑聚合审计。公开目录仍不含 benchmark 文本、原始标识符或绝对路径 | 最终 18 目录均由源 JSONL 行数/hash 与 JSON-derived CSV/Markdown 聚合门禁重算通过 |
| 顶层论文实验队列 | 旧 LongBench 已于 2026-08-29 15:39:55 CST 完成 7,200/7,200，两个 seed 的分析/失败目录和 8.9MB 归档 SHA-256 均通过；旧顶层队列未进入 rule 即退出。strict 队列于 15:46:54 恢复，15:47:18 开始 Qwen 三 seed checkpoint-96 训练，随后顺序重跑 NoLiMa、LongBench、MMLU、IFEval、机制、rule；不含轮询或电源动作 | strict Qwen 三 seed、strict Mistral 三 seed、跨家族数字和总证据审计全部 validated |
| 论文/arXiv 源包 | 构建工具链已通过真实 Tectonic/BibTeX 编译；当前脚手架为 13 页，提交审计 errors=0、pending=4。历史公开包演练选择 410 个文件并对 60 个文件做可移植性重写，净化树当时 121 tests passed、3 个许可证数据测试按设计 skipped；加入 strict block-96 谱系、证据语义、派生清单交叉绑定、一次性队列进度核算与审计后重哈希门禁后，当前完整研究树通过 152 tests。最终 strict evidence 关闭后仍须重新构建净化树并跑完整套件 | 自动数字、完整稿、编译与提交审计通过 |

历史部分剂量训练通过 `run_autodl_fixed100_multiseed_training.sh` 顺序完成并保留审计；它不再是论文主入口。严格训练由 block-96 家族队列顺序跑完所选 seeds；模型 29GB 级文件仅做一次全量 SHA-256 校验，后续条件在 manifest、文件大小和纳秒 mtime 均未变化时复用完整性证明。该优化只减少重复 I/O 和租机空耗，不跳过首次模型校验，也不包含关机命令。

运行环境也属于可复现协议的一部分。顶层队列的父进程保留含 vLLM 的 eval venv，训练子进程必须显式进入含 `datasets`、`trl`、`peft`、`bitsandbytes` 与 `accelerate` 的 train venv，子进程退出后不能污染后续评测环境。2026-08-29 07:39:18 的首次衔接因缺少这层隔离而在训练前门禁停止；没有 checkpoint 或部分 adapter 被接受。修复新增 train-venv 存在性/依赖检查和单测，经 Shell 语法与测试通过后于 07:50:59 从已验证的 seed-1 completion 恢复。恢复后的首单元及随后整个 `20260826` seed 都只训练到冻结的 step 100，并通过 checkpoint、metric trace、canary 和归档校验。

strict 队列第一次在 2026-08-29 15:42:31 CST 同样于 GPU 训练前失败关闭：旧 Qwen parent 预分词元数据没有 `chat_protocol` 字段，block-96 物化器曾把该缺省值写成 JSON `null`，而 preflight 只把“字段缺失”解释为 native，因而把语义等价的 null 错判为协议冲突。修复将 legacy 缺失/null 都显式归一为 `native-system-user-assistant`，并让新物化数据直接写出该值；相反，Mistral 的 `merge-system-into-first-user-v1` 仍必须精确匹配，不会被放宽。新增回归测试后完整测试为 140 passed，远端原失败条件以 96 行、固定 tokenizer 指纹和 cached model SHA-256 重新 preflight 为 `ready`，15:46:54 恢复队列，未接受任何部分 adapter。

两个历史 post-pilot seed 的训练诊断已经同步回本地：每 seed 六单元 × 100 steps，合计 1,200 行，没有缺失或重复 step。`20260826` 六单元的 last-100 median loss 为 $3.06\times10^{-5}$--$7.46\times10^{-5}$，`20260827` 为 $3.27\times10^{-5}$--$1.08\times10^{-4}$；所有单元的 last-100 median completion-token accuracy 都为 1.0，平均约 2.567 秒/step。11/12 个单元触发低于 $10^{-4}$ 的 loss 提示，12/12 都触发 token-accuracy 过拟合提示。它们只说明旧监督 completion 已拟合，并在 realized-subset 修正后降为部分剂量健康诊断；strict NoLiMa、LongBench/rule、第二家族和通用回归才负责判断泛化与遗忘，不能用训练 Loss 代替论文结果。

### 历史 Qwen fixed-100 NoLiMa 诊断（两个 post-pilot seed，不进入 strict 主均值）

固定 Base 的总体 answer / worst-position / position-gap / exact-quote 分别为 17.71% / 7.04% / 40.37 pp / 13.81%。两个历史 post-pilot seed 的 treatment 描述均值如下；Base 只评测一次，不能伪装成训练重复：

| 条件 | Answer | Worst-position | Gap ↓ | Exact quote |
|---|---:|---:|---:|---:|
| Independent + Answer | 14.05% | 2.78% | 45.56 pp | 0.00% |
| Independent + Evidence ID | 15.38% | 5.00% | 39.44 pp | 0.00% |
| Independent + Exact evidence | 18.14% | 5.19% | 49.26 pp | 10.52% |
| Paired + Answer | 13.57% | 2.22% | 42.04 pp | 0.86% |
| Paired + Evidence ID | 16.33% | 4.63% | 44.44 pp | 0.00% |
| Paired + Exact evidence | 18.52% | 6.11% | 51.85 pp | 14.43% |

长度分层的描述性 answer accuracy 进一步显示所有条件都随上下文增长而明显退化：Base 从 1K 的 38.29% 降到 8K 的 12.00% 和 32K 的 2.86%；六个 treatment 的两-seed 均值在 1K 为 26.71%--37.00%，8K 为 9.86%--12.86%，32K 只剩 1.86%--5.86%。Exact-evidence 在 32K 的两个 pairing 条件分别为 4.57% 和 5.86%，仍不足以消除长度退化，也不能把低个位数的绝对正确率写成鲁棒长上下文能力。

预注册的 seed-level 对比给出更稳妥的解释。Exact-evidence 相对 answer-only 在两个 seed 都提高总体答案和引用，二 seed 均值分别为 +4.52 pp 和 +12.05 pp，但 worst 只提高 +3.15 pp、Gap 反而增大 +6.76 pp；没有任何训练条件在两个 seed 都超过固定 Base 的 absolute worst-position。Evidence-ID 相对 answer-only 的方向较温和且一致：Answer +2.05 pp、Worst +2.31 pp、Gap -1.85 pp，但绝对均值仍未超过 Base。Pairing 对 Answer/Worst 的方向在两个 seed 反转，其均值为 +0.29/0.00 pp；只有引用支持相关增益较一致。因此当前 Qwen 证据支持“精确证据监督改善总体回答与可引用性，但收益偏向边缘位置，未修复语义 OOD 的位置鲁棒性”，而不支持“paired 是普遍赢家”或“规则任务修复自动迁移”。

这里的 seed-level Student-$t$ 区间只有两个独立训练重复，绝大多数都很宽；pairing 的 quote 效应区间相对较窄但效应很小，exact-evidence 的 quote 方向在两 seed 一致但区间仍宽。逐样本或 1,050 行不能被当成独立训练重复来制造显著性。上述结论明确标记为历史部分剂量诊断，最终措辞只由三个 strict corrective Qwen seed、三个 prospective Mistral seed、LongBench 和 matched-rule 汇总共同决定。

两个预先冻结规则的失败目录提供了独立的描述性一致性检查。`answer_changes_across_positions` 占可比较 run 内组的 89.33% / 93.90%；`edge_success_middle_failure` 为 46.29% / 50.00%，而反向的 `middle_success_edge_failure` 只有 12.00% / 12.76%。逐行 invalid JSON 为 0.57% / 0.39%，命中输出长度上限为 0.57% / 0.41%，不足以解释 83%--84% 的 answer errors。相对固定 Base 的配对比较中，Base-only answer success 为 4.56% / 3.65%，均高于 treatment-only 的 2.14% / 2.63%；treatment quote recovery 只有 1.46% / 1.29%。这些比例各自保留 row、group 或 cross-run 分母，只用于说明已报告的模式，不能混合成一个率或用来增加 post-hoc 显著性主张。

学习率曲线还暴露了一个容易误述但不改变 matched 对比的实现细节：`train_qlora.py` 保留原始 `max_steps=2000` 的 cosine scheduler horizon，并以 `warmup_ratio=0.03` 计算 60 个 warmup steps。历史部分剂量 callback 在 step 100 停止；strict block-complete callback 机械地改为 step 96，使一轮恰好消费全部 96 行，包含前 60 个 warmup step 和 36 个 post-warmup step。论文训练表、复现卡和曲线必须分别报告两者，主结果只读取 checkpoint-96，不能只写模糊的“3% warmup”或混用 checkpoint。

规则集的“32K”是窗口档位而不是允许挤占解码空间的字面输入长度。Qwen 冻结生成目标为 32,768，实际最长渲染输入为 32,581，加 176 输出上限后仍留 11 Token；Mistral 因 tokenizer 差异把同一档位的生成目标固定为 32,512（31.75K，与 Qwen 目标差 0.78%）。每个家族在数据完成门禁和每次 vLLM 加载前都会重新渲染全部提示，只有 `max_prompt_tokens + max_new_tokens <= 32768` 才能进入 GPU 评测；精确分布和最大样本 ID 写入审计文件，绝不依赖静默截断。

多 seed NoLiMa 使用 `run_autodl_nolima_multiseed.sh`：同一模型家族的 base 只生成一次，随后评测每个 seed 的六个 adapter；每个 seed 独立产生 factorial 分析和 5,000 次 group bootstrap。修正版论文主表通过 `generate_paper_results.py` 只读取 `primary_training_seed_summary=true` 的 seed-level JSON 自动生成；其中 Qwen source status 必须为 `corrective`、Mistral 必须为 `confirmatory`，并明确记录组合并非 confirmatory-only。脚本另行哈希探索性 fixed-100 Qwen 规则分析，仅生成正文中明确标注为 exploratory 的分数范围/worst-position 宏，不能把它混入任何 strict primary 均值。所有输入和生成的 TeX 都保存 SHA-256。除七个单元的分数表外，生成器还输出冻结的 pairing 主效应、监督主效应和 pairing×supervision 关键交互，以 independently trained seed 为单位给出 Student-t 区间；正文不能靠挑最高单元替代因子效应分析。

Qwen 探索性 seed 的规则结果冻结后，`run_autodl_qwen_seed1_completion_queue.sh` 依次完成 NoLiMa、LongBench、MMLU、IFEval 和 NoLiMa 机制诊断。它是实际实验工作队列，不是 watcher/定时器；任一门禁失败即保留可恢复结果并停止后续套件，全部通过后仍不会关机。

历史 `run_autodl_qwen_confirmatory_queue.sh` 的 fixed-100 输出只保留为诊断。主实验由 `run_autodl_qwen_block96_completion_queue.sh` 和对应 Mistral block-96 队列完成三 seed 六条件、NoLiMa、LongBench、规则评测、代表性回归/机制和 seed-level 汇总；Qwen 明确标为 corrective，Mistral 标为 prospective confirmatory。跨家族步骤自动生成论文数字，以及由 strict seed-level NoLiMa JSON 直接绘制的主位置图。该图同时生成 PDF/SVG/PNG、精确 CSV、alt text 和输入/输出哈希 manifest；视觉上裁剪到有效概率范围的宽区间不会覆盖 CSV 中的原始小样本 Student-t 区间。所有队列都是可恢复的实际工作队列，不包含轮询、定时或关机逻辑。

`run_autodl_strict_block96_full_queue.sh` 是修正后的唯一顶层主实验入口：它先要求历史 sampler 审计明确拒绝 fixed-100 strict claim，再顺序执行 strict Qwen、strict Mistral、跨家族论文数字、18 个显式选定的 strict 失败目录、有效 GPU 时间账本和总证据门禁。末尾重新核验所有训练/评测压缩包的相邻 SHA-256，并直接哈希跨家族统计、回归/机制报告、论文数字、主图/CSV/alt text 与 `compute_accounting`，生成 `results/full_paper_evidence_manifest.json`；只有该标签存在时才允许 `final_release_ready=true`。已验证阶段只有在 schema、状态、strict 标志和 completion 中记录的每个 artifact hash 都匹配时才跳过；否则重跑或失败关闭。队列中没有 `sleep`、轮询、cron 或系统电源动作，全部完成后实例仍保持运行以供最终产物审计。

进度与费用 ETA 使用 `estimate_eval_progress.py` 从实际 JSONL 行数和日志中最后一个实测速率计算，输出测量时间、各 run 行数、剩余行数、完成比例、预计完成时刻和剩余租金。该脚本只在人工/代理低频检查时运行一次，不驻留轮询。

论文工具链不再依赖本机预装完整 TeX Live：`install_tectonic.sh` 固定官方 Tectonic 0.17.0 Linux GNU archive 和解包后二进制的 SHA-256，安装到仓库外缓存；`build_paper_pdf.sh` 支持传统 LaTeX/BibTeX 或 Tectonic，记录引擎版本与输入/输出哈希。实际脚手架编译已自动运行 BibTeX，生成非空 PDF 和 `main.bbl`，且无未解析引用、Overfull 或 Underfull 警告。该通过只证明构建可复现，不会绕过仍含 `PENDING` 的提交门禁。

公开复现包也已在最终证据尚未生成前用临时 validated stub 做结构演练：选择逻辑忽略明确排除环境（例如 `.venv`）中的符号链接，但对任何将入包的源码符号链接仍 fail closed；测试夹具不再因示例密码字面量触发自身安全门禁；缺少不允许再分发的 NoLiMa/IFEval 原文时，相应数据一致性测试明确 skipped 并提示按冻结 manifest 重建。纳入两个 Qwen 确认 seed 的训练指标/曲线、确认性 NoLiMa 安全分析、CSV/Markdown canonical JSON 重算门禁、重绘工具以及单/多目录重打包审计后，最新净化包含 410 个文件，通过 Python 编译和全部 Shell 语法，并对 60 个文件进行了有清单记录的路径可移植性重写；实测 121 项通过，3 项仅因第三方 payload 未入包而按设计 skipped，且无 warnings，完整研究树为 124 passed、0 warnings。临时证据 stub 和预检目录已移入回收站，不能被最终打包门禁误认成真实结果；最终净化包仍必须在真实完整 evidence manifest 下重新运行全套测试。

最终算力披露不再靠人工相加：`summarize_compute_accounting.py` 只接受显式列出的 strict training/eval 根目录，主队列以 `--expected-training-step 96` 校验 checkpoint-96 canary，并接受通过 schema 门禁的 `selection_complete` vLLM/IFEval run metadata；复用的 Base 事件按不可变运行身份去重。它分别输出训练步数、评测样本数、有效 GPU 秒/小时及按冻结时价折算的成本下界。该数字明确只是 trainer/vLLM engine active-time 下界，不能冒充 AutoDL 账单；模型加载、CPU 分析、打包、实例闲置和平台计费粒度必须单列说明。完整队列关闭证据后才生成最终账本并写入论文。

初步 full-evidence manifest 不能直接触发正式公开包：最终复核必须用 `--require-evidence-label compute_accounting` 重跑证据审计，且该标签和对应文件必须同时存在，使 manifest 明确写出 `final_release_ready=true`。净化结构演练只能显式传入 `--preflight-allow-incomplete-evidence`，其 selection manifest 会永久记录 `evidence_completeness_preflight_bypass=true`；这种临时树即使测试通过也不能被当成最终 release。

探索性 Qwen 规则冻结结果进一步说明为什么不能只报总体答案分数：六个训练单元的总体答案正确率为 97.33%--99.95%，但 `paired_answer` 的 mean worst-position 只有 87.33%，position gap 为 12.67 pp；相反，`independent_evidence` 的对应值为 99.67% 和 0.33 pp。按预注册的“所有训练条件至少 98% 且范围小于 2 pp”判据，该结果并未形式化判为全面饱和，因为最弱训练单元仍有可区分误差。它仍是探索性 seed，不能据此选择获胜条件或进入确认性均值；其作用是证明总体均值会隐藏位置弱点，并为 NoLiMa 和确认性因子效应分析提供必要性。

同一探索性 seed 的 NoLiMa-Hard 结果给出相反的自然语义 OOD 图景：Base 的答案正确率为 17.71%、mean worst-position 为 7.04%、position gap 为 40.37 pp；六个训练单元的答案正确率仅为 13.14%--16.86%，mean worst-position 为 1.11%--5.56%，没有一个单元超过 Base。配对主效应在答案上为 +1.49 pp（10-case 聚类区间 +0.35--+2.60 pp，Holm $p=0.096$），在精确引用上为 +8.19 pp（+6.16--+10.25 pp，Holm $p=0.0064$），但答案 position gap 同时增加 2.47 pp，worst-position 没有可靠改善。证据 ID 与精确证据监督相对答案监督都显著提高引用，并分别把 gap 降低 11.30 与 12.22 pp，却没有提高总体答案正确率。该结果只能作为探索性证据：它表明规则分布内修复没有自动转化为语义检索，且可验证引用与答案鲁棒性是可分离目标；主结论仍必须由三个 strict corrective Qwen seed 与三个 prospective Mistral seed 决定。

机制诊断进一步把这种失败分解为“定位不到”和“长干扰下不会用”两个环节。Base 的 free-answer 为 17.71%，locate-only 精确引用为 5.71%，oracle-long 只有 0.67%，而移除干扰的 oracle-short 为 100%；四个 corner treatments 的 oracle-short 仍为 98%--100%，说明只给短金证据时的推理/答案生成基本完好。`independent_evidence` 与 `paired_evidence` 把 locate-only 精确引用提高到 27.52% 与 28.10%（支持证据定位率 36.57% 与 48.76%），却没有提高 free-answer，且 oracle-long 分别为 0% 与 1.33%。`paired_answer` 是唯一把 oracle-long 明显推到 19.33% 的单元（10-case 聚类 95% 区间 10.00%--30.67%），但其 free-answer 仍仅 16.86%，没有超过 Base。探索性解释因此是：短证据推理不是主要瓶颈；精确证据监督改善可验证定位但不足以解决长干扰利用，而 paired-answer 可能改善已提供证据在长上下文中的利用，却仍受原始定位限制。该分解只来自一个代表性 Qwen seed，不作为多 seed 因果结论，后续 Mistral 代表 seed 会按同一协议复现。

同一 seed 的 LongBench 自然迁移结果也不支持“规则集高分等于普遍提升”：Base 的三任务 pooled 最大参考答案 token-F1 为 43.68，六个训练单元为 40.81--44.32。`paired_answer` 与 `paired_evidence_id` 分别只比 Base 高 0.63 和 0.46 pp，区间均跨 0；三个 independent 单元和 `paired_evidence` 则低 1.44--2.87 pp。按预冻结的整体 contrast family 校正后，pairing 在 evidence-ID 监督下相对 independent 提高 2.88 pp（95% 配对 bootstrap 区间 +1.13--+4.73，Holm $p=0.030$），但 exact-evidence 相对 answer 的 pooled 主效应为 -2.27 pp（-3.84---0.77，Holm $p=0.034$）。这仍是单个探索性训练 seed 的问题级推断，不能代替训练-seed 复现；目前最稳妥的表述是，配对可能缓解 evidence-ID 条件下的自然任务损失，但更长的精确引用目标没有带来 pooled QA 收益。

MMLU 的首轮结构化分数曾显示 Base 49.15%、训练单元 64.71%--66.60%，但该差异被直接审计为格式混杂：Base 有 11,738/14,042 行撞到 32-token 上限，完整 JSON 率仅 73.69%，而高证据监督单元接近 100%。对同一冻结生成应用不要求 JSON 闭合的窄选项提取器后，99.98% 以上的行可提取选项，Base 为 66.93%，训练单元为 65.92%--66.96%；六个单元相对 Base 的配对区间下界均高于预注册 -2 pp 门槛，全部通过非劣。最弱的 `independent_evidence_id` 为 -1.01 pp（95% 区间 -1.32 至 -0.71 pp），说明未见灾难性遗忘，但也不把格式服从改善写成知识提升。旧分数保留在归档中作为评测失效诊断，论文表只读取格式鲁棒 completion 指向的新分析。

IFEval 给出更保守的指令保持结论：Base 的官方 strict-prompt accuracy 为 72.46%，六个训练单元相对 Base 的点差为 -2.03 至 -0.55 pp；在同一 541 prompts 上进行配对、按约束数分层的 5,000 次 bootstrap 后，所有 Holm 校正差异检验均为 $p=1$，没有检测到处理差异。然而每个 strict-prompt 差值的 95% 区间下界都低于冻结的 -2 pp 门槛，因此不能宣称已经证明非劣。该结果写成“没有显著退化证据，但 2 pp 保持保证仍不确定”，不把非显著误写成等价，也不为追求通过而事后改变 margin。

许可证安全的描述性失败审计进一步量化了这个差异，但不作为额外显著性检验：探索性 rule 中，11.83% 的 run 内位置等价组会随位置改变解析答案，8.64% 呈现“至少一个边缘成功、至少一个内部失败”；NoLiMa 对应比例升至 94.10% 和 47.43%。按 6,300 个 NoLiMa Base--treatment 配对比较计，Base-only 答案成功为 4.76%，treatment-only 成功仅 2.02%；rule 的 25,200 个比较中则分别为 0.12% 和 29.97%。LongBench 不具备位置等价组，目录只报告适用的行级和 Base--treatment 比较：4,200 个生成中 67.07% 未达到答案正确判据、0.90% 同时触及输出上限并形成无效 JSON，3,600 个匹配比较中 Base-only 和 treatment-only 成功分别为 4.69% 与 4.03%。这些不同 scope 的比例不互相相加，且探索性分类规则是在看到该 seed 后补充、随后才为全部确认性运行冻结；它只能说明为何需要跨 seed/模型验证，不能独立选择获胜 treatment。
