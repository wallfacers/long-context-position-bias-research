# 失败案例分类与公开审计协议

冻结补充日期：2026-08-29。该分类是在 Qwen 探索性结果之后、任何 Qwen 确认性或 Mistral 结果之前固定的描述性分析；它不新增显著性检验，也不能用于挑选主结果条件。

## 三种统计单位

| Scope | 分母 | 用途 |
|---|---|---|
| `row` | 目录内全部模型条件的生成行 | 输出格式、答案、引用及截断错误 |
| `group` | 每个 run 内至少含三个位置的等价组 | 位置曲线内部的反转和答案不稳定 |
| `cross_run` | Base 与每个 treatment 按 `sample_id` 对齐的比较 | 同一样本上的退化、恢复与引用恢复 |

不同 scope 的错误率不能相加或直接比较。一个样本也可以同时属于多个类别，例如既是 `invalid_json`，也是 `answer_wrong`；这些比例不是互斥分解。

## 固定类别

| 类别 | Scope | 判定 |
|---|---|---|
| `invalid_json` | row | 输出未通过冻结 JSON 解析门禁 |
| `answer_wrong` | row | 冻结答案评分为错误 |
| `answer_correct_quote_wrong` | row | 答案正确，但适用的精确引用评分失败 |
| `answer_wrong_quote_correct` | row | 答案错误，但适用的精确引用评分通过 |
| `generation_hit_length_cap` | row | `finish_reason=length` |
| `edge_success_middle_failure` | group | 至少一个首/尾位置成功，同时至少一个内部位置失败 |
| `middle_success_edge_failure` | group | 至少一个内部位置成功，同时至少一个首/尾位置失败 |
| `answer_changes_across_positions` | group | 同组不同位置的解析答案不一致 |
| `base_only_answer_success` | cross_run | Base 正确而 treatment 错误 |
| `treatment_only_answer_success` | cross_run | Base 错误而 treatment 正确 |
| `treatment_quote_recovery` | cross_run | Base 的适用引用失败而 treatment 通过 |

若结果行明确把引用标为不适用，引用失败和引用恢复类别都跳过该行；不允许把 N/A 编码成失败。受控规则集与 NoLiMa 的共同评测提示都要求精确引用，因此这些套件中的零引用分数是观察到的失败，不是因为 answer-only 或 evidence-ID treatment 在训练阶段没有使用精确引用目标。NoLiMa 的人工 evidence ID 则明确不适用。

## 选择与防泄漏

`scripts/analyze_failure_cases.py` 先穷举并统计所有候选，再按类别、run、group 和 sample 的固定顺序排序，每类最多保留五个审计样本。公开文件仅包含：

- 结构字段和布尔评分；
- 行、组、配对比较的分母与错误率；
- 源 JSONL 的行数和 SHA-256；
- sample/group/case/book 标识符的 SHA-256。

公开目录不包含 prompt、question、target、生成文本、解析答案、证据引用、benchmark 答案、原始标识符、内容派生哈希或绝对路径。源结果所有者仍可对私有标识符计算同一 SHA-256 以回溯案例。

最终 `scripts/audit_failure_case_catalogs.py` 使用重复 `--manifest` 接收冻结的 primary 集合，要求 18 个 strict block-96 目录全部存在，并重新计算每个 catalog 输出、错误率、源 JSONL 行数及哈希：Qwen 三 seed × rule/NoLiMa/LongBench 九份，Mistral 三 seed ×同三套评测九份。递归 `--results-root` 仍用于单一结果树的局部审计，但与显式清单互斥；因此同仓库里的 fixed-100 历史诊断不会膨胀或污染 primary 计数。任一缺失、重复、源结果变化、路径越界、符号链接或禁发字段都会阻断总证据 manifest 与公开包。

对补充协议冻结前已经完成并打包的探索性套件，只能使用 `scripts/retrofit_failure_catalog.sh` 回填：必须先验证 `RESULTS_READY_FOR_AGENT_REVIEW`、旧压缩包及相邻 SHA，并直接调用逐源 catalog audit 重算源行、源哈希和 canonical 视图；失败时不会改 completion 或旧包。新压缩包确认包含 catalog 后才替换旧包并生成新 SHA。正在运行或部分完成的目录不会被修改。

若既有安全 JSON 目录只存在 CSV/Markdown 表示层缺陷，`scripts/rerender_failure_catalog_views.py` 是唯一允许的重绘路径：它从 canonical license-safe JSON 精确重算两种视图，拒绝任何原始文本字段，并最后更新 manifest。聚合审计会自行重算预期字节，因此即使错误视图被重新哈希也无法通过；重绘后仍须使用上述 retrofit 流程重新打包并通过全量 catalog audit。

含多个 seed catalog、且 completion 已准确列出全部 manifest 的既有套件使用 `scripts/repack_failure_catalog_artifact.sh`：它在替换压缩包前重算指定数量的全部 catalog、验证 completion 的 exact manifest set，把聚合 audit 一并封入替换包，并原子更新相邻 SHA。任一目录不 canonical、来源哈希变化、引用集合不一致或归档成员不足都会保留原压缩包。
