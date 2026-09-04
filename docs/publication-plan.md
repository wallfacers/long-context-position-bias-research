# 论文发布与投稿计划

**预算口径修正（2026-08-29 14:58 CST）。** realized-subset 审计发现原 fixed-100 随机子集不是完整四位置事实块，因此下段 14:17:11 的 ¥400 快照只保留为历史记录。新的 `configs/autodl_strict_block96_budget.json` 已按旧 LongBench 剩余 1,796 条、完成套件的实测吞吐和 ¥2.78/GPU-hour 前瞻计算：完整 strict Qwen+Mistral 顺序队列约 135.52 GPU 小时，GPU ¥376.74，加 ¥15 恢复预留后的期望额 ¥391.74，25% contingency 上限 ¥489.67，建议储备 ¥500；期望完成约 2026-09-04 06:29 CST，上界约 2026-09-05 16:21 CST。该数字包含两模型各三 seed 主实验和代表性回归/机制诊断，是储备上限而非消费目标；已完成产物只有通过 schema/hash gate 才会跳过。论文发布、arXiv 上传本身不收取投稿费，会议注册、差旅或开放获取费用不计入 GPU 实验预算。

**历史预算（不再采用，2026-08-29 14:17:11）。** 当时尚未发现 realized-subset 缺陷，fixed-100 剩余工作被估为 103.97 GPU 小时，GPU 小计 ¥289.05、含恢复预留期望 ¥314.05、25% contingency 上限 ¥392.56。该快照及其 `--skip`、`--remaining-units` 假设仅用于解释为什么后来必须重新估价，不能用于当前充值、ETA 或论文算力披露；当前唯一资金基线是上段 strict block-96 的 ¥500 储备上限。

更新时间：2026-08-29。会议日期和注册费会变化，投稿前必须重新核对目标 venue 的官方 CFP。

## 投稿目标

本项目计划以论文形式发布。论文的核心主张不是再次发现 Lost-in-the-Middle/U 型曲线，而是：

> 在事实、曝光次数、上下文位置和训练 Token 预算匹配的条件下，分离训练样本的配对方式与监督粒度对答案正确性、证据可验证性和 worst-position 鲁棒性的因果影响。

这里的“匹配”只允许由 96 行 strict block-complete 主实验支持：每 seed 24 个事实各四次曝光、四位置各 24 行，pairing 间输入 Token 总量完全相等。原 fixed-100 结果因实际 sampler 只形成 11--15 个不完整跨位置重复事实而降为探索性敏感性分析；论文会公开该实现修正及触发审计，绝不把旧结果继续包装成严格配对实验。

发布分为两层：

1. 先按顶会/期刊标准完成三 seed、第二模型家族、自然语义 OOD、机制诊断、回归、统计和复现审计，再发布 arXiv 完整技术报告；
2. 以同一完整证据为基础准备 ARR/ACL Findings 或主会版本；Workshop 只作为可选反馈渠道，不再充当降低实验门槛的停止点。

用户已于 2026-08-28 将执行目标提升为“先达到顶会/期刊级完整性，再发布 arXiv”。因此当前不再以单模型、单 seed 的 Workshop 最低门槛作为停止点；确认性矩阵、第二家族、自然 OOD、机制诊断、通用能力回归和复现审计的冻结规则见[顶会级完整性协议](top-tier-completion-protocol.md)。

## 三种发布形式的区别

| 形式 | 是否同行评审 | 是否正式论文集 | 典型篇幅 | 适合当前项目的条件 |
|---|---|---|---|---|
| arXiv 预印本 | 否，仅做内容审核 | 否 | 无会议页数限制 | 实验和报告已经自洽、希望尽快公开与确立时间戳 |
| Workshop Paper | 通常是 | 取决于 archival/non-archival | 各 Workshop 自定，常见 4/8 页 | 主题匹配、实验尚聚焦，适合获取同行反馈 |
| ARR Short Paper | 是 | 被 venue 接收后进入论文集 | 4 页正文；最终稿通常多 1 页 | 有一个聚焦、完整、能在 4 页内证明的贡献 |

Workshop 必须逐个确认是否归档。若后续还要把同一核心结果扩写后投主会，优先选择 non-archival 展示；同一结果一旦作为 archival paper 正式发表，不能把高度重叠版本再次投稿。

## arXiv 准备清单

- 完整论文：标题、作者、摘要、方法、实验、相关工作、局限性和参考文献；
- 可编译 LaTeX 源码、`.bib`/`.bbl`、全部图片和必要的样式文件；
- 建议以 `cs.CL` 为主分类，是否 cross-list 到 `cs.LG` 在提交时再确认；
- arXiv 账号；首次向该领域投稿时可能需要 endorsement；
- 所有作者确认作者顺序、署名、单位、联系邮箱和公开版本；
- 选择不可撤销的发布许可前，核对未来目标会议/期刊政策；
- 清理源码中的注释、绝对路径、密钥、日志、内部链接和不需要的文件；
- 最终 evidence manifest 必须含经要求并哈希的 `compute_accounting`，且 `final_release_ready=true`；arXiv 打包器拒绝初步实验 manifest；
- 最终 PDF、图表数字和代码仓库版本相互一致。

arXiv 官方说明投稿对作者免费；文章会形成永久记录，可以继续发布 v2/v3，但不能把修订稿作为另一篇新投稿。官方入口：[Submission Guidelines](https://info.arxiv.org/help/submit/index.html)、[Endorsement](https://info.arxiv.org/help/endorsement.html)、[Licenses](https://info.arxiv.org/help/license/index.html)。

## ARR Short Paper 准备清单

- ACL 官方匿名模板，Short Paper 正文最多 4 页，参考文献和允许的补充材料按当期 CFP；
- 双向匿名：正文、补充材料和匿名代码链接不能泄露作者身份；
- 独立的 Limitations 部分与 Responsible NLP Checklist；
- OpenReview 账号、完整作者列表和利益冲突信息；
- 逐样本结果、随机种子、统计检验、训练配置和可复现说明；
- 投稿期间不得同时投另一个同行评审的 archival venue；
- 所有作者按当期规则注册为潜在 reviewer，并按时完成被分配的评审；
- 收到 ARR reviews/meta-review 后，选择修改后进入下一轮，或 commit 到参加 ARR 的会议。

官方流程：[ARR Authors Guide](https://aclrollingreview.org/authors)、[ARR CFP](https://aclrollingreview.org/cfp)、[Dates and Venues](https://aclrollingreview.org/dates)。截至 2026-08-28，NAACL 2027/COLING 2027 的最后 ARR 轮截止日期为 2026-10-12；ACL 2027 对应 2027 年 1 月轮，具体日期仍应以官方页面为准。

## 费用与预算

投稿和评审阶段通常不收 submission fee；arXiv 明确免费。真正的强制成本一般发生在论文被接收后：至少一名作者必须注册并展示论文，否则可能不能进入会议日程或论文集。

以下仅用 ACL 2026 官方早鸟价作为量级参考，不是 2027 报价：

| 被接收后的参会方式 | Student | Academic | Industry |
|---|---:|---:|---:|
| 主会论文、线上 | USD 300 | USD 450 | USD 600 |
| 1 天 Workshop 论文、线上 | USD 175 | USD 200 | USD 225 |
| 主会论文、线下 3 天 | USD 550 | USD 850 | USD 1,150 |
| 1 天 Workshop 论文、线下 | USD 350 | USD 450 | USD 550 |

按 USD/CNY 7.0--7.5 粗估，线上 Workshop 作者注册约人民币 1,200--1,700 元，线上主会约 2,100--4,500 元；线下还要另计机票、住宿、签证和本地交通。会议可能提供地区减免、D&I subsidy、学生志愿者或学生旅行资助，必须在通知后尽早申请。价格参考：[ACL 2026 Registration](https://2026.aclweb.org/registration/)。

预算原则：论文未接收前只预留实验算力和少量必要 API 费用；接收后再支付作者注册，不提前购买不可退差旅。若资金紧张，首选 arXiv 免费发布，再投支持 non-archival/线上展示和补助的 Workshop。

## 本项目的投稿门槛

### 历史最低门槛（本项目不在此停止）

- 六条件 matched 结果完成，困难切片没有全面饱和；
- 至少报告逐位置曲线、worst-position、gap、证据指标和置信区间；
- 排除训练/测试事实重叠、模板捷径和输出格式假提升；
- 代码、数据生成协议、配置和原始聚合结果可复现；
- 明确单模型/单种子/合成数据等局限。

### 当前顶会/期刊级目标（全部为硬要求）

- 关键条件至少 3 个随机种子；
- 至少第二个模型家族；
- 自然文档或公开长上下文 benchmark；
- locate-only、oracle evidence 等检索/推理机制诊断；
- paired bootstrap、McNemar 或等价的配对显著性分析；
- 若六条件都接近 100%，先增加释义、无词面重叠、矛盾/强干扰和不可回答样本。

## 执行顺序

1. 完成 strict Qwen/Mistral 三 seed 全矩阵并冻结全部逐样本证据；
2. 自动生成主表、位置曲线、训练曲线、失败案例、算力账本与 claim-to-artifact manifest；
3. 按预注册降级规则写入正结果、负结果、模型交互和局限性，不按分数追加赢家选择；
4. 构建并审计完整技术报告、公开复现包和确定性 arXiv 源包；
5. 目标 venue 确认后重新核对 CFP、匿名、预印本、归档和注册政策，并由完整稿派生篇幅受限版本；
6. 所有作者确认后再执行 arXiv 最终提交或 OpenReview 投稿。
