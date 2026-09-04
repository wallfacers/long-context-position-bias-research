# 研究文档

## 项目目标

本项目研究长上下文语言模型是否能够与位置无关地检索和利用上下文，以及通过训练、蒸馏、位置编码校准或上下文编排，能否降低最弱位置上的性能损失。

我们关心的不是模型能否接受某个长度的输入，而是它在该长度下能否可靠完成需要检索、整合和推理的任务。因此，项目会明确区分：

- **声明上下文窗口**：接口或模型配置允许输入的最大 Token 数。
- **有效上下文长度**：在指定任务、干扰和准确率阈值下仍然可用的长度。
- **位置偏差**：相同信息仅因位置变化而产生的性能差异。
- **Lost in the Middle**：开头和末尾优于中间的特定位置曲线。
- **上下文退化**：随长度、干扰或任务复杂度增加而出现的整体能力下降，未必呈 U 形。

## 当前结论

1. 2026 年的位置偏差仍然存在，但不能统一概括成固定 U 形。
2. 这通常是检索与利用失败，不等同于中间 Token 从 KV Cache 中消失。
3. 输入占模型窗口不超过约一半时，U 形更常见；逼近窗口上限后可能转为近因或距离偏置。
4. 分隔符、文档结构、干扰内容、字面重合、多跳深度和 query 位置都会改变曲线。
5. 针对性训练能够显著缓解特定分布上的问题，但通用根治尚未得到证明。
6. 目前针对性 SFT 和位置蒸馏的直接证据强于普通 RLHF；专门的位置鲁棒性 RL 值得实验验证。

## 阅读顺序

1. [研究问题与边界](research-questions.md)
2. [文献综述](literature-review.md)
3. [论文索引](papers.md)
4. [实验协议](experimental-protocol.md)
5. [Qwen2.5-7B 单 seed 消融 pilot](pilot-qwen25-7b.md)
6. [数据准备与 API 使用边界](data-preparation.md)
7. [AutoDL 上机即跑手册](autodl-runbook.md)
8. [正式 matched 消融运行手册](formal-matched-runbook.md)
9. [训练与干预方案](training-strategies.md)
10. [研究路线图](roadmap.md)
11. [论文发布与投稿计划](publication-plan.md)
12. [顶会级完整性协议与自主执行门槛](top-tier-completion-protocol.md)
13. [复现卡与 claim-to-artifact 层级](reproducibility-card.md)
14. [论文主张到证据产物映射](claim-to-artifact-map.md)
15. [失败案例分类与公开审计协议](failure-case-taxonomy.md)

## 文档约定

- 论文链接统一优先使用 alphaXiv。
- “已接收/已发表”和“预印本”分开标注。
- 论文观察记为“证据”，尚未验证的解释记为“假设”。
- 所有模型比较同时报告长度、相对窗口占比、位置、任务和 filler，避免只报平均分。
- 研究状态基准日期为 **2026-08-29**；动态实验进度以[顶会级完整性协议](top-tier-completion-protocol.md)为准。
