# 研究路线图

> 执行状态（2026-08-29 15:48 CST）：realized-sampler 审计发现原 fixed-100 数据子集不具备完整四位置事实块，因此其 Qwen 结果全部降级为历史部分剂量诊断。旧 LongBench 尾段已以 7,200/7,200、分析/失败目录/归档 SHA-256 全通过的状态冻结；strict block-96 顶层队列于 15:46:54 从失败关闭门禁恢复，15:47:18 开始 Qwen corrective 三-seed 六条件训练。论文主实验为 Qwen corrective replication 与 Mistral prospective replication：每模型三 seed、每 seed 六条件、每条件 24 facts × 4 positions = 96 rows/steps。阶段 3 的 RL 与阶段 4 的架构改造不属于本稿必须项，除非冻结结果触发预注册扩展规则。动态状态、预算和停止规则以[顶会级完整性协议](top-tier-completion-protocol.md)为准。

## 阶段 0：复现基础现象

目标：确认评测管线能重现 U 形、近因偏置和 filler 交互。

- 实现位置等价组生成器；
- 实现 KV、无字面重合检索和双跳任务；
- 支持 8K、32K、64K 三档长度；
- 复现 Lost in the Middle 三点曲线；
- 增加七点位置网格和相对窗口长度；
- 输出逐样本 JSONL、聚合表和置信区间。

完成条件：至少两个模型上得到稳定、可重复的位置曲线，三次运行方向一致。

## 阶段 1：机制诊断

目标：区分位置、距离、结构与干扰因素。

- 改变 query 位置；
- 移除或替换文档分隔符；
- 比较 neutral、same-format 和 answer-bearing filler；
- 运行仅检索、自由推理和 Oracle 证据三联实验；
- 可访问内部状态时记录注意力、logit lens 或隐藏位置通道；
- 对 API 模型只做行为结论，不推断不可见内部机制。

完成条件：能够用受控实验判断主要失败来自检索、推理还是 filler 干扰。

## 阶段 2：监督训练基线

目标：建立计算成本较低、可解释的训练基线。

- 普通长上下文 SFT；
- 位置均衡 SFT；
- 答案加证据引用监督；
- Pos2Distill 风格跨位置 KL；
- 未见长度、位置和领域测试。

完成条件：中间提升不是由边缘退化造成，且最弱位置和位置 gap 同时改善。

## 阶段 3：位置强化学习

目标：验证优化最差位置是否优于优化平均准确率。

- 建立确定性答案与证据 verifier；
- 实现位置组奖励；
- 比较 mean、min 和 CVaR 奖励；
- 检查奖励投机、格式退化和通用能力损失；
- 与等 Token 预算 SFT/蒸馏比较。

完成条件：RL 在未见位置或未见 filler 上显著超过 SFT，而不是只拟合训练分布。

## 阶段 4：架构与系统组合

目标：测量训练、位置编码和上下文编排能否叠加。

- 逐层 RoPE scaling；
- 文档重排；
- RAG 压缩；
- 分层摘要与分块推理；
- 在相同延迟和 Token 成本下比较收益。

完成条件：形成模型能力、推理成本和可靠性三者的 Pareto 曲线。

## 阶段 5：发布

计划产物：

- 可复现的数据生成器；
- 位置 × 长度 × filler 评测套件；
- 原始逐样本结果和统计脚本；
- 训练配置及 checkpoint 元数据；
- 技术报告，明确已发表证据与本项目实验的边界；
- 模型卡中的有效上下文长度和最弱位置指标。

发布路径与费用、材料、归档策略见[论文发布与投稿计划](publication-plan.md)。当前停止条件不是 Workshop 最低门槛：先补齐多随机种子、第二模型家族、自然语义 OOD、机制诊断、统计检验和复现审计，形成顶会/期刊级完整技术报告并发布 arXiv；随后以同一冻结证据准备 ARR/ACL Findings 或主会版本，Workshop 仅作为可选反馈渠道。

## 近期任务

1. **已完成并降级为历史诊断：** Qwen fixed-100 三 seed 训练及其规则、NoLiMa、LongBench、MMLU、IFEval 与机制结果；
2. **已完成：** 三 seed strict block-96 数据物化、位置/事实块/Token 匹配审计，以及 Qwen corrective、Mistral prospective 标签冻结；
3. **进行中：** strict Qwen 三 seed × 六条件训练与 rule/NoLiMa/LongBench；旧 LongBench 尾段已完整冻结且不进入 strict 主均值；
4. **待同一队列自动衔接：** strict Mistral 三 seed × 六条件训练与同构评测；
5. **待同一队列自动衔接：** 两家族代表 seed 的 MMLU、IFEval、locate/oracle 机制诊断；
6. **已准备门禁：** 18 个严格失败目录、跨 seed/家族统计、有效 GPU 时间账本、claim-to-artifact 证据清单；
7. **进行中：** 论文、可读主图、公开复现包与 arXiv 源包；实验数字会由 strict 证据自动替换，最终作者元数据只阻塞提交动作，不阻塞其余准备。
