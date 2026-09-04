# 实验协议

> **2026-08-29 realized-subset 修正。** 原 1,024-row / fixed-100 实现只随机消费 100 行，实际没有形成完整四位置事实块，故只保留为探索性敏感性分析。论文主实验现使用每 seed 24 facts × 4 replicas = 96 rows 的 strict block-complete 子集，四位置各 24 行、pairing 输入 Token 总量相等，并从 checkpoint-96 重跑。完整审计、post-hoc Qwen/前瞻 Mistral 的统计标签和执行状态以[顶会级完整性协议](top-tier-completion-protocol.md)为准。

## 一、实验原则

实验的基本单元不是单个 prompt，而是一个**位置等价组**：问题、目标证据和 filler 内容保持一致，只改变目标证据的位置。所有位置比较必须在同一组内配对完成。

主实验采用以下因子设计：

```text
模型
  × 上下文长度
  × 目标相对位置
  × filler 类型
  × 任务类型
  × query 位置
  × 随机种子
```

## 二、自变量

### 上下文长度

同时记录绝对长度和相对长度：

```text
absolute_tokens ∈ {8K, 32K, 64K, 128K}
relative_length = actual_input_tokens / advertised_context_window
```

若模型不支持某个长度，只测试其合法范围，不通过截断伪造结果。

### 证据位置

第一阶段使用七点网格：

```text
position ∈ {0%, 10%, 25%, 50%, 75%, 90%, 100%}
```

多证据任务分别控制：

- 所有证据聚集在同一区域；
- 证据分散于开头、中间、末尾；
- 保持绝对位置但改变证据间距离；
- 保持证据间距离但整体平移。

### filler 类型

| 类型 | 内容 | 目的 |
|---|---|---|
| neutral | Wikipedia、新闻或无关代码 | 测量纯长度与位置效应 |
| same-domain | 同领域但无答案的材料 | 测量语义干扰 |
| same-format | 与目标相同格式的问题或记录 | 测量结构干扰 |
| answer-bearing | 带候选答案或完整推理过程的样例 | 测量抢答和错误证据绑定 |
| adversarial | 与目标高度相似但结论错误 | 测量鲁棒性与引用忠实度 |

### 任务类型

1. 唯一键值精确检索；
2. 无字面重合的语义定位；
3. 多文档问答；
4. 两跳与三跳证据组合；
5. 时间顺序和状态更新；
6. 长代码中的定义、调用和跨文件依赖；
7. 长文档摘要中的事实覆盖与引用。

## 三、输出协议

要求模型返回结构化结果：

```json
{
  "answer": "...",
  "evidence_ids": ["doc-07", "doc-12"],
  "evidence_quotes": ["short supporting span"],
  "confidence": 0.0
}
```

证据引用既是评测信号，也是防止模型根据 filler 或先验猜答案的约束。引用只要求最短支持片段，避免复制长篇版权内容。

## 四、指标

### 任务指标

- Exact Match、F1 或任务准确率；
- 证据文档召回率与精确率；
- 引用支持率；
- 多跳证据完整率；
- 格式有效率和拒答率。

### 位置指标

设位置集合为 \(P\)，位置 \(p\) 的准确率为 \(A_p\)：

```text
mean_accuracy       = mean(A_p)
worst_position      = min(A_p)
position_gap        = max(A_p) - min(A_p)
middle_penalty      = (A_first + A_last) / 2 - A_middle
position_variance   = variance(A_p)
```

如果末尾始终最好，另外报告准确率相对于“证据到 query 距离”的回归斜率。这样即使 U 形消失，也能捕获距离偏置。

### 有效上下文长度

先固定短上下文基线 \(A_0\)，再定义阈值，例如：

```text
ECL_90 = 最大长度 L，使所有位置的准确率至少达到 0.9 × A_0
```

必须使用所有位置或最弱位置，不能只用平均值定义 ECL。

## 五、检索—推理解耦

每个推理样本生成四个版本：

1. **自由任务**：模型自行检索并推理；
2. **仅定位（locate-only）**：保留原长上下文，只返回证据 ID 与精确引用，不求最终答案；
3. **长上下文 Oracle（oracle-long）**：保留长干扰上下文，把正确证据移到 query 后的 oracle 区，再要求回答；
4. **短上下文 Oracle（oracle-short）**：只提供正确证据与问题，再要求回答。

诊断规则：

- 自由任务差、locate-only 差、oracle-long 稳定：主要是检索瓶颈；
- locate-only 稳定、自由任务差：主要是证据整合或答案生成瓶颈；
- oracle-long 差而 oracle-short 稳定：即使证据显式给出，长干扰或上下文长度仍妨碍证据利用；
- oracle-long 与 oracle-short 都差：优先检查纯推理、输出协议或任务定义，不把失败归因于位置检索；
- 只有自由任务和 locate-only 才解释原证据位置曲线；oracle 条件用于机制差值，不把末尾 oracle 的位置误当作原证据位置。

## 六、统计设计

- 自建确认性主集每个条件至少 200 个位置等价组；外部基准使用其全部冻结样本，并按其真正独立的语义 case 聚类推断。少于 200 个独立组的外部门控和机制诊断必须明确标成受限证据，不能把重复 book、长度或位置当成独立语义题扩大样本量。
- 每个模型家族固定 3 个训练/数据 seed。原 Qwen fixed-100 三 seed 全部只进入历史部分剂量描述表和谱系审计；strict block-96 Qwen 三 seed 全部进入 corrective 主均值，Mistral 三 seed 作为修正协议下的 prospective confirmatory replication。两家族所有跨 seed 区间均以 seed 为训练重复，不能用逐样本行数虚增自由度；合并结果明确标记为 corrective + confirmatory，而非伪称 confirmatory-only。
- 位置差异使用配对 bootstrap 置信区间；二分类结果可增加 McNemar 检验。
- 多位置、多长度比较使用 Holm 校正。
- 同时报告效应量和 95% 置信区间，不只报告显著性。
- API 模型记录模型标识、日期、区域、参数、thinking 开关和系统提示词。

## 七、数据切分与防泄漏

- 按原始事实或文档划分 train/dev/test，而不是按位置变体随机划分。
- 同一原始样本的所有位置排列只能属于同一个 split。
- 测试集包含未见领域、未见长度和未见位置组合。
- 训练中出现过的模板需在测试集中改写，检查格式捷径。
- 单独保留无字面重合和对抗干扰测试集。

## 八、候选基线与本论文冻结范围

下面是完整研究路线中可比较的干预家族，不代表在看过结果后可以追加到当前确认性矩阵。本文是处理归因研究而不是新的 SOTA 方法竞赛；冻结的论文基线是未训练 Base，加上事实曝光、序列数和输入 Token 预算匹配的 `independent/paired × answer/evidence-ID/exact-evidence` 六个单元。其中 independent 单元已经构成位置直方图匹配的长上下文 SFT 控制。Pos2Distill、重排、RAG、RoPE 或注意力校准改变目标函数、推理计算或输入，因此在相关工作中按干预类别比较，除非另行预注册和增加预算，否则不混入本轮因果矩阵。

1. 原始模型，无额外训练；
2. 普通短上下文 SFT；
3. 普通长上下文 SFT；
4. 位置均衡 SFT；
5. Pos2Distill 风格位置蒸馏；
6. 文档随机排列与按预测位置重排；
7. RAG 压缩或分层摘要；
8. 可用时加入 RoPE/注意力校准。

## 九、结果表模板

| Model | Method | Task | Filler | Length | First | Middle | Last | Worst | Gap | Evidence Recall |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Template only | Base | KV | neutral | 8K | — | — | — | — | — | — |

所有主结论必须能够回溯到逐样本结果，不只保存聚合表。

## 十、训练审计与可复现性

每个训练 run 必须同时保存：完整 CLI 参数、随机种子、基座模型 revision 与文件 hash、训练数据与 tokenizer 指纹、Python/CUDA/驱动/GPU/依赖版本、逐 optimizer step 的学习率与 Loss、gradient norm、checkpoint 恢复点、最终 adapter hash。任何重启都不能把原始运行记录静默覆盖；最终导出的逐步序列必须检查缺步和重复 step。

训练图至少包含逐步原始值和明确标注平滑方法的趋势线。Loss 与 gradient norm 跨数量级时使用对数轴；图、源 CSV/JSONL、绘图参数和矢量格式同时发布。训练 completion-only Loss 接近零只能说明对监督 completion 的拟合，必须以未参与训练的 dev/test 位置等价组判断泛化。answer-only 与 evidence-supervised 的 completion 长度和目标空间不同，因此跨监督变体的绝对训练 Loss 不作为处理效应；它只用于各 run 内部的收敛、异常和过拟合诊断。

历史 Qwen fixed-100 运行用于记录初始停止规则、困难评测选择和 implementation correction，不作为 strict pairing 显著性结论。修正后的 Qwen 主均值使用三个 block-complete corrective seed，Mistral 使用三个在修正后尚未查看结果的 confirmatory seed；两者均把 seed 作为统计层级，并将逐样本预测、配对 bootstrap 抽样索引、置信区间和多重比较校正结果一并保留。相同 seed 与环境不保证量化 CUDA 内核逐 bit 一致，因此可复现目标是哈希可验证的输入、配置和代码，以及统计上等价的逐样本结论。
