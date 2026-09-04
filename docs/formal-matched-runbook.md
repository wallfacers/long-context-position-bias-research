# 正式 matched 消融运行手册

> **预算、采样与训练步数勘误。** 本页保留最初 2,000-step 规划及随后 fixed-100 实现的历史记录，不能作为当前执行命令。2026-08-29 realized-subset 审计证明 fixed-100 只形成 11--15 个不完整跨位置重复事实，0 个完整四位置块；论文主实验已改为六条件 × strict block-complete 96 rows/steps，保留原 2,000-step cosine horizon（60-step warmup）。当前唯一执行协议以[顶会级完整性协议](top-tier-completion-protocol.md)为准；旧 checkpoint-100 只作为敏感性诊断。

## 目的与停止条件

正式阶段验证两个问题：

1. 在控制事实集合、每事实曝光次数、filler 视图和输入 Token 预算后，同一事实跨位置配对是否真正降低 worst-position 与 position gap；
2. 改善来自答案记忆、证据 ID 定位，还是精确引用监督，并且它能否跨未见长度、干扰类型和任务难度保持。

主矩阵是 `pairing ∈ {independent, paired}` × `supervision ∈ {answer, evidence_id, evidence}` × 三个数据/训练 seed。第一 seed 是付费门控：若 matched 审计失败、训练异常、或六条件完整测试无法复现任何有意义的 pairing/supervision 差异，则先诊断，不启动其余两个 seed。

这轮不是自然领域 OOD 的最终证明。现有 test 的未见 32K、same-format、answer-bearing 与训练未见位置属于长度/干扰/组合分布外。自然语义、跨领域和无字面重合数据是下一阶段的独立扩展，不能与本轮规则数据混写。

## 一、本地 CPU 准备

在租 GPU 前执行：

```bash
bash scripts/prepare_formal_matched_data.sh

python3 scripts/estimate_autodl_budget.py \
  --config configs/autodl_formal_matched_budget.json
```

默认会生成三个 seed 的六份原始 SFT JSONL 和 Arrow 预分词数据，并校验 SHA-256。预期预算输出是：GPU 计算约 ¥258.94，固定预留 ¥25，期望 ¥283.94；加 20% 缓冲后 ¥340.73，建议充值 ¥350。这个估算基于已完成 pilot 的 2.8884 秒/step 和 1.7671 秒/评测样本。

该预算包括 18 个 2,000-step adapter、base 加 18 个 adapter 的 4,200 样本完整测试，以及 5 个 checkpoint 的 9,000 样本机制诊断。它不包括自然语义 API 数据，也不包括把每格从 50 个组扩到 200 个组。

按单台 5090 串行执行的实测校准，时间拆分如下：

| 阶段 | GPU 小时 | 估算费用 |
|---|---:|---:|
| 18 个正式训练 run | 30.68 h | ¥85.30 |
| base + 18 adapters 完整测试 | 40.12 h | ¥111.54 |
| 5 个机制诊断 run | 22.34 h | ¥62.10 |
| 合计 GPU 任务窗口 | 93.14 h | ¥258.94 |

因此全部串行约 3.9 天纯计算，连同人工门控按 4～5 天排期。第一 seed 门控约为 10.2 小时训练加 14.8 小时完整评测，即约 25 小时；通过后才支付其余 seeds。多机并行可缩短墙钟时间但不降低 GPU-hours 或总费用。若已保存且身份哈希一致的 base 结果可复用，最终评测可再省约 2.1 小时和 ¥5.9。

## 二、上传后只做只读预检

模型、代码和数据同步到 AutoDL 后，先核对三个 seed 的 manifest 与本地模型 revision。正式训练输出约定为：

```text
outputs/formal_matched/
  seed_20260825/{six variants}
  seed_20260826/{six variants}
  seed_20260827/{six variants}
```

第一个 seed 先跑一个可恢复的 100-step canary：

```bash
bash scripts/run_sft_variant.sh \
  --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --data-dir data/formal_matched_qwen25_7b/seed_20260825 \
  --output-root outputs/formal_matched/seed_20260825 \
  --variant paired_evidence \
  --seed 20260825 \
  --canary
```

检查显存、step 时间、Loss、learning rate、gradient norm 和 checkpoint 后，完整队列会从 canary checkpoint 继续，不重复已完成 step：

```bash
bash scripts/run_autodl_training_queue.sh \
  --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --data-dir data/formal_matched_qwen25_7b/seed_20260825 \
  --output-root outputs/formal_matched/seed_20260825 \
  --variants independent_answer,independent_evidence_id,independent_evidence,paired_answer,paired_evidence_id,paired_evidence \
  --seed 20260825 \
  --result-bundle /root/autodl-tmp/formal-train-seed-20260825.tar.gz \
  --status-file /root/autodl-tmp/formal-train-seed-20260825.status
```

脚本不会执行系统关机，也不会假设 `shutdown -h` 能停止计费。产物校验完成后，必须在 AutoDL 控制台显式关机/停止实例。

## 三、第一 seed 付费门控

先评测 base 与第一 seed 六个条件：

```bash
bash scripts/run_autodl_formal_eval.sh \
  --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --adapter-root outputs/formal_matched \
  --seeds 20260825 \
  --output-dir results/formal_matched
```

只有以下项目全部成立才启动 seed 2/3：

- 六个训练输出、逐 step 指标和 adapter hash 完整；
- 结果是 84 个 test cell × 每格 50 个组，未混入训练事实；
- JSON 截断率没有重新升高；
- 答案、证据 ID、精确引用、worst-position、gap 都分别报告；
- evidence-ID-only 的位置位于 answer-only 与 ID+quote 之间或给出可解释反例；
- 效果不是只来自单一 task、8K 或 neutral filler。

## 四、剩余 seeds 与最终评测

对 `20260826`、`20260827` 分别运行同一训练队列，替换 data/output/seed/status/bundle 参数。完成后重新调用正式评测脚本：

```bash
bash scripts/run_autodl_formal_eval.sh \
  --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --adapter-root outputs/formal_matched \
  --seeds 20260826,20260827 \
  --checkpoint-name checkpoint-100 \
  --reuse-base-dir results/test_full \
  --output-dir results/formal_matched_confirmatory
```

已有 base 通过数据 hash、模型路径和逐样本数审计后复制复用，不重新花 GPU 计算；seed 1 独立冻结为探索性结果，只补 seed 2/3 的 12 个确认性 run。每个 seed 自动生成 5,000 次位置等价组 bootstrap、完整 2×3 主效应/交互和论文图；跨 seed 汇总把训练 seed 作为统计层级，不能把 4,200 条同一模型预测冒充训练重复。

## 五、机制诊断

诊断只预注册 base 和第一 seed 的四个 2×2 主角条件，ID-only 已在完整测试中作为中间机制条件：

```bash
bash scripts/run_autodl_formal_eval.sh \
  --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --data data/formal_matched_qwen25_7b/eval/test_diagnostics.jsonl \
  --manifest data/formal_matched_qwen25_7b/manifest.json \
  --adapter-root outputs/formal_matched \
  --seeds 20260825 \
  --variants independent_answer,independent_evidence,paired_answer,paired_evidence \
  --output-dir results/formal_diagnostics \
  --artifact /root/autodl-tmp/position-bias-formal-diagnostics.tar.gz \
  --status /root/autodl-tmp/position-bias-formal-diagnostics.status
```

解释顺序固定为：locate-only 判断检索，oracle-long 判断长干扰下证据利用，oracle-short 判断证据已知后的纯推理/输出。不要只因自由任务出现 U 形就宣称“注意力丢失中间信息”；也不要只因 answer-only 很高就宣称完成了可引用、可验证的长上下文检索。

## 六、饱和门控与 NoLiMa 自然 OOD

2026-08-28 的第一 seed checkpoint-100 早期部分结果曾显示四个单元的总体答案准确率超过 99%，但冻结后的完整七-run、4,200 行/条件分析不支持“六条件全面饱和”。六个训练单元的总体答案正确率为 97.33%--99.95%；`paired_answer` 的 mean worst-position 只有 87.33%，gap 为 12.67 pp，而 `independent_evidence` 为 99.67% 和 0.33 pp。预注册的“所有单元至少 98% 且总体范围小于 2 pp”门禁为 false。正确表述是：多个单元的总体分数接近天花板，但总体均值会隐藏仍可辨识的位置失败；第一 seed 仍只能作为探索性证据，不能单独排序方法。

在启动 seed 2/3 前，先运行官方 NoLiMa-Hard 派生的无词面重叠位置等价 gate：

```bash
python3 scripts/prepare_nolima_ood.py \
  --needle-set third_party/NoLiMa/data/needlesets/needle_set_hard.json \
  --haystack-dir third_party/NoLiMa/data/haystack/rand_shuffle \
  --output data/ood_nolima/hard_gate.jsonl \
  --manifest data/ood_nolima/hard_gate.manifest.json \
  --audit data/ood_nolima/hard_gate.audit.json \
  --lengths 1024,8192,32000 \
  --positions 0,0.1,0.25,0.5,0.75,0.9,1 \
  --local-files-only

bash scripts/run_autodl_nolima_gate.sh \
  --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct
```

gate 固定为 10 个 NoLiMa-Hard case × 5 本官方 haystack × 3 个长度 × 7 个位置，共 1,050 行/模型。组内固定人物、问题、书籍切片和长度，只移动原始 needle；不加入人为 evidence ID。NoLiMa 数据遵循 Adobe Research License，仅可用于非商业研究，发布复现材料时必须保留许可与归属。

NoLiMa-Hard 先决定论文的可辨识主张，但在顶会完整性目标下仍扩展 seed 2/3 与第二模型，用确认性重复判断“存在差异”或“没有可检测差异”。若所有方法在自然 OOD 仍无差异，则论文结论应降级为“100-step 训练能让多个规则单元达到近天花板总体分数，但位置修复没有稳定迁移”，不能宣称某种 pairing 或监督粒度更优。

## 七、2026-08-28 确认性修订

早期预算按每个条件 2,000 steps 估计；seed-1 探索实验在 checkpoint-100 已出现 completion Loss 近零、token accuracy 接近 1，且多个 evidence-supervised 单元的总体规则分数接近天花板。完整冻结结果仍保留 worst-position 失败，因此 100 steps 不是“全面解决任务”的证据，而是避免在监督完成项已拟合后继续增加事实记忆风险的固定预算。因此顶会确认阶段冻结为：seed-2/3 与第二模型均运行等预算 100 steps，不做测试集 checkpoint 选择；seed-1 明确标记为产生该停止规则的探索性 pilot。该修订及负结果处理以[顶会级完整性协议](top-tier-completion-protocol.md)为准。

规则 gate 完成后还要运行冻结的 LongBench v1 自然多文档迁移集：

```bash
bash scripts/run_autodl_longbench_transfer.sh \
  --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct
```

该集合包含 HotpotQA、2WikiMQA、MuSiQue 各 200 题，使用官方多参考答案 QA token-F1。它没有 gold evidence 位置标注，因此只用于自然任务迁移，不能用于位置因果或精确引用主张。
