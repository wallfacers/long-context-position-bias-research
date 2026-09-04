# Long-Context Position Bias Research

研究长上下文语言模型中的位置偏差、`Lost in the Middle`、有效上下文长度，以及针对性训练能否获得位置稳健的检索与推理能力。

项目当前处于 strict block-96 主实验执行阶段：realized-sampler 审计已将原 fixed-100 输出降级为历史部分剂量诊断；Qwen corrective replication 与 Mistral prospective replication 各用三 seed 运行 matched $2\times3$ 消融、自然/OOD、机制、通用能力回归和跨家族统计。研究结论、论文记录、实验协议与训练方案统一维护在 [`docs/`](docs/index.md)。

## 目录

```text
.
├── configs/       # 后续实验配置
├── data/          # 本地数据；大文件不提交
├── docs/          # 研究文档与论文索引
├── results/       # 原始结果与聚合指标
├── scripts/       # 数据生成、训练和评测入口
└── src/           # 可复用研究代码
```

## 当前研究判断

截至 2026-08-29，位置相关的上下文利用失败仍未被普遍解决。固定的 U 形曲线并非唯一形态：输入较短时常见首因与近因共同造成的中段低谷；输入接近窗口上限时，首因优势可能消失，转为偏向输入末尾的距离效应。

针对性 SFT、位置蒸馏、证据图数据合成和位置编码校准已经能在特定测试上显著缩小位置差距，但尚无充分证据证明这种能力可以跨长度、任务、干扰分布和模型家族稳定泛化。

本项目的历史 fixed-100 Qwen NoLiMa 诊断与该判断一致：没有训练条件稳定超过固定 Base 的 worst-position；exact-evidence 监督提高总体答案和引用，却扩大平均位置 gap，pairing 对答案与 worst-position 的方向跨 seed 反转。但 sampler 审计证明这些运行没有形成完整四位置事实块，因此它们不能进入主因果均值。最终主张只读取正在执行的三-seed strict Qwen、三-seed prospective Mistral、matched rule、NoLiMa、LongBench 与预冻结回归/机制证据。

## 文档入口

- [研究问题与边界](docs/research-questions.md)
- [文献综述](docs/literature-review.md)
- [论文索引](docs/papers.md)
- [实验协议](docs/experimental-protocol.md)
- [Qwen2.5-7B 单 seed 消融 pilot](docs/pilot-qwen25-7b.md)
- [数据准备与 API 使用边界](docs/data-preparation.md)
- [AutoDL 上机即跑手册](docs/autodl-runbook.md)
- [训练与干预方案](docs/training-strategies.md)
- [研究路线图](docs/roadmap.md)
- [顶会级完整性协议与当前状态](docs/top-tier-completion-protocol.md)
- [论文发布、arXiv 与投稿计划](docs/publication-plan.md)
- [复现卡与 claim-to-artifact 层级](docs/reproducibility-card.md)
- [论文主张到证据产物映射](docs/claim-to-artifact-map.md)
- [失败案例分类与公开审计协议](docs/failure-case-taxonomy.md)

## 轻量复现门禁

无需 CUDA 或模型权重即可验证分析、统计、打包与论文工具链的单元测试：

```bash
python3 -m pip install -r requirements-test.txt
PYTHONPATH="$PWD" python3 -m pytest -q
```

公开包不重新分发 NoLiMa 与 IFEval 等第三方 benchmark 原文；未按冻结 manifest 重建这些 payload 时，三项原文一致性测试会明确标为 skipped，其余测试必须通过。完整研究树中这些测试不得跳过。
