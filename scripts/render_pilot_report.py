#!/usr/bin/env python3
"""Render a Chinese paper-pilot report from validated ablation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


RUN_ORDER = (
    "base",
    "paired_evidence",
    "paired_answer",
    "independent_evidence",
    "independent_answer",
)
RUN_LABELS = {
    "base": "Base",
    "paired_evidence": "Paired + evidence",
    "paired_answer": "Paired + answer",
    "independent_evidence": "Independent + evidence",
    "independent_answer": "Independent + answer",
}
CONTRAST_LABELS = {
    "paired_minus_independent_main_effect": "Paired − independent 主效应",
    "evidence_minus_answer_main_effect": "Evidence − answer 主效应",
    "pairing_x_supervision_interaction": "Pairing × supervision 交互",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percent(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def pp(value: float, digits: int = 1) -> str:
    return f"{value * 100:+.{digits}f} pp"


def interval_effect(item: dict[str, float]) -> str:
    return (
        f"{pp(item['estimate'])} "
        f"[{pp(item['ci95_low'])}, {pp(item['ci95_high'])}]"
    )


def check_mark(value: bool) -> str:
    return "通过" if value else "未通过"


def render(
    analysis_path: Path, validation_path: Path, cost_ledger_path: Path | None = None
) -> str:
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if analysis.get("schema_version") != "position-ablation-analysis-v1":
        raise SystemExit("Unexpected analysis schema")
    if validation.get("status") != "validated":
        raise SystemExit("Evaluation validation report is not validated")
    expected_runs = set(RUN_ORDER)
    if set(analysis.get("run_summaries", {})) != expected_runs:
        raise SystemExit("Analysis does not contain the five expected runs")
    if set(analysis.get("rows_per_run", {}).values()) != {4200}:
        raise SystemExit("Paper report requires exactly 4,200 rows per run")

    summaries = analysis["run_summaries"]
    generation = analysis["generation_diagnostics"]
    intervals = analysis["run_summary_intervals"]
    screening = {
        item["run_name"]: item
        for item in analysis["exploratory_screening"]["results"]
    }
    passed = [name for name in RUN_ORDER[1:] if screening[name]["passes_all_exploratory_checks"]]
    best_worst = max(RUN_ORDER, key=lambda name: summaries[name]["mean_worst_position_accuracy"])
    smallest_gap = min(RUN_ORDER, key=lambda name: summaries[name]["mean_position_gap"])
    evidence_effect = analysis["contrasts"]["evidence_minus_answer_main_effect"][
        "statistics"
    ]
    pairing_effect = analysis["contrasts"]["paired_minus_independent_main_effect"][
        "statistics"
    ]
    execution = validation["execution"]
    cost_ledger = (
        json.loads(cost_ledger_path.read_text(encoding="utf-8"))
        if cost_ledger_path is not None
        else None
    )

    lines = [
        "# Qwen2.5-7B 长上下文位置偏差消融：单 seed pilot 结果",
        "",
        "> 本报告由完整逐样本结果自动生成。本轮是方向筛选，不构成跨模型、跨数据或跨 seed 的最终显著性结论。",
        "",
        "## 验收与执行摘要",
        "",
        f"- 完整矩阵：{validation['matrix']['total_samples']:,} 条预测，5 个 run 各 4,200 条；84 个条件格各 50 条。",
        f"- 配对单位：同一事实、问题和 filler 的七位置等价组；bootstrap 按 task × filler × length 分层。",
        f"- 统计：{analysis['bootstrap']['replicates']:,} 次配对组 bootstrap，seed={analysis['bootstrap']['seed']}，保存全部抽样索引。",
        f"- 全量评测墙钟：{execution['wall_hours']:.2f} 小时；按 ¥{execution['hourly_rate_cny']:.2f}/小时估算 ¥{execution['estimated_cost_cny']:.2f}。这不含此前训练、gate 和平台空闲时间。",
        f"- 输入报告 SHA-256：analysis `{sha256_file(analysis_path)}`；validation `{sha256_file(validation_path)}`。",
        "",
        "## 五个条件的核心结果",
        "",
        "| 条件 | 答案 | 证据 ID | 精确引用 | 引用可支持 | JSON | Length-stop | 最弱位置 | Gap | 首尾 | 中间 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if cost_ledger is not None:
        totals = cost_ledger["totals"]
        lines.insert(
            9,
            f"- 已记录训练、gate、诊断与正式评测任务窗口合计：{totals['recorded_task_wall_hours']:.2f} 小时，估算 ¥{totals['estimated_recorded_task_cost_cny']:.2f}；这是任务账本，不是 AutoDL 发票。",
        )
    for run_name in RUN_ORDER:
        item = summaries[run_name]
        lines.append(
            "| "
            + " | ".join(
                [
                    RUN_LABELS[run_name],
                    percent(item["answer_correct"]),
                    percent(item["evidence_ids_correct"]),
                    percent(item["evidence_quotes_correct"]),
                    percent(item["all_predicted_quotes_supported"]),
                    percent(item["valid_json"]),
                    percent(generation[run_name]["finish_reason_length_rate"]),
                    percent(item["mean_worst_position_accuracy"]),
                    percent(item["mean_position_gap"]),
                    percent(item["mean_edge_accuracy"]),
                    percent(item["mean_middle_accuracy"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "`最弱位置`和 `Gap` 先在每个 task × filler × length 条件内计算，再对 12 个条件等权平均。完整 95% 区间见 `analysis/ablation_analysis.json` 与 `analysis/ablation_contrasts.csv`。",
            "`Length-stop` 表示输出撞到 176-Token 上限；各条件的截断率和平均输出长度随 `analysis/position_cells.csv` 发布。",
            "",
            "## 相对 Base 的探索性工程门槛",
            "",
            "| 条件 | Gap 降幅 | 最弱位置增益 | 平均答案 Δ | 首尾 Δ | Gap≥50% | Worst≥+10pp | 答案≥−2pp | 首尾≥−2pp | JSON≥99% | 全部 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run_name in RUN_ORDER[1:]:
        item = screening[run_name]
        checks = item["checks"]
        reduction = item["gap_reduction_fraction"]
        lines.append(
            "| "
            + " | ".join(
                [
                    RUN_LABELS[run_name],
                    percent(reduction) if reduction is not None else "不可计算",
                    pp(item["worst_position_gain"]),
                    pp(item["mean_answer_delta"]),
                    pp(item["edge_accuracy_delta"]),
                    check_mark(checks["gap_reduction_at_least_50pct"]),
                    check_mark(checks["worst_position_gain_at_least_10pp"]),
                    check_mark(checks["mean_answer_drop_no_more_than_2pp"]),
                    check_mark(checks["edge_accuracy_drop_no_more_than_2pp"]),
                    check_mark(checks["valid_json_at_least_99pct"]),
                    check_mark(item["passes_all_exploratory_checks"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 2×2 处理效应",
            "",
            "下表为百分点效应 `[配对 bootstrap 95% CI]`。Gap 的负值表示差距缩小；其他准确率指标的正值表示提高。",
            "",
            "| 对比 | 答案 Δ | 证据 ID Δ | 精确引用 Δ | 最弱位置 Δ | Gap Δ | 首尾 Δ | JSON Δ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    contrast_names = (
        "paired_minus_independent_main_effect",
        "evidence_minus_answer_main_effect",
        "pairing_x_supervision_interaction",
    )
    contrast_stats = (
        "answer_correct",
        "evidence_ids_correct",
        "evidence_quotes_correct",
        "mean_worst_position_accuracy",
        "mean_position_gap",
        "mean_edge_accuracy",
        "valid_json",
    )
    for contrast_name in contrast_names:
        statistics = analysis["contrasts"][contrast_name]["statistics"]
        lines.append(
            "| "
            + " | ".join(
                [CONTRAST_LABELS[contrast_name]]
                + [interval_effect(statistics[statistic]) for statistic in contrast_stats]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "Holm 校正后的探索性 p 值保存在机器可读对比表中；本报告优先解释效应量与区间，不把单 seed bootstrap 当成最终论文显著性。",
            "",
            "## 自动化结论摘要",
            "",
            (
                "- 满足全部探索性工程门槛的条件："
                + ("、".join(RUN_LABELS[name] for name in passed) if passed else "无")
                + "。"
            ),
            f"- 最弱位置准确率最高：{RUN_LABELS[best_worst]}（{percent(summaries[best_worst]['mean_worst_position_accuracy'])}）。",
            f"- 平均位置 gap 最小：{RUN_LABELS[smallest_gap]}（{percent(summaries[smallest_gap]['mean_position_gap'])}）。",
            f"- Evidence − answer 的答案主效应：{interval_effect(evidence_effect['answer_correct'])}；精确引用主效应：{interval_effect(evidence_effect['evidence_quotes_correct'])}。",
            f"- Paired − independent 的答案主效应：{interval_effect(pairing_effect['answer_correct'])}；gap 主效应：{interval_effect(pairing_effect['mean_position_gap'])}。",
            "",
            "## 可复现性与结论边界",
            "",
            "- 逐样本预测、run 身份、条件格、bootstrap 抽样索引、原始 CSV、PNG/SVG 图与遥测必须随包发布；本报告不是数据替代品。",
            "- Evidence/answer 在各 pairing 层内共享相同输入，适合估计监督目标的处理效应；completion 长度不同，训练 Loss 不可横向比较。",
            "- Paired 训练覆盖 250 个事实的四个位置，independent 覆盖 1,000 个事实的单位置，因此 pairing 对比同时改变了事实多样性，不能归因于唯一机制。",
            "- 数据是规则生成的 KV 与聚集式两跳任务；32K、未见 filler 和未见位置可测试迁移，但不能代表自然文档、代码、分散多跳或 64K/128K。",
            "- 若方向保留，主实验至少需要三个训练/数据 seed、自然语义与对抗数据、更多模型，并对训练事实多样性做额外控制。",
            "",
            "## 图表与源数据",
            "",
            "- `figures/position_curves.svg`：2 任务 × 2 长度的七位置曲线与配对 bootstrap 区间。",
            "- `figures/ablation_summary.svg`：平均答案、最弱位置和 gap 的估计与区间。",
            "- `analysis/position_cells.csv`：五个 run 的 84 个原始条件格。",
            "- `analysis/ablation_contrasts.csv`：全部效应、95% 区间、bootstrap p 与 Holm 校正。",
            "- `reproducibility/`：代码快照、输入哈希、依赖、GPU 信息与进度遥测。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--cost-ledger", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rendered = render(args.analysis, args.validation, args.cost_ledger)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"Wrote pilot report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
