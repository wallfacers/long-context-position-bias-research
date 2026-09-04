#!/usr/bin/env python3
"""Generate traceable LaTeX result macros and the main table from seed-level JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


VARIANTS = (
    "independent_answer",
    "independent_evidence_id",
    "independent_evidence",
    "paired_answer",
    "paired_evidence_id",
    "paired_evidence",
)
TABLE_RUNS = ("base", *VARIANTS)
SUPERVISION_LABELS = {
    "answer": "Answer",
    "evidence_id": "Evidence ID",
    "evidence": "Exact evidence",
}
KEY_FACTORIAL_CONTRASTS = (
    ("paired_minus_independent_main_effect", "Paired $-$ independent"),
    ("evidence_id_minus_answer_main_effect", "Evidence ID $-$ answer"),
    ("evidence_minus_answer_main_effect", "Exact evidence $-$ answer"),
    ("pairing_x_evidence_vs_answer", r"Pairing $\times$ exact-vs-answer"),
)
EXPECTED_PRIMARY_STATUSES = {
    "Qwen2.5-7B": ["corrective"],
    "Mistral-7B-v0.3": ["confirmatory"],
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_seed_analysis(path: Path, expected_kind: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "seed-level-analysis-v1":
        raise ValueError(f"Not a seed-level analysis: {path}")
    if payload.get("analysis_kind") != expected_kind:
        raise ValueError(f"{path} is {payload.get('analysis_kind')}, expected {expected_kind}")
    if not (
        payload.get("primary_training_seed_summary")
        or payload.get("confirmatory_only_primary_summary")
    ):
        raise ValueError(f"{path} includes pilot runs in its primary summary")
    return payload


def read_analysis(path: Path, expected_schema: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != expected_schema:
        raise ValueError(f"Unexpected analysis schema at {path}")
    return payload


def validate_primary_designations(
    reports: tuple[dict[str, Any], ...],
) -> dict[str, list[str]]:
    status_maps = [report.get("primary_statuses_by_family") for report in reports]
    if any(status_map != status_maps[0] for status_map in status_maps[1:]):
        raise ValueError("Rule, NoLiMa, and LongBench primary status maps differ")
    if status_maps[0] != EXPECTED_PRIMARY_STATUSES:
        raise ValueError(
            "Strict paper inputs must label Qwen corrective and Mistral confirmatory; "
            f"found {status_maps[0]}"
        )
    for report in reports:
        if report.get("confirmatory_only_primary_summary") is not False:
            raise ValueError(
                "Corrective Qwen plus confirmatory Mistral cannot be labeled confirmatory-only"
            )
    return status_maps[0]


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def percent(value: float) -> str:
    return f"{100 * value:.1f}"


def mean(family: dict[str, Any], run: str, statistic: str) -> float:
    return float(family[f"run:{run}|{statistic}"]["mean"])


def optional_mean(
    family: dict[str, Any], run: str, statistic: str
) -> float | None:
    item = family.get(f"run:{run}|{statistic}")
    if item is None or item.get("mean") is None:
        return None
    return float(item["mean"])


def effect_interval(
    family: dict[str, Any], contrast: str, statistic: str
) -> str:
    item = family[f"contrast:{contrast}|{statistic}"]
    if item.get("ci95_low") is None or item.get("ci95_high") is None:
        raise ValueError(
            f"Factorial contrast {contrast}/{statistic} has no seed-level interval"
        )
    return (
        f"{100 * float(item['mean']):+.1f} "
        f"[{100 * float(item['ci95_low']):+.1f}, "
        f"{100 * float(item['ci95_high']):+.1f}]"
    )


def exploratory_rule_metrics(report: dict[str, Any]) -> dict[str, float]:
    intervals = report["run_summary_intervals"]

    def estimate(run: str, statistic: str) -> float:
        value = intervals[run][statistic]["estimate"]
        if value is None:
            raise ValueError(f"Exploratory rule metric is null: {run}/{statistic}")
        return float(value)

    trained_answers = [estimate(run, "answer_correct") for run in VARIANTS]
    return {
        "ExploratoryRuleAnswerMin": min(trained_answers),
        "ExploratoryRuleAnswerMax": max(trained_answers),
        "ExploratoryPairedAnswerWorst": estimate(
            "paired_answer", "mean_worst_answer_accuracy"
        ),
        "ExploratoryPairedAnswerGap": estimate(
            "paired_answer", "mean_answer_position_gap"
        ),
        "ExploratoryIndependentEvidenceWorst": estimate(
            "independent_evidence", "mean_worst_answer_accuracy"
        ),
        "ExploratoryIndependentEvidenceGap": estimate(
            "independent_evidence", "mean_answer_position_gap"
        ),
    }


def build_rows(rule: dict[str, Any]) -> list[str]:
    rows = []
    for family_name, family in sorted(rule["families"].items()):
        for run in TABLE_RUNS:
            if run == "base":
                pairing, target = "--", "--"
            else:
                pairing, target_name = run.split("_", 1)
                pairing = pairing.title()
                target = SUPERVISION_LABELS[target_name]
            quote_accuracy = optional_mean(family, run, "evidence_quotes_correct")
            rows.append(
                " & ".join(
                    (
                        latex_escape(family_name),
                        pairing,
                        target,
                        percent(mean(family, run, "answer_correct")),
                        percent(mean(family, run, "mean_worst_answer_accuracy")),
                        percent(mean(family, run, "mean_answer_position_gap")),
                        "--" if quote_accuracy is None else percent(quote_accuracy),
                    )
                )
                + r" \\"
            )
        rows.append(r"\addlinespace")
    return rows[:-1]


def build_longbench_rows(report: dict[str, Any]) -> list[str]:
    rows = []
    slices = (
        "longbench_hotpotqa",
        "longbench_2wikimqa",
        "longbench_musique",
        "overall",
    )
    for family_name, family in sorted(report["families"].items()):
        for run in TABLE_RUNS:
            if run == "base":
                pairing, target = "--", "--"
            else:
                pairing, target_name = run.split("_", 1)
                pairing = pairing.title()
                target = SUPERVISION_LABELS[target_name]
            rows.append(
                " & ".join(
                    (
                        latex_escape(family_name),
                        pairing,
                        target,
                        *(percent(mean(family, run, slice_name)) for slice_name in slices),
                    )
                )
                + r" \\"
            )
        rows.append(r"\addlinespace")
    return rows[:-1]


def build_factorial_contrast_rows(
    rule: dict[str, Any], nolima: dict[str, Any], longbench: dict[str, Any]
) -> list[str]:
    rows = []
    for family_name in sorted(rule["families"]):
        rule_family = rule["families"][family_name]
        nolima_family = nolima["families"][family_name]
        longbench_family = longbench["families"][family_name]
        for contrast, label in KEY_FACTORIAL_CONTRASTS:
            rows.append(
                " & ".join(
                    (
                        latex_escape(family_name),
                        label,
                        effect_interval(
                            rule_family, contrast, "mean_worst_answer_accuracy"
                        ),
                        effect_interval(
                            rule_family, contrast, "mean_answer_position_gap"
                        ),
                        effect_interval(
                            nolima_family, contrast, "mean_worst_answer_accuracy"
                        ),
                        effect_interval(
                            nolima_family, contrast, "mean_answer_position_gap"
                        ),
                        effect_interval(longbench_family, contrast, "overall"),
                    )
                )
                + r" \\"
            )
        rows.append(r"\addlinespace")
    return rows[:-1]


def treatment_labels(run: str) -> tuple[str, str]:
    if run == "base":
        return "--", "--"
    pairing, target_name = run.split("_", 1)
    return pairing.title(), SUPERVISION_LABELS[target_name]


def build_regression_rows(
    reports: list[tuple[str, dict[str, Any], dict[str, Any]]],
) -> list[str]:
    rows = []
    for family_name, mmlu, ifeval in reports:
        for run in TABLE_RUNS:
            pairing, target = treatment_labels(run)
            mmlu_ni = (
                "--"
                if run == "base"
                else (
                    "Pass"
                    if mmlu["noninferiority_to_base"][run][
                        "passes_if_ci95_low_above_margin"
                    ]
                    else "Fail"
                )
            )
            ifeval_ni = (
                "--"
                if run == "base"
                else (
                    "Pass"
                    if ifeval["noninferiority_to_base_strict_prompt"][run][
                        "passes_if_ci95_low_above_margin"
                    ]
                    else "Fail"
                )
            )
            rows.append(
                " & ".join(
                    (
                        latex_escape(family_name),
                        pairing,
                        target,
                        percent(float(mmlu["run_intervals"][run]["estimate"])),
                        mmlu_ni,
                        percent(
                            float(ifeval["run_intervals"][run]["strict_prompt"]["estimate"])
                        ),
                        ifeval_ni,
                    )
                )
                + r" \\"
            )
        rows.append(r"\addlinespace")
    return rows[:-1]


def build_mechanism_rows(
    reports: list[tuple[str, dict[str, Any]]],
) -> list[str]:
    runs = (
        "base",
        "independent_answer",
        "independent_evidence",
        "paired_answer",
        "paired_evidence",
    )
    metrics = (
        "free_answer",
        "locate_quote",
        "oracle_long_answer",
        "oracle_short_answer",
    )
    rows = []
    for family_name, report in reports:
        for run in runs:
            pairing, target = treatment_labels(run)
            rows.append(
                " & ".join(
                    (
                        latex_escape(family_name),
                        pairing,
                        target,
                        *(
                            percent(float(report["run_intervals"][run][metric]["estimate"]))
                            for metric in metrics
                        ),
                    )
                )
                + r" \\"
            )
        rows.append(r"\addlinespace")
    return rows[:-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rule", type=Path, required=True)
    parser.add_argument("--nolima", type=Path, required=True)
    parser.add_argument("--longbench", type=Path, required=True)
    parser.add_argument("--qwen-exploratory-rule", type=Path, required=True)
    parser.add_argument("--qwen-mmlu", type=Path, required=True)
    parser.add_argument("--qwen-ifeval", type=Path, required=True)
    parser.add_argument("--qwen-mechanisms", type=Path, required=True)
    parser.add_argument("--mistral-mmlu", type=Path, required=True)
    parser.add_argument("--mistral-ifeval", type=Path, required=True)
    parser.add_argument("--mistral-mechanisms", type=Path, required=True)
    parser.add_argument("--output-tex", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    rule = read_seed_analysis(args.rule, "factorial")
    nolima = read_seed_analysis(args.nolima, "factorial")
    longbench = read_seed_analysis(args.longbench, "natural_transfer")
    qwen_exploratory_rule = read_analysis(
        args.qwen_exploratory_rule, "matched-factorial-analysis-v1"
    )
    qwen_mmlu = read_analysis(args.qwen_mmlu, "general-regression-analysis-v1")
    qwen_ifeval = read_analysis(args.qwen_ifeval, "ifeval-regression-analysis-v1")
    qwen_mechanisms = read_analysis(
        args.qwen_mechanisms, "nolima-mechanism-analysis-v1"
    )
    mistral_mmlu = read_analysis(args.mistral_mmlu, "general-regression-analysis-v1")
    mistral_ifeval = read_analysis(args.mistral_ifeval, "ifeval-regression-analysis-v1")
    mistral_mechanisms = read_analysis(
        args.mistral_mechanisms, "nolima-mechanism-analysis-v1"
    )
    family_sets = [set(report["families"]) for report in (rule, nolima, longbench)]
    if family_sets[1:] != family_sets[:-1]:
        raise SystemExit("Rule, NoLiMa, and LongBench family sets differ")
    primary_statuses = validate_primary_designations((rule, nolima, longbench))

    metrics = exploratory_rule_metrics(qwen_exploratory_rule)
    lines = ["% Generated from audited seed-level result JSON; do not hand-edit."]
    for name, value in metrics.items():
        lines.append(rf"\newcommand{{\{name}}}{{{percent(value)}\%}}")
    lines.append(r"\newcommand{\GeneratedMainRows}{%")
    lines.extend(build_rows(rule))
    lines.append("}")
    lines.append(r"\newcommand{\GeneratedNoLiMaRows}{%")
    lines.extend(build_rows(nolima))
    lines.append("}")
    lines.append(r"\newcommand{\GeneratedLongBenchRows}{%")
    lines.extend(build_longbench_rows(longbench))
    lines.append("}")
    lines.append(r"\newcommand{\GeneratedFactorialContrastRows}{%")
    lines.extend(build_factorial_contrast_rows(rule, nolima, longbench))
    lines.append("}")
    lines.append(r"\newcommand{\GeneratedRegressionRows}{%")
    lines.extend(
        build_regression_rows(
            [
                ("Qwen2.5-7B", qwen_mmlu, qwen_ifeval),
                ("Mistral-7B-v0.3", mistral_mmlu, mistral_ifeval),
            ]
        )
    )
    lines.append("}")
    lines.append(r"\newcommand{\GeneratedMechanismRows}{%")
    lines.extend(
        build_mechanism_rows(
            [
                ("Qwen2.5-7B", qwen_mechanisms),
                ("Mistral-7B-v0.3", mistral_mechanisms),
            ]
        )
    )
    lines.append("}")
    rendered = "\n".join(lines) + "\n"
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text(rendered, encoding="utf-8")

    sources = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in {
            "rule": args.rule,
            "nolima": args.nolima,
            "longbench": args.longbench,
            "qwen_exploratory_rule": args.qwen_exploratory_rule,
            "qwen_mmlu": args.qwen_mmlu,
            "qwen_ifeval": args.qwen_ifeval,
            "qwen_mechanisms": args.qwen_mechanisms,
            "mistral_mmlu": args.mistral_mmlu,
            "mistral_ifeval": args.mistral_ifeval,
            "mistral_mechanisms": args.mistral_mechanisms,
        }.items()
    }
    manifest = {
        "schema_version": "paper-results-generation-v1",
        "status": "validated",
        "confirmatory_only": False,
        "primary_summaries_confirmatory_only": False,
        "corrective_plus_confirmatory_primary": True,
        "primary_statuses_by_family": primary_statuses,
        "labeled_exploratory_sources": ["qwen_exploratory_rule"],
        "families": sorted(family_sets[0]),
        "metrics": metrics,
        "sources": sources,
        "output_tex_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Generated paper results for {', '.join(sorted(family_sets[0]))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
