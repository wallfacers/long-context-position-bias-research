#!/usr/bin/env python3
"""Paired, subject-stratified analysis of short-context capability regression."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence


VARIANTS = (
    "independent_answer",
    "independent_evidence_id",
    "independent_evidence",
    "paired_answer",
    "paired_evidence_id",
    "paired_evidence",
)
RUN_ORDER = ("base", *VARIANTS)
OPTION_RE = re.compile(r"^[\s\(\[]*([ABCD])[\s\)\].,:;!-]*$", re.IGNORECASE)
JSON_ANSWER_RE = re.compile(
    r'''["']answer["']\s*:\s*["']?([ABCD])["']?''', re.IGNORECASE
)
EXPLICIT_ANSWER_RE = re.compile(
    r"\b(?:the\s+)?(?:answer|option|choice)\s*(?:is|=|:)?\s*[\(\[]?([ABCD])[\)\]]?\b",
    re.IGNORECASE,
)


def contrast_definitions() -> dict[str, dict[str, float]]:
    contrasts = {
        f"{variant}_minus_base": {variant: 1.0, "base": -1.0}
        for variant in VARIANTS
    }
    for supervision in ("answer", "evidence_id", "evidence"):
        contrasts[f"paired_minus_independent_at_{supervision}"] = {
            f"paired_{supervision}": 1.0,
            f"independent_{supervision}": -1.0,
        }
    return contrasts


CONTRASTS = contrast_definitions()


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be CANONICAL_NAME=JSONL")
    name, path = value.split("=", 1)
    if name not in RUN_ORDER:
        raise argparse.ArgumentTypeError(f"Unknown run: {name}")
    return name, Path(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    coordinate = (len(ordered) - 1) * probability
    lower, upper = math.floor(coordinate), math.ceil(coordinate)
    if lower == upper:
        return ordered[lower]
    fraction = coordinate - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_p_two_sided(values: Sequence[float]) -> float:
    denominator = len(values) + 1
    return min(
        1.0,
        2
        * min(
            (sum(value <= 0 for value in values) + 1) / denominator,
            (sum(value >= 0 for value in values) + 1) / denominator,
        ),
    )


def holm_adjust(items: Sequence[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(items, key=lambda item: item[1])
    adjusted = {}
    running = 0.0
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - index) * value))
        adjusted[name] = running
    return adjusted


def weighted(values: dict[str, float], weights: dict[str, float]) -> float:
    return sum(values[run] * coefficient for run, coefficient in weights.items())


def target_option(row: dict[str, Any]) -> str:
    target = row.get("target")
    answer = target.get("answer") if isinstance(target, dict) else None
    normalized = str(answer or "").strip().upper()
    if normalized not in {"A", "B", "C", "D"}:
        raise ValueError(f"target answer is not one option letter: {answer!r}")
    return normalized


def extract_option(row: dict[str, Any]) -> tuple[str | None, str]:
    """Extract an option without making valid JSON a condition of correctness.

    The shared long-context evaluator requests JSON. Base models can emit the
    correct option near the start and then hit a short generation cap before
    closing the object. Treating those rows as knowledge errors confounds
    MMLU with schema obedience, so the frozen raw generation is rescored with
    a deliberately narrow, auditable extractor.
    """

    parsed = row.get("parsed")
    if isinstance(parsed, dict):
        answer = str(parsed.get("answer") or "").strip().upper()
        if answer in {"A", "B", "C", "D"}:
            return answer, "parsed_json"
    generated = str(row.get("generated_text") or "")
    match = JSON_ANSWER_RE.search(generated)
    if match:
        return match.group(1).upper(), "truncated_or_embedded_json"
    match = OPTION_RE.fullmatch(generated)
    if match:
        return match.group(1).upper(), "bare_option"
    matches = {match.group(1).upper() for match in EXPLICIT_ANSWER_RE.finditer(generated)}
    if len(matches) == 1:
        return next(iter(matches)), "explicit_answer_phrase"
    return None, "unextractable"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--noninferiority-margin", type=float, default=0.02)
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args()
    if (
        args.bootstrap_replicates < 100
        or not 0 < args.noninferiority_margin < 1
        or args.progress_every < 1
    ):
        raise SystemExit("Invalid bootstrap count or noninferiority margin")
    paths = dict(args.run)
    if set(paths) != set(RUN_ORDER) or len(args.run) != len(RUN_ORDER):
        raise SystemExit("Exactly one path is required for all seven canonical runs")
    raw = {run: read_jsonl(paths[run]) for run in RUN_ORDER}
    counts = {run: len(rows) for run, rows in raw.items()}
    if len(set(counts.values())) != 1 or not next(iter(counts.values())):
        raise SystemExit(f"Runs must be non-empty and equal-sized: {counts}")
    indexed = {}
    reference_ids: set[str] | None = None
    for run, rows in raw.items():
        table = {str(row["sample_id"]): row for row in rows}
        if len(table) != len(rows):
            raise SystemExit(f"{run}: duplicate sample IDs")
        if reference_ids is None:
            reference_ids = set(table)
        elif set(table) != reference_ids:
            raise SystemExit(f"{run}: sample IDs differ from base")
        indexed[run] = table
    assert reference_ids is not None
    by_subject: dict[str, list[str]] = defaultdict(list)
    option_scores: dict[str, dict[str, float]] = {run: {} for run in RUN_ORDER}
    extraction_sources: dict[str, Counter[str]] = {run: Counter() for run in RUN_ORDER}
    stored_score_disagreements = Counter()
    for sample_id in sorted(reference_ids):
        subject = str(indexed["base"][sample_id]["task"]).removeprefix("mmlu_")
        by_subject[subject].append(sample_id)
        for run in RUN_ORDER:
            row = indexed[run][sample_id]
            try:
                target = target_option(row)
            except ValueError as error:
                raise SystemExit(f"{run}/{sample_id}: {error}") from error
            prediction, source = extract_option(row)
            extraction_sources[run][source] += 1
            score = float(prediction == target)
            option_scores[run][sample_id] = score
            stored = row.get("answer_score")
            if stored in (0, 0.0, 1, 1.0) and float(stored) != score:
                stored_score_disagreements[run] += 1

    def estimates(selection: dict[str, list[str]]) -> dict[str, float]:
        selected = [sample_id for subject in sorted(selection) for sample_id in selection[subject]]
        return {
            run: sum(option_scores[run][sample_id] for sample_id in selected) / len(selected)
            for run in RUN_ORDER
        }

    point_overall = estimates(dict(by_subject))
    point_subject = {
        subject: estimates({subject: ids}) for subject, ids in sorted(by_subject.items())
    }
    run_samples = {run: [] for run in RUN_ORDER}
    contrast_samples = {name: [] for name in CONTRASTS}
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    indices_path = args.output_dir / "paired_bootstrap_indices.jsonl.gz"
    bootstrap_started = time.monotonic()
    with gzip.open(indices_path, "wt", encoding="utf-8") as handle:
        for replicate in range(args.bootstrap_replicates):
            selection = {
                subject: [ids[rng.randrange(len(ids))] for _ in ids]
                for subject, ids in sorted(by_subject.items())
            }
            handle.write(
                json.dumps(
                    {
                        "replicate": replicate,
                        "subject_indices": {
                            subject: [by_subject[subject].index(value) for value in values]
                            for subject, values in selection.items()
                        },
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )
            values = estimates(selection)
            for run, value in values.items():
                run_samples[run].append(value)
            for name, weights in CONTRASTS.items():
                contrast_samples[name].append(weighted(values, weights))
            completed = replicate + 1
            if completed % args.progress_every == 0 or completed == args.bootstrap_replicates:
                elapsed = time.monotonic() - bootstrap_started
                remaining = args.bootstrap_replicates - completed
                print(
                    f"bootstrap progress={completed}/{args.bootstrap_replicates} "
                    f"elapsed={elapsed:.1f}s eta={elapsed / completed * remaining:.1f}s",
                    flush=True,
                )

    intervals = {
        run: {
            "estimate": estimate,
            "ci95_low": percentile(run_samples[run], 0.025),
            "ci95_high": percentile(run_samples[run], 0.975),
            "n": len(reference_ids),
        }
        for run, estimate in point_overall.items()
    }
    contrasts = {}
    p_values = []
    for name, weights in CONTRASTS.items():
        samples = contrast_samples[name]
        p_value = bootstrap_p_two_sided(samples)
        p_values.append((name, p_value))
        contrasts[name] = {
            "estimate": weighted(point_overall, weights),
            "ci95_low": percentile(samples, 0.025),
            "ci95_high": percentile(samples, 0.975),
            "bootstrap_p_two_sided": p_value,
            "weights": weights,
        }
    for name, value in holm_adjust(p_values).items():
        contrasts[name]["holm_adjusted_p"] = value

    margin = args.noninferiority_margin
    noninferiority = {}
    for run in VARIANTS:
        item = contrasts[f"{run}_minus_base"]
        noninferiority[run] = {
            "margin": -margin,
            "difference": item["estimate"],
            "ci95_low": item["ci95_low"],
            "passes_if_ci95_low_above_margin": item["ci95_low"] > -margin,
        }
    report = {
        "schema_version": "general-regression-analysis-v1",
        "benchmark": "MMLU full test, zero-shot format-robust generative option-letter accuracy",
        "scoring_protocol": {
            "name": "format-robust-option-extraction-v1",
            "correctness_requires_valid_json": False,
            "precedence": [
                "parsed_json",
                "truncated_or_embedded_json",
                "bare_option",
                "unambiguous_explicit_answer_phrase",
            ],
            "note": "Stored answer_score is diagnostic only and is not used for MMLU correctness.",
        },
        "rows_per_run": counts,
        "source_sha256": {run: sha256_file(paths[run]) for run in RUN_ORDER},
        "subjects": {subject: len(ids) for subject, ids in sorted(by_subject.items())},
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "seed": args.seed,
            "unit": "question paired across runs and stratified by MMLU subject",
            "indices": indices_path.name,
        },
        "run_intervals": intervals,
        "subject_accuracy": point_subject,
        "contrasts": contrasts,
        "noninferiority_to_base": noninferiority,
        "generation_diagnostics": {
            run: {
                "valid_json_rate": sum(bool(row["valid_json"]) for row in rows) / len(rows),
                "option_extraction_rate": 1
                - extraction_sources[run]["unextractable"] / len(rows),
                "option_extraction_source_counts": dict(sorted(extraction_sources[run].items())),
                "stored_answer_score_disagreements": stored_score_disagreements[run],
                "finish_reason_counts": dict(
                    sorted(Counter(str(row.get("finish_reason", "unknown")) for row in rows).items())
                ),
            }
            for run, rows in raw.items()
        },
    }
    (args.output_dir / "general_regression_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "general_regression_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("run_name", "estimate", "ci95_low", "ci95_high", "n"),
        )
        writer.writeheader()
        for run in RUN_ORDER:
            writer.writerow({"run_name": run, **intervals[run]})
    print(f"Wrote short-context regression analysis to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
