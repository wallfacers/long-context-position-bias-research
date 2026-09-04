#!/usr/bin/env bash
set -Eeuo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
qwen_model=""
mistral_model=""
status_path="/root/autodl-tmp/strict-block96-full-queue.status"

usage() {
  echo "Usage: $0 --qwen-model LOCAL_DIR --mistral-model LOCAL_DIR [--status FILE]"
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --qwen-model) qwen_model="$2"; shift 2 ;;
    --mistral-model) mistral_model="$2"; shift 2 ;;
    --status) status_path="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ ! -s "$qwen_model/model_manifest.json" || ! -s "$mistral_model/model_manifest.json" ]]; then
  echo "Both manifested local model snapshots are required" >&2
  exit 2
fi

mkdir -p "$(dirname "$status_path")"
on_exit() {
  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    printf 'failed exit_code=%s finished_at=%s\n' "$rc" "$(date -u +%FT%TZ)" > "$status_path"
    echo "Strict block-96 full queue stopped at an audit gate; completed stages remain resumable."
  fi
}
trap on_exit EXIT
cd "$root_dir"

printf 'running stage=qwen_strict_block96 started_at=%s\n' "$(date -u +%FT%TZ)" > "$status_path"
if ! python3 - "$root_dir" "$root_dir/results/qwen_block96_completion.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, path = Path(sys.argv[1]), Path(sys.argv[2])
if not path.is_file():
    raise SystemExit(1)
payload = json.loads(path.read_text(encoding="utf-8"))
if (
    payload.get("schema_version") != "qwen-strict-block96-completion-v1"
    or payload.get("status") != "validated"
    or payload.get("strict_realized_matching") is not True
):
    raise SystemExit(1)
hashes = payload.get("artifact_sha256", {})
if not hashes:
    raise SystemExit(1)
for relative, expected in hashes.items():
    artifact = root / relative
    if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
        raise SystemExit(1)
PY
then
  bash scripts/run_autodl_qwen_block96_completion_queue.sh --model "$qwen_model"
fi

printf 'running stage=mistral_strict_block96 started_at=%s\n' "$(date -u +%FT%TZ)" > "$status_path"
if ! python3 - "$root_dir" "$root_dir/results/mistral_block96_completion.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, path = Path(sys.argv[1]), Path(sys.argv[2])
if not path.is_file():
    raise SystemExit(1)
payload = json.loads(path.read_text(encoding="utf-8"))
if (
    payload.get("schema_version") != "mistral-strict-block96-completion-v1"
    or payload.get("status") != "validated"
    or payload.get("strict_realized_matching") is not True
    or payload.get("prospective_under_corrected_protocol") is not True
):
    raise SystemExit(1)
hashes = payload.get("artifact_sha256", {})
if not hashes:
    raise SystemExit(1)
for relative, expected in hashes.items():
    artifact = root / relative
    if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
        raise SystemExit(1)
PY
then
  bash scripts/run_autodl_mistral_block96_completion_queue.sh --model "$mistral_model"
fi

python3 - "$root_dir" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
paths = [
    root / "results/qwen_block96_completion.json",
    root / "results/mistral_block96_completion.json",
    root / "results/cross_family_block96/rule/seed_level_analysis.json",
    root / "results/cross_family_block96/nolima/seed_level_analysis.json",
    root / "results/cross_family_block96/longbench/seed_level_analysis.json",
    root / "paper/generated/results.manifest.json",
    root / "paper/figures/factorial_position_curves.manifest.json",
]
for path in paths:
    if not path.is_file():
        raise SystemExit(f"Missing strict full-queue evidence: {path}")
report = {
    "schema_version": "strict-block96-experiment-completion-v1",
    "status": "validated",
    "artifacts": [str(path.relative_to(root)) for path in paths],
    "artifact_sha256": {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    },
    "paper_primary_protocol": "strict block-complete 96-row training subsets",
    "historical_fixed100_primary_eligible": False,
}
(root / "results/strict_block96_experiment_completion.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

printf 'running stage=strict_failure_catalog_audit started_at=%s\n' "$(date -u +%FT%TZ)" > "$status_path"
failure_manifest_args=()
for family in qwen mistral; do
  for suite in rule nolima longbench; do
    for seed in 20260825 20260826 20260827; do
      failure_manifest_args+=(
        --manifest "$root_dir/results/${family}_block96_${suite}/failure_cases_seed_${seed}/failure_case_catalog.manifest.json"
      )
    done
  done
done
python3 scripts/audit_failure_case_catalogs.py \
  --project-root "$root_dir" \
  "${failure_manifest_args[@]}" \
  --expected-catalogs 18 \
  --output "$root_dir/results/strict_block96_failure_case_catalog_audit.json"

printf 'running stage=strict_compute_accounting started_at=%s\n' "$(date -u +%FT%TZ)" > "$status_path"
training_accounting_args=()
for family_root in qwen_block96 mistral_block96; do
  for seed in 20260825 20260826 20260827; do
    training_accounting_args+=(--training-root "$root_dir/outputs/$family_root/seed_$seed")
  done
done
eval_accounting_args=()
for family in qwen mistral; do
  for suite in rule nolima longbench mmlu ifeval nolima_mechanisms; do
    eval_accounting_args+=(--eval-root "$root_dir/results/${family}_block96_${suite}")
  done
done
python3 scripts/summarize_compute_accounting.py \
  --project-root "$root_dir" \
  "${training_accounting_args[@]}" \
  "${eval_accounting_args[@]}" \
  --expected-training-step 96 \
  --hourly-rate 2.78 \
  --output "$root_dir/results/compute_accounting.json"

printf 'running stage=strict_full_evidence_integrity started_at=%s\n' "$(date -u +%FT%TZ)" > "$status_path"
artifact_args=()
for family in qwen mistral; do
  artifact_root="/root/autodl-tmp/${family}-block96-completion-artifacts"
  for seed in 20260825 20260826 20260827; do
    artifact_args+=(--artifact "${family}_training_${seed}=$artifact_root/training/fixed96-seed-$seed.tar.gz")
  done
  for suite in nolima longbench mmlu ifeval nolima_mechanisms rule; do
    case "$suite" in
      nolima) artifact_name="nolima-multiseed.tar.gz" ;;
      longbench) artifact_name="longbench-multiseed.tar.gz" ;;
      mmlu) artifact_name="mmlu-regression.tar.gz" ;;
      ifeval) artifact_name="ifeval-regression.tar.gz" ;;
      nolima_mechanisms) artifact_name="nolima-mechanisms.tar.gz" ;;
      rule) artifact_name="rule-multiseed.tar.gz" ;;
    esac
    artifact_args+=(--artifact "${family}_${suite}=$artifact_root/$artifact_name")
  done
done
evidence_args=(
  --evidence "experiment_completion=$root_dir/results/strict_block96_experiment_completion.json"
  --evidence "failure_case_catalog_audit=$root_dir/results/strict_block96_failure_case_catalog_audit.json"
  --evidence "compute_accounting=$root_dir/results/compute_accounting.json"
  --evidence "strict_budget=$root_dir/configs/autodl_strict_block96_budget.json"
  --evidence "qwen_fixed100_realized_subset_audit=$root_dir/results/training_subset_audits/qwen_fixed100_realized_subset.json"
  --evidence "qwen_block96_realized_subset_audit=$root_dir/results/training_subset_audits/qwen_block96_realized_subset.json"
  --evidence "mistral_block96_realized_subset_audit=$root_dir/results/training_subset_audits/mistral_block96_realized_subset.json"
  --evidence "cross_family_rule=$root_dir/results/cross_family_block96/rule/seed_level_analysis.json"
  --evidence "cross_family_nolima=$root_dir/results/cross_family_block96/nolima/seed_level_analysis.json"
  --evidence "cross_family_longbench=$root_dir/results/cross_family_block96/longbench/seed_level_analysis.json"
  --evidence "qwen_mmlu=$root_dir/results/qwen_block96_mmlu/general_regression_analysis/general_regression_analysis.json"
  --evidence "qwen_ifeval=$root_dir/results/qwen_block96_ifeval/official_analysis/ifeval_analysis.json"
  --evidence "qwen_mechanisms=$root_dir/results/qwen_block96_nolima_mechanisms/mechanism_analysis/nolima_mechanism_analysis.json"
  --evidence "mistral_mmlu=$root_dir/results/mistral_block96_mmlu/general_regression_analysis/general_regression_analysis.json"
  --evidence "mistral_ifeval=$root_dir/results/mistral_block96_ifeval/official_analysis/ifeval_analysis.json"
  --evidence "mistral_mechanisms=$root_dir/results/mistral_block96_nolima_mechanisms/mechanism_analysis/nolima_mechanism_analysis.json"
  --evidence "paper_results_tex=$root_dir/paper/generated/results.tex"
  --evidence "paper_results_manifest=$root_dir/paper/generated/results.manifest.json"
  --evidence "paper_figure_pdf=$root_dir/paper/figures/factorial_position_curves.pdf"
  --evidence "paper_figure_svg=$root_dir/paper/figures/factorial_position_curves.svg"
  --evidence "paper_figure_png=$root_dir/paper/figures/factorial_position_curves.png"
  --evidence "paper_figure_csv=$root_dir/paper/figures/factorial_position_curves.csv"
  --evidence "paper_figure_alt=$root_dir/paper/figures/factorial_position_curves.alt.txt"
  --evidence "paper_figure_manifest=$root_dir/paper/figures/factorial_position_curves.manifest.json"
)
python3 scripts/audit_full_paper_completion.py \
  --project-root "$root_dir" \
  "${artifact_args[@]}" \
  "${evidence_args[@]}" \
  --require-evidence-label compute_accounting \
  --output "$root_dir/results/full_paper_evidence_manifest.json"

printf 'validated exit_code=0 finished_at=%s\n' "$(date -u +%FT%TZ)" > "$status_path"
echo "Strict Qwen/Mistral block-96 experiment matrix, failure catalogs, compute accounting, and full evidence manifest passed audit."
echo "Instance intentionally remains running; no power action was issued."
trap - EXIT
