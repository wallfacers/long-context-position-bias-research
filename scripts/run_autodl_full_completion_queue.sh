#!/usr/bin/env bash
set -Eeuo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
qwen_model=""
mistral_model=""
status_path="/root/autodl-tmp/full-paper-completion-queue.status"

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
if [[ ! -s "$qwen_model/config.json" || ! -s "$qwen_model/model_manifest.json" ]]; then
  echo "A complete manifested Qwen snapshot is required" >&2
  exit 2
fi
if [[ ! -s "$mistral_model/config.json" ]]; then
  echo "A complete local Mistral snapshot is required" >&2
  exit 2
fi

mkdir -p "$(dirname "$status_path")"
on_exit() {
  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    printf 'failed exit_code=%s finished_at=%s\n' "$rc" "$(date -u +%FT%TZ)" > "$status_path"
    echo "Full paper queue stopped at a failed audit gate; completed rows and checkpoints remain resumable."
  fi
}
trap on_exit EXIT
cd "$root_dir"

validated_json() {
  python3 - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

path, schema = Path(sys.argv[1]), sys.argv[2]
if not path.is_file():
    raise SystemExit(1)
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "validated" or payload.get("schema_version") != schema:
    raise SystemExit(1)
PY
}

formal_completion="$root_dir/results/formal_s100_seed1_frozen/completion.json"
if ! validated_json "$formal_completion" "formal-s100-seed1-frozen-v1"; then
  printf 'running stage=finalize_qwen_seed1_formal started_at=%s\n' "$(date -u +%FT%TZ)" > "$status_path"
  bash scripts/finalize_qwen_seed1_formal.sh
fi
validated_json "$formal_completion" "formal-s100-seed1-frozen-v1"

seed1_completion="$root_dir/results/qwen_seed1_completion.json"
if ! validated_json "$seed1_completion" "qwen-seed1-completion-queue-v1"; then
  printf 'running stage=qwen_seed1_ood_transfer_regression started_at=%s\n' "$(date -u +%FT%TZ)" > "$status_path"
  bash scripts/run_autodl_qwen_seed1_completion_queue.sh --model "$qwen_model"
fi
validated_json "$seed1_completion" "qwen-seed1-completion-queue-v1"

qwen_confirmation="$root_dir/results/qwen_confirmatory_completion.json"
if ! validated_json "$qwen_confirmation" "qwen-confirmatory-completion-v1"; then
  printf 'running stage=qwen_confirmatory_training_and_eval started_at=%s\n' "$(date -u +%FT%TZ)" > "$status_path"
  bash scripts/run_autodl_qwen_confirmatory_queue.sh --model "$qwen_model"
fi
validated_json "$qwen_confirmation" "qwen-confirmatory-completion-v1"

mistral_completion="$root_dir/results/mistral_second_family_completion.json"
if ! validated_json "$mistral_completion" "mistral-second-family-completion-v1"; then
  printf 'running stage=mistral_three_seed_replication started_at=%s\n' "$(date -u +%FT%TZ)" > "$status_path"
  bash scripts/run_autodl_mistral_completion_queue.sh --model "$mistral_model"
fi
validated_json "$mistral_completion" "mistral-second-family-completion-v1"

python3 - "$root_dir/results" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
paths = {
    "qwen_seed1": root / "qwen_seed1_completion.json",
    "qwen_confirmatory": root / "qwen_confirmatory_completion.json",
    "mistral_second_family": root / "mistral_second_family_completion.json",
}
payload = {
    "schema_version": "full-paper-experiment-completion-v1",
    "status": "validated",
    "components": {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    },
}
(root / "full_paper_experiment_completion.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

python3 scripts/audit_failure_case_catalogs.py \
  --project-root "$root_dir" \
  --results-root "$root_dir/results" \
  --expected-catalogs 18 \
  --output "$root_dir/results/failure_case_catalog_audit.json"

artifact_args=(
  --artifact "qwen_formal_seed1=/root/autodl-tmp/qwen-formal-s100-seed1-frozen.tar.gz"
  --artifact "qwen_seed1_nolima=/root/autodl-tmp/qwen-seed1-completion-artifacts/nolima-hard-gate.tar.gz"
  --artifact "qwen_seed1_longbench=/root/autodl-tmp/qwen-seed1-completion-artifacts/longbench-transfer.tar.gz"
  --artifact "qwen_seed1_mmlu=/root/autodl-tmp/qwen-seed1-completion-artifacts/mmlu-regression.tar.gz"
  --artifact "qwen_seed1_ifeval=/root/autodl-tmp/qwen-seed1-completion-artifacts/ifeval-regression.tar.gz"
  --artifact "qwen_seed1_mechanisms=/root/autodl-tmp/qwen-seed1-completion-artifacts/nolima-mechanisms.tar.gz"
  --artifact "qwen_seed2_training=/root/autodl-tmp/qwen-confirmatory-artifacts/training/fixed100-seed-20260826.tar.gz"
  --artifact "qwen_seed3_training=/root/autodl-tmp/qwen-confirmatory-artifacts/training/fixed100-seed-20260827.tar.gz"
  --artifact "qwen_confirmatory_nolima=/root/autodl-tmp/qwen-confirmatory-artifacts/nolima-multiseed.tar.gz"
  --artifact "qwen_confirmatory_longbench=/root/autodl-tmp/qwen-confirmatory-artifacts/longbench-multiseed.tar.gz"
  --artifact "qwen_confirmatory_rule=/root/autodl-tmp/qwen-confirmatory-artifacts/rule-multiseed.tar.gz"
  --artifact "mistral_seed1_training=/root/autodl-tmp/mistral-completion-artifacts/training/fixed100-seed-20260825.tar.gz"
  --artifact "mistral_seed2_training=/root/autodl-tmp/mistral-completion-artifacts/training/fixed100-seed-20260826.tar.gz"
  --artifact "mistral_seed3_training=/root/autodl-tmp/mistral-completion-artifacts/training/fixed100-seed-20260827.tar.gz"
  --artifact "mistral_nolima=/root/autodl-tmp/mistral-completion-artifacts/nolima-multiseed.tar.gz"
  --artifact "mistral_longbench=/root/autodl-tmp/mistral-completion-artifacts/longbench-multiseed.tar.gz"
  --artifact "mistral_mmlu=/root/autodl-tmp/mistral-completion-artifacts/mmlu-regression.tar.gz"
  --artifact "mistral_ifeval=/root/autodl-tmp/mistral-completion-artifacts/ifeval-regression.tar.gz"
  --artifact "mistral_mechanisms=/root/autodl-tmp/mistral-completion-artifacts/nolima-mechanisms.tar.gz"
  --artifact "mistral_rule=/root/autodl-tmp/mistral-completion-artifacts/rule-multiseed.tar.gz"
)
evidence_args=(
  --evidence "experiment_completion=$root_dir/results/full_paper_experiment_completion.json"
  --evidence "failure_case_catalog_audit=$root_dir/results/failure_case_catalog_audit.json"
  --evidence "cross_family_rule=$root_dir/results/cross_family/rule/seed_level_analysis.json"
  --evidence "cross_family_nolima=$root_dir/results/cross_family/nolima/seed_level_analysis.json"
  --evidence "cross_family_longbench=$root_dir/results/cross_family/longbench/seed_level_analysis.json"
  --evidence "qwen_mmlu=$root_dir/results/mmlu_regression_seed1/general_regression_analysis_format_robust/general_regression_analysis.json"
  --evidence "qwen_ifeval=$root_dir/results/ifeval_regression_seed1/official_analysis/ifeval_analysis.json"
  --evidence "qwen_mechanisms=$root_dir/results/nolima_mechanisms_seed1/mechanism_analysis/nolima_mechanism_analysis.json"
  --evidence "mistral_mmlu=$root_dir/results/mistral_mmlu/general_regression_analysis/general_regression_analysis.json"
  --evidence "mistral_ifeval=$root_dir/results/mistral_ifeval/official_analysis/ifeval_analysis.json"
  --evidence "mistral_mechanisms=$root_dir/results/mistral_nolima_mechanisms/mechanism_analysis/nolima_mechanism_analysis.json"
  --evidence "paper_results_tex=$root_dir/paper/generated/results.tex"
  --evidence "paper_results_manifest=$root_dir/paper/generated/results.manifest.json"
  --evidence "paper_figure_pdf=$root_dir/paper/figures/factorial_position_curves.pdf"
  --evidence "paper_figure_svg=$root_dir/paper/figures/factorial_position_curves.svg"
  --evidence "paper_figure_png=$root_dir/paper/figures/factorial_position_curves.png"
  --evidence "paper_figure_csv=$root_dir/paper/figures/factorial_position_curves.csv"
  --evidence "paper_figure_alt=$root_dir/paper/figures/factorial_position_curves.alt.txt"
  --evidence "paper_figure_manifest=$root_dir/paper/figures/factorial_position_curves.manifest.json"
)
printf 'running stage=full_evidence_integrity_audit started_at=%s\n' "$(date -u +%FT%TZ)" > "$status_path"
python3 scripts/audit_full_paper_completion.py \
  --project-root "$root_dir" \
  "${artifact_args[@]}" \
  "${evidence_args[@]}" \
  --output "$root_dir/results/full_paper_evidence_manifest.json"

printf 'validated exit_code=0 finished_at=%s\n' "$(date -u +%FT%TZ)" > "$status_path"
echo "All Qwen and Mistral experiment packages and the paper evidence manifest passed audit."
echo "Instance intentionally remains running; no shutdown command was issued."
trap - EXIT
