#!/usr/bin/env bash
set -Eeuo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir=""
seeds_csv="20260826,20260827"
status_path="/root/autodl-tmp/qwen-confirmatory-queue.status"
artifact_dir="/root/autodl-tmp/qwen-confirmatory-artifacts"

usage() {
  echo "Usage: $0 --model LOCAL_DIR [--seeds CSV] [--status FILE] [--artifact-dir DIR]"
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) model_dir="$2"; shift 2 ;;
    --seeds) seeds_csv="$2"; shift 2 ;;
    --status) status_path="$2"; shift 2 ;;
    --artifact-dir) artifact_dir="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -s "$model_dir/config.json" || ! -s "$model_dir/model_manifest.json" ]]; then
  echo "A complete manifested model is required" >&2
  exit 2
fi
seed1_completion="$root_dir/results/qwen_seed1_completion.json"
if [[ ! -s "$seed1_completion" ]]; then
  echo "Qwen seed-1 OOD/transfer/regression queue must finish first" >&2
  exit 2
fi
python3 - "$seed1_completion" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
if p.get("status") != "validated" or len(p.get("suites", [])) != 5:
    raise SystemExit("Qwen seed-1 completion gate failed")
PY

mkdir -p "$artifact_dir" "$(dirname "$status_path")"
on_exit() {
  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    printf 'failed exit_code=%s finished_at=%s\n' "$rc" "$(date -u +%FT%TZ)" > "$status_path"
    echo "Qwen confirmatory queue failed; audited completed stages and resumable rows remain intact."
  fi
}
trap on_exit EXIT
cd "$root_dir"

printf 'running stage=training seeds=%s started_at=%s\n' "$seeds_csv" "$(date -u +%FT%TZ)" > "$status_path"
bash scripts/run_autodl_fixed100_multiseed_training.sh \
  --model "$model_dir" \
  --data-root "$root_dir/data/formal_matched_qwen25_7b" \
  --output-root "$root_dir/outputs/formal_matched" \
  --seeds "$seeds_csv" \
  --artifact-dir "$artifact_dir/training" \
  --status "$artifact_dir/training.status"

printf 'running stage=nolima seeds=%s started_at=%s\n' "$seeds_csv" "$(date -u +%FT%TZ)" > "$status_path"
bash scripts/run_autodl_nolima_multiseed.sh \
  --model "$model_dir" \
  --adapter-root "$root_dir/outputs/formal_matched" \
  --seeds "$seeds_csv" \
  --reuse-base-dir "$root_dir/results/nolima_hard_gate_seed1" \
  --output-dir "$root_dir/results/qwen_confirmatory_nolima" \
  --artifact "$artifact_dir/nolima-multiseed.tar.gz" \
  --status "$artifact_dir/nolima-multiseed.status"

printf 'running stage=longbench seeds=%s started_at=%s\n' "$seeds_csv" "$(date -u +%FT%TZ)" > "$status_path"
bash scripts/run_autodl_longbench_multiseed.sh \
  --model "$model_dir" \
  --adapter-root "$root_dir/outputs/formal_matched" \
  --seeds "$seeds_csv" \
  --reuse-base-dir "$root_dir/results/longbench_transfer_seed1" \
  --output-dir "$root_dir/results/qwen_confirmatory_longbench" \
  --artifact "$artifact_dir/longbench-multiseed.tar.gz" \
  --status "$artifact_dir/longbench-multiseed.status"

printf 'running stage=rule seeds=%s started_at=%s\n' "$seeds_csv" "$(date -u +%FT%TZ)" > "$status_path"
bash scripts/run_autodl_formal_eval.sh \
  --model "$model_dir" \
  --data "$root_dir/data/pilot_qwen25_7b/raw/test.jsonl" \
  --manifest "$root_dir/data/pilot_qwen25_7b/manifest.json" \
  --adapter-root "$root_dir/outputs/formal_matched" \
  --seeds "$seeds_csv" \
  --reuse-base-dir "$root_dir/results/formal_s100_seed1_frozen" \
  --output-dir "$root_dir/results/qwen_confirmatory_rule" \
  --artifact "$artifact_dir/rule-multiseed.tar.gz" \
  --status "$artifact_dir/rule-multiseed.status"

IFS=',' read -r -a seeds <<< "$seeds_csv"
if [[ "${#seeds[@]}" -ne 2 ]]; then
  echo "The frozen Qwen confirmatory analysis requires exactly two post-pilot seeds" >&2
  exit 2
fi
for index in 0 1; do
  seeds[$index]="${seeds[$index]//[[:space:]]/}"
done
if [[ "${seeds[*]}" != "20260826 20260827" ]]; then
  echo "Frozen Qwen confirmatory seeds must be 20260826,20260827 in that order" >&2
  exit 2
fi
mkdir -p "$root_dir/results/qwen_seed_level"
rule_args=(--analysis "Qwen2.5-7B:${seeds[0]}:confirmatory:$root_dir/results/qwen_confirmatory_rule/analysis_seed_${seeds[0]}/factorial_analysis.json")
rule_args+=(--analysis "Qwen2.5-7B:${seeds[1]}:confirmatory:$root_dir/results/qwen_confirmatory_rule/analysis_seed_${seeds[1]}/factorial_analysis.json")
rule_args+=(--analysis "Qwen2.5-7B:20260825:pilot:$root_dir/results/formal_s100_seed1_frozen/analysis/factorial_analysis.json")
python3 scripts/aggregate_seed_level_results.py "${rule_args[@]}" \
  --output-dir "$root_dir/results/qwen_seed_level/rule"

nolima_args=(--analysis "Qwen2.5-7B:${seeds[0]}:confirmatory:$root_dir/results/qwen_confirmatory_nolima/analysis_seed_${seeds[0]}/factorial_analysis.json")
nolima_args+=(--analysis "Qwen2.5-7B:${seeds[1]}:confirmatory:$root_dir/results/qwen_confirmatory_nolima/analysis_seed_${seeds[1]}/factorial_analysis.json")
nolima_args+=(--analysis "Qwen2.5-7B:20260825:pilot:$root_dir/results/nolima_hard_gate_seed1/factorial_analysis/factorial_analysis.json")
python3 scripts/aggregate_seed_level_results.py "${nolima_args[@]}" \
  --output-dir "$root_dir/results/qwen_seed_level/nolima"

longbench_args=(--analysis "Qwen2.5-7B:${seeds[0]}:confirmatory:$root_dir/results/qwen_confirmatory_longbench/analysis_seed_${seeds[0]}/transfer_analysis.json")
longbench_args+=(--analysis "Qwen2.5-7B:${seeds[1]}:confirmatory:$root_dir/results/qwen_confirmatory_longbench/analysis_seed_${seeds[1]}/transfer_analysis.json")
longbench_args+=(--analysis "Qwen2.5-7B:20260825:pilot:$root_dir/results/longbench_transfer_seed1/transfer_analysis/transfer_analysis.json")
python3 scripts/aggregate_seed_level_results.py "${longbench_args[@]}" \
  --output-dir "$root_dir/results/qwen_seed_level/longbench"

python3 - "$root_dir/results" "$seeds_csv" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
required = [
    root / "qwen_confirmatory_nolima/completion.json",
    root / "qwen_confirmatory_longbench/completion.json",
    root / "qwen_confirmatory_rule/completion.json",
    root / "qwen_seed_level/rule/seed_level_analysis.json",
    root / "qwen_seed_level/nolima/seed_level_analysis.json",
    root / "qwen_seed_level/longbench/seed_level_analysis.json",
]
for path in required:
    if not path.is_file():
        raise SystemExit(f"Missing Qwen confirmatory artifact: {path}")
report = {
    "schema_version": "qwen-confirmatory-completion-v1",
    "status": "validated",
    "confirmatory_seeds": [int(x.strip()) for x in sys.argv[2].split(",")],
    "pilot_seed_excluded_from_primary": 20260825,
    "artifacts": [str(path.relative_to(root)) for path in required],
}
(root / "qwen_confirmatory_completion.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

printf 'validated exit_code=0 seeds=%s finished_at=%s\n' \
  "$seeds_csv" "$(date -u +%FT%TZ)" > "$status_path"
echo "Qwen confirmatory seeds completed training, OOD, transfer, rule, and seed-level audit."
echo "Instance intentionally remains running; no shutdown command was issued."
trap - EXIT
