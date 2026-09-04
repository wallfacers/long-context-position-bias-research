#!/usr/bin/env bash
set -Eeuo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir=""
data_path="$root_dir/data/regression_mmlu/test.jsonl"
data_manifest="$root_dir/data/regression_mmlu/manifest.json"
adapter_root="$root_dir/outputs/formal_matched/seed_20260825"
output_dir="$root_dir/results/mmlu_regression_seed1"
artifact_path="/root/autodl-tmp/position-bias-mmlu-regression.tar.gz"
status_path="/root/autodl-tmp/position-bias-mmlu-regression.status"
checkpoint_name="checkpoint-100"
run_label="s100"
seed="20260825"
variants=(
  independent_answer
  independent_evidence_id
  independent_evidence
  paired_answer
  paired_evidence_id
  paired_evidence
)

usage() {
  echo "Usage: $0 --model LOCAL_DIR [--data JSONL] [--manifest JSON] [--adapter-root DIR] [--checkpoint-name NAME] [--run-label LABEL] [--output-dir DIR] [--artifact FILE] [--status FILE]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) model_dir="$2"; shift 2 ;;
    --data) data_path="$2"; shift 2 ;;
    --manifest) data_manifest="$2"; shift 2 ;;
    --adapter-root) adapter_root="$2"; shift 2 ;;
    --checkpoint-name) checkpoint_name="$2"; shift 2 ;;
    --run-label) run_label="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --artifact) artifact_path="$2"; shift 2 ;;
    --status) status_path="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -d "$model_dir" || ! -s "$model_dir/config.json" ]]; then
  echo "--model must be a complete local model directory" >&2
  exit 2
fi
if [[ ! -s "$data_path" || ! -s "$data_manifest" ]]; then
  echo "MMLU JSONL and manifest are required" >&2
  exit 2
fi
if [[ ! "$run_label" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "--run-label contains unsafe characters" >&2
  exit 2
fi

run_args=(--run base)
adapter_configs=()
adapter_models=()
for variant in "${variants[@]}"; do
  adapter="$adapter_root/$variant/$checkpoint_name"
  if [[ ! -s "$adapter/adapter_config.json" || ! -s "$adapter/adapter_model.safetensors" ]]; then
    echo "Missing $checkpoint_name adapter for $variant: $adapter" >&2
    exit 2
  fi
  run_args+=(--run "${variant}_${run_label}=$adapter")
  adapter_configs+=("$adapter/adapter_config.json")
  adapter_models+=("$adapter/adapter_model.safetensors")
done

case "$(realpath -m "$output_dir")/" in
  "$(realpath "$root_dir")/"*) ;;
  *) echo "--output-dir must be inside the project root" >&2; exit 2 ;;
esac

expected_rows="$(python3 - "$data_path" "$data_manifest" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

data = Path(sys.argv[1])
manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
digest = hashlib.sha256(data.read_bytes()).hexdigest()
if manifest.get("status") != "validated" or manifest.get("output_sha256") != digest:
    raise SystemExit("MMLU manifest/hash validation failed")
if manifest.get("dataset_revision") != "c30699e8356da336a370243923dbaf21066bb9fe":
    raise SystemExit("MMLU dataset revision differs from preregistration")
if int(manifest.get("subjects", 0)) != 57:
    raise SystemExit("MMLU subject count differs")
print(int(manifest["rows"]))
PY
)"
if [[ "$expected_rows" -ne 14042 ]]; then
  echo "Expected full 14,042-row MMLU test split, found $expected_rows" >&2
  exit 2
fi

mkdir -p "$output_dir/logs" "$output_dir/reproducibility" \
  "$(dirname "$artifact_path")" "$(dirname "$status_path")"
log_file="$output_dir/logs/eval-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$log_file") 2>&1

on_exit() {
  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    printf 'failed exit_code=%s finished_at=%s\n' "$rc" "$(date -u +%FT%TZ)" > "$status_path"
    echo "MMLU regression failed; resumable JSONL outputs were preserved."
  fi
}
trap on_exit EXIT
printf 'running stage=generation_and_regression_analysis started_at=%s\n' "$(date -u +%FT%TZ)" > "$status_path"

cd "$root_dir"
python3 -m pip freeze > "$output_dir/reproducibility/pip-freeze.txt"
nvidia-smi -q > "$output_dir/reproducibility/nvidia-smi-q.txt"
uname -a > "$output_dir/reproducibility/uname.txt"
sha256sum "$data_path" "$data_manifest" "$model_dir/config.json" \
  "${adapter_configs[@]}" "${adapter_models[@]}" \
  > "$output_dir/reproducibility/input-sha256.txt"

export VLLM_USE_FLASHINFER_SAMPLER=0
python3 scripts/evaluate_suite_vllm.py \
  --model "$model_dir" \
  --data "$data_path" \
  --output-dir "$output_dir" \
  "${run_args[@]}" \
  --batch-size 32 \
  --max-model-len 2048 \
  --max-num-seqs 32 \
  --max-new-tokens 32 \
  --gpu-memory-utilization 0.88 \
  --max-lora-rank 16 \
  --enforce-eager \
  --seed "$seed"

result_files=()
for run_name in base \
  "independent_answer_${run_label}" "independent_evidence_id_${run_label}" "independent_evidence_${run_label}" \
  "paired_answer_${run_label}" "paired_evidence_id_${run_label}" "paired_evidence_${run_label}"; do
  result="$output_dir/$run_name.jsonl"
  count="$(python3 - "$result" <<'PY'
import sys
from pathlib import Path
print(sum(1 for line in Path(sys.argv[1]).open(encoding="utf-8") if line.strip()))
PY
)"
  if [[ "$count" -ne "$expected_rows" ]]; then
    echo "$run_name: expected $expected_rows rows, found $count" >&2
    exit 1
  fi
  result_files+=("$result")
done

python3 scripts/aggregate_results.py "${result_files[@]}" --output "$output_dir/summary.json"
analysis_dir="$output_dir/general_regression_analysis"
python3 scripts/analyze_general_regression.py \
  --run "base=$output_dir/base.jsonl" \
  --run "independent_answer=$output_dir/independent_answer_${run_label}.jsonl" \
  --run "independent_evidence_id=$output_dir/independent_evidence_id_${run_label}.jsonl" \
  --run "independent_evidence=$output_dir/independent_evidence_${run_label}.jsonl" \
  --run "paired_answer=$output_dir/paired_answer_${run_label}.jsonl" \
  --run "paired_evidence_id=$output_dir/paired_evidence_id_${run_label}.jsonl" \
  --run "paired_evidence=$output_dir/paired_evidence_${run_label}.jsonl" \
  --output-dir "$analysis_dir" \
  --bootstrap-replicates 5000 \
  --seed 20260828 \
  --noninferiority-margin 0.02

python3 - "$output_dir" "$expected_rows" "$checkpoint_name" "$run_label" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
payload = {
    "schema_version": "mmlu-regression-completion-v1",
    "status": "validated",
    "rows_per_run": int(sys.argv[2]),
    "checkpoint_name": sys.argv[3],
    "run_label": sys.argv[4],
    "protocol": "full MMLU test, zero-shot format-robust generative option-letter accuracy",
    "scoring_protocol": "format-robust-option-extraction-v1",
    "paired_analysis": "general_regression_analysis/general_regression_analysis.json",
    "bootstrap_replicates": 5000,
    "noninferiority_margin": 0.02,
}
(output / "completion.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

output_rel="$(realpath -m --relative-to="$root_dir" "$output_dir")"
tar -C "$root_dir" -czf "$artifact_path" "$output_rel"
sha256sum "$artifact_path" > "$artifact_path.sha256"
sha256sum -c "$artifact_path.sha256"
printf 'validated exit_code=0 artifact=%s finished_at=%s\n' "$artifact_path" "$(date -u +%FT%TZ)" > "$status_path"
touch "$output_dir/RESULTS_READY_FOR_AGENT_REVIEW"
echo "MMLU short-context regression completed and packaged."
echo "Instance intentionally remains running; no shutdown command was issued."
trap - EXIT
