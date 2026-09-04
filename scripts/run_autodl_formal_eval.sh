#!/usr/bin/env bash
set -Eeuo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir=""
data_path="$root_dir/data/pilot_qwen25_7b/raw/test.jsonl"
data_manifest="$root_dir/data/pilot_qwen25_7b/manifest.json"
adapter_root="$root_dir/outputs/formal_matched"
seeds_csv="20260825,20260826,20260827"
variants_csv="independent_answer,independent_evidence_id,independent_evidence,paired_answer,paired_evidence_id,paired_evidence"
output_dir="$root_dir/results/formal_matched"
artifact_path="/root/autodl-tmp/position-bias-formal-eval.tar.gz"
status_path="/root/autodl-tmp/position-bias-formal-eval.status"
include_base=1
max_samples=""
checkpoint_name="checkpoint-100"
reuse_base_dir=""

usage() {
  echo "Usage: $0 --model LOCAL_DIR [--data JSONL] [--manifest JSON] [--adapter-root DIR] [--seeds CSV] [--variants CSV] [--checkpoint-name NAME] [--reuse-base-dir DIR] [--output-dir DIR] [--artifact FILE] [--status FILE] [--include-base 0|1] [--max-samples N]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) model_dir="$2"; shift 2 ;;
    --data) data_path="$2"; shift 2 ;;
    --manifest) data_manifest="$2"; shift 2 ;;
    --adapter-root) adapter_root="$2"; shift 2 ;;
    --seeds) seeds_csv="$2"; shift 2 ;;
    --variants) variants_csv="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --artifact) artifact_path="$2"; shift 2 ;;
    --status) status_path="$2"; shift 2 ;;
    --include-base) include_base="$2"; shift 2 ;;
    --max-samples) max_samples="$2"; shift 2 ;;
    --checkpoint-name) checkpoint_name="$2"; shift 2 ;;
    --reuse-base-dir) reuse_base_dir="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -d "$model_dir" || ! -s "$data_path" || ! -s "$data_manifest" ]]; then
  echo "Model directory, evaluation JSONL, and matching manifest are required" >&2
  exit 2
fi
if [[ "$include_base" != "0" && "$include_base" != "1" ]]; then
  echo "--include-base must be 0 or 1" >&2
  exit 2
fi

IFS=',' read -r -a seeds <<< "$seeds_csv"
IFS=',' read -r -a variants <<< "$variants_csv"
run_args=()
run_names=()
adapter_configs=()
adapter_models=()
if [[ "$include_base" == "1" ]]; then
  run_names+=(base)
  if [[ -z "$reuse_base_dir" ]]; then
    run_args+=(--run base)
  fi
fi
for seed in "${seeds[@]}"; do
  seed="${seed//[[:space:]]/}"
  for variant in "${variants[@]}"; do
    variant="${variant//[[:space:]]/}"
    adapter="$adapter_root/seed_$seed/$variant/$checkpoint_name"
    if [[ ! -s "$adapter/adapter_config.json" || ! -s "$adapter/adapter_model.safetensors" ]]; then
      echo "Missing completed adapter checkpoint: $adapter" >&2
      exit 2
    fi
    run_name="s${seed}_${variant}"
    run_args+=(--run "$run_name=$adapter")
    run_names+=("$run_name")
    adapter_configs+=("$adapter/adapter_config.json")
    adapter_models+=("$adapter/adapter_model.safetensors")
  done
done
if [[ "${#run_names[@]}" -eq 0 ]]; then
  echo "No evaluation runs selected" >&2
  exit 2
fi

case "$(realpath -m "$output_dir")/" in
  "$(realpath "$root_dir")/"*) ;;
  *) echo "--output-dir must be inside the project root for safe packaging" >&2; exit 2 ;;
esac

export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p "$output_dir/logs" "$output_dir/reproducibility" "$(dirname "$status_path")" "$(dirname "$artifact_path")"
printf 'running started_at=%s checkpoint=%s\n' "$(date -u +%FT%TZ)" "$checkpoint_name" > "$status_path"
log_file="$output_dir/logs/eval-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$log_file") 2>&1

on_exit() {
  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    printf 'failed exit_code=%s finished_at=%s\n' "$rc" "$(date -u +%FT%TZ)" > "$status_path"
    echo "Evaluation failed; JSONL outputs are resumable and were left in place."
  fi
}
trap on_exit EXIT
printf 'running stage=generation_and_factorial_analysis started_at=%s\n' "$(date -u +%FT%TZ)" > "$status_path"

cd "$root_dir"
if [[ -n "$reuse_base_dir" ]]; then
  base_run_json="$reuse_base_dir/base.jsonl.run.json"
  if [[ ! -s "$base_run_json" && -s "$reuse_base_dir/source_run_metadata/base.jsonl.run.json" ]]; then
    # Frozen result sets normalize run metadata into source_run_metadata/.
    base_run_json="$reuse_base_dir/source_run_metadata/base.jsonl.run.json"
  fi
  python3 - "$reuse_base_dir/base.jsonl" "$base_run_json" "$data_path" "$model_dir" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

result, metadata_path, data, model = map(Path, sys.argv[1:])
if not result.is_file() or not metadata_path.is_file():
    raise SystemExit("Reusable base result and metadata are required")
count = sum(1 for line in result.open(encoding="utf-8") if line.strip())
source_count = sum(1 for line in data.open(encoding="utf-8") if line.strip())
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
digest = hashlib.sha256(data.read_bytes()).hexdigest()
if count != source_count or metadata.get("data_sha256") != digest:
    raise SystemExit("Reusable base result does not match the selected data")
if os.path.realpath(metadata.get("model", "")) != os.path.realpath(model):
    raise SystemExit("Reusable base result does not match the local model")
print(f"Validated reusable base result with {count} rows")
PY
  cp "$reuse_base_dir/base.jsonl" "$output_dir/base.jsonl"
  cp "$base_run_json" "$output_dir/base.jsonl.run.json"
fi
python3 scripts/preflight_autodl.py \
  --mode eval \
  --model "$model_dir" \
  --data "$data_path" \
  --manifest "$data_manifest" \
  --require-model-manifest \
  --output "$output_dir"

python3 -m pip freeze > "$output_dir/reproducibility/pip-freeze.txt"
nvidia-smi -q > "$output_dir/reproducibility/nvidia-smi-q.txt"
uname -a > "$output_dir/reproducibility/uname.txt"
sha256sum "$data_path" "$data_manifest" "$model_dir/config.json" \
  "${adapter_configs[@]}" "${adapter_models[@]}" \
  > "$output_dir/reproducibility/input-sha256.txt"

eval_args=(
  --model "$model_dir"
  --data "$data_path"
  --output-dir "$output_dir"
  "${run_args[@]}"
  --batch-size 4
  --max-model-len 32768
  --max-num-seqs 4
  --max-new-tokens 176
  --gpu-memory-utilization 0.88
  --max-lora-rank 16
  --seed 20260825
)
if [[ -n "$max_samples" ]]; then
  eval_args+=(--max-samples "$max_samples")
fi
python3 scripts/evaluate_suite_vllm.py "${eval_args[@]}"

result_files=()
for run_name in "${run_names[@]}"; do
  result_files+=("$output_dir/$run_name.jsonl")
done
python3 scripts/aggregate_results.py "${result_files[@]}" --output "$output_dir/summary.json"
expected_variants="independent_answer,independent_evidence_id,independent_evidence,paired_answer,paired_evidence_id,paired_evidence"
normalized_variants="$(IFS=,; echo "${variants[*]}")"
if [[ "$include_base" == "1" && "$normalized_variants" == "$expected_variants" ]]; then
  for seed in "${seeds[@]}"; do
    seed="${seed//[[:space:]]/}"
    analysis_dir="$output_dir/analysis_seed_$seed"
    python3 scripts/analyze_factorial_results.py \
      --run "base=$output_dir/base.jsonl" \
      --run "independent_answer=$output_dir/s${seed}_independent_answer.jsonl" \
      --run "independent_evidence_id=$output_dir/s${seed}_independent_evidence_id.jsonl" \
      --run "independent_evidence=$output_dir/s${seed}_independent_evidence.jsonl" \
      --run "paired_answer=$output_dir/s${seed}_paired_answer.jsonl" \
      --run "paired_evidence_id=$output_dir/s${seed}_paired_evidence_id.jsonl" \
      --run "paired_evidence=$output_dir/s${seed}_paired_evidence.jsonl" \
      --output-dir "$analysis_dir" \
      --bootstrap-replicates 5000 \
      --seed 20260828
    python3 scripts/plot_factorial_results.py \
      --analysis "$analysis_dir/factorial_analysis.json" \
      --output-dir "$output_dir/figures_seed_$seed"
    python3 scripts/analyze_failure_cases.py \
      --run "base=$output_dir/base.jsonl" \
      --run "independent_answer=$output_dir/s${seed}_independent_answer.jsonl" \
      --run "independent_evidence_id=$output_dir/s${seed}_independent_evidence_id.jsonl" \
      --run "independent_evidence=$output_dir/s${seed}_independent_evidence.jsonl" \
      --run "paired_answer=$output_dir/s${seed}_paired_answer.jsonl" \
      --run "paired_evidence_id=$output_dir/s${seed}_paired_evidence_id.jsonl" \
      --run "paired_evidence=$output_dir/s${seed}_paired_evidence.jsonl" \
      --output-dir "$output_dir/failure_cases_seed_$seed" \
      --max-examples 5
  done
fi
python3 - "$data_path" "$max_samples" "$output_dir" "$checkpoint_name" "${run_names[@]}" <<'PY'
import json
import sys
from pathlib import Path

data_path = Path(sys.argv[1])
limit = int(sys.argv[2]) if sys.argv[2] else None
output_dir = Path(sys.argv[3])
checkpoint_name = sys.argv[4]
run_names = sys.argv[5:]
source_rows = sum(1 for line in data_path.open(encoding="utf-8") if line.strip())
expected = min(source_rows, limit) if limit is not None else source_rows
for name in run_names:
    path = output_dir / f"{name}.jsonl"
    count = sum(1 for line in path.open(encoding="utf-8") if line.strip())
    if count != expected:
        raise SystemExit(f"{name}: expected {expected} rows, found {count}")
report = {
    "schema_version": "formal-eval-completion-v1",
    "status": "validated",
    "data": str(data_path.resolve()),
    "rows_per_run": expected,
    "runs": run_names,
    "checkpoint_name": checkpoint_name,
    "failure_case_catalogs": sorted(
        path.relative_to(output_dir).as_posix()
        for path in output_dir.glob(
            "failure_cases_seed_*/failure_case_catalog.manifest.json"
        )
    ),
}
(output_dir / "completion.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

output_rel="$(realpath -m --relative-to="$root_dir" "$output_dir")"
tar -C "$root_dir" -czf "$artifact_path" "$output_rel"
sha256sum "$artifact_path" > "$artifact_path.sha256"
sha256sum -c "$artifact_path.sha256"
printf 'validated exit_code=0 artifact=%s finished_at=%s\n' "$artifact_path" "$(date -u +%FT%TZ)" > "$status_path"
touch "$output_dir/RESULTS_READY_FOR_AGENT_REVIEW"
echo "Formal evaluation completed and packaged."
echo "Instance intentionally remains running; stop billing explicitly in the AutoDL console after artifact review."
trap - EXIT
