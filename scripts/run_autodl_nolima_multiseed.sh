#!/usr/bin/env bash
set -Eeuo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir=""
data_path="$root_dir/data/ood_nolima/hard_gate.jsonl"
data_manifest="$root_dir/data/ood_nolima/hard_gate.manifest.json"
adapter_root=""
seeds_csv=""
checkpoint_name="checkpoint-100"
reuse_base_dir=""
output_dir="$root_dir/results/nolima_hard_multiseed"
artifact_path="/root/autodl-tmp/position-bias-nolima-multiseed.tar.gz"
status_path="/root/autodl-tmp/position-bias-nolima-multiseed.status"
variants=(
  independent_answer independent_evidence_id independent_evidence
  paired_answer paired_evidence_id paired_evidence
)

usage() {
  echo "Usage: $0 --model LOCAL_DIR --adapter-root DIR --seeds CSV [--checkpoint-name NAME] [--reuse-base-dir DIR] [--data JSONL] [--manifest JSON] [--output-dir DIR] [--artifact FILE] [--status FILE]"
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) model_dir="$2"; shift 2 ;;
    --data) data_path="$2"; shift 2 ;;
    --manifest) data_manifest="$2"; shift 2 ;;
    --adapter-root) adapter_root="$2"; shift 2 ;;
    --seeds) seeds_csv="$2"; shift 2 ;;
    --checkpoint-name) checkpoint_name="$2"; shift 2 ;;
    --reuse-base-dir) reuse_base_dir="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --artifact) artifact_path="$2"; shift 2 ;;
    --status) status_path="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -s "$model_dir/config.json" || -z "$adapter_root" || -z "$seeds_csv" ]]; then
  usage >&2
  exit 2
fi
if [[ ! -s "$data_path" || ! -s "$data_manifest" ]]; then
  echo "Frozen NoLiMa data and manifest are required" >&2
  exit 2
fi
case "$(realpath -m "$output_dir")/" in
  "$(realpath "$root_dir")/"*) ;;
  *) echo "--output-dir must be inside the project root" >&2; exit 2 ;;
esac

expected_rows="$(python3 - "$data_path" "$data_manifest" <<'PY'
import collections, hashlib, json, sys
from pathlib import Path
data, manifest_path = map(Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("status") != "validated" or manifest.get("output_sha256") != hashlib.sha256(data.read_bytes()).hexdigest():
    raise SystemExit("NoLiMa data lineage check failed")
rows = [json.loads(line) for line in data.open(encoding="utf-8") if line.strip()]
groups = collections.Counter(row["group_id"] for row in rows)
if len(groups) != 150 or collections.Counter(groups.values()) != {7: 150}:
    raise SystemExit("NoLiMa position-equivalent groups differ from the frozen protocol")
print(len(rows))
PY
)"
if [[ "$expected_rows" -ne 1050 ]]; then
  echo "Expected the frozen 1,050-row NoLiMa gate" >&2
  exit 2
fi

IFS=',' read -r -a seeds <<< "$seeds_csv"
run_args=()
run_names=(base)
if [[ -z "$reuse_base_dir" ]]; then
  run_args+=(--run base)
fi
adapter_configs=()
adapter_models=()
for seed in "${seeds[@]}"; do
  seed="${seed//[[:space:]]/}"
  if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "Invalid seed: $seed" >&2
    exit 2
  fi
  for variant in "${variants[@]}"; do
    adapter="$adapter_root/seed_$seed/$variant/$checkpoint_name"
    if [[ ! -s "$adapter/adapter_config.json" || ! -s "$adapter/adapter_model.safetensors" ]]; then
      echo "Missing adapter checkpoint: $adapter" >&2
      exit 2
    fi
    name="s${seed}_${variant}"
    run_args+=(--run "$name=$adapter")
    run_names+=("$name")
    adapter_configs+=("$adapter/adapter_config.json")
    adapter_models+=("$adapter/adapter_model.safetensors")
  done
done

mkdir -p "$output_dir/logs" "$output_dir/reproducibility" \
  "$(dirname "$artifact_path")" "$(dirname "$status_path")"
printf 'running started_at=%s checkpoint=%s seeds=%s\n' \
  "$(date -u +%FT%TZ)" "$checkpoint_name" "$seeds_csv" > "$status_path"
on_exit() {
  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    printf 'failed exit_code=%s finished_at=%s\n' "$rc" "$(date -u +%FT%TZ)" > "$status_path"
    echo "NoLiMa multi-seed evaluation failed; resumable outputs were preserved."
  fi
}
trap on_exit EXIT
printf 'running stage=generation_and_seed_analysis started_at=%s\n' "$(date -u +%FT%TZ)" > "$status_path"

cd "$root_dir"
if [[ -n "$reuse_base_dir" ]]; then
  python3 - "$reuse_base_dir/base.jsonl" "$reuse_base_dir/base.jsonl.run.json" "$data_path" "$model_dir" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
result, metadata_path, data, model = map(Path, sys.argv[1:])
if not result.is_file() or not metadata_path.is_file():
    raise SystemExit("Reusable NoLiMa base result is missing")
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
if sum(1 for line in result.open(encoding="utf-8") if line.strip()) != 1050:
    raise SystemExit("Reusable NoLiMa base is incomplete")
if metadata.get("data_sha256") != hashlib.sha256(data.read_bytes()).hexdigest():
    raise SystemExit("Reusable NoLiMa base has different data")
if os.path.realpath(metadata.get("model", "")) != os.path.realpath(model):
    raise SystemExit("Reusable NoLiMa base has different model")
PY
  cp "$reuse_base_dir/base.jsonl" "$output_dir/base.jsonl"
  cp "$reuse_base_dir/base.jsonl.run.json" "$output_dir/base.jsonl.run.json"
fi
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
  --batch-size 4 \
  --max-model-len 32768 \
  --max-num-seqs 4 \
  --max-new-tokens 176 \
  --gpu-memory-utilization 0.88 \
  --max-lora-rank 16 \
  --enforce-eager \
  --seed 20260825

result_files=()
for name in "${run_names[@]}"; do
  file="$output_dir/$name.jsonl"
  if [[ "$(wc -l < "$file")" -ne 1050 ]]; then
    echo "Incomplete NoLiMa result: $name" >&2
    exit 1
  fi
  result_files+=("$file")
done
python3 scripts/aggregate_results.py "${result_files[@]}" --output "$output_dir/summary.json"

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
    --seed 20260828 \
    --cluster-source-data "$data_path" \
    --cluster-key metadata.case_id \
    --cluster-strata-key task \
    --expected-clusters 10
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

python3 - "$output_dir" "$checkpoint_name" "$seeds_csv" <<'PY'
import json, sys
from pathlib import Path
output = Path(sys.argv[1])
payload = {
    "schema_version": "nolima-multiseed-completion-v1",
    "status": "validated",
    "checkpoint_name": sys.argv[2],
    "seeds": [int(value.strip()) for value in sys.argv[3].split(",") if value.strip()],
    "rows_per_run": 1050,
    "groups_per_run": 150,
    "positions_per_group": 7,
    "bootstrap_replicates_per_seed": 5000,
    "failure_case_catalogs": sorted(
        path.relative_to(output).as_posix()
        for path in output.glob("failure_cases_seed_*/failure_case_catalog.manifest.json")
    ),
}
(output / "completion.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

output_rel="$(realpath -m --relative-to="$root_dir" "$output_dir")"
tar -C "$root_dir" -czf "$artifact_path" "$output_rel"
sha256sum "$artifact_path" > "$artifact_path.sha256"
sha256sum -c "$artifact_path.sha256"
printf 'validated exit_code=0 artifact=%s finished_at=%s\n' \
  "$artifact_path" "$(date -u +%FT%TZ)" > "$status_path"
touch "$output_dir/RESULTS_READY_FOR_AGENT_REVIEW"
echo "NoLiMa multi-seed evaluation completed and packaged."
echo "Instance intentionally remains running; no shutdown command was issued."
trap - EXIT
