#!/usr/bin/env bash
set -Eeuo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir=""
data_path="$root_dir/data/ood_nolima/hard_gate.jsonl"
data_manifest="$root_dir/data/ood_nolima/hard_gate.manifest.json"
adapter_root="$root_dir/outputs/formal_matched/seed_20260825"
output_dir="$root_dir/results/nolima_hard_gate_seed1"
artifact_path="/root/autodl-tmp/position-bias-nolima-hard-gate.tar.gz"
status_path="/root/autodl-tmp/position-bias-nolima-hard-gate.status"
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
  echo "Usage: $0 --model LOCAL_DIR [--data JSONL] [--manifest JSON] [--adapter-root DIR] [--output-dir DIR] [--artifact FILE] [--status FILE]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) model_dir="$2"; shift 2 ;;
    --data) data_path="$2"; shift 2 ;;
    --manifest) data_manifest="$2"; shift 2 ;;
    --adapter-root) adapter_root="$2"; shift 2 ;;
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
  echo "NoLiMa gate JSONL and manifest are required" >&2
  exit 2
fi

run_args=(--run base)
adapter_configs=()
adapter_models=()
for variant in "${variants[@]}"; do
  adapter="$adapter_root/$variant/checkpoint-100"
  if [[ ! -s "$adapter/adapter_config.json" || ! -s "$adapter/adapter_model.safetensors" ]]; then
    echo "Missing checkpoint-100 adapter for $variant: $adapter" >&2
    exit 2
  fi
  run_args+=(--run "${variant}_s100=$adapter")
  adapter_configs+=("$adapter/adapter_config.json")
  adapter_models+=("$adapter/adapter_model.safetensors")
done

case "$(realpath -m "$output_dir")/" in
  "$(realpath "$root_dir")/"*) ;;
  *) echo "--output-dir must be inside the project root" >&2; exit 2 ;;
esac

expected_rows="$(python3 - "$data_path" <<'PY'
import sys
from pathlib import Path
print(sum(1 for line in Path(sys.argv[1]).open(encoding="utf-8") if line.strip()))
PY
)"
if [[ "$expected_rows" -ne 1050 ]]; then
  echo "Expected the preregistered 1,050-row NoLiMa-Hard gate, found $expected_rows" >&2
  exit 2
fi

mkdir -p "$output_dir/logs" "$output_dir/reproducibility" "$(dirname "$artifact_path")" "$(dirname "$status_path")"
log_file="$output_dir/logs/eval-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$log_file") 2>&1

on_exit() {
  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    printf 'failed exit_code=%s finished_at=%s\n' "$rc" "$(date -u +%FT%TZ)" > "$status_path"
    echo "NoLiMa gate failed; resumable JSONL outputs were preserved."
  fi
}
trap on_exit EXIT
printf 'running stage=generation_and_factorial_analysis started_at=%s\n' "$(date -u +%FT%TZ)" > "$status_path"

cd "$root_dir"
python3 - "$data_path" "$data_manifest" <<'PY'
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

data = Path(sys.argv[1])
manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
digest = hashlib.sha256(data.read_bytes()).hexdigest()
if manifest.get("status") != "validated" or manifest.get("output_sha256") != digest:
    raise SystemExit("NoLiMa manifest/hash validation failed")
groups = defaultdict(list)
for line in data.open(encoding="utf-8"):
    if line.strip():
        row = json.loads(line)
        groups[row["group_id"]].append(row)
if len(groups) != 150 or Counter(len(rows) for rows in groups.values()) != {7: 150}:
    raise SystemExit("NoLiMa matched-group validation failed")
print(f"Validated NoLiMa gate: rows=1050 groups={len(groups)}")
PY

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
  --seed "$seed"

result_files=()
for run_name in base \
  independent_answer_s100 independent_evidence_id_s100 independent_evidence_s100 \
  paired_answer_s100 paired_evidence_id_s100 paired_evidence_s100; do
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
analysis_dir="$output_dir/factorial_analysis"
python3 scripts/analyze_factorial_results.py \
  --run "base=$output_dir/base.jsonl" \
  --run "independent_answer=$output_dir/independent_answer_s100.jsonl" \
  --run "independent_evidence_id=$output_dir/independent_evidence_id_s100.jsonl" \
  --run "independent_evidence=$output_dir/independent_evidence_s100.jsonl" \
  --run "paired_answer=$output_dir/paired_answer_s100.jsonl" \
  --run "paired_evidence_id=$output_dir/paired_evidence_id_s100.jsonl" \
  --run "paired_evidence=$output_dir/paired_evidence_s100.jsonl" \
  --output-dir "$analysis_dir" \
  --bootstrap-replicates 5000 \
  --seed 20260828 \
  --cluster-source-data "$data_path" \
  --cluster-key metadata.case_id \
  --cluster-strata-key task \
  --expected-clusters 10
python3 scripts/plot_factorial_results.py \
  --analysis "$analysis_dir/factorial_analysis.json" \
  --output-dir "$output_dir/figures"
python3 scripts/analyze_failure_cases.py \
  --run "base=$output_dir/base.jsonl" \
  --run "independent_answer=$output_dir/independent_answer_s100.jsonl" \
  --run "independent_evidence_id=$output_dir/independent_evidence_id_s100.jsonl" \
  --run "independent_evidence=$output_dir/independent_evidence_s100.jsonl" \
  --run "paired_answer=$output_dir/paired_answer_s100.jsonl" \
  --run "paired_evidence_id=$output_dir/paired_evidence_id_s100.jsonl" \
  --run "paired_evidence=$output_dir/paired_evidence_s100.jsonl" \
  --output-dir "$output_dir/failure_cases" \
  --max-examples 5
python3 - "$output_dir" "$expected_rows" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
payload = {
    "schema_version": "nolima-hard-gate-completion-v2",
    "status": "validated",
    "rows_per_run": int(sys.argv[2]),
    "runs": [
        "base",
        "independent_answer_s100",
        "independent_evidence_id_s100",
        "independent_evidence_s100",
        "paired_answer_s100",
        "paired_evidence_id_s100",
        "paired_evidence_s100",
    ],
    "paired_factorial_analysis": "factorial_analysis/factorial_analysis.json",
    "publication_figures": "figures/figures.metadata.json",
    "failure_case_catalog": "failure_cases/failure_case_catalog.manifest.json",
    "bootstrap_replicates": 5000,
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
echo "NoLiMa-Hard gate completed and packaged."
echo "Instance intentionally remains running; no shutdown command was issued."
trap - EXIT
