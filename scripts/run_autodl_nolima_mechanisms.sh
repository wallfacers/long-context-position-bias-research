#!/usr/bin/env bash
set -Eeuo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir=""
source_data="$root_dir/data/ood_nolima/hard_gate.jsonl"
source_manifest="$root_dir/data/ood_nolima/hard_gate.manifest.json"
diagnostic_data="$root_dir/data/ood_nolima/hard_gate_diagnostics.jsonl"
diagnostic_manifest="$root_dir/data/ood_nolima/hard_gate_diagnostics.manifest.json"
free_result_dir="$root_dir/results/nolima_hard_gate_seed1"
adapter_root="$root_dir/outputs/formal_matched/seed_20260825"
output_dir="$root_dir/results/nolima_mechanisms_seed1"
artifact_path="/root/autodl-tmp/position-bias-nolima-mechanisms.tar.gz"
status_path="/root/autodl-tmp/position-bias-nolima-mechanisms.status"
checkpoint_name="checkpoint-100"
run_label="s100"
seed="20260825"
free_seed=""
variants=(independent_answer independent_evidence paired_answer paired_evidence)

usage() {
  echo "Usage: $0 --model LOCAL_DIR [--source-data JSONL] [--source-manifest JSON] [--diagnostic-data JSONL] [--diagnostic-manifest JSON] [--free-result-dir DIR] [--free-seed INT] [--adapter-root DIR] [--checkpoint-name NAME] [--run-label LABEL] [--output-dir DIR] [--artifact FILE] [--status FILE]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) model_dir="$2"; shift 2 ;;
    --source-data) source_data="$2"; shift 2 ;;
    --source-manifest) source_manifest="$2"; shift 2 ;;
    --diagnostic-data) diagnostic_data="$2"; shift 2 ;;
    --diagnostic-manifest) diagnostic_manifest="$2"; shift 2 ;;
    --free-result-dir) free_result_dir="$2"; shift 2 ;;
    --free-seed) free_seed="$2"; shift 2 ;;
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
if [[ -n "$free_seed" && ! "$free_seed" =~ ^[0-9]+$ ]]; then
  echo "--free-seed must be an integer" >&2
  exit 2
fi
if [[ ! "$run_label" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "--run-label contains unsafe characters" >&2
  exit 2
fi
for required in "$source_data" "$source_manifest" "$diagnostic_data" "$diagnostic_manifest"; do
  if [[ ! -s "$required" ]]; then
    echo "Missing required NoLiMa mechanism input: $required" >&2
    exit 2
  fi
done

run_args=(--run base)
run_names=(base)
adapter_configs=()
adapter_models=()
for variant in "${variants[@]}"; do
  adapter="$adapter_root/$variant/$checkpoint_name"
  if [[ ! -s "$adapter/adapter_config.json" || ! -s "$adapter/adapter_model.safetensors" ]]; then
    echo "Missing $checkpoint_name adapter: $adapter" >&2
    exit 2
  fi
  run_name="${variant}_${run_label}"
  run_args+=(--run "$run_name=$adapter")
  run_names+=("$run_name")
  adapter_configs+=("$adapter/adapter_config.json")
  adapter_models+=("$adapter/adapter_model.safetensors")
done

case "$(realpath -m "$output_dir")/" in
  "$(realpath "$root_dir")/"*) ;;
  *) echo "--output-dir must be inside the project root" >&2; exit 2 ;;
esac

python3 - "$source_data" "$source_manifest" "$diagnostic_data" "$diagnostic_manifest" <<'PY'
import collections
import hashlib
import json
import sys
from pathlib import Path

source, source_manifest_path, diagnostic, diagnostic_manifest_path = map(Path, sys.argv[1:])
source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
diagnostic_manifest = json.loads(diagnostic_manifest_path.read_text(encoding="utf-8"))
if source_manifest.get("output_sha256") != hashlib.sha256(source.read_bytes()).hexdigest():
    raise SystemExit("NoLiMa source hash validation failed")
if diagnostic_manifest.get("source_sha256") != hashlib.sha256(source.read_bytes()).hexdigest():
    raise SystemExit("NoLiMa diagnostic source lineage failed")
if diagnostic_manifest.get("output_sha256") != hashlib.sha256(diagnostic.read_bytes()).hexdigest():
    raise SystemExit("NoLiMa diagnostic hash validation failed")
counts = collections.Counter()
with diagnostic.open(encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            counts[json.loads(line)["evaluation_mode"]] += 1
expected = {"locate_only": 1050, "oracle_long": 150, "oracle_short": 150}
if dict(counts) != expected:
    raise SystemExit(f"NoLiMa diagnostic mode counts differ: {dict(counts)}")
print("Validated frozen NoLiMa mechanism inputs")
PY

free_file() {
  local variant="$1"
  if [[ "$variant" == "base" ]]; then
    printf '%s/base.jsonl' "$free_result_dir"
  elif [[ -n "$free_seed" ]]; then
    printf '%s/s%s_%s.jsonl' "$free_result_dir" "$free_seed" "$variant"
  else
    printf '%s/%s_s100.jsonl' "$free_result_dir" "$variant"
  fi
}
free_names=(base independent_answer independent_evidence paired_answer paired_evidence)
for run_name in "${free_names[@]}"; do
  file="$(free_file "$run_name")"
  if [[ ! -s "$file" || "$(wc -l < "$file")" -ne 1050 ]]; then
    echo "Free NoLiMa gate result is incomplete: $file" >&2
    exit 2
  fi
done

mkdir -p "$output_dir/logs" "$output_dir/reproducibility" \
  "$(dirname "$artifact_path")" "$(dirname "$status_path")"
log_file="$output_dir/logs/eval-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$log_file") 2>&1

on_exit() {
  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    printf 'failed exit_code=%s finished_at=%s\n' "$rc" "$(date -u +%FT%TZ)" > "$status_path"
    echo "NoLiMa mechanism evaluation failed; resumable outputs were preserved."
  fi
}
trap on_exit EXIT
printf 'running stage=generation_and_mechanism_analysis started_at=%s\n' "$(date -u +%FT%TZ)" > "$status_path"

cd "$root_dir"
python3 -m pip freeze > "$output_dir/reproducibility/pip-freeze.txt"
nvidia-smi -q > "$output_dir/reproducibility/nvidia-smi-q.txt"
uname -a > "$output_dir/reproducibility/uname.txt"
sha256sum "$source_data" "$source_manifest" "$diagnostic_data" \
  "$diagnostic_manifest" "$model_dir/config.json" \
  "${adapter_configs[@]}" "${adapter_models[@]}" \
  > "$output_dir/reproducibility/input-sha256.txt"

export VLLM_USE_FLASHINFER_SAMPLER=0
python3 scripts/evaluate_suite_vllm.py \
  --model "$model_dir" \
  --data "$diagnostic_data" \
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

for run_name in "${run_names[@]}"; do
  if [[ "$(wc -l < "$output_dir/$run_name.jsonl")" -ne 1350 ]]; then
    echo "Diagnostic result incomplete: $run_name" >&2
    exit 1
  fi
done

analysis_dir="$output_dir/mechanism_analysis"
python3 scripts/analyze_nolima_mechanisms.py \
  --source-data "$source_data" \
  --diagnostic-data "$diagnostic_data" \
  --free-run "base=$(free_file base)" \
  --free-run "independent_answer=$(free_file independent_answer)" \
  --free-run "independent_evidence=$(free_file independent_evidence)" \
  --free-run "paired_answer=$(free_file paired_answer)" \
  --free-run "paired_evidence=$(free_file paired_evidence)" \
  --diagnostic-run "base=$output_dir/base.jsonl" \
  --diagnostic-run "independent_answer=$output_dir/independent_answer_${run_label}.jsonl" \
  --diagnostic-run "independent_evidence=$output_dir/independent_evidence_${run_label}.jsonl" \
  --diagnostic-run "paired_answer=$output_dir/paired_answer_${run_label}.jsonl" \
  --diagnostic-run "paired_evidence=$output_dir/paired_evidence_${run_label}.jsonl" \
  --output-dir "$analysis_dir" \
  --bootstrap-replicates 5000 \
  --seed 20260828
python3 scripts/plot_nolima_mechanisms.py \
  --analysis "$analysis_dir/nolima_mechanism_analysis.json" \
  --output-dir "$output_dir/figures"

python3 - "$output_dir" "$checkpoint_name" "$run_label" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
payload = {
    "schema_version": "nolima-mechanism-completion-v1",
    "status": "validated",
    "checkpoint_name": sys.argv[2],
    "run_label": sys.argv[3],
    "runs": 5,
    "rows_per_run": 1350,
    "case_clusters": 10,
    "oracle_deduplicated_across_positions": True,
    "analysis": "mechanism_analysis/nolima_mechanism_analysis.json",
    "figures": "figures/figures.metadata.json",
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
echo "NoLiMa mechanism decomposition completed and packaged."
echo "Instance intentionally remains running; no shutdown command was issued."
trap - EXIT
