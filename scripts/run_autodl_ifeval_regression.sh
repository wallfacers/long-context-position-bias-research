#!/usr/bin/env bash
set -Eeuo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir=""
data_path="$root_dir/data/regression_ifeval/input_data.jsonl"
data_manifest="$root_dir/data/regression_ifeval/manifest.json"
official_root="$root_dir/third_party/google-research/instruction_following_eval"
nltk_data="$root_dir/third_party/nltk_data"
adapter_root="$root_dir/outputs/formal_matched/seed_20260825"
output_dir="$root_dir/results/ifeval_regression_seed1"
artifact_path="/root/autodl-tmp/position-bias-ifeval-regression.tar.gz"
status_path="/root/autodl-tmp/position-bias-ifeval-regression.status"
checkpoint_name="checkpoint-100"
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
  echo "Usage: $0 --model LOCAL_DIR [--data JSONL] [--manifest JSON] [--official-root DIR] [--adapter-root DIR] [--checkpoint-name NAME] [--output-dir DIR] [--artifact FILE] [--status FILE] [--seed INT]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) model_dir="$2"; shift 2 ;;
    --data) data_path="$2"; shift 2 ;;
    --manifest) data_manifest="$2"; shift 2 ;;
    --official-root) official_root="$2"; shift 2 ;;
    --adapter-root) adapter_root="$2"; shift 2 ;;
    --checkpoint-name) checkpoint_name="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --artifact) artifact_path="$2"; shift 2 ;;
    --status) status_path="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -d "$model_dir" || ! -s "$model_dir/config.json" ]]; then
  echo "--model must be a complete local model directory" >&2
  exit 2
fi
if [[ ! -s "$data_path" || ! -s "$data_manifest" ]]; then
  echo "IFEval JSONL and manifest are required" >&2
  exit 2
fi
for official_file in data/input_data.jsonl evaluation_lib.py instructions.py \
  instructions_registry.py instructions_util.py; do
  if [[ ! -s "$official_root/$official_file" ]]; then
    echo "Missing official IFEval file: $official_root/$official_file" >&2
    exit 2
  fi
done
if [[ ! -s "$nltk_data/punkt-tab-english.sha256" ]]; then
  echo "Missing frozen NLTK punkt_tab hash manifest" >&2
  exit 2
fi
(cd "$root_dir" && sha256sum -c third_party/nltk_data/punkt-tab-english.sha256)
export NLTK_DATA="$nltk_data"

run_args=(--run base)
adapter_configs=()
adapter_models=()
for variant in "${variants[@]}"; do
  adapter="$adapter_root/$variant/$checkpoint_name"
  if [[ ! -s "$adapter/adapter_config.json" || ! -s "$adapter/adapter_model.safetensors" ]]; then
    echo "Missing $checkpoint_name adapter for $variant: $adapter" >&2
    exit 2
  fi
  run_args+=(--run "$variant=$adapter")
  adapter_configs+=("$adapter/adapter_config.json")
  adapter_models+=("$adapter/adapter_model.safetensors")
done

case "$(realpath -m "$output_dir")/" in
  "$(realpath "$root_dir")/"*) ;;
  *) echo "--output-dir must be inside the project root" >&2; exit 2 ;;
esac

expected_rows="$(python3 - "$data_path" "$data_manifest" "$official_root/data/input_data.jsonl" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

data, manifest_path, official = map(Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
data_hash = hashlib.sha256(data.read_bytes()).hexdigest()
official_hash = hashlib.sha256(official.read_bytes()).hexdigest()
if manifest.get("status") != "validated" or manifest.get("output_sha256") != data_hash:
    raise SystemExit("IFEval manifest/frozen-data hash validation failed")
if manifest.get("official_revision") != "041338718b4e8151372fd63677104c65b73a0a4e":
    raise SystemExit("IFEval official revision differs from preregistration")
if manifest.get("official_source_sha256") != official_hash:
    raise SystemExit("Official IFEval input differs from frozen source")
if int(manifest.get("instruction_instances", 0)) != 834:
    raise SystemExit("IFEval instruction instance count differs")
print(int(manifest["rows"]))
PY
)"
if [[ "$expected_rows" -ne 541 ]]; then
  echo "Expected all 541 official IFEval prompts, found $expected_rows" >&2
  exit 2
fi

python3 - <<'PY'
import absl
import immutabledict
import langdetect
import nltk
print("Official IFEval dependencies import successfully")
PY

mkdir -p "$output_dir/logs" "$output_dir/reproducibility" \
  "$(dirname "$artifact_path")" "$(dirname "$status_path")"
log_file="$output_dir/logs/eval-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$log_file") 2>&1

on_exit() {
  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    printf 'failed exit_code=%s finished_at=%s\n' "$rc" "$(date -u +%FT%TZ)" > "$status_path"
    echo "IFEval regression failed; resumable generation JSONLs were preserved."
  fi
}
trap on_exit EXIT
printf 'running stage=generation_and_official_scoring started_at=%s\n' "$(date -u +%FT%TZ)" > "$status_path"

cd "$root_dir"
python3 -m pip freeze > "$output_dir/reproducibility/pip-freeze.txt"
nvidia-smi -q > "$output_dir/reproducibility/nvidia-smi-q.txt"
uname -a > "$output_dir/reproducibility/uname.txt"
sha256sum "$data_path" "$data_manifest" "$model_dir/config.json" \
  "$official_root/data/input_data.jsonl" "$official_root/evaluation_lib.py" \
  "$official_root/instructions.py" "$official_root/instructions_registry.py" \
  "$official_root/instructions_util.py" "$nltk_data/punkt-tab-english.sha256" \
  "${adapter_configs[@]}" "${adapter_models[@]}" \
  > "$output_dir/reproducibility/input-sha256.txt"

export VLLM_USE_FLASHINFER_SAMPLER=0
python3 scripts/evaluate_ifeval_suite_vllm.py \
  --model "$model_dir" \
  --data "$data_path" \
  --output-dir "$output_dir/generations" \
  "${run_args[@]}" \
  --batch-size 8 \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --max-new-tokens 1024 \
  --gpu-memory-utilization 0.88 \
  --max-lora-rank 16 \
  --seed "$seed"

score_args=()
for run_name in base "${variants[@]}"; do
  result="$output_dir/generations/$run_name.jsonl"
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
  score_args+=(--run "$run_name=$result")
done

analysis_dir="$output_dir/official_analysis"
python3 scripts/score_ifeval_results.py \
  --official-root "$official_root" \
  --input-data "$official_root/data/input_data.jsonl" \
  "${score_args[@]}" \
  --output-dir "$analysis_dir" \
  --bootstrap-replicates 5000 \
  --seed 20260828 \
  --noninferiority-margin 0.02

python3 - "$output_dir" "$expected_rows" "$checkpoint_name" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
analysis = json.loads(
    (output / "official_analysis/ifeval_analysis.json").read_text(encoding="utf-8")
)
if analysis.get("prompts") != int(sys.argv[2]) or analysis.get("instruction_instances") != 834:
    raise SystemExit("Official IFEval analysis completeness check failed")
payload = {
    "schema_version": "ifeval-regression-completion-v1",
    "status": "validated",
    "rows_per_run": int(sys.argv[2]),
    "checkpoint_name": sys.argv[3],
    "instruction_instances": 834,
    "protocol": "official IFEval strict/loose prompt- and instruction-level scoring",
    "paired_analysis": "official_analysis/ifeval_analysis.json",
    "bootstrap_replicates": 5000,
    "noninferiority_margin": 0.02,
    "unconstrained_generation": True,
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
echo "IFEval instruction-following regression completed and packaged."
echo "Instance intentionally remains running; no shutdown command was issued."
trap - EXIT
