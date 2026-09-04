#!/usr/bin/env bash
set -Eeuo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir=""
data_dir=""
output_root=""
seed=""
artifact_path=""
status_path=""
model_attestation=""
train_venv="${POSITION_BIAS_TRAIN_VENV:-/root/autodl-tmp/venvs/train}"
fixed_steps=100
variants=(
  independent_answer
  independent_evidence_id
  independent_evidence
  paired_answer
  paired_evidence_id
  paired_evidence
)

usage() {
  echo "Usage: $0 --model LOCAL_DIR --data-dir DIR --output-root DIR --seed INT --artifact FILE --status FILE [--model-attestation FILE] [--train-venv DIR] [--fixed-steps INT]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) model_dir="$2"; shift 2 ;;
    --data-dir) data_dir="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --artifact) artifact_path="$2"; shift 2 ;;
    --status) status_path="$2"; shift 2 ;;
    --model-attestation) model_attestation="$2"; shift 2 ;;
    --train-venv) train_venv="$2"; shift 2 ;;
    --fixed-steps) fixed_steps="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -d "$model_dir" || ! -s "$model_dir/config.json" ]]; then
  echo "--model must be a complete local model directory" >&2
  exit 2
fi
if [[ ! -d "$data_dir" || ! -s "$data_dir/manifest.json" ]]; then
  echo "--data-dir must contain manifest.json" >&2
  exit 2
fi
if [[ -z "$output_root" || -z "$seed" || -z "$artifact_path" || -z "$status_path" ]]; then
  usage >&2
  exit 2
fi
if [[ ! "$fixed_steps" =~ ^[1-9][0-9]*$ || "$fixed_steps" -ge 2000 ]]; then
  echo "--fixed-steps must be an integer in [1, 1999]" >&2
  exit 2
fi
if [[ ! -x "$train_venv/bin/python" || ! -s "$train_venv/bin/activate" ]]; then
  echo "Training virtual environment is missing: $train_venv" >&2
  exit 2
fi
# The top-level paper queue intentionally runs in the vLLM evaluation
# environment. Training is a child process, so activating the dedicated
# QLoRA environment here does not alter the parent's later evaluation PATH.
source "$train_venv/bin/activate"
hash -r
case "$(realpath -m "$output_root")/" in
  "$(realpath "$root_dir")/"*) ;;
  *) echo "--output-root must be inside the project root" >&2; exit 2 ;;
esac

mkdir -p "$output_root" "$(dirname "$artifact_path")" "$(dirname "$status_path")"
printf 'running seed=%s started_at=%s\n' "$seed" "$(date -u +%FT%TZ)" > "$status_path"

on_exit() {
  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    printf 'failed exit_code=%s seed=%s finished_at=%s\n' "$rc" "$seed" "$(date -u +%FT%TZ)" > "$status_path"
    echo "Fixed-$fixed_steps training queue failed; checkpoints and logs were preserved for resume."
  fi
}
trap on_exit EXIT
printf 'running stage=training_conditions seed=%s started_at=%s\n' "$seed" "$(date -u +%FT%TZ)" > "$status_path"

cd "$root_dir"
if [[ -z "$model_attestation" ]]; then
  model_attestation="$output_root/model_integrity_attestation.json"
fi
if [[ ! -s "$model_attestation" ]]; then
  python3 scripts/preflight_autodl.py \
    --mode train \
    --model "$model_dir" \
    --data "$data_dir/tokenized/${variants[0]}" \
    --manifest "$data_dir/manifest.json" \
    --require-model-manifest \
    --write-model-attestation "$model_attestation" \
    --output "$output_root"
fi
for variant in "${variants[@]}"; do
  printf 'BEGIN %s seed=%s %s\n' "$variant" "$seed" "$(date -u +%FT%TZ)"
  bash scripts/run_sft_variant.sh \
    --model "$model_dir" \
    --variant "$variant" \
    --data-dir "$data_dir" \
    --output-root "$output_root" \
    --seed "$seed" \
    --model-attestation "$model_attestation" \
    --fixed-steps "$fixed_steps" \
    --canary
  printf 'DONE %s seed=%s %s\n' "$variant" "$seed" "$(date -u +%FT%TZ)"
done

diagnostics_dir="$output_root/training_diagnostics"
mkdir -p "$diagnostics_dir"
python3 - "$output_root" "$data_dir/manifest.json" "$model_dir" "$seed" "$diagnostics_dir/fixed${fixed_steps}-audit.json" "$fixed_steps" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
data_manifest = Path(sys.argv[2])
model = Path(sys.argv[3])
seed = int(sys.argv[4])
audit_path = Path(sys.argv[5])
fixed_steps = int(sys.argv[6])
variants = (
    "independent_answer",
    "independent_evidence_id",
    "independent_evidence",
    "paired_answer",
    "paired_evidence_id",
    "paired_evidence",
)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

runs = {}
for variant in variants:
    root = output_root / variant
    checkpoint = root / f"checkpoint-{fixed_steps}"
    required = {
        "adapter_config": checkpoint / "adapter_config.json",
        "adapter_model": checkpoint / "adapter_model.safetensors",
        "trainer_state": checkpoint / "trainer_state.json",
        "canary_completion": root / "CANARY_COMPLETE.json",
        "run_config": root / "run_config.json",
    }
    missing = [name for name, path in required.items() if not path.is_file() or not path.stat().st_size]
    if missing:
        raise SystemExit(f"{variant}: missing fixed-{fixed_steps} outputs: {missing}")
    state = json.loads(required["trainer_state"].read_text(encoding="utf-8"))
    completion = json.loads(required["canary_completion"].read_text(encoding="utf-8"))
    config = json.loads(required["run_config"].read_text(encoding="utf-8"))
    if int(state.get("global_step", -1)) != fixed_steps or int(completion.get("global_step", -1)) != fixed_steps:
        raise SystemExit(f"{variant}: global_step is not exactly {fixed_steps}")
    arguments = config.get("arguments", {})
    if int(arguments.get("stop_after_steps", -1)) != fixed_steps or int(arguments.get("seed", -1)) != seed:
        raise SystemExit(f"{variant}: stopping rule or seed differs from preregistration")
    runs[variant] = {
        "global_step": fixed_steps,
        "files": {
            name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in required.items()
        },
    }

payload = {
    "schema_version": f"fixed{fixed_steps}-training-audit-v1",
    "status": "validated",
    "seed": seed,
    "model_config_sha256": sha256(model / "config.json"),
    "data_manifest_sha256": sha256(data_manifest),
    "optimizer_steps": fixed_steps,
    "stopping_rule": f"exactly {fixed_steps} optimizer steps; no test-set checkpoint selection",
    "runs": runs,
}
audit_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"Validated six fixed-{fixed_steps} training runs for seed {seed}")
PY

metric_args=()
for variant in "${variants[@]}"; do
  metric_args+=(--variant "$variant")
done
python3 scripts/export_training_metrics.py \
  --output-root "$output_root" \
  --diagnostics-dir "$diagnostics_dir" \
  --expected-steps "$fixed_steps" \
  --completion-record-name CANARY_COMPLETE.json \
  --require-complete \
  "${metric_args[@]}"
python3 scripts/plot_training_curves.py \
  --diagnostics-dir "$diagnostics_dir" \
  --ema-span 20 \
  --dpi 300 \
  --scheduler-horizon 2000 \
  --warmup-steps 60 \
  --title "$(basename "$model_dir") fixed-$fixed_steps QLoRA diagnostics (seed $seed)"
python3 scripts/capture_reproducibility.py \
  --project-root "$root_dir" \
  --model "$model_dir" \
  --data-manifest "$data_dir/manifest.json" \
  --output-root "$output_root" \
  --output "$diagnostics_dir/reproducibility.json"

output_rel="$(realpath -m --relative-to="$root_dir" "$output_root")"
data_manifest_rel="$(realpath -m --relative-to="$root_dir" "$data_dir/manifest.json")"
tar -C "$root_dir" -czf "$artifact_path" \
  "$output_rel" \
  "$data_manifest_rel" \
  scripts/run_autodl_fixed100_training_queue.sh \
  scripts/run_sft_variant.sh \
  scripts/train_qlora.py \
  scripts/export_training_metrics.py \
  scripts/plot_training_curves.py \
  scripts/capture_reproducibility.py
sha256sum "$artifact_path" > "$artifact_path.sha256"
sha256sum -c "$artifact_path.sha256"
printf 'validated exit_code=0 seed=%s artifact=%s finished_at=%s\n' \
  "$seed" "$artifact_path" "$(date -u +%FT%TZ)" > "$status_path"
touch "$output_root/FIXED${fixed_steps}_RESULTS_READY_FOR_AGENT_REVIEW"
echo "Six fixed-$fixed_steps training variants completed, audited, and packaged."
echo "Instance intentionally remains running; no shutdown command was issued."
trap - EXIT
