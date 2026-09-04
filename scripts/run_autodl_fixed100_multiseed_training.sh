#!/usr/bin/env bash
set -Eeuo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir=""
data_root=""
output_root=""
seeds_csv="20260825,20260826,20260827"
artifact_dir="/root/autodl-tmp/fixed100-multiseed-artifacts"
status_path="/root/autodl-tmp/fixed100-multiseed-training.status"
fixed_steps=100

usage() {
  echo "Usage: $0 --model LOCAL_DIR --data-root DIR --output-root DIR [--seeds CSV] [--artifact-dir DIR] [--status FILE] [--fixed-steps INT]"
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) model_dir="$2"; shift 2 ;;
    --data-root) data_root="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    --seeds) seeds_csv="$2"; shift 2 ;;
    --artifact-dir) artifact_dir="$2"; shift 2 ;;
    --status) status_path="$2"; shift 2 ;;
    --fixed-steps) fixed_steps="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -s "$model_dir/config.json" || ! -s "$model_dir/model_manifest.json" ]]; then
  echo "A complete, manifested local model is required" >&2
  exit 2
fi
if [[ ! -d "$data_root" || -z "$output_root" ]]; then
  usage >&2
  exit 2
fi
if [[ ! "$fixed_steps" =~ ^[1-9][0-9]*$ || "$fixed_steps" -ge 2000 ]]; then
  echo "--fixed-steps must be an integer in [1, 1999]" >&2
  exit 2
fi
for target in "$data_root" "$output_root"; do
  case "$(realpath -m "$target")/" in
    "$(realpath "$root_dir")/"*) ;;
    *) echo "Data and output roots must be inside the project root: $target" >&2; exit 2 ;;
  esac
done

IFS=',' read -r -a seeds <<< "$seeds_csv"
if [[ "${#seeds[@]}" -eq 0 ]]; then
  echo "At least one seed is required" >&2
  exit 2
fi
for seed in "${seeds[@]}"; do
  seed="${seed//[[:space:]]/}"
  if [[ ! "$seed" =~ ^[0-9]+$ || ! -s "$data_root/seed_$seed/manifest.json" ]]; then
    echo "Missing or invalid frozen data for seed $seed" >&2
    exit 2
  fi
done

mkdir -p "$output_root" "$artifact_dir" "$(dirname "$status_path")"
printf 'running seeds=%s started_at=%s\n' "$seeds_csv" "$(date -u +%FT%TZ)" > "$status_path"
on_exit() {
  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    printf 'failed exit_code=%s seeds=%s finished_at=%s\n' \
      "$rc" "$seeds_csv" "$(date -u +%FT%TZ)" > "$status_path"
    echo "Multi-seed training stopped on a failed seed; completed checkpoints remain resumable."
  fi
}
trap on_exit EXIT
printf 'running stage=training_seeds seeds=%s started_at=%s\n' "$seeds_csv" "$(date -u +%FT%TZ)" > "$status_path"

cd "$root_dir"
if [[ -s "$model_dir/model_integrity_attestation.json" ]]; then
  model_attestation="$model_dir/model_integrity_attestation.json"
else
  model_attestation="$output_root/model_integrity_attestation.json"
fi
for seed in "${seeds[@]}"; do
  seed="${seed//[[:space:]]/}"
  bash scripts/run_autodl_fixed100_training_queue.sh \
    --model "$model_dir" \
    --data-dir "$data_root/seed_$seed" \
    --output-root "$output_root/seed_$seed" \
    --seed "$seed" \
    --artifact "$artifact_dir/fixed${fixed_steps}-seed-$seed.tar.gz" \
    --status "$artifact_dir/fixed${fixed_steps}-seed-$seed.status" \
    --model-attestation "$model_attestation" \
    --fixed-steps "$fixed_steps"
done

python3 - "$output_root" "$artifact_dir" "$seeds_csv" "$fixed_steps" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output, artifacts = Path(sys.argv[1]), Path(sys.argv[2])
seeds = [int(value.strip()) for value in sys.argv[3].split(",") if value.strip()]
fixed_steps = int(sys.argv[4])
records = {}
for seed in seeds:
    audit = output / f"seed_{seed}" / "training_diagnostics" / f"fixed{fixed_steps}-audit.json"
    artifact = artifacts / f"fixed{fixed_steps}-seed-{seed}.tar.gz"
    checksum = artifact.with_suffix(artifact.suffix + ".sha256")
    if not audit.is_file() or not artifact.is_file() or not checksum.is_file():
        raise SystemExit(f"Seed {seed} lacks audited training outputs")
    payload = json.loads(audit.read_text(encoding="utf-8"))
    if payload.get("status") != "validated" or len(payload.get("runs", {})) != 6:
        raise SystemExit(f"Seed {seed} audit failed")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if checksum.read_text(encoding="utf-8").split()[0] != digest:
        raise SystemExit(f"Seed {seed} package hash failed")
    records[str(seed)] = {
        "audit": str(audit.resolve()),
        "artifact": str(artifact.resolve()),
        "artifact_sha256": digest,
    }
report = {
    "schema_version": f"fixed{fixed_steps}-multiseed-training-completion-v1",
    "status": "validated",
    "seeds": seeds,
    "conditions_per_seed": 6,
    "optimizer_steps": fixed_steps,
    "checkpoint_selection": "none",
    "records": records,
}
(output / "multiseed_completion.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

printf 'validated exit_code=0 seeds=%s finished_at=%s\n' \
  "$seeds_csv" "$(date -u +%FT%TZ)" > "$status_path"
touch "$output_root/MULTISEED_RESULTS_READY_FOR_AGENT_REVIEW"
echo "All requested fixed-$fixed_steps seeds completed and passed audit."
echo "Instance intentionally remains running; no shutdown command was issued."
trap - EXIT
