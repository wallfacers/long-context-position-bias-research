#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="${1:-$ROOT_DIR/position-bias-training-results.tar.gz}"
MODEL_DIR="${2:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs}"
DATA_MANIFEST="${DATA_MANIFEST:-$ROOT_DIR/data/pilot_qwen25_7b/manifest.json}"
VARIANT_CSV="${VARIANT_CSV:-independent_answer,paired_answer,independent_evidence,paired_evidence}"
IFS=',' read -r -a VARIANTS <<< "$VARIANT_CSV"

case "$(realpath -m "$OUTPUT_ROOT")/" in
  "$(realpath "$ROOT_DIR")/"*) ;;
  *) echo "OUTPUT_ROOT must be inside the project root for safe packaging" >&2; exit 2 ;;
esac
case "$(realpath -m "$DATA_MANIFEST")" in
  "$(realpath "$ROOT_DIR")/"*) ;;
  *) echo "DATA_MANIFEST must be inside the project root for safe packaging" >&2; exit 2 ;;
esac
if [[ ! -s "$DATA_MANIFEST" || "${#VARIANTS[@]}" -eq 0 ]]; then
  echo "DATA_MANIFEST must exist and VARIANT_CSV must be non-empty" >&2
  exit 2
fi

for variant in "${VARIANTS[@]}"; do
  result="$OUTPUT_ROOT/$variant/TRAINING_COMPLETE.json"
  adapter="$OUTPUT_ROOT/$variant/final_adapter/adapter_config.json"
  if [[ ! -s "$result" || ! -s "$adapter" ]]; then
    echo "Training output is incomplete for $variant" >&2
    exit 1
  fi
done

if [[ -z "$MODEL_DIR" ]]; then
  MODEL_DIR="$(python3 - "$OUTPUT_ROOT/${VARIANTS[0]}/run_config.json" <<'PY'
import json
import sys
from pathlib import Path

print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["model"])
PY
)"
fi
if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Cannot resolve the local base model for reproducibility capture: $MODEL_DIR" >&2
  exit 1
fi

DIAGNOSTICS_DIR="$OUTPUT_ROOT/training_diagnostics"
python3 "$ROOT_DIR/scripts/export_training_metrics.py" \
  --output-root "$OUTPUT_ROOT" \
  --diagnostics-dir "$DIAGNOSTICS_DIR" \
  --expected-steps 2000 \
  --require-complete
python3 "$ROOT_DIR/scripts/plot_training_curves.py" \
  --diagnostics-dir "$DIAGNOSTICS_DIR" \
  --ema-span 50 \
  --dpi 300
python3 "$ROOT_DIR/scripts/capture_reproducibility.py" \
  --project-root "$ROOT_DIR" \
  --model "$MODEL_DIR" \
  --data-manifest "$DATA_MANIFEST" \
  --output-root "$OUTPUT_ROOT" \
  --output "$DIAGNOSTICS_DIR/reproducibility.json" \
  --telemetry /root/autodl-tmp/training-telemetry.live.csv \
  --loss-health /root/autodl-tmp/training-loss-health.live.jsonl \
  --queue-log /root/autodl-tmp/train-all.log

ARTIFACT_MANIFEST="$DIAGNOSTICS_DIR/publication_artifacts.sha256"
ARTIFACT_MANIFEST_TMP="${ARTIFACT_MANIFEST}.tmp-$$"
OUTPUT_ROOT_REL="$(realpath -m --relative-to="$ROOT_DIR" "$OUTPUT_ROOT")"
DATA_MANIFEST_REL="$(realpath -m --relative-to="$ROOT_DIR" "$DATA_MANIFEST")"
DATA_ROOT="$(dirname "$DATA_MANIFEST")"
PACKAGE_DATA_PATHS=("$DATA_MANIFEST_REL")
for variant in "${VARIANTS[@]}"; do
  metadata="$DATA_ROOT/tokenized/$variant/pretokenization.json"
  if [[ -s "$metadata" ]]; then
    PACKAGE_DATA_PATHS+=("$(realpath -m --relative-to="$ROOT_DIR" "$metadata")")
  fi
done
(
  cd "$ROOT_DIR"
  find "$OUTPUT_ROOT_REL" -type f \
    ! -path '*/checkpoint-*/*' \
    ! -name 'publication_artifacts.sha256' \
    ! -name 'publication_artifacts.sha256.tmp-*' \
    -print0 \
    | sort -z \
    | xargs -0 -r sha256sum > "$ARTIFACT_MANIFEST_TMP"
)
mv "$ARTIFACT_MANIFEST_TMP" "$ARTIFACT_MANIFEST"

OUTPUT="$(realpath -m "$OUTPUT")"
mkdir -p "$(dirname "$OUTPUT")"
tar \
  --exclude='checkpoint-*' \
  --exclude='READY_TO_STOP_AUTODL_*' \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  -C "$ROOT_DIR" \
  -czf "$OUTPUT" \
  "$OUTPUT_ROOT_REL" \
  scripts \
  src \
  configs \
  docs \
  tests \
  README.md \
  pyproject.toml \
  requirements-train.txt \
  requirements-eval.txt \
  "${PACKAGE_DATA_PATHS[@]}"
(
  cd "$(dirname "$OUTPUT")"
  sha256sum "$(basename "$OUTPUT")" > "$(basename "$OUTPUT").sha256"
)

touch "$ROOT_DIR/READY_TO_STOP_AUTODL_AFTER_ALL_TRAINING"
ls -lh "$OUTPUT" "$OUTPUT.sha256"
echo "All ${#VARIANTS[@]} adapters and publication diagnostics verified and packaged."
