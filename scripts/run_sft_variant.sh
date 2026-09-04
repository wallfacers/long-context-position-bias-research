#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR=""
VARIANT=""
DATA_DIR="$ROOT_DIR/data/pilot_qwen25_7b"
OUTPUT_ROOT="$ROOT_DIR/outputs"
CANARY=0
MANIFEST="$DATA_DIR/manifest.json"
TRAIN_SEED=20260825
MODEL_ATTESTATION=""
FIXED_STEPS=100

usage() {
  echo "Usage: $0 --model LOCAL_DIR --variant NAME [--data-dir DIR] [--output-root DIR] [--seed INT] [--model-attestation FILE] [--fixed-steps INT] [--canary]"
  echo "Variants: independent_answer independent_evidence_id independent_evidence paired_answer paired_evidence_id paired_evidence"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_DIR="$2"; shift 2 ;;
    --variant) VARIANT="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; MANIFEST="$2/manifest.json"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --seed) TRAIN_SEED="$2"; shift 2 ;;
    --model-attestation) MODEL_ATTESTATION="$2"; shift 2 ;;
    --fixed-steps) FIXED_STEPS="$2"; shift 2 ;;
    --canary) CANARY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$MODEL_DIR" || -z "$VARIANT" ]]; then
  usage >&2
  exit 2
fi
if [[ ! "$FIXED_STEPS" =~ ^[1-9][0-9]*$ || "$FIXED_STEPS" -ge 2000 ]]; then
  echo "--fixed-steps must be an integer in [1, 1999]" >&2
  exit 2
fi
case "$VARIANT" in
  independent_answer|independent_evidence_id|independent_evidence|paired_answer|paired_evidence_id|paired_evidence) ;;
  *) echo "Invalid variant: $VARIANT" >&2; exit 2 ;;
esac

TOKENIZED_DATA="$DATA_DIR/tokenized/$VARIANT"
RAW_DATA="$DATA_DIR/sft/$VARIANT.jsonl"
if [[ -d "$TOKENIZED_DATA" ]]; then
  TRAIN_DATA="$TOKENIZED_DATA"
elif [[ -f "$RAW_DATA" ]]; then
  TRAIN_DATA="$RAW_DATA"
else
  echo "No tokenized or raw training data for $VARIANT" >&2
  exit 1
fi

OUTPUT_DIR="$OUTPUT_ROOT/$VARIANT"
mkdir -p "$OUTPUT_DIR/logs"

PREFLIGHT_ARGS=(
  --mode train
  --model "$MODEL_DIR"
  --data "$TRAIN_DATA"
  --manifest "$MANIFEST"
  --require-model-manifest
  --output "$OUTPUT_DIR"
)
if [[ -n "$MODEL_ATTESTATION" ]]; then
  PREFLIGHT_ARGS+=(--model-attestation "$MODEL_ATTESTATION")
fi
python3 "$ROOT_DIR/scripts/preflight_autodl.py" \
  "${PREFLIGHT_ARGS[@]}"

TRAIN_ARGS=(
  --model "$MODEL_DIR"
  --data "$TRAIN_DATA"
  --output "$OUTPUT_DIR"
  --max-steps 2000
  --save-steps "$FIXED_STEPS"
  --seed "$TRAIN_SEED"
  --resume auto
)
if [[ "$CANARY" -eq 1 ]]; then
  TRAIN_ARGS+=(--stop-after-steps "$FIXED_STEPS")
fi

LOG_FILE="$OUTPUT_DIR/logs/$(date -u +%Y%m%dT%H%M%SZ).log"
python3 "$ROOT_DIR/scripts/train_qlora.py" "${TRAIN_ARGS[@]}" 2>&1 | tee "$LOG_FILE"

if [[ "$CANARY" -eq 1 ]]; then
  touch "$OUTPUT_DIR/READY_TO_STOP_AUTODL_AFTER_FIXED_STEP"
  echo "Fixed-$FIXED_STEPS checkpoint completed. A parent queue may continue with later conditions; do not stop mid-queue."
else
  touch "$OUTPUT_DIR/READY_TO_STOP_AUTODL_AFTER_TRAINING"
  echo "Training completed. Review the enclosing queue before making any instance-state decision."
fi
echo "No shutdown command was issued. AutoDL instance state remains a separate explicit action."
