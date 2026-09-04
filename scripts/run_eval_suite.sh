#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR=""
DATA_DIR="$ROOT_DIR/data/pilot_qwen25_7b"
OUTPUT_DIR="$ROOT_DIR/results/pilot"
CANARY=0
RUNS=()

usage() {
  echo "Usage: $0 --model LOCAL_DIR --run NAME[=ADAPTER_DIR] [--run ...] [--canary]"
  echo "Example: $0 --model /data/Qwen2.5-7B-Instruct --run base --run paired=/data/adapter"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_DIR="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --run) RUNS+=("$2"); shift 2 ;;
    --canary) CANARY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ -z "$MODEL_DIR" || "${#RUNS[@]}" -eq 0 ]]; then
  usage >&2
  exit 2
fi

TEST_DATA="$DATA_DIR/raw/test.jsonl"
MANIFEST="$DATA_DIR/manifest.json"
mkdir -p "$OUTPUT_DIR/logs"
python3 "$ROOT_DIR/scripts/preflight_autodl.py" \
  --mode eval \
  --model "$MODEL_DIR" \
  --data "$TEST_DATA" \
  --manifest "$MANIFEST" \
  --require-model-manifest \
  --output "$OUTPUT_DIR"

EVAL_ARGS=(
  --model "$MODEL_DIR"
  --data "$TEST_DATA"
  --output-dir "$OUTPUT_DIR"
)
for run in "${RUNS[@]}"; do
  EVAL_ARGS+=(--run "$run")
done
if [[ "$CANARY" -eq 1 ]]; then
  CANARY_PER_RUN=$((100 / ${#RUNS[@]}))
  if [[ "$CANARY_PER_RUN" -lt 1 ]]; then
    CANARY_PER_RUN=1
  fi
  EVAL_ARGS+=(--max-samples "$CANARY_PER_RUN")
fi

LOG_FILE="$OUTPUT_DIR/logs/$(date -u +%Y%m%dT%H%M%SZ).log"
python3 "$ROOT_DIR/scripts/evaluate_suite_vllm.py" "${EVAL_ARGS[@]}" 2>&1 | tee "$LOG_FILE"

if [[ "$CANARY" -eq 1 ]]; then
  touch "$OUTPUT_DIR/READY_TO_STOP_AUTODL_AFTER_EVAL_CANARY"
  echo "Evaluation canary completed on $CANARY_PER_RUN spread samples per run."
else
  touch "$OUTPUT_DIR/READY_TO_STOP_AUTODL_AFTER_EVALUATION"
  echo "Evaluation completed. Confirm the JSONL files are persisted."
fi
echo "Stop the instance in the AutoDL console; do not rely on shutdown -h to stop billing."
