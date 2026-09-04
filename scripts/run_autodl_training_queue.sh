#!/usr/bin/env bash
set -u -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR=""
RESULT_BUNDLE="/root/autodl-tmp/position-bias-training-results.tar.gz"
STATUS_FILE="/root/autodl-tmp/train-all.exit"
VARIANTS=(paired_evidence paired_answer independent_evidence independent_answer)
DATA_DIR="$ROOT_DIR/data/pilot_qwen25_7b"
OUTPUT_ROOT="$ROOT_DIR/outputs"
TRAIN_SEED=20260825

usage() {
  echo "Usage: $0 --model LOCAL_DIR [--data-dir DIR] [--output-root DIR] [--variants CSV] [--seed INT] [--result-bundle FILE] [--status-file FILE]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_DIR="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --variants) IFS=',' read -r -a VARIANTS <<< "$2"; shift 2 ;;
    --seed) TRAIN_SEED="$2"; shift 2 ;;
    --result-bundle) RESULT_BUNDLE="$2"; shift 2 ;;
    --status-file) STATUS_FILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$MODEL_DIR" || ! -d "$MODEL_DIR" ]]; then
  echo "--model must point to a complete local model directory" >&2
  exit 2
fi
if [[ ! -s "$DATA_DIR/manifest.json" || "${#VARIANTS[@]}" -eq 0 ]]; then
  echo "--data-dir must contain manifest.json and --variants must be non-empty" >&2
  exit 2
fi

write_status() {
  local code="$1"
  local temporary="${STATUS_FILE}.tmp-$$"
  mkdir -p "$(dirname "$STATUS_FILE")"
  printf '%s\n' "$code" > "$temporary"
  mv "$temporary" "$STATUS_FILE"
}

rm -f "$STATUS_FILE"
for variant in "${VARIANTS[@]}"; do
  printf 'BEGIN %s %s\n' "$variant" "$(date -u +%FT%TZ)"
  if bash "$ROOT_DIR/scripts/run_sft_variant.sh" \
    --model "$MODEL_DIR" \
    --variant "$variant" \
    --data-dir "$DATA_DIR" \
    --output-root "$OUTPUT_ROOT" \
    --seed "$TRAIN_SEED"; then
    :
  else
    code=$?
    printf 'FAILED %s rc=%s %s\n' "$variant" "$code" "$(date -u +%FT%TZ)" >&2
    write_status "$code"
    sync
    exit "$code"
  fi
  printf 'DONE %s %s\n' "$variant" "$(date -u +%FT%TZ)"
done

variant_csv="$(IFS=,; echo "${VARIANTS[*]}")"
if ! OUTPUT_ROOT="$OUTPUT_ROOT" \
  DATA_MANIFEST="$DATA_DIR/manifest.json" \
  VARIANT_CSV="$variant_csv" \
  bash "$ROOT_DIR/scripts/finalize_autodl_training.sh" "$RESULT_BUNDLE" "$MODEL_DIR"; then
  echo "Final result validation or packaging failed" >&2
  write_status 1
  sync
  exit 1
fi

checksum_file="${RESULT_BUNDLE}.sha256"
if ! (
  cd "$(dirname "$RESULT_BUNDLE")"
  sha256sum -c "$(basename "$checksum_file")"
); then
  echo "Final result bundle checksum verification failed" >&2
  write_status 1
  sync
  exit 1
fi

touch "$ROOT_DIR/RESULTS_READY_FOR_AGENT_REVIEW"
write_status 0
sync
printf 'RESULTS READY FOR AGENT REVIEW %s\n' "$(date -u +%FT%TZ)"
