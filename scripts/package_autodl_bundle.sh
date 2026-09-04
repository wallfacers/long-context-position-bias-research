#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="$ROOT_DIR/autodl-position-bias-bundle.tar.gz"
VARIANT="all"
INCLUDE_TEST=0

usage() {
  echo "Usage: $0 [--output FILE.tar.gz] [--variant NAME|all] [--include-test]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) OUTPUT="$2"; shift 2 ;;
    --variant) VARIANT="$2"; shift 2 ;;
    --include-test) INCLUDE_TEST=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$VARIANT" in
  all|independent_answer|paired_answer|independent_evidence|paired_evidence) ;;
  *) echo "Invalid variant: $VARIANT" >&2; exit 2 ;;
esac

OUTPUT="$(realpath -m "$OUTPUT")"
mkdir -p "$(dirname "$OUTPUT")"
FILES=(
  pyproject.toml
  requirements-train.txt
  requirements-eval.txt
  configs
  docs
  scripts
  src
  data/pilot_qwen25_7b/manifest.json
)

if [[ "$VARIANT" == "all" ]]; then
  FILES+=(data/pilot_qwen25_7b/sft)
  if [[ -d "$ROOT_DIR/data/pilot_qwen25_7b/tokenized" ]]; then
    FILES+=(data/pilot_qwen25_7b/tokenized)
  fi
else
  FILES+=("data/pilot_qwen25_7b/sft/$VARIANT.jsonl")
  if [[ -d "$ROOT_DIR/data/pilot_qwen25_7b/tokenized/$VARIANT" ]]; then
    FILES+=("data/pilot_qwen25_7b/tokenized/$VARIANT")
  fi
fi
if [[ "$INCLUDE_TEST" -eq 1 ]]; then
  FILES+=(data/pilot_qwen25_7b/raw/dev.jsonl data/pilot_qwen25_7b/raw/test.jsonl)
fi

for path in "${FILES[@]}"; do
  if [[ ! -e "$ROOT_DIR/$path" ]]; then
    echo "Missing bundle input: $ROOT_DIR/$path" >&2
    exit 1
  fi
done

tar --exclude='__pycache__' --exclude='*.pyc' -C "$ROOT_DIR" -czf "$OUTPUT" "${FILES[@]}"
(
  cd "$(dirname "$OUTPUT")"
  sha256sum "$(basename "$OUTPUT")" > "$(basename "$OUTPUT").sha256"
)
ls -lh "$OUTPUT" "$OUTPUT.sha256"
echo "Verify after upload with: sha256sum -c $(basename "$OUTPUT").sha256"
