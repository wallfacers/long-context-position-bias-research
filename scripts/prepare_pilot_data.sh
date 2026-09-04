#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
tokenizer_name=${TOKENIZER_NAME:-Qwen/Qwen2.5-7B-Instruct}
tokenizer_revision=${TOKENIZER_REVISION:-a09a35458c702b33eeacc393d103063234e8bc28}
data_root=${DATA_ROOT:-"${repo_root}/data/pilot_qwen25_7b"}
seed=${DATA_SEED:-20260825}

common_args=(
  --tokenizer "${tokenizer_name}"
  --seed "${seed}"
  --words-per-document 48
  --overwrite
)

common_args+=(--tokenizer-revision "${tokenizer_revision}")
if [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  common_args+=(--local-files-only)
fi

mkdir -p "${data_root}/raw" "${data_root}/sft"

python3 "${repo_root}/scripts/generate_synthetic_data.py" \
  --output "${data_root}/raw/train.jsonl" \
  --split train \
  --groups-per-condition 625 \
  --tasks kv,two_hop \
  --fillers neutral \
  --lengths 8K \
  --positions 0,25,50,100 \
  "${common_args[@]}"

python3 "${repo_root}/scripts/prepare_sft_variants.py" \
  --input "${data_root}/raw/train.jsonl" \
  --output-dir "${data_root}/sft" \
  --paired-groups 250 \
  --independent-groups 1000 \
  --max-token-budget-gap 0.02 \
  --seed "${seed}" \
  --overwrite

python3 "${repo_root}/scripts/generate_synthetic_data.py" \
  --output "${data_root}/raw/dev.jsonl" \
  --split dev \
  --groups-per-condition 50 \
  --tasks kv,two_hop \
  --fillers neutral \
  --lengths 8K \
  --positions 0,10,25,50,75,90,100 \
  "${common_args[@]}"

python3 "${repo_root}/scripts/generate_synthetic_data.py" \
  --output "${data_root}/raw/test.jsonl" \
  --split test \
  --groups-per-condition 50 \
  --tasks kv,two_hop \
  --fillers neutral,same_format,answer_bearing \
  --lengths 8K,32K \
  --positions 0,10,25,50,75,90,100 \
  "${common_args[@]}"

python3 "${repo_root}/scripts/validate_dataset.py" \
  "${data_root}/raw/train.jsonl" \
  "${data_root}/raw/dev.jsonl" \
  "${data_root}/raw/test.jsonl"

python3 "${repo_root}/scripts/validate_dataset.py" "${data_root}"/sft/*.jsonl

python3 "${repo_root}/scripts/build_data_manifest.py" \
  --root "${data_root}" \
  --output "${data_root}/manifest.json"

echo "Prepared and validated pilot data under ${data_root}"
