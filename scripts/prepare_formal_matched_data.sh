#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
formal_root="${FORMAL_ROOT:-$repo_root/data/formal_matched_qwen25_7b}"
tokenizer_name="${TOKENIZER_NAME:-Qwen/Qwen2.5-7B-Instruct}"
tokenizer_revision="${TOKENIZER_REVISION:-a09a35458c702b33eeacc393d103063234e8bc28}"
data_seeds_csv="${DATA_SEEDS:-20260825,20260826,20260827}"
facts_per_condition="${FACTS_PER_CONDITION:-128}"
pretokenize="${PRETOKENIZE:-1}"
force="${FORCE:-0}"
source_test="${SOURCE_TEST:-$repo_root/data/pilot_qwen25_7b/raw/test.jsonl}"
variants=(
  independent_answer independent_evidence_id independent_evidence
  paired_answer paired_evidence_id paired_evidence
)

if [[ "$pretokenize" != "0" && "$pretokenize" != "1" ]]; then
  echo "PRETOKENIZE must be 0 or 1" >&2
  exit 2
fi
if [[ "$force" != "0" && "$force" != "1" ]]; then
  echo "FORCE must be 0 or 1" >&2
  exit 2
fi
if [[ ! -s "$source_test" ]]; then
  echo "Missing source evaluation data: $source_test" >&2
  exit 2
fi

common_tokenizer_args=(
  --tokenizer "$tokenizer_name"
  --tokenizer-revision "$tokenizer_revision"
  --local-files-only
)
overwrite_args=()
if [[ "$force" == "1" ]]; then
  overwrite_args+=(--overwrite)
fi

IFS=',' read -r -a data_seeds <<< "$data_seeds_csv"
if [[ "${#data_seeds[@]}" -eq 0 ]]; then
  echo "DATA_SEEDS must be a non-empty comma-separated list" >&2
  exit 2
fi

mkdir -p "$formal_root/eval"
for seed in "${data_seeds[@]}"; do
  seed="${seed//[[:space:]]/}"
  if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "Invalid integer seed: $seed" >&2
    exit 2
  fi
  seed_root="$formal_root/seed_$seed"
  bank="$seed_root/raw/train_bank.jsonl"
  mkdir -p "$seed_root/raw" "$seed_root/sft" "$seed_root/tokenized"

  if [[ ! -s "$bank" || "$force" == "1" ]]; then
    python3 "$repo_root/scripts/generate_matched_training_bank.py" \
      --output "$bank" \
      --split train \
      --facts-per-condition "$facts_per_condition" \
      --replicas-per-fact 4 \
      --tasks kv,two_hop \
      --fillers neutral \
      --lengths 8K \
      --positions 0,25,50,100 \
      --seed "$seed" \
      --words-per-document 48 \
      "${common_tokenizer_args[@]}" \
      "${overwrite_args[@]}"
  fi

  if [[ ! -s "$seed_root/sft/matched-design.json" || "$force" == "1" ]]; then
    python3 "$repo_root/scripts/prepare_matched_sft_variants.py" \
      --input "$bank" \
      --output-dir "$seed_root/sft" \
      --seed "$seed" \
      --max-token-budget-gap 0.002 \
      "${overwrite_args[@]}"
  fi

  python3 "$repo_root/scripts/validate_dataset.py" "$bank" "$seed_root"/sft/*.jsonl
  python3 "$repo_root/scripts/audit_matched_training_design.py" \
    "$seed_root"/sft/*.jsonl \
    --max-token-budget-gap 0.002 \
    --output "$seed_root/sft/matched-audit.json"

  if [[ "$pretokenize" == "1" ]]; then
    for variant in "${variants[@]}"; do
      tokenized="$seed_root/tokenized/$variant"
      if [[ ! -s "$tokenized/pretokenization.json" || "$force" == "1" ]]; then
        tokenize_overwrite=()
        if [[ "$force" == "1" && -e "$tokenized" ]]; then
          tokenize_overwrite+=(--overwrite)
        fi
        python3 "$repo_root/scripts/pretokenize_sft.py" \
          "$seed_root/sft/$variant.jsonl" \
          "$tokenized" \
          --tokenizer "$tokenizer_name" \
          --tokenizer-revision "$tokenizer_revision" \
          --max-length 8320 \
          --local-files-only \
          "${tokenize_overwrite[@]}"
      fi
    done
  fi

  python3 "$repo_root/scripts/build_data_manifest.py" \
    --root "$seed_root" \
    --output "$seed_root/manifest.json"
done

diagnostic="$formal_root/eval/test_diagnostics.jsonl"
if [[ ! -s "$diagnostic" || "$force" == "1" ]]; then
  python3 "$repo_root/scripts/prepare_diagnostic_eval.py" \
    --input "$source_test" \
    --output "$diagnostic" \
    --modes locate_only,oracle_long,oracle_short \
    "${common_tokenizer_args[@]}" \
    "${overwrite_args[@]}"
fi
python3 "$repo_root/scripts/validate_dataset.py" "$diagnostic"
python3 "$repo_root/scripts/build_data_manifest.py" \
  --root "$formal_root" \
  --output "$formal_root/manifest.json"

echo "Prepared formal matched matrix under $formal_root"
echo "No external teacher API data was used."
