#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/models/Qwen2.5-7B-Instruct}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/pilot_qwen25_7b}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/results/test_full}"
ARTIFACT_PATH="${ARTIFACT_PATH:-/root/autodl-tmp/position-bias-test-full.tar.gz}"
STATUS_PATH="${STATUS_PATH:-/root/autodl-tmp/position-bias-test-full.status}"
HOURLY_RATE_CNY="${HOURLY_RATE_CNY:-2.78}"
EXPECTED_PER_RUN=4200
RUN_NAMES=(base paired_evidence paired_answer independent_evidence independent_answer)
START_EPOCH="$(date +%s)"

export VLLM_USE_FLASHINFER_SAMPLER=0

mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/reproducibility"
rm -f "$STATUS_PATH"
LOG_FILE="$OUTPUT_DIR/logs/full-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG_FILE") 2>&1

finalize_on_exit() {
  local rc=$?
  set +e
  if [[ "$rc" -ne 0 ]]; then
    echo "failed exit_code=$rc finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS_PATH"
    tar -C "$ROOT_DIR" -czf "$ARTIFACT_PATH.partial" results/test_full
    sha256sum "$ARTIFACT_PATH.partial" > "$ARTIFACT_PATH.partial.sha256"
    echo "Full evaluation stopped; resumable rows and diagnostics were preserved."
  fi
}
trap finalize_on_exit EXIT

cd "$ROOT_DIR"
python3 scripts/preflight_autodl.py \
  --mode eval \
  --model "$MODEL_DIR" \
  --data "$DATA_DIR/raw/test.jsonl" \
  --manifest "$DATA_DIR/manifest.json" \
  --require-model-manifest \
  --output "$OUTPUT_DIR"

python3 -m pip freeze > "$OUTPUT_DIR/reproducibility/pip-freeze.txt"
nvidia-smi -q > "$OUTPUT_DIR/reproducibility/nvidia-smi-q.txt"
uname -a > "$OUTPUT_DIR/reproducibility/uname.txt"
cp scripts/evaluate_suite_vllm.py scripts/evaluate_vllm.py \
  scripts/aggregate_results.py scripts/finalize_eval_full.py \
  "$OUTPUT_DIR/reproducibility/"
sha256sum \
  "$DATA_DIR/raw/test.jsonl" "$MODEL_DIR/config.json" \
  outputs/paired_evidence/final_adapter/adapter_config.json \
  outputs/paired_answer/final_adapter/adapter_config.json \
  outputs/independent_evidence/final_adapter/adapter_config.json \
  outputs/independent_answer/final_adapter/adapter_config.json \
  > "$OUTPUT_DIR/reproducibility/input-sha256.txt"

python3 scripts/evaluate_suite_vllm.py \
  --model "$MODEL_DIR" \
  --data "$DATA_DIR/raw/test.jsonl" \
  --output-dir "$OUTPUT_DIR" \
  --run base \
  --run paired_evidence=outputs/paired_evidence/final_adapter \
  --run paired_answer=outputs/paired_answer/final_adapter \
  --run independent_evidence=outputs/independent_evidence/final_adapter \
  --run independent_answer=outputs/independent_answer/final_adapter \
  --batch-size 4 \
  --max-model-len 32768 \
  --max-num-seqs 4 \
  --max-new-tokens 176 \
  --gpu-memory-utilization 0.88 \
  --max-lora-rank 16 \
  --seed 20260825

python3 scripts/aggregate_results.py \
  "$OUTPUT_DIR"/*.jsonl \
  --output "$OUTPUT_DIR/summary.json"

END_EPOCH="$(date +%s)"
WALL_SECONDS=$((END_EPOCH - START_EPOCH))
FINALIZE_ARGS=()
for run_name in "${RUN_NAMES[@]}"; do
  FINALIZE_ARGS+=(--run "$run_name")
done
python3 scripts/finalize_eval_full.py \
  --results-dir "$OUTPUT_DIR" \
  "${FINALIZE_ARGS[@]}" \
  --expected-per-run "$EXPECTED_PER_RUN" \
  --expected-cells 84 \
  --expected-per-cell 50 \
  --wall-seconds "$WALL_SECONDS" \
  --hourly-rate-cny "$HOURLY_RATE_CNY" \
  --output "$OUTPUT_DIR/validation-report.json"

tar -C "$ROOT_DIR" -czf "$ARTIFACT_PATH" results/test_full
sha256sum "$ARTIFACT_PATH" > "$ARTIFACT_PATH.sha256"
sha256sum -c "$ARTIFACT_PATH.sha256"
echo "validated exit_code=0 artifact=$ARTIFACT_PATH finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS_PATH"
touch "$OUTPUT_DIR/RESULTS_READY_FOR_AGENT_REVIEW"
echo "Full evaluation completed, validated, and packaged."
echo "Instance intentionally left running for agent validation and explicit shutdown."
trap - EXIT
