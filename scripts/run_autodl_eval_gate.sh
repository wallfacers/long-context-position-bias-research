#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/models/Qwen2.5-7B-Instruct}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/pilot_qwen25_7b}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/results/dev_gate}"
ARTIFACT_PATH="${ARTIFACT_PATH:-/root/autodl-tmp/position-bias-dev-gate.tar.gz}"
STATUS_PATH="${STATUS_PATH:-/root/autodl-tmp/position-bias-dev-gate.status}"
HOURLY_RATE_CNY="${HOURLY_RATE_CNY:-2.78}"
EXPECTED_PER_RUN=20
RUN_NAMES=(base paired_evidence paired_answer independent_evidence independent_answer)
START_EPOCH="$(date +%s)"

# RTX 5090 D (SM 12.0) is supported by the CUDA 13 vLLM stack, but the
# bundled FlashInfer sampler can mis-detect its architecture.  Keep the vLLM
# engine and attention backend while using vLLM's native top-k/top-p sampler.
export VLLM_USE_FLASHINFER_SAMPLER=0

mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/reproducibility"
rm -f "$STATUS_PATH"
LOG_FILE="$OUTPUT_DIR/logs/gate-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG_FILE") 2>&1

finalize_on_exit() {
  local rc=$?
  set +e
  if [[ "$rc" -ne 0 ]]; then
    echo "failed exit_code=$rc finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS_PATH"
    tar -C "$ROOT_DIR" -czf "$ARTIFACT_PATH.failed" results/dev_gate
    sha256sum "$ARTIFACT_PATH.failed" > "$ARTIFACT_PATH.failed.sha256"
    echo "Evaluation gate failed; diagnostics were preserved."
  fi
}
trap finalize_on_exit EXIT

cd "$ROOT_DIR"
python3 scripts/preflight_autodl.py \
  --mode eval \
  --model "$MODEL_DIR" \
  --data "$DATA_DIR/raw/dev.jsonl" \
  --manifest "$DATA_DIR/manifest.json" \
  --require-model-manifest \
  --output "$OUTPUT_DIR"

python3 -m pip freeze > "$OUTPUT_DIR/reproducibility/pip-freeze.txt"
nvidia-smi -q > "$OUTPUT_DIR/reproducibility/nvidia-smi-q.txt"
uname -a > "$OUTPUT_DIR/reproducibility/uname.txt"
cp scripts/evaluate_suite_vllm.py scripts/evaluate_vllm.py \
  scripts/aggregate_results.py scripts/finalize_eval_gate.py \
  "$OUTPUT_DIR/reproducibility/"
sha256sum \
  "$DATA_DIR/raw/dev.jsonl" "$DATA_DIR/raw/test.jsonl" \
  "$MODEL_DIR/config.json" \
  outputs/paired_evidence/final_adapter/adapter_config.json \
  outputs/paired_answer/final_adapter/adapter_config.json \
  outputs/independent_evidence/final_adapter/adapter_config.json \
  outputs/independent_answer/final_adapter/adapter_config.json \
  > "$OUTPUT_DIR/reproducibility/input-sha256.txt"

python3 scripts/evaluate_suite_vllm.py \
  --model "$MODEL_DIR" \
  --data "$DATA_DIR/raw/dev.jsonl" \
  --output-dir "$OUTPUT_DIR" \
  --run base \
  --run paired_evidence=outputs/paired_evidence/final_adapter \
  --run paired_answer=outputs/paired_answer/final_adapter \
  --run independent_evidence=outputs/independent_evidence/final_adapter \
  --run independent_answer=outputs/independent_answer/final_adapter \
  --max-samples "$EXPECTED_PER_RUN" \
  --batch-size 4 \
  --max-model-len 9216 \
  --max-num-seqs 4 \
  --max-new-tokens 128 \
  --gpu-memory-utilization 0.88 \
  --max-lora-rank 16 \
  --enforce-eager \
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
python3 scripts/finalize_eval_gate.py \
  --results-dir "$OUTPUT_DIR" \
  --dev-data "$DATA_DIR/raw/dev.jsonl" \
  --test-data "$DATA_DIR/raw/test.jsonl" \
  "${FINALIZE_ARGS[@]}" \
  --expected-per-run "$EXPECTED_PER_RUN" \
  --wall-seconds "$WALL_SECONDS" \
  --hourly-rate-cny "$HOURLY_RATE_CNY" \
  --output "$OUTPUT_DIR/gate-report.json"

tar -C "$ROOT_DIR" -czf "$ARTIFACT_PATH" results/dev_gate
sha256sum "$ARTIFACT_PATH" > "$ARTIFACT_PATH.sha256"
sha256sum -c "$ARTIFACT_PATH.sha256"
echo "validated exit_code=0 artifact=$ARTIFACT_PATH finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS_PATH"
touch "$OUTPUT_DIR/READY_TO_STOP_AUTODL_AFTER_EVAL_GATE"
echo "Evaluation gate completed, validated, and packaged."
echo "Instance intentionally left running for agent validation and explicit shutdown."
trap - EXIT
