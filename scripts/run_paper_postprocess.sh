#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/results/test_full}"
ARTIFACT_PATH="${ARTIFACT_PATH:-/root/autodl-tmp/position-bias-test-full.tar.gz}"
FULL_STATUS_PATH="${FULL_STATUS_PATH:-/root/autodl-tmp/position-bias-test-full.status}"
POSTPROCESS_STATUS_PATH="${POSTPROCESS_STATUS_PATH:-/root/autodl-tmp/position-bias-paper-postprocess.status}"
BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-2000}"
SEED="${SEED:-20260825}"
TEMP_ARTIFACT="${ARTIFACT_PATH}.postprocess-$$.tmp"

finalize_on_exit() {
  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    echo "failed exit_code=$rc finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      > "$POSTPROCESS_STATUS_PATH"
    rm -f "$TEMP_ARTIFACT"
    echo "Paper post-processing failed; the validated raw evaluation remains intact."
  fi
}
trap finalize_on_exit EXIT

if [[ ! -f "$FULL_STATUS_PATH" ]] || ! grep -q '^validated exit_code=0 ' "$FULL_STATUS_PATH"; then
  echo "Full evaluation has not passed its validation gate: $FULL_STATUS_PATH" >&2
  exit 1
fi
if [[ ! -f "$RESULTS_DIR/RESULTS_READY_FOR_AGENT_REVIEW" ]]; then
  echo "Missing agent-review readiness marker in $RESULTS_DIR" >&2
  exit 1
fi

rm -f "$POSTPROCESS_STATUS_PATH"
mkdir -p "$RESULTS_DIR/analysis" "$RESULTS_DIR/figures" "$RESULTS_DIR/reproducibility"
cp \
  "$ROOT_DIR/scripts/analyze_position_ablation.py" \
  "$ROOT_DIR/scripts/audit_pilot_completion.py" \
  "$ROOT_DIR/scripts/compare_generation_caps.py" \
  "$ROOT_DIR/scripts/build_pilot_cost_ledger.py" \
  "$ROOT_DIR/scripts/finalize_eval_full.py" \
  "$ROOT_DIR/scripts/plot_position_ablation.py" \
  "$ROOT_DIR/scripts/render_pilot_report.py" \
  "$ROOT_DIR/scripts/run_paper_postprocess.sh" \
  "$ROOT_DIR/scripts/watch_autodl_eval_postprocess.sh" \
  "$ROOT_DIR/docs/pilot-qwen25-7b.md" \
  "$RESULTS_DIR/reproducibility/"

cd "$ROOT_DIR"
python3 scripts/compare_generation_caps.py \
  --lower-dir "$ROOT_DIR/results/test_full_cap128_diagnostic" \
  --higher-dir "$RESULTS_DIR" \
  --run base \
  --sample-limit 84 \
  --expected-cells 84 \
  --output "$RESULTS_DIR/reproducibility/generation-cap-diagnostic.json"

python3 scripts/analyze_position_ablation.py \
  --results-dir "$RESULTS_DIR" \
  --output-dir "$RESULTS_DIR/analysis" \
  --bootstrap-replicates "$BOOTSTRAP_REPLICATES" \
  --seed "$SEED"

python3 scripts/plot_position_ablation.py \
  --analysis "$RESULTS_DIR/analysis/ablation_analysis.json" \
  --output-dir "$RESULTS_DIR/figures"

python3 scripts/build_pilot_cost_ledger.py \
  --project-root "$ROOT_DIR" \
  --external-root /root/autodl-tmp \
  --hourly-rate-cny 2.78 \
  --output "$RESULTS_DIR/reproducibility/cost-ledger.json"

python3 scripts/render_pilot_report.py \
  --analysis "$RESULTS_DIR/analysis/ablation_analysis.json" \
  --validation "$RESULTS_DIR/validation-report.json" \
  --cost-ledger "$RESULTS_DIR/reproducibility/cost-ledger.json" \
  --output "$RESULTS_DIR/paper-pilot-report.md"

python3 -m unittest discover -s tests -v

python3 scripts/audit_pilot_completion.py \
  --project-root "$ROOT_DIR" \
  --results-dir "$RESULTS_DIR" \
  --output "$RESULTS_DIR/completion-audit.json"

MANIFEST_PATH="$RESULTS_DIR/reproducibility/artifact-manifest.sha256"
find results/test_full -type f \
  ! -path '*/reproducibility/artifact-manifest.sha256' \
  -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$MANIFEST_PATH"
sha256sum -c "$MANIFEST_PATH"

tar -C "$ROOT_DIR" -czf "$TEMP_ARTIFACT" results/test_full
mv "$TEMP_ARTIFACT" "$ARTIFACT_PATH"
sha256sum "$ARTIFACT_PATH" > "$ARTIFACT_PATH.sha256"
sha256sum -c "$ARTIFACT_PATH.sha256"
echo "validated exit_code=0 artifact=$ARTIFACT_PATH finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$POSTPROCESS_STATUS_PATH"
touch "$RESULTS_DIR/PAPER_ARTIFACT_READY_FOR_AGENT_REVIEW"
echo "Paper analysis, figures, report, manifest, tests, and package validated."
echo "Instance intentionally left running for agent review and explicit shutdown."
trap - EXIT
