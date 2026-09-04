#!/usr/bin/env bash
set -Eeuo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir=""
seeds_csv="20260825,20260826,20260827"
data_root="$root_dir/data/formal_block96_qwen25_7b"
output_root="$root_dir/outputs/qwen_block96"
status_path="/root/autodl-tmp/qwen-block96-completion-queue.status"
artifact_dir="/root/autodl-tmp/qwen-block96-completion-artifacts"
train_venv="${POSITION_BIAS_TRAIN_VENV:-/root/autodl-tmp/venvs/train}"
checkpoint_name="checkpoint-96"

usage() {
  echo "Usage: $0 --model LOCAL_DIR [--data-root DIR] [--output-root DIR] [--status FILE] [--artifact-dir DIR] [--train-venv DIR]"
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) model_dir="$2"; shift 2 ;;
    --data-root) data_root="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    --status) status_path="$2"; shift 2 ;;
    --artifact-dir) artifact_dir="$2"; shift 2 ;;
    --train-venv) train_venv="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -s "$model_dir/config.json" || ! -s "$model_dir/model_manifest.json" ]]; then
  echo "A complete manifested Qwen model is required" >&2
  exit 2
fi
if [[ ! -x "$train_venv/bin/python" ]]; then
  echo "Training environment is missing: $train_venv" >&2
  exit 2
fi
for target in "$data_root" "$output_root"; do
  case "$(realpath -m "$target")/" in
    "$(realpath "$root_dir")/"*) ;;
    *) echo "Data/output roots must be inside the project: $target" >&2; exit 2 ;;
  esac
done
python3 - "$data_root/completion.json" "$seeds_csv" <<'PY'
import hashlib, json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit("Strict block-complete Qwen data are missing")
payload = json.loads(path.read_text(encoding="utf-8"))
seeds = [int(value) for value in sys.argv[2].split(",")]
if (
    payload.get("schema_version") != "block-complete-sft-multiseed-v1"
    or payload.get("status") != "validated"
    or payload.get("seeds") != seeds
    or payload.get("rows_per_variant") != 96
    or payload.get("strict_realized_matching") is not True
):
    raise SystemExit("Strict block-complete Qwen data gate failed")
root = path.parent
for seed in seeds:
    record = payload.get("seed_completions", {}).get(str(seed), {})
    if not (
        record.get("schema_version") == "block-complete-sft-seed-v1"
        and record.get("status") == "validated"
        and int(record.get("seed", -1)) == seed
        and int(record.get("rows_per_variant", -1)) == 96
        and int(record.get("optimizer_steps", -1)) == 96
        and record.get("positions_per_variant")
        == {"p000": 24, "p025": 24, "p050": 24, "p100": 24}
        and record.get("strict_realized_matching") is True
        and record.get("complete_fact_blocks") is True
    ):
        raise SystemExit(f"Strict materialized Qwen seed record failed: seed={seed}")
    seed_root = root / f"seed_{seed}"
    lineage = {
        "manifest_sha256": seed_root / "manifest.json",
        "matched_audit_sha256": seed_root / "matched-audit.json",
        "selection_sha256": seed_root / "selection.json",
    }
    for field, artifact in lineage.items():
        if (
            not artifact.is_file()
            or record.get(field) != hashlib.sha256(artifact.read_bytes()).hexdigest()
        ):
            raise SystemExit(
                f"Strict materialized Qwen completion lineage hash mismatch: seed={seed} field={field}"
            )
PY

mkdir -p "$artifact_dir" "$(dirname "$status_path")" "$root_dir/results/training_subset_audits"
on_exit() {
  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    printf 'failed exit_code=%s finished_at=%s\n' "$rc" "$(date -u +%FT%TZ)" > "$status_path"
    echo "Qwen block-96 correction queue failed; audited completed stages remain resumable."
  fi
}
trap on_exit EXIT
cd "$root_dir"

printf 'running stage=historical_realized_subset_audit started_at=%s\n' "$(date -u +%FT%TZ)" > "$status_path"
"$train_venv/bin/python" scripts/audit_realized_training_subset.py \
  --data-root "$root_dir/data/formal_matched_qwen25_7b" \
  --training-output-root "$root_dir/outputs/formal_matched" \
  --seeds "$seeds_csv" \
  --steps 100 \
  --output "$root_dir/results/training_subset_audits/qwen_fixed100_realized_subset.json"
python3 - "$root_dir/results/training_subset_audits/qwen_fixed100_realized_subset.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
assessment = payload.get("claim_assessment", {})
strict_by_seed = assessment.get("strict_realized_fixed_step_matching_by_seed", {})
if payload.get("status") != "validated":
    raise SystemExit("Historical fixed-100 realized-subset audit was not validated")
if assessment.get("strict_realized_fixed_step_matching_all_seeds") is not False:
    raise SystemExit("Historical fixed-100 correction premise changed unexpectedly")
if not strict_by_seed or any(strict_by_seed.values()):
    raise SystemExit("Every historical Qwen fixed-100 seed must remain classified non-strict")
if assessment.get("recommended_action") != "retrain_from_materialized_block_complete_subsets":
    raise SystemExit("Historical fixed-100 audit no longer recommends the preregistered correction")
PY

printf 'running stage=block96_training seeds=%s started_at=%s\n' "$seeds_csv" "$(date -u +%FT%TZ)" > "$status_path"
bash scripts/run_autodl_fixed100_multiseed_training.sh \
  --model "$model_dir" \
  --data-root "$data_root" \
  --output-root "$output_root" \
  --seeds "$seeds_csv" \
  --fixed-steps 96 \
  --artifact-dir "$artifact_dir/training" \
  --status "$artifact_dir/training.status"

printf 'running stage=block96_realized_subset_audit started_at=%s\n' "$(date -u +%FT%TZ)" > "$status_path"
"$train_venv/bin/python" scripts/audit_realized_training_subset.py \
  --data-root "$data_root" \
  --training-output-root "$output_root" \
  --seeds "$seeds_csv" \
  --steps 96 \
  --output "$root_dir/results/training_subset_audits/qwen_block96_realized_subset.json"
python3 - "$root_dir/results/training_subset_audits/qwen_block96_realized_subset.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("status") != "validated" or not payload["claim_assessment"].get(
    "strict_realized_fixed_step_matching_all_seeds"
):
    raise SystemExit("Block-96 realized-subset audit did not pass strict matching")
PY

printf 'running stage=nolima seeds=%s started_at=%s\n' "$seeds_csv" "$(date -u +%FT%TZ)" > "$status_path"
bash scripts/run_autodl_nolima_multiseed.sh \
  --model "$model_dir" \
  --adapter-root "$output_root" \
  --checkpoint-name "$checkpoint_name" \
  --seeds "$seeds_csv" \
  --reuse-base-dir "$root_dir/results/nolima_hard_gate_seed1" \
  --output-dir "$root_dir/results/qwen_block96_nolima" \
  --artifact "$artifact_dir/nolima-multiseed.tar.gz" \
  --status "$artifact_dir/nolima-multiseed.status"

printf 'running stage=longbench seeds=%s started_at=%s\n' "$seeds_csv" "$(date -u +%FT%TZ)" > "$status_path"
bash scripts/run_autodl_longbench_multiseed.sh \
  --model "$model_dir" \
  --adapter-root "$output_root" \
  --checkpoint-name "$checkpoint_name" \
  --seeds "$seeds_csv" \
  --reuse-base-dir "$root_dir/results/longbench_transfer_seed1" \
  --output-dir "$root_dir/results/qwen_block96_longbench" \
  --artifact "$artifact_dir/longbench-multiseed.tar.gz" \
  --status "$artifact_dir/longbench-multiseed.status"

representative_seed="20260825"
representative_adapters="$output_root/seed_$representative_seed"
printf 'running stage=mmlu seed=%s started_at=%s\n' "$representative_seed" "$(date -u +%FT%TZ)" > "$status_path"
bash scripts/run_autodl_mmlu_regression.sh \
  --model "$model_dir" \
  --adapter-root "$representative_adapters" \
  --checkpoint-name "$checkpoint_name" \
  --run-label block96 \
  --output-dir "$root_dir/results/qwen_block96_mmlu" \
  --artifact "$artifact_dir/mmlu-regression.tar.gz" \
  --status "$artifact_dir/mmlu-regression.status"

printf 'running stage=ifeval seed=%s started_at=%s\n' "$representative_seed" "$(date -u +%FT%TZ)" > "$status_path"
bash scripts/run_autodl_ifeval_regression.sh \
  --model "$model_dir" \
  --adapter-root "$representative_adapters" \
  --checkpoint-name "$checkpoint_name" \
  --output-dir "$root_dir/results/qwen_block96_ifeval" \
  --artifact "$artifact_dir/ifeval-regression.tar.gz" \
  --status "$artifact_dir/ifeval-regression.status" \
  --seed "$representative_seed"

printf 'running stage=nolima_mechanisms seed=%s started_at=%s\n' "$representative_seed" "$(date -u +%FT%TZ)" > "$status_path"
bash scripts/run_autodl_nolima_mechanisms.sh \
  --model "$model_dir" \
  --free-result-dir "$root_dir/results/qwen_block96_nolima" \
  --free-seed "$representative_seed" \
  --adapter-root "$representative_adapters" \
  --checkpoint-name "$checkpoint_name" \
  --run-label block96 \
  --output-dir "$root_dir/results/qwen_block96_nolima_mechanisms" \
  --artifact "$artifact_dir/nolima-mechanisms.tar.gz" \
  --status "$artifact_dir/nolima-mechanisms.status"

printf 'running stage=rule seeds=%s started_at=%s\n' "$seeds_csv" "$(date -u +%FT%TZ)" > "$status_path"
bash scripts/run_autodl_formal_eval.sh \
  --model "$model_dir" \
  --data "$root_dir/data/pilot_qwen25_7b/raw/test.jsonl" \
  --manifest "$root_dir/data/pilot_qwen25_7b/manifest.json" \
  --adapter-root "$output_root" \
  --checkpoint-name "$checkpoint_name" \
  --seeds "$seeds_csv" \
  --reuse-base-dir "$root_dir/results/formal_s100_seed1_frozen" \
  --output-dir "$root_dir/results/qwen_block96_rule" \
  --artifact "$artifact_dir/rule-multiseed.tar.gz" \
  --status "$artifact_dir/rule-multiseed.status"

mkdir -p "$root_dir/results/qwen_block96_seed_level"
rule_args=()
nolima_args=()
longbench_args=()
IFS=',' read -r -a seeds <<< "$seeds_csv"
for seed in "${seeds[@]}"; do
  seed="${seed//[[:space:]]/}"
  rule_args+=(--analysis "Qwen2.5-7B:$seed:corrective:$root_dir/results/qwen_block96_rule/analysis_seed_$seed/factorial_analysis.json")
  nolima_args+=(--analysis "Qwen2.5-7B:$seed:corrective:$root_dir/results/qwen_block96_nolima/analysis_seed_$seed/factorial_analysis.json")
  longbench_args+=(--analysis "Qwen2.5-7B:$seed:corrective:$root_dir/results/qwen_block96_longbench/analysis_seed_$seed/transfer_analysis.json")
done
python3 scripts/aggregate_seed_level_results.py "${rule_args[@]}" \
  --output-dir "$root_dir/results/qwen_block96_seed_level/rule"
python3 scripts/aggregate_seed_level_results.py "${nolima_args[@]}" \
  --output-dir "$root_dir/results/qwen_block96_seed_level/nolima"
python3 scripts/aggregate_seed_level_results.py "${longbench_args[@]}" \
  --output-dir "$root_dir/results/qwen_block96_seed_level/longbench"

python3 - "$root_dir" "$seeds_csv" "$checkpoint_name" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
paths = [
    root / "data/formal_block96_qwen25_7b/completion.json",
    root / "results/training_subset_audits/qwen_fixed100_realized_subset.json",
    root / "results/training_subset_audits/qwen_block96_realized_subset.json",
    root / "outputs/qwen_block96/multiseed_completion.json",
    root / "results/qwen_block96_nolima/completion.json",
    root / "results/qwen_block96_longbench/completion.json",
    root / "results/qwen_block96_mmlu/completion.json",
    root / "results/qwen_block96_ifeval/completion.json",
    root / "results/qwen_block96_nolima_mechanisms/completion.json",
    root / "results/qwen_block96_rule/completion.json",
    root / "results/qwen_block96_seed_level/rule/seed_level_analysis.json",
    root / "results/qwen_block96_seed_level/nolima/seed_level_analysis.json",
    root / "results/qwen_block96_seed_level/longbench/seed_level_analysis.json",
]
for path in paths:
    if not path.is_file():
        raise SystemExit(f"Missing Qwen block-96 completion evidence: {path}")
def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
report = {
    "schema_version": "qwen-strict-block96-completion-v1",
    "status": "validated",
    "seeds": [int(value) for value in sys.argv[2].split(",")],
    "checkpoint_name": sys.argv[3],
    "optimizer_steps": 96,
    "rows_per_training_variant": 96,
    "strict_realized_matching": True,
    "exploratory_fixed100_retained_with_caveat": True,
    "artifacts": [str(path.relative_to(root)) for path in paths],
    "artifact_sha256": {str(path.relative_to(root)): sha256(path) for path in paths},
}
(root / "results/qwen_block96_completion.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

printf 'validated exit_code=0 seeds=%s finished_at=%s\n' "$seeds_csv" "$(date -u +%FT%TZ)" > "$status_path"
echo "Qwen strict block-96 training, OOD, transfer, regression, mechanism, and rule suites passed audit."
echo "Instance intentionally remains running; no power action was issued."
trap - EXIT
