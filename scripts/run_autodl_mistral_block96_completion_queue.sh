#!/usr/bin/env bash
set -Eeuo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir=""
formal_source_root="$root_dir/data/formal_matched_mistral7b_v03"
formal_root="$root_dir/data/formal_block96_mistral7b_v03"
nolima_root="$root_dir/data/ood_nolima_mistral7b_v03"
output_root="$root_dir/outputs/mistral_block96"
seeds_csv="20260825,20260826,20260827"
status_path="/root/autodl-tmp/mistral-block96-completion-queue.status"
artifact_dir="/root/autodl-tmp/mistral-block96-completion-artifacts"
train_venv="${POSITION_BIAS_TRAIN_VENV:-/root/autodl-tmp/venvs/train}"
checkpoint_name="checkpoint-96"

usage() {
  echo "Usage: $0 --model LOCAL_DIR [--formal-source-root DIR] [--formal-root DIR] [--nolima-root DIR] [--output-root DIR] [--status FILE] [--artifact-dir DIR] [--train-venv DIR]"
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) model_dir="$2"; shift 2 ;;
    --formal-source-root) formal_source_root="$2"; shift 2 ;;
    --formal-root) formal_root="$2"; shift 2 ;;
    --nolima-root) nolima_root="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    --status) status_path="$2"; shift 2 ;;
    --artifact-dir) artifact_dir="$2"; shift 2 ;;
    --train-venv) train_venv="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -s "$model_dir/config.json" || ! -s "$model_dir/model_manifest.json" ]]; then
  echo "A complete manifested Mistral snapshot is required" >&2
  exit 2
fi
if [[ ! -x "$train_venv/bin/python" || ! -s "$formal_source_root/completion.json" ]]; then
  echo "Training environment and validated Mistral source data are required" >&2
  exit 2
fi
if [[ ! -s "$root_dir/results/qwen_block96_completion.json" ]]; then
  echo "Strict Qwen block-96 queue must finish first" >&2
  exit 2
fi
python3 - "$formal_source_root/completion.json" "$model_dir/model_manifest.json" "$root_dir/results/qwen_block96_completion.json" "$seeds_csv" <<'PY'
import json, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
model = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
qwen = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
seeds = [int(value) for value in sys.argv[4].split(",")]
if data.get("status") != "validated" or data.get("training_seeds") != seeds:
    raise SystemExit("Mistral source data seed gate failed")
if data.get("revision") != model.get("revision"):
    raise SystemExit("Mistral source data/model revisions differ")
if qwen.get("schema_version") != "qwen-strict-block96-completion-v1" or qwen.get("status") != "validated":
    raise SystemExit("Strict Qwen completion gate failed")
if seeds != [20260825, 20260826, 20260827]:
    raise SystemExit("Strict Mistral seeds changed")
PY

mkdir -p "$artifact_dir" "$(dirname "$status_path")" "$root_dir/results/training_subset_audits"
on_exit() {
  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    printf 'failed exit_code=%s finished_at=%s\n' "$rc" "$(date -u +%FT%TZ)" > "$status_path"
    echo "Mistral block-96 queue failed; audited completed stages remain resumable."
  fi
}
trap on_exit EXIT
cd "$root_dir"

printf 'running stage=materialize_block96_data started_at=%s\n' "$(date -u +%FT%TZ)" > "$status_path"
"$train_venv/bin/python" scripts/materialize_block_complete_sft.py \
  --source-root "$formal_source_root" \
  --output-root "$formal_root" \
  --seeds "$seeds_csv" \
  --facts-per-stratum 3 \
  --expected-rows 96

printf 'running stage=block96_training seeds=%s started_at=%s\n' "$seeds_csv" "$(date -u +%FT%TZ)" > "$status_path"
bash scripts/run_autodl_fixed100_multiseed_training.sh \
  --model "$model_dir" \
  --data-root "$formal_root" \
  --output-root "$output_root" \
  --seeds "$seeds_csv" \
  --fixed-steps 96 \
  --artifact-dir "$artifact_dir/training" \
  --status "$artifact_dir/training.status"

printf 'running stage=realized_subset_audit started_at=%s\n' "$(date -u +%FT%TZ)" > "$status_path"
"$train_venv/bin/python" scripts/audit_realized_training_subset.py \
  --data-root "$formal_root" \
  --training-output-root "$output_root" \
  --seeds "$seeds_csv" \
  --steps 96 \
  --output "$root_dir/results/training_subset_audits/mistral_block96_realized_subset.json"
python3 - "$root_dir/results/training_subset_audits/mistral_block96_realized_subset.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("status") != "validated" or not payload["claim_assessment"].get(
    "strict_realized_fixed_step_matching_all_seeds"
):
    raise SystemExit("Mistral block-96 realized-subset gate failed")
PY

printf 'running stage=nolima seeds=%s started_at=%s\n' "$seeds_csv" "$(date -u +%FT%TZ)" > "$status_path"
bash scripts/run_autodl_nolima_multiseed.sh \
  --model "$model_dir" \
  --data "$nolima_root/hard_gate.jsonl" \
  --manifest "$nolima_root/hard_gate.manifest.json" \
  --adapter-root "$output_root" \
  --checkpoint-name "$checkpoint_name" \
  --seeds "$seeds_csv" \
  --output-dir "$root_dir/results/mistral_block96_nolima" \
  --artifact "$artifact_dir/nolima-multiseed.tar.gz" \
  --status "$artifact_dir/nolima-multiseed.status"

printf 'running stage=longbench seeds=%s started_at=%s\n' "$seeds_csv" "$(date -u +%FT%TZ)" > "$status_path"
bash scripts/run_autodl_longbench_multiseed.sh \
  --model "$model_dir" \
  --adapter-root "$output_root" \
  --checkpoint-name "$checkpoint_name" \
  --seeds "$seeds_csv" \
  --output-dir "$root_dir/results/mistral_block96_longbench" \
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
  --output-dir "$root_dir/results/mistral_block96_mmlu" \
  --artifact "$artifact_dir/mmlu-regression.tar.gz" \
  --status "$artifact_dir/mmlu-regression.status"

printf 'running stage=ifeval seed=%s started_at=%s\n' "$representative_seed" "$(date -u +%FT%TZ)" > "$status_path"
bash scripts/run_autodl_ifeval_regression.sh \
  --model "$model_dir" \
  --adapter-root "$representative_adapters" \
  --checkpoint-name "$checkpoint_name" \
  --output-dir "$root_dir/results/mistral_block96_ifeval" \
  --artifact "$artifact_dir/ifeval-regression.tar.gz" \
  --status "$artifact_dir/ifeval-regression.status" \
  --seed "$representative_seed"

printf 'running stage=nolima_mechanisms seed=%s started_at=%s\n' "$representative_seed" "$(date -u +%FT%TZ)" > "$status_path"
bash scripts/run_autodl_nolima_mechanisms.sh \
  --model "$model_dir" \
  --source-data "$nolima_root/hard_gate.jsonl" \
  --source-manifest "$nolima_root/hard_gate.manifest.json" \
  --diagnostic-data "$nolima_root/hard_gate_diagnostics.jsonl" \
  --diagnostic-manifest "$nolima_root/hard_gate_diagnostics.manifest.json" \
  --free-result-dir "$root_dir/results/mistral_block96_nolima" \
  --free-seed "$representative_seed" \
  --adapter-root "$representative_adapters" \
  --checkpoint-name "$checkpoint_name" \
  --run-label block96 \
  --output-dir "$root_dir/results/mistral_block96_nolima_mechanisms" \
  --artifact "$artifact_dir/nolima-mechanisms.tar.gz" \
  --status "$artifact_dir/nolima-mechanisms.status"

printf 'running stage=rule seeds=%s started_at=%s\n' "$seeds_csv" "$(date -u +%FT%TZ)" > "$status_path"
bash scripts/run_autodl_formal_eval.sh \
  --model "$model_dir" \
  --data "$formal_source_root/eval/test.jsonl" \
  --manifest "$formal_source_root/manifest.json" \
  --adapter-root "$output_root" \
  --checkpoint-name "$checkpoint_name" \
  --seeds "$seeds_csv" \
  --output-dir "$root_dir/results/mistral_block96_rule" \
  --artifact "$artifact_dir/rule-multiseed.tar.gz" \
  --status "$artifact_dir/rule-multiseed.status"

mkdir -p "$root_dir/results/mistral_block96_seed_level" "$root_dir/results/cross_family_block96"
m_rule=()
m_nolima=()
m_longbench=()
q_rule=()
q_nolima=()
q_longbench=()
IFS=',' read -r -a seeds <<< "$seeds_csv"
for seed in "${seeds[@]}"; do
  seed="${seed//[[:space:]]/}"
  m_rule+=(--analysis "Mistral-7B-v0.3:$seed:confirmatory:$root_dir/results/mistral_block96_rule/analysis_seed_$seed/factorial_analysis.json")
  m_nolima+=(--analysis "Mistral-7B-v0.3:$seed:confirmatory:$root_dir/results/mistral_block96_nolima/analysis_seed_$seed/factorial_analysis.json")
  m_longbench+=(--analysis "Mistral-7B-v0.3:$seed:confirmatory:$root_dir/results/mistral_block96_longbench/analysis_seed_$seed/transfer_analysis.json")
  q_rule+=(--analysis "Qwen2.5-7B:$seed:corrective:$root_dir/results/qwen_block96_rule/analysis_seed_$seed/factorial_analysis.json")
  q_nolima+=(--analysis "Qwen2.5-7B:$seed:corrective:$root_dir/results/qwen_block96_nolima/analysis_seed_$seed/factorial_analysis.json")
  q_longbench+=(--analysis "Qwen2.5-7B:$seed:corrective:$root_dir/results/qwen_block96_longbench/analysis_seed_$seed/transfer_analysis.json")
done
python3 scripts/aggregate_seed_level_results.py "${m_rule[@]}" --output-dir "$root_dir/results/mistral_block96_seed_level/rule"
python3 scripts/aggregate_seed_level_results.py "${m_nolima[@]}" --output-dir "$root_dir/results/mistral_block96_seed_level/nolima"
python3 scripts/aggregate_seed_level_results.py "${m_longbench[@]}" --output-dir "$root_dir/results/mistral_block96_seed_level/longbench"
python3 scripts/aggregate_seed_level_results.py "${q_rule[@]}" "${m_rule[@]}" --output-dir "$root_dir/results/cross_family_block96/rule"
python3 scripts/aggregate_seed_level_results.py "${q_nolima[@]}" "${m_nolima[@]}" --output-dir "$root_dir/results/cross_family_block96/nolima"
python3 scripts/aggregate_seed_level_results.py "${q_longbench[@]}" "${m_longbench[@]}" --output-dir "$root_dir/results/cross_family_block96/longbench"

python3 scripts/generate_paper_results.py \
  --rule "$root_dir/results/cross_family_block96/rule/seed_level_analysis.json" \
  --nolima "$root_dir/results/cross_family_block96/nolima/seed_level_analysis.json" \
  --longbench "$root_dir/results/cross_family_block96/longbench/seed_level_analysis.json" \
  --qwen-exploratory-rule "$root_dir/results/formal_s100_seed1_frozen/analysis/factorial_analysis.json" \
  --qwen-mmlu "$root_dir/results/qwen_block96_mmlu/general_regression_analysis/general_regression_analysis.json" \
  --qwen-ifeval "$root_dir/results/qwen_block96_ifeval/official_analysis/ifeval_analysis.json" \
  --qwen-mechanisms "$root_dir/results/qwen_block96_nolima_mechanisms/mechanism_analysis/nolima_mechanism_analysis.json" \
  --mistral-mmlu "$root_dir/results/mistral_block96_mmlu/general_regression_analysis/general_regression_analysis.json" \
  --mistral-ifeval "$root_dir/results/mistral_block96_ifeval/official_analysis/ifeval_analysis.json" \
  --mistral-mechanisms "$root_dir/results/mistral_block96_nolima_mechanisms/mechanism_analysis/nolima_mechanism_analysis.json" \
  --output-tex "$root_dir/paper/generated/results.tex" \
  --output-manifest "$root_dir/paper/generated/results.manifest.json"

python3 scripts/plot_seed_level_factorial_results.py \
  --analysis "$root_dir/results/cross_family_block96/nolima/seed_level_analysis.json" \
  --output-dir "$root_dir/paper/figures" \
  --basename factorial_position_curves

python3 - "$root_dir" "$seeds_csv" "$checkpoint_name" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
paths = [
    root / "data/formal_block96_mistral7b_v03/completion.json",
    root / "results/training_subset_audits/mistral_block96_realized_subset.json",
    root / "outputs/mistral_block96/multiseed_completion.json",
    root / "results/mistral_block96_nolima/completion.json",
    root / "results/mistral_block96_longbench/completion.json",
    root / "results/mistral_block96_mmlu/completion.json",
    root / "results/mistral_block96_ifeval/completion.json",
    root / "results/mistral_block96_nolima_mechanisms/completion.json",
    root / "results/mistral_block96_rule/completion.json",
    root / "results/cross_family_block96/rule/seed_level_analysis.json",
    root / "results/cross_family_block96/nolima/seed_level_analysis.json",
    root / "results/cross_family_block96/longbench/seed_level_analysis.json",
    root / "paper/generated/results.tex",
    root / "paper/generated/results.manifest.json",
    root / "paper/figures/factorial_position_curves.manifest.json",
]
for path in paths:
    if not path.is_file():
        raise SystemExit(f"Missing Mistral block-96 completion evidence: {path}")
def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
report = {
    "schema_version": "mistral-strict-block96-completion-v1",
    "status": "validated",
    "seeds": [int(value) for value in sys.argv[2].split(",")],
    "checkpoint_name": sys.argv[3],
    "optimizer_steps": 96,
    "rows_per_training_variant": 96,
    "strict_realized_matching": True,
    "prospective_under_corrected_protocol": True,
    "artifacts": [str(path.relative_to(root)) for path in paths],
    "artifact_sha256": {str(path.relative_to(root)): sha256(path) for path in paths},
}
(root / "results/mistral_block96_completion.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

printf 'validated exit_code=0 seeds=%s finished_at=%s\n' "$seeds_csv" "$(date -u +%FT%TZ)" > "$status_path"
echo "Mistral prospective block-96 replication and corrected cross-family paper artifacts passed audit."
echo "Instance intentionally remains running; no power action was issued."
trap - EXIT
