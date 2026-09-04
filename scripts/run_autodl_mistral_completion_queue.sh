#!/usr/bin/env bash
set -Eeuo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir=""
formal_root="$root_dir/data/formal_matched_mistral7b_v03"
nolima_root="$root_dir/data/ood_nolima_mistral7b_v03"
output_root="$root_dir/outputs/mistral7b_v03"
seeds_csv="20260825,20260826,20260827"
status_path="/root/autodl-tmp/mistral-completion-queue.status"
artifact_dir="/root/autodl-tmp/mistral-completion-artifacts"

usage() {
  echo "Usage: $0 --model LOCAL_DIR [--formal-root DIR] [--nolima-root DIR] [--output-root DIR] [--seeds CSV] [--status FILE] [--artifact-dir DIR]"
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) model_dir="$2"; shift 2 ;;
    --formal-root) formal_root="$2"; shift 2 ;;
    --nolima-root) nolima_root="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    --seeds) seeds_csv="$2"; shift 2 ;;
    --status) status_path="$2"; shift 2 ;;
    --artifact-dir) artifact_dir="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -s "$model_dir/config.json" || ! -s "$model_dir/model_manifest.json" ]]; then
  echo "A complete manifested Mistral snapshot is required" >&2
  exit 2
fi
if [[ ! -s "$formal_root/completion.json" ]]; then
  echo "Mistral tokenizer-specific data completion gate is missing" >&2
  exit 2
fi
if [[ ! -s "$root_dir/results/qwen_confirmatory_completion.json" ]]; then
  echo "Qwen confirmatory queue must finish before the second-family GPU queue" >&2
  exit 2
fi
python3 - "$formal_root/completion.json" "$model_dir/model_manifest.json" "$seeds_csv" <<'PY'
import hashlib, json, sys
from pathlib import Path
data = json.load(open(sys.argv[1], encoding="utf-8"))
model = json.load(open(sys.argv[2], encoding="utf-8"))
seeds = [int(x.strip()) for x in sys.argv[3].split(",") if x.strip()]
if data.get("status") != "validated" or data.get("training_seeds") != seeds:
    raise SystemExit("Mistral family data seeds differ from the frozen queue")
if data.get("revision") != model.get("revision"):
    raise SystemExit("Mistral data/model revisions differ")
if len(seeds) != 3:
    raise SystemExit("The second-family confirmation requires exactly three seeds")
if seeds != [20260825, 20260826, 20260827]:
    raise SystemExit("The frozen second-family seeds or their order changed")
source_record = data.get("files", {}).get("nolima_source_download_manifest")
if not source_record:
    raise SystemExit("Mistral family data lacks the frozen NoLiMa source-download attestation")
source_path = Path(source_record["path"])
if not source_path.is_file():
    raise SystemExit("Frozen NoLiMa source-download manifest is missing")
actual = hashlib.sha256(source_path.read_bytes()).hexdigest()
if actual != source_record.get("sha256"):
    raise SystemExit("Frozen NoLiMa source-download manifest hash differs")
source = json.loads(source_path.read_text(encoding="utf-8"))
if (
    source.get("schema_version") != "nolima-frozen-source-download-v1"
    or source.get("status") != "validated"
):
    raise SystemExit("Frozen NoLiMa source-download attestation has not passed")
PY

mkdir -p "$artifact_dir" "$(dirname "$status_path")"
on_exit() {
  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    printf 'failed exit_code=%s finished_at=%s\n' "$rc" "$(date -u +%FT%TZ)" > "$status_path"
    echo "Mistral completion queue failed; audited completed stages and resumable rows remain intact."
  fi
}
trap on_exit EXIT
cd "$root_dir"

printf 'running stage=training seeds=%s started_at=%s\n' "$seeds_csv" "$(date -u +%FT%TZ)" > "$status_path"
bash scripts/run_autodl_fixed100_multiseed_training.sh \
  --model "$model_dir" \
  --data-root "$formal_root" \
  --output-root "$output_root" \
  --seeds "$seeds_csv" \
  --artifact-dir "$artifact_dir/training" \
  --status "$artifact_dir/training.status"

printf 'running stage=nolima seeds=%s started_at=%s\n' "$seeds_csv" "$(date -u +%FT%TZ)" > "$status_path"
bash scripts/run_autodl_nolima_multiseed.sh \
  --model "$model_dir" \
  --data "$nolima_root/hard_gate.jsonl" \
  --manifest "$nolima_root/hard_gate.manifest.json" \
  --adapter-root "$output_root" \
  --seeds "$seeds_csv" \
  --output-dir "$root_dir/results/mistral_nolima" \
  --artifact "$artifact_dir/nolima-multiseed.tar.gz" \
  --status "$artifact_dir/nolima-multiseed.status"

printf 'running stage=longbench seeds=%s started_at=%s\n' "$seeds_csv" "$(date -u +%FT%TZ)" > "$status_path"
bash scripts/run_autodl_longbench_multiseed.sh \
  --model "$model_dir" \
  --adapter-root "$output_root" \
  --seeds "$seeds_csv" \
  --output-dir "$root_dir/results/mistral_longbench" \
  --artifact "$artifact_dir/longbench-multiseed.tar.gz" \
  --status "$artifact_dir/longbench-multiseed.status"

representative_seed="${seeds_csv%%,*}"
representative_seed="${representative_seed//[[:space:]]/}"
representative_adapters="$output_root/seed_$representative_seed"
printf 'running stage=mmlu seed=%s started_at=%s\n' "$representative_seed" "$(date -u +%FT%TZ)" > "$status_path"
bash scripts/run_autodl_mmlu_regression.sh \
  --model "$model_dir" \
  --adapter-root "$representative_adapters" \
  --output-dir "$root_dir/results/mistral_mmlu" \
  --artifact "$artifact_dir/mmlu-regression.tar.gz" \
  --status "$artifact_dir/mmlu-regression.status"

printf 'running stage=ifeval seed=%s started_at=%s\n' "$representative_seed" "$(date -u +%FT%TZ)" > "$status_path"
bash scripts/run_autodl_ifeval_regression.sh \
  --model "$model_dir" \
  --adapter-root "$representative_adapters" \
  --output-dir "$root_dir/results/mistral_ifeval" \
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
  --free-result-dir "$root_dir/results/mistral_nolima" \
  --free-seed "$representative_seed" \
  --adapter-root "$representative_adapters" \
  --output-dir "$root_dir/results/mistral_nolima_mechanisms" \
  --artifact "$artifact_dir/nolima-mechanisms.tar.gz" \
  --status "$artifact_dir/nolima-mechanisms.status"

printf 'running stage=rule seeds=%s started_at=%s\n' "$seeds_csv" "$(date -u +%FT%TZ)" > "$status_path"
bash scripts/run_autodl_formal_eval.sh \
  --model "$model_dir" \
  --data "$formal_root/eval/test.jsonl" \
  --manifest "$formal_root/manifest.json" \
  --adapter-root "$output_root" \
  --seeds "$seeds_csv" \
  --output-dir "$root_dir/results/mistral_rule" \
  --artifact "$artifact_dir/rule-multiseed.tar.gz" \
  --status "$artifact_dir/rule-multiseed.status"

IFS=',' read -r -a seeds <<< "$seeds_csv"
for index in "${!seeds[@]}"; do seeds[$index]="${seeds[$index]//[[:space:]]/}"; done
mkdir -p "$root_dir/results/mistral_seed_level" "$root_dir/results/cross_family"
m_rule=()
m_nolima=()
m_longbench=()
for seed in "${seeds[@]}"; do
  m_rule+=(--analysis "Mistral-7B-v0.3:$seed:confirmatory:$root_dir/results/mistral_rule/analysis_seed_$seed/factorial_analysis.json")
  m_nolima+=(--analysis "Mistral-7B-v0.3:$seed:confirmatory:$root_dir/results/mistral_nolima/analysis_seed_$seed/factorial_analysis.json")
  m_longbench+=(--analysis "Mistral-7B-v0.3:$seed:confirmatory:$root_dir/results/mistral_longbench/analysis_seed_$seed/transfer_analysis.json")
done
python3 scripts/aggregate_seed_level_results.py "${m_rule[@]}" --output-dir "$root_dir/results/mistral_seed_level/rule"
python3 scripts/aggregate_seed_level_results.py "${m_nolima[@]}" --output-dir "$root_dir/results/mistral_seed_level/nolima"
python3 scripts/aggregate_seed_level_results.py "${m_longbench[@]}" --output-dir "$root_dir/results/mistral_seed_level/longbench"

q_rule=(
  --analysis "Qwen2.5-7B:20260825:pilot:$root_dir/results/formal_s100_seed1_frozen/analysis/factorial_analysis.json"
  --analysis "Qwen2.5-7B:20260826:confirmatory:$root_dir/results/qwen_confirmatory_rule/analysis_seed_20260826/factorial_analysis.json"
  --analysis "Qwen2.5-7B:20260827:confirmatory:$root_dir/results/qwen_confirmatory_rule/analysis_seed_20260827/factorial_analysis.json"
)
q_nolima=(
  --analysis "Qwen2.5-7B:20260825:pilot:$root_dir/results/nolima_hard_gate_seed1/factorial_analysis/factorial_analysis.json"
  --analysis "Qwen2.5-7B:20260826:confirmatory:$root_dir/results/qwen_confirmatory_nolima/analysis_seed_20260826/factorial_analysis.json"
  --analysis "Qwen2.5-7B:20260827:confirmatory:$root_dir/results/qwen_confirmatory_nolima/analysis_seed_20260827/factorial_analysis.json"
)
q_longbench=(
  --analysis "Qwen2.5-7B:20260825:pilot:$root_dir/results/longbench_transfer_seed1/transfer_analysis/transfer_analysis.json"
  --analysis "Qwen2.5-7B:20260826:confirmatory:$root_dir/results/qwen_confirmatory_longbench/analysis_seed_20260826/transfer_analysis.json"
  --analysis "Qwen2.5-7B:20260827:confirmatory:$root_dir/results/qwen_confirmatory_longbench/analysis_seed_20260827/transfer_analysis.json"
)
python3 scripts/aggregate_seed_level_results.py "${q_rule[@]}" "${m_rule[@]}" --output-dir "$root_dir/results/cross_family/rule"
python3 scripts/aggregate_seed_level_results.py "${q_nolima[@]}" "${m_nolima[@]}" --output-dir "$root_dir/results/cross_family/nolima"
python3 scripts/aggregate_seed_level_results.py "${q_longbench[@]}" "${m_longbench[@]}" --output-dir "$root_dir/results/cross_family/longbench"

python3 scripts/generate_paper_results.py \
  --rule "$root_dir/results/cross_family/rule/seed_level_analysis.json" \
  --nolima "$root_dir/results/cross_family/nolima/seed_level_analysis.json" \
  --longbench "$root_dir/results/cross_family/longbench/seed_level_analysis.json" \
  --qwen-exploratory-rule "$root_dir/results/formal_s100_seed1_frozen/analysis/factorial_analysis.json" \
  --qwen-mmlu "$root_dir/results/mmlu_regression_seed1/general_regression_analysis_format_robust/general_regression_analysis.json" \
  --qwen-ifeval "$root_dir/results/ifeval_regression_seed1/official_analysis/ifeval_analysis.json" \
  --qwen-mechanisms "$root_dir/results/nolima_mechanisms_seed1/mechanism_analysis/nolima_mechanism_analysis.json" \
  --mistral-mmlu "$root_dir/results/mistral_mmlu/general_regression_analysis/general_regression_analysis.json" \
  --mistral-ifeval "$root_dir/results/mistral_ifeval/official_analysis/ifeval_analysis.json" \
  --mistral-mechanisms "$root_dir/results/mistral_nolima_mechanisms/mechanism_analysis/nolima_mechanism_analysis.json" \
  --output-tex "$root_dir/paper/generated/results.tex" \
  --output-manifest "$root_dir/paper/generated/results.manifest.json"

python3 scripts/plot_seed_level_factorial_results.py \
  --analysis "$root_dir/results/cross_family/nolima/seed_level_analysis.json" \
  --output-dir "$root_dir/paper/figures" \
  --basename factorial_position_curves

python3 - "$root_dir/results" "$root_dir/paper/generated/results.manifest.json" "$seeds_csv" "$formal_root/completion.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, paper_manifest, data_completion = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[4])
required = [
    data_completion,
    root / "mistral_nolima/completion.json",
    root / "mistral_longbench/completion.json",
    root / "mistral_rule/completion.json",
    root / "mistral_mmlu/completion.json",
    root / "mistral_ifeval/completion.json",
    root / "mistral_nolima_mechanisms/completion.json",
    root / "cross_family/rule/seed_level_analysis.json",
    root / "cross_family/nolima/seed_level_analysis.json",
    root / "cross_family/longbench/seed_level_analysis.json",
    paper_manifest,
    root.parent / "paper/figures/factorial_position_curves.manifest.json",
]
for path in required:
    if not path.is_file():
        raise SystemExit(f"Missing second-family completion artifact: {path}")
def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
report = {
    "schema_version": "mistral-second-family-completion-v1",
    "status": "validated",
    "confirmatory_seeds": [int(x.strip()) for x in sys.argv[3].split(",")],
    "artifacts": [str(path) for path in required],
    "artifact_sha256": {str(path): sha256(path) for path in required},
    "paper_numbers_generated": True,
}
(root / "mistral_second_family_completion.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

printf 'validated exit_code=0 seeds=%s finished_at=%s\n' \
  "$seeds_csv" "$(date -u +%FT%TZ)" > "$status_path"
echo "Mistral three-seed replication and cross-family paper-number audit completed."
echo "Instance intentionally remains running; no shutdown command was issued."
trap - EXIT
