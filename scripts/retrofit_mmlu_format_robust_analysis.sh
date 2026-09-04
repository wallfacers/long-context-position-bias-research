#!/usr/bin/env bash
set -Eeuo pipefail

root_dir=""
result_dir=""
artifact_path=""
expected_rows="14042"
bootstrap_replicates="5000"
analysis_relative="general_regression_analysis_format_robust"

usage() {
  echo "Usage: $0 --project-root DIR --result-dir DIR --artifact FILE [--expected-rows N] [--bootstrap-replicates N]"
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-root) root_dir="$2"; shift 2 ;;
    --result-dir) result_dir="$2"; shift 2 ;;
    --artifact) artifact_path="$2"; shift 2 ;;
    --expected-rows) expected_rows="$2"; shift 2 ;;
    --bootstrap-replicates) bootstrap_replicates="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ -z "$root_dir" || -z "$result_dir" || -z "$artifact_path" ]]; then
  usage >&2
  exit 2
fi
if [[ ! "$expected_rows" =~ ^[1-9][0-9]*$ || ! "$bootstrap_replicates" =~ ^[1-9][0-9]*$ ]]; then
  echo "Expected rows and bootstrap replicates must be positive integers" >&2
  exit 2
fi

root_dir="$(realpath "$root_dir")"
result_dir="$(realpath "$result_dir")"
artifact_path="$(realpath "$artifact_path")"
case "$result_dir/" in
  "$root_dir/"*) ;;
  *) echo "Result directory must be inside the project root" >&2; exit 2 ;;
esac
if [[ ! -f "$result_dir/RESULTS_READY_FOR_AGENT_REVIEW" ]]; then
  echo "Validated suite-ready marker is required; an active/partial suite is never retrofitted" >&2
  exit 2
fi
if [[ ! -s "$artifact_path" || ! -s "$artifact_path.sha256" ]]; then
  echo "Existing artifact and adjacent checksum are required" >&2
  exit 2
fi
sha256sum -c "$artifact_path.sha256"

completion="$result_dir/completion.json"
if [[ ! -s "$completion" ]]; then
  echo "Validated MMLU completion is required" >&2
  exit 2
fi
runs=(
  base
  independent_answer
  independent_evidence_id
  independent_evidence
  paired_answer
  paired_evidence_id
  paired_evidence
)
files=(
  base.jsonl
  independent_answer_s100.jsonl
  independent_evidence_id_s100.jsonl
  independent_evidence_s100.jsonl
  paired_answer_s100.jsonl
  paired_evidence_id_s100.jsonl
  paired_evidence_s100.jsonl
)
run_args=()
for index in "${!runs[@]}"; do
  path="$result_dir/${files[$index]}"
  if [[ ! -s "$path" ]]; then
    echo "Missing MMLU result: $path" >&2
    exit 2
  fi
  count="$(awk 'NF {count++} END {print count+0}' "$path")"
  if [[ "$count" -ne "$expected_rows" ]]; then
    echo "${runs[$index]}: expected $expected_rows rows, found $count" >&2
    exit 2
  fi
  run_args+=(--run "${runs[$index]}=$path")
done

analysis_dir="$result_dir/$analysis_relative"
python3 "$root_dir/scripts/analyze_general_regression.py" \
  "${run_args[@]}" \
  --output-dir "$analysis_dir" \
  --bootstrap-replicates "$bootstrap_replicates" \
  --seed 20260828 \
  --noninferiority-margin 0.02

python3 - "$analysis_dir/general_regression_analysis.json" "$expected_rows" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = int(sys.argv[2])
if report.get("schema_version") != "general-regression-analysis-v1":
    raise SystemExit("Unexpected regression schema")
if report.get("scoring_protocol", {}).get("name") != "format-robust-option-extraction-v1":
    raise SystemExit("Format-robust scoring protocol is missing")
if set(report.get("rows_per_run", {}).values()) != {expected}:
    raise SystemExit("Unexpected format-robust row counts")
if len(report.get("source_sha256", {})) != 7:
    raise SystemExit("All seven source hashes are required")
for run, item in report.get("generation_diagnostics", {}).items():
    if float(item.get("option_extraction_rate", 0.0)) < 0.99:
        raise SystemExit(f"{run}: option extraction rate is below 99%")
PY

completion_backup="$(mktemp "$result_dir/.completion.mmlu-rescore-backup.XXXXXX")"
artifact_temporary="$(mktemp "$(dirname "$artifact_path")/.$(basename "$artifact_path").mmlu-rescore.XXXXXX")"
checksum_temporary="$(mktemp "$(dirname "$artifact_path")/.$(basename "$artifact_path").sha256.mmlu-rescore.XXXXXX")"
cp "$completion" "$completion_backup"
artifact_moved=0
cleanup() {
  local rc=$?
  if [[ "$rc" -ne 0 && "$artifact_moved" -eq 0 && -s "$completion_backup" ]]; then
    mv -f "$completion_backup" "$completion"
  fi
  rm -f "$completion_backup" "$artifact_temporary" "$checksum_temporary"
  exit "$rc"
}
trap cleanup EXIT

python3 - "$completion" "$analysis_relative" "$bootstrap_replicates" <<'PY'
import json
import os
import sys
from pathlib import Path

completion = Path(sys.argv[1])
analysis_relative = sys.argv[2]
bootstrap_replicates = int(sys.argv[3])
report = json.loads(completion.read_text(encoding="utf-8"))
if report.get("status") != "validated" or report.get("schema_version") != "mmlu-regression-completion-v1":
    raise SystemExit("MMLU completion is not validated")
current = report.get("paired_analysis")
if current != analysis_relative:
    report["legacy_format_constrained_analysis"] = current
report["paired_analysis"] = f"{analysis_relative}/general_regression_analysis.json"
report["protocol"] = "full MMLU test, zero-shot format-robust generative option-letter accuracy"
report["scoring_protocol"] = "format-robust-option-extraction-v1"
report["bootstrap_replicates"] = bootstrap_replicates
temporary = completion.with_name(completion.name + f".tmp-{os.getpid()}")
temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(completion)
PY

result_relative="$(realpath --relative-to="$root_dir" "$result_dir")"
tar -C "$root_dir" -czf "$artifact_temporary" "$result_relative"
python3 - "$artifact_temporary" "$result_relative/$analysis_relative/general_regression_analysis.json" "$result_relative/completion.json" <<'PY'
import json
import sys
import tarfile

with tarfile.open(sys.argv[1], "r:gz") as archive:
    names = set(archive.getnames())
    if sys.argv[2] not in names or sys.argv[3] not in names:
        raise SystemExit("Repacked artifact lacks robust analysis or completion")
    completion = json.load(archive.extractfile(sys.argv[3]))
    if completion.get("scoring_protocol") != "format-robust-option-extraction-v1":
        raise SystemExit("Repacked completion does not select robust scoring")
PY
new_digest="$(sha256sum "$artifact_temporary" | cut -d' ' -f1)"
printf '%s  %s\n' "$new_digest" "$artifact_path" > "$checksum_temporary"
mv -f "$artifact_temporary" "$artifact_path"
artifact_moved=1
mv -f "$checksum_temporary" "$artifact_path.sha256"
sha256sum -c "$artifact_path.sha256"
rm -f "$completion_backup"
trap - EXIT
echo "Retrofitted format-robust MMLU analysis into $artifact_path"
