#!/usr/bin/env bash
set -Eeuo pipefail

root_dir=""
result_dir=""
artifact_path=""
expected_catalogs=""
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  echo "Usage: $0 --project-root DIR --result-dir DIR --artifact FILE --expected-catalogs N"
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-root) root_dir="$2"; shift 2 ;;
    --result-dir) result_dir="$2"; shift 2 ;;
    --artifact) artifact_path="$2"; shift 2 ;;
    --expected-catalogs) expected_catalogs="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ -z "$root_dir" || -z "$result_dir" || -z "$artifact_path" || ! "$expected_catalogs" =~ ^[1-9][0-9]*$ ]]; then
  usage >&2
  exit 2
fi
root_dir="$(realpath "$root_dir")"
result_dir="$(realpath "$result_dir")"
artifact_path="$(realpath "$artifact_path")"
case "$result_dir/" in
  "$root_dir/"*) ;;
  *) echo "Result directory must be inside the project root" >&2; exit 2 ;;
esac
if [[ ! -f "$result_dir/RESULTS_READY_FOR_AGENT_REVIEW" || ! -s "$result_dir/completion.json" ]]; then
  echo "Validated suite-ready marker and completion are required" >&2
  exit 2
fi
if [[ ! -s "$artifact_path" || ! -s "$artifact_path.sha256" ]]; then
  echo "Existing artifact and adjacent checksum are required" >&2
  exit 2
fi
sha256sum -c "$artifact_path.sha256"

audit_path="$result_dir/failure_case_catalog_audit.json"
audit_temporary="$(mktemp "$result_dir/.failure-catalog-audit.XXXXXX.json")"
artifact_temporary="$(mktemp "$(dirname "$artifact_path")/.$(basename "$artifact_path").catalog-repack.XXXXXX")"
checksum_temporary="$(mktemp "$(dirname "$artifact_path")/.$(basename "$artifact_path").sha256.catalog-repack.XXXXXX")"
audit_backup=""
artifact_moved=0
cleanup() {
  local rc=$?
  if [[ "$rc" -ne 0 && "$artifact_moved" -eq 0 && -n "$audit_backup" && -s "$audit_backup" ]]; then
    mv -f "$audit_backup" "$audit_path"
  fi
  if [[ "$rc" -ne 0 && "$artifact_moved" -eq 0 && -z "$audit_backup" ]]; then
    rm -f "$audit_path"
  fi
  rm -f "$audit_temporary" "$artifact_temporary" "$checksum_temporary" "$audit_backup"
  exit "$rc"
}
trap cleanup EXIT

python3 "$script_dir/audit_failure_case_catalogs.py" \
  --project-root "$root_dir" \
  --results-root "$result_dir" \
  --expected-catalogs "$expected_catalogs" \
  --output "$audit_temporary"
python3 - "$result_dir" "$expected_catalogs" <<'PY'
import json, sys
from pathlib import Path

result = Path(sys.argv[1])
expected = int(sys.argv[2])
completion = json.loads((result / "completion.json").read_text(encoding="utf-8"))
if completion.get("status") != "validated" or not completion.get("schema_version"):
    raise SystemExit("Suite completion is not validated")
found = sorted(
    path.relative_to(result).as_posix()
    for path in result.rglob("failure_case_catalog.manifest.json")
)
if len(found) != expected:
    raise SystemExit("Completion/catalog count differs")
if expected == 1 and completion.get("failure_case_catalog") == found[0]:
    pass
elif completion.get("failure_case_catalogs") == found:
    pass
else:
    raise SystemExit("Completion does not reference the exact catalog set")
PY

if [[ -e "$audit_path" ]]; then
  audit_backup="$(mktemp "$result_dir/.failure-catalog-audit-backup.XXXXXX.json")"
  cp "$audit_path" "$audit_backup"
fi
mv -f "$audit_temporary" "$audit_path"

result_relative="$(realpath --relative-to="$root_dir" "$result_dir")"
tar -C "$root_dir" -czf "$artifact_temporary" "$result_relative"
python3 - "$artifact_temporary" "$result_relative" "$expected_catalogs" <<'PY'
import sys, tarfile

artifact, result, expected = sys.argv[1], sys.argv[2], int(sys.argv[3])
with tarfile.open(artifact, "r:gz") as archive:
    names = set(archive.getnames())
required = {
    f"{result}/completion.json",
    f"{result}/RESULTS_READY_FOR_AGENT_REVIEW",
    f"{result}/failure_case_catalog_audit.json",
}
if not required.issubset(names):
    raise SystemExit("Replacement artifact lacks completion, ready marker, or audit")
catalogs = [
    name for name in names
    if name.startswith(result + "/") and name.endswith("/failure_case_catalog.manifest.json")
]
if len(catalogs) != expected:
    raise SystemExit("Replacement artifact has the wrong catalog count")
PY
new_digest="$(sha256sum "$artifact_temporary" | cut -d' ' -f1)"
printf '%s  %s\n' "$new_digest" "$artifact_path" > "$checksum_temporary"
mv -f "$artifact_temporary" "$artifact_path"
artifact_moved=1
mv -f "$checksum_temporary" "$artifact_path.sha256"
sha256sum -c "$artifact_path.sha256"
rm -f "$audit_backup"
trap - EXIT
echo "Repacked $expected_catalogs audited failure catalogs into $artifact_path"
