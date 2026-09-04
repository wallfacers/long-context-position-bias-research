#!/usr/bin/env bash
set -Eeuo pipefail

root_dir=""
result_dir=""
artifact_path=""
catalog_relative="failure_cases/failure_case_catalog.manifest.json"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  echo "Usage: $0 --project-root DIR --result-dir DIR --artifact FILE [--catalog-relative PATH]"
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-root) root_dir="$2"; shift 2 ;;
    --result-dir) result_dir="$2"; shift 2 ;;
    --artifact) artifact_path="$2"; shift 2 ;;
    --catalog-relative) catalog_relative="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ -z "$root_dir" || -z "$result_dir" || -z "$artifact_path" ]]; then
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
catalog="$result_dir/$catalog_relative"
if [[ ! -s "$completion" || ! -s "$catalog" ]]; then
  echo "Validated completion and failure catalog manifest are required" >&2
  exit 2
fi

python3 - "$root_dir" "$catalog" "$script_dir" <<'PY'
import sys
from pathlib import Path

root, catalog, script_dir = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, script_dir)
from audit_failure_case_catalogs import audit_catalog

audit_catalog(catalog, root)
PY

completion_backup="$(mktemp "$result_dir/.completion.retrofit-backup.XXXXXX")"
artifact_temporary="$(mktemp "$(dirname "$artifact_path")/.$(basename "$artifact_path").retrofit.XXXXXX")"
checksum_temporary="$(mktemp "$(dirname "$artifact_path")/.$(basename "$artifact_path").sha256.retrofit.XXXXXX")"
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

python3 - "$completion" "$catalog" "$catalog_relative" <<'PY'
import json
import os
import sys
from pathlib import Path

completion, catalog, relative = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
report = json.loads(completion.read_text(encoding="utf-8"))
manifest = json.loads(catalog.read_text(encoding="utf-8"))
if report.get("status") != "validated" or not report.get("schema_version"):
    raise SystemExit("Suite completion is not validated")
if (
    manifest.get("status") != "validated"
    or manifest.get("schema_version") != "failure-case-catalog-manifest-v1"
):
    raise SystemExit("Failure catalog manifest is not validated")
existing = report.get("failure_case_catalog")
if existing not in (None, relative):
    raise SystemExit("Completion already references a different failure catalog")
report["failure_case_catalog"] = relative
temporary = completion.with_name(completion.name + f".tmp-{os.getpid()}")
temporary.write_text(
    json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
temporary.replace(completion)
PY

result_relative="$(realpath --relative-to="$root_dir" "$result_dir")"
tar -C "$root_dir" -czf "$artifact_temporary" "$result_relative"
python3 - "$artifact_temporary" "$result_relative/$catalog_relative" <<'PY'
import sys
import tarfile

with tarfile.open(sys.argv[1], "r:gz") as archive:
    if sys.argv[2] not in archive.getnames():
        raise SystemExit("Repacked artifact does not contain the failure catalog")
PY
new_digest="$(sha256sum "$artifact_temporary" | cut -d' ' -f1)"
printf '%s  %s\n' "$new_digest" "$artifact_path" > "$checksum_temporary"
mv -f "$artifact_temporary" "$artifact_path"
artifact_moved=1
mv -f "$checksum_temporary" "$artifact_path.sha256"
sha256sum -c "$artifact_path.sha256"
rm -f "$completion_backup"
trap - EXIT
echo "Retrofitted validated failure catalog into $artifact_path"
