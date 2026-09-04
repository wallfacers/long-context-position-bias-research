#!/usr/bin/env bash
set -Eeuo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir=""
adapter_root="$root_dir/outputs/formal_matched/seed_20260825"
formal_completion="$root_dir/results/formal_s100_seed1_frozen/completion.json"
status_path="/root/autodl-tmp/qwen-seed1-completion-queue.status"
artifact_dir="/root/autodl-tmp/qwen-seed1-completion-artifacts"

usage() {
  echo "Usage: $0 --model LOCAL_DIR [--adapter-root DIR] [--formal-completion JSON] [--status FILE] [--artifact-dir DIR]"
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) model_dir="$2"; shift 2 ;;
    --adapter-root) adapter_root="$2"; shift 2 ;;
    --formal-completion) formal_completion="$2"; shift 2 ;;
    --status) status_path="$2"; shift 2 ;;
    --artifact-dir) artifact_dir="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -s "$model_dir/config.json" || ! -s "$model_dir/model_manifest.json" ]]; then
  echo "A complete manifested model is required" >&2
  exit 2
fi
if [[ ! -s "$formal_completion" ]]; then
  echo "The frozen formal seed-1 completion gate has not passed: $formal_completion" >&2
  exit 2
fi
python3 - "$formal_completion" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("status") != "validated" or payload.get("rows_per_run") != 4200:
    raise SystemExit("Frozen formal result has not passed its completion audit")
PY

mkdir -p "$artifact_dir" "$(dirname "$status_path")"
printf 'running stage=nolima_gate started_at=%s\n' "$(date -u +%FT%TZ)" > "$status_path"
on_exit() {
  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    printf 'failed exit_code=%s finished_at=%s\n' "$rc" "$(date -u +%FT%TZ)" > "$status_path"
    echo "Qwen seed-1 completion queue failed; completed suites and resumable rows were preserved."
  fi
}
trap on_exit EXIT

cd "$root_dir"
printf 'running stage=nolima_gate started_at=%s\n' "$(date -u +%FT%TZ)" > "$status_path"
bash scripts/run_autodl_nolima_gate.sh \
  --model "$model_dir" \
  --adapter-root "$adapter_root" \
  --output-dir "$root_dir/results/nolima_hard_gate_seed1" \
  --artifact "$artifact_dir/nolima-hard-gate.tar.gz" \
  --status "$artifact_dir/nolima-hard-gate.status"

printf 'running stage=longbench started_at=%s\n' "$(date -u +%FT%TZ)" > "$status_path"
bash scripts/run_autodl_longbench_transfer.sh \
  --model "$model_dir" \
  --adapter-root "$adapter_root" \
  --output-dir "$root_dir/results/longbench_transfer_seed1" \
  --artifact "$artifact_dir/longbench-transfer.tar.gz" \
  --status "$artifact_dir/longbench-transfer.status"

printf 'running stage=mmlu started_at=%s\n' "$(date -u +%FT%TZ)" > "$status_path"
bash scripts/run_autodl_mmlu_regression.sh \
  --model "$model_dir" \
  --adapter-root "$adapter_root" \
  --output-dir "$root_dir/results/mmlu_regression_seed1" \
  --artifact "$artifact_dir/mmlu-regression.tar.gz" \
  --status "$artifact_dir/mmlu-regression.status"

printf 'running stage=ifeval started_at=%s\n' "$(date -u +%FT%TZ)" > "$status_path"
bash scripts/run_autodl_ifeval_regression.sh \
  --model "$model_dir" \
  --adapter-root "$adapter_root" \
  --output-dir "$root_dir/results/ifeval_regression_seed1" \
  --artifact "$artifact_dir/ifeval-regression.tar.gz" \
  --status "$artifact_dir/ifeval-regression.status" \
  --seed 20260825

printf 'running stage=nolima_mechanisms started_at=%s\n' "$(date -u +%FT%TZ)" > "$status_path"
bash scripts/run_autodl_nolima_mechanisms.sh \
  --model "$model_dir" \
  --free-result-dir "$root_dir/results/nolima_hard_gate_seed1" \
  --adapter-root "$adapter_root" \
  --output-dir "$root_dir/results/nolima_mechanisms_seed1" \
  --artifact "$artifact_dir/nolima-mechanisms.tar.gz" \
  --status "$artifact_dir/nolima-mechanisms.status"

python3 - "$root_dir/results" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
expected = {
    "nolima_hard_gate_seed1/completion.json": "nolima-hard-gate-completion-v2",
    "longbench_transfer_seed1/completion.json": "longbench-natural-transfer-completion-v1",
    "mmlu_regression_seed1/completion.json": "mmlu-regression-completion-v1",
    "ifeval_regression_seed1/completion.json": "ifeval-regression-completion-v1",
    "nolima_mechanisms_seed1/completion.json": "nolima-mechanism-completion-v1",
}
records = {}
for relative, schema in expected.items():
    path = root / relative
    if not path.is_file():
        raise SystemExit(f"Missing suite completion: {relative}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "validated" or payload.get("schema_version") != schema:
        raise SystemExit(f"Suite completion failed: {relative}")
    records[relative] = payload
report = {
    "schema_version": "qwen-seed1-completion-queue-v1",
    "status": "validated",
    "suites": list(expected),
    "records": records,
}
(root / "qwen_seed1_completion.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

printf 'validated exit_code=0 finished_at=%s\n' "$(date -u +%FT%TZ)" > "$status_path"
echo "Qwen seed-1 OOD, transfer, regression, and mechanism suites all passed audit."
echo "Instance intentionally remains running; no shutdown command was issued."
trap - EXIT
