#!/usr/bin/env bash
set -Eeuo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
base_dir="$root_dir/results/test_full"
paired_evidence_dir="$root_dir/results/formal_s100_full_gate"
five_dir="$root_dir/results/formal_s100_seed1"
output_dir="$root_dir/results/formal_s100_seed1_frozen"
artifact_path="/root/autodl-tmp/qwen-formal-s100-seed1-frozen.tar.gz"
status_path="/root/autodl-tmp/qwen-formal-s100-seed1-frozen.status"
model_dir="/root/autodl-tmp/models/Qwen2.5-7B-Instruct"

usage() {
  echo "Usage: $0 [--base-dir DIR] [--paired-evidence-dir DIR] [--five-dir DIR] [--output-dir DIR] [--model DIR] [--artifact FILE] [--status FILE]"
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-dir) base_dir="$2"; shift 2 ;;
    --paired-evidence-dir) paired_evidence_dir="$2"; shift 2 ;;
    --five-dir) five_dir="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --model) model_dir="$2"; shift 2 ;;
    --artifact) artifact_path="$2"; shift 2 ;;
    --status) status_path="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -s "$model_dir/model_manifest.json" || ! -s "$model_dir/model_integrity_attestation.json" ]]; then
  echo "Pinned model manifest and integrity attestation are required" >&2
  exit 2
fi

case "$(realpath -m "$output_dir")/" in
  "$(realpath "$root_dir")/"*) ;;
  *) echo "--output-dir must be inside the project root" >&2; exit 2 ;;
esac

canonical=(
  base
  independent_answer
  independent_evidence_id
  independent_evidence
  paired_answer
  paired_evidence_id
  paired_evidence
)
sources=(
  "$base_dir/base.jsonl"
  "$five_dir/independent_answer_s100.jsonl"
  "$five_dir/independent_evidence_id_s100.jsonl"
  "$five_dir/independent_evidence_s100.jsonl"
  "$five_dir/paired_answer_s100.jsonl"
  "$five_dir/paired_evidence_id_s100.jsonl"
  "$paired_evidence_dir/paired_evidence_s100.jsonl"
)

python3 - "${sources[@]}" <<'PY'
import json
import sys
from pathlib import Path

paths = [Path(value) for value in sys.argv[1:]]
reference_ids = None
identities = []
for path in paths:
    metadata_path = path.with_suffix(path.suffix + ".run.json")
    if not path.is_file() or not metadata_path.is_file():
        raise SystemExit(f"Missing result or run identity: {path}")
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if len(rows) != 4200:
        raise SystemExit(f"{path}: expected 4,200 rows, found {len(rows)}")
    ids = {row["sample_id"] for row in rows}
    if len(ids) != 4200:
        raise SystemExit(f"{path}: duplicate sample IDs")
    if reference_ids is None:
        reference_ids = ids
    elif ids != reference_ids:
        raise SystemExit(f"{path}: sample IDs differ from base")
    identities.append(json.loads(metadata_path.read_text(encoding="utf-8")))
for field in ("data_sha256", "selection_sha256", "model"):
    values = {identity.get(field) for identity in identities}
    if len(values) != 1:
        raise SystemExit(f"Run identity mismatch in {field}: {values}")
for identity in identities:
    if int(identity.get("selected_samples", 4200)) != 4200:
        raise SystemExit("A source run did not select the complete formal test")
print("Validated seven complete, identity-matched source runs")
PY

mkdir -p "$output_dir/source_run_metadata" "$output_dir/reproducibility" \
  "$output_dir/analysis" "$output_dir/figures" \
  "$(dirname "$artifact_path")" "$(dirname "$status_path")"
printf 'running started_at=%s\n' "$(date -u +%FT%TZ)" > "$status_path"
on_exit() {
  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    printf 'failed exit_code=%s finished_at=%s\n' "$rc" "$(date -u +%FT%TZ)" > "$status_path"
  fi
}
trap on_exit EXIT

python3 - "$root_dir/data/pilot_qwen25_7b/raw/test.jsonl" "$model_dir" \
  "$output_dir/reproducibility" "${sources[@]}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path.cwd() / "src"))
from position_bias_research.chat_protocol import (
    apply_chat_template,
    selected_protocol_for_tokenizer,
)

data, model, output = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sources = [Path(value) for value in sys.argv[4:]]

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

manifest_path = model / "model_manifest.json"
attestation_path = model / "model_integrity_attestation.json"
chat_audit_path = model / "chat_protocol_audit.json"
pinned_config_path = Path.cwd() / "configs/qwen25_7b_model.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
pinned_config = json.loads(pinned_config_path.read_text(encoding="utf-8"))
manifest_sha = sha256(manifest_path)
if (
    manifest.get("schema_version") != "local-model-manifest-v1"
    or attestation.get("schema_version") != "model-integrity-attestation-v1"
    or Path(attestation.get("model", "")).resolve() != model.resolve()
    or attestation.get("manifest_sha256") != manifest_sha
    or attestation.get("revision") != manifest.get("revision")
):
    raise SystemExit("Pinned Qwen model attestation identity failed")
for record in attestation.get("file_state", []):
    path = model / record["path"]
    stat = path.stat()
    if stat.st_size != int(record["bytes"]) or stat.st_mtime_ns != int(record["mtime_ns"]):
        raise SystemExit(f"Model file changed after attestation: {path}")

rows = [json.loads(line) for line in data.open(encoding="utf-8") if line.strip()]
if len(rows) != 4200 or len({row["sample_id"] for row in rows}) != 4200:
    raise SystemExit("Formal prompt source is not the frozen 4,200-row test")
tokenizer = AutoTokenizer.from_pretrained(
    str(model), local_files_only=True, trust_remote_code=False
)
tokenizer_payload = {
    "backend": tokenizer.backend_tokenizer.to_str(),
    "chat_template": tokenizer.chat_template,
    "special_tokens_map": tokenizer.special_tokens_map,
}
tokenizer_fingerprint = hashlib.sha256(
    json.dumps(tokenizer_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
).hexdigest()
if (
    pinned_config.get("revision") != manifest.get("revision")
    or pinned_config.get("tokenizer_fingerprint") != tokenizer_fingerprint
):
    raise SystemExit("Pinned Qwen config differs from the attested model/tokenizer")
lengths = []
for row in rows:
    rendered = apply_chat_template(
        tokenizer,
        [
            {"role": "system", "content": row["system_prompt"]},
            {"role": "user", "content": row["prompt"]},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    lengths.append(
        (
            len(tokenizer.encode(rendered, add_special_tokens=False)),
            str(row["sample_id"]),
        )
    )
lengths.sort()
max_model_len, max_new_tokens = 32768, 176
maximum, maximum_sample_id = lengths[-1]
if maximum + max_new_tokens > max_model_len:
    raise SystemExit("Post-hoc exact prompt audit exceeds the frozen context budget")
data_sha = sha256(data)
selection_sha = hashlib.sha256(
    json.dumps(
        [row["sample_id"] for row in rows],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
prompt_audit = {
    "schema_version": "eval-prompt-length-audit-v1",
    "status": "validated",
    "audit_timing": "post-hoc CPU re-render; the evaluation process predated this gate",
    "data_sha256": data_sha,
    "model_revision": manifest.get("revision"),
    "model_manifest_sha256": manifest_sha,
    "model_integrity_attestation_sha256": sha256(attestation_path),
    "tokenizer_fingerprint": tokenizer_fingerprint,
    "pinned_model_config_sha256": sha256(pinned_config_path),
    "chat_protocol": selected_protocol_for_tokenizer(tokenizer),
    "chat_protocol_audit_sha256": sha256(chat_audit_path) if chat_audit_path.is_file() else None,
    "selected_samples": len(rows),
    "min_prompt_tokens": lengths[0][0],
    "median_prompt_tokens": lengths[len(lengths) // 2][0],
    "p95_prompt_tokens": lengths[int(0.95 * (len(lengths) - 1))][0],
    "max_prompt_tokens": maximum,
    "max_prompt_sample_id": maximum_sample_id,
    "max_new_tokens": max_new_tokens,
    "max_model_len": max_model_len,
    "unused_context_tokens_at_maximum": max_model_len - max_new_tokens - maximum,
}
prompt_path = output / "prompt-length-audit.json"
prompt_path.write_text(
    json.dumps(prompt_audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

run_records = {}
for source in sources:
    metadata_path = source.with_suffix(source.suffix + ".run.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("data_sha256") != data_sha:
        raise SystemExit(f"Historical run data hash differs: {metadata_path}")
    if metadata.get("selection_sha256") not in (None, selection_sha):
        raise SystemExit(f"Historical run selection differs: {metadata_path}")
    if Path(metadata.get("model", "")).resolve() != model.resolve():
        raise SystemExit(f"Historical run model path differs: {metadata_path}")
    run_records[source.name] = {
        "result_sha256": sha256(source),
        "run_metadata_sha256": sha256(metadata_path),
        "adapter": metadata.get("adapter"),
        "adapter_sha256_at_generation": metadata.get("adapter_sha256"),
        "selected_samples": metadata.get("selected_samples", 4200),
    }
lineage = {
    "schema_version": "posthoc-evaluation-lineage-attestation-v1",
    "status": "validated",
    "scope": "Qwen exploratory seed-1 formal evaluation started before strict lineage gate",
    "limitation": (
        "This binds saved historical run identities and results to a freshly stat-validated "
        "pinned model and exact CPU prompt re-render; it is not represented as a pre-run gate."
    ),
    "data_sha256": data_sha,
    "selection_sha256": selection_sha,
    "model_revision": manifest.get("revision"),
    "model_manifest_sha256": manifest_sha,
    "model_integrity_attestation_sha256": sha256(attestation_path),
    "tokenizer_fingerprint": tokenizer_fingerprint,
    "pinned_model_config_sha256": sha256(pinned_config_path),
    "prompt_length_audit_sha256": sha256(prompt_path),
    "runs": run_records,
}
(output / "generation-lineage-attestation.json").write_text(
    json.dumps(lineage, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(
    f"Validated post-hoc Qwen lineage and prompt budget: "
    f"{maximum}+{max_new_tokens}<={max_model_len}"
)
PY

for index in "${!canonical[@]}"; do
  name="${canonical[$index]}"
  source="${sources[$index]}"
  cp "$source" "$output_dir/$name.jsonl"
  cp "$source.run.json" "$output_dir/source_run_metadata/$name.jsonl.run.json"
done

sha256sum "${sources[@]}" > "$output_dir/reproducibility/source-result-sha256.txt"
for source in "${sources[@]}"; do
  sha256sum "$source.run.json" >> "$output_dir/reproducibility/source-result-sha256.txt"
done
python3 -m pip freeze > "$output_dir/reproducibility/pip-freeze.txt"
uname -a > "$output_dir/reproducibility/uname.txt"

result_args=()
analysis_args=()
for name in "${canonical[@]}"; do
  result_args+=("$output_dir/$name.jsonl")
  analysis_args+=(--run "$name=$output_dir/$name.jsonl")
done
python3 "$root_dir/scripts/aggregate_results.py" "${result_args[@]}" \
  --output "$output_dir/summary.json"
python3 "$root_dir/scripts/analyze_factorial_results.py" \
  "${analysis_args[@]}" \
  --output-dir "$output_dir/analysis" \
  --bootstrap-replicates 5000 \
  --seed 20260828
python3 "$root_dir/scripts/plot_factorial_results.py" \
  --analysis "$output_dir/analysis/factorial_analysis.json" \
  --output-dir "$output_dir/figures"
python3 "$root_dir/scripts/analyze_failure_cases.py" \
  "${analysis_args[@]}" \
  --output-dir "$output_dir/failure_cases" \
  --max-examples 5

python3 - "$output_dir" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
runs = (
    "base", "independent_answer", "independent_evidence_id",
    "independent_evidence", "paired_answer", "paired_evidence_id", "paired_evidence",
)
hashes = {
    run: hashlib.sha256((output / f"{run}.jsonl").read_bytes()).hexdigest()
    for run in runs
}
payload = {
    "schema_version": "formal-s100-seed1-frozen-v1",
    "status": "validated",
    "model_family": "Qwen2.5-7B-Instruct",
    "training_seed": 20260825,
    "pilot_status": "exploratory; fixed-100 stopping rule was chosen after this seed",
    "rows_per_run": 4200,
    "runs": list(runs),
    "result_sha256": hashes,
    "bootstrap_replicates": 5000,
    "analysis": "analysis/factorial_analysis.json",
    "figures": "figures/figures.metadata.json",
    "failure_case_catalog": "failure_cases/failure_case_catalog.manifest.json",
    "prompt_length_audit": {
        "path": "reproducibility/prompt-length-audit.json",
        "sha256": hashlib.sha256(
            (output / "reproducibility/prompt-length-audit.json").read_bytes()
        ).hexdigest(),
    },
    "generation_lineage_attestation": {
        "path": "reproducibility/generation-lineage-attestation.json",
        "sha256": hashlib.sha256(
            (output / "reproducibility/generation-lineage-attestation.json").read_bytes()
        ).hexdigest(),
    },
}
(output / "completion.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

output_rel="$(realpath -m --relative-to="$root_dir" "$output_dir")"
tar -C "$root_dir" -czf "$artifact_path" "$output_rel"
sha256sum "$artifact_path" > "$artifact_path.sha256"
sha256sum -c "$artifact_path.sha256"
printf 'validated exit_code=0 artifact=%s finished_at=%s\n' "$artifact_path" "$(date -u +%FT%TZ)" > "$status_path"
touch "$output_dir/RESULTS_READY_FOR_AGENT_REVIEW"
echo "Qwen seed-1 formal factorial results frozen, analyzed, plotted, and packaged."
echo "Instance intentionally remains running; no shutdown command was issued."
trap - EXIT
