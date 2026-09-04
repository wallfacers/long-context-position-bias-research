#!/usr/bin/env bash
set -Eeuo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir=""
model_config=""
formal_root=""
nolima_root=""
seeds_csv="20260825,20260826,20260827"
status_path="/root/autodl-tmp/model-family-data-prep.status"
force=0
formal_long_tokens=32512
evaluation_max_model_len=32768
evaluation_max_new_tokens=176
variants=(
  independent_answer independent_evidence_id independent_evidence
  paired_answer paired_evidence_id paired_evidence
)

usage() {
  echo "Usage: $0 --model LOCAL_DIR --model-config JSON --formal-root DIR --nolima-root DIR [--seeds CSV] [--status FILE] [--force]"
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) model_dir="$2"; shift 2 ;;
    --model-config) model_config="$2"; shift 2 ;;
    --formal-root) formal_root="$2"; shift 2 ;;
    --nolima-root) nolima_root="$2"; shift 2 ;;
    --seeds) seeds_csv="$2"; shift 2 ;;
    --status) status_path="$2"; shift 2 ;;
    --force) force=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -d "$model_dir" || ! -s "$model_dir/config.json" || ! -s "$model_config" ]]; then
  echo "Complete local model and pinned model config are required" >&2
  exit 2
fi
if [[ -z "$formal_root" || -z "$nolima_root" ]]; then
  usage >&2
  exit 2
fi
for target in "$formal_root" "$nolima_root"; do
  case "$(realpath -m "$target")/" in
    "$(realpath "$root_dir")/"*) ;;
    *) echo "Data roots must be inside the project root: $target" >&2; exit 2 ;;
  esac
done

revision="$(python3 - "$model_config" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload["revision"])
PY
)"
if [[ ! "$revision" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Model config must pin a 40-character revision" >&2
  exit 2
fi

mkdir -p "$(dirname "$status_path")" "$formal_root/eval" "$nolima_root"
printf 'running started_at=%s revision=%s\n' "$(date -u +%FT%TZ)" "$revision" > "$status_path"
on_exit() {
  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    printf 'failed exit_code=%s finished_at=%s\n' "$rc" "$(date -u +%FT%TZ)" > "$status_path"
    echo "Model-family data preparation failed; completed immutable inputs were preserved."
  fi
}
trap on_exit EXIT

cd "$root_dir"
# Fail before the 29GB-scale full-hash pass if the model's native chat
# template drops roles or cannot provide a prefix-safe completion mask.
python3 scripts/audit_chat_protocol.py \
  --tokenizer "$model_dir" \
  --revision "$revision" \
  --local-files-only \
  --output "$model_dir/chat_protocol_audit.json"
python3 - "$model_dir/chat_protocol_audit.json" "$model_config" <<'PY'
import hashlib
import json
import sys

audit_path, config_path = sys.argv[1:]
audit = json.load(open(audit_path, encoding="utf-8"))
config = json.load(open(config_path, encoding="utf-8"))
actual_sha = hashlib.sha256(open(audit_path, "rb").read()).hexdigest()
if audit.get("status") != "passed":
    raise SystemExit("Chat-protocol audit has not passed")
if audit.get("selected_protocol") != config.get("chat_protocol"):
    raise SystemExit("Selected chat protocol differs from pinned model config")
if actual_sha != config.get("chat_protocol_audit_sha256"):
    raise SystemExit("Chat-protocol audit hash differs from pinned model config")
print(f"Validated pinned chat protocol {audit['selected_protocol']} sha256={actual_sha}")
PY
mkdir -p "$formal_root/reproducibility"
cp "$model_dir/chat_protocol_audit.json" \
  "$formal_root/reproducibility/chat_protocol_audit.json"

if [[ ! -s "$model_dir/model_manifest.json" ]]; then
  python3 scripts/stage_model.py \
    --config "$model_config" \
    --output "$model_dir" \
    --manifest-only
fi
python3 - "$model_dir/model_manifest.json" "$model_config" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
config = json.load(open(sys.argv[2], encoding="utf-8"))
if manifest.get("revision") != config.get("revision"):
    raise SystemExit("Model manifest revision differs from pinned config")
print(f"Validated model revision {manifest['revision']}")
PY

overwrite=()
if [[ "$force" -eq 1 ]]; then
  overwrite+=(--overwrite)
fi
tokenizer_args=(
  --tokenizer "$model_dir"
  --tokenizer-revision "$revision"
  --local-files-only
)

eval_test="$formal_root/eval/test.jsonl"
if [[ ! -s "$eval_test" || "$force" -eq 1 ]]; then
  python3 scripts/generate_synthetic_data.py \
    --output "$eval_test" \
    --split test \
    --groups-per-condition 50 \
    --tasks kv,two_hop \
    --fillers neutral,same_format,answer_bearing \
    --lengths "8K,$formal_long_tokens" \
    --positions 0,10,25,50,75,90,100 \
    --seed 20260825 \
    --words-per-document 48 \
    "${tokenizer_args[@]}" \
    "${overwrite[@]}"
fi
python3 scripts/validate_dataset.py "$eval_test"
python3 - "$eval_test" "$model_dir" "$formal_long_tokens" \
  "$evaluation_max_model_len" "$evaluation_max_new_tokens" <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))
from transformers import AutoTokenizer
from position_bias_research.chat_protocol import apply_chat_template

data, model = Path(sys.argv[1]), Path(sys.argv[2])
formal_long_tokens = int(sys.argv[3])
max_model_len = int(sys.argv[4])
max_new_tokens = int(sys.argv[5])
rows = [json.loads(line) for line in data.open(encoding="utf-8") if line.strip()]
if len(rows) != 4200 or len({row["group_id"] for row in rows}) != 600:
    raise SystemExit("Model-family formal test is incomplete")
if {int(row["target_tokens"]) for row in rows} != {8192, formal_long_tokens}:
    raise SystemExit("Model-family formal test lengths differ from the frozen safe slices")
tokenizer = AutoTokenizer.from_pretrained(
    str(model), local_files_only=True, trust_remote_code=False
)
lengths = []
maximum_sample_id = None
current_maximum = -1
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
    length = len(tokenizer.encode(rendered, add_special_tokens=False))
    lengths.append(length)
    if length > current_maximum:
        current_maximum = length
        maximum_sample_id = row["sample_id"]
ordered = sorted(lengths)
maximum = ordered[-1]
if maximum + max_new_tokens > max_model_len:
    raise SystemExit(
        f"Longest rendered prompt is {maximum} tokens; "
        f"{maximum}+{max_new_tokens}>{max_model_len}"
    )
audit = {
    "schema_version": "family-formal-prompt-length-audit-v1",
    "status": "validated",
    "rows": len(rows),
    "groups": len({row["group_id"] for row in rows}),
    "target_token_slices": [8192, formal_long_tokens],
    "length_labels": {"8192": "8K", str(formal_long_tokens): "32K-safe"},
    "min_prompt_tokens": ordered[0],
    "median_prompt_tokens": ordered[len(ordered) // 2],
    "p95_prompt_tokens": ordered[int(0.95 * (len(ordered) - 1))],
    "max_prompt_tokens": maximum,
    "max_prompt_sample_id": maximum_sample_id,
    "max_new_tokens": max_new_tokens,
    "max_model_len": max_model_len,
    "unused_context_tokens_at_maximum": max_model_len - max_new_tokens - maximum,
}
output = data.parent / "prompt_length_audit.json"
temporary = output.with_name(output.name + ".tmp")
temporary.write_text(
    json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
temporary.replace(output)
print(
    f"Validated formal test: rows=4200 groups=600 "
    f"max_prompt_tokens={maximum}+{max_new_tokens}<={max_model_len}"
)
PY

IFS=',' read -r -a seeds <<< "$seeds_csv"
for seed in "${seeds[@]}"; do
  seed="${seed//[[:space:]]/}"
  if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "Invalid seed: $seed" >&2
    exit 2
  fi
  seed_root="$formal_root/seed_$seed"
  bank="$seed_root/raw/train_bank.jsonl"
  mkdir -p "$seed_root/raw" "$seed_root/sft" "$seed_root/tokenized"
  if [[ ! -s "$bank" || "$force" -eq 1 ]]; then
    python3 scripts/generate_matched_training_bank.py \
      --output "$bank" \
      --split train \
      --facts-per-condition 128 \
      --replicas-per-fact 4 \
      --tasks kv,two_hop \
      --fillers neutral \
      --lengths 8K \
      --positions 0,25,50,100 \
      --seed "$seed" \
      --words-per-document 48 \
      "${tokenizer_args[@]}" \
      "${overwrite[@]}"
  fi
  if [[ ! -s "$seed_root/sft/matched-design.json" || "$force" -eq 1 ]]; then
    python3 scripts/prepare_matched_sft_variants.py \
      --input "$bank" \
      --output-dir "$seed_root/sft" \
      --seed "$seed" \
      --max-token-budget-gap 0.002 \
      "${overwrite[@]}"
  fi
  python3 scripts/validate_dataset.py "$bank" "$seed_root"/sft/*.jsonl
  python3 scripts/audit_matched_training_design.py \
    "$seed_root"/sft/*.jsonl \
    --max-token-budget-gap 0.002 \
    --output "$seed_root/sft/matched-audit.json"
  for variant in "${variants[@]}"; do
    tokenized="$seed_root/tokenized/$variant"
    if [[ ! -s "$tokenized/pretokenization.json" || "$force" -eq 1 ]]; then
      tokenize_overwrite=()
      if [[ -e "$tokenized" ]]; then
        if [[ "$force" -eq 1 ]]; then
          tokenize_overwrite+=(--overwrite)
        else
          echo "Incomplete pretokenized directory requires explicit --force: $tokenized" >&2
          exit 2
        fi
      fi
      python3 scripts/pretokenize_sft.py \
        "$seed_root/sft/$variant.jsonl" \
        "$tokenized" \
        --tokenizer "$model_dir" \
        --tokenizer-revision "$revision" \
        --max-length 8320 \
        --local-files-only \
        "${tokenize_overwrite[@]}"
    fi
  done
  python3 scripts/build_data_manifest.py \
    --root "$seed_root" \
    --output "$seed_root/manifest.json" \
    --model-config "$model_config"
done
python3 scripts/build_data_manifest.py \
  --root "$formal_root" \
  --output "$formal_root/manifest.json" \
  --model-config "$model_config"

needle_set="$root_dir/third_party/NoLiMa/data/needlesets/needle_set_hard.json"
haystack_dir="$root_dir/third_party/NoLiMa/data/haystack/rand_shuffle"
source_download_manifest="$root_dir/third_party/NoLiMa/data/frozen-source-download-manifest.json"
if [[ ! -s "$needle_set" || ! -d "$haystack_dir" || ! -s "$source_download_manifest" ]]; then
  echo "Frozen official NoLiMa sources and their download manifest are required; run scripts/fetch_nolima_sources.py" >&2
  exit 2
fi
python3 - "$source_download_manifest" "$needle_set" "$haystack_dir" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path, needle_path, book_dir = map(Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if (
    manifest.get("schema_version") != "nolima-frozen-source-download-v1"
    or manifest.get("status") != "validated"
    or manifest.get("official_repository_revision")
    != "cb14780b249fecf2851127b2101a062c1b2c6430"
    or manifest.get("dataset_revision")
    != "378115b1f136b6ba78f90f78682bc55f70ec3ddd"
):
    raise SystemExit("NoLiMa source-download manifest identity differs from the frozen protocol")

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

if sha256(needle_path) != manifest["needle"]["sha256"]:
    raise SystemExit("Frozen NoLiMa needle SHA-256 differs")
for index, record in sorted(manifest["books"].items()):
    path = book_dir / f"rand_book_{index}.txt"
    if not path.is_file() or sha256(path) != record["combined_sha256"]:
        raise SystemExit(f"Frozen NoLiMa book differs: {path}")
print("Validated pinned NoLiMa repository/dataset identity and final source hashes")
PY
nolima_data="$nolima_root/hard_gate.jsonl"
nolima_manifest="$nolima_root/hard_gate.manifest.json"
nolima_audit="$nolima_root/hard_gate.audit.json"
if [[ ! -s "$nolima_data" || "$force" -eq 1 ]]; then
  python3 scripts/prepare_nolima_ood.py \
    --needle-set "$needle_set" \
    --haystack-dir "$haystack_dir" \
    --output "$nolima_data" \
    --manifest "$nolima_manifest" \
    --audit "$nolima_audit" \
    --lengths 1024,8192,32000 \
    --positions 0,0.1,0.25,0.5,0.75,0.9,1 \
    --tokenizer "$model_dir" \
    --tokenizer-revision "$revision" \
    --local-files-only \
    "${overwrite[@]}"
fi
diagnostic_data="$nolima_root/hard_gate_diagnostics.jsonl"
diagnostic_manifest="$nolima_root/hard_gate_diagnostics.manifest.json"
if [[ ! -s "$diagnostic_data" || "$force" -eq 1 ]]; then
  python3 scripts/prepare_nolima_diagnostics.py \
    --input "$nolima_data" \
    --output "$diagnostic_data" \
    --manifest "$diagnostic_manifest" \
    --tokenizer "$model_dir" \
    --tokenizer-revision "$revision" \
    --local-files-only \
    "${overwrite[@]}"
fi

python3 - "$formal_root" "$nolima_root" "$model_config" "$revision" "$seeds_csv" "$source_download_manifest" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

formal, nolima, model_config, revision = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
seeds = [int(value.strip()) for value in sys.argv[5].split(",") if value.strip()]
source_download_manifest = Path(sys.argv[6])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
paths = {
    "formal_manifest": formal / "manifest.json",
    "formal_test": formal / "eval/test.jsonl",
    "formal_prompt_length_audit": formal / "eval/prompt_length_audit.json",
    "nolima_manifest": nolima / "hard_gate.manifest.json",
    "nolima_data": nolima / "hard_gate.jsonl",
    "nolima_diagnostic_manifest": nolima / "hard_gate_diagnostics.manifest.json",
    "nolima_diagnostics": nolima / "hard_gate_diagnostics.jsonl",
    "nolima_source_download_manifest": source_download_manifest,
    "model_config": model_config,
    "chat_protocol_audit": formal / "reproducibility/chat_protocol_audit.json",
}
for name, path in paths.items():
    if not path.is_file() or not path.stat().st_size:
        raise SystemExit(f"Missing final family-data artifact: {name}={path}")
payload = {
    "schema_version": "model-family-data-completion-v1",
    "status": "validated",
    "revision": revision,
    "training_seeds": seeds,
    "formal_test_rows": 4200,
    "nolima_rows": 1050,
    "nolima_diagnostic_rows": 1350,
    "files": {
        name: {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for name, path in paths.items()
    },
}
(formal / "completion.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print("Validated complete separately tokenized model-family data package")
PY

printf 'validated exit_code=0 formal_root=%s nolima_root=%s finished_at=%s\n' \
  "$formal_root" "$nolima_root" "$(date -u +%FT%TZ)" > "$status_path"
echo "Model-family formal and NoLiMa data are prepared and audited."
echo "Instance intentionally remains running; no shutdown command was issued."
trap - EXIT
