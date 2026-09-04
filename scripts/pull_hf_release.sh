#!/usr/bin/env bash
set -Eeuo pipefail
# Home runbook: pull the staged HF release from the AutoDL instance, verify,
# then upload to the private HF dataset repo.
#
# Prereqs:
#   1. AutoDL instance booted; `ssh autodl-pbias` works (alias in ~/.ssh/config)
#   2. HF token exported:  export HF_TOKEN=hf_...
# Usage:
#   bash scripts/pull_hf_release.sh          # pull + verify only
#   bash scripts/pull_hf_release.sh --upload # pull + verify + upload
cd "$(dirname "$0")/.."

rel_local="artifacts/hf-release"
rel_remote="autodl-pbias:/root/autodl-tmp/hf-release"

mkdir -p "$rel_local/adapters"
echo "== parallel rsync from AutoDL =="
rsync -a "$rel_remote/adapters/qwen_block96/"     "$rel_local/adapters/qwen_block96/" &
rsync -a "$rel_remote/adapters/mistral_block96/"  "$rel_local/adapters/mistral_block96/" &
rsync -a "$rel_remote/results/"                   "$rel_local/results/" &
rsync -a "$rel_remote/paper/"                     "$rel_local/paper/" &
wait

echo "== verify =="
remote_count=$(ssh autodl-pbias 'find /root/autodl-tmp/hf-release -type f | wc -l')
local_count=$(find "$rel_local" -type f | wc -l)
# local adds README.md; hardlinked staging has no extra files
echo "remote files: $remote_count / local files: $local_count (expect local = remote + 1)"
if (( local_count < remote_count )); then
  echo "Local tree is incomplete; re-run this script." >&2
  exit 1
fi
remote_size=$(ssh autodl-pbias 'du -sb /root/autodl-tmp/hf-release | cut -f1')
local_size=$(du -sb "$rel_local" | cut -f1)
echo "remote bytes: $remote_size / local bytes: $local_size (README adds ~2KB)"
if (( local_size + 100000 < remote_size )); then
  echo "Local tree is smaller than remote; re-run this script." >&2
  exit 1
fi
echo "pull verified: $(du -sh "$rel_local" | cut -f1)"

if [[ "${1:-}" == "--upload" ]]; then
  echo "== upload to HF =="
  if [[ -x .venv/bin/python ]]; then
    PY=.venv/bin/python
  else
    PY=python3
    "$PY" -c "import huggingface_hub" 2>/dev/null || python3 -m pip install --user -q huggingface_hub
  fi
  "$PY" scripts/upload_hf_release.py
else
  echo "Pull done. To upload: export HF_TOKEN=... && bash scripts/pull_hf_release.sh --upload"
fi
