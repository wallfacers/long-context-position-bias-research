#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FULL_STATUS_PATH="${FULL_STATUS_PATH:-/root/autodl-tmp/position-bias-test-full.status}"
WATCH_STATUS_PATH="${WATCH_STATUS_PATH:-/root/autodl-tmp/position-bias-paper-watch.status}"
POLL_SECONDS="${POLL_SECONDS:-1800}"

if [[ "${ALLOW_LEGACY_WATCHER:-0}" != "1" ]]; then
  echo "Legacy machine-side polling is disabled; use explicit low-frequency agent checks." >&2
  exit 2
fi
if [[ "$POLL_SECONDS" -lt 300 ]]; then
  echo "POLL_SECONDS must be at least 300" >&2
  exit 2
fi

echo "waiting started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$WATCH_STATUS_PATH"
while [[ ! -f "$FULL_STATUS_PATH" ]]; do
  sleep "$POLL_SECONDS"
done

if ! grep -q '^validated exit_code=0 ' "$FULL_STATUS_PATH"; then
  echo "full_evaluation_failed observed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "$WATCH_STATUS_PATH"
  exit 1
fi

echo "postprocessing started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$WATCH_STATUS_PATH"
if bash "$ROOT_DIR/scripts/run_paper_postprocess.sh"; then
  echo "ready_for_agent_review finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "$WATCH_STATUS_PATH"
else
  rc=$?
  echo "postprocessing_failed exit_code=$rc finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "$WATCH_STATUS_PATH"
  exit "$rc"
fi

echo "Paper artifacts are ready for primary-agent validation."
echo "Instance intentionally remains running."
