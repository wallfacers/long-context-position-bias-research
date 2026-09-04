#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_SOCKET="${SSH_SOCKET:-/tmp/position-bias-autodl-ssh.sock}"
REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_WATCH_STATUS="${REMOTE_WATCH_STATUS:-/root/autodl-tmp/position-bias-paper-watch.status}"
REMOTE_ARTIFACT="${REMOTE_ARTIFACT:-/root/autodl-tmp/position-bias-test-full.tar.gz}"
DESTINATION="${DESTINATION:-$ROOT_DIR/artifacts/position-bias-test-full.tar.gz}"
LOCAL_STATUS="${LOCAL_STATUS:-$ROOT_DIR/artifacts/position-bias-test-full.download.status}"
POLL_SECONDS="${POLL_SECONDS:-1800}"
TEMP_ARTIFACT="${DESTINATION}.partial-$$"
TEMP_CHECKSUM="${DESTINATION}.sha256.partial-$$"

cleanup_on_exit() {
  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    rm -f "$TEMP_ARTIFACT" "$TEMP_CHECKSUM"
  fi
}
trap cleanup_on_exit EXIT

if [[ "${ALLOW_LEGACY_WATCHER:-0}" != "1" ]]; then
  echo "Legacy polling is disabled. Current runs use explicit low-frequency agent checks." >&2
  exit 2
fi
if [[ -z "$REMOTE_HOST" ]]; then
  echo "REMOTE_HOST must be supplied explicitly; no machine endpoint is stored in the repository" >&2
  exit 2
fi
if [[ "$POLL_SECONDS" -lt 300 ]]; then
  echo "POLL_SECONDS must be at least 300" >&2
  exit 2
fi
mkdir -p "$(dirname "$DESTINATION")"
echo "waiting started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOCAL_STATUS"

while true; do
  if [[ ! -S "$SSH_SOCKET" ]]; then
    echo "ssh_control_socket_missing observed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      > "$LOCAL_STATUS"
    exit 1
  fi
  remote_status="$({
    ssh -S "$SSH_SOCKET" -p "$REMOTE_PORT" "$REMOTE_HOST" \
      "test -f '$REMOTE_WATCH_STATUS' && cat '$REMOTE_WATCH_STATUS' || true"
  } 2>&1)" || {
    echo "ssh_check_failed observed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      > "$LOCAL_STATUS"
    exit 1
  }
  if [[ "$remote_status" == ready_for_agent_review* ]]; then
    break
  fi
  if [[ "$remote_status" == *failed* ]]; then
    echo "remote_failed detail=$remote_status observed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      > "$LOCAL_STATUS"
    exit 1
  fi
  sleep "$POLL_SECONDS"
done

echo "downloading started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOCAL_STATUS"
scp -q -o "ControlPath=$SSH_SOCKET" -P "$REMOTE_PORT" \
  "$REMOTE_HOST:$REMOTE_ARTIFACT" "$TEMP_ARTIFACT"
scp -q -o "ControlPath=$SSH_SOCKET" -P "$REMOTE_PORT" \
  "$REMOTE_HOST:$REMOTE_ARTIFACT.sha256" "$TEMP_CHECKSUM"

expected_sha="$(awk 'NR == 1 {print $1}' "$TEMP_CHECKSUM")"
actual_sha="$(sha256sum "$TEMP_ARTIFACT" | awk '{print $1}')"
if [[ ! "$expected_sha" =~ ^[0-9a-f]{64}$ ]] || [[ "$actual_sha" != "$expected_sha" ]]; then
  echo "checksum_failed expected=$expected_sha actual=$actual_sha" > "$LOCAL_STATUS"
  exit 1
fi
mv "$TEMP_ARTIFACT" "$DESTINATION"
printf '%s  %s\n' "$actual_sha" "$DESTINATION" > "$DESTINATION.sha256"
mv "$TEMP_CHECKSUM" "$DESTINATION.remote.sha256"
echo "verified sha256=$actual_sha finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$LOCAL_STATUS"
echo "Verified paper artifact downloaded to $DESTINATION"
echo "Instance remains available for primary-agent review."
trap - EXIT
