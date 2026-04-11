#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="default"
POD="daulet-test"
REMOTE_REPO="/workspace/comma_video_compression_challenge"
MAIN_PID="81610"
OUT_DIR="$(cd "$(dirname "$0")" && pwd)"

is_alive() {
  kubectl -n "$NAMESPACE" exec "$POD" -- test -d "/proc/${MAIN_PID}" >/dev/null 2>&1
}

# Wait until the remote pipeline shell exits.
while is_alive; do
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] remote pipeline pid ${MAIN_PID} still running"
  sleep 30
done

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] remote pipeline exited, collecting artifacts"

pull() {
  local rel="$1"
  local src="${REMOTE_REPO}/${rel}"
  local dst="${OUT_DIR}/$(basename "$rel")"
  if kubectl -n "$NAMESPACE" exec "$POD" -- test -f "$src" >/dev/null 2>&1; then
    kubectl -n "$NAMESPACE" exec "$POD" -- cat "$src" > "$dst"
    echo "pulled $rel -> $dst"
  else
    echo "missing $rel"
  fi
}

pull "submissions/neural_inflate/report.txt"
pull "submissions/neural_inflate/archive.zip"
pull "submissions/neural_inflate/ren_model.pt"
pull "submissions/neural_inflate/ren_model.int8.bz2"
pull "submissions/neural_inflate/ren_model.pt.bz2"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] collection complete"
