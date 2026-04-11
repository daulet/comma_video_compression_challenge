#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="/Users/dzhanguzin/dev/personal/comma_video_compression_challenge/tmp/hou2-neural-inflate"
REMOTE_COPY="/Users/dzhanguzin/dev/personal/comma_video_compression_challenge/tmp/hou2-neural-inflate-remote-final"
STATE_TMP="/tmp/watch_collect_state.$$"
CMD_TMP="/tmp/watch_collect_cmd.$$"
mkdir -p "$OUT_DIR"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

# Run a command with a hard timeout by supervising and killing the child.
run_with_timeout() {
  local timeout_s="$1"
  shift
  : >"$CMD_TMP"
  ( "$@" ) >"$CMD_TMP" 2>/dev/null &
  local cpid=$!
  local t=0
  while kill -0 "$cpid" 2>/dev/null; do
    sleep 1
    t=$((t + 1))
    if (( t >= timeout_s )); then
      kill "$cpid" 2>/dev/null || true
      wait "$cpid" 2>/dev/null || true
      return 124
    fi
  done

  if wait "$cpid"; then
    return 0
  fi
  return 1
}

probe_state() {
  if run_with_timeout 12 kubectl -n default exec daulet-test -- /bin/bash -lc 'pid=$(cat /tmp/run_neural_inflate_pipeline.pid 2>/dev/null || echo ""); if [[ -z "$pid" ]]; then echo NO_PID; elif test -d /proc/$pid; then echo RUNNING:$pid; else echo EXITED:$pid; fi'; then
    cat "$CMD_TMP"
    return 0
  fi
  return 1
}

copy_submission_dir() {
  rm -rf "$REMOTE_COPY"
  run_with_timeout 45 kubectl -n default cp daulet-test:/workspace/comma_video_compression_challenge/submissions/neural_inflate "$REMOTE_COPY"
}

copy_log_tail() {
  run_with_timeout 15 kubectl -n default exec daulet-test -- /bin/bash -lc 'tail -n 200 /tmp/run_neural_inflate_pipeline.log' > "$OUT_DIR/run_neural_inflate_pipeline.log.tail"
}

# Wait until detached run exits, tolerating transient failures.
while true; do
  state=""
  for _ in {1..5}; do
    if state="$(probe_state)"; then
      break
    fi
    sleep 3
  done

  if [[ -z "$state" ]]; then
    log "probe failed (transient), retrying"
    sleep 10
    continue
  fi

  state="$(echo "$state" | tr -d '\r' | head -n 1)"
  log "state=$state"
  if [[ "$state" == EXITED:* || "$state" == NO_PID ]]; then
    break
  fi

  sleep 30
done

# Pull full submission dir via kubectl cp with retries.
for i in {1..30}; do
  if copy_submission_dir; then
    log "kubectl cp succeeded"
    break
  fi
  log "kubectl cp retry $i"
  sleep 5
  if [[ $i -eq 30 ]]; then
    log "kubectl cp failed after retries"
    exit 2
  fi
done

# Copy known artifacts when present.
for f in report.txt archive.zip ren_model.pt ren_model.int8.bz2 ren_model.pt.bz2; do
  if [[ -f "$REMOTE_COPY/$f" ]]; then
    cp -f "$REMOTE_COPY/$f" "$OUT_DIR/$f"
    log "copied $f"
  else
    log "missing $f"
  fi
done

for i in {1..10}; do
  if copy_log_tail; then
    log "saved pipeline log tail"
    break
  fi
  sleep 3
done

rm -f "$STATE_TMP" "$CMD_TMP"
log "done"
