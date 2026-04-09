#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONTEXT="hou2-prod1"
NAMESPACE="default"
POD="daulet-test"
REMOTE_REPO="/workspace/comma_video_compression_challenge"

SYNC="1"
VIDEO_NAMES_FILE="public_test_video_names.txt"
GT_DIR="videos"
EPOCHS="60"
BATCH_SIZE="1"
MAX_VIDEOS="0"
MAX_FRAMES_PER_VIDEO="0"
FRAME_STRIDE="1"
VAL_RATIO="0.1"
MIN_VAL_PAIRS="64"

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --context <name>              kubectl context (default: ${CONTEXT})
  --namespace <ns>              pod namespace (default: ${NAMESPACE})
  --pod <name>                  pod name (default: ${POD})
  --remote-repo <path>          repo path inside pod (default: ${REMOTE_REPO})
  --no-sync                     skip local->pod rsync via tar stream
  --video-names-file <path>     path inside repo (default: ${VIDEO_NAMES_FILE})
  --gt-dir <path>               path inside repo (default: ${GT_DIR})
  --epochs <n>                  REN epochs (default: ${EPOCHS})
  --batch-size <n>              REN batch size (default: ${BATCH_SIZE})
  --max-videos <n>              cap training videos, 0=all (default: ${MAX_VIDEOS})
  --max-frames-per-video <n>    cap frames/video, 0=all (default: ${MAX_FRAMES_PER_VIDEO})
  --frame-stride <n>            keep every n-th frame (default: ${FRAME_STRIDE})
  --val-ratio <f>               validation ratio (default: ${VAL_RATIO})
  --min-val-pairs <n>           minimum val pairs (default: ${MIN_VAL_PAIRS})
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --context) CONTEXT="$2"; shift 2 ;;
    --namespace) NAMESPACE="$2"; shift 2 ;;
    --pod) POD="$2"; shift 2 ;;
    --remote-repo) REMOTE_REPO="$2"; shift 2 ;;
    --no-sync) SYNC="0"; shift ;;
    --video-names-file) VIDEO_NAMES_FILE="$2"; shift 2 ;;
    --gt-dir) GT_DIR="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --max-videos) MAX_VIDEOS="$2"; shift 2 ;;
    --max-frames-per-video) MAX_FRAMES_PER_VIDEO="$2"; shift 2 ;;
    --frame-stride) FRAME_STRIDE="$2"; shift 2 ;;
    --val-ratio) VAL_RATIO="$2"; shift 2 ;;
    --min-val-pairs) MIN_VAL_PAIRS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

current_context="$(kubectl config current-context)"
if [[ "$current_context" != "$CONTEXT" ]]; then
  echo "Switching kubectl context: ${current_context} -> ${CONTEXT}"
  kubectl config use-context "$CONTEXT" >/dev/null
fi

kubectl -n "$NAMESPACE" get pod "$POD" >/dev/null

if [[ "$SYNC" == "1" ]]; then
  echo "Syncing local repo into pod ${NAMESPACE}/${POD}:${REMOTE_REPO} ..."
  tar \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='tmp' \
    --exclude='submissions/*/archive' \
    --exclude='submissions/*/archive.zip' \
    -C "$LOCAL_REPO" -cf - . \
    | kubectl -n "$NAMESPACE" exec -i "$POD" -- /bin/bash -lc "mkdir -p '$REMOTE_REPO' && tar -xf - -C '$REMOTE_REPO'"
fi

REMOTE_VIDEO_NAMES="${REMOTE_REPO}/${VIDEO_NAMES_FILE}"
REMOTE_GT_DIR="${REMOTE_REPO}/${GT_DIR}"

read -r -d '' REMOTE_CMD <<EOF || true
set -euo pipefail
cd "$REMOTE_REPO"

uv sync --group cu130
source .venv/bin/activate

bash submissions/neural_inflate/compress.sh \
  --in-dir "$REMOTE_GT_DIR" \
  --video-names-file "$REMOTE_VIDEO_NAMES"

python -m submissions.neural_inflate.train_ren \
  --gt-dir "$REMOTE_GT_DIR" \
  --compressed-dir "$REMOTE_REPO/submissions/neural_inflate/archive" \
  --video-names-file "$REMOTE_VIDEO_NAMES" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --max-videos "$MAX_VIDEOS" \
  --max-frames-per-video "$MAX_FRAMES_PER_VIDEO" \
  --frame-stride "$FRAME_STRIDE" \
  --val-ratio "$VAL_RATIO" \
  --min-val-pairs "$MIN_VAL_PAIRS" \
  --save-path "$REMOTE_REPO/submissions/neural_inflate/ren_model.pt"

python -m submissions.neural_inflate.pack_ren \
  --input "$REMOTE_REPO/submissions/neural_inflate/ren_model.pt" \
  --output-int8 "$REMOTE_REPO/submissions/neural_inflate/ren_model.int8.bz2" \
  --output-f16 "$REMOTE_REPO/submissions/neural_inflate/ren_model.pt.bz2" \
  --verify

bash submissions/neural_inflate/compress.sh \
  --skip-encode \
  --in-dir "$REMOTE_GT_DIR" \
  --video-names-file "$REMOTE_VIDEO_NAMES"

bash evaluate.sh \
  --submission-dir "$REMOTE_REPO/submissions/neural_inflate" \
  --uncompressed-dir "$REMOTE_GT_DIR" \
  --video-names-file "$REMOTE_VIDEO_NAMES" \
  --device cuda
EOF

echo "Running neural_inflate pipeline in pod ${NAMESPACE}/${POD} ..."
kubectl -n "$NAMESPACE" exec "$POD" -- /bin/bash -lc "$REMOTE_CMD"

ARTIFACT_DIR="$LOCAL_REPO/tmp/hou2-neural-inflate"
mkdir -p "$ARTIFACT_DIR"

pull_artifact() {
  local rel="$1"
  local remote_path="${REMOTE_REPO}/${rel}"
  local dst="$ARTIFACT_DIR/$(basename "$rel")"
  if kubectl -n "$NAMESPACE" exec "$POD" -- test -f "$remote_path" >/dev/null 2>&1; then
    kubectl -n "$NAMESPACE" exec "$POD" -- cat "$remote_path" >"$dst"
    echo "Pulled: $dst"
  fi
}

pull_artifact "submissions/neural_inflate/report.txt"
pull_artifact "submissions/neural_inflate/archive.zip"
pull_artifact "submissions/neural_inflate/ren_model.pt"
pull_artifact "submissions/neural_inflate/ren_model.int8.bz2"
pull_artifact "submissions/neural_inflate/ren_model.pt.bz2"

echo "Done. Artifacts (if present): $ARTIFACT_DIR"
