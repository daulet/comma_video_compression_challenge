#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PD="$(cd "${HERE}/../.." && pwd)"
TMP_DIR="${PD}/tmp/neural_inflate"

IN_DIR="${PD}/videos"
VIDEO_NAMES_FILE="${PD}/public_test_video_names.txt"
ARCHIVE_DIR="${HERE}/archive"
JOBS="1"
SKIP_ENCODE="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in-dir|--in_dir)
      IN_DIR="${2%/}"; shift 2 ;;
    --jobs)
      JOBS="$2"; shift 2 ;;
    --video-names-file|--video_names_file)
      VIDEO_NAMES_FILE="$2"; shift 2 ;;
    --skip-encode)
      SKIP_ENCODE="1"; shift ;;
    *)
      echo "Unknown arg: $1" >&2
      echo "Usage: $0 [--in-dir <dir>] [--jobs <n>] [--video-names-file <file>] [--skip-encode]" >&2
      exit 2 ;;
  esac
done

if [[ "$SKIP_ENCODE" == "0" ]]; then
  rm -rf "$ARCHIVE_DIR"
  mkdir -p "$ARCHIVE_DIR"
  mkdir -p "$TMP_DIR"

  export IN_DIR ARCHIVE_DIR PD

  head -n "$(wc -l < "$VIDEO_NAMES_FILE")" "$VIDEO_NAMES_FILE" | xargs -P"$JOBS" -I{} bash -lc '
    rel="$1"
    [[ -z "$rel" ]] && exit 0

    IN="${IN_DIR}/${rel}"
    BASE="${rel%.*}"
    OUT="${ARCHIVE_DIR}/${BASE}.mkv"
    PRE_IN="'"${TMP_DIR}"'/${BASE}.pre.mkv"

    mkdir -p "$(dirname "$OUT")" "$(dirname "$PRE_IN")"
    echo "→ ${IN}  →  ${OUT}"

    # Step 1: ROI preprocess — denoise outside driving corridor
    rm -f "$PRE_IN"
    python "'"${HERE}"'/preprocess.py" \
      --input "$IN" \
      --output "$PRE_IN" \
      --outside-luma-denoise 2.5 \
      --outside-chroma-mode medium \
      --feather-radius 24 \
      --outside-blend 0.50

    # Step 2: Downscale + AV1 encode
    FFMPEG="${PD}/ffmpeg-new"
    [ ! -x "$FFMPEG" ] && FFMPEG="ffmpeg"
    export LD_LIBRARY_PATH="${PD}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    "$FFMPEG" -nostdin -y -hide_banner -loglevel warning \
      -r 20 -fflags +genpts -i "$PRE_IN" \
      -vf "scale=trunc(iw*0.45/2)*2:trunc(ih*0.45/2)*2:flags=lanczos" \
      -pix_fmt yuv420p -c:v libsvtav1 -preset 0 -crf 33 \
      -svtav1-params "film-grain=22:keyint=180:scd=0" \
      -r 20 "$OUT"

    rm -f "$PRE_IN"
  ' _ {}
else
  if [[ ! -d "$ARCHIVE_DIR" ]]; then
    echo "ERROR: $ARCHIVE_DIR does not exist; run without --skip-encode first" >&2
    exit 1
  fi
fi

# Add model artifacts when present.
for model in ren_model.int8.bz2 ren_model.pt.bz2 ren_model.pt; do
  if [[ -f "${HERE}/${model}" ]]; then
    cp -f "${HERE}/${model}" "${ARCHIVE_DIR}/${model}"
    echo "Included model artifact: ${model}"
  fi
done

if [[ -z "$(find "${ARCHIVE_DIR}" -type f -print -quit)" ]]; then
  echo "ERROR: ${ARCHIVE_DIR} has no files to package." >&2
  echo "Run without --skip-encode first, or place model/video artifacts in archive/." >&2
  exit 1
fi

# zip archive
rm -f "${HERE}/archive.zip"
cd "$ARCHIVE_DIR"
if command -v zip &>/dev/null; then
  zip -r "${HERE}/archive.zip" .
else
  python3 -c "
import os, zipfile
with zipfile.ZipFile('${HERE}/archive.zip', 'w', zipfile.ZIP_STORED) as zf:
    for root, _, files in os.walk('.'):
        for f in files:
            p = os.path.join(root, f)
            zf.write(p)
"
fi

echo "Compressed to ${HERE}/archive.zip"
