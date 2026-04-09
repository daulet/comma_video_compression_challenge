# neural_inflate hou2 runbook

Last updated: 2026-04-09

## Objective

Resume and iterate on a `neural_inflate`-style submission using GPU training on `hou2-prod1` (`daulet-test` pod), with support for training on additional dataset videos.

## Current baseline context

- Leaderboard reference (checked 2026-04-09): `neural_inflate` is top (`1.89`).
- Strategy:
1. ROI preprocessing + aggressive AV1 downscale encode.
2. Lanczos inflate.
3. Train a tiny REN residual model with PoseNet/SegNet-aware loss.
4. Ship compressed REN weights in archive.

## Files added/updated for this workflow

- [`submissions/neural_inflate/train_ren.py`](/Users/dzhanguzin/dev/personal/comma_video_compression_challenge/submissions/neural_inflate/train_ren.py)
  - Reworked to support multi-video training via `--gt-dir`, `--compressed-dir`, `--video-names-file`.
  - Added frame/video caps and split controls (`--max-videos`, `--max-frames-per-video`, `--val-ratio`, etc.).
- [`submissions/neural_inflate/pack_ren.py`](/Users/dzhanguzin/dev/personal/comma_video_compression_challenge/submissions/neural_inflate/pack_ren.py)
  - Packs `ren_model.pt` into:
    - `ren_model.int8.bz2` (custom format consumed by `inflate.py`)
    - `ren_model.pt.bz2` (float16 backup)
- [`submissions/neural_inflate/compress.sh`](/Users/dzhanguzin/dev/personal/comma_video_compression_challenge/submissions/neural_inflate/compress.sh)
  - Added `--skip-encode` mode (repackage model quickly without re-encoding videos).
  - Includes REN model artifacts in `archive/` automatically when present.
  - Clear failure if nothing exists to package.
- [`scripts/hou2_neural_inflate.sh`](/Users/dzhanguzin/dev/personal/comma_video_compression_challenge/scripts/hou2_neural_inflate.sh)
  - End-to-end remote run: sync -> compress -> train -> pack -> repackage -> eval.
  - Pulls artifacts back to local `tmp/hou2-neural-inflate/`.
- [`submissions/neural_inflate/README.md`](/Users/dzhanguzin/dev/personal/comma_video_compression_challenge/submissions/neural_inflate/README.md)
  - Local and cluster usage notes.

## One-command run (recommended)

From repo root:

```bash
bash scripts/hou2_neural_inflate.sh \
  --context hou2-prod1 \
  --namespace default \
  --pod daulet-test \
  --epochs 80 \
  --batch-size 2 \
  --max-frames-per-video 900
```

Outputs copied locally (when present):

- `tmp/hou2-neural-inflate/report.txt`
- `tmp/hou2-neural-inflate/archive.zip`
- `tmp/hou2-neural-inflate/ren_model.pt`
- `tmp/hou2-neural-inflate/ren_model.int8.bz2`
- `tmp/hou2-neural-inflate/ren_model.pt.bz2`

## Resume checklist

1. Verify cluster context:

```bash
kubectl config current-context
```

2. Verify pod:

```bash
kubectl -n default get pod daulet-test -o wide
kubectl -n default exec daulet-test -- nvidia-smi
```

3. Re-run the pipeline command above (adjust hyperparams only).
4. Compare new `report.txt` against previous runs.

## Using extra videos for training

If additional dataset videos are available in pod/local repo:

- Put videos under a directory (default: `videos/`).
- Provide a names file (same format as `public_test_video_names.txt`, one relative path per line).

Run with overrides:

```bash
bash scripts/hou2_neural_inflate.sh \
  --gt-dir videos_full \
  --video-names-file train_video_names.txt \
  --epochs 100 \
  --batch-size 2 \
  --max-videos 0 \
  --max-frames-per-video 1200
```

## Manual step-by-step (inside pod)

If the wrapper script is not used:

```bash
cd /workspace/comma_video_compression_challenge
uv sync --group cu130
source .venv/bin/activate

bash submissions/neural_inflate/compress.sh \
  --in-dir ./videos \
  --video-names-file ./public_test_video_names.txt

python -m submissions.neural_inflate.train_ren \
  --gt-dir ./videos \
  --compressed-dir ./submissions/neural_inflate/archive \
  --video-names-file ./public_test_video_names.txt \
  --epochs 60 \
  --batch-size 1 \
  --save-path ./submissions/neural_inflate/ren_model.pt

python -m submissions.neural_inflate.pack_ren \
  --input ./submissions/neural_inflate/ren_model.pt \
  --output-int8 ./submissions/neural_inflate/ren_model.int8.bz2 \
  --output-f16 ./submissions/neural_inflate/ren_model.pt.bz2 \
  --verify

bash submissions/neural_inflate/compress.sh --skip-encode

bash evaluate.sh \
  --submission-dir ./submissions/neural_inflate \
  --uncompressed-dir ./videos \
  --video-names-file ./public_test_video_names.txt \
  --device cuda
```

## Known gotchas

- `compress.sh --skip-encode` requires existing files in `submissions/neural_inflate/archive/` (video outputs or model artifacts).
- If `kubectl cp` is flaky, use `kubectl exec ... cat <file> > <local_path>` (the wrapper script already does this).
- If training OOMs:
  - lower `--batch-size`
  - lower `--max-frames-per-video`
  - reduce `--features`

## Suggested next experiments

1. Pretrain REN on multiple videos, then finetune on target video only.
2. Sweep encode params (`CRF`, scale factor) and retrain REN for each point.
3. Sweep `--w-temp` for better temporal stability.
4. Sweep REN width (`--features 24/32/40`) vs archive size impact.
