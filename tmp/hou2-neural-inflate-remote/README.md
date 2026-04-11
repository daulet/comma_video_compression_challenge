# neural_inflate

Top leaderboard concept (as of **2026-04-09**):

1. ROI-aware prefiltering + aggressive AV1 downscale encode (`45%`, `CRF 33`, film grain synthesis).
2. Inflate by Lanczos upscale.
3. Learn a tiny residual model (REN) to optimize challenge metrics (PoseNet + SegNet proxies), not pixel fidelity.
4. Package REN weights with strong compression (`int8 + bz2`).

## Local workflow

From repo root:

```bash
# 1) Encode videos into submissions/neural_inflate/archive/*.mkv
bash submissions/neural_inflate/compress.sh

# 2) Train REN (single or multi-video)
python -m submissions.neural_inflate.train_ren \
  --gt-dir ./videos \
  --compressed-dir ./submissions/neural_inflate/archive \
  --video-names-file ./public_test_video_names.txt \
  --epochs 60 \
  --batch-size 1

# 3) Pack checkpoint for inference archive
python -m submissions.neural_inflate.pack_ren \
  --input ./submissions/neural_inflate/ren_model.pt \
  --output-int8 ./submissions/neural_inflate/ren_model.int8.bz2 \
  --output-f16 ./submissions/neural_inflate/ren_model.pt.bz2 \
  --verify

# 4) Rebuild archive.zip without re-encoding video stream
bash submissions/neural_inflate/compress.sh --skip-encode

# 5) Evaluate
bash evaluate.sh --submission-dir ./submissions/neural_inflate --device cuda
```

## hou2-prod1 workflow (daulet-test)

```bash
# Runs sync -> compress -> train -> pack -> repackage -> eval inside pod
bash scripts/hou2_neural_inflate.sh \
  --context hou2-prod1 \
  --namespace default \
  --pod daulet-test \
  --epochs 80 \
  --batch-size 2 \
  --max-frames-per-video 900
```

Artifacts are copied back to `tmp/hou2-neural-inflate/`.

## Experiment ideas

1. More training data: include extra dataset videos in both `--gt-dir` and `--video-names-file`; keep `--max-frames-per-video` to control RAM.
2. Temporal robustness: increase `--w-temp` slightly (`0.005 -> 0.01`) to reduce frame-to-frame correction flicker.
3. Hard-example focus: sample more from turns/complex traffic segments by duplicating those ranges in the video list.
4. Rate-quality sweeps: grid over `CRF` and scale factor (`0.42..0.50`) while retraining REN per setting.
5. Model capacity sweep: REN `--features` (`24, 32, 40`) and pack size impact.
6. Domain transfer: pretrain REN on several videos, then short finetune on target video only for final submission.
