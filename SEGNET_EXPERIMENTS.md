# SegNet Experiments

This note captures the helper scripts used for the SegNet-focused exploration and the exact commands for the two main reproduction targets:

- best `100 * segnet_dist + 25 * rate` result using per-frame image storage
- best video-coded SegNet experiment from the overlay branch

These are exploratory helpers. They are not part of the official `compress.py` / `eval.py` experiment loop unless explicitly ported there.

## Scripts

- `generate_segnet_palette_candidate.py`
  Generate a raw candidate video from SegNet low-res content (`palette`, `overlay`, or `orig_low`) and evaluate its distortion directly.
- `evaluate_segnet_overlay_codec.py`
  Generate low-res overlay frames, encode them with a real video codec, decode/inflate them, and evaluate distortion.
- `optimize_segnet_visualization.py`
  Train tiny renderers from SegNet outputs back to RGB-like low-res frames and measure SegNet agreement.
- `optimize_segnet_latents.py`
  Directly optimize low-res or full-res latent RGB frames against SegNet argmax.
- `sweep_segnet_downscale_images.py`
  Sweep source mode, scale, image codec, and quality for the partial objective `100 * segnet_dist + 25 * rate`.

## Best Partial Result

Goal:

`100 * segnet_dist + 25 * rate`

Best full-clip result found in this branch:

- `source_mode=overlay`
- `alpha=0.99`
- downscale SegNet-preprocessed `512x384` frame to `256x192`
- store each scored odd frame as lossy `webp`
- `quality=75`
- restore with `lanczos`
- inflate with `bilinear`

Reproduce:

```bash
./.venv/bin/python sweep_segnet_downscale_images.py \
  --device cpu \
  --source-modes overlay \
  --alphas 0.99 \
  --scales 0.5 \
  --codecs webp \
  --qualities 75 \
  --down-resamples lanczos \
  --restore-resamples lanczos \
  --inflate-resamples bilinear \
  --unsharp-percents 0 \
  --report artifacts/repro_segnet_seg_rate_best_webp.json
```

Expected result:

- `segnet_dist = 0.01067296`
- `rate = 0.02392383`
- `seg_rate_score = 1.66539217`

## Best Video-Coded Overlay Result

There are two reasonable targets from the video-coded overlay branch.

### Best Partial `segnet + rate` Among Video-Coded Smokes

This one had the best `100 * segnet_dist + 25 * rate` value among the saved video-coded smoke tests, but PoseNet was poor.

Reproduce:

```bash
./.venv/bin/python evaluate_segnet_overlay_codec.py \
  --device cpu \
  --output-dir artifacts/repro_segnet_overlay_codec_best_partial_video \
  --render-mode overlay \
  --alpha 0.995 \
  --codec libsvtav1 \
  --crf 24 \
  --preset 0 \
  --gop 180 \
  --pix-fmt yuv420p \
  --ffmpeg-threads 1 \
  --inflate-mode nearest \
  --prefilter 'hqdn3d=1.5:0:0:0' \
  --max-frames 128
```

Expected partial result:

- `segnet_dist = 0.00319036`
- `rate = 0.00439416`
- `100 * segnet_dist + 25 * rate = 0.428890`

### Best Full Score Among Video-Coded Smokes

This one had the best overall saved smoke-test score from the overlay-video branch.

Reproduce:

```bash
./.venv/bin/python evaluate_segnet_overlay_codec.py \
  --device cpu \
  --output-dir artifacts/repro_segnet_overlay_codec_best_full_video \
  --render-mode overlay \
  --alpha 0.995 \
  --codec libx265 \
  --crf 18 \
  --preset slow \
  --gop 240 \
  --pix-fmt yuv444p \
  --ffmpeg-threads 1 \
  --inflate-mode nearest \
  --max-frames 128
```

Expected result:

- `score = 1.43978739`
- `segnet_dist = 0.00228802`
- `rate = 0.02493881`

## Notes

- The `palette` branch was not competitive once measured on the `segnet + rate` objective.
- The useful SegNet-rate pocket was around `0.5x` semantic resolution.
- The helper reports written under `artifacts/` are intentionally not committed.
