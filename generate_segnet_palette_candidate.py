#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F

from frame_utils import camera_size
from model_diagnostics import count_frames_for_path, iter_rgb_frames
from modules import DistortionNet, posenet_sd_path, segnet_sd_path

HERE = Path(__file__).resolve().parent

PALETTES = {
  "current": np.array([
    [27, 32, 46],
    [230, 84, 59],
    [243, 190, 55],
    [70, 164, 255],
    [63, 196, 142],
  ], dtype=np.uint8),
  "cmykw": np.array([
    [0, 0, 0],
    [0, 255, 255],
    [255, 0, 255],
    [255, 255, 0],
    [255, 255, 255],
  ], dtype=np.uint8),
  "search9": np.array([
    [56, 18, 149],
    [20, 32, 235],
    [185, 76, 241],
    [214, 38, 123],
    [207, 58, 53],
  ], dtype=np.uint8),
}


def _pick_device(device_arg: str | None) -> torch.device:
  if device_arg is not None:
    return torch.device(device_arg)
  if torch.cuda.is_available():
    return torch.device("cuda", 0)
  if torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


def _open_preview(output_path: Path, fps: int) -> tuple[av.container.OutputContainer, av.video.stream.VideoStream]:
  output_path.parent.mkdir(parents=True, exist_ok=True)
  container = av.open(str(output_path), mode="w")
  stream = container.add_stream("libx264", rate=fps)
  stream.width = camera_size[0]
  stream.height = camera_size[1]
  stream.pix_fmt = "yuv420p"
  stream.options = {"crf": "18", "preset": "medium"}
  return container, stream


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Generate a SegNet-argmax palette candidate and evaluate its distortion.")
  parser.add_argument("--input", type=Path, default=HERE / "videos" / "0.mkv")
  parser.add_argument("--output-dir", type=Path, default=HERE / "artifacts" / "segnet_palette_candidate")
  parser.add_argument("--palette", choices=sorted(PALETTES), default="search9")
  parser.add_argument("--render-mode", choices=["palette", "overlay", "orig_low"], default="palette")
  parser.add_argument("--alpha", type=float, default=0.99, help="Overlay weight on the original low-res frame when render-mode=overlay.")
  parser.add_argument("--upsample-mode", choices=["nearest", "bilinear"], default="nearest")
  parser.add_argument("--device", type=str, default="cpu", help="cpu|cuda|mps")
  parser.add_argument("--fps", type=int, default=20)
  parser.add_argument("--seg-batch-size", type=int, default=16)
  parser.add_argument("--eval-batch-size", type=int, default=16)
  parser.add_argument("--max-frames", type=int, default=None)
  parser.add_argument("--write-preview", action="store_true")
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  device = _pick_device(args.device)
  palette = PALETTES[args.palette]
  if not (0.0 <= args.alpha <= 1.0):
    raise ValueError(f"--alpha must be in [0, 1], got {args.alpha}")

  output_dir = args.output_dir
  output_dir.mkdir(parents=True, exist_ok=True)
  raw_path = output_dir / "0.raw"
  report_path = output_dir / "report.json"
  preview_path = output_dir / "preview.mp4"

  n_frames = count_frames_for_path(args.input)
  if args.max_frames is not None:
    n_frames = min(n_frames, args.max_frames)
  if n_frames < 2:
    raise ValueError(f"Need at least 2 frames, found {n_frames}")

  print(f"Loading DistortionNet on {device}", flush=True)
  distortion_net = DistortionNet().eval().to(device)
  distortion_net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)

  preview = None
  preview_stream = None
  if args.write_preview:
    preview, preview_stream = _open_preview(preview_path, fps=args.fps)

  frame_iter = iter_rgb_frames(args.input, max_frames=n_frames)
  pending_orig: list[torch.Tensor] = []
  pending_cmp: list[torch.Tensor] = []
  batch_orig: list[torch.Tensor] = []
  batch_cmp: list[torch.Tensor] = []

  posenet_sum = 0.0
  segnet_sum = 0.0
  n_samples = 0
  processed_frames = 0

  def flush_eval() -> None:
    nonlocal posenet_sum, segnet_sum, n_samples, batch_orig, batch_cmp
    if not batch_orig:
      return
    gt = torch.stack(batch_orig, dim=0).to(device)
    cmp = torch.stack(batch_cmp, dim=0).to(device)
    with torch.inference_mode():
      posenet_dist, segnet_dist = distortion_net.compute_distortion(gt, cmp)
    posenet_sum += float(posenet_dist.sum().item())
    segnet_sum += float(segnet_dist.sum().item())
    n_samples += gt.shape[0]
    batch_orig = []
    batch_cmp = []

  with raw_path.open("wb") as raw_f:
    try:
      while True:
        frames = []
        try:
          while len(frames) < args.seg_batch_size:
            frames.append(next(frame_iter))
        except StopIteration:
          pass
        if not frames:
          break

        x = torch.stack([frame.permute(2, 0, 1) for frame in frames], dim=0).unsqueeze(1).float().to(device)
        with torch.inference_mode():
          orig_low = distortion_net.segnet.preprocess_input(x)
          seg_logits = distortion_net.segnet(orig_low)
        seg_classes = seg_logits.argmax(dim=1).cpu().numpy().astype(np.uint8)

        seg_rgb = palette[seg_classes]
        seg_rgb_t = torch.from_numpy(seg_rgb).permute(0, 3, 1, 2).float()
        if args.render_mode == "palette":
          low_rgb = seg_rgb_t.to(device)
        elif args.render_mode == "overlay":
          low_rgb = args.alpha * orig_low + (1.0 - args.alpha) * seg_rgb_t.to(device)
        elif args.render_mode == "orig_low":
          low_rgb = orig_low
        else:
          raise ValueError(f"unsupported render mode: {args.render_mode}")

        if args.upsample_mode == "nearest":
          up = F.interpolate(low_rgb, size=(camera_size[1], camera_size[0]), mode="nearest")
        else:
          up = F.interpolate(low_rgb, size=(camera_size[1], camera_size[0]), mode="bilinear", align_corners=False)
        candidate_frames = up.permute(0, 2, 3, 1).to(torch.uint8).cpu()

        for orig_frame, cmp_frame in zip(frames, candidate_frames, strict=True):
          raw_f.write(cmp_frame.contiguous().numpy().tobytes())

          if preview is not None and preview_stream is not None:
            video_frame = av.VideoFrame.from_ndarray(cmp_frame.numpy(), format="rgb24")
            for packet in preview_stream.encode(video_frame):
              preview.mux(packet)

          pending_orig.append(orig_frame)
          pending_cmp.append(cmp_frame)
          if len(pending_orig) == 2:
            batch_orig.append(torch.stack(pending_orig, dim=0))
            batch_cmp.append(torch.stack(pending_cmp, dim=0))
            pending_orig = []
            pending_cmp = []
          if len(batch_orig) >= args.eval_batch_size:
            flush_eval()

          processed_frames += 1
          if processed_frames % 64 == 0 or processed_frames == n_frames:
            print(f"  processed {processed_frames}/{n_frames} frames", flush=True)
    finally:
      flush_eval()
      if preview is not None and preview_stream is not None:
        for packet in preview_stream.encode():
          preview.mux(packet)
        preview.close()

  posenet_dist = posenet_sum / max(n_samples, 1)
  segnet_dist = segnet_sum / max(n_samples, 1)
  report = {
    "input": str(args.input),
    "palette_name": args.palette,
    "palette_rgb": palette.tolist(),
    "render_mode": args.render_mode,
    "alpha": args.alpha,
    "upsample_mode": args.upsample_mode,
    "raw_path": str(raw_path),
    "preview_path": str(preview_path) if args.write_preview else None,
    "n_frames": processed_frames,
    "n_samples": n_samples,
    "posenet_dist": posenet_dist,
    "segnet_dist": segnet_dist,
    "distortion_only_score": 100 * segnet_dist + math.sqrt(10 * posenet_dist),
    "raw_bytes": raw_path.stat().st_size,
  }
  report_path.write_text(json.dumps(report, indent=2) + "\n")

  print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
  main()
