#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from model_diagnostics import load_models, pick_device
from posenet_saliency import compute_saliency, find_pair, normalize_map

HERE = Path(__file__).resolve().parent

PANEL_W = 256
PANEL_H = 192
HEADER_H = 72
N_COLS = 7
N_ROWS = 2
CANVAS_W = N_COLS * PANEL_W
CANVAS_H = HEADER_H + N_ROWS * PANEL_H

BG = (12, 14, 20)
FG = (236, 239, 244)
MUTED = (155, 161, 176)
GRID = (54, 61, 79)


def _heatmap_rgb(saliency: np.ndarray) -> np.ndarray:
  s = normalize_map(saliency)
  r = (255.0 * np.clip(s * 1.6, 0.0, 1.0)).astype(np.uint8)
  g = (255.0 * np.clip((s - 0.25) * 1.5, 0.0, 1.0)).astype(np.uint8)
  b = (255.0 * np.clip((s - 0.65) * 2.5, 0.0, 1.0)).astype(np.uint8)
  return np.stack([r, g, b], axis=-1)


def _overlay(frame: np.ndarray, saliency: np.ndarray, alpha: float) -> np.ndarray:
  heat = _heatmap_rgb(saliency).astype(np.float32)
  base = frame.astype(np.float32)
  weight = normalize_map(saliency)[..., None] * alpha
  return np.clip(base * (1.0 - weight) + heat * weight, 0.0, 255.0).astype(np.uint8)


def _resize(rgb: np.ndarray) -> np.ndarray:
  return np.array(Image.fromarray(rgb).resize((PANEL_W, PANEL_H), resample=Image.Resampling.BILINEAR))


def _render_sheet(
  prev_frame: torch.Tensor,
  curr_frame: torch.Tensor,
  pose_vec: np.ndarray,
  prev_saliency: np.ndarray,
  curr_saliency: np.ndarray,
  pair_index: int,
  fps: int,
  output_path: Path,
  alpha: float,
) -> None:
  canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
  draw = ImageDraw.Draw(canvas)
  font = ImageFont.load_default()
  small_font = ImageFont.load_default()

  draw.text((18, 16), f"PoseNet gradient saliency, pair {pair_index:04d} -> frames {pair_index:04d},{pair_index+1:04d}", font=font, fill=FG)
  draw.text((18, 36), f"time {(pair_index + 1) / fps:6.2f}s | output values {[round(float(v), 4) for v in pose_vec.tolist()]}", font=small_font, fill=MUTED)

  base_prev = prev_frame.numpy()
  base_curr = curr_frame.numpy()
  prev_overlays = [base_prev] + [_overlay(base_prev, prev_saliency[idx], alpha=alpha) for idx in range(6)]
  curr_overlays = [base_curr] + [_overlay(base_curr, curr_saliency[idx], alpha=alpha) for idx in range(6)]

  titles = ["raw frame", "p0", "p1", "p2", "p3", "p4", "p5"]
  row_titles = ["previous frame", "current frame"]

  for row, images in enumerate([prev_overlays, curr_overlays]):
    y0 = HEADER_H + row * PANEL_H
    draw.text((18, y0 + 10), row_titles[row], font=font, fill=FG)
    for col, image in enumerate(images):
      x0 = col * PANEL_W
      panel = Image.fromarray(_resize(image))
      canvas.paste(panel, (x0, y0))
      draw.rectangle((x0, y0, x0 + PANEL_W - 1, y0 + PANEL_H - 1), outline=GRID, width=1)
      title = titles[col] if col == 0 else f"{titles[col]} = {pose_vec[col - 1]:+.4f}"
      draw.text((x0 + 10, y0 + 10), title, font=small_font, fill=FG)

  output_path.parent.mkdir(parents=True, exist_ok=True)
  canvas.save(output_path)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Render gradient saliency overlays for PoseNet outputs p0..p5.")
  parser.add_argument("--input", type=Path, default=HERE / "videos" / "0.mkv")
  parser.add_argument("--output", type=Path, default=HERE / "artifacts" / "posenet_saliency.png")
  parser.add_argument("--device", type=str, default="cpu", help="cpu|cuda|mps")
  parser.add_argument("--fps", type=int, default=20)
  parser.add_argument("--pair-index", type=int, default=None, help="Use an explicit pair index instead of auto-selection.")
  parser.add_argument(
    "--select",
    choices=["max_norm", "max_abs_p0", "max_abs_p1", "max_abs_p2", "max_abs_p3", "max_abs_p4", "max_abs_p5"],
    default="max_norm",
    help="Auto-selection rule if --pair-index is omitted.",
  )
  parser.add_argument("--max-frames", type=int, default=None)
  parser.add_argument("--alpha", type=float, default=0.72)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  device = pick_device(args.device)
  print(f"Loading PoseNet on {device}", flush=True)
  posenet, _ = load_models(device)

  if args.pair_index is None:
    print(f"Scanning {args.input} for selector {args.select}", flush=True)
  else:
    print(f"Using explicit pair index {args.pair_index}", flush=True)
  pair_index, prev_frame, curr_frame, coarse_pose = find_pair(
    input_path=args.input,
    posenet=posenet,
    device=device,
    selector=args.select,
    pair_index=args.pair_index,
    max_frames=args.max_frames,
  )
  print(f"Selected pair {pair_index} with coarse pose {np.round(coarse_pose, 4).tolist()}", flush=True)

  print("Computing gradient saliency for p0..p5", flush=True)
  pose_vec, prev_saliency, curr_saliency = compute_saliency(posenet, prev_frame, curr_frame, device)
  _render_sheet(
    prev_frame=prev_frame,
    curr_frame=curr_frame,
    pose_vec=pose_vec,
    prev_saliency=prev_saliency,
    curr_saliency=curr_saliency,
    pair_index=pair_index,
    fps=args.fps,
    output_path=args.output,
    alpha=args.alpha,
  )
  print(f"Wrote {args.output} ({args.output.stat().st_size:,} bytes)", flush=True)


if __name__ == "__main__":
  main()
