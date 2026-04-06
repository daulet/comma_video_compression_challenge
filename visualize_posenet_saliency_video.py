#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from model_diagnostics import iter_rgb_frames, load_models, pick_device
from posenet_saliency import POSE_SALIENCY_COLORS, compute_saliency, find_pair, normalize_map, winner_overlay

HERE = Path(__file__).resolve().parent

PANEL_W = 640
PANEL_H = 480
SIDE_W = 320
HEADER_H = 56
CANVAS_W = PANEL_W * 2 + SIDE_W
CANVAS_H = PANEL_H + HEADER_H

BG = (12, 14, 20)
FG = (236, 239, 244)
MUTED = (155, 161, 176)
GRID = (54, 61, 79)
def _load_window(input_path: Path, start_frame: int, n_frames: int) -> list:
  frames = []
  stop = start_frame + n_frames
  for idx, frame in enumerate(iter_rgb_frames(input_path)):
    if idx < start_frame:
      continue
    if idx >= stop:
      break
    frames.append(frame)
  if len(frames) < 2:
    raise ValueError(f"Need at least 2 frames in requested window, found {len(frames)}")
  return frames


def _resize(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
  return np.array(Image.fromarray(rgb).resize(size, resample=Image.Resampling.BILINEAR))


def _draw_legend(
  draw: ImageDraw.ImageDraw,
  x0: int,
  y0: int,
  pose_vec: np.ndarray,
  font: ImageFont.ImageFont,
  small_font: ImageFont.ImageFont,
) -> None:
  draw.text((x0, y0), "Winner-take-all saliency", font=font, fill=FG)
  draw.text((x0, y0 + 22), "color = dominant PoseNet dim", font=small_font, fill=MUTED)
  for idx, color in enumerate(POSE_SALIENCY_COLORS.tolist()):
    top = y0 + 56 + idx * 42
    draw.rounded_rectangle((x0, top, x0 + 18, top + 18), radius=3, fill=tuple(color))
    draw.text((x0 + 28, top - 1), f"p{idx}: {pose_vec[idx]:+.4f}", font=small_font, fill=FG)


def _render_frame(
  prev_overlay: np.ndarray,
  curr_overlay: np.ndarray,
  pose_vec: np.ndarray,
  pair_index: int,
  n_pairs: int,
  global_pair_index: int,
  fps: int,
  font: ImageFont.ImageFont,
  small_font: ImageFont.ImageFont,
) -> np.ndarray:
  canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
  draw = ImageDraw.Draw(canvas)

  draw.text((18, 16), f"pair {global_pair_index:04d} ({pair_index + 1}/{n_pairs})", font=font, fill=FG)
  draw.text((220, 16), f"time {(global_pair_index + 1) / fps:6.2f}s", font=font, fill=FG)
  draw.text((380, 16), "overlay intensity = normalized saliency strength", font=small_font, fill=MUTED)

  prev_panel = Image.fromarray(_resize(prev_overlay, (PANEL_W, PANEL_H)))
  curr_panel = Image.fromarray(_resize(curr_overlay, (PANEL_W, PANEL_H)))
  canvas.paste(prev_panel, (0, HEADER_H))
  canvas.paste(curr_panel, (PANEL_W, HEADER_H))

  draw.rectangle((0, HEADER_H, PANEL_W - 1, CANVAS_H - 1), outline=GRID, width=1)
  draw.rectangle((PANEL_W, HEADER_H, PANEL_W * 2 - 1, CANVAS_H - 1), outline=GRID, width=1)
  draw.text((18, HEADER_H + 14), "previous frame overlay", font=font, fill=FG)
  draw.text((PANEL_W + 18, HEADER_H + 14), "current frame overlay", font=font, fill=FG)
  _draw_legend(draw, PANEL_W * 2 + 18, HEADER_H + 18, pose_vec, font, small_font)
  return np.array(canvas)


def _open_output(output_path: Path, fps: int):
  output_path.parent.mkdir(parents=True, exist_ok=True)
  container = av.open(str(output_path), mode="w")
  stream = container.add_stream("libx264", rate=fps)
  stream.width = CANVAS_W
  stream.height = CANVAS_H
  stream.pix_fmt = "yuv420p"
  stream.options = {"crf": "18", "preset": "medium"}
  return container, stream


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Render a combined-color PoseNet saliency video overlay.")
  parser.add_argument("--input", type=Path, default=HERE / "videos" / "0.mkv")
  parser.add_argument("--output", type=Path, default=HERE / "artifacts" / "posenet_saliency_window.mp4")
  parser.add_argument("--device", type=str, default="cpu", help="cpu|cuda|mps")
  parser.add_argument("--fps", type=int, default=20)
  parser.add_argument("--pair-index", type=int, default=None)
  parser.add_argument(
    "--select",
    choices=["max_norm", "max_abs_p0", "max_abs_p1", "max_abs_p2", "max_abs_p3", "max_abs_p4", "max_abs_p5"],
    default="max_norm",
  )
  parser.add_argument("--window-pairs", type=int, default=48)
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
  center_pair, _, _, coarse_pose = find_pair(
    input_path=args.input,
    posenet=posenet,
    device=device,
    selector=args.select,
    pair_index=args.pair_index,
    max_frames=args.max_frames,
  )
  print(f"Selected center pair {center_pair} with coarse pose {np.round(coarse_pose, 4).tolist()}", flush=True)

  half = args.window_pairs // 2
  start_pair = max(0, center_pair - half)
  start_frame = start_pair
  n_frames = args.window_pairs + 1
  print(f"Loading window: start_pair={start_pair}, frames={n_frames}", flush=True)
  frames = _load_window(args.input, start_frame=start_frame, n_frames=n_frames)

  font = ImageFont.load_default()
  small_font = ImageFont.load_default()
  container, stream = _open_output(args.output, fps=args.fps)

  try:
    print("Rendering combined-color saliency video...", flush=True)
    for local_pair in range(len(frames) - 1):
      prev_frame = frames[local_pair]
      curr_frame = frames[local_pair + 1]
      pose_vec, prev_saliency, curr_saliency = compute_saliency(posenet, prev_frame, curr_frame, device)
      prev_overlay = winner_overlay(prev_frame.numpy(), prev_saliency, alpha=args.alpha)
      curr_overlay = winner_overlay(curr_frame.numpy(), curr_saliency, alpha=args.alpha)
      dashboard = _render_frame(
        prev_overlay=prev_overlay,
        curr_overlay=curr_overlay,
        pose_vec=pose_vec,
        pair_index=local_pair,
        n_pairs=len(frames) - 1,
        global_pair_index=start_pair + local_pair,
        fps=args.fps,
        font=font,
        small_font=small_font,
      )
      video_frame = av.VideoFrame.from_ndarray(dashboard, format="rgb24")
      for packet in stream.encode(video_frame):
        container.mux(packet)
  finally:
    for packet in stream.encode():
      container.mux(packet)
    container.close()

  print(f"Wrote {args.output} ({args.output.stat().st_size:,} bytes)", flush=True)


if __name__ == "__main__":
  main()
