#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from model_diagnostics import pick_device, render_single_dashboard

HERE = Path(__file__).resolve().parent


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Render a reusable SegNet/PoseNet dashboard for one video.")
  parser.add_argument("--input", type=Path, default=HERE / "videos" / "0.mkv")
  parser.add_argument("--output", type=Path, default=HERE / "artifacts" / "model_outputs_0.mp4")
  parser.add_argument("--device", type=str, default="cpu", help="cpu|cuda|mps")
  parser.add_argument("--fps", type=int, default=20)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--max-frames", type=int, default=None)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  device = pick_device(args.device)
  print(f"Rendering single-view dashboard from {args.input} on {device}", flush=True)
  render_single_dashboard(
    input_path=args.input,
    output_path=args.output,
    device=device,
    fps=args.fps,
    max_frames=args.max_frames,
    batch_size=args.batch_size,
  )
  print(f"Wrote {args.output} ({args.output.stat().st_size:,} bytes)", flush=True)


if __name__ == "__main__":
  main()
