#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from model_diagnostics import pick_device, render_comparison_dashboard

HERE = Path(__file__).resolve().parent


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Render an original-vs-candidate SegNet/PoseNet comparison dashboard.")
  parser.add_argument("--reference", type=Path, default=HERE / "videos" / "0.mkv")
  parser.add_argument("--candidate", type=Path, required=True)
  parser.add_argument("--output", type=Path, default=HERE / "artifacts" / "model_compare_0.mp4")
  parser.add_argument("--device", type=str, default="cpu", help="cpu|cuda|mps")
  parser.add_argument("--fps", type=int, default=20)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--max-frames", type=int, default=None)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  device = pick_device(args.device)
  print(
    f"Rendering comparison dashboard: reference={args.reference} candidate={args.candidate} on {device}",
    flush=True,
  )
  render_comparison_dashboard(
    reference_path=args.reference,
    candidate_path=args.candidate,
    output_path=args.output,
    device=device,
    fps=args.fps,
    max_frames=args.max_frames,
    batch_size=args.batch_size,
  )
  print(f"Wrote {args.output} ({args.output.stat().st_size:,} bytes)", flush=True)


if __name__ == "__main__":
  main()
