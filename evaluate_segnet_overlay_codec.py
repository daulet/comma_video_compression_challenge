#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F

from frame_utils import camera_size
from generate_segnet_palette_candidate import PALETTES, _pick_device
from model_diagnostics import count_frames_for_path, iter_rgb_frames
from modules import DistortionNet, posenet_sd_path, segnet_sd_path

HERE = Path(__file__).resolve().parent
LOW_W = 512
LOW_H = 384
SVTAV1_ENC = Path("/tmp/SVT-AV1/Bin/Release/SvtAv1EncApp")


def _iter_candidate_rgb_frames(path: Path, max_frames: int | None = None):
  container = av.open(str(path))
  stream = container.streams.video[0]
  try:
    for idx, frame in enumerate(container.decode(stream)):
      if max_frames is not None and idx >= max_frames:
        break
      yield torch.from_numpy(frame.to_ndarray(format="rgb24").copy())
  finally:
    container.close()


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Encode low-res SegNet overlay video and score it with official distortion metrics.")
  parser.add_argument("--input", type=Path, default=HERE / "videos" / "0.mkv")
  parser.add_argument("--output-dir", type=Path, default=HERE / "artifacts" / "segnet_overlay_codec")
  parser.add_argument("--palette", choices=sorted(PALETTES), default="search9")
  parser.add_argument("--render-mode", choices=["palette", "overlay", "orig_low"], default="overlay")
  parser.add_argument("--alpha", type=float, default=0.995)
  parser.add_argument("--alpha-even", type=float, default=None)
  parser.add_argument("--alpha-odd", type=float, default=None)
  parser.add_argument("--codec", choices=["libx265", "libx264", "libsvtav1"], default="libx265")
  parser.add_argument("--crf", type=int, default=28)
  parser.add_argument("--preset", type=str, default="slow")
  parser.add_argument("--gop", type=int, default=240)
  parser.add_argument("--pix-fmt", type=str, default="yuv420p")
  parser.add_argument("--lossless", action="store_true")
  parser.add_argument("--ffmpeg-threads", type=int, default=0)
  parser.add_argument("--fps", type=int, default=20)
  parser.add_argument("--prefilter", type=str, default=None)
  parser.add_argument("--seg-batch-size", type=int, default=16)
  parser.add_argument("--eval-batch-size", type=int, default=16)
  parser.add_argument("--device", type=str, default="cpu", help="cpu|cuda|mps")
  parser.add_argument("--max-frames", type=int, default=None)
  parser.add_argument("--inflate-mode", choices=["nearest", "bilinear"], default="nearest")
  parser.add_argument("--keep-lowraw", action="store_true")
  return parser.parse_args()


def _encode_ffmpeg(
  *,
  lowraw_path: Path,
  encoded_path: Path,
  fps: int,
  codec: str,
  crf: int,
  preset: str,
  gop: int,
  pix_fmt: str,
  lossless: bool,
  ffmpeg_threads: int,
  prefilter: str | None,
) -> None:
  encoded_path.parent.mkdir(parents=True, exist_ok=True)
  if codec == "libsvtav1":
    if lossless:
      raise ValueError("lossless is not supported for libsvtav1 in this helper")
    if pix_fmt != "yuv420p":
      raise ValueError(f"external SVT-AV1 helper currently only supports yuv420p, got {pix_fmt}")
    if not SVTAV1_ENC.exists():
      raise FileNotFoundError(f"Missing SVT-AV1 encoder: {SVTAV1_ENC}")
    ivf_path = encoded_path.with_suffix(".ivf")
    ffmpeg_cmd = [
      "ffmpeg",
      "-y",
      "-hide_banner",
      "-loglevel",
      "error",
      "-f",
      "rawvideo",
      "-pix_fmt",
      "rgb24",
      "-s:v",
      f"{LOW_W}x{LOW_H}",
      "-r",
      str(fps),
      "-i",
      str(lowraw_path),
      "-an",
    ]
    if prefilter:
      ffmpeg_cmd.extend(["-vf", prefilter])
    ffmpeg_cmd.extend([
      "-pix_fmt",
      pix_fmt,
      "-f",
      "rawvideo",
      "-r",
      str(fps),
      "pipe:1",
    ])
    svt_cmd = [
      str(SVTAV1_ENC),
      "-i",
      "stdin",
      "-b",
      str(ivf_path),
      "-w",
      str(LOW_W),
      "-h",
      str(LOW_H),
      "--fps",
      str(fps),
      "--input-depth",
      "8",
      "--preset",
      str(preset),
      "--crf",
      str(crf),
      "--keyint",
      str(gop),
      "--scd",
      "0",
      "--lp",
      str(ffmpeg_threads if ffmpeg_threads > 0 else min(8, len(__import__("os").sched_getaffinity(0)))),
    ]
    with subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE) as ffmpeg_proc:
      try:
        subprocess.run(svt_cmd, stdin=ffmpeg_proc.stdout, check=True)
      finally:
        if ffmpeg_proc.stdout is not None:
          ffmpeg_proc.stdout.close()
        ffmpeg_proc.wait()
    subprocess.run([
      "ffmpeg",
      "-y",
      "-hide_banner",
      "-loglevel",
      "error",
      "-i",
      str(ivf_path),
      "-c:v",
      "copy",
      str(encoded_path),
    ], check=True)
    ivf_path.unlink(missing_ok=True)
    return
  cmd = [
    "ffmpeg",
    "-y",
    "-hide_banner",
    "-loglevel",
    "error",
    "-f",
    "rawvideo",
    "-pix_fmt",
    "rgb24",
    "-s:v",
    f"{LOW_W}x{LOW_H}",
    "-r",
    str(fps),
    "-i",
    str(lowraw_path),
    "-an",
    "-pix_fmt",
    pix_fmt,
    "-g",
    str(gop),
    "-keyint_min",
    str(gop),
    "-sc_threshold",
    "0",
    "-c:v",
    codec,
  ]
  if ffmpeg_threads > 0:
    cmd.extend(["-threads", str(ffmpeg_threads)])
  if prefilter:
    cmd.extend(["-vf", prefilter])
  if codec in {"libx265", "libx264"}:
    cmd.extend(["-preset", preset])
    if codec == "libx265":
      params = [f"keyint={gop}", f"min-keyint={gop}", "scenecut=0"]
      if lossless:
        params.append("lossless=1")
      else:
        cmd.extend(["-crf", str(crf)])
      cmd.extend(["-x265-params", ":".join(params)])
    else:
      if lossless:
        cmd.extend(["-qp", "0"])
      else:
        cmd.extend(["-crf", str(crf)])
  else:
    raise ValueError(f"unsupported codec {codec}")
  cmd.append(str(encoded_path))
  subprocess.run(cmd, check=True)


def _upsample_frame(frame: torch.Tensor, mode: str) -> torch.Tensor:
  x = frame.permute(2, 0, 1).unsqueeze(0).float()
  if mode == "nearest":
    up = F.interpolate(x, size=(camera_size[1], camera_size[0]), mode="nearest")
  else:
    up = F.interpolate(x, size=(camera_size[1], camera_size[0]), mode="bilinear", align_corners=False)
  return up[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8)


def main() -> None:
  args = _parse_args()
  if not (0.0 <= args.alpha <= 1.0):
    raise ValueError(f"--alpha must be in [0,1], got {args.alpha}")
  if args.alpha_even is not None and not (0.0 <= args.alpha_even <= 1.0):
    raise ValueError(f"--alpha-even must be in [0,1], got {args.alpha_even}")
  if args.alpha_odd is not None and not (0.0 <= args.alpha_odd <= 1.0):
    raise ValueError(f"--alpha-odd must be in [0,1], got {args.alpha_odd}")

  device = _pick_device(args.device)
  palette = PALETTES[args.palette]

  output_dir = args.output_dir
  output_dir.mkdir(parents=True, exist_ok=True)
  lowraw_path = output_dir / "overlay_512x384.raw"
  encoded_path = output_dir / "archive.mkv"
  report_path = output_dir / "report.json"

  n_frames = count_frames_for_path(args.input)
  if args.max_frames is not None:
    n_frames = min(n_frames, args.max_frames)
  if n_frames < 2:
    raise ValueError(f"Need at least 2 frames, found {n_frames}")

  print(f"Loading DistortionNet on {device}", flush=True)
  distortion_net = DistortionNet().eval().to(device)
  distortion_net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)

  print(f"Generating low-res {args.render_mode} frames to {lowraw_path}", flush=True)
  frame_iter = iter_rgb_frames(args.input, max_frames=n_frames)
  processed_frames = 0
  with lowraw_path.open("wb") as low_f:
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
      seg_rgb = torch.from_numpy(palette[seg_classes]).permute(0, 3, 1, 2).float().to(device)

      if args.render_mode == "palette":
        low_rgb = seg_rgb
      elif args.render_mode == "overlay":
        alpha_values = []
        for batch_offset in range(len(frames)):
          frame_idx = processed_frames + batch_offset
          if frame_idx % 2 == 0 and args.alpha_even is not None:
            alpha_values.append(args.alpha_even)
          elif frame_idx % 2 == 1 and args.alpha_odd is not None:
            alpha_values.append(args.alpha_odd)
          else:
            alpha_values.append(args.alpha)
        alpha = torch.tensor(alpha_values, dtype=torch.float32, device=device).view(-1, 1, 1, 1)
        low_rgb = alpha * orig_low + (1.0 - alpha) * seg_rgb
      elif args.render_mode == "orig_low":
        low_rgb = orig_low
      else:
        raise ValueError(f"unsupported render mode {args.render_mode}")

      low_uint8 = low_rgb.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8).cpu()
      for frame in low_uint8:
        low_f.write(frame.contiguous().numpy().tobytes())
        processed_frames += 1
        if processed_frames % 64 == 0 or processed_frames == n_frames:
          print(f"  lowraw {processed_frames}/{n_frames} frames", flush=True)

  if processed_frames != n_frames:
    raise ValueError(f"expected {n_frames} generated frames, got {processed_frames}")

  print(f"Encoding {encoded_path.name} with {args.codec} crf={args.crf} preset={args.preset}", flush=True)
  _encode_ffmpeg(
    lowraw_path=lowraw_path,
    encoded_path=encoded_path,
    fps=args.fps,
    codec=args.codec,
    crf=args.crf,
    preset=args.preset,
    gop=args.gop,
    pix_fmt=args.pix_fmt,
    lossless=args.lossless,
    ffmpeg_threads=args.ffmpeg_threads,
    prefilter=args.prefilter,
  )

  encoded_frames = count_frames_for_path(encoded_path)
  if encoded_frames != n_frames:
    raise ValueError(f"encoded frame count mismatch: expected {n_frames}, got {encoded_frames}")

  print("Evaluating official distortion on decoded low-res candidate", flush=True)
  gt_iter = iter_rgb_frames(args.input, max_frames=n_frames)
  cmp_iter = _iter_candidate_rgb_frames(encoded_path, max_frames=n_frames)
  pending_gt: list[torch.Tensor] = []
  pending_cmp: list[torch.Tensor] = []
  batch_gt: list[torch.Tensor] = []
  batch_cmp: list[torch.Tensor] = []
  posenet_sum = 0.0
  segnet_sum = 0.0
  n_samples = 0
  compared_frames = 0

  def flush_eval() -> None:
    nonlocal posenet_sum, segnet_sum, n_samples, batch_gt, batch_cmp
    if not batch_gt:
      return
    gt = torch.stack(batch_gt, dim=0).to(device)
    cmp = torch.stack(batch_cmp, dim=0).to(device)
    with torch.inference_mode():
      posenet_dist, segnet_dist = distortion_net.compute_distortion(gt, cmp)
    posenet_sum += float(posenet_dist.sum().item())
    segnet_sum += float(segnet_dist.sum().item())
    n_samples += gt.shape[0]
    batch_gt = []
    batch_cmp = []

  for gt_frame, cmp_low in zip(gt_iter, cmp_iter):
    cmp_frame = _upsample_frame(cmp_low, mode=args.inflate_mode)
    pending_gt.append(gt_frame)
    pending_cmp.append(cmp_frame)
    if len(pending_gt) == 2:
      batch_gt.append(torch.stack(pending_gt, dim=0))
      batch_cmp.append(torch.stack(pending_cmp, dim=0))
      pending_gt = []
      pending_cmp = []
    if len(batch_gt) >= args.eval_batch_size:
      flush_eval()
    compared_frames += 1
    if compared_frames % 64 == 0 or compared_frames == n_frames:
      print(f"  eval {compared_frames}/{n_frames} frames", flush=True)
  flush_eval()

  if compared_frames != n_frames:
    raise ValueError(f"evaluation frame mismatch: expected {n_frames}, got {compared_frames}")

  posenet_dist = posenet_sum / max(n_samples, 1)
  segnet_dist = segnet_sum / max(n_samples, 1)
  archive_bytes = encoded_path.stat().st_size
  original_bytes = args.input.stat().st_size
  rate = archive_bytes / original_bytes
  score = 100 * segnet_dist + math.sqrt(10 * posenet_dist) + 25 * rate
  report = {
    "input": str(args.input),
    "palette_name": args.palette,
    "palette_rgb": palette.tolist(),
    "render_mode": args.render_mode,
    "alpha": args.alpha,
    "alpha_even": args.alpha_even,
    "alpha_odd": args.alpha_odd,
    "codec": args.codec,
    "crf": args.crf,
    "preset": args.preset,
    "gop": args.gop,
    "pix_fmt": args.pix_fmt,
    "lossless": args.lossless,
    "ffmpeg_threads": args.ffmpeg_threads,
    "prefilter": args.prefilter,
    "inflate_mode": args.inflate_mode,
    "encoded_path": str(encoded_path),
    "n_frames": n_frames,
    "n_samples": n_samples,
    "archive_bytes": archive_bytes,
    "original_bytes": original_bytes,
    "rate": rate,
    "posenet_dist": posenet_dist,
    "segnet_dist": segnet_dist,
    "score": score,
  }
  report_path.write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report, indent=2), flush=True)

  if not args.keep_lowraw:
    lowraw_path.unlink(missing_ok=True)


if __name__ == "__main__":
  main()
