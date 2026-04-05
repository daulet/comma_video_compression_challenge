#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import shutil
import subprocess
import zipfile
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F

from frame_utils import camera_size, yuv420_to_rgb

HERE = Path(__file__).resolve().parent
ENCODE_PRESET = "medium"
ENCODE_GOP = 120
ENCODE_BFRAMES = 4
REFINER_FEATURES = 13
REFINER_SAMPLE_STRIDE = 16
REFINER_RIDGE = 1e-2
REFINER_RESIDUAL_CLAMP = 24.0 / 255.0


def load_video_names(video_names_file: Path) -> list[str]:
  return [
    line.strip()
    for line in video_names_file.read_text().splitlines()
    if line.strip()
  ]


def _reset_dir(path: Path) -> None:
  if path.exists():
    shutil.rmtree(path)
  path.mkdir(parents=True, exist_ok=True)


def _run(cmd: list[str]) -> None:
  subprocess.run(cmd, check=True)


def _resize_rgb(t: torch.Tensor) -> torch.Tensor:
  target_w, target_h = camera_size
  h, w, _ = t.shape
  if h == target_h and w == target_w:
    return t
  x = t.permute(2, 0, 1).unsqueeze(0).float()
  x = F.interpolate(x, size=(target_h, target_w), mode="bicubic", align_corners=False)
  return x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().to(torch.uint8)


def _iter_video_tensors(src: Path) -> itertools.chain[torch.Tensor]:
  fmt = "hevc" if src.suffix == ".hevc" else None
  container = av.open(str(src), format=fmt)
  stream = container.streams.video[0]
  try:
    for frame in container.decode(stream):
      yield _resize_rgb(yuv420_to_rgb(frame)).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
  finally:
    container.close()


def _refiner_features(base: torch.Tensor, prev_base: torch.Tensor) -> torch.Tensor:
  blur = F.avg_pool2d(F.pad(base, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)
  edge = base - blur
  delta = base - prev_base
  bias = torch.ones((1, 1, base.shape[2], base.shape[3]), dtype=base.dtype, device=base.device)
  return torch.cat([base, edge, delta, delta.abs(), bias], dim=1)


def _refiner_path(dst_video: Path) -> Path:
  return dst_video.with_suffix(".refiner.bin")


def _fit_temporal_refiner(src: Path, encoded: Path, dst: Path) -> None:
  print(f"Fitting temporal refiner for {encoded}")
  xtx = torch.zeros((REFINER_FEATURES, REFINER_FEATURES), dtype=torch.float64)
  xty = torch.zeros((REFINER_FEATURES, 3), dtype=torch.float64)
  prev_base = None
  n_frames = 0

  for gt, base in zip(_iter_video_tensors(src), _iter_video_tensors(encoded)):
    if prev_base is None:
      prev_base = base
    feats = _refiner_features(base, prev_base)[..., ::REFINER_SAMPLE_STRIDE, ::REFINER_SAMPLE_STRIDE]
    target = (gt - base)[..., ::REFINER_SAMPLE_STRIDE, ::REFINER_SAMPLE_STRIDE]
    x = feats.squeeze(0).permute(1, 2, 0).reshape(-1, REFINER_FEATURES).to(dtype=torch.float64)
    y = target.squeeze(0).permute(1, 2, 0).reshape(-1, 3).to(dtype=torch.float64)
    xtx += x.T @ x
    xty += x.T @ y
    prev_base = base
    n_frames += 1

  ridge = REFINER_RIDGE * torch.eye(REFINER_FEATURES, dtype=torch.float64)
  weights = torch.linalg.solve(xtx + ridge, xty).to(dtype=torch.float16).cpu().numpy()
  dst.write_bytes(weights.tobytes())
  print(f"Saved temporal refiner ({n_frames} frames, {dst.stat().st_size} bytes)")


def _load_temporal_refiner(path: Path) -> torch.Tensor | None:
  if not path.exists():
    return None
  arr = np.frombuffer(path.read_bytes(), dtype=np.float16)
  if arr.size != REFINER_FEATURES * 3:
    raise ValueError(f"Unexpected refiner size for {path}: {arr.size}")
  return torch.tensor(arr.reshape(REFINER_FEATURES, 3), dtype=torch.float32)


def _apply_temporal_refiner(base: torch.Tensor, prev_base: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
  if weights is None:
    return base
  feats = _refiner_features(base, prev_base).squeeze(0).permute(1, 2, 0)
  residual = torch.matmul(feats, weights).clamp_(-REFINER_RESIDUAL_CLAMP, REFINER_RESIDUAL_CLAMP)
  refined = (base.squeeze(0).permute(1, 2, 0) + residual).clamp_(0.0, 1.0)
  return refined.permute(2, 0, 1).unsqueeze(0)


def _encode_one_video(src: Path, dst: Path, scale_factor: float, crf: int) -> None:
  dst.parent.mkdir(parents=True, exist_ok=True)
  scale_expr = (
    f"scale=trunc(iw*{scale_factor}/2)*2:trunc(ih*{scale_factor}/2)*2:flags=lanczos"
  )
  _run(
    [
      "ffmpeg",
      "-nostdin",
      "-y",
      "-hide_banner",
      "-loglevel",
      "warning",
      "-r",
      "20",
      "-fflags",
      "+genpts",
      "-i",
      str(src),
      "-vf",
      scale_expr,
      "-pix_fmt",
      "yuv420p",
      "-c:v",
      "libx265",
      "-preset",
      ENCODE_PRESET,
      "-crf",
      str(crf),
      "-g",
      str(ENCODE_GOP),
      "-bf",
      str(ENCODE_BFRAMES),
      "-x265-params",
      (
        f"keyint={ENCODE_GOP}:min-keyint=24:scenecut=40:"
        "aq-mode=3:qcomp=0.70:deblock=-1,-1:log-level=warning"
      ),
      "-r",
      "20",
      str(dst),
    ]
  )


def compress_videos(
  in_dir: Path,
  video_names: list[str],
  archive_dir: Path,
  archive_zip: Path,
  *,
  scale_factor: float = 0.45,
  crf: int = 30,
) -> None:
  _reset_dir(archive_dir)

  for rel in video_names:
    base = Path(rel).with_suffix("")
    src = in_dir / rel
    dst = archive_dir / f"{base}.mkv"
    if not src.exists():
      raise FileNotFoundError(f"Missing source video: {src}")
    print(f"Encoding {src} -> {dst}")
    _encode_one_video(src, dst, scale_factor=scale_factor, crf=crf)
    _fit_temporal_refiner(src, dst, _refiner_path(dst))

  archive_zip.parent.mkdir(parents=True, exist_ok=True)
  if archive_zip.exists():
    archive_zip.unlink()
  with zipfile.ZipFile(archive_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(archive_dir.rglob("*")):
      if path.is_file():
        zf.write(path, arcname=str(path.relative_to(archive_dir)))
  print(f"Compressed to {archive_zip}")


def _decode_and_resize_to_raw(src: Path, dst: Path, weights: torch.Tensor | None) -> int:
  fmt = "hevc" if src.suffix == ".hevc" else None
  container = av.open(str(src), format=fmt)
  stream = container.streams.video[0]
  dst.parent.mkdir(parents=True, exist_ok=True)
  n = 0
  prev_base = None
  with dst.open("wb") as f:
    for frame in container.decode(stream):
      base = _resize_rgb(yuv420_to_rgb(frame)).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
      if prev_base is None:
        prev_base = base
      t = _apply_temporal_refiner(base, prev_base, weights)
      f.write(
        t.mul(255.0).clamp_(0.0, 255.0).squeeze(0).permute(1, 2, 0).round().to(torch.uint8).contiguous().numpy().tobytes()
      )
      prev_base = base
      n += 1
  container.close()
  return n


def inflate_archive(archive_dir: Path, output_dir: Path, video_names: list[str]) -> None:
  _reset_dir(output_dir)
  for rel in video_names:
    base = Path(rel).with_suffix("")
    src = archive_dir / f"{base}.mkv"
    dst = output_dir / f"{base}.raw"
    if not src.exists():
      raise FileNotFoundError(f"Missing encoded video in archive: {src}")
    weights = _load_temporal_refiner(_refiner_path(src))
    print(f"Decoding + resizing {src} -> {dst}")
    n = _decode_and_resize_to_raw(src, dst, weights)
    print(f"Saved {n} frames")


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Single-file compression/inflate entrypoint.")
  subparsers = parser.add_subparsers(dest="command", required=True)

  compress_parser = subparsers.add_parser("compress", help="compress input videos into archive.zip")
  compress_parser.add_argument("--in-dir", type=Path, default=HERE / "videos")
  compress_parser.add_argument("--video-names-file", type=Path, default=HERE / "public_test_video_names.txt")
  compress_parser.add_argument("--archive-dir", type=Path, default=HERE / "autoresearch_work" / "archive_build")
  compress_parser.add_argument("--archive-zip", type=Path, default=HERE / "autoresearch_work" / "archive.zip")
  compress_parser.add_argument("--scale-factor", type=float, default=0.45)
  compress_parser.add_argument("--crf", type=int, default=30)

  inflate_parser = subparsers.add_parser("inflate", help="inflate archive dir into .raw files")
  inflate_parser.add_argument("--archive-dir", type=Path, required=True)
  inflate_parser.add_argument("--out-dir", type=Path, default=HERE / "autoresearch_work" / "inflated")
  inflate_parser.add_argument("--video-names-file", type=Path, default=HERE / "public_test_video_names.txt")

  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  video_names = load_video_names(args.video_names_file)

  if args.command == "compress":
    compress_videos(
      in_dir=args.in_dir,
      video_names=video_names,
      archive_dir=args.archive_dir,
      archive_zip=args.archive_zip,
      scale_factor=args.scale_factor,
      crf=args.crf,
    )
    return

  if args.command == "inflate":
    inflate_archive(
      archive_dir=args.archive_dir,
      output_dir=args.out_dir,
      video_names=video_names,
    )
    return

  raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
  main()
