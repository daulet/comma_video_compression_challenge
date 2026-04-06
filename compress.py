#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F

from frame_utils import camera_size, yuv420_to_rgb

HERE = Path(__file__).resolve().parent
ENCODE_PRESET = "slow"
ENCODE_GOP = 180
ENCODE_BFRAMES = 4
LINEAR_REFINER_FEATURES = 13
LINEAR_REFINER_SAMPLE_STRIDE = 16
REFINER_RIDGE = 1e-2
LINEAR_REFINER_RESIDUAL_CLAMP = 24.0 / 255.0
MLP_REFINER_FEATURES = 19
MLP_REFINER_HIDDEN = 5
MLP_REFINER_SAMPLE_STRIDE = 32
MLP_REFINER_MAX_SAMPLES = 200_000
MLP_REFINER_STEPS = 192
MLP_REFINER_BATCH_SIZE = 8192
MLP_REFINER_LR = 3e-2
MLP_REFINER_RESIDUAL_CLAMP = 10.0 / 255.0


@dataclass(frozen=True)
class TemporalRefiner:
  linear: torch.Tensor
  mlp_w1: torch.Tensor
  mlp_b1: torch.Tensor
  mlp_w2: torch.Tensor
  mlp_b2: torch.Tensor


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


def _linear_refiner_features(base: torch.Tensor, prev_base: torch.Tensor) -> torch.Tensor:
  blur = F.avg_pool2d(F.pad(base, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)
  edge = base - blur
  delta = base - prev_base
  bias = torch.ones((1, 1, base.shape[2], base.shape[3]), dtype=base.dtype, device=base.device)
  return torch.cat([base, edge, delta, delta.abs(), bias], dim=1)


def _mlp_refiner_features(
  base: torch.Tensor,
  prev_base: torch.Tensor,
  prev_prev_base: torch.Tensor,
  linear_residual: torch.Tensor,
  prev_linear_residual: torch.Tensor,
) -> torch.Tensor:
  delta = base - prev_base
  prev_delta = prev_base - prev_prev_base
  linear = linear_residual.permute(2, 0, 1).unsqueeze(0)
  prev_linear = prev_linear_residual.permute(2, 0, 1).unsqueeze(0)
  bias = torch.ones((1, 1, base.shape[2], base.shape[3]), dtype=base.dtype, device=base.device)
  return torch.cat([base, delta, delta.abs(), prev_delta, linear, prev_linear, bias], dim=1)


def _refiner_path(dst_video: Path) -> Path:
  return dst_video.with_suffix(".refiner.bin")


def _predict_linear_refiner_residual(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
  return torch.matmul(features.squeeze(0).permute(1, 2, 0), weights).clamp_(
    -LINEAR_REFINER_RESIDUAL_CLAMP, LINEAR_REFINER_RESIDUAL_CLAMP
  )


def _predict_mlp_refiner_residual(features: torch.Tensor, refiner: TemporalRefiner) -> torch.Tensor:
  hidden = torch.tanh(torch.matmul(features.squeeze(0).permute(1, 2, 0), refiner.mlp_w1) + refiner.mlp_b1)
  return torch.matmul(hidden, refiner.mlp_w2).add_(refiner.mlp_b2).clamp_(
    -MLP_REFINER_RESIDUAL_CLAMP, MLP_REFINER_RESIDUAL_CLAMP
  )


def _refine_frame(base: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
  refined = (base.squeeze(0).permute(1, 2, 0) + residual).clamp_(0.0, 1.0)
  return refined.permute(2, 0, 1).unsqueeze(0)


def _zero_residual_like(base: torch.Tensor) -> torch.Tensor:
  return torch.zeros((base.shape[2], base.shape[3], 3), dtype=base.dtype, device=base.device)


def _fit_mlp_temporal_refiner(features: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, ...]:
  if features.shape[0] > MLP_REFINER_MAX_SAMPLES:
    generator = torch.Generator().manual_seed(0)
    keep = torch.randperm(features.shape[0], generator=generator)[:MLP_REFINER_MAX_SAMPLES]
    features = features[keep]
    targets = targets[keep]

  generator = torch.Generator().manual_seed(0)
  w1 = (torch.randn((MLP_REFINER_FEATURES, MLP_REFINER_HIDDEN), generator=generator) * 0.05).requires_grad_()
  b1 = torch.zeros((MLP_REFINER_HIDDEN,), dtype=torch.float32, requires_grad=True)
  w2 = (torch.randn((MLP_REFINER_HIDDEN, 3), generator=generator) * 0.05).requires_grad_()
  b2 = torch.zeros((3,), dtype=torch.float32, requires_grad=True)
  params = [w1, b1, w2, b2]
  optimizer = torch.optim.Adam(params, lr=MLP_REFINER_LR)
  batch_size = min(MLP_REFINER_BATCH_SIZE, features.shape[0])

  for _ in range(MLP_REFINER_STEPS):
    batch_idx = torch.randint(features.shape[0], (batch_size,), generator=generator)
    xb = features[batch_idx]
    yb = targets[batch_idx]
    hidden = torch.tanh(xb @ w1 + b1)
    pred = (hidden @ w2 + b2).clamp(-MLP_REFINER_RESIDUAL_CLAMP, MLP_REFINER_RESIDUAL_CLAMP)
    loss = F.smooth_l1_loss(pred, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

  with torch.inference_mode():
    hidden = torch.tanh(features @ w1 + b1)
    pred = (hidden @ w2 + b2).clamp(-MLP_REFINER_RESIDUAL_CLAMP, MLP_REFINER_RESIDUAL_CLAMP)
    loss = F.smooth_l1_loss(pred, targets).item()
    print(f"Fitted tiny temporal MLP on {features.shape[0]} samples (loss={loss:.6f})")

  return (
    w1.detach().to(dtype=torch.float16),
    b1.detach().to(dtype=torch.float16),
    w2.detach().to(dtype=torch.float16),
    b2.detach().to(dtype=torch.float16),
  )


def _save_temporal_refiner(dst: Path, refiner: TemporalRefiner) -> None:
  arrays = [
    refiner.linear.cpu().numpy().reshape(-1),
    refiner.mlp_w1.cpu().numpy().reshape(-1),
    refiner.mlp_b1.cpu().numpy().reshape(-1),
    refiner.mlp_w2.cpu().numpy().reshape(-1),
    refiner.mlp_b2.cpu().numpy().reshape(-1),
  ]
  dst.write_bytes(np.concatenate(arrays).astype(np.float16).tobytes())


def _fit_temporal_refiner(src: Path, encoded: Path, dst: Path) -> None:
  print(f"Fitting temporal refiner for {encoded}")
  xtx = torch.zeros((LINEAR_REFINER_FEATURES, LINEAR_REFINER_FEATURES), dtype=torch.float64)
  xty = torch.zeros((LINEAR_REFINER_FEATURES, 3), dtype=torch.float64)
  n_frames = 0
  prev_base = None

  for gt, base in zip(_iter_video_tensors(src), _iter_video_tensors(encoded)):
    if prev_base is None:
      prev_base = base
    feats = _linear_refiner_features(base, prev_base)[..., ::LINEAR_REFINER_SAMPLE_STRIDE, ::LINEAR_REFINER_SAMPLE_STRIDE]
    target = (gt - base)[..., ::LINEAR_REFINER_SAMPLE_STRIDE, ::LINEAR_REFINER_SAMPLE_STRIDE]
    x = feats.squeeze(0).permute(1, 2, 0).reshape(-1, LINEAR_REFINER_FEATURES).to(dtype=torch.float64)
    y = target.squeeze(0).permute(1, 2, 0).reshape(-1, 3).to(dtype=torch.float64)
    xtx += x.T @ x
    xty += x.T @ y
    prev_base = base
    n_frames += 1

  ridge = REFINER_RIDGE * torch.eye(LINEAR_REFINER_FEATURES, dtype=torch.float64)
  linear_weights = torch.linalg.solve(xtx + ridge, xty).to(dtype=torch.float32)

  mlp_features = []
  mlp_targets = []
  prev_base = None
  prev_prev_base = None
  prev_linear_residual = None
  for gt, base in zip(_iter_video_tensors(src), _iter_video_tensors(encoded)):
    if prev_base is None:
      prev_base = base
    if prev_prev_base is None:
      prev_prev_base = prev_base
    if prev_linear_residual is None:
      prev_linear_residual = _zero_residual_like(base)
    linear_feats = _linear_refiner_features(base, prev_base)
    linear_residual = _predict_linear_refiner_residual(linear_feats, linear_weights)
    linear_refined = _refine_frame(base, linear_residual)
    feats = _mlp_refiner_features(base, prev_base, prev_prev_base, linear_residual, prev_linear_residual)
    target = (gt - linear_refined).clamp_(-MLP_REFINER_RESIDUAL_CLAMP, MLP_REFINER_RESIDUAL_CLAMP)
    x = feats[..., ::MLP_REFINER_SAMPLE_STRIDE, ::MLP_REFINER_SAMPLE_STRIDE]
    y = target[..., ::MLP_REFINER_SAMPLE_STRIDE, ::MLP_REFINER_SAMPLE_STRIDE]
    mlp_features.append(x.squeeze(0).permute(1, 2, 0).reshape(-1, MLP_REFINER_FEATURES))
    mlp_targets.append(y.squeeze(0).permute(1, 2, 0).reshape(-1, 3))
    prev_prev_base = prev_base
    prev_base = base
    prev_linear_residual = linear_residual

  mlp_w1, mlp_b1, mlp_w2, mlp_b2 = _fit_mlp_temporal_refiner(
    torch.cat(mlp_features).to(dtype=torch.float32),
    torch.cat(mlp_targets).to(dtype=torch.float32),
  )
  refiner = TemporalRefiner(
    linear=linear_weights.to(dtype=torch.float16),
    mlp_w1=mlp_w1,
    mlp_b1=mlp_b1,
    mlp_w2=mlp_w2,
    mlp_b2=mlp_b2,
  )
  _save_temporal_refiner(dst, refiner)
  print(f"Saved temporal refiner ({n_frames} frames, {dst.stat().st_size} bytes)")


def _load_temporal_refiner(path: Path) -> TemporalRefiner | None:
  if not path.exists():
    return None
  arr = np.frombuffer(path.read_bytes(), dtype=np.float16)
  expected = (
    LINEAR_REFINER_FEATURES * 3
    + MLP_REFINER_FEATURES * MLP_REFINER_HIDDEN
    + MLP_REFINER_HIDDEN
    + MLP_REFINER_HIDDEN * 3
    + 3
  )
  if arr.size != expected:
    raise ValueError(f"Unexpected refiner size for {path}: {arr.size}")
  offset = 0

  def take(count: int, shape: tuple[int, ...]) -> torch.Tensor:
    nonlocal offset
    out = torch.tensor(arr[offset:offset + count].reshape(shape), dtype=torch.float32)
    offset += count
    return out

  return TemporalRefiner(
    linear=take(LINEAR_REFINER_FEATURES * 3, (LINEAR_REFINER_FEATURES, 3)),
    mlp_w1=take(MLP_REFINER_FEATURES * MLP_REFINER_HIDDEN, (MLP_REFINER_FEATURES, MLP_REFINER_HIDDEN)),
    mlp_b1=take(MLP_REFINER_HIDDEN, (MLP_REFINER_HIDDEN,)),
    mlp_w2=take(MLP_REFINER_HIDDEN * 3, (MLP_REFINER_HIDDEN, 3)),
    mlp_b2=take(3, (3,)),
  )


def _apply_temporal_refiner(
  base: torch.Tensor,
  prev_base: torch.Tensor,
  prev_prev_base: torch.Tensor,
  prev_linear_residual: torch.Tensor,
  refiner: TemporalRefiner | None,
) -> tuple[torch.Tensor, torch.Tensor]:
  if refiner is None:
    return base, _zero_residual_like(base)
  linear_features = _linear_refiner_features(base, prev_base)
  linear_residual = _predict_linear_refiner_residual(linear_features, refiner.linear)
  mlp_features = _mlp_refiner_features(base, prev_base, prev_prev_base, linear_residual, prev_linear_residual)
  residual = linear_residual + _predict_mlp_refiner_residual(mlp_features, refiner)
  return _refine_frame(base, residual), linear_residual


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
        f"keyint={ENCODE_GOP}:min-keyint=24:scenecut=0:"
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


def _decode_and_resize_to_raw(src: Path, dst: Path, weights: TemporalRefiner | None) -> int:
  fmt = "hevc" if src.suffix == ".hevc" else None
  container = av.open(str(src), format=fmt)
  stream = container.streams.video[0]
  dst.parent.mkdir(parents=True, exist_ok=True)
  n = 0
  prev_base = None
  prev_prev_base = None
  prev_linear_residual = None
  with dst.open("wb") as f:
    for frame in container.decode(stream):
      base = _resize_rgb(yuv420_to_rgb(frame)).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
      if prev_base is None:
        prev_base = base
      if prev_prev_base is None:
        prev_prev_base = prev_base
      if prev_linear_residual is None:
        prev_linear_residual = _zero_residual_like(base)
      t, linear_residual = _apply_temporal_refiner(base, prev_base, prev_prev_base, prev_linear_residual, weights)
      prev_prev_base = prev_base
      f.write(
        t.mul(255.0).clamp_(0.0, 255.0).squeeze(0).permute(1, 2, 0).round().to(torch.uint8).contiguous().numpy().tobytes()
      )
      prev_base = base
      prev_linear_residual = linear_residual
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
