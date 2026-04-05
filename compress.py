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
MLP_REFINER_FEATURES = 16
MLP_REFINER_HIDDEN = 4
MLP_REFINER_SAMPLE_STRIDE = 32
MLP_REFINER_MAX_SAMPLES = 200_000
MLP_REFINER_STEPS = 128
MLP_REFINER_BATCH_SIZE = 8192
MLP_REFINER_LR = 3e-2
MLP_REFINER_RESIDUAL_CLAMP = 10.0 / 255.0
MLP_INIT_W1 = [
  [-0.05078125, -0.0341796875, -0.006595611572265625, -0.01413726806640625],
  [0.043212890625, 0.0772705078125, -0.0184478759765625, -0.088623046875],
  [0.00907135009765625, -0.01983642578125, 0.0024166107177734375, 0.037017822265625],
  [-0.10430908203125, -0.0701904296875, -0.0269775390625, -0.048004150390625],
  [-0.099853515625, 0.0287322998046875, 0.09033203125, 0.0594482421875],
  [-0.0181427001953125, 0.03912353515625, 0.037017822265625, 0.10394287109375],
  [0.03253173828125, -0.024566650390625, -0.032562255859375, -0.0311279296875],
  [0.05718994140625, 0.091064453125, 0.010284423828125, -0.0733642578125],
  [-0.02752685546875, -0.0072479248046875, -0.0267791748046875, -0.042388916015625],
  [0.047332763671875, -0.1612548828125, -0.07086181640625, 0.051788330078125],
  [0.09564208984375, 0.1793212890625, 0.06500244140625, 0.058441162109375],
  [0.09320068359375, -0.053741455078125, -0.1165771484375, -0.0460205078125],
  [0.041748046875, 1.349609375, -0.056396484375, 0.67236328125],
  [0.045623779296875, -1.5673828125, 0.2034912109375, -0.7255859375],
  [-0.046539306640625, -0.002361297607421875, 0.37890625, -0.046722412109375],
  [-0.06695556640625, -0.047821044921875, 0.00921630859375, -0.0160675048828125],
]
MLP_INIT_B1 = [0.00962066650390625, 0.013885498046875, -0.081787109375, 0.01151275634765625]
MLP_INIT_W2 = [
  [-0.0234832763671875, -0.006908416748046875, 0.057464599609375],
  [0.07470703125, 0.01273345947265625, 0.009918212890625],
  [-0.038970947265625, -0.030731201171875, 0.07421875],
  [0.018829345703125, 0.0258026123046875, -0.0037899017333984375],
]
MLP_INIT_B2 = [-0.00275421142578125, -0.0018014907836914062, 0.00930023193359375]


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
) -> torch.Tensor:
  delta = base - prev_base
  prev_delta = prev_base - prev_prev_base
  linear = linear_residual.permute(2, 0, 1).unsqueeze(0)
  bias = torch.ones((1, 1, base.shape[2], base.shape[3]), dtype=base.dtype, device=base.device)
  return torch.cat([base, delta, delta.abs(), prev_delta, linear, bias], dim=1)


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


def _fit_mlp_temporal_refiner(features: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, ...]:
  if features.shape[0] > MLP_REFINER_MAX_SAMPLES:
    generator = torch.Generator().manual_seed(0)
    keep = torch.randperm(features.shape[0], generator=generator)[:MLP_REFINER_MAX_SAMPLES]
    features = features[keep]
    targets = targets[keep]

  generator = torch.Generator().manual_seed(0)
  w1 = torch.tensor(MLP_INIT_W1, dtype=torch.float32, requires_grad=True)
  b1 = torch.tensor(MLP_INIT_B1, dtype=torch.float32, requires_grad=True)
  w2 = torch.tensor(MLP_INIT_W2, dtype=torch.float32, requires_grad=True)
  b2 = torch.tensor(MLP_INIT_B2, dtype=torch.float32, requires_grad=True)
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
  for gt, base in zip(_iter_video_tensors(src), _iter_video_tensors(encoded)):
    if prev_base is None:
      prev_base = base
    if prev_prev_base is None:
      prev_prev_base = prev_base
    linear_feats = _linear_refiner_features(base, prev_base)
    linear_residual = _predict_linear_refiner_residual(linear_feats, linear_weights)
    linear_refined = _refine_frame(base, linear_residual)
    feats = _mlp_refiner_features(base, prev_base, prev_prev_base, linear_residual)
    target = (gt - linear_refined).clamp_(-MLP_REFINER_RESIDUAL_CLAMP, MLP_REFINER_RESIDUAL_CLAMP)
    x = feats[..., ::MLP_REFINER_SAMPLE_STRIDE, ::MLP_REFINER_SAMPLE_STRIDE]
    y = target[..., ::MLP_REFINER_SAMPLE_STRIDE, ::MLP_REFINER_SAMPLE_STRIDE]
    mlp_features.append(x.squeeze(0).permute(1, 2, 0).reshape(-1, MLP_REFINER_FEATURES))
    mlp_targets.append(y.squeeze(0).permute(1, 2, 0).reshape(-1, 3))
    prev_prev_base = prev_base
    prev_base = base

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
  refiner: TemporalRefiner | None,
) -> torch.Tensor:
  if refiner is None:
    return base
  linear_features = _linear_refiner_features(base, prev_base)
  linear_residual = _predict_linear_refiner_residual(linear_features, refiner.linear)
  mlp_features = _mlp_refiner_features(base, prev_base, prev_prev_base, linear_residual)
  residual = linear_residual + _predict_mlp_refiner_residual(mlp_features, refiner)
  return _refine_frame(base, residual)


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
  with dst.open("wb") as f:
    for frame in container.decode(stream):
      base = _resize_rgb(yuv420_to_rgb(frame)).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
      if prev_base is None:
        prev_base = base
      if prev_prev_base is None:
        prev_prev_base = prev_base
      t = _apply_temporal_refiner(base, prev_base, prev_prev_base, weights)
      prev_prev_base = prev_base
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
