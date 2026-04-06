#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import random
import shutil
import subprocess
import zipfile
from pathlib import Path

import av
import torch
import torch.nn as nn
import torch.nn.functional as F

from frame_utils import camera_size, yuv420_to_rgb, segnet_model_input_size

HERE = Path(__file__).resolve().parent

UNSHARP_KERNEL = torch.tensor([
  [1., 8., 28., 56., 70., 56., 28., 8., 1.],
  [8., 64., 224., 448., 560., 448., 224., 64., 8.],
  [28., 224., 784., 1568., 1960., 1568., 784., 224., 28.],
  [56., 448., 1568., 3136., 3920., 3136., 1568., 448., 56.],
  [70., 560., 1960., 3920., 4900., 3920., 1960., 560., 70.],
  [56., 448., 1568., 3136., 3920., 3136., 1568., 448., 56.],
  [28., 224., 784., 1568., 1960., 1568., 784., 224., 28.],
  [8., 64., 224., 448., 560., 448., 224., 64., 8.],
  [1., 8., 28., 56., 70., 56., 28., 8., 1.],
], dtype=torch.float32) / 65536.0

SCALE_FACTOR = 0.45
CRF = 34
GAMMA_BOOST = 1.0


class LinearFilter(nn.Module):
  def __init__(self, kernel_size: int = 9):
    super().__init__()
    self.kernel_size = kernel_size
    self.pad = kernel_size // 2
    self.weight = nn.Parameter(torch.zeros(3, 1, kernel_size, kernel_size))
    with torch.no_grad():
      self.weight[:, :, kernel_size // 2, kernel_size // 2] = 1.0
      unsharp_k = UNSHARP_KERNEL.unsqueeze(0).expand(3, 1, 9, 9)
      identity = torch.zeros(3, 1, 9, 9)
      identity[:, :, 4, 4] = 1.0
      self.weight.copy_(1.85 * identity - 0.85 * unsharp_k)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x_padded = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="reflect")
    return F.conv2d(x_padded, self.weight, groups=3)


def _count_params(model: nn.Module) -> int:
  return sum(p.numel() for p in model.parameters())


def load_video_names(video_names_file: Path) -> list[str]:
  return [line.strip() for line in video_names_file.read_text().splitlines() if line.strip()]


def _reset_dir(path: Path) -> None:
  if path.exists():
    shutil.rmtree(path)
  path.mkdir(parents=True, exist_ok=True)


def _run(cmd: list[str]) -> None:
  subprocess.run(cmd, check=True)


def _decode_video_frames(src: Path) -> list[torch.Tensor]:
  fmt = "hevc" if src.suffix == ".hevc" else None
  container = av.open(str(src), format=fmt)
  stream = container.streams.video[0]
  frames = [yuv420_to_rgb(frame) for frame in container.decode(stream)]
  container.close()
  return frames


def _bicubic_upsample(t: torch.Tensor, target_h: int, target_w: int, inverse_gamma: bool = True) -> torch.Tensor:
  x = t.permute(2, 0, 1).unsqueeze(0).float()
  if x.shape[2] != target_h or x.shape[3] != target_w:
    x = F.interpolate(x, size=(target_h, target_w), mode="bicubic", align_corners=False)
  x = x.clamp(0, 255)
  if inverse_gamma and GAMMA_BOOST != 1.0:
    x = (x / 255.0).pow(GAMMA_BOOST) * 255.0
    x = x.clamp(0, 255)
  return x


def _apply_unsharp(x: torch.Tensor, strength: float = 0.85) -> torch.Tensor:
  kernel = UNSHARP_KERNEL.expand(3, 1, 9, 9)
  x_padded = F.pad(x, (4, 4, 4, 4), mode="reflect")
  blur = F.conv2d(x_padded, kernel, groups=3)
  return (x + strength * (x - blur)).clamp(0, 255)


def _train_filter_with_eval_loss(
  original_frames: list[torch.Tensor],
  upsampled_chw: list[torch.Tensor],
  n_iters: int = 300,
  batch_size: int = 2,
  lr: float = 1e-3,
) -> LinearFilter:
  from modules import DistortionNet, posenet_sd_path, segnet_sd_path

  eval_h, eval_w = segnet_model_input_size[1], segnet_model_input_size[0]
  n_frames = len(original_frames)

  print("  Loading eval models...", flush=True)
  distortion_net = DistortionNet().eval()
  distortion_net.load_state_dicts(posenet_sd_path, segnet_sd_path, "cpu")
  for p in distortion_net.parameters():
    p.requires_grad_(False)

  print("  Preparing eval-resolution frames...", flush=True)
  original_eval = []
  upsampled_eval = []
  for i in range(n_frames):
    o = original_frames[i].permute(2, 0, 1).unsqueeze(0).float()
    o = F.interpolate(o, size=(eval_h, eval_w), mode="bilinear", align_corners=False)
    original_eval.append(o)
    u = F.interpolate(upsampled_chw[i], size=(eval_h, eval_w), mode="bilinear", align_corners=False)
    upsampled_eval.append(u)

  print("  Pre-computing original eval outputs...", flush=True)
  orig_posenet_outs = []
  orig_segnet_outs = []
  with torch.no_grad():
    for i in range(n_frames - 1):
      pair = torch.stack([
        original_eval[i].squeeze(0).permute(1, 2, 0),
        original_eval[i + 1].squeeze(0).permute(1, 2, 0),
      ], dim=0).unsqueeze(0)
      po, so = distortion_net(pair)
      orig_posenet_outs.append({k: v.detach() for k, v in po.items()})
      orig_segnet_outs.append(so.detach())
  print(f"  Pre-computed {len(orig_posenet_outs)} pairs", flush=True)

  filt = LinearFilter(kernel_size=9)
  print(f"  LinearFilter: {_count_params(filt)} params (init from unsharp)", flush=True)
  optimizer = torch.optim.Adam(filt.parameters(), lr=lr)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iters)

  filt.train()
  running_loss = 0.0
  running_pose = 0.0
  running_seg = 0.0

  for it in range(n_iters):
    pair_indices = [random.randint(0, n_frames - 2) for _ in range(batch_size)]

    filtered_pairs = []
    orig_pose_batch = []
    orig_seg_batch = []

    for i in pair_indices:
      f_i = filt(upsampled_eval[i])
      f_i1 = filt(upsampled_eval[i + 1])

      pair = torch.stack([
        f_i.squeeze(0).permute(1, 2, 0),
        f_i1.squeeze(0).permute(1, 2, 0),
      ], dim=0).unsqueeze(0)
      filtered_pairs.append(pair)
      orig_pose_batch.append(orig_posenet_outs[i])
      orig_seg_batch.append(orig_segnet_outs[i])

    filtered_batch = torch.cat(filtered_pairs, dim=0)

    enh_po, enh_so = distortion_net(filtered_batch)

    orig_pose = torch.cat([orig_pose_batch[j]["pose"] for j in range(batch_size)])
    loss_posenet = F.mse_loss(enh_po["pose"][:, :6], orig_pose[:, :6])

    orig_seg = torch.cat(orig_seg_batch)
    orig_classes = orig_seg.argmax(dim=1)
    loss_segnet = F.cross_entropy(enh_so, orig_classes)

    loss = 10.0 * loss_posenet + 1.0 * loss_segnet

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    running_loss += loss.item()
    running_pose += loss_posenet.item()
    running_seg += loss_segnet.item()

    if (it + 1) % 50 == 0:
      n = 50
      print(
        f"  iter {it+1}/{n_iters}: loss={running_loss/n:.4f} pose={running_pose/n:.6f} seg={running_seg/n:.4f} lr={scheduler.get_last_lr()[0]:.6f}",
        flush=True,
      )
      running_loss = running_pose = running_seg = 0.0

  filt.eval()
  return filt


def _save_filter(filt: LinearFilter, path: Path) -> int:
  state = {k: v.half() for k, v in filt.state_dict().items()}
  buf = io.BytesIO()
  torch.save(state, buf)
  data = buf.getvalue()
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(data)
  print(f"Saved LinearFilter: {len(data)} bytes", flush=True)
  return len(data)


def _load_filter(path: Path) -> LinearFilter:
  state = torch.load(path, map_location="cpu", weights_only=True)
  state = {k: v.float() for k, v in state.items()}
  filt = LinearFilter(kernel_size=9)
  filt.load_state_dict(state)
  filt.eval()
  return filt


SVTAV1_ENC = Path("/tmp/SVT-AV1/Bin/Release/SvtAv1EncApp")


def _build_road_mask(frame_idx: int, width: int, height: int, feather: int = 48) -> torch.Tensor:
  from PIL import Image, ImageDraw, ImageFilter

  segments = [
    (0, 299, [(0.14, 0.52), (0.82, 0.48), (0.98, 1.00), (0.05, 1.00)]),
    (300, 599, [(0.10, 0.50), (0.76, 0.47), (0.92, 1.00), (0.00, 1.00)]),
    (600, 899, [(0.18, 0.50), (0.84, 0.47), (0.98, 1.00), (0.06, 1.00)]),
    (900, 1199, [(0.22, 0.52), (0.90, 0.49), (1.00, 1.00), (0.10, 1.00)]),
  ]
  poly = [(0.15 * width, 0.52 * height), (0.85 * width, 0.48 * height), (width, height), (0, height)]
  for start, end, p in segments:
    if start <= frame_idx <= end:
      poly = [(x * width, y * height) for x, y in p]
      break
  img = Image.new("L", (width, height), 0)
  ImageDraw.Draw(img).polygon(poly, fill=255)
  if feather > 0:
    img = img.filter(ImageFilter.GaussianBlur(radius=feather))
  mask = torch.frombuffer(memoryview(img.tobytes()), dtype=torch.uint8).clone().view(height, width).float() / 255.0
  return mask.unsqueeze(0).unsqueeze(0)


def _roi_preprocess_video(src: Path, dst: Path, denoise_strength: float = 2.5, blend: float = 0.60) -> None:
  container = av.open(str(src))
  stream = container.streams.video[0]
  w, h = stream.width, stream.height

  out = av.open(str(dst), mode="w")
  out_stream = out.add_stream("ffv1", rate=20)
  out_stream.width, out_stream.height, out_stream.pix_fmt = w, h, "yuv420p"

  ks = 3 if denoise_strength <= 2.0 else 5
  sigma = max(0.1, denoise_strength * 0.35)
  coords = torch.arange(ks) - ks // 2
  g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
  kernel_1d = (g / g.sum()).float()
  kernel_2d = torch.outer(kernel_1d, kernel_1d).view(1, 1, ks, ks)
  luma_blend = min(0.9, denoise_strength / 3.0)

  for idx, frame in enumerate(container.decode(stream)):
    rgb = yuv420_to_rgb(frame)
    x = rgb.permute(2, 0, 1).float().unsqueeze(0)

    mask = _build_road_mask(idx, w, h)
    outside_alpha = (1.0 - mask) * blend

    r, g_ch, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y = 0.299 * r + 0.587 * g_ch + 0.114 * b
    u = (b - y) / 1.772 + 128.0
    v = (r - y) / 1.402 + 128.0

    y_blur = F.conv2d(y, kernel_2d, padding=ks // 2)
    y_denoised = (1 - luma_blend) * y + luma_blend * y_blur
    u_blur = F.avg_pool2d(u, kernel_size=5, stride=1, padding=2)
    v_blur = F.avg_pool2d(v, kernel_size=5, stride=1, padding=2)

    r2 = y_denoised + 1.402 * (v_blur - 128.0)
    g2 = y_denoised - 0.344136 * (u_blur - 128.0) - 0.714136 * (v_blur - 128.0)
    b2 = y_denoised + 1.772 * (u_blur - 128.0)
    denoised = torch.cat([r2, g2, b2], dim=1)

    mixed = x * (1.0 - outside_alpha) + denoised * outside_alpha
    out_rgb = mixed.clamp(0, 255).round().to(torch.uint8).squeeze(0).permute(1, 2, 0).cpu().numpy()

    vf = av.VideoFrame.from_ndarray(out_rgb, format="rgb24")
    for pkt in out_stream.encode(vf):
      out.mux(pkt)

  for pkt in out_stream.encode():
    out.mux(pkt)
  out.close()
  container.close()
  print(f"  ROI preprocessed {idx+1} frames", flush=True)


def _encode_one_video(src: Path, dst: Path, scale_factor: float, crf: int) -> None:
  dst.parent.mkdir(parents=True, exist_ok=True)
  vf = f"scale=trunc(iw*{scale_factor}/2)*2:trunc(ih*{scale_factor}/2)*2:flags=lanczos,hqdn3d=1.5:0:0:0"

  if SVTAV1_ENC.exists():
    import subprocess as sp

    ivf = dst.with_suffix(".ivf")
    ow, oh = camera_size
    sw = int(ow * scale_factor) // 2 * 2
    sh = int(oh * scale_factor) // 2 * 2

    ffmpeg_cmd = (
      f'ffmpeg -nostdin -y -hide_banner -loglevel warning '
      f'-r 20 -fflags +genpts -i "{src}" '
      f'-vf "{vf}" -pix_fmt yuv420p -f rawvideo -r 20 pipe:1'
    )
    svt_cmd = (
      f'"{SVTAV1_ENC}" -i stdin -b "{ivf}" '
      f'-w {sw} -h {sh} --fps 20 --input-depth 8 '
      f'--preset 0 --crf {crf} --keyint 180 --scd 0 '
      f'--film-grain 22 --film-grain-denoise 0 '
      f'--lp {min(8, len(__import__("os").sched_getaffinity(0)))}'
    )
    sp.run(f"{ffmpeg_cmd} | {svt_cmd}", shell=True, check=True)

    _run([
      "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "warning",
      "-i", str(ivf), "-c:v", "copy", "-r", "20", str(dst),
    ])
    ivf.unlink()
  else:
    _run([
      "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "warning",
      "-r", "20", "-fflags", "+genpts", "-i", str(src),
      "-vf", vf,
      "-c:v", "libx265", "-preset", "slow", "-crf", str(crf),
      "-x265-params", "keyint=180:min-keyint=1:scenecut=40:bframes=4:b-adapt=2:rc-lookahead=40:frame-threads=4:log-level=warning",
      "-r", "20", str(dst),
    ])


def compress_videos(
  in_dir: Path,
  video_names: list[str],
  archive_dir: Path,
  archive_zip: Path,
  *,
  scale_factor: float = SCALE_FACTOR,
  crf: int = CRF,
) -> None:
  _reset_dir(archive_dir)
  target_w, target_h = camera_size

  for rel in video_names:
    base = Path(rel).with_suffix("")
    src = in_dir / rel
    dst = archive_dir / f"{base}.mkv"
    if not src.exists():
      raise FileNotFoundError(f"Missing source video: {src}")

    roi_tmp = archive_dir / f"{base}.roi.mkv"
    print(f"ROI preprocessing {src}...", flush=True)
    _roi_preprocess_video(src, roi_tmp)

    print(f"Encoding -> {dst}", flush=True)
    _encode_one_video(roi_tmp, dst, scale_factor=scale_factor, crf=crf)
    roi_tmp.unlink()

    print("Decoding original frames...", flush=True)
    original_frames = _decode_video_frames(in_dir / rel)
    print(f"  {len(original_frames)} frames", flush=True)

    print("Decoding + upsampling compressed frames...", flush=True)
    compressed_frames = _decode_video_frames(dst)
    upsampled_chw = [_bicubic_upsample(f, target_h, target_w) for f in compressed_frames]
    print(f"  {len(compressed_frames)} frames", flush=True)

    print("Training LinearFilter with eval model loss...", flush=True)
    filt = _train_filter_with_eval_loss(original_frames, upsampled_chw)

    _save_filter(filt, archive_dir / "linear_filter.pt")
    del original_frames, compressed_frames, upsampled_chw

  archive_zip.parent.mkdir(parents=True, exist_ok=True)
  if archive_zip.exists():
    archive_zip.unlink()
  with zipfile.ZipFile(archive_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(archive_dir.rglob("*")):
      if path.is_file():
        zf.write(path, arcname=str(path.relative_to(archive_dir)))
  print(f"Archive: {archive_zip} ({archive_zip.stat().st_size:,} bytes)", flush=True)


def _decode_and_restore_to_raw(src: Path, dst: Path, filt: LinearFilter | None) -> int:
  target_w, target_h = camera_size
  fmt = "hevc" if src.suffix == ".hevc" else None
  container = av.open(str(src), format=fmt)
  stream = container.streams.video[0]
  dst.parent.mkdir(parents=True, exist_ok=True)

  n = 0
  with torch.inference_mode(), dst.open("wb") as f:
    for frame in container.decode(stream):
      t = yuv420_to_rgb(frame)
      x = _bicubic_upsample(t, target_h, target_w)

      if filt is not None:
        x = filt(x)
      else:
        x = _apply_unsharp(x)

      out = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().to(torch.uint8)
      f.write(out.contiguous().numpy().tobytes())
      n += 1

  container.close()
  return n


def inflate_archive(archive_dir: Path, output_dir: Path, video_names: list[str]) -> None:
  _reset_dir(output_dir)
  filt_path = archive_dir / "linear_filter.pt"
  filt = _load_filter(filt_path) if filt_path.exists() else None
  if filt:
    print(f"Loaded LinearFilter ({_count_params(filt)} params)", flush=True)

  for rel in video_names:
    base = Path(rel).with_suffix("")
    src = archive_dir / f"{base}.mkv"
    dst = output_dir / f"{base}.raw"
    if not src.exists():
      raise FileNotFoundError(f"Missing encoded video in archive: {src}")
    print(f"Decoding + restoring {src} -> {dst}", flush=True)
    n = _decode_and_restore_to_raw(src, dst, filt)
    print(f"Saved {n} frames", flush=True)


def create_viewable_video(raw_path: Path, output_path: Path, fps: int = 20) -> None:
  w, h = camera_size
  output_path.parent.mkdir(parents=True, exist_ok=True)
  _run([
    "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "warning",
    "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
    "-i", str(raw_path),
    "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
    "-r", str(fps), str(output_path),
  ])
  print(f"Viewable video: {output_path} ({output_path.stat().st_size:,} bytes)", flush=True)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(dest="command", required=True)
  cp = subparsers.add_parser("compress")
  cp.add_argument("--in-dir", type=Path, default=HERE / "videos")
  cp.add_argument("--video-names-file", type=Path, default=HERE / "public_test_video_names.txt")
  cp.add_argument("--archive-dir", type=Path, default=HERE / "autoresearch_work" / "archive_build")
  cp.add_argument("--archive-zip", type=Path, default=HERE / "autoresearch_work" / "archive.zip")
  cp.add_argument("--scale-factor", type=float, default=SCALE_FACTOR)
  cp.add_argument("--crf", type=int, default=CRF)
  ip = subparsers.add_parser("inflate")
  ip.add_argument("--archive-dir", type=Path, required=True)
  ip.add_argument("--out-dir", type=Path, default=HERE / "autoresearch_work" / "inflated")
  ip.add_argument("--video-names-file", type=Path, default=HERE / "public_test_video_names.txt")
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
  elif args.command == "inflate":
    inflate_archive(archive_dir=args.archive_dir, output_dir=args.out_dir, video_names=video_names)


if __name__ == "__main__":
  main()
