#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

import av
import torch
import torch.nn.functional as F

from frame_utils import camera_size, yuv420_to_rgb

HERE = Path(__file__).resolve().parent


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
      "-c:v",
      "libx265",
      "-preset",
      "ultrafast",
      "-crf",
      str(crf),
      "-g",
      "1",
      "-bf",
      "0",
      "-x265-params",
      "keyint=1:min-keyint=1:scenecut=0:frame-threads=4:log-level=warning",
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

  archive_zip.parent.mkdir(parents=True, exist_ok=True)
  if archive_zip.exists():
    archive_zip.unlink()
  with zipfile.ZipFile(archive_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(archive_dir.rglob("*")):
      if path.is_file():
        zf.write(path, arcname=str(path.relative_to(archive_dir)))
  print(f"Compressed to {archive_zip}")


def _decode_and_resize_to_raw(src: Path, dst: Path) -> int:
  target_w, target_h = camera_size
  fmt = "hevc" if src.suffix == ".hevc" else None
  container = av.open(str(src), format=fmt)
  stream = container.streams.video[0]
  dst.parent.mkdir(parents=True, exist_ok=True)
  n = 0
  with dst.open("wb") as f:
    for frame in container.decode(stream):
      t = yuv420_to_rgb(frame)  # (H, W, 3), uint8
      h, w, _ = t.shape
      if h != target_h or w != target_w:
        x = t.permute(2, 0, 1).unsqueeze(0).float()
        x = F.interpolate(x, size=(target_h, target_w), mode="bicubic", align_corners=False)
        t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().to(torch.uint8)
      f.write(t.contiguous().numpy().tobytes())
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
    print(f"Decoding + resizing {src} -> {dst}")
    n = _decode_and_resize_to_raw(src, dst)
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
