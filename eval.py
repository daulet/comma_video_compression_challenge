#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import shutil
import zipfile
from pathlib import Path

import torch
from tqdm import tqdm

from compress import compress_videos, inflate_archive, load_video_names
from frame_utils import DaliVideoDataset, AVVideoDataset, TensorVideoDataset, camera_size, seq_len
from modules import DistortionNet, segnet_sd_path, posenet_sd_path

HERE = Path(__file__).resolve().parent


def _pick_device(device_arg: str | None) -> torch.device:
  if device_arg is not None:
    return torch.device(device_arg)
  if torch.cuda.is_available():
    return torch.device("cuda", int(os.getenv("LOCAL_RANK", "0")))
  if torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


def _prepare_workdir(work_dir: Path) -> tuple[Path, Path, Path]:
  archive_build_dir = work_dir / "archive_build"
  archive_dir = work_dir / "archive"
  inflated_dir = work_dir / "inflated"
  if work_dir.exists():
    shutil.rmtree(work_dir)
  work_dir.mkdir(parents=True, exist_ok=True)
  return archive_build_dir, archive_dir, inflated_dir


def _unzip_archive(archive_zip: Path, archive_dir: Path) -> None:
  archive_dir.mkdir(parents=True, exist_ok=True)
  with zipfile.ZipFile(archive_zip, "r") as zf:
    zf.extractall(archive_dir)


def _evaluate_distortion(
  *,
  device: torch.device,
  video_names: list[str],
  uncompressed_dir: Path,
  inflated_dir: Path,
  batch_size: int,
  num_threads: int,
  prefetch_queue_depth: int,
  seed: int,
) -> tuple[float, float, int]:
  if device.type == "cuda":
    default_dataset_class = DaliVideoDataset
  else:
    default_dataset_class = AVVideoDataset

  distortion_net = DistortionNet().eval().to(device=device)
  distortion_net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)

  ds_gt = default_dataset_class(
    video_names,
    data_dir=uncompressed_dir,
    batch_size=batch_size,
    device=device,
    num_threads=num_threads,
    seed=seed,
    prefetch_queue_depth=prefetch_queue_depth,
  )
  ds_gt.prepare_data()
  dl_gt = torch.utils.data.DataLoader(ds_gt, batch_size=None, num_workers=0)

  ds_comp = TensorVideoDataset(
    video_names,
    data_dir=inflated_dir,
    batch_size=batch_size,
    device=device,
    num_threads=num_threads,
    seed=seed,
    prefetch_queue_depth=prefetch_queue_depth,
  )
  ds_comp.prepare_data()
  dl_comp = torch.utils.data.DataLoader(ds_comp, batch_size=None, num_workers=0)

  posenet_dists = torch.zeros([], device=device)
  segnet_dists = torch.zeros([], device=device)
  batch_sizes = torch.zeros([], device=device)

  with torch.inference_mode():
    for (_, _, batch_gt), (_, _, batch_comp) in tqdm(zip(dl_gt, dl_comp)):
      batch_gt = batch_gt.to(device)
      batch_comp = batch_comp.to(device)
      expected_shape = [seq_len, camera_size[1], camera_size[0], 3]
      assert list(batch_comp.shape)[1:] == expected_shape, f"unexpected batch shape: {batch_comp.shape}"
      assert batch_gt.shape == batch_comp.shape, (
        f"ground truth and compressed batch shape mismatch: {batch_gt.shape} vs {batch_comp.shape}"
      )
      posenet_dist, segnet_dist = distortion_net.compute_distortion(batch_gt, batch_comp)
      assert posenet_dist.shape == (batch_gt.shape[0],) and segnet_dist.shape == (batch_gt.shape[0],), (
        f"unexpected distortion shapes: {posenet_dist.shape}, {segnet_dist.shape}"
      )
      posenet_dists += posenet_dist.sum()
      segnet_dists += segnet_dist.sum()
      batch_sizes += batch_gt.shape[0]

  n_samples = int(batch_sizes.item())
  return (posenet_dists / batch_sizes).item(), (segnet_dists / batch_sizes).item(), n_samples


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Autoresearch eval harness for comma video compression challenge."
  )
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--num-threads", type=int, default=2)
  parser.add_argument("--prefetch-queue-depth", type=int, default=4)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--device", type=str, default="cpu", help="cpu|cuda|mps")
  parser.add_argument("--video-names-file", type=Path, default=HERE / "public_test_video_names.txt")
  parser.add_argument("--uncompressed-dir", type=Path, default=HERE / "videos")
  parser.add_argument("--work-dir", type=Path, default=HERE / "autoresearch_work")
  parser.add_argument("--report", type=Path, default=HERE / "autoresearch_work" / "report.txt")
  parser.add_argument("--allow-multi-video", action="store_true")
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  enforce_single_video = not args.allow_multi_video

  device = _pick_device(args.device)
  video_names = load_video_names(args.video_names_file)

  if enforce_single_video and len(video_names) != 1:
    raise ValueError(
      f"Expected exactly one video in {args.video_names_file}, found {len(video_names)}."
    )

  archive_build_dir, archive_dir, inflated_dir = _prepare_workdir(args.work_dir)
  archive_zip = args.work_dir / "archive.zip"

  print("=== Evaluation config ===")
  print(f"  batch_size: {args.batch_size}")
  print(f"  device: {device}")
  print(f"  num_threads: {args.num_threads}")
  print(f"  prefetch_queue_depth: {args.prefetch_queue_depth}")
  print(f"  seed: {args.seed}")
  print(f"  video_names_file: {args.video_names_file}")
  print(f"  uncompressed_dir: {args.uncompressed_dir}")
  print(f"  work_dir: {args.work_dir}")
  print(f"  report: {args.report}")

  compress_videos(
    in_dir=args.uncompressed_dir,
    video_names=video_names,
    archive_dir=archive_build_dir,
    archive_zip=archive_zip,
  )
  _unzip_archive(archive_zip, archive_dir)
  inflate_archive(archive_dir, inflated_dir, video_names)

  posenet_dist, segnet_dist, n_samples = _evaluate_distortion(
    device=device,
    video_names=video_names,
    uncompressed_dir=args.uncompressed_dir,
    inflated_dir=inflated_dir,
    batch_size=args.batch_size,
    num_threads=args.num_threads,
    prefetch_queue_depth=args.prefetch_queue_depth,
    seed=args.seed,
  )

  compressed_size = archive_zip.stat().st_size
  uncompressed_size = sum(file.stat().st_size for file in args.uncompressed_dir.rglob("*") if file.is_file())
  rate = compressed_size / uncompressed_size
  score = 100 * segnet_dist + math.sqrt(posenet_dist * 10) + 25 * rate

  lines = [
    f"=== Evaluation results over {n_samples} samples ===",
    f"  Average PoseNet Distortion: {posenet_dist:.8f}",
    f"  Average SegNet Distortion: {segnet_dist:.8f}",
    f"  Submission file size: {compressed_size:,} bytes",
    f"  Original uncompressed size: {uncompressed_size:,} bytes",
    f"  Compression Rate: {rate:.8f}",
    f"  Final score: 100*segnet_dist + √(10*posenet_dist) + 25*rate = {score:.2f}",
    "---",
    f"score: {score:.8f}",
    f"segnet_dist: {segnet_dist:.8f}",
    f"posenet_dist: {posenet_dist:.8f}",
    f"rate: {rate:.8f}",
    f"archive_bytes: {compressed_size}",
  ]
  print("\n".join(lines))

  args.report.parent.mkdir(parents=True, exist_ok=True)
  with args.report.open("w") as f:
    f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
  main()
