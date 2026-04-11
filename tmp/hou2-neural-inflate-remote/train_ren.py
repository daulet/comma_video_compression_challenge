#!/usr/bin/env python3
"""Train REN (Residual Enhancement Network) for neural_inflate.

This script trains on (compressed -> original) frame pairs and optimizes
against challenge proxies (PoseNet + SegNet) plus temporal smoothness.

It supports multiple videos and can use any GT/compressed dataset layout,
as long as compressed files follow `<compressed_dir>/<video_name_without_ext>.mkv`.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from frame_utils import camera_size, yuv420_to_rgb  # noqa: E402
from modules import DistortionNet, posenet_sd_path, segnet_sd_path  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class REN(nn.Module):
  def __init__(self, features: int = 32):
    super().__init__()
    self.down = nn.PixelUnshuffle(2)
    self.body = nn.Sequential(
      nn.Conv2d(12, features, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(features, features, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(features, features, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(features, 12, 3, padding=1),
    )
    self.up = nn.PixelShuffle(2)
    nn.init.zeros_(self.body[-1].weight)
    nn.init.zeros_(self.body[-1].bias)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x_norm = x / 255.0
    residual = self.up(self.body(self.down(x_norm)))
    return (x_norm + residual).clamp(0, 1) * 255.0


@dataclass
class VideoPair:
  name: str
  comp_frames: list[torch.Tensor]  # each frame is HWC uint8
  gt_frames: list[torch.Tensor]    # each frame is HWC uint8


class ConsecutivePairDataset(Dataset):
  def __init__(self, stores: list[VideoPair], refs: list[tuple[int, int]]):
    self.stores = stores
    self.refs = refs

  def __len__(self) -> int:
    return len(self.refs)

  def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    vid_idx, frame_idx = self.refs[idx]
    store = self.stores[vid_idx]

    ca = store.comp_frames[frame_idx].permute(2, 0, 1).float()
    cb = store.comp_frames[frame_idx + 1].permute(2, 0, 1).float()
    ga = store.gt_frames[frame_idx].permute(2, 0, 1).float()
    gb = store.gt_frames[frame_idx + 1].permute(2, 0, 1).float()
    return ca, cb, ga, gb


@torch.no_grad()
def decode_all_frames(
  video_path: Path,
  target_size: tuple[int, int] | None = None,
  lanczos: bool = False,
  max_frames: int = 0,
  frame_stride: int = 1,
) -> list[torch.Tensor]:
  fmt = "hevc" if video_path.suffix == ".hevc" else None
  container = av.open(str(video_path), format=fmt)
  stream = container.streams.video[0]
  frames: list[torch.Tensor] = []

  target_w = target_h = None
  if target_size is not None:
    target_w, target_h = target_size

  for frame_idx, frame in enumerate(container.decode(stream)):
    if frame_stride > 1 and (frame_idx % frame_stride) != 0:
      continue

    t = yuv420_to_rgb(frame)
    if target_w and target_h and (t.shape[1] != target_w or t.shape[0] != target_h):
      if lanczos:
        pil = Image.fromarray(t.numpy())
        pil = pil.resize((target_w, target_h), Image.LANCZOS)
        t = torch.from_numpy(np.array(pil))
      else:
        t = (
          F.interpolate(
            t.permute(2, 0, 1).unsqueeze(0).float(),
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
          )
          .clamp(0, 255)
          .squeeze(0)
          .permute(1, 2, 0)
          .round()
          .to(torch.uint8)
        )

    frames.append(t)
    if max_frames > 0 and len(frames) >= max_frames:
      break

  container.close()
  return frames


def compute_loss(
  model: REN,
  posenet: nn.Module,
  segnet: nn.Module,
  comp_a: torch.Tensor,
  comp_b: torch.Tensor,
  gt_a: torch.Tensor,
  gt_b: torch.Tensor,
  w_seg: float,
  w_temp: float,
) -> tuple[torch.Tensor, float, float, float]:
  inf_a = model(comp_a)
  inf_b = model(comp_b)

  pair_inf = torch.stack([inf_a.permute(0, 2, 3, 1), inf_b.permute(0, 2, 3, 1)], dim=1)
  pair_gt = torch.stack([gt_a.permute(0, 2, 3, 1), gt_b.permute(0, 2, 3, 1)], dim=1)

  posenet_in_inf = posenet.preprocess_input(pair_inf.permute(0, 1, 4, 2, 3))
  with torch.no_grad():
    posenet_in_gt = posenet.preprocess_input(pair_gt.permute(0, 1, 4, 2, 3))
    posenet_out_gt = posenet(posenet_in_gt)
  posenet_out_inf = posenet(posenet_in_inf)
  loss_pose = sum(
    F.mse_loss(
      posenet_out_inf[h.name][..., : h.out // 2],
      posenet_out_gt[h.name][..., : h.out // 2],
    )
    for h in posenet.hydra.heads
  )

  segnet_in_inf = segnet.preprocess_input(pair_inf.permute(0, 1, 4, 2, 3))
  with torch.no_grad():
    segnet_in_gt = segnet.preprocess_input(pair_gt.permute(0, 1, 4, 2, 3))
    logits_gt = segnet(segnet_in_gt)
  logits_inf = segnet(segnet_in_inf)
  loss_seg = F.kl_div(
    F.log_softmax(logits_inf, dim=1),
    F.softmax(logits_gt, dim=1),
    reduction="batchmean",
  )

  corr_a = (inf_a - comp_a) / 255.0
  corr_b = (inf_b - comp_b) / 255.0
  loss_temp = F.l1_loss(corr_a, corr_b)

  loss = loss_pose + w_seg * loss_seg + w_temp * loss_temp
  return loss, float(loss_pose.item()), float(loss_seg.item()), float(loss_temp.item())


def _read_video_names(gt_dir: Path, video_names_file: Path | None) -> list[str]:
  if video_names_file is not None:
    if not video_names_file.exists():
      raise FileNotFoundError(f"video names file not found: {video_names_file}")
    names = [line.strip() for line in video_names_file.read_text().splitlines() if line.strip()]
    if names:
      return names

  # Fallback: use all .mkv under gt_dir
  return sorted(str(p.relative_to(gt_dir)) for p in gt_dir.rglob("*.mkv"))


def _find_compressed_path(rel_name: str, compressed_dir: Path) -> Path | None:
  rel = Path(rel_name)
  base = rel.with_suffix("")
  candidates = [
    compressed_dir / f"{base}.mkv",
    compressed_dir / rel,
    compressed_dir / f"{base}.hevc",
  ]
  for c in candidates:
    if c.exists():
      return c
  return None


def load_video_pairs(args: argparse.Namespace) -> list[VideoPair]:
  names = _read_video_names(args.gt_dir, args.video_names_file)
  if args.max_videos > 0:
    names = names[: args.max_videos]

  if not names:
    raise RuntimeError("No videos found for training")

  w, h = camera_size
  stores: list[VideoPair] = []

  print(f"Loading {len(names)} video(s)...")
  for i, rel_name in enumerate(names, start=1):
    gt_path = args.gt_dir / rel_name
    if not gt_path.exists():
      raise FileNotFoundError(f"GT video not found: {gt_path}")

    comp_path = _find_compressed_path(rel_name, args.compressed_dir)
    if comp_path is None:
      raise FileNotFoundError(
        f"Compressed counterpart not found for {rel_name} under {args.compressed_dir}"
      )

    print(f"  [{i}/{len(names)}] {rel_name}")
    print(f"    gt:   {gt_path}")
    print(f"    comp: {comp_path}")

    gt_frames = decode_all_frames(
      gt_path,
      target_size=None,
      lanczos=False,
      max_frames=args.max_frames_per_video,
      frame_stride=args.frame_stride,
    )
    comp_frames = decode_all_frames(
      comp_path,
      target_size=(w, h),
      lanczos=True,
      max_frames=args.max_frames_per_video,
      frame_stride=args.frame_stride,
    )

    n = min(len(gt_frames), len(comp_frames))
    if n < 2:
      print(f"    skipped: only {n} aligned frame(s)")
      continue

    gt_frames = gt_frames[:n]
    comp_frames = comp_frames[:n]
    stores.append(VideoPair(name=rel_name, comp_frames=comp_frames, gt_frames=gt_frames))
    print(f"    aligned frames: {n} -> pairs: {n - 1}")

  if not stores:
    raise RuntimeError("No usable videos after alignment")

  return stores


def split_refs(
  refs: list[tuple[int, int]], seed: int, val_ratio: float, min_val_pairs: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
  if len(refs) < 2:
    raise RuntimeError("Need at least 2 frame pairs total to make train/val split")

  shuffled = list(refs)
  rng = random.Random(seed)
  rng.shuffle(shuffled)

  n_val = int(round(len(shuffled) * val_ratio))
  n_val = max(n_val, min_val_pairs)
  n_val = min(n_val, len(shuffled) - 1)

  val_refs = shuffled[:n_val]
  train_refs = shuffled[n_val:]
  return train_refs, val_refs


def _to_device(x: torch.Tensor, non_blocking: bool) -> torch.Tensor:
  return x.to(DEVICE, non_blocking=non_blocking)


def train(args: argparse.Namespace) -> None:
  print(f"Device: {DEVICE}")
  print(f"Torch: {torch.__version__}")

  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed_all(args.seed)

  stores = load_video_pairs(args)

  refs: list[tuple[int, int]] = []
  for vid_idx, store in enumerate(stores):
    refs.extend((vid_idx, i) for i in range(len(store.gt_frames) - 1))

  train_refs, val_refs = split_refs(
    refs,
    seed=args.seed,
    val_ratio=args.val_ratio,
    min_val_pairs=args.min_val_pairs,
  )

  print(f"Total pairs: {len(refs)}")
  print(f"Train pairs: {len(train_refs)}")
  print(f"Val pairs:   {len(val_refs)}")

  train_ds = ConsecutivePairDataset(stores, train_refs)
  val_ds = ConsecutivePairDataset(stores, val_refs)

  pin = DEVICE.type == "cuda"
  train_loader = DataLoader(
    train_ds,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=pin,
    drop_last=True,
    persistent_workers=args.num_workers > 0,
  )
  val_loader = DataLoader(
    val_ds,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=pin,
    drop_last=False,
    persistent_workers=args.num_workers > 0,
  )

  model = REN(features=args.features).to(DEVICE)
  n_params = sum(p.numel() for p in model.parameters())
  print(f"Model params: {n_params:,}")

  print("Loading DistortionNet (frozen)...")
  distortion_net = DistortionNet().to(DEVICE).eval()
  distortion_net.load_state_dicts(posenet_sd_path, segnet_sd_path, DEVICE)
  for p in distortion_net.parameters():
    p.requires_grad_(False)
  posenet = distortion_net.posenet
  segnet = distortion_net.segnet

  optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=args.epochs,
    eta_min=args.lr * 0.01,
  )

  args.save_path.parent.mkdir(parents=True, exist_ok=True)

  # Calibrate seg weight from identity behavior unless overridden.
  ca, cb, ga, gb = train_ds[0]
  ca = ca.unsqueeze(0).to(DEVICE)
  cb = cb.unsqueeze(0).to(DEVICE)
  ga = ga.unsqueeze(0).to(DEVICE)
  gb = gb.unsqueeze(0).to(DEVICE)
  model.train()
  _, lp0, ls0, lt0 = compute_loss(model, posenet, segnet, ca, cb, ga, gb, 1.0, args.w_temp)

  w_seg = args.w_seg
  if w_seg is None:
    if ls0 <= 0:
      w_seg = 0.1
    else:
      w_seg = max(0.01, min(10.0, lp0 / ls0))

  print("Initial loss calibration:")
  print(f"  pose={lp0:.6f} seg={ls0:.6f} temp={lt0:.6f}")
  print(f"  weights: w_seg={w_seg:.4f}, w_temp={args.w_temp:.4f}")

  if DEVICE.type == "cuda":
    torch.cuda.empty_cache()

  best_val = float("inf")
  print(f"\nTraining for {args.epochs} epochs (batch_size={args.batch_size})")

  for epoch in range(1, args.epochs + 1):
    model.train()
    train_loss = train_pose = train_seg = train_temp = 0.0
    n_batches = 0

    for comp_a, comp_b, gt_a, gt_b in train_loader:
      comp_a = _to_device(comp_a, non_blocking=pin)
      comp_b = _to_device(comp_b, non_blocking=pin)
      gt_a = _to_device(gt_a, non_blocking=pin)
      gt_b = _to_device(gt_b, non_blocking=pin)

      optimizer.zero_grad(set_to_none=True)
      loss, lp, ls, lt = compute_loss(
        model,
        posenet,
        segnet,
        comp_a,
        comp_b,
        gt_a,
        gt_b,
        w_seg,
        args.w_temp,
      )
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()

      train_loss += float(loss.item())
      train_pose += lp
      train_seg += ls
      train_temp += lt
      n_batches += 1

    scheduler.step()

    denom = max(1, n_batches)
    train_loss /= denom
    train_pose /= denom
    train_seg /= denom
    train_temp /= denom

    do_val = (epoch == 1) or (epoch == args.epochs) or (epoch % args.val_every == 0)
    if not do_val:
      print(
        f"Epoch {epoch:03d}/{args.epochs} "
        f"train={train_loss:.6f} "
        f"(pose={train_pose:.6f} seg={train_seg:.6f} temp={train_temp:.6f})"
      )
      continue

    model.eval()
    val_loss = val_pose = val_seg = val_temp = 0.0
    n_val = 0
    with torch.no_grad():
      for comp_a, comp_b, gt_a, gt_b in val_loader:
        comp_a = _to_device(comp_a, non_blocking=pin)
        comp_b = _to_device(comp_b, non_blocking=pin)
        gt_a = _to_device(gt_a, non_blocking=pin)
        gt_b = _to_device(gt_b, non_blocking=pin)

        loss, lp, ls, lt = compute_loss(
          model,
          posenet,
          segnet,
          comp_a,
          comp_b,
          gt_a,
          gt_b,
          w_seg,
          args.w_temp,
        )
        val_loss += float(loss.item())
        val_pose += lp
        val_seg += ls
        val_temp += lt
        n_val += 1

    val_denom = max(1, n_val)
    val_loss /= val_denom
    val_pose /= val_denom
    val_seg /= val_denom
    val_temp /= val_denom

    marker = ""
    if val_loss < best_val:
      best_val = val_loss
      torch.save(model.state_dict(), args.save_path)
      marker = "  <- saved"

    print(
      f"Epoch {epoch:03d}/{args.epochs} "
      f"train={train_loss:.6f} (pose={train_pose:.6f} seg={train_seg:.6f} temp={train_temp:.6f}) "
      f"val={val_loss:.6f} (pose={val_pose:.6f} seg={val_seg:.6f} temp={val_temp:.6f}) "
      f"lr={scheduler.get_last_lr()[0]:.2e}{marker}"
    )

  size_kb = args.save_path.stat().st_size / 1024
  print(f"\nBest val_loss: {best_val:.6f}")
  print(f"Saved model: {args.save_path} ({size_kb:.1f} KB)")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Train REN for neural_inflate")
  parser.add_argument("--epochs", type=int, default=60)
  parser.add_argument("--batch-size", type=int, default=1)
  parser.add_argument("--lr", type=float, default=5e-4)
  parser.add_argument("--features", type=int, default=32)
  parser.add_argument("--num-workers", type=int, default=0)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--val-every", type=int, default=5)
  parser.add_argument("--val-ratio", type=float, default=0.1)
  parser.add_argument("--min-val-pairs", type=int, default=64)
  parser.add_argument("--w-seg", type=float, default=None)
  parser.add_argument("--w-temp", type=float, default=0.005)

  parser.add_argument("--gt-dir", type=Path, default=ROOT / "videos")
  parser.add_argument("--compressed-dir", type=Path, default=HERE / "archive")
  parser.add_argument("--video-names-file", type=Path, default=ROOT / "public_test_video_names.txt")
  parser.add_argument("--max-videos", type=int, default=0)
  parser.add_argument("--max-frames-per-video", type=int, default=0)
  parser.add_argument("--frame-stride", type=int, default=1)

  parser.add_argument("--save-path", type=Path, default=HERE / "ren_model.pt")

  args = parser.parse_args()
  if args.batch_size <= 0:
    raise ValueError("--batch-size must be > 0")
  if args.frame_stride <= 0:
    raise ValueError("--frame-stride must be > 0")
  if not (0.0 <= args.val_ratio < 1.0):
    raise ValueError("--val-ratio must be in [0, 1)")
  return args


if __name__ == "__main__":
  train(parse_args())
