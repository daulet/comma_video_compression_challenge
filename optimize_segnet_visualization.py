#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from frame_utils import camera_size
from model_diagnostics import iter_rgb_frames
from modules import SegNet, segnet_sd_path
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
LOW_W = 512
LOW_H = 384

DEFAULT_PALETTE = np.array([
  [56, 18, 149],
  [20, 32, 235],
  [185, 76, 241],
  [214, 38, 123],
  [207, 58, 53],
], dtype=np.float32)


def pick_device(device_arg: str | None) -> torch.device:
  if device_arg:
    return torch.device(device_arg)
  if torch.cuda.is_available():
    return torch.device("cuda", 0)
  if torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


def seed_everything(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)


def _extract_scored_frames(input_path: Path, max_scored_frames: int | None) -> list[torch.Tensor]:
  scored = []
  for idx, frame in enumerate(iter_rgb_frames(input_path)):
    if idx % 2 == 1:
      scored.append(frame)
      if max_scored_frames is not None and len(scored) >= max_scored_frames:
        break
  if not scored:
    raise ValueError("No scored frames extracted")
  return scored


@dataclass
class CachedBatch:
  probs: torch.Tensor
  logits: torch.Tensor
  targets: torch.Tensor
  logit_mean: torch.Tensor
  logit_std: torch.Tensor


def _cache_targets(
  *,
  input_path: Path,
  segnet: SegNet,
  device: torch.device,
  max_scored_frames: int | None,
  batch_size: int,
) -> CachedBatch:
  scored_frames = _extract_scored_frames(input_path, max_scored_frames)
  probs_chunks = []
  logits_chunks = []
  target_chunks = []

  with torch.inference_mode():
    for start in range(0, len(scored_frames), batch_size):
      batch = scored_frames[start : start + batch_size]
      x = torch.stack([frame.permute(2, 0, 1) for frame in batch], dim=0).unsqueeze(1).float().to(device)
      logits = segnet(segnet.preprocess_input(x))
      probs = logits.softmax(dim=1)
      probs_chunks.append(probs.cpu().to(torch.float16))
      logits_chunks.append(logits.cpu().to(torch.float16))
      target_chunks.append(logits.argmax(dim=1).cpu().to(torch.uint8))

  return CachedBatch(
    probs=torch.cat(probs_chunks, dim=0),
    logits=torch.cat(logits_chunks, dim=0),
    targets=torch.cat(target_chunks, dim=0),
    logit_mean=torch.zeros(5, dtype=torch.float32),
    logit_std=torch.ones(5, dtype=torch.float32),
  )


def finalize_cache_stats(cached: CachedBatch) -> CachedBatch:
  logits_f32 = cached.logits.to(torch.float32)
  mean = logits_f32.mean(dim=(0, 2, 3))
  std = logits_f32.std(dim=(0, 2, 3)).clamp_min(1e-3)
  cached.logit_mean = mean
  cached.logit_std = std
  return cached


def select_features(
  cached: CachedBatch,
  idx: slice | torch.Tensor,
  *,
  device: torch.device,
  feature_type: str,
) -> torch.Tensor:
  if feature_type == "probs":
    return cached.probs[idx].to(device=device, dtype=torch.float32)
  if feature_type == "logits":
    logits = cached.logits[idx].to(device=device, dtype=torch.float32)
    mean = cached.logit_mean.to(device=device, dtype=torch.float32).view(1, 5, 1, 1)
    std = cached.logit_std.to(device=device, dtype=torch.float32).view(1, 5, 1, 1)
    return (logits - mean) / std
  raise ValueError(f"unsupported feature_type {feature_type}")


class PaletteRenderer(nn.Module):
  def __init__(self, init_palette: np.ndarray):
    super().__init__()
    init01 = torch.tensor(init_palette.T / 255.0, dtype=torch.float32).clamp(1e-4, 1.0 - 1e-4)
    init = torch.logit(init01).view(1, 3, 5, 1, 1)
    self.colors = nn.Parameter(init)

  def forward(self, probs: torch.Tensor) -> torch.Tensor:
    colors = self.colors.sigmoid() * 255.0
    return (probs.unsqueeze(1) * colors).sum(dim=2)


class LinearRenderer(nn.Module):
  def __init__(self):
    super().__init__()
    self.proj = nn.Conv2d(5, 3, kernel_size=1)

  def forward(self, probs: torch.Tensor) -> torch.Tensor:
    return self.proj(probs).sigmoid() * 255.0


class PixelMLPRenderer(nn.Module):
  def __init__(self, hidden: int):
    super().__init__()
    self.net = nn.Sequential(
      nn.Conv2d(5, hidden, kernel_size=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(hidden, 3, kernel_size=1),
    )

  def forward(self, probs: torch.Tensor) -> torch.Tensor:
    return self.net(probs).sigmoid() * 255.0


class ConvRenderer(nn.Module):
  def __init__(self, hidden: int):
    super().__init__()
    self.net = nn.Sequential(
      nn.Conv2d(5, hidden, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(hidden, 3, kernel_size=1),
    )

  def forward(self, probs: torch.Tensor) -> torch.Tensor:
    return self.net(probs).sigmoid() * 255.0


def build_renderer(kind: str, hidden: int) -> nn.Module:
  if kind == "palette":
    return PaletteRenderer(DEFAULT_PALETTE)
  if kind == "linear":
    return LinearRenderer()
  if kind == "pixel_mlp":
    return PixelMLPRenderer(hidden)
  if kind == "conv":
    return ConvRenderer(hidden)
  raise ValueError(f"unknown renderer {kind}")


def roundtrip_lowres(
  low_rgb: torch.Tensor,
  *,
  upsample_mode: str,
) -> torch.Tensor:
  if upsample_mode == "bilinear":
    full = F.interpolate(low_rgb, size=(camera_size[1], camera_size[0]), mode="bilinear", align_corners=False)
  elif upsample_mode == "nearest":
    full = F.interpolate(low_rgb, size=(camera_size[1], camera_size[0]), mode="nearest")
  else:
    raise ValueError(f"unsupported upsample_mode {upsample_mode}")
  return F.interpolate(full, size=(LOW_H, LOW_W), mode="bilinear", align_corners=False)


def evaluate_renderer(
  *,
  segnet: SegNet,
  renderer: nn.Module,
  cached: CachedBatch,
  device: torch.device,
  batch_size: int,
  upsample_mode: str,
  feature_type: str,
) -> tuple[float, float]:
  renderer.eval()
  total_ce = 0.0
  total_err = 0.0
  total_pixels = 0
  n_items = cached.targets.shape[0]

  with torch.inference_mode():
    for start in range(0, n_items, batch_size):
      features = select_features(cached, slice(start, start + batch_size), device=device, feature_type=feature_type)
      targets = cached.targets[start : start + batch_size].to(device=device, dtype=torch.long)
      low_rgb = renderer(features)
      rerun_in = roundtrip_lowres(low_rgb, upsample_mode=upsample_mode)
      logits = segnet(rerun_in)
      ce = F.cross_entropy(logits, targets, reduction="mean")
      err = (logits.argmax(dim=1) != targets).float().mean()
      total_ce += float(ce.item()) * features.shape[0]
      total_err += float(err.item()) * features.shape[0]
      total_pixels += features.shape[0]
  return total_ce / total_pixels, total_err / total_pixels


def train_renderer(
  *,
  segnet: SegNet,
  renderer: nn.Module,
  cached: CachedBatch,
  device: torch.device,
  batch_size: int,
  epochs: int,
  lr: float,
  upsample_mode: str,
  soft_loss_weight: float,
  feature_type: str,
) -> dict:
  renderer = renderer.to(device)
  params = [p for p in renderer.parameters() if p.requires_grad]
  optimizer = torch.optim.Adam(params, lr=lr)
  n_items = cached.targets.shape[0]
  best_state = None
  best_err = math.inf
  history = []

  for epoch in range(epochs):
    renderer.train()
    order = torch.randperm(n_items)
    for start in range(0, n_items, batch_size):
      idx = order[start : start + batch_size]
      features = select_features(cached, idx, device=device, feature_type=feature_type)
      orig_logits = cached.logits[idx].to(device=device, dtype=torch.float32)
      targets = cached.targets[idx].to(device=device, dtype=torch.long)

      low_rgb = renderer(features)
      rerun_in = roundtrip_lowres(low_rgb, upsample_mode=upsample_mode)
      logits = segnet(rerun_in)
      ce = F.cross_entropy(logits, targets)
      loss = ce
      if soft_loss_weight > 0.0:
        teacher = orig_logits.softmax(dim=1)
        soft = F.kl_div(F.log_softmax(logits, dim=1), teacher, reduction="none").mean()
        loss = loss + soft_loss_weight * soft

      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()

    ce_eval, err_eval = evaluate_renderer(
      segnet=segnet,
      renderer=renderer,
      cached=cached,
      device=device,
      batch_size=batch_size,
      upsample_mode=upsample_mode,
      feature_type=feature_type,
    )
    history.append({
      "epoch": epoch + 1,
      "cross_entropy": ce_eval,
      "segnet_dist": err_eval,
      "seg_score": 100.0 * err_eval,
    })
    print(
      f"epoch={epoch + 1} ce={ce_eval:.6f} segnet_dist={err_eval:.6f} seg_score={100.0 * err_eval:.4f}",
      flush=True,
    )
    if err_eval < best_err:
      best_err = err_eval
      best_state = {k: v.detach().cpu().clone() for k, v in renderer.state_dict().items()}

  if best_state is not None:
    renderer.load_state_dict(best_state)
  ce_eval, err_eval = evaluate_renderer(
    segnet=segnet,
    renderer=renderer,
    cached=cached,
    device=device,
    batch_size=batch_size,
    upsample_mode=upsample_mode,
    feature_type=feature_type,
  )
  return {
    "best_segnet_dist": err_eval,
    "best_seg_score": 100.0 * err_eval,
    "history": history,
    "state_dict": {k: v.cpu() for k, v in renderer.state_dict().items()},
  }


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Optimize a SegNet-visualization renderer for low SegNet disagreement.")
  parser.add_argument("--input", type=Path, default=HERE / "videos" / "0.mkv")
  parser.add_argument("--renderer", choices=["palette", "linear", "pixel_mlp", "conv"], default="pixel_mlp")
  parser.add_argument("--device", type=str, default="cpu", help="cpu|cuda|mps")
  parser.add_argument("--cache-batch-size", type=int, default=8)
  parser.add_argument("--batch-size", type=int, default=4)
  parser.add_argument("--epochs", type=int, default=6)
  parser.add_argument("--hidden", type=int, default=16)
  parser.add_argument("--lr", type=float, default=3e-3)
  parser.add_argument("--soft-loss-weight", type=float, default=0.1)
  parser.add_argument("--max-scored-frames", type=int, default=64)
  parser.add_argument("--feature-type", choices=["probs", "logits"], default="probs")
  parser.add_argument("--seed", type=int, default=123)
  parser.add_argument("--upsample-mode", choices=["bilinear", "nearest"], default="bilinear")
  parser.add_argument("--report", type=Path, default=HERE / "artifacts" / "segnet_opt_report.json")
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  seed_everything(args.seed)
  torch.set_num_threads(1)
  device = pick_device(args.device)
  print(f"device={device}", flush=True)

  segnet = SegNet().eval().to(device)
  segnet.load_state_dict(load_file(segnet_sd_path, device=str(device)))

  print(f"caching scored frames from {args.input}", flush=True)
  cached = _cache_targets(
    input_path=args.input,
    segnet=segnet,
    device=device,
    max_scored_frames=args.max_scored_frames,
    batch_size=args.cache_batch_size,
  )
  cached = finalize_cache_stats(cached)
  print(
    f"cached n={cached.targets.shape[0]} odd frames at {LOW_W}x{LOW_H}, initial target pixels={cached.targets.numel()}",
    flush=True,
  )

  renderer = build_renderer(args.renderer, args.hidden)
  if isinstance(renderer, PaletteRenderer):
    with torch.no_grad():
      init01 = torch.tensor(DEFAULT_PALETTE.T / 255.0, dtype=torch.float32).clamp(1e-4, 1.0 - 1e-4)
      renderer.colors.copy_(torch.logit(init01).view(1, 3, 5, 1, 1))

  baseline_ce, baseline_err = evaluate_renderer(
    segnet=segnet,
    renderer=renderer.to(device),
    cached=cached,
    device=device,
    batch_size=args.batch_size,
    upsample_mode=args.upsample_mode,
    feature_type=args.feature_type,
  )
  print(
    f"baseline ce={baseline_ce:.6f} segnet_dist={baseline_err:.6f} seg_score={100.0 * baseline_err:.4f}",
    flush=True,
  )

  result = train_renderer(
    segnet=segnet,
    renderer=renderer,
    cached=cached,
    device=device,
    batch_size=args.batch_size,
    epochs=args.epochs,
    lr=args.lr,
    upsample_mode=args.upsample_mode,
    soft_loss_weight=args.soft_loss_weight,
    feature_type=args.feature_type,
  )

  serializable = {
    "input": str(args.input),
    "renderer": args.renderer,
    "upsample_mode": args.upsample_mode,
    "hidden": args.hidden,
    "feature_type": args.feature_type,
    "lr": args.lr,
    "soft_loss_weight": args.soft_loss_weight,
    "max_scored_frames": args.max_scored_frames,
    "best_segnet_dist": result["best_segnet_dist"],
    "best_seg_score": result["best_seg_score"],
    "history": result["history"],
  }
  args.report.parent.mkdir(parents=True, exist_ok=True)
  args.report.write_text(json.dumps(serializable, indent=2) + "\n")
  print(json.dumps(serializable, indent=2), flush=True)


if __name__ == "__main__":
  main()
