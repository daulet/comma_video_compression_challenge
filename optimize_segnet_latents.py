#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from frame_utils import camera_size
from modules import SegNet, segnet_sd_path
from safetensors.torch import load_file
from optimize_segnet_visualization import (
  DEFAULT_PALETTE,
  _cache_targets,
  finalize_cache_stats,
  pick_device,
  roundtrip_lowres,
  select_features,
  seed_everything,
)

HERE = Path(__file__).resolve().parent


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Directly optimize scored SegNet-visualization frames against SegNet argmax.")
  parser.add_argument("--input", type=Path, default=HERE / "videos" / "0.mkv")
  parser.add_argument("--device", type=str, default="cpu", help="cpu|cuda|mps")
  parser.add_argument("--cache-batch-size", type=int, default=8)
  parser.add_argument("--batch-size", type=int, default=4)
  parser.add_argument("--epochs", type=int, default=8)
  parser.add_argument("--lr", type=float, default=1e-2)
  parser.add_argument("--soft-loss-weight", type=float, default=0.05)
  parser.add_argument("--anchor-weight", type=float, default=0.02)
  parser.add_argument("--tv-weight", type=float, default=1e-4)
  parser.add_argument("--max-scored-frames", type=int, default=32)
  parser.add_argument("--feature-type", choices=["probs", "logits"], default="probs")
  parser.add_argument("--latent-res", choices=["low", "full"], default="low")
  parser.add_argument("--seed", type=int, default=123)
  parser.add_argument("--upsample-mode", choices=["bilinear", "nearest"], default="nearest")
  parser.add_argument("--report", type=Path, default=HERE / "artifacts" / "segnet_latent_report.json")
  return parser.parse_args()


def _tv_loss(x: torch.Tensor) -> torch.Tensor:
  dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
  dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
  return dx + dy


def evaluate_latents(
  *,
  segnet: SegNet,
  cached,
  latent: torch.Tensor,
  anchor: torch.Tensor,
  device: torch.device,
  batch_size: int,
  upsample_mode: str,
  latent_res: str,
) -> tuple[float, float]:
  total_ce = 0.0
  total_err = 0.0
  n_items = cached.targets.shape[0]

  with torch.inference_mode():
    for start in range(0, n_items, batch_size):
      targets = cached.targets[start : start + batch_size].to(device=device, dtype=torch.long)
      batch_latent = latent[start : start + batch_size].sigmoid() * 255.0
      if latent_res == "low":
        rerun_in = roundtrip_lowres(batch_latent, upsample_mode=upsample_mode)
      elif latent_res == "full":
        rerun_in = F.interpolate(batch_latent, size=(384, 512), mode="bilinear", align_corners=False)
      else:
        raise ValueError(f"unsupported latent_res {latent_res}")
      logits = segnet(rerun_in)
      ce = F.cross_entropy(logits, targets, reduction="mean")
      err = (logits.argmax(dim=1) != targets).float().mean()
      total_ce += float(ce.item()) * targets.shape[0]
      total_err += float(err.item()) * targets.shape[0]
  return total_ce / n_items, total_err / n_items


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
    f"cached n={cached.targets.shape[0]} odd frames at 512x384, initial target pixels={cached.targets.numel()}",
    flush=True,
  )

  with torch.no_grad():
    features = select_features(cached, slice(0, cached.targets.shape[0]), device=torch.device("cpu"), feature_type=args.feature_type)
    colors = torch.tensor(DEFAULT_PALETTE.T, dtype=torch.float32).view(1, 3, 5, 1, 1)
    init_low = (features.unsqueeze(1) * colors).sum(dim=2).clamp(1e-4, 255.0 - 1e-4)
    if args.latent_res == "low":
      init_rgb = init_low
    else:
      if args.upsample_mode == "nearest":
        init_rgb = F.interpolate(init_low, size=(camera_size[1], camera_size[0]), mode="nearest")
      else:
        init_rgb = F.interpolate(init_low, size=(camera_size[1], camera_size[0]), mode="bilinear", align_corners=False)
    latent = torch.logit((init_rgb / 255.0).clamp(1e-4, 1.0 - 1e-4))

  anchor = init_rgb.to(device)
  latent = torch.nn.Parameter(latent.to(device))
  optimizer = torch.optim.Adam([latent], lr=args.lr)

  baseline_ce, baseline_err = evaluate_latents(
    segnet=segnet,
    cached=cached,
    latent=latent,
    anchor=anchor,
    device=device,
    batch_size=args.batch_size,
    upsample_mode=args.upsample_mode,
    latent_res=args.latent_res,
  )
  print(
    f"baseline ce={baseline_ce:.6f} segnet_dist={baseline_err:.6f} seg_score={100.0 * baseline_err:.4f}",
    flush=True,
  )

  best_state = latent.detach().cpu().clone()
  best_err = baseline_err
  history = [{
    "epoch": 0,
    "cross_entropy": baseline_ce,
    "segnet_dist": baseline_err,
    "seg_score": 100.0 * baseline_err,
  }]

  n_items = cached.targets.shape[0]
  for epoch in range(args.epochs):
    order = torch.randperm(n_items)
    for start in range(0, n_items, args.batch_size):
      idx = order[start : start + args.batch_size].to(device)
      targets = cached.targets[idx.cpu()].to(device=device, dtype=torch.long)
      teacher = cached.logits[idx.cpu()].to(device=device, dtype=torch.float32).softmax(dim=1)
      rgb = latent[idx].sigmoid() * 255.0
      if args.latent_res == "low":
        rerun_in = roundtrip_lowres(rgb, upsample_mode=args.upsample_mode)
      else:
        rerun_in = F.interpolate(rgb, size=(384, 512), mode="bilinear", align_corners=False)
      logits = segnet(rerun_in)
      ce = F.cross_entropy(logits, targets)
      soft = F.kl_div(F.log_softmax(logits, dim=1), teacher, reduction="none").mean()
      anchor_loss = F.mse_loss(rgb, anchor[idx])
      tv = _tv_loss(rgb / 255.0)
      loss = ce + args.soft_loss_weight * soft + args.anchor_weight * anchor_loss + args.tv_weight * tv
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()

    ce_eval, err_eval = evaluate_latents(
      segnet=segnet,
      cached=cached,
      latent=latent,
      anchor=anchor,
      device=device,
      batch_size=args.batch_size,
      upsample_mode=args.upsample_mode,
      latent_res=args.latent_res,
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
      best_state = latent.detach().cpu().clone()

  latent.data.copy_(best_state.to(device))
  ce_eval, err_eval = evaluate_latents(
    segnet=segnet,
    cached=cached,
    latent=latent,
    anchor=anchor,
    device=device,
    batch_size=args.batch_size,
    upsample_mode=args.upsample_mode,
    latent_res=args.latent_res,
  )
  result = {
    "input": str(args.input),
    "feature_type": args.feature_type,
    "latent_res": args.latent_res,
    "upsample_mode": args.upsample_mode,
    "lr": args.lr,
    "soft_loss_weight": args.soft_loss_weight,
    "anchor_weight": args.anchor_weight,
    "tv_weight": args.tv_weight,
    "max_scored_frames": args.max_scored_frames,
    "best_segnet_dist": err_eval,
    "best_seg_score": 100.0 * err_eval,
    "history": history,
  }
  args.report.parent.mkdir(parents=True, exist_ok=True)
  args.report.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
  main()
