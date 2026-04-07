#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import random
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
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
TEMPORAL_LINEAR_FEATURES = 13
TEMPORAL_LINEAR_SAMPLE_STRIDE = 24
TEMPORAL_RIDGE = 1e-2
TEMPORAL_LINEAR_RESIDUAL_CLAMP = 20.0 / 255.0
TEMPORAL_MLP_FEATURES = 19
TEMPORAL_MLP_HIDDEN = 4
TEMPORAL_MLP_SAMPLE_STRIDE = 32
TEMPORAL_MLP_MAX_SAMPLES = 160_000
TEMPORAL_MLP_STEPS = 160
TEMPORAL_MLP_BATCH_SIZE = 8192
TEMPORAL_MLP_LR = 3e-2
TEMPORAL_MLP_RESIDUAL_CLAMP = 8.0 / 255.0


@dataclass(frozen=True)
class TemporalRefiner:
  linear: torch.Tensor
  mlp_w1: torch.Tensor
  mlp_b1: torch.Tensor
  mlp_w2: torch.Tensor
  mlp_b2: torch.Tensor


class LinearFilter(nn.Module):
  def __init__(self, kernel_size: int = 9, init_strength: float = 0.85):
    super().__init__()
    self.kernel_size = kernel_size
    self.pad = kernel_size // 2
    self.weight = nn.Parameter(torch.zeros(3, 1, kernel_size, kernel_size))
    with torch.no_grad():
      self.weight[:, :, kernel_size // 2, kernel_size // 2] = 1.0
      unsharp_k = UNSHARP_KERNEL.unsqueeze(0).expand(3, 1, 9, 9)
      identity = torch.zeros(3, 1, 9, 9)
      identity[:, :, 4, 4] = 1.0
      self.weight.copy_((1.0 + init_strength) * identity - init_strength * unsharp_k)

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


def _edge_gate_features(x: torch.Tensor) -> torch.Tensor:
  y = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
  blur = F.avg_pool2d(F.pad(y, (1, 1, 1, 1), mode="reflect"), kernel_size=3, stride=1)
  edge = (y - blur).abs()
  edge = F.avg_pool2d(F.pad(edge, (1, 1, 1, 1), mode="reflect"), kernel_size=3, stride=1)
  scale = edge.mean(dim=(2, 3), keepdim=True).clamp_min(1e-3)
  return edge / scale


def _edge_gate_map(x: torch.Tensor, gate_scale: torch.Tensor, gate_bias: torch.Tensor) -> torch.Tensor:
  return torch.sigmoid(gate_scale.view(1, 1, 1, 1) * (_edge_gate_features(x) - gate_bias.view(1, 1, 1, 1)))


def _apply_gated_filters(
  x: torch.Tensor,
  smooth_filt: LinearFilter,
  detail_filt: LinearFilter,
  gate_scale: torch.Tensor,
  gate_bias: torch.Tensor,
) -> torch.Tensor:
  gate = _edge_gate_map(x, gate_scale, gate_bias)
  smooth = smooth_filt(x)
  detail = detail_filt(x)
  return smooth + gate * (detail - smooth)


def _boundary_priority_map(classes: torch.Tensor) -> torch.Tensor:
  boundary = torch.zeros_like(classes, dtype=torch.bool)
  boundary[:, 1:, :] |= classes[:, 1:, :] != classes[:, :-1, :]
  boundary[:, :-1, :] |= classes[:, 1:, :] != classes[:, :-1, :]
  boundary[:, :, 1:] |= classes[:, :, 1:] != classes[:, :, :-1]
  boundary[:, :, :-1] |= classes[:, :, 1:] != classes[:, :, :-1]
  return 1.0 + 2.0 * boundary.float()


def _segmentation_margin_loss(
  logits: torch.Tensor,
  target_classes: torch.Tensor,
  pixel_weights: torch.Tensor,
  margin: float = 0.25,
) -> torch.Tensor:
  target_logits = logits.gather(1, target_classes.unsqueeze(1)).squeeze(1)
  class_mask = F.one_hot(target_classes, num_classes=logits.shape[1]).permute(0, 3, 1, 2).bool()
  other_logits = logits.masked_fill(class_mask, float("-inf")).amax(dim=1)
  loss = F.relu(margin - (target_logits - other_logits))
  return (loss * pixel_weights).sum() / pixel_weights.sum().clamp_min(1.0)


def _train_filter_bank_with_eval_loss(
  original_frames: list[torch.Tensor],
  upsampled_chw: list[torch.Tensor],
  n_iters: int = 300,
  batch_size: int = 2,
  lr: float = 1e-3,
) -> tuple[LinearFilter, LinearFilter, LinearFilter, LinearFilter, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  from modules import DistortionNet, posenet_sd_path, segnet_sd_path

  eval_h, eval_w = segnet_model_input_size[1], segnet_model_input_size[0]
  n_pairs = len(original_frames) // 2

  print("  Loading eval models...", flush=True)
  distortion_net = DistortionNet().eval()
  distortion_net.load_state_dicts(posenet_sd_path, segnet_sd_path, "cpu")
  for p in distortion_net.parameters():
    p.requires_grad_(False)

  print("  Preparing eval-resolution frames...", flush=True)
  upsampled_eval = []
  original_eval = []
  for original, upsampled in zip(original_frames, upsampled_chw, strict=True):
    o = original.permute(2, 0, 1).unsqueeze(0).float()
    original_eval.append(F.interpolate(o, size=(eval_h, eval_w), mode="bilinear", align_corners=False))
    upsampled_eval.append(F.interpolate(upsampled, size=(eval_h, eval_w), mode="bilinear", align_corners=False))

  print("  Pre-computing original eval outputs...", flush=True)
  orig_pose_targets = []
  orig_seg_classes = []
  orig_seg_weights = []
  with torch.no_grad():
    for pair_idx in range(n_pairs):
      i = pair_idx * 2
      pair = torch.stack([
        original_eval[i].squeeze(0).permute(1, 2, 0),
        original_eval[i + 1].squeeze(0).permute(1, 2, 0),
      ], dim=0).unsqueeze(0)
      po, so = distortion_net(pair)
      classes = so.argmax(dim=1).detach().to(torch.uint8)
      orig_pose_targets.append(po["pose"][:, :6].detach())
      orig_seg_classes.append(classes)
      orig_seg_weights.append(_boundary_priority_map(classes.long()))
  print(f"  Pre-computed {len(orig_pose_targets)} evaluator pairs", flush=True)

  even_smooth = LinearFilter(kernel_size=9, init_strength=0.15)
  even_detail = LinearFilter(kernel_size=9, init_strength=0.95)
  odd_smooth = LinearFilter(kernel_size=9, init_strength=0.15)
  odd_detail = LinearFilter(kernel_size=9, init_strength=0.95)
  even_gate_scale = nn.Parameter(torch.tensor(2.0))
  even_gate_bias = nn.Parameter(torch.tensor(1.5))
  odd_gate_scale = nn.Parameter(torch.tensor(2.0))
  odd_gate_bias = nn.Parameter(torch.tensor(1.5))
  print(
    (
      "  Gated LinearFilter bank: "
      f"even=({_count_params(even_smooth)}+{_count_params(even_detail)}) "
      f"odd=({_count_params(odd_smooth)}+{_count_params(odd_detail)}) params"
    ),
    flush=True,
  )
  optimizer = torch.optim.Adam(
    (
      list(even_smooth.parameters())
      + list(even_detail.parameters())
      + list(odd_smooth.parameters())
      + list(odd_detail.parameters())
      + [even_gate_scale, even_gate_bias, odd_gate_scale, odd_gate_bias]
    ),
    lr=lr,
  )
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iters)

  even_smooth.train()
  even_detail.train()
  odd_smooth.train()
  odd_detail.train()
  running_loss = 0.0
  running_pose = 0.0
  running_seg = 0.0

  for it in range(n_iters):
    pair_indices = [random.randrange(n_pairs) for _ in range(batch_size)]

    filtered_pairs = []
    pose_targets = []
    seg_targets = []
    seg_weights = []

    for pair_idx in pair_indices:
      i = pair_idx * 2
      f_i = _apply_gated_filters(upsampled_eval[i], even_smooth, even_detail, even_gate_scale, even_gate_bias)
      f_i1 = _apply_gated_filters(upsampled_eval[i + 1], odd_smooth, odd_detail, odd_gate_scale, odd_gate_bias)

      pair = torch.stack([
        f_i.squeeze(0).permute(1, 2, 0),
        f_i1.squeeze(0).permute(1, 2, 0),
      ], dim=0).unsqueeze(0)
      filtered_pairs.append(pair)
      pose_targets.append(orig_pose_targets[pair_idx])
      seg_targets.append(orig_seg_classes[pair_idx])
      seg_weights.append(orig_seg_weights[pair_idx])

    filtered_batch = torch.cat(filtered_pairs, dim=0)

    enh_po, enh_so = distortion_net(filtered_batch)

    pose_target = torch.cat(pose_targets, dim=0)
    loss_posenet = F.mse_loss(enh_po["pose"][:, :6], pose_target)

    target_classes = torch.cat(seg_targets, dim=0).long()
    pixel_weights = torch.cat(seg_weights, dim=0)
    loss_segnet = _segmentation_margin_loss(enh_so, target_classes, pixel_weights)

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

  even_smooth.eval()
  even_detail.eval()
  odd_smooth.eval()
  odd_detail.eval()
  return (
    even_smooth,
    even_detail,
    odd_smooth,
    odd_detail,
    even_gate_scale.detach(),
    even_gate_bias.detach(),
    odd_gate_scale.detach(),
    odd_gate_bias.detach(),
  )


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


def _save_filter_bank(
  even_smooth: LinearFilter,
  even_detail: LinearFilter,
  odd_smooth: LinearFilter,
  odd_detail: LinearFilter,
  even_gate_scale: torch.Tensor,
  even_gate_bias: torch.Tensor,
  odd_gate_scale: torch.Tensor,
  odd_gate_bias: torch.Tensor,
  path: Path,
) -> int:
  state = {
    "even_smooth": {k: v.half() for k, v in even_smooth.state_dict().items()},
    "even_detail": {k: v.half() for k, v in even_detail.state_dict().items()},
    "odd_smooth": {k: v.half() for k, v in odd_smooth.state_dict().items()},
    "odd_detail": {k: v.half() for k, v in odd_detail.state_dict().items()},
    "gate": {
      "even_scale": even_gate_scale.half(),
      "even_bias": even_gate_bias.half(),
      "odd_scale": odd_gate_scale.half(),
      "odd_bias": odd_gate_bias.half(),
    },
  }
  buf = io.BytesIO()
  torch.save(state, buf)
  data = buf.getvalue()
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(data)
  print(f"Saved gated LinearFilter bank: {len(data)} bytes", flush=True)
  return len(data)


def _load_filter_bank(path: Path) -> tuple[LinearFilter, LinearFilter, LinearFilter, LinearFilter, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  state = torch.load(path, map_location="cpu", weights_only=True)
  if "even" in state and "odd" in state:
    even_filt = LinearFilter(kernel_size=9)
    odd_filt = LinearFilter(kernel_size=9)
    even_filt.load_state_dict({k: v.float() for k, v in state["even"].items()})
    odd_filt.load_state_dict({k: v.float() for k, v in state["odd"].items()})
    even_filt.eval()
    odd_filt.eval()
    return (
      even_filt,
      even_filt,
      odd_filt,
      odd_filt,
      torch.tensor(16.0),
      torch.tensor(-8.0),
      torch.tensor(16.0),
      torch.tensor(-8.0),
    )
  even_smooth = LinearFilter(kernel_size=9)
  even_detail = LinearFilter(kernel_size=9)
  odd_smooth = LinearFilter(kernel_size=9)
  odd_detail = LinearFilter(kernel_size=9)
  even_smooth.load_state_dict({k: v.float() for k, v in state["even_smooth"].items()})
  even_detail.load_state_dict({k: v.float() for k, v in state["even_detail"].items()})
  odd_smooth.load_state_dict({k: v.float() for k, v in state["odd_smooth"].items()})
  odd_detail.load_state_dict({k: v.float() for k, v in state["odd_detail"].items()})
  even_smooth.eval()
  even_detail.eval()
  odd_smooth.eval()
  odd_detail.eval()
  gate = state["gate"]
  return (
    even_smooth,
    even_detail,
    odd_smooth,
    odd_detail,
    gate["even_scale"].float(),
    gate["even_bias"].float(),
    gate["odd_scale"].float(),
    gate["odd_bias"].float(),
  )


def _refiner_path(dst_video: Path) -> Path:
  return dst_video.with_suffix(".refiner.bin")


def _normalize_frame(x: torch.Tensor) -> torch.Tensor:
  return x.clamp(0, 255).div(255.0)


def _zero_residual_like(base: torch.Tensor) -> torch.Tensor:
  return torch.zeros((base.shape[2], base.shape[3], 3), dtype=base.dtype, device=base.device)


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


def _predict_linear_refiner_residual(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
  return torch.matmul(features.squeeze(0).permute(1, 2, 0), weights).clamp_(
    -TEMPORAL_LINEAR_RESIDUAL_CLAMP, TEMPORAL_LINEAR_RESIDUAL_CLAMP
  )


def _predict_mlp_refiner_residual(features: torch.Tensor, refiner: TemporalRefiner) -> torch.Tensor:
  hidden = torch.tanh(torch.matmul(features.squeeze(0).permute(1, 2, 0), refiner.mlp_w1) + refiner.mlp_b1)
  return torch.matmul(hidden, refiner.mlp_w2).add_(refiner.mlp_b2).clamp_(
    -TEMPORAL_MLP_RESIDUAL_CLAMP, TEMPORAL_MLP_RESIDUAL_CLAMP
  )


def _refine_frame(base: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
  refined = (base.squeeze(0).permute(1, 2, 0) + residual).clamp_(0.0, 1.0)
  return refined.permute(2, 0, 1).unsqueeze(0)


def _fit_mlp_temporal_refiner(features: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, ...]:
  if features.shape[0] > TEMPORAL_MLP_MAX_SAMPLES:
    generator = torch.Generator().manual_seed(0)
    keep = torch.randperm(features.shape[0], generator=generator)[:TEMPORAL_MLP_MAX_SAMPLES]
    features = features[keep]
    targets = targets[keep]

  generator = torch.Generator().manual_seed(0)
  w1 = (torch.randn((TEMPORAL_MLP_FEATURES, TEMPORAL_MLP_HIDDEN), generator=generator) * 0.05).requires_grad_()
  b1 = torch.zeros((TEMPORAL_MLP_HIDDEN,), dtype=torch.float32, requires_grad=True)
  w2 = (torch.randn((TEMPORAL_MLP_HIDDEN, 3), generator=generator) * 0.05).requires_grad_()
  b2 = torch.zeros((3,), dtype=torch.float32, requires_grad=True)
  params = [w1, b1, w2, b2]
  optimizer = torch.optim.Adam(params, lr=TEMPORAL_MLP_LR)
  batch_size = min(TEMPORAL_MLP_BATCH_SIZE, features.shape[0])

  for _ in range(TEMPORAL_MLP_STEPS):
    batch_idx = torch.randint(features.shape[0], (batch_size,), generator=generator)
    xb = features[batch_idx]
    yb = targets[batch_idx]
    hidden = torch.tanh(xb @ w1 + b1)
    pred = (hidden @ w2 + b2).clamp(-TEMPORAL_MLP_RESIDUAL_CLAMP, TEMPORAL_MLP_RESIDUAL_CLAMP)
    loss = F.smooth_l1_loss(pred, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

  with torch.inference_mode():
    hidden = torch.tanh(features @ w1 + b1)
    pred = (hidden @ w2 + b2).clamp(-TEMPORAL_MLP_RESIDUAL_CLAMP, TEMPORAL_MLP_RESIDUAL_CLAMP)
    loss = F.smooth_l1_loss(pred, targets).item()
    print(f"Fitted temporal MLP on {features.shape[0]} samples (loss={loss:.6f})", flush=True)

  return (
    w1.detach().to(dtype=torch.float16),
    b1.detach().to(dtype=torch.float16),
    w2.detach().to(dtype=torch.float16),
    b2.detach().to(dtype=torch.float16),
  )


def _save_temporal_refiner(path: Path, refiner: TemporalRefiner) -> int:
  arrays = [
    refiner.linear.cpu().numpy().reshape(-1),
    refiner.mlp_w1.cpu().numpy().reshape(-1),
    refiner.mlp_b1.cpu().numpy().reshape(-1),
    refiner.mlp_w2.cpu().numpy().reshape(-1),
    refiner.mlp_b2.cpu().numpy().reshape(-1),
  ]
  data = np.concatenate(arrays).astype(np.float16).tobytes()
  path.write_bytes(data)
  print(f"Saved temporal refiner: {len(data)} bytes", flush=True)
  return len(data)


def _load_temporal_refiner(path: Path) -> TemporalRefiner | None:
  if not path.exists():
    return None
  arr = np.frombuffer(path.read_bytes(), dtype=np.float16)
  expected = (
    TEMPORAL_LINEAR_FEATURES * 3
    + TEMPORAL_MLP_FEATURES * TEMPORAL_MLP_HIDDEN
    + TEMPORAL_MLP_HIDDEN
    + TEMPORAL_MLP_HIDDEN * 3
    + 3
  )
  if arr.size != expected:
    raise ValueError(f"Unexpected temporal refiner size for {path}: {arr.size}")
  offset = 0

  def take(count: int, shape: tuple[int, ...]) -> torch.Tensor:
    nonlocal offset
    out = torch.tensor(arr[offset:offset + count].reshape(shape), dtype=torch.float32)
    offset += count
    return out

  return TemporalRefiner(
    linear=take(TEMPORAL_LINEAR_FEATURES * 3, (TEMPORAL_LINEAR_FEATURES, 3)),
    mlp_w1=take(TEMPORAL_MLP_FEATURES * TEMPORAL_MLP_HIDDEN, (TEMPORAL_MLP_FEATURES, TEMPORAL_MLP_HIDDEN)),
    mlp_b1=take(TEMPORAL_MLP_HIDDEN, (TEMPORAL_MLP_HIDDEN,)),
    mlp_w2=take(TEMPORAL_MLP_HIDDEN * 3, (TEMPORAL_MLP_HIDDEN, 3)),
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


def _fit_temporal_refiner(
  original_frames: list[torch.Tensor],
  upsampled_chw: list[torch.Tensor],
  even_filt: LinearFilter,
  odd_filt: LinearFilter,
  path: Path,
) -> None:
  xtx = torch.zeros((TEMPORAL_LINEAR_FEATURES, TEMPORAL_LINEAR_FEATURES), dtype=torch.float64)
  xty = torch.zeros((TEMPORAL_LINEAR_FEATURES, 3), dtype=torch.float64)
  prev_base = None

  with torch.inference_mode():
    for idx, (original, upsampled) in enumerate(zip(original_frames, upsampled_chw, strict=True)):
      filt = even_filt if (idx % 2 == 0) else odd_filt
      base = _normalize_frame(filt(upsampled))
      target = original.permute(2, 0, 1).unsqueeze(0).float().div(255.0)
      if prev_base is None:
        prev_base = base
      feats = _linear_refiner_features(base, prev_base)[..., ::TEMPORAL_LINEAR_SAMPLE_STRIDE, ::TEMPORAL_LINEAR_SAMPLE_STRIDE]
      residual = (target - base)[..., ::TEMPORAL_LINEAR_SAMPLE_STRIDE, ::TEMPORAL_LINEAR_SAMPLE_STRIDE]
      x = feats.squeeze(0).permute(1, 2, 0).reshape(-1, TEMPORAL_LINEAR_FEATURES).to(dtype=torch.float64)
      y = residual.squeeze(0).permute(1, 2, 0).reshape(-1, 3).to(dtype=torch.float64)
      xtx += x.T @ x
      xty += x.T @ y
      prev_base = base

  ridge = TEMPORAL_RIDGE * torch.eye(TEMPORAL_LINEAR_FEATURES, dtype=torch.float64)
  linear_weights = torch.linalg.solve(xtx + ridge, xty).to(dtype=torch.float32)

  mlp_features = []
  mlp_targets = []
  prev_base = None
  prev_prev_base = None
  prev_linear_residual = None
  with torch.inference_mode():
    for idx, (original, upsampled) in enumerate(zip(original_frames, upsampled_chw, strict=True)):
      filt = even_filt if (idx % 2 == 0) else odd_filt
      base = _normalize_frame(filt(upsampled))
      target = original.permute(2, 0, 1).unsqueeze(0).float().div(255.0)
      if prev_base is None:
        prev_base = base
      if prev_prev_base is None:
        prev_prev_base = prev_base
      if prev_linear_residual is None:
        prev_linear_residual = _zero_residual_like(base)
      linear_features = _linear_refiner_features(base, prev_base)
      linear_residual = _predict_linear_refiner_residual(linear_features, linear_weights)
      linear_refined = _refine_frame(base, linear_residual)
      feats = _mlp_refiner_features(base, prev_base, prev_prev_base, linear_residual, prev_linear_residual)
      residual = (target - linear_refined).clamp_(-TEMPORAL_MLP_RESIDUAL_CLAMP, TEMPORAL_MLP_RESIDUAL_CLAMP)
      x = feats[..., ::TEMPORAL_MLP_SAMPLE_STRIDE, ::TEMPORAL_MLP_SAMPLE_STRIDE]
      y = residual[..., ::TEMPORAL_MLP_SAMPLE_STRIDE, ::TEMPORAL_MLP_SAMPLE_STRIDE]
      mlp_features.append(x.squeeze(0).permute(1, 2, 0).reshape(-1, TEMPORAL_MLP_FEATURES))
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
  _save_temporal_refiner(path, refiner)


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

    print("Training parity-specific LinearFilter bank with eval model loss...", flush=True)
    (
      even_smooth,
      even_detail,
      odd_smooth,
      odd_detail,
      even_gate_scale,
      even_gate_bias,
      odd_gate_scale,
      odd_gate_bias,
    ) = _train_filter_bank_with_eval_loss(original_frames, upsampled_chw)

    _save_filter_bank(
      even_smooth,
      even_detail,
      odd_smooth,
      odd_detail,
      even_gate_scale,
      even_gate_bias,
      odd_gate_scale,
      odd_gate_bias,
      archive_dir / "linear_filter_bank.pt",
    )
    del original_frames, compressed_frames, upsampled_chw

  archive_zip.parent.mkdir(parents=True, exist_ok=True)
  if archive_zip.exists():
    archive_zip.unlink()
  with zipfile.ZipFile(archive_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(archive_dir.rglob("*")):
      if path.is_file():
        zf.write(path, arcname=str(path.relative_to(archive_dir)))
  print(f"Archive: {archive_zip} ({archive_zip.stat().st_size:,} bytes)", flush=True)


def _decode_and_restore_to_raw(
  src: Path,
  dst: Path,
  even_smooth: LinearFilter | None,
  even_detail: LinearFilter | None,
  odd_smooth: LinearFilter | None,
  odd_detail: LinearFilter | None,
  even_gate_scale: torch.Tensor | None,
  even_gate_bias: torch.Tensor | None,
  odd_gate_scale: torch.Tensor | None,
  odd_gate_bias: torch.Tensor | None,
  refiner: TemporalRefiner | None,
) -> int:
  target_w, target_h = camera_size
  fmt = "hevc" if src.suffix == ".hevc" else None
  container = av.open(str(src), format=fmt)
  stream = container.streams.video[0]
  dst.parent.mkdir(parents=True, exist_ok=True)

  n = 0
  prev_base = None
  prev_prev_base = None
  prev_linear_residual = None
  with torch.inference_mode(), dst.open("wb") as f:
    for frame in container.decode(stream):
      t = yuv420_to_rgb(frame)
      x = _bicubic_upsample(t, target_h, target_w)

      if (n % 2 == 0) and even_smooth is not None and even_detail is not None and even_gate_scale is not None and even_gate_bias is not None:
        x = _apply_gated_filters(x, even_smooth, even_detail, even_gate_scale, even_gate_bias)
      elif (n % 2 == 1) and odd_smooth is not None and odd_detail is not None and odd_gate_scale is not None and odd_gate_bias is not None:
        x = _apply_gated_filters(x, odd_smooth, odd_detail, odd_gate_scale, odd_gate_bias)
      else:
        x = _apply_unsharp(x)

      base = _normalize_frame(x)
      if prev_base is None:
        prev_base = base
      if prev_prev_base is None:
        prev_prev_base = prev_base
      if prev_linear_residual is None:
        prev_linear_residual = _zero_residual_like(base)
      refined, linear_residual = _apply_temporal_refiner(base, prev_base, prev_prev_base, prev_linear_residual, refiner)
      out = refined.mul(255.0).clamp_(0.0, 255.0).squeeze(0).permute(1, 2, 0).round().to(torch.uint8)
      f.write(out.contiguous().numpy().tobytes())
      prev_prev_base = prev_base
      prev_base = base
      prev_linear_residual = linear_residual
      n += 1

  container.close()
  return n


def inflate_archive(archive_dir: Path, output_dir: Path, video_names: list[str]) -> None:
  _reset_dir(output_dir)
  bank_path = archive_dir / "linear_filter_bank.pt"
  filt_path = archive_dir / "linear_filter.pt"
  even_smooth = even_detail = odd_smooth = odd_detail = None
  even_gate_scale = even_gate_bias = odd_gate_scale = odd_gate_bias = None
  if bank_path.exists():
    (
      even_smooth,
      even_detail,
      odd_smooth,
      odd_detail,
      even_gate_scale,
      even_gate_bias,
      odd_gate_scale,
      odd_gate_bias,
    ) = _load_filter_bank(bank_path)
    print(
      (
        "Loaded gated LinearFilter bank "
        f"(even={_count_params(even_smooth)}+{_count_params(even_detail)} "
        f"odd={_count_params(odd_smooth)}+{_count_params(odd_detail)} params)"
      ),
      flush=True,
    )
  elif filt_path.exists():
    even_smooth = even_detail = odd_smooth = odd_detail = _load_filter(filt_path)
    even_gate_scale = odd_gate_scale = torch.tensor(16.0)
    even_gate_bias = odd_gate_bias = torch.tensor(-8.0)
    print(f"Loaded LinearFilter ({_count_params(even_smooth)} params)", flush=True)

  for rel in video_names:
    base = Path(rel).with_suffix("")
    src = archive_dir / f"{base}.mkv"
    dst = output_dir / f"{base}.raw"
    if not src.exists():
      raise FileNotFoundError(f"Missing encoded video in archive: {src}")
    refiner = _load_temporal_refiner(_refiner_path(src))
    if refiner is not None:
      print("Loaded temporal refiner", flush=True)
    print(f"Decoding + restoring {src} -> {dst}", flush=True)
    n = _decode_and_restore_to_raw(
      src,
      dst,
      even_smooth,
      even_detail,
      odd_smooth,
      odd_detail,
      even_gate_scale,
      even_gate_bias,
      odd_gate_scale,
      odd_gate_bias,
      refiner,
    )
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
