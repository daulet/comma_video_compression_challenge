#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from frame_utils import segnet_model_input_size
from model_diagnostics import iter_rgb_frames

POSE_SALIENCY_COLORS = np.array([
  [231, 76, 60],
  [241, 196, 15],
  [46, 204, 113],
  [52, 152, 219],
  [155, 89, 182],
  [26, 188, 156],
], dtype=np.uint8)


def pose_vector(posenet, prev_frame: torch.Tensor, curr_frame: torch.Tensor, device: torch.device) -> np.ndarray:
  with torch.inference_mode():
    pair = torch.stack([
      prev_frame.permute(2, 0, 1),
      curr_frame.permute(2, 0, 1),
    ], dim=0).unsqueeze(0).float().to(device)
    out = posenet(posenet.preprocess_input(pair))["pose"][0, :6]
  return out.detach().cpu().numpy().astype(np.float32)


def pose_preprocess_grad(pair_btchw: torch.Tensor) -> torch.Tensor:
  batch_size, seq_len, channels, _, _ = pair_btchw.shape
  x = pair_btchw.view(batch_size * seq_len, channels, pair_btchw.shape[-2], pair_btchw.shape[-1])
  x = F.interpolate(x, size=(segnet_model_input_size[1], segnet_model_input_size[0]), mode="bilinear", align_corners=False)

  h = x.shape[-2]
  w = x.shape[-1]
  h2 = h // 2
  w2 = w // 2
  x = x[..., : 2 * h2, : 2 * w2]

  r = x[:, 0]
  g = x[:, 1]
  b = x[:, 2]
  y = (0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 255.0)
  u = ((b - y) / 1.772 + 128.0).clamp(0.0, 255.0)
  v = ((r - y) / 1.402 + 128.0).clamp(0.0, 255.0)

  u_sub = (u[:, 0::2, 0::2] + u[:, 1::2, 0::2] + u[:, 0::2, 1::2] + u[:, 1::2, 1::2]) * 0.25
  v_sub = (v[:, 0::2, 0::2] + v[:, 1::2, 0::2] + v[:, 0::2, 1::2] + v[:, 1::2, 1::2]) * 0.25
  y00 = y[:, 0::2, 0::2]
  y10 = y[:, 1::2, 0::2]
  y01 = y[:, 0::2, 1::2]
  y11 = y[:, 1::2, 1::2]

  yuv6 = torch.stack([y00, y10, y01, y11, u_sub, v_sub], dim=1)
  return yuv6.view(batch_size, seq_len * 6, h2, w2)


def select_metric(pose_vec: np.ndarray, selector: str) -> float:
  if selector == "max_norm":
    return float(np.linalg.norm(pose_vec))
  idx = int(selector.removeprefix("max_abs_p"))
  return float(abs(pose_vec[idx]))


def find_pair(
  *,
  input_path: Path,
  posenet,
  device: torch.device,
  selector: str,
  pair_index: int | None,
  max_frames: int | None,
) -> tuple[int, torch.Tensor, torch.Tensor, np.ndarray]:
  prev_frame = None
  best_metric = -1.0
  best = None

  for idx, frame in enumerate(iter_rgb_frames(input_path, max_frames=max_frames)):
    if prev_frame is None:
      prev_frame = frame
      continue
    pose_vec = pose_vector(posenet, prev_frame, frame, device)
    current_pair = idx - 1
    if pair_index is not None:
      if current_pair == pair_index:
        return current_pair, prev_frame.clone(), frame.clone(), pose_vec
    else:
      metric = select_metric(pose_vec, selector)
      if metric > best_metric:
        best_metric = metric
        best = (current_pair, prev_frame.clone(), frame.clone(), pose_vec.copy())
    prev_frame = frame

  if best is None:
    raise ValueError("Failed to locate a valid frame pair")
  return best


def normalize_map(saliency: np.ndarray) -> np.ndarray:
  saliency = np.maximum(saliency, 0.0)
  hi = np.percentile(saliency, 99.5)
  if hi <= 1e-12:
    return np.zeros_like(saliency, dtype=np.float32)
  return np.clip(saliency / hi, 0.0, 1.0).astype(np.float32)


def compute_saliency(
  posenet,
  prev_frame: torch.Tensor,
  curr_frame: torch.Tensor,
  device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  pair = torch.stack([
    prev_frame.permute(2, 0, 1),
    curr_frame.permute(2, 0, 1),
  ], dim=0).unsqueeze(0).float().to(device).detach().requires_grad_(True)

  pose = posenet(pose_preprocess_grad(pair))["pose"][0, :6]
  prev_maps = []
  curr_maps = []

  for dim in range(6):
    grad = torch.autograd.grad(pose[dim], pair, retain_graph=True)[0][0].abs().mean(dim=1).cpu().numpy()
    prev_maps.append(grad[0])
    curr_maps.append(grad[1])

  return pose.detach().cpu().numpy().astype(np.float32), np.stack(prev_maps), np.stack(curr_maps)


def winner_overlay(
  frame: np.ndarray,
  saliency_maps: np.ndarray,
  alpha: float,
  colors: np.ndarray | None = None,
) -> np.ndarray:
  palette = POSE_SALIENCY_COLORS if colors is None else colors
  norm_maps = np.stack([normalize_map(m) for m in saliency_maps], axis=0)
  winner = norm_maps.argmax(axis=0)
  strength = norm_maps.max(axis=0)
  color = palette[winner].astype(np.float32)
  base = frame.astype(np.float32)
  weight = (strength * alpha)[..., None]
  return np.clip(base * (1.0 - weight) + color * weight, 0.0, 255.0).astype(np.uint8)
