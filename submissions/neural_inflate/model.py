#!/usr/bin/env python3
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _coord_grid(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
  ys = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
  xs = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
  yy, xx = torch.meshgrid(ys, xs, indexing="ij")
  return torch.stack((xx, yy), dim=0).unsqueeze(0)


class SepConv(nn.Module):
  def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
    super().__init__()
    self.depthwise = nn.Conv2d(
      in_ch,
      in_ch,
      kernel_size=3,
      stride=stride,
      padding=1,
      groups=in_ch,
      bias=True,
    )
    self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.pointwise(self.depthwise(x))


class ResBlock(nn.Module):
  def __init__(self, channels: int, expand: int = 2):
    super().__init__()
    hidden = channels * expand
    self.conv1 = SepConv(channels, hidden)
    self.act = nn.ReLU(inplace=True)
    self.conv2 = SepConv(hidden, channels)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    residual = self.conv2(self.act(self.conv1(x)))
    return x + residual


class SepConvGRUCell(nn.Module):
  def __init__(self, channels: int):
    super().__init__()
    pair_ch = channels * 2
    self.gates_dw = nn.Conv2d(pair_ch, pair_ch, kernel_size=3, padding=1, groups=pair_ch)
    self.gates_pw = nn.Conv2d(pair_ch, channels * 2, kernel_size=1)
    self.cand_dw = nn.Conv2d(pair_ch, pair_ch, kernel_size=3, padding=1, groups=pair_ch)
    self.cand_pw = nn.Conv2d(pair_ch, channels, kernel_size=1)

  def forward(self, x: torch.Tensor, h: torch.Tensor | None) -> torch.Tensor:
    if h is None:
      h = torch.zeros_like(x)

    xh = torch.cat((x, h), dim=1)
    z, r = self.gates_pw(self.gates_dw(xh)).chunk(2, dim=1)
    z = torch.sigmoid(z)
    r = torch.sigmoid(r)

    xr = torch.cat((x, r * h), dim=1)
    cand = torch.tanh(self.cand_pw(self.cand_dw(xr)))
    return (1.0 - z) * h + z * cand


class REN(nn.Module):
  def __init__(self, features: int = 24):
    super().__init__()
    self.features = features
    self.down = nn.PixelUnshuffle(2)
    self.up = nn.PixelShuffle(2)

    input_ch = (3 + 3 + 3 + 2) * 4
    bottleneck = features * 2

    self.stem = nn.Conv2d(input_ch, features, kernel_size=3, padding=1)
    self.enc1 = nn.Sequential(
      ResBlock(features),
      ResBlock(features),
    )
    self.down2 = nn.Sequential(
      SepConv(features, bottleneck, stride=2),
      nn.ReLU(inplace=True),
    )
    self.enc2 = nn.Sequential(
      ResBlock(bottleneck),
      ResBlock(bottleneck),
    )
    self.gru = SepConvGRUCell(bottleneck)
    self.up1 = nn.Sequential(
      nn.Conv2d(bottleneck, features * 4, kernel_size=1),
      nn.PixelShuffle(2),
      nn.ReLU(inplace=True),
    )
    self.fuse = nn.Sequential(
      nn.Conv2d(features * 2, features, kernel_size=1),
      nn.ReLU(inplace=True),
      ResBlock(features),
      ResBlock(features),
    )
    self.head = nn.Conv2d(features, 12, kernel_size=3, padding=1)

    nn.init.zeros_(self.head.weight)
    nn.init.zeros_(self.head.bias)

  def _coords(self, batch: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return _coord_grid(height, width, device, dtype).expand(batch, -1, -1, -1)

  def step(
    self,
    x: torch.Tensor,
    prev_rgb: torch.Tensor | None = None,
    prev_state: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    x_norm = x / 255.0
    prev_norm = x_norm if prev_rgb is None else prev_rgb / 255.0
    delta = x_norm - prev_norm
    coords = self._coords(x.shape[0], x.shape[-2], x.shape[-1], x.device, x.dtype)

    inp = torch.cat((x_norm, prev_norm, delta, coords), dim=1)
    x1 = torch.relu(self.stem(self.down(inp)))
    x1 = self.enc1(x1)

    x2 = self.down2(x1)
    x2 = self.enc2(x2)
    state = self.gru(x2, prev_state)

    up = self.up1(state)
    if up.shape[-2:] != x1.shape[-2:]:
      up = F.interpolate(up, size=x1.shape[-2:], mode="bilinear", align_corners=False)
    fused = self.fuse(torch.cat((up, x1), dim=1))
    residual = self.up(self.head(fused))
    out = (x_norm + residual).clamp(0.0, 1.0) * 255.0
    return out, state

  def forward(
    self,
    x: torch.Tensor,
    prev_rgb: torch.Tensor | None = None,
    prev_state: torch.Tensor | None = None,
    return_state: bool = False,
  ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    out, state = self.step(x, prev_rgb=prev_rgb, prev_state=prev_state)
    if return_state:
      return out, state
    return out

  def forward_pair(self, comp_a: torch.Tensor, comp_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    inf_a, state = self.step(comp_a)
    inf_b, _ = self.step(comp_b, prev_rgb=inf_a, prev_state=state)
    return inf_a, inf_b


def infer_features_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int:
  weight = state_dict.get("stem.weight")
  if weight is None:
    raise KeyError("Unsupported model state_dict: missing stem.weight")
  return int(weight.shape[0])
