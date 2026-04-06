#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import av
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file

from frame_utils import camera_size, frame_count, yuv420_to_rgb
from modules import PoseNet, SegNet, posenet_sd_path, segnet_sd_path

SINGLE_PANEL_W = 640
SINGLE_PANEL_H = 480
SINGLE_SIDE_W = 320
SINGLE_HEADER_H = 56
SINGLE_CANVAS_W = SINGLE_PANEL_W * 2 + SINGLE_SIDE_W
SINGLE_CANVAS_H = SINGLE_PANEL_H + SINGLE_HEADER_H

COMPARE_PANEL_W = 512
COMPARE_PANEL_H = 384
COMPARE_SIDE_W = 360
COMPARE_HEADER_H = 60
COMPARE_CANVAS_W = COMPARE_PANEL_W * 2 + COMPARE_SIDE_W
COMPARE_CANVAS_H = COMPARE_PANEL_H * 2 + COMPARE_HEADER_H

SEG_PALETTE = np.array([
  [27, 32, 46],
  [230, 84, 59],
  [243, 190, 55],
  [70, 164, 255],
  [63, 196, 142],
], dtype=np.uint8)

BG = (12, 14, 20)
FG = (236, 239, 244)
MUTED = (155, 161, 176)
GRID = (54, 61, 79)
REF = (121, 192, 255)
CMP = (255, 177, 66)
POS = (90, 205, 127)
NEG = (239, 83, 80)


def pick_device(device_arg: str | None) -> torch.device:
  if device_arg:
    return torch.device(device_arg)
  if torch.cuda.is_available():
    return torch.device("cuda", 0)
  if torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


def _raw_frame_count(path: Path) -> int:
  width, height = camera_size
  frame_bytes = width * height * 3
  size = path.stat().st_size
  if size % frame_bytes != 0:
    raise ValueError(f"{path} size {size} is not divisible by raw frame size {frame_bytes}")
  return size // frame_bytes


def count_frames_for_path(path: Path) -> int:
  if path.suffix == ".raw":
    return _raw_frame_count(path)
  return frame_count(str(path))


def iter_rgb_frames(path: Path, max_frames: int | None = None) -> Iterator[torch.Tensor]:
  if path.suffix == ".raw":
    width, height = camera_size
    n_frames = _raw_frame_count(path)
    if max_frames is not None:
      n_frames = min(n_frames, max_frames)
    mm = np.memmap(path, dtype=np.uint8, mode="r", shape=(n_frames, height, width, 3))
    try:
      for idx in range(n_frames):
        yield torch.from_numpy(mm[idx].copy())
    finally:
      del mm
    return

  fmt = "hevc" if path.suffix == ".hevc" else None
  container = av.open(str(path), format=fmt)
  stream = container.streams.video[0]
  try:
    for idx, frame in enumerate(container.decode(stream)):
      if max_frames is not None and idx >= max_frames:
        break
      yield yuv420_to_rgb(frame)
  finally:
    container.close()


def load_models(device: torch.device) -> tuple[PoseNet, SegNet]:
  posenet = PoseNet().eval().to(device)
  segnet = SegNet().eval().to(device)
  posenet.load_state_dict(load_file(posenet_sd_path, device=str(device)))
  segnet.load_state_dict(load_file(segnet_sd_path, device=str(device)))
  return posenet, segnet


def _take_batch(it: Iterator[torch.Tensor], batch_size: int) -> list[torch.Tensor]:
  batch = []
  try:
    while len(batch) < batch_size:
      batch.append(next(it))
  except StopIteration:
    pass
  return batch


def _stack_frames_for_seg(frames: list[torch.Tensor], device: torch.device) -> torch.Tensor:
  x = torch.stack([frame.permute(2, 0, 1) for frame in frames], dim=0).unsqueeze(1).float()
  return x.to(device)


def _analyze_batch(
  frames: list[torch.Tensor],
  prev_frame: torch.Tensor | None,
  posenet: PoseNet,
  segnet: SegNet,
  device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray], torch.Tensor | None]:
  if not frames:
    return [], [], prev_frame

  with torch.inference_mode():
    seg_in = _stack_frames_for_seg(frames, device)
    seg_logits = segnet(segnet.preprocess_input(seg_in))
  seg_classes = [cls.astype(np.uint8) for cls in seg_logits.argmax(dim=1).cpu().numpy()]

  pose_vectors: list[np.ndarray | None] = [None] * len(frames)
  pair_tensors = []
  pair_slots = []
  if prev_frame is None:
    pose_vectors[0] = np.zeros(6, dtype=np.float32)
  else:
    pair_tensors.append(torch.stack([prev_frame.permute(2, 0, 1), frames[0].permute(2, 0, 1)], dim=0))
    pair_slots.append(0)

  for idx in range(1, len(frames)):
    pair_tensors.append(torch.stack([frames[idx - 1].permute(2, 0, 1), frames[idx].permute(2, 0, 1)], dim=0))
    pair_slots.append(idx)

  if pair_tensors:
    with torch.inference_mode():
      pose_in = torch.stack(pair_tensors, dim=0).float().to(device)
      pose_out = posenet(posenet.preprocess_input(pose_in))["pose"][:, :6].cpu().numpy().astype(np.float32)
    for slot, vec in zip(pair_slots, pose_out, strict=True):
      pose_vectors[slot] = vec

  resolved = [vec if vec is not None else np.zeros(6, dtype=np.float32) for vec in pose_vectors]
  return seg_classes, resolved, frames[-1]


def _resize_rgb(rgb: np.ndarray, size: tuple[int, int], resample: int) -> np.ndarray:
  return np.array(Image.fromarray(rgb).resize(size, resample=resample))


def _seg_to_rgb(seg_classes: np.ndarray) -> np.ndarray:
  return SEG_PALETTE[seg_classes]


def _draw_history(
  draw: ImageDraw.ImageDraw,
  x0: int,
  y0: int,
  width: int,
  height: int,
  values: list[float],
  color: tuple[int, int, int],
  label: str,
  font: ImageFont.ImageFont,
) -> None:
  draw.text((x0, y0 - 20), label, font=font, fill=MUTED)
  draw.rectangle((x0, y0, x0 + width, y0 + height), outline=GRID, width=1)
  if len(values) < 2:
    return
  vmax = max(max(values), 1e-6)
  points = []
  start = max(0, len(values) - width)
  for idx, value in enumerate(values[start:]):
    px = x0 + idx
    py = y0 + height - 1 - int((value / vmax) * (height - 6))
    points.append((px, py))
  if len(points) >= 2:
    draw.line(points, fill=color, width=2)


def _draw_seg_legend(
  draw: ImageDraw.ImageDraw,
  x0: int,
  y0: int,
  font: ImageFont.ImageFont,
) -> None:
  draw.text((x0, y0), "SegNet argmax classes", font=font, fill=FG)
  for idx, color in enumerate(SEG_PALETTE.tolist()):
    top = y0 + 24 + idx * 22
    draw.rounded_rectangle((x0, top, x0 + 16, top + 16), radius=3, fill=tuple(color))
    draw.text((x0 + 24, top - 1), f"class {idx}", font=font, fill=MUTED)


def _render_single_frame(
  rgb: np.ndarray,
  seg_rgb: np.ndarray,
  pose_vec: np.ndarray,
  pose_history: list[float],
  frame_idx: int,
  n_frames: int,
  fps: int,
  max_abs_pose: float,
  font: ImageFont.ImageFont,
  small_font: ImageFont.ImageFont,
) -> np.ndarray:
  canvas = Image.new("RGB", (SINGLE_CANVAS_W, SINGLE_CANVAS_H), BG)
  draw = ImageDraw.Draw(canvas)

  draw.text((18, 16), f"frame {frame_idx:04d}/{n_frames - 1:04d}", font=font, fill=FG)
  draw.text((220, 16), f"time {frame_idx / fps:6.2f}s", font=font, fill=FG)
  draw.text((380, 16), "SegNet argmax palette + PoseNet pair output", font=small_font, fill=MUTED)

  rgb_panel = Image.fromarray(_resize_rgb(rgb, (SINGLE_PANEL_W, SINGLE_PANEL_H), Image.Resampling.BILINEAR))
  seg_panel = Image.fromarray(_resize_rgb(seg_rgb, (SINGLE_PANEL_W, SINGLE_PANEL_H), Image.Resampling.NEAREST))
  canvas.paste(rgb_panel, (0, SINGLE_HEADER_H))
  canvas.paste(seg_panel, (SINGLE_PANEL_W, SINGLE_HEADER_H))

  draw.rectangle((0, SINGLE_HEADER_H, SINGLE_PANEL_W - 1, SINGLE_CANVAS_H - 1), outline=GRID, width=1)
  draw.rectangle((SINGLE_PANEL_W, SINGLE_HEADER_H, SINGLE_PANEL_W * 2 - 1, SINGLE_CANVAS_H - 1), outline=GRID, width=1)
  draw.text((18, SINGLE_HEADER_H + 14), "RGB frame", font=font, fill=FG)
  draw.text((SINGLE_PANEL_W + 18, SINGLE_HEADER_H + 14), "SegNet argmax", font=font, fill=FG)

  _draw_seg_legend(draw, SINGLE_PANEL_W + 18, SINGLE_HEADER_H + SINGLE_PANEL_H - 146, small_font)

  x0 = SINGLE_PANEL_W * 2 + 16
  y0 = SINGLE_HEADER_H + 14
  draw.text((x0, y0), "PoseNet output", font=font, fill=FG)
  draw.text((x0, y0 + 22), "pair(prev, curr), dims 0..5", font=small_font, fill=MUTED)

  bar_left = x0 + 20
  bar_top = y0 + 64
  bar_width = SINGLE_SIDE_W - 60
  bar_height = 32
  zero_x = bar_left + bar_width // 2
  scale = max(max_abs_pose, 1e-6)

  for idx, value in enumerate(pose_vec.tolist()):
    top = bar_top + idx * 52
    draw.line((bar_left, top + bar_height // 2, bar_left + bar_width, top + bar_height // 2), fill=GRID, width=1)
    draw.line((zero_x, top, zero_x, top + bar_height), fill=(200, 200, 200), width=1)
    span = int((value / scale) * (bar_width // 2 - 8))
    if span >= 0:
      rect = [zero_x, top + 5, zero_x + span, top + bar_height - 5]
      color = POS
    else:
      rect = [zero_x + span, top + 5, zero_x, top + bar_height - 5]
      color = NEG
    if rect[0] != rect[2]:
      draw.rounded_rectangle(rect, radius=4, fill=color)
    draw.text((bar_left, top - 18), f"p{idx}: {value:+.4f}", font=small_font, fill=FG)

  _draw_history(draw, x0 + 20, y0 + 396, SINGLE_SIDE_W - 40, 72, pose_history, REF, "pose magnitude history", small_font)
  return np.array(canvas)


def _render_compare_frame(
  ref_rgb: np.ndarray,
  cmp_rgb: np.ndarray,
  ref_seg_rgb: np.ndarray,
  cmp_seg_rgb: np.ndarray,
  ref_pose: np.ndarray,
  cmp_pose: np.ndarray,
  seg_disagreement: float,
  pose_mse: float,
  pose_history: list[float],
  seg_history: list[float],
  frame_idx: int,
  n_frames: int,
  fps: int,
  max_abs_pose: float,
  max_abs_delta: float,
  font: ImageFont.ImageFont,
  small_font: ImageFont.ImageFont,
) -> np.ndarray:
  canvas = Image.new("RGB", (COMPARE_CANVAS_W, COMPARE_CANVAS_H), BG)
  draw = ImageDraw.Draw(canvas)

  draw.text((18, 16), f"frame {frame_idx:04d}/{n_frames - 1:04d}", font=font, fill=FG)
  draw.text((220, 16), f"time {frame_idx / fps:6.2f}s", font=font, fill=FG)
  draw.text((400, 16), "reference vs reconstruction", font=small_font, fill=MUTED)

  panels = [
    ("reference RGB", ref_rgb, (0, COMPARE_HEADER_H), Image.Resampling.BILINEAR),
    ("candidate RGB", cmp_rgb, (COMPARE_PANEL_W, COMPARE_HEADER_H), Image.Resampling.BILINEAR),
    ("reference SegNet argmax", ref_seg_rgb, (0, COMPARE_HEADER_H + COMPARE_PANEL_H), Image.Resampling.NEAREST),
    ("candidate SegNet argmax", cmp_seg_rgb, (COMPARE_PANEL_W, COMPARE_HEADER_H + COMPARE_PANEL_H), Image.Resampling.NEAREST),
  ]
  for title, image, (x0, y0), resample in panels:
    panel = Image.fromarray(_resize_rgb(image, (COMPARE_PANEL_W, COMPARE_PANEL_H), resample))
    canvas.paste(panel, (x0, y0))
    draw.rectangle((x0, y0, x0 + COMPARE_PANEL_W - 1, y0 + COMPARE_PANEL_H - 1), outline=GRID, width=1)
    draw.text((x0 + 16, y0 + 12), title, font=font, fill=FG)

  x0 = COMPARE_PANEL_W * 2 + 16
  y0 = COMPARE_HEADER_H + 14
  draw.text((x0, y0), "Current metrics", font=font, fill=FG)
  draw.text((x0, y0 + 26), f"Seg disagreement: {seg_disagreement:.4f}", font=small_font, fill=FG)
  draw.text((x0, y0 + 44), f"Pose MSE: {pose_mse:.6f}", font=small_font, fill=FG)

  scale = max(max_abs_pose, 1e-6)
  delta_scale = max(max_abs_delta, 1e-6)
  bar_left = x0 + 18
  bar_width = COMPARE_SIDE_W - 48
  for idx, (ref_value, cmp_value) in enumerate(zip(ref_pose.tolist(), cmp_pose.tolist(), strict=True)):
    top = y0 + 86 + idx * 62
    zero_x = bar_left + bar_width // 2
    draw.text((bar_left, top - 18), f"p{idx}: ref {ref_value:+.4f} cmp {cmp_value:+.4f}", font=small_font, fill=FG)
    for row, value, color in [(0, ref_value, REF), (18, cmp_value, CMP)]:
      row_top = top + row
      draw.line((bar_left, row_top + 8, bar_left + bar_width, row_top + 8), fill=GRID, width=1)
      draw.line((zero_x, row_top - 2, zero_x, row_top + 18), fill=(200, 200, 200), width=1)
      span = int((value / scale) * (bar_width // 2 - 8))
      if span >= 0:
        rect = [zero_x, row_top + 2, zero_x + span, row_top + 14]
      else:
        rect = [zero_x + span, row_top + 2, zero_x, row_top + 14]
      if rect[0] != rect[2]:
        draw.rounded_rectangle(rect, radius=3, fill=color)
    delta = cmp_value - ref_value
    delta_top = top + 38
    draw.text((bar_left, delta_top), f"d {delta:+.4f}", font=small_font, fill=MUTED)
    delta_span = int((delta / delta_scale) * (bar_width // 2 - 8))
    zero_x = bar_left + bar_width // 2
    draw.line((bar_left, delta_top + 18, bar_left + bar_width, delta_top + 18), fill=GRID, width=1)
    draw.line((zero_x, delta_top + 10, zero_x, delta_top + 26), fill=(160, 160, 160), width=1)
    if delta_span >= 0:
      rect = [zero_x, delta_top + 12, zero_x + delta_span, delta_top + 24]
      color = POS
    else:
      rect = [zero_x + delta_span, delta_top + 12, zero_x, delta_top + 24]
      color = NEG
    if rect[0] != rect[2]:
      draw.rounded_rectangle(rect, radius=3, fill=color)

  _draw_history(draw, x0 + 18, y0 + 486, COMPARE_SIDE_W - 36, 90, pose_history, REF, "Pose MSE history", small_font)
  _draw_history(draw, x0 + 18, y0 + 612, COMPARE_SIDE_W - 36, 90, seg_history, CMP, "Seg disagreement history", small_font)
  _draw_seg_legend(draw, x0 + 18, y0 + 720, small_font)
  return np.array(canvas)


def _open_output(output_path: Path, width: int, height: int, fps: int) -> tuple[av.container.OutputContainer, av.video.stream.VideoStream]:
  output_path.parent.mkdir(parents=True, exist_ok=True)
  container = av.open(str(output_path), mode="w")
  stream = container.add_stream("libx264", rate=fps)
  stream.width = width
  stream.height = height
  stream.pix_fmt = "yuv420p"
  stream.options = {"crf": "18", "preset": "medium"}
  return container, stream


def render_single_dashboard(
  *,
  input_path: Path,
  output_path: Path,
  device: torch.device,
  fps: int,
  max_frames: int | None = None,
  batch_size: int = 16,
) -> None:
  n_frames = count_frames_for_path(input_path)
  if max_frames is not None:
    n_frames = min(n_frames, max_frames)
  if n_frames < 2:
    raise ValueError(f"Need at least 2 frames in {input_path}, found {n_frames}")

  posenet, segnet = load_models(device)
  frame_iter = iter_rgb_frames(input_path, max_frames=n_frames)
  container, stream = _open_output(output_path, SINGLE_CANVAS_W, SINGLE_CANVAS_H, fps)
  font = ImageFont.load_default()
  small_font = ImageFont.load_default()
  pose_history: list[float] = []
  max_abs_pose = 1e-6
  prev_frame = None
  frame_idx = 0

  try:
    while True:
      batch = _take_batch(frame_iter, batch_size)
      if not batch:
        break
      seg_classes, pose_vectors, prev_frame = _analyze_batch(batch, prev_frame, posenet, segnet, device)
      for frame, seg_cls, pose_vec in zip(batch, seg_classes, pose_vectors, strict=True):
        max_abs_pose = max(max_abs_pose, float(np.max(np.abs(pose_vec))))
        pose_history.append(float(np.linalg.norm(pose_vec)))
        dashboard = _render_single_frame(
          rgb=frame.numpy(),
          seg_rgb=_seg_to_rgb(seg_cls),
          pose_vec=pose_vec,
          pose_history=pose_history,
          frame_idx=frame_idx,
          n_frames=n_frames,
          fps=fps,
          max_abs_pose=max_abs_pose,
          font=font,
          small_font=small_font,
        )
        video_frame = av.VideoFrame.from_ndarray(dashboard, format="rgb24")
        for packet in stream.encode(video_frame):
          container.mux(packet)
        frame_idx += 1
  finally:
    for packet in stream.encode():
      container.mux(packet)
    container.close()


def render_comparison_dashboard(
  *,
  reference_path: Path,
  candidate_path: Path,
  output_path: Path,
  device: torch.device,
  fps: int,
  max_frames: int | None = None,
  batch_size: int = 16,
) -> None:
  n_ref = count_frames_for_path(reference_path)
  n_cmp = count_frames_for_path(candidate_path)
  n_frames = min(n_ref, n_cmp)
  if max_frames is not None:
    n_frames = min(n_frames, max_frames)
  if n_frames < 2:
    raise ValueError(f"Need at least 2 aligned frames, found {n_frames}")

  posenet, segnet = load_models(device)
  ref_iter = iter_rgb_frames(reference_path, max_frames=n_frames)
  cmp_iter = iter_rgb_frames(candidate_path, max_frames=n_frames)
  container, stream = _open_output(output_path, COMPARE_CANVAS_W, COMPARE_CANVAS_H, fps)
  font = ImageFont.load_default()
  small_font = ImageFont.load_default()
  pose_history: list[float] = []
  seg_history: list[float] = []
  max_abs_pose = 1e-6
  max_abs_delta = 1e-6
  prev_ref = None
  prev_cmp = None
  frame_idx = 0

  try:
    while True:
      ref_batch = _take_batch(ref_iter, batch_size)
      cmp_batch = _take_batch(cmp_iter, batch_size)
      if not ref_batch and not cmp_batch:
        break
      if len(ref_batch) != len(cmp_batch):
        raise ValueError("Reference and candidate streams have different lengths")

      ref_seg, ref_pose, prev_ref = _analyze_batch(ref_batch, prev_ref, posenet, segnet, device)
      cmp_seg, cmp_pose, prev_cmp = _analyze_batch(cmp_batch, prev_cmp, posenet, segnet, device)

      for ref_frame, cmp_frame, ref_seg_cls, cmp_seg_cls, ref_pose_vec, cmp_pose_vec in zip(
        ref_batch,
        cmp_batch,
        ref_seg,
        cmp_seg,
        ref_pose,
        cmp_pose,
        strict=True,
      ):
        seg_disagreement = float(np.mean(ref_seg_cls != cmp_seg_cls))
        pose_mse = float(np.mean((ref_pose_vec - cmp_pose_vec) ** 2))
        pose_history.append(pose_mse)
        seg_history.append(seg_disagreement)
        max_abs_pose = max(
          max_abs_pose,
          float(np.max(np.abs(ref_pose_vec))),
          float(np.max(np.abs(cmp_pose_vec))),
        )
        max_abs_delta = max(max_abs_delta, float(np.max(np.abs(cmp_pose_vec - ref_pose_vec))))

        dashboard = _render_compare_frame(
          ref_rgb=ref_frame.numpy(),
          cmp_rgb=cmp_frame.numpy(),
          ref_seg_rgb=_seg_to_rgb(ref_seg_cls),
          cmp_seg_rgb=_seg_to_rgb(cmp_seg_cls),
          ref_pose=ref_pose_vec,
          cmp_pose=cmp_pose_vec,
          seg_disagreement=seg_disagreement,
          pose_mse=pose_mse,
          pose_history=pose_history,
          seg_history=seg_history,
          frame_idx=frame_idx,
          n_frames=n_frames,
          fps=fps,
          max_abs_pose=max_abs_pose,
          max_abs_delta=max_abs_delta,
          font=font,
          small_font=small_font,
        )
        video_frame = av.VideoFrame.from_ndarray(dashboard, format="rgb24")
        for packet in stream.encode(video_frame):
          container.mux(packet)
        frame_idx += 1
  finally:
    for packet in stream.encode():
      container.mux(packet)
    container.close()
