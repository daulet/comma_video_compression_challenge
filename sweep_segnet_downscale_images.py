#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from safetensors.torch import load_file

from frame_utils import camera_size, segnet_model_input_size
from model_diagnostics import count_frames_for_path
from model_diagnostics import iter_rgb_frames
from modules import SegNet, segnet_sd_path

HERE = Path(__file__).resolve().parent
BASE_W, BASE_H = segnet_model_input_size

PALETTES = {
  "current": np.array([
    [27, 32, 46],
    [230, 84, 59],
    [243, 190, 55],
    [70, 164, 255],
    [63, 196, 142],
  ], dtype=np.uint8),
  "search9": np.array([
    [56, 18, 149],
    [20, 32, 235],
    [185, 76, 241],
    [214, 38, 123],
    [207, 58, 53],
  ], dtype=np.uint8),
}


RESAMPLE_MAP = {
  "nearest": Image.Resampling.NEAREST,
  "box": Image.Resampling.BOX,
  "bilinear": Image.Resampling.BILINEAR,
  "bicubic": Image.Resampling.BICUBIC,
  "lanczos": Image.Resampling.LANCZOS,
}


def pick_device(device_arg: str | None) -> torch.device:
  if device_arg:
    return torch.device(device_arg)
  if torch.cuda.is_available():
    return torch.device("cuda", 0)
  if torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


@dataclass
class SegCache:
  orig_low_frames: list[np.ndarray]
  seg_classes: list[np.ndarray]
  targets: torch.Tensor
  total_video_frames: int


def _encode_image(img: Image.Image, codec: str, quality: int | None) -> bytes:
  buf = io.BytesIO()
  if codec == "png":
    img.save(buf, format="PNG", compress_level=9, optimize=True)
  elif codec == "webp_lossless":
    img.save(buf, format="WEBP", lossless=True, quality=100, method=6)
  elif codec == "webp":
    if quality is None:
      raise ValueError("quality is required for lossy webp")
    img.save(buf, format="WEBP", lossless=False, quality=quality, method=6)
  elif codec == "jpeg":
    if quality is None:
      raise ValueError("quality is required for jpeg")
    img.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True, subsampling=0)
  else:
    raise ValueError(f"unsupported codec {codec}")
  return buf.getvalue()


def _decode_image(payload: bytes) -> Image.Image:
  with Image.open(io.BytesIO(payload)) as img:
    return img.convert("RGB")


def _even_round(value: float) -> int:
  rounded = int(round(value))
  if rounded % 2 == 1:
    rounded += 1
  return max(8, rounded)


def _cache_scored_frames(
  *,
  input_path: Path,
  segnet: SegNet,
  device: torch.device,
  batch_size: int,
  max_scored_frames: int | None,
) -> SegCache:
  total_video_frames = count_frames_for_path(input_path)
  orig_low_frames: list[np.ndarray] = []
  seg_classes: list[np.ndarray] = []
  targets: list[torch.Tensor] = []
  pending = []

  def flush(batch_frames: list[torch.Tensor]) -> None:
    if not batch_frames:
      return
    x = torch.stack([frame.permute(2, 0, 1) for frame in batch_frames], dim=0).unsqueeze(1).float().to(device)
    with torch.inference_mode():
      orig_low = segnet.preprocess_input(x)
      logits = segnet(orig_low)
    low_uint8 = orig_low.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    orig_low_frames.extend([frame.copy() for frame in low_uint8])
    seg_classes.extend([frame.copy() for frame in logits.argmax(dim=1).cpu().numpy().astype(np.uint8)])
    targets.append(logits.argmax(dim=1).cpu().to(torch.uint8))

  scored = 0
  for idx, frame in enumerate(iter_rgb_frames(input_path)):
    if idx % 2 == 0:
      continue
    pending.append(frame)
    scored += 1
    if len(pending) >= batch_size:
      flush(pending)
      pending = []
    if max_scored_frames is not None and scored >= max_scored_frames:
      break
  flush(pending)

  if not orig_low_frames:
    raise ValueError("No scored odd frames found")

  return SegCache(
    orig_low_frames=orig_low_frames,
    seg_classes=seg_classes,
    targets=torch.cat(targets, dim=0),
    total_video_frames=total_video_frames,
  )


def _render_source_low(
  *,
  orig_low_rgb: np.ndarray,
  seg_class: np.ndarray,
  source_mode: str,
  palette: np.ndarray,
  alpha: float,
) -> np.ndarray:
  if source_mode == "orig_low":
    return orig_low_rgb
  palette_rgb = palette[seg_class]
  if source_mode == "palette":
    return palette_rgb
  if source_mode == "overlay":
    blend = alpha * orig_low_rgb.astype(np.float32) + (1.0 - alpha) * palette_rgb.astype(np.float32)
    return np.rint(blend).clip(0, 255).astype(np.uint8)
  raise ValueError(f"unsupported source_mode {source_mode}")


def _render_candidate_full(
  low_rgb: np.ndarray,
  *,
  stored_w: int,
  stored_h: int,
  down_resample: str,
  restore_resample: str,
  inflate_resample: str,
  codec: str,
  quality: int | None,
  unsharp_percent: int,
) -> tuple[np.ndarray, int]:
  img = Image.fromarray(low_rgb, mode="RGB")
  if (stored_w, stored_h) != (BASE_W, BASE_H):
    img = img.resize((stored_w, stored_h), resample=RESAMPLE_MAP[down_resample])
  payload = _encode_image(img, codec, quality)
  decoded = _decode_image(payload)
  if decoded.size != (BASE_W, BASE_H):
    decoded = decoded.resize((BASE_W, BASE_H), resample=RESAMPLE_MAP[restore_resample])
  if unsharp_percent > 0:
    decoded = decoded.filter(ImageFilter.UnsharpMask(radius=1.0, percent=unsharp_percent, threshold=0))
  full = decoded.resize(camera_size, resample=RESAMPLE_MAP[inflate_resample])
  return np.array(full, dtype=np.uint8, copy=True), len(payload)


def evaluate_config(
  *,
  segnet: SegNet,
  cache: SegCache,
  device: torch.device,
  batch_size: int,
  scale: float,
  source_mode: str,
  palette: np.ndarray,
  alpha: float,
  codec: str,
  quality: int | None,
  down_resample: str,
  restore_resample: str,
  inflate_resample: str,
  unsharp_percent: int,
  original_bytes: int,
) -> dict:
  stored_w = _even_round(BASE_W * scale)
  stored_h = _even_round(BASE_H * scale)

  total_bytes = 0
  total_err = 0.0
  total_items = 0
  batch_frames: list[torch.Tensor] = []
  batch_targets: list[torch.Tensor] = []

  def flush() -> None:
    nonlocal total_err, total_items, batch_frames, batch_targets
    if not batch_frames:
      return
    x = torch.stack(batch_frames, dim=0).unsqueeze(1).float().to(device)
    y = torch.stack(batch_targets, dim=0).to(device=device, dtype=torch.long)
    with torch.inference_mode():
      logits = segnet(segnet.preprocess_input(x))
    err = (logits.argmax(dim=1) != y).float().mean(dim=(1, 2))
    total_err += float(err.sum().item())
    total_items += x.shape[0]
    batch_frames = []
    batch_targets = []

  for orig_low_rgb, seg_class, target in zip(cache.orig_low_frames, cache.seg_classes, cache.targets, strict=True):
    low_rgb = _render_source_low(
      orig_low_rgb=orig_low_rgb,
      seg_class=seg_class,
      source_mode=source_mode,
      palette=palette,
      alpha=alpha,
    )
    full_rgb, payload_bytes = _render_candidate_full(
      low_rgb,
      stored_w=stored_w,
      stored_h=stored_h,
      down_resample=down_resample,
      restore_resample=restore_resample,
      inflate_resample=inflate_resample,
      codec=codec,
      quality=quality,
      unsharp_percent=unsharp_percent,
    )
    total_bytes += payload_bytes
    batch_frames.append(torch.from_numpy(full_rgb).permute(2, 0, 1))
    batch_targets.append(target)
    if len(batch_frames) >= batch_size:
      flush()
  flush()

  segnet_dist = total_err / max(total_items, 1)
  total_odd_frames = cache.total_video_frames // 2
  estimated_full_bytes = total_bytes * total_odd_frames / max(total_items, 1)
  rate = estimated_full_bytes / original_bytes
  return {
    "scale": scale,
    "source_mode": source_mode,
    "alpha": alpha,
    "stored_w": stored_w,
    "stored_h": stored_h,
    "codec": codec,
    "quality": quality,
    "down_resample": down_resample,
    "restore_resample": restore_resample,
    "inflate_resample": inflate_resample,
    "unsharp_percent": unsharp_percent,
    "odd_frames": total_items,
    "odd_image_bytes": total_bytes,
    "bytes_per_odd_frame": total_bytes / max(total_items, 1),
    "estimated_full_bytes": estimated_full_bytes,
    "rate": rate,
    "segnet_dist": segnet_dist,
    "seg_score": 100.0 * segnet_dist,
    "seg_rate_score": 100.0 * segnet_dist + 25.0 * rate,
  }


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Sweep SegNet-only downscale/image configurations on scored odd frames.")
  parser.add_argument("--input", type=Path, default=HERE / "videos" / "0.mkv")
  parser.add_argument("--device", type=str, default="cpu", help="cpu|cuda|mps")
  parser.add_argument("--cache-batch-size", type=int, default=8)
  parser.add_argument("--eval-batch-size", type=int, default=8)
  parser.add_argument("--max-scored-frames", type=int, default=None)
  parser.add_argument("--source-modes", nargs="+", choices=["orig_low", "overlay", "palette"], default=["orig_low"])
  parser.add_argument("--palette", choices=sorted(PALETTES), default="search9")
  parser.add_argument("--alphas", type=float, nargs="+", default=[0.995])
  parser.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.875, 0.75, 0.625, 0.5])
  parser.add_argument("--codecs", nargs="+", choices=["png", "webp_lossless", "webp", "jpeg"], default=["png", "webp_lossless"])
  parser.add_argument("--qualities", type=int, nargs="+", default=[95, 90, 85, 80, 70, 60])
  parser.add_argument("--down-resamples", nargs="+", choices=sorted(RESAMPLE_MAP), default=["lanczos"])
  parser.add_argument("--restore-resamples", nargs="+", choices=sorted(RESAMPLE_MAP), default=["lanczos"])
  parser.add_argument("--inflate-resamples", nargs="+", choices=sorted(RESAMPLE_MAP), default=["bilinear"])
  parser.add_argument("--unsharp-percents", type=int, nargs="+", default=[0])
  parser.add_argument("--report", type=Path, default=HERE / "artifacts" / "segnet_downscale_sweep.json")
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  torch.set_num_threads(1)
  device = pick_device(args.device)
  palette = PALETTES[args.palette]
  original_bytes = args.input.stat().st_size
  print(f"device={device}", flush=True)

  segnet = SegNet().eval().to(device)
  segnet.load_state_dict(load_file(segnet_sd_path, device=str(device)))

  print(f"caching odd scored frames from {args.input}", flush=True)
  cache = _cache_scored_frames(
    input_path=args.input,
    segnet=segnet,
    device=device,
    batch_size=args.cache_batch_size,
    max_scored_frames=args.max_scored_frames,
  )
  print(
    f"cached {len(cache.orig_low_frames)} odd frames at {BASE_W}x{BASE_H}",
    flush=True,
  )

  results = []
  for source_mode in args.source_modes:
    alphas = args.alphas if source_mode == "overlay" else [1.0]
    for alpha in alphas:
      for scale in args.scales:
        for codec in args.codecs:
          qualities = args.qualities if codec in {"webp", "jpeg"} else [None]
          for quality in qualities:
            for down_resample in args.down_resamples:
              for restore_resample in args.restore_resamples:
                for inflate_resample in args.inflate_resamples:
                  for unsharp_percent in args.unsharp_percents:
                    print(
                      (
                        f"eval source={source_mode} alpha={alpha:.4f} scale={scale:.4f} "
                        f"codec={codec} quality={quality} down={down_resample} "
                        f"restore={restore_resample} inflate={inflate_resample} "
                        f"unsharp={unsharp_percent}"
                      ),
                      flush=True,
                    )
                    result = evaluate_config(
                      segnet=segnet,
                      cache=cache,
                      device=device,
                      batch_size=args.eval_batch_size,
                      scale=scale,
                      source_mode=source_mode,
                      palette=palette,
                      alpha=alpha,
                      codec=codec,
                      quality=quality,
                      down_resample=down_resample,
                      restore_resample=restore_resample,
                      inflate_resample=inflate_resample,
                      unsharp_percent=unsharp_percent,
                      original_bytes=original_bytes,
                    )
                    print(
                      (
                        f"  seg_rate_score={result['seg_rate_score']:.4f} "
                        f"seg_score={result['seg_score']:.4f} "
                        f"rate={result['rate']:.4f} "
                        f"est_bytes={result['estimated_full_bytes']:.0f}"
                      ),
                      flush=True,
                    )
                    results.append(result)

  results.sort(key=lambda row: (row["seg_rate_score"], row["segnet_dist"], row["estimated_full_bytes"]))
  payload = {
    "input": str(args.input),
    "max_scored_frames": args.max_scored_frames,
    "n_results": len(results),
    "results": results,
  }
  args.report.parent.mkdir(parents=True, exist_ok=True)
  args.report.write_text(json.dumps(payload, indent=2) + "\n")
  print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
  main()
