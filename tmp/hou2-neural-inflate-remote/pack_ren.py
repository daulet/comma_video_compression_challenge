#!/usr/bin/env python3
"""Pack REN checkpoints into submission-friendly formats.

Outputs:
- int8 custom container + bz2: ren_model.int8.bz2
- float16 torch checkpoint + bz2: ren_model.pt.bz2
"""

from __future__ import annotations

import argparse
import bz2
import io
import struct
from pathlib import Path

import numpy as np
import torch


def _tensor_to_int8(t: torch.Tensor) -> tuple[np.ndarray, float]:
  arr = t.detach().cpu().float().numpy()
  max_abs = float(np.max(np.abs(arr))) if arr.size else 0.0
  if max_abs < 1e-12:
    scale = 1.0
    q = np.zeros_like(arr, dtype=np.int8)
  else:
    scale = max_abs / 127.0
    q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
  return q, scale


def save_int8_bz2(state_dict: dict[str, torch.Tensor], output_path: Path) -> None:
  buf = io.BytesIO()
  buf.write(struct.pack("<I", len(state_dict)))

  for name, tensor in state_dict.items():
    q, scale = _tensor_to_int8(tensor)
    name_b = name.encode("utf-8")
    shape = list(q.shape)

    buf.write(struct.pack("<I", len(name_b)))
    buf.write(name_b)
    buf.write(struct.pack("<I", len(shape)))
    for dim in shape:
      buf.write(struct.pack("<I", int(dim)))
    buf.write(struct.pack("<f", float(scale)))

    data = q.tobytes(order="C")
    buf.write(struct.pack("<I", len(data)))
    buf.write(data)

  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_bytes(bz2.compress(buf.getvalue(), compresslevel=9))


def save_f16_bz2(state_dict: dict[str, torch.Tensor], output_path: Path) -> None:
  sd_f16 = {k: v.detach().cpu().to(torch.float16) for k, v in state_dict.items()}
  payload = io.BytesIO()
  torch.save(sd_f16, payload)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_bytes(bz2.compress(payload.getvalue(), compresslevel=9))


def load_int8_bz2(path: Path) -> dict[str, torch.Tensor]:
  raw = bz2.decompress(path.read_bytes())
  buf = io.BytesIO(raw)
  n_tensors = struct.unpack("<I", buf.read(4))[0]

  sd: dict[str, torch.Tensor] = {}
  for _ in range(n_tensors):
    name_len = struct.unpack("<I", buf.read(4))[0]
    name = buf.read(name_len).decode("utf-8")

    n_dims = struct.unpack("<I", buf.read(4))[0]
    shape = [struct.unpack("<I", buf.read(4))[0] for _ in range(n_dims)]
    scale = struct.unpack("<f", buf.read(4))[0]

    data_len = struct.unpack("<I", buf.read(4))[0]
    q = np.frombuffer(buf.read(data_len), dtype=np.int8).reshape(shape)
    sd[name] = torch.from_numpy(q.astype(np.float32) * scale)

  return sd


def main() -> None:
  parser = argparse.ArgumentParser(description="Pack REN checkpoint for neural_inflate")
  parser.add_argument("--input", type=Path, required=True, help="Path to ren_model.pt")
  parser.add_argument("--output-int8", type=Path, default=None, help="Output path for ren_model.int8.bz2")
  parser.add_argument("--output-f16", type=Path, default=None, help="Output path for ren_model.pt.bz2")
  parser.add_argument("--skip-f16", action="store_true", help="Skip writing float16 bz2")
  parser.add_argument("--verify", action="store_true", help="Verify int8 dequantization error")
  args = parser.parse_args()

  if not args.input.exists():
    raise FileNotFoundError(f"Input checkpoint not found: {args.input}")

  out_int8 = args.output_int8 or args.input.with_suffix(".int8.bz2")
  out_f16 = args.output_f16 or args.input.with_suffix(".pt.bz2")

  state_dict = torch.load(args.input, map_location="cpu", weights_only=True)
  if not isinstance(state_dict, dict):
    raise TypeError("Expected state_dict dictionary")

  save_int8_bz2(state_dict, out_int8)
  if not args.skip_f16:
    save_f16_bz2(state_dict, out_f16)

  in_size = args.input.stat().st_size
  int8_size = out_int8.stat().st_size
  print(f"input:       {args.input} ({in_size:,} bytes)")
  print(f"output int8: {out_int8} ({int8_size:,} bytes)")

  if not args.skip_f16:
    f16_size = out_f16.stat().st_size
    print(f"output f16:  {out_f16} ({f16_size:,} bytes)")

  if args.verify:
    recon = load_int8_bz2(out_int8)
    worst_abs = 0.0
    mean_abs = 0.0
    n = 0
    for name, ref in state_dict.items():
      q = recon[name]
      diff = (ref.detach().cpu().float() - q).abs()
      worst_abs = max(worst_abs, float(diff.max().item()))
      mean_abs += float(diff.mean().item())
      n += 1
    mean_abs = mean_abs / max(1, n)
    print(f"verify: mean_abs={mean_abs:.6e}, worst_abs={worst_abs:.6e}")


if __name__ == "__main__":
  main()
