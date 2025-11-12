#!/usr/bin/env python
import os, mmap, torch, warnings
from typing import List
from pathlib import Path
import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
warnings.filterwarnings("ignore", message=r"Please set `reader_name`.*", category=Warning, module=r"nvidia\.dali\.plugin\.base_iterator")

seq_len = 2
camera_size = (1164, 874)
camera_fl = 910.
fcam_model_input_size = ecam_model_input_size = (512, 256)
segnet_model_input_size = (512, 384)
fcam_model_cy = 47.6
ecam_model_cy = 0.5 * (256 + fcam_model_cy)
fcam_model_fl, ecam_model_fl = 910.0, 455.0
camera_intrinsics = torch.tensor([
  [camera_fl ,          0., 0.5 * camera_size[0]],
  [         0., camera_fl, 0.5 * camera_size[1]],
  [         0.,           0.,          1.]
])
fcam_model_intrinsics  = torch.tensor([
  [fcam_model_fl,  0.0,  0.5 * fcam_model_input_size[0]],
  [0.0,  fcam_model_fl,                fcam_model_cy],
  [0.0,  0.0,                                   1.0]])
ecam_model_intrinsics = torch.tensor([
  [ecam_model_fl,  0.0,  0.5 * ecam_model_input_size [0]],
  [0.0,  ecam_model_fl,     ecam_model_cy],
  [0.0,  0.0,                                     1.0]])

def affine_transform_image(image, from_size, to_size, from_intrinsics, to_intrinsics, mode="bilinear", align_corners=True):
  fx_f, fy_f, cx_f, cy_f = from_intrinsics[0,0].item(), from_intrinsics[1,1].item(), from_intrinsics[0,2].item(), from_intrinsics[1,2].item()
  fx_t, fy_t, cx_t, cy_t = to_intrinsics[0,0].item(), to_intrinsics[1,1].item(), to_intrinsics[0,2].item(), to_intrinsics[1,2].item()
  sx, sy = (fx_t / fx_f), (fy_t / fy_f)
  w_crop, h_crop = int(round(to_size[0] / sx)), int(round(to_size[1] / sy))
  x0, y0 = int(round(cx_f - cx_t / sx)), int(round(cy_f - cy_t / sy))
  x0, y0 = max(0, min(from_size[0] - w_crop, x0)), max(0, min(from_size[1] - h_crop, y0))
  image = image[..., y0:y0+h_crop, x0:x0+w_crop]
  return torch.nn.functional.interpolate(image, size=(to_size[1], to_size[0]), mode=mode, align_corners=align_corners)

@pipeline_def
def inbuf_video_pipe():
  vid = fn.experimental.inputs.video(
    name="inbuf",
    sequence_length=2,
    device="mixed",
    no_copy=True,
    blocking=False,
    last_sequence_policy="partial",
  )
  return vid

def hevc_buffer_mmap(path: str):
  f = open(path, "rb")
  mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
  mv = memoryview(mm)
  return mv, (mm, f)

def hevc_frame_count(path: str) -> int:
  # assumes one slice per frame x265 default
  with open(path, 'rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as b:
    frames, i = 0, 0
    find = b.find
    while True:
      j = find(b'\x00\x00\x01', i)
      if j < 0: return frames
      p = j + 3
      if ((b[p] >> 1) & 0x3F) <= 31:   # any VCL slice
        frames += 1
      i = p

@torch.no_grad()
def rgb_to_yuv6(rgb_chw: torch.Tensor) -> torch.Tensor:
  H, W = rgb_chw.shape[-2], rgb_chw.shape[-1]
  H2, W2 = H // 2, W // 2
  rgb = rgb_chw[..., : , :2*H2, :2*W2]

  R = rgb[..., 0, :, :]
  G = rgb[..., 1, :, :]
  B = rgb[..., 2, :, :]

  kYR, kYG, kYB = 0.299, 0.587, 0.114
  Y = (R * kYR + G * kYG + B * kYB).clamp_(0.0, 255.0)
  U = ((B - Y) / 1.772 + 128.0).clamp_(0.0, 255.0)
  V = ((R - Y) / 1.402 + 128.0).clamp_(0.0, 255.0)

  U_sub = (
    U[..., 0::2, 0::2] + U[..., 1::2, 0::2] +
    U[..., 0::2, 1::2] + U[..., 1::2, 1::2]
  ) * 0.25
  V_sub = (
    V[..., 0::2, 0::2] + V[..., 1::2, 0::2] +
    V[..., 0::2, 1::2] + V[..., 1::2, 1::2]
  ) * 0.25

  y00 = Y[..., 0::2, 0::2]
  y10 = Y[..., 1::2, 0::2]
  y01 = Y[..., 0::2, 1::2]
  y11 = Y[..., 1::2, 1::2]
  return torch.stack([y00, y10, y01, y11, U_sub, V_sub], dim=-3)

class DaliHevcDataset(torch.utils.data.IterableDataset):
  def __init__(self, file_names: List[str], archive_path:Path, data_dir:Path, batch_size: int, device_id: int, num_threads: int = 2, seed: int = 123, prefetch_queue_depth: int = 4):
    super().__init__()
    self.all_file_names = file_names
    self.archive_path = archive_path
    self.batch_size = batch_size
    self.device_id = device_id
    self.data_dir = data_dir
    self.num_threads = num_threads
    self.seed = seed
    self.prefetch_queue_depth = prefetch_queue_depth
    self.rank, self.world_size = self.get_dist_info()
    self.file_names = self.all_file_names[self.rank::self.world_size]
    self.paths = [str(data_dir / fn) for fn in self.file_names]

  def prepare_data(self):
    if self.device_id == 0:
      if not all((self.data_dir / fn).exists() for fn in self.file_names):
        import zipfile
        print(f"decompressing archive {self.archive_path} to {self.data_dir}...")
        with zipfile.ZipFile(self.archive_path, 'r') as zip_ref:
          zip_ref.extractall(self.data_dir)
        if self.world_size > 1:
          torch.distributed.barrier()
    assert all((self.data_dir / fn).exists() for fn in self.file_names)
    print(f"DALI HEVC dataset on rank {self.rank} with {len(self.paths)} files.")

  def get_dist_info(self):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
      return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))

  def __iter__(self):
    for path in self.paths:
      frames_per_file = hevc_frame_count(path)
      num_sequences = frames_per_file // seq_len
      pipe_size = (num_sequences // self.batch_size) * self.batch_size
      pipe = inbuf_video_pipe(batch_size=self.batch_size, num_threads=self.num_threads, device_id=self.device_id, prefetch_queue_depth=self.prefetch_queue_depth)
      pipe.build()
      mv, (mm, f) = hevc_buffer_mmap(path)
      pipe.feed_input("inbuf", [mv])
      it = DALIGenericIterator([pipe], output_map=["video"], size=pipe_size, auto_reset=False, last_batch_policy=LastBatchPolicy.PARTIAL, last_batch_padded=False, prepare_first_batch=False)

      for idx, data in enumerate(it):
        vid = data[0]["video"]
        yield path, idx, vid

      torch.cuda.synchronize()
      it.reset()
      del it, pipe
      mv.release()
      mm.close()
      f.close()

if __name__ == "__main__":
  batch_size = 32
  device_id = 0
  files = ['b0c9d2329ad1606b|2018-07-29--11-17-20/7/video.hevc']
  uncompressed_archive_path = Path('./test_videos.zip')
  uncompressed_data_dir = Path('./deflated_test_videos/')
  ds = DaliHevcDataset(files, archive_path=uncompressed_archive_path, data_dir=uncompressed_data_dir, batch_size=batch_size, device_id=device_id)
  ds.prepare_data()
  for i, (path, idx, batch) in enumerate(ds):
    assert batch.shape == (batch_size, seq_len, camera_size[1], camera_size[0], 3), f"unexpected batch shape: {batch.shape}"
    print(i, batch.shape)