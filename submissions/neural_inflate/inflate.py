#!/usr/bin/env python
import os, av, torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from frame_utils import camera_size, yuv420_to_rgb

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HERE = os.path.dirname(os.path.abspath(__file__))



class REN(nn.Module):
    def __init__(self, features=32):
        super().__init__()
        self.down = nn.PixelUnshuffle(2)
        self.body = nn.Sequential(
            nn.Conv2d(12, features, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(features, 12, 3, padding=1),
        )
        self.up = nn.PixelShuffle(2)

    def forward(self, x):
        x_norm = x / 255.0
        residual = self.up(self.body(self.down(x_norm)))
        return (x_norm + residual).clamp(0, 1) * 255.0


MODEL = REN(features=32).to(DEVICE).eval()
MODEL.load_state_dict(torch.load(
    os.path.join(HERE, 'ren_model.pt'),
    map_location=DEVICE, weights_only=True
))


def decode_and_resize_to_file(video_path: str, dst: str):
    target_w, target_h = camera_size
    fmt = 'hevc' if video_path.endswith('.hevc') else None
    container = av.open(video_path, format=fmt)
    stream = container.streams.video[0]
    n = 0
    with open(dst, 'wb') as f:
        for frame in container.decode(stream):
            t = yuv420_to_rgb(frame)
            H, W, _ = t.shape
            if H != target_h or W != target_w:
                pil = Image.fromarray(t.numpy())
                pil = pil.resize((target_w, target_h), Image.LANCZOS)
                x = torch.from_numpy(np.array(pil)).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
                with torch.no_grad():
                    x = MODEL(x)
                t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
            f.write(t.contiguous().numpy().tobytes())
            n += 1
    container.close()
    return n


if __name__ == "__main__":
    import sys
    src, dst = sys.argv[1], sys.argv[2]
    n = decode_and_resize_to_file(src, dst)
    print(f"saved {n} frames")
