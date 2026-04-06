#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from model_diagnostics import (
  _analyze_batch,
  _resize_rgb,
  _seg_to_rgb,
  _take_batch,
  count_frames_for_path,
  iter_rgb_frames,
  load_models,
  pick_device,
)
from posenet_saliency import POSE_SALIENCY_COLORS, compute_saliency, find_pair, winner_overlay

HERE = Path(__file__).resolve().parent

RGB_SIZE = (512, 384)
SEG_SIZE = (512, 384)
SAL_SIZE = (512, 384)


def _save_rgb(path: Path, rgb: np.ndarray, *, quality: int = 80) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  Image.fromarray(rgb).save(path, format="JPEG", quality=quality, optimize=True)


def _save_seg(path: Path, rgb: np.ndarray) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  Image.fromarray(rgb).save(path, format="PNG", optimize=True)


def _rel(path: Path, root: Path) -> str:
  return path.relative_to(root).as_posix()


def _round_vec(values: np.ndarray, digits: int = 6) -> list[float]:
  return [round(float(v), digits) for v in values.tolist()]


def _write_html(output_dir: Path, data: dict) -> None:
  html_path = output_dir / "index.html"
  payload = json.dumps(data, separators=(",", ":"))
  html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model Diagnostics Dashboard</title>
  <style>
    :root {{
      --bg: #0c0e14;
      --panel: #151923;
      --panel2: #1b2130;
      --border: #31394b;
      --text: #eceff4;
      --muted: #9ba1b0;
      --ref: #79c0ff;
      --cmp: #ffb142;
      --good: #5acd7f;
      --bad: #ef5350;
      --accent: #ffd166;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background: radial-gradient(circle at top, #172034 0%, var(--bg) 50%);
      color: var(--text);
    }}
    .shell {{
      max-width: 1600px;
      margin: 0 auto;
      padding: 20px;
    }}
    .topbar {{
      display: grid;
      grid-template-columns: 1fr auto auto auto;
      gap: 16px;
      align-items: center;
      margin-bottom: 16px;
      padding: 16px 18px;
      background: rgba(21, 25, 35, 0.92);
      border: 1px solid var(--border);
      border-radius: 18px;
      backdrop-filter: blur(10px);
    }}
    .title h1 {{
      margin: 0 0 6px;
      font-size: 24px;
      letter-spacing: 0.01em;
    }}
    .title p {{
      margin: 0;
      color: var(--muted);
      font-size: 14px;
    }}
    .button-row {{
      display: flex;
      gap: 10px;
      align-items: center;
      justify-content: flex-end;
    }}
    button, select {{
      appearance: none;
      border: 1px solid var(--border);
      background: var(--panel2);
      color: var(--text);
      border-radius: 999px;
      padding: 10px 16px;
      font: inherit;
    }}
    button {{
      cursor: pointer;
    }}
    .controls {{
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 14px;
      align-items: center;
      margin-bottom: 16px;
      padding: 16px 18px;
      background: rgba(21, 25, 35, 0.92);
      border: 1px solid var(--border);
      border-radius: 18px;
    }}
    .slider-wrap {{
      display: grid;
      gap: 6px;
    }}
    input[type=range] {{
      width: 100%;
      accent-color: var(--accent);
    }}
    .meta {{
      display: flex;
      gap: 14px;
      color: var(--muted);
      font-size: 13px;
      justify-content: flex-end;
      flex-wrap: wrap;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(0, 1.6fr) minmax(360px, 0.9fr);
      gap: 16px;
    }}
    .panel {{
      background: rgba(21, 25, 35, 0.92);
      border: 1px solid var(--border);
      border-radius: 18px;
      overflow: hidden;
    }}
    .panel-header {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--border);
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .grid4 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0;
    }}
    figure {{
      margin: 0;
      position: relative;
      border-right: 1px solid var(--border);
      border-bottom: 1px solid var(--border);
      min-height: 240px;
      background: #0b0e14;
    }}
    figure:nth-child(2n) {{
      border-right: 0;
    }}
    figure:nth-last-child(-n+2) {{
      border-bottom: 0;
    }}
    img {{
      display: block;
      width: 100%;
      height: auto;
      aspect-ratio: 4 / 3;
      object-fit: cover;
      background: #090b10;
    }}
    figcaption {{
      position: absolute;
      left: 10px;
      top: 10px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(8, 10, 15, 0.75);
      border: 1px solid rgba(255, 255, 255, 0.12);
      font-size: 12px;
      color: var(--text);
    }}
    .sidebar {{
      display: grid;
      gap: 16px;
      align-content: start;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      padding: 16px;
    }}
    .metric-card {{
      padding: 14px;
      border-radius: 14px;
      background: var(--panel2);
      border: 1px solid var(--border);
    }}
    .metric-card .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 8px;
    }}
    .metric-card .value {{
      font-size: 28px;
      font-weight: 700;
    }}
    .metric-card .detail {{
      margin-top: 6px;
      color: var(--muted);
      font-size: 13px;
    }}
    .pose-grid {{
      padding: 16px;
      display: grid;
      gap: 12px;
    }}
    .pose-row {{
      display: grid;
      grid-template-columns: 42px 1fr 78px;
      gap: 10px;
      align-items: center;
      font-size: 13px;
    }}
    .track {{
      position: relative;
      height: 28px;
      border-radius: 999px;
      background: var(--panel2);
      border: 1px solid var(--border);
      overflow: hidden;
    }}
    .zero {{
      position: absolute;
      left: 50%;
      top: 0;
      bottom: 0;
      width: 1px;
      background: rgba(255, 255, 255, 0.18);
    }}
    .bar {{
      position: absolute;
      top: 4px;
      bottom: 4px;
      border-radius: 999px;
      opacity: 0.88;
    }}
    .bar.ref {{ background: var(--ref); }}
    .bar.cmp {{ background: var(--cmp); opacity: 0.72; }}
    .chart-panel {{
      padding: 16px;
    }}
    .chart-title {{
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 8px;
    }}
    canvas {{
      width: 100%;
      height: 180px;
      display: block;
      background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0));
      border: 1px solid var(--border);
      border-radius: 12px;
    }}
    .saliency-box {{
      padding: 16px;
      display: grid;
      gap: 12px;
    }}
    .legend {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      font-size: 12px;
      color: var(--muted);
    }}
    .legend-item {{
      display: flex;
      gap: 8px;
      align-items: center;
      padding: 8px 10px;
      background: var(--panel2);
      border: 1px solid var(--border);
      border-radius: 12px;
    }}
    .swatch {{
      width: 14px;
      height: 14px;
      border-radius: 999px;
      flex: 0 0 auto;
    }}
    .notice {{
      padding: 12px 14px;
      border-radius: 12px;
      background: rgba(255, 209, 102, 0.08);
      border: 1px solid rgba(255, 209, 102, 0.18);
      color: #ffd98b;
      font-size: 13px;
    }}
    .hidden {{ display: none; }}
    @media (max-width: 1100px) {{
      .topbar {{
        grid-template-columns: 1fr;
      }}
      .layout {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="topbar">
      <div class="title">
        <h1>Model Diagnostics Dashboard</h1>
        <p id="subtitle"></p>
      </div>
      <div class="button-row">
        <button id="play">Play</button>
        <button id="prev">Prev</button>
        <button id="next">Next</button>
      </div>
      <select id="viewMode">
        <option value="compare">Compare</option>
        <option value="saliency">Saliency</option>
      </select>
      <div class="meta">
        <span id="frameMeta"></span>
        <span id="saliencyMeta"></span>
      </div>
    </div>

    <div class="controls">
      <div id="frameLabel">Frame 0</div>
      <div class="slider-wrap">
        <input id="scrubber" type="range" min="0" max="0" step="1" value="0">
        <div id="timeLabel"></div>
      </div>
      <div class="meta">
        <span id="refPath"></span>
        <span id="cmpPath"></span>
      </div>
    </div>

    <div class="layout">
      <div class="panel">
        <div class="panel-header">
          <span id="panelTitle">Reference vs candidate</span>
          <span id="panelNote"></span>
        </div>
        <div id="compareGrid" class="grid4">
          <figure><img id="refRgb" alt=""><figcaption>Reference RGB</figcaption></figure>
          <figure><img id="cmpRgb" alt=""><figcaption>Candidate RGB</figcaption></figure>
          <figure><img id="refSeg" alt=""><figcaption>Reference SegNet argmax</figcaption></figure>
          <figure><img id="cmpSeg" alt=""><figcaption>Candidate SegNet argmax</figcaption></figure>
        </div>
        <div id="saliencyGrid" class="grid4 hidden">
          <figure><img id="salPrevRaw" alt=""><figcaption>Previous frame</figcaption></figure>
          <figure><img id="salPrevOverlay" alt=""><figcaption>Previous saliency</figcaption></figure>
          <figure><img id="salCurrRaw" alt=""><figcaption>Current frame</figcaption></figure>
          <figure><img id="salCurrOverlay" alt=""><figcaption>Current saliency</figcaption></figure>
        </div>
      </div>

      <div class="sidebar">
        <div class="panel">
          <div class="panel-header">
            <span>Current Metrics</span>
            <span id="metricMode"></span>
          </div>
          <div class="metric-grid">
            <div class="metric-card">
              <div class="label">Seg disagreement</div>
              <div id="segValue" class="value">0.0000</div>
              <div class="detail">fraction of pixels with different argmax class</div>
            </div>
            <div class="metric-card">
              <div class="label">Pose MSE</div>
              <div id="poseValue" class="value">0.000000</div>
              <div class="detail">mean squared error on PoseNet dims p0..p5</div>
            </div>
          </div>
        </div>

        <div class="panel">
          <div class="panel-header">
            <span>PoseNet Values</span>
            <span>ref vs candidate</span>
          </div>
          <div id="poseRows" class="pose-grid"></div>
        </div>

        <div class="panel chart-panel">
          <div class="chart-title">Pose MSE history</div>
          <canvas id="poseChart" width="520" height="180"></canvas>
        </div>

        <div class="panel chart-panel">
          <div class="chart-title">Seg disagreement history</div>
          <canvas id="segChart" width="520" height="180"></canvas>
        </div>

        <div class="panel saliency-box">
          <div class="panel-header" style="padding:0;border:0;">
            <span>Saliency Legend</span>
            <span>Winner-take-all colors</span>
          </div>
          <div class="legend" id="legend"></div>
          <div id="saliencyNotice" class="notice"></div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const DATA = {payload};
    const state = {{
      frame: 0,
      playing: false,
      timer: null,
      view: "compare",
    }};

    const el = {{
      subtitle: document.getElementById("subtitle"),
      frameMeta: document.getElementById("frameMeta"),
      saliencyMeta: document.getElementById("saliencyMeta"),
      refPath: document.getElementById("refPath"),
      cmpPath: document.getElementById("cmpPath"),
      frameLabel: document.getElementById("frameLabel"),
      timeLabel: document.getElementById("timeLabel"),
      scrubber: document.getElementById("scrubber"),
      play: document.getElementById("play"),
      prev: document.getElementById("prev"),
      next: document.getElementById("next"),
      viewMode: document.getElementById("viewMode"),
      panelTitle: document.getElementById("panelTitle"),
      panelNote: document.getElementById("panelNote"),
      compareGrid: document.getElementById("compareGrid"),
      saliencyGrid: document.getElementById("saliencyGrid"),
      refRgb: document.getElementById("refRgb"),
      cmpRgb: document.getElementById("cmpRgb"),
      refSeg: document.getElementById("refSeg"),
      cmpSeg: document.getElementById("cmpSeg"),
      salPrevRaw: document.getElementById("salPrevRaw"),
      salPrevOverlay: document.getElementById("salPrevOverlay"),
      salCurrRaw: document.getElementById("salCurrRaw"),
      salCurrOverlay: document.getElementById("salCurrOverlay"),
      segValue: document.getElementById("segValue"),
      poseValue: document.getElementById("poseValue"),
      poseRows: document.getElementById("poseRows"),
      metricMode: document.getElementById("metricMode"),
      poseChart: document.getElementById("poseChart"),
      segChart: document.getElementById("segChart"),
      legend: document.getElementById("legend"),
      saliencyNotice: document.getElementById("saliencyNotice"),
    }};

    el.subtitle.textContent = `${{DATA.reference_path}} vs ${{DATA.candidate_path}}`;
    el.refPath.textContent = `ref: ${{DATA.reference_path}}`;
    el.cmpPath.textContent = `cmp: ${{DATA.candidate_path}}`;
    el.scrubber.max = String(DATA.frames.length - 1);
    el.scrubber.value = "0";
    el.frameMeta.textContent = `${{DATA.frames.length}} frames exported`;
    el.saliencyMeta.textContent = `saliency window centered at pair ${{DATA.saliency.center_pair}}`;

    DATA.pose_colors.forEach((hex, idx) => {{
      const item = document.createElement("div");
      item.className = "legend-item";
      item.innerHTML = `<span class="swatch" style="background:${{hex}}"></span><span>p${{idx}}</span>`;
      el.legend.appendChild(item);
    }});

    function formatNumber(value, digits) {{
      return Number(value).toFixed(digits);
    }}

    function setPlay(on) {{
      state.playing = on;
      el.play.textContent = on ? "Pause" : "Play";
      if (state.timer) {{
        window.clearInterval(state.timer);
        state.timer = null;
      }}
      if (on) {{
        state.timer = window.setInterval(() => {{
          if (state.frame >= DATA.frames.length - 1) {{
            setPlay(false);
            return;
          }}
          setFrame(state.frame + 1);
        }}, Math.max(40, Math.round(1000 / DATA.fps)));
      }}
    }}

    function setView(view) {{
      state.view = view;
      el.viewMode.value = view;
      const compare = view === "compare";
      el.compareGrid.classList.toggle("hidden", !compare);
      el.saliencyGrid.classList.toggle("hidden", compare);
      el.panelTitle.textContent = compare ? "Reference vs candidate" : "Reference PoseNet saliency";
      el.panelNote.textContent = compare ? "RGB + SegNet argmax" : "winner-take-all p0..p5 overlay";
      el.metricMode.textContent = compare ? "compare" : "compare + saliency";
      render();
    }}

    function setFrame(index) {{
      const clamped = Math.max(0, Math.min(DATA.frames.length - 1, index));
      state.frame = clamped;
      el.scrubber.value = String(clamped);
      render();
    }}

    function pathFor(relativePath) {{
      return relativePath ? relativePath : "";
    }}

    function renderPoseRows(frame) {{
      el.poseRows.innerHTML = "";
      const scale = Math.max(DATA.max_abs_pose, 1e-6);
      frame.ref_pose.forEach((refValue, idx) => {{
        const cmpValue = frame.cmp_pose[idx];
        const refPct = Math.abs(refValue) / scale * 50;
        const cmpPct = Math.abs(cmpValue) / scale * 50;
        const refLeft = refValue >= 0 ? 50 : 50 - refPct;
        const cmpLeft = cmpValue >= 0 ? 50 : 50 - cmpPct;
        const row = document.createElement("div");
        row.className = "pose-row";
        row.innerHTML = `
          <div>p${{idx}}</div>
          <div class="track">
            <div class="zero"></div>
            <div class="bar ref" style="left:${{refLeft}}%;width:${{refPct}}%;"></div>
            <div class="bar cmp" style="left:${{cmpLeft}}%;width:${{cmpPct}}%;"></div>
          </div>
          <div>${{formatNumber(refValue, 4)}} / ${{formatNumber(cmpValue, 4)}}</div>
        `;
        el.poseRows.appendChild(row);
      }});
    }}

    function drawHistory(canvas, values, color, currentIndex) {{
      const ctx = canvas.getContext("2d");
      const width = canvas.width;
      const height = canvas.height;
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#101520";
      ctx.fillRect(0, 0, width, height);

      ctx.strokeStyle = "rgba(255,255,255,0.08)";
      ctx.lineWidth = 1;
      for (let i = 1; i < 4; i += 1) {{
        const y = Math.round((height / 4) * i) + 0.5;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }}

      const maxValue = Math.max(...values, 1e-6);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      values.forEach((value, idx) => {{
        const x = values.length === 1 ? 0 : (idx / (values.length - 1)) * (width - 1);
        const y = height - 10 - (value / maxValue) * (height - 20);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }});
      ctx.stroke();

      const x = values.length === 1 ? 0 : (currentIndex / (values.length - 1)) * (width - 1);
      ctx.strokeStyle = "rgba(255,209,102,0.95)";
      ctx.beginPath();
      ctx.moveTo(x + 0.5, 0);
      ctx.lineTo(x + 0.5, height);
      ctx.stroke();
    }}

    function render() {{
      const frame = DATA.frames[state.frame];
      el.frameLabel.textContent = `Frame ${{String(frame.frame_idx).padStart(4, "0")}}`;
      el.timeLabel.textContent = `time ${{formatNumber(frame.frame_idx / DATA.fps, 2)}}s`;
      el.segValue.textContent = formatNumber(frame.seg_disagreement, 4);
      el.poseValue.textContent = formatNumber(frame.pose_mse, 6);
      el.refRgb.src = pathFor(frame.ref_rgb);
      el.cmpRgb.src = pathFor(frame.cmp_rgb);
      el.refSeg.src = pathFor(frame.ref_seg);
      el.cmpSeg.src = pathFor(frame.cmp_seg);

      const saliency = frame.saliency;
      const hasSaliency = saliency && saliency.prev_raw;
      if (hasSaliency) {{
        el.salPrevRaw.src = pathFor(saliency.prev_raw);
        el.salPrevOverlay.src = pathFor(saliency.prev_overlay);
        el.salCurrRaw.src = pathFor(saliency.curr_raw);
        el.salCurrOverlay.src = pathFor(saliency.curr_overlay);
        el.saliencyNotice.textContent = `pair ${{saliency.pair_idx}} available: current pose = [${{saliency.pose.map(v => formatNumber(v, 4)).join(", ")}}]`;
      }} else {{
        el.salPrevRaw.removeAttribute("src");
        el.salPrevOverlay.removeAttribute("src");
        el.salCurrRaw.removeAttribute("src");
        el.salCurrOverlay.removeAttribute("src");
        el.saliencyNotice.textContent = "Saliency overlays were only precomputed for the selected window. Scrub near the center pair or regenerate with a larger --saliency-window-pairs value.";
      }}

      renderPoseRows(frame);
      drawHistory(el.poseChart, DATA.pose_history, "#79c0ff", state.frame);
      drawHistory(el.segChart, DATA.seg_history, "#ffb142", state.frame);
    }}

    el.scrubber.addEventListener("input", (event) => setFrame(Number(event.target.value)));
    el.play.addEventListener("click", () => setPlay(!state.playing));
    el.prev.addEventListener("click", () => setFrame(state.frame - 1));
    el.next.addEventListener("click", () => setFrame(state.frame + 1));
    el.viewMode.addEventListener("change", (event) => setView(event.target.value));
    window.addEventListener("keydown", (event) => {{
      if (event.key === "ArrowLeft") setFrame(state.frame - 1);
      if (event.key === "ArrowRight") setFrame(state.frame + 1);
      if (event.key === " ") {{
        event.preventDefault();
        setPlay(!state.playing);
      }}
      if (event.key.toLowerCase() === "v") {{
        setView(state.view === "compare" ? "saliency" : "compare");
      }}
    }});

    setView("compare");
    render();
  </script>
</body>
</html>
"""
  html_path.write_text(html)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Build an interactive local HTML dashboard for model diagnostics.")
  parser.add_argument("--reference", type=Path, default=HERE / "videos" / "0.mkv")
  parser.add_argument("--candidate", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, default=HERE / "artifacts" / "model_dashboard")
  parser.add_argument("--device", type=str, default="cpu", help="cpu|cuda|mps")
  parser.add_argument("--fps", type=int, default=20)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--max-frames", type=int, default=None)
  parser.add_argument("--rgb-quality", type=int, default=80)
  parser.add_argument("--saliency-window-pairs", type=int, default=48)
  parser.add_argument("--pair-index", type=int, default=None)
  parser.add_argument(
    "--select",
    choices=["max_norm", "max_abs_p0", "max_abs_p1", "max_abs_p2", "max_abs_p3", "max_abs_p4", "max_abs_p5"],
    default="max_norm",
  )
  parser.add_argument("--saliency-alpha", type=float, default=0.72)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  device = pick_device(args.device)
  print(f"Building dashboard on {device}: ref={args.reference} cmp={args.candidate}", flush=True)

  n_ref = count_frames_for_path(args.reference)
  n_cmp = count_frames_for_path(args.candidate)
  n_frames = min(n_ref, n_cmp)
  if args.max_frames is not None:
    n_frames = min(n_frames, args.max_frames)
  if n_frames < 2:
    raise ValueError(f"Need at least 2 aligned frames, found {n_frames}")

  output_dir = args.output_dir
  assets_dir = output_dir / "assets"
  assets_dir.mkdir(parents=True, exist_ok=True)

  posenet, segnet = load_models(device)
  ref_iter = iter_rgb_frames(args.reference, max_frames=n_frames)
  cmp_iter = iter_rgb_frames(args.candidate, max_frames=n_frames)
  prev_ref = None
  prev_cmp = None
  frame_idx = 0
  max_abs_pose = 1e-6
  pose_history: list[float] = []
  seg_history: list[float] = []
  frames: list[dict] = []

  print(f"Exporting {n_frames} frames of comparison assets", flush=True)
  while True:
    ref_batch = _take_batch(ref_iter, args.batch_size)
    cmp_batch = _take_batch(cmp_iter, args.batch_size)
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

      ref_rgb = _resize_rgb(ref_frame.numpy(), RGB_SIZE, Image.Resampling.BILINEAR)
      cmp_rgb = _resize_rgb(cmp_frame.numpy(), RGB_SIZE, Image.Resampling.BILINEAR)
      ref_seg_rgb = _resize_rgb(_seg_to_rgb(ref_seg_cls), SEG_SIZE, Image.Resampling.NEAREST)
      cmp_seg_rgb = _resize_rgb(_seg_to_rgb(cmp_seg_cls), SEG_SIZE, Image.Resampling.NEAREST)

      ref_rgb_path = assets_dir / f"ref_rgb_{frame_idx:04d}.jpg"
      cmp_rgb_path = assets_dir / f"cmp_rgb_{frame_idx:04d}.jpg"
      ref_seg_path = assets_dir / f"ref_seg_{frame_idx:04d}.png"
      cmp_seg_path = assets_dir / f"cmp_seg_{frame_idx:04d}.png"
      _save_rgb(ref_rgb_path, ref_rgb, quality=args.rgb_quality)
      _save_rgb(cmp_rgb_path, cmp_rgb, quality=args.rgb_quality)
      _save_seg(ref_seg_path, ref_seg_rgb)
      _save_seg(cmp_seg_path, cmp_seg_rgb)

      frames.append({
        "frame_idx": frame_idx,
        "ref_rgb": _rel(ref_rgb_path, output_dir),
        "cmp_rgb": _rel(cmp_rgb_path, output_dir),
        "ref_seg": _rel(ref_seg_path, output_dir),
        "cmp_seg": _rel(cmp_seg_path, output_dir),
        "ref_pose": _round_vec(ref_pose_vec),
        "cmp_pose": _round_vec(cmp_pose_vec),
        "seg_disagreement": round(seg_disagreement, 6),
        "pose_mse": round(pose_mse, 8),
        "saliency": None,
      })
      frame_idx += 1
      if frame_idx % 64 == 0 or frame_idx == n_frames:
        print(f"  exported {frame_idx}/{n_frames} frames", flush=True)

  print("Selecting saliency window", flush=True)
  center_pair, _, _, coarse_pose = find_pair(
    input_path=args.reference,
    posenet=posenet,
    device=device,
    selector=args.select,
    pair_index=args.pair_index,
    max_frames=n_frames,
  )
  half = args.saliency_window_pairs // 2
  start_pair = max(0, center_pair - half)
  stop_pair = min(n_frames - 2, start_pair + args.saliency_window_pairs - 1)
  start_pair = max(0, stop_pair - args.saliency_window_pairs + 1)
  print(
    f"Precomputing saliency for pairs {start_pair}..{stop_pair} around center {center_pair} with coarse pose {np.round(coarse_pose, 4).tolist()}",
    flush=True,
  )

  saliency_frames = list(iter_rgb_frames(args.reference, max_frames=stop_pair + 2))
  for pair_idx in range(start_pair, stop_pair + 1):
    prev_frame = saliency_frames[pair_idx]
    curr_frame = saliency_frames[pair_idx + 1]
    pose_vec, prev_saliency, curr_saliency = compute_saliency(posenet, prev_frame, curr_frame, device)
    prev_raw = _resize_rgb(prev_frame.numpy(), SAL_SIZE, Image.Resampling.BILINEAR)
    curr_raw = _resize_rgb(curr_frame.numpy(), SAL_SIZE, Image.Resampling.BILINEAR)
    prev_overlay = _resize_rgb(winner_overlay(prev_frame.numpy(), prev_saliency, alpha=args.saliency_alpha), SAL_SIZE, Image.Resampling.BILINEAR)
    curr_overlay = _resize_rgb(winner_overlay(curr_frame.numpy(), curr_saliency, alpha=args.saliency_alpha), SAL_SIZE, Image.Resampling.BILINEAR)

    prev_raw_path = assets_dir / f"sal_prev_raw_{pair_idx:04d}.jpg"
    curr_raw_path = assets_dir / f"sal_curr_raw_{pair_idx:04d}.jpg"
    prev_overlay_path = assets_dir / f"sal_prev_overlay_{pair_idx:04d}.jpg"
    curr_overlay_path = assets_dir / f"sal_curr_overlay_{pair_idx:04d}.jpg"
    _save_rgb(prev_raw_path, prev_raw, quality=args.rgb_quality)
    _save_rgb(curr_raw_path, curr_raw, quality=args.rgb_quality)
    _save_rgb(prev_overlay_path, prev_overlay, quality=args.rgb_quality)
    _save_rgb(curr_overlay_path, curr_overlay, quality=args.rgb_quality)

    frames[pair_idx + 1]["saliency"] = {
      "pair_idx": pair_idx,
      "prev_raw": _rel(prev_raw_path, output_dir),
      "curr_raw": _rel(curr_raw_path, output_dir),
      "prev_overlay": _rel(prev_overlay_path, output_dir),
      "curr_overlay": _rel(curr_overlay_path, output_dir),
      "pose": _round_vec(pose_vec),
    }
    local_count = pair_idx - start_pair + 1
    total = stop_pair - start_pair + 1
    if local_count % 8 == 0 or pair_idx == stop_pair:
      print(f"  saliency {local_count}/{total} pairs", flush=True)

  data = {
    "reference_path": str(args.reference),
    "candidate_path": str(args.candidate),
    "fps": args.fps,
    "frames": frames,
    "pose_history": [round(v, 8) for v in pose_history],
    "seg_history": [round(v, 6) for v in seg_history],
    "max_abs_pose": round(max_abs_pose, 6),
    "saliency": {
      "center_pair": center_pair,
      "start_pair": start_pair,
      "stop_pair": stop_pair,
    },
    "pose_colors": [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in POSE_SALIENCY_COLORS.tolist()],
  }
  _write_html(output_dir, data)
  print(f"Wrote {output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
  main()
