# Bubble Tracking Project

## Overview

Use SAM2 (Segment Anything Model 2) masks to analyze bubble dynamics in liquid fuel cells.

**Input**: Continuous photos taken at certain frequency on a liquid fuel cell in a fixed direction and fixed position

**Output**: Per-frame bubble masks, total bubble area over time, and split/merge event logs derived from mask topology

## Quick Start

```bash
# Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Verify installation
python scripts/check_env.py
```

## Project Structure

```
bubble_tracking/
├── src/bubble_tracking/    # Library code
├── scripts/                # CLI entrypoints
├── notebooks/              # Jupyter demos
├── docs/                   # Documentation
├── data/                   # Datasets (not versioned)
├── outputs/                # Run artifacts (not versioned)
└── checkpoints/            # Model weights (not versioned)
```

## Phases

### Phase 1: SAM2 Validation
1. Reproduce sample dataset of SAM2
2. Reproduce a biomedical research using SAM2 for tracking

**Documentation**: [docs/phase1.md](docs/phase1.md)

### Phase 2+: Bubble-Specific Pipeline
- Pipeline documentation: [docs/phase2_pipeline.md](docs/phase2_pipeline.md)
- Original requirement notes: [docs/next_steps_26-02-21.md](docs/next_steps_26-02-21.md)

### Phase 3: Prompting + Preprocessing Stabilization (Current)

Current status summary:
- Root cause of grainy background in preprocessing was double contrast stretching (noise got over-amplified).
- Fix applied: remove normalization in the CV keyframe segmentation path (`--cv-keyframe-no-normalize`), which yields a smooth background and cleaner masks.
- Strategy evaluation:
1. Event-driven reprompting: best in theory, but practical runs OOM due to too many events (not reliable on RTX 4090 for full runs).
2. Global SAM2 auto-mask reprompting: unstable; SAM2 often ignores small/weak bubbles.
3. Classic CV segmentation inspired by `picture_automatrix` (with normalization removed): best practical result, captures nearly all bubbles.
- Current operating mode: CV keyframe reprompting every 100 frames, balancing speed and accuracy.
- Current observed result: good visual tracking with >95% bubble coverage.
- Known limitation: small/weak bubbles can flicker (appear/disappear across adjacent frames) due to SAM2 confidence instability on low-contrast contours.

### Run phase 2 pipeline

**1) Environment**
```bash
cd /path/to/bubble_tracking

python3.11 -m venv venv311
source venv311/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

If you want GPU Torch (recommended):
```bash
pip install --upgrade --force-reinstall --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch torchvision
```

Install SAM2 package from local repo:
```bash
cd segment-anything-2
pip install -e .
cd ..
```

Download checkpoint:
```bash
mkdir -p checkpoints
wget -O checkpoints/sam2_hiera_tiny.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt
```

Create data and ouptputs directory.
```bash
mkdir -p data
mkdir -p outputs
```

Build OpenCV with CUDA support for faster flow computation (Optional):
```bash
cd third_party
./build.sh
```

Quick check:
```bash
python scripts/check_env.py
```

**2) Normalize raw filenames to numeric frames**
(Needed if raw names look like `1 (1526).jpg`.)
```bash
RAW_DIR=/path/to/Air1
OUT_DIR=data/bubbles/frames

python - <<'PY'
import os, re
from pathlib import Path
import shutil

raw = Path(os.environ["RAW_DIR"])
out = Path(os.environ["OUT_DIR"])
out.mkdir(parents=True, exist_ok=True)

pat = re.compile(r'(?:\((\d+)\)|(\d+))\.(jpg|jpeg|png)$', re.I)
items = []
for n in os.listdir(raw):
    m = pat.search(n)
    if m:
        idx = int(m.group(1) or m.group(2))
        items.append((idx, n))
items.sort()

for idx, name in items:
    src = raw / name
    dst = out / f"{idx:06d}.jpg"
    if not dst.exists():
        shutil.copy2(src, dst)
print(f"normalized frames: {len(items)} -> {out}")
PY
```

**3) Create ROI files (one-time)**
```bash
python scripts/select_roi.py \
  --frames data/bubbles/frames \
  --frame-idx 1 \
  --out data/interim/air1/roi.json
```

```bash
python scripts/select_polygon_roi.py \
  --frames data/bubbles/frames \
  --frame-idx 1 \
  --out-json data/interim/air1/roi_poly.json \
  --out-mask data/interim/air1/roi_mask.png
```

**4) Run pipeline (SAM2 auto-search + flow tracking)**
```bash
python scripts/run_sam2_prompted_flow.py \
  --frames data/bubbles/frames \ 
  --reference data/bubbles/frames/000001.jpg \
  --output outputs/flow_tracking_sam2_auto_full \
  --roi data/interim/air1/roi.json \
  --roi-mask data/interim/air1/roi_mask.png \
  --prompt-mode sam2 \
  --sam2-checkpoint checkpoints/sam2_hiera_tiny.pt \
  --sam2-model-cfg sam2_hiera_t.yaml \
  --sam2-mask-scale 0.5 \
  --sam2-points-per-side 16 \
  --sam2-device cuda \
  --fps 4 \
  --mm-per-pixel 0.0428 \
  --log-every 25
```

**5) Summarize first 900 frames (per-second stats + plots)**
```bash
python scripts/summarize_run_stats.py \
  --run-dir outputs/flow_tracking_sam2_auto_full \
  --max-frames 900
```

**Expected outputs**
- `outputs/flow_tracking_sam2_auto_full/tracks.csv`
- `outputs/flow_tracking_sam2_auto_full/events.json`
- `outputs/flow_tracking_sam2_auto_full/area_timeseries.csv`
- `outputs/flow_tracking_sam2_auto_full/tracking_overlay.mp4`
- `outputs/flow_tracking_sam2_auto_full/per_second_area.csv`
- `outputs/flow_tracking_sam2_auto_full/per_second_events.csv`
- `outputs/flow_tracking_sam2_auto_full/per_second_summary.png`

# Run Phase 3 pipeline
Set up environments like phase 2.

Update data directories in as well as in the commands.

1. Prepare data and select roi
```bash
source venv/bin/activate

python scripts/copy_and_compress_frames.py

# case2 roi
python scripts/select_roi.py \
  --frames data/bubbles/case2_air1/frames_0p5 \
  --frame-idx 1 \
  --out data/interim/case2_air1/roi_0p5.json

python scripts/select_polygon_roi.py \
  --frames data/bubbles/case2_air1/frames_0p5 \
  --frame-idx 1 \
  --out-json data/interim/case2_air1/roi_poly_0p5.json \
  --out-mask data/interim/case2_air1/roi_mask_0p5.png
```

2. Tracking (cv reprompting at 100 frames interval, SAM2 base+ tracking)

```bash
# case 2 tracking and velocity visualization

python scripts/run_sam2_prompted_flow.py \
  --frames data/bubbles/case2_air1/frames_0p5 \
  --reference data/bubbles/case2_air1/frames_0p5/000001.jpg \
  --output outputs/case2_air1_sam2_cv_keyframe100_nonorm_centroid_delta \
  --roi data/interim/case2_air1/roi_0p5.json \
  --roi-mask data/interim/case2_air1/roi_mask_0p5.png \
  --prompt-mode sam2 \
  --sam2-checkpoint checkpoints/sam2_hiera_base_plus.pt \
  --sam2-model-cfg sam2_hiera_b+.yaml \
  --sam2-device cuda \
  --cv-keyframe-prompts \
  --cv-keyframe-interval 100 \
  --cv-keyframe-start-idx 127 \
  --cv-keyframe-preprocess-mode default \
  --cv-keyframe-diff-sigma 0.5 \
  --cv-keyframe-no-normalize \
  --cv-keyframe-min-area 50 \
  --cv-keyframe-threshold-offset 7 \
  --cv-keyframe-debug-dir outputs/case2_air1_sam2_cv_keyframe100_nonorm_centroid_delta/cv_keyframe_debug \
  --use-flow \
  --velocity-mode centroid_delta \
  --preprocess-mode raw \
  --preprocess-raw-sigma 1.0 \
  --fps 4 \
  --mm-per-pixel 0.0856
```
3. Produce velocity overlay

```bash
python scripts/export_velocity_overlay.py \
  --run-dir outputs/case2_air1_sam2_cv_keyframe100_nonorm_centroid_delta \
  --frames data/bubbles/case2_air1/frames_0p5 \
  --output outputs/case2_air1_sam2_cv_keyframe100_nonorm_centroid_delta/velocity_overlay.mp4 \
  --fps 4 \
  --label-units mm_s
```

## Current Scripts and Logic Map

### Main scripts

| Script | Purpose |
|---|---|
| `scripts/run_sam2_prompted_flow.py` | Main Phase 3 runner. Builds SAM2 prompts, propagates SAM2 masks through video, then runs tracking/events/velocity export. |
| `scripts/run_flow_tracking.py` | Generic flow-tracking runner (can run with external masks or SAM2 auto/propagate modes). |
| `scripts/export_velocity_overlay.py` | Renders per-object velocity labels/arrows on original frames and writes overlay video. |
| `scripts/run_reprompted_tracking.py` | Older experimental keyframe re-prompting pipeline (kept for experiments/ablation). |
| `scripts/select_roi.py` / `scripts/select_polygon_roi.py` | Generate rectangular ROI JSON and polygon ROI mask files. |

### Step-by-step logic locations

1. **Preprocessing**
- `src/bubble_tracking/flow_tracking.py`
- `preprocess_frame()` and `preprocess_frame_with_debug()`
- Modes:
  - `raw` (used in current Phase 3 tracking stage)
  - `default`
  - `automatrix` (reference-inspired mode)

2. **Optical Flow**
- `src/bubble_tracking/flow_tracking.py`
- `FlowEngine.compute()` (Farneback flow)
- Flow usage is controlled by tracking config (`use_flow`), exposed in CLI as `--use-flow/--no-use-flow`.

3. **SAM2 Tracking / Mask Propagation**
- `scripts/run_sam2_prompted_flow.py`
  - `_run_sam2_propagation()` runs SAM2 video predictor propagation.
  - `_find_sam2_prompt()` handles prompt-once SAM2 auto-search mode.
- `src/bubble_tracking/sam2_adapter.py`
  - `SAM2Runner.run()` executes SAM2 video propagation and returns per-frame masks.

4. **Reprompting**
- Current production path:
  - `scripts/run_sam2_prompted_flow.py`
  - `_build_cv_keyframe_mask_prompts()` creates CV masks at keyframes and uses them as SAM2 mask prompts.
  - Recommended setting used in current runs: keyframe interval of 100 frames.
- Experimental path:
  - `scripts/run_reprompted_tracking.py` (kept for research/testing).

5. **Post-processing SAM2 Masks + Tracking IDs + Events (split/merge/birth/death)**
- `src/bubble_tracking/flow_tracking.py`
  - `track_frames()` performs connected-component labeling, association, ID assignment, and event logging.
  - Core helpers:
    - `_overlap_counts()`
    - `_build_links()`
    - `_confirm_persistent_components()`
    - `_warp_labels()`
- Events are written to `events.json` and include split/merge/birth/death counts.

6. **Mask-only Topology Event Analysis (without flow ID tracking)**
- `src/bubble_tracking/mask_events.py`
  - `analyze_masks_dir()` computes area timeseries and topology events directly from mask transitions.
