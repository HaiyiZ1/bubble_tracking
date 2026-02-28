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
