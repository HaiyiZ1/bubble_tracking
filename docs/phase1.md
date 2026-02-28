# Phase 1: SAM2 Reproduction & Biomedical Tracking Validation

## Overview

Phase 1 establishes a working SAM2 pipeline and validates it against a biomedical tracking paper.

## Quick Start

### 1. Environment Setup

```bash
cd /home/aaron/Downloads/bubble_tracking

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify environment
python scripts/check_env.py
```

### 2. Install SAM2

```bash
# Clone SAM2 repository
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
cd ..

# Download checkpoint (smallest for CPU testing)
mkdir -p checkpoints
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt
cd ..
```

### 3. Run Sample Pipeline

```bash
# Get sample video frames
python scripts/get_sample_videos.py

# Extract frames (if needed)
python scripts/extract_frames.py --input data/raw/sample.mp4 --output data/interim/sample_frames

# Run SAM2 tracking
python scripts/run_sam2_video.py \
    --frames data/interim/sample_frames \
    --prompts data/prompts/sample_prompts.yaml \
    --output outputs/sample_run
```

## Phase 1.1: SAM2 Sample Run

### Acceptance Criteria

- [ ] Pipeline runs end-to-end on sample video
- [ ] Produces per-frame masks in `outputs/<run_id>/masks/`
- [ ] Generates `tracks.json` with object IDs, centroids, and areas
- [ ] Creates `overlay.mp4` visualization
- [ ] Deterministic: same inputs produce same outputs

### Output Structure

```
outputs/<run_id>/
├── run_config.yaml      # Full configuration for reproducibility
├── masks/
│   ├── 000000.png       # Per-frame mask (instance IDs encoded)
│   ├── 000001.png
│   └── ...
├── tracks.json          # Tracking results
└── overlay.mp4          # Visualization video
```

### tracks.json Schema

```json
{
  "metadata": {
    "num_frames": 100,
    "num_objects": 3,
    "created_at": "2024-01-01T00:00:00Z"
  },
  "objects": {
    "1": {
      "frames": {
        "0": {"centroid": [100, 200], "area": 1500, "bbox": [80, 180, 120, 220]},
        "1": {"centroid": [102, 198], "area": 1480, "bbox": [82, 178, 122, 218]}
      }
    }
  }
}
```

## Phase 1.2: Biomedical Reproduction

See [phase1_paper_selection.md](phase1_paper_selection.md) for paper selection rationale.

### Steps

1. Download dataset: `python scripts/get_biomed_dataset.py`
2. Run reproduction: `python scripts/run_biomed_repro.py`
3. View report: `docs/phase1_biomed_repro_report.md`

### Acceptance Criteria

- [ ] Dataset loader works for chosen dataset
- [ ] Metrics match paper (within reasonable tolerance)
- [ ] Qualitative results show good temporal consistency

### Findings / Known Limitations (important for unlabeled bubble data)

- **SAM2 is prompt-based (not fully unsupervised)**: it needs a point/box/mask prompt to know what to track. For unlabeled bubble videos, you’ll need either:
  - manual initialization (clicks/boxes) on the first frame, or
  - an automatic prompt generator (e.g., blob detection) to initialize objects.
- **Mitosis / object birth is not automatic**: in CTC HeLa, newly appearing/reproduced cells are not assigned new IDs automatically; SAM2 propagates only the objects you initialize.
- **Tracking can collapse (all masks become empty)**: on CTC HeLa (`01`), we observed a failure mode where predictions become all-background starting at **frame 34** (no detections from that point onward). This is a propagation failure, not just an evaluation artifact.
  - **Mitigation (recommended for bubbles)**: add a track-management layer that detects failures (e.g., total mask area → 0, sudden drops) and triggers **automatic re-initialization** using an auto-segmentation/auto-prompt step on the current frame.
- **Evaluation note (CTC)**: CTC `SEG/` ground truth can be sparse and may not align with `TRA/` tracking masks. For tracking validation, prefer evaluating against `TRA/` masks when available.

## CPU-Only Notes

Default settings are optimized for CPU:
- Short clips (30-100 frames)
- Downsampled resolution (480p)
- Smallest SAM2 checkpoint (sam2_hiera_tiny)

For full runs on GPU, modify `--resolution` and `--checkpoint` flags.

