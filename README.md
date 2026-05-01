# Bubble Tracking Project

This README is focused on **reproducible publishing runs**.

For AI-assisted local pipeline setup, follow the guidance in `AGENTS.md`.

## Environment Setup

```bash
cd /path/to/bubble_tracking

python3.11 -m venv venv311
source venv311/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt

# Install SAM2 package from local submodule
pip install -e segment-anything-2

# Optional but recommended for GPU runs
pip install --upgrade --force-reinstall --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch torchvision

python scripts/check_env.py
```

Required checkpoints in `checkpoints/`:
- `sam2_hiera_tiny.pt`
- `sam2_hiera_base_plus.pt`

## Publishing Reproduction Workflow (Case1/2/3 Air1)

### 1) Relabel datasets so frame `000001` is the first usable bubble frame

```bash
python scripts/relabel_publishing_air1.py --case case1_air1 case2_air1 case3_air1
```

This creates:
- `data/bubbles/case1_air1_publish`
- `data/bubbles/case2_air1_publish`
- `data/bubbles/case3_air1_publish`

Each relabeled root contains:
- `frames/`
- `frames_0p5/`
- `frame_map.csv`
- `publishing_relabel_config.json`

### 2) Run tracking with CV keyframe prompts and debug outputs

#### Case 1 (High Porosity-Air1)

```bash
python scripts/run_sam2_prompted_flow.py \
  --frames data/bubbles/case1_air1_publish/frames_0p5 \
  --reference data/bubbles/case1_air1_publish/frames_0p5/000001.jpg \
  --output outputs/case1_air1_publish \
  --roi data/interim/bubbles_air1_072_271/roi_0p5.json \
  --roi-mask data/interim/bubbles_air1_072_271/roi_mask_0p5.png \
  --prompt-mode sam2 \
  --sam2-checkpoint checkpoints/sam2_hiera_tiny.pt \
  --sam2-model-cfg sam2_hiera_t.yaml \
  --sam2-device cuda \
  --cv-keyframe-prompts \
  --cv-keyframe-interval 160 \
  --cv-keyframe-start-idx 10 \
  --cv-keyframe-preprocess-mode default \
  --cv-keyframe-diff-sigma 0.5 \
  --cv-keyframe-no-normalize \
  --cv-keyframe-min-area 50 \
  --cv-keyframe-threshold-offset 7 \
  --cv-keyframe-debug-dir outputs/case1_air1_publish_debug \
  --use-flow \
  --velocity-mode centroid_delta \
  --preprocess-mode raw \
  --preprocess-raw-sigma 1.0 \
  --debounce-birth-frames 2 \
  --debounce-death-frames 3 \
  --fps 4 \
  --mm-per-pixel 0.08563
```

#### Case 2 (Low Porosity-Air1)

```bash
python scripts/run_sam2_prompted_flow.py \
  --frames data/bubbles/case2_air1_publish/frames_0p5 \
  --reference data/bubbles/case2_air1_publish/frames_0p5/000001.jpg \
  --output outputs/case2_air1_publish \
  --roi data/interim/case2_air1/roi_0p5.json \
  --roi-mask data/interim/case2_air1/roi_mask_0p5.png \
  --prompt-mode sam2 \
  --sam2-checkpoint checkpoints/sam2_hiera_base_plus.pt \
  --sam2-model-cfg sam2_hiera_b+.yaml \
  --sam2-device cuda \
  --cv-keyframe-prompts \
  --cv-keyframe-interval 160 \
  --cv-keyframe-start-idx 2 \
  --cv-keyframe-preprocess-mode default \
  --cv-keyframe-diff-sigma 0.5 \
  --cv-keyframe-no-normalize \
  --cv-keyframe-min-area 50 \
  --cv-keyframe-threshold-offset 7 \
  --cv-keyframe-debug-dir outputs/case2_air1_publish_debug \
  --use-flow \
  --velocity-mode centroid_delta \
  --preprocess-mode raw \
  --preprocess-raw-sigma 1.0 \
  --debounce-birth-frames 2 \
  --debounce-death-frames 3 \
  --fps 4 \
  --mm-per-pixel 0.08695
```

#### Case 3 (Medium Porosity-Air1)

```bash
python scripts/run_sam2_prompted_flow.py \
  --frames data/bubbles/case3_air1_publish/frames_0p5 \
  --reference data/bubbles/case3_air1_publish/frames_0p5/000001.jpg \
  --output outputs/case3_air1_publish \
  --roi data/interim/case3_air1/roi_0p5.json \
  --roi-mask data/interim/case3_air1/roi_mask_0p5.png \
  --prompt-mode sam2 \
  --sam2-checkpoint checkpoints/sam2_hiera_base_plus.pt \
  --sam2-model-cfg sam2_hiera_b+.yaml \
  --sam2-device cuda \
  --cv-keyframe-prompts \
  --cv-keyframe-interval 160 \
  --cv-keyframe-start-idx 10 \
  --cv-keyframe-preprocess-mode default \
  --cv-keyframe-diff-sigma 0.5 \
  --cv-keyframe-no-normalize \
  --cv-keyframe-min-area 50 \
  --cv-keyframe-threshold-offset 7 \
  --cv-keyframe-debug-dir outputs/case3_air1_publish_debug \
  --use-flow \
  --velocity-mode centroid_delta \
  --preprocess-mode raw \
  --preprocess-raw-sigma 1.0 \
  --debounce-birth-frames 2 \
  --debounce-death-frames 3 \
  --fps 4 \
  --mm-per-pixel 0.08264
```

### 3) Export publishing overlay videos

```bash
python scripts/export_publishing_tracking_video.py \
  --case case1_air1 \
  --run-dir outputs/case1_air1_publish \
  --frames data/bubbles/case1_air1_publish/frames_0p5

python scripts/export_publishing_tracking_video.py \
  --case case2_air1 \
  --run-dir outputs/case2_air1_publish \
  --frames data/bubbles/case2_air1_publish/frames_0p5

python scripts/export_publishing_tracking_video.py \
  --case case3_air1 \
  --run-dir outputs/case3_air1_publish \
  --frames data/bubbles/case3_air1_publish/frames_0p5
```

Each run writes:
- `tracking_overlay_publish_1x.mp4`
- `tracking_overlay_publish_10x.mp4`

## Flag Meanings (`scripts/run_sam2_prompted_flow.py`)

| Flag | Meaning |
|---|---|
| `--frames` | Ordered frame directory used as input sequence. |
| `--reference` | Zero-bubble reference frame for preprocessing/flow comparison. |
| `--output` | Output run directory for masks, tracks, events, summaries, and videos. |
| `--roi` | Rectangular ROI JSON (`x1,y1,x2,y2`). |
| `--roi-mask` | Polygon ROI mask PNG used for in-ROI filtering. |
| `--prompt-mode sam2` | Use SAM2 prompt/propagation path. |
| `--sam2-checkpoint` | SAM2 checkpoint file path. |
| `--sam2-model-cfg` | SAM2 config name/path (for example `sam2_hiera_b+.yaml`). |
| `--sam2-device` | Inference device (`cuda` or `cpu`). |
| `--cv-keyframe-prompts` | Enable CV-derived keyframe masks as SAM2 prompts. |
| `--cv-keyframe-interval` | Keyframe spacing (in frames). |
| `--cv-keyframe-start-idx` | First keyframe index for prompting in relabeled data. |
| `--cv-keyframe-preprocess-mode` | Preprocess mode for keyframe prompt generation (`raw/default/automatrix`). |
| `--cv-keyframe-diff-sigma` | Gaussian sigma applied to keyframe diff image. |
| `--cv-keyframe-no-normalize` | Disable normalization in keyframe CV preprocessing. |
| `--cv-keyframe-min-area` | Minimum component area kept as prompt mask. |
| `--cv-keyframe-threshold-offset` | Threshold offset used by CV keyframe segmentation. |
| `--cv-keyframe-debug-dir` | Directory for keyframe debug artifacts (preprocessed image, mask, overlay). |
| `--use-flow` | Enable optical-flow-assisted association/tracking. |
| `--velocity-mode centroid_delta` | Compute velocity from centroid displacement between linked frames. |
| `--preprocess-mode raw` | Tracking-stage preprocessing mode. |
| `--preprocess-raw-sigma` | Gaussian smoothing sigma used by `raw` preprocessing mode. |
| `--debounce-birth-frames` | Require N consecutive detections before confirming object birth. |
| `--debounce-death-frames` | Require N consecutive misses before confirming object death. |
| `--fps` | Acquisition frame rate used for time/velocity conversion. |
| `--mm-per-pixel` | Spatial calibration scale (mm per pixel). |

## Additional Visualization Commands

These are optional post-processing commands commonly used for reviewer-facing figures/tables.

### Velocity overlay videos from completed runs

```bash
python scripts/export_velocity_overlay.py \
  --run-dir outputs/case1_air1_publish \
  --frames data/bubbles/case1_air1_publish/frames_0p5 \
  --output outputs/case1_air1_publish/velocity_overlay.mp4 \
  --fps 4 \
  --label-units mm_s

python scripts/export_velocity_overlay.py \
  --run-dir outputs/case2_air1_publish \
  --frames data/bubbles/case2_air1_publish/frames_0p5 \
  --output outputs/case2_air1_publish/velocity_overlay.mp4 \
  --fps 4 \
  --label-units mm_s

python scripts/export_velocity_overlay.py \
  --run-dir outputs/case3_air1_publish \
  --frames data/bubbles/case3_air1_publish/frames_0p5 \
  --output outputs/case3_air1_publish/velocity_overlay.mp4 \
  --fps 4 \
  --label-units mm_s
```

### Vanished-area moving-window plots (2s and 5s)

```bash
python scripts/plot_vanished_area_moving_window.py --run-dir outputs/case1_air1_publish
python scripts/plot_vanished_area_moving_window.py --run-dir outputs/case2_air1_publish
python scripts/plot_vanished_area_moving_window.py --run-dir outputs/case3_air1_publish
```

### Velocity normalization and area scatter figures

1. Generate `bubble_velocity_area_metrics.csv` plus base histogram/scatter plots:

```bash
python scripts/plot_velocity_norm_summary.py \
  --run-dir outputs/case1_air1_publish \
  --output-dir outputs/case1_air1_publish \
  --v-ref-1 0.23 \
  --v-ref-2 7.0

python scripts/plot_velocity_norm_summary.py \
  --run-dir outputs/case2_air1_publish \
  --output-dir outputs/case2_air1_publish \
  --v-ref-1 0.23 \
  --v-ref-2 7.0

python scripts/plot_velocity_norm_summary.py \
  --run-dir outputs/case3_air1_publish \
  --output-dir outputs/case3_air1_publish \
  --v-ref-1 0.23 \
  --v-ref-2 7.0
```

2. Generate log-log scatter with marginal PDFs from the metrics CSV:

```bash
python scripts/plot_velocity_area_logxy_with_marginal_pdfs.py \
  --run-dir outputs/case1_air1_publish \
  --x-column max_v_over_v_ref_1 \
  --v-ref-index 1 \
  --v-ref-value 0.23 \
  --output outputs/case1_air1_publish/velocity_area_scatter_logxy_vref1_with_pdfs.png

python scripts/plot_velocity_area_logxy_with_marginal_pdfs.py \
  --run-dir outputs/case2_air1_publish \
  --x-column max_v_over_v_ref_1 \
  --v-ref-index 1 \
  --v-ref-value 0.23 \
  --output outputs/case2_air1_publish/velocity_area_scatter_logxy_vref1_with_pdfs.png

python scripts/plot_velocity_area_logxy_with_marginal_pdfs.py \
  --run-dir outputs/case3_air1_publish \
  --x-column max_v_over_v_ref_1 \
  --v-ref-index 1 \
  --v-ref-value 0.23 \
  --output outputs/case3_air1_publish/velocity_area_scatter_logxy_vref1_with_pdfs.png

# Optional second reference velocity view (v_ref_2 = 7.0 mm/s)
python scripts/plot_velocity_area_logxy_with_marginal_pdfs.py \
  --run-dir outputs/case1_air1_publish \
  --x-column max_v_over_v_ref_2 \
  --v-ref-index 2 \
  --v-ref-value 7.0 \
  --output outputs/case1_air1_publish/velocity_area_scatter_logxy_vref2_with_pdfs.png

python scripts/plot_velocity_area_logxy_with_marginal_pdfs.py \
  --run-dir outputs/case2_air1_publish \
  --x-column max_v_over_v_ref_2 \
  --v-ref-index 2 \
  --v-ref-value 7.0 \
  --output outputs/case2_air1_publish/velocity_area_scatter_logxy_vref2_with_pdfs.png

python scripts/plot_velocity_area_logxy_with_marginal_pdfs.py \
  --run-dir outputs/case3_air1_publish \
  --x-column max_v_over_v_ref_2 \
  --v-ref-index 2 \
  --v-ref-value 7.0 \
  --output outputs/case3_air1_publish/velocity_area_scatter_logxy_vref2_with_pdfs.png
```

### Case-level max-velocity summary CSV exports

```bash
python scripts/export_bubble_max_velocity_summary.py --output-dir outputs
```
