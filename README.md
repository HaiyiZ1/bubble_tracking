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

| Model | Download |
|---|---|
| `sam2_hiera_tiny.pt` | https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt |
| `sam2_hiera_small.pt` | https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt |
| `sam2_hiera_base_plus.pt` | https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt |
| `sam2_hiera_large.pt` | https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt |

The v1 publishing re-runs for Case 2 and Case 3 require `base_plus`. Download all four if reproducing from scratch.

## Datasets

Original, compressed and key frame debug image data:

| Asset | Link |
|---|---|
| `case1_air1_publish` dataset folder (`frames/`, `frames_0p5/`) | `https://unisydneyedu-my.sharepoint.com/:u:/g/personal/xdin9349_uni_sydney_edu_au/IQCYCknlCOsHT7T-PGBzWd3QAZJx17SLKlpcbR6fcJf4z_E?e=lcTNXe` |
| `case2_air1_publish` dataset folder (`frames/`, `frames_0p5/`) | `https://unisydneyedu-my.sharepoint.com/:u:/g/personal/xdin9349_uni_sydney_edu_au/IQBEdhWOq7z3QZIpR4Mjq73hAefo5AoivczdwKRCjTc7n6s?e=9Ukf11` |
| `case3_air1_publish` dataset folder (`frames/`, `frames_0p5/`) | `https://unisydneyedu-my.sharepoint.com/:u:/g/personal/xdin9349_uni_sydney_edu_au/IQDTxBBWiBMxQ72MiSqd4SxrAUkna7jhhIV4WjJlUE8uPHk?e=E6nGmD` |
| Case 1 keyframe debug artifacts | `https://unisydneyedu-my.sharepoint.com/:u:/g/personal/xdin9349_uni_sydney_edu_au/IQB2TCsQTs3jRYVfMmmAWUVpAZv0UD5IK0MzON9ku-XF6A8?e=nJkozF` |
| Case 2 keyframe debug artifacts | `https://unisydneyedu-my.sharepoint.com/:u:/g/personal/xdin9349_uni_sydney_edu_au/IQAqbiZPpCA2RLb785Q5SkGRAdw38Ek0kQix4Bu2Au_QCCg?e=EnNick` |
| Case 3 keyframe debug artifacts | `https://unisydneyedu-my.sharepoint.com/:u:/g/personal/xdin9349_uni_sydney_edu_au/IQAjNRhfyIisSoQX9E8iyzRhAbP5gJnCsUIRlge_Q9wHLnw?e=k9eKBN` |
| ROI masks | `https://unisydneyedu-my.sharepoint.com/:u:/g/personal/xdin9349_uni_sydney_edu_au/IQAjNRhfyIisSoQX9E8iyzRhAbP5gJnCsUIRlge_Q9wHLnw?e=k9eKBN` |

## meta.yaml schema

`meta.yaml` (repo root) is the source of truth for per-case calibration. Key fields:

- Top-level `fps: 4` — acquisition frame rate used by Phase 3 for time/velocity conversion.
- Per-case `scale: {mm: <float>, pixel: <float>}` — spatial calibration; Phase 3 derives `mm_per_pixel` from this automatically.
- Per-air `start_frame` — frame index offset per air condition.

There are no `--mm-per-pixel` or `--fps` CLI flags in the new pipeline.

## Publishing Reproduction Workflow (Case 1/2/3 Air1)

The pipeline is split into three self-contained phases. Run them in order for each case.

### Determinism

Phase 2 (SAM2 tracking) produces byte-identical `sam2_masks/`, `tracks.csv`, and `events.json` across runs only when these env vars are set:

```bash
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

`CUBLAS_WORKSPACE_CONFIG=:4096:8` pins the cuBLAS workspace to a fixed size, preventing the non-deterministic algorithm selection that SAM2's attention kernels would otherwise trigger. Set these once per shell session before running Phase 2, or prefix them inline on each Phase 2 invocation as shown below.

### Phase 1 — Preprocess

Downscales frames, applies ROI, writes a preprocessed frame set into `<run-dir>/`.

```bash
# Case 1 (High Porosity Air1)
python -m phase1_preprocess --case case1.air1 \
  --raw-frames data/bubbles/case1_air1_publish/frames \
  --run-dir outputs/case1_air1_publish \
  --roi-from data/interim/bubbles_air1_072_271/roi_0p5.json \
  --roi-mask-from data/interim/bubbles_air1_072_271/roi_mask_0p5.png

# Case 2 (Low Porosity Air1)
python -m phase1_preprocess --case case2.air1 \
  --raw-frames data/bubbles/case2_air1_publish/frames \
  --run-dir outputs/case2_air1_publish \
  --roi-from data/interim/case2_air1/roi_0p5.json \
  --roi-mask-from data/interim/case2_air1/roi_mask_0p5.png

# Case 3 (Medium Porosity Air1)
python -m phase1_preprocess --case case3.air1 \
  --raw-frames data/bubbles/case3_air1_publish/frames \
  --run-dir outputs/case3_air1_publish \
  --roi-from data/interim/case3_air1/roi_0p5.json \
  --roi-mask-from data/interim/case3_air1/roi_mask_0p5.png
```

To select ROI interactively (omit `--roi-from` / `--roi-mask-from`); the phase will open an interactive picker and save the results into the run directory.

### Phase 2 — Track

Runs SAM2 + CV keyframe prompting. Writes `sam2_masks/`, `tracks.csv`, `events.json` into `<run-dir>/`.

Set determinism env vars before running (see Determinism section above), or prefix inline:

```bash
# Case 1 (High Porosity Air1)
PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m phase2_track \
  --run-dir outputs/case1_air1_publish \
  --sam2-model tiny \
  --cv-keyframe-start-idx 10 \
  --cv-keyframe-normalize-lightmap \
  --no-sam2-offload-video-to-cpu

# Case 2 (Low Porosity Air1)
PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m phase2_track \
  --run-dir outputs/case2_air1_publish \
  --sam2-model base_plus \
  --cv-keyframe-start-idx 2 \
  --no-sam2-offload-video-to-cpu

# Case 3 (Medium Porosity Air1)
PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m phase2_track \
  --run-dir outputs/case3_air1_publish \
  --sam2-model base_plus \
  --cv-keyframe-start-idx 10 \
  --no-sam2-offload-video-to-cpu
```

### Phase 3 — Postprocess

Computes velocity/area metrics, generates figures, and exports overlay videos. Reads from `<run-dir>/` and writes outputs to `<run-dir>/phase3/`.

```bash
python -m phase3_postprocess --run-dir outputs/case1_air1_publish
python -m phase3_postprocess --run-dir outputs/case2_air1_publish
python -m phase3_postprocess --run-dir outputs/case3_air1_publish
```

**Output path migration note:** Consumers reading `bubble_velocity_area_metrics.csv` or other summary files directly from `<run-dir>/` (the legacy layout) must migrate to `<run-dir>/phase3/`.

## Flag Meanings

| Phase | Flag | Meaning |
|---|---|---|
| 1 | `--case` | meta.yaml lookup key, e.g. `case1.air1` |
| 1 | `--raw-frames` | Input frames directory (relabeled or raw) |
| 1 | `--run-dir` | Output run directory root |
| 1 | `--meta` | meta.yaml path (default `./meta.yaml`) |
| 1 | `--downscale` | Resize factor; default 0.5 |
| 1 | `--roi-from`, `--roi-mask-from` | Skip interactive ROI selection by importing existing files |
| 2 | `--run-dir` | Run directory from Phase 1 |
| 2 | `--sam2-model` | One of `tiny`, `small`, `base_plus`, `large` |
| 2 | `--cv-keyframe-start-idx` | First keyframe index (1-based, relabeled) |
| 2 | `--device` | `cuda` / `cpu`; auto-detect default |
| 2 | `--no-sam2-offload-video-to-cpu` | Disable CPU offload (default: on for OOM safety) |
| 2 | `--cv-keyframe-extra-indices` | Additional manual keyframes alongside the regular interval |
| 2 | `--cv-keyframe-threshold-offset` | CV threshold offset; default 7 |
| 2 | `--cv-keyframe-min-area` | Min connected-component area in CV keyframe prompt; default 50 |
| 2 | `--cv-keyframe-normalize-lightmap` | Enable light-map normalization (default off; pass to match historical case1/2/3 outputs) |
| 3 | `--run-dir` | Run directory from Phase 2 |
| 3 | `--v-ref-1`, `--v-ref-2` | Reference velocities for normalization (defaults 0.23, 7.0) |
