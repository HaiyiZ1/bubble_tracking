# Findings & Decisions

## Tracking Plan (2026-02-22)
- Reference design: `docs/plans/2026-02-22-bubble-tracking-design.md`
- Assumptions: zero-bubble reference image; stationary camera; FPS constant within a dataset (default 4 FPS for current), but varies across datasets; crowding increases over time with frequent merge/split events later
- ID handling: create new ID on merge/split; terminate ID on disappearance; persist per-track birth/death reasons and lineage links for later visualization
- Pipeline notes from `next_steps_26-02-21.md`:
  - Preprocess frames (CLAHE for contrast, Gaussian blur for denoise)
  - Dense optical flow per frame pair -> velocity field
  - Instance segmentation per frame (SAM2) with per-frame IDs
  - Flow-guided overlap matching for tracking; overlap-only likely fails at high velocity
  - Per-bubble velocity: median flow inside mask
  - Area from pixel counts; scale 100 mm = 2335 pixels
- Current gap: `run_flow_tracking.py` segments via reference-diff thresholding; SAM2 masks are not wired into the flow tracker yet
- New direction: use SAM2 masks from `outputs/bubbles_air1_072_271_auto_search/masks` for per-frame segmentation while keeping flow computed from preprocessed frames
- ROI JSON (`data/interim/bubbles_air1_072_271/roi.json`) is a bounding box; ROI mask exists separately for polygon ROI

## Code Notes (2026-02-22)
- `picture_automatrix.py` is a single-frame preprocessing + segmentation script with GUI ROI selection and hard-coded Windows paths; no flow/tracking yet
- `samples/picture_automatrix.py` matches `picture_automatrix.py` (duplicate example)
- Codebase has a `src/bubble_tracking` package for core logic; likely target for reusable pipeline code
- `src/bubble_tracking/track_builder.py` only aggregates per-object stats from fixed mask IDs; no cross-frame matching logic
- `src/bubble_tracking/reprompt_matcher.py` implements IoU + centroid matching with split/birth/death logic, but no explicit merge handling
- `src/bubble_tracking/mask_events.py` already provides connected-component overlap event detection (split/merge/birth/death) on binary masks
- `src/bubble_tracking/frame_io.py` provides frame/mask loading, sorted filename parsing, and overlay video creation utilities
- `src/bubble_tracking/visualization.py` can draw mask overlays and annotate IDs; useful for quick sanity-check outputs
- `src/bubble_tracking/visualization.py` already includes matplotlib plots for trajectories and area-over-time
- `scripts/select_roi.py` provides an interactive ROI selector that saves ROI to JSON; could replace GUI calls in `picture_automatrix.py`
- `src/bubble_tracking/sam2_adapter.py` constructs `FrameData` with centroid/area/bbox via named args, so adding optional velocity to `FrameData` can be backward compatible
- `requirements.txt` does not include scipy or scikit-image; prefer OpenCV/Numpy equivalents unless we update deps
- Existing CLI scripts (e.g., `scripts/run_reprompted_tracking.py`) use argparse + `sys.path` insert for `src/` imports; can follow same pattern for new runner

## Implementation Notes (2026-02-22)
- Added `src/bubble_tracking/flow_tracking.py` with diff-based preprocessing, segmentation, flow-guided matching, and outputs (tracks/events/area timeseries)
- Added `scripts/run_flow_tracking.py` CLI with ROI support, FPS/scale params, and overlay video/frames options
- Extended `FrameData` to include optional velocity (pixels/sec)
- Flow estimation currently uses CPU `cv2.calcOpticalFlowFarneback` only; no CUDA/GPU path yet
- Added `--masks-dir` support to `run_flow_tracking.py` to use SAM2 masks for segmentation while keeping flow from preprocessed frames
- SAM2 masks are re-labeled per frame via connected components (foreground = mask > 0); requires mask/frame indices to match
- Added `--sam2-auto` in `run_flow_tracking.py` to generate SAM2 auto-masks for every frame (saved to `output/sam2_masks`) and feed them into the flow tracker
- Added `--sam2-propagate` to auto-search a prompt and run SAM2 video propagation; backfills empty masks for pre-prompt frames
- Prompt search reports OOM downgrades (points-per-side / mask-scale) and saves them in `sam2_auto_config.json`
- Added minimal runner `scripts/run_sam2_prompted_flow.py` to seed SAM2 propagation via classic segmentation or SAM2 auto-search, then run flow tracking on the masks
- SAM2 `SAM2AutomaticMaskGenerator` supports `points_per_batch`, `crop_n_layers`, and `crop_n_points_downscale_factor` for memory control; `output_mode='binary_mask'` is noted as memory-heavy for large resolutions

## GPU Acceleration Decision (2026-02-22)
- Target GPU path: CUDA Farneback optical flow with auto CPU fallback (warn once)
- Expected speedup: flow step ~5–20x on RTX 4090; end-to-end ~2–8x depending on ROI/IO
- Requires OpenCV built with CUDA; otherwise auto fallback to CPU

## GPU Flow Implementation (2026-02-22)
- Added CUDA Farneback flow backend with auto fallback to CPU in `src/bubble_tracking/flow_tracking.py`
- Added CLI flags: `--flow-backend {auto,cpu}` and `--cuda-device`
- `events.json` now includes `runtime` metadata for actual flow backend used

## Lineage Metadata (2026-03-24)
- `TrackedObject` now carries lifecycle metadata: `birth_reason`, `birth_frame_idx`, `birth_parent_ids`, `death_reason`, `death_frame_idx`, `death_child_ids`
- Lifecycle metadata is derived from the final event list after debounce/relink processing so track annotations match the saved post-processed IDs
- Birth reason mapping:
  - `injection`: track appears without a split/merge parent event
  - `split`: track is listed in a split event's `child_ids`
  - `merge`: track is listed as a merge event's `child_id`
- Death reason mapping:
  - `death`: track disappears without producing split/merge children
  - `split`: track is the parent in a split event
  - `merge`: track is one of the parents in a merge event
- Tracks still present in the final frame keep no death metadata because the sequence ends before an observed disappearance
- Synthetic validation with a split -> merge -> death mask sequence confirmed the expected lineage metadata:
  - parent track: `injection` birth, `split` death
  - split children: `split` birth, `merge` death
  - merged child: `merge` birth, `death` death

## Publishing Prep Findings (2026-03-24)
- Existing Air1 case roots on disk:
  - case1 current source root is the generic `data/bubbles` tree (`frames/`, `frames_0p5/`, `map.csv`)
  - case2 source root is `data/bubbles/case2_air1`
  - case3 source root is `data/bubbles/case3_air1`
- Requested inclusive first frames for relabeling:
  - case1 Air1: original frame 71
  - case2 Air1: original frame 127
  - case3 Air1: original frame 215
- Current run configs confirm the old pipeline used the unrelabelled datasets:
  - case1 frames: `data/bubbles/frames_0p5`, prompt start idx 80
  - case2 frames: `data/bubbles/case2_air1/frames_0p5`, prompt start idx 127
  - case3 frames: `data/bubbles/case3_air1/frames_0p5`, prompt start idx 224
- After relabeling, the equivalent prompt-start indices shift by the discarded-frame offsets:
  - case1: old idx 80 -> new idx 10
  - case2: old idx 127 -> new idx 1
  - case3: old idx 224 -> new idx 10
- Publishing overlays should stay separate from the tracking pipeline. Best fit is a standalone exporter that reads an existing run dir (`tracks.json` + `masks/`) plus the relabeled frames dir and writes:
  - a 1x video at acquisition speed (`fps=4`)
  - a 10x video at `fps=40`
  - both capped to the first 5 minutes (`1200` frames at 4 FPS)
- Existing `create_tracking_video()` always draws `Frame {idx}`. To keep pipeline behavior unchanged, publishing text overlays should live in a new export script rather than altering default pipeline video output.

## Progress Logging (2026-02-22)
- Added `--log-every` CLI flag to print one progress line per N frames
- Progress line includes stage (`segment` or `flow+match`) and processed/total frames

## Diagnostics (2026-02-22)
- Added `scripts/inspect_frame21.py` to compare baseline, reference, normalized diff, current mask, and CLAHE mask
- Added `preprocess_frame_with_debug` helper in `src/bubble_tracking/flow_tracking.py` to expose normalization internals

## Noise Reduction Controls (2026-02-22)
- Added CLI flags for CLAHE (`--use-clahe`, `--clahe-clip-limit`, `--clahe-tile-grid`)
- Added compact-blob filter flags (`--compact-max-area`, `--compact-min-dark-delta`) to drop small components that are not dark enough

## Temporal Persistence Gate (2026-02-22)
- Added temporal persistence gate with 1-frame backfill to suppress flicker noise
- Confirmation rule: keep component if it overlaps previous confirmed frame or next frame (flow-warped overlap)
- Added CLI flags: `--persistence-min-overlap-px`, `--persistence-max-area`

## Automatrix Reference Notes (2026-02-22)
- `picture_automatrix.py` uses SciPy `gaussian_filter`, `griddata` (cubic), and `zoom` for light-matrix interpolation
- Segmentation uses `opening(binary_mask, disk(4))`, then `remove_small_objects(label(binary_mask), min_size=500)` (opening is effectively ignored)
- Median filter uses `skimage.filters.median` with `disk(12)` on the cleaned binary mask
- `requirements.txt` does not list `scipy` or `scikit-image`; strict mode will require user install

## Automatrix Mode Implementation (2026-02-22)
- Added `preprocess.mode` (`default` or `automatrix`) to select strict automatrix pipeline
- Implemented automatrix preprocessing and segmentation with SciPy/Skimage in flow tracking
- Added CLI flag `--preprocess-mode` to `scripts/run_flow_tracking.py`
- Added guards for zero dynamic range in automatrix normalization to avoid NaNs

## Edge Strength Gate (2026-02-22)
- Added boundary gradient gate using Sobel magnitude on normalized grayscale
- Keeps components whose boundary median strength exceeds `edge_min_strength`
- Optional `edge_min_fraction` enforces a minimum fraction of boundary pixels above threshold
- CLI flags: `--edge-min-strength`, `--edge-boundary-radius`, `--edge-min-fraction`

## Edge Strength Diagnostics (2026-02-22)
- Added `scripts/export_edge_strength_diagnostics.py` to export per-frame overlays and edge metrics (CSV/JSON)
- Reads `events.json` to reuse preprocess/segment config and supports frame ranges

## Environment Notes (2026-02-22)
- Default `python` in this shell does not have `cv2`; use the project venv (e.g., `venv311/bin/python`) to inspect OpenCV build
- `venv311` has `opencv-python` 4.11.0; `cv2.cuda` exists but `getCudaEnabledDeviceCount()` returns 0 and `cv2.cuda_FarnebackOpticalFlow` is unavailable (no CUDA build)
- On some distros, `libdc1394-22-dev` is replaced by `libdc1394-dev`; it is optional for OpenCV build
- `ninja-build` install triggered NVIDIA DKMS autoinstall for new kernel and failed; apt is left with unconfigured kernel packages until DKMS succeeds
- DKMS log shows missing NVIDIA source headers (`os-interface.h`, `nv-firmware.h`), suggesting `nvidia-dkms-550`/`nvidia-kernel-source-550` needs reinstall before `dpkg --configure -a` can succeed
- DKMS autoinstall for kernel 6.17.0-14 fails because `/var/crash/nvidia-kernel-source-550.0.crash` already exists; delete the crash file before re-running `dpkg --configure -a`

## Goal Pivot (2026-02-13)
- Project objective changed from persistent bubble identity tracking to:
  - total bubble area over time from SAM2 masks
  - split/merge event detection from mask topology
- Implication: stable object IDs are no longer required for primary metrics.

## Requirements
- Run SAM2 on bubble dataset with auto-prompts
- Use frames 72–271 (first ~70 frames are empty; first bubble appears around frame 72)
- Dataset location: `data/bubbles/Air1/` (JPG frames) (folder exists; `data/bubles` does not)
- No ground-truth labels; optional minimal manual prompts acceptable
- GPU is available (RTX 4090, CUDA 12.x)

## Dataset Prep (Air1 Full)
- Raw frames live at `/home/aaron/Downloads/Air1` and are named like `1 (1526).jpg`
- Total frames: 1747, indices 1..1747
- Loader expects trailing digits before extension, so raw names must be normalized
- Use `scripts/generate_auto_prompts.py --rebuild-subset` to copy/normalize into a new frames directory (writes `map.csv`)

## Research Findings
 - Repo scripts include `scripts/run_sam2_video.py` (generic SAM2 video runner) and `scripts/run_biomed_repro.py` (dataset-specific DAVIS/CTC).
 - No bubble-specific script found via grep; likely need to adapt `run_sam2_video.py`.
 - Prompt YAML format (from `data/raw/prompts/synthetic_prompts.yaml`): list of prompts with `frame_idx`, `object_id`, and `points` (each point has `label`, `x`, `y`).
 - PromptSpec supports point prompts and optional bounding boxes; mask prompts are in-memory only (not serialized to YAML).
- Frame loader uses regex to extract trailing digits before file extension and sorts numerically; `1 (1000).jpg` will be treated as frame index 1000.
- Bubble frames filenames are like `1 (550).jpg`; current regex expects digits immediately before `.jpg`, so these files are NOT matched. We must rename or copy to a clean numeric pattern for loading.
 - Parsed filenames with pattern `\\((\\d+)\\)`: 1747 frames indexed 1..1747; frame 72 exists.
 - Frame 72 stats (grayscale): shape 3672x5496, mean ~138, median 210.
- Otsu threshold on frame 72 yields many tiny components; inverted Otsu with min_area filtering suggests ~10 components at min_area=200, ~6 at 500, ~4 at 1000+.
- venv has CPU-only PyTorch: torch 2.9.1+cpu, cuda_version=None, torch.cuda.is_available()=False.
- CUDA PyTorch install from cu121 index failed (no matching distribution), likely because venv uses Python 3.14 (unsupported for CUDA wheels).
- System Python 3.11 and 3.12 are available; current venv uses Python 3.14.
- requirements.txt pins CPU-only torch via `--extra-index-url https://download.pytorch.org/whl/cpu` and `torch==2.9.1+cpu`.
- New venv311 has torch 2.5.1+cu121 and numpy 1.26.4, but torch.cuda.is_available() is False; local `nvidia-smi` fails with NVML error.
- Fixed GPU tensor -> numpy conversion in `sam2_adapter.py` to avoid `TypeError: can't convert cuda:0 device type tensor to numpy`.
- New subset created: frames 72–271 copied to `data/interim/bubbles_air1_072_271/frames` with map.csv.
- ROI auto-prompts (middle 10%, bottom 10%) on frame 72 keep 4 components with min_area=200.
- `sam2_adapter.py` currently computes per-object properties only when ROI masking is enabled, and only for the last object in loop scope. This under-populates `tracks.json` without ROI and can silently skew stats.
- Existing `reprompt_matcher.py` reports births/deaths/splits but does not model merges; merge detection needs to be added in mask-level post-processing.
- Existing runners already emit per-frame mask PNGs, which is enough to compute:
  - framewise total area (`count_nonzero(mask > 0)`)
  - connected-component transitions for split/merge detection

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Use frames 72–172 | First non-empty frames start around 72 |
| Use GPU | CPU is too slow for 1700 frames; GPU available |
| Start with auto-prompts | No labels; manual prompts unstable due to quick morphology changes |
| Normalize subset to numeric filenames | Loader requires digits before extension |
| Auto-prompts: Otsu-inverted + min_area=200 | Produces ~10 components on frame 72 |
| Extend subset to 72–271 | Use first 200 frames after bubble start |
| ROI gating: middle 10%, bottom 10% | Constrain prompts to birth region; reduce electrodes/background |
| Re-prompting with automask (N=5) | Use keyframe re-prompting to capture births and reduce drift |
| Event detection from connected components, not SAM2 object IDs | IDs can be unstable after re-prompts and are not the target metric |
| Use overlap linkage (IoU/containment thresholds) between adjacent masks | Captures split/merge transitions with minimal assumptions |
| Add configurable event thresholds at CLI level | Enables tuning sensitivity without code edits (`--event-*` args) |

## Implementation Findings (2026-02-13)
- Added new module: `src/bubble_tracking/mask_events.py`
  - Computes per-frame area time series from binary foreground (`mask > 0`)
  - Detects `split`, `merge`, `birth`, `death` events via connected-component overlap linkage
  - Supports both in-memory mask dicts and on-disk mask directories
  - Writes CSV-friendly records for plotting/post-analysis
- Integrated mask analytics into runners:
  - `scripts/run_sam2_video.py`
  - `scripts/run_reprompted_tracking.py`
  - `scripts/run_sam2_prompt_then_track.py`
- New run artifacts per output directory:
  - `area_timeseries.json`
  - `area_timeseries.csv`
  - `events.json`
  - `stats.json` now includes `bubble_area`, `mask_event_summary`, and `mask_event_config`
- Fixed SAM2 adapter bug in `src/bubble_tracking/sam2_adapter.py`:
  - per-object properties are now computed every frame regardless of ROI usage
  - properties are computed for each object in `object_ids_out` (not just the last loop variable)

## Validation Results
- `py_compile` passed for updated scripts and modules.
- Synthetic three-frame mask scenario produced expected topology events:
  - split (frame 0 -> 1): one component to two
  - merge (frame 1 -> 2): two components to one
  - event summary: `{'num_splits': 1, 'num_merges': 1, 'num_births': 0, 'num_deaths': 0}`
- Real-mask smoke test on `outputs/bubbles_air1_072_172_auto_gpu/masks` completed successfully:
  - frames analyzed: 101
  - event summary: `{'num_splits': 4, 'num_merges': 3, 'num_births': 0, 'num_deaths': 6}`
  - mean total area: `3961784.02 px`

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| session-catchup.py not found at expected path | Created planning files from templates and proceeded |
| `data/bubles/Air1` not found | Actual folder is `data/bubbles/Air1` |
| `git status` unavailable at project root | Continue without repo-level diff checks in this directory |
| `ModuleNotFoundError: bubble_tracking` in ad-hoc validation script | Re-ran with `PYTHONPATH=src` |

## Resources
- `data/bubbles/Air1/`
- `data/raw/prompts/synthetic_prompts.yaml`
- `scripts/run_sam2_video.py`

## Visual/Browser Findings
- N/A

---
*Update this file after every 2 view/browser/search operations*
*This prevents visual information from being lost*

## Recovery Findings (2026-03-09)
- `scripts/run_sam2_prompted_flow.py` had reverted to an older 539-line version (prompt-once only).
- `src/bubble_tracking/flow_tracking.py` had lost:
  - `raw` preprocess mode,
  - `use_flow` toggle,
  - `velocity_mode` (`centroid_delta`) support.
- `scripts/run_flow_tracking.py` had lost matching CLI flags for the above runtime options.

### Restored (Core)
- `src/bubble_tracking/flow_tracking.py`
  - Added `PreprocessConfig.raw_sigma`.
  - Added `TrackingConfig.use_flow` and `TrackingConfig.velocity_mode`.
  - Added raw-diff preprocess path (no light-map normalization).
  - Added centroid-delta velocity updater for one-to-one associations.
  - Added flow-disabled path (identity overlap matching) and runtime metadata fields.
- `scripts/run_flow_tracking.py`
  - Restored CLI: `--preprocess-mode raw`, `--preprocess-raw-sigma`, `--use-flow/--no-use-flow`, `--velocity-mode`.
- `scripts/run_sam2_prompted_flow.py`
  - Restored CLI + logic for `--cv-keyframe-prompts`.
  - Restored CV keyframe segmentation options:
    `--cv-keyframe-interval`, `--cv-keyframe-start-idx`,
    `--cv-keyframe-preprocess-mode`, `--cv-keyframe-diff-sigma`,
    `--cv-keyframe-use-clahe`, `--cv-keyframe-disable-light-map`,
    `--cv-keyframe-no-normalize`, `--cv-keyframe-min-area`,
    `--cv-keyframe-threshold-offset`.
  - Restored keyframe debug outputs: `--cv-keyframe-debug-dir` with overlay/mask/preprocessed images.
  - Restored tracking-stage options passthrough: `--use-flow`, `--velocity-mode`, `--preprocess-mode`, `--preprocess-raw-sigma`.

### Deferred
- Advanced `event_reprompt` and `sam2_global_union_reprompt` branches were not reintroduced in this recovery pass because the active case2/case3 pipeline uses CV-keyframe prompts and centroid-delta velocity.

## Velocity-Normalized Plotting (2026-03-09)
- Added `scripts/plot_velocity_norm_summary.py` for Phase 3 velocity visualizations.
- Outputs per run:
  - `velocity_norm_histograms.png` (4 subplots):
    - `v_x / v_ref_1`, `v_y / v_ref_1`, `v_x / v_ref_2`, `v_y / v_ref_2`
  - `velocity_area_scatter.png` (2 subplots):
    - `max(V / v_ref_1)` vs `(area_per_bubble / roi_area)`
    - `max(V / v_ref_2)` vs `(area_per_bubble / roi_area)`
- Implementation details:
  - `max(V / v_ref_i)` is computed per bubble using max `velocity_mag_mm_s` across its lifecycle.
  - `area_per_bubble` currently uses `max(area_px)` per bubble lifecycle (configurable with `--area-stat`).
  - `roi_area` is resolved from `sam2_prompted_flow_config.json` (`roi_mask` preferred, then `roi_json` bbox).
- Updated `scripts/plot_velocity_norm_summary.py` histogram binning to support automatic bin selection (`--bins auto`, now default) and regenerated outputs for all three runs.
- Extended `plot_velocity_norm_summary.py` scatter outputs:
  - `velocity_area_scatter.png` (linear axes)
  - `velocity_area_scatter_logx.png` (log x)
  - `velocity_area_scatter_logxy.png` (log x + log y)
- For log plots, non-positive points are filtered and dropped count is annotated per subplot.

## Velocity PDF Fitting (2026-03-09)
- Extended `scripts/plot_velocity_norm_summary.py` to estimate empirical density + fitted PDFs for normalized velocity components.
- Added output artifacts per run:
  - `velocity_norm_pdf_fit.png` (hist density + KDE + best-fit PDF overlay)
  - `velocity_distribution_fit.csv` (top-3 fits by AIC for each variable)
  - `velocity_distribution_best.json` (rank-1 distribution for each variable)
- Candidate distributions: `norm`, `laplace`, `logistic`, `cauchy`, `t`.
- Fitting metric: AIC (ranking), with BIC and KS test exported for diagnostics.
- Across current case runs (`case1_air1`, `case2_air1`, `case3_air1`, `case3_air4`), the best family is consistently Student's t for all four normalized components.

## Scatter Trendline Update (2026-03-09)
- Updated `scripts/plot_velocity_norm_summary.py` scatter plots to include a best-fit line on each subplot.
- Fit behavior by axis mode:
  - linear axes: linear fit in `(x, y)`
  - log-y: linear fit in `(x, log10(y))`
  - log-x: linear fit in `(log10(x), y)`
  - log-x/log-y: linear fit in `(log10(x), log10(y))` (power-law in original axes)
- Legend now shows `best fit (R²=...)` for each subplot.
- Added a new scatter export: `velocity_area_scatter_logy.png` (x linear, y log).

## Residual Saturation Comparison (2026-03-12)
- Compared SAM2 pipeline metric vs Haiyi traditional CV metric for:
  - `case1_air1`, `case2_air1`, `case3_air1`, `case3_air4`.
- SAM2-side metric used:
  - `mean(total_area_px of last 4 frames) / roi_area_px`.
- Haiyi-side metric used:
  - `traditional_CV_residual_saturation_from_haiyi.csv` value (last-frame total area / ROI area).
- Results (SAM2 mean-last4 vs Haiyi):
  - case1_air1: `0.039353876` vs `0.188410775` (−79.11%)
  - case2_air1: `0.177874755` vs `0.816576803` (−78.22%)
  - case3_air1: `0.134526110` vs `0.498604706` (−73.02%)
  - case3_air4: `0.181956360` vs `0.764480873` (−76.20%)

## Residual Saturation Comparison (Haiyi v2, 2026-03-12)
- Updated source file: `traditional_CV_residual_saturation_from_haiyi_v2.csv`.
- Compared SAM2 last-frame ratio vs Haiyi last-frame ratio:
  - SAM2 metric: `last_frame_total_area_px / roi_area_px`.
  - Haiyi metric: residual saturation from v2 CSV.
- Results:
  - case1_air1: `0.039335842` vs `0.065190128` (−39.66%)
  - case2_air1: `0.177783975` vs `0.205777354` (−13.60%)
  - case3_air1: `0.134695188` vs `0.149082807` (−9.65%)
  - case3_air4: `0.181749569` vs `0.228579781` (−20.49%)

## Flicker Debounce Postprocess (2026-03-15)
- Added ID-level temporal debounce to tracking outputs (postprocess on saved instance masks):
  - Birth debounce: suppress short-lived IDs using `birth_confirm_frames`.
  - Death debounce: bridge short disappearances by copying previous-frame ID mask up to `death_confirm_frames - 1` gaps.
- Implementation location:
  - `src/bubble_tracking/flow_tracking.py`
  - `_smooth_presence_1d()` for 1D on/off gating.
  - `_debounce_saved_masks()` for mask rewriting + debounced area series + debounced birth/death events.
- Exposed CLI flags in both runners:
  - `--debounce-birth-frames`
  - `--debounce-death-frames`
- Notes:
  - Default values are `1` (disabled), preserving old behavior unless explicitly enabled.
  - Debounce rewrites saved masks, recomputes `area_timeseries`, replaces birth/death event entries with debounced ones, and rebuilds `tracks.json/csv` frame data from the debounced masks for consistency.
  - Added ID re-linking across short gaps: if a new ID appears shortly after an old ID disappears and centroid/area are compatible, the new ID is remapped to the old ID before debounce.
  - Re-link controls:
    - `relink_max_centroid_dist_px` (default `20`)
    - `relink_min_area_ratio` (default `0.35`)
    - `relink_max_area_ratio` (default `2.8`)

## Case1 LogXY + Marginal PDFs (2026-03-16)
- Added script `scripts/plot_velocity_area_logxy_with_marginal_pdfs.py`.
- Purpose: keep the original scatter plots unchanged and generate a separate figure containing:
  - main log-log scatter for `x = max(V / v_ref_2)` and `y = max(area_per_bubble / roi_area)`
  - top marginal PDF for `x`
  - right marginal PDF for `y`
- Output generated:
  - `outputs/case1_air1_sam2_cv_keyframe160_nonorm_centroid_delta/velocity_area_scatter_logxy_vref2_with_pdfs.png`
- Styling update applied:
  - explicit marginal labels "PDF of ..." for both top/right panels
  - mathtext/LaTeX-style labels/titles for subscripts and symbols
  - bold main figure title
  - added coordinate labels for three points on each PDF (left end, peak, right end)
  - forced `max(A_bubble/A_ROI)` value formatting to decimal (`0.xx` style), not percent

## Bubble Summary CSV Export (2026-03-18)
- Added `scripts/export_bubble_max_velocity_summary.py` to export per-bubble summaries for `case1_air1`, `case2_air1`, and `case3_air1`.
- Output files:
  - `outputs/case1_air1_A_bubble_max_velocity.csv`
  - `outputs/case2_air1_A_bubble_max_velocity.csv`
  - `outputs/case3_air1_A_bubble_max_velocity.csv`
- Column interpretation:
  - `A_bubble_avg` = mean `area_px` over all appearances of the bubble.
  - `A_bubble_max` = max `area_px` over all appearances.
  - `max_velocity_x` / `max_velocity_y` use the maximum absolute component speed across frames, converted from `tracks.csv` pixel-per-second columns to pixel/frame with `fps`.
  - `max_velocity` uses the real per-frame magnitude maximum from `velocity_mag_px_s`, then converts to pixel/frame.
  - `time_of_appearance in s` = `frame_amount_of_appearance / fps`.
- Note: the user prompt said "14 columns" but listed 15 headings, so the exporter writes all 15 listed fields.
