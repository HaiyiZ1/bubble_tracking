# Progress Log

## Session: 2026-02-02

### Phase 1: Requirements & Discovery
- **Status:** complete
- **Started:** 2026-02-02 20:30
- Actions taken:
  - Read planning-with-files skill instructions
  - Initialized planning files (task_plan.md, findings.md, progress.md)
  - Captured user constraints (frames 72–172, auto-prompts, GPU available)
- Files created/modified:
  - task_plan.md (created)
  - findings.md (created)
  - progress.md (created)

### Phase 2: Planning & Structure
- **Status:** complete
- Actions taken:
  - Identified `scripts/run_sam2_video.py` and prompt YAML format
  - Chose auto-prompt approach (Otsu-inverted + min_area=200)
  - Planned numeric subset directory for frames 72–172
- Files created/modified:
  - task_plan.md (updated)
  - findings.md (updated)

### Phase 3: Implementation
- **Status:** in_progress
- **Started:** 2026-02-02 20:44
- Actions taken:
  - Copied frames 72–172 to `data/interim/bubbles_air1_072_172/frames` with numeric names
  - Created `scripts/generate_auto_prompts.py`
  - Generated auto prompts and debug overlay for frame 0
  - Attempted SAM2 run on GPU; failed due to CPU-only torch build
  - Installed SAM2 in venv311 and updated visualization to larger, colored labels and optional overlay frame export
  - Created new 72–271 (200 frames) subset and ROI-gated auto-prompts
  - Tightened ROI width to middle 10% and regenerated prompts
  - Added polygon ROI selection and mask support for auto-prompts
  - Added SAM2 automask prompts and stats output
  - Wrote design doc for re-prompting every 5 frames (N=5)
  - Implemented hybrid matcher and re-prompted tracking script
- Files created/modified:
  - scripts/generate_auto_prompts.py (created)
  - data/interim/bubbles_air1_072_172/frames/ (created)
  - data/interim/bubbles_air1_072_172/map.csv (created)
  - data/interim/bubbles_air1_072_172/prompts_auto.yaml (created)
  - data/interim/bubbles_air1_072_172/debug_prompt_frame0.jpg (created)
  - src/bubble_tracking/visualization.py (updated)
  - scripts/run_sam2_video.py (updated)
  - data/interim/bubbles_air1_072_271/frames/ (created)
  - data/interim/bubbles_air1_072_271/map.csv (created)
  - data/interim/bubbles_air1_072_271/prompts_auto.yaml (created)
  - data/interim/bubbles_air1_072_271/debug_prompt_frame0.jpg (created)
  - scripts/generate_auto_prompts.py (updated)
  - src/bubble_tracking/reprompt_matcher.py (created)
  - scripts/run_reprompted_tracking.py (created)

## Session: 2026-02-13

### Phase 1: Requirements & Discovery
- **Status:** complete
- **Started:** 2026-02-13
- Actions taken:
  - Captured user goal pivot to aggregate bubble-area dynamics and split/merge detection
  - Audited existing SAM2 runner outputs and mask artifacts
  - Located core integration points and identified a SAM2 adapter frame-property bug
- Files created/modified:
  - task_plan.md (updated)
  - findings.md (updated)
  - progress.md (updated)

### Phase 2: Design
- **Status:** complete
- Actions taken:
  - Chosen event model: connected components + overlap linkage between adjacent frames
  - Defined output contracts: `area_timeseries.json`, `area_timeseries.csv`, `events.json`
  - Preserved compatibility with existing `tracks.json` and `stats.json`
- Files created/modified:
  - task_plan.md (updated)
  - findings.md (updated)

### Phase 3: Implementation
- **Status:** in_progress
- Actions taken:
  - Added `src/bubble_tracking/mask_events.py` for area time series and split/merge/birth/death event detection
  - Integrated mask analytics into:
    - `scripts/run_sam2_video.py`
    - `scripts/run_reprompted_tracking.py`
    - `scripts/run_sam2_prompt_then_track.py`
  - Added CLI controls for event thresholds (`--event-min-area`, overlap/IoU options)
  - Fixed per-object frame property bug in `src/bubble_tracking/sam2_adapter.py`
- Files created/modified:
  - src/bubble_tracking/mask_events.py (created)
  - src/bubble_tracking/sam2_adapter.py (updated)
  - scripts/run_sam2_video.py (updated)
  - scripts/run_reprompted_tracking.py (updated)
  - scripts/run_sam2_prompt_then_track.py (updated)
  - README.md (updated objective language)
  - task_plan.md (updated)
  - findings.md (updated)
  - progress.md (updated)

### Phase 4: Testing & Verification
- **Status:** complete
- Actions taken:
  - Ran syntax validation with `python -m py_compile` on all changed files
  - Ran synthetic mask test to verify split and merge detection behavior
  - Confirmed generation contracts for `area_timeseries.json`, `area_timeseries.csv`, `events.json`
- Files created/modified:
  - (runtime checks only; no new files committed from tests)

## Session: 2026-02-22

### Phase 1: Code & Data Review
- **Status:** complete
- **Started:** 2026-02-22
- Actions taken:
  - Attempted planning-with-files session catchup; script missing at `${CLAUDE_PLUGIN_ROOT}/scripts`
  - Updated task plan for flow-guided mask tracking phases
  - Reviewed `next_steps_26-02-21.md` and logged pipeline assumptions
  - Reviewed `picture_automatrix.py` and `samples/picture_automatrix.py`
  - Confirmed core package location at `src/bubble_tracking`
  - Reviewed `src/bubble_tracking/track_builder.py` (mask ID aggregation only)
  - Reviewed `src/bubble_tracking/reprompt_matcher.py` and `src/bubble_tracking/mask_events.py`
  - Reviewed `src/bubble_tracking/frame_io.py` and `src/bubble_tracking/visualization.py`
  - Reviewed `scripts/select_roi.py` for ROI selection utility
  - Reviewed `src/bubble_tracking/sam2_adapter.py` (FrameData usage)
  - Reviewed `requirements.txt` and existing runner script patterns
  - Updated findings with new tracking plan summary
- Files created/modified:
  - task_plan.md (updated)
  - findings.md (updated)
  - progress.md (updated)

### Phase 2: Preprocessing + Segmentation
- **Status:** complete
- **Started:** 2026-02-22
- Actions taken:
  - Implemented diff-based preprocessing and segmentation in `src/bubble_tracking/flow_tracking.py`
  - Added zero-bubble reference handling with optional ROI cropping
- Files created/modified:
  - src/bubble_tracking/flow_tracking.py (created)
  - src/bubble_tracking/schemas.py (updated)

### Phase 3: Flow-Guided Matching + Events
- **Status:** complete
- **Started:** 2026-02-22
- Actions taken:
  - Added Farneback optical flow, label warping, and overlap-based linking
  - Added ID assignment rules for births/splits/merges/deaths and event logging
- Files created/modified:
  - src/bubble_tracking/flow_tracking.py (created)

### Phase 4: Outputs + Validation
- **Status:** in_progress
- **Started:** 2026-02-22
- Actions taken:
  - Added outputs for `tracks.json`, `tracks.csv`, `events.json`, and `area_timeseries.*`
  - Added `scripts/run_flow_tracking.py` runner with overlay video/frames options
  - Ran `py_compile` on new module and runner
  - Fixed histogram computation in segmentation to avoid NumPy bin mismatch error
  - Re-read brainstorming + planning-with-files skills (planning path fallback used)
  - Captured GPU Farneback + auto fallback requirement for flow acceleration
  - Added stage-based progress logging with `--log-every` option
- Files created/modified:
  - scripts/run_flow_tracking.py (created)
  - src/bubble_tracking/flow_tracking.py (updated)
  - progress.md (updated)

### Phase 5: GPU Flow Acceleration
- **Status:** in_progress
- **Started:** 2026-02-22
- Actions taken:
  - Wrote GPU flow design doc
  - Added CUDA Farneback flow backend with auto fallback and runtime metadata
  - Added CLI flags for flow backend and CUDA device selection
  - Ran `py_compile` on updated flow tracking module and runner
  - Re-read planning-with-files skill (fallback path used)
  - Re-read brainstorming skill for GPU setup work
  - Attempted OpenCV CUDA check with default Python; `cv2` not installed in that interpreter
  - Checked OpenCV build in `venv311` (no CUDA Farneback; device count 0)
  - Added progress logging design doc
  - Re-read planning-with-files and brainstorming skills for noise reduction work
  - Attempted session catchup; script missing at `${CLAUDE_PLUGIN_ROOT}/scripts`
  - Re-read `task_plan.md` and `findings.md` before noise reduction work
- Files created/modified:
  - docs/plans/2026-02-22-gpu-flow-design.md (created)
  - docs/plans/2026-02-22-progress-logging-design.md (created)
  - src/bubble_tracking/flow_tracking.py (updated)
  - scripts/run_flow_tracking.py (updated)
  - findings.md (updated)
  - task_plan.md (updated)
  - progress.md (updated)

### Phase 6: Noise Reduction (Lighting Drift)
- **Status:** in_progress
- **Started:** 2026-02-22
- Actions taken:
  - Added frame diagnostics script with CLAHE A/B comparison
  - Added debug preprocessing helper for normalized diff images

### Phase 7: SAM2 Mask Integration (Flow Tracking)
- **Status:** in_progress
- **Started:** 2026-02-22
- Actions taken:
  - Re-ran planning-with-files session catchup using fallback path (`/home/aaron/.codex/skills/.../session-catchup.py`)
  - Confirmed Air1 raw frames are named like `1 (N).jpg` and require normalization
  - Counted Air1 frames: 1747 (indices 1..1747)
  - Updated plan/findings with full-dataset prep approach
  - Full dataset SAM2 run killed during propagation (~59%); likely OOM due to full-res frame loading
  - Checked existing plotting helpers; `src/bubble_tracking/visualization.py` has matplotlib plots for trajectories and area-over-time
  - Added `scripts/summarize_run_stats.py` to compute per-second area and merge/split stats (first 900 frames) and plot summary
  - Attempted planning-with-files catchup via `${CLAUDE_PLUGIN_ROOT}` path (missing); used codex skill path
  - Captured user direction to integrate SAM2 masks into flow tracking (use `outputs/bubbles_air1_072_271_auto_search/masks`)

### Phase 7: SAM2 Mask Integration (Flow Tracking)
- **Status:** in_progress
- **Started:** 2026-02-22
- Actions taken:
  - Added `--masks-dir` option to `scripts/run_flow_tracking.py`
  - Added mask/frame index alignment and SAM2 mask relabeling in `src/bubble_tracking/flow_tracking.py`
  - Re-labels SAM2 masks per frame via connected components before flow-guided matching
  - Verified test frame filenames are `000000.jpg`.. and can align to mask indices
  - Added `--sam2-auto` mode to generate SAM2 auto-masks per frame and save to `output/sam2_masks`
  - Added SAM2 auto-mask CLI parameters and `sam2_auto_config.json` output
  - Wrote design doc `docs/plans/2026-02-22-sam2-automask-flow-integration-design.md`
  - Attempted git check for committing design doc; repo not initialized
  - Added `--sam2-propagate` (prompt search + propagation) with OOM downgrade logging and mask backfill
  - Added `scripts/run_sam2_prompted_flow.py` (clean end-to-end runner with classic or SAM2 prompt)
  - Wrote design doc `docs/plans/2026-02-22-sam2-prompted-flow-design.md`
  - Ran `py_compile` on diagnostics and flow tracking updates
  - Generated frame-21 diagnostic panel in `outputs/flow_tracking_test/diagnostics/frame_21_panel.png`
  - Added CLAHE CLI flags to flow tracking runner
  - Added compact-blob filter CLI flags to flow tracking runner
  - Generated per-frame overlay images for flow tracking runs (200 frames each)
  - Designed and implemented temporal persistence gate with 1-frame backfill
  - Added CLI flags for temporal persistence gating
  - Ran `py_compile` after persistence-gate changes
  - Wrote temporal persistence gate design doc
  - Started design for strict `picture_automatrix.py` preprocessing mode
  - Reviewed `picture_automatrix.py` to capture exact preprocessing/segmentation steps
  - Noted `scipy`/`scikit-image` not in `requirements.txt` for strict mode
  - Implemented strict automatrix preprocess/segment mode with CLI selector
  - Added automatrix mode design doc
  - Added zero-range guards for automatrix normalization to avoid NaNs
  - Started design for edge-strength gating to suppress vague blobs
  - Implemented edge-strength gating for component boundaries
  - Added CLI flags for edge-strength gating
  - Added edge-strength gate design doc
  - Added edge-strength diagnostics export script
  - Added edge-strength diagnostics design doc
- Files created/modified:
  - scripts/inspect_frame21.py (created)
  - src/bubble_tracking/flow_tracking.py (updated)
  - scripts/run_flow_tracking.py (updated)
  - outputs/flow_tracking_test/overlay_frames (created)
  - outputs/flow_tracking_test/overlay_frames.mp4 (created)
  - outputs/flow_tracking_test_clahe/overlay_frames (created)
  - outputs/flow_tracking_test_clahe/overlay_frames.mp4 (created)
  - outputs/flow_tracking_test_clahe_with_compact/overlay_frames (created)
  - outputs/flow_tracking_test_clahe_with_compact/overlay_frames.mp4 (created)
  - src/bubble_tracking/flow_tracking.py (updated)
  - scripts/run_flow_tracking.py (updated)
  - docs/plans/2026-02-22-temporal-persistence-gate-design.md (created)
  - docs/plans/2026-02-22-automatrix-mode-design.md (created)
  - docs/plans/2026-02-22-edge-strength-gate-design.md (created)
  - scripts/export_edge_strength_diagnostics.py (created)
  - docs/plans/2026-02-22-edge-strength-diagnostics-design.md (created)
  - findings.md (updated)
  - task_plan.md (updated)
  - progress.md (updated)

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| `py_compile` changed modules/scripts | Updated Python files | No syntax/import-time errors | Passed | pass |
| `py_compile` flow tracking module | `src/bubble_tracking/flow_tracking.py`, `scripts/run_flow_tracking.py` | No syntax/import-time errors | Passed | pass |
| `py_compile` flow tracking (GPU edits) | `src/bubble_tracking/flow_tracking.py`, `scripts/run_flow_tracking.py` | No syntax/import-time errors | Passed | pass |
| `py_compile` CLAHE/compact CLI updates | `src/bubble_tracking/flow_tracking.py`, `scripts/run_flow_tracking.py` | No syntax/import-time errors | Passed | pass |
| `py_compile` persistence gating updates | `src/bubble_tracking/flow_tracking.py`, `scripts/run_flow_tracking.py` | No syntax/import-time errors | Passed | pass |
| `py_compile` automatrix mode updates | `src/bubble_tracking/flow_tracking.py`, `scripts/run_flow_tracking.py`, `scripts/inspect_frame21.py` | No syntax/import-time errors | Passed | pass |
| `py_compile` automatrix guard updates | `src/bubble_tracking/flow_tracking.py` | No syntax/import-time errors | Passed | pass |
| `py_compile` edge gate updates | `src/bubble_tracking/flow_tracking.py`, `scripts/run_flow_tracking.py`, `scripts/inspect_frame21.py` | No syntax/import-time errors | Passed | pass |
| `py_compile` edge diagnostics script | `scripts/export_edge_strength_diagnostics.py` | No syntax/import-time errors | Passed | pass |
| Synthetic split/merge scenario (`analyze_masks_by_frame`) | 3 handcrafted masks (split then merge) | `num_splits=1`, `num_merges=1` | Matched expected | pass |
| Real mask directory analysis (`analyze_masks_dir`) | `outputs/bubbles_air1_072_172_auto_gpu/masks` | Complete without crash; return summary counts | 101 frames analyzed; splits=4, merges=3 | pass |

## Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-02-02 20:31 | session-catchup.py not found at $HOME/.claude/plugins/planning-with-files | 1 | Proceeded with template-based planning files |
| 2026-02-02 20:36 | `data/bubles/Air1` not found | 1 | Use `data/bubbles/Air1` |
| 2026-02-02 20:39 | Frame regex skipped filenames like `1 (550).jpg` | 1 | Rename/copy to numeric filenames before running |
| 2026-02-02 20:41 | ModuleNotFoundError: cv2 (system python) | 1 | Use venv python |
| 2026-02-02 20:43 | apply_patch failed to find lines in findings.md | 1 | Viewed file and re-applied patch |
| 2026-02-02 20:46 | Torch not compiled with CUDA enabled | 1 | Install CUDA-enabled PyTorch or run on CPU |
| 2026-02-02 20:48 | pip cu121 torch install failed (no matching distribution) | 1 | Python 3.14 likely unsupported for CUDA wheels |
| 2026-02-02 20:50 | pip install requirements timed out (10s) | 1 | Re-ran with longer timeout |
| 2026-02-02 20:55 | torch.cuda.is_available() False; nvidia-smi NVML error | 1 | GPU not accessible in current shell |
| 2026-02-02 21:10 | TypeError: can't convert cuda tensor to numpy | 1 | Convert masks_out to CPU numpy in sam2_adapter |
| 2026-02-13 | `git status` failed in `/home/aaron/Downloads/bubble_tracking` (not a git repository) | 1 | Continued without git-derived status for this directory |
| 2026-02-13 | `ModuleNotFoundError: bubble_tracking` in ad-hoc test | 1 | Re-ran with `PYTHONPATH=src` |
| 2026-02-22 | `session-catchup.py` not found at `${CLAUDE_PLUGIN_ROOT}/scripts` | 1 | Proceeded without catchup; documented state in planning files |
| 2026-02-22 | `ValueError` from `np.histogram` (bins broadcast mismatch) | 1 | Switched to `cv2.calcHist` in segmentation |
| 2026-02-22 | `E: Unable to locate package libdc1394-22-dev` | 1 | Use `libdc1394-dev` or skip dependency |
| 2026-02-22 | `dpkg` errors during `ninja-build` install due to NVIDIA DKMS build failure | 1 | Remove stale crash file and re-run `dpkg --configure -a`; check DKMS log if it persists |
| 2026-02-22 | DKMS build failed: missing `os-interface.h`/`nv-firmware.h` | 1 | Reinstall `nvidia-dkms-550`/`nvidia-kernel-source-550`, then rerun `dpkg --configure -a` |
| 2026-02-22 | DKMS autoinstall blocked by existing crash file | 1 | Delete `/var/crash/nvidia-kernel-source-550.0.crash` then rerun `dpkg --configure -a` |
| 2026-02-22 | `session-catchup.py` not found at `${CLAUDE_PLUGIN_ROOT}/scripts` | 3 | Proceeded without catchup; logged state in planning files |
| 2026-02-22 | `session-catchup.py` not found at `${CLAUDE_PLUGIN_ROOT}/scripts` | 4 | Proceeded without catchup; logged state in planning files |
| 2026-02-22 | `session-catchup.py` not found at `${CLAUDE_PLUGIN_ROOT}/scripts` | 5 | Proceeded without catchup; logged state in planning files |
| 2026-02-22 | `session-catchup.py` not found at `${CLAUDE_PLUGIN_ROOT}/scripts` | 7 | Proceeded without catchup; logged state in planning files |
| 2026-02-22 | `FileNotFoundError` for `outputs/flow_tracking_test_edge_persist/events.json` | 1 | Run diagnostics with correct output dir |
| 2026-02-22 | `session-catchup.py` not found at `${CLAUDE_PLUGIN_ROOT}/scripts` | 8 | Proceeded without catchup; logged state in planning files |
| 2026-02-22 | `session-catchup.py` not found at `${CLAUDE_PLUGIN_ROOT}/scripts` | 4 | Proceeded without catchup; logged state in planning files |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Phase 5 (GPU Flow Acceleration) |
| Where am I going? | Run a short sample to confirm GPU or fallback behavior |
| What's the goal? | Implement flow-guided mask tracking with merge/split events and per-frame area |
| What have I learned? | See findings.md |
| What have I done? | Added CUDA Farneback backend with auto fallback and CLI flags |

---
*Update after completing each phase or encountering errors*

## Session: 2026-03-09 (Recovery after script overwrite)

### Recovery Audit
- **Status:** complete
- Actions taken:
  - Confirmed `run_sam2_prompted_flow.py` reverted to older prompt-once version.
  - Confirmed missing runtime features in `flow_tracking.py` and `run_flow_tracking.py`.

### Recovery Implementation
- **Status:** complete (core)
- Actions taken:
  - Patched `src/bubble_tracking/flow_tracking.py` to restore raw preprocess, flow toggle, and centroid-delta velocity.
  - Patched `scripts/run_flow_tracking.py` to expose restored CLI controls.
  - Patched `scripts/run_sam2_prompted_flow.py` to restore CV-keyframe prompt workflow and debug overlays.

### Verification
- **Status:** complete
- Actions taken:
  - `python -m py_compile src/bubble_tracking/flow_tracking.py scripts/run_flow_tracking.py scripts/run_sam2_prompted_flow.py`
  - Verified help output with venv Python includes restored flags for both scripts.

### Files modified
- `src/bubble_tracking/flow_tracking.py`
- `scripts/run_flow_tracking.py`
- `scripts/run_sam2_prompted_flow.py`

### Session: 2026-03-09 (Velocity normalized plots)
- **Status:** complete
- Actions taken:
  - Added `scripts/plot_velocity_norm_summary.py`.
  - Generated requested plots for:
    - `outputs/case2_air1_sam2_cv_keyframe100_nonorm_centroid_delta`
    - `outputs/case3_air1_sam2_cv_keyframe100_nonorm_centroid_delta`
    - `outputs/case3_air4_sam2_cv_keyframe100_nonorm_centroid_delta`
- Generated files per run:
  - `velocity_norm_histograms.png`
  - `velocity_area_scatter.png`
  - `bubble_velocity_area_metrics.csv`
- Updated velocity histogram plotting to automatic binning (`--bins auto`) and regenerated all three run visualizations.
- Generated log-axis scatter variants (`logx`, `logxy`) for case2_air1, case3_air1, case3_air4 run directories.

### Session: 2026-03-09 (Velocity PDF + distribution fit)
- **Status:** complete
- Actions taken:
  - Extended `scripts/plot_velocity_norm_summary.py` with PDF overlays and parametric distribution fitting.
  - Added outputs:
    - `velocity_norm_pdf_fit.png`
    - `velocity_distribution_fit.csv`
    - `velocity_distribution_best.json`
  - Regenerated fit outputs for:
    - `outputs/case3_air1_sam2_cv_keyframe100_nonorm_centroid_delta`
    - `outputs/case3_air4_sam2_cv_keyframe100_nonorm_centroid_delta`
  - Confirmed best-fit family across case1/case2/case3 runs is Student's t for all normalized velocity components.

### Files modified
- `scripts/plot_velocity_norm_summary.py`
- `findings.md`
- `progress.md`

### Session: 2026-03-09 (Scatter best-fit line)
- **Status:** complete
- Actions taken:
  - Added best-fit line overlay to all velocity-area scatter subplots.
  - Added legend entry with `R²` on each subplot.
  - Regenerated plot outputs for:
    - `case1_air1_sam2_cv_keyframe160_nonorm_centroid_delta`
    - `case2_air1_sam2_cv_keyframe100_nonorm_centroid_delta`
    - `case3_air1_sam2_cv_keyframe100_nonorm_centroid_delta`
    - `case3_air4_sam2_cv_keyframe100_nonorm_centroid_delta`

### Files modified
- `scripts/plot_velocity_norm_summary.py`
- `findings.md`
- `progress.md`

### Session: 2026-03-09 (Add log-y scatter output)
- **Status:** complete
- Actions taken:
  - Added `velocity_area_scatter_logy.png` generation to `plot_velocity_norm_summary.py`.
  - Regenerated plots for:
    - `case1_air1_sam2_cv_keyframe160_nonorm_centroid_delta`
    - `case2_air1_sam2_cv_keyframe100_nonorm_centroid_delta`
    - `case3_air1_sam2_cv_keyframe100_nonorm_centroid_delta`
    - `case3_air4_sam2_cv_keyframe100_nonorm_centroid_delta`

### Files modified
- `scripts/plot_velocity_norm_summary.py`
- `findings.md`
- `progress.md`
