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

## Session: 2026-03-24

### Phase: Track Lineage Metadata
- **Status:** complete
- **Started:** 2026-03-24 19:35 AEDT
- Actions taken:
  - Re-read planning and brainstorming skill instructions before changing tracking behavior
  - Extended `TrackedObject` with birth/death lifecycle metadata fields for lineage export
  - Added a post-processing annotation pass in `src/bubble_tracking/flow_tracking.py` to map final events onto per-track birth/death reasons and parent/child IDs
  - Preserved correctness after debounce/relink by annotating lifecycle metadata after the final mask/result rebuild
  - Ran `venv311/bin/python -m py_compile` on the changed tracking modules and runners
  - Ran a synthetic split -> merge -> death tracking scenario and verified the resulting lifecycle metadata in `tracks.json`
- Files created/modified:
  - src/bubble_tracking/schemas.py (updated)
  - src/bubble_tracking/flow_tracking.py (updated)
  - task_plan.md (updated)
  - findings.md (updated)
  - progress.md (updated)

### Phase: Publishing Prep Workflow
- **Status:** complete
- **Started:** 2026-03-24 20:10 AEDT
- Actions taken:
  - Audited current case roots and confirmed case1 still points at the generic `data/bubbles` tree while case2/case3 already have dedicated roots
  - Captured publishing presets for case1/case2/case3 Air1, including first usable frame numbers, titles, and shifted keyframe prompt indices after relabeling
  - Added `src/bubble_tracking/publishing.py` to centralize publishing case presets
  - Added `scripts/relabel_publishing_air1.py` to copy only the post-start frames into new relabeled publishing roots and emit `frame_map.csv` plus `publishing_relabel_config.json`
  - Added `scripts/export_publishing_tracking_video.py` to export 1x and 10x tracking overlays with a title banner, elapsed-time text, and a 5-minute cap, without changing default pipeline video behavior
  - Updated `README.md` with the new relabel + rerun + publishing-overlay workflow and adjusted `--cv-keyframe-start-idx` values
  - Ran syntax checks, CLI help checks, a temp-directory relabel test, and a temp-directory publishing-overlay export test
- Files created/modified:
  - src/bubble_tracking/publishing.py (created)
  - scripts/relabel_publishing_air1.py (created)
  - scripts/export_publishing_tracking_video.py (created)
  - README.md (updated)
  - task_plan.md (updated)
  - findings.md (updated)
  - progress.md (updated)
  - Added mask/frame index alignment and SAM2 mask relabeling in `src/bubble_tracking/flow_tracking.py`

### Phase: Stateful Debounce In Tracking Loop
- **Status:** complete
- **Started:** 2026-03-24 21:05 AEDT
- Actions taken:
  - Re-read brainstorming and planning-with-files skill instructions before changing core tracking behavior
  - Replaced post-hoc debounce/rebuild in `track_frames()` with in-loop tentative-birth confirmation and missing-track grace handling
  - Added frame buffering so tentative 1-frame flickers can be removed before masks/results are flushed to disk
  - Kept confirmed tracks alive through short absences by persisting the same ID and carrying forward a fallback mask
  - Preserved explicit split/merge lineage behavior with immediate parent death and child birth events
  - Re-sorted final events and reused lifecycle annotation so `birth_reason` / `death_reason` match the new stable track history
  - Patched death metadata so `death_frame_idx` records the first missing frame rather than the later confirmation frame
  - Ran `venv311/bin/python -m py_compile` on the updated tracking modules and runners
  - Ran targeted synthetic scenarios for flicker suppression, internal-gap continuity, terminal-gap handling, and split/merge lineage
  - Verified that true deaths now backdate to the first missing frame and retro-trim any grace-only masks from `tracks.json`, saved mask PNGs, and `area_timeseries`
  - Verified that gap masks are still carried forward and preserved when the bubble reappears within the grace window
- Files created/modified:
  - src/bubble_tracking/flow_tracking.py (updated)
  - task_plan.md (updated)
  - findings.md (updated)
  - progress.md (updated)

### Phase: Vanished Area Moving-Window Visualization
- **Status:** complete
- **Started:** 2026-03-25
- Actions taken:
  - Added `scripts/plot_vanished_area_moving_window.py` to compute and plot vanished bubble area using only true `death` events
  - Implemented centered 2s (`+-1s`) and 5s (`+-2.5s`) windows sampled at integer-second centers
  - Added four output plots with two x-axis variants (seconds and frames)
  - Added CSV export with per-second center frame and both window totals
  - Ran `venv311/bin/python -m py_compile scripts/plot_vanished_area_moving_window.py`
  - Ran a synthetic event test to verify:
    - only `death` events are counted
    - all four PNGs are generated
    - CSV values match centered-window expectations
- Files created/modified:
  - scripts/plot_vanished_area_moving_window.py (created)
  - task_plan.md (updated)
  - findings.md (updated)
  - progress.md (updated)

### Phase: Publish Velocity/Lineage Exports
- **Status:** complete
- **Started:** 2026-03-25
- Actions taken:
  - Updated `scripts/export_bubble_max_velocity_summary.py` to target publish run dirs (`case1/2/3_air1_publish`) by default
  - Added lineage columns to summary CSV from `tracks.json` (`birth_reason`, `death_reason`, parent/child IDs)
  - Verified exporter output and CSV headers by writing summaries to a temp output directory
  - Verified publish run metric generation with `scripts/plot_velocity_norm_summary.py` to produce `bubble_velocity_area_metrics.csv`
- Files created/modified:
  - scripts/export_bubble_max_velocity_summary.py (updated)
  - task_plan.md (updated)
  - findings.md (updated)
  - progress.md (updated)

### Phase: Framewise Injection/Vanish Area CSV Export
- **Status:** complete
- **Started:** 2026-03-27
- Actions taken:
  - Added `scripts/export_injection_vanished_area_timeseries.py`
  - Implemented per-frame aggregation from `events.json`:
    - `birth` events -> total injection-birth area per frame
    - `death` events -> total vanished-death area per frame
  - Implemented px^2 -> mm^2 conversion using run-specific `mm_per_pixel`
  - Exported CSVs into:
    - `outputs/case1_air1_publish/injection_vanished_area_timeseries.csv`
    - `outputs/case2_air1_publish/injection_vanished_area_timeseries.csv`
    - `outputs/case3_air1_publish/injection_vanished_area_timeseries.csv`
  - Verified headers and row counts:
    - case1: 1678 lines (1677 frames + header)
    - case2: 1313 lines (1312 frames + header)
    - case3: 1399 lines (1398 frames + header)
- Files created/modified:
  - scripts/export_injection_vanished_area_timeseries.py (created)
  - task_plan.md (updated)
  - findings.md (updated)
  - progress.md (updated)
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

### Session: 2026-03-12 (Residual saturation comparison)
- **Status:** complete
- Actions taken:
  - Parsed `traditional_CV_residual_saturation_from_haiyi.csv`.
  - Computed SAM2 metric per run as mean of last 4 `total_area_px` values divided by ROI mask area.
  - Compared against Haiyi residual saturation values for:
    - case1_air1, case2_air1, case3_air1, case3_air4.
  - Observed SAM2 metric is lower than Haiyi across all four cases by ~73% to ~79%.

### Files modified
- `findings.md`
- `progress.md`

### Session: 2026-03-15 (Birth/Death debounce for flicker)
- **Status:** complete
- Actions taken:
  - Added debounce config to tracking (`DebounceConfig`) and wired into `TrackingConfig`.
  - Implemented mask postprocess debounce in `flow_tracking.py`:
    - suppress short ON runs (birth debounce)
    - fill short OFF gaps (death debounce hold)
    - recompute `area_timeseries` from debounced masks
    - regenerate birth/death events from debounced masks
    - rebuild track frame records from debounced masks (consistent overlays/tracks CSV)
  - Added CLI flags to both runners:
    - `--debounce-birth-frames`
    - `--debounce-death-frames`
    - `--debounce-relink-max-centroid-dist-px`
    - `--debounce-relink-min-area-ratio`
    - `--debounce-relink-max-area-ratio`
  - Added gap re-link logic so short-gap reappearances can keep old IDs instead of creating new IDs.
  - Verified with `py_compile` and `--help` checks in `venv311`.

### Files modified
- `src/bubble_tracking/flow_tracking.py`
- `scripts/run_sam2_prompted_flow.py`
- `scripts/run_flow_tracking.py`
- `findings.md`
- `progress.md`

### Session: 2026-03-16 (Case1 scatter with marginal PDFs)
- **Status:** complete
- Actions taken:
  - Added `scripts/plot_velocity_area_logxy_with_marginal_pdfs.py`.
  - Generated a new non-overwriting case1 figure with:
    - log-log scatter using only `max_v_over_v_ref_2`
    - top PDF of x variable
    - right PDF of y variable
  - Output:
    - `outputs/case1_air1_sam2_cv_keyframe160_nonorm_centroid_delta/velocity_area_scatter_logxy_vref2_with_pdfs.png`

### Files modified
- `scripts/plot_velocity_area_logxy_with_marginal_pdfs.py`
- `findings.md`
- `progress.md`

### Session: 2026-04-13 (Case2 Air1 65/70 threshold variants)
- **Status:** complete
- Actions taken:
  - Re-ran exporter with `mm2-threshold=5.0` and ratio thresholds:
    - `0.65`
    - `0.70`
  - Used temporary output directory `/tmp/case2_threshold_exports_65_70`.
  - Copied only case2 outputs into `outputs/`.
  - Verified output file presence, 16-column headers, and row counts.

### Files modified
- `task_plan.md`
- `findings.md`
- `progress.md`

### Session: 2026-04-19 (Applied id49 crop v4 and rerendered frame 0304/0234)
- **Status:** complete
- Actions taken:
  - Confirmed input files exist:
    - `000233_manual_mask_id49_crop_10pct_after_manual_change_v4.csv`
    - `000233_manual_mask_id49_crop_10pct_meta.json`
    - `000233_manual_mask_id48_void_filled.png`
  - Read crop bounds from metadata and validated edited CSV shape/values.
  - Merged edited crop into full-frame mask with protected-ID checks.
  - Wrote merged mask:
    - `000233_manual_mask_id48_void_filled_id49_from_crop_v4.png`
  - Re-rendered final frame image using merged mask and `x_offset=-70`.
  - Verification:
    - merged IDs: `[0,16,48,49]`
    - `pixels_id49=21993`
    - render: `IDs rendered: 3`, `clamped_labels=0`

### Files modified
- `task_plan.md`
- `findings.md`
- `progress.md`

### Session: 2026-04-10 (Three escaping threshold-variant CSVs)
- **Status:** complete
- Actions taken:
  - Re-read:
    - `journal1_three_escaping_cases_subcount_outputCSV_requirements.md`
    - `pixel_mm_conversion_standard.md`
  - Confirmed `scripts/export_three_escaping_subcount_timeseries.py` supports configurable thresholds and non-overwriting output template.
  - Regenerated three new variant outputs using:
    - `--ratio-threshold 0.5`
    - `--mm2-threshold 5.0`
    - `--output-template '{case_key}_air1_time_series_subcount_three_escaping_situation_threshold_50%_and_5mm2.csv'`
  - Verified all three files exist and share the expected 16-column schema.

### Files modified
- `task_plan.md`
- `findings.md`
- `progress.md`

### Session: 2026-04-13 (Case2 Air1 40/60/80 threshold variants)
- **Status:** complete
- Actions taken:
  - Re-read planning files and key requirements context.
  - Generated threshold variants with `mm2-threshold=5.0` and ratio thresholds:
    - `0.4`
    - `0.6`
    - `0.8`
  - Used temporary output directory `/tmp/case2_threshold_exports` and copied only case2 files to `outputs/`.
  - Verified output file presence, 16-column headers, and row counts.

### Files modified
- `task_plan.md`
- `findings.md`
- `progress.md`

### Session: 2026-04-06 (Three escaping situations subcount CSVs)
- **Status:** complete
- Actions taken:
  - Read requirements from `journal1_three_escaping_cases_subcount_outputCSV_requirements.md`.
  - Read conversion key from `pixel_mm_conversion_standard.md` and logged the downscaled 0.5 per-case ratios as a persistent key in findings.
  - Added `scripts/export_three_escaping_subcount_timeseries.py` implementing all 16 requested columns.
  - Implemented split/merge escape formulas with threshold:
    - `max(20% area term, 3.14 mm^2 converted to px^2)` per case.
  - Generated output files:
    - `outputs/case1_air1_time_series_subcount_three_escaping_situation.csv`
    - `outputs/case2_air1_time_series_subcount_three_escaping_situation.csv`
    - `outputs/case3_air1_time_series_subcount_three_escaping_situation.csv`
  - Validated:
    - headers and initial rows,
    - nonzero vanished/split/merge rows and parent-child string formatting.

### Files modified
- `scripts/export_three_escaping_subcount_timeseries.py` (created)
- `task_plan.md` (updated)
- `findings.md` (updated)
- `progress.md` (updated)

### Session: 2026-03-16 (Label/LaTeX/title styling update)
- **Status:** complete
- Actions taken:
  - Updated case1 marginal-PDF plot script with:
    - explicit "PDF of ..." labels for both marginal PDFs
    - mathtext/LaTeX style for title and axis labels
    - bold main title
  - Regenerated:
    - `outputs/case1_air1_sam2_cv_keyframe160_nonorm_centroid_delta/velocity_area_scatter_logxy_vref2_with_pdfs.png`
  - Confirmed original `velocity_area_scatter_logxy.png` remains unchanged.

### Files modified
- `scripts/plot_velocity_area_logxy_with_marginal_pdfs.py`
- `findings.md`
- `progress.md`

### Session: 2026-03-18 (Per-bubble max velocity CSV export)
- **Status:** complete
- Actions taken:
  - Added `scripts/export_bubble_max_velocity_summary.py`.
  - Exported three requested summary CSVs for:
    - `case1_air1`
    - `case2_air1`
    - `case3_air1`
  - Verified headers and sample rows in all three outputs.
  - Chosen interpretation:
    - component maxima use absolute `x/y` speed per frame
    - total max velocity uses the real magnitude column per frame
    - appearance time = appearance frame count divided by `fps`

### Files modified
- `scripts/export_bubble_max_velocity_summary.py`
- `findings.md`
- `progress.md`

### Session: 2026-03-16 (PDF coordinate labels + decimal ratio format)
- **Status:** complete
- Actions taken:
  - Updated right-marginal ratio values to decimal formatting (no percent style).
  - Added coordinate annotations for each PDF at 3 points:
    - left end
    - peak
    - right end
  - Regenerated:
    - `outputs/case1_air1_sam2_cv_keyframe160_nonorm_centroid_delta/velocity_area_scatter_logxy_vref2_with_pdfs.png`

### Files modified
- `scripts/plot_velocity_area_logxy_with_marginal_pdfs.py`
- `findings.md`
- `progress.md`

### Session: 2026-04-18 (Case1 publish frames 301-311 label relayout)
- **Status:** complete
- Actions taken:
  - Confirmed target output folder:
    - `outputs/case1_air1_publish_bubble_numbering_size_enlarge_frames_with_original_frame_numbering_301_to_311`
  - Confirmed requested source frame range mapping using relabel metadata:
    - original `301..311` -> relabeled frame numbers `231..241` -> frame indices `230..240`
  - Confirmed original annotation behavior from:
    - `src/bubble_tracking/visualization.py::draw_tracking_info`
  - Prepared export strategy to render 11 still frames from case1 publish artifacts with requested label changes:
    - labels outside below-right of bbox
    - black text, transparent background
    - size scaled by `60/36`
  - First generation attempt failed with system python (`cv2` missing); switched to `./venv/bin/python`.
  - Generated and wrote 11 frames for original frame numbers `301..311`.
  - Verified output directory contains exactly 11 `.jpg` files and filenames include original frame numbers.

### Files modified
- `task_plan.md`
- `findings.md`
- `progress.md`

### Session: 2026-04-18 (Case1 publish label position correction to “WANT”)
- **Status:** complete
- Actions taken:
  - Applied follow-up location correction on the same 11 files in:
    - `outputs/case1_air1_publish_bubble_numbering_size_enlarge_frames_with_original_frame_numbering_301_to_311`
  - Regenerated original-frame `301..311` overlays with corrected anchor:
    - horizontal: `tx = bbox.x_max` (directly below corner; removed right shift)
    - vertical: `ty = bbox.y_max + 4 + text_height`
  - Kept unchanged:
    - black text
    - transparent label background
    - size scaling `60/36` (`font_scale=1.5`, `font_thickness=3`)
  - Verification:
    - exactly 11 output files present
    - render reported `clamped_labels=0`

### Files modified
- `task_plan.md`
- `findings.md`
- `progress.md`

### Session: 2026-04-18 (Manual coordinate export for label placement)
- **Status:** complete
- Actions taken:
  - Exported per-bubble coordinate handoff CSV for original frames `301..311`.
  - Included:
    - bbox right-bottom corner coordinates
    - label text size metrics
    - relative and absolute coordinates for current and alternative placements
  - Output:
    - `outputs/case1_air1_publish_bubble_numbering_size_enlarge_frames_with_original_frame_numbering_301_to_311/label_relative_coordinates_original_301_to_311.csv`
  - Row count:
    - `54`

### Files modified
- `task_plan.md`
- `findings.md`
- `progress.md`

### Session: 2026-04-19 (Times-style font regeneration for 11 frames)
- **Status:** complete
- Actions taken:
  - Checked available fonts and confirmed Times New Roman is not installed.
  - Selected Times-compatible `Nimbus Roman` as fallback.
  - Regenerated all 11 target frames in-place using PIL TrueType text rendering.
  - Preserved latest placement (`x_offset=-90`) and black transparent label styling.
  - Verified run output:
    - regenerated `11` frames
    - `clamped_labels=0`

### Files modified
- `task_plan.md`
- `findings.md`
- `progress.md`

### Session: 2026-04-19 (Manual mask copy + rerender setup for frame 0304/0234)
- **Status:** complete
- Actions taken:
  - Located the correct source mask for relabeled frame 0234: `outputs/case1_air1_publish/masks/000233.png`.
  - Copied it to user manual-edit folder as `000233_manual_mask.png`.
  - Added `scripts/render_case1_0234_from_manual_mask.py` to regenerate one frame from edited mask.
  - Implemented recomputation of bbox/centroid from edited mask IDs, preserving original numbering IDs.
  - Ran one verification render, producing:
    - `outputs/case1_air1_publish_original_0304_relabeled_0234_manual_change/case1_air1_publish_original_0304_relabeled_0234_label_outside_black_text_scale_60_over_36.jpg`
  - Verification output:
    - `IDs rendered: 3`
    - `clamped_labels=0`

### Files modified
- `scripts/render_case1_0234_from_manual_mask.py` (created)
- `task_plan.md`
- `findings.md`
- `progress.md`

### Session: 2026-04-19 (Implemented matrix export/import + rerender workflow for frame 0304/0234)
- **Status:** complete
- Actions taken:
  - Added `scripts/matrix_roundtrip_case1_0234.py` with `export`/`import` subcommands.
  - Exported editable full-frame binary matrices for `#48` and `#49` into manual folder.
  - Added strict import validations (shape, binary values, empties/NaN/non-int) and protected-pixel guard.
  - Implemented ID-preserving rebuild logic with overlap policy (`#49` wins).
  - Verified export dimensions (`1836 x 2748` both CSVs).
  - Ran synthetic edit test and confirmed:
    - IDs unchanged: `[0,16,48,49]`
    - #49 bbox/centroid moved downward.
  - Ran existing renderer from edited mask and produced test image successfully.

### Files modified
- `scripts/matrix_roundtrip_case1_0234.py` (created)
- `task_plan.md`
- `findings.md`
- `progress.md`

### Session: 2026-04-19 (Compact CSV-edit workflow for frame 0304/0234)
- **Status:** complete
- Actions taken:
  - Re-read `brainstorming` and `planning-with-files` skills per user-invoked workflow.
  - Reviewed existing scripts:
    - `scripts/matrix_roundtrip_case1_0234.py`
    - `scripts/render_case1_0234_from_manual_mask.py`
  - Confirmed user blocker: full-frame CSV width (`2748` columns) exceeds viewer limit (`1000`).
  - Extended `scripts/matrix_roundtrip_case1_0234.py` with `prepare-compact` subcommand.
  - Implemented compact workflow outputs:
    - `000233_manual_mask_id48_void_filled.png`
    - `000233_manual_mask_id49_crop_10pct_before_manual_change.csv`
    - `000233_manual_mask_id49_crop_10pct_meta.json`
  - `prepare-compact` run stats:
    - `id48_hole_fill_pixels_added=1677`
    - crop bounds:
      - `row_start=1499`, `row_end_exclusive=1767`
      - `col_start=1296`, `col_end_exclusive=1470`
    - cropped CSV shape: `268 x 174`
  - Re-rendered frame with updated mask using existing style (`x_offset=-70`, black text, Times-style fallback):
    - `case1_air1_publish_original_0304_relabeled_0234_label_outside_black_text_scale_60_over_36.jpg`
  - Verification:
    - IDs unchanged: `[0,16,48,49]`
    - `#16` unchanged, `#49` unchanged, `#48` increased by hole-fill pixels
    - `clamped_labels=0`
  - Encountered and fixed one parser issue:
    - `argparse` help string failed on `%`; escaped to `%%`.

### Files modified
- `scripts/matrix_roundtrip_case1_0234.py` (updated)
- `task_plan.md`
- `findings.md`
- `progress.md`

### Session: 2026-04-19 (Applied id49 crop v3 and rerendered frame 0304/0234)
- **Status:** complete
- Actions taken:
  - Confirmed input files exist:
    - `000233_manual_mask_id49_crop_10pct_after_manual_change_v3.csv`
    - `000233_manual_mask_id49_crop_10pct_meta.json`
    - `000233_manual_mask_id48_void_filled.png`
  - Read crop bounds from metadata and validated edited CSV shape/values.
  - Merged edited crop into full-frame mask with protected-ID checks.
  - Wrote merged mask:
    - `000233_manual_mask_id48_void_filled_id49_from_crop_v3.png`
  - Re-rendered final frame image using merged mask and `x_offset=-70`.
  - Verification:
    - merged IDs: `[0,16,48,49]`
    - `pixels_id49=21932`
    - render: `IDs rendered: 3`, `clamped_labels=0`

### Files modified
- `task_plan.md`
- `findings.md`
- `progress.md`

### Session: 2026-04-19 (Case2 Air1 SAM2 self-reprompt @160)
- Started experiment setup for requested comparison vs CV-keyframe reprompt baseline.
- Confirmed baseline run metadata and keyframe schedule from:
  - `outputs/case2_air1_publish/sam2_prompted_flow_config.json`
  - `outputs/case2_air1_publish/sam2_cv_keyframe_prompt_summary.json`
- Baseline key details recorded:
  - keyframe interval `160`
  - start index `2`
  - first prompt index `2`
  - first prompt area `35365`
- Next action: implement a dedicated self-mask-reprompt script and run it on `case2_air1`.

### Files modified
- `task_plan.md`
- `findings.md`
- `progress.md`
