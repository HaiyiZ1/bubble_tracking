# Task Plan: Bubble Tracking (Flow + Masks) (2026-02-22)

## Goal
Implement the flow-guided mask tracking pipeline described in `docs/plans/2026-02-22-bubble-tracking-design.md` for bubble datasets, focusing on births, splits, merges, deaths, per-frame area, and per-track lineage metadata.

## Current Phase
Phase 8

## Phases

### Phase 1: Code & Data Review
- [x] Inspect `next_steps_26-02-21.md` and `picture_automatrix.py` for integration points
- [x] Identify current data I/O, mask formats, and outputs
- **Status:** complete

### Phase 2: Preprocessing + Segmentation
- [x] Implement/confirm thresholding and cleanup pipeline
- [x] Add zero-bubble reference baseline handling
- **Status:** complete

### Phase 3: Flow-Guided Matching + Events
- [x] Add optical flow warp (Farneback or DIS)
- [x] Compute overlap matrix and match by IoU + gating
- [x] Emit events (birth/death/split/merge) with simple log
- [x] Assign IDs (new on split/merge as needed)
- **Status:** complete

### Phase 4: Outputs + Validation
- [x] Save annotated overlay frames
- [x] Save CSV/JSON of events and area
- [x] Add per-frame progress logging (stage + processed/total)
- [x] Annotate tracked IDs with birth/death reasons and lineage links for later visualization
- [ ] Run on small sample; sanity check counts and runtime
- **Status:** in_progress

### Phase 5: GPU Flow Acceleration
- [x] Add CUDA Farneback flow path with auto CPU fallback and warning
- [x] Add CLI flag to control flow backend (auto/cpu) and CUDA device selection
- [x] Emit runtime flow backend info in outputs
- [ ] Run a short sample to confirm GPU path or fallback behavior
- **Status:** in_progress

### Phase 6: Noise Reduction (Lighting Drift)
- [x] Verify false positives on frame 21 (component stats vs reference)
- [x] Add filtering/normalization to suppress lighting-change artifacts
- [x] Add CLAHE + compact-blob filter flags for reproducible runs
- [x] Design temporal persistence gate for flicker noise
- [x] Implement temporal persistence gate with 1-frame backfill
- [x] Add CLI flags for temporal persistence gating
- [x] Add strict `picture_automatrix.py` preprocessing mode
- [x] Add edge-strength gating for weak contours
- [x] Add CLI flags for edge-strength gating
- [x] Add edge-strength diagnostics export (overlays + CSV/JSON)
- [ ] Validate improved masks on a short subset
- **Status:** in_progress

### Phase 7: SAM2 Mask Integration (Flow Tracking)
- [x] Add `--masks-dir` option to load SAM2 masks per frame
- [x] Align mask frames with input frames; validate shapes/ROI
- [x] Re-label components per frame (connected components) to avoid relying on SAM2 IDs
- [x] Keep flow computation on preprocessed frames; use SAM2 masks for segmentation only
- [x] Add `--sam2-auto` option to generate per-frame SAM2 auto-masks
- [x] Emit `sam2_auto_config.json` with SAM2 parameters
- [x] Add `--sam2-propagate` option for prompt search + SAM2 propagation
- [x] Create minimal end-to-end runner `scripts/run_sam2_prompted_flow.py`
- [ ] Run a short subset for sanity check (events + overlay)
- [ ] Normalize full Air1 dataset filenames and run SAM2 auto-search on full set
- [x] Add per-second stats + plot script for first 900 frames
- **Status:** in_progress

### Phase 8: Publishing Prep (Case1-3 Air1)
- [x] Add a standalone relabel step that discards pre-bubble frames and rewrites case1/case2/case3 Air1 frame roots
- [x] Add a standalone publishing overlay exporter with fixed titles, elapsed-time text, 5-minute cap, and 1x/10x outputs
- [x] Document the adjusted prompt/frame offsets for relabeled datasets so pipeline commands can be re-run safely
- **Status:** complete

## Assumptions
- Zero-bubble reference image exists
- Camera is stationary
- FPS constant per dataset (default 4 fps for current)
- Crowding increases over time; frequent merges/splits later

## Errors Encountered (Current Task)
| Error | Attempt | Resolution |
|-------|---------|------------|
| `session-catchup.py` not found at `${CLAUDE_PLUGIN_ROOT}/scripts` | 1 | Proceeded without catchup; logged state in planning files |
| `ValueError` in `np.histogram` during segmentation (NumPy bins mismatch) | 1 | Switched to `cv2.calcHist` for histogram computation |
| `ModuleNotFoundError: cv2` when checking OpenCV build | 1 | Use the project venv Python (e.g., `venv311/bin/python`) |
| `E: Unable to locate package libdc1394-22-dev` during OpenCV deps install | 1 | Use `libdc1394-dev` or skip (FireWire optional) |
| `dpkg` failed during `ninja-build` install due to NVIDIA DKMS error | 1 | Remove stale crash file and re-run `dpkg --configure -a`; inspect DKMS log if it fails |
| NVIDIA DKMS autoinstall fails: `Cannot create report ... nvidia-kernel-source-550.0.crash` | 1 | Delete crash file and re-run `dpkg --configure -a` |
| `session-catchup.py` not found at `${CLAUDE_PLUGIN_ROOT}/scripts` (repeat) | 2 | Proceeded without catchup; logged state in planning files |
| `session-catchup.py` not found at `${CLAUDE_PLUGIN_ROOT}/scripts` (repeat) | 3 | Proceeded without catchup; logged state in planning files |
| `session-catchup.py` not found at `${CLAUDE_PLUGIN_ROOT}/scripts` (repeat) | 4 | Proceeded without catchup; logged state in planning files |
| `session-catchup.py` not found at `${CLAUDE_PLUGIN_ROOT}/scripts` (repeat) | 5 | Proceeded without catchup; logged state in planning files |
| `session-catchup.py` not found at `${CLAUDE_PLUGIN_ROOT}/scripts` (repeat) | 6 | Proceeded without catchup; logged state in planning files |
| `session-catchup.py` not found at `${CLAUDE_PLUGIN_ROOT}/scripts` (repeat) | 7 | Proceeded without catchup; logged state in planning files |
| `session-catchup.py` not found at `${CLAUDE_PLUGIN_ROOT}/scripts` (repeat) | 8 | Proceeded without catchup; logged state in planning files |
| `FileNotFoundError` for `outputs/flow_tracking_test_edge_persist/events.json` | 1 | Run diagnostics with correct output dir |
| `session-catchup.py` not found at `${CLAUDE_PLUGIN_ROOT}/scripts` (repeat) | 4 | Proceeded without catchup; logged state in planning files |
| `session-catchup.py` not found at `${CLAUDE_PLUGIN_ROOT}/scripts` | 9 | Used `/home/aaron/.codex/skills/planning-with-files/scripts/session-catchup.py` |
| `session-catchup.py` not found at `${CLAUDE_PLUGIN_ROOT}/scripts` | 10 | Used `/home/aaron/.codex/skills/planning-with-files/scripts/session-catchup.py` |
| `git rev-parse --is-inside-work-tree` failed (not a git repo) | 1 | Skipped commit for design doc |
| SAM2 automask OOM (`torch.OutOfMemoryError` in `SAM2AutomaticMaskGenerator.generate`) | 1 | Reduce `--sam2-mask-scale`/`--sam2-points-per-side` or add `points_per_batch` tuning |
| Full-dataset SAM2 propagation killed mid-run | 1 | Likely OOM (video frames loaded at full resolution); run in chunks or downscale |
| Synthetic split validation produced one component instead of two | 1 | Added a 1-pixel gap between child masks so connected-components emitted a real split |

## Archived Plan (2026-02-13)

# Task Plan: Bubble Area Dynamics + Split/Merge Events

## Goal
Pivot the project from per-object bubble tracking to mask-driven analysis:
1) compute total bubble area over time from SAM2 masks, and
2) detect split/merge events from frame-to-frame mask topology.

## Current Phase
Phase 5

## Phases

### Phase 1: Requirements & Discovery
- [x] Capture new objective (aggregate area + split/merge)
- [x] Audit current runner outputs and mask availability
- [x] Identify integration points in `run_sam2_video.py`, `run_reprompted_tracking.py`, and `sam2_adapter.py`
- **Status:** complete

### Phase 2: Design
- [x] Select event detection strategy based on connected components + overlap linkage
- [x] Define output artifacts (`area_timeseries.json`, `area_timeseries.csv`, `events.json`)
- [x] Keep compatibility with existing `tracks.json`/`stats.json` outputs
- **Status:** complete

### Phase 3: Implementation
- [x] Add mask analysis module (area series + split/merge detection)
- [x] Integrate analysis outputs into runner scripts
- [x] Fix `sam2_adapter` frame property bug (properties only computed under ROI branch)
- **Status:** complete

### Phase 4: Testing & Verification
- [x] Run synthetic validation of split/merge logic
- [x] Run script-level dry/smoke validation (imports + CLI path)
- [x] Confirm output file contracts and JSON schema sanity
- **Status:** complete

### Phase 5: Delivery
- [ ] Summarize implementation details and assumptions
- [ ] List next tunable thresholds for split/merge robustness
- **Status:** pending

## Key Questions
1. Should split/merge be detected on every adjacent frame or only keyframes? (default: adjacent frame pairs)
2. What minimum overlap should define parent-child linkage? (default: conservative overlap thresholds)
3. Should tiny components be ignored as noise? (default: yes, configurable min area)

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Use SAM2 masks as primary data product | Aligns with new objective; avoids fragile persistent IDs |
| Compute area from binary union (`mask > 0`) | Measures total bubble occupancy regardless of SAM2 instance IDs |
| Detect split/merge via connected components + overlap linkage | Works even when SAM2 object IDs drift/restart |
| Emit analysis as separate artifacts | Keeps backward compatibility with existing pipeline outputs |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| `git status` failed in project root (`not a git repository`) | 1 | Continue without git-based diffing in this directory |
| `ModuleNotFoundError: bubble_tracking` during synthetic test | 1 | Re-run with `PYTHONPATH=src` |

## Notes
- Update phase status as you progress: pending -> in_progress -> complete
- Re-read this plan before major decisions
- Log all errors and threshold assumptions

## Session: 2026-03-09 Recovery (Post-overwrite)

### Recovery Phase A: Loss Audit
- [x] Verify which files reverted after overwrite
- [x] Confirm missing features against current run commands (case2/case3)
- **Status:** complete

### Recovery Phase B: Reapply Core Runtime Features
- [x] Restore raw preprocess mode + sigma control in tracking core
- [x] Restore flow on/off control and velocity source mode (`flow_median` / `centroid_delta`)
- [x] Restore SAM2 CV-keyframe prompt mode in `run_sam2_prompted_flow.py`
- [x] Restore CV keyframe debug overlay export option
- [ ] Re-implement advanced event/global reprompt modes (deferred unless needed)
- **Status:** in_progress

### Recovery Phase C: Validate and Resume Runs
- [x] Syntax check patched files
- [x] Confirm CLI flags required by current case2/case3 commands
- [ ] Run full end-to-end tracking on case2/case3 (user runtime)
- **Status:** in_progress
