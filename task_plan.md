# Task Plan: Bubble Tracking (Flow + Masks) (2026-02-22)

## Goal
Implement the flow-guided mask tracking pipeline described in `docs/plans/2026-02-22-bubble-tracking-design.md` for bubble datasets, focusing on births, splits, merges, deaths, per-frame area, and per-track lineage metadata.

## Current Phase
Phase 17

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

### Phase 9: Stateful Debounce In Tracking Loop
- [x] Replace post-hoc birth/death debounce with in-loop tentative-birth and missing-track grace handling
- [x] Preserve continuous IDs and masks through short flicker gaps
- [x] Keep split/merge transitions explicit for lineage metadata
- [x] Validate synthetic flicker, internal-gap, terminal-gap, and split/merge scenarios
- **Status:** complete

### Phase 10: Vanished Area Moving-Window Visualization
- [x] Add standalone plotting script for moving-window vanished area from true death events
- [x] Generate four plots (2s/5s windows x seconds/frames axes)
- [x] Validate with synthetic event stream and output artifacts
- **Status:** complete

### Phase 11: Publish Velocity/Lineage Exports
- [x] Update max-velocity summary export for case1-3 publish run directories
- [x] Include lifecycle lineage fields (`birth_reason`, `death_reason`, parents/children IDs) in summary CSV
- [x] Validate publish-run metric generation and plotting command flow
- **Status:** complete

### Phase 12: Publish Vanished-Area Log-Scale Plots
- [x] Extend vanished-area plotting script to emit additional log-y variants
- [x] Generate log-scale vanished-area plots for case1/2/3 Air1 publish outputs
- [x] Verify all expected `_logy` files are present in each publish directory
- **Status:** complete

### Phase 13: Framewise Injection/Vanish Area CSV Export
- [x] Add standalone exporter for per-frame injection-birth and vanished-death area totals
- [x] Convert per-frame totals from px^2 to mm^2 using each run's own `mm_per_pixel`
- [x] Generate CSVs for case1/2/3 air1 publish directories and verify frame-aligned row counts
- **Status:** complete

### Phase 14: Three Escaping Situation Subcount CSV Export
- [x] Read `journal1_three_escaping_cases_subcount_outputCSV_requirements.md` and implement all 16 required columns
- [x] Use downscaled conversion key from `pixel_mm_conversion_standard.md` for px^2 <-> mm^2 and 3.14 mm^2 threshold conversion
- [x] Generate:
  - `outputs/case1_air1_time_series_subcount_three_escaping_situation.csv`
  - `outputs/case2_air1_time_series_subcount_three_escaping_situation.csv`
  - `outputs/case3_air1_time_series_subcount_three_escaping_situation.csv`
- [x] Validate nonzero split/merge/death rows and parent-child formatting
- **Status:** complete

### Phase 15: Threshold-Variant Three Escaping Subcount CSV Export
- [x] Keep original three CSV outputs unchanged and export three additional threshold-variant files
- [x] Apply threshold updates:
  - split/merge ratio threshold: `20%` -> `50%`
  - split/merge absolute threshold: `3.14 mm^2` -> `5.0 mm^2`
- [x] Generate:
  - `outputs/case1_air1_time_series_subcount_three_escaping_situation_threshold_50%_and_5mm2.csv`
  - `outputs/case2_air1_time_series_subcount_three_escaping_situation_threshold_50%_and_5mm2.csv`
  - `outputs/case3_air1_time_series_subcount_three_escaping_situation_threshold_50%_and_5mm2.csv`
- [x] Verify headers, row counts, and file presence for all three variant outputs
- **Status:** complete

### Phase 16: Case2 Air1 Multi-Threshold Subcount CSV Export (5 mm^2)
- [x] Generate three new case2-only threshold variants at `40%`, `60%`, and `80%`
- [x] Keep existing `50%` case2 output and all prior base outputs untouched
- [x] Produce:
  - `outputs/case2_air1_time_series_subcount_three_escaping_situation_threshold_40%_and_5mm2.csv`
  - `outputs/case2_air1_time_series_subcount_three_escaping_situation_threshold_60%_and_5mm2.csv`
  - `outputs/case2_air1_time_series_subcount_three_escaping_situation_threshold_80%_and_5mm2.csv`
- [x] Verify each file has the expected 16-column schema and frame-aligned row count
- **Status:** complete

### Phase 17: Case2 Air1 Additional Threshold CSV Export (65%/70%, 5 mm^2)
- [x] Generate two new case2-only threshold variants at `65%` and `70%`
- [x] Keep existing `40/50/60/80%` case2 outputs untouched
- [x] Produce:
  - `outputs/case2_air1_time_series_subcount_three_escaping_situation_threshold_65%_and_5mm2.csv`
  - `outputs/case2_air1_time_series_subcount_three_escaping_situation_threshold_70%_and_5mm2.csv`
- [x] Verify each file has the expected 16-column schema and frame-aligned row count
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

## Session: 2026-04-18 (Case1 Publish Label Relayout, Frames 301-311)

### Goal
Create 11 modified still frames for original frame numbers `301..311` (inclusive), with bubble ID labels moved outside each bubble bounding box to below-right, black text on transparent background, and label scale multiplied by `60/36`.

### Current Phase
Phase 3

### Phases

### Phase 1: Source Mapping + Rendering Strategy
- [x] Confirm output target directory path and empty state
- [x] Confirm original-to-relabeled frame mapping for case1 publish data
- [x] Confirm label/bbox drawing implementation details in visualization code
- **Status:** complete

### Phase 2: Generate Requested Frames
- [x] Render frames for original numbers `301..311`
- [x] Apply requested label placement/style/size changes
- [x] Include original frame number in each output filename
- **Status:** complete

### Phase 3: Verification
- [x] Verify exactly 11 files were produced in the target folder
- [x] Verify output filenames include original frame numbers `301..311`
- **Status:** complete

### Decisions Made
| Decision | Rationale |
|----------|-----------|
| Use `outputs/case1_air1_publish/tracks.json` + relabeled `frames_0p5` + `masks/` to render stills | Ensures frame/object alignment with `case1_air1_publish` results |
| Map original to relabeled via `frame_map.csv` | Avoids hardcoding assumptions and keeps traceability to source frame numbers |
| Use scaled font settings from original overlay defaults (`0.9, 2` -> `1.5, 3`) | Meets `60/36` size multiplier while preserving visual consistency |

### Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| Incorrect shell form for session catchup command (`(command -v ...)`) | 1 | Re-ran with command substitution: `$(command -v ...)` |
| `ModuleNotFoundError: No module named 'cv2'` using system `python3` | 1 | Re-ran generation with project interpreter `./venv/bin/python` (OpenCV installed) |

## Session: 2026-04-18 (Follow-up Label Position Correction)

### Goal
Adjust the 11 generated frames so labels are still outside near the right-bottom corner, but directly below the corner instead of below-right.

### Phases

### Phase 1: Requirement Confirmation
- [x] Confirm requested shift: remove rightward offset, keep below-corner placement
- **Status:** complete

### Phase 2: Regenerate Frames In-Place
- [x] Re-render original frame numbers `301..311`
- [x] Keep style constraints unchanged (black text, transparent background, scale `60/36`)
- [x] Move label anchor from `x_max + offset` to `x_max`
- **Status:** complete

### Phase 3: Verify Outputs
- [x] Confirm exactly 11 files present in target directory
- [x] Confirm files overwritten with corrected placement
- [x] Confirm no label clamp was required (`clamped_labels=0`)
- **Status:** complete

### Decisions Made
| Decision | Rationale |
|----------|-----------|
| Use direct x-anchor at bbox right edge (`tx = x_max`) | Matches user-requested “just below right-bottom corner” instead of below-right |
| Keep filenames unchanged | Preserves traceability and avoids producing duplicate variants |

## Session: 2026-04-18 (Coordinate Handoff for Manual Rerun)

### Goal
Provide precise relative and absolute label coordinates for original frames `301..311` so user can manually edit placement and rerun.

### Phases

### Phase 1: Export Coordinates
- [x] Extract per-object bbox right-bottom corners for frames `301..311`
- [x] Compute current and alternative relative offsets
- [x] Save handoff CSV in the target output folder
- **Status:** complete

### Decisions Made
| Decision | Rationale |
|----------|-----------|
| Export both relative offsets and absolute `cv2.putText` anchors | Lets user choose manual placement strategy without re-deriving formulas |
| Include two “WANT” options (right-edge aligned, centered-under-corner) | Clarifies ambiguity in visual intent and avoids another rerun loop |

## Session: 2026-04-18 (Auto Placement Retry with Negative Offset)

### Goal
Apply user-requested automatic label placement update by using a negative horizontal offset (`-30`) instead of manual coordinate editing.

### Phases

### Phase 1: Regenerate with Offset
- [x] Re-render original frames `301..311` in-place
- [x] Apply `tx = x_max - 30` with prior vertical anchor retained
- [x] Keep style unchanged (black text, transparent background, scale `60/36`)
- **Status:** complete

### Phase 2: Verify
- [x] Confirm all 11 files regenerated
- [x] Confirm no coordinate clamping (`clamped_labels=0`)
- **Status:** complete

### Decisions Made
| Decision | Rationale |
|----------|-----------|
| Use `x_offset = -30` as requested | Directly matches user instruction for first retry |
| Keep output filenames unchanged and overwrite in-place | Minimizes churn and keeps the same review location |

## Session: 2026-04-19 (Bubble Numbering Font Change)

### Goal
Change bubble numbering typography in the 11 target frames to a Times-style font.

### Phases

### Phase 1: Font Availability Check
- [x] Search system/user font directories for Times New Roman font files
- [x] Confirm Times New Roman is not installed on this machine
- [x] Select closest installed Times-compatible fallback (`Nimbus Roman`)
- **Status:** complete

### Phase 2: Regenerate Frames with New Font
- [x] Re-render all 11 frames in-place
- [x] Keep placement/color/size intent unchanged
- [x] Draw labels with PIL TrueType renderer using `NimbusRoman-Regular.otf`
- **Status:** complete

### Phase 3: Verification
- [x] Verify all 11 files regenerated
- [x] Verify no text clamping (`clamped_labels=0`)
- **Status:** complete

### Decisions Made
| Decision | Rationale |
|----------|-----------|
| Use `NimbusRoman-Regular.otf` | Closest available Times-family serif on this system when Times New Roman is unavailable |
| Preserve `x_offset=-90` from latest accepted placement | Avoid changing layout while changing font only |

## Session: 2026-04-19 (Single-Frame Manual Mask Workflow for Frame 0304/0234)

### Goal
Support manual editing of relabeled frame 0234 mask and regenerate the annotated image with stable bubble IDs and recomputed geometry from edited mask (notably #49).

### Phases

### Phase 1: Prepare Manual-Edit Inputs
- [x] Identify source mask for relabeled frame 0234 (`masks/000233.png`)
- [x] Copy mask into user manual folder as `000233_manual_mask.png`
- **Status:** complete

### Phase 2: Build Rerender Path
- [x] Add single-frame rerender script that uses edited ID mask directly
- [x] Recompute bbox and centroid from edited mask for each ID (including #49)
- [x] Preserve numbering by reading IDs from mask values (no recount)
- **Status:** complete

### Phase 3: Verify Baseline Run
- [x] Run script once against copied mask to validate output path and rendering
- [x] Confirm output written in manual folder with expected filename
- **Status:** complete

### Decisions Made
| Decision | Rationale |
|----------|-----------|
| Recompute geometry from edited mask pixels instead of tracks.json frame data | Guarantees #49 bbox/centroid reflect manual mask edits |
| Keep IDs from mask values | Preserves numbering exactly without re-indexing |
| Provide reusable script for rerun after each manual tweak | Minimizes turnaround for iterative manual edits |

## Session: 2026-04-19 (Implemented Full-Frame Binary Matrix Round-Trip for 0304/0234)

### Goal
Implement export/import round-trip between ID mask PNG and human-editable full-frame binary CSV matrices for IDs #48 and #49, then regenerate frame with ID-preserving geometry recomputation.

### Phases

### Phase 1: Script Implementation
- [x] Added `scripts/matrix_roundtrip_case1_0234.py` with `export` and `import` subcommands
- [x] Implemented raw full-frame CSV export for `#48` and `#49` (no headers)
- [x] Implemented metadata JSON export with shape/dtype/ID mapping
- [x] Implemented import rebuild preserving non-48/49 IDs and overlap policy (`#49` wins)
- **Status:** complete

### Phase 2: Validation
- [x] Verified export files exist and matrix shape is exactly `1836 x 2748`
- [x] Ran synthetic import test with #49 vertical extension and #48 hole fill
- [x] Verified IDs unchanged and #49 geometry updated (`y_max` and centroid y increased)
- [x] Verified render from edited mask succeeds via existing renderer script
- **Status:** complete

### Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| False-positive protected-pixel conflicts during import validation | 1 | Replaced boolean conflict logic with `np.isin` + `np.logical_*`; re-ran tests successfully |

### Decisions Made
| Decision | Rationale |
|----------|-----------|
| Keep matrix representation as two full-frame binary CSVs (`id48`, `id49`) | Matches user requirement for human-editable 2D binary matrices while preserving IDs |
| Keep existing renderer script for final image output | Preserves accepted annotation styling and avoids duplicate rendering paths |

## Session: 2026-04-19 (Compact Editable Matrix Workflow for 0304/0234)

### Goal
Replace full-frame manual editing with a compact workflow: auto-fill enclosed `#48` voids, export a 10%-expanded cropped `#49` matrix plus crop origin metadata, and regenerate the annotated frame with existing style settings.

### Phases

### Phase 1: Prepare Compact Workflow
- [x] Add script support to fill enclosed background holes inside `#48`
- [x] Add script support to export 10%-expanded cropped `#49` binary matrix and crop metadata
- [x] Keep outputs in the same manual folder for direct user editing
- **Status:** complete

### Phase 2: Generate Artifacts
- [x] Produce filled-mask output for user review
- [x] Produce compact `#49` CSV and crop metadata (row/col origin for later merge)
- [x] Regenerate the annotated frame from the updated mask with existing style (`x_offset=-70`, black text, Times-style serif fallback)
- **Status:** complete

### Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| `argparse` help formatting error (`badly formed help string`) caused by `%` in help text | 1 | Escaped `%` as `%%` in subcommand help string and reran successfully |

### Decisions Made
| Decision | Rationale |
|----------|-----------|
| Fill `#48` holes only on background pixels (`ID 0`) | Prevents accidental overwrite of other tracked IDs while satisfying "foreground surrounds background" void filling |
| Export `#49` as bbox-based crop with 10% margin each side | Keeps editable matrix small enough for limited CSV viewers while preserving local context |

## Session: 2026-04-19 (Apply Edited #49 Crop v3 and Rerender 0304/0234)

### Goal
Use the manually edited cropped CSV (`000233_manual_mask_id49_crop_10pct_after_manual_change_v3.csv`) as the new `#49` region, merge it back into the full-frame mask, and regenerate the annotated frame.

### Phases

### Phase 1: Merge Edited Crop to Full Mask
- [x] Read crop bounds from `000233_manual_mask_id49_crop_10pct_meta.json`
- [x] Validate edited CSV shape and binary values
- [x] Merge edited `#49` crop into full-frame mask while preserving other IDs
- **Status:** complete

### Phase 2: Regenerate Frame
- [x] Render frame from merged mask with existing style (`x_offset=-70`)
- [x] Confirm output image is written and labels are not clamped
- **Status:** complete

### Decisions Made
| Decision | Rationale |
|----------|-----------|
| Merge into the `#48`-void-filled base mask (`000233_manual_mask_id48_void_filled.png`) | Preserves prior accepted `#48` hole-fill edits while updating only `#49` from manual CSV |
| Enforce protected-ID check during merge (`allowed: 0,48,49`) | Prevents accidental paint-over on unrelated object IDs when applying edited crop |

## Session: 2026-04-19 (Apply Edited #49 Crop v4 and Rerender 0304/0234)

### Goal
Use the manually edited cropped CSV (`000233_manual_mask_id49_crop_10pct_after_manual_change_v4.csv`) as the new `#49` region, merge it into the full-frame mask, and regenerate the annotated frame.

### Phases

### Phase 1: Merge Edited Crop to Full Mask
- [x] Read crop bounds from `000233_manual_mask_id49_crop_10pct_meta.json`
- [x] Validate edited CSV shape and binary values
- [x] Merge edited `#49` crop into full-frame mask while preserving other IDs
- **Status:** complete

### Phase 2: Regenerate Frame
- [x] Render frame from merged mask with existing style (`x_offset=-70`)
- [x] Confirm output image is written and labels are not clamped
- **Status:** complete

### Decisions Made
| Decision | Rationale |
|----------|-----------|
| Keep the same merge base (`000233_manual_mask_id48_void_filled.png`) for v4 | Ensures only the requested `#49` crop revision changed between v3 and v4 runs |
| Write a distinct merged mask artifact for v4 | Preserves traceability across manual-edit iterations (`v3` vs `v4`) |

## Session: 2026-04-19 (Case2 Air1 SAM2 Self-Prompted Chunking @160)

### Goal
Run a chunked SAM2 self-reprompt experiment on `case2_air1` with chunk interval 160, then compare outputs against existing CV-keyframe reprompt run at `outputs/case2_air1_publish`.

### Phases

### Phase 1: Confirm Existing Pipeline and Baseline
- [x] Verify CV-keyframe baseline run artifacts and parameters
- [x] Identify first prompt frame/index and keyframe cadence from baseline config
- **Status:** complete

### Phase 2: Implement Self-Reprompt Runner
- [ ] Add script to run SAM2 in chunks where next chunk prompt mask comes from previous chunk SAM2 output
- [ ] Support interval=160 and preserve per-object IDs over chunks
- [ ] Save masks, tracks, area timeseries, events, and run config metadata
- **Status:** in_progress

### Phase 3: Execute Experiment and Compare
- [ ] Run self-reprompt script on `data/bubbles/case2_air1_publish/frames_0p5`
- [ ] Compute side-by-side summary vs `outputs/case2_air1_publish` (objects/events/area dynamics)
- [ ] Report comparison and practical interpretation
- **Status:** pending

### Decisions Made
| Decision | Rationale |
|----------|-----------|
| Use SAM2 mask prompts (not point prompts) for chunk handoff | Matches requested "self-prompting from SAM2 masks" behavior directly |
| Seed first chunk from baseline CV keyframe prompt mask/frame | Keeps start condition aligned for fair comparison |

### Phase Update (2026-04-19)
- Phase 2 implementation completed:
  - Added `scripts/run_sam2_self_prompted_flow.py` for chunked SAM2 self-mask re-prompting.
  - Script compiles (`python -m py_compile`).
- Phase 3 execution status:
  - Sandbox run blocked for CUDA (`No CUDA GPUs are available`).
  - Escalated GPU run request was rejected in-session.
  - CPU fallback start confirmed but full run not completed here due runtime constraints.
- **Status:** awaiting manual GPU execution to complete comparison.

### Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| `ModuleNotFoundError: No module named 'cv2'` when using system `python` | 1 | Switched to project env interpreter: `/home/aaron/Downloads/bubble_tracking/venv311/bin/python` |
| `RuntimeError: No CUDA GPUs are available` in sandbox | 2 | Requested escalated GPU execution; request was rejected, so manual run command prepared for user |
