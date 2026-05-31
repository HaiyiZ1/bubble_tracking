"""CLI entry for Phase 2.

Usage:
    python -m phase2_track \\
        --run-dir outputs/case1_air1_publish \\
        --sam2-model {tiny,small,base_plus,large} \\
        [--device cuda] \\
        [--no-sam2-offload-video-to-cpu]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from .schemas import (
    PREPROCESS_MODE,
    PREPROCESS_RAW_SIGMA,
    DEBOUNCE_BIRTH_FRAMES,
    DEBOUNCE_DEATH_FRAMES,
    SAM2_FILL_HOLE_AREA,
    SAM2_MODELS,
)
from bubble_common.schemas import (
    PreprocessConfig,
    SegmentationConfig,
    DebounceConfig,
    TrackingConfig,
    ObjectPrompt,
    PointPrompt,
    PromptSpec,
)

# Video FPS for output overlays (not a user-facing flag in v1).
_VIDEO_FPS = 10.0


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser("phase2_track")
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--sam2-model", required=True, choices=tuple(SAM2_MODELS))
    p.add_argument("--device", default=None, help="cuda | cpu (auto-detect by default)")

    # offload defaults True; --no-sam2-offload-video-to-cpu disables.
    p.add_argument(
        "--sam2-offload-video-to-cpu", dest="sam2_offload_video_to_cpu",
        action="store_true", default=True,
    )
    p.add_argument(
        "--no-sam2-offload-video-to-cpu", dest="sam2_offload_video_to_cpu",
        action="store_false",
    )
    return p.parse_args()


def _load_cv_keyframe_prompts(phase1_dir: Path) -> tuple:
    """Read cv_keyframe_prompts/ written by phase1.

    Returns (PromptSpec, list_of_keyframe_frame_indices).
    """
    prompts_dir = phase1_dir / "cv_keyframe_prompts"
    if not prompts_dir.exists():
        raise FileNotFoundError(
            f"{prompts_dir} not found. Run phase1 with --cv-keyframe-start-idx first."
        )
    meta = json.loads((prompts_dir / "prompts.json").read_text())
    prompts = []
    keyframe_frame_indices = []
    for kf in meta["keyframes"]:
        frame_idx = int(kf["frame_idx"])
        keyframe_frame_indices.append(frame_idx)
        mask_png = prompts_dir / f"{frame_idx:06d}.png"
        instance_mask = cv2.imread(str(mask_png), cv2.IMREAD_UNCHANGED)
        if instance_mask is None:
            raise ValueError(f"Failed to read {mask_png}")
        for obj in kf["objects"]:
            object_id = int(obj["object_id"])
            obj_mask = (instance_mask == object_id).astype(np.uint8)
            negative_points = [
                PointPrompt(x=float(x), y=float(y), label=0)
                for x, y in obj.get("negative_points", [])
            ]
            prompts.append(ObjectPrompt(
                object_id=object_id,
                frame_idx=frame_idx,
                mask=obj_mask,
                points=negative_points if negative_points else [],
            ))
    return PromptSpec(prompts=prompts), keyframe_frame_indices


def main() -> None:
    from bubble_common.run_config import load as load_run_config, mark_phase_done

    # Phase 2's own modules (lazy import to keep startup fast for --help).
    from .sam2 import (
        SAM2Config,
        SAM2Runner,
        sorted_frame_paths,
        save_mask,
    )
    from .flow import track_frames, load_roi_json
    from .events import save_area_timeseries_csv, save_tracks
    from .visualize import create_tracking_video, create_sam2_overlay_video

    args = _parse()

    # --- Determinism: make SAM2 + cv2 + numpy reproducible across runs. ---
    # Without these, GPU SAM2 propagation drifts between identical-input runs
    # (cuDNN/cuBLAS heuristics), and multi-threaded OpenCV optical-flow can
    # produce slightly different results in track_frames.
    # NOTE: PYTHONHASHSEED must be set in the calling shell environment
    # (CUBLAS_WORKSPACE_CONFIG too), not here — Python reads them at startup
    # before main() runs. We set them anyway as a safety net for child procs.
    import os as _os
    import cv2 as _cv2
    import numpy as _np
    import torch as _torch
    _os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    _os.environ.setdefault("PYTHONHASHSEED", "0")
    _cv2.setNumThreads(1)
    _np.random.seed(0)
    _torch.manual_seed(0)
    _torch.cuda.manual_seed_all(0)
    _torch.backends.cudnn.deterministic = True
    _torch.backends.cudnn.benchmark = False

    # --- Load phase1 run config ---
    cfg = load_run_config(args.run_dir)
    if not cfg.phase1_done:
        sys.exit(f"{args.run_dir}/run_config.json: phase1_done=False. Run phase1 first.")
    if cfg.cv_keyframes is None:
        sys.exit(
            f"{args.run_dir}/run_config.json: cv_keyframes is missing. "
            "Re-run phase1 with --cv-keyframe-start-idx to build CV-keyframe prompts."
        )

    phase2_dir = args.run_dir / "phase2"
    phase2_dir.mkdir(parents=True, exist_ok=True)

    # --- Resolve paths from run_config ---
    frames_dir: str = cfg.frames
    reference_path: str = cfg.reference
    fps: float = cfg.fps
    mm_per_pixel: float = cfg.mm_per_pixel

    # --- Load ROI and frames ---
    roi = load_roi_json(cfg.roi_json) if cfg.roi_json else None

    frame_paths = sorted_frame_paths(frames_dir)
    if not frame_paths:
        sys.exit(f"No frames found in {frames_dir}")

    reference_full = cv2.imread(reference_path)
    if reference_full is None:
        sys.exit(f"Failed to read reference image: {reference_path}")

    # Build ROI mask from roi_mask PNG (generated by phase1).
    roi_mask = None
    if cfg.roi_mask:
        roi_mask_img = cv2.imread(cfg.roi_mask, cv2.IMREAD_GRAYSCALE)
        if roi_mask_img is None:
            sys.exit(f"Failed to read roi_mask: {cfg.roi_mask}")
        # Ensure roi_mask matches frame shape.
        first_frame = cv2.imread(frame_paths[0])
        if first_frame is None:
            sys.exit(f"Failed to read first frame: {frame_paths[0]}")
        if roi_mask_img.shape != first_frame.shape[:2]:
            roi_mask_img = cv2.resize(
                roi_mask_img,
                (first_frame.shape[1], first_frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        roi_mask = roi_mask_img

    # --- Resolve SAM2 model paths from the registry ---
    ckpt, model_cfg_name = SAM2_MODELS[args.sam2_model]
    sam2_masks_dir = str(phase2_dir / "sam2_masks")

    # --- Step 3: Load CV-keyframe prompts from disk (built by phase1) ---
    phase1_dir = args.run_dir / "phase1"
    print(f"[phase2] Loading CV-keyframe prompts from {phase1_dir / 'cv_keyframe_prompts'} ...")
    prompt_spec, keyframe_frame_indices = _load_cv_keyframe_prompts(phase1_dir)
    print(f"[phase2] {len(prompt_spec.prompts)} CV-keyframe prompts loaded.")

    # --- Step 4: Run SAM2 propagation -> phase2_dir/sam2_masks/ ---
    print(f"[phase2] Running SAM2 propagation (model={args.sam2_model}) ...")
    sam2_cfg = SAM2Config(
        checkpoint_path=ckpt,
        model_cfg=model_cfg_name,
        device=args.device if args.device is not None else (
            "cuda" if _torch.cuda.is_available() else "cpu"
        ),
        fill_hole_area=SAM2_FILL_HOLE_AREA,
        offload_video_to_cpu=args.sam2_offload_video_to_cpu,
    )
    runner = SAM2Runner(sam2_cfg)
    _tracking_result_sam2, masks_by_frame = runner.run(
        frames_dir=frames_dir,
        prompts=prompt_spec,
        roi_mask=roi_mask,
    )

    # Save SAM2 masks to disk.
    # masks_by_frame uses 0-based SAM2 internal indices; remap to the
    # actual 1-based filename numbers from frame_paths (matching the oracle's
    # _save_masks_for_frames logic in scripts/run_sam2_prompted_flow.py).
    # Frames not reached by propagation receive an empty (all-zero) mask so
    # that track_frames sees a complete, contiguous set of mask files.
    Path(sam2_masks_dir).mkdir(parents=True, exist_ok=True)
    first_frame_img = cv2.imread(frame_paths[0])
    if first_frame_img is None:
        sys.exit(f"Failed to read first frame for empty-mask shape: {frame_paths[0]}")
    import numpy as _np
    empty_mask = _np.zeros(first_frame_img.shape[:2], dtype=_np.uint16)
    import re as _re
    _frame_num_re = _re.compile(r"(\d+)\.(jpg|jpeg|png)$", _re.IGNORECASE)
    for sam2_idx, fp in enumerate(frame_paths):
        m = _frame_num_re.search(fp)
        file_num = int(m.group(1)) if m else (sam2_idx + 1)
        mask = masks_by_frame.get(sam2_idx, empty_mask)
        save_mask(mask, Path(sam2_masks_dir) / f"{file_num:06d}.png")
    print(f"[phase2] SAM2 masks saved to {sam2_masks_dir}")

    # --- Step 5: Run tracking ---
    print("[phase2] Running flow tracking ...")
    tracking_cfg = TrackingConfig(
        fps=fps,
        mm_per_pixel=mm_per_pixel,
        use_flow=True,
        velocity_mode="centroid_delta",
        log_every=10,
        debounce=DebounceConfig(
            birth_confirm_frames=max(1, int(DEBOUNCE_BIRTH_FRAMES)),
            death_confirm_frames=max(1, int(DEBOUNCE_DEATH_FRAMES)),
        ),
        preprocess=PreprocessConfig(
            mode=PREPROCESS_MODE,
            raw_sigma=PREPROCESS_RAW_SIGMA,
        ),
        segment=SegmentationConfig(min_area=cfg.cv_keyframes.get("min_area", 50)),
    )
    outputs = track_frames(
        frames_dir=frames_dir,
        reference_path=reference_path,
        output_dir=str(phase2_dir),
        masks_dir=sam2_masks_dir,
        roi=roi,
        config=tracking_cfg,
    )
    print(
        f"[phase2] Tracking done. "
        f"Objects={outputs['result'].num_objects} "
        f"Events={outputs['event_summary']}"
    )

    # --- Step 6: area_timeseries already written by track_frames ---
    # track_frames writes area_timeseries.csv and area_timeseries.json itself.

    # --- Step 8: SAM2 overlay video ---
    sam2_overlay_path = phase2_dir / "sam2_overlay.mp4"
    print(f"[phase2] Creating SAM2 overlay video -> {sam2_overlay_path}")
    create_sam2_overlay_video(
        frames_dir=frames_dir,
        masks_dir=sam2_masks_dir,
        out_path=str(sam2_overlay_path),
        fps=_VIDEO_FPS,
        alpha=0.4,
    )

    # --- Step 9: Tracking overlay video ---
    tracking_overlay_path = phase2_dir / "tracking_overlay.mp4"
    print(f"[phase2] Creating tracking overlay video -> {tracking_overlay_path}")
    create_tracking_video(
        frames_dir=frames_dir,
        masks_dir=outputs["masks_dir"],
        result=outputs["result"],
        output_path=str(tracking_overlay_path),
        fps=_VIDEO_FPS,
        alpha=0.4,
    )

    # --- Step 10: Write phase2_config.json for audit ---
    phase2_config = {
        "run_dir": str(args.run_dir),
        "sam2_model": args.sam2_model,
        "sam2_checkpoint": ckpt,
        "sam2_model_cfg": model_cfg_name,
        "sam2_device": sam2_cfg.device,
        "sam2_fill_hole_area": SAM2_FILL_HOLE_AREA,
        "sam2_offload_video_to_cpu": args.sam2_offload_video_to_cpu,
        "cv_keyframe": {
            "num_prompts": len(prompt_spec.prompts),
            "keyframe_frame_indices": keyframe_frame_indices,
            "phase1_config": cfg.cv_keyframes,
        },
        "tracking": {
            "fps": fps,
            "mm_per_pixel": mm_per_pixel,
            "preprocess_mode": PREPROCESS_MODE,
            "preprocess_raw_sigma": PREPROCESS_RAW_SIGMA,
            "debounce_birth_frames": DEBOUNCE_BIRTH_FRAMES,
            "debounce_death_frames": DEBOUNCE_DEATH_FRAMES,
            "velocity_mode": "centroid_delta",
            "use_flow": True,
        },
    }
    phase2_config_path = phase2_dir / "phase2_config.json"
    phase2_config_path.write_text(json.dumps(phase2_config, indent=2) + "\n")
    print(f"[phase2] Config written to {phase2_config_path}")

    # --- Step 11: Mark phase2 done ---
    mark_phase_done(args.run_dir, 2)
    print("[phase2] Done. phase2_done=True written to run_config.json")


if __name__ == "__main__":
    main()
