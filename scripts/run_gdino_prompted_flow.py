#!/usr/bin/env python3
"""
Run flow tracking using SAM2 propagation seeded by Grounding DINO text-prompt keyframes.

Pipeline per keyframe (every --gdino-keyframe-interval frames):
  1. Run Grounding DINO on the frame with the given text prompt -> bounding boxes.
  2. Feed each accepted box into SAM2 image predictor -> per-box refined mask.
  3. Union the per-box masks into a single binary mask.
  4. Supply the union mask as one SAM2 video-predictor mask prompt (object_id=1).

Downstream (SAM2 video propagation, flow tracking, debounce, events, overlay)
is identical to run_sam2_prompted_flow.py --cv-keyframe-prompts so the three
runners (CV / self-reprompt / GDINO) are directly comparable.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bubble_tracking.flow_tracking import (  # noqa: E402
    DebounceConfig,
    PreprocessConfig,
    SegmentationConfig,
    TrackingConfig,
    load_roi_json,
    track_frames,
)
from bubble_tracking.frame_io import save_mask, sorted_frame_paths  # noqa: E402
from bubble_tracking.sam2_adapter import create_runner  # noqa: E402
from bubble_tracking.schemas import ObjectPrompt, PromptSpec  # noqa: E402
from bubble_tracking.visualization import create_tracking_video, draw_mask_overlay  # noqa: E402


def _frame_index_from_path(path: str) -> int:
    match = re.search(r"(\d+)\.(?:jpg|jpeg|png)$", Path(path).name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Failed to parse frame index from {path}")
    return int(match.group(1))


def _load_roi_mask(mask_path: str, shape: tuple) -> np.ndarray:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read ROI mask: {mask_path}")
    if mask.shape[:2] != shape[:2]:
        raise ValueError("ROI mask size does not match frame size")
    return mask


def _roi_mask_from_json(roi: dict, shape: tuple) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    mask[roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]] = 255
    return mask


def _should_log(idx: int, total: int, log_every: int) -> bool:
    if log_every <= 0:
        return False
    if idx == total - 1:
        return True
    return (idx % log_every) == 0


def _boxes_in_roi(boxes_xyxy: np.ndarray, roi_mask: Optional[np.ndarray]) -> np.ndarray:
    if roi_mask is None or boxes_xyxy.size == 0:
        return np.ones(len(boxes_xyxy), dtype=bool)
    h, w = roi_mask.shape[:2]
    cx = ((boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) * 0.5).astype(np.int32)
    cy = ((boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) * 0.5).astype(np.int32)
    cx = np.clip(cx, 0, w - 1)
    cy = np.clip(cy, 0, h - 1)
    return roi_mask[cy, cx] > 0


def _save_gdino_keyframe_debug(
    debug_dir: str,
    frame_full: np.ndarray,
    union_mask: np.ndarray,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    base_idx: int,
    prompt_idx: int,
) -> Dict[str, str]:
    os.makedirs(debug_dir, exist_ok=True)
    prefix = f"base_{base_idx:06d}_prompt_{prompt_idx:06d}"

    mask_u8 = (union_mask.astype(np.uint8) * 255)
    inst = np.zeros(union_mask.shape[:2], dtype=np.int32)
    inst[union_mask > 0] = 1
    overlay = draw_mask_overlay(frame_full, inst, colors={1: (0, 255, 0)}, alpha=0.45)
    for (x1, y1, x2, y2), sc in zip(boxes_xyxy.astype(int), scores):
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(
            overlay,
            f"{float(sc):.2f}",
            (int(x1), max(0, int(y1) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    overlay_path = os.path.join(debug_dir, f"{prefix}_overlay.jpg")
    mask_path = os.path.join(debug_dir, f"{prefix}_mask.png")
    cv2.imwrite(overlay_path, overlay)
    cv2.imwrite(mask_path, mask_u8)
    return {
        "overlay": os.path.abspath(overlay_path),
        "mask": os.path.abspath(mask_path),
    }


def _build_gdino_keyframe_mask_prompts(
    frame_paths: List[str],
    roi_mask: Optional[np.ndarray],
    sam2_checkpoint: str,
    sam2_model_cfg: str,
    device: str,
    text_prompt: str,
    gdino_model_id: str,
    box_threshold: float,
    text_threshold: float,
    keyframe_interval: int,
    keyframe_start_idx: int,
    min_area: int,
    debug_dir: Optional[str],
    log_every: int,
) -> Tuple[PromptSpec, Optional[int], List[dict]]:
    if keyframe_interval <= 0:
        raise ValueError("--gdino-keyframe-interval must be > 0")
    if keyframe_start_idx < 0:
        raise ValueError("--gdino-keyframe-start-idx must be >= 0")

    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except Exception as exc:
        raise ImportError(
            "GDINO keyframe prompts require `transformers`, `pillow`, and the SAM2 package."
        ) from exc

    gdino_processor = AutoProcessor.from_pretrained(gdino_model_id)
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(gdino_model_id).to(device)
    gdino_model.eval()

    sam2_base = build_sam2(sam2_model_cfg, sam2_checkpoint, device=device)
    sam2_img_predictor = SAM2ImagePredictor(sam2_base)

    # GDINO expects a lowercase phrase ending with "."
    text_norm = text_prompt.strip().lower()
    if not text_norm.endswith("."):
        text_norm = text_norm + "."

    prompts: List[ObjectPrompt] = []
    logs: List[dict] = []
    total = len(frame_paths)
    debug_root = os.path.abspath(debug_dir) if debug_dir else None
    if debug_root:
        os.makedirs(debug_root, exist_ok=True)

    use_cuda_autocast = str(device).lower().startswith("cuda")

    for base_idx in range(keyframe_start_idx, total, keyframe_interval):
        segment_end = min(base_idx + keyframe_interval, total)
        selected = None

        for search_idx in range(base_idx, segment_end):
            frame_full = cv2.imread(frame_paths[search_idx])
            if frame_full is None:
                continue
            frame_rgb = cv2.cvtColor(frame_full, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            target_size = (frame_full.shape[0], frame_full.shape[1])

            inputs = gdino_processor(
                images=pil_image,
                text=text_norm,
                return_tensors="pt",
            ).to(device)

            with torch.inference_mode():
                outputs = gdino_model(**inputs)

            results = gdino_processor.post_process_grounded_object_detection(
                outputs,
                input_ids=inputs.input_ids,
                threshold=float(box_threshold),
                text_threshold=float(text_threshold),
                target_sizes=[target_size],
            )
            res = results[0]
            boxes_xyxy = res["boxes"].detach().cpu().numpy().astype(np.float32)
            scores = res["scores"].detach().cpu().numpy().astype(np.float32)

            if boxes_xyxy.size == 0:
                continue

            in_roi = _boxes_in_roi(boxes_xyxy, roi_mask)
            boxes_xyxy = boxes_xyxy[in_roi]
            scores = scores[in_roi]
            if boxes_xyxy.size == 0:
                continue

            # Box -> per-box refined mask via SAM2 image predictor
            sam2_img_predictor.set_image(frame_rgb)
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_cuda_autocast
                else _NullCtx()
            )
            union = np.zeros(frame_full.shape[:2], dtype=bool)
            with torch.inference_mode(), autocast_ctx:
                for box in boxes_xyxy:
                    masks_out, _, _ = sam2_img_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=box[None, :],
                        multimask_output=False,
                    )
                    # SAM2ImagePredictor returns numpy; shape is (1, H, W) or (H, W)
                    m = np.asarray(masks_out)
                    if m.ndim == 3:
                        m = m[0]
                    union |= m > 0

            if roi_mask is not None:
                union &= (roi_mask > 0)

            area = int(union.sum())
            if area < min_area:
                continue

            debug_paths = {}
            if debug_root:
                debug_paths = _save_gdino_keyframe_debug(
                    debug_dir=debug_root,
                    frame_full=frame_full,
                    union_mask=union,
                    boxes_xyxy=boxes_xyxy,
                    scores=scores,
                    base_idx=base_idx,
                    prompt_idx=search_idx,
                )

            selected = {
                "base_idx": int(base_idx),
                "prompt_idx": int(search_idx),
                "segment_end": int(segment_end),
                "num_boxes": int(len(boxes_xyxy)),
                "min_score": float(scores.min()) if scores.size else None,
                "max_score": float(scores.max()) if scores.size else None,
                "area": int(area),
                "mask": union.astype(np.uint8),
                "debug_paths": debug_paths,
            }
            break

        if selected is None:
            logs.append({
                "base_idx": int(base_idx),
                "segment_end": int(segment_end),
                "status": "no_gdino_mask",
            })
        else:
            prompt_idx = int(selected["prompt_idx"])
            prompts.append(
                ObjectPrompt(
                    object_id=1,
                    frame_idx=prompt_idx,
                    mask=selected["mask"],
                )
            )
            log_entry = {
                "base_idx": int(selected["base_idx"]),
                "prompt_idx": int(prompt_idx),
                "segment_end": int(selected["segment_end"]),
                "area": int(selected["area"]),
                "num_boxes": int(selected["num_boxes"]),
                "min_score": selected["min_score"],
                "max_score": selected["max_score"],
                "status": "ok",
            }
            if selected.get("debug_paths"):
                log_entry["debug"] = selected["debug_paths"]
            logs.append(log_entry)

        if _should_log(base_idx, total, log_every):
            print(f"[progress] stage=gdino_keyframe_prompt frame {base_idx + 1}/{total}")

    # Free GPU memory held by the image predictor + GDINO before video propagation starts
    del sam2_img_predictor
    del sam2_base
    del gdino_model
    del gdino_processor
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
    except Exception:
        pass

    if not prompts:
        raise RuntimeError(
            "No GDINO keyframe prompts found. Try lowering --gdino-box-threshold or "
            "--gdino-min-area, or revising --gdino-text-prompt."
        )

    prompts.sort(key=lambda p: int(p.frame_idx))
    first_idx = int(prompts[0].frame_idx)
    return PromptSpec(prompts=prompts), first_idx, logs


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *_):
        return False


def _run_sam2_propagation(
    frames_dir: str,
    prompts: PromptSpec,
    roi_mask: Optional[np.ndarray],
    checkpoint: str,
    model_cfg: str,
    device: str,
    fill_hole_area: int = 8,
) -> Dict[int, np.ndarray]:
    import torch

    runner = create_runner(
        checkpoint_path=checkpoint,
        model_cfg=model_cfg,
        device=device,
        fill_hole_area=fill_hole_area,
    )
    try:
        _, masks_by_frame = runner.run(
            frames_dir=frames_dir,
            prompts=prompts,
            output_dir=None,
            roi_mask=roi_mask,
        )
    except torch.OutOfMemoryError as exc:
        print(
            "[sam2] OOM during propagation; consider lower resolution or smaller model.",
            file=sys.stderr,
        )
        raise exc
    return masks_by_frame


def _save_masks_for_frames(
    frame_paths: List[str],
    masks_by_frame: Dict[int, np.ndarray],
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    frame_indices = [_frame_index_from_path(p) for p in frame_paths]
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        raise ValueError(f"Failed to read frame: {frame_paths[0]}")
    empty = np.zeros(first_frame.shape[:2], dtype=np.uint8)
    for idx, frame_idx in enumerate(frame_indices):
        mask = masks_by_frame.get(idx, empty)
        save_mask(mask, os.path.join(output_dir, f"{frame_idx:06d}.png"))


def main() -> None:
    parser = argparse.ArgumentParser(description="GDINO-prompted SAM2 flow tracking")
    parser.add_argument("--frames", required=True, help="Frames directory")
    parser.add_argument("--reference", required=True, help="Zero-bubble reference image")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--roi", default=None, help="ROI JSON path (x1,y1,x2,y2)")
    parser.add_argument("--roi-mask", default=None, help="ROI mask PNG (optional polygon mask)")
    parser.add_argument("--sam2-checkpoint", default="checkpoints/sam2_hiera_tiny.pt", help="SAM2 checkpoint path")
    parser.add_argument("--sam2-model-cfg", default="sam2_hiera_t.yaml", help="SAM2 model cfg name")
    parser.add_argument("--sam2-device", default="cuda", help="SAM2/GDINO device (cuda or cpu)")
    parser.add_argument(
        "--sam2-fill-hole-area",
        type=int,
        default=8,
        help="SAM2 background-hole fill area in low-res masks (upstream default 8). Set to 0 to preserve genuine holes.",
    )
    parser.add_argument(
        "--gdino-model-id",
        default="IDEA-Research/grounding-dino-tiny",
        help="HuggingFace model ID for Grounding DINO",
    )
    parser.add_argument(
        "--gdino-text-prompt",
        default="round bubble.",
        help="Text prompt for GDINO (lowercase, trailing period)",
    )
    parser.add_argument(
        "--gdino-box-threshold",
        type=float,
        default=0.25,
        help="GDINO box confidence threshold",
    )
    parser.add_argument(
        "--gdino-text-threshold",
        type=float,
        default=0.25,
        help="GDINO text-match threshold",
    )
    parser.add_argument(
        "--gdino-keyframe-interval",
        type=int,
        default=160,
        help="Keyframe interval for GDINO prompts",
    )
    parser.add_argument(
        "--gdino-keyframe-start-idx",
        type=int,
        default=0,
        help="Start frame index for GDINO prompts",
    )
    parser.add_argument(
        "--gdino-min-area",
        type=int,
        default=50,
        help="Minimum total area (px) of union mask to accept a keyframe",
    )
    parser.add_argument(
        "--gdino-keyframe-debug-dir",
        default=None,
        help="Optional directory for GDINO keyframe overlay and mask artifacts",
    )
    parser.add_argument("--min-area", type=int, default=500, help="Min component area for downstream tracking")
    parser.add_argument("--fps", type=float, default=4.0, help="Frames per second")
    parser.add_argument("--mm-per-pixel", type=float, default=None, help="Scale in mm per pixel")
    parser.add_argument(
        "--preprocess-mode",
        choices=["raw", "default", "automatrix"],
        default="raw",
        help="Preprocess mode for optical flow/tracking stage",
    )
    parser.add_argument(
        "--preprocess-raw-sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma used in raw preprocess mode (0 disables blur)",
    )
    parser.add_argument(
        "--use-flow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable optical flow for association and velocity estimation",
    )
    parser.add_argument(
        "--velocity-mode",
        choices=["flow_median", "centroid_delta"],
        default="flow_median",
        help="Per-object velocity source",
    )
    parser.add_argument("--debounce-birth-frames", type=int, default=1)
    parser.add_argument("--debounce-death-frames", type=int, default=1)
    parser.add_argument("--debounce-relink-max-centroid-dist-px", type=float, default=20.0)
    parser.add_argument("--debounce-relink-min-area-ratio", type=float, default=0.35)
    parser.add_argument("--debounce-relink-max-area-ratio", type=float, default=2.8)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--no-video", action="store_true", help="Skip overlay video")
    parser.add_argument("--video-fps", type=float, default=10.0, help="FPS for overlay video")
    parser.add_argument("--overlay-dir", default=None, help="Optional directory to save overlay frames")
    args = parser.parse_args()

    roi = load_roi_json(args.roi) if args.roi else None
    frame_paths = sorted_frame_paths(args.frames)
    if not frame_paths:
        raise ValueError(f"No frames found in {args.frames}")

    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        raise ValueError("Failed to read first frame")

    roi_mask = None
    if args.roi_mask:
        roi_mask = _load_roi_mask(args.roi_mask, first_frame.shape)
    elif roi is not None:
        roi_mask = _roi_mask_from_json(roi, first_frame.shape)

    sam2_masks_dir = os.path.join(args.output, "sam2_masks")
    os.makedirs(args.output, exist_ok=True)

    prompt_spec, prompt_frame_idx, gdino_logs = _build_gdino_keyframe_mask_prompts(
        frame_paths=frame_paths,
        roi_mask=roi_mask,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_model_cfg=args.sam2_model_cfg,
        device=args.sam2_device,
        text_prompt=args.gdino_text_prompt,
        gdino_model_id=args.gdino_model_id,
        box_threshold=args.gdino_box_threshold,
        text_threshold=args.gdino_text_threshold,
        keyframe_interval=args.gdino_keyframe_interval,
        keyframe_start_idx=args.gdino_keyframe_start_idx,
        min_area=args.gdino_min_area,
        debug_dir=args.gdino_keyframe_debug_dir,
        log_every=args.log_every,
    )

    masks_by_frame = _run_sam2_propagation(
        frames_dir=args.frames,
        prompts=prompt_spec,
        roi_mask=roi_mask,
        checkpoint=args.sam2_checkpoint,
        model_cfg=args.sam2_model_cfg,
        device=args.sam2_device,
        fill_hole_area=args.sam2_fill_hole_area,
    )
    _save_masks_for_frames(frame_paths, masks_by_frame, sam2_masks_dir)

    frame_indices = [_frame_index_from_path(p) for p in frame_paths]
    prompt_frame_number: Optional[int] = None
    if prompt_frame_idx is not None and prompt_frame_idx < len(frame_indices):
        prompt_frame_number = int(frame_indices[prompt_frame_idx])

    config_out = {
        "frames_dir": os.path.abspath(args.frames),
        "reference": os.path.abspath(args.reference),
        "output_dir": os.path.abspath(args.output),
        "roi_json": os.path.abspath(args.roi) if args.roi else None,
        "roi_mask": os.path.abspath(args.roi_mask) if args.roi_mask else None,
        "prompt_mode": "gdino_keyframe_prompts",
        "prompt_frame_idx": int(prompt_frame_idx) if prompt_frame_idx is not None else None,
        "prompt_frame_number": int(prompt_frame_number) if prompt_frame_number is not None else None,
        "sam2_checkpoint": args.sam2_checkpoint,
        "sam2_model_cfg": args.sam2_model_cfg,
        "sam2_device": args.sam2_device,
        "gdino": {
            "model_id": args.gdino_model_id,
            "text_prompt": args.gdino_text_prompt,
            "box_threshold": float(args.gdino_box_threshold),
            "text_threshold": float(args.gdino_text_threshold),
            "keyframe_interval": int(args.gdino_keyframe_interval),
            "keyframe_start_idx": int(args.gdino_keyframe_start_idx),
            "min_area": int(args.gdino_min_area),
            "debug_dir": os.path.abspath(args.gdino_keyframe_debug_dir) if args.gdino_keyframe_debug_dir else None,
            "num_prompts": int(len(prompt_spec.prompts)),
            "keyframe_logs": gdino_logs,
        },
        "tracking": {
            "use_flow": bool(args.use_flow),
            "velocity_mode": args.velocity_mode,
            "debounce_birth_frames": int(args.debounce_birth_frames),
            "debounce_death_frames": int(args.debounce_death_frames),
            "debounce_relink_max_centroid_dist_px": float(args.debounce_relink_max_centroid_dist_px),
            "debounce_relink_min_area_ratio": float(args.debounce_relink_min_area_ratio),
            "debounce_relink_max_area_ratio": float(args.debounce_relink_max_area_ratio),
            "fps": float(args.fps),
            "mm_per_pixel": args.mm_per_pixel,
            "preprocess_mode": args.preprocess_mode,
            "preprocess_raw_sigma": float(args.preprocess_raw_sigma),
            "min_area": int(args.min_area),
        },
    }
    with open(os.path.join(args.output, "gdino_prompted_flow_config.json"), "w") as f:
        json.dump(config_out, f, indent=2)

    tracking_cfg = TrackingConfig(
        fps=args.fps,
        mm_per_pixel=args.mm_per_pixel,
        use_flow=args.use_flow,
        velocity_mode=args.velocity_mode,
        log_every=args.log_every,
        debounce=DebounceConfig(
            birth_confirm_frames=max(1, int(args.debounce_birth_frames)),
            death_confirm_frames=max(1, int(args.debounce_death_frames)),
            relink_max_centroid_dist_px=float(args.debounce_relink_max_centroid_dist_px),
            relink_min_area_ratio=float(args.debounce_relink_min_area_ratio),
            relink_max_area_ratio=float(args.debounce_relink_max_area_ratio),
        ),
        preprocess=PreprocessConfig(
            mode=args.preprocess_mode,
            raw_sigma=args.preprocess_raw_sigma,
        ),
        segment=SegmentationConfig(min_area=args.min_area),
    )
    outputs = track_frames(
        frames_dir=args.frames,
        reference_path=args.reference,
        output_dir=args.output,
        masks_dir=sam2_masks_dir,
        roi=roi,
        config=tracking_cfg,
    )

    with open(os.path.join(args.output, "gdino_keyframe_prompt_summary.json"), "w") as f:
        json.dump({
            "num_prompts": int(len(prompt_spec.prompts)),
            "text_prompt": args.gdino_text_prompt,
            "box_threshold": float(args.gdino_box_threshold),
            "text_threshold": float(args.gdino_text_threshold),
            "keyframe_logs": gdino_logs,
        }, f, indent=2)

    if not args.no_video:
        output_path = Path(args.output) / "tracking_overlay.mp4"
        create_tracking_video(
            frames_dir=args.frames,
            masks_dir=outputs["masks_dir"],
            result=outputs["result"],
            output_path=str(output_path),
            fps=args.video_fps,
            alpha=0.4,
            overlay_dir=args.overlay_dir,
        )

    print("Tracking complete.")
    print(f"Prompt mode: gdino_keyframe_prompts")
    print(f"Text prompt: {args.gdino_text_prompt!r}")
    print(f"Keyframe prompts kept: {len(prompt_spec.prompts)}")
    print(f"Prompt frame: {prompt_frame_number if prompt_frame_number is not None else '-'}")
    print(f"Objects: {outputs['result'].num_objects}")
    print(f"Events: {outputs['event_summary']}")
    print(f"Masks: {outputs['masks_dir']}")


if __name__ == "__main__":
    main()
