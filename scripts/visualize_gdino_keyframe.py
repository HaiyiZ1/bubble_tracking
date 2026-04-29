#!/usr/bin/env python3
"""
Visualize GDINO keyframe detection + SAM2 box-to-mask refinement on a single frame.

Produces one PNG with three panels:
  1. Raw GDINO boxes (red) with confidence scores.
  2. Per-box SAM2 masks in distinct colors + boxes.
  3. Union mask (green) -- exactly what run_gdino_prompted_flow.py would feed
     to SAM2 video predictor at this keyframe.

This does NOT propagate or track; it's a fast sanity check for prompt/threshold quality.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _distinct_color(i: int) -> tuple:
    palette = [
        (0, 255, 0), (0, 200, 255), (255, 200, 0), (255, 0, 255),
        (0, 255, 200), (255, 100, 0), (100, 255, 0), (0, 100, 255),
        (200, 0, 255), (255, 255, 0), (0, 255, 255), (150, 150, 255),
    ]
    return palette[i % len(palette)]


def _overlay_mask(frame_bgr: np.ndarray, mask: np.ndarray, color: tuple, alpha: float = 0.45) -> np.ndarray:
    out = frame_bgr.copy()
    layer = np.zeros_like(out)
    layer[mask > 0] = color
    cv2.addWeighted(layer, alpha, out, 1.0 - alpha * 0.0, 0, out)
    out[mask > 0] = cv2.addWeighted(out, 1 - alpha, layer, alpha, 0)[mask > 0]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize GDINO+SAM2 detection on one frame")
    parser.add_argument("--frame", required=True, help="Path to the frame image")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--roi-mask", default=None, help="Optional ROI mask PNG to constrain box centers")
    parser.add_argument("--gdino-model-id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--gdino-text-prompt", default="round bubble.")
    parser.add_argument("--gdino-box-threshold", type=float, default=0.25)
    parser.add_argument("--gdino-text-threshold", type=float, default=0.25)
    parser.add_argument("--sam2-checkpoint", default="checkpoints/sam2_hiera_base_plus.pt")
    parser.add_argument("--sam2-model-cfg", default="sam2_hiera_b+.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-width", type=int, default=1600, help="Downscale wide frames for panel output")
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    frame_bgr = cv2.imread(args.frame)
    if frame_bgr is None:
        raise SystemExit(f"Failed to read frame: {args.frame}")
    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(frame_rgb)

    roi_mask = None
    if args.roi_mask:
        roi_mask = cv2.imread(args.roi_mask, cv2.IMREAD_GRAYSCALE)
        if roi_mask is None or roi_mask.shape[:2] != (h, w):
            raise SystemExit("ROI mask missing or size mismatch")

    print(f"[viz] frame: {args.frame}  shape=({h}, {w})  device={args.device}")

    text = args.gdino_text_prompt.strip().lower()
    if not text.endswith("."):
        text = text + "."

    proc = AutoProcessor.from_pretrained(args.gdino_model_id)
    gdino = AutoModelForZeroShotObjectDetection.from_pretrained(args.gdino_model_id).to(args.device).eval()

    inputs = proc(images=pil, text=text, return_tensors="pt").to(args.device)
    with torch.inference_mode():
        outputs = gdino(**inputs)
    results = proc.post_process_grounded_object_detection(
        outputs, input_ids=inputs.input_ids,
        threshold=float(args.gdino_box_threshold),
        text_threshold=float(args.gdino_text_threshold),
        target_sizes=[(h, w)],
    )
    boxes = results[0]["boxes"].detach().cpu().numpy().astype(np.float32)
    scores = results[0]["scores"].detach().cpu().numpy().astype(np.float32)
    print(f"[viz] GDINO boxes: {len(boxes)}  "
          f"score_range=({float(scores.min()) if scores.size else None}, "
          f"{float(scores.max()) if scores.size else None})")

    if roi_mask is not None and boxes.size:
        cx = ((boxes[:, 0] + boxes[:, 2]) * 0.5).astype(np.int32).clip(0, w - 1)
        cy = ((boxes[:, 1] + boxes[:, 3]) * 0.5).astype(np.int32).clip(0, h - 1)
        keep = roi_mask[cy, cx] > 0
        dropped = int((~keep).sum())
        boxes = boxes[keep]
        scores = scores[keep]
        print(f"[viz] ROI filter kept {len(boxes)} boxes, dropped {dropped}")

    # Panel 1: raw boxes
    p1 = frame_bgr.copy()
    for (x1, y1, x2, y2), s in zip(boxes.astype(int), scores):
        cv2.rectangle(p1, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(p1, f"{float(s):.2f}", (int(x1), max(0, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    # Panels 2 and 3: per-box masks + union
    sam2_base = build_sam2(args.sam2_model_cfg, args.sam2_checkpoint, device=args.device)
    predictor = SAM2ImagePredictor(sam2_base)
    predictor.set_image(frame_rgb)

    per_box = frame_bgr.copy()
    union = np.zeros((h, w), dtype=bool)
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if args.device.startswith("cuda") else _null())
    with torch.inference_mode(), autocast_ctx:
        for i, box in enumerate(boxes):
            masks_out, _, _ = predictor.predict(
                point_coords=None, point_labels=None,
                box=box[None, :], multimask_output=False,
            )
            m = np.asarray(masks_out)
            if m.ndim == 3:
                m = m[0]
            m = m > 0
            union |= m
            color = _distinct_color(i)
            per_box = _overlay_mask(per_box, m, color, alpha=0.45)
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(per_box, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    p3 = _overlay_mask(frame_bgr, union.astype(np.uint8), (0, 255, 0), alpha=0.5)
    # Add captions
    for panel, caption in [
        (p1, f"GDINO boxes ({len(boxes)}) - text='{text}'"),
        (per_box, f"SAM2 per-box masks"),
        (p3, f"UNION mask (area={int(union.sum())} px) <- what SAM2 video sees"),
    ]:
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(panel, caption, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)

    # Stack vertically (they share width); downscale for file size
    combo = np.vstack([p1, per_box, p3])
    if combo.shape[1] > args.max_width:
        scale = args.max_width / combo.shape[1]
        combo = cv2.resize(combo, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), combo)
    print(f"[viz] wrote {out_path.resolve()}")


class _null:
    def __enter__(self): return None
    def __exit__(self, *_): return False


if __name__ == "__main__":
    main()
