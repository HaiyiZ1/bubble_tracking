#!/usr/bin/env python3
"""
Run SAM2 video tracking pipeline.

Usage:
    python scripts/run_sam2_video.py \\
        --frames data/raw/synthetic_frames \\
        --prompts data/raw/prompts/synthetic_prompts.yaml \\
        --output outputs/test_run

    python scripts/run_sam2_video.py \\
        --frames data/interim/video_frames \\
        --prompts prompts.yaml \\
        --checkpoint checkpoints/sam2_hiera_tiny.pt \\
        --output outputs/my_run
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
import cv2

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_prompts(prompts_path: str):
    """Load prompts from YAML file."""
    from bubble_tracking.schemas import PromptSpec

    with open(prompts_path, "r") as f:
        data = yaml.safe_load(f)
    return PromptSpec.from_dict(data)


def save_run_config(output_dir: str, config: dict) -> None:
    """Save run configuration for reproducibility."""
    config_path = os.path.join(output_dir, "run_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def main():
    parser = argparse.ArgumentParser(
        description="Run SAM2 video tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--frames",
        type=str,
        required=True,
        help="Directory containing input frames (PNG/JPG)",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        required=True,
        help="Path to prompts YAML file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: outputs/<timestamp>)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/sam2_hiera_tiny.pt",
        help="Path to SAM2 checkpoint",
    )
    parser.add_argument(
        "--roi-mask",
        type=str,
        default=None,
        help="Optional ROI mask PNG (nonzero=keep)",
    )
    parser.add_argument(
        "--model-cfg",
        type=str,
        default="sam2_hiera_t.yaml",
        help="SAM2 model configuration name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cpu/cuda, default: auto-detect)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="FPS for output video",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="Mask overlay opacity (0-1)",
    )
    parser.add_argument(
        "--label-scale",
        type=float,
        default=0.9,
        help="Label font scale for IDs",
    )
    parser.add_argument(
        "--label-thickness",
        type=int,
        default=2,
        help="Label font thickness",
    )
    parser.add_argument(
        "--overlay-dir",
        type=str,
        default=None,
        help="Optional directory to save per-frame overlay images",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip creating overlay video",
    )
    parser.add_argument(
        "--event-min-area",
        type=int,
        default=25,
        help="Minimum connected-component area for split/merge analysis",
    )
    parser.add_argument(
        "--event-min-overlap-px",
        type=int,
        default=10,
        help="Minimum overlap pixels required to link components across frames",
    )
    parser.add_argument(
        "--event-min-link-iou",
        type=float,
        default=0.05,
        help="Minimum IoU for parent-child component linkage",
    )
    parser.add_argument(
        "--event-min-link-overlap-ratio",
        type=float,
        default=0.25,
        help="Minimum overlap/min(area) ratio for parent-child linkage",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs without running SAM2",
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.isdir(args.frames):
        print(f"Error: Frames directory not found: {args.frames}")
        sys.exit(1)

    if not os.path.exists(args.prompts):
        print(f"Error: Prompts file not found: {args.prompts}")
        sys.exit(1)

    # Set output directory
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join("outputs", f"run_{timestamp}")

    os.makedirs(args.output, exist_ok=True)
    masks_dir = os.path.join(args.output, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    # Load prompts
    prompts = load_prompts(args.prompts)
    print(f"Loaded {len(prompts.prompts)} object prompts")

    # Save run configuration
    run_config = {
        "frames_dir": os.path.abspath(args.frames),
        "prompts_file": os.path.abspath(args.prompts),
        "checkpoint": args.checkpoint,
        "model_cfg": args.model_cfg,
        "device": args.device or "auto",
        "created_at": datetime.now().isoformat(),
        "num_prompts": len(prompts.prompts),
        "event_analysis": {
            "min_component_area": args.event_min_area,
            "min_overlap_pixels": args.event_min_overlap_px,
            "min_link_iou": args.event_min_link_iou,
            "min_link_overlap_ratio": args.event_min_link_overlap_ratio,
        },
    }
    if args.roi_mask:
        run_config["roi_mask"] = os.path.abspath(args.roi_mask)
    save_run_config(args.output, run_config)

    if args.dry_run:
        print("\n[Dry run] Configuration validated successfully")
        print(f"Output would be written to: {args.output}")
        return

    # Check for SAM2
    if not os.path.exists(args.checkpoint):
        print(f"\nError: SAM2 checkpoint not found: {args.checkpoint}")
        print("\nTo download SAM2 checkpoint:")
        print("  mkdir -p checkpoints")
        print("  cd checkpoints")
        print("  wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt")
        print("\nOr run with --dry-run to validate configuration without SAM2")
        sys.exit(1)

    # Import SAM2 adapter
    try:
        from bubble_tracking.sam2_adapter import create_runner
        from bubble_tracking.track_builder import save_tracks, compute_track_statistics
        from bubble_tracking.mask_events import analyze_masks_by_frame, save_area_timeseries_csv
        from bubble_tracking.visualization import create_tracking_video
    except ImportError as e:
        print(f"Error importing bubble_tracking modules: {e}")
        print("Make sure you've installed dependencies: pip install -r requirements.txt")
        sys.exit(1)

    # Run SAM2
    print(f"\nRunning SAM2 tracking...")
    print(f"  Frames: {args.frames}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device: {args.device or 'auto'}")

    roi_mask = None
    if args.roi_mask:
        roi_mask = cv2.imread(args.roi_mask, cv2.IMREAD_GRAYSCALE)
        if roi_mask is None:
            print(f"Error: ROI mask not found or unreadable: {args.roi_mask}")
            sys.exit(1)

    runner = create_runner(
        checkpoint_path=args.checkpoint,
        model_cfg=args.model_cfg,
        device=args.device,
    )

    result, masks = runner.run(
        frames_dir=args.frames,
        prompts=prompts,
        output_dir=args.output,
        roi_mask=roi_mask,
    )

    # Save tracks
    tracks_path = os.path.join(args.output, "tracks.json")
    save_tracks(result, tracks_path)
    print(f"Saved tracks to: {tracks_path}")

    # Compute and save statistics
    stats = compute_track_statistics(result)

    # Compute mask-level area dynamics and split/merge events
    mask_analysis = analyze_masks_by_frame(
        masks_by_frame=masks,
        min_component_area=args.event_min_area,
        min_overlap_pixels=args.event_min_overlap_px,
        min_link_iou=args.event_min_link_iou,
        min_link_overlap_ratio=args.event_min_link_overlap_ratio,
    )
    area_json_path = os.path.join(args.output, "area_timeseries.json")
    with open(area_json_path, "w") as f:
        json.dump(mask_analysis["area_timeseries"], f, indent=2)
    area_csv_path = os.path.join(args.output, "area_timeseries.csv")
    save_area_timeseries_csv(mask_analysis["area_timeseries"], area_csv_path)
    events_path = os.path.join(args.output, "events.json")
    with open(events_path, "w") as f:
        json.dump(
            {
                "metadata": mask_analysis["analysis_metadata"],
                "summary": mask_analysis["event_summary"],
                "events": mask_analysis["events"],
            },
            f,
            indent=2,
        )

    # Fold new aggregate metrics into stats for one-stop summary.
    stats["bubble_area"] = mask_analysis["area_summary"]
    stats["mask_event_summary"] = mask_analysis["event_summary"]
    stats["mask_event_config"] = mask_analysis["analysis_metadata"]

    stats_path = os.path.join(args.output, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to: {stats_path}")
    print(f"Saved area timeseries to: {area_json_path} and {area_csv_path}")
    print(f"Saved split/merge events to: {events_path}")

    # Create overlay video
    if not args.no_video:
        video_path = os.path.join(args.output, "overlay.mp4")
        print(f"Creating overlay video: {video_path}")
        create_tracking_video(
            frames_dir=args.frames,
            masks_dir=masks_dir,
            result=result,
            output_path=video_path,
            fps=args.fps,
            alpha=args.alpha,
            overlay_dir=args.overlay_dir,
            label_scale=args.label_scale,
            label_thickness=args.label_thickness,
        )

    print(f"\nDone! Results saved to: {args.output}")
    print(f"  - tracks.json: {result.num_objects} objects, {result.num_frames} frames")
    print(f"  - masks/: per-frame segmentation masks")
    print(
        f"  - area_timeseries: {len(mask_analysis['area_timeseries'])} frames "
        f"(mean area {mask_analysis['area_summary']['mean_total_area_px']:.1f}px)"
    )
    print(
        f"  - events.json: splits={mask_analysis['event_summary']['num_splits']}, "
        f"merges={mask_analysis['event_summary']['num_merges']}"
    )
    if not args.no_video:
        print(f"  - overlay.mp4: visualization video")


if __name__ == "__main__":
    main()
