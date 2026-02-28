#!/usr/bin/env python3
"""
Run biomedical dataset reproduction for Phase 1.2.

Usage:
    # Run on DAVIS 2017 (default: single sequence for smoke test)
    python scripts/run_biomed_repro.py --dataset davis2017 --sequence blackswan

    # Run on full DAVIS validation set
    python scripts/run_biomed_repro.py --dataset davis2017 --all

    # Run on CTC HeLa
    python scripts/run_biomed_repro.py --dataset ctc-hela --sequence 01
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_davis_reproduction(args):
    """Run SAM2 reproduction on DAVIS 2017."""
    from bubble_tracking.datasets import DAVISDataset
    from bubble_tracking.eval.davis_eval import evaluate_davis_sequence, print_davis_results
    from bubble_tracking.schemas import PromptSpec, ObjectPrompt

    # Load dataset
    dataset = DAVISDataset(args.data_path, split="val")
    print(f"Loaded DAVIS 2017: {len(dataset.sequences)} sequences")

    # Determine sequences to process
    if args.all:
        sequences = dataset.sequences
    elif args.sequence:
        sequences = [args.sequence]
    else:
        # Default: first sequence for smoke test
        sequences = dataset.sequences[:1] if dataset.sequences else []

    if not sequences:
        print("No sequences to process")
        return

    print(f"Processing {len(sequences)} sequence(s): {sequences}")

    # Prepare output directory
    output_dir = Path(args.output) / "davis2017"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for seq_name in sequences:
        print(f"\n{'='*50}")
        print(f"Processing: {seq_name}")
        print("=" * 50)

        try:
            seq = dataset.get_sequence(seq_name)
            seq_output_dir = output_dir / seq_name
            masks_output_dir = seq_output_dir / "masks"
            masks_output_dir.mkdir(parents=True, exist_ok=True)

            if args.dry_run:
                print(f"  [Dry run] Would process {seq.num_frames} frames")
                print(f"  Objects: {seq.object_ids}")
                continue

            # Get first-frame prompts from GT masks
            first_frame_prompts = dataset.get_first_frame_prompts(seq_name)
            print(f"  Objects from first frame: {list(first_frame_prompts.keys())}")

            # Build prompts from first frame GT masks (mask prompts; best for VOS)
            prompts = []
            for obj_id, mask in first_frame_prompts.items():
                prompts.append(ObjectPrompt(
                    object_id=obj_id,
                    frame_idx=0,
                    mask=(mask > 0),
                ))

            prompt_spec = PromptSpec(prompts=prompts)

            # Save prompts
            prompts_path = seq_output_dir / "prompts.yaml"
            with open(prompts_path, "w") as f:
                yaml.dump(prompt_spec.to_dict(), f)

            # Check if SAM2 is available
            if args.skip_sam2:
                print("  [Skip SAM2] Generating mock predictions for evaluation test")
                # Copy GT as "predictions" for evaluation smoke test
                import cv2
                for frame_idx in range(seq.num_frames):
                    gt_mask = dataset.load_mask(seq_name, frame_idx)
                    out_path = masks_output_dir / f"{frame_idx:06d}.png"
                    cv2.imwrite(str(out_path), gt_mask)
            else:
                # Run SAM2
                try:
                    from bubble_tracking.sam2_adapter import create_runner
                    from bubble_tracking.track_builder import save_tracks
                    from bubble_tracking.visualization import create_tracking_video

                    runner = create_runner(
                        checkpoint_path=args.checkpoint,
                        model_cfg=args.model_cfg,
                    )

                    result, masks = runner.run(
                        frames_dir=seq.frames_dir,
                        prompts=prompt_spec,
                        output_dir=str(seq_output_dir),
                    )

                    tracks_path = seq_output_dir / "tracks.json"
                    save_tracks(result, str(tracks_path))
                    print(f"  Tracked {result.num_objects} objects over {result.num_frames} frames")

                    # Generate overlay video
                    overlay_path = seq_output_dir / "overlay.mp4"
                    print(f"  Creating overlay video...")
                    create_tracking_video(
                        frames_dir=seq.frames_dir,
                        masks_dir=str(masks_output_dir),
                        result=result,
                        output_path=str(overlay_path),
                        fps=10.0,
                    )
                    print(f"  Saved overlay to: {overlay_path}")

                except ImportError as e:
                    print(f"  [Warning] SAM2 not available: {e}")
                    print("  Run with --skip-sam2 to test evaluation pipeline with GT masks")
                    continue

            # Evaluate
            print("  Evaluating...")
            metrics = evaluate_davis_sequence(
                pred_masks_dir=str(masks_output_dir),
                gt_masks_dir=seq.masks_dir,
            )
            results[seq_name] = metrics
            print(f"  J={metrics['J']:.3f}, F={metrics['F']:.3f}, J&F={metrics['J&F']:.3f}")

        except Exception as e:
            print(f"  [Error] {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    if results:
        print_davis_results(results)

        # Save results
        results_path = output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")


def run_ctc_reproduction(args):
    """Run SAM2 reproduction on CTC dataset."""
    import cv2
    from bubble_tracking.datasets import CTCDataset
    from bubble_tracking.eval.ctc_eval import evaluate_ctc_sequence, print_ctc_results

    # Load dataset
    dataset = CTCDataset(args.data_path)
    print(f"Loaded CTC dataset: {len(dataset.sequences)} sequences")

    # Determine sequences to process
    if args.all:
        sequences = dataset.sequences
    elif args.sequence:
        sequences = [args.sequence]
    else:
        sequences = dataset.sequences[:1] if dataset.sequences else []

    if not sequences:
        print("No sequences to process")
        return

    print(f"Processing {len(sequences)} sequence(s): {sequences}")

    # Prepare output
    output_dir = Path(args.output) / "ctc-hela"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for seq_name in sequences:
        print(f"\n{'='*50}")
        print(f"Processing: {seq_name}")
        print("=" * 50)

        try:
            seq = dataset.get_sequence(seq_name)
            seq_output_dir = output_dir / seq_name
            masks_output_dir = seq_output_dir / "masks"
            masks_output_dir.mkdir(parents=True, exist_ok=True)

            if args.dry_run:
                print(f"  [Dry run] Would process {seq.num_frames} frames")
                continue

            # For now, copy GT for smoke test
            if args.skip_sam2:
                print("  [Skip SAM2] Using GT masks for evaluation smoke test")
                import cv2

                if seq.gt_seg_dir:
                    for frame_idx in range(seq.num_frames):
                        gt_mask = dataset.load_gt_seg(seq_name, frame_idx)
                        if gt_mask is not None:
                            out_path = masks_output_dir / f"{frame_idx:06d}.png"
                            cv2.imwrite(str(out_path), gt_mask)
            else:
                # Run SAM2 on CTC data
                try:
                    from bubble_tracking.sam2_adapter import create_runner
                    from bubble_tracking.track_builder import save_tracks
                    from bubble_tracking.visualization import create_tracking_video
                    from bubble_tracking.schemas import PromptSpec, ObjectPrompt

                    # Get object IDs from first frame GT
                    object_ids = dataset.get_object_ids_from_first_frame(seq_name)
                    print(f"  Objects from first frame: {object_ids}")

                    # Build prompts from first frame GT
                    prompts = []
                    first_mask = dataset.load_gt_tra(seq_name, 0)
                    if first_mask is None:
                        first_mask = dataset.load_gt_seg(seq_name, 0)
                    if first_mask is not None:
                        for obj_id in object_ids:
                            obj_mask = (first_mask == obj_id)
                            if int(obj_mask.sum()) == 0:
                                continue
                            prompts.append(ObjectPrompt(
                                object_id=obj_id,
                                frame_idx=0,
                                mask=obj_mask,
                            ))

                    prompt_spec = PromptSpec(prompts=prompts)

                    # Extract frames to JPEG for SAM2
                    frames_dir = seq_output_dir / "frames"
                    frames_dir.mkdir(parents=True, exist_ok=True)
                    print(f"  Extracting frames to JPEG...")
                    for frame_idx in range(seq.num_frames):
                        frame = dataset.load_frame(seq_name, frame_idx)
                        cv2.imwrite(str(frames_dir / f"{frame_idx:06d}.jpg"), frame)

                    runner = create_runner(
                        checkpoint_path=args.checkpoint,
                        model_cfg=args.model_cfg,
                    )

                    result, masks = runner.run(
                        frames_dir=str(frames_dir),
                        prompts=prompt_spec,
                        output_dir=str(seq_output_dir),
                    )

                    tracks_path = seq_output_dir / "tracks.json"
                    save_tracks(result, str(tracks_path))
                    print(f"  Tracked {result.num_objects} objects over {result.num_frames} frames")

                    # Generate overlay video
                    overlay_path = seq_output_dir / "overlay.mp4"
                    print(f"  Creating overlay video...")
                    create_tracking_video(
                        frames_dir=str(frames_dir),
                        masks_dir=str(masks_output_dir),
                        result=result,
                        output_path=str(overlay_path),
                        fps=10.0,
                    )
                    print(f"  Saved overlay to: {overlay_path}")

                except ImportError as e:
                    print(f"  [Warning] SAM2 not available: {e}")
                    print("  Run with --skip-sam2 to test evaluation pipeline")
                    continue

            # Evaluate (prefer TRA since SEG is sparse in this dataset)
            gt_dir = seq.gt_tra_dir or seq.gt_seg_dir
            if gt_dir:
                print("  Evaluating...")
                metrics = evaluate_ctc_sequence(
                    pred_masks_dir=str(masks_output_dir),
                    gt_masks_dir=gt_dir,
                )
                results[seq_name] = metrics
                print(f"  SEG={metrics['SEG']:.3f} ({metrics['num_frames']} frames)")

        except Exception as e:
            print(f"  [Error] {e}")
            import traceback
            traceback.print_exc()

    if results:
        print_ctc_results(results)

        results_path = output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run biomedical reproduction for Phase 1.2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["davis2017", "ctc-hela"],
        default="davis2017",
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to dataset (default: data/biomed/<dataset>)",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Single sequence to process",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all sequences",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/biomed_repro",
        help="Output directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/sam2_hiera_tiny.pt",
        help="SAM2 checkpoint path",
    )
    parser.add_argument(
        "--model-cfg",
        type=str,
        default="sam2_hiera_t.yaml",
        help="SAM2 model config",
    )
    parser.add_argument(
        "--skip-sam2",
        action="store_true",
        help="Skip SAM2 inference, use GT masks for evaluation smoke test",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup without running inference",
    )

    args = parser.parse_args()

    # Set default data paths
    if args.data_path is None:
        if args.dataset == "davis2017":
            args.data_path = "data/biomed/davis2017"
        elif args.dataset == "ctc-hela":
            args.data_path = "data/biomed/ctc-hela/DIC-C2DH-HeLa"

    # Validate data path
    if not os.path.exists(args.data_path):
        print(f"Dataset not found: {args.data_path}")
        print(f"\nDownload with: python scripts/get_biomed_dataset.py --dataset {args.dataset.replace('2017', '2017-val')}")
        sys.exit(1)

    print(f"Dataset: {args.dataset}")
    print(f"Data path: {args.data_path}")
    print(f"Output: {args.output}")

    if args.dataset == "davis2017":
        run_davis_reproduction(args)
    elif args.dataset == "ctc-hela":
        run_ctc_reproduction(args)


if __name__ == "__main__":
    main()

