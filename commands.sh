python scripts/run_reprompted_tracking.py \
  --frames data/interim/bubbles_air1_072_271/frames \
  --roi-mask data/interim/bubbles_air1_072_271/roi_mask.png \
  --output outputs/reprompt_recover_cuda_200 \
  --checkpoint checkpoints/sam2_hiera_tiny.pt \
  --model-cfg sam2_hiera_t.yaml \
  --device cuda \
  --recover-only \
  --keyframe-interval 120 \
  --points-per-side 32 \
  --min-points-per-side 8 \
  --mask-scale 0.5 \
  --min-area 20

/home/aaron/Downloads/bubble_tracking/venv311/bin/python scripts/run_sam2_prompted_flow.py \
    --frames data/interim/bubbles_air1_072_271/frames \
    --output outputs/test_reprompt_recover_200 \
    --roi data/interim/bubbles_air1_072_271/roi.json \
    --roi-mask data/interim/bubbles_air1_072_271/roi_mask.png \
    --prompt-mode sam2 \
    --sam2-checkpoint checkpoints/sam2_hiera_tiny.pt \
    --sam2-model-cfg sam2_hiera_t.yaml \
    --sam2-mask-scale 1.0 \
    --sam2-points-per-side 16 \
    --sam2-device cuda \
    --fps 4 \
    --mm-per-pixel 0.0856 \
    --recover-only \
    --keyframe-interval 120


/home/aaron/Downloads/bubble_tracking/venv311/bin/python scripts/run_reprompted_tracking.py \
  --frames data/interim/bubbles_air1_072_271/frames \
  --output outputs/test_reprompt_recover_200 \
  --checkpoint checkpoints/sam2_hiera_tiny.pt \
  --model-cfg sam2_hiera_t.yaml \
  --device cuda \
  --recover-only \
  --keyframe-interval 120 \
  --roi-mask data/interim/bubbles_air1_072_271/roi_mask.png \
  --mask-scale 0.5 \
  --points-per-side 32 \
  --min-points-per-side 8 \
  --min-area 20 

# new preprocessing + original prompt once + tracking
python scripts/run_sam2_prompted_flow.py \
  --frames data/interim/bubbles_air1_072_271/frames \
  --reference data/bubbles/frames/000001.jpg \
  --output outputs/prompt_once_raw_200 \
  --roi data/interim/bubbles_air1_072_271/roi.json \
  --roi-mask data/interim/bubbles_air1_072_271/roi_mask.png \
  --prompt-mode sam2 \
  --sam2-checkpoint checkpoints/sam2_hiera_tiny.pt \
  --sam2-model-cfg sam2_hiera_t.yaml \
  --sam2-device cuda \
  --sam2-mask-scale 0.5 \
  --sam2-points-per-side 16 \
  --preprocess-mode raw \
  --preprocess-raw-sigma 1.0 \
  --fps 4 \
  --mm-per-pixel 0.0856


# Pass 1: original prompt-once SAM2 + tracking.
# Build reprompts from pass-1 tracked IDs (weak/missing).
# Pass 2: rerun SAM2 with combined prompts, then final tracking output.

python scripts/run_sam2_prompted_flow.py \
  --frames data/bubbles/frames_0p5 \
  --reference data/bubbles/frames_0p5/000001.jpg \
  --output outputs/sam2_prompted_flow_event_reprompt_0p5 \
  --roi data/interim/bubbles_air1_072_271/roi_0p5.json \
  --roi-mask data/interim/bubbles_air1_072_271/roi_mask_0p5.png \
  --prompt-mode sam2 \
  --sam2-checkpoint checkpoints/sam2_hiera_tiny.pt \
  --sam2-model-cfg sam2_hiera_t.yaml \
  --sam2-device cuda \
  --sam2-mask-scale 0.5 \
  --sam2-points-per-side 16 \
  --preprocess-mode raw \
  --preprocess-raw-sigma 1.0 \
  --event-reprompt \
  --event-reprompt-interval 120 \
  --event-reprompt-missing-frames 6 \
  --event-reprompt-weak-area-drop-ratio 0.35 \
  --event-reprompt-weak-consecutive 3 \
  --event-reprompt-seed-tracks \
  --event-reprompt-seed-min-lifetime 5 \
  --fps 4 \
  --mm-per-pixel 0.0856
# This OOM


python scripts/run_sam2_prompted_flow.py \
  --frames data/bubbles/frames_0p5 \
  --reference data/bubbles/frames_0p5/000001.jpg \
  --output outputs/sam2_prompted_flow_event_reprompt_0p5 \
  --roi data/interim/bubbles_air1_072_271/roi_0p5.json \
  --roi-mask data/interim/bubbles_air1_072_271/roi_mask_0p5.png \
  --prompt-mode sam2 \
  --sam2-checkpoint checkpoints/sam2_hiera_tiny.pt \
  --sam2-model-cfg sam2_hiera_t.yaml \
  --sam2-device cuda \
  --sam2-mask-scale 0.5 \
  --sam2-points-per-side 16 \
  --preprocess-mode raw \
  --preprocess-raw-sigma 1.0 \
  --event-reprompt \
  --no-event-reprompt-seed-tracks \
  --event-reprompt-max-prompts-per-keyframe 4 \
  --event-reprompt-interval 240 \
  --event-reprompt-missing-frames 6 \
  --event-reprompt-weak-area-drop-ratio 0.35 \
  --event-reprompt-weak-consecutive 3 \
  --fps 4 \
  --mm-per-pixel 0.0856

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/run_sam2_prompted_flow.py \
  --frames data/bubbles/frames_0p5 \
  --reference data/bubbles/frames_0p5/000001.jpg \
  --output outputs/sam2_prompted_flow_event_reprompt_0p5 \
  --roi data/interim/bubbles_air1_072_271/roi_0p5.json \
  --roi-mask data/interim/bubbles_air1_072_271/roi_mask_0p5.png \
  --prompt-mode sam2 \
  --sam2-checkpoint checkpoints/sam2_hiera_tiny.pt \
  --sam2-model-cfg sam2_hiera_t.yaml \
  --sam2-device cuda \
  --sam2-mask-scale 0.5 \
  --sam2-points-per-side 16 \
  --preprocess-mode raw \
  --preprocess-raw-sigma 1.0 \
  --event-reprompt \
  --no-event-reprompt-seed-tracks \
  --event-reprompt-max-prompts-per-keyframe 4 \
  --event-reprompt-interval 240 \
  --event-reprompt-missing-frames 6 \
  --event-reprompt-weak-area-drop-ratio 0.35 \
  --event-reprompt-weak-consecutive 3 \
  --fps 4 \
  --mm-per-pixel 0.0856

# stil oom



python scripts/run_sam2_prompted_flow.py \
  --frames data/bubbles/frames_0p5 \
  --reference data/bubbles/frames_0p5/000001.jpg \
  --output outputs/sam2_prompted_flow_event_reprompt_0p5_pass2only \
  --roi data/interim/bubbles_air1_072_271/roi_0p5.json \
  --roi-mask data/interim/bubbles_air1_072_271/roi_mask_0p5.png \
  --prompt-mode sam2 \
  --sam2-checkpoint checkpoints/sam2_hiera_tiny.pt \
  --sam2-model-cfg sam2_hiera_t.yaml \
  --sam2-device cuda \
  --sam2-mask-scale 0.5 \
  --sam2-points-per-side 16 \
  --preprocess-mode raw \
  --preprocess-raw-sigma 1.0 \
  --event-reprompt \
  --event-reprompt-pass1-tracks outputs/sam2_prompted_flow_event_reprompt_0p5/event_reprompt_pass1_tracking/tracks.json \
  --no-event-reprompt-seed-tracks \
  --event-reprompt-max-prompts-per-keyframe 2 \
  --event-reprompt-interval 240 \
  --event-reprompt-missing-frames 6 \
  --event-reprompt-weak-area-drop-ratio 0.35 \
  --event-reprompt-weak-consecutive 3 \
  --fps 4 \
  --mm-per-pixel 0.0856

# still ooms


# global reprompt
/home/aaron/Downloads/bubble_tracking/venv311/bin/python scripts/run_sam2_prompted_flow.py \
  --frames data/bubbles/frames_0p5 \
  --reference data/bubbles/frames_0p5/000001.jpg \
  --output outputs/sam2_global_union_0p5 \
  --roi data/interim/bubbles_air1_072_271/roi_0p5.json \
  --roi-mask data/interim/bubbles_air1_072_271/roi_mask_0p5.png \
  --prompt-mode sam2 \
  --sam2-global-union-reprompt \
  --sam2-global-union-interval 120 \
  --sam2-checkpoint checkpoints/sam2_hiera_tiny.pt \
  --sam2-model-cfg sam2_hiera_t.yaml \
  --sam2-device cuda \
  --sam2-mask-scale 0.5 \
  --sam2-min-mask-scale 0.25 \
  --sam2-points-per-side 16 \
  --sam2-min-points-per-side 8 \
  --sam2-min-area 20 \
  --sam2-max-area-ratio 0.5 \
  --preprocess-mode raw \
  --preprocess-raw-sigma 1.0 \
  --fps 4 \
  --mm-per-pixel 0.0856

# promising, missing small and weak contoured bubbles

# try larger model
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/aaron/Downloads/bubble_tracking/venv311/bin/python scripts/run_sam2_prompted_flow.py \
  --frames data/bubbles/frames_0p5 \
  --reference data/bubbles/frames_0p5/000001.jpg \
  --output outputs/sam2_global_union_baseplus_kf100 \
  --roi data/interim/bubbles_air1_072_271/roi_0p5.json \
  --roi-mask data/interim/bubbles_air1_072_271/roi_mask_0p5.png \
  --prompt-mode sam2 \
  --sam2-checkpoint checkpoints/sam2_hiera_base_plus.pt \
  --sam2-model-cfg sam2_hiera_b+.yaml \
  --sam2-device cuda \
  --sam2-global-union-reprompt \
  --sam2-global-union-interval 100 \
  --sam2-mask-scale 0.5 \
  --sam2-min-mask-scale 0.5 \
  --sam2-points-per-side 16 \
  --sam2-min-points-per-side 8 \
  --sam2-min-area 20 \
  --sam2-max-area-ratio 0.5 \
  --preprocess-mode raw \
  --preprocess-raw-sigma 1.0 \
  --fps 4 \
  --mm-per-pixel 0.0856

# not really better
# SAM2 is ingoring a lot of bubbles

# use cv segmentation to reprompt
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/aaron/Downloads/bubble_tracking/venv311/bin/python scripts/run_sam2_prompted_flow.py \
  --frames data/bubbles/frames_0p5 \
  --reference data/bubbles/frames_0p5/000001.jpg \
  --output outputs/sam2_cv_keyframe100_min100_baseplus \
  --roi data/interim/bubbles_air1_072_271/roi_0p5.json \
  --roi-mask data/interim/bubbles_air1_072_271/roi_mask_0p5.png \
  --prompt-mode sam2 \
  --sam2-checkpoint checkpoints/sam2_hiera_base_plus.pt \
  --sam2-model-cfg sam2_hiera_b+.yaml \
  --sam2-device cuda \
  --cv-keyframe-prompts \
  --cv-keyframe-interval 100 \
  --cv-keyframe-start-idx 0 \
  --cv-keyframe-preprocess-mode default \
  --cv-keyframe-diff-sigma 0.5 \
  --cv-keyframe-min-area 100 \
  --cv-keyframe-threshold-offset 7 \
  --preprocess-mode raw \
  --preprocess-raw-sigma 1.0 \
  --fps 4 \
  --mm-per-pixel 0.0856


# Single-frame preprocessing inspection 
/home/aaron/Downloads/bubble_tracking/venv311/bin/python scripts/inspect_cv_frame.py \
  --frame 000080.jpg \
  --output-dir outputs/debug_cv_frame80_script \
  --preprocess-mode default \
  --diff-sigma 0.5 \
  --min-area 100 \
  --threshold-offset 7 \
  --no-normalize

# Full CV-keyframe -> SAM2 run with no normalization in CV prompt stage
/home/aaron/Downloads/bubble_tracking/venv311/bin/python scripts/run_sam2_prompted_flow.py \
  --frames data/bubbles/frames_0p5 \
  --reference data/bubbles/frames_0p5/000001.jpg \
  --output outputs/sam2_cv_keyframe100_nonorm \
  --roi data/interim/bubbles_air1_072_271/roi_0p5.json \
  --roi-mask data/interim/bubbles_air1_072_271/roi_mask_0p5.png \
  --prompt-mode sam2 \
  --sam2-checkpoint checkpoints/sam2_hiera_base_plus.pt \
  --sam2-model-cfg sam2_hiera_b+.yaml \
  --sam2-device cuda \
  --cv-keyframe-prompts \
  --cv-keyframe-interval 100 \
  --cv-keyframe-start-idx 0 \
  --cv-keyframe-preprocess-mode default \
  --cv-keyframe-diff-sigma 0.5 \
  --cv-keyframe-no-normalize \
  --cv-keyframe-min-area 100 \
  --cv-keyframe-threshold-offset 7 \
  --cv-keyframe-start-idx 90 \
  --preprocess-mode raw \
  --preprocess-raw-sigma 1.0 \
  --fps 4 \
  --mm-per-pixel 0.0856

# regenerate speed
/home/aaron/Downloads/bubble_tracking/venv311/bin/python scripts/run_flow_tracking.py \
  --frames data/bubbles/frames_0p5 \
  --reference data/bubbles/frames_0p5/000001.jpg \
  --masks-dir outputs/sam2_cv_keyframe100_nonorm/sam2_masks \
  --output outputs/sam2_cv_keyframe100_nonorm_vel_centroid \
  --roi data/interim/bubbles_air1_072_271/roi_0p5.json \
  --preprocess-mode raw \
  --raw-sigma 1.0 \
  --min-area 100 \
  --use-flow \
  --velocity-mode centroid_delta \
  --flow-backend auto \
  --fps 4 \
  --mm-per-pixel 0.0856 \
  --no-video

/home/aaron/Downloads/bubble_tracking/venv311/bin/python scripts/export_velocity_overlay.py \
  --run-dir outputs/sam2_cv_keyframe100_nonorm_vel_centroid \
  --frames data/bubbles/frames_0p5 \
  --output outputs/sam2_cv_keyframe100_nonorm_vel_centroid/velocity_overlay.mp4 \
  --fps 4 \
  --label-units mm_s

# case1 air1 whole pipeline rerun with centroid delta
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/aaron/Downloads/bubble_tracking/venv311/bin/python scripts/run_sam2_prompted_flow.py \
  --frames data/bubbles/frames_0p5 \
  --reference data/bubbles/frames_0p5/000001.jpg \
  --output outputs/case1_air1_sam2_cv_keyframe160_nonorm_centroid_delta \
  --roi data/interim/bubbles_air1_072_271/roi_0p5.json \
  --roi-mask data/interim/bubbles_air1_072_271/roi_mask_0p5.png \
  --prompt-mode sam2 \
  --sam2-checkpoint checkpoints/sam2_hiera_tiny.pt \
  --sam2-model-cfg sam2_hiera_t.yaml \
  --sam2-device cuda \
  --cv-keyframe-prompts \
  --cv-keyframe-interval 160 \
  --cv-keyframe-start-idx 80 \
  --cv-keyframe-preprocess-mode default \
  --cv-keyframe-diff-sigma 0.5 \
  --cv-keyframe-no-normalize \
  --cv-keyframe-min-area 100 \
  --cv-keyframe-threshold-offset 7 \
  --use-flow \
  --velocity-mode centroid_delta \
  --preprocess-mode raw \
  --preprocess-raw-sigma 1.0 \
  --fps 4 \
  --mm-per-pixel 0.0856

/home/aaron/Downloads/bubble_tracking/venv311/bin/python scripts/export_velocity_overlay.py \
  --run-dir outputs/case1_air1_sam2_cv_keyframe160_nonorm_centroid_delta \
  --frames data/bubbles/frames_0p5 \
  --output outputs/case1_air1_sam2_cv_keyframe160_nonorm_centroid_delta/velocity_overlay.mp4 \
  --fps 4 \
  --label-units mm_s

# prepare case 2 and 3 data

python scripts/copy_and_compress_frames.py

# case2 roi
python scripts/select_roi.py \
  --frames data/bubbles/case2_air1/frames_0p5 \
  --frame-idx 1 \
  --out data/interim/case2_air1/roi_0p5.json

python scripts/select_polygon_roi.py \
  --frames data/bubbles/case2_air1/frames_0p5 \
  --frame-idx 1 \
  --out-json data/interim/case2_air1/roi_poly_0p5.json \
  --out-mask data/interim/case2_air1/roi_mask_0p5.png

# case3
python scripts/select_roi.py \
  --frames data/bubbles/case3_air1/frames_0p5 \
  --frame-idx 1 \
  --out data/interim/case3_air1/roi_0p5.json

python scripts/select_polygon_roi.py \
  --frames data/bubbles/case3_air1/frames_0p5 \
  --frame-idx 1 \
  --out-json data/interim/case3_air1/roi_poly_0p5.json \
  --out-mask data/interim/case3_air1/roi_mask_0p5.png

# case 2 tracking and velocity visualization

python scripts/run_sam2_prompted_flow.py \
  --frames data/bubbles/case2_air1/frames_0p5 \
  --reference data/bubbles/case2_air1/frames_0p5/000001.jpg \
  --output outputs/case2_air1_sam2_cv_keyframe100_nonorm_centroid_delta \
  --roi data/interim/case2_air1/roi_0p5.json \
  --roi-mask data/interim/case2_air1/roi_mask_0p5.png \
  --prompt-mode sam2 \
  --sam2-checkpoint checkpoints/sam2_hiera_base_plus.pt \
  --sam2-model-cfg sam2_hiera_b+.yaml \
  --sam2-device cuda \
  --cv-keyframe-prompts \
  --cv-keyframe-interval 100 \
  --cv-keyframe-start-idx 127 \
  --cv-keyframe-preprocess-mode default \
  --cv-keyframe-diff-sigma 0.5 \
  --cv-keyframe-no-normalize \
  --cv-keyframe-min-area 50 \
  --cv-keyframe-threshold-offset 7 \
  --cv-keyframe-debug-dir outputs/case2_air1_sam2_cv_keyframe100_nonorm_centroid_delta/cv_keyframe_debug \
  --use-flow \
  --velocity-mode centroid_delta \
  --preprocess-mode raw \
  --preprocess-raw-sigma 1.0 \
  --fps 4 \
  --mm-per-pixel 0.0856

python scripts/export_velocity_overlay.py \
  --run-dir outputs/case2_air1_sam2_cv_keyframe100_nonorm_centroid_delta \
  --frames data/bubbles/case2_air1/frames_0p5 \
  --output outputs/case2_air1_sam2_cv_keyframe100_nonorm_centroid_delta/velocity_overlay.mp4 \
  --fps 4 \
  --label-units mm_s

# case 3

python scripts/run_sam2_prompted_flow.py \
  --frames data/bubbles/case3_air1/frames_0p5 \
  --reference data/bubbles/case3_air1/frames_0p5/000001.jpg \
  --output outputs/case3_air1_sam2_cv_keyframe100_nonorm_centroid_delta_run_2 \
  --roi data/interim/case3_air1/roi_0p5.json \
  --roi-mask data/interim/case3_air1/roi_mask_0p5.png \
  --prompt-mode sam2 \
  --sam2-checkpoint checkpoints/sam2_hiera_base_plus.pt \
  --sam2-model-cfg sam2_hiera_b+.yaml  \
  --sam2-device cuda \
  --cv-keyframe-prompts \
  --cv-keyframe-interval 100 \
  --cv-keyframe-start-idx 224 \
  --cv-keyframe-preprocess-mode default \
  --cv-keyframe-diff-sigma 0.5 \
  --cv-keyframe-no-normalize \
  --cv-keyframe-min-area 50 \
  --cv-keyframe-threshold-offset 7 \
  --cv-keyframe-debug-dir outputs/case3_air1_sam2_cv_keyframe100_nonorm_centroid_delta/cv_keyframe_debug \
  --use-flow \
  --velocity-mode centroid_delta \
  --preprocess-mode raw \
  --preprocess-raw-sigma 1.0 \
  --fps 4 \
  --mm-per-pixel 0.0856


python scripts/export_velocity_overlay.py \
  --run-dir outputs/case3_air1_sam2_cv_keyframe100_nonorm_centroid_delta \
  --frames data/bubbles/case3_air1/frames_0p5 \
  --output outputs/case3_air1_sam2_cv_keyframe100_nonorm_centroid_delta/velocity_overlay.mp4 \
  --fps 4 \
  --label-units mm_s

# case 3 air 4
python scripts/copy_and_compress_frames.py

python scripts/run_sam2_prompted_flow.py \
  --frames data/bubbles/case3_air4/frames_0p5 \
  --reference data/bubbles/case3_air1/frames_0p5/000001.jpg \
  --output outputs/case3_air4_sam2_cv_keyframe100_nonorm_centroid_delta \
  --roi data/interim/case3_air1/roi_0p5.json \
  --roi-mask data/interim/case3_air1/roi_mask_0p5.png \
  --prompt-mode sam2 \
  --sam2-checkpoint checkpoints/sam2_hiera_base_plus.pt \
  --sam2-model-cfg sam2_hiera_b+.yaml  \
  --sam2-device cuda \
  --cv-keyframe-prompts \
  --cv-keyframe-interval 100 \
  --cv-keyframe-start-idx 0 \
  --cv-keyframe-preprocess-mode default \
  --cv-keyframe-diff-sigma 0.5 \
  --cv-keyframe-no-normalize \
  --cv-keyframe-min-area 50 \
  --cv-keyframe-threshold-offset 7 \
  --cv-keyframe-debug-dir outputs/case3_air4_sam2_cv_keyframe100_nonorm_centroid_delta/cv_keyframe_debug \
  --use-flow \
  --velocity-mode centroid_delta \
  --preprocess-mode raw \
  --preprocess-raw-sigma 1.0 \
  --fps 4 \
  --mm-per-pixel 0.0856


python scripts/export_velocity_overlay.py \
  --run-dir outputs/case3_air4_sam2_cv_keyframe100_nonorm_centroid_delta \
  --frames data/bubbles/case3_air1/frames_0p5 \
  --output outputs/case3_air4_sam2_cv_keyframe100_nonorm_centroid_delta/velocity_overlay.mp4 \
  --fps 4 \
  --label-units mm_s


# ---Publishing run

# relabel data (start from frame 71 for case 1 air 1, 127 for case 2 air 1, 215 for case 3 air 1)
python scripts/relabel_publishing_air1.py --case case1_air1 case2_air1 case3_air1



# re-run pipeline with updated video output
## with debounce options to reduce flickering

# case 1 air 1
python scripts/run_sam2_prompted_flow.py \
  --frames data/bubbles/case1_air1_publish/frames_0p5 \
  --reference data/bubbles/case1_air1_publish/frames_0p5/000001.jpg \
  --output outputs/case1_air1_publish \
  --roi data/interim/bubbles_air1_072_271/roi_0p5.json \
  --roi-mask data/interim/bubbles_air1_072_271/roi_mask_0p5.png \
  --prompt-mode sam2 \
  --sam2-checkpoint checkpoints/sam2_hiera_tiny.pt \
  --sam2-model-cfg sam2_hiera_t.yaml \
  --sam2-device cuda \
  --cv-keyframe-prompts \
  --cv-keyframe-interval 160 \
  --cv-keyframe-start-idx 10 \
  --cv-keyframe-preprocess-mode default \
  --cv-keyframe-diff-sigma 0.5 \
  --cv-keyframe-no-normalize \
  --cv-keyframe-min-area 50 \
  --cv-keyframe-threshold-offset 7 \
  --cv-keyframe-debug-dir outputs/case1_air1_publish_debug \
  --use-flow \
  --velocity-mode centroid_delta \
  --preprocess-mode raw \
  --preprocess-raw-sigma 1.0 \
  --debounce-birth-frames 2 \
  --debounce-death-frames 3 \
  --fps 4 \
  --mm-per-pixel 0.08563


# case 2 air 1
python scripts/run_sam2_prompted_flow.py \
  --frames data/bubbles/case2_air1_publish/frames_0p5 \
  --reference data/bubbles/case2_air1_publish/frames_0p5/000001.jpg \
  --output outputs/case2_air1_publish \
  --roi data/interim/case2_air1/roi_0p5.json \
  --roi-mask data/interim/case2_air1/roi_mask_0p5.png \
  --prompt-mode sam2 \
  --sam2-checkpoint checkpoints/sam2_hiera_base_plus.pt \
  --sam2-model-cfg sam2_hiera_b+.yaml \
  --sam2-device cuda \
  --cv-keyframe-prompts \
  --cv-keyframe-interval 160 \
  --cv-keyframe-start-idx 2 \
  --cv-keyframe-preprocess-mode default \
  --cv-keyframe-diff-sigma 0.5 \
  --cv-keyframe-no-normalize \
  --cv-keyframe-min-area 50 \
  --cv-keyframe-threshold-offset 7 \
  --cv-keyframe-debug-dir outputs/case2_air1_publish_debug \
  --use-flow \
  --velocity-mode centroid_delta \
  --preprocess-mode raw \
  --preprocess-raw-sigma 1.0 \
  --debounce-birth-frames 2 \
  --debounce-death-frames 3 \
  --fps 4 \
  --mm-per-pixel 0.08695

# case 3 air 1
python scripts/run_sam2_prompted_flow.py \
  --frames data/bubbles/case3_air1_publish/frames_0p5 \
  --reference data/bubbles/case3_air1_publish/frames_0p5/000001.jpg \
  --output outputs/case3_air1_publish \
  --roi data/interim/case3_air1/roi_0p5.json \
  --roi-mask data/interim/case3_air1/roi_mask_0p5.png \
  --prompt-mode sam2 \
  --sam2-checkpoint checkpoints/sam2_hiera_base_plus.pt \
  --sam2-model-cfg sam2_hiera_b+.yaml  \
  --sam2-device cuda \
  --cv-keyframe-prompts \
  --cv-keyframe-interval 160 \
  --cv-keyframe-start-idx 10 \
  --cv-keyframe-preprocess-mode default \
  --cv-keyframe-diff-sigma 0.5 \
  --cv-keyframe-no-normalize \
  --cv-keyframe-min-area 50 \
  --cv-keyframe-threshold-offset 7 \
  --cv-keyframe-debug-dir outputs/case3_air1_publish_debug \
  --use-flow \
  --velocity-mode centroid_delta \
  --preprocess-mode raw \
  --preprocess-raw-sigma 1.0 \
  --debounce-birth-frames 2 \
  --debounce-death-frames 3 \
  --fps 4 \
  --mm-per-pixel 0.08264

# generate publishing videos
python scripts/export_publishing_tracking_video.py \
  --case case1_air1 \
  --run-dir outputs/case1_air1_publish \
  --frames data/bubbles/case1_air1_publish/frames_0p5

python scripts/export_publishing_tracking_video.py \
  --case case2_air1 \
  --run-dir outputs/case2_air1_publish \
  --frames data/bubbles/case2_air1_publish/frames_0p5

python scripts/export_publishing_tracking_video.py \
  --case case3_air1 \
  --run-dir outputs/case3_air1_publish \
  --frames data/bubbles/case3_air1_publish/frames_0p5

# moving average of vanished area (2s/5s) window

## case 1
python scripts/plot_vanished_area_moving_window.py \
  --run-dir outputs/case1_air1_publish

## case 2
python scripts/plot_vanished_area_moving_window.py \
  --run-dir outputs/case2_air1_publish

## case 3
python scripts/plot_vanished_area_moving_window.py \
  --run-dir outputs/case3_air1_publish


# previous plots and summaries

# 1) Generate bubble_velocity_area_metrics.csv for each publish run
for case in case1 case2 case3; do
  venv311/bin/python scripts/plot_velocity_norm_summary.py \
    --run-dir "outputs/${case}_air1_publish" \
    --output-dir "outputs/${case}_air1_publish" \
    --v-ref-1 0.23 \
    --v-ref-2 7.0
done


# Velocity area logxy with marginal pdfs
# 2) Plot logxy + marginal PDFs for v_ref_1
for case in case1 case2 case3; do
  venv311/bin/python scripts/plot_velocity_area_logxy_with_marginal_pdfs.py \
    --run-dir "outputs/${case}_air1_publish" \
    --x-column max_v_over_v_ref_1 \
    --v-ref-index 1 \
    --v-ref-value 0.23 \
    --output "outputs/${case}_air1_publish/velocity_area_scatter_logxy_vref1_with_pdfs.png"
done

# 3) Plot logxy + marginal PDFs for v_ref_2
for case in case1 case2 case3; do
  venv311/bin/python scripts/plot_velocity_area_logxy_with_marginal_pdfs.py \
    --run-dir "outputs/${case}_air1_publish" \
    --x-column max_v_over_v_ref_2 \
    --v-ref-index 2 \
    --v-ref-value 7.0 \
    --output "outputs/${case}_air1_publish/velocity_area_scatter_logxy_vref2_with_pdfs.png"
done

# 4) Regenerate per-bubble max-velocity summary CSVs (now includes lineage fields)
venv311/bin/python scripts/export_bubble_max_velocity_summary.py --output-dir outputs



# self chunked runs

/home/aaron/Downloads/bubble_tracking/venv311/bin/python scripts/run_sam2_self_prompted_flow.py \
  --frames data/bubbles/case2_air1_publish/frames_0p5 \
  --reference data/bubbles/case2_air1_publish/frames_0p5/000001.jpg \
  --output outputs/case2_air1_publish_self_prompt160_manual \
  --roi data/interim/case2_air1/roi_0p5.json \
  --roi-mask data/interim/case2_air1/roi_mask_0p5.png \
  --seed-mask outputs/case2_air1_publish_debug/base_000002_prompt_000002_mask.png \
  --seed-frame-idx 2 \
  --chunk-interval 160 \
  --sam2-checkpoint checkpoints/sam2_hiera_base_plus.pt \
  --sam2-model-cfg sam2_hiera_b+.yaml \
  --sam2-device cuda \
  --use-flow \
  --velocity-mode centroid_delta \
  --preprocess-mode raw \
  --preprocess-raw-sigma 1.0 \
  --debounce-birth-frames 2 \
  --debounce-death-frames 3 \
  --fps 4 \
  --mm-per-pixel 0.08695 \
  --log-every 80


# gdino text-prompted keyframes (baseline comparison vs cv-keyframe and self-reprompt)

/home/aaron/Downloads/bubble_tracking/venv311/bin/python scripts/run_gdino_prompted_flow.py \
  --frames data/bubbles/case2_air1_publish/frames_0p5 \
  --reference data/bubbles/case2_air1_publish/frames_0p5/000001.jpg \
  --output outputs/case2_air1_publish_gdino_keyframe160 \
  --roi data/interim/case2_air1/roi_0p5.json \
  --roi-mask data/interim/case2_air1/roi_mask_0p5.png \
  --sam2-checkpoint checkpoints/sam2_hiera_base_plus.pt \
  --sam2-model-cfg sam2_hiera_b+.yaml \
  --sam2-device cuda \
  --gdino-model-id IDEA-Research/grounding-dino-tiny \
  --gdino-text-prompt "round bubble." \
  --gdino-box-threshold 0.25 \
  --gdino-text-threshold 0.25 \
  --gdino-keyframe-interval 160 \
  --gdino-keyframe-start-idx 2 \
  --gdino-min-area 50 \
  --gdino-keyframe-debug-dir outputs/case2_air1_publish_gdino_keyframe160_debug \
  --use-flow \
  --velocity-mode centroid_delta \
  --preprocess-mode raw \
  --preprocess-raw-sigma 1.0 \
  --debounce-birth-frames 2 \
  --debounce-death-frames 3 \
  --fps 4 \
  --mm-per-pixel 0.08695 \
  --log-every 80


### --- zero degree ---

cd /home/aaron/Downloads/bubble_tracking

VENV=/home/aaron/Downloads/bubble_tracking/venv311/bin/python
RAW=/home/aaron/Downloads/20250326_0degree_1
CASE=zero_degree
DATA_ROOT=data/bubbles/${CASE}_publish
INTERIM_ROOT=data/interim/${CASE}
RUN_DIR=outputs/${CASE}_publish
DEBUG_DIR=outputs/${CASE}_publish_debug

FPS=4
MM_PER_PIXEL=0.08695   # set this to your calibrated value
TITLE="0 Degree-Air1"

# 1) Prepare dataset (frames + frames_0p5, numeric naming)
$VENV - <<'PY'
from pathlib import Path
import re, cv2
raw = Path("/home/aaron/Downloads/20250326_0degree_1")
out = Path("data/bubbles/zero_degree_publish")
frames = out / "frames"
frames_0p5 = out / "frames_0p5"
frames.mkdir(parents=True, exist_ok=True)
frames_0p5.mkdir(parents=True, exist_ok=True)

exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
imgs = [p for p in raw.iterdir() if p.is_file() and p.suffix.lower() in exts]
def key(p):
    nums = re.findall(r"\d+", p.stem)
    return (int(nums[-1]) if nums else 10**12, p.name.lower())
imgs = sorted(imgs, key=key)

n = 0
for i, p in enumerate(imgs, start=1):
    im = cv2.imread(str(p))
    if im is None:
        continue
    name = f"{i:06d}.jpg"
    cv2.imwrite(str(frames / name), im, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    h, w = im.shape[:2]
    half = cv2.resize(im, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(frames_0p5 / name), half, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    n += 1
print(f"Wrote {n} frames to {out}")
PY

# 2) Create ROI files on 0.5x frames (interactive)
$VENV scripts/select_roi.py \
  --frames ${DATA_ROOT}/frames_0p5 \
  --frame-idx 1 \
  --out ${INTERIM_ROOT}/roi_0p5.json

$VENV scripts/select_polygon_roi.py \
  --frames ${DATA_ROOT}/frames_0p5 \
  --frame-idx 1 \
  --out-json ${INTERIM_ROOT}/roi_poly_0p5.json \
  --out-mask ${INTERIM_ROOT}/roi_mask_0p5.png

# 3a) split to two subsets
$VENV - <<'PY'
from pathlib import Path
import os

src = Path("data/bubbles/zero_degree_publish/frames_0p5")
p1 = Path("data/bubbles/zero_degree_publish_part1/frames_0p5")
p2 = Path("data/bubbles/zero_degree_publish_part2/frames_0p5")
split = 1280

p1.mkdir(parents=True, exist_ok=True)
p2.mkdir(parents=True, exist_ok=True)

files = sorted(src.glob("*.jpg"))
for i, f in enumerate(files, start=1):
    dst = p1 / f.name if i <= split else p2 / f.name
    if not dst.exists():
        os.link(f, dst)
print(f"part1={len(list(p1.glob('*.jpg')))} part2={len(list(p2.glob('*.jpg')))}")
PY

# 3) Run pipeline (same style as case1/2/3 publish)
$VENV scripts/run_sam2_prompted_flow.py \
  --frames ${DATA_ROOT}/frames_0p5 \
  --reference ${DATA_ROOT}/frames_0p5/000001.jpg \
  --output ${RUN_DIR} \
  --roi ${INTERIM_ROOT}/roi_0p5.json \
  --roi-mask ${INTERIM_ROOT}/roi_mask_0p5.png \
  --prompt-mode sam2 \
  --sam2-checkpoint checkpoints/sam2_hiera_base_plus.pt \
  --sam2-model-cfg sam2_hiera_b+.yaml \
  --sam2-device cuda \
  --cv-keyframe-prompts \
  --cv-keyframe-interval 160 \
  --cv-keyframe-start-idx 0 \
  --cv-keyframe-preprocess-mode default \
  --cv-keyframe-diff-sigma 0.5 \
  --cv-keyframe-no-normalize \
  --cv-keyframe-min-area 50 \
  --cv-keyframe-threshold-offset 3 \
  --cv-keyframe-debug-dir ${DEBUG_DIR} \
  --use-flow \
  --velocity-mode centroid_delta \
  --preprocess-mode raw \
  --preprocess-raw-sigma 1.0 \
  --debounce-birth-frames 2 \
  --debounce-death-frames 3 \
  --fps ${FPS} \
  --mm-per-pixel ${MM_PER_PIXEL}

## stuck at keyframes, it has missing bubble area, lowering offset < 3 will produce false positive bubbles

### attempted fix 1: contrast normalization
$VENV scripts/run_sam2_prompted_flow.py \
  --frames ${DATA_ROOT}/frames_0p5 \
  --reference ${DATA_ROOT}/frames_0p5/000001.jpg \
  --output ${RUN_DIR} \
  --roi ${INTERIM_ROOT}/roi_0p5.json \
  --roi-mask ${INTERIM_ROOT}/roi_mask_0p5.png \
  --prompt-mode sam2 \
  --sam2-checkpoint checkpoints/sam2_hiera_base_plus.pt \
  --sam2-model-cfg sam2_hiera_b+.yaml \
  --sam2-device cuda \
  --cv-keyframe-prompts \
  --cv-keyframe-interval 160 \
  --cv-keyframe-start-idx 0 \
  --cv-keyframe-preprocess-mode default \
  --cv-keyframe-diff-sigma 0.5 \
  --cv-keyframe-disable-light-map \
  --cv-keyframe-min-area 50 \
  --cv-keyframe-threshold-offset 3 \
  --cv-keyframe-debug-dir "${DEBUG_DIR}_contrast_normalize" \
  --use-flow \
  --velocity-mode centroid_delta \
  --preprocess-mode raw \
  --preprocess-raw-sigma 1.0 \
  --debounce-birth-frames 2 \
  --debounce-death-frames 3 \
  --fps ${FPS} \
  --mm-per-pixel ${MM_PER_PIXEL}

# this already improves the result. No false positives. Still has bubble that's 

### Attempted fix 2: light map noramlization + contrast normalization
$VENV scripts/run_sam2_prompted_flow.py \
  --frames ${DATA_ROOT}/frames_0p5 \
  --reference ${DATA_ROOT}/frames_0p5/000001.jpg \
  --output ${RUN_DIR} \
  --roi ${INTERIM_ROOT}/roi_0p5.json \
  --roi-mask ${INTERIM_ROOT}/roi_mask_0p5.png \
  --prompt-mode sam2 \
  --sam2-checkpoint checkpoints/sam2_hiera_base_plus.pt \
  --sam2-model-cfg sam2_hiera_b+.yaml \
  --sam2-device cuda \
  --cv-keyframe-prompts \
  --cv-keyframe-interval 160 \
  --cv-keyframe-start-idx 0 \
  --cv-keyframe-preprocess-mode default \
  --cv-keyframe-diff-sigma 0.5 \
  --cv-keyframe-min-area 50 \
  --cv-keyframe-threshold-offset 3 \
  --cv-keyframe-debug-dir "${DEBUG_DIR}_contrast_lightmap_normalize" \
  --use-flow \
  --velocity-mode centroid_delta \
  --preprocess-mode raw \
  --preprocess-raw-sigma 1.0 \
  --debounce-birth-frames 2 \
  --debounce-death-frames 3 \
  --fps ${FPS} \
  --mm-per-pixel ${MM_PER_PIXEL}

# Open gap fixed

## Two-part runs


# 4) 1x / 10x tracking videos
$VENV scripts/export_publishing_tracking_video.py \
  --run-dir ${RUN_DIR} \
  --frames ${DATA_ROOT}/frames_0p5 \
  --title "${TITLE}"

# 5) Visualizations
$VENV scripts/export_velocity_overlay.py \
  --run-dir ${RUN_DIR} \
  --frames ${DATA_ROOT}/frames_0p5 \
  --output ${RUN_DIR}/velocity_overlay.mp4 \
  --fps ${FPS} \
  --label-units mm_s

$VENV scripts/plot_vanished_area_moving_window.py \
  --run-dir ${RUN_DIR}

$VENV scripts/plot_velocity_norm_summary.py \
  --run-dir ${RUN_DIR} \
  --output-dir ${RUN_DIR} \
  --v-ref-1 0.23 \
  --v-ref-2 7.0

$VENV scripts/plot_velocity_area_logxy_with_marginal_pdfs.py \
  --run-dir ${RUN_DIR} \
  --x-column max_v_over_v_ref_1 \
  --v-ref-index 1 \
  --v-ref-value 0.23 \
  --output ${RUN_DIR}/velocity_area_scatter_logxy_vref1_with_pdfs.png

$VENV scripts/plot_velocity_area_logxy_with_marginal_pdfs.py \
  --run-dir ${RUN_DIR} \
  --x-column max_v_over_v_ref_2 \
  --v-ref-index 2 \
  --v-ref-value 7.0 \
  --output ${RUN_DIR}/velocity_area_scatter_logxy_vref2_with_pdfs.png

# 6) Escape area table (same 16-column format as case1/2/3 threshold variant)
$VENV - <<'PY'
from pathlib import Path
import scripts.export_three_escaping_subcount_timeseries as t

case_key = "case0degree"
run_dir = Path("outputs/zero_degree_publish")
out_csv = Path("outputs/zero_degree_time_series_subcount_three_escaping_situation_threshold_50%_and_5mm2.csv")

t.MM_PER_PIXEL_BY_CASE[case_key] = 0.08695  # use same calibrated value as run
t._export_case(
    case_key=case_key,
    run_dir=run_dir,
    output_csv=out_csv,
    ratio_threshold=0.5,
    mm2_threshold=5.0,
)
print(f"Wrote: {out_csv}")
PY
