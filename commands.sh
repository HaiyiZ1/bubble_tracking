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
  --output outputs/case3_air1_sam2_cv_keyframe100_nonorm_centroid_delta \
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