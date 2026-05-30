#!/usr/bin/env bash
# Run the 12-cell matrix: {b+, large} x {extras yes/no} x {min_area 428, 68, 27}
# All runs: full 1677 frames, threshold_offset=6, --no-video, offset=6.
# Large rows use --sam2-offload-video-to-cpu to avoid OOM.
set -euo pipefail
cd /home/aaron/Downloads/bubble_tracking

PY=/home/aaron/Downloads/bubble_tracking/venv311/bin/python
FRAMES=data/bubbles/case1_air1_publish/frames_0p5
REF=${FRAMES}/000001.jpg
ROI=data/interim/bubbles_air1_072_271/roi_0p5.json
ROIM=data/interim/bubbles_air1_072_271/roi_mask_0p5.png
FPS=4
MMPP=0.08563
EXTRAS="228 229 230 231 232 233 234 235 236"   # original 299-307

# Pre-flight: scrap superseded 1300-frame and threshold_offset=7 runs.
rm -rf \
  outputs/case1_air1_publish_3_14_l outputs/case1_air1_publish_3_14_l_debug \
  outputs/case1_air1_publish_0_5_l outputs/case1_air1_publish_0_5_l_debug \
  outputs/case1_air1_publish_full_l_offload outputs/case1_air1_publish_full_l_offload_debug

run_cell () {
  local model="$1"          # l | bp
  local extras="$2"         # yes | no
  local minarea="$3"        # 428 | 68 | 27
  local tag="matrix_${model}_${extras}_min${minarea}_off6"
  local out="outputs/${tag}"
  local dbg="outputs/${tag}_debug"

  local ckpt cfg offload_arg
  if [[ "$model" == "l" ]]; then
    ckpt=checkpoints/sam2_hiera_large.pt; cfg=sam2_hiera_l.yaml
    offload_arg="--sam2-offload-video-to-cpu"
  else
    ckpt=checkpoints/sam2_hiera_base_plus.pt; cfg=sam2_hiera_b+.yaml
    offload_arg=""
  fi

  local extras_arg=""
  [[ "$extras" == "yes" ]] && extras_arg="--cv-keyframe-extra-indices ${EXTRAS}"

  echo "===== $(date '+%H:%M:%S') START ${tag} ====="
  SECONDS=0
  $PY scripts/run_sam2_prompted_flow.py \
    --frames "$FRAMES" \
    --reference "$REF" \
    --output "$out" \
    --roi "$ROI" \
    --roi-mask "$ROIM" \
    --prompt-mode sam2 \
    --sam2-checkpoint "$ckpt" \
    --sam2-model-cfg "$cfg" \
    --sam2-device cuda \
    $offload_arg \
    --cv-keyframe-prompts \
    --cv-keyframe-interval 160 \
    --cv-keyframe-start-idx 10 \
    --cv-keyframe-preprocess-mode default \
    --cv-keyframe-diff-sigma 0.5 \
    --cv-keyframe-no-normalize \
    --cv-keyframe-min-area "$minarea" \
    --cv-keyframe-threshold-offset 6 \
    --cv-keyframe-debug-dir "$dbg" \
    $extras_arg \
    --no-video \
    --use-flow \
    --velocity-mode centroid_delta \
    --preprocess-mode raw \
    --preprocess-raw-sigma 1.0 \
    --debounce-birth-frames 2 \
    --debounce-death-frames 3 \
    --fps $FPS \
    --mm-per-pixel $MMPP
  echo "===== $(date '+%H:%M:%S') DONE  ${tag} (${SECONDS}s) ====="
  sleep 5
}

# 12 rows, base+ first (faster) so large failures fail late.
run_cell bp  no  428
run_cell bp  yes 428
run_cell bp  no  68
run_cell bp  yes 68
run_cell bp  no  27
run_cell bp  yes 27
run_cell l   no  428
run_cell l   yes 428
run_cell l   no  68
run_cell l   yes 68
run_cell l   no  27
run_cell l   yes 27

echo "ALL DONE"
