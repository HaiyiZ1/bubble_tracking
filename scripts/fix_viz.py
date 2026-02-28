import json
from bubble_tracking.schemas import TrackingResult
from bubble_tracking.visualization import create_tracking_video

tracks = "outputs/bubbles_air1_072_172_auto_gpu/tracks.json"
with open(tracks, "r") as f:
    result = TrackingResult.from_dict(json.load(f))

create_tracking_video(
    frames_dir="data/interim/bubbles_air1_072_172/frames",
    masks_dir="outputs/bubbles_air1_072_172_auto_gpu/masks",
    result=result,
    output_path="outputs/bubbles_air1_072_172_auto_gpu/overlay_readable.mp4",
    fps=10,
    alpha=0.35,
    overlay_dir="outputs/bubbles_air1_072_172_auto_gpu/overlay_frames",
    label_scale=1.2,
    label_thickness=3,
)