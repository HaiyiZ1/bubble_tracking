# GPU Farneback Flow Acceleration (2026-02-22)

## Goal
Speed up optical flow computation by using OpenCV CUDA Farneback when available, with a safe CPU fallback.

## Scope
- Only replace the optical flow step with a CUDA-backed implementation.
- Keep preprocessing, segmentation, event detection, and outputs unchanged.
- Add runtime metadata to report which backend actually ran.

## Approach
- Add a small flow backend helper that selects between CUDA Farneback and CPU Farneback.
- Default backend is `auto`: if CUDA is available and OpenCV exposes `cv2.cuda_FarnebackOpticalFlow`, use GPU; otherwise warn once and fall back to CPU.
- Provide `--flow-backend {auto,cpu}` and `--cuda-device` CLI flags to control behavior.

## Error Handling
- If CUDA init fails (missing module, no device, invalid device index, or `cv2.error`), print a single warning and switch to CPU for the rest of the run.
- If CUDA flow fails mid-run, fall back to CPU and continue.

## Outputs
- Record actual backend in `events.json` under a `runtime` section:
  - `flow_backend` (cuda/cpu)
  - `flow_backend_requested` (auto/cpu)
  - `cuda_device` (integer)

## Testing
- Run a short subset with `--flow-backend auto` and confirm `events.json` reports `flow_backend=cuda` when CUDA is available.
- Run with `--flow-backend cpu` to ensure forced CPU behavior still works.
