# Progress Logging (2026-02-22)

## Goal
Provide lightweight per-frame progress output with stage and processed/total frames.

## Scope
- Print one line every N frames (and the last frame).
- Avoid extra dependencies (no tqdm).
- Allow disabling via CLI.

## Approach
- Add `log_every` to `TrackingConfig` with a CLI flag `--log-every`.
- Track total frames and emit a single line per interval:
  - Example: `[progress] stage=flow+match frame 50/200`
- Use `segment` stage for the first frame and `flow+match` when optical flow and matching run.

## Error Handling
- If `log_every <= 0`, skip logging.
- Logging is best-effort (no impact on pipeline outputs).

## Testing
- Run a short subset with `--log-every 5` and confirm progress lines appear.
- Run with `--log-every 0` to ensure logging is suppressed.
