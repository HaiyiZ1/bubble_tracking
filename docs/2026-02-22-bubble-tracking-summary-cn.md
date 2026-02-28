# 气泡跟踪管线说明（中文简版）

## 现状与问题
- **经典 reference-diff 分割**：基于零气泡参考帧做差分+阈值分割，噪声无法可靠去除（光照变化、轻微纹理会被当作气泡）。
- **SAM2 自动跟踪**：在前若干帧表现良好，但在 N 帧后容易丢失目标；后续新出现气泡无法自动被发现。
- **下一步方向**：需要更鲁棒的 prompt 策略，计划用**经典分割做“再发现”**，并把新气泡作为提示重新启动 SAM2 传播。

## Pipeline 实现（当前版本）
1) **预处理**  
   - 使用零气泡参考帧；可选 CLAHE/去噪等（用于光照归一化）。
   - ROI 限定处理区域（支持矩形 ROI 或多边形 mask）。
2) **光流计算**  
   - 使用 Farneback（CPU 或 CUDA）计算每帧速度场。
3) **分割（SAM2）**  
   - `sam2 auto-search` 在 ROI 内搜索提示点/区域，找到目标后开始传播。
   - `propagate` 在整段视频中生成每帧掩膜。
4) **匹配与事件**  
   - 基于掩膜与光流进行帧间关联，统计 birth/death/merge/split。
5) **输出统计**  
   - 记录每个气泡的面积、速度向量与时间序列。

## SAM2 使用方式
- 采用 **auto-search + propagate** 流程；当前版本**不做自动再发现**。  
  → 因此新气泡出现后可能无法被追踪，且在长序列中容易丢失。
- ROI 能减少误检与加速。
- 可能出现 OOM（全分辨率视频过大），建议**下采样后再运行**（例如 0.5x）。

## 输出与结果指标
### 1) 时间轴（Time Frame）
- `tracks.csv` / `tracks.json`：每帧包含 `frame_idx` 与 `time_s`。
- `time_s = frame_idx / fps`。

### 2) 单气泡面积（Per-bubble Area）
来自 `tracks.csv` / `tracks.json`：
- `area_px`：像素面积
- `area_mm2`：若提供 `--mm-per-pixel`，会给出物理单位面积

### 3) 速度向量（Moving Velocity Vector）
来自 `tracks.csv` / `tracks.json`：
- `velocity_x_px_s`, `velocity_y_px_s`, `velocity_mag_px_s`
- 若提供 `--mm-per-pixel`，同时有 `velocity_*_mm_s`

### 4) 总气泡面积随时间变化
来自 `area_timeseries.csv` / `area_timeseries.json`：
- 每帧总面积（气泡掩膜并集面积）
- 可按秒聚合为曲线（见下方“按秒汇总”）

### 5) merge/split 频率随时间变化
来自 `events.json`：
- 记录 `merge`, `split`, `birth`, `death` 事件与帧号
- 可按秒统计频率

## 按秒汇总口径（建议）
使用 `second = floor(time_s)` 作为桶：
- **merge/split 频率**：每秒事件计数（merge_count/s, split_count/s）
- **总面积变化**：对 `area_timeseries` 按秒取平均或最大值

## 下一步计划
- **结合经典分割进行再发现**：在传播期间定期或触发式运行 reference-diff 分割，
  把新气泡作为 prompt 追加给 SAM2，再继续传播。
- 目标：解决后期新气泡出现与丢失问题，提高长序列鲁棒性。
