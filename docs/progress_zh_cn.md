# 当前进度与发现（简体中文）

## 概览

本项目 Phase 1 目标是验证并跑通基于 **SAM2 (Segment Anything Model 2)** 的视频分割/跟踪流程，并用公开数据集做初步复现与定量/定性验证。目前我们已经完成：

- **SAM2 基础管线跑通**：帧序列输入 → 提示（prompt）初始化 → 视频传播 → 输出 mask、`tracks.json`、统计与可视化视频 overlay。
- **基准/生物数据集验证已运行**：
  - DAVIS 2017（VOS 基准）
  - CTC HeLa（Cell Tracking Challenge，细胞序列）
- 记录并确认了 SAM2 在“无标注/目标会分裂或新生”的场景下的关键局限。
- 评估了 CPU-only 的实际耗时，结论是 **CPU 运行不具备工程可行性**，需要迁移到 **Colab/GPU**。

---

## 运行环境与基础事实

- **当前机器**：CPU-only（无 CUDA GPU）
- **SAM2 输入帧格式限制**：官方视频加载器默认只读取 `.jpg/.jpeg`
  - 因此我们已将合成数据与抽帧输出统一为 **JPEG**。
- **Mask “看起来全黑”**：mask 文件是 **16-bit (uint16)** 的实例 ID 图；例如 object id=128 时，在 0~65535 的显示范围内会显得非常暗，视觉上像“全黑”，但数据是有效的。

---

## 结果记录与解释

### 1) SAM2 合成数据（synthetic）跑通

- 输入：合成的 50 帧序列（3 个移动圆形目标）
- 输出：`outputs/test_run/`
  - `masks/`（每帧实例 mask）
  - `tracks.json`（每个 object 每帧的 centroid/area/bbox）
  - `stats.json`（统计）
  - `overlay.mp4`（可视化）
- CPU 耗时：约 **~2 分钟 / 50 帧**（量级参考）

解释：
- 该步骤主要用于验证**工程链路**：提示→传播→落盘→可视化→统计都能工作。

---

### 2) DAVIS 2017（SAM2官方/VOS 基准）

- 运行命令（示例）：

```bash
python scripts/run_biomed_repro.py --dataset davis2017
```

- 默认跑 1 个序列（例如 `bike-packing`），输出路径：
  - `outputs/biomed_repro/davis2017/<sequence>/masks/`
  - `outputs/biomed_repro/davis2017/<sequence>/tracks.json`
  - `outputs/biomed_repro/davis2017/<sequence>/overlay.mp4`
  - 汇总：`outputs/biomed_repro/davis2017/results.json`

- 结果（单序列示例 `bike-packing`）：
  - **J = 0.438**
  - **F = 0.439**
  - **J&F = 0.438**

解释（为什么与论文常见值差距较大）：
- 我们使用的是 **CPU + tiny checkpoint**，且属于最小化复现（非官方完整评测配置）。
- 早期我们使用 bbox prompt，后续已切换为 **first-frame mask prompt（更接近标准 semi-supervised VOS 协议）**，这通常能提高质量；但 CPU+tiny 仍可能明显低于论文报告的 GPU/更大模型结果。
- 我们修复了 DAVIS GT 的读取：DAVIS 的标注是 **palette/index PNG**，如果按 RGB 读会导致评估维度错误。

---

### 3) CTC HeLa（细胞跟踪挑战数据）

- 运行命令（示例）：

```bash
python scripts/run_biomed_repro.py --dataset ctc-hela --sequence 01
```

- 输出路径：
  - `outputs/biomed_repro/ctc-hela/01/masks/`
  - `outputs/biomed_repro/ctc-hela/01/tracks.json`
  - `outputs/biomed_repro/ctc-hela/01/overlay.mp4`
  - 汇总：`outputs/biomed_repro/ctc-hela/results.json`

- 结果（`01` 序列）：
  - **SEG = 0.027（84 frames）**
  - CPU 耗时：约 **~9 分钟 / 84 帧**（约 6~7 秒/帧量级）

关键现象与解释：
- **跟踪崩溃（collapse）**：在该 run 中，我们观察到从 **frame 34** 开始预测 mask 变为全 0（背景），之后持续丢失（tracks 中对应 frame 的 detections 变为 0）。
  - 这说明问题不只是“新生细胞没有自动分配 ID”，而是 **传播过程整体失败**（模型失去目标后无法自恢复）。
- **细胞分裂/新生（mitosis）不会自动产生新 track**：
  - SAM2 是 **prompt-based**：它会传播你初始化的目标，但不会“凭空发现新对象并自动编号”。
  - 对于会分裂/新生的目标（细胞、气泡的产生/消失/分裂/合并），需要额外的**轨迹管理层**来处理“birth/split/merge”。

CTC 评估注意事项：
- CTC 的 `SEG/` 标注在该数据中是 **稀疏帧**，且与 `TRA/` 的 tracking mask 可能不对齐。
- 当前我们倾向使用 `TRA/` 作为评估基准（更贴近 tracking 验证）。

---

## SAM2 的关键局限（对气泡数据很重要）

1. **不是完全无监督（unsupervised）**
   - 必须有 prompt（点/框/mask）告诉模型“追踪什么”。
   - 对无标注气泡数据，需要：
     - 人工在首帧点击/框选（半自动），或
     - 自动生成 prompt（例如传统 CV blob 检测、阈值/边缘/连通域、或训练一个检测器）。

2. **不会自动处理新生目标 / 分裂产生的新对象**
   - 分裂后“新对象”不会自动分配新 ID。

3. **传播可能崩溃且不会自动恢复**
   - 如 CTC HeLa `01` 在 frame 34 后 mask 变全 0。
   - 需要工程上增加“失败检测 + 自动重初始化”的机制：
     - 例如检测 total mask area→0、或对象数量骤降等；
     - 触发自动重提示（auto-prompt）并继续传播。

---

## CPU 运行不可行：需要迁移到 Colab/GPU

从实际耗时看：
- 合成数据：~2 分钟 / 50 帧
- DAVIS：~1 分钟级 / 70 帧（不同序列略有差异）
- CTC HeLa：~9 分钟 / 84 帧（更慢）

对真实气泡数据（更长序列、更高分辨率、更多目标）而言，CPU 运行会非常慢，无法迭代。

结论：
- 后续需要迁移到 **Google Colab（或本地/远程 GPU）** 来跑：
  - 更长序列、更大模型 checkpoint
  - 更频繁的实验迭代（尤其是加入自动重初始化与 split/merge 逻辑）

---

## 下一步建议（面向气泡数据）

- **(1) 设计“自动 prompt 生成”**：面向 bubble 的图像特性，先做轻量级 blob/边缘/阈值/连通域候选。
- **(2) 增加“失败检测 + 自动重初始化”**：解决像 frame 34 这种 collapse。
- **(3) 增加 split/merge/birth/death 事件建模**：输出更符合需求的“分裂/合并图结构”。
- **(4) 迁移到 Colab/GPU**：保证迭代速度与可扩展性。


