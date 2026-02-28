# Phase 1.2: Biomedical SAM2 Paper Selection

## Selection Criteria

We evaluated candidate papers based on:

1. **Public dataset** with clear train/val/test splits
2. **Clear prompting strategy** described in the paper
3. **Code released** or sufficient detail to reimplement
4. **Standard metrics** (Dice, IoU, J&F)
5. **CPU-friendly** subset available for smoke testing

## Candidates Evaluated

### 1. Medical SAM 2 (MedSAM2)
- **Paper**: "Medical SAM 2: Segment Medical Images as Video via Segment Anything Model 2"
- **Repository**: https://github.com/ImprintLab/Medical-SAM2
- **Datasets**: REFUGE (fundus imaging), BTCV (CT organs)
- **Metrics**: Dice, IoU
- **Pros**: Official code, multiple datasets, clear training pipeline
- **Cons**: Focuses on 2D/3D segmentation rather than video tracking

### 2. Biomedical-SAM-2
- **Paper**: SAM2 adaptation for biomedical imaging
- **Repository**: https://github.com/ZhilingYan/Biomedical-SAM-2
- **Datasets**: BUSI (Breast Ultrasound), Kvasir-SEG (polyp)
- **Metrics**: Dice, IoU
- **Pros**: Multiple modalities, click prompts
- **Cons**: Not video-based tracking

### 3. SAM2 on Cell Tracking Challenge (CTC)
- **Paper**: Various benchmarking studies
- **Datasets**: Cell Tracking Challenge sequences (DIC-C2DH-HeLa, PhC-C2DL-PSC, etc.)
- **Metrics**: SEG, TRA, AOGM (CTC standard metrics)
- **Pros**: True video tracking, standardized benchmarks, public datasets
- **Cons**: No single authoritative SAM2-based paper yet

### 4. SAM2 Video Object Segmentation Benchmarks
- **Paper**: Original SAM2 paper includes VOS benchmarks
- **Datasets**: DAVIS 2017, YouTube-VOS
- **Metrics**: J&F (Jaccard and F-measure)
- **Pros**: Well-established benchmark, part of SAM2 official evaluation
- **Cons**: Not biomedical (general video objects)

## Selected Approach: Two-Track Validation

Given our goal of validating SAM2's tracking accuracy for bubble tracking, we'll implement:

### Track A: DAVIS 2017 VOS Benchmark (Primary)
**Rationale**: This is the standard benchmark SAM2 was evaluated on. Reproducing results on DAVIS validates our pipeline against known ground truth before applying to biomedical data.

- **Dataset**: DAVIS 2017 val set (30 sequences)
- **Metrics**: J (Jaccard), F (boundary F-measure), J&F (mean)
- **Expected results**: J&F ~82-89% depending on SAM2 variant
- **Download**: https://davischallenge.org/

### Track B: Cell Tracking Challenge Subset (Biomedical)
**Rationale**: Closest to bubble tracking (microscopy, moving/dividing objects, temporal tracking). We'll evaluate SAM2 zero-shot on CTC sequences.

- **Dataset**: DIC-C2DH-HeLa (Phase contrast HeLa cells)
- **Metrics**: SEG (segmentation accuracy), TRA (tracking accuracy)
- **Expected**: Baseline comparison with CTC leaderboard methods
- **Download**: http://celltrackingchallenge.net/

## Implementation Plan

### Track A: DAVIS 2017

```bash
# Download DAVIS 2017 (validation set only for efficiency)
python scripts/get_biomed_dataset.py --dataset davis2017-val

# Run SAM2 with first-frame prompts (simulating semi-supervised VOS)
python scripts/run_biomed_repro.py --dataset davis2017 --split val

# Evaluate using official DAVIS toolkit
python -m bubble_tracking.eval.davis_eval --pred outputs/davis2017 --gt data/davis2017
```

### Track B: CTC HeLa

```bash
# Download CTC HeLa sequence
python scripts/get_biomed_dataset.py --dataset ctc-hela

# Run SAM2 with first-frame GT masks as prompts
python scripts/run_biomed_repro.py --dataset ctc-hela

# Evaluate using CTC metrics
python -m bubble_tracking.eval.ctc_eval --pred outputs/ctc-hela --gt data/ctc-hela
```

## Success Criteria

1. **DAVIS 2017**: Achieve J&F within 5% of reported SAM2 paper results
2. **CTC HeLa**: Produce segmentation overlays with reasonable temporal consistency
3. **Pipeline**: End-to-end reproduction completes on CPU with subset data

## References

- [SAM2 Paper](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/)
- [DAVIS Challenge](https://davischallenge.org/)
- [Cell Tracking Challenge](http://celltrackingchallenge.net/)
- [Medical SAM 2](https://github.com/ImprintLab/Medical-SAM2)

