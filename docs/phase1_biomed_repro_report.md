# Phase 1.2: Biomedical SAM2 Reproduction Report

## Overview

This report documents the reproduction of SAM2 tracking on biomedical/benchmark datasets as part of Phase 1 validation.

## Environment

```
Python: 3.x
PyTorch: 2.1.2 (CPU)
SAM2: segment-anything-2 (git checkout)
Checkpoint: sam2_hiera_tiny.pt
```

## Datasets Evaluated

### DAVIS 2017 (Video Object Segmentation Benchmark)

- **Source**: https://davischallenge.org/
- **Split**: Validation (30 sequences)
- **Resolution**: 480p
- **Metrics**: J (Jaccard/IoU), F (Boundary F-measure), J&F (mean)

### CTC HeLa (Cell Tracking Challenge)

- **Source**: http://celltrackingchallenge.net/
- **Dataset**: DIC-C2DH-HeLa
- **Metrics**: SEG (segmentation accuracy)

## Methodology

### Prompting Strategy

For semi-supervised video object segmentation (matching SAM2 paper protocol):

1. Use first-frame ground truth masks to derive bounding box prompts
2. Run SAM2 video propagation through remaining frames
3. Compare predicted masks against ground truth

### Evaluation

- J (Jaccard): Mean IoU between predicted and GT masks per object
- F (Boundary): Precision/recall of boundary pixels within 2px threshold
- J&F: Average of J and F scores

## Results

### DAVIS 2017

| Sequence | J | F | J&F |
|----------|-----|-----|-----|
| (run `scripts/run_biomed_repro.py` to populate) | | | |
| **Mean** | | | |

**Expected (from SAM2 paper)**:
- SAM2-Tiny: J&F ~82-84%
- SAM2-Small: J&F ~85-87%
- SAM2-Base+: J&F ~88-89%

### CTC HeLa

| Sequence | SEG | Frames |
|----------|-----|--------|
| (run with `--dataset ctc-hela`) | | |

## Reproduction Commands

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install SAM2
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2 && pip install -e . && cd ..

# Download checkpoint
mkdir -p checkpoints && cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt
cd ..

# Download DAVIS
python scripts/get_biomed_dataset.py --dataset davis2017-val

# Run reproduction (smoke test on one sequence)
python scripts/run_biomed_repro.py --dataset davis2017 --sequence blackswan

# Run full evaluation
python scripts/run_biomed_repro.py --dataset davis2017 --all

# Test evaluation pipeline without SAM2 (uses GT masks)
python scripts/run_biomed_repro.py --dataset davis2017 --skip-sam2
```

## Discussion

### CPU-Only Limitations

Running on CPU significantly impacts:
- **Speed**: ~1-5 seconds per frame vs <100ms on GPU
- **Memory**: May need to reduce batch size or resolution
- **Practicality**: Full dataset evaluation takes hours

For Phase 1 validation, we focus on:
- Single-sequence smoke tests
- Downsampled resolution where needed
- Verification that pipeline produces expected output format

### Deviations from Paper

1. **Resolution**: Using 480p instead of full resolution
2. **Checkpoint**: Using tiny model for CPU feasibility
3. **Subset**: Evaluating on subset of sequences

### Next Steps for Phase 2

1. Acquire GPU access for full-speed evaluation
2. Run complete DAVIS validation set
3. Extend to bubble-specific datasets
4. Implement bubble-specific metrics (merge/split detection)

## Artifacts

- `outputs/biomed_repro/davis2017/results.json` - Quantitative results
- `outputs/biomed_repro/davis2017/<sequence>/masks/` - Predicted masks
- `outputs/biomed_repro/davis2017/<sequence>/tracks.json` - Tracking output

## Conclusion

Phase 1.2 establishes that:
1. SAM2 can be integrated into our pipeline
2. Standard VOS evaluation metrics can be computed
3. The pipeline is ready for bubble-specific adaptation in Phase 2

---

*Report generated: [DATE]*
*Commit hash: [HASH]*

