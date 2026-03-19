# Experimental Results Summary (LoViF, BAPPS, PieAPP)

This document summarizes the currently completed experiments in this workspace and is intended as a paper-writing draft.

## 1. Experiment Scope and Checkpoints

- **Main model family**: SIQA dual-backbone FR-IQA (DINOv3 + CLIP).
- **LoViF protocol**: 5-fold stratified evaluation on internal LoViF/SIQA setting.
- **BAPPS protocol**: 2AFC validation split, hard accuracy + soft agreement.
- **PieAPP protocol**: official test split (`40` references, `600` distorted images), report `SROCC / KRCC / PLCC`, with additional `5PL` mapping and `|PLCC(5PL)|`.

Primary result files:
- LoViF: `/data/SIQAdn2/workdirs/siqa_dual_dinov3_clip/fold_*/oof_metrics.json`
- BAPPS (model): `/data/SIQAv3/BAPPS2afc/results/bapps_2afc_summary.json`
- BAPPS (classical baselines): `/data/SIQAv3/BAPPS2afc/baseline/results/bapps_2afc_metrics_summary.json`
- PieAPP: `/data/SIQAv3/PieAPP2afc/results/pieapp_test_metrics.json`

---

## 2. Main Quantitative Results

### 2.1 LoViF (5-fold OOF)

From 5 fold-level `oof_metrics.json` files:

| Metric | Mean | Std |
|---|---:|---:|
| SROCC | 0.8570 | 0.0265 |
| PLCC | 0.8673 | 0.0204 |
| Score (0.6*SROCC + 0.4*PLCC) | 0.8611 | 0.0241 |
| MAE | 0.6293 | 0.0543 |
| RMSE | 0.8146 | 0.0481 |

Per-fold `score` values: `0.8661, 0.8837, 0.8753, 0.8656, 0.8149`.

### 2.2 BAPPS (2AFC, validation split)

Main SIQA model (checkpoint used in script: `fold_1/checkpoints/best.pth`):

- Overall **2AFC accuracy**: **0.68435**
- Overall **soft agreement**: **0.64991**
- Total pairs: `36,344`

Per-subset 2AFC accuracy:

| Subset | Accuracy |
|---|---:|
| cnn | 0.83284 |
| traditional | 0.77754 |
| superres | 0.69731 |
| frameinterp | 0.64778 |
| color | 0.60297 |
| deblur | 0.59661 |

Baseline metrics (same BAPPS split):

| Baseline | Overall 2AFC Accuracy |
|---|---:|
| DISTS | 0.72293 |
| LPIPS | 0.69910 |
| FSIM | 0.67139 |
| PSNR | 0.66660 |
| SSIM | 0.64096 |
| VIF | 0.58133 |

Observation: current SIQA setting on BAPPS is below best baseline (`DISTS`) by `0.03858` absolute 2AFC accuracy.

### 2.3 PieAPP (official test split)

From `/data/SIQAv3/PieAPP2afc/results/pieapp_test_metrics.json`:

- `num_references = 40`
- `num_samples = 600`
- Direction alignment used: `pred_error = -pred_quality` (to match PieAPP error-style labels)

Final metrics:

| Metric | Value |
|---|---:|
| SROCC | 0.59562 |
| KRCC | 0.42753 |
| PLCC (raw) | 0.64096 |
| PLCC (after 5PL) | 0.64105 |
| \|PLCC(5PL)\| | 0.64105 |

---

## 3. Interpretation for Paper Writing

1. **LoViF in-domain performance is strong and stable** across folds (`SROCC ~0.86`, `PLCC ~0.87`), supporting the effectiveness of the dual-backbone + semantic gating design on the target training domain.
2. **Cross-dataset generalization remains moderate** on PieAPP and BAPPS, indicating a non-trivial domain gap between LoViF/SIQA training distribution and perceptual distortions in standard IQA benchmarks.
3. **On BAPPS, hand-crafted/perceptual baselines are still competitive**, especially `DISTS`; this suggests future work should further improve pairwise preference calibration and distortion-type coverage.
4. **5PL mapping has limited gain on PieAPP PLCC** in current setup (very small change from raw PLCC), implying that the main bottleneck is likely representation/domain mismatch rather than monotonic score calibration.

---

## 4. Reproducibility Notes

- BAPPS and PieAPP were evaluated with the checkpoint:
  - `/data/SIQAdn2/workdirs/siqa_dual_dinov3_clip/fold_1/checkpoints/best.pth`
- BAPPS summary is based on hard 2AFC decision metric (`argmax`-style pairwise choice), plus soft agreement.
- PieAPP PLCC was reported both before and after `5PL`; paper-ready value can follow your requested convention `|PLCC(5PL)|`.

If needed, this summary can be converted directly into a **paper table format** (LaTeX `tabular`) in the next step.
