# paperFigExample

This folder contains a script to generate figure-support data for one selected LoViF pair.

## What it does

Given one `(Ref, Dist)` pair, it computes:
- Baseline metrics: `PSNR`, `SSIM`, `FSIM`, `VIF`, `LPIPS_VGG_PIQ`, `DISTS`
- Your model score: `DSS-SQA` (from checkpoint)

Then it calibrates baseline raw scores to a `0-5` quality range using LoViF train set (`Train_scores.xlsx`) and exports figure-ready tables.

## Script

- `eval_single_pair_for_paper_fig.py`

## Default target pair

- Ref: `/data/dataset/LoViF/Train/Ref/000087.png`
- Dist: `/data/dataset/LoViF/Train/Dist/000087.png`

## Run

```bash
source /data/miniforge3/bin/activate lovif
python /data/SIQAv3/paperFigExample/eval_single_pair_for_paper_fig.py \
  --ref /data/dataset/LoViF/Train/Ref/000004.png \
  --dist /data/dataset/LoViF/Train/Dist/000004.png \
  --ckpt /data/SIQAdn2/workdirs/siqa_dual_dinov3_clip/fold_1/checkpoints/best.pth \
  --out_dir /data/SIQAv3/paperFigExample
```

## Outputs

- `single_pair_scores_for_plot.csv`: metric bars (`raw_score`, `mapped_0_5`)
- `single_pair_result.json`: full machine-readable output
- `metric_mapping_details.json`: mapping fit details and calibration correlations
- `single_pair_readout.md`: quick human-readable summary table

## Note on LPIPS variant

`LPIPS_VGG_PIQ` here uses `piq.LPIPS` (VGG-based LPIPS implementation).
