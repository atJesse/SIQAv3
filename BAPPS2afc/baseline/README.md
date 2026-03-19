# BAPPS 2AFC Baseline Metrics

Evaluate BAPPS `2afc/val` with baseline quality metrics:
- `PSNR`
- `SSIM`
- `FSIM`
- `VIF`
- `LPIPS`
- `DISTS`

## 2AFC rule
For each triplet `(ref, p0, p1)`:
- Human choice uses `judge_p1`: `1 if judge_p1 >= 0.5 else 0`
- Metric choice:
  - High-better metrics (`PSNR/SSIM/FSIM/VIF`): choose `0` if `score_p0 > score_p1` else `1`
  - Low-better metrics (`LPIPS/DISTS`): choose `0` if `score_p0 < score_p1` else `1`
- 2AFC accuracy = proportion of metric choice matching human choice.

## Run
```bash
source /data/miniforge3/etc/profile.d/conda.sh
conda activate lovif
cd /data/SIQAv3
python BAPPS2afc/baseline/eval_bapps_2afc_baselines.py \
  --bapps_root /data/dataset/BAPPS/dataset/2afc/val \
  --out_dir /data/SIQAv3/BAPPS2afc/baseline/results
```

## Outputs
- `bapps_2afc_metrics_all_pairs.csv`
- `bapps_2afc_metrics_subset_summary.csv`
- `bapps_2afc_metrics_overall_summary.csv`
- `bapps_2afc_metrics_summary.json`
- `readme.txt`
