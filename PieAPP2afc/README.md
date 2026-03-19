# PieAPP Test Evaluation

This folder evaluates a trained SIQA checkpoint on PieAPP test split and reports:
- `SROCC`
- `KRCC`
- `PLCC(raw)`
- `PLCC(5PL)` and `|PLCC(5PL)|`

## Key protocol note
PieAPP `per_image_score` is an **error/dissimilarity** score (higher = worse), while SIQA model output here is quality-like (higher = better). So the script aligns direction by:

- `pred_error = -pred_quality`

Correlations are computed against `gt_error`.

## Run
```bash
source /data/miniforge3/bin/activate lovif
HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=0 \
python /data/SIQAv3/PieAPP2afc/eval_pieapp_test.py \
  --config /data/SIQAv3/configs/siqa_base.yaml \
  --ckpt /data/SIQAdn2/workdirs/siqa_dual_dinov3_clip/fold_1/checkpoints/best.pth \
  --pieapp_root /data/dataset/PieAPP/PieAPP_dataset_CVPR_2018 \
  --out_dir /data/SIQAv3/PieAPP2afc/results
```

## Outputs
- `results/pieapp_test_predictions.csv`: per-image prediction table
- `results/pieapp_test_metrics.json`: metric summary + 5PL params
- `results/summary.txt`: plain-text summary
