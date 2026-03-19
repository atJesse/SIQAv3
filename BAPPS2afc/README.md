# BAPPS 2AFC Evaluation

This folder contains scripts for evaluating SIQA models on Berkeley-Adobe Perceptual Patch Similarity (BAPPS) `2afc/val`.

## Rule used (hard 2AFC accuracy)
For each sample `(ref, p0, p1)`:
- Model scores: `score_p0 = Model(ref, p0)`, `score_p1 = Model(ref, p1)`
- Model choice: `0 if score_p0 > score_p1 else 1`
- `judge` in BAPPS is interpreted as preference probability for `p1`
- Human choice: `1 if judge_p1 >= 0.5 else 0`
- Correct if model choice equals human choice.

Note: in this `2afc/val` split, `judge == 0.5` does not appear based on distribution scan.

## Run
```bash
source /data/miniforge3/etc/profile.d/conda.sh
conda activate lovif
cd /data/SIQAv3
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0
python BAPPS2afc/eval_bapps_2afc.py \
  --config /data/SIQAv3/configs/siqa_base.yaml \
  --ckpt /data/SIQAdn2/workdirs/siqa_dual_dinov3_clip/fold_1/checkpoints/best.pth \
  --bapps_root /data/dataset/BAPPS/dataset/2afc/val \
  --out_dir /data/SIQAv3/BAPPS2afc/results
```

### Environment notes
- Mainland China: recommend setting
  - `HF_ENDPOINT=https://hf-mirror.com`
  - `HF_HUB_ENABLE_HF_TRANSFER=0`
- Overseas servers: usually no need to set these two env vars; default HuggingFace endpoint should work.

The script now prints startup diagnostics:
- online/offline capability
- current `HF_ENDPOINT` and `HF_HUB_ENABLE_HF_TRANSFER`
- cache hit status for CLIP/DINOv3 weights

## Outputs
- `bapps_2afc_all_scores.csv`: all pairs with `judge_p1`, `score_p0`, `score_p1`, model/human choice, correctness.
- `per_subset/<subset>_scores.csv`: per-subset detailed scores.
- `bapps_2afc_subset_summary.csv`: per-subset summary.
- `bapps_2afc_summary.json`: overall + subset metrics.
- `readme.txt`: run info and core metrics.
