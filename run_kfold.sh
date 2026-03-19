#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/siqa_base.yaml}"
NUM_FOLDS="${NUM_FOLDS:-5}"
AGGREGATE_INFERENCE="${AGGREGATE_INFERENCE:-1}"
FOLD_WEIGHTS="${FOLD_WEIGHTS:-}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH"
  exit 1
fi

if [[ -z "${HF_ENDPOINT:-}" ]]; then
  export HF_ENDPOINT="https://hf-mirror.com"
fi
if [[ -z "${HF_HUB_ENABLE_HF_TRANSFER:-}" ]]; then
  export HF_HUB_ENABLE_HF_TRANSFER="0"
fi

WORK_DIR=$(python3 - "$CONFIG_PATH" <<'PY'
import sys
import yaml
cfg_path = sys.argv[1]
with open(cfg_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
print(cfg['output']['work_dir'])
PY
)

echo "[KFold] config=$CONFIG_PATH"
echo "[KFold] num_folds=$NUM_FOLDS"
echo "[KFold] work_dir=$WORK_DIR"
echo "[KFold] HF_ENDPOINT=$HF_ENDPOINT"
echo "[KFold] HF_HUB_ENABLE_HF_TRANSFER=$HF_HUB_ENABLE_HF_TRANSFER"

for ((fold=0; fold<NUM_FOLDS; fold++)); do
  echo "[KFold] ===== Fold $fold/$((NUM_FOLDS-1)) ====="
  TRAIN_CMD=(python3 train_siqa.py --config "$CONFIG_PATH" --num_folds "$NUM_FOLDS" --fold "$fold")
  if [[ "$DRY_RUN" == "1" ]]; then
    TRAIN_CMD+=(--dry_run)
  fi
  "${TRAIN_CMD[@]}"

  if [[ "$AGGREGATE_INFERENCE" == "1" ]]; then
    FOLD_CKPT="$WORK_DIR/fold_${fold}/checkpoints/best.pth"
    FOLD_OUT="$WORK_DIR/fold_${fold}/submission"
    python3 infer_val_submission.py --config "$CONFIG_PATH" --ckpt "$FOLD_CKPT" --out_dir "$FOLD_OUT"
  fi
done

AGG_CMD=(python3 tools/kfold_aggregate.py --work_dir "$WORK_DIR" --num_folds "$NUM_FOLDS")
if [[ "$AGGREGATE_INFERENCE" == "1" ]]; then
  AGG_CMD+=(--aggregate_inference)
fi
if [[ -n "$FOLD_WEIGHTS" ]]; then
  AGG_CMD+=(--weights "$FOLD_WEIGHTS")
fi
"${AGG_CMD[@]}"

echo "[KFold] Done. Summary: $WORK_DIR/kfold_summary.json"
