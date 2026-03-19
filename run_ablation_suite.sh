#!/usr/bin/env bash
set -euo pipefail

BASE_CONFIG="${1:-configs/siqa_base.yaml}"
NUM_FOLDS="${NUM_FOLDS:-5}"
AGGREGATE_INFERENCE="${AGGREGATE_INFERENCE:-1}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_DONE="${SKIP_DONE:-1}"
ABLATION_EXPERIMENTS="${ABLATION_EXPERIMENTS:-baseline,clip_only,structure_only,no_mult,no_gate}"

if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "Base config not found: $BASE_CONFIG"
  exit 1
fi

if [[ -z "${HF_ENDPOINT:-}" ]]; then
  export HF_ENDPOINT="https://hf-mirror.com"
fi
if [[ -z "${HF_HUB_ENABLE_HF_TRANSFER:-}" ]]; then
  export HF_HUB_ENABLE_HF_TRANSFER="0"
fi

readarray -t PATHS < <(python3 - "$BASE_CONFIG" <<'PY'
import os
import sys
import yaml
base_cfg = sys.argv[1]
with open(base_cfg, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
base_work_dir = cfg['output']['work_dir']
suite_root = os.environ.get('ABLATION_ROOT', f"{base_work_dir}_ablation_suite")
config_root = os.path.join(suite_root, 'configs')
os.makedirs(config_root, exist_ok=True)
print(suite_root)
print(config_root)
PY
)

SUITE_ROOT="${PATHS[0]}"
CONFIG_ROOT="${PATHS[1]}"

IFS=',' read -r -a EXP_LIST <<< "$ABLATION_EXPERIMENTS"

echo "[Ablation] base_config=$BASE_CONFIG"
echo "[Ablation] suite_root=$SUITE_ROOT"
echo "[Ablation] config_root=$CONFIG_ROOT"
echo "[Ablation] experiments=$ABLATION_EXPERIMENTS"
echo "[Ablation] num_folds=$NUM_FOLDS"
echo "[Ablation] aggregate_inference=$AGGREGATE_INFERENCE"
echo "[Ablation] dry_run=$DRY_RUN"
echo "[Ablation] skip_done=$SKIP_DONE"
echo "[Ablation] HF_ENDPOINT=$HF_ENDPOINT"

for exp_raw in "${EXP_LIST[@]}"; do
  exp="$(echo "$exp_raw" | xargs)"
  [[ -z "$exp" ]] && continue

  EXP_DIR="$SUITE_ROOT/$exp"
  CFG_PATH="$CONFIG_ROOT/${exp}.yaml"
  SUMMARY_PATH="$EXP_DIR/kfold_summary.json"

  if [[ "$SKIP_DONE" == "1" && -f "$SUMMARY_PATH" ]]; then
    echo "[Ablation] Skip finished experiment: $exp"
    continue
  fi

  echo "[Ablation] ===== Building config: $exp ====="
  python3 - "$BASE_CONFIG" "$CFG_PATH" "$EXP_DIR" "$exp" <<'PY'
import os
import sys
import yaml

base_path, out_path, exp_dir, exp = sys.argv[1:5]

with open(base_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

model = cfg.setdefault('model', {})
model.setdefault('ablation_mode', 'full')
model.setdefault('clip_mult_enabled', True)
model.setdefault('semantic_gate_enabled', True)
model.setdefault('semantic_gate_mode', 'hard')

if exp == 'baseline':
    model['ablation_mode'] = 'full'
    model['clip_mult_enabled'] = True
    model['semantic_gate_enabled'] = True
elif exp == 'clip_only':
    model['ablation_mode'] = 'clip_only'
    model['clip_mult_enabled'] = True
    model['semantic_gate_enabled'] = True
elif exp == 'structure_only':
    model['ablation_mode'] = 'structure_only'
    model['clip_mult_enabled'] = True
    model['semantic_gate_enabled'] = True
elif exp == 'no_mult':
    model['ablation_mode'] = 'full'
    model['clip_mult_enabled'] = False
    model['semantic_gate_enabled'] = True
elif exp == 'no_gate':
    model['ablation_mode'] = 'full'
    model['clip_mult_enabled'] = True
    model['semantic_gate_enabled'] = False
    model['semantic_gate_mode'] = 'off'
else:
    raise ValueError(f'Unknown experiment name: {exp}')

cfg.setdefault('output', {})
cfg['output']['work_dir'] = exp_dir

os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

print(f'Wrote config: {out_path}')
PY

  echo "[Ablation] ===== Running: $exp ====="
  NUM_FOLDS="$NUM_FOLDS" AGGREGATE_INFERENCE="$AGGREGATE_INFERENCE" DRY_RUN="$DRY_RUN" \
    bash run_kfold.sh "$CFG_PATH"
done

python3 tools/ablation_collect.py \
  --suite_root "$SUITE_ROOT" \
  --experiments "$ABLATION_EXPERIMENTS" \
  --out_csv "$SUITE_ROOT/ablation_summary.csv" \
  --out_md "$SUITE_ROOT/ablation_summary.md"

echo "[Ablation] Done."
echo "[Ablation] CSV summary: $SUITE_ROOT/ablation_summary.csv"
echo "[Ablation] Markdown summary: $SUITE_ROOT/ablation_summary.md"
