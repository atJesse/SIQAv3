# SIQA Dual-Backbone FR-IQA (Swin + CLIP)

## English

This project is a semantic full-reference IQA pipeline with very small training data (510 pairs), score labels `0..5`, and official metrics `SROCC / PLCC`.

### 1) Core Architecture (Teacher-Requested)
Implemented in `siqa/model.py` as a Siamese network with **two frozen pretrained backbones**:
- Backbone A (structure-aware): `Swin-T` (`swin_tiny_patch4_window7_224`, ImageNet pretrained)
- Backbone B (semantic-aware): `CLIP Vision` (default `clip_vit_l14_336`, can switch to `clip_vit_b32`)

For each pair `(Ref, Dist)`, model extracts:
- $F_{swin\_ref}$, $F_{swin\_dist}$
- $F_{clip\_ref}$, $F_{clip\_dist}$

### 2) Explicit Cosine Similarity Injection
To break the “quality-only shortcut”, the model explicitly injects CLIP cosine similarity:

$$
S_{cos}=\frac{F_{clip\_ref}\cdot F_{clip\_dist}}{\|F_{clip\_ref}\|\,\|F_{clip\_dist}\|}
$$

Fusion uses:
- $Diff_{swin}=|F_{swin\_ref}-F_{swin\_dist}|$
- $Diff_{clip}=|F_{clip\_ref}-F_{clip\_dist}|$
- CLIP multiplication branch: $F_{mult\_clip}=\tilde{F}_{clip\_ref}\odot\tilde{F}_{clip\_dist}$
- Safe default (no feature-dim inflation):
  - $Fused=Concat(F_{swin\_ref},F_{swin\_dist},Diff_{swin},F_{clip\_ref},Diff_{clip},F_{mult\_clip},S_{cos})$
  - (`F_{clip_dist}` is replaced by `F_{mult_clip}`)

Then a required anti-overfitting bottleneck is applied:
- `Linear -> BN -> SiLU -> Dropout(0.5)`
- default `bottleneck_dim=256`

### 3) Semantic Gate / Veto Mechanism
In inference/eval (`model.eval()`), a hard semantic gate is enabled:
- If `S_cos < semantic_gate_threshold` (default `0.4`), force logits toward class `0`
- This makes semantically unrelated pairs strongly biased to near-zero scores

Now supports rollback-friendly gate modes:
- `semantic_gate_mode: off` → disable gate
- `semantic_gate_mode: hard` → single-threshold hard veto
- `semantic_gate_mode: soft` → dual-threshold soft gate (recommended)

Soft gate behavior (default):
- `S_cos < semantic_gate_threshold` (e.g., `0.4`) → hard veto to class `0`
- `semantic_gate_threshold <= S_cos < semantic_gate_high_threshold` (e.g., `0.5`) → soft logit penalty toward lower scores
- `S_cos >= semantic_gate_high_threshold` → no gate intervention

### 4) Training Objective
`train_siqa.py` still uses hybrid objective:

$$
\mathcal{L}=\lambda_{ce}\mathcal{L}_{CE}(logits,y_{cls})+\lambda_{mse}\mathcal{L}_{MSE}(\hat{y},y)
$$

Prediction score:

$$
\hat{y}=\sum_{i=0}^{5} i\cdot softmax(logits)_i
$$

### 5) China-Friendly Download / Offline First
Because servers in China may have unstable external access:
- CLIP supports local-first loading:
  - `model.clip_local_dir`
  - `model.clip_local_files_only: true`
- Swin supports local checkpoint loading:
  - `model.swin_local_path`

If no local files are given:
- CLIP may require `HF_ENDPOINT=https://hf-mirror.com`
- Swin may download from `torchvision` model hub once

### 6) New Model Config Keys
In `configs/siqa_base.yaml`:
- `model.swin_name`
- `model.clip_name`
- `model.freeze_backbones`
- `model.swin_local_path`
- `model.clip_local_dir`
- `model.clip_local_files_only`
- `model.clip_interpolate_pos_encoding`
- `model.clip_mult_enabled`
- `model.clip_mult_replace_raw`
- `model.clip_mult_l2_norm`
- `model.bottleneck_dim`
- `model.bottleneck_dropout`
- `model.semantic_gate_enabled`
- `model.semantic_gate_mode`
- `model.semantic_gate_threshold`
- `model.semantic_gate_high_threshold`
- `model.gate_logit_strength`
- `model.soft_gate_logit_strength`

### 7) Quick Start
Install:
```bash
pip install -r requirements.txt
```

Train:
```bash
python3 train_siqa.py --config configs/siqa_base.yaml
```

Train with HF mirror (for China):
```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1
python3 train_siqa.py --config configs/siqa_base.yaml
```

Submission inference:
```bash
python3 infer_val_submission.py \
  --config configs/siqa_base.yaml \
  --ckpt /data/SIQAdn2/workdirs/siqa_dual_swin_clip/checkpoints/best.pth \
  --out_dir /data/SIQAdn2/submission
```

### 8) CLIP Semantic Distribution Analysis Tool
To analyze the relationship between discrete labels (`0..5`) and CLIP cosine similarity on all training pairs:

```bash
python3 tools/analyze_clip_semantic_distribution.py \
  --config configs/siqa_base.yaml \
  --output_dir tools/output_clip_semantic_analysis
```

Outputs:
- `clip_cosine_per_pair.csv`: per-pair `img_name, score, score_cls, cos_sim`
- `clip_cosine_class_summary.csv`: class-wise cosine range/percentiles/statistics
- `clip_cosine_global_summary.json`: Pearson/Spearman/linear-fit (`R^2`)
- `clip_score_relationship.png`: visualization (scatter + trend)

### 9) 5-Parameter Logistic Mapping (VQEG-style)
For end-score calibration (especially to stretch both extremes near `0` and `5` smoothly), use:

```bash
python3 tools/logistic_5pl_mapping.py \
  --train_pred_csv /data/SIQAdn2/workdirs/siqa_dual_swin_clip/train_predictions.csv \
  --label_xlsx /data/SIQA/Data/Train/Train/Train_scores.xlsx \
  --test_pred_csv /data/SIQAdn2/submission/prediction.csv \
  --out_csv /data/SIQAdn2/submission/prediction_mapped.csv \
  --out_json /data/SIQAdn2/submission/logistic_5pl_params.json
```

Outputs:
- mapped prediction CSV (`Score` after 5PL calibration)
- fitted beta parameters (`b1..b5`) and before/after metrics in JSON

---

## 中文

这是一个小样本（510 对）语义全参考 IQA 项目，标签范围 `0..5`，评价指标是 `SROCC / PLCC`。

### 1) 按老师意见实现的核心架构
`siqa/model.py` 已改为 **双骨干孪生网络**，并默认冻结两套预训练参数：
- Backbone A（结构感知）：`Swin-T`（`swin_tiny_patch4_window7_224`，ImageNet 预训练）
- Backbone B（语义对齐）：`CLIP Vision`（默认 `clip_vit_l14_336`，可改 `clip_vit_b32`）

对每个 `(Ref, Dist)`，提取四组特征：
- $F_{swin\_ref}$, $F_{swin\_dist}$
- $F_{clip\_ref}$, $F_{clip\_dist}$

### 2) 显式引入 CLIP 余弦相似度
为打破“只看画质、不看语义”的捷径，显式加入：

$$
S_{cos}=\frac{F_{clip\_ref}\cdot F_{clip\_dist}}{\|F_{clip\_ref}\|\,\|F_{clip\_dist}\|}
$$

融合方式：
- $Diff_{swin}=|F_{swin\_ref}-F_{swin\_dist}|$
- $Diff_{clip}=|F_{clip\_ref}-F_{clip\_dist}|$
- CLIP 点乘分支：$F_{mult\_clip}=\tilde{F}_{clip\_ref}\odot\tilde{F}_{clip\_dist}$
- 默认安全融合（不增加总维度）：
  - $Fused=Concat(F_{swin\_ref},F_{swin\_dist},Diff_{swin},F_{clip\_ref},Diff_{clip},F_{mult\_clip},S_{cos})$
  - （即用 `F_{mult_clip}` 替换原先的 `F_{clip_dist}` 分支）

然后立即经过防过拟合瓶颈层：
- `Linear -> BN -> SiLU -> Dropout(0.5)`
- 默认降维到 `256`（可改 `128`）

### 3) 语义门控/否决机制
在推理与验证阶段（`model.eval()`）启用硬阈值门控：
- 当 `S_cos < semantic_gate_threshold`（默认 `0.4`）
- 直接把 logits 强制偏向 `0` 分类别

这样对“语义完全不相关但画质看起来好”的样本会更强地压低分数。

目前支持可回退门控模式：
- `semantic_gate_mode: off`：关闭门控
- `semantic_gate_mode: hard`：单阈值硬门控
- `semantic_gate_mode: soft`：双阈值软门控（推荐）

软门控（默认）逻辑：
- `S_cos < semantic_gate_threshold`（如 `0.4`）：执行硬否决，强压到 `0` 分方向
- `semantic_gate_threshold <= S_cos < semantic_gate_high_threshold`（如 `0.5`）：执行软惩罚，温和下压分数
- `S_cos >= semantic_gate_high_threshold`：不干预

### 4) 损失与输出
训练仍采用混合损失：

$$
\mathcal{L}=\lambda_{ce}\mathcal{L}_{CE}(logits,y_{cls})+\lambda_{mse}\mathcal{L}_{MSE}(\hat{y},y)
$$

输出分数：

$$
\hat{y}=\sum_{i=0}^{5} i\cdot softmax(logits)_i
$$

### 5) 中国服务器下载与离线优先
考虑国内网络环境，已支持本地优先：
- CLIP 本地优先：
  - `model.clip_local_dir`
  - `model.clip_local_files_only: true`
- Swin 本地权重：
  - `model.swin_local_path`

若未提供本地文件：
- CLIP 建议设置 `HF_ENDPOINT=https://hf-mirror.com`
- Swin 可能在首次从 `torchvision` 下载一次权重

### 6) 新增配置项
`configs/siqa_base.yaml` 已新增：
- `model.swin_name`
- `model.clip_name`
- `model.freeze_backbones`
- `model.swin_local_path`
- `model.clip_local_dir`
- `model.clip_local_files_only`
- `model.clip_interpolate_pos_encoding`
- `model.clip_mult_enabled`
- `model.clip_mult_replace_raw`
- `model.clip_mult_l2_norm`
- `model.bottleneck_dim`
- `model.bottleneck_dropout`
- `model.semantic_gate_enabled`
- `model.semantic_gate_mode`
- `model.semantic_gate_threshold`
- `model.semantic_gate_high_threshold`
- `model.gate_logit_strength`
- `model.soft_gate_logit_strength`

### 7) 常用命令
安装依赖：
```bash
pip install -r requirements.txt
```

训练：
```bash
python3 train_siqa.py --config configs/siqa_base.yaml
```

国内镜像训练：
```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1
python3 train_siqa.py --config configs/siqa_base.yaml
```

提交式推理：
```bash
python3 infer_val_submission.py \
  --config configs/siqa_base.yaml \
  --ckpt /data/SIQAdn2/workdirs/siqa_dual_swin_clip/checkpoints/best.pth \
  --out_dir /data/SIQAdn2/submission
```

### 8) CLIP 语义分布分析工具
用于分析训练集 510 对图像中，离散评分（`0..5`）与 CLIP 余弦相似度之间的关系：

```bash
python3 tools/analyze_clip_semantic_distribution.py \
  --config configs/siqa_base.yaml \
  --output_dir tools/output_clip_semantic_analysis
```

输出文件：
- `clip_cosine_per_pair.csv`：逐样本 `img_name, score, score_cls, cos_sim`
- `clip_cosine_class_summary.csv`：各分数类别余弦相似度范围/分位数/统计量
- `clip_cosine_global_summary.json`：Pearson/Spearman/线性拟合（含 `R^2`）
- `clip_score_relationship.png`：可视化图（散点 + 趋势）

### 9) 5 参数逻辑斯谛映射（VQEG 标准后处理）
用于把模型分数做平滑校准（尤其把两端更自然地拉向 `0/5`）：

```bash
python3 tools/logistic_5pl_mapping.py \
  --train_pred_csv /data/SIQAdn2/workdirs/siqa_dual_swin_clip/train_predictions.csv \
  --label_xlsx /data/SIQA/Data/Train/Train/Train_scores.xlsx \
  --test_pred_csv /data/SIQAdn2/submission/prediction.csv \
  --out_csv /data/SIQAdn2/submission/prediction_mapped.csv \
  --out_json /data/SIQAdn2/submission/logistic_5pl_params.json
```

输出：
- 映射后的预测 CSV（`Score` 为 5PL 校准后结果）
- 拟合得到的 `b1..b5` 参数及映射前后指标对比 JSON
