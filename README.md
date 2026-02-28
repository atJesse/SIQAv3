# SIQA Challenge Pipeline (FR-IQA, Semantic-Focused)

## English

This repository provides a reproducible pipeline for a semantic full-reference IQA challenge:
- Input: paired `(Ref, Dist)` images.
- Label: score in `0..5`.
- Official metric: `0.6 * SROCC + 0.4 * PLCC`.

### 1) Architecture
The model is implemented in `siqa/model.py`:
- Backbone: `DINOv2` (default `dinov2_vits14`) used in siamese form for both `Ref` and `Dist`.
- Feature fusion: concatenate `[feat_ref, feat_dist, abs(feat_ref - feat_dist)]`.
- Head: MLP classifier producing 6 logits (for score classes `0..5`).
- Continuous prediction: expected score from class probabilities:

$$
\hat{y}=\sum_{i=0}^{5} i\cdot \mathrm{softmax}(\mathrm{logits})_i
$$

### 2) Loss Design
Training loss in `train_siqa.py` is a weighted hybrid:

$$
\mathcal{L}=\lambda_{ce}\,\mathcal{L}_{CE}(\text{logits}, y_{cls})+\lambda_{mse}\,\mathcal{L}_{MSE}(\hat{y}, y)
$$

Current config (`configs/siqa_base.yaml`):
- `loss_weight_ce = 1.0`
- `loss_weight_mse = 1.0`
- `label_smoothing = 0.05`

### 3) Data Preprocessing
Implemented in `siqa/dataset.py`:
- Parse `Train_scores.xlsx` via `openpyxl`.
- Paired load by filename from `Ref/` and `Dist/`.
- Different image sizes/aspect ratios are handled by `ResizePadSquare(image_size)`:
  - keep aspect ratio,
  - resize by long side,
  - zero-pad to square.
- Normalization uses ImageNet mean/std.

### 4) Training Behavior Observed
From `workdirs/siqa_dinov2_vits14/train.log`:
- Training loss decreases steadily (approx `3.5 -> 0.3`).
- Validation score peaks early (`best score = 0.7936` around epoch 5), then fluctuates lower.

This indicates early overfitting under small-data setting: optimization loss keeps improving while ranking/generalization metrics stop improving.

### 5) Script Roles
- `train_siqa.py`: train/val/checkpoint pipeline.
- `predict_siqa.py`: generic inference utility (uses training score table style dataset).
- `infer_val_submission.py`: **submission-oriented inference script** for Val/Test folders; writes:
  - `prediction.csv`
  - `readme.txt`

For final submission-style inference, use `infer_val_submission.py`.

### 6) Quick Start
Install:
```bash
pip install -r requirements.txt
```

Train:
```bash
python3 train_siqa.py --config configs/siqa_base.yaml
```

Submission inference:
```bash
python3 infer_val_submission.py \
  --config configs/siqa_base.yaml \
  --ckpt /data/SIQA/workdirs/siqa_dinov2_vits14/checkpoints/best.pth \
  --out_dir /data/SIQA/submission
```

---

## 中文

本仓库提供语义全参考 IQA 挑战赛的可复现流程：
- 输入：成对图像 `(Ref, Dist)`。
- 标签：`0..5` 分。
- 官方指标：`0.6 * SROCC + 0.4 * PLCC`。

### 1) 架构说明
模型在 `siqa/model.py` 中：
- 骨干：`DINOv2`（默认 `dinov2_vits14`），Ref/Dist 共享同一骨干（siamese）。
- 融合：拼接 `[feat_ref, feat_dist, abs(feat_ref - feat_dist)]`。
- 头部：MLP 输出 6 维 logits（对应 `0..5` 类）。
- 连续分数：对 logits 做 softmax 后计算期望分数：

$$
\hat{y}=\sum_{i=0}^{5} i\cdot \mathrm{softmax}(\mathrm{logits})_i
$$

### 2) Loss 设计
训练损失在 `train_siqa.py` 中采用加权混合损失：

$$
\mathcal{L}=\lambda_{ce}\,\mathcal{L}_{CE}(\text{logits}, y_{cls})+\lambda_{mse}\,\mathcal{L}_{MSE}(\hat{y}, y)
$$

当前配置（`configs/siqa_base.yaml`）：
- `loss_weight_ce = 1.0`
- `loss_weight_mse = 1.0`
- `label_smoothing = 0.05`

### 3) 数据预处理
在 `siqa/dataset.py` 中实现：
- 使用 `openpyxl` 读取 `Train_scores.xlsx`。
- 按文件名在 `Ref/` 与 `Dist/` 做配对读取。
- 针对尺寸/比例不一致，使用 `ResizePadSquare(image_size)`：
  - 等比缩放（按长边），
  - 再补边到方形。
- 最后做 ImageNet 均值方差归一化。

### 4) 当前观察到的训练现象
根据 `workdirs/siqa_dinov2_vits14/train.log`：
- 训练损失持续下降（约 `3.5 -> 0.3`）。
- 验证指标在早期达到峰值（`best score = 0.7936`，约第 5 epoch），后续波动下降。

这说明小样本场景下出现了早期过拟合：loss 继续下降，但排序/泛化指标不再提升。

### 5) 脚本分工
- `train_siqa.py`：训练/验证/checkpoint。
- `predict_siqa.py`：通用推理脚本（按训练集分数表式数据组织）。
- `infer_val_submission.py`：**面向提交的推理脚本**，直接输出：
  - `prediction.csv`
  - `readme.txt`

最终提交流程建议使用 `infer_val_submission.py`。

### 6) 快速使用
安装：
```bash
pip install -r requirements.txt
```

训练：
```bash
python3 train_siqa.py --config configs/siqa_base.yaml
```

提交式推理：
```bash
python3 infer_val_submission.py \
  --config configs/siqa_base.yaml \
  --ckpt /data/SIQA/workdirs/siqa_dinov2_vits14/checkpoints/best.pth \
  --out_dir /data/SIQA/submission
```
