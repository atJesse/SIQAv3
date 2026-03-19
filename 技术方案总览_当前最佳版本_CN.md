# SIQAv3 当前代码版本技术方案总览（中文）

> 本文档只描述 **当前 `/data/SIQAv3` 代码仓库中可验证的实现与默认配置**。  
> 不再保留无法从当前代码直接核验的历史线上分数、旧路径或旧仓库描述。

---

## 0. 当前仓库定位

- 仓库路径：`/data/SIQAv3`
- 任务类型：全参考图像质量评估（FR-IQA）
- 输入：一对图像 `(Ref, Dist)`
- 标签范围：`0..5`
- 当前默认训练数据：LoViF 训练集（510 对）
- 当前代码默认主结构：`DINOv3 + CLIP`

---

## 1. 当前默认配置（与 `configs/siqa_base.yaml` 一致）

配置文件：`/data/SIQAv3/configs/siqa_base.yaml`

### 1.1 数据路径
- `ref_dir`: `/data/dataset/LoViF/Train/Ref`
- `dist_dir`: `/data/dataset/LoViF/Train/Dist`
- `score_file`: `/data/dataset/LoViF/Train/Train_scores.xlsx`
- `image_size`: `224`

### 1.2 模型默认项
- `structure_backbone: vit_base_patch16_dinov3`
- `clip_name: clip_vit_l14_336`
- `ablation_mode: full`
- `freeze_backbones: true`
- `clip_mult_enabled: true`
- `clip_mult_replace_raw: true`
- `clip_mult_l2_norm: true`
- `semantic_gate_enabled: true`
- `semantic_gate_mode: hard`
- `semantic_gate_threshold: 0.4`
- `semantic_gate_high_threshold: 0.5`

### 1.3 训练默认项
- `epochs: 10`
- `batch_size: 32`
- `num_workers: 4`
- `lr: 2e-4`
- `weight_decay: 1e-2`
- `grad_clip: 1.0`
- `val_ratio: 0.05`
- `auto_resume: true`

### 1.4 输出目录
- `output.work_dir: /data/SIQAdn2/workdirs/siqa_dual_dinov3_clip`

---

## 2. 模型架构（与 `siqa/model.py` 一致）

实现文件：`/data/SIQAv3/siqa/model.py`

### 2.1 双骨干 Siamese
- 结构分支（默认）：`DINOv3-Base`
  - `vit_base_patch16_dinov3`
- 回退可选结构分支：`Swin-T`
  - `swin_tiny_patch4_window7_224`
- 语义分支：`CLIP Vision`
  - 默认：`clip_vit_l14_336`

对 `(Ref, Dist)`，共享权重提取：
- `F_s_ref`, `F_s_dist`
- `F_c_ref`, `F_c_dist`

### 2.2 融合特征
- 结构差分：
  $$Diff_s = |F_{s\_ref} - F_{s\_dist}|$$
- 语义差分：
  $$Diff_c = |F_{c\_ref} - F_{c\_dist}|$$
- 语义余弦相似度：
  $$S_{cos}=\frac{F_{c\_ref}\cdot F_{c\_dist}}{\|F_{c\_ref}\|\,\|F_{c\_dist}\|}$$
- 语义点乘分支：
  - 默认先做 L2 归一化，再逐元素相乘：
  $$F_{mult\_c}=\tilde{F}_{c\_ref}\odot\tilde{F}_{c\_dist}$$

### 2.3 默认融合顺序
当前默认 `clip_mult_replace_raw=true`，因此默认拼接为：

$$
Fused = Concat(F_{s\_ref}, F_{s\_dist}, Diff_s, F_{c\_ref}, Diff_c, F_{mult\_c}, S_{cos})
$$

即：
- `F_c_dist` 在默认配置下 **不直接进入最终拼接**
- 它被 `F_mult_c` 分支替代

### 2.4 Bottleneck 与输出头
- `Bottleneck`：`Linear -> BatchNorm1d -> SiLU -> Dropout(0.5)`
- 默认 `bottleneck_dim = 256`
- `Head`：
  - `Linear(256 -> 128) -> SiLU -> Dropout(0.2) -> Linear(128 -> 6)`
- 输出 6 类 logits，对应 `0..5`
- 连续分数用期望计算：
  $$\hat{y}=\sum_{i=0}^{5} i\cdot softmax(logits)_i$$

---

## 3. 语义门控（与代码一致）

### 3.1 触发条件
- 仅在 `model.eval()` 时生效
- 且 `semantic_gate_enabled = true`
- 且 `ablation_mode != structure_only`

### 3.2 当前默认门控
- `semantic_gate_mode = hard`
- 若 `S_cos < semantic_gate_threshold`（默认 `0.4`）
- 则直接把 logits 强压向 `class 0`

### 3.3 软门控支持
代码也支持：
- `semantic_gate_mode = soft`
- `S_cos < low_th`：硬否决
- `low_th <= S_cos < high_th`：软惩罚
- `S_cos >= high_th`：不干预

---

## 4. 损失与训练流程（与 `train_siqa.py` 一致）

实现文件：`/data/SIQAv3/train_siqa.py`

### 4.1 混合损失
$$
\mathcal{L}=\lambda_{ce}\mathcal{L}_{CE}(logits,y_{cls})+\lambda_{mse}\mathcal{L}_{MSE}(\hat{y},y)
$$

默认：
- `loss_weight_ce = 1.0`
- `loss_weight_mse = 1.0`
- `label_smoothing = 0`

### 4.2 数据划分
- 支持原始 holdout 划分
- 支持 `--num_folds 5 --fold i` 的分层 K-fold 训练
- K-fold 分层依据：`score_cls = round(score)`

### 4.3 优化器与调度器
- Optimizer：`AdamW`
- Scheduler：`CosineAnnealingLR`

---

## 5. 数据预处理（与 `siqa/dataset.py` 一致）

实现文件：`/data/SIQAv3/siqa/dataset.py`

### 5.1 评估/推理预处理
- `ResizePadSquare(224)`
- `ToTensor()`
- `Normalize(mean, std)`

### 5.2 训练预处理
- `ResizePadSquare(224)`
- `RandomHorizontalFlip(0.5)`
- `ToTensor()`
- `Normalize(mean, std)`

### 5.3 默认归一化
- `mean = [0.48145466, 0.4578275, 0.40821073]`
- `std  = [0.26862954, 0.26130258, 0.27577711]`

---

## 6. 推理、提交与集成（与现有脚本一致）

### 6.1 单 checkpoint 提交推理
脚本：`/data/SIQAv3/infer_val_submission.py`

输出：
- `prediction.csv`，列名为：`picture_name,Score`
- `readme.txt`

示例：
```bash
python3 infer_val_submission.py \
  --config configs/siqa_base.yaml \
  --ckpt /data/SIQAdn2/workdirs/siqa_dual_dinov3_clip/fold_1/checkpoints/best.pth \
  --out_dir /data/SIQAdn2/submission
```

### 6.2 K-fold 结果文件
`run_kfold.sh` 与 `tools/kfold_aggregate.py` 会生成：
- `fold_i/checkpoints/best.pth`
- `fold_i/oof_val_predictions.csv`
- `kfold_oof_predictions.csv`
- `kfold_summary.json`
- `kfold_prediction_mean.csv`
- `kfold_prediction_weighted.csv`

---

## 7. 5PL 后处理（与脚本接口一致）

脚本：`/data/SIQAv3/tools/logistic_5pl_mapping.py`

该脚本要求：
- `--train_pred_csv`：训练集预测 CSV
- `--label_xlsx` 或 `--label_csv`
- `--test_pred_csv`：待映射测试/提交预测

当前仓库内最直接可用的训练预测文件示例是：
- `/data/SIQAdn2/workdirs/siqa_dual_dinov3_clip/kfold_oof_predictions.csv`

LoViF 标签文件示例：
- `/data/dataset/LoViF/Train/Train_scores.xlsx`

示例命令：
```bash
python3 tools/logistic_5pl_mapping.py \
  --train_pred_csv /data/SIQAdn2/workdirs/siqa_dual_dinov3_clip/kfold_oof_predictions.csv \
  --label_xlsx /data/dataset/LoViF/Train/Train_scores.xlsx \
  --test_pred_csv /data/SIQAdn2/submission/prediction.csv \
  --out_csv /data/SIQAdn2/submission/prediction_mapped.csv \
  --out_json /data/SIQAdn2/submission/logistic_5pl_params.json
```

---

## 8. 分析工具（与当前仓库一致）

### 8.1 CLIP 语义分布分析
脚本：`/data/SIQAv3/tools/analyze_clip_semantic_distribution.py`

典型输出：
- `clip_cosine_per_pair.csv`
- `clip_cosine_class_summary.csv`
- `clip_cosine_global_summary.json`
- `clip_score_relationship.png`

### 8.2 BAPPS 2AFC 评测
目录：`/data/SIQAv3/BAPPS2afc`

### 8.3 PieAPP 测试集评测
目录：`/data/SIQAv3/PieAPP2afc`

---

## 9. 与旧文档相比的关键修正

本文件已经移除并修正以下旧内容：
- 旧仓库路径 `/data/SIQAdn2/...` 作为源码路径的描述
- 旧数据路径 `/data/SIQA/Data/...`
- 旧默认主干 `Swin-T` 的表述
- 旧输出目录 `siqa_dual_swin_clip`
- `clip_local_files_only=true` 的过期描述（当前默认是 `false`）
- 不可直接从当前代码核验的历史线上分数

---

## 10. 文档说明

本文档基于以下当前代码文件核对：
- `configs/siqa_base.yaml`
- `siqa/model.py`
- `siqa/dataset.py`
- `train_siqa.py`
- `infer_val_submission.py`
- `tools/logistic_5pl_mapping.py`

因此它描述的是 **当前代码版本**，可直接作为后续实验与论文写作的技术说明基线。
