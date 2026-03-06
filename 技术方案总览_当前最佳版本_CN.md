# SIQAdn2 当前最佳版本技术方案总览（中文）

## 0. 当前阶段结果
- 本版本线上/评测结果：
  - `FinalScore: 0.9108`
  - `SROCC: 0.9072`
  - `PLCC: 0.9163`
- 当前目标：在保持语义一致性约束能力的前提下，继续提升 `SROCC / PLCC`，减少“语义不相关但给高分”的错误。

---

## 1. 任务定义与目标

### 1.1 任务定义
- 任务类型：全参考图像质量评估（FR-IQA, Full-Reference IQA）
- 输入：一对图像 `(Ref, Dist)`
- 标签：离散分数 `0,1,2,3,4,5`
- 评价指标：
  - 主指标：`FinalScore = 0.6 * SROCC + 0.4 * PLCC`

### 1.2 关键挑战
- 训练集极小（510 对），易过拟合
- 传统差值特征易“捷径学习”：只看画质，不看语义
- 语义完全不相关样本（应接近 0 分）需要强约束

### 1.3 本方案核心目标
1. 同时建模结构质量与语义一致性
2. 强化“语义否决”能力，避免语义不一致高分
3. 通过后处理（5 参数逻辑斯谛映射）提升分数端点校准能力

---

## 2. 数据与路径规范

## 2.1 当前配置文件
- 配置文件：`/data/SIQAdn2/configs/siqa_base.yaml`

### 2.2 数据路径（当前默认）
- 训练参考图：`/data/SIQA/Data/Train/Train/Ref`
- 训练失真图：`/data/SIQA/Data/Train/Train/Dist`
- 训练标签表：`/data/SIQA/Data/Train/Train/Train_scores.xlsx`

### 2.3 训练标签格式
- 文件：`Train_scores.xlsx`
- 每行至少两列：`(img_name, score)`
- `img_name` 支持不带扩展名，读取时自动补 `.png`
- `score` 为 `0..5` 的数值

### 2.4 推理输出格式
- `infer_val_submission.py` 输出：
  - `prediction.csv`，列名：`picture_name,Score`
  - `readme.txt`

---

## 3. 模型架构（当前主干方案）

实现文件：`/data/SIQAdn2/siqa/model.py`

### 3.1 双骨干 Siamese
- Backbone A（结构感知）：`Swin-T`
  - `swin_tiny_patch4_window7_224`
- Backbone B（语义对齐）：`CLIP Vision`
  - 当前默认：`clip_vit_l14_336`

Ref 与 Dist 分别通过两条共享权重分支，得到：
- `F_swin_ref, F_swin_dist`
- `F_clip_ref, F_clip_dist`

### 3.2 融合特征（含点乘增强）
- 结构差值：
  $$Diff_{swin}=|F_{swin\_ref} - F_{swin\_dist}|$$
- 语义差值：
  $$Diff_{clip}=|F_{clip\_ref} - F_{clip\_dist}|$$
- 语义余弦相似度：
  $$S_{cos}=\frac{F_{clip\_ref}\cdot F_{clip\_dist}}{\|F_{clip\_ref}\|\|F_{clip\_dist}\|}$$
- 点乘增强（策略二）：
  $$F_{mult\_clip}=\tilde{F}_{clip\_ref}\odot\tilde{F}_{clip\_dist}$$
  其中 $\tilde{F}$ 表示可选 L2 归一化后的特征（当前默认启用归一化）。

#### 当前默认“安全融合”（不增总维度）
- 用 `F_mult_clip` 替换原 `F_clip_dist` 分支：
  $$Fused=Concat(F_{swin\_ref},F_{swin\_dist},Diff_{swin},F_{clip\_ref},Diff_{clip},F_{mult\_clip},S_{cos})$$
- 目的：在 510 对小样本下尽量不膨胀参数量，降低过拟合风险。

### 3.3 Bottleneck + 预测头
- Bottleneck：`Linear -> BN -> SiLU -> Dropout(0.5)`
- 默认降维：`bottleneck_dim=256`
- 分类头输出 6 维 logits（对应 `0..5`）
- 连续分数输出（期望分）：
  $$\hat{y}=\sum_{i=0}^{5} i\cdot softmax(logits)_i$$

---

## 4. 语义门控/否决机制（可开关可回退）

### 4.1 触发位置
- 仅在 `model.eval()`（验证/推理）阶段触发

### 4.2 门控模式
- `semantic_gate_mode: off`
  - 关闭门控
- `semantic_gate_mode: hard`
  - 单阈值硬门控：`S_cos < semantic_gate_threshold`（默认 0.4）
  - 直接将 logits 强压向 0 分类别
- `semantic_gate_mode: soft`
  - 双阈值软门控（推荐探索）
  - `S_cos < low_th`：硬否决
  - `low_th <= S_cos < high_th`：软惩罚（温和下压分数）
  - `S_cos >= high_th`：不干预

### 4.3 当前默认
- 配置中默认：`semantic_gate_mode: hard`
- 即当前线上最好成绩对应的是“硬门控基线 + 乘法增强版本”

---

## 5. 损失设计与优化

实现文件：`/data/SIQAdn2/train_siqa.py`

### 5.1 混合损失
$$
\mathcal{L}=\lambda_{ce}\,\mathcal{L}_{CE}(logits,y_{cls})+\lambda_{mse}\,\mathcal{L}_{MSE}(\hat{y},y)
$$
- 当前默认：
  - `loss_weight_ce = 1.0`
  - `loss_weight_mse = 1.0`
  - `label_smoothing = 0`

### 5.2 优化器与学习率
- Optimizer: `AdamW`
- Scheduler: `CosineAnnealingLR`
- 当前默认：
  - `lr = 2e-4`
  - `weight_decay = 1e-2`
  - `grad_clip = 1.0`
  - `epochs = 10`
  - `batch_size = 32`

---

## 6. 数据预处理与划分

实现文件：`/data/SIQAdn2/siqa/dataset.py`

### 6.1 预处理
- `ResizePadSquare(image_size=224)`
  - 保持长宽比缩放，补边成方图
- 训练增强：`RandomHorizontalFlip(0.5)`
- 归一化（当前默认）：
  - mean: `[0.48145466, 0.4578275, 0.40821073]`
  - std: `[0.26862954, 0.26130258, 0.27577711]`

### 6.2 数据划分
- `stratified_split_indices` 基于分数离散桶分层划分
- 当前默认：`val_ratio = 0.05`

---

## 7. 复现性、日志与检查点

### 7.1 复现性
- 全局 seed：`seed=3407`
- `random / numpy / torch / cuda` 全部设定
- DataLoader worker seed 固定
- `cudnn.deterministic=True, cudnn.benchmark=False`

### 7.2 Checkpoint
- 输出目录：`/data/SIQAdn2/workdirs/siqa_dual_swin_clip`
- 保存：
  - `checkpoints/last.pth`
  - `checkpoints/best.pth`
- `auto_resume=true` 时自动尝试 `last.pth`
- 若模型结构不兼容旧 ckpt，会自动跳过并重新开始（防崩）

### 7.3 Debug日志
- 训练日志：`train.log`
- 支持输出：
  - batch shape
  - 学习率
  - GPU 显存
  - 门控触发比例（总/硬/软）

---

## 8. 推理与后处理

## 8.1 提交推理
- 脚本：`/data/SIQAdn2/infer_val_submission.py`
- 输入：`Ref/Dist` 文件夹
- 输出：`prediction.csv`（`picture_name,Score`）

### 8.2 VQEG 风格 5 参数逻辑斯谛映射（后处理）
- 脚本：`/data/SIQAdn2/tools/logistic_5pl_mapping.py`
- 功能：
  1. 用训练集 `(预测值, 真值)` 拟合 5PL 参数 `b1..b5`
  2. 把该映射应用到测试/提交预测值
  3. 生成映射后 `csv` 与参数 `json`

5PL 形式：
$$
y=b_1\left(0.5-\frac{1}{1+e^{b_2(x-b_3)}}\right)+b_4x+b_5
$$

- 输出（示例）：
  - `/data/SIQAdn2/submission/prediction_mapped.csv`
  - `/data/SIQAdn2/submission/logistic_5pl_params.json`

---

## 9. 分析工具与辅助实验

### 9.1 CLIP 语义分布分析工具
- 脚本：`/data/SIQAdn2/tools/analyze_clip_semantic_distribution.py`
- 功能：
  - 统计每个分数类别的 `S_cos` 分布范围、分位数
  - 计算 `Pearson/Spearman/R^2`
  - 输出图表与 CSV 供阈值分析

### 9.2 典型输出
- `clip_cosine_per_pair.csv`
- `clip_cosine_class_summary.csv`
- `clip_cosine_global_summary.json`
- `clip_score_relationship.png`

---

## 10. 关键配置参数总表（当前）

来自 `configs/siqa_base.yaml`：

### 10.1 `data`
- `ref_dir`, `dist_dir`, `score_file`
- `image_size=224`
- `normalize_mean/std`（CLIP风格）

### 10.2 `model`
- 基础：
  - `swin_name=swin_tiny_patch4_window7_224`
  - `clip_name=clip_vit_l14_336`
  - `num_classes=6`
  - `freeze_backbones=true`
- 离线加载：
  - `swin_local_path`
  - `clip_local_dir`
  - `clip_local_files_only=true`
- 乘法分支：
  - `clip_mult_enabled=true`
  - `clip_mult_replace_raw=true`
  - `clip_mult_l2_norm=true`
- 瓶颈：
  - `bottleneck_dim=256`
  - `bottleneck_dropout=0.5`
- 门控：
  - `semantic_gate_enabled=true`
  - `semantic_gate_mode=hard`
  - `semantic_gate_threshold=0.4`
  - `semantic_gate_high_threshold=0.5`
  - `gate_logit_strength=12.0`
  - `soft_gate_logit_strength=6.0`

### 10.3 `train`
- `epochs=10`, `batch_size=32`, `num_workers=4`
- `lr=2e-4`, `weight_decay=1e-2`
- `grad_clip=1.0`
- `val_ratio=0.05`
- `auto_resume=true`

---

## 11. 常用命令（当前版本）

### 11.1 训练
```bash
python3 train_siqa.py --config configs/siqa_base.yaml
```

### 11.2 推理
```bash
python3 infer_val_submission.py \
  --config configs/siqa_base.yaml \
  --ckpt /data/SIQAdn2/workdirs/siqa_dual_swin_clip/checkpoints/best.pth \
  --out_dir /data/SIQAdn2/submission
```

### 11.3 5PL 后处理
```bash
python3 tools/logistic_5pl_mapping.py \
  --train_pred_csv /data/SIQAdn2/workdirs/siqa_dual_swin_clip/train_predictions.csv \
  --label_xlsx /data/SIQA/Data/Train/Train/Train_scores.xlsx \
  --test_pred_csv /data/SIQAdn2/submission/prediction.csv \
  --out_csv /data/SIQAdn2/submission/prediction_mapped.csv \
  --out_json /data/SIQAdn2/submission/logistic_5pl_params.json
```

---

## 12. 下一步冲分建议（简版）
1. 用当前最好配置作为基线，固定随机种子复现实验
2. 仅改一个变量做消融：
   - `semantic_gate_mode: hard -> soft`
   - `semantic_gate_high_threshold` 在 `0.45~0.55` 搜索
3. 比较是否使用 5PL 后处理对提交结果提升幅度
4. 保留最优配置与日志，形成可复现提交链路

---

> 文档说明：本文件基于当前代码与 `siqa_base.yaml` 实际内容整理，适用于当前最佳版本归档与后续冲分实验管理。
