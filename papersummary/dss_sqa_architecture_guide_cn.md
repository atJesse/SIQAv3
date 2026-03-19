# DSS-SQA 架构图绘制指南（与 `/data/SIQAv3` 代码一致）

> 对应实现：`/data/SIQAv3/siqa/model.py`（`SiameseSemanticIQA`）  
> 训练入口：`/data/SIQAv3/train_siqa.py`  
> 预处理：`/data/SIQAv3/siqa/dataset.py`

---

## 1. 一句话总览

DSS-SQA 是一个 **Siamese 双骨干全参考 IQA**：
- 结构分支：`DINOv3`（默认）或 `Swin-T`（可切换）
- 语义分支：`CLIP-ViT`
- 对 `(Ref, Dist)` 提取结构+语义特征后，做 `差分 + 点乘融合 + 余弦注入`，再经过 `Bottleneck + 分类头`，输出 `0..5` 的期望分数。

---

## 2. 输入与预处理（画图时可单独画在最左侧）

代码口径（默认配置 `configs/siqa_base.yaml`）：
- 输入：`Ref image` 与 `Dist image`
- `build_eval_transform`：`ResizePadSquare(224) -> ToTensor -> Normalize(mean,std)`
- 其中 `mean/std` 默认是 CLIP 风格：
  - `mean=[0.48145466, 0.4578275, 0.40821073]`
  - `std=[0.26862954, 0.26130258, 0.27577711]`

注意：
- CLIP 分支在 `extract_clip_features` 内部会再检查尺寸，若不等于 CLIP config 的 `image_size`（如 L14-336），会做一次 `bilinear interpolate` 到 CLIP 需要的尺寸。

---

## 3. 主干结构（Siamese，权重共享）

对每个输入对 `(Ref, Dist)`，同一套 backbone 权重分别处理两张图。

### 3.1 结构分支（Structure Backbone）

- 默认：`vit_base_patch16_dinov3`（`timm`）
- 备选：`swin_tiny_patch4_window7_224`
- 输出：
  - `F_s_ref`（Ref 的结构特征）
  - `F_s_dist`（Dist 的结构特征）

### 3.2 语义分支（CLIP Backbone）

- 默认：`clip_vit_l14_336`
- 输出：
  - `F_c_ref`（Ref 的语义特征）
  - `F_c_dist`（Dist 的语义特征）

---

## 4. 融合细节（重点：点乘融合的位置）

这一部分是你画图最需要强调的。

### 4.1 先算差分

- 结构差分：
  - `Diff_s = |F_s_ref - F_s_dist|`
- 语义差分：
  - `Diff_c = |F_c_ref - F_c_dist|`

### 4.2 点乘融合（你提到的关键点）

在代码里位置是：
1. 先拿到 `F_c_ref` 和 `F_c_dist`
2. 计算 `mult_clip`

默认配置 `clip_mult_enabled=true` 且 `clip_mult_l2_norm=true` 时：
- `mult_clip = L2Norm(F_c_ref) ⊙ L2Norm(F_c_dist)`

若 `clip_mult_l2_norm=false` 才是原始逐元素乘：
- `mult_clip = F_c_ref ⊙ F_c_dist`

### 4.3 余弦相似度注入

- `S_cos = cosine(F_c_ref, F_c_dist)`（标量）
- 作为一个 1 维特征拼接进融合向量（不是后处理单独分支）。

### 4.4 最终拼接向量（默认配置下）

默认 `clip_mult_replace_raw=true`，因此 **用 `mult_clip` 替代 `F_c_dist`**：

`F_fused = Concat(F_s_ref, F_s_dist, Diff_s, F_c_ref, Diff_c, mult_clip, S_cos)`

这点请在图里明确标注：“`F_c_dist` 被乘法分支替换（默认）”。

---

## 5. 维度与头部（建议在图右侧标维度）

代码公式：
- `clip_branch_dim = clip_dim * (3 if clip_mult_replace_raw else 4)`
- `fusion_dim = structure_dim * 3 + clip_branch_dim + 1`

默认主干常见维度（DINOv3-Base + CLIP-L/14）：
- `structure_dim ≈ 768`
- `clip_dim ≈ 1024`
- `clip_mult_replace_raw=true` 时：
  - `clip_branch_dim = 3*1024 = 3072`
  - `fusion_dim = 3*768 + 3072 + 1 = 5377`

后续头部：
1. `Bottleneck`: `Linear(fusion_dim -> 256) -> BN -> SiLU -> Dropout(0.5)`
2. `Head`: `Linear(256 -> 128) -> SiLU -> Dropout(0.2) -> Linear(128 -> 6)`
3. `Softmax(logits)` 后做期望：
   - `score = Σ_i p_i * i`，`i in {0,1,2,3,4,5}`

---

## 6. Semantic Gate / Veto（推理态生效）

语义门控只在：
- `model.eval()` 且 `semantic_gate_enabled=true` 时启用。

默认 `semantic_gate_mode=hard`：
- 若 `S_cos < semantic_gate_threshold`（默认 `0.4`）
- 强制 logits 偏向 `class 0`（极低分）

软门控（`mode=soft`）代码也支持：
- `low_th <= S_cos < high_th` 施加渐进惩罚
- `S_cos < low_th` 仍是硬压低

注意：若 `ablation_mode=structure_only`，会关闭 semantic gate 的应用逻辑。

---

## 7. 训练目标（图中可写在 Loss 框）

`train_siqa.py` 中：
- 分类损失：`CE(logits, score_cls)`
- 回归损失：`MSE(score_pred, score_float)`
- 总损失：
  - `L = λ_ce * CE + λ_mse * MSE`
- 默认：`λ_ce=1.0`, `λ_mse=1.0`

标签定义：
- `score_cls = round(score)`
- `score_float` 是原始分数（0~5）

---

## 8. 消融开关（画图时可用虚线表示）

`ablation_mode`：
- `full`：全分支
- `clip_only`：结构特征置零（仅语义）
- `structure_only`：语义特征置零（仅结构）

`clip_mult_*`：
- `clip_mult_enabled`：是否启用乘法分支
- `clip_mult_replace_raw`：是否用乘法分支替代 `F_c_dist`
- `clip_mult_l2_norm`：乘法前是否先做 L2 归一化

---

## 9. 你可以直接照着画的图层布局（推荐）

### 左到右布局

1. **Input Block**：`Ref` / `Dist`
2. **Preprocess Block**：`ResizePadSquare + Normalize`
3. **Dual Siamese Encoders**：
   - 上路：Structure backbone（DINOv3 / Swin）
   - 下路：CLIP backbone
4. **Feature Interaction Block**：
   - `Diff_s`, `Diff_c`
   - `mult_clip`（高亮：L2Norm 后逐元素乘）
   - `S_cos`（单标量注入）
5. **Fusion Concat Block**：显示默认拼接项
6. **Bottleneck + Head**
7. **Score Output (0~5)**
8. **Semantic Gate (Inference only)**：挂在 Head 后 logits 侧边

### 视觉高亮建议

- 用红框标你要强调的创新：
  1) `mult_clip` 分支  
  2) `S_cos` 注入  
  3) `semantic gate`
- 在 `mult_clip` 边上写：
  - `mult_clip = L2(F_c_ref) ⊙ L2(F_c_dist)`（default）
- 在 Concat 边上写：
  - `F_c_dist replaced by mult_clip (default)`

---

## 10. 论文图注可直接参考（中文草案）

“我们提出的 DSS-SQA 采用结构-语义解耦的双视觉 Siamese 架构。结构分支使用 DINOv3（可切换 Swin），语义分支使用 CLIP。针对参考图与失真图，模型同时建模结构差分与语义差分，并在语义侧引入归一化逐元素乘法交互与余弦相似度标量注入。融合特征经 bottleneck 与分类头输出离散质量分布，再通过期望得到连续质量分。推理阶段进一步采用基于语义相似度的门控机制，对语义不一致样本施加低分约束，从而提升语义一致性评估能力。”

---

## 11. 代码一致性核对结论（给你放心用）

以上描述逐项对应 `siqa/model.py` 当前实现，特别是：
- 点乘分支位置与公式
- `clip_mult_replace_raw=true` 下替代关系
- 余弦注入在 concat 内而非后处理
- gate 仅在 `eval` 生效

都已按代码核对。可直接用于架构图绘制与论文方法段。
