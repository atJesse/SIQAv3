# 实验结果总结（LoViF / BAPPS / PieAPP）

本文档用于论文撰写阶段，汇总当前工作区中已经完成并落盘的实验结果。

## 1）实验范围与对应结果文件

- **主模型**：SIQA 双骨干 FR-IQA（DINOv3 + CLIP）。
- **LoViF**：5-fold 分层评估（OOF 指标）。
- **BAPPS**：2AFC 验证集评测（硬准确率 + soft agreement）。
- **PieAPP**：官方 test 划分（40 个参考图、600 个失真图），报告 `SROCC / KRCC / PLCC`，并执行 `5PL` 映射后给出 `|PLCC(5PL)|`。

主要结果来源：
- LoViF：`/data/SIQAdn2/workdirs/siqa_dual_dinov3_clip/fold_*/oof_metrics.json`
- BAPPS（主模型）：`/data/SIQAv3/BAPPS2afc/results/bapps_2afc_summary.json`
- BAPPS（传统/感知 baseline）：`/data/SIQAv3/BAPPS2afc/baseline/results/bapps_2afc_metrics_summary.json`
- PieAPP：`/data/SIQAv3/PieAPP2afc/results/pieapp_test_metrics.json`

---

## 2）核心量化结果

### 2.1 LoViF（5-fold OOF）

由 5 个 fold 的 `oof_metrics.json` 汇总得到：

| 指标 | 均值 | 标准差 |
|---|---:|---:|
| SROCC | 0.8570 | 0.0265 |
| PLCC | 0.8673 | 0.0204 |
| Score（0.6*SROCC + 0.4*PLCC） | 0.8611 | 0.0241 |
| MAE | 0.6293 | 0.0543 |
| RMSE | 0.8146 | 0.0481 |

各折 `score`：`0.8661, 0.8837, 0.8753, 0.8656, 0.8149`。

### 2.2 BAPPS（2AFC，val 划分）

主模型（脚本中使用 checkpoint：`fold_1/checkpoints/best.pth`）总体结果：

- 总体 **2AFC accuracy**：**0.68435**
- 总体 **soft agreement**：**0.64991**
- 总样本对数：`36,344`

分子集准确率：

| 子集 | 2AFC Accuracy |
|---|---:|
| cnn | 0.83284 |
| traditional | 0.77754 |
| superres | 0.69731 |
| frameinterp | 0.64778 |
| color | 0.60297 |
| deblur | 0.59661 |

Baseline（同一 BAPPS 划分）总体准确率：

| Baseline | Overall 2AFC Accuracy |
|---|---:|
| DISTS | 0.72293 |
| LPIPS | 0.69910 |
| FSIM | 0.67139 |
| PSNR | 0.66660 |
| SSIM | 0.64096 |
| VIF | 0.58133 |

结论：当前 SIQA 在 BAPPS 上低于最优 baseline（`DISTS`）约 `0.03858`（绝对准确率差）。

### 2.3 PieAPP（official test）

来自 `/data/SIQAv3/PieAPP2afc/results/pieapp_test_metrics.json`：

- `num_references = 40`
- `num_samples = 600`
- 方向对齐方式：`pred_error = -pred_quality`（与 PieAPP 的 error 风格标签对齐）

最终指标：

| 指标 | 数值 |
|---|---:|
| SROCC | 0.59562 |
| KRCC | 0.42753 |
| PLCC（raw） | 0.64096 |
| PLCC（5PL 后） | 0.64105 |
| \|PLCC(5PL)\| | 0.64105 |

---

## 3）论文写作可用结论（建议表述）

1. **LoViF 域内表现较强且稳定**：5-fold 下 `SROCC≈0.86`、`PLCC≈0.87`，说明双骨干 + 语义门控在目标训练域有效。
2. **跨数据集泛化为中等水平**：在 PieAPP、BAPPS 上仍存在明显 domain gap。
3. **BAPPS 上传统/感知指标仍有竞争力**：尤其 `DISTS`，提示后续可加强 pairwise 偏好建模与失真类型覆盖。
4. **PieAPP 上 5PL 提升有限**：当前从 `PLCC(raw)` 到 `PLCC(5PL)` 提升非常小，瓶颈更可能在特征域泛化而非单纯分数单调映射。

---

## 4）复现实验备注

- BAPPS 与 PieAPP 均使用：
  - `/data/SIQAdn2/workdirs/siqa_dual_dinov3_clip/fold_1/checkpoints/best.pth`
- BAPPS 当前为硬判决 2AFC 口径（并同时输出 soft agreement）。
- PieAPP 已按你的要求采用 `5PL` 后 PLCC，并保留 `|PLCC(5PL)|` 作为论文可汇报值。

如需，我可以下一步直接生成一版 **LaTeX 表格**（含三数据集主表 + BAPPS 子集附表）。
