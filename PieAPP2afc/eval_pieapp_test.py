import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from scipy.optimize import curve_fit
from scipy.stats import kendalltau, pearsonr, spearmanr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from siqa.dataset import build_eval_transform
from siqa.model import SiameseSemanticIQA


@dataclass
class PieAPPRecord:
    ref_image: str
    dist_image: str
    gt_error: float
    pred_quality: float
    pred_error: float


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SIQA checkpoint on PieAPP test split")
    parser.add_argument("--config", type=str, default="/data/SIQAv3/configs/siqa_base.yaml")
    parser.add_argument("--ckpt", type=str, default="/data/SIQAdn2/workdirs/siqa_dual_dinov3_clip/fold_1/checkpoints/best.pth")
    parser.add_argument("--pieapp_root", type=str, default="/data/dataset/PieAPP/PieAPP_dataset_CVPR_2018")
    parser.add_argument("--out_dir", type=str, default="/data/SIQAv3/PieAPP2afc/results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_refs", type=int, default=0, help="Optional debug cap; 0 means all test references")
    parser.add_argument("--max_images_per_ref", type=int, default=0, help="Optional debug cap; 0 means all per-image rows")
    return parser.parse_args()


def five_param_logistic(x: np.ndarray, b1: float, b2: float, b3: float, b4: float, b5: float) -> np.ndarray:
    z = np.clip(b2 * (x - b3), -60.0, 60.0)
    return b1 * (0.5 - 1.0 / (1.0 + np.exp(z))) + b4 * x + b5


def fit_5pl_and_map(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p0 = np.array(
        [
            float(np.max(y) - np.min(y)),
            1.0,
            float(np.median(x)),
            1.0,
            float(np.mean(y)),
        ],
        dtype=np.float64,
    )
    params, _ = curve_fit(
        five_param_logistic,
        x.astype(np.float64),
        y.astype(np.float64),
        p0=p0,
        maxfev=200000,
    )
    mapped = five_param_logistic(x.astype(np.float64), *params)
    return params, mapped


def safe_corr(a: np.ndarray, b: np.ndarray, method: str) -> float:
    if len(a) < 2:
        return 0.0
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return 0.0

    if method == "pearson":
        return float(pearsonr(a, b)[0])
    if method == "spearman":
        return float(spearmanr(a, b)[0])
    if method == "kendall":
        return float(kendalltau(a, b)[0])
    raise ValueError(f"Unsupported method: {method}")


def build_model(cfg: Dict, ckpt_path: str, device: torch.device) -> SiameseSemanticIQA:
    structure_backbone = cfg["model"].get("structure_backbone", cfg["model"].get("swin_name", "swin_tiny_patch4_window7_224"))
    model = SiameseSemanticIQA(
        num_classes=cfg["model"]["num_classes"],
        structure_backbone=structure_backbone,
        ablation_mode=cfg["model"].get("ablation_mode", "full"),
        swin_name=cfg["model"].get("swin_name", "swin_tiny_patch4_window7_224"),
        clip_name=cfg["model"].get("clip_name", "clip_vit_b32"),
        freeze_backbones=False,
        swin_local_path=cfg["model"].get("swin_local_path", ""),
        clip_local_dir=cfg["model"].get("clip_local_dir", ""),
        clip_local_files_only=cfg["model"].get("clip_local_files_only", False),
        clip_interpolate_pos_encoding=cfg["model"].get("clip_interpolate_pos_encoding", True),
        clip_mult_enabled=cfg["model"].get("clip_mult_enabled", True),
        clip_mult_replace_raw=cfg["model"].get("clip_mult_replace_raw", True),
        clip_mult_l2_norm=cfg["model"].get("clip_mult_l2_norm", True),
        bottleneck_dim=cfg["model"].get("bottleneck_dim", 256),
        bottleneck_dropout=cfg["model"].get("bottleneck_dropout", 0.5),
        semantic_gate_enabled=cfg["model"].get("semantic_gate_enabled", True),
        semantic_gate_threshold=cfg["model"].get("semantic_gate_threshold", 0.4),
        semantic_gate_high_threshold=cfg["model"].get("semantic_gate_high_threshold", 0.5),
        semantic_gate_mode=cfg["model"].get("semantic_gate_mode", "hard"),
        gate_logit_strength=cfg["model"].get("gate_logit_strength", 12.0),
        soft_gate_logit_strength=cfg["model"].get("soft_gate_logit_strength", 6.0),
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model_state = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(model_state)
    model.eval()
    return model


def read_test_refs(test_list_path: str, max_refs: int) -> List[str]:
    refs: List[str] = []
    with open(test_list_path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if not name:
                continue
            refs.append(name)
    if max_refs > 0:
        refs = refs[:max_refs]
    return refs


def read_per_image_score_csv(csv_path: str, max_images_per_ref: int) -> List[Tuple[str, str, float]]:
    rows: List[Tuple[str, str, float]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return rows
        for row in reader:
            if len(row) < 3:
                continue
            ref_name = row[0].strip()
            dist_name = row[1].strip()
            gt_error = float(row[2].strip())
            rows.append((ref_name, dist_name, gt_error))
            if max_images_per_ref > 0 and len(rows) >= max_images_per_ref:
                break
    return rows


def score_pair(model: SiameseSemanticIQA, transform, device: torch.device, ref_path: str, dist_path: str) -> float:
    ref_img = Image.open(ref_path).convert("RGB")
    dist_img = Image.open(dist_path).convert("RGB")
    ref_t = transform(ref_img).unsqueeze(0).to(device)
    dist_t = transform(dist_img).unsqueeze(0).to(device)
    with torch.no_grad():
        _, score = model(ref_t, dist_t)
    return float(score.item())


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but torch.cuda.is_available() is False")

    print("[pieapp] loading model...")
    model = build_model(cfg, args.ckpt, device)

    mean = cfg["data"].get("normalize_mean", [0.48145466, 0.4578275, 0.40821073])
    std = cfg["data"].get("normalize_std", [0.26862954, 0.26130258, 0.27577711])
    transform = build_eval_transform(cfg["data"]["image_size"], mean=mean, std=std)

    test_list_path = os.path.join(args.pieapp_root, "test_reference_list.txt")
    ref_root = os.path.join(args.pieapp_root, "reference_images", "test")
    dist_root = os.path.join(args.pieapp_root, "distorted_images", "test")
    label_root = os.path.join(args.pieapp_root, "labels", "test")

    for path in [test_list_path, ref_root, dist_root, label_root]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required path: {path}")

    test_refs = read_test_refs(test_list_path, args.max_refs)
    if not test_refs:
        raise RuntimeError("No test references found")

    print(f"[pieapp] refs={len(test_refs)} | device={device}")
    start = time.time()

    records: List[PieAPPRecord] = []

    for idx, ref_name in enumerate(test_refs, start=1):
        ref_stem = os.path.splitext(ref_name)[0]
        ref_path = os.path.join(ref_root, ref_name)
        score_csv = os.path.join(label_root, f"{ref_stem}_per_image_score.csv")
        dist_dir = os.path.join(dist_root, ref_stem)

        if not os.path.isfile(ref_path):
            raise FileNotFoundError(f"Missing reference image: {ref_path}")
        if not os.path.isfile(score_csv):
            raise FileNotFoundError(f"Missing per-image score csv: {score_csv}")
        if not os.path.isdir(dist_dir):
            raise FileNotFoundError(f"Missing distorted image dir: {dist_dir}")

        per_image_rows = read_per_image_score_csv(score_csv, args.max_images_per_ref)
        if not per_image_rows:
            continue

        for csv_ref_name, dist_name, gt_error in per_image_rows:
            if csv_ref_name != ref_name:
                raise RuntimeError(
                    f"CSV ref mismatch in {score_csv}: expected {ref_name}, got {csv_ref_name}"
                )

            dist_path = os.path.join(dist_dir, dist_name)
            if not os.path.isfile(dist_path):
                raise FileNotFoundError(f"Missing distorted image: {dist_path}")

            pred_quality = score_pair(model, transform, device, ref_path, dist_path)
            pred_error = -pred_quality
            records.append(
                PieAPPRecord(
                    ref_image=ref_name,
                    dist_image=dist_name,
                    gt_error=float(gt_error),
                    pred_quality=pred_quality,
                    pred_error=pred_error,
                )
            )

        print(f"[pieapp] processed ref {idx}/{len(test_refs)}: {ref_name} | samples={len(per_image_rows)}")

    if not records:
        raise RuntimeError("No records generated")

    y_true_error = np.array([r.gt_error for r in records], dtype=np.float64)
    y_pred_error = np.array([r.pred_error for r in records], dtype=np.float64)

    srocc = safe_corr(y_true_error, y_pred_error, method="spearman")
    krcc = safe_corr(y_true_error, y_pred_error, method="kendall")
    plcc_raw = safe_corr(y_true_error, y_pred_error, method="pearson")

    params_5pl, y_pred_mapped = fit_5pl_and_map(y_pred_error, y_true_error)
    plcc_5pl = safe_corr(y_true_error, y_pred_mapped, method="pearson")
    plcc_5pl_abs = abs(plcc_5pl)

    elapsed = time.time() - start

    pred_csv = os.path.join(args.out_dir, "pieapp_test_predictions.csv")
    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ref_image", "dist_image", "gt_error", "pred_quality", "pred_error", "pred_error_5pl_mapped"])
        for record, mapped in zip(records, y_pred_mapped.tolist()):
            writer.writerow(
                [
                    record.ref_image,
                    record.dist_image,
                    record.gt_error,
                    record.pred_quality,
                    record.pred_error,
                    float(mapped),
                ]
            )

    metrics = {
        "dataset": "PieAPP_test",
        "num_references": len(test_refs),
        "num_samples": len(records),
        "direction_note": "Model outputs quality-like score; converted to error-like score via pred_error = -pred_quality",
        "srocc": float(srocc),
        "krcc": float(krcc),
        "plcc_raw": float(plcc_raw),
        "plcc_5pl": float(plcc_5pl),
        "plcc_5pl_abs": float(plcc_5pl_abs),
        "five_pl_params": {
            "b1": float(params_5pl[0]),
            "b2": float(params_5pl[1]),
            "b3": float(params_5pl[2]),
            "b4": float(params_5pl[3]),
            "b5": float(params_5pl[4]),
        },
        "config": args.config,
        "checkpoint": args.ckpt,
        "pieapp_root": args.pieapp_root,
        "elapsed_sec": float(elapsed),
    }

    metrics_json = os.path.join(args.out_dir, "pieapp_test_metrics.json")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    summary_txt = os.path.join(args.out_dir, "summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("PieAPP test evaluation summary\n")
        f.write(f"samples: {len(records)}\n")
        f.write(f"references: {len(test_refs)}\n")
        f.write(f"SROCC: {srocc:.6f}\n")
        f.write(f"KRCC: {krcc:.6f}\n")
        f.write(f"PLCC(raw): {plcc_raw:.6f}\n")
        f.write(f"PLCC(5PL): {plcc_5pl:.6f}\n")
        f.write(f"|PLCC(5PL)|: {plcc_5pl_abs:.6f}\n")

    print("[pieapp] ===== Done =====")
    print(f"[pieapp] samples={len(records)} refs={len(test_refs)} elapsed_sec={elapsed:.2f}")
    print(f"[pieapp] SROCC={srocc:.6f} KRCC={krcc:.6f} PLCC(raw)={plcc_raw:.6f}")
    print(f"[pieapp] PLCC(5PL)={plcc_5pl:.6f} |PLCC(5PL)|={plcc_5pl_abs:.6f}")
    print(f"[pieapp] outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
