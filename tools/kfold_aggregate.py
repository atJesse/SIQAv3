import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from siqa.utils import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--weights", type=str, default="", help="Optional comma-separated fold weights, e.g. 0.2,0.2,0.2,0.2,0.2")
    parser.add_argument("--aggregate_inference", action="store_true", help="Aggregate per-fold inference CSVs under fold_i/submission/prediction.csv")
    parser.add_argument("--inference_relpath", type=str, default="submission/prediction.csv")
    return parser.parse_args()


def load_oof_csv(path: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    names: List[str] = []
    y_true: List[float] = []
    y_pred: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.append(row["img_name"])
            y_true.append(float(row["y_true"]))
            y_pred.append(float(row["y_pred"]))
    return names, np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)


def load_prediction_csv(path: str) -> Dict[str, float]:
    pred: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred[row["picture_name"]] = float(row["Score"])
    return pred


def parse_weights(weights_str: str, num_folds: int, work_dir: str) -> np.ndarray:
    if weights_str:
        values = [float(x.strip()) for x in weights_str.split(",") if x.strip()]
        if len(values) != num_folds:
            raise ValueError(f"weights length mismatch: expected {num_folds}, got {len(values)}")
        arr = np.asarray(values, dtype=np.float64)
    else:
        values = []
        for fold in range(num_folds):
            metric_path = os.path.join(work_dir, f"fold_{fold}", "best_metrics.json")
            score = 1.0
            if os.path.exists(metric_path):
                with open(metric_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                score = float(payload.get("best_score", 1.0))
            values.append(max(score, 1e-8))
        arr = np.asarray(values, dtype=np.float64)

    total = float(arr.sum())
    if total <= 0.0:
        raise ValueError("Invalid fold weights: sum must be > 0")
    return arr / total


def main():
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)

    all_names: List[str] = []
    all_true_list: List[np.ndarray] = []
    all_pred_list: List[np.ndarray] = []
    fold_sizes: Dict[str, int] = {}

    for fold in range(args.num_folds):
        oof_path = os.path.join(args.work_dir, f"fold_{fold}", "oof_val_predictions.csv")
        if not os.path.exists(oof_path):
            raise FileNotFoundError(f"Missing OOF file for fold {fold}: {oof_path}")
        names, y_true, y_pred = load_oof_csv(oof_path)
        all_names.extend(names)
        all_true_list.append(y_true)
        all_pred_list.append(y_pred)
        fold_sizes[f"fold_{fold}"] = len(names)

    y_true = np.concatenate(all_true_list)
    y_pred = np.concatenate(all_pred_list)
    oof_metrics = compute_metrics(y_true, y_pred)

    oof_csv_out = os.path.join(args.work_dir, "kfold_oof_predictions.csv")
    with open(oof_csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["img_name", "y_true", "y_pred"])
        for name, true_score, pred_score in zip(all_names, y_true.tolist(), y_pred.tolist()):
            writer.writerow([name, float(true_score), float(pred_score)])

    summary = {
        "num_folds": args.num_folds,
        "fold_sizes": fold_sizes,
        "oof_metrics": oof_metrics,
        "oof_count": int(len(y_true)),
    }

    if args.aggregate_inference:
        fold_preds: List[Dict[str, float]] = []
        for fold in range(args.num_folds):
            pred_path = os.path.join(args.work_dir, f"fold_{fold}", args.inference_relpath)
            if not os.path.exists(pred_path):
                raise FileNotFoundError(f"Missing inference prediction for fold {fold}: {pred_path}")
            fold_preds.append(load_prediction_csv(pred_path))

        first_names = sorted(fold_preds[0].keys())
        for idx in range(1, len(fold_preds)):
            other_names = sorted(fold_preds[idx].keys())
            if other_names != first_names:
                raise RuntimeError("Per-fold prediction files do not share identical image names")

        matrix = np.zeros((len(first_names), args.num_folds), dtype=np.float64)
        for fold in range(args.num_folds):
            for row_idx, name in enumerate(first_names):
                matrix[row_idx, fold] = fold_preds[fold][name]

        mean_scores = matrix.mean(axis=1)
        mean_out = os.path.join(args.work_dir, "kfold_prediction_mean.csv")
        with open(mean_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["picture_name", "Score"])
            for name, score in zip(first_names, mean_scores.tolist()):
                writer.writerow([name, float(score)])

        weights = parse_weights(args.weights, args.num_folds, args.work_dir)
        weighted_scores = matrix @ weights
        weighted_out = os.path.join(args.work_dir, "kfold_prediction_weighted.csv")
        with open(weighted_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["picture_name", "Score"])
            for name, score in zip(first_names, weighted_scores.tolist()):
                writer.writerow([name, float(score)])

        summary["inference_ensemble"] = {
            "mean_csv": mean_out,
            "weighted_csv": weighted_out,
            "normalized_weights": weights.tolist(),
            "num_images": len(first_names),
        }

    summary_path = os.path.join(args.work_dir, "kfold_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved OOF predictions: {oof_csv_out}")
    print(f"Saved OOF summary: {summary_path}")
    print(
        "OOF metrics | "
        f"score={oof_metrics['score']:.4f} "
        f"srocc={oof_metrics['srocc']:.4f} "
        f"plcc={oof_metrics['plcc']:.4f}"
    )
    if args.aggregate_inference:
        print("Saved ensemble CSVs: kfold_prediction_mean.csv, kfold_prediction_weighted.csv")


if __name__ == "__main__":
    main()
