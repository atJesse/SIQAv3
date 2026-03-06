import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from siqa.dataset import read_score_table


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit VQEG-style 5-parameter logistic mapping on train predictions and apply to test predictions."
    )
    parser.add_argument("--train_pred_csv", type=str, required=True, help="CSV with training predictions")
    parser.add_argument("--label_xlsx", type=str, default="", help="Train label xlsx (e.g., Train_scores.xlsx)")
    parser.add_argument("--label_csv", type=str, default="", help="Alternative label csv with (name, score)")
    parser.add_argument("--test_pred_csv", type=str, required=True, help="CSV with test/submission predictions")
    parser.add_argument("--out_csv", type=str, default="tools/output_logistic_mapping/prediction_mapped.csv")
    parser.add_argument("--out_json", type=str, default="tools/output_logistic_mapping/logistic_5pl_params.json")
    parser.add_argument("--clip_min", type=float, default=0.0)
    parser.add_argument("--clip_max", type=float, default=5.0)
    return parser.parse_args()


def auto_find_name_score_cols(fieldnames: List[str]) -> Tuple[str, str]:
    lower_map = {k.lower(): k for k in fieldnames}

    name_candidates = ["img_name", "image_name", "picture_name", "name", "filename"]
    score_candidates = ["pred_score", "score", "prediction", "mos", "y_pred"]

    name_col = None
    score_col = None

    for candidate in name_candidates:
        if candidate in lower_map:
            name_col = lower_map[candidate]
            break

    for candidate in score_candidates:
        if candidate in lower_map:
            score_col = lower_map[candidate]
            break

    if name_col is None:
        name_col = fieldnames[0]
    if score_col is None:
        score_col = fieldnames[1] if len(fieldnames) > 1 else fieldnames[0]

    return name_col, score_col


def read_prediction_csv(path: str) -> List[Tuple[str, float]]:
    rows: List[Tuple[str, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"No header found in prediction csv: {path}")
        name_col, score_col = auto_find_name_score_cols(reader.fieldnames)
        for row in reader:
            name = str(row[name_col]).strip()
            score = float(row[score_col])
            rows.append((name, score))
    if not rows:
        raise RuntimeError(f"No data found in prediction csv: {path}")
    return rows


def read_label_csv(path: str) -> Dict[str, float]:
    labels: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"No header found in label csv: {path}")
        name_col, score_col = auto_find_name_score_cols(reader.fieldnames)
        for row in reader:
            name = str(row[name_col]).strip()
            if not name.lower().endswith(".png"):
                name = f"{name}.png"
            labels[name] = float(row[score_col])
    if not labels:
        raise RuntimeError(f"No labels found in label csv: {path}")
    return labels


def build_label_dict(label_xlsx: str, label_csv: str) -> Dict[str, float]:
    if label_xlsx:
        pairs = read_score_table(label_xlsx)
        return {name: float(score) for name, score in pairs}
    if label_csv:
        return read_label_csv(label_csv)
    raise ValueError("Either --label_xlsx or --label_csv must be provided")


def five_param_logistic(x: np.ndarray, b1: float, b2: float, b3: float, b4: float, b5: float) -> np.ndarray:
    z = np.clip(b2 * (x - b3), -60.0, 60.0)
    return b1 * (0.5 - 1.0 / (1.0 + np.exp(z))) + b4 * x + b5


def safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    return ranks


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    plcc = safe_corrcoef(y_true, y_pred)
    srocc = safe_corrcoef(rankdata(y_true), rankdata(y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {
        "plcc": plcc,
        "srocc": srocc,
        "rmse": rmse,
        "mae": mae,
        "score": 0.6 * srocc + 0.4 * plcc,
    }


def fit_logistic_5pl(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    try:
        from scipy.optimize import curve_fit
    except Exception as exc:
        raise RuntimeError(
            "scipy is required for 5-parameter logistic fitting. Please install scipy first."
        ) from exc

    x = x.astype(np.float64)
    y = y.astype(np.float64)

    p0 = np.array([
        float(np.max(y) - np.min(y)),
        1.0,
        float(np.median(x)),
        1.0,
        float(np.mean(y)),
    ], dtype=np.float64)

    params, _ = curve_fit(
        five_param_logistic,
        x,
        y,
        p0=p0,
        maxfev=200000,
    )

    y_fit = five_param_logistic(x, *params)
    metrics = compute_metrics(y, y_fit)
    return params, metrics


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)

    train_pred = read_prediction_csv(args.train_pred_csv)
    labels = build_label_dict(args.label_xlsx, args.label_csv)

    matched_names: List[str] = []
    y_true_list: List[float] = []
    y_pred_list: List[float] = []

    for name, pred in train_pred:
        if name in labels:
            matched_names.append(name)
            y_true_list.append(labels[name])
            y_pred_list.append(pred)

    if len(y_true_list) < 20:
        raise RuntimeError(
            f"Too few matched samples for fitting: {len(y_true_list)}. "
            "Please check image names in train prediction csv and labels."
        )

    y_true = np.array(y_true_list, dtype=np.float64)
    y_pred = np.array(y_pred_list, dtype=np.float64)

    metrics_before = compute_metrics(y_true, y_pred)
    params, metrics_after = fit_logistic_5pl(y_pred, y_true)

    test_pred_rows = read_prediction_csv(args.test_pred_csv)
    mapped_rows = []
    for name, pred in test_pred_rows:
        mapped = float(five_param_logistic(np.array([pred], dtype=np.float64), *params)[0])
        mapped = float(np.clip(mapped, args.clip_min, args.clip_max))
        mapped_rows.append((name, mapped, float(pred)))

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["picture_name", "Score", "raw_score"])
        writer.writerows(mapped_rows)

    out_info = {
        "train_pred_csv": args.train_pred_csv,
        "test_pred_csv": args.test_pred_csv,
        "label_xlsx": args.label_xlsx,
        "label_csv": args.label_csv,
        "matched_samples": len(y_true_list),
        "metrics_before_mapping": metrics_before,
        "metrics_after_mapping": metrics_after,
        "beta": {
            "b1": float(params[0]),
            "b2": float(params[1]),
            "b3": float(params[2]),
            "b4": float(params[3]),
            "b5": float(params[4]),
        },
        "function": "y = b1 * (0.5 - 1 / (1 + exp(b2*(x-b3)))) + b4*x + b5",
        "clip_range": [args.clip_min, args.clip_max],
        "output_csv": args.out_csv,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out_info, f, indent=2, ensure_ascii=False)

    print("=== 5PL logistic mapping complete ===")
    print(f"Matched training samples: {len(y_true_list)}")
    print(
        "Before mapping | score={:.4f}, srocc={:.4f}, plcc={:.4f}, rmse={:.4f}".format(
            metrics_before["score"], metrics_before["srocc"], metrics_before["plcc"], metrics_before["rmse"]
        )
    )
    print(
        "After mapping  | score={:.4f}, srocc={:.4f}, plcc={:.4f}, rmse={:.4f}".format(
            metrics_after["score"], metrics_after["srocc"], metrics_after["plcc"], metrics_after["rmse"]
        )
    )
    print(
        "Betas: b1={:.6f}, b2={:.6f}, b3={:.6f}, b4={:.6f}, b5={:.6f}".format(
            params[0], params[1], params[2], params[3], params[4]
        )
    )
    print(f"Saved mapped csv: {args.out_csv}")
    print(f"Saved params json: {args.out_json}")


if __name__ == "__main__":
    main()
