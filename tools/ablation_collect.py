import argparse
import csv
import json
import os
from typing import Dict, List

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite_root", type=str, required=True)
    parser.add_argument("--experiments", type=str, required=True, help="Comma-separated experiment names")
    parser.add_argument("--out_csv", type=str, default="")
    parser.add_argument("--out_md", type=str, default="")
    return parser.parse_args()


def safe_load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def gather_fold_metric_values(exp_dir: str, key: str) -> List[float]:
    values: List[float] = []
    if not os.path.isdir(exp_dir):
        return values
    for name in sorted(os.listdir(exp_dir)):
        if not name.startswith("fold_"):
            continue
        metric_path = os.path.join(exp_dir, name, "oof_metrics.json")
        payload = safe_load_json(metric_path)
        if payload is None:
            continue
        if key in payload:
            values.append(float(payload[key]))
    return values


def main():
    args = parse_args()
    experiments = [x.strip() for x in args.experiments.split(",") if x.strip()]
    if not experiments:
        raise ValueError("No experiments provided")

    out_csv = args.out_csv or os.path.join(args.suite_root, "ablation_summary.csv")
    out_md = args.out_md or os.path.join(args.suite_root, "ablation_summary.md")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    rows: List[Dict[str, str]] = []

    for exp in experiments:
        exp_dir = os.path.join(args.suite_root, exp)
        summary_path = os.path.join(exp_dir, "kfold_summary.json")
        summary = safe_load_json(summary_path)

        row: Dict[str, str] = {
            "experiment": exp,
            "status": "done" if summary is not None else "missing",
            "work_dir": exp_dir,
            "num_folds": "",
            "oof_count": "",
            "score": "",
            "srocc": "",
            "plcc": "",
            "mae": "",
            "rmse": "",
            "fold_score_mean": "",
            "fold_score_std": "",
            "fold_srocc_mean": "",
            "fold_srocc_std": "",
            "fold_plcc_mean": "",
            "fold_plcc_std": "",
            "mean_csv": "",
            "weighted_csv": "",
        }

        if summary is not None:
            oof_metrics = summary.get("oof_metrics", {})
            row["num_folds"] = str(summary.get("num_folds", ""))
            row["oof_count"] = str(summary.get("oof_count", ""))
            for key in ["score", "srocc", "plcc", "mae", "rmse"]:
                if key in oof_metrics:
                    row[key] = f"{float(oof_metrics[key]):.6f}"

            fold_score = gather_fold_metric_values(exp_dir, "score")
            fold_srocc = gather_fold_metric_values(exp_dir, "srocc")
            fold_plcc = gather_fold_metric_values(exp_dir, "plcc")
            if fold_score:
                row["fold_score_mean"] = f"{float(np.mean(fold_score)):.6f}"
                row["fold_score_std"] = f"{float(np.std(fold_score)):.6f}"
            if fold_srocc:
                row["fold_srocc_mean"] = f"{float(np.mean(fold_srocc)):.6f}"
                row["fold_srocc_std"] = f"{float(np.std(fold_srocc)):.6f}"
            if fold_plcc:
                row["fold_plcc_mean"] = f"{float(np.mean(fold_plcc)):.6f}"
                row["fold_plcc_std"] = f"{float(np.std(fold_plcc)):.6f}"

            ensemble = summary.get("inference_ensemble", {})
            row["mean_csv"] = str(ensemble.get("mean_csv", ""))
            row["weighted_csv"] = str(ensemble.get("weighted_csv", ""))

        rows.append(row)

    fieldnames = [
        "experiment",
        "status",
        "num_folds",
        "oof_count",
        "score",
        "srocc",
        "plcc",
        "mae",
        "rmse",
        "fold_score_mean",
        "fold_score_std",
        "fold_srocc_mean",
        "fold_srocc_std",
        "fold_plcc_mean",
        "fold_plcc_std",
        "mean_csv",
        "weighted_csv",
        "work_dir",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Ablation Summary\n\n")
        f.write("| experiment | score | srocc | plcc | mae | rmse | fold_score_mean±std | fold_srocc_mean±std | fold_plcc_mean±std | status |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            fs = (
                f"{row['fold_score_mean']}±{row['fold_score_std']}"
                if row["fold_score_mean"] and row["fold_score_std"]
                else ""
            )
            fr = (
                f"{row['fold_srocc_mean']}±{row['fold_srocc_std']}"
                if row["fold_srocc_mean"] and row["fold_srocc_std"]
                else ""
            )
            fp = (
                f"{row['fold_plcc_mean']}±{row['fold_plcc_std']}"
                if row["fold_plcc_mean"] and row["fold_plcc_std"]
                else ""
            )
            f.write(
                f"| {row['experiment']} | {row['score']} | {row['srocc']} | {row['plcc']} | {row['mae']} | {row['rmse']} | {fs} | {fr} | {fp} | {row['status']} |\n"
            )

    print(f"Saved CSV summary: {out_csv}")
    print(f"Saved Markdown summary: {out_md}")


if __name__ == "__main__":
    main()
