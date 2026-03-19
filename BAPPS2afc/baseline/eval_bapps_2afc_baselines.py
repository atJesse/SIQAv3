import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import piq
import torch
from PIL import Image


@dataclass
class MetricRecord:
    subset: str
    sample_id: str
    metric: str
    judge_p1: float
    human_choice: int
    model_choice: int
    score_p0: float
    score_p1: float
    margin_p0_minus_p1: float
    correct: int
    soft_agreement: float


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate 2AFC accuracy on BAPPS using classical/perceptual IQA metrics")
    parser.add_argument("--bapps_root", type=str, default="/data/dataset/BAPPS/dataset/2afc/val")
    parser.add_argument("--out_dir", type=str, default="/data/SIQAv3/BAPPS2afc/baseline/results")
    parser.add_argument(
        "--subsets",
        type=str,
        default="",
        help="Comma-separated subset names. Empty = all subsets under bapps_root",
    )
    parser.add_argument(
        "--max_samples_per_subset",
        type=int,
        default=0,
        help="Optional cap for quick debugging; 0 means all samples",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu",
    )
    return parser.parse_args()


def list_subsets(root: str, subsets_arg: str) -> List[str]:
    if subsets_arg.strip():
        return [x.strip() for x in subsets_arg.split(",") if x.strip()]
    return sorted([x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x))])


def load_img(path: str, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t


def compute_metric_scores(ref: torch.Tensor, dist: torch.Tensor, metric_name: str, lpips_model, dists_model) -> float:
    if metric_name == "PSNR":
        return float(piq.psnr(ref, dist, data_range=1.0).item())
    if metric_name == "SSIM":
        return float(piq.ssim(ref, dist, data_range=1.0).item())
    if metric_name == "FSIM":
        return float(piq.fsim(ref, dist, data_range=1.0, chromatic=True).item())
    if metric_name == "VIF":
        return float(piq.vif_p(ref, dist, data_range=1.0).item())
    if metric_name == "LPIPS":
        return float(lpips_model(ref, dist).item())
    if metric_name == "DISTS":
        return float(dists_model(ref, dist).item())
    raise ValueError(f"Unsupported metric: {metric_name}")


def summarize(records: List[MetricRecord]) -> Dict[str, float]:
    if not records:
        return {
            "count": 0,
            "accuracy_2afc": 0.0,
            "soft_agreement_mean": 0.0,
            "model_choose_p0_ratio": 0.0,
            "human_choose_p0_ratio": 0.0,
            "mean_margin_p0_minus_p1": 0.0,
        }

    count = len(records)
    correct = sum(r.correct for r in records)
    return {
        "count": count,
        "accuracy_2afc": correct / count,
        "soft_agreement_mean": float(np.mean([r.soft_agreement for r in records])),
        "model_choose_p0_ratio": float(np.mean([1.0 if r.model_choice == 0 else 0.0 for r in records])),
        "human_choose_p0_ratio": float(np.mean([1.0 if r.human_choice == 0 else 0.0 for r in records])),
        "mean_margin_p0_minus_p1": float(np.mean([r.margin_p0_minus_p1 for r in records])),
    }


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but torch.cuda.is_available() is False")

    metric_names = ["PSNR", "SSIM", "FSIM", "VIF", "LPIPS", "DISTS"]
    high_better = {"PSNR", "SSIM", "FSIM", "VIF"}

    lpips_model = piq.LPIPS(reduction="mean").to(device).eval()
    dists_model = piq.DISTS(reduction="mean").to(device).eval()

    subsets = list_subsets(args.bapps_root, args.subsets)
    if not subsets:
        raise RuntimeError("No subsets found")

    start = time.time()
    all_records: List[MetricRecord] = []

    for subset in subsets:
        subset_dir = os.path.join(args.bapps_root, subset)
        judge_dir = os.path.join(subset_dir, "judge")
        p0_dir = os.path.join(subset_dir, "p0")
        p1_dir = os.path.join(subset_dir, "p1")
        ref_dir = os.path.join(subset_dir, "ref")

        for d in [judge_dir, p0_dir, p1_dir, ref_dir]:
            if not os.path.isdir(d):
                raise FileNotFoundError(f"Missing dir: {d}")

        judge_files = sorted([x for x in os.listdir(judge_dir) if x.endswith(".npy")])
        if args.max_samples_per_subset > 0:
            judge_files = judge_files[: args.max_samples_per_subset]

        subset_records: List[MetricRecord] = []

        for idx, npy_name in enumerate(judge_files, start=1):
            sample_id = os.path.splitext(npy_name)[0]
            ref_path = os.path.join(ref_dir, f"{sample_id}.png")
            p0_path = os.path.join(p0_dir, f"{sample_id}.png")
            p1_path = os.path.join(p1_dir, f"{sample_id}.png")
            judge_path = os.path.join(judge_dir, npy_name)

            judge_p1 = float(np.squeeze(np.load(judge_path)))
            human_choice = 1 if judge_p1 >= 0.5 else 0

            ref_t = load_img(ref_path, device)
            p0_t = load_img(p0_path, device)
            p1_t = load_img(p1_path, device)

            with torch.no_grad():
                for metric in metric_names:
                    score_p0 = compute_metric_scores(ref_t, p0_t, metric, lpips_model, dists_model)
                    score_p1 = compute_metric_scores(ref_t, p1_t, metric, lpips_model, dists_model)

                    if metric in high_better:
                        model_choice = 0 if score_p0 > score_p1 else 1
                    else:
                        model_choice = 0 if score_p0 < score_p1 else 1

                    correct = int(model_choice == human_choice)
                    soft_agreement = judge_p1 if model_choice == 1 else (1.0 - judge_p1)

                    record = MetricRecord(
                        subset=subset,
                        sample_id=sample_id,
                        metric=metric,
                        judge_p1=judge_p1,
                        human_choice=human_choice,
                        model_choice=model_choice,
                        score_p0=score_p0,
                        score_p1=score_p1,
                        margin_p0_minus_p1=score_p0 - score_p1,
                        correct=correct,
                        soft_agreement=soft_agreement,
                    )
                    subset_records.append(record)
                    all_records.append(record)

            if idx % 500 == 0:
                print(f"[subset={subset}] processed {idx}/{len(judge_files)} samples")

        print(f"[subset={subset}] done samples={len(judge_files)}")

    elapsed = time.time() - start

    all_csv = os.path.join(args.out_dir, "bapps_2afc_metrics_all_pairs.csv")
    with open(all_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "subset",
                "sample_id",
                "metric",
                "judge_p1",
                "human_choice",
                "model_choice",
                "score_p0",
                "score_p1",
                "margin_p0_minus_p1",
                "correct",
                "soft_agreement",
            ]
        )
        for r in all_records:
            writer.writerow(
                [
                    r.subset,
                    r.sample_id,
                    r.metric,
                    r.judge_p1,
                    r.human_choice,
                    r.model_choice,
                    r.score_p0,
                    r.score_p1,
                    r.margin_p0_minus_p1,
                    r.correct,
                    r.soft_agreement,
                ]
            )

    subset_summary_rows = []
    overall_summary_rows = []

    for subset in subsets:
        subset_records = [r for r in all_records if r.subset == subset]
        for metric in metric_names:
            metric_records = [r for r in subset_records if r.metric == metric]
            stats = summarize(metric_records)
            row = {"subset": subset, "metric": metric, **stats}
            subset_summary_rows.append(row)

    for metric in metric_names:
        metric_records = [r for r in all_records if r.metric == metric]
        stats = summarize(metric_records)
        row = {"subset": "ALL", "metric": metric, **stats}
        overall_summary_rows.append(row)

    subset_csv = os.path.join(args.out_dir, "bapps_2afc_metrics_subset_summary.csv")
    with open(subset_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "subset",
                "metric",
                "count",
                "accuracy_2afc",
                "soft_agreement_mean",
                "model_choose_p0_ratio",
                "human_choose_p0_ratio",
                "mean_margin_p0_minus_p1",
            ]
        )
        for r in subset_summary_rows:
            writer.writerow(
                [
                    r["subset"],
                    r["metric"],
                    r["count"],
                    r["accuracy_2afc"],
                    r["soft_agreement_mean"],
                    r["model_choose_p0_ratio"],
                    r["human_choose_p0_ratio"],
                    r["mean_margin_p0_minus_p1"],
                ]
            )

    overall_csv = os.path.join(args.out_dir, "bapps_2afc_metrics_overall_summary.csv")
    with open(overall_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "metric",
                "count",
                "accuracy_2afc",
                "soft_agreement_mean",
                "model_choose_p0_ratio",
                "human_choose_p0_ratio",
                "mean_margin_p0_minus_p1",
            ]
        )
        for r in overall_summary_rows:
            writer.writerow(
                [
                    r["metric"],
                    r["count"],
                    r["accuracy_2afc"],
                    r["soft_agreement_mean"],
                    r["model_choose_p0_ratio"],
                    r["human_choose_p0_ratio"],
                    r["mean_margin_p0_minus_p1"],
                ]
            )

    summary_json = os.path.join(args.out_dir, "bapps_2afc_metrics_summary.json")
    payload = {
        "meta": {
            "bapps_root": args.bapps_root,
            "subsets": subsets,
            "num_samples_per_subset": args.max_samples_per_subset,
            "device": str(device),
            "elapsed_sec": elapsed,
            "runtime_per_pair_metric_sec": elapsed / max(1, len(all_records)),
            "metrics": metric_names,
        },
        "overall": overall_summary_rows,
        "per_subset": subset_summary_rows,
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    report_txt = os.path.join(args.out_dir, "readme.txt")
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("BAPPS 2AFC baseline metric evaluation\n")
        f.write(f"bapps_root: {args.bapps_root}\n")
        f.write(f"device: {device}\n")
        f.write(f"elapsed_sec: {elapsed:.2f}\n")
        f.write("Overall accuracy by metric:\n")
        for r in overall_summary_rows:
            f.write(f"  {r['metric']}: acc={r['accuracy_2afc']:.6f}, soft={r['soft_agreement_mean']:.6f}\n")

    print("Saved:")
    print(f"- {all_csv}")
    print(f"- {subset_csv}")
    print(f"- {overall_csv}")
    print(f"- {summary_json}")
    print(f"- {report_txt}")


if __name__ == "__main__":
    main()
