import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from siqa.dataset import PairSample, SemanticIQADataset, build_eval_transform, read_score_table


@dataclass
class SampleRecord:
    name: str
    score: float
    score_cls: int
    cos_sim: float


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze CLIP semantic cosine similarity vs IQA score")
    parser.add_argument("--config", type=str, default="configs/siqa_base.yaml")
    parser.add_argument("--output_dir", type=str, default="tools/output_clip_semantic_analysis")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means use all samples")
    parser.add_argument("--force_online", action="store_true", help="Ignore local-files-only and allow online CLIP loading")
    return parser.parse_args()


def resolve_norm_stats(cfg: Dict) -> Tuple[list, list]:
    data_cfg = cfg["data"]
    mean = data_cfg.get("normalize_mean", [0.48145466, 0.4578275, 0.40821073])
    std = data_cfg.get("normalize_std", [0.26862954, 0.26130258, 0.27577711])
    return mean, std


def resolve_clip_model_id(clip_name: str) -> str:
    alias_map = {
        "clip_vit_b32": "openai/clip-vit-base-patch32",
        "clip_vit_l14": "openai/clip-vit-large-patch14",
        "clip_vit_l14_336": "openai/clip-vit-large-patch14-336",
    }
    if clip_name in alias_map:
        return alias_map[clip_name]
    if "/" in clip_name:
        return clip_name
    raise ValueError(
        "Unsupported CLIP backbone alias. Use one of "
        "{'clip_vit_b32','clip_vit_l14','clip_vit_l14_336'} or a HuggingFace model id."
    )


def build_clip_model(cfg: Dict, force_online: bool = False):
    from transformers import CLIPVisionModel

    model_cfg = cfg.get("model", {})
    clip_name = model_cfg.get("clip_name", "clip_vit_l14_336")
    clip_local_dir = model_cfg.get("clip_local_dir", "")
    clip_local_files_only = bool(model_cfg.get("clip_local_files_only", False)) and not force_online

    if clip_local_dir and os.path.isdir(clip_local_dir) and not force_online:
        model_id = clip_local_dir
        local_files_only = True
    else:
        model_id = resolve_clip_model_id(clip_name)
        local_files_only = clip_local_files_only

    model = CLIPVisionModel.from_pretrained(model_id, local_files_only=local_files_only)
    model.eval()
    return model, model_id


def extract_clip_features(model, x: torch.Tensor, interpolate_pos_encoding: bool = True) -> torch.Tensor:
    outputs = model(pixel_values=x, interpolate_pos_encoding=interpolate_pos_encoding)
    if outputs.pooler_output is not None:
        return outputs.pooler_output
    return outputs.last_hidden_state[:, 0, :]


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def summarize_by_class(records: List[SampleRecord]) -> List[Dict]:
    out = []
    for cls in range(6):
        cls_vals = np.array([r.cos_sim for r in records if r.score_cls == cls], dtype=np.float64)
        if len(cls_vals) == 0:
            out.append(
                {
                    "score_cls": cls,
                    "count": 0,
                    "min": np.nan,
                    "max": np.nan,
                    "mean": np.nan,
                    "std": np.nan,
                    "p05": np.nan,
                    "p25": np.nan,
                    "p50": np.nan,
                    "p75": np.nan,
                    "p95": np.nan,
                }
            )
            continue

        out.append(
            {
                "score_cls": cls,
                "count": int(len(cls_vals)),
                "min": float(np.min(cls_vals)),
                "max": float(np.max(cls_vals)),
                "mean": float(np.mean(cls_vals)),
                "std": float(np.std(cls_vals)),
                "p05": float(np.percentile(cls_vals, 5)),
                "p25": float(np.percentile(cls_vals, 25)),
                "p50": float(np.percentile(cls_vals, 50)),
                "p75": float(np.percentile(cls_vals, 75)),
                "p95": float(np.percentile(cls_vals, 95)),
            }
        )
    return out


def save_raw_csv(path: str, records: List[SampleRecord]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["img_name", "score", "score_cls", "cos_sim"])
        for r in records:
            writer.writerow([r.name, f"{r.score:.6f}", r.score_cls, f"{r.cos_sim:.8f}"])


def save_class_summary_csv(path: str, rows: List[Dict]) -> None:
    keys = ["score_cls", "count", "min", "max", "mean", "std", "p05", "p25", "p50", "p75", "p95"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_plots(output_dir: str, records: List[SampleRecord], class_rows: List[Dict], summary: Dict) -> str:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return "matplotlib not available; skip plotting"

    scores = np.array([r.score for r in records], dtype=np.float64)
    cls_scores = np.array([r.score_cls for r in records], dtype=np.int64)
    cos_vals = np.array([r.cos_sim for r in records], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=140)

    jitter = (np.random.default_rng(3407).random(len(cls_scores)) - 0.5) * 0.12
    axes[0].scatter(cls_scores + jitter, cos_vals, s=14, alpha=0.65)
    axes[0].set_title("CLIP Cosine Similarity by Score Class")
    axes[0].set_xlabel("Score class")
    axes[0].set_ylabel("Cosine similarity")
    axes[0].set_xticks([0, 1, 2, 3, 4, 5])
    axes[0].grid(alpha=0.25)

    xs = np.array([row["score_cls"] for row in class_rows if row["count"] > 0], dtype=np.float64)
    ys = np.array([row["mean"] for row in class_rows if row["count"] > 0], dtype=np.float64)
    if len(xs) >= 2:
        axes[0].plot(xs, ys, color="red", linewidth=2, marker="o", label="Class mean")
        axes[0].legend(loc="best")

    axes[1].scatter(cos_vals, scores, s=14, alpha=0.6)
    if len(cos_vals) >= 2:
        slope = summary.get("linear_fit", {}).get("slope", 0.0)
        intercept = summary.get("linear_fit", {}).get("intercept", 0.0)
        x_line = np.linspace(float(np.min(cos_vals)), float(np.max(cos_vals)), 120)
        y_line = slope * x_line + intercept
        axes[1].plot(x_line, y_line, color="orange", linewidth=2, label="Linear fit")
        axes[1].legend(loc="best")
    axes[1].set_title("Score vs CLIP Cosine Similarity")
    axes[1].set_xlabel("Cosine similarity")
    axes[1].set_ylabel("Score")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    plot_path = os.path.join(output_dir, "clip_score_relationship.png")
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    pairs = [PairSample(name=n, score=s) for n, s in read_score_table(cfg["data"]["score_file"])]
    if args.max_samples > 0:
        pairs = pairs[: args.max_samples]

    mean, std = resolve_norm_stats(cfg)
    ds = SemanticIQADataset(
        pairs,
        ref_dir=cfg["data"]["ref_dir"],
        dist_dir=cfg["data"]["dist_dir"],
        transform=build_eval_transform(cfg["data"]["image_size"], mean=mean, std=std),
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, model_id = build_clip_model(cfg, force_online=args.force_online)
    clip_model = clip_model.to(device)

    interpolate = bool(cfg.get("model", {}).get("clip_interpolate_pos_encoding", True))

    records: List[SampleRecord] = []
    with torch.no_grad():
        for batch in loader:
            ref = batch["ref"].to(device, non_blocking=True)
            dist = batch["dist"].to(device, non_blocking=True)
            score = batch["score"].cpu().numpy()
            names = batch["name"]

            feat_ref = extract_clip_features(clip_model, ref, interpolate_pos_encoding=interpolate)
            feat_dist = extract_clip_features(clip_model, dist, interpolate_pos_encoding=interpolate)
            cos_sim = torch.nn.functional.cosine_similarity(feat_ref, feat_dist, dim=1).detach().cpu().numpy()

            for name, s, c in zip(names, score.tolist(), cos_sim.tolist()):
                records.append(SampleRecord(name=name, score=float(s), score_cls=int(round(float(s))), cos_sim=float(c)))

    if not records:
        raise RuntimeError("No records found for analysis.")

    class_rows = summarize_by_class(records)

    score_vals = np.array([r.score for r in records], dtype=np.float64)
    cos_vals = np.array([r.cos_sim for r in records], dtype=np.float64)

    pearson = safe_corr(score_vals, cos_vals)
    spearman = safe_corr(rankdata(score_vals), rankdata(cos_vals))

    if len(cos_vals) >= 2:
        slope, intercept = np.polyfit(cos_vals, score_vals, 1)
        y_hat = slope * cos_vals + intercept
        ss_res = float(np.sum((score_vals - y_hat) ** 2))
        ss_tot = float(np.sum((score_vals - np.mean(score_vals)) ** 2))
        r2 = 0.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    else:
        slope, intercept, r2 = 0.0, 0.0, 0.0

    summary = {
        "n_samples": len(records),
        "clip_model_source": model_id,
        "pearson_score_cos": float(pearson),
        "spearman_score_cos": float(spearman),
        "linear_fit": {
            "slope": float(slope),
            "intercept": float(intercept),
            "r2": float(r2),
        },
    }

    raw_csv = os.path.join(args.output_dir, "clip_cosine_per_pair.csv")
    cls_csv = os.path.join(args.output_dir, "clip_cosine_class_summary.csv")
    summary_json = os.path.join(args.output_dir, "clip_cosine_global_summary.json")

    save_raw_csv(raw_csv, records)
    save_class_summary_csv(cls_csv, class_rows)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    plot_info = save_plots(args.output_dir, records, class_rows, summary)

    print("=== CLIP semantic cosine analysis done ===")
    print(f"Samples: {len(records)}")
    print(f"Model source: {model_id}")
    print(f"Pearson(score, cos): {summary['pearson_score_cos']:.6f}")
    print(f"Spearman(score, cos): {summary['spearman_score_cos']:.6f}")
    print(f"Linear fit: score = {summary['linear_fit']['slope']:.6f} * cos + {summary['linear_fit']['intercept']:.6f}")
    print(f"Linear R^2: {summary['linear_fit']['r2']:.6f}")
    print(f"Saved raw csv: {raw_csv}")
    print(f"Saved class summary csv: {cls_csv}")
    print(f"Saved global summary: {summary_json}")
    print(f"Plot output: {plot_info}")


if __name__ == "__main__":
    main()
