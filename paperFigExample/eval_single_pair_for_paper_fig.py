import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import piq
import torch
import yaml
from PIL import Image
from scipy.optimize import curve_fit

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from siqa.dataset import ResizePadSquare, build_eval_transform, read_score_table
from siqa.model import SiameseSemanticIQA


@dataclass
class MetricMapping:
    name: str
    higher_is_better: bool
    method: str
    params: List[float]
    raw_min: float
    raw_max: float
    raw_mean: float
    raw_std: float
    corr_raw_plcc: float
    corr_raw_srocc: float
    corr_mapped_plcc: float
    corr_mapped_srocc: float


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one LoViF pair with baselines and DSS-SQA, then map to 0-5")
    parser.add_argument("--ref", type=str, default="/data/dataset/LoViF/Train/Ref/000087.png")
    parser.add_argument("--dist", type=str, default="/data/dataset/LoViF/Train/Dist/000087.png")
    parser.add_argument("--train_ref_dir", type=str, default="/data/dataset/LoViF/Train/Ref")
    parser.add_argument("--train_dist_dir", type=str, default="/data/dataset/LoViF/Train/Dist")
    parser.add_argument("--train_scores_xlsx", type=str, default="/data/dataset/LoViF/Train/Train_scores.xlsx")
    parser.add_argument("--config", type=str, default="/data/SIQAv3/configs/siqa_base.yaml")
    parser.add_argument("--ckpt", type=str, default="/data/SIQAdn2/workdirs/siqa_dual_dinov3_clip/fold_1/checkpoints/best.pth")
    parser.add_argument("--out_dir", type=str, default="/data/SIQAv3/paperFigExample")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_train_samples", type=int, default=0, help="0 means use all train samples for mapping")
    parser.add_argument("--metric_image_size", type=int, default=256, help="Resize side length for baseline metrics")
    return parser.parse_args()


def five_param_logistic(x: np.ndarray, b1: float, b2: float, b3: float, b4: float, b5: float) -> np.ndarray:
    z = np.clip(b2 * (x - b3), -60.0, 60.0)
    return b1 * (0.5 - 1.0 / (1.0 + np.exp(z))) + b4 * x + b5


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
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


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    return safe_pearson(rankdata(x), rankdata(y))


def fit_mapping(raw_scores: np.ndarray, mos_scores: np.ndarray, higher_is_better: bool) -> Tuple[Callable[[np.ndarray], np.ndarray], MetricMapping]:
    x = raw_scores.astype(np.float64)
    y = mos_scores.astype(np.float64)

    if not higher_is_better:
        x_for_fit = -x
    else:
        x_for_fit = x

    p0 = np.array(
        [
            float(np.max(y) - np.min(y)),
            1.0,
            float(np.median(x_for_fit)),
            1.0,
            float(np.mean(y)),
        ],
        dtype=np.float64,
    )

    method = "5pl"
    params: List[float]

    try:
        fit_params, _ = curve_fit(
            five_param_logistic,
            x_for_fit,
            y,
            p0=p0,
            maxfev=200000,
        )
        params = [float(v) for v in fit_params]

        def mapper(v: np.ndarray) -> np.ndarray:
            vv = v.astype(np.float64)
            if not higher_is_better:
                vv = -vv
            out = five_param_logistic(vv, *fit_params)
            return np.clip(out, 0.0, 5.0)

    except Exception:
        method = "minmax"
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        if x_max <= x_min:
            x_max = x_min + 1e-6

        params = [x_min, x_max]

        def mapper(v: np.ndarray) -> np.ndarray:
            vv = v.astype(np.float64)
            if higher_is_better:
                out = 5.0 * (vv - x_min) / (x_max - x_min)
            else:
                out = 5.0 * (x_max - vv) / (x_max - x_min)
            return np.clip(out, 0.0, 5.0)

    mapped_train = mapper(x)
    corr_raw_plcc = safe_pearson(y, x if higher_is_better else -x)
    corr_raw_srocc = safe_spearman(y, x if higher_is_better else -x)
    corr_mapped_plcc = safe_pearson(y, mapped_train)
    corr_mapped_srocc = safe_spearman(y, mapped_train)

    mapping = MetricMapping(
        name="",
        higher_is_better=higher_is_better,
        method=method,
        params=params,
        raw_min=float(np.min(x)),
        raw_max=float(np.max(x)),
        raw_mean=float(np.mean(x)),
        raw_std=float(np.std(x)),
        corr_raw_plcc=float(corr_raw_plcc),
        corr_raw_srocc=float(corr_raw_srocc),
        corr_mapped_plcc=float(corr_mapped_plcc),
        corr_mapped_srocc=float(corr_mapped_srocc),
    )
    return mapper, mapping


def load_img(path: str, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def load_img_for_metrics(path: str, device: torch.device, image_size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = ResizePadSquare(image_size)(img)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def build_dss_model(cfg: Dict, ckpt_path: str, device: torch.device) -> SiameseSemanticIQA:
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


def score_dss(model: SiameseSemanticIQA, transform, device: torch.device, ref_path: str, dist_path: str) -> float:
    ref_img = Image.open(ref_path).convert("RGB")
    dist_img = Image.open(dist_path).convert("RGB")
    ref_t = transform(ref_img).unsqueeze(0).to(device)
    dist_t = transform(dist_img).unsqueeze(0).to(device)
    with torch.no_grad():
        _, score = model(ref_t, dist_t)
    return float(score.item())


def compute_raw_metrics(
    ref_t: torch.Tensor,
    dist_t: torch.Tensor,
    lpips_model,
    dists_model,
) -> Dict[str, float]:
    out = {
        "PSNR": float(piq.psnr(ref_t, dist_t, data_range=1.0).item()),
        "SSIM": float(piq.ssim(ref_t, dist_t, data_range=1.0).item()),
        "FSIM": float(piq.fsim(ref_t, dist_t, data_range=1.0, chromatic=True).item()),
        "VIF": float(piq.vif_p(ref_t, dist_t, data_range=1.0).item()),
        "LPIPS_VGG_PIQ": float(lpips_model(ref_t, dist_t).item()),
        "DISTS": float(dists_model(ref_t, dist_t).item()),
    }
    return out


def quality_percentile(values: np.ndarray, x: float, higher_is_better: bool) -> float:
    arr = values.astype(np.float64)
    if higher_is_better:
        return float(np.mean(arr <= x) * 100.0)
    return float(np.mean(arr >= x) * 100.0)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but torch.cuda.is_available() is False")

    for p in [args.ref, args.dist, args.train_ref_dir, args.train_dist_dir, args.train_scores_xlsx, args.config, args.ckpt]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing path: {p}")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    mean = cfg["data"].get("normalize_mean", [0.48145466, 0.4578275, 0.40821073])
    std = cfg["data"].get("normalize_std", [0.26862954, 0.26130258, 0.27577711])
    transform = build_eval_transform(cfg["data"]["image_size"], mean=mean, std=std)

    print("[paper-fig] loading models...")
    lpips_model = piq.LPIPS(reduction="mean").to(device).eval()
    dists_model = piq.DISTS(reduction="mean").to(device).eval()
    dss_model = build_dss_model(cfg, args.ckpt, device)

    score_pairs = read_score_table(args.train_scores_xlsx)
    train_items: List[Tuple[str, float]] = []
    for name, mos in score_pairs:
        ref_path = os.path.join(args.train_ref_dir, name)
        dist_path = os.path.join(args.train_dist_dir, name)
        if os.path.isfile(ref_path) and os.path.isfile(dist_path):
            train_items.append((name, float(mos)))

    if args.max_train_samples > 0:
        train_items = train_items[: args.max_train_samples]

    if len(train_items) < 30:
        raise RuntimeError(f"Too few train items for mapping: {len(train_items)}")

    mos_array = np.array([mos for _, mos in train_items], dtype=np.float64)

    metric_higher_better = {
        "PSNR": True,
        "SSIM": True,
        "FSIM": True,
        "VIF": True,
        "LPIPS_VGG_PIQ": False,
        "DISTS": False,
    }

    train_raw: Dict[str, List[float]] = {k: [] for k in metric_higher_better.keys()}
    train_dss: List[float] = []

    print(f"[paper-fig] computing train raw metrics for mapping... n={len(train_items)}")
    for idx, (name, _mos) in enumerate(train_items, start=1):
        ref_path = os.path.join(args.train_ref_dir, name)
        dist_path = os.path.join(args.train_dist_dir, name)

        ref_t = load_img_for_metrics(ref_path, device, args.metric_image_size)
        dist_t = load_img_for_metrics(dist_path, device, args.metric_image_size)

        with torch.no_grad():
            raw = compute_raw_metrics(ref_t, dist_t, lpips_model, dists_model)

        for k, v in raw.items():
            train_raw[k].append(float(v))

        dss_score = score_dss(dss_model, transform, device, ref_path, dist_path)
        train_dss.append(float(dss_score))

        if idx % 100 == 0 or idx == len(train_items):
            print(f"[paper-fig] processed {idx}/{len(train_items)}")

    mapping_functions: Dict[str, Callable[[np.ndarray], np.ndarray]] = {}
    mapping_info: Dict[str, MetricMapping] = {}
    mapped_train_by_metric: Dict[str, np.ndarray] = {}

    for metric_name, high_better in metric_higher_better.items():
        raw_arr = np.array(train_raw[metric_name], dtype=np.float64)
        mapper, info = fit_mapping(raw_arr, mos_array, higher_is_better=high_better)
        info.name = metric_name
        mapping_functions[metric_name] = mapper
        mapping_info[metric_name] = info
        mapped_train_by_metric[metric_name] = mapper(raw_arr)

    baseline_metric_order = ["PSNR", "SSIM", "FSIM", "VIF", "LPIPS_VGG_PIQ", "DISTS"]
    baseline_mapped_matrix = np.stack([mapped_train_by_metric[m] for m in baseline_metric_order], axis=1)
    baseline_mapped_avg = np.mean(baseline_mapped_matrix, axis=1)
    dss_train_arr = np.array(train_dss, dtype=np.float64)

    contrast_rows: List[Dict[str, float]] = []
    for i, (name, mos) in enumerate(train_items):
        avg_base = float(baseline_mapped_avg[i])
        dss_val = float(dss_train_arr[i])
        mos_val = float(mos)
        contrast_gap = float(avg_base - dss_val)
        baseline_abs_err = float(abs(avg_base - mos_val))
        dss_abs_err = float(abs(dss_val - mos_val))
        alignment_gain = float(baseline_abs_err - dss_abs_err)
        contrast_rows.append(
            {
                "name": name,
                "mos": mos_val,
                "baseline_avg_mapped": avg_base,
                "dss_score": dss_val,
                "contrast_gap": contrast_gap,
                "baseline_abs_err": baseline_abs_err,
                "dss_abs_err": dss_abs_err,
                "alignment_gain": alignment_gain,
            }
        )

    contrast_rows.sort(key=lambda x: (x["alignment_gain"], x["contrast_gap"]), reverse=True)

    contrast_csv = os.path.join(args.out_dir, "top_contrast_candidates.csv")
    with open(contrast_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "name",
                "mos",
                "baseline_avg_mapped",
                "dss_score",
                "contrast_gap",
                "baseline_abs_err",
                "dss_abs_err",
                "alignment_gain",
            ]
        )
        for rank, row in enumerate(contrast_rows[:50], start=1):
            writer.writerow(
                [
                    rank,
                    row["name"],
                    row["mos"],
                    row["baseline_avg_mapped"],
                    row["dss_score"],
                    row["contrast_gap"],
                    row["baseline_abs_err"],
                    row["dss_abs_err"],
                    row["alignment_gain"],
                ]
            )

    pair_name = os.path.basename(args.dist)
    current_pair_rank = None
    for rank, row in enumerate(contrast_rows, start=1):
        if row["name"] == pair_name:
            current_pair_rank = rank
            break

    ref_t_case = load_img_for_metrics(args.ref, device, args.metric_image_size)
    dist_t_case = load_img_for_metrics(args.dist, device, args.metric_image_size)
    with torch.no_grad():
        case_raw = compute_raw_metrics(ref_t_case, dist_t_case, lpips_model, dists_model)

    case_dss = score_dss(dss_model, transform, device, args.ref, args.dist)

    gt_mos = None
    label_dict = {name: float(mos) for name, mos in score_pairs}
    if pair_name in label_dict:
        gt_mos = label_dict[pair_name]

    rows = []
    for metric_name in ["PSNR", "SSIM", "FSIM", "VIF", "LPIPS_VGG_PIQ", "DISTS"]:
        raw_value = float(case_raw[metric_name])
        mapped_value = float(mapping_functions[metric_name](np.array([raw_value], dtype=np.float64))[0])
        percentile = quality_percentile(
            np.array(train_raw[metric_name], dtype=np.float64),
            raw_value,
            higher_is_better=metric_higher_better[metric_name],
        )
        rows.append(
            {
                "metric": metric_name,
                "raw_score": raw_value,
                "mapped_0_5": mapped_value,
                "quality_percentile_in_train": percentile,
                "mapping_method": mapping_info[metric_name].method,
            }
        )

    dss_percentile = float(np.mean(np.array(train_dss, dtype=np.float64) <= case_dss) * 100.0)
    rows.append(
        {
            "metric": "DSS-SQA",
            "raw_score": float(case_dss),
            "mapped_0_5": float(np.clip(case_dss, 0.0, 5.0)),
            "quality_percentile_in_train": dss_percentile,
            "mapping_method": "native_model_output",
        }
    )

    csv_path = os.path.join(args.out_dir, "single_pair_scores_for_plot.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "raw_score", "mapped_0_5", "quality_percentile_in_train", "mapping_method"])
        for r in rows:
            writer.writerow([r["metric"], r["raw_score"], r["mapped_0_5"], r["quality_percentile_in_train"], r["mapping_method"]])

    mapping_json = os.path.join(args.out_dir, "metric_mapping_details.json")
    mapping_dump = {}
    for k, info in mapping_info.items():
        mapping_dump[k] = {
            "higher_is_better": info.higher_is_better,
            "method": info.method,
            "params": info.params,
            "raw_min": info.raw_min,
            "raw_max": info.raw_max,
            "raw_mean": info.raw_mean,
            "raw_std": info.raw_std,
            "corr_raw_plcc": info.corr_raw_plcc,
            "corr_raw_srocc": info.corr_raw_srocc,
            "corr_mapped_plcc": info.corr_mapped_plcc,
            "corr_mapped_srocc": info.corr_mapped_srocc,
        }

    with open(mapping_json, "w", encoding="utf-8") as f:
        json.dump(mapping_dump, f, indent=2, ensure_ascii=False)

    result_json = os.path.join(args.out_dir, "single_pair_result.json")
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "pair": {
                    "ref": args.ref,
                    "dist": args.dist,
                    "name": pair_name,
                    "mos_if_available": gt_mos,
                },
                "checkpoint": args.ckpt,
                "config": args.config,
                "device": str(device),
                "train_samples_used_for_mapping": len(train_items),
                "metric_image_size": args.metric_image_size,
                "baseline_preprocess": "ResizePadSquare(image_size)+ToTensor in [0,1] (no normalization)",
                "dss_preprocess": "build_eval_transform(image_size from config): ResizePadSquare+ToTensor+Normalize(mean,std)",
                "current_pair_contrast_rank": current_pair_rank,
                "metrics": rows,
                "note": "LPIPS here is piq.LPIPS, which is VGG-based LPIPS variant.",
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    md_path = os.path.join(args.out_dir, "single_pair_readout.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Single Pair Metric Readout\n\n")
        f.write(f"- Ref: `{args.ref}`\n")
        f.write(f"- Dist: `{args.dist}`\n")
        if gt_mos is not None:
            f.write(f"- MOS (from Train_scores.xlsx): `{gt_mos:.4f}`\n")
        else:
            f.write("- MOS: not found in score table\n")
        f.write(f"- Mapping calibration set size: `{len(train_items)}`\n\n")
        if current_pair_rank is not None:
            f.write(f"- Contrast rank in train set (higher is more eye-catching): `{current_pair_rank}` / `{len(train_items)}`\n")
            f.write(f"- Top contrast candidates file: `{contrast_csv}`\n\n")
        f.write("| Metric | Raw | Mapped(0-5) | Quality Percentile(Train) |\n")
        f.write("|---|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['metric']} | {r['raw_score']:.6f} | {r['mapped_0_5']:.6f} | {r['quality_percentile_in_train']:.2f}% |\n"
            )

    print("[paper-fig] done")
    print(f"[paper-fig] outputs:\n- {csv_path}\n- {result_json}\n- {mapping_json}\n- {md_path}\n- {contrast_csv}")


if __name__ == "__main__":
    main()
