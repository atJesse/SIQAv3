import argparse
import csv
import glob
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import yaml
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from siqa.dataset import build_eval_transform
from siqa.model import SiameseSemanticIQA


@dataclass
class PairRecord:
    subset: str
    sample_id: str
    judge_p1: float
    human_choice: int
    model_choice: int
    score_p0: float
    score_p1: float
    margin_p0_minus_p1: float
    correct: int
    soft_agreement: float


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SIQA model on BAPPS 2AFC val subsets")
    parser.add_argument("--config", type=str, default="/data/SIQAv3/configs/siqa_base.yaml")
    parser.add_argument("--ckpt", type=str, default="/data/SIQAdn2/workdirs/siqa_dual_dinov3_clip/fold_1/checkpoints/best.pth")
    parser.add_argument("--bapps_root", type=str, default="/data/dataset/BAPPS/dataset/2afc/val")
    parser.add_argument("--out_dir", type=str, default="/data/SIQAv3/BAPPS2afc/results")
    parser.add_argument(
        "--subsets",
        type=str,
        default="",
        help="Comma-separated subset names. Empty means all folders under bapps_root.",
    )
    parser.add_argument(
        "--max_samples_per_subset",
        type=int,
        default=0,
        help="Optional cap for quick debugging; 0 means use all samples.",
    )
    return parser.parse_args()


def resolve_clip_model_id(clip_name: str) -> str:
    alias_map = {
        "clip_vit_b32": "openai/clip-vit-base-patch32",
        "clip_vit_l14": "openai/clip-vit-large-patch14",
        "clip_vit_l14_336": "openai/clip-vit-large-patch14-336",
    }
    if clip_name in alias_map:
        return alias_map[clip_name]
    return clip_name


def hf_repo_to_cache_glob(repo_id: str) -> str:
    return repo_id.replace("/", "--")


def cache_hit_for_repo(repo_id: str) -> bool:
    repo_glob = hf_repo_to_cache_glob(repo_id)
    roots = [
        "/root/.cache/huggingface/hub",
        os.path.expanduser("~/.cache/huggingface/hub"),
        "/data/.cache/huggingface/hub",
    ]
    for root in roots:
        if not os.path.isdir(root):
            continue
        pattern = os.path.join(root, f"models--{repo_glob}*")
        for model_dir in glob.glob(pattern):
            snapshots = glob.glob(os.path.join(model_dir, "snapshots", "*"))
            blobs = glob.glob(os.path.join(model_dir, "blobs", "*"))
            if snapshots or blobs:
                return True
    return False


def print_runtime_diagnostics(cfg: Dict) -> None:
    model_cfg = cfg.get("model", {})

    hf_endpoint = os.getenv("HF_ENDPOINT", "<unset>")
    hf_transfer = os.getenv("HF_HUB_ENABLE_HF_TRANSFER", "<unset>")
    clip_name = model_cfg.get("clip_name", "clip_vit_l14_336")
    clip_model_id = resolve_clip_model_id(clip_name)
    structure_backbone = model_cfg.get("structure_backbone", model_cfg.get("swin_name", "swin_tiny_patch4_window7_224"))

    clip_local_dir = model_cfg.get("clip_local_dir", "")
    clip_local_files_only = bool(model_cfg.get("clip_local_files_only", False))
    clip_local_dir_exists = bool(clip_local_dir) and os.path.isdir(clip_local_dir)
    clip_cache_hit = cache_hit_for_repo(clip_model_id)

    dino_cache_hit = False
    if "dinov3" in structure_backbone:
        dino_cache_hit = cache_hit_for_repo(f"timm/{structure_backbone}")

    clip_offline_ready = clip_local_dir_exists or clip_local_files_only or clip_cache_hit
    structure_offline_ready = True
    if "dinov3" in structure_backbone:
        structure_offline_ready = dino_cache_hit

    offline_ready = clip_offline_ready and structure_offline_ready
    mode = "offline-capable" if offline_ready else "online-required"

    print("[startup] ===== Runtime Diagnostics =====")
    print(f"[startup] HF_ENDPOINT={hf_endpoint}")
    print(f"[startup] HF_HUB_ENABLE_HF_TRANSFER={hf_transfer}")
    print(f"[startup] structure_backbone={structure_backbone}")
    print(f"[startup] clip_model_id={clip_model_id}")
    print(f"[startup] clip_local_dir={clip_local_dir or '<empty>'} | exists={clip_local_dir_exists}")
    print(f"[startup] clip_local_files_only={clip_local_files_only}")
    print(f"[startup] cache_hit_clip={clip_cache_hit}")
    if "dinov3" in structure_backbone:
        print(f"[startup] cache_hit_dinov3={dino_cache_hit}")
    print(f"[startup] mode={mode}")
    if not offline_ready:
        print(
            "[startup] hint: for mainland China, set HF_ENDPOINT=https://hf-mirror.com "
            "and HF_HUB_ENABLE_HF_TRANSFER=0 before running."
        )
    print("[startup] =================================")


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


def list_subsets(root: str, subsets_arg: str) -> List[str]:
    if subsets_arg.strip():
        return [x.strip() for x in subsets_arg.split(",") if x.strip()]

    names = []
    for item in sorted(os.listdir(root)):
        path = os.path.join(root, item)
        if os.path.isdir(path):
            names.append(item)
    return names


def load_judge_value(path: str) -> float:
    arr = np.load(path)
    value = float(np.squeeze(arr))
    return value


def score_pair(model: SiameseSemanticIQA, transform, device: torch.device, ref_path: str, dist_path: str) -> float:
    ref_img = Image.open(ref_path).convert("RGB")
    dist_img = Image.open(dist_path).convert("RGB")
    ref_t = transform(ref_img).unsqueeze(0).to(device)
    dist_t = transform(dist_img).unsqueeze(0).to(device)
    with torch.no_grad():
        _, score = model(ref_t, dist_t)
    return float(score.item())


def evaluate_subset(
    subset: str,
    subset_dir: str,
    model: SiameseSemanticIQA,
    transform,
    device: torch.device,
    max_samples_per_subset: int,
) -> List[PairRecord]:
    judge_dir = os.path.join(subset_dir, "judge")
    p0_dir = os.path.join(subset_dir, "p0")
    p1_dir = os.path.join(subset_dir, "p1")
    ref_dir = os.path.join(subset_dir, "ref")

    for must_dir in [judge_dir, p0_dir, p1_dir, ref_dir]:
        if not os.path.isdir(must_dir):
            raise FileNotFoundError(f"Missing directory: {must_dir}")

    judge_files = sorted([x for x in os.listdir(judge_dir) if x.endswith(".npy")])
    if max_samples_per_subset > 0:
        judge_files = judge_files[:max_samples_per_subset]

    records: List[PairRecord] = []
    for npy_name in judge_files:
        sample_id = os.path.splitext(npy_name)[0]
        ref_path = os.path.join(ref_dir, f"{sample_id}.png")
        p0_path = os.path.join(p0_dir, f"{sample_id}.png")
        p1_path = os.path.join(p1_dir, f"{sample_id}.png")
        judge_path = os.path.join(judge_dir, npy_name)

        for p in [ref_path, p0_path, p1_path, judge_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing sample file: {p}")

        judge_p1 = load_judge_value(judge_path)
        human_choice = 1 if judge_p1 >= 0.5 else 0

        score_p0 = score_pair(model, transform, device, ref_path, p0_path)
        score_p1 = score_pair(model, transform, device, ref_path, p1_path)

        model_choice = 0 if score_p0 > score_p1 else 1
        correct = int(model_choice == human_choice)

        soft_agreement = judge_p1 if model_choice == 1 else (1.0 - judge_p1)

        records.append(
            PairRecord(
                subset=subset,
                sample_id=sample_id,
                judge_p1=judge_p1,
                human_choice=human_choice,
                model_choice=model_choice,
                score_p0=score_p0,
                score_p1=score_p1,
                margin_p0_minus_p1=score_p0 - score_p1,
                correct=correct,
                soft_agreement=soft_agreement,
            )
        )

    return records


def summarize_records(records: List[PairRecord]) -> Dict[str, float]:
    if not records:
        return {
            "count": 0,
            "accuracy_2afc": 0.0,
            "soft_agreement_mean": 0.0,
            "mean_margin_p0_minus_p1": 0.0,
            "model_choose_p0_ratio": 0.0,
            "human_choose_p0_ratio": 0.0,
        }

    count = len(records)
    correct = sum(r.correct for r in records)
    soft = sum(r.soft_agreement for r in records)
    mean_margin = float(np.mean([r.margin_p0_minus_p1 for r in records]))
    model_choose_p0_ratio = float(np.mean([1.0 if r.model_choice == 0 else 0.0 for r in records]))
    human_choose_p0_ratio = float(np.mean([1.0 if r.human_choice == 0 else 0.0 for r in records]))

    return {
        "count": count,
        "accuracy_2afc": correct / count,
        "soft_agreement_mean": soft / count,
        "mean_margin_p0_minus_p1": mean_margin,
        "model_choose_p0_ratio": model_choose_p0_ratio,
        "human_choose_p0_ratio": human_choose_p0_ratio,
    }


def save_records_csv(path: str, records: List[PairRecord]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "subset",
                "sample_id",
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
        for r in records:
            writer.writerow(
                [
                    r.subset,
                    r.sample_id,
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


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    per_subset_dir = os.path.join(args.out_dir, "per_subset")
    os.makedirs(per_subset_dir, exist_ok=True)

    if not os.path.isdir(args.bapps_root):
        raise FileNotFoundError(f"BAPPS root not found: {args.bapps_root}")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print_runtime_diagnostics(cfg)

    mean = cfg["data"].get("normalize_mean", [0.48145466, 0.4578275, 0.40821073])
    std = cfg["data"].get("normalize_std", [0.26862954, 0.26130258, 0.27577711])
    transform = build_eval_transform(cfg["data"]["image_size"], mean=mean, std=std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, args.ckpt, device)

    subsets = list_subsets(args.bapps_root, args.subsets)
    if not subsets:
        raise RuntimeError("No subsets found for evaluation")

    start = time.time()
    all_records: List[PairRecord] = []
    subset_summary_rows: List[Dict[str, float]] = []

    for subset in subsets:
        subset_dir = os.path.join(args.bapps_root, subset)
        if not os.path.isdir(subset_dir):
            raise FileNotFoundError(f"Subset dir not found: {subset_dir}")

        records = evaluate_subset(
            subset=subset,
            subset_dir=subset_dir,
            model=model,
            transform=transform,
            device=device,
            max_samples_per_subset=args.max_samples_per_subset,
        )
        all_records.extend(records)

        sub_summary = summarize_records(records)
        sub_summary["subset"] = subset
        subset_summary_rows.append(sub_summary)

        subset_csv = os.path.join(per_subset_dir, f"{subset}_scores.csv")
        save_records_csv(subset_csv, records)
        print(
            f"[subset={subset}] count={sub_summary['count']} "
            f"acc={sub_summary['accuracy_2afc']:.4f} "
            f"soft={sub_summary['soft_agreement_mean']:.4f}"
        )

    elapsed = time.time() - start

    all_csv = os.path.join(args.out_dir, "bapps_2afc_all_scores.csv")
    save_records_csv(all_csv, all_records)

    overall = summarize_records(all_records)
    overall["subsets"] = subsets
    overall["elapsed_sec"] = elapsed
    overall["runtime_per_pair_sec"] = elapsed / max(1, len(all_records))
    overall["config"] = args.config
    overall["ckpt"] = args.ckpt
    overall["bapps_root"] = args.bapps_root
    overall["device"] = str(device)
    overall["max_samples_per_subset"] = args.max_samples_per_subset

    subset_summary_csv = os.path.join(args.out_dir, "bapps_2afc_subset_summary.csv")
    with open(subset_summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "subset",
                "count",
                "accuracy_2afc",
                "soft_agreement_mean",
                "mean_margin_p0_minus_p1",
                "model_choose_p0_ratio",
                "human_choose_p0_ratio",
            ]
        )
        for row in subset_summary_rows:
            writer.writerow(
                [
                    row["subset"],
                    row["count"],
                    row["accuracy_2afc"],
                    row["soft_agreement_mean"],
                    row["mean_margin_p0_minus_p1"],
                    row["model_choose_p0_ratio"],
                    row["human_choose_p0_ratio"],
                ]
            )

    summary_json = os.path.join(args.out_dir, "bapps_2afc_summary.json")
    payload = {
        "overall": overall,
        "per_subset": subset_summary_rows,
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    info_txt = os.path.join(args.out_dir, "readme.txt")
    with open(info_txt, "w", encoding="utf-8") as f:
        f.write("BAPPS 2AFC evaluation (hard accuracy)\n")
        f.write(f"config: {args.config}\n")
        f.write(f"ckpt: {args.ckpt}\n")
        f.write(f"bapps_root: {args.bapps_root}\n")
        f.write(f"device: {device}\n")
        f.write(f"total_pairs: {overall['count']}\n")
        f.write(f"accuracy_2afc: {overall['accuracy_2afc']:.6f}\n")
        f.write(f"soft_agreement_mean: {overall['soft_agreement_mean']:.6f}\n")
        f.write(f"elapsed_sec: {elapsed:.2f}\n")
        f.write(f"runtime_per_pair_sec: {overall['runtime_per_pair_sec']:.6f}\n")

    print("Saved:")
    print(f"- {all_csv}")
    print(f"- {subset_summary_csv}")
    print(f"- {summary_json}")
    print(f"- {info_txt}")
    print(
        "Overall 2AFC hard accuracy: "
        f"{overall['accuracy_2afc']:.4f} "
        f"(count={overall['count']}, soft={overall['soft_agreement_mean']:.4f})"
    )


if __name__ == "__main__":
    main()
