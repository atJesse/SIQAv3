import argparse
import csv
import os

import torch
import yaml
from torch.utils.data import DataLoader

from siqa.dataset import PairSample, SemanticIQADataset, build_eval_transform, read_score_table
from siqa.model import SiameseSemanticIQA


def resolve_norm_stats(cfg):
    data_cfg = cfg["data"]
    mean = data_cfg.get("normalize_mean", [0.48145466, 0.4578275, 0.40821073])
    std = data_cfg.get("normalize_std", [0.26862954, 0.26130258, 0.27577711])
    return mean, std


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/siqa_base.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output", type=str, default="predictions.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    mean, std = resolve_norm_stats(cfg)
    samples = [PairSample(name=n, score=s) for n, s in read_score_table(data_cfg["score_file"])]
    ds = SemanticIQADataset(
        samples,
        ref_dir=data_cfg["ref_dir"],
        dist_dir=data_cfg["dist_dir"],
        transform=build_eval_transform(data_cfg["image_size"], mean=mean, std=std),
    )
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["num_workers"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseSemanticIQA(
        num_classes=cfg["model"]["num_classes"],
        swin_name=cfg["model"].get("swin_name", "swin_tiny_patch4_window7_224"),
        clip_name=cfg["model"].get("clip_name", "clip_vit_b32"),
        freeze_backbones=False,
        swin_local_path=cfg["model"].get("swin_local_path", ""),
        clip_local_dir=cfg["model"].get("clip_local_dir", ""),
        clip_local_files_only=cfg["model"].get("clip_local_files_only", False),
        clip_interpolate_pos_encoding=cfg["model"].get("clip_interpolate_pos_encoding", True),
        bottleneck_dim=cfg["model"].get("bottleneck_dim", 256),
        bottleneck_dropout=cfg["model"].get("bottleneck_dropout", 0.5),
        semantic_gate_enabled=cfg["model"].get("semantic_gate_enabled", True),
        semantic_gate_threshold=cfg["model"].get("semantic_gate_threshold", 0.4),
        gate_logit_strength=cfg["model"].get("gate_logit_strength", 12.0),
    ).to(device)

    state = torch.load(args.ckpt, map_location=device)
    model_state = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(model_state)
    model.eval()

    rows = []
    with torch.no_grad():
        for batch in loader:
            ref = batch["ref"].to(device)
            dist = batch["dist"].to(device)
            _, score = model(ref, dist)
            for name, s in zip(batch["name"], score.cpu().numpy()):
                rows.append((name, float(s)))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["img_name", "pred_score"])
        writer.writerows(rows)

    print(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    main()
