import argparse
import csv
import glob
import os
import time

import torch
import yaml
from PIL import Image

from siqa.dataset import build_eval_transform
from siqa.model import SiameseSemanticIQA


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/siqa_base.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--ref_dir", type=str, default="/data/SIQA/Data/Val/Val/Ref")
    parser.add_argument("--dist_dir", type=str, default="/data/SIQA/Data/Val/Val/Dist")
    parser.add_argument("--out_dir", type=str, default="/data/SIQA/submission")
    parser.add_argument("--order_file", type=str, default="", help="Optional file listing required image order (one name per line or first CSV column).")
    return parser.parse_args()


def load_order(args):
    if args.order_file and os.path.exists(args.order_file):
        names = []
        with open(args.order_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                first = line.split(",")[0].strip()
                if first.lower().endswith(".png"):
                    names.append(first)
        if names:
            return names

    return sorted([os.path.basename(p) for p in glob.glob(os.path.join(args.dist_dir, "*.png"))])


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "prediction.csv")
    out_txt = os.path.join(args.out_dir, "readme.txt")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    names = load_order(args)
    if not names:
        raise RuntimeError("No images found for inference.")

    for name in names:
        if not os.path.exists(os.path.join(args.ref_dir, name)):
            raise FileNotFoundError(f"Missing ref image: {name}")
        if not os.path.exists(os.path.join(args.dist_dir, name)):
            raise FileNotFoundError(f"Missing dist image: {name}")

    mean = cfg["data"].get("normalize_mean", [0.48145466, 0.4578275, 0.40821073])
    std = cfg["data"].get("normalize_std", [0.26862954, 0.26130258, 0.27577711])
    transform = build_eval_transform(cfg["data"]["image_size"], mean=mean, std=std)

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
    start = time.time()
    with torch.no_grad():
        for name in names:
            ref = Image.open(os.path.join(args.ref_dir, name)).convert("RGB")
            dist = Image.open(os.path.join(args.dist_dir, name)).convert("RGB")
            ref_t = transform(ref).unsqueeze(0).to(device)
            dist_t = transform(dist).unsqueeze(0).to(device)
            _, score = model(ref_t, dist_t)
            rows.append((name, float(score.item())))
    elapsed = time.time() - start
    runtime_per_image = elapsed / len(rows)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["picture_name", "Score"])
        writer.writerows(rows)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"runtime per video [s] : {runtime_per_image:.4f}\n")
        f.write(f"CPU[1] / GPU[0] : {0 if torch.cuda.is_available() else 1}\n")
        f.write("Extra Data [1] / No Extra Data [0] : 1\n")
        f.write("Other description : Dual-backbone Siamese IQA (Swin+CLIP) with cosine semantic feature and semantic gate veto.\n")

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_txt}")
    print(f"Images: {len(rows)} | runtime per image: {runtime_per_image:.4f}s")


if __name__ == "__main__":
    main()
