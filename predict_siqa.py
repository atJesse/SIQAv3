import argparse
import csv
import os

import torch
import yaml
from torch.utils.data import DataLoader

from siqa.dataset import PairSample, SemanticIQADataset, build_eval_transform, read_score_table
from siqa.model import SiameseSemanticIQA


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
    samples = [PairSample(name=n, score=s) for n, s in read_score_table(data_cfg["score_file"])]
    ds = SemanticIQADataset(
        samples,
        ref_dir=data_cfg["ref_dir"],
        dist_dir=data_cfg["dist_dir"],
        transform=build_eval_transform(data_cfg["image_size"]),
    )
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["num_workers"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseSemanticIQA(
        num_classes=cfg["model"]["num_classes"],
        backbone_name=cfg["model"]["backbone"],
        freeze_backbone=False,
    ).to(device)

    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
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
