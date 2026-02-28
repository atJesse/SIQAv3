import argparse
import json
import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from siqa.dataset import (
    PairSample,
    SemanticIQADataset,
    build_eval_transform,
    build_train_transform,
    read_score_table,
    stratified_split_indices,
)
from siqa.model import SiameseSemanticIQA
from siqa.utils import build_logger, compute_metrics, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/siqa_base.yaml")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg: Dict):
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    pairs = [PairSample(name=n, score=s) for n, s in read_score_table(data_cfg["score_file"])]
    scores = [p.score for p in pairs]
    tr_idx, va_idx = stratified_split_indices(scores, val_ratio=train_cfg["val_ratio"], seed=cfg["seed"])

    train_samples = [pairs[i] for i in tr_idx]
    val_samples = [pairs[i] for i in va_idx]

    train_ds = SemanticIQADataset(
        train_samples,
        ref_dir=data_cfg["ref_dir"],
        dist_dir=data_cfg["dist_dir"],
        transform=build_train_transform(data_cfg["image_size"]),
    )
    val_ds = SemanticIQADataset(
        val_samples,
        ref_dir=data_cfg["ref_dir"],
        dist_dir=data_cfg["dist_dir"],
        transform=build_eval_transform(data_cfg["image_size"]),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
    )
    return train_loader, val_loader


def evaluate(model, loader, device):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            ref = batch["ref"].to(device, non_blocking=True)
            dist = batch["dist"].to(device, non_blocking=True)
            y = batch["score"].cpu().numpy()
            _, pred = model(ref, dist)
            all_pred.append(pred.detach().cpu().numpy())
            all_true.append(y)

    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    return compute_metrics(y_true, y_pred)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    os.makedirs(cfg["output"]["work_dir"], exist_ok=True)
    ckpt_dir = os.path.join(cfg["output"]["work_dir"], "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = build_logger(os.path.join(cfg["output"]["work_dir"], "train.log"))

    logger.info("Config loaded from %s", args.config)
    logger.info("Seed = %d", cfg["seed"])

    train_loader, val_loader = build_dataloaders(cfg)
    logger.info("Train size = %d | Val size = %d", len(train_loader.dataset), len(val_loader.dataset))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseSemanticIQA(
        num_classes=cfg["model"]["num_classes"],
        backbone_name=cfg["model"]["backbone"],
        freeze_backbone=cfg["model"]["freeze_backbone"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])

    ce_loss = nn.CrossEntropyLoss(label_smoothing=cfg["train"].get("label_smoothing", 0.0))
    mse_loss = nn.MSELoss()

    best_score = -1.0
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            ref = batch["ref"].to(device, non_blocking=True)
            dist = batch["dist"].to(device, non_blocking=True)
            y_cls = batch["score_cls"].to(device, non_blocking=True)
            y_reg = batch["score"].to(device, non_blocking=True)

            logits, pred = model(ref, dist)
            loss = cfg["train"]["loss_weight_ce"] * ce_loss(logits, y_cls)
            loss += cfg["train"]["loss_weight_mse"] * mse_loss(pred, y_reg)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg["train"]["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optimizer.step()

            running_loss += float(loss.item())
            if step % cfg["debug"]["log_every"] == 0:
                logger.info("Epoch %d Step %d | Loss %.4f", epoch, step, running_loss / step)

            if args.dry_run and step >= 2:
                break

        scheduler.step()

        metrics = evaluate(model, val_loader, device)
        logger.info(
            "Epoch %d done | train_loss %.4f | val_score %.4f | srocc %.4f | plcc %.4f",
            epoch,
            running_loss / max(1, step),
            metrics["score"],
            metrics["srocc"],
            metrics["plcc"],
        )

        last_ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metrics": metrics,
            "config": cfg,
        }
        torch.save(last_ckpt, os.path.join(ckpt_dir, "last.pth"))

        if metrics["score"] > best_score:
            best_score = metrics["score"]
            torch.save(last_ckpt, os.path.join(ckpt_dir, "best.pth"))

        if args.dry_run:
            logger.info("Dry-run mode: stopping early.")
            break

    with open(os.path.join(cfg["output"]["work_dir"], "best_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"best_score": best_score}, f, indent=2)
    logger.info("Training finished. Best score = %.4f", best_score)


if __name__ == "__main__":
    main()
