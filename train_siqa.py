import argparse
import json
import os
import random
from typing import Dict, Tuple

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
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _resolve_norm_stats(cfg: Dict) -> Tuple[list, list]:
    data_cfg = cfg["data"]
    default_mean = [0.48145466, 0.4578275, 0.40821073]
    default_std = [0.26862954, 0.26130258, 0.27577711]
    mean = data_cfg.get("normalize_mean", default_mean)
    std = data_cfg.get("normalize_std", default_std)
    return mean, std


def build_dataloaders(cfg: Dict):
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    mean, std = _resolve_norm_stats(cfg)

    pairs = [PairSample(name=n, score=s) for n, s in read_score_table(data_cfg["score_file"])]
    scores = [p.score for p in pairs]
    tr_idx, va_idx = stratified_split_indices(scores, val_ratio=train_cfg["val_ratio"], seed=cfg["seed"])

    train_samples = [pairs[i] for i in tr_idx]
    val_samples = [pairs[i] for i in va_idx]

    train_ds = SemanticIQADataset(
        train_samples,
        ref_dir=data_cfg["ref_dir"],
        dist_dir=data_cfg["dist_dir"],
        transform=build_train_transform(data_cfg["image_size"], mean=mean, std=std),
    )
    val_ds = SemanticIQADataset(
        val_samples,
        ref_dir=data_cfg["ref_dir"],
        dist_dir=data_cfg["dist_dir"],
        transform=build_eval_transform(data_cfg["image_size"], mean=mean, std=std),
    )

    generator = torch.Generator()
    generator.manual_seed(cfg["seed"])
    pin_memory = bool(train_cfg.get("pin_memory", True))

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    return train_loader, val_loader


def evaluate(model, loader, device):
    model.eval()
    all_pred, all_true = [], []
    gate_trigger_count = 0
    hard_gate_trigger_count = 0
    soft_gate_trigger_count = 0
    sample_count = 0
    cos_values = []
    with torch.no_grad():
        for batch in loader:
            ref = batch["ref"].to(device, non_blocking=True)
            dist = batch["dist"].to(device, non_blocking=True)
            y = batch["score"].cpu().numpy()
            _, pred, aux = model(ref, dist, return_aux=True)
            all_pred.append(pred.detach().cpu().numpy())
            all_true.append(y)
            gate_mask = aux["gate_mask"]
            hard_gate_mask = aux.get("hard_gate_mask", torch.zeros_like(gate_mask, dtype=torch.bool))
            soft_gate_mask = aux.get("soft_gate_mask", torch.zeros_like(gate_mask, dtype=torch.bool))
            gate_trigger_count += int(gate_mask.sum().item())
            hard_gate_trigger_count += int(hard_gate_mask.sum().item())
            soft_gate_trigger_count += int(soft_gate_mask.sum().item())
            sample_count += int(gate_mask.numel())
            cos_values.append(aux["cos_sim"].detach().cpu().numpy())

    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    metrics = compute_metrics(y_true, y_pred)
    if sample_count > 0:
        metrics["gate_trigger_ratio"] = gate_trigger_count / sample_count
        metrics["hard_gate_trigger_ratio"] = hard_gate_trigger_count / sample_count
        metrics["soft_gate_trigger_ratio"] = soft_gate_trigger_count / sample_count
    else:
        metrics["gate_trigger_ratio"] = 0.0
        metrics["hard_gate_trigger_ratio"] = 0.0
        metrics["soft_gate_trigger_ratio"] = 0.0
    if cos_values:
        metrics["cos_sim_mean"] = float(np.mean(np.concatenate(cos_values)))
    else:
        metrics["cos_sim_mean"] = 0.0
    return metrics


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
    logger.info("Debug config = %s", cfg.get("debug", {}))

    train_loader, val_loader = build_dataloaders(cfg)
    logger.info("Train size = %d | Val size = %d", len(train_loader.dataset), len(val_loader.dataset))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "Building model backbones: Swin=%s, CLIP=%s (first run may download pretrained weights)",
        cfg["model"]["swin_name"],
        cfg["model"]["clip_name"],
    )
    model = SiameseSemanticIQA(
        num_classes=cfg["model"]["num_classes"],
        swin_name=cfg["model"].get("swin_name", "swin_tiny_patch4_window7_224"),
        clip_name=cfg["model"].get("clip_name", "clip_vit_b32"),
        freeze_backbones=cfg["model"].get("freeze_backbones", True),
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

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])

    ce_loss = nn.CrossEntropyLoss(label_smoothing=cfg["train"].get("label_smoothing", 0.0))
    mse_loss = nn.MSELoss()

    best_score = -1.0
    start_epoch = 1
    resume_path = args.resume
    if not resume_path and cfg["train"].get("auto_resume", True):
        auto_last = os.path.join(ckpt_dir, "last.pth")
        if os.path.exists(auto_last):
            resume_path = auto_last

    if resume_path:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        try:
            state = torch.load(resume_path, map_location=device)
            model.load_state_dict(state["model"])
            if "optimizer" in state:
                optimizer.load_state_dict(state["optimizer"])
            if "scheduler" in state:
                scheduler.load_state_dict(state["scheduler"])
            start_epoch = int(state.get("epoch", 0)) + 1
            best_score = float(state.get("metrics", {}).get("score", best_score))
            logger.info("Resumed from checkpoint: %s | start_epoch=%d | best_score=%.4f", resume_path, start_epoch, best_score)
        except Exception as exc:
            if args.resume:
                raise RuntimeError(f"Failed to resume from explicit checkpoint: {resume_path}") from exc
            logger.warning(
                "Auto-resume checkpoint is incompatible with current model and will be ignored: %s | reason=%s",
                resume_path,
                exc,
            )
            start_epoch = 1
            best_score = -1.0

    for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
        model.train()
        running_loss = 0.0
        steps_done = 0
        debug_cfg = cfg.get("debug", {})

        for step, batch in enumerate(train_loader, start=1):
            steps_done = step
            ref = batch["ref"].to(device, non_blocking=True)
            dist = batch["dist"].to(device, non_blocking=True)
            y_cls = batch["score_cls"].to(device, non_blocking=True)
            y_reg = batch["score"].to(device, non_blocking=True)

            if epoch == start_epoch and step == 1 and debug_cfg.get("print_batch_shapes", True):
                logger.info(
                    "Batch debug | ref=%s dist=%s y_cls=%s y_reg=%s",
                    tuple(ref.shape),
                    tuple(dist.shape),
                    tuple(y_cls.shape),
                    tuple(y_reg.shape),
                )

            logits, pred = model(ref, dist)
            loss = cfg["train"]["loss_weight_ce"] * ce_loss(logits, y_cls)
            loss += cfg["train"]["loss_weight_mse"] * mse_loss(pred, y_reg)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg["train"]["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optimizer.step()

            running_loss += float(loss.item())
            if step % debug_cfg.get("log_every", 10) == 0:
                lr = optimizer.param_groups[0]["lr"]
                log_msg = f"Epoch {epoch} Step {step} | Loss {running_loss / step:.4f} | LR {lr:.6e}"
                if torch.cuda.is_available() and debug_cfg.get("log_gpu_mem", True):
                    mem_alloc = torch.cuda.memory_allocated(device) / (1024**3)
                    mem_resv = torch.cuda.memory_reserved(device) / (1024**3)
                    log_msg += f" | GPU mem {mem_alloc:.2f}/{mem_resv:.2f} GB"
                logger.info(log_msg)

            if args.dry_run and step >= 2:
                break

        scheduler.step()

        metrics = evaluate(model, val_loader, device)
        logger.info(
            "Epoch %d done | train_loss %.4f | val_score %.4f | srocc %.4f | plcc %.4f | gate_ratio %.2f%% (hard %.2f%%, soft %.2f%%) | cos_mean %.4f",
            epoch,
            running_loss / max(1, steps_done),
            metrics["score"],
            metrics["srocc"],
            metrics["plcc"],
            100.0 * metrics.get("gate_trigger_ratio", 0.0),
            100.0 * metrics.get("hard_gate_trigger_ratio", 0.0),
            100.0 * metrics.get("soft_gate_trigger_ratio", 0.0),
            metrics.get("cos_sim_mean", 0.0),
        )

        last_ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metrics": metrics,
            "config": cfg,
        }
        last_ckpt_path = os.path.join(ckpt_dir, "last.pth")
        torch.save(last_ckpt, last_ckpt_path)
        if not os.path.exists(last_ckpt_path) or os.path.getsize(last_ckpt_path) == 0:
            raise RuntimeError(f"Checkpoint save failed: {last_ckpt_path}")

        if metrics["score"] > best_score:
            best_score = metrics["score"]
            best_ckpt_path = os.path.join(ckpt_dir, "best.pth")
            torch.save(last_ckpt, best_ckpt_path)
            if not os.path.exists(best_ckpt_path) or os.path.getsize(best_ckpt_path) == 0:
                raise RuntimeError(f"Best checkpoint save failed: {best_ckpt_path}")

        if args.dry_run:
            logger.info("Dry-run mode: stopping early.")
            break

    with open(os.path.join(cfg["output"]["work_dir"], "best_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"best_score": best_score}, f, indent=2)
    logger.info("Training finished. Best score = %.4f", best_score)


if __name__ == "__main__":
    main()
