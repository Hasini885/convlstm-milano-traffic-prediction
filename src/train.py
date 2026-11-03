"""Train the ConvLSTM forecaster with early stopping, LR scheduling and best-checkpoint saving.

Usage::

    python -m src.train --config configs/default.yaml [--set training.epochs=20 ...]
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .dataset import build_dataloaders, build_datasets
from .model import build_model, residual_positions
from .utils import add_common_args, get_device, get_logger, load_config, resolve_path, save_json, set_seed


def masked_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None, kind: str = "mse", huber_delta: float = 1.0
) -> torch.Tensor:
    """MSE or Huber loss averaged over the cells where ``mask`` (an ``(H, W)`` bool) is True."""
    if kind == "mse":
        err = (pred - target) ** 2
    elif kind == "huber":
        err = F.huber_loss(pred, target, reduction="none", delta=huber_delta)
    else:
        raise ValueError(f"Unknown loss {kind!r}")
    if mask is None:
        return err.mean()
    m = mask.to(err.dtype)
    return (err * m).sum() / (m.sum() * (err.numel() // m.numel())).clamp_min(1)


class EarlyStopping:
    """Stop when the monitored value has not improved by ``min_delta`` for ``patience`` epochs."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        self.patience, self.min_delta = patience, min_delta
        self.best = math.inf
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        """Record ``value``; return True if it is a new best."""
        if value < self.best - self.min_delta:
            self.best, self.bad_epochs = value, 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_cfg: dict[str, Any],
    mask: torch.Tensor | None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    grad_clip: float | None = None,
    max_batches: int | None = None,
) -> float:
    """Run one pass over ``loader``. Trains if ``optimizer`` is given, otherwise only evaluates.

    Returns:
        Mean loss per batch.
    """
    training = optimizer is not None
    model.train(training)
    total, n = 0.0, 0
    grad_ctx = contextlib.nullcontext() if training else torch.no_grad()
    with grad_ctx:
        for b, (x, y) in enumerate(loader):
            if max_batches is not None and b >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                pred = model(x)
            loss = masked_loss(pred.float(), y, mask, loss_cfg["loss"], loss_cfg.get("huber_delta", 1.0))
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
            total += loss.item()
            n += 1
    return total / max(n, 1)


def train(cfg: dict[str, Any]) -> Path:
    """Train a model according to ``cfg`` and return the path of the best checkpoint."""
    out_dir = resolve_path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("train", out_dir / "train.log")
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))
    tcfg = cfg["training"]
    save_json(cfg, out_dir / "config_used.json")

    bundle = build_datasets(cfg)
    loaders = build_dataloaders(cfg, bundle.datasets)
    mask = bundle.mask.to(device) if bundle.mask is not None else None
    logger.info(f"Device: {device} | windows train/val/test = "
                + "/".join(str(len(bundle.datasets[s])) for s in ("train", "val", "test")))
    if mask is not None:
        logger.info(f"Loss mask: {int(mask.sum())}/{mask.numel()} cells active in the training period")

    residual = residual_positions(bundle.input_channels, bundle.target_channels)
    if cfg["model"].get("residual", False) and residual is None:
        logger.info("model.residual requested, but some target features are not inputs; disabling it")
    model = build_model(cfg, len(bundle.input_channels), len(bundle.target_channels), residual).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n_params:,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 0.0))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=tcfg.get("scheduler_factor", 0.5),
        patience=tcfg.get("scheduler_patience", 5), min_lr=tcfg.get("min_lr", 1e-6))
    use_amp = bool(tcfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    stopper = EarlyStopping(tcfg.get("patience", 10), tcfg.get("min_delta", 0.0))
    ckpt_path = out_dir / "best_model.pt"
    history_path = out_dir / "history.csv"

    with open(history_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "lr", "seconds"])

    max_seconds = tcfg.get("max_hours") and tcfg["max_hours"] * 3600
    run_start = time.time()
    for epoch in range(1, tcfg["epochs"] + 1):
        t0 = time.time()
        train_loss = run_epoch(model, loaders["train"], device, tcfg, mask, optimizer, scaler, use_amp,
                               tcfg.get("grad_clip"), tcfg.get("max_train_batches"))
        val_loss = run_epoch(model, loaders["val"], device, tcfg, mask, use_amp=use_amp,
                             max_batches=tcfg.get("max_eval_batches"))
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        improved = stopper.step(val_loss)
        if improved:
            torch.save({
                "model_state": model.state_dict(),
                "config": cfg,
                "epoch": epoch,
                "val_loss": val_loss,
                "normalizer": bundle.normalizer.state_dict(),
                "input_channels": bundle.input_channels,
                "target_channels": bundle.target_channels,
                "residual_channels": residual if cfg["model"].get("residual", False) else None,
                "features": bundle.meta["features"],
            }, ckpt_path)
        logger.info(f"Epoch {epoch:3d} | train {train_loss:.6f} | val {val_loss:.6f} | lr {lr:.2e} | "
                    f"{elapsed:.1f}s{'  * saved best' if improved else ''}")
        with open(history_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, lr, round(elapsed, 2)])

        if stopper.should_stop:
            logger.info(f"Early stopping: no val improvement for {stopper.patience} epochs")
            break
        # Stop before the next epoch would exceed the time budget (e.g. a Kaggle session limit).
        if max_seconds and time.time() - run_start + elapsed > max_seconds:
            logger.info(f"Stopping: time budget of {tcfg['max_hours']} h reached")
            break

    logger.info(f"Best val loss {stopper.best:.6f}; checkpoint: {ckpt_path}")
    return ckpt_path


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Train the ConvLSTM forecaster."))
    args = parser.parse_args()
    train(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
