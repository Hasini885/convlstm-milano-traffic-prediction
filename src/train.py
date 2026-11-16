"""Train the ConvLSTM forecaster with early stopping, LR scheduling and best-checkpoint saving.

Usage::

    python -m src.train --config configs/default.yaml [--resume] [--set training.epochs=20 ...]
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import math
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .dataset import build_dataloaders, build_datasets
from .model import build_model, residual_positions
from .utils import add_common_args, get_device, get_logger, load_config, resolve_path, save_json, set_seed

LAST_CHECKPOINT = "last_checkpoint.pt"


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
    """Stop when the monitored value has not improved *noticeably* for ``patience`` epochs.

    Any improvement is a new best (and is checkpointed), but only an improvement of more than
    ``min_delta + min_delta_rel * best`` resets the patience counter. Without the relative
    threshold, improvements in the seventh significant digit keep a converged run alive.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, min_delta_rel: float = 0.0) -> None:
        self.patience, self.min_delta, self.min_delta_rel = patience, min_delta, min_delta_rel
        self.best = math.inf
        self.reference = math.inf  # last value that counted as a noticeable improvement
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        """Record ``value``; return True if it is a new best."""
        if value < self.reference * (1 - self.min_delta_rel) - self.min_delta:
            self.reference, self.bad_epochs = value, 0
        else:
            self.bad_epochs += 1
        if value < self.best:
            self.best = value
            return True
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience

    def state_dict(self) -> dict[str, float]:
        return {"best": self.best, "reference": self.reference, "bad_epochs": self.bad_epochs}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.best, self.reference, self.bad_epochs = state["best"], state["reference"], int(state["bad_epochs"])


def run_epoch(
    model: nn.Module | None,
    loader: DataLoader,
    device: torch.device,
    loss_cfg: dict[str, Any],
    mask: torch.Tensor | None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    grad_clip: float | None = None,
    max_batches: int | None = None,
    persistence_channels: Sequence[int] | None = None,
) -> float:
    """Run one pass over ``loader``. Trains if ``optimizer`` is given, otherwise only evaluates.

    With ``model=None`` the persistence forecast (the last input frame of ``persistence_channels``,
    repeated for every horizon) is scored instead, which gives the loss a reference point.

    Returns:
        Mean loss per sample.
    """
    training = optimizer is not None
    if model is not None:
        model.train(training)
    total, n = 0.0, 0
    grad_ctx = contextlib.nullcontext() if training else torch.no_grad()
    with grad_ctx:
        for b, (x, y) in enumerate(loader):
            if max_batches is not None and b >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if model is None:
                pred = x[:, -1:, list(persistence_channels)].expand_as(y)
            else:
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
            # Weight by batch size so a short final batch does not count as much as a full one.
            total += loss.item() * x.shape[0]
            n += x.shape[0]
    return total / max(n, 1)


def train(cfg: dict[str, Any], resume: bool = False) -> Path:
    """Train a model according to ``cfg`` and return the path of the best checkpoint.

    With ``resume=True`` and a ``last_checkpoint.pt`` in the output directory, training continues
    from the end of the last finished epoch (model, optimiser, scheduler and early-stopping state).
    """
    out_dir = resolve_path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    last_path = out_dir / LAST_CHECKPOINT
    resume = resume and last_path.exists()
    logger = get_logger("train", out_dir / "train.log", append=resume)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # input shapes are fixed, so let cuDNN pick the fastest kernels
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
    model = build_model(cfg, bundle.datasets["train"].num_input_channels, len(bundle.target_channels), residual).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n_params:,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 0.0))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=tcfg.get("scheduler_factor", 0.5),
        patience=tcfg.get("scheduler_patience", 5), min_lr=tcfg.get("min_lr", 1e-6))
    use_amp = bool(tcfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    stopper = EarlyStopping(tcfg.get("patience", 10), tcfg.get("min_delta", 0.0), tcfg.get("min_delta_rel", 0.0))
    ckpt_path = out_dir / "best_model.pt"
    history_path = out_dir / "history.csv"

    start_epoch = 1
    if resume:
        state = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        if scaler is not None and state.get("scaler_state"):
            scaler.load_state_dict(state["scaler_state"])
        stopper.load_state_dict(state["stopper_state"])
        start_epoch = state["epoch"] + 1
        logger.info(f"Resumed from {last_path} after epoch {state['epoch']} (best val loss {stopper.best:.4e})")
    else:
        with open(history_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "lr", "seconds",
                                    "train_vs_persistence", "val_vs_persistence"])

    # The loss of "predict no change" on each split. Raw losses from different periods are not
    # comparable (the December validation period is easier than November, so val < train even for
    # persistence); as a fraction of the same split's persistence loss they are, and anything below
    # 1.0 beats persistence.
    persistence_loss = persistence_train = None
    if residual is not None:
        persistence_loss = run_epoch(None, loaders["val"], device, tcfg, mask,
                                     max_batches=tcfg.get("max_eval_batches"), persistence_channels=residual)
        persistence_train = run_epoch(None, loaders["train"], device, tcfg, mask,
                                      max_batches=tcfg.get("max_train_batches"), persistence_channels=residual)
        logger.info(f"Reference: persistence scores a train loss of {persistence_train:.4e} "
                    f"and a val loss of {persistence_loss:.4e}")

    def ratio(loss: float, reference: float | None) -> float | str:
        return round(loss / reference, 5) if reference else ""

    def vs_persistence(val_loss: float) -> str:
        return f" ({val_loss / persistence_loss:.3f}x persistence)" if persistence_loss else ""

    def vs_persistence(val_loss: float) -> str:
        return f" ({val_loss / persistence_loss:.3f}x persistence)" if persistence_loss else ""

    max_seconds = tcfg.get("max_hours") and tcfg["max_hours"] * 3600
    run_start = time.time()
    for epoch in range(start_epoch, tcfg["epochs"] + 1):
        if stopper.should_stop:  # a resumed run that had already stopped
            break
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]  # the rate this epoch trains with
        train_loss = run_epoch(model, loaders["train"], device, tcfg, mask, optimizer, scaler, use_amp,
                               tcfg.get("grad_clip"), tcfg.get("max_train_batches"))
        val_loss = run_epoch(model, loaders["val"], device, tcfg, mask, use_amp=use_amp,
                             max_batches=tcfg.get("max_eval_batches"))
        scheduler.step(val_loss)
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
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "stopper_state": stopper.state_dict(),
            "epoch": epoch,
        }, last_path)
        logger.info(f"Epoch {epoch:3d} | train {train_loss:.4e} | val {val_loss:.4e}{vs_persistence(val_loss)} | "
                    f"lr {lr:.2e} | {elapsed:.1f}s{'  * saved best' if improved else ''}")
        with open(history_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, lr, round(elapsed, 2),
                                    ratio(train_loss, persistence_train), ratio(val_loss, persistence_loss)])

        if stopper.should_stop:
            logger.info(f"Early stopping: no noticeable val improvement for {stopper.patience} epochs")
            break
        # Stop before the next epoch would exceed the time budget (e.g. a Kaggle session limit).
        if max_seconds and time.time() - run_start + elapsed > max_seconds:
            logger.info(f"Stopping: time budget of {tcfg['max_hours']} h reached (continue with --resume)")
            break

    logger.info(f"Best val loss {stopper.best:.4e}{vs_persistence(stopper.best)}; checkpoint: {ckpt_path}")
    return ckpt_path


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Train the ConvLSTM forecaster."))
    parser.add_argument("--resume", action="store_true",
                        help="Continue from <output_dir>/last_checkpoint.pt if it exists")
    args = parser.parse_args()
    train(load_config(args.config, args.overrides), args.resume)


if __name__ == "__main__":
    main()
