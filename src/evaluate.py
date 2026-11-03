"""Evaluate a trained checkpoint on the test split against naive baselines, and make plots.

Metrics (RMSE, MAE, MAPE) are computed in the original units after denormalisation, per
prediction horizon and per target channel. Only cells that were active during training count
when ``data.mask_inactive`` is on. MAPE only counts cells whose true value exceeds
``evaluation.mape_threshold``, because it is undefined or explosive near zero.

Baselines (need every target feature to also be an input feature):
  * ``persistence``: repeat the last observed frame for every horizon.
  * ``window_mean``: the mean of the ``t_in`` input frames (historical average).

Usage::

    python -m src.evaluate --config configs/default.yaml [--checkpoint path/to/best_model.pt]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from .dataset import MinMaxNormalizer, build_dataloaders, build_datasets  # noqa: E402
from .model import build_model  # noqa: E402
from .utils import add_common_args, get_device, get_logger, load_config, resolve_path, save_json, set_seed  # noqa: E402

# Plot styling: fixed categorical order, single-hue sequential ramps, recessive axes.
SURFACE = "#fcfcfb"
TEXT, TEXT_MUTED = "#0b0b0b", "#52514e"
SERIES = {"Actual": "#2a78d6", "ConvLSTM": "#eb6834", "Persistence": "#1baf7a"}
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
SEQ_ORANGE = LinearSegmentedColormap.from_list("seq_orange", ["#fcfcfb", "#f9d3c0", "#f19a72", "#eb6834", "#b9471c", "#7a2c0f"])


class MetricAccumulator:
    """Streaming RMSE / MAE / MAPE per (horizon, channel), so the test set never has to sit in memory."""

    def __init__(self, t_out: int, n_channels: int, mape_threshold: float) -> None:
        shape = (t_out, n_channels)
        self.sse, self.sae, self.n = np.zeros(shape), np.zeros(shape), np.zeros(shape)
        self.sape, self.n_ape = np.zeros(shape), np.zeros(shape)
        self.mape_threshold = mape_threshold

    def update(self, pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor | None) -> None:
        """Add a batch. ``pred``/``true``: ``(B, T_out, C, H, W)`` in original units; ``mask``: ``(H, W)`` bool."""
        pred, true = pred.double().flatten(-2), true.double().flatten(-2)
        if mask is not None:
            keep = mask.flatten()
            pred, true = pred[..., keep], true[..., keep]
        err = pred - true
        dims = (0, 3)
        self.sse += (err ** 2).sum(dims).cpu().numpy()
        self.sae += err.abs().sum(dims).cpu().numpy()
        self.n += err.shape[0] * err.shape[3]
        ape_ok = true > self.mape_threshold
        self.sape += torch.where(ape_ok, err.abs() / true.clamp_min(1e-12), torch.zeros_like(err)).sum(dims).cpu().numpy()
        self.n_ape += ape_ok.sum(dims).cpu().numpy()

    @staticmethod
    def _summarise(sse: np.ndarray, sae: np.ndarray, n: np.ndarray, sape: np.ndarray, n_ape: np.ndarray) -> dict[str, float]:
        return {
            "RMSE": float(np.sqrt(sse.sum() / n.sum())),
            "MAE": float(sae.sum() / n.sum()),
            "MAPE_%": float(100 * sape.sum() / n_ape.sum()) if n_ape.sum() else float("nan"),
        }

    def results(self, channel_names: Sequence[str]) -> dict[str, Any]:
        """Overall, per-horizon and per-channel metrics."""
        arrays = (self.sse, self.sae, self.n, self.sape, self.n_ape)
        return {
            "overall": self._summarise(*arrays),
            "per_horizon": [
                {"horizon": h + 1, **self._summarise(*(a[h] for a in arrays))} for h in range(self.sse.shape[0])],
            "per_channel": {
                name: self._summarise(*(a[:, c] for a in arrays)) for c, name in enumerate(channel_names)},
        }


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c3")
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.grid(True, color="#ecebe7", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_heatmaps(samples: list[dict[str, Any]], channel: str, units_label: str, out_path: Path) -> None:
    """One row per sample: actual, ConvLSTM prediction and absolute error (first horizon)."""
    fig, axes = plt.subplots(len(samples), 3, figsize=(13, 4.1 * len(samples)), squeeze=False, facecolor=SURFACE)
    for row, s in enumerate(samples):
        vmax = float(np.percentile(s["actual"], 99.5)) or 1.0
        err = np.abs(s["pred"] - s["actual"])
        panels = [
            (s["actual"], f"Actual · {s['time']}", SEQ_BLUE, vmax),
            (s["pred"], "ConvLSTM prediction", SEQ_BLUE, vmax),
            (err, "Absolute error", SEQ_ORANGE, float(np.percentile(err, 99.5)) or 1.0),
        ]
        for ax, (img, title, cmap, vm) in zip(axes[row], panels):
            im = ax.imshow(img, origin="lower", cmap=cmap, vmin=0, vmax=vm, interpolation="nearest")
            ax.set_title(title, color=TEXT, fontsize=11, loc="left")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.outline.set_visible(False)
            cb.ax.tick_params(colors=TEXT_MUTED, labelsize=8)
            cb.set_label(units_label, color=TEXT_MUTED, fontsize=8)
    fig.suptitle(f"{channel}: predicted vs actual (north up, color capped at the 99.5th percentile)",
                 color=TEXT, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def plot_cell_series(times: Sequence[pd.Timestamp], series: dict[str, np.ndarray], title: str, units_label: str, out_path: Path) -> None:
    """Time series of actual vs predicted vs persistence for one cell (first horizon)."""
    fig, ax = plt.subplots(figsize=(13, 4.2), facecolor=SURFACE)
    _style_axes(ax)
    for name, values in series.items():
        ax.plot(times, values, color=SERIES[name], linewidth=2 if name != "Persistence" else 1.4,
                label=name, alpha=1.0 if name != "Persistence" else 0.85)
    ax.set_title(title, color=TEXT, fontsize=12, loc="left")
    ax.set_ylabel(units_label, color=TEXT_MUTED)
    ax.set_ylim(bottom=0)
    leg = ax.legend(frameon=False, loc="upper left", ncol=len(series))
    for text in leg.get_texts():
        text.set_color(TEXT)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def plot_history(history_csv: Path, out_path: Path) -> None:
    """Training and validation loss per epoch."""
    hist = pd.read_csv(history_csv)
    if hist.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    _style_axes(ax)
    ax.plot(hist["epoch"], hist["train_loss"], color=SERIES["Actual"], linewidth=2, label="Train")
    ax.plot(hist["epoch"], hist["val_loss"], color=SERIES["ConvLSTM"], linewidth=2, label="Validation")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Epoch", color=TEXT_MUTED)
    ax.set_title("Loss (normalised scale)", color=TEXT, fontsize=12, loc="left")
    leg = ax.legend(frameon=False)
    for text in leg.get_texts():
        text.set_color(TEXT)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def _pick_cell(activity: np.ndarray, mask: torch.Tensor | None, rng: np.random.Generator, fixed: Sequence[int] | None) -> tuple[int, int]:
    """Use the configured cell if given, else a random cell from the 10% most active ones."""
    if fixed:
        return int(fixed[0]), int(fixed[1])
    valid = activity if mask is None else np.where(mask.numpy(), activity, -np.inf)
    threshold = np.percentile(valid[np.isfinite(valid)], 90)
    candidates = np.argwhere(valid >= threshold)
    r, c = candidates[rng.integers(len(candidates))]
    return int(r), int(c)


@torch.no_grad()
def evaluate(cfg: dict[str, Any], checkpoint: str | Path | None = None) -> dict[str, Any]:
    """Evaluate ``checkpoint`` (default ``<output_dir>/best_model.pt``) on the test split."""
    out_dir = resolve_path(cfg["paths"]["output_dir"])
    ckpt_path = resolve_path(checkpoint) if checkpoint else out_dir / "best_model.pt"
    logger = get_logger("evaluate", out_dir / "evaluate.log")
    device = get_device(cfg.get("device", "auto"))
    ecfg = cfg["evaluation"]
    set_seed(cfg.get("seed", 42))

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    # Use the data/model settings the model was trained with; paths and evaluation settings come
    # from the current config.
    run_cfg = {**ckpt["config"], "paths": cfg["paths"], "evaluation": ecfg}
    run_cfg["training"] = {**ckpt["config"]["training"], "batch_size": cfg["training"]["batch_size"]}
    normalizer = MinMaxNormalizer.from_state_dict(ckpt["normalizer"])

    bundle = build_datasets(run_cfg, normalizer=normalizer)
    test_ds = bundle.datasets["test"]
    loader = build_dataloaders(run_cfg, {"test": test_ds})["test"]
    in_ch, tgt_ch = bundle.input_channels, bundle.target_channels
    target_names = [bundle.meta["features"][i] for i in tgt_ch]
    residual = ckpt.get("residual_channels")

    model = build_model(run_cfg, len(in_ch), len(tgt_ch), residual).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.info(f"Loaded {ckpt_path} (epoch {ckpt['epoch']}, val loss {ckpt['val_loss']:.6f}); "
                f"{len(test_ds)} test windows on {device}")

    t_out = run_cfg["data"]["t_out"]
    baseline_pos = [in_ch.index(t) for t in tgt_ch] if all(t in in_ch for t in tgt_ch) else None
    mask = bundle.mask
    mask_dev = mask.to(device) if mask is not None else None
    accs = {"ConvLSTM": MetricAccumulator(t_out, len(tgt_ch), ecfg.get("mape_threshold", 1.0))}
    if baseline_pos is not None:
        accs["Persistence"] = MetricAccumulator(t_out, len(tgt_ch), ecfg.get("mape_threshold", 1.0))
        accs["WindowMean"] = MetricAccumulator(t_out, len(tgt_ch), ecfg.get("mape_threshold", 1.0))
    else:
        logger.info("Baselines skipped: not every target feature is an input feature")

    rng = np.random.default_rng(cfg.get("seed", 42))
    cell = _pick_cell(bundle.activity, mask, rng, ecfg.get("plot_cell"))
    sample_ids = set(rng.choice(len(test_ds), size=min(ecfg.get("num_plot_samples", 3), len(test_ds)), replace=False).tolist())
    samples: list[dict[str, Any]] = []
    cell_series: dict[str, list[float]] = {"Actual": [], "ConvLSTM": [], "Persistence": []}
    origin = pd.Timestamp(bundle.meta["origin"])
    step = pd.Timedelta(minutes=bundle.meta["interval_minutes"])

    offset = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        pred_raw = normalizer.denormalize(pred.float(), tgt_ch).clamp_min(0)  # traffic is non-negative
        y_raw = normalizer.denormalize(y, tgt_ch)
        accs["ConvLSTM"].update(pred_raw, y_raw, mask_dev)

        persist = None
        if baseline_pos is not None:
            x_raw = normalizer.denormalize(x, in_ch)[:, :, baseline_pos]
            persist = x_raw[:, -1:].expand(-1, t_out, -1, -1, -1)
            accs["Persistence"].update(persist, y_raw, mask_dev)
            accs["WindowMean"].update(x_raw.mean(dim=1, keepdim=True).expand(-1, t_out, -1, -1, -1), y_raw, mask_dev)

        r, c = cell
        cell_series["Actual"] += y_raw[:, 0, 0, r, c].cpu().tolist()
        cell_series["ConvLSTM"] += pred_raw[:, 0, 0, r, c].cpu().tolist()
        if persist is not None:
            cell_series["Persistence"] += persist[:, 0, 0, r, c].cpu().tolist()

        for b in range(x.shape[0]):
            if offset + b in sample_ids:
                samples.append({
                    "actual": y_raw[b, 0, 0].cpu().numpy(),
                    "pred": pred_raw[b, 0, 0].cpu().numpy(),
                    "time": (origin + step * test_ds.target_time_index(offset + b)).strftime("%Y-%m-%d %H:%M"),
                })
        offset += x.shape[0]

    results = {name: acc.results(target_names) for name, acc in accs.items()}
    results["notes"] = {
        "units": "original dataset units (unitless activity counts)",
        "mask_inactive": mask is not None,
        "active_cells": int(mask.sum()) if mask is not None else None,
        "mape_threshold": ecfg.get("mape_threshold", 1.0),
        "test_windows": len(test_ds),
        "checkpoint": str(ckpt_path),
    }
    save_json(results, out_dir / "metrics.json")

    rows = []
    for name, res in results.items():
        if name == "notes":
            continue
        for h in res["per_horizon"]:
            rows.append({"model": name, **h})
    with open(out_dir / "metrics_per_horizon.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "horizon", "RMSE", "MAE", "MAPE_%"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"{'model':<12} {'horizon':>7} {'RMSE':>12} {'MAE':>12} {'MAPE %':>9}")
    for row in rows:
        logger.info(f"{row['model']:<12} {row['horizon']:>7} {row['RMSE']:>12.4f} {row['MAE']:>12.4f} {row['MAPE_%']:>9.2f}")
    for name, res in results.items():
        if name != "notes":
            o = res["overall"]
            logger.info(f"{name:<12} overall  RMSE {o['RMSE']:.4f} | MAE {o['MAE']:.4f} | MAPE {o['MAPE_%']:.2f}%")

    channel = target_names[0]
    units = f"{channel} activity"
    plot_heatmaps(sorted(samples, key=lambda s: s["time"]), channel, units, out_dir / "heatmaps.png")
    times = [origin + step * test_ds.target_time_index(i) for i in range(len(test_ds))]
    series = {k: np.asarray(v) for k, v in cell_series.items() if v}
    crop = run_cfg["data"].get("crop")
    r_abs, c_abs = (cell[0] + crop[0], cell[1] + crop[2]) if crop else cell
    cell_id = r_abs * bundle.meta["grid_size"] + c_abs + 1
    plot_cell_series(times, series, f"{channel} at CellID {cell_id} (row {r_abs}, col {c_abs}), 1-step-ahead test predictions",
                     units, out_dir / "cell_timeseries.png")
    if (out_dir / "history.csv").exists():
        plot_history(out_dir / "history.csv", out_dir / "loss_curve.png")
    logger.info(f"Metrics and plots written to {out_dir}")
    return results


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Evaluate a trained ConvLSTM checkpoint."))
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path (default: <output_dir>/best_model.pt)")
    args = parser.parse_args()
    evaluate(load_config(args.config, args.overrides), args.checkpoint)


if __name__ == "__main__":
    main()
