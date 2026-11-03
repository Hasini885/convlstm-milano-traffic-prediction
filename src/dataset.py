"""Preprocessing and PyTorch datasets for the Telecom Italia "Milano" SMS/Call/Internet data.

Pipeline
--------
1. ``preprocess``: stream the daily raw files, aggregate over country codes, and scatter every
   record into a ``(T, C, H, W)`` float32 tensor stored as ``.npy`` (memory-mapped, so the full
   ~1.8 GB, 5-channel tensor never has to fit in RAM at once).
2. ``build_datasets``: chronologically split the time axis, fit min-max statistics on the
   training segment only, and wrap each segment in a sliding-window ``MilanoTrafficDataset``.

Run ``python -m src.dataset --config configs/default.yaml`` to execute step 1.

Missing values
--------------
The raw data is *sparse*: a (cell, timestamp, country) row is only present when there was
activity, and individual fields are left empty when that activity type was zero. So a missing
value means "no recorded activity", not "unknown". The default ``fill_method: zero`` uses that
interpretation. ``fill_method: ffill_zero`` forward-fills each cell over time and zero-fills
whatever is still missing at the start. Cells with no activity at all in the training period can
also be excluded from the loss and the metrics (``data.mask_inactive``).
"""
from __future__ import annotations

import argparse
import gzip
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .utils import add_common_args, get_logger, load_config, load_json, resolve_path, save_json

FEATURES: tuple[str, ...] = ("smsin", "smsout", "callin", "callout", "internet")
LOCAL_TZ = "Europe/Rome"
TENSOR_FILE = "tensor.npy"
META_FILE = "meta.json"

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
# The headerless Harvard Dataverse files have this column order.
_HEADERLESS_COLUMNS = ["cell_id", "time", "countrycode", *FEATURES]
# Header aliases (lower-cased, non-alphanumerics stripped) -> canonical name, for CSV variants
# such as the Kaggle mirror ("datetime,CellID,countrycode,smsin,...").
_COLUMN_ALIASES = {
    "cellid": "cell_id", "squareid": "cell_id", "gridid": "cell_id",
    "datetime": "time", "timeinterval": "time", "time": "time",
    "countrycode": "countrycode",
    "smsin": "smsin", "smsout": "smsout", "callin": "callin", "callout": "callout",
    "internet": "internet",
}


# --------------------------------------------------------------------------------------------
# Raw file handling
# --------------------------------------------------------------------------------------------
def cell_id_to_rowcol(cell_ids: np.ndarray, grid_size: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Map 1-based Milano ``CellID`` values to ``(row, col)`` grid indices.

    IDs run west->east along a row, then row by row from south to north, so
    ``row = (id - 1) // grid_size`` (row 0 is the southern edge) and ``col = (id - 1) % grid_size``.
    Plot with ``origin="lower"`` to get north at the top.
    """
    idx = np.asarray(cell_ids, dtype=np.int64) - 1
    return idx // grid_size, idx % grid_size


def discover_raw_files(raw_dir: str | Path) -> list[tuple[pd.Timestamp, Path]]:
    """Return ``(date, path)`` for every daily ``sms-call-internet-mi-YYYY-MM-DD.{txt,csv}[.gz]`` file, sorted by date."""
    raw_dir = Path(raw_dir)
    found: list[tuple[pd.Timestamp, Path]] = []
    for path in raw_dir.iterdir():
        name = path.name.lower()
        if not name.endswith((".txt", ".csv", ".txt.gz", ".csv.gz")) or "sms-call-internet" not in name:
            continue
        match = _DATE_RE.search(path.name)
        if match:
            found.append((pd.Timestamp(match.group(1)), path))
    if not found:
        raise FileNotFoundError(f"No 'sms-call-internet-mi-YYYY-MM-DD' files found in {raw_dir}")
    return sorted(found, key=lambda item: item[0])


def _to_local_naive(time_col: pd.Series) -> pd.Series:
    """Parse raw timestamps (epoch milliseconds in UTC, or datetime strings) into naive Milan local time."""
    if pd.api.types.is_numeric_dtype(time_col):
        ts = pd.to_datetime(time_col, unit="ms", utc=True)
    else:
        ts = pd.to_datetime(time_col, utc=False)
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    return ts


def read_raw_file(path: str | Path, features: Sequence[str] = FEATURES) -> pd.DataFrame:
    """Read one daily file and aggregate it over country codes.

    Handles both the original headerless, tab-separated Dataverse ``.txt`` files and CSVs with a
    header row (for example the Kaggle mirror).

    Returns:
        DataFrame with columns ``time`` (naive local datetime), ``cell_id`` and one column per
        feature. Each (time, cell) pair appears once; a feature is NaN when it had no records.
    """
    path = Path(path)
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
    sep = "\t" if "\t" in first_line else ","
    has_header = any(ch.isalpha() for ch in first_line.split(sep)[0])

    if has_header:
        df = pd.read_csv(path, sep=sep)
        df.columns = [_COLUMN_ALIASES.get(re.sub(r"[^a-z0-9]", "", c.lower()), c) for c in df.columns]
    else:
        df = pd.read_csv(path, sep=sep, header=None, names=_HEADERLESS_COLUMNS)

    missing = {"cell_id", "time", *features} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}; found {list(df.columns)}")

    df = df[["time", "cell_id", *features]]
    df["time"] = _to_local_naive(df["time"])
    # min_count=1 keeps NaN when every country row for that field was empty.
    return df.groupby(["time", "cell_id"], as_index=False, sort=False)[list(features)].sum(min_count=1)


def _forward_fill_time(tensor: np.ndarray, rows_per_chunk: int = 10) -> None:
    """Forward-fill NaNs along the time axis in place, per (channel, cell), then zero-fill the rest."""
    T, C, H, W = tensor.shape
    t_idx = np.arange(T)[:, None]
    for c in range(C):
        for r0 in range(0, H, rows_per_chunk):
            r1 = min(r0 + rows_per_chunk, H)
            block = np.array(tensor[:, c, r0:r1, :]).reshape(T, -1)
            last_valid = np.where(np.isnan(block), 0, t_idx)
            np.maximum.accumulate(last_valid, axis=0, out=last_valid)
            block = block[last_valid, np.arange(block.shape[1])]
            tensor[:, c, r0:r1, :] = np.nan_to_num(block, nan=0.0).reshape(T, r1 - r0, W)


def preprocess(
    raw_dir: str | Path,
    processed_dir: str | Path,
    features: Sequence[str] = FEATURES,
    interval_minutes: int = 10,
    grid_size: int = 100,
    max_days: int | None = None,
    fill_method: str = "zero",
    logger: Any | None = None,
) -> Path:
    """Convert the daily raw files into one ``(T, C, H, W)`` float32 tensor plus a metadata file.

    Time index 0 is local midnight of the first file's date. Each record is placed at its absolute
    time index, so rows that spill into a neighbouring day (because of the UTC->local shift) still
    land in the right slot.

    Args:
        raw_dir: Directory containing the daily raw files.
        processed_dir: Output directory for ``tensor.npy`` and ``meta.json``.
        features: Channels to store, in this order.
        interval_minutes: Temporal resolution of the data (10 in the original release).
        grid_size: Grid side length (100 for Milano).
        max_days: Only use the first N days (for quick experiments).
        fill_method: ``"zero"`` or ``"ffill_zero"`` (see the module docstring).

    Returns:
        Path to the written tensor.
    """
    if fill_method not in {"zero", "ffill_zero"}:
        raise ValueError(f"Unknown fill_method {fill_method!r}")
    log = logger.info if logger else print
    features = list(features)
    files = discover_raw_files(raw_dir)
    if max_days:
        files = files[:max_days]

    origin = files[0][0]
    end = files[-1][0] + pd.Timedelta(days=1)
    step = pd.Timedelta(minutes=interval_minutes)
    T = int((end - origin) / step)
    C, H, W = len(features), grid_size, grid_size

    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / TENSOR_FILE
    log(f"Building tensor ({T}, {C}, {H}, {W}) from {len(files)} files "
        f"[{origin.date()} .. {files[-1][0].date()}] -> {out_path}")

    tensor = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(T, C, H, W))
    tensor[:] = 0.0 if fill_method == "zero" else np.nan

    n_out_of_range = n_bad_cell = n_off_grid = 0
    for i, (date, path) in enumerate(files):
        df = read_raw_file(path, features)
        if i == 0:
            diffs = np.diff(np.sort(df["time"].unique()))
            if len(diffs) and pd.Timedelta(diffs.min()) != step:
                log(f"WARNING: smallest timestamp gap in {path.name} is {pd.Timedelta(diffs.min())}, "
                    f"but interval_minutes={interval_minutes}. Fix data.interval_minutes.")

        offset = (df["time"] - origin) / step
        t = np.floor(offset.to_numpy()).astype(np.int64)
        n_off_grid += int((offset.to_numpy() != t).sum())
        cell = df["cell_id"].to_numpy(np.int64)
        valid_t = (t >= 0) & (t < T)
        valid_cell = (cell >= 1) & (cell <= H * W)
        n_out_of_range += int((~valid_t).sum())
        n_bad_cell += int((valid_t & ~valid_cell).sum())
        keep = valid_t & valid_cell
        t, (r, c) = t[keep], cell_id_to_rowcol(cell[keep], grid_size)

        for ch, feat in enumerate(features):
            vals = df[feat].to_numpy(np.float32)[keep]
            if fill_method == "zero":
                vals = np.nan_to_num(vals, nan=0.0)
            cur = tensor[t, ch, r, c]
            # (time, cell) pairs are unique within a file; accumulate in case a neighbouring file
            # contributed to the same slot.
            tensor[t, ch, r, c] = np.where(np.isnan(cur), vals, cur + np.nan_to_num(vals, nan=0.0))
        log(f"  [{i + 1}/{len(files)}] {path.name}: {keep.sum():,} (time, cell) records")

    if fill_method == "ffill_zero":
        log("Forward-filling missing values along time, then zero-filling the remainder ...")
        _forward_fill_time(tensor)
    tensor.flush()

    if n_out_of_range or n_bad_cell or n_off_grid:
        log(f"Note: dropped {n_out_of_range:,} rows outside the time range, {n_bad_cell:,} rows with "
            f"invalid CellID; {n_off_grid:,} rows were not aligned to the time grid (floored).")

    meta = {
        "features": features,
        "shape": [T, C, H, W],
        "origin": origin.isoformat(),
        "interval_minutes": interval_minutes,
        "grid_size": grid_size,
        "fill_method": fill_method,
        "days": [d.date().isoformat() for d, _ in files],
    }
    save_json(meta, processed_dir / META_FILE)
    log(f"Done. Metadata written to {processed_dir / META_FILE}")
    return out_path


# --------------------------------------------------------------------------------------------
# Normalisation and statistics
# --------------------------------------------------------------------------------------------
@dataclass
class MinMaxNormalizer:
    """Per-channel min-max scaling to [0, 1], with an optional ``log1p`` applied first.

    ``mins`` and ``maxs`` are indexed by channel in the *processed tensor*, so the same object
    serves any subset of input or target channels via the ``channels`` argument.
    """

    mins: np.ndarray
    maxs: np.ndarray
    log_transform: bool = False

    def _params(self, channels: Sequence[int], like: np.ndarray | torch.Tensor) -> tuple[Any, Any]:
        lo = self.mins[list(channels)].astype(np.float32)
        scale = (self.maxs[list(channels)] - lo).astype(np.float32)
        scale = np.where(scale > 0, scale, 1.0).astype(np.float32)
        lo, scale = lo.reshape(-1, 1, 1), scale.reshape(-1, 1, 1)
        if isinstance(like, torch.Tensor):
            return (torch.as_tensor(lo, dtype=like.dtype, device=like.device),
                    torch.as_tensor(scale, dtype=like.dtype, device=like.device))
        return lo, scale

    def normalize(self, x: np.ndarray | torch.Tensor, channels: Sequence[int]) -> Any:
        """Scale ``x`` of shape ``(..., len(channels), H, W)`` to roughly [0, 1]."""
        if self.log_transform:
            x = torch.log1p(x) if isinstance(x, torch.Tensor) else np.log1p(x)
        lo, scale = self._params(channels, x)
        return (x - lo) / scale

    def denormalize(self, x: np.ndarray | torch.Tensor, channels: Sequence[int]) -> Any:
        """Inverse of :meth:`normalize`, returning values in the original units."""
        lo, scale = self._params(channels, x)
        x = x * scale + lo
        if self.log_transform:
            x = torch.expm1(x) if isinstance(x, torch.Tensor) else np.expm1(x)
        return x

    def state_dict(self) -> dict[str, Any]:
        return {"mins": self.mins.tolist(), "maxs": self.maxs.tolist(), "log_transform": self.log_transform}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "MinMaxNormalizer":
        return cls(np.asarray(state["mins"], dtype=np.float64), np.asarray(state["maxs"], dtype=np.float64),
                   bool(state["log_transform"]))


def compute_train_statistics(
    tensor: np.ndarray, t_end: int, target_channels: Sequence[int], log_transform: bool = False, chunk: int = 256
) -> tuple[MinMaxNormalizer, np.ndarray]:
    """Fit the normalizer and the mean activity map on ``tensor[:t_end]`` (the training segment only).

    Returns:
        ``(normalizer, activity)``, where ``activity`` is the ``(H, W)`` mean of the target
        channels over the training period. Cells where it is 0 had no activity at all.
    """
    C, H, W = tensor.shape[1:]
    mins = np.full(C, np.inf)
    maxs = np.full(C, -np.inf)
    activity = np.zeros((H, W), dtype=np.float64)
    for start in range(0, t_end, chunk):
        block = np.asarray(tensor[start:min(start + chunk, t_end)], dtype=np.float64)
        activity += block[:, list(target_channels)].sum(axis=(0, 1))
        if log_transform:
            block = np.log1p(block)
        mins = np.minimum(mins, block.min(axis=(0, 2, 3)))
        maxs = np.maximum(maxs, block.max(axis=(0, 2, 3)))
    activity /= t_end * len(target_channels)
    return MinMaxNormalizer(mins, maxs, log_transform), activity.astype(np.float32)


def split_bounds(num_steps: int, ratios: Sequence[float]) -> list[tuple[int, int]]:
    """Chronological ``[(start, end), ...]`` time ranges for train/val/test."""
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"split must be three ratios summing to 1, got {ratios}")
    n_train = int(num_steps * ratios[0])
    n_val = int(num_steps * ratios[1])
    return [(0, n_train), (n_train, n_train + n_val), (n_train + n_val, num_steps)]


# --------------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------------
class MilanoTrafficDataset(Dataset):
    """Sliding windows over one chronological segment of the processed tensor.

    Every window lies entirely inside ``[t_start, t_end)``, so no target from a later split ever
    appears in an earlier one. Item ``i`` is::

        X: (t_in,  C_in,  H, W)   frames t0 .. t0+t_in-1         (normalised)
        Y: (t_out, C_out, H, W)   frames t0+t_in .. t0+t_in+t_out-1 (normalised)

    with ``t0 = t_start + i * stride``. The tensor is opened lazily as a memmap inside each worker,
    so DataLoader workers do not pickle the whole array.
    """

    def __init__(
        self,
        tensor_path: str | Path,
        t_start: int,
        t_end: int,
        t_in: int,
        t_out: int,
        input_channels: Sequence[int],
        target_channels: Sequence[int],
        normalizer: MinMaxNormalizer,
        stride: int = 1,
        crop: Sequence[int] | None = None,
    ) -> None:
        self.tensor_path = str(tensor_path)
        self.t_start, self.t_end = t_start, t_end
        self.t_in, self.t_out, self.stride = t_in, t_out, stride
        self.input_channels = list(input_channels)
        self.target_channels = list(target_channels)
        self.normalizer = normalizer
        self.crop = tuple(crop) if crop else None
        self.num_windows = (t_end - t_start - t_in - t_out) // stride + 1
        if self.num_windows <= 0:
            raise ValueError(f"Segment [{t_start}, {t_end}) is too short for t_in={t_in}, t_out={t_out}")
        self._array: np.ndarray | None = None

    def __len__(self) -> int:
        return self.num_windows

    def _data(self) -> np.ndarray:
        if self._array is None:
            arr = np.load(self.tensor_path, mmap_mode="r")
            if self.crop:
                r0, r1, c0, c1 = self.crop
                arr = arr[:, :, r0:r1, c0:c1]
            self._array = arr
        return self._array

    def target_time_index(self, i: int) -> int:
        """Absolute time index of the first target frame of window ``i``."""
        return self.t_start + i * self.stride + self.t_in

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        arr = self._data()
        t0 = self.t_start + i * self.stride
        x = np.asarray(arr[t0:t0 + self.t_in][:, self.input_channels], dtype=np.float32)
        y = np.asarray(arr[t0 + self.t_in:t0 + self.t_in + self.t_out][:, self.target_channels], dtype=np.float32)
        x = self.normalizer.normalize(x, self.input_channels).astype(np.float32)
        y = self.normalizer.normalize(y, self.target_channels).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


@dataclass
class DataBundle:
    """Everything produced by :func:`build_datasets`."""

    datasets: dict[str, MilanoTrafficDataset]
    normalizer: MinMaxNormalizer
    mask: torch.Tensor | None  # (H, W) bool over the (cropped) grid, or None
    activity: np.ndarray  # (H, W) mean training activity of the target channels (cropped)
    meta: dict[str, Any]
    input_channels: list[int]
    target_channels: list[int]


def _channel_indices(names: Sequence[str], available: Sequence[str]) -> list[int]:
    unknown = [n for n in names if n not in available]
    if unknown:
        raise ValueError(f"Features {unknown} not in the processed tensor {list(available)}; re-run preprocessing")
    return [list(available).index(n) for n in names]


def build_datasets(cfg: dict[str, Any], normalizer: MinMaxNormalizer | None = None) -> DataBundle:
    """Create train/val/test datasets from the processed tensor described by ``cfg``.

    Args:
        cfg: Full config dict.
        normalizer: Fitted normalizer to reuse (e.g. from a checkpoint). If ``None``, it is fitted
            on the training segment.
    """
    dcfg = cfg["data"]
    processed_dir = resolve_path(cfg["paths"]["processed_dir"])
    tensor_path = processed_dir / TENSOR_FILE
    if not tensor_path.exists():
        raise FileNotFoundError(f"{tensor_path} not found. Run `python -m src.dataset` first.")
    meta = load_json(processed_dir / META_FILE)
    input_channels = _channel_indices(dcfg["input_features"], meta["features"])
    target_channels = _channel_indices(dcfg["target_features"], meta["features"])

    tensor = np.load(tensor_path, mmap_mode="r")
    bounds = split_bounds(tensor.shape[0], dcfg["split"])
    fitted, activity = compute_train_statistics(
        tensor, bounds[0][1], target_channels, bool(dcfg.get("log_transform", False)))
    normalizer = normalizer or fitted

    crop = dcfg.get("crop")
    if crop:
        r0, r1, c0, c1 = crop
        activity = activity[r0:r1, c0:c1]
    mask = torch.from_numpy(activity > 0) if dcfg.get("mask_inactive", True) else None

    datasets = {
        name: MilanoTrafficDataset(
            tensor_path, start, end, dcfg["t_in"], dcfg["t_out"], input_channels, target_channels,
            normalizer, stride=dcfg.get("stride", 1) if name == "train" else 1, crop=crop)
        for name, (start, end) in zip(("train", "val", "test"), bounds)
    }
    return DataBundle(datasets, normalizer, mask, activity, meta, input_channels, target_channels)


def build_dataloaders(cfg: dict[str, Any], datasets: dict[str, MilanoTrafficDataset]) -> dict[str, DataLoader]:
    """Wrap datasets in DataLoaders. Only the training loader is shuffled."""
    workers = int(cfg["data"].get("num_workers", 0))
    return {
        name: DataLoader(
            ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=(name == "train"),
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=workers > 0,
            drop_last=False,
        )
        for name, ds in datasets.items()
    }


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Preprocess raw Milano files into a tensor."))
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    dcfg = cfg["data"]
    preprocess(
        raw_dir=resolve_path(cfg["paths"]["raw_dir"]),
        processed_dir=resolve_path(cfg["paths"]["processed_dir"]),
        features=dcfg["features"],
        interval_minutes=dcfg["interval_minutes"],
        grid_size=dcfg["grid_size"],
        max_days=dcfg.get("max_days"),
        fill_method=dcfg.get("fill_method", "zero"),
        logger=get_logger("preprocess"),
    )


if __name__ == "__main__":
    main()
