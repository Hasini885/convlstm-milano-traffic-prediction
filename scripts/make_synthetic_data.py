"""Generate synthetic daily files in the exact Harvard Dataverse format, for pipeline testing.

Each file is tab-separated with no header and the columns::

    square_id  time_interval(ms since epoch, UTC)  country_code  smsin  smsout  callin  callout  internet

Like the real release, it has several country-code rows per (cell, time), empty fields for zero
activity, and some (cell, time) pairs with no row at all. The signal is a few spatial hotspots
scaled by a daily cycle plus noise, so a model has something learnable.

Usage::

    python scripts/make_synthetic_data.py --out data/synthetic/raw --days 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_SCALE = {"smsin": 3.0, "smsout": 2.0, "callin": 2.5, "callout": 2.5, "internet": 60.0}


def make_day(date: pd.Timestamp, grid: int, rng: np.random.Generator, hotspots: np.ndarray) -> pd.DataFrame:
    """Build one day of synthetic records."""
    steps = 144
    yy, xx = np.mgrid[0:grid, 0:grid]
    spatial = np.full((grid, grid), 0.05)
    for cy, cx, sigma, amp in hotspots:
        spatial += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))

    local_times = pd.date_range(date, periods=steps, freq="10min", tz="Europe/Rome")
    hours = (local_times.hour + local_times.minute / 60).to_numpy()
    daily = 0.25 + 0.75 * np.clip(np.sin((hours - 6) / 24 * 2 * np.pi), 0, None) ** 1.5  # (steps,)
    epoch_ms = (local_times.tz_convert("UTC").as_unit("ms").asi8)

    base = daily[:, None, None] * spatial[None]  # (steps, grid, grid)
    t_idx, r_idx, c_idx = np.meshgrid(np.arange(steps), np.arange(grid), np.arange(grid), indexing="ij")
    cell_ids = (r_idx * grid + c_idx + 1).ravel()
    frame = {"square_id": cell_ids, "time": epoch_ms[t_idx.ravel()], "country": 39}
    for feat, scale in FEATURE_SCALE.items():
        frame[feat] = (base.ravel() * scale * rng.lognormal(0, 0.25, base.size)).astype(np.float32)
    df = pd.DataFrame(frame)

    # Extra rows for foreign country codes (aggregated away during preprocessing).
    extra = df.sample(frac=0.08, random_state=int(rng.integers(1 << 31))).copy()
    extra["country"] = rng.choice([33, 44, 49, 86], size=len(extra))
    for feat in FEATURE_SCALE:
        extra[feat] *= 0.1
    df = pd.concat([df, extra], ignore_index=True)

    # Sparsity: tiny values become empty fields, and some rows disappear entirely.
    for feat in FEATURE_SCALE:
        df.loc[df[feat] < 0.05, feat] = np.nan
    df = df[rng.random(len(df)) > 0.03]
    return df.sort_values(["time", "square_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="data/synthetic/raw", help="Output directory")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--start", default="2013-11-01")
    parser.add_argument("--grid", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    n_hot = 12
    hotspots = np.column_stack([
        rng.uniform(20, args.grid - 20, n_hot), rng.uniform(20, args.grid - 20, n_hot),
        rng.uniform(2, 8, n_hot), rng.uniform(1, 6, n_hot)])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for d in range(args.days):
        date = pd.Timestamp(args.start) + pd.Timedelta(days=d)
        path = out / f"sms-call-internet-mi-{date.date()}.txt"
        make_day(date, args.grid, rng, hotspots).to_csv(path, sep="\t", header=False, index=False, float_format="%.4f")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
