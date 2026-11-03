# ConvLSTM Cellular Traffic Forecasting: Telecom Italia "Milano"

A complete, reproducible PyTorch project that predicts future mobile traffic (SMS, calls,
internet) on the 100×100 Milano grid with a spatio-temporal **ConvLSTM**. It covers raw-data
preprocessing, the model, training with early stopping, evaluation against naive baselines, and
visualisation.

---

## Table of contents

1. [Project status](#1-project-status)
2. [Project structure](#2-project-structure)
3. [Quick start](#3-quick-start)
4. [Environment and installation](#4-environment-and-installation)
5. [The dataset](#5-the-dataset)
6. [Preprocessing pipeline](#6-preprocessing-pipeline)
7. [Dataset, splits and normalisation](#7-dataset-splits-and-normalisation)
8. [Model architecture](#8-model-architecture)
9. [Training](#9-training)
10. [Evaluation, metrics and baselines](#10-evaluation-metrics-and-baselines)
11. [Configuration reference](#11-configuration-reference)
12. [Command-line usage](#12-command-line-usage)
13. [Results](#13-results)
14. [Design decisions](#14-design-decisions)
15. [Code reference](#15-code-reference)
16. [Output files](#16-output-files)
17. [Troubleshooting](#17-troubleshooting)
18. [Known limitations and next steps](#18-known-limitations-and-next-steps)

---

## 1. Project status

| Item | Status |
|---|---|
| Preprocessing (`src/dataset.py`) | Done; checked against a direct pandas aggregation |
| Model (`src/model.py`) | Done; forward/backward checked for 1 and 5 inputs, multi-channel output, `t_out` 1 and 3 |
| Training (`src/train.py`) | Done; runs end to end on CPU |
| Evaluation and plots (`src/evaluate.py`) | Done; metrics per horizon and channel, baselines, 3 plots |
| Synthetic data generator | Done; writes files in the exact Dataverse raw format |
| **Real Milano data** | **Done.** Full run on Kaggle (Tesla T4): 62 days, full 100×100 grid. Results in §13 and `outputs/kaggle/`. |

---

## 2. Project structure

```
convlstm_milano/
├── README.md                     # this file
├── requirements.txt              # pinned dependencies
├── configs/
│   ├── default.yaml              # full-data hyperparameters (every option, with comments)
│   └── smoke.yaml                # small, fast CPU test; inherits default.yaml via `_base_`
├── src/
│   ├── __init__.py
│   ├── utils.py                  # config loading + CLI overrides, seeding, logging, device, JSON I/O
│   ├── dataset.py                # raw parsing, tensor building, normalisation, Dataset, DataLoaders
│   ├── model.py                  # ConvLSTMCell, ConvLSTM, PredictionHead, ConvLSTMForecaster
│   ├── train.py                  # masked loss, early stopping, LR scheduling, checkpointing
│   └── evaluate.py               # streaming metrics, baselines, heatmaps / time-series / loss plots
├── scripts/
│   └── make_synthetic_data.py    # fake daily files in the raw Dataverse format
├── data/
│   ├── raw/                      # put the real daily files here
│   ├── processed/                # tensor.npy + meta.json for the real data
│   └── synthetic/                # created by the smoke test
│       ├── raw/                  # 3 synthetic daily .txt files (~232 MB)
│       └── processed/            # synthetic tensor (432, 5, 100, 100) (~83 MB)
└── outputs/
    ├── smoke/                    # checkpoint, logs, metrics and plots from the smoke run
    └── smoke_preview/            # re-rendered plots (integer epoch ticks on the loss curve)
```

Code size: about 1,530 lines (`dataset.py` 488, `evaluate.py` 314, `model.py` 243,
`train.py` 194, `utils.py` 122, `make_synthetic_data.py` 85, configs 82).
`desktop.ini` and `__pycache__/` are Windows and Python artefacts and can be ignored.

---

## 3. Quick start

From the project root:

```bash
pip install -r requirements.txt

# A. No download needed: run the full pipeline on synthetic data (under a minute on CPU)
python scripts/make_synthetic_data.py --out data/synthetic/raw --days 3
python -m src.dataset  --config configs/smoke.yaml
python -m src.train    --config configs/smoke.yaml
python -m src.evaluate --config configs/smoke.yaml

# B. Real data: put the daily files in data/raw/, then
python -m src.dataset  --config configs/default.yaml
python -m src.train    --config configs/default.yaml
python -m src.evaluate --config configs/default.yaml
```

---

### Running on Kaggle (GPU, real data)

The full real-data run is set up as a Kaggle notebook: `kaggle/convlstm_milano_kaggle.ipynb`
(configuration: `configs/kaggle.yaml`).

- **Data:** the Kaggle dataset [`dkgmgo/telecom-italia-milan`](https://www.kaggle.com/datasets/dkgmgo/telecom-italia-milan).
  All 62 daily files match the official Harvard Dataverse release byte for byte (checked
  against the Dataverse file list). This avoids the Dataverse guestbook form that blocks
  automated downloads.
- **Code:** uploaded as the private dataset `hasinisaimaram/convlstm-milano-code` (a zip of this
  folder).
- **Notebook:** `hasinisaimaram/convlstm-milano-traffic-prediction` (private, GPU). It finds the
  code and the raw files under `/kaggle/input` and detects the time step. It then preprocesses,
  trains (budget `max_hours: 8.5`, within Kaggle's 12-hour limit), evaluates, and displays the
  metrics and plots.

To re-run after changing the code: re-zip the project and run
`kaggle datasets version -p <folder> -m "update"`, then `kaggle kernels push -p <kernel folder>`.
On Windows, call the Kaggle CLI with **relative** paths; it garbles absolute ones.

## 4. Environment and installation

`requirements.txt`:

| Package | Version | Used for |
|---|---|---|
| torch | 2.12.1 | model, training, metrics |
| numpy | 1.26.4 | tensors, memmaps |
| pandas | 2.2.0 | reading raw files, timestamps |
| scikit-learn | 1.4.0 | pinned as requested; **not used** (metrics are computed incrementally in PyTorch) |
| matplotlib | 3.9.2 | plots |
| pyyaml | 6.0.3 | configs |

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

For a GPU, install a CUDA build of PyTorch from <https://pytorch.org/get-started/locally/>. The
code picks CUDA automatically when it is available (`device: auto`).

**Verified environment:** Windows 11, Python 3.11.9, CPU-only PyTorch 2.12.1, no GPU.

---

## 5. The dataset

**Telecom Italia Big Data Challenge, "Telecommunications - SMS, Call, Internet - MI"**,
Harvard Dataverse, doi:10.7910/DVN/EGZHFV.

- **Space:** the city of Milan, split into a 100×100 grid of 10,000 square cells (`CellID` 1–10000).
- **Time:** 10-minute intervals, 2013-11-01 to 2014-01-01: 62 daily files, 144 steps per day,
  **T = 8,928** steps.
- **Features (channels):** `smsin`, `smsout`, `callin`, `callout`, `internet`. These are
  anonymised activity levels, not raw counts.
- **Size:** about 300 MB per unzipped daily file. The processed tensor with all five channels is
  8928 × 5 × 100 × 100 float32 ≈ **1.8 GB** (about 360 MB for one channel).

### How to download

1. Open the Dataverse page above (search "Telecommunications - SMS, Call, Internet - MI").
2. Download the daily files `sms-call-internet-mi-2013-11-01.txt` …
   `sms-call-internet-mi-2014-01-01.txt` and unzip them into `data/raw/`.
3. Optional: `milano-grid.geojson` from the same collection, to check the grid orientation.

### Supported raw formats (detected automatically)

| Source | Format |
|---|---|
| Harvard Dataverse (original) | tab-separated `.txt`, **no header**: `square_id, time_interval (epoch ms, UTC), country_code, sms_in, sms_out, call_in, call_out, internet` |
| CSV mirrors (e.g. Kaggle) | `.csv` with a header such as `datetime,CellID,countrycode,smsin,smsout,callin,callout,internet` |

Header names are matched case-insensitively, ignoring punctuation (`CellID`, `square_id` and
`gridid` all map to the cell ID). Files must have `sms-call-internet` and a `YYYY-MM-DD` date in
their name.

### Raw-data quirks the pipeline handles

- **One row per (cell, time, country code).** Rows are summed over country codes.
- **Sparse records.** A row only exists when there was activity, and individual fields are empty
  when that activity type was zero.
- **UTC timestamps.** Converted to Milan local time (Europe/Rome). No DST change falls inside
  Nov–Dec 2013.

---

## 6. Preprocessing pipeline

```bash
python -m src.dataset --config configs/default.yaml
```

Implemented by `preprocess()` in `src/dataset.py`:

1. **Discover files** in `paths.raw_dir`, sorted by the date in the file name.
   `data.max_days` keeps only the first N days.
2. **Read each day** (streamed, one file at a time) with `read_raw_file()`: detect the
   separator and header, convert timestamps to naive Milan local time, keep the requested
   features, and **sum over country codes** (`min_count=1`, so all-empty fields stay NaN).
3. **Map cells to the grid** with `cell_id_to_rowcol()`:
   `row = (CellID − 1) // 100`, `col = (CellID − 1) % 100`. IDs run west→east, then
   south→north, so row 0 is the southern edge and plots use `origin="lower"`.
4. **Place each record at its absolute time index**: `t = floor((time − origin) / 10 min)`,
   where `origin` is local midnight of the first file's date. Records that spill into a
   neighbouring day (because of the UTC→local shift) still land in the right slot.
5. **Write** into a float32 `.npy` memmap of shape **`(T, C, 100, 100)`**, so the full tensor
   never has to fit in RAM at once.
6. **Fill missing values** (`data.fill_method`):
   - `zero` (default): missing means no activity, so it becomes 0.
   - `ffill_zero`: forward-fill each cell over time, then zero-fill the remaining leading gaps.
7. **Sanity checks and logging.** A warning is logged if the smallest timestamp gap doesn't
   match `interval_minutes` (some mirrors are hourly). Counts of dropped rows (outside the time
   range, invalid CellID) and floored, misaligned timestamps are reported.
8. **Metadata.** `meta.json` records the features, shape, origin, interval, grid size, fill
   method and the list of days.

All five features are stored, so `input_features` and `target_features` can be changed later
without re-running preprocessing.

---

## 7. Dataset, splits and normalisation

Implemented by `build_datasets()` and `MilanoTrafficDataset`:

- **Chronological split** of the time axis: 70% / 15% / 15% for train / val / test
  (`data.split`). Every sliding window lies **entirely inside one split**, so no target from a
  later period leaks into an earlier one.
- **Sliding windows** (stride `data.stride` for training, 1 for val/test):
  - `X`: `(T_in, C_in, H, W)`, frames `t0 … t0+T_in−1`
  - `Y`: `(T_out, C_out, H, W)`, frames `t0+T_in … t0+T_in+T_out−1`
  - Batched by the DataLoader: `X (B, T_in, C_in, H, W)`, `Y (B, T_out, C_out, H, W)`
  - Defaults: `T_in = 6` (1 hour), `T_out = 1` (10 minutes ahead)
- **Normalisation:** per-channel min-max to [0, 1] (`MinMaxNormalizer`), fitted on the
  **training segment only**. With `data.log_transform: true`, `log1p` is applied first. The
  fitted statistics are saved in the checkpoint and reused at evaluation time.
- **Activity mask:** cells whose mean target activity over the training period is exactly 0 are
  excluded from the loss and the metrics (`data.mask_inactive: true`).
- **Spatial crop:** `data.crop: [r0, r1, c0, c1]` trains on a sub-grid (for quick experiments
  or low memory).
- **Loading:** the tensor is opened lazily as a read-only memmap inside each DataLoader worker,
  so workers don't pickle the whole array. Only the training loader is shuffled.

---

## 8. Model architecture

```
Input X  (B, T_in, C_in, H, W)
   │
   ▼
ConvLSTM layer 1  (C_in → 64, 3×3)  ── h sequence (B, T_in, 64, H, W)
   │
   ▼
ConvLSTM layer 2  (64 → 64, 3×3)    ── final hidden state h_T (B, 64, H, W)
   │
   ▼
PredictionHead
   ├─ 1×1 Conv2d: 64 → T_out·C_out                 (coarse forecast)
   └─ refinement: [coarse ‖ h_T] → 3×3 Conv → ReLU → 3×3 Conv → + coarse
   │
   ▼
Output Ŷ  (B, T_out, C_out, H, W)
```

### `ConvLSTMCell(input_dim, hidden_dim, kernel_size, bias)`

- `forward(input_tensor, cur_state)`: concatenates the input and `h_{t−1}` along channels,
  applies **one** `Conv2d` with `4·hidden_dim` outputs, and splits it into the
  `i, f, o, g` gates:
  - `i, f, o = σ(·)`, `g = tanh(·)`
  - `c_t = f ⊙ c_{t−1} + i ⊙ g`
  - `h_t = o ⊙ tanh(c_t)`
- "Same" padding (the kernel must be odd), so H and W are preserved.
- The forget-gate bias starts at 1 to help gradients flow early in training.
- `init_hidden()` returns zero `(h, c)` states.

### `ConvLSTM(input_dim, hidden_dim, kernel_size, num_layers, bias, return_all_layers)`

- Stacks `num_layers` cells. `hidden_dim` and `kernel_size` can be a single value or a list with
  one entry per layer.
- Input `(B, T, C, H, W)`. Returns `(layer_outputs, last_states)`: the full hidden sequence and
  the final `(h, c)`, for the last layer only or for all layers.

### `PredictionHead(hidden_dim, out_channels, t_out, refine_layers, refine_channels)`

- A 1×1 projection to `t_out · out_channels`, plus an optional residual stack of
  `refine_layers` 3×3 convolutions (sees both the coarse output and `h_T`).
- **Direct multi-horizon** forecasting: all `T_out` frames come out in one pass, with no
  autoregressive rollout.

### `ConvLSTMForecaster`

- Encoder plus head. Optional `residual` mode predicts the change from the last observed frame
  (`ŷ = x_last + head(h_T)`). It is **off by default**; see §13.

### Parameter counts

| Configuration | Parameters |
|---|---|
| Default (hidden 64, 2 layers, refine 2×32), 1 input channel | 464,290 |
| Default, 5 input channels | 473,506 |
| Smoke config (hidden 16, refine 16), 1 input channel | 30,978 |

---

## 9. Training

```bash
python -m src.train --config configs/default.yaml
```

| Component | Setting |
|---|---|
| Loss | masked **MSE** (default) or masked **Huber** (`training.loss: huber`, `huber_delta` on the normalised scale) |
| Optimiser | Adam, lr `1e-3`, optional weight decay |
| Scheduler | `ReduceLROnPlateau` on val loss (factor 0.5, patience 5, min lr 1e-6) |
| Early stopping | patience 10 on val loss (`min_delta` configurable) |
| Epochs | up to 100 |
| Batch size | 16 |
| Gradient clipping | max norm 1.0 |
| Mixed precision | AMP float16 + GradScaler, CUDA only |
| Device | CUDA if available, otherwise CPU (`device: auto`) |
| Seeding | Python, NumPy and PyTorch (`seed: 42`) |
| Debug caps | `max_train_batches`, `max_eval_batches` |

Each epoch logs train loss, val loss, learning rate and duration to the console,
`outputs/<run>/train.log` and `outputs/<run>/history.csv`. When the validation loss improves,
the checkpoint `outputs/<run>/best_model.pt` is overwritten.

### Checkpoint contents

`model_state`, `config`, `epoch`, `val_loss`, `normalizer` (per-channel mins/maxs and the log
flag), `input_channels`, `target_channels`, `residual_channels` and `features`. It contains only
plain types, so it loads with `torch.load(..., weights_only=True)`.

### Memory and speed

On the full 100×100 grid with hidden 64 and batch 16, the activations take a few GB. On a
smaller GPU, lower `training.batch_size` or `model.hidden_dim`, or use `data.crop`. Full-grid
training on CPU works but is very slow.

---

## 10. Evaluation, metrics and baselines

```bash
python -m src.evaluate --config configs/default.yaml [--checkpoint path/to/best_model.pt]
```

- Model and data settings come from the **checkpoint's** config. Paths, evaluation settings and
  batch size come from the current config.
- Predictions and targets are **denormalised to original units**. Predictions are clamped to
  ≥ 0, because traffic can't be negative.
- Metrics are accumulated incrementally (the test set never sits in memory) and reported
  **overall, per prediction horizon, and per target channel**:
  - **RMSE**, **MAE**
  - **MAPE (%)**, counting only cells whose true value is above `evaluation.mape_threshold`
    (default 1.0), because MAPE is undefined at zero.
- Only active (unmasked) cells are scored when `mask_inactive` is on.

### Baselines

Both need every target feature to also be an input feature; otherwise they are skipped.

| Name | Prediction |
|---|---|
| `Persistence` | the last observed frame, repeated for every horizon ("previous timestep") |
| `WindowMean` | the mean of the `T_in` input frames (historical average) |

### Plots

| File | Content |
|---|---|
| `heatmaps.png` | for `num_plot_samples` random test frames: actual, ConvLSTM prediction and absolute error (north up; blue ramp for traffic, orange ramp for error, colour capped at the 99.5th percentile) |
| `cell_timeseries.png` | 1-step-ahead test series for one cell: actual vs ConvLSTM vs persistence. The cell is `evaluation.plot_cell` or a random cell from the 10% busiest. |
| `loss_curve.png` | train and validation loss per epoch (log scale) |

---

## 11. Configuration reference

Configs are YAML. `smoke.yaml` inherits `default.yaml` through `_base_: default.yaml` and
overrides only what it needs. Relative paths are resolved against the project root.

### Top level

| Key | Default | Meaning |
|---|---|---|
| `seed` | 42 | RNG seed |
| `device` | auto | `auto`, `cuda` or `cpu` |

### `paths`

| Key | Default | Meaning |
|---|---|---|
| `raw_dir` | `data/raw` | raw daily files |
| `processed_dir` | `data/processed` | `tensor.npy` and `meta.json` |
| `output_dir` | `outputs/default` | checkpoint, logs, metrics, plots |

### `data`

| Key | Default | Meaning |
|---|---|---|
| `features` | all 5 | channels stored in the processed tensor |
| `input_features` | `[internet]` | model inputs, e.g. all five |
| `target_features` | `[internet]` | predicted channels |
| `interval_minutes` | 10 | time resolution (60 for hourly mirrors) |
| `grid_size` | 100 | grid side length |
| `max_days` | null | preprocess only the first N days |
| `fill_method` | zero | `zero` or `ffill_zero` |
| `log_transform` | false | apply `log1p` before min-max scaling |
| `mask_inactive` | true | exclude never-active cells from loss and metrics |
| `t_in` | 6 | input frames (1 hour) |
| `t_out` | 1 | predicted frames |
| `stride` | 1 | step between training windows |
| `split` | `[0.70, 0.15, 0.15]` | chronological train/val/test ratios |
| `crop` | null | `[r0, r1, c0, c1]` sub-grid |
| `num_workers` | 0 | DataLoader workers (0 is safest on Windows) |

### `model`

| Key | Default | Meaning |
|---|---|---|
| `hidden_dim` | 64 | int, or a list with one entry per layer |
| `num_layers` | 2 | stacked ConvLSTM cells |
| `kernel_size` | `[3, 3]` | odd convolution kernel |
| `bias` | true | gate-convolution bias |
| `refine_layers` | 2 | 3×3 refinement convs in the head (0 = plain 1×1 projection) |
| `refine_channels` | 32 | refinement width |
| `residual` | false | predict the change from the last frame |

### `training`

| Key | Default | Meaning |
|---|---|---|
| `batch_size` | 16 | |
| `epochs` | 100 | maximum epochs |
| `lr` | 1e-3 | Adam learning rate |
| `weight_decay` | 0.0 | |
| `loss` | mse | `mse` or `huber` |
| `huber_delta` | 0.05 | on the normalised scale |
| `patience` | 10 | early-stopping patience |
| `min_delta` | 0.0 | minimum improvement |
| `scheduler_patience` | 5 | ReduceLROnPlateau |
| `scheduler_factor` | 0.5 | |
| `min_lr` | 1e-6 | |
| `grad_clip` | 1.0 | max gradient norm |
| `amp` | true | mixed precision (CUDA only) |
| `max_train_batches` | null | debug cap |
| `max_eval_batches` | null | debug cap |

### `evaluation`

| Key | Default | Meaning |
|---|---|---|
| `mape_threshold` | 1.0 | minimum true value counted in MAPE |
| `num_plot_samples` | 3 | heatmap rows |
| `plot_cell` | null | `[row, col]` in the (cropped) grid; null = random busy cell |

### `smoke.yaml` overrides

Synthetic data paths, `output_dir: outputs/smoke`, `max_days: 3`, `crop: [34, 66, 34, 66]`
(32×32), `hidden_dim: 16`, `refine_channels: 16`, `batch_size: 8`, `epochs: 3`, `amp: false`.

---

## 12. Command-line usage

Every entry point takes `--config <file>` and `--set key=value ...`. Values are parsed as YAML,
so numbers, booleans, lists and `null` all work. Run commands from the project root.

| Command | Purpose |
|---|---|
| `python -m src.dataset --config ...` | build `tensor.npy` and `meta.json` |
| `python -m src.train --config ...` | train and save the best checkpoint |
| `python -m src.evaluate --config ... [--checkpoint ...]` | metrics and plots |
| `python scripts/make_synthetic_data.py --out DIR --days N [--start 2013-11-01 --grid 100 --seed 0]` | synthetic raw files |

Examples:

```bash
# all five features in, internet out, 3 steps (30 min) ahead
python -m src.train --config configs/default.yaml \
  --set data.input_features=[smsin,smsout,callin,callout,internet] data.t_out=3 paths.output_dir=outputs/all5_h3

# log scaling + Huber loss
python -m src.train --config configs/default.yaml --set data.log_transform=true training.loss=huber

# quick check on the first week of real data, on a sub-grid
python -m src.dataset --config configs/default.yaml --set data.max_days=7
python -m src.train   --config configs/default.yaml --set data.crop=[30,70,30,70] training.epochs=5

# smoke test on the real data instead of synthetic
python -m src.dataset --config configs/smoke.yaml --set paths.raw_dir=data/raw paths.processed_dir=data/processed
```

Remember to pass the same `--set` overrides (at least `paths.output_dir`) to `src.evaluate` that
you used for training.

---

## 13. Results

### Real data: full run on Kaggle (`outputs/kaggle/`)

Setup:

| Item | Value |
|---|---|
| Data | Telecom Italia Milano, all 62 days (2013-11-01 to 2014-01-01), 10-minute steps, full 100×100 grid |
| Tensor | (8928, 5, 100, 100); preprocessing took about 11 minutes |
| Task | internet traffic, 6 frames (1 hour) in → 1 frame (10 minutes) ahead |
| Windows | 6,243 train / 1,333 val / 1,334 test. The test period is about 2013-12-23 to 2014-01-01 (Christmas and New Year). |
| Model | default config: 2 ConvLSTM layers, hidden 64, 3×3 kernels, refinement head; 464,290 parameters |
| Hardware | Kaggle Tesla T4, PyTorch 2.11 (CUDA), mixed precision |
| Training | about 146 s per epoch; early stopping after epoch 51 (best: epoch 41); about 2 h 5 min of training, 2 h 18 min for the whole notebook |
| Active cells | 10,000 / 10,000 (none masked) |

Test-set results, in original units:

| Model | RMSE | MAE | MAPE % |
|---|---|---|---|
| **ConvLSTM** | **8.4745** | **3.9071** | **11.99** |
| Persistence (last frame) | 9.3922 | 4.3344 | 12.56 |
| Window mean (last hour) | 10.3727 | 4.4394 | 12.13 |
| *ConvLSTM vs persistence* | *−9.8%* | *−9.9%* | *−4.6%* |
| *ConvLSTM vs window mean* | *−18.3%* | *−12.0%* | *−1.2%* |

The ConvLSTM beats both baselines on all three metrics. Persistence is the stronger baseline
here (unlike on the synthetic data), as expected for real 10-minute traffic.

Observations:

- **Errors are concentrated in the dense city centre.** That is where traffic levels, and
  therefore absolute errors, are highest (`heatmaps.png`).
- **Validation loss is below training loss throughout.** The later periods (mid to late
  December, then the holidays) have lower overall traffic, so their errors are smaller on the
  normalised scale. It is not a sign of leakage.
- **The normalised losses are very small (~1e-6).** Min-max scaling with a few very busy cells
  pushes most values close to 0. This is the main motivation for trying
  `data.log_transform: true` next.

### Synthetic data: smoke test and verification

The results in the rest of this section come from **synthetic data** (`scripts/make_synthetic_data.py`). It has 12
Gaussian traffic hotspots scaled by a daily cycle, with lognormal noise. Like the real data, it
has extra country-code rows, empty fields for tiny values, and about 3% of rows dropped.

### Correctness checks

- **Values:** a spot check of one (time, cell) internet value matched the raw file exactly
  (6.6311 = 6.6311), and the day total of the tensor matched a direct pandas groupby.
- **Timezone:** epoch `1383260400000` ms (2013-10-31 23:00 UTC) maps to 2013-11-01 00:00 Milan
  time, which is time index 0.
- **Forward fill:** `ffill_zero` leaves 0 NaNs. Zero cells go from 15.9% (zero-fill) to 3.0%
  (forward-fill) on day 1.
- **Model shapes:** `(2, 6, 5, 20, 24)` → `(2, 3, 2, 20, 24)` with per-layer hidden sizes
  `[8, 12]`, refinement and residual on. The backward pass works.
- **Splits:** 3 days = 432 steps, giving 296 / 58 / 60 train / val / test windows.

### Smoke run (`configs/smoke.yaml`, 3 epochs, artefacts in `outputs/smoke/`)

| Epoch | Train loss | Val loss | Time |
|---|---|---|---|
| 1 | 0.002776 | 0.001553 | 3.7 s |
| 2 | 0.000487 | **0.000580** (best) | 3.5 s |
| 3 | 0.000387 | 0.000640 | 3.5 s |

| Model | RMSE | MAE | MAPE % |
|---|---|---|---|
| ConvLSTM (3 epochs, undertrained) | 19.36 | 10.68 | 60.96 |
| Persistence | 19.95 | 9.67 | 33.61 |
| WindowMean | 16.44 | 8.03 | 28.22 |

### Ablations (same smoke setup, 12 epochs)

| Setting | ConvLSTM RMSE | MAE | MAPE % |
|---|---|---|---|
| `residual: false` (current default) | **15.07** | 7.60 | 29.55 |
| `log_transform: true`, `residual: true` | 15.25 | **7.20** | **22.60** |
| `log_transform: true`, `residual: false` | **14.94** | 7.31 | 26.22 |
| `residual: true`, plain min-max | 20.4: stuck at the persistence level | | |
| Persistence / WindowMean (reference) | 19.95 / 16.44 | 9.67 / 8.03 | 33.61 / 28.22 |

A 20-epoch run with `t_out=3` and all five input channels also confirmed that metrics are
reported per horizon. That run used the old `residual: true` setting, which stalled at the
persistence level (RMSE ~20).

**Takeaways:**

- Trained for 12 epochs, the ConvLSTM beats both baselines on RMSE and MAE.
- The residual mode stalls with plain min-max scaling, so it is off by default.
- `log1p` scaling improves MAPE.
- On synthetic data the window mean is a strong baseline, because the noise is independent at
  every step. These numbers show the pipeline learns; they do **not** predict real-data
  performance.

---

## 14. Design decisions

| Decision | Choice | Reason |
|---|---|---|
| Missing values | zero-fill (default); forward-fill as an option | Missing rows and fields in this dataset mean no activity. Forward-filling would make up traffic that didn't happen. |
| Inactive cells | masked from loss and metrics | Never-active cells add only trivially easy zeros. |
| Country codes | summed | The target is total traffic per cell. |
| Timestamps | UTC → Europe/Rome, absolute indexing | Days line up with the local calendar, and boundary spill-over lands correctly. |
| Normalisation | per-channel min-max fitted on train only, optional `log1p` | Prevents leakage; `log1p` handles heavy tails. |
| Split | chronological, windows confined to one split | No temporal leakage. |
| Multi-step output | direct, all horizons at once | Simple and fast, with no error accumulation from an autoregressive rollout. |
| Storage | float32 `.npy` memmap | The full 1.8 GB tensor never has to fit in RAM. |
| MAPE | only true values > threshold | Undefined or explosive near zero. |
| Predictions | clamped to ≥ 0 at evaluation | Traffic can't be negative. |
| Residual head | off | It stalled at persistence in tests. |
| `num_workers` | 0 by default | Safest on Windows; raise it on Linux. |

---

## 15. Code reference

### `src/utils.py`

| Function | Purpose |
|---|---|
| `load_config(path, overrides)` | load YAML, follow `_base_`, apply dotted `key=value` overrides |
| `add_common_args(parser)` | add `--config` and `--set` |
| `resolve_path(path)` | anchor relative paths at the project root |
| `set_seed(seed)` | seed Python, NumPy and PyTorch |
| `get_device(pref)` | CUDA if available, else CPU |
| `get_logger(name, log_file)` | console and file logger |
| `save_json` / `load_json` | JSON I/O |

### `src/dataset.py`

| Name | Purpose |
|---|---|
| `FEATURES` | `("smsin", "smsout", "callin", "callout", "internet")` |
| `cell_id_to_rowcol(ids, grid_size)` | CellID → `(row, col)` |
| `discover_raw_files(raw_dir)` | find and sort daily files by date |
| `read_raw_file(path, features)` | parse one day (either format), local time, sum over countries |
| `preprocess(...)` | build `tensor.npy` and `meta.json` |
| `MinMaxNormalizer` | `normalize` / `denormalize` (NumPy or torch), `state_dict` / `from_state_dict` |
| `compute_train_statistics(...)` | fit the normalizer and the activity map on the training segment |
| `split_bounds(T, ratios)` | chronological split ranges |
| `MilanoTrafficDataset` | sliding windows over one split; `target_time_index(i)` |
| `DataBundle` | datasets, normalizer, mask, activity, meta, channel indices |
| `build_datasets(cfg, normalizer)` | create the train/val/test datasets |
| `build_dataloaders(cfg, datasets)` | wrap them in DataLoaders |

### `src/model.py`

| Name | Purpose |
|---|---|
| `ConvLSTMCell` | single cell (`forward`, `init_hidden`) |
| `ConvLSTM` | multi-layer encoder |
| `PredictionHead` | 1×1 projection plus optional 3×3 refinement |
| `ConvLSTMForecaster` | full model |
| `build_model(cfg, in_dim, out_dim, residual)` | build the model from the config |
| `residual_positions(in_ch, tgt_ch)` | positions of the target channels within the inputs |

### `src/train.py`

| Name | Purpose |
|---|---|
| `masked_loss(pred, target, mask, kind, delta)` | MSE or Huber over unmasked cells |
| `EarlyStopping` | patience-based stopping |
| `run_epoch(...)` | one train or eval pass (AMP, clipping, batch caps) |
| `train(cfg)` | full training loop; returns the checkpoint path |

### `src/evaluate.py`

| Name | Purpose |
|---|---|
| `MetricAccumulator` | streaming RMSE/MAE/MAPE per (horizon, channel) |
| `plot_heatmaps`, `plot_cell_series`, `plot_history` | the three figures |
| `evaluate(cfg, checkpoint)` | full evaluation; writes metrics and plots |

---

## 16. Output files

Every run writes these to `paths.output_dir`:

| File | Written by | Content |
|---|---|---|
| `best_model.pt` | train | best checkpoint (see §9) |
| `config_used.json` | train | exact config of the run |
| `history.csv` | train | epoch, train_loss, val_loss, lr, seconds |
| `train.log` | train | training log |
| `metrics.json` | evaluate | overall, per-horizon and per-channel metrics for all models, plus notes (units, active cells, MAPE threshold, test windows, checkpoint) |
| `metrics_per_horizon.csv` | evaluate | model, horizon, RMSE, MAE, MAPE_% |
| `evaluate.log` | evaluate | evaluation log and metrics table |
| `heatmaps.png`, `cell_timeseries.png`, `loss_curve.png` | evaluate | plots |

Preprocessing writes `tensor.npy` and `meta.json` to `paths.processed_dir`.

---

## 17. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `OSError: [Errno 22] Invalid argument: '...png'` during evaluation | Windows locks a PNG that is open in an image viewer. Close the viewer, or evaluate into another folder with `--set paths.output_dir=...` and `--checkpoint`. |
| `No 'sms-call-internet-mi-YYYY-MM-DD' files found` | Wrong `paths.raw_dir`, or the file names lack the date pattern. |
| `tensor.npy not found` | Run `python -m src.dataset` first, with the same `processed_dir`. |
| `Features [...] not in the processed tensor` | Re-run preprocessing with those features in `data.features`. |
| Warning: smallest timestamp gap is 1:00:00 | Hourly mirror; set `data.interval_minutes: 60`. |
| `Segment ... is too short for t_in/t_out` | Too few days for the split; increase `max_days` or reduce `t_in`/`t_out`. |
| CUDA out of memory | Lower `batch_size` or `hidden_dim`, or use `data.crop`; AMP is on by default. |
| Pandas "Pyarrow will become a required dependency" warning | Harmless; `pip install pyarrow` silences it. |
| Model stuck at persistence-level error | Make sure `model.residual: false`; try `data.log_transform: true`. |
| DataLoader hangs on Windows with workers > 0 | Keep `num_workers: 0`, or run from a proper script entry point (already done). |

---

## 18. Known limitations and next steps

- **One real-data configuration run so far** (internet only, 1 step ahead). Next experiments:
  - `log_transform: true`
  - all five features as input
  - `t_out: 3` or `6` for 30–60 minutes ahead
- **No GPU on the development machine.** Full-grid training is done on Kaggle (§3).
- **Grid orientation** (IDs running south→north) should be checked against
  `milano-grid.geojson` if exact geography matters.
- **Possible extensions:**
  - a daily-seasonal baseline (same time yesterday)
  - calendar and time-of-day features
  - an encoder–decoder / seq2seq head for longer horizons
  - a Conv3d refinement head
  - hyperparameter sweeps
  - multi-target training on all five channels
  - TensorBoard logging
- `scikit-learn` is pinned as requested but unused; it can be removed from `requirements.txt`.
