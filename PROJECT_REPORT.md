# Project Report and Presentation Guide

**Spatio-Temporal ConvLSTM for Cellular Traffic Prediction: Telecom Italia "Milano" Dataset**

This file brings together everything about the project in one place:

- what was built
- every finding and check made along the way
- the final results on the real data
- what to show your professor, and how to answer likely questions

Related documents:

| File | Purpose |
|---|---|
| `README.md` | technical reference: setup, commands, configuration |
| `PROJECT_OVERVIEW.md` | plain-language explanation of the project |
| `outputs/kaggle/` | the real-data results: metrics, plots, logs, checkpoint |

Links:

- GitHub repository: <https://github.com/Hasini885/convlstm-milano-traffic-prediction>
- Kaggle notebook (full run): <https://www.kaggle.com/code/hasinisaimaram/convlstm-milano-traffic-prediction>

---

## Contents

1. [Summary in one page](#1-summary-in-one-page)
2. [Problem statement and motivation](#2-problem-statement-and-motivation)
3. [Dataset: description and findings](#3-dataset-description-and-findings)
4. [Preprocessing: what was done and why](#4-preprocessing-what-was-done-and-why)
5. [Model: ConvLSTM architecture](#5-model-convlstm-architecture)
6. [Training setup](#6-training-setup)
7. [Evaluation method and baselines](#7-evaluation-method-and-baselines)
8. [Development and verification findings (synthetic data)](#8-development-and-verification-findings-synthetic-data)
9. [Final results on the real data](#9-final-results-on-the-real-data)
10. [Analysis and interpretation of the results](#10-analysis-and-interpretation-of-the-results)
11. [Engineering findings and problems solved](#11-engineering-findings-and-problems-solved)
12. [Limitations](#12-limitations)
13. [Future work and suggested experiments](#13-future-work-and-suggested-experiments)
14. [What to show your professor](#14-what-to-show-your-professor)
15. [Likely questions, with answers](#15-likely-questions-with-answers)
16. [How to reproduce everything](#16-how-to-reproduce-everything)
17. [Project timeline](#17-project-timeline)
18. [References and data citation](#18-references-and-data-citation)

---

## 1. Summary in one page

| Item | Value |
|---|---|
| Goal | Predict mobile internet traffic on every cell of a 100×100 grid over Milan, 10 minutes ahead, from the previous 1 hour |
| Data | Telecom Italia Big Data Challenge: 62 days (2013-11-01 to 2014-01-01), 10-minute resolution, 10,000 cells, 5 activity types |
| Model | 2-layer ConvLSTM (hidden 64, 3×3 kernels) plus a convolutional prediction head; 464,290 parameters |
| Training | Kaggle Tesla T4 GPU, masked MSE, Adam, ReduceLROnPlateau, early stopping; 51 epochs, about 2 h 5 min |
| Test period | 2013-12-23 17:40 to 2014-01-01 23:50 (Christmas and New Year), 1,334 forecasts × 10,000 cells |
| **Result** | **ConvLSTM RMSE 8.47 / MAE 3.91 / MAPE 11.99%** |
| Best baseline | Persistence (repeat the last frame): RMSE 9.39 / MAE 4.33 / MAPE 12.56% |
| Improvement | **−9.8% RMSE, −9.9% MAE, −4.6% MAPE** vs persistence; −18.3% RMSE vs the 1-hour window mean |

**In one sentence:** a ConvLSTM trained on the full real Milan dataset predicts 10-minute-ahead
internet traffic across the whole city about 10% more accurately than the strongest naive
baseline, with no data leakage between training and test periods.

---

## 2. Problem statement and motivation

Mobile networks have to provide capacity where and when people use it. Traffic changes quickly
in both **space** (a crowd moves from the station to the stadium) and **time** (daily rhythms,
evening peaks, holidays).

If an operator can forecast traffic for each area a few minutes ahead, it can act before
congestion happens:

- move bandwidth to where it will be needed
- switch cells on and off to save energy
- balance load between cells

**Research question:** given the traffic on a 100×100 grid over the last hour, can a
spatio-temporal deep-learning model predict the traffic for the next 10 minutes better than
simple rules such as "the next 10 minutes will look like the last 10"?

**Why this is a spatio-temporal problem:** each 10-minute snapshot of the city is an image
(a 100×100 "pixel" grid where brightness is traffic), and the sequence of snapshots is a video.
Forecasting is predicting the next frame, which needs a model that understands spatial
neighbourhoods and temporal dynamics at the same time.

---

## 3. Dataset: description and findings

### 3.1 Description

- **Source:** Telecom Italia Big Data Challenge, "Telecommunications - SMS, Call, Internet - MI",
  Harvard Dataverse, doi:10.7910/DVN/EGZHFV.
- **Spatial unit:** Milan is split into 100×100 = 10,000 square cells of about 235 m × 235 m,
  numbered with `CellID` 1–10,000.
- **Temporal unit:** 10-minute intervals. That is 144 per day; over 62 days, T = 8,928 time steps.
- **Five activity types (channels):** `smsin`, `smsout`, `callin`, `callout`, `internet`. The
  values are anonymised, scaled activity levels, not raw counts.
- **Raw format:** one file per day (`sms-call-internet-mi-YYYY-MM-DD.txt`), tab-separated, no
  header. Columns: cell ID, time interval (milliseconds since epoch, UTC), country code, then
  the five activity values.

### 3.2 Findings about the data

1. **One row per (cell, time, country code).** Traffic is split by the country of the foreign
   party, so rows must be **summed over country codes** to get the total per cell.
2. **The data is sparse.** A row only exists when there was activity, and individual fields are
   empty when that activity type was zero. A missing value therefore means "no activity", not
   "unknown". This drove the missing-value decision (§4).
3. **Timestamps are UTC.** They must be converted to Milan local time (Europe/Rome, UTC+1 in
   winter) so that days and daily cycles line up with local life. The first record,
   `1383260400000` ms, is 2013-10-31 23:00 UTC, which is 2013-11-01 00:00 in Milan.
4. **Size.** About 300–360 MB per day uncompressed, 20.8 GB in total. The first day holds
   1,439,982 (time, cell) records after aggregation.
5. **Every cell is active.** All 10,000 cells had some internet activity during the training
   period, so the "inactive cell" mask removed nothing on the real data.
6. **Access.** Harvard Dataverse requires a **guestbook form** (personal details) before any
   download, which blocks automated downloads (HTTP 400: "You may not download this file
   without the required Guestbook response"). The API also rejects Python's default user agent
   (HTTP 403).
7. **Kaggle mirror verified.** The Kaggle dataset `dkgmgo/telecom-italia-milan` contains all 62
   daily files plus `milano-grid.geojson`. **All 62 files match the official Dataverse release
   byte for byte** (file sizes compared against the Dataverse API listing), so the experiments
   use the original, complete data.
8. **Other mirrors are incomplete or different.** For example, `marcodena/mobile-phone-activity`
   holds only 7 days in CSV form, and `freckled/telecom` holds two large merged CSVs. These were
   not used.

---

## 4. Preprocessing: what was done and why

Implemented in `src/dataset.py` (`preprocess()`).

| Step | What it does | Why |
|---|---|---|
| 1. Discover files | Finds the daily files and sorts them by the date in the name | Supports `.txt`, `.csv` and gzip-compressed versions |
| 2. Read and aggregate | Reads each day; sums rows over country codes (empty fields stay empty unless at least one value exists) | Total traffic per cell |
| 3. Time conversion | UTC epoch ms → Milan local time | Daily cycles match local time |
| 4. Grid mapping | `row = (CellID−1) // 100`, `col = (CellID−1) % 100` | IDs run west→east, then south→north; row 0 is the southern edge, so plots use `origin="lower"` (north up) |
| 5. Absolute time index | `t = (time − origin) / 10 min`; origin = local midnight of the first day | Records that cross midnight (because of the UTC shift) land in the right slot |
| 6. Tensor storage | float32 `.npy` memory-mapped file of shape **(T, C, H, W) = (8928, 5, 100, 100)**, about 1.8 GB | The full tensor never needs to fit in RAM |
| 7. Missing values | **Zero-fill** (default); forward-fill + zero-fill available | Missing = no activity (finding 3.2.2); forward-filling would invent traffic |
| 8. Metadata | `meta.json`: features, shape, origin, interval, fill method, list of days | Reproducibility |

Then, when building the datasets (`build_datasets()`):

- **Chronological split 70/15/15** of the time axis:
  - Train: 2013-11-01 00:00 → 2013-12-14 09:30 (6,249 steps)
  - Validation: 2013-12-14 09:30 → 2013-12-23 16:40 (1,339 steps)
  - Test: 2013-12-23 16:40 → 2014-01-02 00:00 (1,340 steps)
- **Every sliding window lies entirely inside one split,** so no target from a later period
  is ever seen during training (no temporal leakage).
- **Min-max scaling to [0, 1] per channel, fitted on the training segment only,** so no test
  statistics leak into training.
- **Sliding windows:** input X = 6 frames (1 hour), target Y = the next 1 frame (10 minutes).
  That gives 6,243 training, 1,333 validation and 1,334 test windows.
- **Mask:** cells with zero activity over the entire training period are excluded from loss and
  metrics. (On the real data none were excluded; see finding 3.2.5.)

---

## 5. Model: ConvLSTM architecture

Implemented in `src/model.py`.

```
Input  X: (B, 6, 1, 100, 100)        6 past frames of internet traffic
  │
  ▼
ConvLSTM layer 1 (1 → 64 channels, 3×3)
  │   hidden-state sequence (B, 6, 64, 100, 100)
  ▼
ConvLSTM layer 2 (64 → 64 channels, 3×3)
  │   final hidden state h_T (B, 64, 100, 100)
  ▼
Prediction head
  ├─ 1×1 Conv: 64 → 1                        coarse forecast
  └─ refinement: [coarse ‖ h_T] → 3×3 Conv (32) → ReLU → 3×3 Conv (1) → + coarse
  │
  ▼
Output Ŷ: (B, 1, 1, 100, 100)        next frame
```

### 5.1 ConvLSTM cell (Shi et al., 2015)

The input and the previous hidden state are concatenated along channels. **One** convolution
produces all four gates:

- input gate `i = σ(·)`, forget gate `f = σ(·)`, output gate `o = σ(·)`
- candidate `g = tanh(·)`
- cell state `c_t = f ⊙ c_{t−1} + i ⊙ g`
- hidden state `h_t = o ⊙ tanh(c_t)`

It differs from a normal LSTM in that the states are **images** (64 × 100 × 100), and the
gates are computed with **convolutions** instead of fully connected layers. Each cell therefore
decides what to remember based on itself *and its neighbours*.

Details:

- "Same" padding keeps the 100×100 size.
- The forget-gate bias starts at 1, a standard trick for stable early training.
- Hidden states start at zero.

### 5.2 Prediction head

- A 1×1 convolution maps the 64 hidden channels to the forecast. A small residual stack of
  3×3 convolutions then refines it spatially; recent papers suggest such a stack sharpens the
  outputs.
- Multi-step forecasting (if `t_out > 1`) is **direct**: all future frames come out at once,
  with no autoregressive error build-up.

### 5.3 Size

| Configuration | Parameters |
|---|---|
| Default, 1 input channel (used in the final run) | **464,290** |
| Default, 5 input channels | 473,506 |
| Smoke-test configuration (hidden 16) | 30,978 |

---

## 6. Training setup

Implemented in `src/train.py`; configuration in `configs/default.yaml` and `configs/kaggle.yaml`.

| Setting | Value |
|---|---|
| Loss | masked mean squared error (MSE) on the normalised scale |
| Optimiser | Adam, learning rate 1e-3 |
| LR scheduler | ReduceLROnPlateau on validation loss: factor 0.5, patience 5, min 1e-6 |
| Early stopping | patience 10 epochs on validation loss |
| Max epochs | 100 |
| Batch size | 16 |
| Gradient clipping | max norm 1.0 |
| Mixed precision | yes (float16 autocast + GradScaler on CUDA) |
| Time budget | 8.5 h (stop cleanly before Kaggle's 12 h limit, so evaluation always runs) |
| Seed | 42 (Python, NumPy, PyTorch) |
| Hardware | Kaggle **Tesla T4** GPU, PyTorch 2.11 (CUDA 13.0) |
| Checkpointing | best model by validation loss → `best_model.pt` (weights, config, normaliser, channel info) |

---

## 7. Evaluation method and baselines

Implemented in `src/evaluate.py`.

- Predictions are **converted back to original units** before scoring, and clamped to ≥ 0
  because traffic can't be negative.
- **Metrics,** over all test windows × all active cells:
  - **RMSE** (root mean squared error): penalises large errors more.
  - **MAE** (mean absolute error): the average size of the error.
  - **MAPE** (mean absolute percentage error): relative error, counting only cells whose true
    value is > 1.0, because MAPE explodes near zero.
- Metrics are reported overall, per prediction horizon and per channel, and accumulated
  incrementally so the test set never has to sit in memory.
- **Baselines:**
  - **Persistence:** the prediction is the last observed frame ("the next 10 minutes look like
    the last 10"). Very strong for short-term forecasting.
  - **Window mean:** the prediction is the average of the 6 input frames (historical average).
- **Plots:**
  - `heatmaps.png`: actual, predicted and absolute error for 3 random test times
  - `cell_timeseries.png`: test-period series for a busy cell
  - `loss_curve.png`: training and validation loss per epoch

---

## 8. Development and verification findings (synthetic data)

Before using GPU time, the whole pipeline was built and tested on the development laptop
(Windows 11, Python 3.11.9, CPU-only PyTorch 2.12.1, no GPU). A **synthetic data generator**
(`scripts/make_synthetic_data.py`) writes files in the *exact* raw Dataverse format:

- 12 Gaussian traffic hotspots scaled by a daily cycle, with log-normal noise
- extra foreign country-code rows
- empty fields for tiny values
- about 3% of rows removed

### 8.1 Correctness checks (all passed)

| Check | Result |
|---|---|
| A single (time, cell) value in the tensor vs the raw file | identical (6.6311 = 6.6311) |
| A full day's total in the tensor vs a direct pandas groupby | identical |
| UTC → Milan conversion | `1383260400000` ms → 2013-11-01 00:00 local = time index 0 |
| Forward-fill mode | 0 NaNs left; zero cells drop from 15.9% (zero-fill) to 3.0% (forward-fill) on day 1 |
| Gzip input (`.txt.gz`) | tensor **identical** to the one built from plain `.txt` |
| Model shapes | (2, 6, 5, 20, 24) → (2, 3, 2, 20, 24) with per-layer hidden sizes [8, 12], refinement and residual mode; backward pass OK |
| Splits on 3 days (432 steps) | 296 / 58 / 60 train / val / test windows |
| Time budget | training stopped cleanly when the budget was reached and kept the best checkpoint |

### 8.2 Smoke-test results (3 days, 32×32 crop, hidden 16)

**3-epoch run** (`outputs/smoke/`):

| Epoch | Train loss | Val loss |
|---|---|---|
| 1 | 0.002776 | 0.001553 |
| 2 | 0.000487 | **0.000580** (best) |
| 3 | 0.000387 | 0.000640 |

Test: ConvLSTM RMSE 19.36 vs persistence 19.95 vs window mean 16.44. The model was undertrained
after only 3 epochs.

**Ablation study (12 epochs each):**

| Setting | ConvLSTM RMSE | MAE | MAPE % |
|---|---|---|---|
| `residual: false` (chosen default) | **15.07** | 7.60 | 29.55 |
| `log_transform: true`, `residual: true` | 15.25 | **7.20** | **22.60** |
| `log_transform: true`, `residual: false` | **14.94** | 7.31 | 26.22 |
| `residual: true`, plain min-max | ~20.4 (stuck at persistence level) | | |
| *Persistence / window mean (reference)* | *19.95 / 16.44* | *9.67 / 8.03* | *33.61 / 28.22* |

**Findings from development:**

1. **The pipeline learns.** After 12 epochs the ConvLSTM beats both baselines on RMSE.
2. **Residual mode doesn't work with min-max scaling.** Predicting the *change* from the last
   frame made the model collapse to the persistence solution (RMSE ~20.4), so it was **turned
   off by default**.
3. **Log scaling helps MAPE.** `log_transform: true` gave the best MAPE (22.6%), because it
   spreads out small values. This is the most promising next experiment on the real data.
4. **Synthetic data is not a performance predictor.** Its noise is independent at every step,
   which makes the window mean an unusually strong baseline there. On real data, persistence is
   the stronger baseline (§9).
5. A multi-horizon test (`t_out = 3`, all five features as input) confirmed that per-horizon
   metrics work; the error grows with the horizon, as expected.

---

## 9. Final results on the real data

**Run:** Kaggle notebook `hasinisaimaram/convlstm-milano-traffic-prediction`, version 1,
2026-11-03. All artefacts are in `outputs/kaggle/`.

### 9.1 Run details

| Item | Value |
|---|---|
| Data | all 62 days, 10-minute steps (detected automatically), 10,000 cells |
| Tensor | (8928, 5, 100, 100); preprocessing took **about 11 minutes** for 62 files |
| Input → target | internet → internet; 6 frames in, 1 frame out |
| Windows | 6,243 train / 1,333 val / 1,334 test |
| Active cells | 10,000 / 10,000 |
| GPU | Tesla T4 |
| Epoch time | about **146.8 s** on average (6,243 training windows + 1,333 validation windows) |
| Epochs | 51 (early stopping); **best epoch 41** |
| Learning rate | 1e-3, halved at epochs 19, 33 and 47 (final 1.25e-4) |
| Training time | **124.8 min** (about 2 h 5 min); whole notebook about 2 h 18 min |
| Best validation loss | 1.829e-6 (normalised MSE); training loss at that epoch 6.06e-6 |

### 9.2 Test-set metrics (original units)

| Model | RMSE | MAE | MAPE % |
|---|---|---|---|
| **ConvLSTM** | **8.4745** | **3.9071** | **11.99** |
| Persistence (last frame) | 9.3922 | 4.3344 | 12.56 |
| Window mean (last hour) | 10.3727 | 4.4394 | 12.13 |

### 9.3 Improvement of the ConvLSTM

| Compared with | RMSE | MAE | MAPE |
|---|---|---|---|
| Persistence | **−9.8%** | **−9.9%** | **−4.6%** |
| Window mean | **−18.3%** | **−12.0%** | **−1.2%** |

**The ConvLSTM beats both baselines on all three metrics.**

### 9.4 Plots (in `outputs/kaggle/`)

- **`heatmaps.png`** (test times 2013-12-27 19:10, 2013-12-29 19:00 and 2013-12-30 21:20): the
  predicted maps reproduce Milan's spatial structure, with the dense city centre and the
  suburban hotspots (stations, shopping areas, the airport region). The absolute error is
  concentrated in the busy centre.
- **`cell_timeseries.png`** (CellID 4128, row 41, col 27, a busy cell): over the test period the
  prediction follows the daily cycle (night lows around 80–100, daytime highs of 250–330) and
  the holiday days.
- **`loss_curve.png`**: training loss drops sharply after epoch 1 and then slowly improves;
  validation loss settles early and flattens, so early stopping triggered at epoch 51.

---

## 10. Analysis and interpretation of the results

1. **Beating persistence is the key result.** For 10-minute-ahead traffic, "repeat the last
   frame" is very hard to beat, because traffic changes little in 10 minutes. A 10% lower RMSE
   than persistence shows the ConvLSTM really uses the spatial and temporal context: trends
   over the hour, and neighbouring cells.
2. **Persistence is stronger than the window mean on real data** (RMSE 9.39 vs 10.37). Real
   traffic is smooth over 10 minutes but drifts over an hour, so averaging the whole hour lags
   behind trends. On synthetic data it was the other way round, because there the noise was
   random at every step.
3. **The MAPE improvement is smaller (−4.6%) than the RMSE improvement (−9.8%).** RMSE is
   dominated by busy cells with large absolute values, where the ConvLSTM helps most. MAPE
   weights every cell equally in relative terms, and low-traffic cells, where relative errors
   are noisier, count as much as busy ones.
4. **Errors are concentrated in the city centre.** Absolute errors scale with traffic level, so
   the busiest cells have the largest errors (see the heatmaps). In relative terms the model is
   consistent across the city.
5. **Validation loss is below training loss for the whole run.** This is *not* leakage: the
   split is strictly chronological and the scaling uses training statistics only. The
   validation period (mid to late December) has lower overall traffic than November, so the
   squared errors are smaller on the normalised scale.
6. **The test period is unusual and hard.** Christmas and New Year have different behaviour
   from the training period (less commuting, holiday patterns). The model still beats both
   baselines there, which suggests it generalises.
7. **Most learning happens in the first epoch,** and later epochs refine it slowly. The
   normalised losses are very small (~1e-6) because min-max scaling with a few extremely busy
   cells compresses most values towards 0. This limits how much the loss can tell the model
   about low-traffic cells, and is the main motivation for trying log scaling.

---

## 11. Engineering findings and problems solved

| Problem | Finding | Solution |
|---|---|---|
| Raw data has one row per country code | Totals per cell need aggregation | Sum over country codes during reading |
| Sparse records, empty fields | Missing means no activity | Zero-fill by default; forward-fill as an option |
| UTC timestamps | Days would be shifted by an hour | Convert to Europe/Rome; absolute time indexing |
| 1.8 GB tensor | Too big to keep in RAM comfortably | Memory-mapped `.npy`, opened lazily in each DataLoader worker |
| Possible temporal leakage | Random splits leak the future | Chronological split; windows confined to one split; scaling fitted on training only |
| Residual model collapsed to persistence | Found in the ablation | Turned residual mode off by default |
| Dataverse guestbook blocks downloads | HTTP 400 without a guestbook response; 403 for the Python user agent | Used the Kaggle mirror, verified byte for byte against the Dataverse file list |
| Disk space for raw data | 20.8 GB uncompressed | Optional gzip-on-download; preprocessing reads `.gz` directly |
| Kaggle's 12-hour session limit | Training must not stop evaluation from running | `training.max_hours` budget (8.5 h) |
| No GPU on the laptop | Full-grid training is impractical on CPU | Ran on a Kaggle T4 GPU |
| Windows locks open PNGs | `OSError: [Errno 22]` when re-saving a plot open in a viewer | Write to another folder, or close the viewer |
| Kaggle CLI on Windows | Garbles absolute paths | Call it with relative paths |
| Kaggle may not auto-extract the code zip | Code might be missing | The notebook extracts the zip itself if needed |
| Unstable home internet | The status watcher stopped on a DNS error | Watcher retries on network errors (the Kaggle run itself was unaffected) |

---

## 12. Limitations

1. **One configuration on real data.** Internet only, one step (10 minutes) ahead, plain
   min-max scaling. No hyperparameter search yet.
2. **Single test period,** and it is a holiday period. Results on a normal working week might
   differ.
3. **The scaling compresses low-traffic cells** (tiny normalised losses); see §10.7.
4. **No external information:** no calendar or holiday features, no weather, no events.
5. **Grid orientation** (IDs running south→north) is inferred from the dataset documentation and
   should be checked against `milano-grid.geojson` if exact geography matters.
6. **Point forecasts only,** with no uncertainty estimate.
7. **Compute:** training used a free Kaggle T4; larger models or longer horizons would need more
   GPU time.

---

## 13. Future work and suggested experiments

Each experiment is one Kaggle run (about 2 hours) with a single config change:

| Experiment | Config change | Question it answers |
|---|---|---|
| **Log scaling** | `data.log_transform=true` | Does log1p scaling improve accuracy, especially on low-traffic cells? (most promising) |
| **All five features as input** | `data.input_features=[smsin,smsout,callin,callout,internet]` | Do SMS and call activity help predict internet traffic? |
| **Longer horizons** | `data.t_out=3` (30 min) or `6` (1 h) | How quickly does accuracy decay with the horizon? Per-horizon metrics are built in. |
| **Longer history** | `data.t_in=12` or `18` | Does more past context help? |
| **Huber loss** | `training.loss=huber` | More robust to traffic spikes? |
| **Model size** | `model.hidden_dim=32/128`, `model.num_layers=1/3` | Accuracy vs compute trade-off |
| **Other targets** | `data.target_features=[callin]` | Does the method transfer to calls and SMS? |

Further ideas:

- a "same time yesterday" seasonal baseline
- time-of-day and holiday features
- an encoder–decoder (seq2seq) head for long horizons
- comparison with other models: CNN, LSTM, PredRNN, graph neural networks
- uncertainty estimation

---

## 14. What to show your professor

Present it as a story: **problem → data → method → results → what you learned.**

### Suggested presentation (10–12 slides)

| # | Slide | Content | Material |
|---|---|---|---|
| 1 | Title | project title, your name, guide | |
| 2 | Problem and motivation | why operators need traffic forecasts; the research question | §2 |
| 3 | Dataset | Milan grid, 10-minute steps, 62 days, 5 activities; data challenges found | §3; left column of `heatmaps.png` |
| 4 | The key idea | the city as a video → next-frame prediction | §2 |
| 5 | Preprocessing | aggregation, time conversion, grid mapping, tensor (8928, 5, 100, 100); missing values | §4 |
| 6 | Avoiding leakage | chronological split, train-only scaling, windows inside splits | §4 |
| 7 | Model | ConvLSTM cell equations and architecture diagram | §5 |
| 8 | Training | settings table; T4 GPU; 51 epochs; early stopping | §6, `loss_curve.png` |
| 9 | Results | the metrics table; −9.8% RMSE vs persistence | §9, `metrics.json` |
| 10 | Visual results | heatmaps and cell time series | `heatmaps.png`, `cell_timeseries.png` |
| 11 | Analysis | why beating persistence matters; val < train explained; where errors are | §10 |
| 12 | Limitations and future work | §12–13; the experiment table | §12, §13 |

### Live demo (optional, impressive)

1. Open the **Kaggle notebook**. It shows the full real-data run with outputs: data detection,
   preprocessing log, epoch-by-epoch training, metrics tables and plots.
2. Open the **GitHub repository**. It shows clean structure, documentation and reproducibility.
3. Show `configs/default.yaml` and explain that every hyperparameter is configurable with no
   hard-coded paths.

### Points to emphasise

- A **real, full-scale dataset**: 62 days, 10,000 cells, 20.8 GB raw.
- A **clean evaluation**: a chronological split, train-only normalisation, and comparison
  against two baselines.
- **Clear results:** about 10% better than the strongest baseline.
- **Engineering quality:** tested pipeline, configs, logging, checkpoints, documentation.

---

## 15. Likely questions, with answers

**Q1. How does a ConvLSTM cell work?**
Like an LSTM, it keeps a cell state (long-term memory) and a hidden state (output), controlled
by gates. The input gate controls what to add, the forget gate what to erase, and the output
gate what to expose. The candidate g is the new information. The difference is that the states
are 2-D feature maps and the gates are computed with convolutions, so each location's update
depends on its neighbourhood. That captures spatial and temporal patterns together.

**Q2. Why not a normal LSTM or a CNN?**
A CNN sees one frame (or stacked frames) but has no memory of how the sequence evolves. A
normal LSTM would have to flatten the 10,000 cells into a vector, losing the spatial layout,
and it would need an enormous number of parameters. A ConvLSTM keeps the grid structure and
shares weights across space.

**Q3. Why split chronologically rather than randomly?**
With a random split, the training set would contain time steps right before and after each
test step. The model could then "interpolate" from the future, which inflates the results. A
chronological split mirrors real use: train on the past, predict the future.

**Q4. Why fit normalisation on the training data only?**
Using test-period minimum and maximum values would leak information about the future into
training. With training-only statistics, the test data is truly unseen.

**Q5. Why compare with persistence? Why is it hard to beat?**
Traffic changes little in 10 minutes, so "the next frame equals the last frame" is already very
accurate. Any model must beat it to be useful. The ConvLSTM is 9.8% better on RMSE.

**Q6. How did you handle missing values?**
In this dataset a missing record means there was no activity, so zero-filling is the correct
interpretation. Forward-filling would invent traffic that never happened. Forward-fill is
available as an option, and cells with no training activity would be masked (none existed).

**Q7. Why is MAPE computed only where the true value is above 1?**
MAPE divides by the true value. Near zero it explodes and gets dominated by meaningless tiny
values, so cells with true value ≤ 1 are excluded from MAPE (but not from RMSE or MAE).

**Q8. Validation loss is lower than training loss. Is something wrong?**
No. The validation period (mid to late December) has lower overall traffic, so the errors are
smaller in absolute terms. The split is strictly chronological with no overlap, so there is no
leakage.

**Q9. Why does the test period matter?**
The test period is Christmas and New Year, which behaves differently from the November training
data. The model beating the baselines there suggests that it generalises.

**Q10. What does the prediction head do?**
A 1×1 convolution turns the 64 hidden feature maps into a traffic map. Two 3×3 convolutions
then add a residual correction that sharpens spatial detail.

**Q11. How long did training take, and on what?**
On a Kaggle Tesla T4 GPU: about 147 seconds per epoch and 51 epochs, so about 2 hours 5
minutes. Preprocessing the 62 days took about 11 minutes.

**Q12. How would you improve the results?**
Log scaling (the most promising), adding SMS and call features as inputs, time-of-day and
holiday features, longer input history, hyperparameter tuning, and an encoder–decoder for
longer horizons (§13).

**Q13. What were the hardest problems?**
Understanding the raw data (country-code rows, sparsity, UTC times) and getting the full data
(the Dataverse guestbook; solved with a verified Kaggle mirror). Then fitting 20.8 GB of data
and full-grid training into Kaggle's limits (memory-mapped tensors, a training time budget),
and designing a leak-free evaluation.

**Q14. Is the code reusable?**
Yes. All settings are in YAML configs with command-line overrides. It works with any subset of
the five channels, any horizon, sub-grid crops, and both raw-file formats, and the same approach
applies to any "grid over time" problem (weather radar, road traffic, pollution).

> **Preparation tip:** read `src/model.py` (the ConvLSTM cell) and `src/dataset.py` (the
> preprocessing) until you can explain every step in your own words. Professors usually ask
> you to walk through the code.

---

## 16. How to reproduce everything

**Locally (no download, CPU):**

```bash
pip install -r requirements.txt
python scripts/make_synthetic_data.py --out data/synthetic/raw --days 3
python -m src.dataset  --config configs/smoke.yaml
python -m src.train    --config configs/smoke.yaml
python -m src.evaluate --config configs/smoke.yaml
```

**Full real-data run on Kaggle:**

1. Create a notebook with a GPU accelerator.
2. Attach two datasets: `dkgmgo/telecom-italia-milan` (the data) and
   `hasinisaimaram/convlstm-milano-code` (the project code).
3. Run `kaggle/convlstm_milano_kaggle.ipynb` top to bottom. It finds the data, preprocesses,
   trains, evaluates, and shows the metrics and plots.

**With any machine that has the real data in `data/raw/`:**

```bash
python -m src.dataset  --config configs/default.yaml
python -m src.train    --config configs/default.yaml
python -m src.evaluate --config configs/default.yaml
```

Experiments are run by adding `--set key=value`, for example
`--set data.log_transform=true paths.output_dir=outputs/log_scaling`.

---

## 17. Project timeline

| Date | Milestone |
|---|---|
| 2026-10-28 | Project designed and implemented: preprocessing, ConvLSTM model, training, evaluation, configs |
| 2026-10-28 | Verified end to end on synthetic data; ablation study (residual mode, log scaling) |
| 2026-10-29 | Plots reviewed and improved; full README written |
| 2026-11-03 | Plain-language project overview written |
| 2026-11-03 | Kaggle support added: notebook, Kaggle config, gzip input, training time budget, Dataverse downloader |
| 2026-11-03 | Real data located on Kaggle and verified byte for byte against Harvard Dataverse |
| 2026-11-03 | Full real-data training on a Kaggle Tesla T4 (12:26–14:51); ConvLSTM beats both baselines |
| 2026-11-03 | Code and results published on GitHub (public repository) |

---

## 18. References and data citation

- **Dataset:** Telecom Italia (2015). *Telecommunications - SMS, Call, Internet - MI.* Harvard
  Dataverse. doi:10.7910/DVN/EGZHFV.
- **Dataset paper:** Barlacchi, G. et al. (2015). *A multi-source dataset of urban life in the
  city of Milan and the Province of Trentino.* Scientific Data 2, 150055.
  doi:10.1038/sdata.2015.55.
- **ConvLSTM:** Shi, X. et al. (2015). *Convolutional LSTM Network: A Machine Learning Approach
  for Precipitation Nowcasting.* Advances in Neural Information Processing Systems (NeurIPS).
- **LSTM:** Hochreiter, S. and Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural
  Computation 9(8).
- **Adam:** Kingma, D. P. and Ba, J. (2015). *Adam: A Method for Stochastic Optimization.* ICLR.
- **Data access used:** Kaggle dataset `dkgmgo/telecom-italia-milan` (a mirror of the Dataverse
  files, verified identical).
