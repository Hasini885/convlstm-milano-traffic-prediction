# Mock Viva: Likely Questions and Answers

Practice questions for presenting the **ConvLSTM Milano traffic forecasting** project. They are
grouped by topic and go from easy to hard. Every answer is based on this project's code, configs
and real results (`outputs/kaggle/`).

**How to use it:** cover the answer, say yours out loud, then check. The ★ questions are the
ones examiners ask most often. Section 10 has the hard, critical questions; prepare those best.

---

## Key numbers to remember

| What | Value |
|---|---|
| Grid | 100 × 100 = 10,000 cells, about 235 m each |
| Time step | 10 minutes, 144 per day |
| Period | 62 days, 2013-11-01 to 2014-01-01, T = 8,928 steps |
| Channels | smsin, smsout, callin, callout, internet (model uses **internet** only) |
| Tensor | (8928, 5, 100, 100) float32, about 1.8 GB; raw data about 20.8 GB |
| Input → output | 6 frames (1 hour) → 1 frame (10 minutes ahead) |
| Split | 70 / 15 / 15 chronological → 6,243 / 1,333 / 1,334 windows |
| Test period | 2013-12-23 to 2014-01-01 (Christmas and New Year) |
| Model | 2 ConvLSTM layers, hidden 64, 3×3 kernels, refinement head, **464,290 parameters** |
| Training | Tesla T4, about 147 s/epoch, 51 epochs (best 41), about 2 h 5 min |
| **ConvLSTM** | **RMSE 8.47, MAE 3.91, MAPE 11.99%** |
| Persistence | RMSE 9.39, MAE 4.33, MAPE 12.56% |
| Window mean | RMSE 10.37, MAE 4.44, MAPE 12.13% |
| Improvement vs persistence | −9.8% RMSE, −9.9% MAE, −4.6% MAPE |

---

## 1. Problem and motivation

**★ Q1. Explain your project in 30 seconds.**
I predict mobile internet traffic for every part of Milan 10 minutes ahead. The city is a 100×100
grid, and every 10 minutes gives one traffic "image", so the data is a video of the city. I train
a ConvLSTM, which handles space and time together, on 62 days of real Telecom Italia data. On the
test period it is about 10% more accurate than the strongest simple baseline, "repeat the last
frame".

**Q2. Why does anyone need this forecast?**
Operators can act before congestion happens: move capacity where it will be needed, balance load
between cells, and switch off idle equipment at night to save energy. Reacting after a surge has
started is too late for users.

**Q3. Why is it a spatio-temporal problem and not just a time series?**
A cell's traffic depends on its own history (time) and on its neighbours (space): crowds move
between areas, and busy zones spread. Forecasting each cell on its own throws away the
neighbour information.

**Q4. What exactly is your research question?**
Given the last hour of traffic on the grid, can a spatio-temporal deep-learning model predict the
next 10 minutes better than naive rules such as persistence and the hourly mean?

---

## 2. Dataset

**★ Q5. Describe the dataset.**
Telecom Italia Big Data Challenge, "SMS, Call, Internet - MI" (Harvard Dataverse,
doi:10.7910/DVN/EGZHFV). Milan is split into 10,000 square cells. Each cell has five activity
values every 10 minutes for November and December 2013. One file per day, tab-separated, no
header.

**Q6. Are the values real counts?**
No. They are anonymised, scaled activity levels. The patterns are real, but the units are not
bytes or number of calls, so the metrics are in "dataset units".

**★ Q7. What problems did you find in the raw data?**
1. One row per (cell, time, **country code**), so rows must be summed per cell.
2. **Sparse**: a row only exists when there was activity, and empty fields mean zero.
3. Timestamps are **UTC** (epoch ms); Milan is UTC+1 in winter.
4. Size: about 300–360 MB per day, 20.8 GB in total.
5. Access: Dataverse needs a guestbook form, which blocks automated downloads.

**Q8. How did you get the data, and how do you know it's correct?**
I used the Kaggle mirror `dkgmgo/telecom-italia-milan`. I compared the sizes of all 62 files with
the official Dataverse file list and they match byte for byte. Other mirrors were incomplete
(for example only 7 days) and were not used.

**Q9. Why use only the internet channel?**
It is the largest and most important traffic type for capacity planning, and one channel keeps the
first experiment simple. All five channels are stored in the tensor, so using them as input is a
one-line config change (`data.input_features`). That is listed as future work.

**Q10. Were any cells inactive?**
No. All 10,000 cells had internet activity in the training period, so the inactive-cell mask
removed nothing on the real data. It matters for other channels or crops.

---

## 3. Preprocessing

**★ Q11. Walk me through the preprocessing.**
For each daily file (`preprocess()` in `src/dataset.py`):
1. Read it, detect the format, convert UTC → Europe/Rome local time.
2. Sum over country codes.
3. Map CellID to grid position: `row = (id−1) // 100`, `col = (id−1) % 100`.
4. Compute an absolute time index: `t = (time − origin) / 10 min`, where origin is local midnight
   of the first day.
5. Write the values into a memory-mapped float32 array of shape (T, C, 100, 100).
6. Fill missing values with zero, and save `meta.json`.

**★ Q12. How did you handle missing values, and why that way?**
Zero-fill. In this dataset a missing row or field means *no activity*, not "unknown".
Forward-filling would invent traffic that never happened. Forward-fill is still available as an
option (`fill_method: ffill_zero`).

**Q13. Why convert to local time? Would it change the results?**
Human behaviour follows local clocks (commute, lunch, evening). Without the conversion every day
would be shifted by an hour, so "midnight" would actually be 1 a.m. The model could still learn,
but days, plots and any time-of-day features would be misaligned. No daylight-saving change falls
inside Nov–Dec 2013.

**Q14. Why an absolute time index instead of grouping by file?**
After the UTC → local shift, some records in a day's file belong to the neighbouring day. With an
absolute index they still land in the correct slot.

**Q15. Why a memory-mapped `.npy` file?**
The full tensor is about 1.8 GB. A memmap lets the code read only the windows it needs, so the
whole tensor never has to sit in RAM. Each DataLoader worker opens it lazily, read-only.

**Q16. How did you check the preprocessing is correct?**
A single (time, cell) value matched the raw file exactly (6.6311). A full day's total matched a
direct pandas groupby. The first timestamp, 2013-10-31 23:00 UTC, mapped to time index 0
(2013-11-01 00:00 Milan). A gzip input gave an identical tensor.

**Q17. What does the grid orientation look like?**
IDs run west → east, then south → north, so row 0 is the southern edge. Plots use
`origin="lower"` to put north at the top. (This comes from the dataset documentation; it could
be double-checked against `milano-grid.geojson`.)

---

## 4. Splitting, normalisation, windows

**★ Q18. Why a chronological split and not a random one?**
A random split puts training time steps right before and after each test step, so the model
effectively interpolates from the future and the score is inflated. Chronological splitting
matches real use: learn from the past, predict the future. Every window also lies fully inside
one split, so no target crosses a boundary.

**★ Q19. Why fit the normalisation on the training data only?**
The minimum and maximum of the test period are information from the future. Fitting on training
data only keeps the test data truly unseen. The fitted values are saved in the checkpoint and
reused at evaluation.

**Q20. What normalisation did you use?**
Per-channel min-max scaling to [0, 1]. An optional `log1p` before the scaling is available
(`data.log_transform`) but was off in the final run.

**Q21. How are the training examples built?**
A sliding window: X = frames t … t+5 (1 hour), Y = frame t+6. Stride 1. Batch shape is
X (16, 6, 1, 100, 100) and Y (16, 1, 1, 100, 100).

**Q22. Why 6 input frames?**
One hour is enough to see the recent trend at a 10-minute resolution, and it keeps memory and
training time manageable on the full grid. A longer history (12 or 18 frames) is a planned
experiment.

**Q23. How many windows does each split have, and why fewer than the steps?**
6,243 / 1,333 / 1,334. Each split loses `t_in + t_out − 1 = 6` steps at its end, because a window
needs 7 consecutive frames inside the split (6,249 − 6 = 6,243).

---

## 5. The model

**★ Q24. How does a ConvLSTM cell work? Write the equations.**
The input X_t and previous hidden state H_{t−1} are concatenated and passed through **one**
convolution that outputs all four gates:

- i = σ(·) input gate, f = σ(·) forget gate, o = σ(·) output gate, g = tanh(·) candidate
- C_t = f ⊙ C_{t−1} + i ⊙ g
- H_t = o ⊙ tanh(C_t)

It is an LSTM where the states are 2-D feature maps (64 × 100 × 100) and the matrix
multiplications are replaced by convolutions. So each location's memory update depends on its
neighbourhood. (Shi et al., 2015.)

**★ Q25. Why ConvLSTM instead of a plain LSTM or a CNN?**
- A plain LSTM needs the grid flattened into a 10,000-long vector. That loses the spatial layout
  and needs a huge number of parameters.
- A CNN is good at space but has no memory of how the sequence develops (unless you stack frames
  as channels, which treats time crudely).
- A ConvLSTM keeps the grid, shares weights across locations, and models time with gated memory.

**Q26. What does each gate do, intuitively?**
Forget gate: how much old memory to keep. Input gate: how much new information to write.
Candidate g: the new information itself. Output gate: how much of the memory to expose as the
hidden state.

**Q27. Why one convolution with 4 × hidden outputs instead of four convolutions?**
It is mathematically the same as four separate convolutions on the same input, but faster: one
large GPU operation instead of four small ones. The output is then split into i, f, o, g.

**Q28. Why "same" padding and an odd kernel?**
Same padding keeps the output at 100×100, so hidden states can be added and passed step to step.
It needs an odd kernel to pad symmetrically; the code raises an error for even sizes.

**Q29. Why initialise the forget-gate bias to 1?**
σ(1) ≈ 0.73, so at the start the cell mostly keeps its memory. That helps gradients flow through
time early in training. It is a standard trick (Jozefowicz et al., 2015).

**Q30. What is the receptive field of the model?**
Each 3×3 convolution spreads information by one cell. With the recurrence over 6 steps in each
of 2 layers, plus the 3×3 refinement convolutions, a prediction can see roughly a dozen cells in
each direction, a few kilometres. Enough for local crowd movement, not for the whole city.

**★ Q31. What does the prediction head do?**
A 1×1 convolution maps the 64 hidden channels of the last hidden state to a coarse forecast.
Then a small residual block (3×3 conv → ReLU → 3×3 conv, 32 channels) sees the coarse forecast
and the hidden state together and adds a correction to sharpen spatial detail.

**Q32. How would you predict more than one step ahead?**
The head is **direct multi-horizon**: set `t_out = 3` and it outputs all 3 frames at once.
There is no autoregressive feeding back of predictions, so errors don't accumulate step by step.

**Q33. How did you get 464,290 parameters?**
- ConvLSTM layer 1: (1 + 64) × 256 × 3 × 3 + 256 = 150,016
- ConvLSTM layer 2: (64 + 64) × 256 × 3 × 3 + 256 = 295,168
- Head 1×1: 64 × 1 + 1 = 65
- Refinement: (1 + 64) × 32 × 9 + 32 = 18,752 and 32 × 1 × 9 + 1 = 289
- Total = **464,290**. (256 = 4 gates × 64 hidden channels.)

**Q34. What is the "residual" option and why is it off?**
It predicts the *change* from the last frame (ŷ = x_last + head(h)). In early tests it collapsed
to the persistence solution (the same error as persistence), because with
min-max scaling the changes are tiny and predicting zero change is an easy local minimum. So it
is off by default.

---

## 6. Training

**★ Q35. Describe the training setup.**
Masked MSE loss on the normalised scale, Adam (lr 1e-3), ReduceLROnPlateau (halve after 5 epochs
with no improvement, minimum 1e-6), early stopping with patience 10, batch size 16, gradient
clipping at norm 1.0, mixed precision on GPU, seed 42. The best model by validation loss is saved.

**Q36. Why MSE?**
It is the standard regression loss, and it matches RMSE, the main metric. It punishes large errors
more, which matters at busy cells where congestion happens. Huber loss is available as an option
for robustness to spikes.

**Q37. What is the "masked" loss?**
Errors are averaged only over cells marked active (cells with some training-period activity), so
always-zero cells don't dilute the loss. On the real data all cells were active, so it was an
ordinary MSE.

**Q38. Why Adam? Why lr 1e-3?**
Adam adapts the step size per parameter and works well with little tuning; 1e-3 is its standard
default. The scheduler lowered it when progress stalled: halved at epochs 19, 33 and 47, ending at
1.25e-4.

**Q39. Why gradient clipping?**
Recurrent models can have exploding gradients through time. Clipping the gradient norm to 1.0
keeps the updates stable.

**Q40. What is mixed precision and why use it?**
Most operations run in float16 (autocast) and a GradScaler scales the loss so small gradients
don't underflow. It roughly halves memory and speeds up training on the T4, which matters for a
full 100×100 grid with 64 hidden channels.

**★ Q41. How did you prevent overfitting?**
Early stopping on validation loss (stopped at epoch 51, best at 41), a moderately sized model
(464k parameters for about 62 million target values in training), and a separate test set that
was only used once at the end.

**Q42. How long did training take, and on what hardware?**
Kaggle Tesla T4 GPU: about 147 s per epoch, 51 epochs, about 2 h 5 min of training; the whole
notebook (preprocessing 11 min + training + evaluation) about 2 h 18 min. The laptop has no GPU,
so the full-grid run was done on Kaggle.

**Q43. What is `max_hours` for?**
Kaggle sessions stop after 12 hours. The 8.5 h budget makes training stop cleanly in time, so the
best checkpoint is kept and evaluation always runs.

---

## 7. Evaluation and results

**★ Q44. What are your results?**
On the test period, ConvLSTM RMSE 8.47, MAE 3.91, MAPE 11.99%. Persistence: 9.39 / 4.33 / 12.56%.
Window mean: 10.37 / 4.44 / 12.13%. The ConvLSTM beats both baselines on all three metrics: about
10% lower RMSE and MAE than persistence.

**★ Q45. What is persistence and why is it hard to beat?**
It predicts that the next frame equals the last observed frame. Traffic changes little in 10
minutes, so it is already very accurate. Any useful model has to beat it, which is why it is the
main baseline.

**Q46. Why is persistence better than the window mean on real data?**
Real traffic is smooth over 10 minutes but drifts over an hour. The hourly average lags behind
rising or falling trends.

**Q47. Explain RMSE, MAE and MAPE.**
- RMSE: square root of the mean squared error; punishes large errors more.
- MAE: mean absolute error; the typical size of an error.
- MAPE: mean absolute error as a percentage of the true value; scale-free.

All are computed in original units, after denormalising and clamping predictions to ≥ 0.

**★ Q48. Why is MAPE only computed where the true value is above 1?**
MAPE divides by the true value. Near zero it explodes and tiny meaningless values dominate it.
Cells with true value ≤ 1 are excluded from MAPE only; RMSE and MAE still use every cell.

**Q49. Why is the MAPE improvement (−4.6%) smaller than the RMSE improvement (−9.8%)?**
RMSE is dominated by busy cells with large absolute values, where the ConvLSTM helps most. MAPE
weights every cell equally in relative terms, including quieter cells where relative errors are
noisier and the gain is smaller.

**★ Q50. Your validation loss is lower than your training loss. Is that a bug or leakage?**
Neither. The split is strictly chronological and scaling uses training statistics only. The
validation period (mid to late December) has lower overall traffic than November, so errors on
the normalised scale are smaller. Lower loss on an easier period is expected.

**Q51. Why are the loss values so small (about 1e-6)?**
Min-max scaling uses the maximum over all cells, and a few very busy cells set that maximum, so
most normalised values are close to 0 and their squared errors are tiny. That is a known weakness,
and the reason log scaling is the top next experiment.

**Q52. Where does the model make the biggest errors?**
In the dense city centre, where traffic is highest (see the right column of `heatmaps.png`).
Absolute errors grow with traffic level; in relative terms the model is fairly consistent.

**Q53. What do your plots show?**
- `heatmaps.png`: actual vs predicted vs absolute error for 3 test times; the spatial structure
  (centre, hotspots) is reproduced.
- `cell_timeseries.png`: a busy cell (CellID 4128) over the whole test period; the forecast follows
  the daily cycle through Christmas and New Year.
- `loss_curve.png`: big drop after epoch 1, then slow improvement; early stopping at 51.

**Q54. How are metrics computed for 13 million predictions without running out of memory?**
`MetricAccumulator` in `src/evaluate.py` keeps running sums (squared error, absolute error,
counts) per horizon and channel, batch by batch, and computes the final metrics at the end. The
test set never has to sit in memory.

---

## 8. Code and engineering

**★ Q55. Walk me through the code structure.**
- `src/dataset.py`: raw parsing, tensor building, normaliser, Dataset and DataLoaders
- `src/model.py`: ConvLSTMCell → ConvLSTM → PredictionHead → ConvLSTMForecaster
- `src/train.py`: masked loss, early stopping, LR scheduling, AMP, checkpointing
- `src/evaluate.py`: streaming metrics, baselines, plots
- `src/utils.py`: config loading with `--set` overrides, seeding, logging
- `configs/`: `default.yaml` and `kaggle.yaml` (inherits `default.yaml` through `_base_`)

**Q56. How do you make the results reproducible?**
A fixed seed (Python, NumPy, PyTorch), the exact config saved as `config_used.json`, the
normaliser stored in the checkpoint, logged training history, pinned package versions, and a
Kaggle notebook that runs the whole pipeline from raw files.

**Q57. What is in the checkpoint?**
Model weights, the config, best epoch and validation loss, the normaliser's min/max values, and
the channel information. Only plain types, so it loads safely with `weights_only=True`.

**Q58. How does the Dataset read a training window?**
`MilanoTrafficDataset.__getitem__` (`src/dataset.py`) opens the tensor as a read-only memmap
(lazily, once per worker), slices the 6 input frames and the next target frame for the chosen
channels, applies the training-set min-max scaling, and returns them as tensors. Only the
training loader is shuffled.

**Q59. How do you change the inputs, target or horizon without editing code?**
Everything is in the YAML config and can be overridden from the command line, for example
`--set data.input_features=[smsin,smsout,callin,callout,internet] data.t_out=3`. All five
channels are already stored in the tensor, so no re-preprocessing is needed.

**Q60. What was the hardest engineering problem?**
Getting and handling the data: the Dataverse guestbook blocked downloads (solved with a verified
Kaggle mirror), 20.8 GB of raw files (memmap, gzip support), the raw-format quirks (country codes,
sparsity, UTC), and fitting full-grid training into Kaggle's time limit.

---

## 9. Limitations and future work

**★ Q61. What are the limitations of your work?**
1. Only one configuration on real data (internet only, 10 minutes ahead, no hyperparameter search).
2. One run, one seed: no variance or confidence intervals.
3. A single test period, and it is the holidays.
4. Min-max scaling compresses quiet cells.
5. No calendar, weather or event features.
6. Point forecasts only, no uncertainty.
7. Only naive baselines, no other deep models.

**★ Q62. How would you improve the results?**
In order: `log1p` scaling; all five channels as input; time-of-day and holiday features; a longer
input history; a "same time yesterday" seasonal baseline; hyperparameter tuning; an
encoder–decoder for longer horizons. Each is one config change and about 2 hours on Kaggle.

**Q63. How would accuracy change for 30 or 60 minutes ahead?**
It would drop, because the future gets less certain. Persistence degrades even faster, so the
ConvLSTM's relative advantage often grows. The code already reports metrics per horizon
(`data.t_out=3` or `6`); it has not been run on the real data yet.

**Q64. Which other models could you compare against?**
ARIMA or a per-cell LSTM (no spatial context), a 3-D CNN, PredRNN / PredRNN++ (improved ConvLSTM
variants), ST-ResNet, graph neural networks (STGCN, DCRNN), and Transformer-based models.

**Q65. Could this work in a real network?**
Yes, in principle: inference for the whole city is one forward pass, well under a second on a
GPU. It would need live data feeds, periodic retraining as patterns change, and monitoring for
unusual events.

---

## 10. Hard and critical questions

**★ Q66. A 10% improvement in RMSE. Is that significant, or just noise?**
It is measured over 1,334 test times × 10,000 cells (about 13 million predictions), and the
ConvLSTM wins on all three metrics against both baselines, so it is not a fluke of a few
points. But it is one training run with one seed. To claim it formally I would run several seeds
and a paired test (for example Diebold–Mariano) on the per-window errors. That is a fair
limitation to admit.

**Q67. Isn't testing on Christmas and New Year unfair?**
It makes the test harder, not easier: the holiday behaviour (less commuting, the New Year spike)
never appeared in training. Beating the baselines there suggests the model generalises. A normal
working week as a second test period would be a good extra check.

**Q68. Your model only uses the last hour. How does it know about daily patterns?**
It doesn't explicitly. It sees the trend within the hour and the spatial layout, which is enough
for 10 minutes ahead. For longer horizons, time-of-day features or a "same time yesterday" input
would help. That is listed as future work.

**Q69. Why not just predict each cell separately with ARIMA or a small LSTM?**
That ignores the neighbours, which carry information (crowds move, busy areas spread), and it
needs 10,000 separate models. The ConvLSTM shares one set of weights across the city and uses the
spatial context. A per-cell baseline would still be a useful comparison.

**Q70. Did you tune hyperparameters on the test set?**
No. The defaults were chosen beforehand, model selection used only the
validation loss (early stopping, best checkpoint), and the test set was evaluated once at the end.

**Q71. Why is MAPE about 12% when RMSE is "only" 8.5?**
They measure different things. RMSE is in absolute units, and most cells have low traffic, so
typical absolute errors are small. MAPE is relative, and at quieter cells (true values just above
1) an absolute error of 1 or 2 is already a large percentage.

**Q72. Could the model just be copying the last frame?**
No. If it were, its errors would equal persistence's (RMSE 9.39); they are about 10% lower. That
collapse to persistence was actually seen in early tests of the residual mode, which is why
that mode is off.

**Q73. What would you do differently if you started again?**
Use log scaling from the start, add time-of-day features, keep a second test period outside the
holidays, plan several seeds for every experiment, and compare against at least one more deep
model.

**Q74. What did you learn from this project?**
Most of the work in applied deep learning is the data: understanding its quirks, cleaning it,
and building a leak-free evaluation. A strong naive baseline is the honest yardstick. And a clean,
config-driven pipeline makes every new experiment a one-line change.

---

## Before the viva

- Be able to draw the architecture and write the 5 ConvLSTM equations from memory.
- Open `src/model.py` (`ConvLSTMCell.forward`) and `src/dataset.py` (`preprocess`,
  `MilanoTrafficDataset.__getitem__`) and explain each line in your own words.
- Know the results table and the "−9.8% vs persistence" number by heart.
- Have the plots ready: `outputs/kaggle/heatmaps.png`, `cell_timeseries.png`, `loss_curve.png`.
- When you don't know something, say so and say how you would find out. Examiners value that more
  than a guess.
