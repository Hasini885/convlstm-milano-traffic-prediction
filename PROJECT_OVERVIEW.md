# Project Overview: Forecasting Milan's Mobile Traffic with ConvLSTM

A plain-language explanation of what this project does and why. For the full write-up see
[PROJECT_REPORT.md](PROJECT_REPORT.md); for setup, commands and the technical reference see
[README.md](README.md).

## The problem

Mobile networks have to put capacity where people will use it. If an operator knows that in 10
minutes there will be a surge of internet use around the train station, it can shift bandwidth,
power up equipment or balance load ahead of time. If it only reacts after the surge starts,
users get slow connections and dropped calls.

The project asks one question: **given the traffic across a city over the last hour, what will
the traffic be in each part of the city over the next 10 minutes?**

## The data

Telecom Italia released a public dataset for Milan from November to December 2013:

- **The city is split into a 100×100 grid**, so 10,000 squares of about 235 m each. Each square
  has an ID from 1 to 10,000.
- **Every 10 minutes**, each square has five activity values: SMS received, SMS sent, calls
  received, calls made, and internet use. That's 144 snapshots a day for 62 days, about 8,900
  in total.
- The numbers are anonymised activity levels, not exact counts. Only the patterns matter.

The raw files are messy. They come as one file per day with one row per square, time and caller
country. A row only exists when there was activity, and the timestamps are in UTC. Turning this
into something a model can learn from is a large part of the project.

## The key idea: the city as a video

Each 10-minute snapshot can be drawn as a 100×100 image, where every pixel is one square and its
brightness is the traffic. Stack the snapshots in time and you get a **video of the city's
activity**. Busy areas glow during the day, shift toward residential areas in the evening, and
fade at night.

Forecasting traffic then becomes **predicting the next frame of a video** from the last few
frames. Two things matter at once:

- **Space:** a square's traffic is related to its neighbours. A crowd leaving a stadium spills
  into nearby streets.
- **Time:** traffic follows trends and daily rhythms. If it's been rising for an hour, it will
  probably keep rising.

## Why ConvLSTM

There are two standard tools, each covering one of those needs:

- **Convolutional networks (CNNs)** handle space. They look at small neighbourhoods of pixels and
  learn local patterns.
- **LSTMs** handle time. They keep a memory that they update step by step, and learn what to
  remember and what to forget.

A **ConvLSTM** combines them. It is an LSTM whose memory is itself an image, and it updates that
memory with convolutions rather than ordinary matrix multiplications. At every time step it
decides, for each location, what to keep, what to forget and what to output, taking the
neighbouring squares into account. It was introduced in 2015 for rainfall forecasting from radar
images, which is the same kind of problem.

## How the model works

1. **Input:** the last 6 snapshots (1 hour) for each square, plus four "clock" values that tell
   the model the time of day and the day of the week.
2. **Two ConvLSTM layers** read the frames in order and build up a summary: 64 learned feature
   maps per square, describing what is happening there and nearby.
3. **A prediction head** turns that summary into the forecast: the traffic for every square 10
   minutes ahead.
4. **Output:** a predicted 100×100 traffic map.

The final model has about 474,000 parameters, which is small by modern standards.

## The pipeline, step by step

1. **Preprocessing** reads each daily file and adds up the per-country rows. It converts the
   times to Milan local time and places every value at its square's position in the grid and its
   10-minute slot. The result is one large array of shape (time, feature, 100, 100), about
   1.8 GB for all five features. Missing values become zero, because a missing value in this
   dataset means no activity.
2. **Splitting** is done by time: the first 70% of the period is for training, the next 15% for
   validation and the last 15% for testing. The model never sees the future while it learns,
   which matches how it would be used in real life.
3. **Scaling** first takes the logarithm of the traffic, then standardises it (mean 0, spread 1),
   using statistics from the training period only, so nothing about the test period leaks in.
   The logarithm matters because a few city-centre squares are hundreds of times busier than
   the rest; on a log scale, a 10% error counts the same in a quiet suburb and in the centre.
4. **Windowing** cuts the timeline into examples of "1 hour of input → the next 10 minutes" by
   sliding along it one step at a time.
5. **Training** shows the model thousands of these examples. It adjusts its weights to reduce
   the squared error between prediction and reality, and lowers the learning rate when progress
   stalls. It stops early when validation error stops improving noticeably, and saves the best
   version.
6. **Evaluation** turns the test predictions back into original units and measures the error.

## How you judge it: baselines

A forecasting model is only useful if it beats simple rules, so the project compares it against
four:

- **Persistence:** "the next 10 minutes will look like the last 10." This is surprisingly hard
  to beat for short-term traffic.
- **Window mean:** "the next 10 minutes will be the average of the last hour."
- **Same time yesterday** and **same time last week:** the classic seasonal forecasts.

Three error measures are reported:

- **RMSE:** average error, penalising big misses more heavily (dominated by the busiest squares).
- **MAE:** the plain average error.
- **MAPE:** the error as a percentage (every square counts equally).

The project also produces heatmaps (actual vs predicted vs error maps of the city), a
time-series plot for one busy square, and a training curve that shows the model's error as a
fraction of persistence's error.

## How the project developed: three runs

The model was trained three times on the **full real Milano data** (all 62 days, the whole
100×100 grid) on a free Kaggle Tesla T4 GPU, about 1.5–2 hours each.

1. **Run 1** used simple 0–1 scaling. It already beat the baselines, but a review found
   drawbacks: the scaling squashed almost all values close to zero (the training loss was about
   0.000006, so the model got very little signal), the model had no idea of the time of day,
   there were no seasonal baselines, early stopping kept going on meaningless tiny improvements,
   and the training curve was hard to interpret.
2. **Run 2** fixed these: standardised scaling, clock inputs, seasonal baselines, a stricter
   early-stopping rule, resumable training, and a few bugs. Training became better behaved, but
   test accuracy was about the same as run 1 (1.5% worse on RMSE).
3. **Run 3** added the log transform. This is the **best model**.

## Where it stands

On the test period (Christmas and New Year 2013):

| Model | RMSE | MAE | MAPE |
|---|---|---|---|
| **ConvLSTM, run 3 (final)** | **8.47** | **3.78** | **10.59%** |
| ConvLSTM, run 1 | 8.47 | 3.91 | 11.99% |
| Persistence | 9.39 | 4.33 | 12.56% |
| Window mean | 10.37 | 4.44 | 12.13% |
| Same time yesterday | 27.45 | 9.30 | 24.78% |
| Same time last week | 57.01 | 17.40 | 40.56% |

- The final model is **about 10% better than persistence on RMSE, 13% on MAE and 16% on MAPE**.
- Compared with the first run, the percentage error dropped by **11.7%** (11.99% → 10.59%)
  while RMSE stayed the same: the log scale helped the many quieter squares most.
- Measured against persistence, the training and validation errors are nearly identical
  (0.69× and 0.70×), so the model is not overfitting.
- "Same time yesterday / last week" are poor forecasts here: 10-minute traffic is noisy, and the
  holiday fortnight differs from the days before it.

The results, plots and logs are in `outputs/kaggle/` (run 1), `outputs/kaggle_run2/` and
`outputs/kaggle_run3/`; `PROJECT_REPORT.md` explains them in detail.

## Why it's a good project

It covers the full workflow of an applied deep-learning project:

- turning messy real-world data into clean tensors
- splitting by time to avoid leakage
- building a spatio-temporal model
- disciplined training with early stopping, checkpoints and resuming
- honest evaluation against simple and seasonal baselines
- reviewing your own first version, fixing its weaknesses, and reporting what helped and what
  did not

The same approach works for other "grid over time" problems, such as weather radar, traffic
congestion, bike-share demand or air pollution maps.
