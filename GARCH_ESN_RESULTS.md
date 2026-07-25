# GARCH & ESN benchmarks (same dataset/split as HAR.py / QRC_test_1.py)

**Data:** `final_qrc_dataset.csv`, 7 tickers (SPY, JPM, XOM, AAPL, MSFT, NVDA, UNH),
daily, single `split` column (train 2005–~2020, test ~2020–2025, same split
already used by `HAR.py` and `QRC_test_1.py`). Target `y_next` = next-day
log realized variance. QLIKE = mean over test rows (same formula as HAR.py).

## Pooled results (all 7 tickers, all test days)

| Model | MSE | RMSE | MAE | QLIKE | Hit rate |
|---|---|---|---|---|---|
| HAR (already in repo) | — | — | — | 0.396 (as reported) | — |
| **GARCH(1,1)** | 1.2537 | 1.1197 | 0.9241 | **0.4331** | 0.4913 |
| **ESN (N=50)** | 0.9576 | 0.9786 | 0.7655 | **0.8306** | 0.4830 |

- **GARCH(1,1)**: fit once per ticker on the train block (on daily
  `log_return`, not `y` — GARCH models return variance directly), then rolled
  forward through the test block as a true one-step-ahead forecast using the
  fitted recursion + actual realized returns (no refitting per day, no
  look-ahead). Slightly worse than HAR (0.433 vs 0.396) — in line with GARCH
  usually being a weaker RV predictor than HAR-type models in this literature.
- **ESN (N=50)**: fixed random reservoir (never trained) + ridge readout
  fit once on train states, K=3 memory depth (same lag convention as
  HAR's `y_d`/`y_w`/`y_m`: features exclude the current day's own `y`).
  Clearly worse than GARCH/HAR here (0.83) — two tickers (SPY, XOM) have a
  couple of outlier test days that blow up their QLIKE; see per-ticker table.
  **Caveat:** reservoirpy's `Ridge.fit` threw `RuntimeWarning: overflow/
  invalid value encountered in matmul` during fitting. Final predictions
  are finite and in a sane range (checked: no NaN/Inf, distribution close to
  actual `y`), but this warrants a second look — possibly reservoir
  hyperparameters (`sr`, `lr`, `input_scaling`) need retuning for this
  daily/7-ticker data (values live around y≈-9, a very different scale from
  the paper's monthly data ≈-0.6 to -0.8 that these hyperparameters were
  copied from).

## Per-ticker QLIKE

| Ticker | GARCH(1,1) | ESN (N=50) |
|---|---|---|
| AAPL | 0.4544 | 0.6459 |
| JPM | 0.4025 | 0.6287 |
| MSFT | 0.4302 | 0.6513 |
| NVDA | 0.4800 | 0.6869 |
| SPY | 0.5260 | 1.2894 |
| UNH | 0.4132 | 0.6742 |
| XOM | 0.3253 | 1.2380 |

## Files
- `GARCH.py`, `ESN.py` — scripts (same conventions as `HAR.py`)
- `garch_test_predictions.csv`, `esn_test_predictions.csv` — per-row predictions
- `GARCH_model_SPY.png`, `ESN_model_SPY.png` — predicted vs actual (SPY test set)

## Not done here
- Rolling/walk-forward retraining (HAR.py/QRC_test_1.py use a single
  train/test split, not the paper's per-step rolling window — matched that
  here for consistency).
- LSTM: not run against this dataset (only ran it earlier against the
  *paper's* `Data.CSV`, which is a different dataset — see note below).

## Important note on scope
This is a separate, corrected pass. Earlier in this session GARCH/ESN/LSTM
were benchmarked against `Data.CSV` from the original paper's repo
(`LeeQY1996/...`), which is monthly S&P 500 data — **not** this project's
actual dataset. Those earlier results (in `Quantum-Reservoir-Computing/
benchmark_results/`) are a paper-reproduction/validation exercise, not
directly comparable to this team's `final_qrc_dataset.csv`-based QRC1/QRC2
numbers. This file supersedes those for anything about *this* project.
