"""
QRC_test_2.py
-------------
Direct QRC model only -- no HAR anywhere:

    QRC(Open_t, High_t, Close_t, Volume_t, logRV_t)  ->  logRV_{t+1}

Pipeline:
    1. Load final_qrc_dataset.csv
    2. Build y_next target; split by the `split` column
    3. Scale the five raw feature columns (train-only robust z-score + tanh)
    4. Train reservoir + ridge readout; evaluate one-step-ahead on test
    5. Report vs a persistence baseline (delete that one line if unwanted)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.metrics import mean_absolute_error, mean_squared_error

from QRC_model_2 import QRC_Model_2

# =====================================================================
# Config
# =====================================================================
CSV_PATH = "final_qrc_dataset.csv"
DT = 0.1
N_TROTTER = 2
FEEDBACK_SCALE = 1.0
F_BS = [1.0]
RIDGE_PARAM = 1.0e-6
SEED = 0
FIG_DIR = "figs"

# One feature per qubit; the ORDER here is the qubit assignment.
# This list is the single source of truth -- the scaler is built from it.
FEATURE_NAMES = [
    "Open",       # raw price level: non-stationary; watch report() on test
    "High",
    "Close",
    "Volume",     # raw volume: heavy right skew; log1p would encode better
    "y",          # logRV_t
]
NUM_QUBITS = len(FEATURE_NAMES)


# =====================================================================
# Feature scaling
# =====================================================================
class TrainOnlyScaler:
    """Robust z-score (median/IQR) fit on TRAIN rows only, then tanh.

    Median/IQR because financial features have fat tails; tanh bounds the
    encoding to (-1, 1) as required by the RY angle embedding.
    """

    def __init__(self, feature_names, squash=1.0):
        self.feature_names = list(feature_names)
        self.squash = squash
        self.center_ = None
        self.scale_ = None

    def fit(self, df: pd.DataFrame):
        X = df[self.feature_names].to_numpy(dtype=float)
        self.center_ = np.nanmedian(X, axis=0)
        q75, q25 = np.nanpercentile(X, [75, 25], axis=0)
        iqr = q75 - q25
        self.scale_ = np.where(iqr > 0, iqr / 1.349, 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.center_ is None:
            raise RuntimeError("call fit() on the training split first")
        X = df[self.feature_names].to_numpy(dtype=float)
        return np.tanh(self.squash * (X - self.center_) / self.scale_)


def build_qrc_tensors_direct(split_df: pd.DataFrame, scaler: TrainOnlyScaler):
    """Per-ticker aligned arrays. The scaler defines which feature columns
    are used and in what order -- no separate feature_names argument.

    Returns:
        V:       (N, T, F) scaled feature vectors, F == len(scaler.feature_names)
        y_next:  (N, T)    target: logRV at row t+1
        dates:   list of N date arrays
        tickers: (N,) ticker names
    """
    V_list, y_list, date_list, tickers = [], [], [], []
    for ticker, sub in split_df.groupby("ticker", sort=True):
        sub = sub.sort_values("date")
        V_list.append(scaler.transform(sub))
        y_list.append(sub["y_next"].to_numpy(dtype=float))
        date_list.append(sub["date"].to_numpy())
        tickers.append(ticker)

    T = min(len(a) for a in y_list)
    if len(set(len(a) for a in y_list)) > 1:
        print(f"[tensors] unbalanced panel, truncating all tickers to T={T}")

    V = np.stack([v[:T] for v in V_list])
    y = np.stack([a[:T] for a in y_list])
    date_list = [d[:T] for d in date_list]
    return V, y, date_list, np.array(tickers)


def report(V: np.ndarray, feature_names):
    """Sanity-check the encoding range. Any feature with std < ~0.1 is a
    qubit carrying almost no information (saturated or constant)."""
    flat = V.reshape(-1, V.shape[-1])
    print(f"{'feature':>12}  {'min':>7}  {'max':>7}  {'mean':>7}  {'std':>7}")
    for i, name in enumerate(feature_names):
        col = flat[:, i]
        print(f"{name:>12}  {col.min():>7.3f}  {col.max():>7.3f}  "
              f"{col.mean():>7.3f}  {col.std():>7.3f}")


# =====================================================================
# 1. Data
# =====================================================================
dff = pd.read_csv(CSV_PATH)
dff["date"] = pd.to_datetime(dff["date"])
dff = dff.sort_values(["ticker", "date"])

dff["y_next"] = dff.groupby("ticker")["y"].shift(-1)
dff = dff.dropna(subset=["y_next", "split"] + FEATURE_NAMES)

train = dff[dff["split"] == "train"].copy()
test = dff[dff["split"] == "test"].copy()

# =====================================================================
# 2. Feature tensors (scaler fit on TRAIN only)
# =====================================================================
scaler = TrainOnlyScaler(feature_names=FEATURE_NAMES).fit(train)

V_tr, y_tr, _, tickers = build_qrc_tensors_direct(train, scaler)
V_te, y_te, dates_te, _ = build_qrc_tensors_direct(test, scaler)

print("Encoding check -- TRAIN split:")
report(V_tr, FEATURE_NAMES)
print("\nEncoding check -- TEST split (watch for saturated price channels):")
report(V_te, FEATURE_NAMES)

# =====================================================================
# 3. Model
# =====================================================================
qrc_model = QRC_Model_2(
    num_qubits=NUM_QUBITS,
    backends=[qml.device("default.qubit", wires=NUM_QUBITS) for _ in F_BS],
    f_bs=F_BS,
    dt=DT,
    n_trotter=N_TROTTER,
    feedback_scale=FEEDBACK_SCALE,
    ridge_param=RIDGE_PARAM,
    seed=SEED,
)

qrc_model.train(V_tr, y_tr)
qrc_model.fit()

# =====================================================================
# 4. Evaluate one-step-ahead on test
# =====================================================================
preds_log = qrc_model.forward_one_shot(V_te, y_te)   # (N, T-1)

rows = []
for i, tick in enumerate(tickers):
    rows.append(pd.DataFrame({
        "date": dates_te[i][1:],
        "ticker": tick,
        "y_true": y_te[i, 1:],
        "y_qrc": preds_log[i],
    }))
results = pd.concat(rows, ignore_index=True)


def evaluate(y_true, y_pred, label=""):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    ratio = np.exp(y_true) / np.exp(y_pred)
    ql = np.mean(ratio - np.log(ratio) - 1)
    print(f"{label:>12s}  MSE: {mse:.4f}  RMSE: {np.sqrt(mse):.4f}  "
          f"MAE: {mae:.4f}  QLIKE: {ql:.5f}")
    return {"mse": mse, "mae": mae, "qlike": ql}


print("\nTest-set results:")
evaluate(results["y_true"], results["y_qrc"], "QRC direct")
# persistence reference: y_hat_{t+1} = y_t (delete this line if unwanted)
evaluate(results["y_true"], y_te[:, :-1].flatten(), "persistence")

# =====================================================================
# 5. Plots
# =====================================================================
os.makedirs(FIG_DIR, exist_ok=True)


def qlike(y_true, y_pred):
    r = np.exp(y_true) / np.exp(y_pred)
    return np.mean(r - np.log(r) - 1)


for tick, gdf in results.groupby("ticker"):
    gdf = gdf.sort_values("date")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(gdf["date"], gdf["y_true"], color="steelblue", alpha=0.55,
            lw=1.0, label="Actual (log RV)")
    ax.plot(gdf["date"], gdf["y_qrc"], color="seagreen", lw=1.2, alpha=0.9,
            label=f"QRC direct  (QLIKE {qlike(gdf.y_true, gdf.y_qrc):.3f})")
    ax.set_ylabel("log RV")
    ax.set_xlabel("Date")
    ax.set_title(f"{tick} -- test set, direct QRC forecast")
    ax.legend(loc="upper left", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/QRC_direct_{tick}.png", dpi=150)
    plt.close(fig)

per_ticker = results.groupby("ticker").apply(
    lambda gdf: pd.Series({"QRC direct": qlike(gdf.y_true, gdf.y_qrc)})
)
print("\nPer-ticker QLIKE:")
print(per_ticker.round(4).to_string())
