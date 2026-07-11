import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# -----------------------------
# 1. Load dataset
# -----------------------------
dff = pd.read_csv("final_qrc_dataset.csv")

# Optional if your CSV has a date column
if "date" in dff.columns:
    dff["date"] = pd.to_datetime(dff["date"])
    dff = dff.sort_values(["ticker", "date"])

# -----------------------------
# 2. Create HAR features
# -----------------------------
# HAR:
# y_d = yesterday's volatility
# y_w = average volatility over past 5 days
# y_m = average volatility over past 22 days

dff["y_d"] = dff.groupby("ticker")["y"].shift(1)
dff["y_w"] = (
    dff.groupby("ticker")["y"]
    .rolling(window=5)
    .mean()
    .reset_index(level=0, drop=True)
    .shift(1)
)
dff["y_m"] = (
    dff.groupby("ticker")["y"]
    .rolling(window=22)
    .mean()
    .reset_index(level=0, drop=True)
    .shift(1)
)

# Target
dff["y_next"] = dff.groupby("ticker")["y"].shift(-1)

# Remove NaNs from lagging/rolling
dff = dff.dropna(subset=["y_d", "y_w", "y_m", "y_next", "split"])


# -----------------------------
# 3. HAR model function
# -----------------------------
def estimate_har_model(data):
    X = data[["y_d", "y_w", "y_m"]]
    y = data["y_next"]

    X = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, X).fit()
    return model


# -----------------------------
# 4. Train/Test by split label
# -----------------------------

train = dff[dff["split"] == "train"].copy()
test = dff[dff["split"] == "test"].copy()

model = estimate_har_model(train)

def get_HAR(data):
    X_data = sm.add_constant(data[["y_d", "y_w", "y_m"]], has_constant="add")
    data = data.copy()
    data["y_hat_HAR"] = model.predict(X_data)
    # ---- Evaluate in LOG space (what the model actually predicts) ----
    return data

def build_qrc_data_by_tickers(split):
    """
    One entry per ticker (N = number of tickers).

    Returns:
        x_list:     list of N arrays, x_list[n][t]     = realized y_next
        y_HAR_list: list of N arrays, y_HAR_list[n][t] = HAR forecast of that value
        tickers:    list of N ticker names (row order)
    Same-size, same-index convention: residual = x - y_HAR elementwise.
    """
    x_list, y_HAR_list, date_list, tickers = [], [], [], []
    y_actual = []
    for ticker, sub in split.groupby("ticker", sort=True):
        sub = sub.sort_values("date")
        x_list.append(sub["y_next"].to_numpy(dtype=float))
        y_HAR_list.append(sub["y_hat_HAR"].to_numpy(dtype=float))
        y_actual.append(sub)
        date_list.append(sub["date"].to_numpy())
        tickers.append(ticker)
    return np.array(x_list), np.array(y_HAR_list), date_list, np.array(tickers)


def evaluate(y_true_log, y_pred_log, label=""):
    """
    y_true_log: realized y_next on the test rows (log RV)
    y_pred_log: forecast in LOG space (y_hat_HAR, or y_hat_HAR + e_hat_QRC)
    """
    mse  = mean_squared_error(y_true_log, y_pred_log)   # (y_true, y_pred) order
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true_log, y_pred_log)

    # QLIKE on RV scale — ratio must be TRUE / FORECAST
    rv_true = np.exp(y_true_log)
    rv_pred = np.exp(y_pred_log)
    ratio   = rv_true / rv_pred
    qlike   = np.mean(ratio - np.log(ratio) - 1)

    print(f"{label:>12s}  MSE: {mse:.4f}  RMSE: {rmse:.4f}  "
          f"MAE: {mae:.4f}  QLIKE: {qlike:.5f}")
    return {"mse": mse, "rmse": rmse, "mae": mae, "qlike": qlike}


######## Train QRC by each ticker data
from QRC_model import QRC_Model
# Analytic (exact expectation values, no shot noise) — good for debugging
num_qubits = 6
f_bs = [0.1, 0.5]  # two reservoirs, two feedbacks
backends = [qml.device("default.qubit", wires=num_qubits) for _ in f_bs]
W = 60  # window length
qrc_model = QRC_Model(num_qubits=num_qubits, backends=backends, f_bs=f_bs, dt=0.1, ridge_param=1.e-6, seed=0)
X_train = get_HAR(train) 
X_test = get_HAR(test)
x_train, yhar_train, _, tickers = build_qrc_data_by_tickers(X_train)
x_test, yhar_test, dates_test, _ = build_qrc_data_by_tickers(X_test)
qrc_model.train(x_train, yhar_train)
qrc_model.fit()
#### Testing
rows = []

preds_log = qrc_model.forward_one_shot(x_test, yhar_test)  # (N, T-1)
for pred_log, tick, x, yh, d in zip(preds_log, tickers, x_test, yhar_test, dates_test):
    rows.append(pd.DataFrame({
        "date":   d[1:],
        "ticker": tick,
        "y_true": x[1:],          
        "y_har":  yh[1:],     
        "y_qrc":  pred_log,
    }))
results = pd.concat(rows, ignore_index=True)
# ---- evaluate ONCE, on identical rows ----
evaluate(results["y_true"], results["y_har"], label="HAR only")
evaluate(results["y_true"], results["y_qrc"], label="HAR + QRC")

def qlike(y_true_log, y_pred_log):
    r = np.exp(y_true_log) / np.exp(y_pred_log)
    return np.mean(r - np.log(r) - 1)


def plot_ticker(g, tick, outdir="figs"):
    import os; os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(g["date"], g["y_true"], color="steelblue", alpha=0.55, lw=1.0,
            label="Actual (log RV)")
    ax.plot(g["date"], g["y_har"], color="darkorange", lw=1.2,
            label=f"HAR  (QLIKE {qlike(g.y_true, g.y_har):.3f})")
    ax.plot(g["date"], g["y_qrc"], color="seagreen", lw=1.2, alpha=0.9,
            label=f"HAR+QRC  (QLIKE {qlike(g.y_true, g.y_qrc):.3f})")

    ax.set_ylabel("log RV")
    ax.set_xlabel("Date")
    ax.set_title(f"{tick} — test set forecasts")
    ax.legend(loc="upper left", fontsize=10)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(f"{outdir}/HAR_vs_QRC_{tick}.png", dpi=150)
    plt.close(fig)


for tick, g in results.groupby("ticker"):
    plot_ticker(g.sort_values("date"), tick)
    
per_ticker = results.groupby("ticker").apply(
    lambda g: pd.Series({
        "HAR":     qlike(g.y_true, g.y_har),
        "HAR+QRC": qlike(g.y_true, g.y_qrc),
    })
)
ax = per_ticker.plot.bar(figsize=(9, 5), color=["darkorange", "seagreen"], rot=0)
ax.set_ylabel("QLIKE (lower = better)")
ax.set_title("Per-ticker QLIKE — HAR vs HAR+QRC")
plt.tight_layout(); plt.savefig("figs/qlike_per_ticker.png", dpi=150)