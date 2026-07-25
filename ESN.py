import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from reservoirpy.nodes import Reservoir, Ridge

EPS = 1e-8
K = 3           # same memory depth as the paper's QRC (k=3)
N = 50          # reservoir size, matches classical_reservoir.ipynb's RC

# -----------------------------
# 1. Load dataset
# -----------------------------
dff = pd.read_csv("final_qrc_dataset.csv")
dff["date"] = pd.to_datetime(dff["date"])
dff = dff.sort_values(["ticker", "date"])

# Same lag convention as HAR.py/QRC_test_1.py: features are built from
# shift(1..K) of "y" (i.e. *excluding* the current row's own y), target is
# y_next = y shifted by -1. Kept identical so QLIKE/MSE are comparable to
# the already-reported HAR and HAR+QRC numbers.
for k in range(1, K + 1):
    dff[f"l{k}"] = dff.groupby("ticker")["y"].shift(k)
dff = dff.dropna(subset=[f"l{k}" for k in range(1, K + 1)] + ["y_next", "split"])

# -----------------------------
# 2. Fixed random reservoir (never trained), single ridge readout fit on
#    train rows only -- same single train/test split as HAR.py.
# -----------------------------
import reservoirpy as rpy
rpy.set_seed(0)
reservoir = Reservoir(N, input_dim=1, lr=0.6, sr=0.9, input_scaling=1)
reservoir.initialize(np.zeros((K, 1)))


def states_for(df_slice):
    cols = [f"l{k}" for k in range(K, 0, -1)]  # chronological order: oldest lag first
    X = df_slice[cols].values
    out = np.empty((len(df_slice), reservoir.output_dim))
    for i in range(len(df_slice)):
        seq = X[i].reshape(K, 1)
        reservoir.reset()
        out[i, :] = reservoir.run(seq)[-1, :]
    return out


test_parts = []
for ticker, g in dff.groupby("ticker", sort=True):
    g = g.sort_values("date").reset_index(drop=True)
    train_g = g[g["split"] == "train"]
    test_g = g[g["split"] == "test"]

    train_states = states_for(train_g)
    readout = Ridge(ridge=1e-7).fit(train_states, train_g[["y_next"]].values)

    test_states = states_for(test_g)
    pred = readout.run(test_states)[:, 0]

    out = test_g.copy()
    out["y_hat_ESN_next"] = pred
    test_parts.append(out)
    print(f"  {ticker}: done ({len(train_g)} train, {len(test_g)} test)")

test = pd.concat(test_parts, ignore_index=True)

# -----------------------------
# 3. Evaluate (same convention as HAR.py)
# -----------------------------
actual = test["y_next"].values
pred = test["y_hat_ESN_next"].values

mse = mean_squared_error(actual, pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual, pred)

rv_actual = np.exp(actual)
rv_pred = np.exp(pred)
qlike = np.mean(rv_actual / rv_pred - np.log(rv_actual / rv_pred) - 1)


def hit_rate_grouped(df, col):
    rates = []
    for _, g in df.groupby("ticker"):
        pd_dir = (np.diff(g[col].values) > 0)
        ac_dir = (np.diff(g["y_next"].values) > 0)
        rates.append((pd_dir == ac_dir).mean())
    return np.mean(rates)


print(f"\nHit Rate: {hit_rate_grouped(test, 'y_hat_ESN_next'):.4f}")
print(f"MSE: {mse:.4f}  RMSE: {rmse:.4f}  MAE: {mae:.4f}  QLIKE: {qlike:.4f}")

print("\nPer-ticker QLIKE:")
for ticker, g in test.groupby("ticker"):
    a, p = g["y_next"].values, g["y_hat_ESN_next"].values
    ra, rp = np.exp(a), np.exp(p)
    q = np.mean(ra / rp - np.log(ra / rp) - 1)
    print(f"  {ticker:5s}  QLIKE={q:.4f}")

# -----------------------------
# 4. Visualize (SPY, matching HAR.py)
# -----------------------------
import matplotlib.pyplot as plt

spy = test[test["ticker"] == "SPY"]
plt.figure(figsize=(10, 6))
plt.rcParams["font.size"] = 16
plt.plot(spy["date"], spy["y_next"], label="Actual (log RV)", color="steelblue", alpha=0.7)
plt.plot(spy["date"], spy["y_hat_ESN_next"], label="Predicted", color="darkorange")
plt.title("ESN (N=50) Model — Predicted vs Actual log RV (SPY, test set)")
plt.xlabel("Date")
plt.ylabel("log Realized Volatility")
plt.legend()
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig("ESN_model_SPY.png")

test.to_csv("esn_test_predictions.csv", index=False)
