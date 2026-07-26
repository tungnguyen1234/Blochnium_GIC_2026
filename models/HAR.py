import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

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

X_test = sm.add_constant(test[["y_d", "y_w", "y_m"]], has_constant="add")
test["y_hat_HAR_next"] = model.predict(X_test)
test["residual_next"] = test["y_next"] - test["y_hat_HAR_next"]

# ---- Evaluate in LOG space (what the model actually predicts) ----
actual = test["y_next"].values
pred = test["y_hat_HAR_next"].values

mse = mean_squared_error(actual, pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual, pred)
# MAPE is fine in log space since y_next = log(RV+eps) is bounded away from 0,
# but it's not a great metric for log values; QLIKE on raw RV is more standard:
rv_actual = np.exp(actual)          # ≈ RV_next + EPS
rv_pred = np.exp(pred)
qlike = np.mean(rv_actual / rv_pred - np.log(rv_actual / rv_pred) - 1)

# ---- Hit rate per ticker (don't diff across ticker boundaries) ----
def hit_rate_grouped(df):
    rates = []
    for _, g in df.groupby("ticker"):
        pd_dir = (np.diff(g["y_hat_HAR_next"].values) > 0)
        ac_dir = (np.diff(g["y_next"].values) > 0)
        rates.append((pd_dir == ac_dir).mean())
    return np.mean(rates)

print(f"Hit Rate: {hit_rate_grouped(test):.4f}")
print(f"MSE: {mse:.4f}  RMSE: {rmse:.4f}  MAE: {mae:.4f}  QLIKE: {qlike:.4f}")


# -----------------------------
# 5. Visualize predictions vs actual
# -----------------------------

import matplotlib.pyplot as plt
spy = test[test["ticker"] == "SPY"]

plt.figure(figsize=(10, 6))
plt.rcParams["font.size"] = 16
plt.plot(spy["date"], spy["y_next"], label="Actual (log RV)", color="steelblue", alpha=0.7)
plt.plot(spy["date"], spy["y_hat_HAR_next"], label="Predicted", color="darkorange")
plt.title("HAR Model — Predicted vs Actual log RV (SPY, test set)")
plt.xlabel("Date")
plt.ylabel("log Realized Volatility")
plt.legend()
plt.gcf().autofmt_xdate()   # clean date ticks instead of manual xticks
plt.tight_layout()
plt.savefig("HAR_model_SPY.png")