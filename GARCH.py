import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from arch import arch_model

EPS = 1e-8

# -----------------------------
# 1. Load dataset
# -----------------------------
dff = pd.read_csv("final_qrc_dataset.csv")
dff["date"] = pd.to_datetime(dff["date"])
dff = dff.sort_values(["ticker", "date"])

# -----------------------------
# 2. Per-ticker GARCH(1,1) on daily log returns
# -----------------------------
# GARCH models the conditional variance of returns directly, so unlike HAR it
# is fit on log_return (not y). We fit once on the train block per ticker
# (same single train/test split as HAR.py/QRC_test_1.py), then produce true
# one-step-ahead forecasts through the test block using the GARCH(1,1)
# recursion with the fitted (train-only) parameters and the *actual* realized
# returns as they arrive -- no lookahead, no refitting per day.
SCALE = 100.0  # arch converges more reliably on O(1-1000)-scale returns

def garch_forecast_ticker(g):
    g = g.sort_values("date").reset_index(drop=True)
    train = g[g["split"] == "train"]
    n_train = len(train)

    am = arch_model(g["log_return"].values * SCALE, mean="Zero", vol="GARCH", p=1, q=1)
    res = am.fit(first_obs=0, last_obs=n_train, disp="off")

    # analytic one-step-ahead variance forecast for every day using the
    # train-fitted params, rolled forward through the full series
    fc = res.forecast(horizon=1, start=n_train - 1, reindex=False)
    sigma2 = fc.variance.values[:, 0] / (SCALE ** 2)  # forecast of RV_{t+1}

    out = g.iloc[n_train:].copy()
    out["y_hat_GARCH_next"] = np.log(sigma2[:len(out)] + EPS)
    return out


test_parts = []
for ticker, g in dff.groupby("ticker", sort=True):
    test_parts.append(garch_forecast_ticker(g))
test = pd.concat(test_parts, ignore_index=True)

# -----------------------------
# 3. Evaluate (same convention as HAR.py)
# -----------------------------
actual = test["y_next"].values
pred = test["y_hat_GARCH_next"].values

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


print(f"Hit Rate: {hit_rate_grouped(test, 'y_hat_GARCH_next'):.4f}")
print(f"MSE: {mse:.4f}  RMSE: {rmse:.4f}  MAE: {mae:.4f}  QLIKE: {qlike:.4f}")

print("\nPer-ticker QLIKE:")
for ticker, g in test.groupby("ticker"):
    a, p = g["y_next"].values, g["y_hat_GARCH_next"].values
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
plt.plot(spy["date"], spy["y_hat_GARCH_next"], label="Predicted", color="darkorange")
plt.title("GARCH(1,1) Model — Predicted vs Actual log RV (SPY, test set)")
plt.xlabel("Date")
plt.ylabel("log Realized Volatility")
plt.legend()
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig("GARCH_model_SPY.png")

test.to_csv("garch_test_predictions.csv", index=False)
