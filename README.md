# Volatility Forecasting Dataset
[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="150">](https://account.qbraid.com?gitHubUrl=https://github.com/tungnguyen1234/Blochnium_GIC_2026)

This dataset is constructed for **next-day volatility forecasting** and **high-volatility regime prediction** using daily OHLCV market data downloaded from Yahoo Finance.

The dataset supports two prediction tasks:

1. **Regression**: Predict next-day log realized variance.
2. **Classification**: Predict whether the next trading day belongs to a high-volatility regime.

---

# Dataset Columns

| Column | Description |
|---------|-------------|
| **ticker** | Stock ticker symbol (e.g., SPY, JPM, XOM, AAPL, MSFT, NVDA, UNH). |
| **Open** | Opening price of the trading day. |
| **High** | Highest trading price of the day. |
| **Low** | Lowest trading price of the day. |
| **Close** | Closing price of the trading day. |
| **Volume** | Number of shares traded during the day. |
| **log_return** | Daily close-to-close log return, computed as $r_t=\log(P_t/P_{t-1})$. |
| **RV** | Daily realized variance proxy, defined as the squared log return, $RV_t=r_t^2$. |
| **y** | Log-transformed realized variance, $y_t=\log(RV_t+\epsilon)$, where $\epsilon=10^{-8}$ is added for numerical stability. |
| **y_next** | Next-day log realized variance target, computed as $y_{t+1}$. This is the regression target. |
| **high_vol_regime_next** | Binary label indicating whether the next trading day is a high-volatility regime. A value of **1** indicates $y_{t+1}$ exceeds the training-set quantile threshold; otherwise **0**. |
| **regime_threshold** | Quantile threshold $q_\alpha$ used to define the high-volatility regime. The threshold is computed **only from the training set** to prevent data leakage. |
| **split** | Dataset partition used in experiments (`train` or `test`). |

---


# Target Variables

## Regression Target

The regression task predicts

$$y_{t+1}=\log(RV_{t+1}+\epsilon)$$

which represents the next-day log realized variance.

---

## Classification Target

The classification task predicts whether the next trading day belongs to a high-volatility regime.

The regime label is defined as

$$
\text{HighVol}_{t+1}
=
\begin{cases}
1,& y_{t+1}>q_\alpha\\
0,& \text{otherwise}
\end{cases}
$$

where $q_\alpha$ is the $\alpha$-quantile (typically 0.80–0.95) computed **only from the training window**.

This design makes the regime labels **leakage-safe**, since no future observations are used when defining the threshold.

---

# Volatility Proxy

Daily log return is computed as

$$
r_t
=
\log\left(\frac{P_t}{P_{t-1}}\right),
$$

where $P_t$ is the daily closing price.

The realized variance proxy is

$$
RV_t=r_t^2.
$$

Finally, $y_t=\log(RV_t+\epsilon)$ is used to stabilize the distribution before prediction.

---

# Data Source

- **Yahoo Finance**
- Downloaded using the `yfinance` Python package.
- Daily OHLCV data covering **2005–2025**.

The current asset universe consists of:

- SPY
- JPM
- XOM
- AAPL
- MSFT
- NVDA
- UNH

To fully test the QRC model, run: 
`python3 QRC_test.py`
---

# Intended Use

This dataset is designed for machine learning models that perform:

- Next-day volatility forecasting (regression)
- High-volatility regime prediction (classification)
- Hybrid forecasting models combining classical time-series methods with Quantum Reservoir Computing (QRC)


# Current test result
Test-set results:
- HAR only                       MSE: 0.6024  RMSE: 0.7762  MAE: 0.6106  QLIKE: 0.39580
- QRC_model_1 (HAR + QRC)        MSE: 0.5948  RMSE: 0.7712  MAE: 0.6070  QLIKE: 0.38465
- QRC_model_2 (5 qb features)    MSE: 1.3626  RMSE: 1.1673  MAE: 0.9072  QLIKE: 1.35230