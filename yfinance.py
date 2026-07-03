import yfinance as yf
import pandas as pd
import numpy as np

# 1. Choose ticker
ticker = "SPY"

# 2. Download daily Yahoo Finance data
df = yf.download(
    ticker,
    start="2010-01-01",
    end="2025-12-31",
    interval="1d",
    auto_adjust=True,
    progress=False
)

# 3. Clean column names if yfinance returns MultiIndex
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# 4. Keep useful columns
df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

# 5. Create log return
df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

# 6. Volatility proxy option A: squared return
df["rv_squared_return"] = df["log_return"] ** 2

# 7. Volatility proxy option B: Parkinson volatility from High/Low
df["rv_parkinson"] = (np.log(df["High"] / df["Low"]) ** 2) / (4 * np.log(2))

# 8. Log volatility target
eps = 1e-8
df["log_rv"] = np.log(df["rv_parkinson"] + eps)

# 9. Remove missing rows
df = df.dropna()

# 10. Save for team
df.to_csv("spy_yahoo_volatility_data.csv")

print(df.head())
print(df.tail())
print(df.shape)