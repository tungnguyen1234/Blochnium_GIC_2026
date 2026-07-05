import numpy as np
import pandas as pd
import yfinance as yf


TICKERS = ["SPY", "JPM", "XOM", "AAPL", "MSFT", "NVDA", "UNH"]
START = "2005-01-01"
END = "2025-12-31"

EPS = 1e-8
ALPHA = 0.90
TRAIN_RATIO = 0.70


def download_data(ticker):
    df = yf.download(
        ticker,
        start=START,
        end=END,
        auto_adjust=True,
        progress=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df


def add_volatility_target_and_regime(df, alpha=ALPHA):
    df = df.copy()

    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["RV"] = df["log_return"] ** 2
    df["y"] = np.log(df["RV"] + EPS)

    # next-day target
    df["y_next"] = df["y"].shift(-1)

    df = df.dropna()

    train_end = int(len(df) * TRAIN_RATIO)
    train = df.iloc[:train_end]

    # leakage-safe threshold from training window only
    q_alpha = train["y"].quantile(alpha)

    df["high_vol_regime_next"] = (df["y_next"] > q_alpha).astype(int)
    df["regime_threshold"] = q_alpha

    df["split"] = "train"
    df.iloc[train_end:, df.columns.get_loc("split")] = "test"

    return df


all_data = []

for ticker in TICKERS:
    print(f"Processing {ticker}...")

    df = download_data(ticker)
    df = add_volatility_target_and_regime(df)

    df.insert(0, "ticker", ticker)
    all_data.append(df)

final = pd.concat(all_data)
final = final.reset_index().rename(columns={"Date": "date"})

final.to_csv("regime_label_dataset.csv", index=False)

print(final.head())
breakpoint()
print(final.shape)