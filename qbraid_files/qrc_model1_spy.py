"""Reproduce the old QRC_Model experiment for SPY only."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models.QRC_model import QRC_Model


def add_har_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["ticker", "date"])
    grouped = frame.groupby("ticker")["y"]
    frame["y_d"] = grouped.shift(1)
    frame["y_w"] = grouped.transform(lambda s: s.rolling(window=5).mean().shift(1))
    frame["y_m"] = grouped.transform(lambda s: s.rolling(window=22).mean().shift(1))
    frame["y_next"] = grouped.shift(-1)
    return frame.dropna(subset=["y_d", "y_w", "y_m", "y_next", "split"]).copy()


def fit_har(train: pd.DataFrame):
    x_train = sm.add_constant(train[["y_d", "y_w", "y_m"]], has_constant="add")
    return sm.OLS(train["y_next"], x_train).fit()


def add_har_prediction(frame: pd.DataFrame, model) -> pd.DataFrame:
    frame = frame.copy()
    x_all = sm.add_constant(frame[["y_d", "y_w", "y_m"]], has_constant="add")
    frame["y_hat_HAR"] = model.predict(x_all)
    return frame


def qlike(y_true_log, y_pred_log):
    ratio = np.exp(np.asarray(y_true_log) - np.asarray(y_pred_log))
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "qlike": qlike(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def print_metrics(label: str, values: dict):
    print(
        f"{label:>12s}  MSE: {values['mse']:.4f}  RMSE: {values['rmse']:.4f}  "
        f"MAE: {values['mae']:.4f}  QLIKE: {values['qlike']:.5f}  R2: {values['r2']:.5f}"
    )


def plot_predictions(results: pd.DataFrame, out_path: Path, title: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        results["date"],
        results["y_true"],
        color="steelblue",
        alpha=0.55,
        lw=1.0,
        label="Actual (log RV)",
    )
    ax.plot(
        results["date"],
        results["y_har"],
        color="darkorange",
        lw=1.2,
        label=f"HAR (QLIKE {qlike(results.y_true, results.y_har):.3f})",
    )
    ax.plot(
        results["date"],
        results["y_qrc"],
        color="seagreen",
        lw=1.2,
        alpha=0.9,
        label=f"HAR+QRC_Model (QLIKE {qlike(results.y_true, results.y_qrc):.3f})",
    )
    ax.set_ylabel("log RV")
    ax.set_xlabel("Date")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_arrays(frame: pd.DataFrame):
    frame = frame.sort_values("date")
    x = frame["y_next"].to_numpy(dtype=float)[None, :]
    y_har = frame["y_hat_HAR"].to_numpy(dtype=float)[None, :]
    dates = [frame["date"].to_numpy()]
    return x, y_har, dates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "final_qrc_dataset.csv")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--num-qubits", type=int, default=6)
    parser.add_argument("--n-reservoirs", type=int, default=2)
    parser.add_argument("--f-b", type=float, default=0.1)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--ridge-alpha", type=float, default=1.0e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", type=Path, default=THIS_DIR / "results")
    parser.add_argument("--fig-dir", type=Path, default=THIS_DIR / "figs")
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    data = add_har_features(data)
    data = data[data["ticker"].astype(str) == args.ticker].copy()
    train = data[data["split"] == "train"].copy()
    test = data[data["split"] == "test"].copy()
    if train.empty or test.empty:
        raise ValueError(f"Need train and test rows for ticker {args.ticker}.")

    har_model = fit_har(train)
    train = add_har_prediction(train, har_model)
    test = add_har_prediction(test, har_model)
    x_train, yhar_train, _ = build_arrays(train)
    x_test, yhar_test, dates_test = build_arrays(test)

    backends = [
        qml.device("default.qubit", wires=args.num_qubits)
        for _ in range(args.n_reservoirs)
    ]
    qrc_model = QRC_Model(
        num_qubits=args.num_qubits,
        backends=backends,
        f_bs=[args.f_b] * args.n_reservoirs,
        dt=args.dt,
        ridge_param=args.ridge_alpha,
        seed=args.seed,
    )
    qrc_model.train(x_train, yhar_train)
    qrc_model.fit()
    preds_log = qrc_model.forward_one_shot(x_test, yhar_test)[0]

    results = pd.DataFrame(
        {
            "date": dates_test[0][1:],
            "ticker": args.ticker,
            "y_true": x_test[0, 1:],
            "y_har": yhar_test[0, 1:],
            "y_qrc": preds_log,
        }
    )
    results["date"] = pd.to_datetime(results["date"])

    har_metrics = metrics(results["y_true"], results["y_har"])
    qrc_metrics = metrics(results["y_true"], results["y_qrc"])
    print(f"Ticker: {args.ticker}")
    print(f"Train rows: {len(train)}")
    print(f"Test rows:  {len(results)}")
    print(f"Qubits/reservoirs: {args.num_qubits}/{args.n_reservoirs}")
    print_metrics("HAR only", har_metrics)
    print_metrics("HAR+QRC", qrc_metrics)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.results_dir / f"{args.ticker}_qrc_model1_spy_only_predictions.csv"
    out_metrics = args.results_dir / f"{args.ticker}_qrc_model1_spy_only_metrics.csv"
    results.to_csv(out_csv, index=False)
    pd.DataFrame.from_dict(
        {"HAR": har_metrics, "HAR+QRC_Model": qrc_metrics},
        orient="index",
    ).to_csv(out_metrics)

    out_fig = args.fig_dir / f"{args.ticker}_qrc_model1_spy_only.png"
    plot_predictions(results, out_fig, f"{args.ticker} -- QRC_Model SPY-only test forecasts")
    print(f"Saved predictions: {out_csv}")
    print(f"Saved metrics:     {out_metrics}")
    print(f"Saved figure:      {out_fig}")


if __name__ == "__main__":
    main()
