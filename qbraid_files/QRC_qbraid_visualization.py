"""Generate Phase 3 figures and summary table from transition datasets."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit import transpile
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qbraid_files.qbraid_utils import build_qrc_circuit, generate_j
from QRC_run_qbraid import (
    add_har,
    add_targets,
    anchored_transition,
    build_feature_csv,
    download_spy,
    metrics,
    select_feature_rows,
    z_columns,
)


DEFAULT_RESULTS = THIS_DIR / "results"
DEFAULT_FIGS = THIS_DIR / "figs"
DEFAULT_10Q_FEATURES = (
    DEFAULT_RESULTS / "SPY_2026_ionq_ionq_sim_simulator_10q_first100_features.csv"
)
DEFAULT_5Q_FEATURES = (
    DEFAULT_RESULTS / "SPY_2026_qbraid_qbraid_sim_qir-sv_5q_first100_features.csv"
)


def train_readout(train_features: pd.DataFrame, feature_cols, alpha: float):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_features[feature_cols].to_numpy(dtype=float))
    y_train = train_features["residual_target"].to_numpy(dtype=float)
    ridge = Ridge(alpha=alpha)
    ridge.fit(x_train, y_train)
    return scaler, ridge.coef_.astype(float), float(ridge.intercept_)


def train_predictions(train_features, scaler, w_sim, b_sim, feature_cols):
    x_train = scaler.transform(train_features[feature_cols].to_numpy(dtype=float))
    preds = train_features.copy()
    preds["target_date"] = pd.to_datetime(preds["target_date"])
    preds["y_qrc_after"] = preds["y_har_target"] + (x_train @ w_sim + b_sim)
    return preds


def transition_predictions(eval_features, scaler, w_sim, b_sim, feature_cols, args):
    transition_size = min(args.transition_size, max(1, len(eval_features) - 1))
    transition = eval_features.iloc[:transition_size].copy()
    test = eval_features.iloc[transition_size:].copy()

    x_transition = scaler.transform(transition[feature_cols].to_numpy(dtype=float))
    y_transition = transition["residual_target"].to_numpy(dtype=float)
    x_test = scaler.transform(test[feature_cols].to_numpy(dtype=float))

    w_after, b_after = anchored_transition(
        w_sim, b_sim, x_transition, y_transition, args.anchor_l2
    )

    test = test.copy()
    test["target_date"] = pd.to_datetime(test["target_date"])
    test["y_qrc_before"] = test["y_har_target"] + (x_test @ w_sim + b_sim)
    test["y_qrc_after"] = test["y_har_target"] + (x_test @ w_after + b_after)
    return test


def circuit_stats(num_qubits: int, args):
    j_mat = generate_j(num_qubits, args.seed)
    circuit = build_qrc_circuit(
        num_qubits,
        j_mat,
        0.0,
        np.zeros(num_qubits, dtype=float),
        args.dt,
        args.f_b,
        args.n_trotter,
        args.input_scale,
        args.feedback_scale,
    )
    return {
        "logical_depth": int(circuit.depth()),
        "transpiled_depth": int(transpile(circuit, optimization_level=1).depth()),
    }


def plot_actual_vs_predicted(preds: pd.DataFrame, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        preds["target_date"],
        preds["y_target"],
        color="#1f77b4",
        lw=1.4,
        label="Actual",
    )
    ax.plot(
        preds["target_date"],
        preds["y_qrc_after"],
        color="#d8a24a",
        lw=1.4,
        alpha=0.95,
        label="Predicted",
    )
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Log realized volatility")
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.grid(True, color="#9a9a9a", alpha=0.42, linewidth=0.8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_train_and_test(train_preds, test_preds, out_path: Path, title: str):
    train_part = train_preds[["target_date", "y_target", "y_qrc_after"]].copy()
    train_part["split"] = "Train 2025"
    test_part = test_preds[["target_date", "y_target", "y_qrc_after"]].copy()
    test_part["split"] = "Test 2026"
    data = pd.concat([train_part, test_part], ignore_index=True)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(data["target_date"], data["y_target"], color="#1f77b4", lw=1.15, label="Actual")
    ax.plot(
        data["target_date"],
        data["y_qrc_after"],
        color="#d8a24a",
        lw=1.15,
        alpha=0.95,
        label="Predicted",
    )
    split_date = pd.Timestamp(test_part["target_date"].min())
    ax.axvline(split_date, color="#333333", lw=1.0, ls="--", alpha=0.75)
    ax.text(
        split_date,
        ax.get_ylim()[1],
        "  2026 test",
        va="top",
        ha="left",
        fontsize=9,
        color="#333333",
    )
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Log realized volatility")
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(True, color="#9a9a9a", alpha=0.42, linewidth=0.8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_har_style(train_preds, test_preds, out_path: Path, title: str):
    data = pd.concat(
        [
            train_preds[["target_date", "y_target", "y_qrc_after"]],
            test_preds[["target_date", "y_target", "y_qrc_after"]],
        ],
        ignore_index=True,
    )

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.plot(
        data["target_date"],
        data["y_target"],
        color="#6fa3d2",
        lw=1.2,
        alpha=0.85,
        label="Actual (log RV)",
    )
    ax.plot(
        data["target_date"],
        data["y_qrc_after"],
        color="#ff8c00",
        lw=1.4,
        label="Predicted",
    )
    ax.set_title(title, fontsize=19)
    ax.set_xlabel("Date", fontsize=16)
    ax.set_ylabel("log Realized Volatility", fontsize=16)
    ax.legend(loc="upper center", fontsize=16, frameon=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="both", labelsize=14)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_metrics(summary: pd.DataFrame, out_path: Path):
    methods = ["HAR", "5Q QRC", "10Q QRC"]
    rmse = [
        summary.loc["HAR", "rmse"],
        summary.loc["QRC-5", "rmse"],
        summary.loc["QRC-10", "rmse"],
    ]
    qlikes = [
        summary.loc["HAR", "qlike"],
        summary.loc["QRC-5", "qlike"],
        summary.loc["QRC-10", "qlike"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(methods, rmse, color=["#7a7a7a", "#4c78a8", "#f58518"])
    axes[0].set_title("RMSE")
    axes[0].set_ylabel("Lower is better")
    axes[1].bar(methods, qlikes, color=["#7a7a7a", "#4c78a8", "#f58518"])
    axes[1].set_title("QLIKE")
    axes[1].set_ylabel("Lower is better")
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Phase 3 Forecasting Performance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_resources(summary: pd.DataFrame, out_path: Path):
    labels = ["5 qubits", "10 qubits"]
    logical = [summary.loc["QRC-5", "logical_depth"], summary.loc["QRC-10", "logical_depth"]]
    transpiled = [
        summary.loc["QRC-5", "transpiled_depth"],
        summary.loc["QRC-10", "transpiled_depth"],
    ]
    runtime = [
        summary.loc["QRC-5", "mean_runtime_seconds"],
        summary.loc["QRC-10", "mean_runtime_seconds"],
    ]

    x = np.arange(len(labels))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(x - width / 2, logical, width, label="Logical depth")
    axes[0].bar(x + width / 2, transpiled, width, label="Transpiled depth")
    axes[0].set_xticks(x, labels)
    axes[0].set_title("Circuit Depth Scaling")
    axes[0].set_ylabel("Depth")
    axes[0].legend(fontsize=8)
    axes[1].bar(labels, runtime, color=["#4c78a8", "#f58518"])
    axes[1].set_title("Runtime Scaling")
    axes[1].set_ylabel("Mean wall-clock seconds per task")
    fig.suptitle("Phase 3 Resource Scaling")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-07-27")
    parser.add_argument("--train-year", type=int, default=2025)
    parser.add_argument("--eval-start", default="2026-01-01")
    parser.add_argument("--eval-rows", type=int, default=100)
    parser.add_argument("--transition-size", type=int, default=10)
    parser.add_argument("--test-tail", type=int, default=30)
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--n-trotter", type=int, default=2)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--f-b", type=float, default=1.0)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--feedback-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ridge-alpha", type=float, default=1.0e-6)
    parser.add_argument("--anchor-l2", type=float, default=100.0)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIGS)
    parser.add_argument("--features-5q", type=Path, default=DEFAULT_5Q_FEATURES)
    parser.add_argument("--features-10q", type=Path, default=DEFAULT_10Q_FEATURES)
    args = parser.parse_args()

    raw = download_spy(args.start, args.end)
    frame = add_har(add_targets(raw), args.train_year)
    train_rows = select_feature_rows(
        frame,
        start=f"{args.train_year}-01-01",
        end=f"{args.train_year}-12-31",
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    train_5q = build_feature_csv(
        train_rows,
        argparse.Namespace(**vars(args), num_qubits=5, overwrite_features=False),
        args.results_dir / "SPY_2025_local_5q_train_features.csv",
        backend="local",
    )
    train_10q = build_feature_csv(
        train_rows,
        argparse.Namespace(**vars(args), num_qubits=10, overwrite_features=False),
        args.results_dir / "SPY_2025_local_10q_train_features.csv",
        backend="local",
    )

    eval_5q = pd.read_csv(args.features_5q)
    eval_10q = pd.read_csv(args.features_10q)

    cols_5q = z_columns(train_5q)
    cols_10q = z_columns(train_10q)
    scaler_5q, w_5q, b_5q = train_readout(train_5q, cols_5q, args.ridge_alpha)
    scaler_10q, w_10q, b_10q = train_readout(train_10q, cols_10q, args.ridge_alpha)
    train_pred_5q = train_predictions(train_5q, scaler_5q, w_5q, b_5q, cols_5q)
    train_pred_10q = train_predictions(train_10q, scaler_10q, w_10q, b_10q, cols_10q)
    pred_5q = transition_predictions(eval_5q, scaler_5q, w_5q, b_5q, cols_5q, args)
    pred_10q = transition_predictions(eval_10q, scaler_10q, w_10q, b_10q, cols_10q, args)

    train_pred_5q.to_csv(args.results_dir / "phase3_5q_train_predictions.csv", index=False)
    train_pred_10q.to_csv(args.results_dir / "phase3_10q_train_predictions.csv", index=False)
    pred_5q.to_csv(args.results_dir / "phase3_5q_predictions.csv", index=False)
    pred_10q.to_csv(args.results_dir / "phase3_10q_predictions.csv", index=False)

    har_metrics = metrics(pred_10q["y_target"], pred_10q["y_har_target"])
    qrc5_metrics = metrics(pred_5q["y_target"], pred_5q["y_qrc_after"])
    qrc10_metrics = metrics(pred_10q["y_target"], pred_10q["y_qrc_after"])
    stats_5q = circuit_stats(5, args)
    stats_10q = circuit_stats(10, args)

    summary = pd.DataFrame.from_dict(
        {
            "HAR": {
                **har_metrics,
                "qubits": 0,
                "logical_depth": 0,
                "transpiled_depth": 0,
                "shots_per_task": 0,
                "tasks": len(pred_10q) + args.transition_size,
                "total_shots": 0,
                "mean_runtime_seconds": 0.0,
                "backend": "classical",
            },
            "QRC-5": {
                **qrc5_metrics,
                **stats_5q,
                "qubits": 5,
                "shots_per_task": args.shots,
                "tasks": args.eval_rows,
                "total_shots": args.eval_rows * args.shots,
                "mean_runtime_seconds": 0.0,
                "backend": "qBraid QIR simulator",
            },
            "QRC-10": {
                **qrc10_metrics,
                **stats_10q,
                "qubits": 10,
                "shots_per_task": args.shots,
                "tasks": args.eval_rows,
                "total_shots": args.eval_rows * args.shots,
                "mean_runtime_seconds": np.nan,
                "backend": "qBraid simulator",
            },
        },
        orient="index",
    )
    summary.to_csv(args.results_dir / "phase3_summary.csv")

    plot_actual_vs_predicted(
        pred_10q.tail(args.test_tail),
        args.fig_dir / "phase3_figure1_forecast.png",
        "Actual vs Predicted Log Volatility",
    )
    plot_actual_vs_predicted(
        pred_5q,
        args.fig_dir / "phase3_actual_vs_predicted_5q.png",
        "Actual vs Predicted Log Volatility - 5Q QIR Simulator",
    )
    plot_actual_vs_predicted(
        pred_10q,
        args.fig_dir / "phase3_actual_vs_predicted_10q.png",
        "Actual vs Predicted Log Volatility - 10Q Simulator",
    )
    plot_train_and_test(
        train_pred_5q,
        pred_5q,
        args.fig_dir / "phase3_train_2025_test_2026_5q.png",
        "Actual vs Predicted Log Volatility - 2025 Train and 2026 Test - 5Q",
    )
    plot_train_and_test(
        train_pred_10q,
        pred_10q,
        args.fig_dir / "phase3_train_2025_test_2026_10q.png",
        "Actual vs Predicted Log Volatility - 2025 Train and 2026 Test - 10Q",
    )
    plot_har_style(
        train_pred_5q,
        pred_5q,
        args.fig_dir / "phase3_har_style_5q.png",
        "QRC-5 Model -- Predicted vs Actual log RV (SPY)",
    )
    plot_har_style(
        train_pred_10q,
        pred_10q,
        args.fig_dir / "phase3_har_style_10q.png",
        "QRC-10 Model -- Predicted vs Actual log RV (SPY)",
    )
    plot_metrics(summary, args.fig_dir / "phase3_figure2_metrics.png")
    plot_resources(summary, args.fig_dir / "phase3_figure3_resources.png")

    print(summary[["backend", "qubits", "rmse", "qlike", "logical_depth", "transpiled_depth", "shots_per_task", "tasks", "total_shots"]])
    print()
    print(f"Saved: {args.fig_dir / 'phase3_figure1_forecast.png'}")
    print(f"Saved: {args.fig_dir / 'phase3_actual_vs_predicted_5q.png'}")
    print(f"Saved: {args.fig_dir / 'phase3_actual_vs_predicted_10q.png'}")
    print(f"Saved: {args.fig_dir / 'phase3_train_2025_test_2026_5q.png'}")
    print(f"Saved: {args.fig_dir / 'phase3_train_2025_test_2026_10q.png'}")
    print(f"Saved: {args.fig_dir / 'phase3_har_style_5q.png'}")
    print(f"Saved: {args.fig_dir / 'phase3_har_style_10q.png'}")
    print(f"Saved: {args.fig_dir / 'phase3_figure2_metrics.png'}")
    print(f"Saved: {args.fig_dir / 'phase3_figure3_resources.png'}")
    print(f"Saved: {args.results_dir / 'phase3_summary.csv'}")


if __name__ == "__main__":
    main()
