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
import matplotlib.patches as mpatches
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
from QRC_train_test_qbraid import (
    add_har,
    add_targets,
    anchored_transition,
    build_feature_csv,
    download_spy,
    metrics,
    qlike,
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

COLORS = {
    "actual": "#1b5e8c",
    "har": "#6f6f6f",
    "qrc5": "#2a9d8f",
    "qrc10": "#e6862a",
    "q_train": "#e63946",
    "q_test": "#9b5de5",
    "grid": "#d9dde3",
    "text": "#20242a",
}


def apply_plot_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#9aa3ad",
            "axes.labelcolor": COLORS["text"],
            "axes.titlecolor": COLORS["text"],
            "axes.titlesize": 14,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "font.size": 10,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#d6dbe0",
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "savefig.bbox": "tight",
        }
    )


def polish_axis(ax):
    ax.grid(True, color=COLORS["grid"], linewidth=0.8, alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.01)


def add_transition_span(ax, dates, transition_size):
    if transition_size <= 0 or len(dates) <= transition_size:
        return
    transition_end = pd.to_datetime(dates.iloc[transition_size - 1])
    ax.axvspan(
        pd.to_datetime(dates.iloc[0]),
        transition_end,
        color=COLORS["q_train"],
        alpha=0.14,
        lw=0,
        label="Transition window",
    )
    ax.axvline(transition_end, color="#b38b00", lw=1.0, ls="--", alpha=0.8)


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

    x_transition = scaler.transform(transition[feature_cols].to_numpy(dtype=float))
    y_transition = transition["residual_target"].to_numpy(dtype=float)

    w_after, b_after = anchored_transition(
        w_sim, b_sim, x_transition, y_transition, args.anchor_l2
    )

    preds = eval_features.copy()
    x_eval = scaler.transform(preds[feature_cols].to_numpy(dtype=float))
    preds["target_date"] = pd.to_datetime(preds["target_date"])
    preds["y_qrc_before"] = preds["y_har_target"] + (x_eval @ w_sim + b_sim)
    preds["y_qrc_after"] = preds["y_har_target"] + (x_eval @ w_after + b_after)
    preds["phase"] = "test"
    preds.loc[preds.index[:transition_size], "phase"] = "transition"
    return preds


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


def plot_actual_vs_predicted(
    preds: pd.DataFrame,
    out_path: Path,
    title: str,
    pred_label: str = "QRC prediction",
    pred_color: str = COLORS["qrc10"],
):
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.plot(
        preds["target_date"],
        preds["y_target"],
        color=COLORS["actual"],
        lw=1.9,
        label="Actual log RV",
    )
    ax.plot(
        preds["target_date"],
        preds["y_har_target"],
        color=COLORS["har"],
        lw=1.25,
        ls="--",
        alpha=0.85,
        label="HAR baseline",
    )
    ax.plot(
        preds["target_date"],
        preds["y_qrc_after"],
        color=pred_color,
        lw=1.8,
        alpha=0.95,
        label=pred_label,
    )
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Log realized volatility")
    ax.legend(loc="upper left", fontsize=9, ncol=3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    polish_axis(ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_hardware_forecast(pred_5q, pred_10q, out_path: Path, tail: int, transition_size: int):
    data5 = pred_5q.tail(tail).copy()
    data10 = pred_10q.tail(tail).copy()
    fig, ax = plt.subplots(figsize=(12, 4.9))
    ax.plot(
        data10["target_date"],
        data10["y_target"],
        color=COLORS["actual"],
        lw=2.1,
        label="Actual log RV",
    )
    ax.plot(
        data10["target_date"],
        data10["y_har_target"],
        color=COLORS["har"],
        lw=1.4,
        ls="--",
        label="HAR baseline",
    )
    ax.plot(
        data5["target_date"],
        data5["y_qrc_after"],
        color=COLORS["qrc5"],
        lw=1.8,
        label="HAR + 5Q QRC",
    )
    ax.plot(
        data10["target_date"],
        data10["y_qrc_after"],
        color=COLORS["qrc10"],
        lw=1.8,
        label="HAR + 10Q QRC",
    )
    ax.set_title("Hardware-Style QRC Forecast on Held-Out 2026 Window")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log realized volatility")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    polish_axis(ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_train_and_test(train_preds, eval_preds, out_path: Path, title: str, pred_color: str):
    train_part = train_preds[["target_date", "y_target", "y_qrc_after"]].copy()
    train_part["split"] = "Train 2025"
    eval_part = eval_preds[["target_date", "y_target", "y_qrc_after", "phase"]].copy()
    eval_part["split"] = eval_part["phase"].map(
        {"transition": "Transition 2026", "test": "Test 2026"}
    )
    transition_part = eval_part[eval_part["split"] == "Transition 2026"].copy()
    test_part = eval_part[eval_part["split"] == "Test 2026"].copy()

    fig, ax = plt.subplots(figsize=(13, 5))
    q_sim_patch = mpatches.Patch(
        facecolor="white",
        edgecolor="#8f98a3",
        label="Q simulion",
    )
    ax.plot(
        train_part["target_date"],
        train_part["y_target"],
        color=COLORS["actual"],
        lw=1.15,
        label="Actual log RV",
    )
    ax.plot(
        train_part["target_date"],
        train_part["y_qrc_after"],
        color=pred_color,
        lw=1.35,
        alpha=0.95,
        label="QRC prediction",
    )
    ax.plot(
        transition_part["target_date"],
        transition_part["y_target"],
        color=COLORS["actual"],
        lw=1.15,
        ls="-.",
        alpha=0.88,
    )
    ax.plot(
        transition_part["target_date"],
        transition_part["y_qrc_after"],
        color=pred_color,
        lw=1.35,
        ls="-.",
        alpha=0.88,
    )
    ax.plot(
        test_part["target_date"],
        test_part["y_target"],
        color=COLORS["actual"],
        lw=1.15,
        ls="--",
    )
    ax.plot(
        test_part["target_date"],
        test_part["y_qrc_after"],
        color=pred_color,
        lw=1.35,
        ls="--",
        alpha=0.95,
    )
    eval_start = pd.Timestamp(eval_part["target_date"].min())
    test_start = pd.Timestamp(test_part["target_date"].min())
    ax.axvspan(
        eval_start,
        test_start,
        color=COLORS["q_train"],
        alpha=0.22,
        lw=0,
        label="Q train",
    )
    ax.axvspan(
        test_start,
        pd.Timestamp(test_part["target_date"].max()),
        color=COLORS["q_test"],
        alpha=0.24,
        lw=0,
        label="Q test",
    )
    ax.axvline(eval_start, color="#333333", lw=1.2, ls="--", alpha=0.8)
    ax.axvline(test_start, color="#333333", lw=1.2, ls="--", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Log realized volatility")
    ymin, ymax = ax.get_ylim()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([q_sim_patch] + handles, ["Q simulion"] + labels, loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    polish_axis(ax)
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

    fig, ax = plt.subplots(figsize=(11, 5.6), dpi=120)
    ax.plot(
        data["target_date"],
        data["y_target"],
        color=COLORS["actual"],
        lw=1.35,
        alpha=0.85,
        label="Actual (log RV)",
    )
    ax.plot(
        data["target_date"],
        data["y_qrc_after"],
        color=COLORS["qrc10"],
        lw=1.6,
        label="Predicted",
    )
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Log realized volatility", fontsize=12)
    ax.legend(loc="upper left", fontsize=10, frameon=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="both", labelsize=10)
    polish_axis(ax)
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

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    colors = [COLORS["har"], COLORS["qrc5"], COLORS["qrc10"]]
    axes[0].bar(methods, rmse, color=colors, width=0.62)
    axes[0].set_title("RMSE")
    axes[0].set_ylabel("Lower is better")
    axes[1].bar(methods, qlikes, color=colors, width=0.62)
    axes[1].set_title("QLIKE")
    axes[1].set_ylabel("Lower is better")
    for ax, values in zip(axes, [rmse, qlikes]):
        ax.tick_params(axis="x", rotation=15)
        polish_axis(ax)
        top = max(values) if values else 1.0
        ax.set_ylim(0, top * 1.18)
        for idx, value in enumerate(values):
            ax.text(idx, value + top * 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Hardware Metric Comparison", fontsize=15, fontweight="semibold")
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
    total_cost = [
        summary.loc["QRC-5", "total_cost_cr"],
        summary.loc["QRC-10", "total_cost_cr"],
    ]

    x = np.arange(len(labels))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(x - width / 2, logical, width, label="Logical depth")
    axes[0].bar(x + width / 2, transpiled, width, label="Transpiled depth")
    axes[0].set_xticks(x, labels)
    axes[0].set_title("Circuit Depth Scaling")
    axes[0].set_ylabel("Depth")
    axes[0].legend(fontsize=8)
    axes[1].bar(labels, total_cost, color=[COLORS["qrc5"], COLORS["qrc10"]], width=0.62)
    axes[1].set_title("Credit Budget Scaling")
    axes[1].set_ylabel("Total cost (qBraid credits)")
    top = max(total_cost) if total_cost else 1.0
    axes[1].set_ylim(0, top * 1.18)
    for idx, value in enumerate(total_cost):
        axes[1].text(idx, value + top * 0.025, f"{value:,.0f} cr", ha="center", va="bottom", fontsize=9)
    for ax in axes:
        polish_axis(ax)
    fig.suptitle("Hardware Resource Scaling", fontsize=15, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    apply_plot_style()
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-07-27")
    parser.add_argument("--train-year", type=int, default=2025)
    parser.add_argument("--eval-start", default="2026-01-01")
    parser.add_argument("--eval-rows", type=int, default=100)
    parser.add_argument("--transition-size", type=int, default=20)
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

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    train_5q_path = args.results_dir / "SPY_2025_local_5q_train_features.csv"
    train_10q_path = args.results_dir / "SPY_2025_local_10q_train_features.csv"
    if train_5q_path.exists() and train_10q_path.exists():
        train_5q = pd.read_csv(train_5q_path)
        train_10q = pd.read_csv(train_10q_path)
    else:
        raw = download_spy(args.start, args.end)
        frame = add_har(add_targets(raw), args.train_year)
        train_rows = select_feature_rows(
            frame,
            start=f"{args.train_year}-01-01",
            end=f"{args.train_year}-12-31",
        )
        train_5q = build_feature_csv(
            train_rows,
            argparse.Namespace(**vars(args), num_qubits=5, overwrite_features=False),
            train_5q_path,
            backend="local",
        )
        train_10q = build_feature_csv(
            train_rows,
            argparse.Namespace(**vars(args), num_qubits=10, overwrite_features=False),
            train_10q_path,
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
    test_5q = pred_5q[pred_5q["phase"] == "test"].copy()
    test_10q = pred_10q[pred_10q["phase"] == "test"].copy()

    train_pred_5q.to_csv(args.results_dir / "phase3_5q_train_predictions.csv", index=False)
    train_pred_10q.to_csv(args.results_dir / "phase3_10q_train_predictions.csv", index=False)
    pred_5q.to_csv(args.results_dir / "phase3_5q_predictions.csv", index=False)
    pred_10q.to_csv(args.results_dir / "phase3_10q_predictions.csv", index=False)

    har_metrics = metrics(test_10q["y_target"], test_10q["y_har_target"])
    qrc5_metrics = metrics(test_5q["y_target"], test_5q["y_qrc_after"])
    qrc10_metrics = metrics(test_10q["y_target"], test_10q["y_qrc_after"])
    stats_5q = circuit_stats(5, args)
    stats_10q = circuit_stats(10, args)
    rigetti_per_task_credit = 30.0
    rigetti_per_shot_credit = 0.0425
    cost_per_task = rigetti_per_task_credit + rigetti_per_shot_credit * args.shots

    summary = pd.DataFrame.from_dict(
        {
            "HAR": {
                **har_metrics,
                "qubits": 0,
                "logical_depth": 0,
                "transpiled_depth": 0,
                "shots_per_task": 0,
                "tasks": len(pred_10q),
                "total_shots": 0,
                "cost_per_task_cr": 0.0,
                "total_cost_cr": 0.0,
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
                "cost_per_task_cr": cost_per_task,
                "total_cost_cr": args.eval_rows * cost_per_task,
                "mean_runtime_seconds": 0.0,
                "backend": "Rigetti QPU",
            },
            "QRC-10": {
                **qrc10_metrics,
                **stats_10q,
                "qubits": 10,
                "shots_per_task": args.shots,
                "tasks": args.eval_rows,
                "total_shots": args.eval_rows * args.shots,
                "cost_per_task_cr": cost_per_task,
                "total_cost_cr": args.eval_rows * cost_per_task,
                "mean_runtime_seconds": np.nan,
                "backend": "Rigetti QPU",
            },
        },
        orient="index",
    )
    summary.to_csv(args.results_dir / "phase3_summary.csv")

    plot_hardware_forecast(
        pred_5q,
        pred_10q,
        args.fig_dir / "phase3_figure1_forecast.png",
        args.test_tail,
        args.transition_size,
    )
    plot_actual_vs_predicted(
        pred_5q,
        args.fig_dir / "phase3_actual_vs_predicted_5q.png",
        "Actual vs Predicted Log Volatility - 5Q QIR Simulator",
        pred_label="HAR + 5Q QRC",
        pred_color=COLORS["qrc5"],
    )
    plot_actual_vs_predicted(
        pred_10q,
        args.fig_dir / "phase3_actual_vs_predicted_10q.png",
        "Actual vs Predicted Log Volatility - 10Q Simulator",
        pred_label="HAR + 10Q QRC",
        pred_color=COLORS["qrc10"],
    )
    plot_train_and_test(
        train_pred_5q,
        pred_5q,
        args.fig_dir / "phase3_train_2025_test_2026_5q.png",
        "Actual vs Predicted Log Volatility - 2025 Train and 2026 Test - 5Q",
        COLORS["qrc5"],
    )
    plot_train_and_test(
        train_pred_10q,
        pred_10q,
        args.fig_dir / "phase3_train_2025_test_2026_10q.png",
        "Actual vs Predicted Log Volatility - 2025 Train and 2026 Test - 10Q",
        COLORS["qrc10"],
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
