"""Run the connected transition workflow for SPY.

This entrypoint connects two experiment tracks:

1. QRC_Model_1 SPY-only local exact baseline, reproduced from the old code.
2. qBraid measured-feature transition results for 5Q/10Q.

By default it does not submit qBraid jobs. It reuses existing feature CSVs.
Use train_sim_then_qbraid_readout.py if a new qBraid/QPU dataset must be
submitted.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from PIL import Image


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
DEFAULT_HARDWARE_DIR = THIS_DIR / "results"


def run_command(cmd: list[str], cwd: Path):
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def read_metrics(path: Path, prefix: str) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = [f"{prefix}:{idx}" for idx in frame.index]
    return frame


def build_summary(args) -> Path:
    qrc_model1_metrics = (
        args.results_dir / "qrc_model1_connected" / "SPY_qrc_model1_spy_only_metrics.csv"
    )
    phase3_metrics = args.results_dir / "phase3_summary.csv"

    parts = []
    if qrc_model1_metrics.exists():
        parts.append(read_metrics(qrc_model1_metrics, "local"))
    if phase3_metrics.exists():
        phase3 = pd.read_csv(phase3_metrics, index_col=0)
        phase3.index = [f"qbraid:{idx}" for idx in phase3.index]
        parts.append(phase3)
    if not parts:
        raise FileNotFoundError("No metrics were produced by the pipeline.")

    summary = pd.concat(parts, axis=0, sort=False)
    out_path = args.results_dir / "connected_transition_summary.csv"
    summary.to_csv(out_path)
    return out_path


def export_pdf(src: Path, dst: Path):
    if not src.exists():
        raise FileNotFoundError(f"Cannot export missing figure: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.open(src).convert("RGB").save(dst, resolution=160.0)
    print(f"Saved PDF: {dst}", flush=True)


def export_report_pdfs(fig_dir: Path):
    for stem in [
        "phase3_figure1_forecast",
        "phase3_figure2_metrics",
        "phase3_figure3_resources",
        "phase3_train_2025_test_2026_5q",
        "phase3_train_2025_test_2026_10q",
    ]:
        export_pdf(fig_dir / f"{stem}.png", fig_dir / f"{stem}.pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--num-qubits", type=int, default=6)
    parser.add_argument("--n-reservoirs", type=int, default=1)
    parser.add_argument("--eval-rows", type=int, default=100)
    parser.add_argument("--transition-size", type=int, default=20)
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--end", default="2026-07-26")
    parser.add_argument("--results-dir", type=Path, default=THIS_DIR / "results")
    parser.add_argument("--fig-dir", type=Path, default=THIS_DIR / "figs")
    
    parser.add_argument(
        "--export-pdfs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export the main phase3 PNG figures to one-page PDFs.",
    )
    parser.add_argument(
        "--skip-qrc-model1",
        action="store_true",
        help="Only rebuild phase3/qBraid summaries from existing CSVs.",
    )
    parser.add_argument(
        "--skip-phase3",
        action="store_true",
        help="Only run QRC_Model_1 and aggregate existing phase3 metrics.",
    )
    args = parser.parse_args()

    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    
    DEFAULT_HARDWARE_5Q = (
    DEFAULT_HARDWARE_DIR
        / f"SPY_2026_aws_rigetti_qpu_cepheus-1-108q_5q_first{args.transition_size}_features.csv"
    )
    DEFAULT_HARDWARE_10Q = (
        DEFAULT_HARDWARE_DIR
        / f"SPY_2026_aws_rigetti_qpu_cepheus-1-108q_10q_first{args.transition_size}_features.csv"
    )

    if not args.skip_qrc_model1:
        run_command(
            [
                str("python3"),
                str(THIS_DIR / "qbraid_files" / "qrc_model1_spy.py"),
                "--ticker",
                args.ticker,
                "--num-qubits",
                str(args.num_qubits),
                "--n-reservoirs",
                str(args.n_reservoirs),
                "--results-dir",
                str(args.results_dir / "qrc_model1_connected"),
                "--fig-dir",
                str(args.fig_dir / "qrc_model1_connected"),
            ],
            cwd=REPO_ROOT,
        )

    if not args.skip_phase3:
        run_command(
            [
                str("python3"),
                str(THIS_DIR / "qbraid_files" / "QRC_visualization_qbraid.py"),
                "--eval-rows",
                str(args.eval_rows),
                "--transition-size",
                str(args.transition_size),
                "--shots",
                str(args.shots),
                "--end",
                args.end,
                "--features-5q",
                str(DEFAULT_HARDWARE_5Q),
                "--features-10q",
                str(DEFAULT_HARDWARE_10Q),
                "--results-dir",
                str(args.results_dir),
                "--fig-dir",
                str(args.fig_dir),
            ],
            cwd=REPO_ROOT,
        )

    if args.export_pdfs and not args.skip_phase3:
        export_report_pdfs(args.fig_dir)

    summary_path = build_summary(args)
    print(f"\nSaved connected summary: {summary_path}", flush=True)
    print(pd.read_csv(summary_path, index_col=0).to_string(), flush=True)


if __name__ == "__main__":
    main()
