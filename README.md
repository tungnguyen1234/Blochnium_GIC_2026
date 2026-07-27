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
| **split** | Dataset partition used in experiments (`train` or `test`). |
<!-- | **high_vol_regime_next** | Binary label indicating whether the next trading day is a high-volatility regime. A value of **1** indicates $y_{t+1}$ exceeds the training-set quantile threshold; otherwise **0**. |
| **regime_threshold** | Quantile threshold $q_\alpha$ used to define the high-volatility regime. The threshold is computed **only from the training set** to prevent data leakage. | -->

# Instruction



<!-- Real qBraid quantum computer can be run with this notebook via qBraid instance `QRC_test_qbraid_notebook.ipynb` -->

## Setup on qBraid

### 1. Launch the repository

Click the **Launch on qBraid** button above.

The GitHub repository must be public for the launch button to clone it automatically.

### 2. Open a terminal

From the repository root, install the required dependencies:

```bash
pip install -r requirements.txt
```

The project requires a Python environment with packages for:

- Numerical and tabular processing
- Classical machine learning
- Financial data processing
- Qiskit circuit construction and execution
- qBraid quantum backend access

## Experiment 1: Local GPU run
After clicking on launch Qbraid, simulation results can be run with 
```bash
python3 QRC_run_local.py
```

Main outputs and visualizations are:

```text
figs/HAR_vs_QRC_{ticker_id}.png
```

## Experiment 2: Full QPU Rigetti Benchmark

When clicking `Launch QBraid`, open the Terminal and run the following commands to setup the environments

```bash
cd Blochnium_GIC_2026/
pip3 install -r requirements.txt
```

### Step 1: Calling API/QPU to obtain dataset using 5Qubit

```bash
QBRAID_API_KEY="..." OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python3 QRC_train_test_qbraid.py \
  --eval-feature-backend qbraid \
  --qbraid-device-id aws:rigetti:qpu:cepheus-1-108q \
  --num-qubits 5 \
  --eval-rows 100 \
  --transition-size 20 \
  --qbraid-shots 100
```

### Step 2: Calling API/QPU to obtain dataset using 10Qubit

```bash
QBRAID_API_KEY="..." OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python3 QRC_train_test_qbraid.py \
  --eval-feature-backend qbraid \
  --qbraid-device-id aws:rigetti:qpu:cepheus-1-108q \
  --num-qubits 10 \
  --eval-rows 100 \
  --transition-size 20 \
  --qbraid-shots 100
```

### Step 3: Using dataset QPU has obtained to train/adapt results from both 5Qubit and 10Qubit and visualize their 

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python3 QRC_main_qbraid.py \
  --skip-qrc-model1 \
  --eval-rows 100 \
  --transition-size 20 \
  --shots 100
```

Main outputs include

```text
results/connected_transition_summary.csv
results/phase3_summary.csv
results/phase3_5q_predictions.csv
results/phase3_5q_train_predictions.csv
results/phase3_10q_predictions.csv
results/phase3_10q_train_predictions.csv
```

# Target Variables

## Regression Target

The regression task predicts

```math 
y_{t+1}=\log(RV_{t+1}+\epsilon) 
```

which represents the next-day log realized variance.

---

## Classification Target

The classification task predicts whether the next trading day belongs to a high-volatility regime.

The regime label is defined as

```math
\text{HighVol}_{t+1}
=
\begin{cases}
1,& y_{t+1}>q_\alpha\\
0,& \text{otherwise}
\end{cases}
```

where $q_\alpha$ is the $\alpha$-quantile (typically 0.80–0.95) computed **only from the training window**.

This design makes the regime labels **leakage-safe**, since no future observations are used when defining the threshold.

---

# Volatility Proxy

Daily log return is computed as

```math
r_t
=
\log\left(\frac{P_t}{P_{t-1}}\right),
```

where $P_t$ is the daily closing price.

The realized variance proxy is

```math
RV_t=r_t^2.
```

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


---

# Intended Use

This dataset is designed for machine learning models that perform:

- Next-day volatility forecasting (regression)
- High-volatility regime prediction (classification)
- Hybrid forecasting models combining classical time-series methods with Quantum Reservoir Computing (QRC)