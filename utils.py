
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import statsmodels.api as sm

# -----------------------------
# 3. HAR model function
# -----------------------------
def estimate_har_model(data):
    X = data[["y_d", "y_w", "y_m"]]
    y = data["y_next"]

    X = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, X).fit()
    return model

def qlike(y_true_log, y_pred_log):
    r = np.exp(y_true_log) / np.exp(y_pred_log)
    return np.mean(r - np.log(r) - 1)


def plot_ticker(g, tick, outdir="figs"):
    import os; os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(g["date"], g["y_true"], color="steelblue", alpha=0.55, lw=1.0,
            label="Actual (log RV)")
    ax.plot(g["date"], g["y_har"], color="darkorange", lw=1.2,
            label=f"HAR  (QLIKE {qlike(g.y_true, g.y_har):.3f})")
    ax.plot(g["date"], g["y_qrc"], color="seagreen", lw=1.2, alpha=0.9,
            label=f"HAR+QRC  (QLIKE {qlike(g.y_true, g.y_qrc):.3f})")

    ax.set_ylabel("log RV")
    ax.set_xlabel("Date")
    ax.set_title(f"{tick} — test set forecasts")
    ax.legend(loc="upper left", fontsize=10)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(f"{outdir}/HAR_vs_QRC_{tick}.png", dpi=150)
    plt.close(fig)