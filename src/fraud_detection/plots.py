"""All figures for the notebooks and the report, in one consistent style."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

GREEN = "#86BC25"
DARK_GREEN = "#046A38"
BLUE = "#0076A8"
GREY = "#75787B"
LIGHT_GREY = "#D0D0CE"
BLACK = "#1A1A1A"
RED = "#DA291C"
AMBER = "#ED8B00"
SERIES = [GREEN, BLUE, DARK_GREEN, GREY, AMBER, RED]

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": GREY,
        "axes.grid": True,
        "grid.color": LIGHT_GREY,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _pretty(cat: str) -> str:
    return (
        cat.replace("es_", "")
        .replace("andrestaurants", " & restaurants")
        .replace("andtoys", " & toys")
        .replace("andbeauty", " & beauty")
        .replace("hotelservices", "hotel services")
        .replace("otherservices", "other services")
    )


# ----------------------------------------------------------------------------- EDA
def class_balance(df: pd.DataFrame, path: Path) -> Path:
    counts = df["fraud"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars = ax.bar(
        ["Benign (0)", "Fraud (1)"], counts.values, color=[GREY, GREEN], width=0.55
    )
    for b, v in zip(bars, counts.values, strict=True):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v,
            f"{v:,}\n({v / counts.sum():.2%})",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylabel("Transactions")
    ax.set_title("Fraud is 1.2% of transactions: 1 in 83")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


def fraud_by_category(df: pd.DataFrame, path: Path) -> Path:
    g = (
        df.groupby("category", observed=True)["fraud"]
        .agg(["size", "mean", "sum"])
        .sort_values("mean")
    )
    g.index = [_pretty(c) for c in g.index]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    ax1.barh(g.index, g["size"] / len(df) * 100, color=GREY)
    ax1.set_xlabel("Share of all transactions (%)")
    ax1.set_title("Where the volume is")
    ax1.set_xscale("log")
    ax1.grid(axis="y", visible=False)
    colors = [GREEN if v > 0.05 else LIGHT_GREY for v in g["mean"]]
    ax2.barh(g.index, g["mean"] * 100, color=colors)
    for i, (rate, n) in enumerate(zip(g["mean"], g["sum"], strict=True)):
        ax2.text(
            rate * 100 + 1,
            i,
            f"{rate:.1%}  (n={int(n):,})",
            va="center",
            fontsize=9,
            color=BLACK,
        )
    ax2.set_xlabel("Fraud rate (%)")
    ax2.set_title("Where the fraud is")
    ax2.set_xlim(0, 118)
    ax2.grid(axis="y", visible=False)
    fig.suptitle(
        "Fraud concentrates in a handful of low-volume categories; 85% of volume (transportation) has none",
        fontsize=12,
        fontweight="bold",
    )
    return _save(fig, path)


def amount_distribution(df: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.logspace(-1, 4, 60)
    for label, color, name in [(0, GREY, "Benign"), (1, GREEN, "Fraud")]:
        ax.hist(
            df.loc[df.fraud == label, "amount"].clip(lower=0.1),
            bins=bins,
            density=True,
            alpha=0.75,
            color=color,
            label=name,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Transaction amount (log scale)")
    ax.set_ylabel("Density")
    med0 = df.loc[df.fraud == 0, "amount"].median()
    med1 = df.loc[df.fraud == 1, "amount"].median()
    ax.axvline(med0, color=GREY, ls="--", lw=1)
    ax.axvline(med1, color=DARK_GREEN, ls="--", lw=1)
    ax.set_title(
        f"Fraudulent transactions are about {med1 / med0:.0f}x larger (median {med1:,.0f} vs {med0:,.0f})"
    )
    ax.legend()
    return _save(fig, path)


def fraud_over_time(df: pd.DataFrame, path: Path) -> Path:
    g = df.groupby("step")["fraud"].agg(["size", "sum"])
    g["rate"] = g["sum"] / g["size"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(g.index, g["size"], color=GREY, lw=1.6, label="Transactions per step")
    ax.set_ylabel("Transactions per step", color=GREY)
    ax.set_xlabel("Step")
    ax2 = ax.twinx()
    ax2.plot(g.index, g["sum"], color=GREEN, lw=2, label="Fraud per step")
    ax2.set_ylabel("Fraud per step", color=DARK_GREEN)
    ax2.set_ylim(0, g["sum"].max() * 1.6)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax.set_title(
        f"Transactions per step grow while fraud stays at {int(g['sum'].iloc[0])} per step" if g["sum"].nunique() == 1 else "Transactions and fraud per step"
    )
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right")
    return _save(fig, path)


def hour_of_day(df: pd.DataFrame, path: Path) -> Path:
    g = df.assign(h=df.step % 24).groupby("h")["fraud"].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(g.index, g.values, color=LIGHT_GREY, edgecolor=GREY)
    ax.axhline(df.fraud.mean() * 100, color=GREEN, lw=2, label="Overall fraud rate")
    ax.set_xlabel("Hour of day (step mod 24, per the brief's definition of step)")
    ax.set_ylabel("Fraud rate (%)")
    ax.set_ylim(0, g.max() * 1.5)
    ax.set_title("Time-of-day carries no signal: fraud rate is flat across hours")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


def customer_persistence(df: pd.DataFrame, path: Path) -> Path:
    d = df.sort_values(["customer", "step"]).copy()
    d["prev_fraud"] = d.groupby("customer", observed=True)["fraud"].shift(1)
    d = d.dropna(subset=["prev_fraud"])
    rates = d.groupby("prev_fraud")["fraud"].mean() * 100
    fig, ax = plt.subplots(figsize=(6, 3.8))
    bars = ax.bar(
        ["Previous transaction benign", "Previous transaction fraud"],
        [rates[0.0], rates[1.0]],
        color=[GREY, GREEN],
        width=0.55,
    )
    for b, v in zip(bars, [rates[0.0], rates[1.0]], strict=True):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v,
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_ylabel("Fraud rate of the next transaction (%)")
    ax.set_title(
        f"Fraud persists on a compromised account: {rates[1.0] / rates[0.0]:.0f}x higher risk after a fraud"
    )
    ax.set_ylim(0, rates[1.0] * 1.25)
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


def fraud_by_group(df: pd.DataFrame, col: str, path: Path, title: str) -> Path:
    g = df.groupby(col, observed=True)["fraud"].agg(["size", "mean"])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(g.index.astype(str), g["mean"] * 100, color=GREY)
    for i, (n, r) in enumerate(zip(g["size"], g["mean"], strict=True)):
        ax.text(i, r * 100, f"{r:.2%}\nn={n:,}", ha="center", va="bottom", fontsize=8.5)
    ax.set_ylabel("Fraud rate (%)")
    ax.set_xlabel(col)
    ax.set_ylim(0, (g["mean"].max() * 100) * 1.45)
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


# ----------------------------------------------------------------------------- model performance
def pr_curves(
    preds: dict[str, np.ndarray],
    y: np.ndarray,
    path: Path,
    names: dict[str, str] | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5.2))
    for (key, p), color in zip(preds.items(), SERIES, strict=False):
        precision, recall, _ = precision_recall_curve(y, p)
        from sklearn.metrics import average_precision_score

        ap = average_precision_score(y, p)
        label = f"{(names or {}).get(key, key)} (AP={ap:.3f})"
        ax.plot(
            recall,
            precision,
            lw=2 if key in ("lgbm", "xgb") else 1.5,
            color=color,
            label=label,
        )
    ax.axhline(
        y.mean(),
        color=LIGHT_GREY,
        ls="--",
        lw=1,
        label=f"No-skill (fraud rate {y.mean():.1%})",
    )
    ax.set_xlabel("Recall (share of fraud caught)")
    ax.set_ylabel("Precision (share of alerts that are fraud)")
    ax.set_title("Precision-recall on the hold-out period (steps 144-179)")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", fontsize=9)
    return _save(fig, path)


def roc_curves(
    preds: dict[str, np.ndarray],
    y: np.ndarray,
    path: Path,
    names: dict[str, str] | None = None,
) -> Path:
    from sklearn.metrics import roc_auc_score

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for (key, p), color in zip(preds.items(), SERIES, strict=False):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(
            fpr,
            tpr,
            lw=1.8,
            color=color,
            label=f"{(names or {}).get(key, key)} (AUC={roc_auc_score(y, p):.4f})",
        )
    ax.plot([0, 1], [0, 1], color=LIGHT_GREY, ls="--", lw=1)
    ax.set_xlim(0, 0.1)
    ax.set_xlabel("False positive rate (zoomed to 0-10%)")
    ax.set_ylabel("True positive rate")
    ax.set_title(
        "ROC (zoomed): all models look near-perfect, which is why ROC-AUC is not the metric to manage by"
    )
    ax.legend(loc="lower right", fontsize=9)
    return _save(fig, path)


def model_comparison(
    table: pd.DataFrame, path: Path, metric: str = "average_precision"
) -> Path:
    """table: index=model name, columns include cv_{metric}_mean, cv_{metric}_std, test_{metric}."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(table))
    w = 0.38
    ax.bar(
        x - w / 2,
        table[f"cv_{metric}_mean"],
        w,
        yerr=table[f"cv_{metric}_std"],
        color=GREY,
        capsize=3,
        label="Time-series CV (mean +/- sd)",
    )
    ax.bar(
        x + w / 2,
        table[f"test_{metric}"],
        w,
        color=GREEN,
        label="Hold-out (steps 144-179)",
    )
    for i, v in enumerate(table[f"test_{metric}"]):
        ax.text(i + w / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [textwrap.fill(str(i), 16) for i in table.index], rotation=0, fontsize=9
    )
    ax.set_ylabel("Average precision (PR-AUC)")
    ax.set_ylim(0, 1.08)
    ax.set_title("Average precision by model: time-series CV vs hold-out")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=9)
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


def cost_curve(
    sweep: pd.DataFrame, threshold: float, path: Path, capacity: int | None = None
) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.plot(
        sweep["threshold"],
        sweep["net_benefit_per_step"],
        color=GREEN,
        lw=2.2,
        label="Net benefit per step",
    )
    ax.axvline(
        threshold,
        color=BLACK,
        ls="--",
        lw=1.2,
        label=f"Chosen threshold = {threshold:.2f}",
    )
    ax.set_xlabel("Score threshold for raising an alert")
    ax.set_ylabel("Net benefit per step", color=DARK_GREEN)
    ax2 = ax.twinx()
    ax2.plot(
        sweep["threshold"],
        sweep["alerts_per_step"],
        color=GREY,
        lw=1.6,
        label="Alerts per step",
    )
    if capacity:
        ax2.axhline(
            capacity,
            color=AMBER,
            ls=":",
            lw=1.5,
            label=f"Analyst capacity ({capacity}/step)",
        )
    ax2.set_ylabel("Alerts per step", color=GREY)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(
        h1 + h2,
        l1 + l2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        fontsize=9,
        ncol=4,
    )
    ax.set_title("Net benefit and alert volume by score threshold (hold-out period)")
    return _save(fig, path)


def confusion(dm: dict, path: Path, n_steps: int) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    mat = np.array([[dm["tn"], dm["fp"]], [dm["fn"], dm["tp"]]])
    colors = [[LIGHT_GREY, RED], [AMBER, GREEN]]
    for i in range(2):
        for j in range(2):
            ax.add_patch(
                plt.Rectangle((j, 1 - i), 1, 1, color=colors[i][j], alpha=0.85)
            )
            ax.text(
                j + 0.5,
                1 - i + 0.62,
                f"{mat[i, j]:,}",
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color="white" if (i, j) != (0, 0) else BLACK,
            )
            ax.text(
                j + 0.5,
                1 - i + 0.3,
                f"{mat[i, j] / n_steps:,.1f} per step",
                ha="center",
                va="center",
                fontsize=9.5,
                color="white" if (i, j) != (0, 0) else BLACK,
            )
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Predicted benign", "Predicted fraud (alert)"])
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Actual fraud", "Actual benign"])
    ax.grid(False)
    ax.set_title(
        f"Hold-out outcomes at threshold {dm['threshold']:.2f}: precision {dm['precision']:.0%}, recall {dm['recall']:.0%}"
    )
    return _save(fig, path)


def calibration(table: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(
        [0, 1], [0, 1], color=LIGHT_GREY, ls="--", lw=1, label="Perfect calibration"
    )
    ax.plot(
        table["predicted"],
        table["observed"],
        marker="o",
        color=GREEN,
        lw=2,
        label="Champion model",
    )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraud rate")
    ax.set_title("Calibration on the hold-out period")
    ax.legend(loc="upper left")
    return _save(fig, path)


def strategy_comparison(
    table: pd.DataFrame,
    path: Path,
    title: str,
    ylabel: str = "Average precision (hold-out)",
    metric: str = "average_precision",
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [GREEN if i == table[metric].idxmax() else GREY for i in table.index]
    ax.bar(table.index.astype(str), table[metric], color=colors, width=0.6)
    for i, v in enumerate(table[metric]):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=10)
    lo = max(0, table[metric].min() - 0.08)
    ax.set_ylim(lo, min(1.0, table[metric].max() + 0.05))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=9)
    return _save(fig, path)


# ----------------------------------------------------------------------------- interpretation
def shap_summary(
    values: np.ndarray, Xt: pd.DataFrame, path: Path, max_display: int = 15
) -> Path:
    import shap

    fig = plt.figure(figsize=(9, 6.5))
    Xnum = Xt.copy()
    for c in Xnum.columns:
        if str(Xnum[c].dtype) == "category":
            Xnum[c] = Xnum[c].cat.codes.astype(float)
    shap.summary_plot(
        values,
        Xnum,
        max_display=max_display,
        show=False,
        color_bar=True,
        plot_size=None,
    )
    plt.gcf().axes[0].set_title(
        "What drives the fraud score (SHAP, hold-out sample)", fontweight="bold"
    )
    plt.gca().grid(False)
    return _save(fig, path)


def shap_bar(imp: pd.DataFrame, path: Path, top: int = 15) -> Path:
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(
        d["feature"],
        d["share"] * 100,
        color=[GREEN if i < 3 else GREY for i in range(len(d))][::-1],
    )
    for i, v in enumerate(d["share"] * 100):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("Share of total |SHAP| (%)")
    ax.set_title("Feature importance: share of mean |SHAP| (hold-out sample)")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def shap_dependence(
    values: np.ndarray,
    Xt: pd.DataFrame,
    feature: str,
    path: Path,
    color_by: str = "category",
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    j = list(Xt.columns).index(feature)
    x = Xt[feature].to_numpy(dtype=float)
    cats = Xt[color_by].astype(str)
    top = cats.value_counts().index[:6]
    for c, color in zip(top, SERIES, strict=False):
        m = (cats == c).to_numpy()
        ax.scatter(x[m], values[m, j], s=6, alpha=0.5, color=color, label=_pretty(c))
    ax.set_xscale("log")
    ax.set_xlabel(f"{feature} (log scale)")
    ax.set_ylabel(f"SHAP value for {feature}")
    ax.set_title(f"How {feature} moves the score, by {color_by}")
    ax.legend(markerscale=3, fontsize=9, loc="upper left")
    return _save(fig, path)


def local_explanation(example: dict, base_value: float, path: Path) -> Path:
    d = example["drivers"][::-1]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    vals = [x["shap"] for x in d]
    labels = [f"{x['feature']} = {x['value']}" for x in d]
    ax.barh(labels, vals, color=[GREEN if v > 0 else RED for v in vals])
    ax.axvline(0, color=BLACK, lw=0.8)
    ax.set_xlabel("Contribution to log-odds of fraud")
    kind = example["kind"].replace("_", " ")
    ax.set_title(
        f"{kind.title()}: score {example['score']:.2f}, amount {example['amount']:,.2f}, label {example['label']}"
    )
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


# ----------------------------------------------------------------------------- fairness
def group_outcomes_plot(table: pd.DataFrame, path: Path, title: str) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, col, label, color in zip(
        axes,
        ["alert_rate", "false_positive_rate", "recall"],
        ["Alert rate", "False positive rate", "Recall (fraud caught)"],
        [GREY, RED, GREEN],
        strict=True,
    ):
        ax.bar(table["group"].astype(str), table[col] * 100, color=color, width=0.6)
        for i, v in enumerate(table[col] * 100):
            ax.text(
                i,
                v,
                f"{v:.2f}%" if col != "recall" else f"{v:.0f}%",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
        ax.set_title(label, fontsize=11)
        ax.set_ylabel("%")
        ax.set_ylim(0, (table[col].max() * 100) * 1.3 + 1e-9)
        ax.grid(axis="x", visible=False)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    return _save(fig, path)


def stability(table: pd.DataFrame, path: Path, threshold: float) -> Path:
    """Per-step precision, recall and alert rate over the hold-out period."""
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(
        table["step"],
        table["recall"] * 100,
        color=GREEN,
        lw=2,
        marker="o",
        ms=3,
        label="Recall",
    )
    ax.plot(
        table["step"],
        table["precision"] * 100,
        color=BLUE,
        lw=2,
        marker="o",
        ms=3,
        label="Precision",
    )
    ax.set_ylabel("%")
    ax.set_xlabel("Step (hold-out period)")
    ax.set_ylim(0, 105)
    ax2 = ax.twinx()
    ax2.bar(
        table["step"],
        table["alert_rate"] * 100,
        color=LIGHT_GREY,
        width=0.8,
        label="Alert rate (%)",
    )
    ax2.set_ylabel("Alert rate (%)", color=GREY)
    ax2.set_ylim(0, max(table["alert_rate"].max() * 100 * 3, 1))
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=9, ncol=3)
    ax.set_title(
        f"Step-by-step performance at threshold {threshold:.2f} over the hold-out period"
    )
    return _save(fig, path)
