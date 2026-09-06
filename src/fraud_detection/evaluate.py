"""Metrics, threshold selection and the business cost model.

Two families of numbers are reported:

* **Ranking quality** (threshold-free): ROC-AUC and, more importantly for a 1%
  positive rate, average precision (area under the precision-recall curve).
* **Decision quality** at an operating point: precision, recall, alerts per
  step and a net-benefit figure computed from an explicit cost model.

The operating point is chosen on validation predictions (inside the training
window), never on the hold-out set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

from .config import CostAssumptions

DEFAULT_THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)


def ranking_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    precision, recall, thr = precision_recall_curve(y, p)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    out = {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "f1_at_0_5": float(f1_score(y, (p >= 0.5).astype(int), zero_division=0)),
        "best_f1": float(f1[best]),
        "best_f1_threshold": float(thr[best]) if len(thr) else 0.5,
    }
    for k in (0.5, 1.0, 2.0):
        out[f"recall_at_top_{k:g}pct"] = recall_at_top_k(y, p, k / 100)
        out[f"precision_at_top_{k:g}pct"] = precision_at_top_k(y, p, k / 100)
    out["precision_at_recall_0_9"] = precision_at_recall(y, p, 0.9)
    out["recall_at_precision_0_9"] = recall_at_precision(y, p, 0.9)
    for fpr in (0.001, 0.005, 0.01):
        out[f"recall_at_fpr_{fpr:g}"] = recall_at_fpr(y, p, fpr)
    return out


def _top_k_mask(p: np.ndarray, frac: float) -> np.ndarray:
    n = max(int(round(len(p) * frac)), 1)
    order = np.argsort(-p, kind="stable")
    mask = np.zeros(len(p), dtype=bool)
    mask[order[:n]] = True
    return mask


def recall_at_top_k(y, p, frac: float) -> float:
    y = np.asarray(y)
    mask = _top_k_mask(np.asarray(p), frac)
    return float(y[mask].sum() / max(y.sum(), 1))


def precision_at_top_k(y, p, frac: float) -> float:
    y = np.asarray(y)
    mask = _top_k_mask(np.asarray(p), frac)
    return float(y[mask].mean()) if mask.any() else 0.0


def precision_at_recall(y, p, target: float) -> float:
    precision, recall, _ = precision_recall_curve(y, p)
    ok = recall >= target
    return float(precision[ok].max()) if ok.any() else 0.0


def recall_at_precision(y, p, target: float) -> float:
    precision, recall, _ = precision_recall_curve(y, p)
    ok = precision >= target
    return float(recall[ok].max()) if ok.any() else 0.0


def decision_metrics(
    y: np.ndarray,
    p: np.ndarray,
    amount: np.ndarray,
    threshold: float,
    costs: CostAssumptions,
    n_steps: int,
) -> dict[str, float]:
    """Confusion counts and money at a given threshold."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    amount = np.asarray(amount, dtype=float)
    alert = p >= threshold
    tp = int((alert & (y == 1)).sum())
    fp = int((alert & (y == 0)).sum())
    fn = int((~alert & (y == 1)).sum())
    tn = int((~alert & (y == 0)).sum())
    fraud_value_total = float(amount[y == 1].sum())
    fraud_value_caught = float(amount[alert & (y == 1)].sum())
    fraud_value_missed = fraud_value_total - fraud_value_caught
    review_cost = float(alert.sum() * costs.review_cost)
    friction_cost = float(fp * costs.friction_cost)
    net = fraud_value_caught - review_cost - friction_cost
    n_alerts = int(alert.sum())
    return {
        "threshold": float(threshold),
        "n_alerts": n_alerts,
        "alerts_per_step": n_alerts / max(n_steps, 1),
        "alert_rate": n_alerts / len(y),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "false_positive_rate": fp / max(fp + tn, 1),
        "fraud_value_total": fraud_value_total,
        "fraud_value_caught": fraud_value_caught,
        "fraud_value_missed": fraud_value_missed,
        "value_recall": fraud_value_caught / max(fraud_value_total, 1e-9),
        "review_cost": review_cost,
        "friction_cost": friction_cost,
        "net_benefit": net,
        "net_benefit_per_step": net / max(n_steps, 1),
    }


def threshold_sweep(
    y, p, amount, costs: CostAssumptions, n_steps: int, thresholds=DEFAULT_THRESHOLDS
) -> pd.DataFrame:
    rows = [decision_metrics(y, p, amount, t, costs, n_steps) for t in thresholds]
    return pd.DataFrame(rows)


def choose_threshold(
    sweep: pd.DataFrame, costs: CostAssumptions, policy: str = "net_benefit"
) -> float:
    """Pick an operating point from a sweep computed on *validation* predictions.

    * ``net_benefit``: maximise expected net benefit under the cost model.
    * ``capacity``: the lowest threshold whose alert volume fits analyst capacity.
    """
    if policy == "net_benefit":
        return float(sweep.loc[sweep["net_benefit"].idxmax(), "threshold"])
    if policy == "capacity":
        ok = sweep[sweep["alerts_per_step"] <= costs.alerts_per_step_capacity]
        return (
            float(ok["threshold"].min())
            if not ok.empty
            else float(sweep["threshold"].max())
        )
    raise KeyError(policy)


def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    y = np.asarray(y)
    p = np.asarray(p)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    df = pd.DataFrame({"bin": idx, "y": y, "p": p})
    g = df.groupby("bin").agg(
        n=("y", "size"), observed=("y", "mean"), predicted=("p", "mean")
    )
    g["bin_low"] = bins[g.index]
    g["bin_high"] = bins[g.index + 1]
    return g.reset_index(drop=True)


def rules_from_tree(pipeline, feature_names: list[str] | None = None) -> str:
    """Render the rules baseline as readable if/then text (leaf weights = [benign, fraud] counts)."""
    from sklearn.tree import export_text

    prep = pipeline.named_steps["prep"]
    names = (
        list(prep.get_feature_names_out()) if feature_names is None else feature_names
    )
    names = [n.replace("category_", "category=") for n in names]
    return export_text(
        pipeline.named_steps["clf"], feature_names=names, decimals=2, show_weights=True
    )


# --------------------------------------------------------------------------- #
# Robustness views: fixed-FPR recall, bootstrap intervals, segments, stability, cost sensitivity
# --------------------------------------------------------------------------- #
AMOUNT_BANDS = [0, 50, 100, 250, 500, float("inf")]
AMOUNT_BAND_LABELS = ["0-50", "50-100", "100-250", "250-500", "500+"]


def recall_at_fpr(y, p, fpr_target: float) -> float:
    """Recall achievable while keeping the false-positive rate at or below ``fpr_target``."""
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y, p)
    ok = fpr <= fpr_target
    return float(tpr[ok].max()) if ok.any() else 0.0


def bootstrap_ci(
    y,
    p,
    amount,
    threshold: float,
    n_boot: int = 200,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Percentile bootstrap intervals for the headline hold-out metrics."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    amount = np.asarray(amount, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = {
        "roc_auc": [],
        "average_precision": [],
        "precision": [],
        "recall": [],
        "value_recall": [],
    }
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb, ab = y[idx], p[idx], amount[idx]
        if yb.sum() == 0:
            continue
        stats["roc_auc"].append(roc_auc_score(yb, pb))
        stats["average_precision"].append(average_precision_score(yb, pb))
        alert = pb >= threshold
        tp = (alert & (yb == 1)).sum()
        stats["precision"].append(tp / max(alert.sum(), 1))
        stats["recall"].append(tp / max(yb.sum(), 1))
        stats["value_recall"].append(
            ab[alert & (yb == 1)].sum() / max(ab[yb == 1].sum(), 1e-9)
        )
    out = {}
    for k, vals in stats.items():
        lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[k] = {"low": float(lo), "high": float(hi)}
    out["n_boot"] = n_boot
    return out


def amount_band(amount) -> pd.Series:
    return pd.cut(
        pd.Series(np.asarray(amount, dtype=float)),
        bins=AMOUNT_BANDS,
        labels=AMOUNT_BAND_LABELS,
        right=False,
        include_lowest=True,
    )


def segment_performance(segments, y, p, amount, threshold: float) -> pd.DataFrame:
    """Alert rate, precision, recall, false-positive rate and value recall per segment at a threshold."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    amount = np.asarray(amount, dtype=float)
    seg = pd.Series(np.asarray(segments).astype(str))
    alert = p >= threshold
    rows = []
    for name, idx in seg.groupby(seg, observed=True).groups.items():
        idx = np.asarray(idx)
        yy, aa, am = y[idx], alert[idx], amount[idx]
        tp = int((aa & (yy == 1)).sum())
        fp = int((aa & (yy == 0)).sum())
        fn = int((~aa & (yy == 1)).sum())
        tn = int((~aa & (yy == 0)).sum())
        rows.append(
            {
                "segment": name,
                "n": len(idx),
                "fraud": int(yy.sum()),
                "fraud_rate": float(yy.mean()),
                "alerts": int(aa.sum()),
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(tp + fn, 1),
                "false_positive_rate": fp / max(fp + tn, 1),
                "value_recall": float(
                    am[aa & (yy == 1)].sum() / max(am[yy == 1].sum(), 1e-9)
                )
                if yy.sum()
                else float("nan"),
                "fraud_value_missed": float(am[~aa & (yy == 1)].sum()),
            }
        )
    return (
        pd.DataFrame(rows).sort_values("fraud", ascending=False).reset_index(drop=True)
    )


def stability_over_time(steps, y, p, threshold: float) -> pd.DataFrame:
    """Per-step alert rate, precision and recall on a scored period."""
    df = pd.DataFrame(
        {
            "step": np.asarray(steps),
            "y": np.asarray(y).astype(int),
            "alert": np.asarray(p) >= threshold,
        }
    )
    rows = []
    for step, d in df.groupby("step"):
        tp = int((d.alert & (d.y == 1)).sum())
        rows.append(
            {
                "step": int(step),
                "n": len(d),
                "fraud": int(d.y.sum()),
                "alerts": int(d.alert.sum()),
                "alert_rate": float(d.alert.mean()),
                "precision": tp / max(int(d.alert.sum()), 1),
                "recall": tp / max(int(d.y.sum()), 1),
            }
        )
    return pd.DataFrame(rows)


COST_SCENARIOS = [
    ("base", 5.0, 10.0),
    ("cheap review", 2.0, 10.0),
    ("expensive review", 15.0, 10.0),
    ("high friction", 5.0, 50.0),
    ("expensive review, high friction", 15.0, 50.0),
]


def cost_sensitivity(
    y_val,
    amount_val,
    n_steps_val: int,
    y_test,
    amount_test,
    n_steps_test: int,
    oof: dict[str, np.ndarray],
    test_preds: dict[str, np.ndarray],
    base: CostAssumptions,
    scenarios=COST_SCENARIOS,
) -> pd.DataFrame:
    """Re-choose each model's threshold on validation under different cost assumptions and score the hold-out.

    ``oof`` and ``test_preds`` map model keys to validation and hold-out predictions.
    """
    rows = []
    for name, review, friction in scenarios:
        costs = CostAssumptions(
            review_cost=review,
            friction_cost=friction,
            alerts_per_step_capacity=base.alerts_per_step_capacity,
        )
        row = {"scenario": name, "review_cost": review, "friction_cost": friction}
        for key in oof:
            sweep = threshold_sweep(y_val, oof[key], amount_val, costs, n_steps_val)
            thr = choose_threshold(sweep, costs, "net_benefit")
            d = decision_metrics(
                y_test, test_preds[key], amount_test, thr, costs, n_steps_test
            )
            row[f"{key}_threshold"] = thr
            row[f"{key}_recall"] = d["recall"]
            row[f"{key}_alerts_per_step"] = d["alerts_per_step"]
            row[f"{key}_net_benefit"] = d["net_benefit"]
        rows.append(row)
    return pd.DataFrame(rows)
