"""Drift monitoring for a deployed model: population stability index (PSI) on the model
inputs and on the score distribution, plus the alert rate against what was expected.

A reference is built from the training window when the model is trained; a new batch is
compared against it with :func:`compare`. PSI below 0.1 is stable, 0.1 to 0.25 is worth
watching, above 0.25 means the population has moved and the model should be re-validated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import CATEGORICAL_FEATURES

PSI_WATCH = 0.10
PSI_ALERT = 0.25

# Features that take a single value per merchant (or category) and step. Row-level PSI on
# them is dominated by bin edges: a small shift in one large merchant's share moves a whole
# point mass across a bin. Their drift is a function of the merchant and category mix and of
# amounts, which are monitored directly, so they are excluded from the row-level check.
ENTITY_LEVEL_FEATURES = {
    "merch_share_last_24",
    "merch_mean_amount_last_24",
    "cat_mean_amount_last_24",
}


def _psi(ref: np.ndarray, cur: np.ndarray, eps: float = 1e-4) -> float:
    ref = np.clip(np.asarray(ref, dtype=float), eps, None)
    cur = np.clip(np.asarray(cur, dtype=float), eps, None)
    ref = ref / ref.sum()
    cur = cur / cur.sum()
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def _numeric_hist(x: pd.Series, edges: np.ndarray) -> np.ndarray:
    """Counts per reference bin, with values outside the range folded into the end bins and NaN as its own bin."""
    vals = x.to_numpy(dtype=float)
    nan = np.isnan(vals)
    v = np.clip(vals[~nan], edges[0], edges[-1])
    counts = np.histogram(v, bins=edges)[0].astype(float)
    return np.append(counts, float(nan.sum()))


def build_reference(
    X: pd.DataFrame,
    features: list[str],
    scores: np.ndarray,
    threshold: float,
    n_bins: int = 10,
) -> dict:
    """Summarise the training population: decile bins per numeric feature, level shares per categorical, score deciles."""
    ref: dict = {
        "n_rows": int(len(X)),
        "threshold": float(threshold),
        "features": {},
        "entity_level_features": sorted(
            f for f in features if f in ENTITY_LEVEL_FEATURES
        ),
    }
    for f in features:
        if f in ENTITY_LEVEL_FEATURES:
            continue
        if f in CATEGORICAL_FEATURES:
            counts = X[f].astype(str).value_counts()
            ref["features"][f] = {
                "type": "categorical",
                "levels": counts.index.tolist(),
                "ref": (counts / counts.sum()).tolist(),
            }
        else:
            edges = np.unique(
                np.nanquantile(
                    X[f].dropna().to_numpy(dtype=float), np.linspace(0, 1, n_bins + 1)
                )
            )
            hist = _numeric_hist(X[f], edges)
            ref["features"][f] = {
                "type": "numeric",
                "edges": edges.tolist(),
                "ref": (hist / hist.sum()).tolist(),
            }
    scores = np.asarray(scores, dtype=float)
    edges = np.linspace(0, 1, 11)
    hist = np.histogram(np.clip(scores, 0, 1), bins=edges)[0].astype(float)
    ref["score"] = {"edges": edges.tolist(), "ref": (hist / hist.sum()).tolist()}
    ref["expected_alert_rate"] = float((scores >= threshold).mean())
    return ref


def compare(
    X: pd.DataFrame, scores: np.ndarray, reference: dict
) -> tuple[pd.DataFrame, dict]:
    """PSI per input and for the score, and the batch alert rate versus the expected one."""
    rows = []
    for f, spec in reference["features"].items():
        if spec["type"] == "categorical":
            cur = X[f].astype(str).value_counts()
            counts = np.array(
                [cur.get(level, 0) for level in spec["levels"]], dtype=float
            )
            unseen = 1.0 - counts.sum() / max(len(X), 1)
            rows.append(
                {
                    "feature": f,
                    "psi": _psi(spec["ref"], counts),
                    "unseen_share": float(unseen),
                }
            )
        else:
            hist = _numeric_hist(X[f], np.asarray(spec["edges"], dtype=float))
            rows.append(
                {"feature": f, "psi": _psi(spec["ref"], hist), "unseen_share": 0.0}
            )
    score_hist = np.histogram(
        np.clip(np.asarray(scores, dtype=float), 0, 1),
        bins=np.asarray(reference["score"]["edges"]),
    )[0]
    rows.append(
        {
            "feature": "fraud_score",
            "psi": _psi(reference["score"]["ref"], score_hist),
            "unseen_share": 0.0,
        }
    )
    table = (
        pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    )
    table["status"] = np.where(
        table["psi"] >= PSI_ALERT,
        "ALERT",
        np.where(table["psi"] >= PSI_WATCH, "WATCH", "ok"),
    )
    alert_rate = float((np.asarray(scores) >= reference["threshold"]).mean())
    summary = {
        "rows": len(X),
        "alert_rate": alert_rate,
        "expected_alert_rate": reference["expected_alert_rate"],
        "alert_rate_ratio": alert_rate / max(reference["expected_alert_rate"], 1e-9),
        "n_alert": int((table["status"] == "ALERT").sum()),
        "n_watch": int((table["status"] == "WATCH").sum()),
    }
    return table, summary
