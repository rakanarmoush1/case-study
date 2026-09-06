"""Group-level outcome checks for protected attributes.

Fraud models are not credit decisions, but declining or delaying a customer's
payment is still an adverse action. We therefore report, per gender and age
band, how often customers are flagged, how often those flags are wrong and how
much fraud is caught, and we compare the champion with and without the
protected attributes as inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def group_outcomes(
    groups: pd.Series, y: np.ndarray, p: np.ndarray, threshold: float
) -> pd.DataFrame:
    y = np.asarray(y).astype(int)
    alert = np.asarray(p) >= threshold
    df = pd.DataFrame({"group": groups.astype(str).to_numpy(), "y": y, "alert": alert})
    rows = []
    for g, d in df.groupby("group"):
        tp = int((d.alert & (d.y == 1)).sum())
        fp = int((d.alert & (d.y == 0)).sum())
        fn = int((~d.alert & (d.y == 1)).sum())
        tn = int((~d.alert & (d.y == 0)).sum())
        rows.append(
            {
                "group": g,
                "n": len(d),
                "fraud_rate": float(d.y.mean()),
                "alert_rate": float(d.alert.mean()),
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(tp + fn, 1),
                "false_positive_rate": fp / max(fp + tn, 1),
            }
        )
    out = pd.DataFrame(rows)
    ref = out.loc[out["n"].idxmax(), "alert_rate"]
    out["alert_rate_ratio_vs_largest_group"] = out["alert_rate"] / max(ref, 1e-12)
    return out
