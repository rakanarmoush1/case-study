"""Model interpretation: SHAP values for the boosted champion and gain importance."""

from __future__ import annotations

import numpy as np
import pandas as pd


def transformed_frame(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Apply only the preprocessing step of a boosting pipeline."""
    return pipeline.named_steps["prep"].transform(X)


def shap_values(pipeline, X: pd.DataFrame) -> tuple[np.ndarray, float, pd.DataFrame]:
    """Return (shap_values [n, features], expected_value, transformed X) for class 1."""
    import shap

    clf = pipeline.named_steps["clf"]
    Xt = transformed_frame(pipeline, X)
    explainer = shap.TreeExplainer(clf)
    values = explainer.shap_values(Xt)
    if isinstance(values, list):  # older API: one array per class
        values = values[1]
    elif values.ndim == 3:
        values = values[:, :, 1]
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = float(np.asarray(base).ravel()[-1])
    return np.asarray(values), float(base), Xt


def global_importance(values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    imp = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": np.abs(values).mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False)
    imp["share"] = imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()
    return imp.reset_index(drop=True)


def gain_importance(pipeline, feature_names: list[str]) -> pd.DataFrame:
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "booster_"):
        gain = clf.booster_.feature_importance(importance_type="gain")
    else:
        gain = clf.feature_importances_
    imp = pd.DataFrame({"feature": feature_names, "gain": gain}).sort_values(
        "gain", ascending=False
    )
    imp["share"] = imp["gain"] / imp["gain"].sum()
    return imp.reset_index(drop=True)


def local_examples(
    values: np.ndarray,
    Xt: pd.DataFrame,
    y: np.ndarray,
    p: np.ndarray,
    amount: np.ndarray,
    threshold: float,
    top_n: int = 5,
) -> list[dict]:
    """Pick one true positive, one false positive and one false negative and list their top drivers."""
    y = np.asarray(y)
    p = np.asarray(p)
    picks = {}
    tp = np.where((y == 1) & (p >= threshold))[0]
    fp = np.where((y == 0) & (p >= threshold))[0]
    fn = np.where((y == 1) & (p < threshold))[0]
    if len(tp):
        picks["true_positive"] = int(tp[np.argsort(-p[tp])[len(tp) // 2]])
    if len(fp):
        picks["false_positive"] = int(fp[np.argmax(p[fp])])
    if len(fn):
        picks["false_negative"] = int(fn[np.argmax(p[fn])])
    out = []
    for kind, i in picks.items():
        row = values[i]
        order = np.argsort(-np.abs(row))[:top_n]
        out.append(
            {
                "kind": kind,
                "index": i,
                "score": float(p[i]),
                "label": int(y[i]),
                "amount": float(amount[i]),
                "drivers": [
                    {
                        "feature": Xt.columns[j],
                        "value": _fmt(Xt.iloc[i, j]),
                        "shap": float(row[j]),
                    }
                    for j in order
                ],
            }
        )
    return out


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else round(float(v), 3)
    return str(v)
