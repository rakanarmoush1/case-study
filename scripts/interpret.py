"""Explain the saved model and check outcomes by customer group: SHAP global importance,
the amount dependence plot, three explained transactions (a true positive, the
highest-scoring false positive and the highest-scoring miss) and alert, recall and
false-positive rates by gender and age band at the saved threshold. Script version
of notebooks/03_interpretation_fairness_business.ipynb.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fraud_detection import plots
from fraud_detection.config import PROJECT_ROOT, PipelineConfig
from fraud_detection.data import load_clean
from fraud_detection.evaluate import (
    amount_band,
    bootstrap_ci,
    segment_performance,
    stability_over_time,
)
from fraud_detection.fairness import group_outcomes
from fraud_detection.features import build_features
from fraud_detection.interpret import global_importance, local_examples, shap_values


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "fraud.csv")
    ap.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / "production_model.joblib",
    )
    ap.add_argument(
        "--sample",
        type=int,
        default=20000,
        help="Hold-out rows used for the global SHAP figures",
    )
    ap.add_argument(
        "--figures", type=Path, default=PROJECT_ROOT / "reports" / "figures"
    )
    args = ap.parse_args()
    pd.set_option("display.width", 160)

    cfg = PipelineConfig()
    bundle = joblib.load(args.model)
    model, features, thr = bundle["model"], bundle["features"], bundle["threshold"]
    df = build_features(load_clean(args.data))
    test = df[df["step"] >= cfg.split.test_start_step].reset_index(drop=True)
    y, amount = test["fraud"].to_numpy(), test["amount"].to_numpy()
    p = model.predict_proba(test[features])[:, 1]
    print(
        f"{bundle['family']} with {len(features)} features, threshold {thr:.2f}; hold-out alerts {(p >= thr).sum():,} of {len(test):,}"
    )

    rng = np.random.default_rng(cfg.random_state)
    idx = rng.choice(len(test), min(args.sample, len(test)), replace=False)
    vals, base, Xt = shap_values(model, test.iloc[idx][features])
    imp = global_importance(vals, list(Xt.columns))
    print("\n== global importance (share of mean |SHAP|) ==")
    print(imp.head(12).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    args.figures.mkdir(parents=True, exist_ok=True)
    plots.shap_bar(imp, args.figures / "interp_shap_bar.png")
    plots.shap_summary(vals, Xt, args.figures / "interp_shap_summary.png")
    plots.shap_dependence(
        vals, Xt, "amount", args.figures / "interp_shap_dependence_amount.png"
    )

    vals_all, _, Xt_all = shap_values(model, test[features])
    print("\n== explained transactions ==")
    for ex in local_examples(vals_all, Xt_all, y, p, amount, thr):
        print(
            f"{ex['kind']:15s} score {ex['score']:.3f}  label {ex['label']}  amount {ex['amount']:,.2f}"
        )
        for drv in ex["drivers"]:
            print(
                f"    {drv['feature']:26s} = {drv['value']!s:14s} shap {drv['shap']:+.2f}"
            )
        plots.local_explanation(
            ex, base, args.figures / f"interp_local_{ex['kind']}.png"
        )

    print("\n== outcomes by group at the saved threshold ==")
    for col in ["gender", "age"]:
        tbl = group_outcomes(test[col], y, p, thr)
        print(tbl.to_string(index=False, float_format=lambda v: f"{v:.4f}"), "\n")
        plots.group_outcomes_plot(
            tbl,
            args.figures / f"fair_{col}_outcomes.png",
            f"Outcomes by {col} at the production threshold (model without protected attributes)",
        )
    print(
        "== where the model is weak: performance by segment at the saved threshold =="
    )
    cols = [
        "segment",
        "n",
        "fraud",
        "alerts",
        "precision",
        "recall",
        "false_positive_rate",
        "value_recall",
    ]
    print(
        segment_performance(test["category"], y, p, amount, thr)[cols]
        .head(10)
        .to_string(index=False, float_format=lambda v: f"{v:.3f}"),
        "\n",
    )
    print(
        segment_performance(amount_band(amount), y, p, amount, thr)[cols].to_string(
            index=False, float_format=lambda v: f"{v:.3f}"
        ),
        "\n",
    )

    stab = stability_over_time(test["step"], y, p, thr)
    print(f"== stability over the {len(stab)} hold-out steps ==")
    print(
        f"precision {stab.precision.min():.2f}-{stab.precision.max():.2f}, recall {stab.recall.min():.2f}-{stab.recall.max():.2f}, alert rate {stab.alert_rate.min():.2%}-{stab.alert_rate.max():.2%}"
    )
    plots.stability(stab, args.figures / "prod_stability.png", thr)

    ci = bootstrap_ci(y, p, amount, thr, n_boot=200, seed=cfg.random_state)
    print("\n== 95% bootstrap intervals on the hold-out ==")
    for k in ("roc_auc", "average_precision", "precision", "recall", "value_recall"):
        print(f"{k:18s} {ci[k]['low']:.3f} - {ci[k]['high']:.3f}")
    print(f"figures saved to {args.figures}")


if __name__ == "__main__":
    main()
