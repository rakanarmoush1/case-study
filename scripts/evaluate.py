"""Benchmark the model families on the temporal split and compare the best one with the
rules baseline. Thresholds are chosen on the validation folds inside the training
window and applied unchanged to the hold-out period. Script version of
notebooks/02_modelling_and_evaluation.ipynb; the full experiment (imbalance
treatments, ablation, tuning, SHAP) is `fraud-detection run --tune`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_detection import plots
from fraud_detection.config import PROJECT_ROOT, PipelineConfig
from fraud_detection.data import load_clean
from fraud_detection.evaluate import (
    choose_threshold,
    decision_metrics,
    ranking_metrics,
    threshold_sweep,
)
from fraud_detection.features import build_features, feature_columns
from fraud_detection.models import MODEL_SPECS, make_model, positive_scale
from fraud_detection.split import temporal_split, time_series_folds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "fraud.csv")
    ap.add_argument(
        "--families", nargs="+", default=list(MODEL_SPECS), choices=list(MODEL_SPECS)
    )
    ap.add_argument(
        "--quick", action="store_true", help="Smaller models, for a fast check"
    )
    ap.add_argument(
        "--figures", type=Path, default=PROJECT_ROOT / "reports" / "figures"
    )
    args = ap.parse_args()
    pd.set_option("display.width", 200)

    cfg = PipelineConfig(quick=args.quick)
    df = build_features(load_clean(args.data))
    train, test = temporal_split(df, cfg.split)
    features = feature_columns(include_protected=True, include_time=True)
    y_tr, y_te = train["fraud"].to_numpy(), test["fraud"].to_numpy()
    amt_tr, amt_te = train["amount"].to_numpy(), test["amount"].to_numpy()
    n_steps_te = int(test["step"].nunique())
    spw = positive_scale(y_tr)
    folds = list(time_series_folds(train["step"], cfg.split))
    val_mask = np.zeros(len(train), dtype=bool)
    for _, va in folds:
        val_mask[va] = True
    n_steps_val = int(train.loc[val_mask, "step"].nunique())

    rows, preds, decisions = [], {}, {}
    for key in args.families:
        oof = np.full(len(train), np.nan)
        cv_ap = []
        for tr_idx, va_idx in folds:
            m = make_model(key, features, spw, cfg.random_state, cfg.n_jobs, cfg.quick)
            m.fit(train.iloc[tr_idx][features], y_tr[tr_idx])
            oof[va_idx] = m.predict_proba(train.iloc[va_idx][features])[:, 1]
            cv_ap.append(
                ranking_metrics(y_tr[va_idx], oof[va_idx])["average_precision"]
            )
        sweep = threshold_sweep(
            y_tr[val_mask], oof[val_mask], amt_tr[val_mask], cfg.costs, n_steps_val
        )
        thr = choose_threshold(sweep, cfg.costs, "net_benefit")
        m = make_model(key, features, spw, cfg.random_state, cfg.n_jobs, cfg.quick)
        m.fit(train[features], y_tr)
        p = m.predict_proba(test[features])[:, 1]
        r = ranking_metrics(y_te, p)
        d = decision_metrics(y_te, p, amt_te, thr, cfg.costs, n_steps_te)
        preds[key], decisions[key] = p, d
        rows.append(
            {
                "model": MODEL_SPECS[key].name,
                "cv_ap": float(np.mean(cv_ap)),
                "test_ap": r["average_precision"],
                "roc_auc": r["roc_auc"],
                "threshold": thr,
                "precision": d["precision"],
                "recall": d["recall"],
                "alerts_per_step": d["alerts_per_step"],
                "value_stopped": d["value_recall"],
                "net_benefit_per_step": d["net_benefit_per_step"],
            }
        )
        print(
            f"{MODEL_SPECS[key].name:32s} cv AP {rows[-1]['cv_ap']:.3f}  test AP {r['average_precision']:.3f}  precision {d['precision']:.1%}  recall {d['recall']:.1%}  alerts/step {d['alerts_per_step']:.1f}"
        )

    table = pd.DataFrame(rows).set_index("model")
    print("\n== hold-out benchmark (thresholds chosen on validation) ==")
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))

    if "rules" in decisions and len(decisions) > 1:
        best = max(
            (k for k in decisions if k != "rules"),
            key=lambda k: table.loc[MODEL_SPECS[k].name, "cv_ap"],
        )
        r, d = decisions["rules"], decisions[best]
        k = 10000 / len(test)
        comp = pd.DataFrame(
            {
                "rules baseline": [
                    r["n_alerts"] * k,
                    r["fp"] * k,
                    r["tp"] * k,
                    r["fn"] * k,
                    r["recall"],
                    r["value_recall"],
                    r["precision"],
                    r["net_benefit"] * k,
                ],
                MODEL_SPECS[best].name: [
                    d["n_alerts"] * k,
                    d["fp"] * k,
                    d["tp"] * k,
                    d["fn"] * k,
                    d["recall"],
                    d["value_recall"],
                    d["precision"],
                    d["net_benefit"] * k,
                ],
            },
            index=[
                "alerts",
                "false declines",
                "frauds caught",
                "frauds missed",
                "recall",
                "fraud value stopped",
                "precision",
                "net benefit",
            ],
        )
        print(f"\n== per 10,000 transactions: rules vs {MODEL_SPECS[best].name} ==")
        print(comp.to_string(float_format=lambda v: f"{v:,.3f}"))

    args.figures.mkdir(parents=True, exist_ok=True)
    plots.pr_curves(
        preds,
        y_te,
        args.figures / "model_pr_curves.png",
        {k: MODEL_SPECS[k].name for k in preds},
    )
    print(f"\nPR curves saved to {args.figures / 'model_pr_curves.png'}")


if __name__ == "__main__":
    main()
