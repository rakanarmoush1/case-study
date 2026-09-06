"""Train the production fraud model and save it for scoring.

Steps: load and clean the data, build point-in-time features, split by time, fit the
boosted model (tuned parameters from reports/metrics/tuning.json when available),
choose the alert threshold on validation folds with the cost model, report hold-out
performance and save the model bundle.

Example:
    python scripts/train.py --data data/fraud.csv --out models/production_model.joblib
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fraud_detection.config import PROJECT_ROOT, PipelineConfig
from fraud_detection.data import load_clean
from fraud_detection.evaluate import (
    choose_threshold,
    decision_metrics,
    ranking_metrics,
    threshold_sweep,
)
from fraud_detection.features import build_features, feature_columns
from fraud_detection.models import make_model, positive_scale, save_bundle
from fraud_detection.monitoring import build_reference
from fraud_detection.split import temporal_split, time_series_folds


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "fraud.csv")
    ap.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "models" / "production_model.joblib"
    )
    ap.add_argument("--family", choices=["xgb", "lgbm"], default="xgb")
    ap.add_argument(
        "--include-protected",
        action="store_true",
        help="Use age and gender as inputs (off by default)",
    )
    ap.add_argument(
        "--quick", action="store_true", help="Fewer boosting rounds, for a fast check"
    )
    args = ap.parse_args()

    cfg = PipelineConfig(quick=args.quick)
    df = build_features(load_clean(args.data))
    train, test = temporal_split(df, cfg.split)
    features = feature_columns(
        include_protected=args.include_protected, include_time=False
    )
    y = train["fraud"].to_numpy()
    spw = positive_scale(y)
    print(
        f"train steps {train.step.min()}-{train.step.max()} ({len(train):,} rows), hold-out steps {test.step.min()}-{test.step.max()} ({len(test):,} rows)"
    )

    params: dict = {}
    tuning_path = cfg.metrics_dir / "tuning.json"
    if tuning_path.exists():
        saved = json.loads(tuning_path.read_text())
        if saved.get("family") == args.family:
            params = saved["best_params"]
            print(f"using tuned parameters for {args.family} from {tuning_path.name}")

    # Alert threshold from out-of-fold predictions on the expanding-window folds.
    oof = np.full(len(train), np.nan)
    for tr_idx, va_idx in time_series_folds(train["step"], cfg.split):
        model = make_model(
            args.family, features, spw, cfg.random_state, cfg.n_jobs, args.quick, params
        )
        model.fit(train.iloc[tr_idx][features], y[tr_idx])
        oof[va_idx] = model.predict_proba(train.iloc[va_idx][features])[:, 1]
    mask = ~np.isnan(oof)
    sweep = threshold_sweep(
        y[mask],
        oof[mask],
        train["amount"].to_numpy()[mask],
        cfg.costs,
        int(train["step"][mask].nunique()),
    )
    threshold = choose_threshold(sweep, cfg.costs, "net_benefit")
    threshold_capacity = choose_threshold(sweep, cfg.costs, "capacity")
    print(
        f"threshold {threshold:.2f} (max net benefit on validation); capacity policy would use {threshold_capacity:.2f}"
    )

    # Final fit on the whole training window, scored once on the hold-out.
    model = make_model(
        args.family, features, spw, cfg.random_state, cfg.n_jobs, args.quick, params
    )
    model.fit(train[features], y)
    p = model.predict_proba(test[features])[:, 1]
    m = ranking_metrics(test["fraud"].to_numpy(), p)
    d = decision_metrics(
        test["fraud"].to_numpy(),
        p,
        test["amount"].to_numpy(),
        threshold,
        cfg.costs,
        int(test["step"].nunique()),
    )
    print(
        f"hold-out: ROC-AUC {m['roc_auc']:.4f}, average precision {m['average_precision']:.3f}"
    )
    print(
        f"at threshold {threshold:.2f}: precision {d['precision']:.1%}, recall {d['recall']:.1%}, {d['alerts_per_step']:.1f} alerts per step, fraud value stopped {d['value_recall']:.1%}"
    )

    save_bundle(
        args.out,
        model,
        features,
        threshold,
        threshold_capacity,
        args.family,
        metadata={
            "training_rows": len(train),
            "training_steps": [int(train.step.min()), int(train.step.max())],
            "holdout_steps": [int(test.step.min()), int(test.step.max())],
            "params": params,
            "holdout_metrics": {
                "roc_auc": m["roc_auc"],
                "average_precision": m["average_precision"],
                "precision": d["precision"],
                "recall": d["recall"],
                "alert_rate": d["alert_rate"],
                "value_recall": d["value_recall"],
            },
        },
    )
    recent = (
        train["step"] >= cfg.split.test_start_step - cfg.split.cv_fold_length
    ).to_numpy()
    reference = build_reference(
        train.loc[recent, features], features, oof[recent], threshold
    )
    (args.out.parent / "monitoring_reference.json").write_text(
        json.dumps(reference, indent=2)
    )
    print(
        f"saved {args.out}, {args.out.with_suffix('.json').name} and monitoring_reference.json"
    )


if __name__ == "__main__":
    main()
