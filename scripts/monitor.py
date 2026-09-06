"""Check a new batch of transactions for drift against the training reference.

Scores the batch with the saved model, then reports the population stability index (PSI)
of every model input and of the score distribution, plus the alert rate against what the
model produced on validation. PSI above 0.25 means the population has moved and the model
should be re-validated; 0.10 to 0.25 is worth watching.

Example:
    python scripts/monitor.py --input data/fraud.csv --min-step 144
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from fraud_detection.config import PROJECT_ROOT
from fraud_detection.data import load_clean
from fraud_detection.features import build_features
from fraud_detection.monitoring import compare


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Transactions CSV to check (with each customer's history)",
    )
    ap.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / "production_model.joblib",
    )
    ap.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Defaults to monitoring_reference.json next to the model",
    )
    ap.add_argument(
        "--min-step",
        type=int,
        default=None,
        help="Only check rows from this step on; earlier rows feed the history features",
    )
    ap.add_argument(
        "--max-step", type=int, default=None, help="Only check rows up to this step"
    )
    ap.add_argument("--top", type=int, default=12, help="How many inputs to list")
    args = ap.parse_args()
    pd.set_option("display.width", 140)

    bundle = joblib.load(args.model)
    reference = json.loads(
        (args.reference or args.model.parent / "monitoring_reference.json").read_text()
    )
    df = build_features(load_clean(args.input))
    if args.min_step is not None:
        df = df[df["step"] >= args.min_step]
    if args.max_step is not None:
        df = df[df["step"] <= args.max_step]
    scores = bundle["model"].predict_proba(df[bundle["features"]])[:, 1]

    table, summary = compare(df[bundle["features"]], scores, reference)
    print(
        f"batch of {summary['rows']:,} transactions against a reference of {reference['n_rows']:,} training rows"
    )
    print(
        f"alert rate {summary['alert_rate']:.2%} vs expected {summary['expected_alert_rate']:.2%} (ratio {summary['alert_rate_ratio']:.2f})"
    )
    print(f"inputs flagged: {summary['n_alert']} ALERT, {summary['n_watch']} WATCH\n")
    print(
        table.head(args.top).to_string(index=False, float_format=lambda v: f"{v:.3f}")
    )
    if reference.get("entity_level_features"):
        print(
            "\nnot checked row by row (one value per merchant or category and step; covered by the merchant, category and amount checks): "
            + ", ".join(reference["entity_level_features"])
        )


if __name__ == "__main__":
    main()
