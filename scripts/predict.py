"""Score a file of card transactions with the saved production model.

Example:
    python scripts/predict.py --input data/fraud.csv --output scored.csv --min-step 144

The input must use the raw file's schema (step, customer, age, gender, zipcodeOri,
merchant, zipMerchant, category, amount, and optionally fraud). History features are
rebuilt from the file itself, so it should contain each customer's earlier
transactions, exactly as a batch scoring job would receive them. If a fraud column
is present the script also reports how the scores did.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from fraud_detection.config import PROJECT_ROOT, CostAssumptions
from fraud_detection.data import load_clean
from fraud_detection.evaluate import decision_metrics, ranking_metrics
from fraud_detection.features import build_features


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--input", required=True, type=Path, help="Transactions CSV (plain or gzip)"
    )
    ap.add_argument(
        "--output", required=True, type=Path, help="Where to write the scored rows"
    )
    ap.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / "production_model.joblib",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the saved alert threshold",
    )
    ap.add_argument(
        "--min-step",
        type=int,
        default=None,
        help="Only output rows from this step on; earlier rows still feed the history features",
    )
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    model, features = bundle["model"], bundle["features"]
    threshold = bundle["threshold"] if args.threshold is None else args.threshold

    df = build_features(load_clean(args.input))
    df["fraud_score"] = model.predict_proba(df[features])[:, 1]
    df["alert"] = (df["fraud_score"] >= threshold).astype(int)
    if args.min_step is not None:
        df = df[df["step"] >= args.min_step]

    cols = [
        "txn_id",
        "step",
        "customer",
        "merchant",
        "category",
        "amount",
        "fraud_score",
        "alert",
    ]
    df[cols].to_csv(args.output, index=False)
    print(
        f"scored {len(df):,} transactions with {bundle['family']} at threshold {threshold:.2f} -> {args.output}"
    )
    print(
        f"alerts: {int(df['alert'].sum()):,} ({df['alert'].mean():.2%} of transactions)"
    )

    if "fraud" in df.columns:
        y, p, amount = (
            df["fraud"].to_numpy(),
            df["fraud_score"].to_numpy(),
            df["amount"].to_numpy(),
        )
        m = ranking_metrics(y, p)
        d = decision_metrics(
            y, p, amount, threshold, CostAssumptions(), int(df["step"].nunique())
        )
        print(
            f"labels present: ROC-AUC {m['roc_auc']:.4f}, average precision {m['average_precision']:.3f}, "
            f"precision {d['precision']:.1%}, recall {d['recall']:.1%}, fraud value stopped {d['value_recall']:.1%}"
        )


if __name__ == "__main__":
    main()
