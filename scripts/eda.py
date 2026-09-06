"""Exploratory data analysis as a script: prints the data profile and the key tables, and
saves the EDA figures to reports/figures. The same analysis is walked through in
notebooks/01_eda.ipynb.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from fraud_detection import plots
from fraud_detection.config import PROJECT_ROOT
from fraud_detection.data import clean, load_raw, profile
from fraud_detection.evaluate import amount_band


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "fraud.csv")
    ap.add_argument(
        "--figures", type=Path, default=PROJECT_ROOT / "reports" / "figures"
    )
    args = ap.parse_args()
    pd.set_option("display.width", 160)

    raw = load_raw(args.data)
    print("== profile ==")
    print(pd.Series(profile(raw)).to_string(), "\n")
    df = clean(raw)

    print("== fraud by category ==")
    cat = df.groupby("category", observed=True)["fraud"].agg(
        transactions="size", fraud_count="sum", fraud_rate="mean"
    )
    cat["volume_share"] = cat["transactions"] / len(df)
    print(
        cat.sort_values("fraud_rate", ascending=False).to_string(
            float_format=lambda v: f"{v:.4f}"
        ),
        "\n",
    )
    print(
        "merchants with more than one category:",
        int((df.groupby("merchant", observed=True)["category"].nunique() > 1).sum()),
    )

    print("\n== amount ==")
    print(
        df.groupby("fraud")["amount"]
        .describe()[["count", "mean", "50%", "max"]]
        .to_string()
    )
    print("AUC of amount alone:", round(roc_auc_score(df["fraud"], df["amount"]), 3))
    within = {
        c: round(roc_auc_score(g["fraud"], g["amount"]), 3)
        for c, g in df.groupby("category", observed=True)
        if g["fraud"].nunique() > 1
    }
    print("AUC of amount alone within category:", within)
    bands = pd.DataFrame(
        {"band": amount_band(df["amount"]), "fraud": df["fraud"].to_numpy()}
    )
    print("\nfraud rate by amount band:")
    print(
        bands.groupby("band", observed=True)["fraud"]
        .agg(transactions="size", fraud_count="sum", fraud_rate="mean")
        .to_string(float_format=lambda v: f"{v:.4f}")
    )

    print("\n== time ==")
    per_step = df.groupby("step")["fraud"].sum()
    hourly = df.assign(h=df["step"] % 24).groupby("h")["fraud"].mean()
    print("distinct fraud counts per step:", sorted(per_step.unique().tolist()))
    print(f"fraud rate by hour of day: min {hourly.min():.4f}, max {hourly.max():.4f}")

    print("\n== persistence within a customer ==")
    d = df.sort_values(["customer", "step"]).copy()
    d["prev_fraud"] = d.groupby("customer", observed=True)["fraud"].shift(1)
    rates = d.dropna(subset=["prev_fraud"]).groupby("prev_fraud")["fraud"].mean()
    print(
        f"P(fraud | previous transaction fraud) = {rates[1.0]:.3f}; P(fraud | previous benign) = {rates[0.0]:.4f}; ratio {rates[1.0] / rates[0.0]:.0f}x"
    )

    print("\n== protected attributes ==")
    for col in ["gender", "age"]:
        print(
            df.groupby(col, observed=True)["fraud"]
            .agg(n="size", fraud_rate="mean")
            .to_string(float_format=lambda v: f"{v:.4f}"),
            "\n",
        )

    args.figures.mkdir(parents=True, exist_ok=True)
    plots.class_balance(df, args.figures / "eda_class_balance.png")
    plots.fraud_by_category(df, args.figures / "eda_fraud_by_category.png")
    plots.amount_distribution(df, args.figures / "eda_amount_distribution.png")
    plots.fraud_over_time(df, args.figures / "eda_fraud_over_time.png")
    plots.hour_of_day(df, args.figures / "eda_hour_of_day.png")
    plots.customer_persistence(df, args.figures / "eda_customer_persistence.png")
    plots.fraud_by_group(
        df, "age", args.figures / "eda_fraud_by_age.png", "Fraud rate by age band"
    )
    plots.fraud_by_group(
        df,
        "gender",
        args.figures / "eda_fraud_by_gender.png",
        "Fraud rate by gender code",
    )
    print(f"figures saved to {args.figures}")


if __name__ == "__main__":
    main()
