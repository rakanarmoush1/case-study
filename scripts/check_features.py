"""Sanity checks for the point-in-time features, run on the real data.

1. Brute force: for a random sample of transactions, recompute the customer, merchant and
   category history features directly from the raw rows and compare with the vectorised
   version.
2. No future leakage: features for steps before a cut-off must be identical whether or not
   later steps are present in the data.
3. No label leakage: flipping every fraud label must leave every feature unchanged.

Example:
    python scripts/check_features.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_detection.config import PROJECT_ROOT
from fraud_detection.data import load_clean
from fraud_detection.features import ALL_FEATURES, NUMERIC_FEATURES, build_features


def brute_force_check(df: pd.DataFrame, feats: pd.DataFrame, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    for i in rng.choice(len(feats), n, replace=False):
        r = feats.iloc[i]
        t = r["step"]
        hist = df[(df["customer"] == r["customer"]) & (df["step"] < t)]
        if len(hist):
            assert np.isclose(r["cust_prev_mean_amount"], hist["amount"].mean())
            assert np.isclose(r["cust_prev_max_amount"], hist["amount"].max())
            assert r["cust_steps_since_last"] == t - hist["step"].max()
            assert np.isclose(
                r["cust_cat_prev_share"], (hist["category"] == r["category"]).mean()
            )
            assert np.isclose(
                r["cust_merch_prev_share"], (hist["merchant"] == r["merchant"]).mean()
            )
        else:
            assert np.isnan(r["cust_prev_mean_amount"]) and r["cust_is_new"] == 1
        window = hist[hist["step"] >= t - 7]
        assert r["cust_n_last_7"] == len(window)
        assert np.isclose(r["cust_amount_last_7"], window["amount"].sum())
        assert r["cust_first_time_merchant"] == int(
            (hist["merchant"] == r["merchant"]).sum() == 0
        )
        recent = df[(df["step"] < t) & (df["step"] >= t - 24)]
        mwin = recent[recent["merchant"] == r["merchant"]]
        if len(recent):
            assert np.isclose(r["merch_share_last_24"], len(mwin) / len(recent))
        if len(mwin):
            assert np.isclose(r["merch_mean_amount_last_24"], mwin["amount"].mean())
        cwin = recent[recent["category"] == r["category"]]
        if len(cwin):
            assert np.isclose(r["cat_mean_amount_last_24"], cwin["amount"].mean())


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "fraud.csv")
    ap.add_argument(
        "--sample", type=int, default=200, help="Transactions to verify by brute force"
    )
    ap.add_argument(
        "--cutoff", type=int, default=100, help="Step used for the future-leakage check"
    )
    args = ap.parse_args()

    df = load_clean(args.data)
    feats = build_features(df)

    brute_force_check(df, feats, args.sample, seed=0)
    print(
        f"PASS  brute force: {args.sample} sampled transactions match a direct recomputation"
    )

    truncated = build_features(df[df["step"] < args.cutoff].reset_index(drop=True))
    a = (
        feats[feats["step"] < args.cutoff]
        .sort_values("txn_id")[NUMERIC_FEATURES]
        .reset_index(drop=True)
    )
    b = truncated.sort_values("txn_id")[NUMERIC_FEATURES].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    print(
        f"PASS  no future leakage: features for steps < {args.cutoff} are identical with and without later data"
    )

    flipped = df.copy()
    flipped["fraud"] = 1 - flipped["fraud"]
    pd.testing.assert_frame_equal(
        feats[ALL_FEATURES], build_features(flipped)[ALL_FEATURES]
    )
    print("PASS  no label leakage: flipping every label leaves every feature unchanged")


if __name__ == "__main__":
    main()
