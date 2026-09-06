"""Loading, validation and cleaning of the raw transactions file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import AMOUNT, CONSTANT_COLUMNS, RAW_CATEGORICALS, STEP, TARGET

EXPECTED_COLUMNS = [
    "step",
    "customer",
    "age",
    "gender",
    "zipcodeOri",
    "merchant",
    "zipMerchant",
    "category",
    "amount",
    "fraud",
]


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the raw CSV (plain or gzip). Values are wrapped in single quotes in the source file."""
    df = pd.read_csv(path, quotechar="'")
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns and c != TARGET]
    if missing:
        raise ValueError(f"Input file is missing expected columns: {missing}")
    return df


def profile(df: pd.DataFrame) -> dict:
    """Light-weight data quality profile used in the EDA and the report."""
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) == 1]
    return {
        "n_rows": len(df),
        "n_columns": int(df.shape[1]),
        "n_missing_values": int(df.isna().sum().sum()),
        "n_duplicate_rows": int(df.duplicated().sum()),
        "constant_columns": constant_cols,
        "n_steps": int(df[STEP].nunique()),
        "step_min": int(df[STEP].min()),
        "step_max": int(df[STEP].max()),
        "n_customers": int(df["customer"].nunique()),
        "n_merchants": int(df["merchant"].nunique()),
        "n_categories": int(df["category"].nunique()),
        "fraud_count": int(df[TARGET].sum()) if TARGET in df.columns else None,
        "fraud_rate": float(df[TARGET].mean()) if TARGET in df.columns else None,
        "n_zero_amount": int((df[AMOUNT] == 0).sum()),
        "n_negative_amount": int((df[AMOUNT] < 0).sum()),
    }


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the (few) cleaning steps this dataset needs.

    * Drop columns that are constant (both zip codes are a single value).
    * Cast categoricals to pandas ``category`` dtype.
    * Enforce numeric types for step, amount and the target.
    * Keep a stable row order (by step, then original order) so that all
      point-in-time features are deterministic.
    """
    out = df.copy()
    drop = [c for c in CONSTANT_COLUMNS if c in out.columns]
    out = out.drop(columns=drop)

    out[STEP] = out[STEP].astype("int32")
    out[AMOUNT] = out[AMOUNT].astype("float64")
    if TARGET in out.columns:
        out[TARGET] = out[TARGET].astype("int8")
    out["customer"] = out["customer"].astype(str)
    for col in RAW_CATEGORICALS:
        out[col] = out[col].astype(str).astype("category")

    out = out.sort_values(STEP, kind="stable").reset_index(drop=True)
    out.index.name = "txn_id"
    return out.reset_index()


def load_clean(path: str | Path) -> pd.DataFrame:
    return clean(load_raw(path))
