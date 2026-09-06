"""Point-in-time feature engineering.

Every behavioural feature for a transaction at step ``t`` is computed from
transactions at steps **strictly before** ``t``. Transactions inside the same
step carry no ordering information in this dataset, so using "earlier rows in
the same step" would be an arbitrary (and unreproducible) choice; excluding the
current step entirely is the conservative option and mirrors how a batch
feature store would refresh overnight.

No feature uses the fraud label, directly or indirectly. In production,
confirmed-fraud labels arrive with a delay of weeks (chargebacks), so a model
that relied on them at scoring time would not be deployable.

Features are also built to be stationary: rolling windows and shares rather than
cumulative counts. A merchant's "transactions so far" grows with calendar time and
acts as a clock, which a drift monitor would flag on day one and which the model
would extrapolate in production. Cumulative counts are still computed as
intermediates (for shares and novelty flags) but are not model inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    AMOUNT,
    CATEGORY,
    CUSTOMER,
    MERCHANT,
    PROTECTED_ATTRIBUTES,
    STEP,
    TARGET,
)

# --------------------------------------------------------------------------- #
# Feature groups (the single source of truth for the modelling code)
# --------------------------------------------------------------------------- #
NUMERIC_FEATURES: list[str] = [
    # transaction
    "amount",
    "log_amount",
    # time-of-day style features suggested in the brief (kept so their value can be tested)
    "hour_of_day",
    "day_of_week",
    # customer history (point-in-time)
    "cust_prev_mean_amount",
    "cust_prev_std_amount",
    "cust_prev_max_amount",
    "amount_over_cust_mean",
    "amount_z_cust",
    "amount_over_cust_max",
    "cust_steps_since_last",
    "cust_n_last_7",
    "cust_n_last_24",
    "cust_amount_last_7",
    "cust_cat_prev_share",
    "cust_merch_prev_share",
    "cust_first_time_category",
    "cust_first_time_merchant",
    "cust_is_new",
    # merchant / category context over a rolling window (point-in-time)
    "merch_share_last_24",
    "merch_mean_amount_last_24",
    "amount_over_merch_mean",
    "cat_mean_amount_last_24",
    "amount_over_cat_mean",
]

CATEGORICAL_FEATURES: list[str] = ["category", "merchant", "age", "gender"]

# Transparent baseline: the raw fields a fraud analyst could write rules on today.
RULES_FEATURES: list[str] = ["amount", "category"]

ALL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def feature_columns(
    include_protected: bool = True, include_time: bool = True
) -> list[str]:
    cols = list(ALL_FEATURES)
    if not include_protected:
        cols = [c for c in cols if c not in PROTECTED_ATTRIBUTES]
    if not include_time:
        cols = [c for c in cols if c not in ("hour_of_day", "day_of_week")]
    return cols


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _exclusive_cumulative(
    df: pd.DataFrame, keys: list[str], prefix: str
) -> pd.DataFrame:
    """Per (keys, step): count / sum / sum-of-squares / max of amount over steps < current step.

    Returns a frame keyed by ``keys + [step]`` with columns ``{prefix}_n``,
    ``{prefix}_mean``, ``{prefix}_std``, ``{prefix}_max`` (NaN when no history).
    """
    grp = keys + [STEP]
    agg = (
        df.assign(_sq=df[AMOUNT] ** 2)
        .groupby(grp, observed=True)
        .agg(
            n=(AMOUNT, "size"), s=(AMOUNT, "sum"), sq=("_sq", "sum"), mx=(AMOUNT, "max")
        )
        .reset_index()
        .sort_values(grp, kind="stable")
    )
    g = agg.groupby(keys, observed=True)
    cum_n = g["n"].cumsum() - agg["n"]
    cum_s = g["s"].cumsum() - agg["s"]
    cum_sq = g["sq"].cumsum() - agg["sq"]
    cummax_incl = g["mx"].cummax()
    prev_max = cummax_incl.groupby([agg[k] for k in keys], observed=True).shift(1)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cum_n > 0, cum_s / cum_n, np.nan)
        var = np.where(cum_n > 1, cum_sq / cum_n - mean**2, np.nan)
        std = np.sqrt(np.clip(var, 0, None))

    out = agg[grp].copy()
    out[f"{prefix}_n"] = cum_n.to_numpy().astype("float64")
    out[f"{prefix}_mean"] = mean
    out[f"{prefix}_std"] = std
    out[f"{prefix}_max"] = prev_max.to_numpy()
    return out


def _customer_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling-window activity per customer using a dense (customer x step) matrix.

    Windows are *exclusive* of the current step: ``cust_n_last_7`` at step t counts
    transactions in steps [t-7, t-1].
    """
    cust_codes, cust_uniques = pd.factorize(df[CUSTOMER])
    n_cust = len(cust_uniques)
    n_steps = int(df[STEP].max()) + 1
    steps = df[STEP].to_numpy()

    counts = np.zeros((n_cust, n_steps), dtype="float64")
    sums = np.zeros((n_cust, n_steps), dtype="float64")
    np.add.at(counts, (cust_codes, steps), 1.0)
    np.add.at(sums, (cust_codes, steps), df[AMOUNT].to_numpy())

    # exclusive cumulative sums: excl[:, t] = sum over steps < t
    excl_counts = np.concatenate(
        [np.zeros((n_cust, 1)), np.cumsum(counts, axis=1)[:, :-1]], axis=1
    )
    excl_sums = np.concatenate(
        [np.zeros((n_cust, 1)), np.cumsum(sums, axis=1)[:, :-1]], axis=1
    )

    def window(excl: np.ndarray, w: int) -> np.ndarray:
        lag = np.clip(steps - w, 0, None)
        return excl[cust_codes, steps] - excl[cust_codes, lag]

    # last step (< t) at which the customer transacted
    seen = np.where(counts > 0, np.arange(n_steps)[None, :], -1)
    last_seen_incl = np.maximum.accumulate(seen, axis=1)
    last_seen_excl = np.concatenate(
        [np.full((n_cust, 1), -1), last_seen_incl[:, :-1]], axis=1
    )
    last = last_seen_excl[cust_codes, steps].astype("float64")
    steps_since_last = np.where(last >= 0, steps - last, np.nan)

    return pd.DataFrame(
        {
            "cust_n_last_7": window(excl_counts, 7),
            "cust_n_last_24": window(excl_counts, 24),
            "cust_amount_last_7": window(excl_sums, 7),
            "cust_steps_since_last": steps_since_last,
        },
        index=df.index,
    )


def _rolling_by_key(
    df: pd.DataFrame, key: str, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Count and mean amount per ``key`` over steps [t-window, t-1], using a dense (key x step) matrix."""
    codes, uniques = pd.factorize(df[key])
    n_keys = len(uniques)
    n_steps = int(df[STEP].max()) + 1
    steps = df[STEP].to_numpy()
    counts = np.zeros((n_keys, n_steps), dtype="float64")
    sums = np.zeros((n_keys, n_steps), dtype="float64")
    np.add.at(counts, (codes, steps), 1.0)
    np.add.at(sums, (codes, steps), df[AMOUNT].to_numpy())
    excl_counts = np.concatenate(
        [np.zeros((n_keys, 1)), np.cumsum(counts, axis=1)[:, :-1]], axis=1
    )
    excl_sums = np.concatenate(
        [np.zeros((n_keys, 1)), np.cumsum(sums, axis=1)[:, :-1]], axis=1
    )
    lag = np.clip(steps - window, 0, None)
    n = excl_counts[codes, steps] - excl_counts[codes, lag]
    total = excl_sums[codes, steps] - excl_sums[codes, lag]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, total / n, np.nan)
    return n, mean


def _rolling_total(df: pd.DataFrame, window: int) -> np.ndarray:
    """Total number of transactions (all keys) in steps [t-window, t-1] for each row."""
    n_steps = int(df[STEP].max()) + 1
    steps = df[STEP].to_numpy()
    counts = np.bincount(steps, minlength=n_steps).astype("float64")
    excl = np.concatenate([[0.0], np.cumsum(counts)[:-1]])
    lag = np.clip(steps - window, 0, None)
    return excl[steps] - excl[lag]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with all engineered features appended.

    The input must already be cleaned (see ``data.clean``) and contain the whole
    history: features for the test period legitimately use training-period
    history, exactly as a deployed model would.
    """
    out = df.copy()

    # -- transaction-level ---------------------------------------------------
    out["log_amount"] = np.log1p(out[AMOUNT])
    out["hour_of_day"] = (out[STEP] % 24).astype("int16")
    out["day_of_week"] = ((out[STEP] // 24) % 7).astype("int16")

    # -- customer history -----------------------------------------------------
    cust = _exclusive_cumulative(out, [CUSTOMER], "cust_prev")
    out = out.merge(cust, on=[CUSTOMER, STEP], how="left")
    out = out.rename(
        columns={
            "cust_prev_mean": "cust_prev_mean_amount",
            "cust_prev_std": "cust_prev_std_amount",
            "cust_prev_max": "cust_prev_max_amount",
        }
    )
    out["cust_is_new"] = (out["cust_prev_n"] == 0).astype("int8")
    out["amount_over_cust_mean"] = out[AMOUNT] / (out["cust_prev_mean_amount"] + 1.0)
    out["amount_z_cust"] = (out[AMOUNT] - out["cust_prev_mean_amount"]) / (
        out["cust_prev_std_amount"].fillna(0.0) + 1.0
    )
    out["amount_over_cust_max"] = out[AMOUNT] / (out["cust_prev_max_amount"] + 1.0)

    vel = _customer_velocity(out)
    out = pd.concat([out, vel], axis=1)

    cc = _exclusive_cumulative(out, [CUSTOMER, CATEGORY], "cust_cat_prev")[
        [CUSTOMER, CATEGORY, STEP, "cust_cat_prev_n"]
    ]
    out = out.merge(cc, on=[CUSTOMER, CATEGORY, STEP], how="left")
    out["cust_cat_prev_share"] = out["cust_cat_prev_n"] / out["cust_prev_n"].replace(
        0, np.nan
    )
    out["cust_first_time_category"] = (out["cust_cat_prev_n"] == 0).astype("int8")

    cm = _exclusive_cumulative(out, [CUSTOMER, MERCHANT], "cust_merch_prev")[
        [CUSTOMER, MERCHANT, STEP, "cust_merch_prev_n"]
    ]
    out = out.merge(cm, on=[CUSTOMER, MERCHANT, STEP], how="left")
    out["cust_merch_prev_share"] = out["cust_merch_prev_n"] / out[
        "cust_prev_n"
    ].replace(0, np.nan)
    out["cust_first_time_merchant"] = (out["cust_merch_prev_n"] == 0).astype("int8")

    # -- merchant / category context over the last 24 steps ----------------------
    merch_n, out["merch_mean_amount_last_24"] = _rolling_by_key(out, MERCHANT, 24)
    total_n = _rolling_total(out, 24)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["merch_share_last_24"] = np.where(total_n > 0, merch_n / total_n, np.nan)
    out["amount_over_merch_mean"] = out[AMOUNT] / (
        out["merch_mean_amount_last_24"] + 1.0
    )
    _, out["cat_mean_amount_last_24"] = _rolling_by_key(out, CATEGORY, 24)
    out["amount_over_cat_mean"] = out[AMOUNT] / (out["cat_mean_amount_last_24"] + 1.0)

    # merges reset the categorical dtype in some pandas versions; enforce it again
    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].astype("category")
    for col in NUMERIC_FEATURES:
        out[col] = out[col].astype("float64")

    missing = [c for c in ALL_FEATURES if c not in out.columns]
    if missing:
        raise RuntimeError(f"Feature engineering did not produce: {missing}")

    out = out.sort_values("txn_id", kind="stable").reset_index(drop=True)
    return out


def split_xy(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    return df[features], df[TARGET].astype(int)
