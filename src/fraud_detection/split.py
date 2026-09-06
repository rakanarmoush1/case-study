"""Temporal train/test split and time-series cross-validation folds.

Fraud models are always used *forward in time*: they are trained on the past
and score the future. A random split would let the model peek at transactions
that happen after the ones it is evaluated on, which flatters every metric.
The hold-out set is therefore the last block of steps, and cross-validation
inside the training window uses expanding windows that respect time.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from .config import STEP, SplitConfig


def temporal_split(
    df: pd.DataFrame, cfg: SplitConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df[STEP] < cfg.test_start_step]
    test = df[df[STEP] >= cfg.test_start_step]
    if train.empty or test.empty:
        raise ValueError(
            "Temporal split produced an empty partition; check test_start_step."
        )
    return train, test


def time_series_folds(
    steps: pd.Series | np.ndarray, cfg: SplitConfig
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, valid_idx) positional indices for expanding-window CV.

    With the defaults (test starts at 144, 3 folds of 16 steps) the folds are
    validated on steps [96,112), [112,128), [128,144) and trained on everything
    before each validation block.
    """
    steps = np.asarray(steps)
    last = cfg.test_start_step
    for k in range(cfg.n_cv_folds, 0, -1):
        valid_start = last - k * cfg.cv_fold_length
        valid_end = valid_start + cfg.cv_fold_length
        train_idx = np.where(steps < valid_start)[0]
        valid_idx = np.where((steps >= valid_start) & (steps < valid_end))[0]
        if len(train_idx) == 0 or len(valid_idx) == 0:
            continue
        yield train_idx, valid_idx


def random_split_for_comparison(df: pd.DataFrame, test_size: float, seed: int):
    """A stratified *random* split, used only to show how much it over-states performance."""
    from sklearn.model_selection import train_test_split

    from .config import TARGET

    return train_test_split(
        df, test_size=test_size, stratify=df[TARGET], random_state=seed
    )
