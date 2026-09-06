"""Hyper-parameter search for the boosted champion with time-respecting folds.

The search space depends on the champion family (LightGBM or XGBoost). The
objective is the mean average precision over the expanding-window folds, so the
chosen parameters are never influenced by the hold-out period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .config import TARGET, PipelineConfig
from .models import make_model
from .split import time_series_folds

SEARCH_ROUNDS = 400
FINAL_ROUNDS = 900


def _suggest(trial, family: str, spw: float) -> dict:
    common = {
        "n_estimators": SEARCH_ROUNDS,
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, spw, log=True),
    }
    if family == "lgbm":
        common["num_leaves"] = trial.suggest_int("num_leaves", 15, 255, log=True)
        common["min_child_samples"] = trial.suggest_int(
            "min_child_samples", 20, 500, log=True
        )
    elif family == "xgb":
        common["max_depth"] = trial.suggest_int("max_depth", 3, 10)
        common["min_child_weight"] = trial.suggest_float(
            "min_child_weight", 1.0, 50.0, log=True
        )
        common["gamma"] = trial.suggest_float("gamma", 0.0, 5.0)
    else:
        raise KeyError(f"No search space for family {family}")
    return common


def tune_booster(
    family: str,
    train: pd.DataFrame,
    features: list[str],
    spw: float,
    cfg: PipelineConfig,
) -> dict:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folds = list(time_series_folds(train["step"], cfg.split))
    y = train[TARGET].to_numpy()

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial, family, spw)
        scores = []
        for tr_idx, va_idx in folds:
            model = make_model(
                family,
                features,
                spw,
                cfg.random_state,
                cfg.n_jobs,
                quick=False,
                params=params,
            )
            model.fit(train.iloc[tr_idx][features], y[tr_idx])
            p = model.predict_proba(train.iloc[va_idx][features])[:, 1]
            scores.append(average_precision_score(y[va_idx], p))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.random_state)
    )
    study.optimize(objective, n_trials=cfg.n_tuning_trials, show_progress_bar=False)
    best = dict(study.best_params)
    best["n_estimators"] = FINAL_ROUNDS
    return {
        "family": family,
        "best_params": best,
        "best_cv_average_precision": float(study.best_value),
        "n_trials": len(study.trials),
        "trials": [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials
            if t.value is not None
        ],
    }
