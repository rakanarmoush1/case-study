"""Model factory: transparent rules baseline, linear baseline, bagging and boosting.

All models are exposed as scikit-learn compatible pipelines with
``fit(X, y)`` / ``predict_proba(X)`` so that the evaluation code treats them
identically. Tree boosting models consume pandas ``category`` columns natively;
the linear and forest models get a one-hot encoded, imputed (and for the
linear model, scaled) matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RULES_FEATURES


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    description: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "rules": ModelSpec(
        "rules",
        "Rules baseline (depth-4 tree)",
        "Segment rules: a depth-4 decision tree on amount and category (up to 16 segments), each flagged when its historical fraud rate exceeds the threshold. What a fraud team could deploy today.",
    ),
    "logreg": ModelSpec(
        "logreg",
        "Logistic regression",
        "Linear, fully interpretable baseline with class weighting.",
    ),
    "rf": ModelSpec(
        "rf",
        "Random forest",
        "Bagged trees with balanced subsampling; robust, few assumptions.",
    ),
    "mlp": ModelSpec(
        "mlp",
        "Neural network (MLP)",
        "Two hidden layers on one-hot, scaled inputs with early stopping; no class weighting (unsupported), thresholded on validation like the others.",
    ),
    "lgbm": ModelSpec(
        "lgbm",
        "LightGBM",
        "Gradient boosted trees with native categorical handling and class weighting.",
    ),
    "xgb": ModelSpec(
        "xgb",
        "XGBoost",
        "Gradient boosted trees (histogram method) as a second boosting implementation.",
    ),
}


class CategoryAligner(BaseEstimator, TransformerMixin):
    """Select feature columns and pin categorical levels to those seen in training.

    Unseen levels at scoring time become NaN, which the boosting libraries treat
    as a separate branch instead of raising.
    """

    def __init__(self, features: list[str], impute: bool = False):
        self.features = features
        self.impute = impute

    def fit(self, X: pd.DataFrame, y=None):
        self.categorical_ = [c for c in self.features if c in CATEGORICAL_FEATURES]
        self.numeric_ = [c for c in self.features if c not in CATEGORICAL_FEATURES]
        self.categories_ = {
            c: list(X[c].astype("category").cat.categories) for c in self.categorical_
        }
        self.medians_ = X[self.numeric_].median(numeric_only=True).to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X[self.features].copy()
        for c in self.categorical_:
            known = out[c].astype(str)
            known = known.where(known.isin(self.categories_[c]), other=None)
            out[c] = pd.Categorical(known, categories=self.categories_[c])
        for c in self.numeric_:
            out[c] = out[c].astype("float64")
            if self.impute:
                out[c] = out[c].fillna(self.medians_[c])
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.features, dtype=object)


def _onehot_preprocessor(features: list[str], scale: bool) -> ColumnTransformer:
    numeric = [c for c in features if c in NUMERIC_FEATURES]
    categorical = [c for c in features if c in CATEGORICAL_FEATURES]
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), numeric),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, dtype=np.float32
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_model(
    key: str,
    features: list[str],
    scale_pos_weight: float,
    random_state: int = 42,
    n_jobs: int = -1,
    quick: bool = False,
    params: dict | None = None,
) -> Pipeline:
    """Build an untrained pipeline for ``key`` using ``features``.

    ``scale_pos_weight`` is the negative/positive ratio of the training data and
    is used by the boosting models to re-weight the minority class.
    """
    params = dict(params or {})
    if key == "rules":
        feats = [f for f in RULES_FEATURES if f in features]
        return Pipeline(
            [
                ("prep", _onehot_preprocessor(feats, scale=False)),
                (
                    "clf",
                    DecisionTreeClassifier(
                        max_depth=4,
                        min_samples_leaf=200,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if key == "logreg":
        return Pipeline(
            [
                ("prep", _onehot_preprocessor(features, scale=True)),
                (
                    "clf",
                    LogisticRegression(
                        C=params.get("C", 0.5),
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if key == "rf":
        return Pipeline(
            [
                ("prep", _onehot_preprocessor(features, scale=False)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=60 if quick else 400,
                        min_samples_leaf=params.get("min_samples_leaf", 20),
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=n_jobs,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if key == "mlp":
        return Pipeline(
            [
                ("prep", _onehot_preprocessor(features, scale=True)),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        batch_size=2048,
                        learning_rate_init=1e-3,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=5,
                        max_iter=10 if quick else 40,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if key == "lgbm":
        base = dict(
            n_estimators=200 if quick else 900,
            learning_rate=0.03,
            num_leaves=63,
            min_child_samples=100,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=-1,
        )
        base.update(params)
        return Pipeline(
            [("prep", CategoryAligner(features)), ("clf", LGBMClassifier(**base))]
        )
    if key == "xgb":
        base = dict(
            n_estimators=200 if quick else 900,
            learning_rate=0.03,
            max_depth=6,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            enable_categorical=True,
            n_jobs=n_jobs,
            random_state=random_state,
            eval_metric="aucpr",
        )
        base.update(params)
        return Pipeline(
            [("prep", CategoryAligner(features)), ("clf", XGBClassifier(**base))]
        )
    raise KeyError(f"Unknown model key: {key}")


def make_imbalance_variant(
    strategy: str,
    features: list[str],
    scale_pos_weight: float,
    random_state: int = 42,
    n_jobs: int = -1,
    quick: bool = False,
    params: dict | None = None,
):
    """LightGBM under different class-imbalance treatments (same imputed inputs for all).

    strategies: ``none`` (no treatment), ``class_weight`` (scale_pos_weight),
    ``undersample`` (random majority under-sampling to a 1:10 ratio),
    ``smote`` (SMOTE-NC oversampling of the minority class to a 1:10 ratio).
    """
    from imblearn.over_sampling import SMOTENC
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.under_sampling import RandomUnderSampler

    params = dict(params or {})
    lgbm_params = dict(
        n_estimators=200 if quick else 900,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=-1,
    )
    lgbm_params.update(params)
    lgbm_params.pop("scale_pos_weight", None)
    aligner = CategoryAligner(features, impute=True)
    if strategy == "none":
        return ImbPipeline([("prep", aligner), ("clf", LGBMClassifier(**lgbm_params))])
    if strategy == "class_weight":
        return ImbPipeline(
            [
                ("prep", aligner),
                (
                    "clf",
                    LGBMClassifier(scale_pos_weight=scale_pos_weight, **lgbm_params),
                ),
            ]
        )
    if strategy == "undersample":
        return ImbPipeline(
            [
                ("prep", aligner),
                (
                    "sample",
                    RandomUnderSampler(
                        sampling_strategy=0.1, random_state=random_state
                    ),
                ),
                ("clf", LGBMClassifier(**lgbm_params)),
            ]
        )
    if strategy == "smote":
        cat_idx = [i for i, f in enumerate(features) if f in CATEGORICAL_FEATURES]
        return ImbPipeline(
            [
                ("prep", aligner),
                (
                    "sample",
                    SMOTENC(
                        categorical_features=cat_idx,
                        sampling_strategy=0.1,
                        random_state=random_state,
                        k_neighbors=5,
                    ),
                ),
                ("clf", LGBMClassifier(**lgbm_params)),
            ]
        )
    raise KeyError(f"Unknown imbalance strategy: {strategy}")


def positive_scale(y: pd.Series | np.ndarray) -> float:
    y = np.asarray(y)
    pos = max(int(y.sum()), 1)
    return float((len(y) - pos) / pos)


def save_bundle(
    path,
    model,
    features: list[str],
    threshold: float,
    threshold_capacity: float,
    family: str,
    metadata: dict | None = None,
) -> None:
    """Persist the model bundle and a JSON sidecar describing it (inputs, threshold, versions, metrics)."""
    import json
    import platform
    from datetime import UTC, datetime
    from importlib.metadata import version
    from pathlib import Path as _Path

    import joblib

    path = _Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": features,
            "threshold": threshold,
            "threshold_capacity": threshold_capacity,
            "family": family,
        },
        path,
    )
    meta = {
        "family": family,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "features": features,
        "threshold": threshold,
        "threshold_capacity": threshold_capacity,
        "libraries": {
            "python": platform.python_version(),
            **{
                lib: version(lib)
                for lib in ("scikit-learn", "xgboost", "lightgbm", "pandas", "numpy")
            },
        },
    }
    meta.update(metadata or {})
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
