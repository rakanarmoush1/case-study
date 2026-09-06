"""Central configuration for the fraud detection pipeline.

Everything that a reviewer might want to change (paths, split cut-off, cost
assumptions, model sizes) lives here so that the rest of the code base has no
magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TARGET = "fraud"
STEP = "step"
AMOUNT = "amount"
CUSTOMER = "customer"
MERCHANT = "merchant"
CATEGORY = "category"

# Columns that carry no information in this dataset (a single constant value).
CONSTANT_COLUMNS = ("zipcodeOri", "zipMerchant")

# Protected / sensitive attributes. Kept out of the production candidate model
# unless legal and compliance explicitly approve their use (see fairness.py).
PROTECTED_ATTRIBUTES = ("age", "gender")

RAW_CATEGORICALS = ("category", "merchant", "age", "gender")


@dataclass(frozen=True)
class CostAssumptions:
    """Illustrative cost model used to convert scores into a business decision.

    * A caught fraud avoids losing the transaction amount.
    * Every alert consumes analyst time (``review_cost``), whether it is a true
      or a false positive.
    * A false positive additionally creates customer friction (``friction_cost``),
      e.g. a declined card, a phone call, or churn risk.

    All values are illustrative and in the same units as the amount column; the
    data does not state a currency. They are parameters, not findings.
    """

    review_cost: float = 5.0
    friction_cost: float = 10.0
    # Analyst capacity expressed as alerts per step (the time unit of the data).
    alerts_per_step_capacity: int = 60


@dataclass(frozen=True)
class SplitConfig:
    """Temporal split. The data spans steps 0-179; the last 20% of steps are held out."""

    test_start_step: int = 144
    n_cv_folds: int = 3
    cv_fold_length: int = 16  # steps per validation fold inside the training window


@dataclass
class PipelineConfig:
    data_path: Path = PROJECT_ROOT / "data" / "fraud.csv"
    reports_dir: Path = PROJECT_ROOT / "reports"
    models_dir: Path = PROJECT_ROOT / "models"
    random_state: int = 42
    n_jobs: int = -1
    quick: bool = False  # smaller models / subsample for CI and smoke tests
    tune: bool = False  # run Optuna hyper-parameter search for the champion
    n_tuning_trials: int = 25
    split: SplitConfig = field(default_factory=SplitConfig)
    costs: CostAssumptions = field(default_factory=CostAssumptions)

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"

    @property
    def metrics_dir(self) -> Path:
        return self.reports_dir / "metrics"
