"""Command line entry point: ``fraud-detection run`` executes the full experiment pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import PipelineConfig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fraud-detection", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser(
        "run", help="Run the full experiment pipeline and write reports/."
    )
    run_p.add_argument(
        "--data", type=Path, default=None, help="Path to fraud.csv or fraud.csv.gz"
    )
    run_p.add_argument(
        "--reports",
        type=Path,
        default=None,
        help="Output directory (default: reports/)",
    )
    run_p.add_argument(
        "--quick",
        action="store_true",
        help="Smaller models, no SMOTE/tuning (CI and smoke tests)",
    )
    run_p.add_argument(
        "--tune",
        action="store_true",
        help="Run Optuna hyper-parameter search for the champion",
    )
    run_p.add_argument(
        "--trials",
        type=int,
        default=25,
        help="Number of Optuna trials when --tune is set",
    )
    run_p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Use only the first N steps (smoke tests)",
    )

    args = parser.parse_args(argv)
    if args.command == "run":
        cfg = PipelineConfig(
            quick=args.quick, tune=args.tune, n_tuning_trials=args.trials
        )
        if args.data:
            cfg.data_path = args.data
        if args.reports:
            cfg.reports_dir = args.reports
            cfg.models_dir = args.reports / "models"
        from .pipeline import run

        run(cfg, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
