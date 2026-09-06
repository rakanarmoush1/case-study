# Card fraud detection case study

Detecting fraudulent card transactions. The aim was a model that would hold up in a real fraud operation: history features built only from earlier time steps, validation that trains on the past and tests on the future, an alert threshold tied to analyst capacity and the cost of a false decline, and reasons attached to every alert.

## Layout

- `src/fraud_detection/` - the package: data loading, feature engineering, temporal splits, models, evaluation, interpretation, and the `fraud-detection` command that runs the full experiment.
- `scripts/` - the individual steps as plain scripts: `eda.py`, `check_features.py`, `evaluate.py`, `train.py`, `predict.py`, `interpret.py`, `monitor.py`.
- `notebooks/` - the same analysis as executed notebooks: EDA, modelling and evaluation, interpretation and business outcomes.
- `reports/` - the figures and the metrics from the last run.
- `models/` - the trained model, a JSON sidecar describing it, and the drift-monitoring reference.
- `data/` - the transactions file.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Running

```bash
fraud-detection run --tune          # full experiment: benchmark, imbalance and ablation studies, tuning, SHAP
python scripts/eda.py               # data profile, key tables and EDA figures
python scripts/check_features.py    # brute-force, future-leakage and label-leakage checks on the features
python scripts/evaluate.py          # benchmark the model families on the temporal split
python scripts/train.py             # fit the production model, choose the threshold, save it
python scripts/predict.py --input data/fraud.csv --output scored.csv --min-step 144
python scripts/interpret.py         # SHAP explanations, outcomes by customer group, weak segments, stability
python scripts/monitor.py --input data/fraud.csv --min-step 144   # drift check: PSI on every input and on the score
```

The Makefile has a target for each of these.

## Notes

- Nothing in the analysis depends on whether a time step is an hour or a day; operational figures are reported per 10,000 transactions for that reason.
- Age and gender are not inputs to the production model.
- The data does not state a currency, so amounts and costs are reported in its own units.
- History features are rolling windows and shares rather than cumulative counts, so no input drifts with calendar time. Training the model also writes a JSON sidecar (inputs, threshold, library versions, hold-out metrics) and a drift-monitoring reference next to it.
- The cost parameters used to choose the alert threshold are placeholders for the business to set (see `src/fraud_detection/config.py`).

## Deployment notes

- `scripts/predict.py` scores in batch and rebuilds each customer's history from the file it is given. Real-time scoring needs the same history features served from a feature store keyed by customer, merchant and category, refreshed every step.
- The alert threshold is a business parameter, chosen on validation data from analyst capacity and the cost of a false decline; changing it does not require retraining.
- Retrain and re-threshold on a fixed cadence and whenever `scripts/monitor.py` flags drift in the inputs, the score distribution or the alert rate.
