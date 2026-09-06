"""End-to-end experiment pipeline.

Steps (each writes figures and/or metrics under ``reports/``):

1. Load, profile and clean the data; build point-in-time features.
2. Temporal split; benchmark six model families with time-series CV and a
   hold-out period. Operating thresholds are chosen on out-of-fold predictions.
3. Experiments on the champion family: class-imbalance treatments and feature
   ablations, both scored on the last validation fold (never on hold-out).
4. Protected-attribute check: retrain without age/gender and compare.
5. Optional Optuna tuning; final production candidate fit, calibration, SHAP
   interpretation and group-outcome (fairness) tables.
6. Persist the model (with a metadata sidecar and a drift-monitoring reference) and the metrics JSON.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import plots
from .config import AMOUNT, PROTECTED_ATTRIBUTES, STEP, TARGET, PipelineConfig
from .data import clean, load_raw, profile
from .evaluate import (
    amount_band,
    bootstrap_ci,
    calibration_table,
    choose_threshold,
    cost_sensitivity,
    decision_metrics,
    ranking_metrics,
    rules_from_tree,
    segment_performance,
    stability_over_time,
    threshold_sweep,
)
from .fairness import group_outcomes
from .features import CATEGORICAL_FEATURES, build_features, feature_columns
from .interpret import gain_importance, global_importance, local_examples, shap_values
from .models import (
    MODEL_SPECS,
    make_imbalance_variant,
    make_model,
    positive_scale,
    save_bundle,
)
from .monitoring import build_reference
from .split import random_split_for_comparison, temporal_split, time_series_folds

RAW_FEATURES = ["amount", "category", "merchant", "age", "gender"]


def _log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def _mean_std(dicts: list[dict]) -> dict:
    keys = dicts[0].keys()
    out = {}
    for k in keys:
        vals = np.array([d[k] for d in dicts], dtype=float)
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std())
    return out


def _apply_smoke(df: pd.DataFrame, cfg: PipelineConfig, max_steps: int | None):
    if not max_steps:
        return df, cfg
    df = df[df[STEP] < max_steps].reset_index(drop=True)
    from .config import SplitConfig

    split = SplitConfig(
        test_start_step=int(max_steps * 0.8),
        n_cv_folds=2,
        cv_fold_length=max(int(max_steps * 0.1), 1),
    )
    cfg.split = split
    return df, cfg


def run(cfg: PipelineConfig, max_steps: int | None = None) -> dict:
    t0 = time.time()
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    cfg.metrics_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    figs = cfg.figures_dir
    R: dict = {
        "config": {
            **asdict(cfg),
            "data_path": str(cfg.data_path),
            "reports_dir": str(cfg.reports_dir),
            "models_dir": str(cfg.models_dir),
        }
    }

    # ------------------------------------------------------------------ 1. data
    raw = load_raw(cfg.data_path)
    R["data_profile"] = profile(raw)
    df = clean(raw)
    df, cfg = _apply_smoke(df, cfg, max_steps)
    R["config"]["split"] = asdict(cfg.split)
    df = build_features(df)
    _log(f"data loaded and features built: {df.shape}", t0)

    plots.class_balance(df, figs / "eda_class_balance.png")
    plots.fraud_by_category(df, figs / "eda_fraud_by_category.png")
    plots.amount_distribution(df, figs / "eda_amount_distribution.png")
    plots.fraud_over_time(df, figs / "eda_fraud_over_time.png")
    plots.hour_of_day(df, figs / "eda_hour_of_day.png")
    plots.customer_persistence(df, figs / "eda_customer_persistence.png")
    plots.fraud_by_group(
        df, "age", figs / "eda_fraud_by_age.png", "Fraud rate by age band"
    )
    plots.fraud_by_group(
        df, "gender", figs / "eda_fraud_by_gender.png", "Fraud rate by gender code"
    )

    cat_tbl = (
        df.groupby("category", observed=True)[TARGET]
        .agg(transactions="size", fraud_count="sum", fraud_rate="mean")
        .sort_values("fraud_rate", ascending=False)
    )
    cat_tbl["volume_share"] = cat_tbl["transactions"] / len(df)
    cat_tbl["median_amount_fraud"] = (
        df[df[TARGET] == 1].groupby("category", observed=True)[AMOUNT].median()
    )
    cat_tbl["median_amount_benign"] = (
        df[df[TARGET] == 0].groupby("category", observed=True)[AMOUNT].median()
    )
    R["eda"] = {
        "category_table": cat_tbl.reset_index().to_dict(orient="records"),
        "amount_median_fraud": float(df.loc[df[TARGET] == 1, AMOUNT].median()),
        "amount_median_benign": float(df.loc[df[TARGET] == 0, AMOUNT].median()),
        "amount_mean_fraud": float(df.loc[df[TARGET] == 1, AMOUNT].mean()),
        "amount_mean_benign": float(df.loc[df[TARGET] == 0, AMOUNT].mean()),
        "fraud_per_step_unique_values": sorted(
            df.groupby(STEP)[TARGET].sum().unique().tolist()
        ),
        "customers_with_fraud": int(df[df[TARGET] == 1]["customer"].nunique()),
        "merchants_with_fraud": int(df[df[TARGET] == 1]["merchant"].nunique()),
        "gender_table": df.groupby("gender", observed=True)[TARGET]
        .agg(n="size", fraud_rate="mean")
        .reset_index()
        .to_dict(orient="records"),
        "age_table": df.groupby("age", observed=True)[TARGET]
        .agg(n="size", fraud_rate="mean")
        .reset_index()
        .to_dict(orient="records"),
    }
    band_tbl = (
        pd.DataFrame(
            {"band": amount_band(df[AMOUNT]), "fraud": df[TARGET].to_numpy()}
        )
        .groupby("band", observed=True)["fraud"]
        .agg(transactions="size", fraud_count="sum", fraud_rate="mean")
        .reset_index()
    )
    R["eda"]["amount_band_table"] = band_tbl.to_dict(orient="records")
    d = df.sort_values(["customer", STEP]).copy()
    d["prev_fraud"] = d.groupby("customer", observed=True)[TARGET].shift(1)
    d = d.dropna(subset=["prev_fraud"])
    R["eda"]["fraud_rate_after_fraud"] = float(d.loc[d.prev_fraud == 1, TARGET].mean())
    R["eda"]["fraud_rate_after_benign"] = float(d.loc[d.prev_fraud == 0, TARGET].mean())

    # ------------------------------------------------------------------ 2. split + benchmark
    train, test = temporal_split(df, cfg.split)
    features = feature_columns(include_protected=True, include_time=True)
    y_tr = train[TARGET].to_numpy()
    y_te = test[TARGET].to_numpy()
    amt_tr = train[AMOUNT].to_numpy()
    amt_te = test[AMOUNT].to_numpy()
    n_steps_te = int(test[STEP].nunique())
    spw = positive_scale(y_tr)
    folds = list(time_series_folds(train[STEP], cfg.split))
    val_mask = np.zeros(len(train), dtype=bool)
    for _, va in folds:
        val_mask[va] = True
    n_steps_val = int(train.loc[val_mask, STEP].nunique())
    R["split"] = {
        "train_rows": len(train),
        "test_rows": len(test),
        "train_fraud": int(y_tr.sum()),
        "test_fraud": int(y_te.sum()),
        "train_steps": [int(train[STEP].min()), int(train[STEP].max())],
        "test_steps": [int(test[STEP].min()), int(test[STEP].max())],
        "n_cv_folds": len(folds),
        "validation_steps": [
            int(train.loc[val_mask, STEP].min()),
            int(train.loc[val_mask, STEP].max()),
        ],
        "scale_pos_weight": spw,
        "test_fraud_value": float(amt_te[y_te == 1].sum()),
        "test_transactions_per_step": float(len(test) / n_steps_te),
    }
    R["features"] = {
        "all": features,
        "numeric": [f for f in features if f not in CATEGORICAL_FEATURES],
        "categorical": [f for f in features if f in CATEGORICAL_FEATURES],
    }

    R["models"] = {}
    test_preds: dict[str, np.ndarray] = {}
    oof_preds: dict[str, np.ndarray] = {}
    fitted = {}
    for key, spec in MODEL_SPECS.items():
        tk = time.time()
        oof = np.full(len(train), np.nan)
        cv_scores = []
        for tr_idx, va_idx in folds:
            m = make_model(key, features, spw, cfg.random_state, cfg.n_jobs, cfg.quick)
            m.fit(train.iloc[tr_idx][features], y_tr[tr_idx])
            p = m.predict_proba(train.iloc[va_idx][features])[:, 1]
            oof[va_idx] = p
            cv_scores.append(ranking_metrics(y_tr[va_idx], p))
        sweep_val = threshold_sweep(
            y_tr[val_mask], oof[val_mask], amt_tr[val_mask], cfg.costs, n_steps_val
        )
        thr = choose_threshold(sweep_val, cfg.costs, "net_benefit")
        m = make_model(key, features, spw, cfg.random_state, cfg.n_jobs, cfg.quick)
        m.fit(train[features], y_tr)
        p_te = m.predict_proba(test[features])[:, 1]
        test_preds[key] = p_te
        oof_preds[key] = oof
        fitted[key] = m
        R["models"][key] = {
            "name": spec.name,
            "description": spec.description,
            "cv": _mean_std(cv_scores),
            "cv_folds": cv_scores,
            "threshold": thr,
            "test": ranking_metrics(y_te, p_te),
            "decision": decision_metrics(
                y_te, p_te, amt_te, thr, cfg.costs, n_steps_te
            ),
            "fit_seconds": time.time() - tk,
        }
        _log(
            f"model {key}: cv AP={R['models'][key]['cv']['average_precision_mean']:.4f} test AP={R['models'][key]['test']['average_precision']:.4f}",
            t0,
        )
    R["rules_text"] = rules_from_tree(fitted["rules"])

    names = {k: v.name for k, v in MODEL_SPECS.items()}
    plots.pr_curves(test_preds, y_te, figs / "model_pr_curves.png", names)
    plots.roc_curves(test_preds, y_te, figs / "model_roc_curves.png", names)
    comp = pd.DataFrame(
        {
            "cv_average_precision_mean": {
                names[k]: v["cv"]["average_precision_mean"]
                for k, v in R["models"].items()
            },
            "cv_average_precision_std": {
                names[k]: v["cv"]["average_precision_std"]
                for k, v in R["models"].items()
            },
            "test_average_precision": {
                names[k]: v["test"]["average_precision"] for k, v in R["models"].items()
            },
        }
    )
    plots.model_comparison(comp, figs / "model_comparison.png")
    champion_key = max(
        R["models"], key=lambda k: R["models"][k]["cv"]["average_precision_mean"]
    )
    R["champion_family"] = champion_key
    _log(f"champion family by CV average precision: {champion_key}", t0)

    # ------------------------------------------------------------------ 3. experiments on last validation fold
    tr_idx, va_idx = folds[-1]
    Xa, ya = train.iloc[tr_idx], y_tr[tr_idx]
    Xb, yb = train.iloc[va_idx], y_tr[va_idx]
    strategies = ["none", "class_weight", "undersample"] + (
        [] if cfg.quick else ["smote"]
    )
    R["imbalance"] = {}
    for s in strategies:
        tk = time.time()
        m = make_imbalance_variant(
            s, features, spw, cfg.random_state, cfg.n_jobs, cfg.quick
        )
        m.fit(Xa[features], ya)
        p = m.predict_proba(Xb[features])[:, 1]
        R["imbalance"][s] = {**ranking_metrics(yb, p), "fit_seconds": time.time() - tk}
        _log(
            f"imbalance {s}: AP={R['imbalance'][s]['average_precision']:.4f} brier={R['imbalance'][s]['brier']:.4f}",
            t0,
        )
    imb = pd.DataFrame(R["imbalance"]).T
    imb.index = [
        "No treatment",
        "Class weights",
        "Under-sampling 1:10",
        "SMOTE-NC 1:10",
    ][: len(imb)]
    plots.strategy_comparison(
        imb,
        figs / "exp_imbalance_strategies.png",
        "Class-imbalance treatments (LightGBM, last validation fold)",
        "Average precision (validation)",
    )

    ablations = {
        "Raw fields only": RAW_FEATURES,
        "Raw + time-of-day": RAW_FEATURES + ["hour_of_day", "day_of_week"],
        "Full minus time-of-day": feature_columns(True, include_time=False),
        "Full feature set": features,
    }
    R["ablation"] = {}
    for name, cols in ablations.items():
        m = make_model(champion_key, cols, spw, cfg.random_state, cfg.n_jobs, cfg.quick)
        m.fit(Xa[cols], ya)
        p = m.predict_proba(Xb[cols])[:, 1]
        R["ablation"][name] = {**ranking_metrics(yb, p), "n_features": len(cols)}
        _log(f"ablation {name}: AP={R['ablation'][name]['average_precision']:.4f}", t0)
    abl = pd.DataFrame(R["ablation"]).T
    plots.strategy_comparison(
        abl,
        figs / "exp_feature_ablation.png",
        "Feature-set ablation (champion family, last validation fold)",
        "Average precision (validation)",
    )

    # ------------------------------------------------------------------ 4. protected attributes
    features_np = feature_columns(include_protected=False, include_time=False)
    features_p = feature_columns(include_protected=True, include_time=False)
    m_with = make_model(
        champion_key, features_p, spw, cfg.random_state, cfg.n_jobs, cfg.quick
    ).fit(train[features_p], y_tr)
    m_without = make_model(
        champion_key, features_np, spw, cfg.random_state, cfg.n_jobs, cfg.quick
    ).fit(train[features_np], y_tr)
    p_with = m_with.predict_proba(test[features_p])[:, 1]
    p_without = m_without.predict_proba(test[features_np])[:, 1]
    ap_with = average_precision_score(y_te, p_with)
    ap_without = average_precision_score(y_te, p_without)
    R["protected"] = {
        "attributes": list(PROTECTED_ATTRIBUTES),
        "with": {
            "average_precision": float(ap_with),
            "roc_auc": float(roc_auc_score(y_te, p_with)),
        },
        "without": {
            "average_precision": float(ap_without),
            "roc_auc": float(roc_auc_score(y_te, p_without)),
        },
        "ap_drop": float(ap_with - ap_without),
    }
    tolerance = 0.01
    use_protected = (ap_with - ap_without) > tolerance
    R["protected"]["decision"] = (
        "Retain protected attributes pending legal review: removing them costs more than 1pt of average precision."
        if use_protected
        else "Exclude age and gender from the production model: the precision cost is negligible and the regulatory/reputational risk is not."
    )
    _log(f"protected attributes: AP with={ap_with:.4f} without={ap_without:.4f}", t0)

    # ------------------------------------------------------------------ 5. tuning + production candidate
    final_features = features_p if use_protected else features_np
    params = {}
    tuning_path = cfg.metrics_dir / "tuning.json"
    if cfg.tune and champion_key in ("lgbm", "xgb"):
        from .tuning import tune_booster

        R["tuning"] = tune_booster(champion_key, train, final_features, spw, cfg)
        tuning_path.write_text(json.dumps(R["tuning"], indent=2))
        params = R["tuning"]["best_params"]
        _log(
            f"tuning done ({champion_key}): best CV AP={R['tuning']['best_cv_average_precision']:.4f}",
            t0,
        )
    elif tuning_path.exists():
        saved = json.loads(tuning_path.read_text())
        if saved.get("family") == champion_key:
            R["tuning"] = saved
            params = saved["best_params"]
            _log(f"loaded previously tuned parameters for {champion_key}", t0)

    oof = np.full(len(train), np.nan)
    cv_scores = []
    for tr_idx, va_idx in folds:
        m = make_model(
            champion_key,
            final_features,
            spw,
            cfg.random_state,
            cfg.n_jobs,
            cfg.quick,
            params,
        )
        m.fit(train.iloc[tr_idx][final_features], y_tr[tr_idx])
        p = m.predict_proba(train.iloc[va_idx][final_features])[:, 1]
        oof[va_idx] = p
        cv_scores.append(ranking_metrics(y_tr[va_idx], p))
    sweep_val = threshold_sweep(
        y_tr[val_mask], oof[val_mask], amt_tr[val_mask], cfg.costs, n_steps_val
    )
    thr = choose_threshold(sweep_val, cfg.costs, "net_benefit")
    thr_cap = choose_threshold(sweep_val, cfg.costs, "capacity")
    prod = make_model(
        champion_key,
        final_features,
        spw,
        cfg.random_state,
        cfg.n_jobs,
        cfg.quick,
        params,
    )
    prod.fit(train[final_features], y_tr)
    p_prod = prod.predict_proba(test[final_features])[:, 1]
    sweep_test = threshold_sweep(y_te, p_prod, amt_te, cfg.costs, n_steps_te)
    dm = decision_metrics(y_te, p_prod, amt_te, thr, cfg.costs, n_steps_te)
    dm_cap = decision_metrics(y_te, p_prod, amt_te, thr_cap, cfg.costs, n_steps_te)
    dm_rules = R["models"]["rules"]["decision"]
    R["production"] = {
        "family": champion_key,
        "name": MODEL_SPECS[champion_key].name,
        "features": final_features,
        "uses_protected_attributes": bool(use_protected),
        "params": params,
        "cv": _mean_std(cv_scores),
        "threshold": thr,
        "threshold_capacity": thr_cap,
        "test": ranking_metrics(y_te, p_prod),
        "decision": dm,
        "decision_capacity": dm_cap,
        "validation_sweep": sweep_val.to_dict(orient="records"),
        "test_sweep": sweep_test.to_dict(orient="records"),
        "calibration": calibration_table(y_te, p_prod).to_dict(orient="records"),
        "vs_rules": {
            "alerts_per_step_rules": dm_rules["alerts_per_step"],
            "alerts_per_step_model": dm["alerts_per_step"],
            "false_positives_per_step_rules": dm_rules["fp"] / n_steps_te,
            "false_positives_per_step_model": dm["fp"] / n_steps_te,
            "recall_rules": dm_rules["recall"],
            "recall_model": dm["recall"],
            "value_recall_rules": dm_rules["value_recall"],
            "value_recall_model": dm["value_recall"],
            "net_benefit_rules": dm_rules["net_benefit"],
            "net_benefit_model": dm["net_benefit"],
        },
    }
    plots.cost_curve(
        sweep_test,
        thr,
        figs / "prod_cost_curve.png",
        cfg.costs.alerts_per_step_capacity,
    )
    plots.confusion(dm, figs / "prod_confusion.png", n_steps_te)
    plots.calibration(
        pd.DataFrame(R["production"]["calibration"]), figs / "prod_calibration.png"
    )
    _log(
        f"production candidate: test AP={R['production']['test']['average_precision']:.4f} thr={thr:.2f} precision={dm['precision']:.3f} recall={dm['recall']:.3f}",
        t0,
    )

    # ------------------------------------------------------------------ 5b. robustness: uncertainty, segments, stability, cost sensitivity, throughput
    ts = time.time()
    prod.predict_proba(test[final_features])
    scoring_seconds = time.time() - ts
    R["production"]["scoring"] = {
        "rows": len(test),
        "seconds": scoring_seconds,
        "rows_per_second": len(test) / max(scoring_seconds, 1e-9),
    }
    R["production"]["bootstrap_ci"] = bootstrap_ci(
        y_te,
        p_prod,
        amt_te,
        thr,
        n_boot=50 if cfg.quick else 200,
        seed=cfg.random_state,
    )
    R["segments"] = {
        "category": segment_performance(
            test["category"], y_te, p_prod, amt_te, thr
        ).to_dict(orient="records"),
        "amount_band": segment_performance(
            amount_band(amt_te), y_te, p_prod, amt_te, thr
        ).to_dict(orient="records"),
    }
    stab = stability_over_time(test[STEP], y_te, p_prod, thr)
    R["stability"] = stab.to_dict(orient="records")
    plots.stability(stab, figs / "prod_stability.png", thr)
    sens = cost_sensitivity(
        y_tr[val_mask],
        amt_tr[val_mask],
        n_steps_val,
        y_te,
        amt_te,
        n_steps_te,
        {"model": oof[val_mask], "rules": oof_preds["rules"][val_mask]},
        {"model": p_prod, "rules": test_preds["rules"]},
        cfg.costs,
    )
    R["cost_sensitivity"] = sens.to_dict(orient="records")
    _log(
        f"robustness: AP 95% CI {R['production']['bootstrap_ci']['average_precision']['low']:.3f}-{R['production']['bootstrap_ci']['average_precision']['high']:.3f}; scoring {R['production']['scoring']['rows_per_second']:,.0f} rows/s",
        t0,
    )

    # ------------------------------------------------------------------ 6. random split comparison (why temporal matters)
    if not cfg.quick:
        rtr, rte = random_split_for_comparison(
            df, test_size=len(test) / len(df), seed=cfg.random_state
        )
        m = make_model(
            champion_key,
            final_features,
            spw,
            cfg.random_state,
            cfg.n_jobs,
            cfg.quick,
            params,
        )
        m.fit(rtr[final_features], rtr[TARGET])
        pr = m.predict_proba(rte[final_features])[:, 1]
        R["random_split_comparison"] = {
            "random_split_average_precision": float(
                average_precision_score(rte[TARGET], pr)
            ),
            "temporal_split_average_precision": float(
                R["production"]["test"]["average_precision"]
            ),
        }
        _log("random split comparison done", t0)

    # ------------------------------------------------------------------ 7. interpretation
    rng = np.random.default_rng(cfg.random_state)
    n_sample = min(len(test), 3000 if cfg.quick else 20000)
    idx = rng.choice(len(test), n_sample, replace=False)
    sample = test.iloc[idx]
    vals, base, Xt = shap_values(prod, sample[final_features])
    imp = global_importance(vals, list(Xt.columns))
    gimp = gain_importance(prod, final_features)
    plots.shap_summary(vals, Xt, figs / "interp_shap_summary.png")
    plots.shap_bar(imp, figs / "interp_shap_bar.png")
    plots.shap_dependence(
        vals, Xt, "amount", figs / "interp_shap_dependence_amount.png"
    )
    vals_all, _, Xt_all = (
        shap_values(prod, test[final_features]) if not cfg.quick else (vals, base, Xt)
    )
    y_loc = y_te if not cfg.quick else y_te[idx]
    p_loc = p_prod if not cfg.quick else p_prod[idx]
    a_loc = amt_te if not cfg.quick else amt_te[idx]
    examples = local_examples(vals_all, Xt_all, y_loc, p_loc, a_loc, thr)
    for ex in examples:
        plots.local_explanation(ex, base, figs / f"interp_local_{ex['kind']}.png")
    R["interpretation"] = {
        "shap_sample_size": int(n_sample),
        "expected_value": base,
        "global_importance": imp.to_dict(orient="records"),
        "gain_importance": gimp.to_dict(orient="records"),
        "local_examples": examples,
    }
    _log("interpretation done", t0)

    # ------------------------------------------------------------------ 8. fairness / group outcomes
    R["fairness"] = {}
    for col in PROTECTED_ATTRIBUTES:
        tbl_prod = group_outcomes(test[col], y_te, p_prod, thr)
        tbl_with = group_outcomes(test[col], y_te, p_with, thr)
        R["fairness"][col] = {
            "production_model": tbl_prod.to_dict(orient="records"),
            "model_with_protected": tbl_with.to_dict(orient="records"),
        }
        plots.group_outcomes_plot(
            tbl_prod,
            figs / f"fair_{col}_outcomes.png",
            f"Outcomes by {col} at the production threshold (model without protected attributes)",
        )
    _log("fairness tables done", t0)

    # ------------------------------------------------------------------ 9. persist
    save_bundle(
        cfg.models_dir / "production_model.joblib",
        prod,
        final_features,
        thr,
        thr_cap,
        champion_key,
        metadata={
            "training_rows": len(train),
            "training_steps": R["split"]["train_steps"],
            "holdout_steps": R["split"]["test_steps"],
            "params": params,
            "holdout_metrics": {
                k: R["production"]["test"][k]
                for k in ("roc_auc", "average_precision", "brier")
            }
            | {k: dm[k] for k in ("precision", "recall", "alert_rate", "value_recall")},
        },
    )
    recent = (
        train[STEP] >= cfg.split.test_start_step - cfg.split.cv_fold_length
    ).to_numpy()
    reference = build_reference(
        train.loc[recent, final_features], final_features, oof[recent], thr
    )
    (cfg.models_dir / "monitoring_reference.json").write_text(
        json.dumps(reference, indent=2)
    )
    R["runtime_seconds"] = time.time() - t0
    (cfg.metrics_dir / "results.json").write_text(
        json.dumps(R, indent=2, default=_json_default)
    )
    _log("pipeline finished", t0)
    return R


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)
