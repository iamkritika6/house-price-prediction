"""Cross-validate the candidate models and compare them on the validation set.

Run:
    docker compose run --rm hpp python scripts/train.py

The test set is NOT touched here. It is scored once, by scripts/evaluate.py,
after the model choice is final.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from hpp.baselines import MedianBaseline, PricePerSqftBaseline
from hpp.features import add_features
from hpp.metrics import score
from hpp.pipeline import make_model
from hpp.split import TARGET, property_fingerprint

INTERIM = Path("data/interim")
METRICS_DIR = Path("reports/metrics")
SEED = 42
N_FOLDS = 5


def candidates() -> dict[str, object]:
    """The models to compare, cheapest and dumbest first."""
    return {
        "baseline_median": MedianBaseline(),
        "baseline_price_per_sqft": PricePerSqftBaseline(),
        "ridge": make_model(Ridge(alpha=1.0), scale=True),
        "random_forest": make_model(
            RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=SEED,
                n_jobs=-1,
            ),
            scale=False,
        ),
        "lightgbm": make_model(
            LGBMRegressor(
                n_estimators=600,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                random_state=SEED,
                n_jobs=-1,
                verbose=-1,
            ),
            scale=False,
        ),
    }


def cross_validate(name: str, model, X: pd.DataFrame, y: np.ndarray, groups: pd.Series) -> dict:
    """Group k-fold cross-validation, reporting the spread across folds.

    Grouped, so that the same property never appears in a fold's training and
    validation halves at once -- the same reason the main split is grouped.

    The spread matters as much as the average. A single split gives one
    number, and you cannot tell whether a model is better than another or just
    luckier on that particular slice.
    """
    splitter = GroupKFold(n_splits=N_FOLDS)
    fold_scores = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y, groups=groups), 1):
        from sklearn.base import clone

        fitted = clone(model)
        fitted.fit(X.iloc[train_idx], y[train_idx])
        predictions = fitted.predict(X.iloc[valid_idx])
        fold_scores.append(score(y[valid_idx], predictions))

    rmse = np.array([s.rmse_lakh for s in fold_scores])
    mae = np.array([s.mae_lakh for s in fold_scores])
    mape = np.array([s.mape_percent for s in fold_scores])

    return {
        "rmse_lakh_mean": float(rmse.mean()),
        "rmse_lakh_std": float(rmse.std(ddof=1)),
        "mae_lakh_mean": float(mae.mean()),
        "mae_lakh_std": float(mae.std(ddof=1)),
        "mape_percent_mean": float(mape.mean()),
        "mape_percent_std": float(mape.std(ddof=1)),
        "rmse_per_fold": [float(v) for v in rmse],
    }


def main() -> int:
    train = add_features(pd.read_csv(INTERIM / "train.csv"))
    validation = add_features(pd.read_csv(INTERIM / "validation.csv"))

    y_train = train[TARGET].to_numpy(dtype=float)
    y_validation = validation[TARGET].to_numpy(dtype=float)
    groups = property_fingerprint(train)

    print(f"train:      {len(train)} rows")
    print(f"validation: {len(validation)} rows")
    print(f"{N_FOLDS}-fold grouped cross-validation on the training set\n")

    results: dict[str, dict] = {}

    print(f"{'model':26s} {'CV RMSE (lakh)':>22s} {'CV MAE (lakh)':>20s} {'CV MAPE %':>18s} {'fit s':>7s}")
    print("-" * 98)

    for name, model in candidates().items():
        started = time.perf_counter()
        cv = cross_validate(name, model, train, y_train, groups)
        elapsed = time.perf_counter() - started

        from sklearn.base import clone

        fitted = clone(model)
        fitted.fit(train, y_train)
        validation_score = score(y_validation, fitted.predict(validation))

        results[name] = {
            "cross_validation": cv,
            "validation": validation_score.as_dict(),
        }

        print(
            f"{name:26s} "
            f"{cv['rmse_lakh_mean']:10.2f} +/- {cv['rmse_lakh_std']:6.2f} "
            f"{cv['mae_lakh_mean']:11.2f} +/- {cv['mae_lakh_std']:4.2f} "
            f"{cv['mape_percent_mean']:9.1f} +/- {cv['mape_percent_std']:4.1f} "
            f"{elapsed:7.1f}"
        )

    print("\n--- held-out validation set (each model fitted on the full training set) ---")
    for name, result in results.items():
        v = result["validation"]
        print(
            f"{name:26s} RMSE {v['rmse_lakh']:8.2f} lakh  "
            f"MAE {v['mae_lakh']:7.2f} lakh (Rs {v['mae_rupees']:>11,.0f})  "
            f"MAPE {v['mape_percent']:5.1f}%  "
            f"median APE {v['median_ape_percent']:5.1f}%  "
            f"R2 {v['r2']:6.3f}"
        )

    print("\n--- improvement over the baselines, on validation MAE ---")
    median_mae = results["baseline_median"]["validation"]["mae_lakh"]
    ppsf_mae = results["baseline_price_per_sqft"]["validation"]["mae_lakh"]
    for name, result in results.items():
        if name.startswith("baseline"):
            continue
        mae = result["validation"]["mae_lakh"]
        print(
            f"{name:26s} "
            f"{100 * (1 - mae / median_mae):6.1f}% better than predicting the median, "
            f"{100 * (1 - mae / ppsf_mae):6.1f}% better than price-per-sqft"
        )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = METRICS_DIR / "model_comparison.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
