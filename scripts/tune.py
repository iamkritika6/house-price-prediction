"""Try to improve on the first model comparison.

Run:
    docker compose run --rm hpp python scripts/tune.py

Two questions, answered in order:

1. Does encoding locality by its typical price beat one column per locality?
2. Do better LightGBM settings beat the defaults?

Both are decided by grouped cross-validation on the **training set**, and
confirmed once on the validation set. The test set is still not touched.
Tuning against the test set is how projects end up reporting a number that
does not survive contact with new data.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

from hpp.features import add_features
from hpp.metrics import score
from hpp.pipeline import make_model
from hpp.split import TARGET, property_fingerprint

INTERIM = Path("data/interim")
METRICS_DIR = Path("reports/metrics")
SEED = 42
N_FOLDS = 5


def grouped_cv_rmse(model, X, y, groups) -> tuple[float, float]:
    """Mean and standard deviation of RMSE across grouped folds."""
    splitter = GroupKFold(n_splits=N_FOLDS)
    scores = []
    for train_idx, valid_idx in splitter.split(X, y, groups=groups):
        fitted = clone(model)
        fitted.fit(X.iloc[train_idx], y[train_idx])
        scores.append(score(y[valid_idx], fitted.predict(X.iloc[valid_idx])).rmse_lakh)
    values = np.array(scores)
    return float(values.mean()), float(values.std(ddof=1))


def main() -> int:
    train = add_features(pd.read_csv(INTERIM / "train.csv"))
    validation = add_features(pd.read_csv(INTERIM / "validation.csv"))

    y_train = train[TARGET].to_numpy(dtype=float)
    y_validation = validation[TARGET].to_numpy(dtype=float)
    groups = property_fingerprint(train)

    results: dict[str, dict] = {}

    # ------------------------------------------------------------------
    print("=== 1. how should locality be encoded? ===\n")
    print(f"{'model':34s} {'CV RMSE (lakh)':>22s}")
    print("-" * 58)

    encoding_models = {
        "ridge / onehot": make_model(Ridge(alpha=1.0), scale=True, location_encoding="onehot"),
        "ridge / target": make_model(Ridge(alpha=1.0), scale=True, location_encoding="target"),
        "random_forest / onehot": make_model(
            RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
            scale=False,
            location_encoding="onehot",
        ),
        "random_forest / target": make_model(
            RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
            scale=False,
            location_encoding="target",
        ),
        "lightgbm / onehot": make_model(
            LGBMRegressor(n_estimators=600, learning_rate=0.05, random_state=SEED, n_jobs=-1, verbose=-1),
            scale=False,
            location_encoding="onehot",
        ),
        "lightgbm / target": make_model(
            LGBMRegressor(n_estimators=600, learning_rate=0.05, random_state=SEED, n_jobs=-1, verbose=-1),
            scale=False,
            location_encoding="target",
        ),
    }

    encoding_results = {}
    for name, model in encoding_models.items():
        mean, std = grouped_cv_rmse(model, train, y_train, groups)
        encoding_results[name] = {"cv_rmse_mean": mean, "cv_rmse_std": std}
        print(f"{name:34s} {mean:10.2f} +/- {std:6.2f}")

    results["location_encoding"] = encoding_results

    # ------------------------------------------------------------------
    print("\n=== 2. tuning LightGBM ===\n")

    search_space = {
        "regressor__model__n_estimators": [300, 600, 900, 1500],
        "regressor__model__learning_rate": [0.01, 0.02, 0.05, 0.1],
        "regressor__model__num_leaves": [15, 31, 63, 127],
        "regressor__model__min_child_samples": [5, 10, 20, 40],
        "regressor__model__subsample": [0.7, 0.85, 1.0],
        "regressor__model__subsample_freq": [0, 1],
        "regressor__model__colsample_bytree": [0.6, 0.8, 1.0],
        "regressor__model__reg_lambda": [0.0, 1.0, 10.0],
    }

    base = make_model(
        LGBMRegressor(random_state=SEED, n_jobs=-1, verbose=-1),
        scale=False,
        location_encoding="target",
    )

    search = RandomizedSearchCV(
        base,
        param_distributions=search_space,
        n_iter=40,
        scoring="neg_root_mean_squared_error",
        cv=GroupKFold(n_splits=N_FOLDS),
        random_state=SEED,
        n_jobs=1,
        refit=True,
    )

    started = time.perf_counter()
    search.fit(train, y_train, groups=groups)
    elapsed = time.perf_counter() - started
    print(f"searched 40 settings in {elapsed:.0f}s")

    print("\nbest settings found:")
    for key, value in sorted(search.best_params_.items()):
        print(f"  {key.replace('regressor__model__', ''):22s} {value}")

    tuned_mean, tuned_std = grouped_cv_rmse(search.best_estimator_, train, y_train, groups)
    print(f"\ntuned lightgbm CV RMSE: {tuned_mean:.2f} +/- {tuned_std:.2f}")

    results["lightgbm_tuning"] = {
        "best_params": {k.replace("regressor__model__", ""): v for k, v in search.best_params_.items()},
        "cv_rmse_mean": tuned_mean,
        "cv_rmse_std": tuned_std,
        "n_settings_tried": 40,
        "search_seconds": elapsed,
    }

    # ------------------------------------------------------------------
    print("\n=== 3. confirm on the validation set ===\n")

    final_candidates = {
        "lightgbm_default_onehot": encoding_models["lightgbm / onehot"],
        "lightgbm_default_target": encoding_models["lightgbm / target"],
        "lightgbm_tuned_target": search.best_estimator_,
        "random_forest_target": encoding_models["random_forest / target"],
    }

    validation_results = {}
    for name, model in final_candidates.items():
        fitted = clone(model)
        fitted.fit(train, y_train)
        s = score(y_validation, fitted.predict(validation))
        validation_results[name] = s.as_dict()
        print(f"{name:26s} {s.describe()}")

    results["validation"] = validation_results

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = METRICS_DIR / "tuning.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
