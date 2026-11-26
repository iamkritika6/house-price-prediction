"""Score the final model on the test set, once, and produce the figures.

Run:
    docker compose run --rm hpp python scripts/evaluate.py

This is the only script that reads data/interim/test.csv. Every choice --
which model, which settings, which features -- was made before it ran, using
the training and validation sets only.

The final model is refitted on train + validation before being scored. That is
standard: once the choices are fixed, there is no reason to withhold the
validation rows from the model that ships. The test set stays out.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold

from hpp import plots
from hpp.baselines import MedianBaseline, PricePerSqftBaseline
from hpp.features import add_features
from hpp.final_model import DESCRIPTION, make_final_model
from hpp.metrics import score, score_by_band
from hpp.split import TARGET, property_fingerprint

INTERIM = Path("data/interim")
METRICS_DIR = Path("reports/metrics")
SEED = 42


def main() -> int:
    train = add_features(pd.read_csv(INTERIM / "train.csv"))
    validation = add_features(pd.read_csv(INTERIM / "validation.csv"))
    test = add_features(pd.read_csv(INTERIM / "test.csv"))

    fit_on = pd.concat([train, validation], ignore_index=True)
    y_fit = fit_on[TARGET].to_numpy(dtype=float)
    y_test = test[TARGET].to_numpy(dtype=float)

    print(f"final model: {DESCRIPTION}")
    print(f"fitted on {len(fit_on)} rows (train + validation)")
    print(f"scored on {len(test)} test rows, which have not been used before\n")

    # Cross-validated score of the final configuration, on the training set
    # only. Reported so the README can quote a figure with a spread alongside
    # the single test number.
    y_train = train[TARGET].to_numpy(dtype=float)
    groups = property_fingerprint(train)
    splitter = GroupKFold(n_splits=5)
    fold_rmse = []
    for train_idx, valid_idx in splitter.split(train, y_train, groups=groups):
        folded = make_final_model()
        folded.fit(train.iloc[train_idx], y_train[train_idx])
        fold_rmse.append(score(y_train[valid_idx], folded.predict(train.iloc[valid_idx])).rmse_lakh)
    cv_mean = float(np.mean(fold_rmse))
    cv_std = float(np.std(fold_rmse, ddof=1))
    print(f"cross-validated RMSE on the training set: {cv_mean:.2f} +/- {cv_std:.2f} lakh\n")

    model = make_final_model()
    model.fit(fit_on, y_fit)
    predictions = model.predict(test)

    final = score(y_test, predictions)
    print("=== TEST SET RESULT ===")
    print(final.describe())

    # Baselines on the same test set, for context.
    print("\n=== the same test set, for comparison ===")
    comparison = {"final_model": final.as_dict()}
    for name, baseline in (
        ("baseline_median", MedianBaseline()),
        ("baseline_price_per_sqft", PricePerSqftBaseline()),
    ):
        baseline.fit(fit_on, y_fit)
        s = score(y_test, baseline.predict(test))
        comparison[name] = s.as_dict()
        print(f"{name:26s} {s.describe()}")

    improvement_median = 100 * (1 - final.mae_lakh / comparison["baseline_median"]["mae_lakh"])
    improvement_ppsf = 100 * (1 - final.mae_lakh / comparison["baseline_price_per_sqft"]["mae_lakh"])
    print(f"\nfinal model MAE is {improvement_median:.1f}% below the median baseline")
    print(f"final model MAE is {improvement_ppsf:.1f}% below the price-per-sqft baseline")

    # --- where does it fail? ---
    print("\n=== error by price band ===")
    bands = score_by_band(y_test, predictions)
    print(f"{'band':16s} {'n':>6s} {'median APE':>12s} {'MAE (lakh)':>12s}")
    for label, s in bands.items():
        print(f"{label:16s} {s.n:6d} {s.median_ape_percent:11.1f}% {s.mae_lakh:12.2f}")

    # --- seen vs unseen localities ---
    known = set(fit_on["location"].dropna())
    seen_mask = test["location"].isin(known).to_numpy()
    print("\n=== localities the model has seen vs not ===")
    seen_unseen = {}
    for label, mask in (("seen in training", seen_mask), ("never seen", ~seen_mask)):
        if mask.sum() == 0:
            continue
        s = score(y_test[mask], predictions[mask])
        seen_unseen[label] = s.as_dict()
        print(f"{label:20s} n={s.n:5d}  median APE {s.median_ape_percent:5.1f}%  MAE {s.mae_lakh:7.2f} lakh")

    # --- permutation importance, measured on the test set predictions ---
    print("\n=== permutation importance (on the test set) ===")
    result = permutation_importance(
        model,
        test,
        y_test,
        n_repeats=10,
        random_state=SEED,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    )
    feature_names = list(test.columns)
    means = result.importances_mean
    stds = result.importances_std

    order = np.argsort(means)[::-1]
    importance_table = []
    for i in order[:15]:
        print(f"  {feature_names[i]:20s} {means[i]:8.2f} +/- {stds[i]:5.2f} lakh RMSE")
        importance_table.append(
            {"feature": feature_names[i], "mean": float(means[i]), "std": float(stds[i])}
        )

    # --- figures ---
    print("\n=== figures ===")
    for path in (
        plots.predicted_vs_actual(y_test, predictions),
        plots.residuals(y_test, predictions),
        plots.error_by_band(bands),
        plots.importance(feature_names, means, stds),
    ):
        print(f"  wrote {path}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = METRICS_DIR / "test_results.json"
    out.write_text(
        json.dumps(
            {
                "model": DESCRIPTION,
                "cv_rmse_lakh_mean": cv_mean,
                "cv_rmse_lakh_std": cv_std,
                "n_fit_rows": len(fit_on),
                "comparison_on_test": comparison,
                "improvement_vs_median_percent": improvement_median,
                "improvement_vs_price_per_sqft_percent": improvement_ppsf,
                "by_price_band": {k: v.as_dict() for k, v in bands.items()},
                "seen_vs_unseen_locality": seen_unseen,
                "permutation_importance": importance_table,
            },
            indent=2,
        )
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
