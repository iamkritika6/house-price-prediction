"""Explain individual predictions with SHAP.

Run:
    docker compose run --rm hpp python scripts/explain_shap.py

Permutation importance answers "which features matter overall?". SHAP answers
"why did *this* property get *this* price?", which is the question a user of a
valuation tool actually asks.

A note on reading the numbers
-----------------------------
The model is trained on log(price), so SHAP contributions are in log space.
A contribution of +0.2 does not mean "+0.2 lakh" -- it means the price is
multiplied by exp(0.2), about +22%. The script converts them for readability.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from hpp.features import add_features  # noqa: E402
from hpp.final_model import make_final_model  # noqa: E402
from hpp.split import TARGET  # noqa: E402

INTERIM = Path("data/interim")
FIGURE_DIR = Path("reports/figures")
METRICS_DIR = Path("reports/metrics")
N_EXAMPLES = 3


def main() -> int:
    train = add_features(pd.read_csv(INTERIM / "train.csv"))
    validation = add_features(pd.read_csv(INTERIM / "validation.csv"))
    test = add_features(pd.read_csv(INTERIM / "test.csv"))

    fit_on = pd.concat([train, validation], ignore_index=True)
    y_fit = fit_on[TARGET].to_numpy(dtype=float)
    y_test = test[TARGET].to_numpy(dtype=float)

    model = make_final_model()
    model.fit(fit_on, y_fit)

    # The fitted pipeline sits inside the target transformer.
    pipeline = model.regressor_
    preprocessor = pipeline.named_steps["prepare"]
    estimator = pipeline.named_steps["model"]

    X_test = preprocessor.transform(test)
    feature_names = list(preprocessor.get_feature_names_out())

    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_test)

    print(f"explained {X_test.shape[0]} test rows over {X_test.shape[1]} features")

    # --- overall picture ---
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        show=False,
        max_display=15,
    )
    plt.title("SHAP: how each feature moves the predicted price")
    plt.tight_layout()
    beeswarm = FIGURE_DIR / "shap_summary.png"
    plt.savefig(beeswarm, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"wrote {beeswarm}")

    # --- individual explanations ---
    predictions = model.predict(test)
    errors = np.abs(predictions - y_test) / y_test

    chosen = {
        "typical": int(np.argsort(np.abs(errors - np.median(errors)))[0]),
        "best": int(np.argmin(errors)),
        "worst": int(np.argmax(errors)),
    }

    explanations = {}
    print("\n=== individual predictions explained ===")
    for label, idx in chosen.items():
        row = test.iloc[idx]
        contributions = shap_values[idx]
        order = np.argsort(np.abs(contributions))[::-1][:6]

        print(f"\n--- {label} prediction ---")
        print(
            f"  {row['bedrooms']:.0f} bed, {row['total_sqft_num']:.0f} sqft, "
            f"{row['location']}, {row['area_type']}"
        )
        print(f"  actual    {y_test[idx]:8.1f} lakh")
        print(f"  predicted {predictions[idx]:8.1f} lakh   ({100 * errors[idx]:.1f}% out)")
        baseline = float(np.ravel(explainer.expected_value)[0])
        print(f"  starting point (average log price): {baseline:.3f}")

        drivers = []
        for i in order:
            multiplier = float(np.exp(contributions[i]))
            direction = "raises" if contributions[i] > 0 else "lowers"
            print(
                f"    {feature_names[i]:28s} {direction} the price by "
                f"{abs(multiplier - 1) * 100:5.1f}%   (value {X_test[idx, i]:.2f})"
            )
            drivers.append(
                {
                    "feature": feature_names[i],
                    "value": float(X_test[idx, i]),
                    "shap_log": float(contributions[i]),
                    "percent_effect": float((multiplier - 1) * 100),
                }
            )

        explanations[label] = {
            "actual_lakh": float(y_test[idx]),
            "predicted_lakh": float(predictions[idx]),
            "absolute_percentage_error": float(100 * errors[idx]),
            "location": str(row["location"]),
            "bedrooms": float(row["bedrooms"]),
            "total_sqft": float(row["total_sqft_num"]),
            "top_drivers": drivers,
        }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = METRICS_DIR / "shap_examples.json"
    out.write_text(json.dumps(explanations, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
