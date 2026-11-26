"""Model pipelines.

Everything that is *learned* from the data lives inside a scikit-learn
``Pipeline``. That is not a style preference -- it is what makes the
cross-validation honest.

When a pipeline is cross-validated, scikit-learn fits the whole pipeline on
each training fold. So the imputation medians, the set of locality categories
and the scaling factors are all recomputed from that fold alone, and the
held-out fold is genuinely held out.

The common alternative -- impute and encode the whole dataset up front, then
split -- lets every fold peek at statistics computed from the data it is about
to be tested on. The score comes out better and means less. That is the
mistake this file is arranged to make impossible.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hpp.features import BOOLEAN_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES

# Localities with fewer than this many listings in the training fold are
# pooled into a single "infrequent" category rather than getting a column of
# their own. With 1,265 localities over ~7,500 training rows, a locality seen
# twice cannot support its own coefficient; it would just memorise two prices.
MIN_LOCATION_FREQUENCY = 10


def make_preprocessor(*, scale: bool) -> ColumnTransformer:
    """Build the preprocessing step.

    ``scale=True`` for models that care about the size of a feature (ridge
    regression), ``False`` for tree-based models, which split on order and are
    unaffected by scale.
    """
    numeric_steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="median")),
    ]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(
                    # A locality in the test set that never appeared in training
                    # is mapped to the pooled "infrequent" category instead of
                    # raising. Roughly 6% of test rows are in this position.
                    handle_unknown="infrequent_if_exist",
                    min_frequency=MIN_LOCATION_FREQUENCY,
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
            ("boolean", "passthrough", BOOLEAN_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_model(estimator, *, scale: bool, log_target: bool = True):
    """Wrap an estimator in preprocessing and, optionally, a log target.

    Why a log target
    ----------------
    Prices run from 8 lakh to 3,600 lakh and are heavily right-skewed. Fitting
    on the raw price means the handful of very expensive properties dominate
    the loss, and the model buys accuracy at the top of the market with large
    percentage errors on the ordinary flats that make up most of the data.

    Training on ``log(price)`` makes the model care about *proportional* error
    instead. Predictions are converted straight back to lakhs, so every metric
    reported is still in money -- the transform is an internal detail, not a
    change to what is being measured.
    """
    pipeline = Pipeline(
        [
            ("prepare", make_preprocessor(scale=scale)),
            ("model", estimator),
        ]
    )

    if not log_target:
        return pipeline

    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log,
        inverse_func=np.exp,
    )
