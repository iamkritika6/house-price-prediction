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
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from hpp.features import BOOLEAN_FEATURES, NUMERIC_FEATURES

# Localities with fewer than this many listings in the training fold are
# pooled into a single "infrequent" category rather than getting a column of
# their own. With 1,265 localities over ~7,500 training rows, a locality seen
# twice cannot support its own coefficient; it would just memorise two prices.
MIN_LOCATION_FREQUENCY = 10


def make_preprocessor(*, scale: bool, location_encoding: str = "onehot") -> ColumnTransformer:
    """Build the preprocessing step.

    ``scale=True`` for models that care about the size of a feature (ridge
    regression), ``False`` for tree-based models, which split on order and are
    unaffected by scale.

    ``location_encoding`` chooses how the 1,265 localities are represented:

    ``"onehot"``
        One column per locality, with localities seen fewer than
        ``MIN_LOCATION_FREQUENCY`` times pooled into one "infrequent" column.
        Simple and completely safe, but it throws away the ordering -- the
        model has no way to know that two expensive localities resemble each
        other.

    ``"target"``
        Replace each locality with a number summarising its prices in the
        training data. Far more informative, and the obvious way to leak:
        if a row's own price helps compute the value used to predict that
        same row, the model is reading the answer.

        ``TargetEncoder`` avoids this by cross-fitting internally -- the
        encoding applied to a row is computed from *other* rows only. It also
        shrinks localities with few listings towards the overall mean, so a
        locality seen twice does not get a confident price level of its own.
    """
    numeric_steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="median")),
    ]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    area_type = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    if location_encoding == "onehot":
        location = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                (
                    "encode",
                    OneHotEncoder(
                        # A locality in the test set that never appeared in
                        # training is mapped to the pooled "infrequent"
                        # category instead of raising. Roughly 6% of test rows
                        # are in this position.
                        handle_unknown="infrequent_if_exist",
                        min_frequency=MIN_LOCATION_FREQUENCY,
                        sparse_output=False,
                    ),
                ),
            ]
        )
    elif location_encoding == "target":
        location = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                (
                    "encode",
                    TargetEncoder(
                        target_type="continuous",
                        # Passing an explicit cross-validator keeps the
                        # internal cross-fitting reproducible. The older
                        # `random_state` argument does the same thing but is
                        # deprecated in scikit-learn 1.9 and removed in 1.11.
                        cv=KFold(n_splits=5, shuffle=True, random_state=0),
                    ),
                ),
            ]
        )
    else:
        raise ValueError(f"unknown location_encoding: {location_encoding!r}")

    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
            ("boolean", "passthrough", BOOLEAN_FEATURES),
            ("area_type", area_type, ["area_type"]),
            ("location", location, ["location"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_model(
    estimator,
    *,
    scale: bool,
    log_target: bool = True,
    location_encoding: str = "onehot",
):
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
            (
                "prepare",
                make_preprocessor(scale=scale, location_encoding=location_encoding),
            ),
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
