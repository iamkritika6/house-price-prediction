"""Derived features.

Every function here is row-wise and deterministic: it computes a value from
other values in the same row and learns nothing from the dataset. That is why
these can be computed before the split without leaking.

The learned parts of preprocessing -- filling missing values, deciding which
localities are too rare to keep as their own category, scaling -- live in
:mod:`hpp.pipeline`, where scikit-learn fits them on the training fold only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "total_sqft_num",
    "log_sqft",
    "bedrooms",
    "bath",
    "balcony",
    "sqft_per_bedroom",
    "bath_per_bedroom",
]

BOOLEAN_FEATURES = [
    "sqft_was_range",
    "is_rk",
    "has_society",
    "is_ready_to_move",
]

CATEGORICAL_FEATURES = [
    "area_type",
    "location",
]

ALL_FEATURES = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived columns the model uses.

    - ``log_sqft``: prices and sizes are both strongly right-skewed, and the
      relationship between them is closer to straight on a log scale. This
      mainly helps the linear model; the tree models do not need it but are
      not harmed by it.
    - ``sqft_per_bedroom``: separates a spacious three-bedroom flat from a
      cramped one. Size and bedroom count are highly correlated, so their
      ratio carries information neither does alone.
    - ``bath_per_bedroom``: a rough proxy for how well-appointed a property is.
    - ``is_ready_to_move``: the raw ``availability`` column has 81 values, most
      of them possession dates. The distinction that actually affects price is
      whether the buyer can move in now or is buying something unbuilt.
    """
    out = df.copy()

    sqft = out["total_sqft_num"].astype(float)
    bedrooms = out["bedrooms"].astype(float)

    out["log_sqft"] = np.log1p(sqft)
    out["sqft_per_bedroom"] = sqft / bedrooms
    out["bath_per_bedroom"] = out["bath"].astype(float) / bedrooms

    availability = out["availability"].astype("string").str.strip().str.lower()
    out["is_ready_to_move"] = availability.isin(
        {"ready to move", "immediate possession"}
    ).fillna(False)

    for column in BOOLEAN_FEATURES:
        if column in out.columns:
            out[column] = out[column].astype(bool)

    return out
