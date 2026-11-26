"""Baselines that every model has to beat.

A score means nothing on its own. "R-squared 0.80" is only good if something
simpler does worse. Two baselines are used here, deliberately chosen to be
harder to beat as you go down the list.

1. **Median price.** Ignore every feature and always predict the median price
   of the training set. This is the floor. A model that cannot beat it has
   learned nothing at all.

2. **Median price per square foot, by locality.** Work out the typical price
   per square foot in each locality from the training data, then multiply by
   the property's size. This is roughly what an estate agent does in their
   head, and it is genuinely hard to beat. Beating *this* is the claim worth
   making, because it is the claim that the machine learning is earning its
   keep rather than rediscovering arithmetic.

Both are fitted on training data only, exactly like any other model, because
both learn a quantity (a median) from the sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class MedianBaseline(BaseEstimator, RegressorMixin):
    """Always predict the median training price, whatever the features say."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MedianBaseline":
        self.median_ = float(np.median(np.asarray(y, dtype=float)))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "median_")
        return np.full(len(X), self.median_, dtype=float)


class PricePerSqftBaseline(BaseEstimator, RegressorMixin):
    """Predict ``median price per sqft in this locality`` x ``size``.

    Localities with fewer than ``min_listings`` training examples fall back to
    the city-wide median price per square foot, because a median taken from
    one or two listings is noise. Unseen localities use the same fallback.
    """

    def __init__(self, location_column: str = "location", sqft_column: str = "total_sqft_num", min_listings: int = 5):
        self.location_column = location_column
        self.sqft_column = sqft_column
        self.min_listings = min_listings

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "PricePerSqftBaseline":
        y = np.asarray(y, dtype=float)
        sqft = X[self.sqft_column].to_numpy(dtype=float)

        if np.any(sqft <= 0):
            raise ValueError("every listing must have a positive size")

        price_per_sqft = y / sqft
        self.global_rate_ = float(np.median(price_per_sqft))

        frame = pd.DataFrame(
            {"location": X[self.location_column].to_numpy(), "rate": price_per_sqft}
        )
        grouped = frame.groupby("location")["rate"]
        rates = grouped.median()
        counts = grouped.size()

        self.location_rate_ = rates[counts >= self.min_listings].to_dict()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "global_rate_")
        rates = (
            X[self.location_column]
            .map(self.location_rate_)
            .fillna(self.global_rate_)
            .to_numpy(dtype=float)
        )
        return rates * X[self.sqft_column].to_numpy(dtype=float)
