"""Tests for derived features and the preprocessing pipeline.

The leakage tests here are the point of the file. They check not that the
pipeline produces numbers, but that it produces them *using training data
only*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from hpp.features import add_features
from hpp.pipeline import make_model, make_preprocessor


def _frame(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "location": rng.choice(["Whitefield", "Kothanur", "Hebbal"], size=n),
            "area_type": rng.choice(["Super built-up  Area", "Plot  Area"], size=n),
            "availability": rng.choice(["Ready To Move", "18-Dec"], size=n),
            "total_sqft_num": rng.uniform(500, 2500, size=n),
            "bedrooms": rng.integers(1, 4, size=n).astype(float),
            "bath": rng.integers(1, 3, size=n).astype(float),
            "balcony": rng.integers(0, 3, size=n).astype(float),
            "sqft_was_range": rng.random(n) < 0.1,
            "is_rk": np.zeros(n, dtype=bool),
            "has_society": rng.random(n) < 0.5,
            "price_lakh": rng.uniform(30, 300, size=n),
        }
    )


class TestAddFeatures:
    def test_log_sqft_is_monotonic_in_size(self) -> None:
        df = add_features(_frame())
        order_sqft = df["total_sqft_num"].rank()
        order_log = df["log_sqft"].rank()
        assert (order_sqft == order_log).all()

    def test_sqft_per_bedroom(self) -> None:
        df = _frame(n=1)
        df.loc[0, "total_sqft_num"] = 1200.0
        df.loc[0, "bedrooms"] = 3.0
        assert add_features(df)["sqft_per_bedroom"].iloc[0] == pytest.approx(400.0)

    def test_bath_per_bedroom(self) -> None:
        df = _frame(n=1)
        df.loc[0, "bath"] = 2.0
        df.loc[0, "bedrooms"] = 4.0
        assert add_features(df)["bath_per_bedroom"].iloc[0] == pytest.approx(0.5)

    @pytest.mark.parametrize(
        "availability, expected",
        [
            ("Ready To Move", True),
            ("ready to move", True),
            ("Immediate Possession", True),
            ("18-Dec", False),
            ("19-Jun", False),
        ],
    )
    def test_ready_to_move_flag(self, availability: str, expected: bool) -> None:
        df = _frame(n=1)
        df.loc[0, "availability"] = availability
        assert bool(add_features(df)["is_ready_to_move"].iloc[0]) is expected

    def test_original_frame_is_not_modified(self) -> None:
        df = _frame()
        before = df.copy()
        add_features(df)
        pd.testing.assert_frame_equal(df, before)


class TestPreprocessor:
    def test_unknown_encoding_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            make_preprocessor(scale=False, location_encoding="magic")

    @pytest.mark.parametrize("encoding", ["onehot", "target"])
    def test_both_encodings_produce_finite_numbers(self, encoding: str) -> None:
        df = add_features(_frame(n=200))
        y = df["price_lakh"].to_numpy()
        prepared = make_preprocessor(scale=False, location_encoding=encoding).fit_transform(df, y)
        assert np.isfinite(np.asarray(prepared, dtype=float)).all()

    @pytest.mark.parametrize("encoding", ["onehot", "target"])
    def test_a_locality_never_seen_in_training_does_not_crash(self, encoding: str) -> None:
        """About 6% of real test rows are in a locality absent from training."""
        train = add_features(_frame(n=200, seed=1))
        preprocessor = make_preprocessor(scale=False, location_encoding=encoding)
        preprocessor.fit(train, train["price_lakh"].to_numpy())

        unseen = add_features(_frame(n=5, seed=2))
        unseen["location"] = "Somewhere That Does Not Exist"
        prepared = preprocessor.transform(unseen)
        assert np.isfinite(np.asarray(prepared, dtype=float)).all()


class TestNoLeakageInPreprocessing:
    def test_imputation_uses_the_training_median_not_the_new_data(self) -> None:
        """The core leakage test.

        Fit on data whose bathroom median is 2. Then transform new data whose
        own median is 50, with one value missing. The gap must be filled with
        2 -- the training median. If it is filled with 50, the transformer has
        looked at the data it is supposed to be blind to.
        """
        train = add_features(_frame(n=50))
        train["bath"] = 2.0
        preprocessor = make_preprocessor(scale=False, location_encoding="onehot")
        preprocessor.fit(train, train["price_lakh"].to_numpy())

        new = add_features(_frame(n=5, seed=9))
        new["bath"] = 50.0
        new.loc[0, "bath"] = np.nan
        new["bath_per_bedroom"] = new["bath"] / new["bedrooms"]

        prepared = pd.DataFrame(
            preprocessor.transform(new),
            columns=preprocessor.get_feature_names_out(),
        )
        assert prepared["bath"].iloc[0] == pytest.approx(2.0)

    def test_transform_result_does_not_depend_on_other_rows(self) -> None:
        """A fitted transformer must treat each row independently.

        Transforming one row alone must give the same answer as transforming
        it as part of a batch. If it does not, the rows are influencing each
        other and a test prediction depends on the rest of the test set.
        """
        train = add_features(_frame(n=120))
        preprocessor = make_preprocessor(scale=True, location_encoding="target")
        preprocessor.fit(train, train["price_lakh"].to_numpy())

        batch = add_features(_frame(n=10, seed=5))
        together = np.asarray(preprocessor.transform(batch), dtype=float)
        alone = np.asarray(preprocessor.transform(batch.iloc[[3]]), dtype=float)

        np.testing.assert_allclose(together[3], alone[0], rtol=1e-9, atol=1e-9)


class TestLogTarget:
    def test_predictions_come_back_in_lakhs_not_logs(self) -> None:
        """The log transform must be undone before predictions are returned."""
        df = add_features(_frame(n=200))
        y = df["price_lakh"].to_numpy()

        model = make_model(Ridge(alpha=1.0), scale=True)
        model.fit(df, y)
        predictions = model.predict(df)

        # Prices are tens to hundreds of lakhs; logs of them are single digits.
        assert predictions.min() > 5.0
        assert np.median(predictions) > 20.0

    def test_log_target_predictions_are_always_positive(self) -> None:
        """A price cannot be negative; exponentiating guarantees it is not.

        A linear model on the raw price can and does predict negative prices
        for small or unusual properties. This is one concrete reason for the
        log target beyond the skew.
        """
        df = add_features(_frame(n=200))
        y = df["price_lakh"].to_numpy()

        model = make_model(Ridge(alpha=1.0), scale=True, log_target=True)
        model.fit(df, y)

        odd = add_features(_frame(n=20, seed=77))
        odd["total_sqft_num"] = 50.0
        odd["log_sqft"] = np.log1p(odd["total_sqft_num"])
        assert (model.predict(odd) > 0).all()
