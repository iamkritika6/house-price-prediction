"""Tests for the error metrics and the baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hpp.baselines import MedianBaseline, PricePerSqftBaseline
from hpp.metrics import RUPEES_PER_LAKH, score, score_by_band


class TestScore:
    def test_perfect_predictions_have_no_error(self) -> None:
        y = np.array([50.0, 75.0, 120.0])
        result = score(y, y)
        assert result.rmse_lakh == pytest.approx(0.0)
        assert result.mae_lakh == pytest.approx(0.0)
        assert result.mape_percent == pytest.approx(0.0)
        assert result.r2 == pytest.approx(1.0)

    def test_mae_is_the_average_miss(self) -> None:
        y_true = np.array([100.0, 100.0])
        y_pred = np.array([90.0, 120.0])
        assert score(y_true, y_pred).mae_lakh == pytest.approx(15.0)

    def test_rmse_punishes_the_big_miss_harder(self) -> None:
        """Same total error, but concentrated in one prediction."""
        y_true = np.array([100.0, 100.0])
        spread = score(y_true, np.array([90.0, 110.0]))
        concentrated = score(y_true, np.array([100.0, 120.0]))
        assert spread.mae_lakh == concentrated.mae_lakh
        assert concentrated.rmse_lakh > spread.rmse_lakh

    def test_rmse_is_never_below_mae(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(20):
            y_true = rng.uniform(10, 500, size=50)
            y_pred = y_true + rng.normal(0, 30, size=50)
            result = score(y_true, y_pred)
            assert result.rmse_lakh >= result.mae_lakh - 1e-9

    def test_mape_is_relative_not_absolute(self) -> None:
        """Being 10 lakh out matters more on a cheap flat than a mansion."""
        cheap = score(np.array([40.0]), np.array([50.0]))
        expensive = score(np.array([400.0]), np.array([410.0]))
        assert cheap.mae_lakh == expensive.mae_lakh
        assert cheap.mape_percent > expensive.mape_percent

    def test_rupee_conversion(self) -> None:
        result = score(np.array([100.0]), np.array([90.0]))
        assert result.mae_lakh == pytest.approx(10.0)
        assert result.mae_rupees == pytest.approx(10.0 * RUPEES_PER_LAKH)

    def test_median_ape_resists_a_single_disaster(self) -> None:
        y_true = np.array([100.0, 100.0, 100.0, 100.0])
        y_pred = np.array([105.0, 95.0, 102.0, 1000.0])
        result = score(y_true, y_pred)
        assert result.median_ape_percent < result.mape_percent

    def test_shape_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            score(np.array([1.0, 2.0]), np.array([1.0]))

    def test_empty_input_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            score(np.array([]), np.array([]))


class TestScoreByBand:
    def test_bands_partition_the_rows(self) -> None:
        y_true = np.array([30.0, 60.0, 90.0, 200.0, 900.0])
        y_pred = y_true * 1.1
        bands = score_by_band(y_true, y_pred)
        assert sum(s.n for s in bands.values()) == len(y_true)

    def test_empty_bands_are_omitted(self) -> None:
        y = np.array([60.0, 61.0])
        bands = score_by_band(y, y)
        assert len(bands) == 1


class TestMedianBaseline:
    def test_predicts_the_training_median_regardless_of_features(self) -> None:
        X = pd.DataFrame({"anything": [1, 2, 3, 4, 5]})
        y = np.array([10.0, 20.0, 30.0, 40.0, 1000.0])
        model = MedianBaseline().fit(X, y)
        predictions = model.predict(pd.DataFrame({"anything": [99, -99]}))
        assert np.all(predictions == 30.0)

    def test_median_not_mean(self) -> None:
        """The median must not be dragged up by one expensive property."""
        X = pd.DataFrame({"a": range(5)})
        y = np.array([10.0, 10.0, 10.0, 10.0, 5000.0])
        model = MedianBaseline().fit(X, y)
        assert model.predict(X)[0] == 10.0


class TestPricePerSqftBaseline:
    @staticmethod
    def _frame(locations: list[str], sqft: list[float]) -> pd.DataFrame:
        return pd.DataFrame({"location": locations, "total_sqft_num": sqft})

    def test_uses_locality_rate(self) -> None:
        X = self._frame(["A"] * 6 + ["B"] * 6, [1000.0] * 12)
        # Locality A is 0.05 lakh/sqft, locality B is 0.10 lakh/sqft.
        y = np.array([50.0] * 6 + [100.0] * 6)
        model = PricePerSqftBaseline(min_listings=5).fit(X, y)

        predicted = model.predict(self._frame(["A", "B"], [2000.0, 2000.0]))
        assert predicted[0] == pytest.approx(100.0)
        assert predicted[1] == pytest.approx(200.0)

    def test_rare_locality_falls_back_to_city_rate(self) -> None:
        X = self._frame(["A"] * 10 + ["Rare"], [1000.0] * 11)
        y = np.array([50.0] * 10 + [500.0])
        model = PricePerSqftBaseline(min_listings=5).fit(X, y)

        # 'Rare' has one listing, below min_listings, so it must not get its
        # own rate -- one listing is noise, not a local price level.
        assert "Rare" not in model.location_rate_
        predicted = model.predict(self._frame(["Rare"], [1000.0]))
        assert predicted[0] == pytest.approx(model.global_rate_ * 1000.0)

    def test_unseen_locality_falls_back_rather_than_failing(self) -> None:
        X = self._frame(["A"] * 10, [1000.0] * 10)
        y = np.array([50.0] * 10)
        model = PricePerSqftBaseline(min_listings=5).fit(X, y)
        predicted = model.predict(self._frame(["Never Seen"], [1000.0]))
        assert np.isfinite(predicted[0])

    def test_scales_with_size(self) -> None:
        X = self._frame(["A"] * 10, [1000.0] * 10)
        y = np.array([50.0] * 10)
        model = PricePerSqftBaseline(min_listings=5).fit(X, y)
        small, large = model.predict(self._frame(["A", "A"], [500.0, 2000.0]))
        assert large == pytest.approx(4 * small)

    def test_zero_size_is_rejected(self) -> None:
        X = self._frame(["A"], [0.0])
        with pytest.raises(ValueError):
            PricePerSqftBaseline().fit(X, np.array([50.0]))
