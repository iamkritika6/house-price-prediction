"""Tests for the cleaning functions.

The awkward inputs used here are real values taken from the raw file, not
invented ones, so a passing test says something about the actual data.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from hpp.cleaning import (
    clean,
    is_rk,
    parse_bedrooms,
    parse_total_sqft,
    sqft_range_width,
)


class TestParseTotalSqft:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("1056", 1056.0),
            ("1056.0", 1056.0),
            ("  2600  ", 2600.0),
        ],
    )
    def test_plain_numbers(self, raw: str, expected: float) -> None:
        assert parse_total_sqft(raw) == expected

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("2100 - 2850", 2475.0),
            ("1042 - 1105", 1073.5),
            ("3067 - 8156", 5611.5),
        ],
    )
    def test_ranges_become_midpoints(self, raw: str, expected: float) -> None:
        assert parse_total_sqft(raw) == expected

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("34.46Sq. Meter", 34.46 * 10.7639),
            ("4125Perch", 4125 * 272.25),
            ("1Acres", 43560.0),
            ("1Guntha", 1089.0),
            ("1Cents", 435.6),
            ("1Grounds", 2400.0),
            ("1Sq. Yards", 9.0),
        ],
    )
    def test_units_are_converted(self, raw: str, expected: float) -> None:
        result = parse_total_sqft(raw)
        assert result is not None
        assert math.isclose(result, expected, rel_tol=1e-9)

    def test_unit_conversions_are_mutually_consistent(self) -> None:
        """40 Guntha, 100 Cents and 160 Perch should each be one acre."""
        acre = parse_total_sqft("1Acres")
        assert acre is not None
        for text in ("40Guntha", "100Cents", "160Perch"):
            value = parse_total_sqft(text)
            assert value is not None
            assert math.isclose(value, acre, rel_tol=1e-6), text

    @pytest.mark.parametrize("raw", [None, "", "   ", "not a number", float("nan")])
    def test_unparseable_becomes_none(self, raw: object) -> None:
        assert parse_total_sqft(raw) is None

    def test_unknown_unit_is_not_guessed(self) -> None:
        """An unrecognised unit must not fall back to treating it as sq ft."""
        assert parse_total_sqft("100Furlongs") is None


class TestSqftRangeWidth:
    def test_range_width(self) -> None:
        assert sqft_range_width("2100 - 2850") == 750.0

    def test_wide_range_is_visible(self) -> None:
        """The widest real range in the file spans over 5000 sq ft."""
        assert sqft_range_width("3067 - 8156") == 5089.0

    def test_not_a_range(self) -> None:
        assert sqft_range_width("1056") is None
        assert sqft_range_width("34.46Sq. Meter") is None


class TestParseBedrooms:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("2 BHK", 2),
            ("2 Bedroom", 2),
            ("1 RK", 1),
            ("10 BHK", 10),
            ("4 Bedroom", 4),
        ],
    )
    def test_both_conventions_agree(self, raw: str, expected: int) -> None:
        assert parse_bedrooms(raw) == expected

    def test_bhk_and_bedroom_are_the_same_number(self) -> None:
        assert parse_bedrooms("3 BHK") == parse_bedrooms("3 Bedroom")

    @pytest.mark.parametrize("raw", [None, "", "BHK", float("nan")])
    def test_unparseable_becomes_none(self, raw: object) -> None:
        assert parse_bedrooms(raw) is None


class TestIsRk:
    def test_detects_rk(self) -> None:
        assert is_rk("1 RK") is True

    def test_plain_flats_are_not_rk(self) -> None:
        assert is_rk("1 BHK") is False
        assert is_rk("1 Bedroom") is False

    def test_missing_is_not_rk(self) -> None:
        assert is_rk(None) is False


def _raw_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    columns = [
        "area_type",
        "availability",
        "location",
        "size",
        "society",
        "total_sqft",
        "bath",
        "balcony",
        "price",
    ]
    return pd.DataFrame(rows, columns=columns)


class TestClean:
    @staticmethod
    def _row(**overrides: object) -> dict[str, object]:
        row: dict[str, object] = {
            "area_type": "Super built-up  Area",
            "availability": "Ready To Move",
            "location": "Whitefield",
            "size": "2 BHK",
            "society": "Coomee",
            "total_sqft": "1056",
            "bath": 2.0,
            "balcony": 1.0,
            "price": 39.07,
        }
        row.update(overrides)
        return row

    def test_exact_duplicates_are_dropped(self) -> None:
        df = clean(_raw_frame([self._row(), self._row(), self._row(location="Kothanur")]))
        assert len(df) == 2
        assert df.attrs["n_duplicates_dropped"] == 1

    def test_duplicates_dropped_before_anything_else(self) -> None:
        """Deduplication must not depend on parsing succeeding."""
        bad = self._row(total_sqft="nonsense")
        df = clean(_raw_frame([bad, bad]))
        assert df.attrs["n_duplicates_dropped"] == 1

    def test_location_whitespace_is_stripped(self) -> None:
        """' Whitefield' and 'Whitefield' are the same place, not two places."""
        df = clean(_raw_frame([self._row(location=" Whitefield"), self._row(location="Whitefield", price=40.0)]))
        assert set(df["location"]) == {"Whitefield"}

    def test_rows_without_usable_size_are_dropped(self) -> None:
        df = clean(_raw_frame([self._row(), self._row(total_sqft="nonsense", price=50.0)]))
        assert len(df) == 1
        assert df.attrs["n_unusable_dropped"] == 1

    def test_missing_society_becomes_a_flag(self) -> None:
        df = clean(_raw_frame([self._row(society=None), self._row(society="Coomee", price=50.0)]))
        assert list(df["has_society"]) == [False, True]
        assert "society" not in df.columns

    def test_raw_text_columns_are_removed(self) -> None:
        """The unparsed text columns must not survive into modelling."""
        df = clean(_raw_frame([self._row()]))
        for column in ("total_sqft", "size", "price", "society"):
            assert column not in df.columns

    def test_target_is_named_price_lakh(self) -> None:
        df = clean(_raw_frame([self._row(price=39.07)]))
        assert df["price_lakh"].iloc[0] == pytest.approx(39.07)

    def test_bath_and_balcony_become_numeric(self) -> None:
        df = clean(_raw_frame([self._row(bath="2.0", balcony="1.0")]))
        assert df["bath"].dtype.kind == "f"
        assert df["balcony"].dtype.kind == "f"
        assert df["bath"].iloc[0] == pytest.approx(2.0)


class TestPlausibility:
    """The fixed domain rules that remove impossible listings."""

    @staticmethod
    def _row(**overrides: object) -> dict[str, object]:
        return TestClean._row(**overrides)

    def test_one_square_foot_home_is_removed(self) -> None:
        df = clean(_raw_frame([self._row(total_sqft="1")]))
        assert len(df) == 0
        assert df.attrs["n_implausible_dropped"] == 1

    def test_thirty_acre_listing_is_removed(self) -> None:
        df = clean(_raw_frame([self._row(total_sqft="1306800")]))
        assert len(df) == 0

    def test_forty_three_bedrooms_is_removed(self) -> None:
        df = clean(_raw_frame([self._row(size="43 BHK", total_sqft="40000")]))
        assert len(df) == 0

    def test_six_bedrooms_in_six_hundred_feet_is_removed(self) -> None:
        """100 sq ft per bedroom is not a home, it is a data entry error."""
        df = clean(_raw_frame([self._row(size="6 BHK", total_sqft="600")]))
        assert len(df) == 0

    def test_ordinary_listing_survives(self) -> None:
        df = clean(_raw_frame([self._row(size="2 BHK", total_sqft="1056")]))
        assert len(df) == 1
        assert df.attrs["n_implausible_dropped"] == 0

    def test_breakdown_names_each_rule(self) -> None:
        df = clean(_raw_frame([self._row(total_sqft="1")]))
        breakdown = df.attrs["implausible_breakdown"]
        assert breakdown["total_sqft < 100"] == 1
        assert sum(breakdown.values()) >= 1

    def test_rules_do_not_look_at_price(self) -> None:
        """A wildly priced but physically plausible listing must survive.

        Removing rows on the basis of the target is a modelling decision that
        has to be made on training data only, so it must not happen here.
        """
        df = clean(_raw_frame([self._row(price=3600.0)]))
        assert len(df) == 1
