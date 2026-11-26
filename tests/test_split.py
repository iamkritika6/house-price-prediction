"""Tests for the train / validation / test split.

These are the most important tests in the project. If the split leaks, every
number the project reports is wrong, and it is wrong in the flattering
direction, which is the direction nobody notices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hpp.split import (
    TARGET,
    check_no_group_overlap,
    make_splits,
    property_fingerprint,
    unseen_location_rate,
)


def _frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "location": rng.choice([f"Area {i}" for i in range(40)], size=n),
            "area_type": rng.choice(["Super built-up  Area", "Plot  Area"], size=n),
            "total_sqft_num": rng.uniform(400, 3000, size=n).round(1),
            "bedrooms": rng.integers(1, 5, size=n).astype(float),
            "bath": rng.integers(1, 4, size=n).astype(float),
            "balcony": rng.integers(0, 3, size=n).astype(float),
            TARGET: rng.uniform(20, 300, size=n).round(2),
        }
    )


class TestSplitSizes:
    def test_three_sets_are_produced(self) -> None:
        splits = make_splits(_frame())
        assert set(splits.sizes) == {"train", "validation", "test"}

    def test_every_row_ends_up_somewhere_exactly_once(self) -> None:
        df = _frame()
        splits = make_splits(df)
        assert sum(splits.sizes.values()) == len(df)

    def test_proportions_are_roughly_sixty_twenty_twenty(self) -> None:
        df = _frame(n=2000)
        splits = make_splits(df)
        total = len(df)
        assert 0.55 <= splits.sizes["train"] / total <= 0.65
        assert 0.15 <= splits.sizes["validation"] / total <= 0.25
        assert 0.15 <= splits.sizes["test"] / total <= 0.25

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"test_fraction": 0.0},
            {"test_fraction": 1.0},
            {"validation_fraction": 0.0},
            {"test_fraction": 0.6, "validation_fraction": 0.5},
        ],
    )
    def test_impossible_fractions_are_rejected(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            make_splits(_frame(), **kwargs)


class TestNoLeakage:
    def test_no_property_appears_in_two_splits(self) -> None:
        splits = make_splits(_frame())
        assert check_no_group_overlap(splits) == {
            "train_validation": 0,
            "train_test": 0,
            "validation_test": 0,
        }

    def test_duplicated_listings_stay_together(self) -> None:
        """The real reason for grouping: the same property listed twice.

        Build a dataset where every property appears three times with a
        different availability date, then confirm no property straddles the
        split.
        """
        base = _frame(n=200)
        tripled = pd.concat([base, base, base], ignore_index=True)
        tripled["availability"] = ["a"] * 200 + ["b"] * 200 + ["c"] * 200

        splits = make_splits(tripled)
        assert check_no_group_overlap(splits) == {
            "train_validation": 0,
            "train_test": 0,
            "validation_test": 0,
        }

    def test_without_grouping_duplicates_would_have_leaked(self) -> None:
        """Shows the leak this guards against is real, not hypothetical.

        A plain row-wise shuffle of the same tripled data puts copies of the
        same property on both sides almost every time.
        """
        rng = np.random.default_rng(0)
        base = _frame(n=200)
        tripled = pd.concat([base, base, base], ignore_index=True)

        order = rng.permutation(len(tripled))
        cut = int(0.8 * len(tripled))
        naive_train = tripled.iloc[order[:cut]]
        naive_test = tripled.iloc[order[cut:]]

        shared = set(property_fingerprint(naive_train)) & set(
            property_fingerprint(naive_test)
        )
        assert len(shared) > 0


class TestReproducibility:
    def test_same_seed_gives_same_split(self) -> None:
        df = _frame()
        a = make_splits(df, seed=7)
        b = make_splits(df, seed=7)
        pd.testing.assert_frame_equal(a.train, b.train)
        pd.testing.assert_frame_equal(a.test, b.test)

    def test_different_seed_gives_a_different_split(self) -> None:
        df = _frame()
        a = make_splits(df, seed=1)
        b = make_splits(df, seed=2)
        assert not a.test.equals(b.test)


class TestUnseenLocations:
    def test_rate_is_between_zero_and_one(self) -> None:
        splits = make_splits(_frame())
        rates = unseen_location_rate(splits)
        assert set(rates) == {"validation", "test"}
        for value in rates.values():
            assert 0.0 <= value <= 1.0

    def test_locations_unique_to_test_are_counted_as_unseen(self) -> None:
        df = _frame(n=400)
        df.loc[df.index[-1], "location"] = "Nowhere Unique"
        splits = make_splits(df, seed=3)
        rates = unseen_location_rate(splits)
        in_train = "Nowhere Unique" in set(splits.train["location"])
        if not in_train:
            assert rates["validation"] + rates["test"] > 0
