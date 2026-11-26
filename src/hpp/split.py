"""Split the data into training, validation and test sets.

This module exists on its own, and runs before anything is fitted, because the
split is the single thing most easily got wrong in a price prediction project.

The three sets and what each is for
-----------------------------------
``train``
    The model learns from this, and every fitted quantity -- imputation
    medians, encoder categories, scaling factors -- is learned from here only.

``validation``
    Used to compare models and choose settings. Looked at many times, so by the
    end its score is no longer an unbiased estimate: whatever scores best on it
    is partly winning by luck.

``test``
    Scored **once**, at the very end, after all choices are final. This is the
    number that gets reported.

A two-way split, which is what most tutorials use, forces the same data to do
the second and third jobs at once. The reported score is then optimistic,
because the model was selected on the data it is being judged on.

Near-duplicate listings
-----------------------
Exactly duplicated rows are removed during cleaning. But the same property can
still appear twice with a different ``availability`` date -- same locality,
same size, same bedroom count, same price. Those are not independent rows. If
one copy lands in training and another in test, the model is scored on a
property it has already seen.

So the split is made over *groups* rather than rows: listings sharing a
fingerprint of their physical characteristics and price stay together on the
same side of the split.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

TARGET = "price_lakh"

# Columns that identify a property. Two listings agreeing on all of these are
# treated as the same property advertised twice.
FINGERPRINT_COLUMNS = [
    "location",
    "area_type",
    "total_sqft_num",
    "bedrooms",
    "bath",
    "balcony",
    TARGET,
]


@dataclass(frozen=True)
class Splits:
    """The three sets, plus the facts needed to describe how they were made."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    n_groups: int
    seed: int

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }


def property_fingerprint(df: pd.DataFrame) -> pd.Series:
    """A group label identifying the same property advertised more than once."""
    return (
        df[FINGERPRINT_COLUMNS]
        .astype("string")
        .fillna("~missing~")
        .agg("|".join, axis=1)
    )


def make_splits(
    df: pd.DataFrame,
    *,
    seed: int = 42,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.2,
) -> Splits:
    """Split into train / validation / test, grouping near-duplicate listings.

    The fractions are of the whole dataset, so the defaults give roughly
    60 / 20 / 20.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if test_fraction + validation_fraction >= 1:
        raise ValueError("test and validation fractions must leave room for training data")

    groups = property_fingerprint(df)

    # First carve off the test set and do not touch it again.
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    rest_idx, test_idx = next(splitter.split(df, groups=groups))

    rest = df.iloc[rest_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)

    # Then split the remainder into train and validation. The validation
    # fraction is expressed relative to the whole dataset, so it is rescaled.
    relative_validation = validation_fraction / (1.0 - test_fraction)
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=relative_validation, random_state=seed
    )
    rest_groups = property_fingerprint(rest)
    train_idx, validation_idx = next(splitter.split(rest, groups=rest_groups))

    return Splits(
        train=rest.iloc[train_idx].reset_index(drop=True),
        validation=rest.iloc[validation_idx].reset_index(drop=True),
        test=test,
        n_groups=int(groups.nunique()),
        seed=seed,
    )


def check_no_group_overlap(splits: Splits) -> dict[str, int]:
    """Verify that no property appears in more than one split.

    Returns the number of shared fingerprints between each pair. All three
    numbers must be zero; this is asserted in the tests and re-checked by the
    split script, because a silent leak here would invalidate every number the
    project reports.
    """
    train = set(property_fingerprint(splits.train))
    validation = set(property_fingerprint(splits.validation))
    test = set(property_fingerprint(splits.test))
    return {
        "train_validation": len(train & validation),
        "train_test": len(train & test),
        "validation_test": len(validation & test),
    }


def unseen_location_rate(splits: Splits) -> dict[str, float]:
    """Fraction of rows whose location never appears in the training set.

    With 1,265 distinct locations over ~12,600 rows, a long tail of localities
    appears only once or twice. Those rows are genuinely harder: the model has
    no local price level to work from. Reporting this rate makes that visible
    instead of hiding it inside an average.
    """
    known = set(splits.train["location"].dropna())
    out = {}
    for name, frame in (("validation", splits.validation), ("test", splits.test)):
        unseen = ~frame["location"].isin(known)
        out[name] = float(unseen.mean())
    return out
