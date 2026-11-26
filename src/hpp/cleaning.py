"""Turn the raw listings file into a tidy table.

Everything in this module is **deterministic and row-wise**: each function looks
at one value, or at one row, and never at the dataset as a whole. That is a
deliberate rule, not an accident.

Why the rule matters
--------------------
Anything *learned from the data* -- a median used to fill missing values, the
categories used by an encoder, a scaling factor -- must be learned from the
training set only. If it is learned from the whole dataset before splitting,
information from the test set leaks into the model and the test score becomes
optimistic.

Parsing ``"2100 - 2850"`` into ``2475.0`` learns nothing from other rows, so it
is safe to do here, before the split. Filling a missing bathroom count with the
median of the column is *not* safe here, so it lives in the modelling pipeline
instead (see :mod:`hpp.pipeline`).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Conversions into square feet.
#
# All of these are consistent with 1 acre = 43,560 sq ft:
#   1 Guntha  = 1/40 acre   = 1,089 sq ft
#   1 Cent    = 1/100 acre  =   435.6 sq ft
#   1 Perch   = 1/160 acre  =   272.25 sq ft
# A "Ground" is a South Indian unit of 2,400 sq ft.
# A square yard (gaj) is 3ft x 3ft = 9 sq ft.
SQFT_PER_UNIT: dict[str, float] = {
    "sq. meter": 10.7639,
    "sq. yards": 9.0,
    "perch": 272.25,
    "acres": 43560.0,
    "cents": 435.6,
    "guntha": 1089.0,
    "grounds": 2400.0,
}

# Plausibility limits for a residential listing.
#
# These are FIXED domain rules, not thresholds derived from this dataset, and
# none of them looks at the price. That matters: a rule computed from the data
# (say, "drop anything beyond 1.5 x the interquartile range of price per square
# foot") would have to be fitted on the training set only, because it encodes
# something learned from the sample. A rule that says a home cannot be one
# square foot is a statement about homes, so it can be applied up front and
# explained in one sentence.
MIN_SQFT = 100.0
MAX_SQFT = 50_000.0
MAX_BEDROOMS = 20
MIN_SQFT_PER_BEDROOM = 150.0

_RANGE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*$")
_WITH_UNIT = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([A-Za-z. ]+?)\s*$")
_PLAIN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*$")
_LEADING_INT = re.compile(r"^\s*(\d+)")


def parse_total_sqft(value: object) -> float | None:
    """Parse one ``total_sqft`` value into a number of square feet.

    Handles the three shapes that actually occur in the raw file:

    >>> parse_total_sqft("1056")
    1056.0
    >>> parse_total_sqft("2100 - 2850")      # midpoint
    2475.0
    >>> parse_total_sqft("34.46Sq. Meter")   # converted
    370.92...

    Returns ``None`` for anything it does not recognise, so that unparseable
    values become missing rather than silently wrong.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    text = str(value).strip()
    if not text:
        return None

    plain = _PLAIN.match(text)
    if plain:
        return float(plain.group(1))

    span = _RANGE.match(text)
    if span:
        low, high = float(span.group(1)), float(span.group(2))
        # Midpoint. The width is kept separately by `sqft_range_width` so a
        # very wide range can be inspected rather than quietly averaged.
        return (low + high) / 2.0

    with_unit = _WITH_UNIT.match(text)
    if with_unit:
        amount = float(with_unit.group(1))
        unit = with_unit.group(2).strip().lower()
        factor = SQFT_PER_UNIT.get(unit)
        if factor is not None:
            return amount * factor

    return None


def sqft_range_width(value: object) -> float | None:
    """Width of a ``total_sqft`` range, or ``None`` if it is not a range.

    ``"2100 - 2850"`` gives ``750.0``. Used to audit how much precision the
    midpoint assumption throws away: a range of ``3067 - 8156`` is not really
    one property size at all.
    """
    if value is None:
        return None
    span = _RANGE.match(str(value))
    if not span:
        return None
    return float(span.group(2)) - float(span.group(1))


def parse_bedrooms(value: object) -> int | None:
    """Parse ``size`` into a bedroom count.

    The raw column uses two conventions for the same thing -- ``"2 BHK"`` and
    ``"2 Bedroom"`` both mean two bedrooms -- plus ``"1 RK"`` for a
    room-and-kitchen studio. Taking the leading integer handles all three.

    >>> parse_bedrooms("2 BHK"), parse_bedrooms("2 Bedroom"), parse_bedrooms("1 RK")
    (2, 2, 1)
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    match = _LEADING_INT.match(str(value).strip())
    return int(match.group(1)) if match else None


def is_rk(value: object) -> bool:
    """True for a ``"1 RK"`` style studio listing.

    An RK is a single room with a kitchen, which is a smaller thing than a
    one-bedroom flat. :func:`parse_bedrooms` maps both to 1, so this flag keeps
    the difference available to the model instead of discarding it.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return bool(re.search(r"\bRK\b", str(value), flags=re.IGNORECASE))


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw listings table.

    Steps, in order:

    1. Drop exactly duplicated rows. This happens **first**, and before any
       train/test split, because a duplicate that survives into both sides of
       the split lets the model be scored on a row it has memorised.
    2. Parse ``total_sqft`` into numeric square feet.
    3. Parse ``size`` into a bedroom count and an RK flag.
    4. Strip whitespace from ``location`` (the raw file has values such as
       ``" Whitefield"`` with a leading space, which would otherwise be a
       different category from ``"Whitefield"``).
    5. Record whether ``society`` was missing, then drop the column itself.
       It is 41% missing with 2,688 distinct values, so it is far more useful
       as "was this listing attached to a named development?" than as a
       category in its own right.
    6. Drop rows with no usable size or price.
    """
    df = raw.copy()

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_duplicates = before - len(df)

    df["total_sqft_num"] = df["total_sqft"].map(parse_total_sqft)
    df["sqft_range_width"] = df["total_sqft"].map(sqft_range_width)
    df["sqft_was_range"] = df["sqft_range_width"].notna()

    df["bedrooms"] = df["size"].map(parse_bedrooms)
    df["is_rk"] = df["size"].map(is_rk)

    df["location"] = df["location"].astype("string").str.strip()
    df["area_type"] = df["area_type"].astype("string").str.strip()

    df["has_society"] = df["society"].notna()

    # The raw file is read as text so nothing is silently coerced on load, so
    # the genuinely numeric columns are converted explicitly here.
    df["bath"] = pd.to_numeric(df["bath"], errors="coerce")
    df["balcony"] = pd.to_numeric(df["balcony"], errors="coerce")
    df["price_lakh"] = pd.to_numeric(df["price"], errors="coerce")

    # A listing with no parseable size, or no price, cannot be used either to
    # train or to score. Dropping is honest; imputing a size would be inventing
    # the main predictor.
    usable = df["total_sqft_num"].notna() & df["price_lakh"].notna() & df["bedrooms"].notna()
    n_unusable = int((~usable).sum())
    df = df.loc[usable].reset_index(drop=True)

    df = df.drop(columns=["society", "total_sqft", "size", "price"])

    # Implausible rows. Counted individually so the README can say exactly how
    # many rows each rule removed, rather than reporting one opaque total.
    too_small = df["total_sqft_num"] < MIN_SQFT
    too_large = df["total_sqft_num"] > MAX_SQFT
    too_many_beds = df["bedrooms"] > MAX_BEDROOMS
    cramped = (df["total_sqft_num"] / df["bedrooms"]) < MIN_SQFT_PER_BEDROOM

    implausible = too_small | too_large | too_many_beds | cramped
    n_implausible = int(implausible.sum())
    df = df.loc[~implausible].reset_index(drop=True)

    df.attrs["n_duplicates_dropped"] = n_duplicates
    df.attrs["n_unusable_dropped"] = n_unusable
    df.attrs["n_implausible_dropped"] = n_implausible
    df.attrs["implausible_breakdown"] = {
        f"total_sqft < {MIN_SQFT:.0f}": int(too_small.sum()),
        f"total_sqft > {MAX_SQFT:.0f}": int(too_large.sum()),
        f"bedrooms > {MAX_BEDROOMS}": int(too_many_beds.sum()),
        f"sqft per bedroom < {MIN_SQFT_PER_BEDROOM:.0f}": int(cramped.sum()),
    }
    return df
