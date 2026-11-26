"""Clean the raw listings and write the tidy table to data/interim/.

Run:
    docker compose run --rm hpp python scripts/clean_data.py

Also cross-checks our own parsing of ``total_sqft`` against the numeric column
in the OpenML copy of the same dataset. The two were produced independently, so
agreement is evidence that the parsing is right.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

from hpp.cleaning import parse_total_sqft

RAW_CSV = Path("data/raw/Bengaluru_House_Data.csv")
OPENML_ARFF = Path("data/raw/bengaluru_real_estate_price.arff")
OUT_CSV = Path("data/interim/clean.csv")

# Column order inside the OpenML ARFF @data section. It has no `availability`
# column, which OpenML dropped.
ARFF_SQFT_INDEX = 4


def read_arff_total_sqft(path: Path) -> list[float | None]:
    """Pull just the total_sqft column out of the OpenML ARFF file."""
    values: list[float | None] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        in_data = False
        for line in handle:
            if not in_data:
                if line.strip().lower().startswith("@data"):
                    in_data = True
                continue
            if not line.strip():
                continue
            fields = next(csv.reader([line], quotechar="'"))
            raw = fields[ARFF_SQFT_INDEX].strip()
            values.append(None if raw.lower() in {"nan", "?", ""} else float(raw))
    return values


def cross_check(raw: pd.DataFrame) -> None:
    """Compare our parsed total_sqft against OpenML's."""
    if not OPENML_ARFF.exists():
        print("OpenML copy not present; skipping cross-check.")
        return

    theirs = read_arff_total_sqft(OPENML_ARFF)
    if len(theirs) != len(raw):
        print(
            f"Row counts differ (ours {len(raw)}, OpenML {len(theirs)}); "
            "cannot align by position, skipping cross-check."
        )
        return

    ours = raw["total_sqft"].map(parse_total_sqft)
    comparison = pd.DataFrame({"ours": ours.to_numpy(), "theirs": theirs})

    both = comparison.dropna()
    agree = (both["ours"] - both["theirs"]).abs() < 0.01

    print("\n--- cross-check of total_sqft against the OpenML copy ---")
    print(f"rows compared (both non-missing): {len(both)}")
    print(f"agree to within 0.01 sq ft:       {int(agree.sum())}")
    print(f"disagree:                         {int((~agree).sum())}")
    print(f"we parsed, OpenML missing:        {int((comparison['ours'].notna() & comparison['theirs'].isna()).sum())}")
    print(f"OpenML parsed, we did not:        {int((comparison['ours'].isna() & comparison['theirs'].notna()).sum())}")

    disagreements = both.loc[~agree]
    if len(disagreements):
        print("\nexamples of disagreement (raw value, ours, OpenML):")
        for idx in disagreements.index[:10]:
            print(f"  {raw['total_sqft'].iloc[idx]!r:24s} {both['ours'][idx]:12.2f} {both['theirs'][idx]:12.2f}")


def main() -> int:
    from hpp.cleaning import clean

    raw = pd.read_csv(RAW_CSV, dtype=str, keep_default_na=False, na_values=[""])
    print(f"raw:     {raw.shape[0]} rows x {raw.shape[1]} columns")

    cross_check(raw)

    df = clean(raw)
    print("\n--- cleaning ---")
    print(f"duplicate rows dropped:  {df.attrs['n_duplicates_dropped']}")
    print(f"unusable rows dropped:   {df.attrs['n_unusable_dropped']}")
    print(f"implausible rows dropped:{df.attrs['n_implausible_dropped']}")
    for rule, count in df.attrs["implausible_breakdown"].items():
        print(f"    {rule:34s} {count}")
    print(f"remaining:               {len(df)} rows x {df.shape[1]} columns")

    print("\n--- columns after cleaning ---")
    for column in df.columns:
        missing = int(df[column].isna().sum())
        print(f"{column:18s} {str(df[column].dtype):10s} missing={missing}")

    print("\n--- sanity checks on the cleaned data ---")
    print(f"total_sqft_num  min={df['total_sqft_num'].min():.1f}  median={df['total_sqft_num'].median():.1f}  max={df['total_sqft_num'].max():.1f}")
    print(f"bedrooms        min={df['bedrooms'].min()}  median={df['bedrooms'].median()}  max={df['bedrooms'].max()}")
    print(f"price_lakh      min={df['price_lakh'].min():.1f}  median={df['price_lakh'].median():.1f}  max={df['price_lakh'].max():.1f}")
    print(f"listings parsed from a range: {int(df['sqft_was_range'].sum())}")
    print(f"studio (RK) listings:         {int(df['is_rk'].sum())}")
    print(f"distinct locations:           {df['location'].nunique()}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
