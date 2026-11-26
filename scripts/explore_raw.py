"""Print a factual profile of the raw dataset.

Run:
    docker compose run --rm hpp python scripts/explore_raw.py

Every number quoted in the README about the raw data comes from this script,
so the figures can always be regenerated and checked.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

RAW_CSV = Path("data/raw/Bengaluru_House_Data.csv")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    print(f"file:    {RAW_CSV}")
    print(f"sha256:  {sha256(RAW_CSV)}")

    # Read everything as text: the point is to see the data as published,
    # before any type coercion silently changes or drops values.
    df = pd.read_csv(RAW_CSV, dtype=str, keep_default_na=False, na_values=[""])
    print(f"shape:   {df.shape[0]} rows x {df.shape[1]} columns")

    print("\n--- missing values per column ---")
    missing = df.isna().sum()
    for col in df.columns:
        pct = 100 * missing[col] / len(df)
        print(f"{col:14s} {missing[col]:6d}  ({pct:5.2f}%)")

    print("\n--- distinct values per column ---")
    for col in df.columns:
        print(f"{col:14s} {df[col].nunique(dropna=True):6d}")

    print("\n--- exact duplicate rows ---")
    print(f"fully duplicated rows: {df.duplicated().sum()}")

    print("\n--- total_sqft: how messy is it? ---")
    sqft = df["total_sqft"].dropna()
    plain = sqft.str.fullmatch(r"\d+(\.\d+)?")
    print(f"plain numbers:        {int(plain.sum())}")
    print(f"NOT plain numbers:    {int((~plain).sum())}")
    print("\nexamples of non-plain values:")
    for value in sqft[~plain].drop_duplicates().head(12):
        print(f"  {value}")

    print("\n--- size: free text, not a number ---")
    for value, count in df["size"].value_counts().head(8).items():
        print(f"  {value:16s} {count}")

    print("\n--- price (as published) ---")
    price = pd.to_numeric(df["price"], errors="coerce")
    print(f"non-numeric prices:   {int(price.isna().sum() - df['price'].isna().sum())}")
    print(price.describe().to_string())

    print("\n--- area_type ---")
    for value, count in df["area_type"].value_counts().items():
        print(f"  {value:24s} {count}")


if __name__ == "__main__":
    main()
