"""Make the train / validation / test split and save it.

Run:
    docker compose run --rm hpp python scripts/make_splits.py

The split is written to disk so that every later script uses exactly the same
rows. Re-deriving the split inside each script would risk them silently
disagreeing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from hpp.split import (
    TARGET,
    check_no_group_overlap,
    make_splits,
    unseen_location_rate,
)

CLEAN_CSV = Path("data/interim/clean.csv")
OUT_DIR = Path("data/interim")
METRICS_DIR = Path("reports/metrics")
SEED = 42


def main() -> int:
    df = pd.read_csv(CLEAN_CSV)
    print(f"cleaned data: {len(df)} rows")

    splits = make_splits(df, seed=SEED)

    print(f"\ndistinct properties (fingerprint groups): {splits.n_groups}")
    print(f"rows: {len(df)}  ->  {len(df) - splits.n_groups} rows share a property with another row")

    print("\n--- split sizes ---")
    for name, size in splits.sizes.items():
        print(f"{name:12s} {size:6d}  ({100 * size / len(df):5.1f}%)")

    print("\n--- leakage check: shared properties between splits ---")
    overlap = check_no_group_overlap(splits)
    for pair, count in overlap.items():
        print(f"{pair:20s} {count}")
    if any(overlap.values()):
        print("\nLEAK DETECTED - stopping.")
        return 1
    print("all zero: no property appears in more than one split")

    print("\n--- target distribution per split (price in lakhs) ---")
    print(f"{'split':12s} {'median':>9s} {'mean':>9s} {'p90':>9s} {'max':>9s}")
    for name, frame in (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        price = frame[TARGET]
        print(
            f"{name:12s} {price.median():9.1f} {price.mean():9.1f} "
            f"{price.quantile(0.9):9.1f} {price.max():9.1f}"
        )

    print("\n--- locations never seen in training ---")
    rates = unseen_location_rate(splits)
    for name, rate in rates.items():
        frame = splits.validation if name == "validation" else splits.test
        print(f"{name:12s} {100 * rate:5.2f}%  ({int(rate * len(frame))} rows)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        path = OUT_DIR / f"{name}.csv"
        frame.to_csv(path, index=False)
        print(f"\nwrote {path} ({len(frame)} rows)")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "seed": SEED,
        "rows_cleaned": len(df),
        "n_property_groups": splits.n_groups,
        "sizes": splits.sizes,
        "overlap_between_splits": overlap,
        "unseen_location_rate": rates,
    }
    (METRICS_DIR / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {METRICS_DIR / 'split_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
