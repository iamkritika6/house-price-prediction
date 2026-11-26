"""Figures describing the data itself, as opposed to the model's performance.

Run:
    docker compose run --rm hpp python scripts/explore_figures.py

These are drawn from the **training set only**, never the test set. Looking at
a chart is a way of learning something from data, and decisions made after
staring at a plot of the test set would be just as leaky as fitting on it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hpp.cleaning import clean  # noqa: E402
from hpp.split import TARGET  # noqa: E402

RAW_CSV = Path("data/raw/Bengaluru_House_Data.csv")
TRAIN_CSV = Path("data/interim/train.csv")
FIGURE_DIR = Path("reports/figures")

BLUE = "#4878a8"


def save(fig: plt.Figure, name: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def price_distribution(train: pd.DataFrame) -> None:
    """Why the model is trained on log(price) rather than price."""
    price = train[TARGET]

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.8))

    left.hist(price, bins=120, color=BLUE)
    left.axvline(price.median(), color="black", linestyle="--", linewidth=1.3,
                 label=f"median {price.median():.0f}")
    left.axvline(price.mean(), color="crimson", linestyle="--", linewidth=1.3,
                 label=f"mean {price.mean():.0f}")
    left.set_xlabel("asking price (lakh)")
    left.set_ylabel("number of listings")
    left.set_title("As published: a long tail to the right")
    left.legend()
    left.grid(alpha=0.25)

    logged = np.log(price)
    right.hist(logged, bins=80, color=BLUE)
    right.axvline(logged.median(), color="black", linestyle="--", linewidth=1.3,
                  label="median")
    right.axvline(logged.mean(), color="crimson", linestyle="--", linewidth=1.3,
                  label="mean")
    right.set_xlabel("log of asking price")
    right.set_ylabel("number of listings")
    right.set_title("On a log scale: roughly symmetric")
    right.legend()
    right.grid(alpha=0.25)

    fig.suptitle(
        "The mean sits well above the median, so a few expensive properties "
        "dominate any average",
        fontsize=11,
    )
    save(fig, "price_distribution.png")


def size_vs_price(train: pd.DataFrame) -> None:
    """The single strongest relationship in the data, and how area_type cuts it."""
    fig, ax = plt.subplots(figsize=(9.5, 6))

    for area_type, group in train.groupby("area_type"):
        ax.scatter(
            group["total_sqft_num"],
            group[TARGET],
            s=9,
            alpha=0.35,
            edgecolor="none",
            label=f"{area_type.strip()} (n={len(group)})",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("size (sq ft, log scale)")
    ax.set_ylabel("asking price (lakh, log scale)")
    ax.set_title("Size against price, split by how the area was measured")
    ax.legend(markerscale=2.5, fontsize=9)
    ax.grid(alpha=0.25)
    save(fig, "size_vs_price.png")


def top_locations(train: pd.DataFrame) -> None:
    """Price per square foot varies several-fold across the city."""
    rate = train[TARGET] / train["total_sqft_num"]
    frame = pd.DataFrame({"location": train["location"], "rate": rate})

    grouped = frame.groupby("location")["rate"]
    summary = pd.DataFrame({"median_rate": grouped.median(), "n": grouped.size()})
    summary = summary[summary["n"] >= 30].sort_values("median_rate")

    top = pd.concat([summary.head(10), summary.tail(10)])

    fig, ax = plt.subplots(figsize=(9, 8))
    colours = [BLUE] * 10 + ["#a85048"] * 10
    bars = ax.barh(top.index, top["median_rate"] * 100_000, color=colours)
    for bar, count in zip(bars, top["n"]):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"n={count}", va="center", fontsize=8)

    ax.set_xlabel("median asking price per square foot (rupees)")
    ax.set_title(
        "Ten cheapest and ten dearest localities\n"
        "(localities with at least 30 training listings)"
    )
    ax.grid(axis="x", alpha=0.25)
    save(fig, "price_per_sqft_by_location.png")


def where_the_rows_went() -> None:
    """A visual audit of what cleaning removed, and why."""
    raw = pd.read_csv(RAW_CSV, dtype=str, keep_default_na=False, na_values=[""])
    cleaned = clean(raw)

    breakdown = cleaned.attrs["implausible_breakdown"]
    steps = {
        "raw listings": len(raw),
        "exact duplicates": -cleaned.attrs["n_duplicates_dropped"],
        "no usable size or price": -cleaned.attrs["n_unusable_dropped"],
        "under 100 sq ft": -breakdown["total_sqft < 100"],
        "over 50,000 sq ft": -breakdown["total_sqft > 50000"],
        "more than 20 bedrooms": -breakdown["bedrooms > 20"],
        "under 150 sq ft per bedroom": -breakdown["sqft per bedroom < 150"],
    }

    labels = list(steps) + ["remaining"]
    values = list(steps.values()) + [len(cleaned)]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    colours = [BLUE] + ["#a85048"] * (len(values) - 2) + ["#4a8a5a"]
    bars = ax.bar(labels, [abs(v) for v in values], color=colours)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{value:+,}" if value < 0 else f"{value:,}",
                ha="center", fontsize=9)

    ax.set_yscale("log")
    ax.set_ylabel("listings (log scale)")
    ax.set_title("Where the rows went: 13,320 raw listings to 12,612 usable")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    save(fig, "cleaning_audit.png")


def main() -> int:
    train = pd.read_csv(TRAIN_CSV)
    print(f"drawing from the training set only: {len(train)} rows\n")

    price_distribution(train)
    size_vs_price(train)
    top_locations(train)
    where_the_rows_went()
    return 0


if __name__ == "__main__":
    sys.exit(main())
