"""Plots that show where the model fails, not just that it works.

A single accuracy number hides the shape of the errors. These four figures are
chosen because each answers a question an interviewer is likely to ask.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hpp.metrics import Scores  # noqa: E402

FIGURE_DIR = Path("reports/figures")


def _save(fig: plt.Figure, name: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, name: str = "predicted_vs_actual.png") -> Path:
    """Is the model systematically wrong anywhere?

    Points on the diagonal are perfect. Log scales on both axes, because
    prices span 8 lakh to 3,600 lakh and a linear axis would compress almost
    every point into the bottom-left corner.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, s=8, alpha=0.3, edgecolor="none")

    limits = [min(y_true.min(), y_pred.min()) * 0.9, max(y_true.max(), y_pred.max()) * 1.1]
    ax.plot(limits, limits, linestyle="--", linewidth=1.2, color="black", label="perfect prediction")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("actual asking price (lakh)")
    ax.set_ylabel("predicted asking price (lakh)")
    ax.set_title("Predicted vs actual asking price")
    ax.legend()
    ax.grid(alpha=0.25)
    return _save(fig, name)


def residuals(y_true: np.ndarray, y_pred: np.ndarray, name: str = "residuals.png") -> Path:
    """Percentage error against the predicted price.

    If the cloud is centred on zero and the same width everywhere, the model
    is evenly reliable. A funnel shape, or a drift away from zero, says the
    model is biased in part of the market.
    """
    percentage_error = 100.0 * (y_pred - y_true) / y_true

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(y_pred, percentage_error, s=8, alpha=0.3, edgecolor="none")
    ax.axhline(0, linestyle="--", linewidth=1.2, color="black")
    ax.set_xscale("log")
    ax.set_ylim(-150, 300)
    ax.set_xlabel("predicted asking price (lakh, log scale)")
    ax.set_ylabel("error as % of actual price")
    ax.set_title("Where the model is wrong, and by how much")
    ax.grid(alpha=0.25)
    return _save(fig, name)


def error_by_band(bands: dict[str, Scores], name: str = "error_by_price_band.png") -> Path:
    """Median percentage error within each price band, with counts."""
    labels = list(bands)
    values = [bands[label].median_ape_percent for label in labels]
    counts = [bands[label].n for label in labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color="#4878a8")
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"n={count}",
            ha="center",
            fontsize=9,
        )
    ax.set_ylabel("median absolute percentage error")
    ax.set_xlabel("actual asking price")
    ax.set_title("Typical error by price band")
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, name)


def importance(
    names: list[str],
    means: np.ndarray,
    stds: np.ndarray,
    top: int = 20,
    name: str = "permutation_importance.png",
) -> Path:
    """Permutation importance: how much worse does the model get without it?

    Each feature is shuffled in turn and the model re-scored. A feature that
    matters produces a big drop. Unlike a tree's built-in importance, this
    measures effect on *predictions*, and it is measured on held-out data.
    """
    order = np.argsort(means)[-top:]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.32 * len(order))))
    ax.barh(
        [names[i] for i in order],
        means[order],
        xerr=stds[order],
        color="#4878a8",
    )
    ax.set_xlabel("increase in RMSE when this feature is shuffled (lakh)")
    ax.set_title("Permutation importance (measured on held-out data)")
    ax.grid(axis="x", alpha=0.25)
    return _save(fig, name)
