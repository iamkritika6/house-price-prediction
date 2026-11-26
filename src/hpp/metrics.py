"""Error metrics, reported in money rather than in abstract scores.

Why not R-squared alone
-----------------------
R-squared says what fraction of the variance in price the model explains. It
is unitless, it flatters a model on a right-skewed target, and it answers a
question nobody actually asks. "How far out is this valuation likely to be,
in rupees?" is the question.

The three metrics used here
---------------------------
``RMSE``
    Root mean squared error. Squaring before averaging means a few very large
    mistakes dominate it. Report it when big misses are what hurt.

``MAE``
    Mean absolute error. The typical size of a miss, in rupees. Less affected
    by a handful of mansions than RMSE.

RMSE is always at least as large as MAE. The gap between them tells you how
uneven the errors are: close together means the model is consistently a bit
wrong, far apart means it is usually close but occasionally badly wrong.

``MAPE``
    Mean absolute percentage error. Being out by 20 lakh matters far more on a
    40 lakh flat than on a 4 crore house, and a percentage captures that.
    MAPE is asymmetric -- it punishes over-prediction more than
    under-prediction -- so it is reported alongside the other two, never alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RUPEES_PER_LAKH = 100_000


@dataclass(frozen=True)
class Scores:
    """Error of a set of predictions, with the target measured in lakhs."""

    rmse_lakh: float
    mae_lakh: float
    mape_percent: float
    median_ape_percent: float
    r2: float
    n: int

    @property
    def rmse_rupees(self) -> float:
        return self.rmse_lakh * RUPEES_PER_LAKH

    @property
    def mae_rupees(self) -> float:
        return self.mae_lakh * RUPEES_PER_LAKH

    def as_dict(self) -> dict[str, float]:
        return {
            "n": self.n,
            "rmse_lakh": self.rmse_lakh,
            "mae_lakh": self.mae_lakh,
            "rmse_rupees": self.rmse_rupees,
            "mae_rupees": self.mae_rupees,
            "mape_percent": self.mape_percent,
            "median_ape_percent": self.median_ape_percent,
            "r2": self.r2,
        }

    def describe(self) -> str:
        return (
            f"RMSE {self.rmse_lakh:7.2f} lakh (Rs {self.rmse_rupees:,.0f})  "
            f"MAE {self.mae_lakh:7.2f} lakh (Rs {self.mae_rupees:,.0f})  "
            f"MAPE {self.mape_percent:5.1f}%  "
            f"median APE {self.median_ape_percent:5.1f}%  "
            f"R2 {self.r2:6.3f}"
        )


def score(y_true: np.ndarray, y_pred: np.ndarray) -> Scores:
    """Score predictions against the truth, both in lakhs."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("cannot score an empty set of predictions")

    error = y_pred - y_true
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))

    # Prices are all strictly positive in this dataset, but guard anyway:
    # a zero true value would make the percentage error infinite.
    nonzero = y_true != 0
    ape = np.abs(error[nonzero] / y_true[nonzero]) * 100.0
    mape = float(np.mean(ape))
    median_ape = float(np.median(ape))

    total_variance = float(np.sum((y_true - y_true.mean()) ** 2))
    residual_variance = float(np.sum(error**2))
    r2 = 1.0 - residual_variance / total_variance if total_variance > 0 else float("nan")

    return Scores(
        rmse_lakh=rmse,
        mae_lakh=mae,
        mape_percent=mape,
        median_ape_percent=median_ape,
        r2=r2,
        n=int(y_true.size),
    )


def score_by_band(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    edges: tuple[float, ...] = (0, 50, 75, 100, 150, 250, np.inf),
) -> dict[str, Scores]:
    """Score separately within price bands.

    A single average hides where a model fails. Property price models are
    usually good in the middle of the market and poor at the top, and a buyer
    of a 40 lakh flat is not reassured by an average that was rescued by
    accurate predictions on 3 crore houses.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    out: dict[str, Scores] = {}
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (y_true >= low) & (y_true < high)
        if mask.sum() == 0:
            continue
        label = f"{low:.0f}-{high:.0f} lakh" if np.isfinite(high) else f"{low:.0f}+ lakh"
        out[label] = score(y_true[mask], y_pred[mask])
    return out
