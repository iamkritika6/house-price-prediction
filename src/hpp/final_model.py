"""The model that ships, and the reasoning for choosing it.

Chosen: **Random Forest, with locality encoded by its typical price, trained
on log(price).**

Why this one
------------
Cross-validated RMSE on the training set, 5 grouped folds:

===========================================  ==================
model                                        CV RMSE (lakh)
===========================================  ==================
ridge, one-hot locality                      97.59 +/- 11.36
lightgbm, one-hot locality                   94.98 +/- 16.90
random forest, one-hot locality              94.27 +/- 13.98
ridge, target-encoded locality               92.33 +/-  8.01
lightgbm, target-encoded locality            88.63 +/- 12.61
lightgbm, tuned over 40 settings             86.34 +/- 12.92
random forest, target-encoded locality       85.91 +/- 12.62
===========================================  ==================

The notable result is the last two rows. LightGBM was given a 40-setting
randomised search costing 966 seconds, and still did not beat a Random Forest
left on sensible defaults.

That makes the choice **conservative rather than lucky**: the comparison is
deliberately unfair in LightGBM's favour, because only LightGBM was tuned.
Random Forest won anyway, so tuning it as well could only widen the gap, not
reverse it.

What this does not claim
------------------------
The gap between the top few rows is smaller than the fold-to-fold spread, so
this is not evidence that Random Forest is *the* right algorithm for house
price data in general. It is evidence that on this dataset, at this size, the
simpler and faster-to-fit option is at least as good -- and it was chosen on
the training and validation sets, never on the test set.

The one change that genuinely helped was not the algorithm at all. It was
encoding locality by its typical price instead of giving each of 1,265
localities its own column. That improved every model family, in the same
direction, by a similar amount -- which is stronger evidence than any single
comparison, because three different algorithms do not all get lucky the same
way.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor

from hpp.pipeline import make_model

SEED = 42

DESCRIPTION = (
    "Random Forest (300 trees, min_samples_leaf=2), locality target-encoded, "
    "trained on log(price)"
)


def make_final_model():
    """Build the final model, unfitted."""
    return make_model(
        RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=-1,
        ),
        scale=False,
        location_encoding="target",
        log_target=True,
    )
