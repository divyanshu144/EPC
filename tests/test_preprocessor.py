"""Tests for the modelling preprocessor's design matrix.

Regression tests for a numerical failure observed on real data: the one-hot
encoder produced a rank-deficient design matrix, LinearRegression solved it via
least squares, and the resulting minimum-norm coefficients (order 1e11) cancelled
exactly on training rows but exploded on the three test rows containing an unseen
category. Reported R2 was -1.13e15.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from scripts.pipeline.train_models import build_preprocessor

CAT = ["PROPERTY_TYPE", "TENURE_STD", "MAIN_FUEL_STD"]
NUM = ["LOG_FLOOR_AREA"]


def _frame(n: int = 240) -> pd.DataFrame:
    """A frame with several multi-level categoricals, like the real feature set."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "PROPERTY_TYPE": rng.choice(["House", "Flat", "Bungalow"], n),
            "TENURE_STD": rng.choice(["Owner-occupied", "Rented (private)"], n),
            "MAIN_FUEL_STD": rng.choice(["gas", "electric"], n),
            "LOG_FLOOR_AREA": rng.normal(4.5, 0.3, n),
        }
    )


def test_design_matrix_is_full_rank():
    """Each one-hot block sums to 1, so without a dropped level the blocks are
    mutually collinear and collinear with the intercept. That rank deficiency is
    what made the least-squares solution arbitrary along the null space.
    """
    df = _frame()
    matrix = build_preprocessor(CAT, NUM).fit_transform(df)

    rank = np.linalg.matrix_rank(matrix)
    assert rank == matrix.shape[1], (
        f"Design matrix is rank deficient: rank {rank} of {matrix.shape[1]} columns. "
        "Unregularised OLS on a deficient design yields arbitrary null-space "
        "coefficients that explode on rows outside the training row space."
    )


def test_categorical_blocks_drop_one_level():
    """A k-level categorical must contribute k-1 columns, not k."""
    df = _frame()
    pre = build_preprocessor(CAT, NUM).fit(df)
    encoder = pre.named_transformers_["cat"]

    for column, categories in zip(CAT, encoder.categories_):
        expected = len(categories) - 1
        actual = len([c for c in encoder.get_feature_names_out() if c.startswith(column)])
        assert actual == expected, f"{column}: expected {expected} columns, got {actual}"


def test_unseen_category_does_not_explode_predictions():
    """The exact failure seen on real data.

    Three of 25,032 test rows carried an AGE_BAND_STD value absent from train.
    Under handle_unknown='ignore' their block encoded as all zeros, which broke
    the cancellation the huge coefficients depended on, predicting 4.07e10
    against actuals near 50-90.
    """
    train = _frame()
    target = 50 + 10 * train["LOG_FLOOR_AREA"] + np.random.default_rng(1).normal(0, 1, len(train))

    pipe = Pipeline([("pre", build_preprocessor(CAT, NUM)), ("model", LinearRegression())]).fit(
        train[CAT + NUM], target
    )

    unseen = pd.DataFrame(
        {
            "PROPERTY_TYPE": ["Castle"],  # never seen in training
            "TENURE_STD": ["Owner-occupied"],
            "MAIN_FUEL_STD": ["gas"],
            "LOG_FLOOR_AREA": [4.5],
        }
    )
    prediction = pipe.predict(unseen[CAT + NUM])[0]

    assert abs(prediction) < 1_000, (
        f"Prediction for an unseen category exploded to {prediction:.3e}. "
        "This is the rank-deficiency failure recurring."
    )


def test_coefficients_stay_bounded():
    """Coefficients of order 1e11 are the signature of the minimum-norm solution
    to a singular system, not of a fitted model."""
    train = _frame()
    target = 50 + 10 * train["LOG_FLOOR_AREA"]

    pipe = Pipeline([("pre", build_preprocessor(CAT, NUM)), ("model", LinearRegression())]).fit(
        train[CAT + NUM], target
    )

    largest = np.abs(pipe.named_steps["model"].coef_).max()
    assert largest < 1_000, f"Largest coefficient is {largest:.3e}, indicating a singular solve"


@pytest.mark.parametrize("n_levels", [2, 5, 12])
def test_full_rank_holds_across_cardinalities(n_levels):
    """Rank must hold regardless of how many levels a categorical has."""
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "PROPERTY_TYPE": rng.choice([f"cat{i}" for i in range(n_levels)], 300),
            "TENURE_STD": rng.choice(["a", "b"], 300),
            "MAIN_FUEL_STD": rng.choice(["x", "y"], 300),
            "LOG_FLOOR_AREA": rng.normal(4.5, 0.3, 300),
        }
    )
    matrix = build_preprocessor(CAT, NUM).fit_transform(df)
    assert np.linalg.matrix_rank(matrix) == matrix.shape[1]
