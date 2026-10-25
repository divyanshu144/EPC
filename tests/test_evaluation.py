from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ew_housing_energy_impact.ml.evaluation import (
    FEATURES,
    PROPERTY_KEY,
    assert_property_separation,
    build_preprocessor,
    cap_by_property,
    chronological_partitions,
    grouped_folds,
    predictive_inputs,
    property_partitions,
)


def test_property_assignment_is_order_independent_and_excludes_unknowns():
    df = pd.DataFrame({PROPERTY_KEY: [f"p{i}" for i in range(200)] * 2 + [None, "NOUPRN:c1", " "]})
    partition = property_partitions(df)
    shuffled = df.sample(frac=1, random_state=7)
    pd.testing.assert_series_equal(partition, property_partitions(shuffled).sort_index())
    assert partition.iloc[-3:].eq("excluded_unidentified").all()
    assert assert_property_separation(df, partition)["property_overlap"] == 0
    assert partition.groupby(df[PROPERTY_KEY]).nunique().max() == 1


def test_overlap_checker_rejects_corrupted_assignment():
    df = pd.DataFrame({PROPERTY_KEY: ["same", "same", "other"]})
    partitions = pd.Series(["train", "test", "validation"])
    with pytest.raises(ValueError, match="Property overlap"):
        assert_property_separation(df, partitions)


def test_temporal_split_purges_earlier_records_of_future_properties():
    df = pd.DataFrame(
        {
            PROPERTY_KEY: ["a", "b", "b", "c", "c", "c", "NOUPRN:x"],
            "YEAR": [2020, 2020, 2024, 2020, 2024, 2025, 2025],
        }
    )
    partitions = chronological_partitions(df)
    assert partitions.tolist() == [
        "train",
        "excluded_property_overlap",
        "validation",
        "excluded_property_overlap",
        "excluded_property_overlap",
        "test",
        "excluded_unidentified",
    ]
    assert_property_separation(df, partitions)


def test_grouped_cv_and_sampling_keep_repeat_properties_together():
    df = pd.DataFrame({PROPERTY_KEY: [f"p{i}" for i in range(30)] * 3})
    sampled = cap_by_property(df, max_rows=20)
    assert len(sampled) == 18
    assert sampled.groupby(PROPERTY_KEY).size().eq(3).all()
    assert set(sampled.index) == set(cap_by_property(df.sample(frac=1), 20).index)
    for train, validation in grouped_folds(sampled):
        assert not set(sampled.iloc[train][PROPERTY_KEY]) & set(
            sampled.iloc[validation][PROPERTY_KEY]
        )


def test_imputation_is_fit_on_training_only_and_target_columns_are_excluded():
    train = pd.DataFrame(
        {
            "PROPERTY_TYPE": ["House", "Flat", None],
            "AGE_BAND_STD": ["old", "old", "new"],
            "TENURE_STD": ["Unknown"] * 3,
            "MAIN_FUEL_STD": ["gas"] * 3,
            "LOG_FLOOR_AREA": [2.0, 4.0, np.nan],
            "CURRENT_ENERGY_EFFICIENCY": [99, 99, 99],
            "EFFICIENCY_GAP": [1, 1, 1],
        }
    )
    test = train.copy()
    test["LOG_FLOOR_AREA"] = [1000.0, np.nan, np.nan]
    test["PROPERTY_TYPE"] = ["Castle"] * 3
    inputs = predictive_inputs(train)
    assert inputs.columns.tolist() == FEATURES
    pre = build_preprocessor().fit(inputs)
    assert pre.named_transformers_["num"].named_steps["impute"].statistics_[0] == 3.0
    transformed = pre.transform(predictive_inputs(test))
    assert np.isfinite(transformed).all()
    assert pre.named_transformers_["num"].named_steps["impute"].statistics_[0] == 3.0
