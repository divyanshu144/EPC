"""Frozen property partitions and fold-local preprocessing for the new benchmark.

Only identified properties are eligible. A certificate-specific fallback key is
not evidence of a distinct dwelling. Hash partitions are independent of input
ordering and stable when unrelated rows are added. Unknown UPRN linkage errors
remain a source limitation that no splitter can resolve.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROPERTY_KEY = "BUILDING_REFERENCE_NUMBER"
CATEGORICAL = ["PROPERTY_TYPE", "AGE_BAND_STD", "TENURE_STD", "MAIN_FUEL_STD"]
NUMERIC = ["LOG_FLOOR_AREA"]
FEATURES = CATEGORICAL + NUMERIC
SEED = 42


def _digest(key: str, namespace: str) -> str:
    return hashlib.sha256(f"{namespace}:{key}".encode()).hexdigest()


def identified_properties(df: pd.DataFrame) -> pd.Series:
    if PROPERTY_KEY not in df:
        raise ValueError(f"Missing required property identifier: {PROPERTY_KEY}")
    keys = df[PROPERTY_KEY].astype("string").str.strip()
    return keys.notna() & keys.ne("") & ~keys.str.startswith("NOUPRN:", na=False)


def property_partitions(df: pd.DataFrame, seed: int = SEED) -> pd.Series:
    """Assign identified dwellings to 60/20/20 train/validation/test partitions."""
    eligible = identified_properties(df)
    keys = df[PROPERTY_KEY].astype("string").str.strip()
    lookup = {}
    for key in keys.loc[eligible].unique():
        fraction = int(_digest(key, f"property-split-v1:{seed}")[:16], 16) / 2**64
        lookup[key] = "train" if fraction < 0.6 else "validation" if fraction < 0.8 else "test"
    return keys.map(lookup).fillna("excluded_unidentified").rename("partition")


def chronological_partitions(
    df: pd.DataFrame, validation_year: int = 2024, test_year: int = 2025
) -> pd.Series:
    """Evaluate later certificates from properties never seen in earlier partitions.

    Properties in test-year data have ALL earlier records excluded. Validation
    properties likewise have earlier records excluded. This tests future records
    of unseen properties, not revisits to previously observed properties.
    """
    if validation_year >= test_year:
        raise ValueError("validation_year must precede test_year")
    eligible = identified_properties(df)
    years = pd.to_numeric(df["YEAR"], errors="coerce")
    keys = df[PROPERTY_KEY].astype("string").str.strip()
    test_props = set(keys.loc[eligible & years.eq(test_year)])
    val_props = set(keys.loc[eligible & years.eq(validation_year)]) - test_props
    partition = pd.Series("excluded_outside_time_scope", index=df.index, name="partition")
    partition.loc[~eligible] = "excluded_unidentified"
    partition.loc[eligible & years.lt(validation_year)] = "train"
    partition.loc[eligible & keys.isin(test_props | val_props) & years.lt(validation_year)] = (
        "excluded_property_overlap"
    )
    partition.loc[eligible & years.eq(validation_year) & keys.isin(test_props)] = (
        "excluded_property_overlap"
    )
    partition.loc[eligible & years.eq(validation_year) & keys.isin(val_props)] = "validation"
    partition.loc[eligible & years.eq(test_year)] = "test"
    return partition


def assert_property_separation(df: pd.DataFrame, partitions: pd.Series) -> dict:
    """Check the actual assignments, rather than trusting the splitting algorithm."""
    if not partitions.index.equals(df.index):
        raise ValueError("Partition index does not match data index")
    groups = {}
    counts = {}
    for name in ["train", "validation", "test"]:
        part = df.loc[partitions.eq(name)]
        if part.empty:
            raise ValueError(f"Empty {name} partition")
        if not identified_properties(part).all():
            raise ValueError(f"Unidentified properties in {name}")
        groups[name] = set(part[PROPERTY_KEY].astype("string").str.strip())
        counts[name] = {"rows": len(part), "properties": len(groups[name])}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if groups[a] & groups[b]:
            raise ValueError(f"Property overlap between {a} and {b}")
    return {
        "partitions": counts,
        "property_overlap": 0,
        "excluded": partitions[~partitions.isin(groups)].value_counts().to_dict(),
    }


def cap_by_property(df: pd.DataFrame, max_rows: int, seed: int = SEED) -> pd.DataFrame:
    """Deterministic property sampling; never split a property's records to hit a cap."""
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    if not identified_properties(df).all():
        raise ValueError("Sampling requires identified properties")
    if len(df) <= max_rows:
        return df.copy()
    keys = df[PROPERTY_KEY].astype("string").str.strip()
    counts = keys.value_counts()
    order = sorted(counts.index, key=lambda key: _digest(key, f"property-sample-v1:{seed}"))
    selected = []
    total = 0
    for key in order:
        size = int(counts[key])
        if total + size <= max_rows:
            selected.append(key)
            total += size
        if total == max_rows:
            break
    if not selected:
        raise ValueError("No complete property fits within row cap")
    return df.loc[keys.isin(selected)].copy()


def grouped_folds(df: pd.DataFrame, n_splits: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
    if not identified_properties(df).all():
        raise ValueError("Cross-validation requires identified properties")
    groups = df[PROPERTY_KEY].astype("string").str.strip()
    folds = list(GroupKFold(n_splits=n_splits).split(df, groups=groups))
    for train, validation in folds:
        if set(groups.iloc[train]) & set(groups.iloc[validation]):
            raise AssertionError("Property overlap in cross-validation")
    return folds


def predictive_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Use a fixed target-free feature list and sklearn-compatible missing values."""
    result = df[FEATURES].copy()
    for column in CATEGORICAL:
        result[column] = result[column].astype(object).where(result[column].notna(), np.nan)
    for column in NUMERIC:
        result[column] = pd.to_numeric(result[column], errors="coerce").astype(float)
    return result


def build_preprocessor() -> ColumnTransformer:
    categorical = Pipeline(
        [
            (
                "impute",
                SimpleImputer(strategy="constant", fill_value="Unknown", keep_empty_features=True),
            ),
            ("encode", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
        ]
    )
    numeric = Pipeline(
        [
            (
                "impute",
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            ),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer([("cat", categorical, CATEGORICAL), ("num", numeric, NUMERIC)])
