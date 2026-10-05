"""
test_pipeline.py
------------------
Unit tests covering data loading, preprocessing, and prediction
functionality of the AI Flower Classification System.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pandas as pd
import pytest

from data_loader import (
    FEATURE_NAMES,
    TARGET_NAMES,
    dataset_summary,
    get_feature_target_split,
    load_iris_dataframe,
)
from preprocessing import check_duplicates, check_missing_values, scale_features, split_data
from predict import predict_species


@pytest.fixture(scope="module")
def iris_df() -> pd.DataFrame:
    return load_iris_dataframe()


def test_load_iris_dataframe_shape(iris_df: pd.DataFrame) -> None:
    assert iris_df.shape[0] == 150
    assert set(FEATURE_NAMES).issubset(iris_df.columns)


def test_target_names_present(iris_df: pd.DataFrame) -> None:
    assert set(iris_df["species"].unique()) == set(TARGET_NAMES)


def test_dataset_summary(iris_df: pd.DataFrame) -> None:
    summary = dataset_summary(iris_df)
    assert summary["n_samples"] == 150
    assert summary["n_classes"] == 3
    assert summary["missing_values"] == 0


def test_no_missing_values(iris_df: pd.DataFrame) -> None:
    missing = check_missing_values(iris_df)
    assert missing.sum() == 0


def test_duplicate_check(iris_df: pd.DataFrame) -> None:
    dup_count = check_duplicates(iris_df)
    assert dup_count >= 0


def test_feature_target_split(iris_df: pd.DataFrame) -> None:
    x, y = get_feature_target_split(iris_df)
    assert list(x.columns) == FEATURE_NAMES
    assert len(x) == len(y)


def test_split_data_shapes(iris_df: pd.DataFrame) -> None:
    x, y = get_feature_target_split(iris_df)
    x_train, x_test, y_train, y_test = split_data(x, y, test_size=0.2, random_state=42)
    assert len(x_train) + len(x_test) == len(x)
    assert abs(len(x_test) / len(x) - 0.2) < 0.05


def test_scale_features(iris_df: pd.DataFrame) -> None:
    x, y = get_feature_target_split(iris_df)
    x_train, x_test, _, _ = split_data(x, y, test_size=0.2, random_state=42)
    x_train_scaled, x_test_scaled, scaler = scale_features(x_train, x_test)
    assert abs(x_train_scaled.mean().mean()) < 1e-6
    assert scaler is not None


def test_predict_species_returns_valid_output() -> None:
    result = predict_species(5.1, 3.5, 1.4, 0.2)
    assert result["predicted_species"] in TARGET_NAMES
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["probabilities"], dict)
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-6


def test_predict_species_setosa_case() -> None:
    # Classic small-petal measurements strongly associated with setosa
    result = predict_species(5.0, 3.4, 1.5, 0.2)
    assert result["predicted_species"] == "setosa"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
