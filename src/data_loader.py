"""
data_loader.py
----------------
Handles loading of the Iris dataset from scikit-learn and provides
a clean pandas DataFrame interface for the rest of the pipeline.

Author: Rocky Dutta
"""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_iris


FEATURE_NAMES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

TARGET_NAMES = ["setosa", "versicolor", "virginica"]


def load_iris_dataframe() -> pd.DataFrame:
    """Load the Iris dataset and return it as a tidy pandas DataFrame.

    Returns:
        pd.DataFrame: DataFrame with feature columns, a numeric ``target``
        column, and a human-readable ``species`` column.
    """
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=FEATURE_NAMES)
    df["target"] = iris.target
    df["species"] = df["target"].map(dict(enumerate(TARGET_NAMES)))
    return df


def get_feature_target_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into features (X) and target (y).

    Args:
        df: DataFrame produced by :func:`load_iris_dataframe`.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Features and target series.
    """
    x = df[FEATURE_NAMES].copy()
    y = df["target"].copy()
    return x, y


def dataset_summary(df: pd.DataFrame) -> dict:
    """Compute a quick summary dictionary of the dataset.

    Args:
        df: The Iris DataFrame.

    Returns:
        dict: Summary containing shape, missing values, duplicate count,
        and class distribution.
    """
    return {
        "n_samples": int(df.shape[0]),
        "n_features": len(FEATURE_NAMES),
        "n_classes": df["species"].nunique(),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_distribution": df["species"].value_counts().to_dict(),
    }


if __name__ == "__main__":
    data = load_iris_dataframe()
    print(data.head())
    print(dataset_summary(data))
