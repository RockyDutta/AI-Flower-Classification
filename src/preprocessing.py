"""
preprocessing.py
------------------
Data cleaning, feature scaling, and train/test split utilities.

Author: Rocky Dutta
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """Return the count of missing values per column.

    Args:
        df: Input DataFrame.

    Returns:
        pd.Series: Missing value counts indexed by column name.
    """
    return df.isnull().sum()


def check_duplicates(df: pd.DataFrame) -> int:
    """Return the number of duplicate rows in the DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        int: Number of duplicate rows.
    """
    return int(df.duplicated().sum())


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows, keeping the first occurrence.

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame without duplicate rows.
    """
    return df.drop_duplicates().reset_index(drop=True)


def scale_features(
    x_train: pd.DataFrame, x_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit a StandardScaler on training data and transform both splits.

    Args:
        x_train: Training features.
        x_test: Testing features.

    Returns:
        tuple: Scaled training features, scaled testing features, and the
        fitted scaler (for reuse at inference time).
    """
    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(
        scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index
    )
    x_test_scaled = pd.DataFrame(
        scaler.transform(x_test), columns=x_test.columns, index=x_test.index
    )
    return x_train_scaled, x_test_scaled, scaler


def split_data(
    x: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
):
    """Perform a stratified train/test split.

    Args:
        x: Feature DataFrame.
        y: Target Series.
        test_size: Fraction of data reserved for testing.
        random_state: Seed for reproducibility.

    Returns:
        tuple: x_train, x_test, y_train, y_test
    """
    return train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
