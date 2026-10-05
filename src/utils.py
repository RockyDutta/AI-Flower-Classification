"""
utils.py
----------
Shared helper functions used across the pipeline: paths, model
persistence, and small formatting helpers.

Author: Rocky Dutta
"""

from __future__ import annotations

import json
import os
from typing import Any

import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
IMAGES_DIR = os.path.join(BASE_DIR, "images")


def ensure_dir(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def save_model(model: Any, filename: str = "model.pkl") -> str:
    """Persist a trained model object to disk using joblib.

    Args:
        model: The fitted model (or dict bundle) to save.
        filename: Target filename inside the models directory.

    Returns:
        str: Full path to the saved model file.
    """
    ensure_dir(MODELS_DIR)
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, path)
    return path


def load_model(filename: str = "model.pkl") -> Any:
    """Load a previously saved model bundle from disk.

    Args:
        filename: Filename inside the models directory.

    Returns:
        Any: The deserialized model object.
    """
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at {path}. Run `python src/train.py` first."
        )
    return joblib.load(path)


def save_metrics(metrics: dict, filename: str = "metrics.json") -> str:
    """Save an evaluation metrics dictionary to the models directory as JSON."""
    ensure_dir(MODELS_DIR)
    path = os.path.join(MODELS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return path


def load_metrics(filename: str = "metrics.json") -> dict:
    """Load a previously saved metrics JSON file."""
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
