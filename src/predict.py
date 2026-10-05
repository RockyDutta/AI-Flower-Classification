"""
predict.py
------------
Inference utilities for making predictions with the saved model bundle.
Used by both the CLI and the Streamlit web application.

Author: Rocky Dutta
"""

from __future__ import annotations

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from utils import load_model


def predict_species(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
) -> dict:
    """Predict the Iris species for a single flower measurement.

    Args:
        sepal_length: Sepal length in cm.
        sepal_width: Sepal width in cm.
        petal_length: Petal length in cm.
        petal_width: Petal width in cm.

    Returns:
        dict: Prediction result containing the predicted species,
        confidence score, and full probability breakdown.
    """
    bundle = load_model("model.pkl")
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_names = bundle["feature_names"]
    target_names = bundle["target_names"]

    input_df = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=feature_names,
    )
    input_scaled = pd.DataFrame(
        scaler.transform(input_df), columns=feature_names
    )

    prediction_idx = model.predict(input_scaled)[0]
    predicted_species = target_names[prediction_idx]

    probabilities = None
    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_scaled)[0]
        confidence = float(np.max(probabilities))

    result = {
        "predicted_species": predicted_species,
        "confidence": confidence,
        "probabilities": (
            dict(zip(target_names, [float(p) for p in probabilities]))
            if probabilities is not None
            else None
        ),
        "model_used": bundle.get("model_name", "Unknown"),
    }
    return result


if __name__ == "__main__":
    example = predict_species(5.1, 3.5, 1.4, 0.2)
    print(example)
