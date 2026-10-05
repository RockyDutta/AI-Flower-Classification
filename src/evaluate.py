"""
evaluate.py
-------------
Standalone evaluation utilities for inspecting a saved model's
performance without re-running the full training pipeline.

Run directly with:
    python src/evaluate.py

Author: Rocky Dutta
"""

from __future__ import annotations

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data_loader import get_feature_target_split, load_iris_dataframe
from preprocessing import split_data
from utils import load_metrics, load_model


def evaluate_saved_model(random_state: int = 42) -> None:
    """Load the saved model bundle and print an evaluation report."""
    bundle = load_model("model.pkl")
    model = bundle["model"]
    scaler = bundle["scaler"]
    target_names = bundle["target_names"]

    df = load_iris_dataframe()
    x, y = get_feature_target_split(df)
    _, x_test, _, y_test = split_data(x, y, test_size=0.2, random_state=random_state)

    import pandas as pd

    x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)
    preds = model.predict(x_test_scaled)

    print(f"Model: {bundle.get('model_name', 'Unknown')}")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, preds, target_names=target_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    stored_metrics = load_metrics()
    if stored_metrics:
        print("\nStored training metrics summary available in models/metrics.json")


if __name__ == "__main__":
    evaluate_saved_model()
