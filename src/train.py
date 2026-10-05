"""
train.py
----------
End-to-end training script for the AI Flower Classification System.

Pipeline:
    1. Load dataset
    2. Clean data (missing values, duplicates)
    3. Split into train/test sets
    4. Scale features
    5. Train and compare multiple classifiers
    6. Tune KNN's K hyperparameter
    7. Evaluate the best model
    8. Save the final model bundle (model + scaler + metadata) to disk

Run directly with:
    python src/train.py

Author: Rocky Dutta
"""

from __future__ import annotations

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from data_loader import (
    FEATURE_NAMES,
    TARGET_NAMES,
    dataset_summary,
    get_feature_target_split,
    load_iris_dataframe,
)
from preprocessing import drop_duplicates, scale_features, split_data
from utils import save_metrics, save_model
from visualization import (
    generate_all_eda_plots,
    plot_confusion_matrix,
    plot_k_accuracy_curve,
    plot_model_comparison,
)

K_CANDIDATES = [1, 3, 5, 7, 9, 11, 15, 21]
RANDOM_STATE = 42


def build_candidate_models(best_k: int) -> dict:
    """Return a dictionary of candidate models to compare."""
    return {
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=best_k),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
        "Logistic Regression": LogisticRegression(max_iter=1000),
    }


def tune_knn_k(x_train, y_train, x_test, y_test) -> tuple[int, list[float]]:
    """Try several K values for KNN and return the best K plus accuracy list."""
    accuracies = []
    for k in K_CANDIDATES:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        accuracies.append(accuracy_score(y_test, preds))
    best_k = K_CANDIDATES[int(np.argmax(accuracies))]
    return best_k, accuracies


def run_training_pipeline() -> dict:
    """Execute the full training pipeline and return a metrics summary."""
    print("Step 1/8: Loading dataset...")
    df = load_iris_dataframe()
    summary = dataset_summary(df)
    print(f"  Dataset summary: {summary}")

    print("Step 2/8: Cleaning data (duplicates)...")
    df_clean = drop_duplicates(df)
    print(f"  Rows before: {len(df)}, after de-dup: {len(df_clean)}")

    print("Step 3/8: Generating EDA visualizations...")
    generate_all_eda_plots(df_clean, FEATURE_NAMES)

    print("Step 4/8: Splitting data...")
    x, y = get_feature_target_split(df_clean)
    x_train, x_test, y_train, y_test = split_data(x, y, test_size=0.2, random_state=RANDOM_STATE)

    print("Step 5/8: Scaling features...")
    x_train_scaled, x_test_scaled, scaler = scale_features(x_train, x_test)

    print("Step 6/8: Tuning KNN hyperparameter K...")
    best_k, k_accuracies = tune_knn_k(x_train_scaled, y_train, x_test_scaled, y_test)
    plot_k_accuracy_curve(K_CANDIDATES, k_accuracies)
    print(f"  Best K found: {best_k}")

    print("Step 7/8: Training and comparing models...")
    models = build_candidate_models(best_k)
    results = {}
    for name, model in models.items():
        model.fit(x_train_scaled, y_train)
        preds = model.predict(x_test_scaled)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        cv_scores = cross_val_score(model, x_train_scaled, y_train, cv=5)
        results[name] = {
            "accuracy": float(acc),
            "f1_score": float(f1),
            "cv_mean_accuracy": float(cv_scores.mean()),
            "cv_std_accuracy": float(cv_scores.std()),
        }
        print(f"  {name}: accuracy={acc:.4f}, f1={f1:.4f}, cv_mean={cv_scores.mean():.4f}")

    plot_model_comparison(list(results.keys()), [v["accuracy"] for v in results.values()])

    best_model_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = models[best_model_name]
    best_preds = best_model.predict(x_test_scaled)

    print(f"Step 8/8: Best model = {best_model_name}. Saving model bundle...")
    report = classification_report(
        y_test, best_preds, target_names=TARGET_NAMES, output_dict=True
    )
    plot_confusion_matrix(y_test, best_preds, TARGET_NAMES)

    model_bundle = {
        "model": best_model,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "model_name": best_model_name,
        "best_k": best_k,
    }
    save_model(model_bundle, filename="model.pkl")

    metrics_summary = {
        "dataset_summary": summary,
        "best_k": best_k,
        "k_accuracies": dict(zip(K_CANDIDATES, k_accuracies)),
        "model_comparison": results,
        "best_model": best_model_name,
        "classification_report": report,
        "test_accuracy": results[best_model_name]["accuracy"],
    }
    save_metrics(metrics_summary)

    print("\nTraining pipeline complete.")
    print(f"Best Model: {best_model_name} | Test Accuracy: {results[best_model_name]['accuracy']:.4f}")
    return metrics_summary


if __name__ == "__main__":
    run_training_pipeline()
