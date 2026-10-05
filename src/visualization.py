"""
visualization.py
------------------
Generates professional, publication-quality plots for exploratory data
analysis and model evaluation. All plots are saved to the /images
directory.

Author: Rocky Dutta
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"

IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images")


def _ensure_images_dir() -> None:
    os.makedirs(IMAGES_DIR, exist_ok=True)


def plot_histograms(df: pd.DataFrame, feature_cols: list[str]) -> str:
    """Plot histograms for each numeric feature."""
    _ensure_images_dir()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, col in zip(axes.flatten(), feature_cols):
        sns.histplot(df[col], kde=True, ax=ax, color="#4C72B0")
        ax.set_title(f"Distribution of {col.replace('_', ' ').title()}")
    fig.suptitle("Feature Histograms", fontsize=16, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(IMAGES_DIR, "histograms.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_count_plot(df: pd.DataFrame) -> str:
    """Plot a count plot of class distribution."""
    _ensure_images_dir()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(data=df, x="species", hue="species", palette="viridis", ax=ax, legend=False)
    ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
    path = os.path.join(IMAGES_DIR, "count_plot.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_pair_plot(df: pd.DataFrame, feature_cols: list[str]) -> str:
    """Plot a pair plot colored by species."""
    _ensure_images_dir()
    grid = sns.pairplot(
        df[feature_cols + ["species"]], hue="species", palette="viridis", corner=True
    )
    grid.fig.suptitle("Pair Plot of Iris Features", y=1.02, fontsize=16, fontweight="bold")
    path = os.path.join(IMAGES_DIR, "pair_plot.png")
    grid.savefig(path)
    plt.close(grid.fig)
    return path


def plot_correlation_heatmap(df: pd.DataFrame, feature_cols: list[str]) -> str:
    """Plot a correlation heatmap of the features."""
    _ensure_images_dir()
    fig, ax = plt.subplots(figsize=(7, 6))
    corr = df[feature_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, square=True)
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    path = os.path.join(IMAGES_DIR, "correlation_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_box_plots(df: pd.DataFrame, feature_cols: list[str]) -> str:
    """Plot box plots for each feature grouped by species."""
    _ensure_images_dir()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, col in zip(axes.flatten(), feature_cols):
        sns.boxplot(data=df, x="species", y=col, hue="species", palette="viridis", ax=ax, legend=False)
        ax.set_title(f"{col.replace('_', ' ').title()} by Species")
    fig.suptitle("Box Plots", fontsize=16, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(IMAGES_DIR, "box_plots.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_scatter(df: pd.DataFrame) -> str:
    """Plot a scatter plot of petal length vs petal width."""
    _ensure_images_dir()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=df, x="petal_length", y="petal_width", hue="species", palette="viridis", ax=ax, s=70
    )
    ax.set_title("Petal Length vs Petal Width", fontsize=14, fontweight="bold")
    path = os.path.join(IMAGES_DIR, "scatter_plot.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_feature_distribution(df: pd.DataFrame, feature_cols: list[str]) -> str:
    """Plot KDE distributions of each feature grouped by species."""
    _ensure_images_dir()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, col in zip(axes.flatten(), feature_cols):
        sns.kdeplot(data=df, x=col, hue="species", fill=True, palette="viridis", ax=ax)
        ax.set_title(f"{col.replace('_', ' ').title()} Distribution")
    fig.suptitle("Feature Distribution by Species", fontsize=16, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(IMAGES_DIR, "feature_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_confusion_matrix(y_true, y_pred, labels: list[str], filename: str = "confusion_matrix.png") -> str:
    """Plot a confusion matrix heatmap."""
    _ensure_images_dir()
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    path = os.path.join(IMAGES_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_k_accuracy_curve(k_values: list[int], accuracies: list[float]) -> str:
    """Plot accuracy vs K for KNN hyperparameter tuning."""
    _ensure_images_dir()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, accuracies, marker="o", linewidth=2, color="#DD8452")
    best_idx = int(np.argmax(accuracies))
    ax.scatter(
        k_values[best_idx], accuracies[best_idx], color="red", zorder=5, s=100,
        label=f"Best K = {k_values[best_idx]}"
    )
    ax.set_xlabel("K Value")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs K (KNN Hyperparameter Tuning)", fontsize=14, fontweight="bold")
    ax.legend()
    path = os.path.join(IMAGES_DIR, "accuracy_vs_k.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_model_comparison(model_names: list[str], accuracies: list[float]) -> str:
    """Plot a bar chart comparing model accuracies."""
    _ensure_images_dir()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(model_names, accuracies, color=sns.color_palette("viridis", len(model_names)))
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{acc:.3f}", ha="center", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    plt.xticks(rotation=15)
    path = os.path.join(IMAGES_DIR, "model_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_all_eda_plots(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    """Generate the full EDA visualization suite and return saved paths."""
    paths = [
        plot_histograms(df, feature_cols),
        plot_count_plot(df),
        plot_pair_plot(df, feature_cols),
        plot_correlation_heatmap(df, feature_cols),
        plot_box_plots(df, feature_cols),
        plot_scatter(df),
        plot_feature_distribution(df, feature_cols),
    ]
    return paths
