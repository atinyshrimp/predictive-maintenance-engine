"""Evaluation and visualization utilities."""

import logging
from typing import List, Dict, Optional, Any
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
)

from src.config import REPORTS_DIR, ASSETS_DIR

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation and visualization."""

    def __init__(self, save_dir: Optional[Path] = None):
        """
        Initialize evaluator.

        Args:
            save_dir: Directory to save plots (defaults to assets)
        """
        self.save_dir = save_dir or ASSETS_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        save: bool = True,
    ) -> None:
        """
        Plot confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model for title
            save: Whether to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Failure", "Failure"],
            yticklabels=["No Failure", "Failure"],
        )
        plt.title(f"Confusion Matrix - {model_name}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

        if save:
            filepath = self.save_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            logger.info(f"Confusion matrix saved to {filepath}")

        plt.close()

    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_pred_probas: List[np.ndarray],
        model_names: List[str],
        save: bool = True,
    ) -> None:
        """
        Plot ROC curves for multiple models.

        Args:
            y_true: True labels
            y_pred_probas: List of predicted probabilities for each model
            model_names: List of model names
            save: Whether to save the plot
        """
        plt.figure(figsize=(10, 8))

        for y_pred_proba, model_name in zip(y_pred_probas, model_names):
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            plt.plot(
                fpr,
                tpr,
                label=f"{model_name} (AUC = {roc_auc:.3f})",
                linewidth=2,
            )

        plt.plot([0, 1], [0, 1], "k--", label="Random Classifier", linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("ROC Curves - Model Comparison", fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)

        if save:
            filepath = self.save_dir / "roc_curves_comparison.png"
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            logger.info(f"ROC curves saved to {filepath}")

        plt.close()

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = "Model",
        save: bool = True,
    ) -> None:
        """
        Plot precision-recall curve.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            save: Whether to save the plot
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)

        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, linewidth=2, label=model_name)
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.title(f"Precision-Recall Curve - {model_name}", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)

        if save:
            filepath = self.save_dir / f"precision_recall_{model_name.lower().replace(' ', '_')}.png"
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            logger.info(f"Precision-recall curve saved to {filepath}")

        plt.close()

    def plot_feature_importance(
        self,
        feature_names: List[str],
        importance_scores: np.ndarray,
        model_name: str = "Model",
        top_n: int = 20,
        save: bool = True,
    ) -> None:
        """
        Plot feature importance.

        Args:
            feature_names: List of feature names
            importance_scores: Feature importance scores
            model_name: Name of the model
            top_n: Number of top features to display
            save: Whether to save the plot
        """
        # Create DataFrame and sort
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importance_scores}
        ).sort_values("importance", ascending=False)

        # Select top N features
        top_features = importance_df.head(top_n)

        plt.figure(figsize=(10, 8))
        sns.barplot(data=top_features, x="importance", y="feature", palette="viridis")
        plt.title(f"Top {top_n} Feature Importances - {model_name}", fontsize=14)
        plt.xlabel("Importance Score", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.tight_layout()

        if save:
            filepath = self.save_dir / f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            logger.info(f"Feature importance plot saved to {filepath}")

        plt.close()

    def plot_rul_distribution(
        self, train_rul: pd.Series, test_rul: pd.Series, save: bool = True
    ) -> None:
        """
        Plot RUL distribution for train and test sets.

        Args:
            train_rul: Training RUL values
            test_rul: Test RUL values
            save: Whether to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Training RUL distribution
        axes[0].hist(train_rul, bins=50, alpha=0.7, color="blue", edgecolor="black")
        axes[0].set_title("RUL Distribution - Training Set", fontsize=12)
        axes[0].set_xlabel("Remaining Useful Life (cycles)", fontsize=10)
        axes[0].set_ylabel("Frequency", fontsize=10)
        axes[0].grid(alpha=0.3)

        # Test RUL distribution
        axes[1].hist(test_rul, bins=50, alpha=0.7, color="green", edgecolor="black")
        axes[1].set_title("RUL Distribution - Test Set", fontsize=12)
        axes[1].set_xlabel("Remaining Useful Life (cycles)", fontsize=10)
        axes[1].set_ylabel("Frequency", fontsize=10)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = self.save_dir / "rul_distribution.png"
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            logger.info(f"RUL distribution plot saved to {filepath}")

        plt.close()

    def generate_classification_report(
        self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model"
    ) -> str | dict[Any, Any]:
        """
        Generate detailed classification report.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model

        Returns:
            Classification report as dictionary
        """
        report = classification_report(
            y_true, y_pred, target_names=["No Failure", "Failure"], output_dict=True
        )

        logger.info(f"\nClassification Report - {model_name}")
        logger.info(classification_report(y_true, y_pred, target_names=["No Failure", "Failure"]))

        return report

    def compare_models(
        self, models_metrics: Dict[str, Dict[str, float]], save: bool = True
    ) -> None:
        """
        Compare multiple models with bar plots.

        Args:
            models_metrics: Dictionary mapping model names to their metrics
            save: Whether to save the plot
        """
        metrics_df = pd.DataFrame(models_metrics).T

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        metrics = ["accuracy", "precision", "recall", "f1_score"]

        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            if metric in metrics_df.columns:
                metrics_df[metric].plot(kind="bar", ax=ax, color="skyblue", edgecolor="black")
                ax.set_title(f"{metric.replace('_', ' ').title()} Comparison", fontsize=12)
                ax.set_ylabel("Score", fontsize=10)
                ax.set_xlabel("Model", fontsize=10)
                ax.set_ylim([0, 1])
                ax.grid(axis="y", alpha=0.3)
                ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        if save:
            filepath = self.save_dir / "model_comparison.png"
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            logger.info(f"Model comparison plot saved to {filepath}")

        plt.close()


def save_results_to_csv(
    results: pd.DataFrame, filename: str = "model_results.csv"
) -> None:
    """
    Save evaluation results to CSV.

    Args:
        results: DataFrame with results
        filename: Output filename
    """
    filepath = REPORTS_DIR / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(filepath, index=False)
    logger.info(f"Results saved to {filepath}")
