"""Machine learning models for predictive maintenance."""

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import joblib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import Pipeline

from src.config import XGBOOST_PARAMS, RANDOM_FOREST_PARAMS, MODELS_DIR

logger = logging.getLogger(__name__)


class MaintenanceClassifier:
    """Base class for predictive maintenance classifiers."""

    def __init__(self, model_name: str):
        """
        Initialize the classifier.

        Args:
            model_name: Name of the model for logging and saving
        """
        self.model_name = model_name
        self.model = None
        self.metrics = {}

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info(f"Training {self.model_name}")
        
        if self.model is None:
            raise ValueError("Model is not initialized")
        
        self.model.fit(X_train, y_train)
        logger.info(f"{self.model_name} training complete")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features for prediction

        Returns:
            Predicted labels
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features for prediction

        Returns:
            Predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)

    def evaluate(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: True labels

        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]

        self.metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
        }

        logger.info(f"{self.model_name} Evaluation Metrics:")
        for metric, value in self.metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        return self.metrics

    def save_model(self, filepath: Optional[Path] = None) -> None:
        """
        Save the trained model.

        Args:
            filepath: Path to save the model (defaults to models directory)
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        if filepath is None:
            filepath = MODELS_DIR / f"{self.model_name.lower().replace(' ', '_')}.pkl"

        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: Path) -> "MaintenanceClassifier":
        """
        Load a trained model.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded classifier instance
        """
        instance = cls(model_name=filepath.stem)
        instance.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return instance


class XGBoostMaintenanceClassifier(MaintenanceClassifier):
    """XGBoost-based classifier for predictive maintenance."""

    def __init__(
        self,
        use_cost_sensitive: bool = False,
        scale_pos_weight: Optional[float] = None,
        **kwargs,
    ):
        """
        Initialize XGBoost classifier.

        Args:
            use_cost_sensitive: Whether to use cost-sensitive learning
            scale_pos_weight: Weight for positive class (auto-calculated if None)
            **kwargs: Additional XGBoost parameters
        """
        super().__init__(model_name="XGBoost")
        params = {**XGBOOST_PARAMS, **kwargs}

        if use_cost_sensitive and scale_pos_weight is not None:
            params["scale_pos_weight"] = scale_pos_weight
            self.model_name = "XGBoost (Cost-Sensitive)"

        self.model = XGBClassifier(**params)


class RandomForestMaintenanceClassifier(MaintenanceClassifier):
    """Random Forest-based classifier for predictive maintenance."""

    def __init__(self, use_balanced: bool = False, **kwargs):
        """
        Initialize Random Forest classifier.

        Args:
            use_balanced: Whether to use balanced class weights
            **kwargs: Additional Random Forest parameters
        """
        super().__init__(model_name="Random Forest")
        params = {**RANDOM_FOREST_PARAMS, **kwargs}

        if use_balanced:
            params["class_weight"] = "balanced"
            self.model_name = "Random Forest (Balanced)"

        self.model = RandomForestClassifier(**params)


class ImbalanceHandler:
    """Handle class imbalance using various techniques."""

    @staticmethod
    def apply_smote(
        X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE (Synthetic Minority Over-sampling Technique).

        Args:
            X_train: Training features
            y_train: Training labels
            random_state: Random state for reproducibility

        Returns:
            Tuple of (resampled_X, resampled_y)
        """
        logger.info("Applying SMOTE for handling class imbalance")
        smote = SMOTE(random_state=random_state)
        result = smote.fit_resample(X_train, y_train)
        X_resampled, y_resampled = result[0], result[1]

        logger.info(
            f"SMOTE: Original shape {X_train.shape}, "
            f"Resampled shape {X_resampled.shape}"
        )
        return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled, name=y_train.name)

    @staticmethod
    def apply_random_undersampling(
        X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply Random Under-Sampling.

        Args:
            X_train: Training features
            y_train: Training labels
            random_state: Random state for reproducibility

        Returns:
            Tuple of (resampled_X, resampled_y)
        """
        logger.info("Applying Random Under-Sampling for handling class imbalance")
        rus = RandomUnderSampler(random_state=random_state)
        result = rus.fit_resample(X_train, y_train)
        X_resampled, y_resampled = result[0], result[1]

        logger.info(
            f"RUS: Original shape {X_train.shape}, "
            f"Resampled shape {X_resampled.shape}"
        )
        return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled, name=y_train.name)
    
    @staticmethod
    def calculate_scale_pos_weight(y_train: pd.Series) -> float:
        """
        Calculate scale_pos_weight for XGBoost cost-sensitive learning.

        Args:
            y_train: Training labels

        Returns:
            Calculated weight for positive class
        """
        n_negative = (y_train == 0).sum()
        n_positive = (y_train == 1).sum()
        weight = n_negative / n_positive

        logger.info(
            f"Calculated scale_pos_weight: {weight:.2f} "
            f"(negative: {n_negative}, positive: {n_positive})"
        )
        return weight


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    handle_imbalance: str = "none",
) -> Dict[str, MaintenanceClassifier]:
    """
    Train and evaluate multiple models.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        handle_imbalance: Method to handle imbalance ('none', 'smote', 'undersample', 'cost_sensitive')

    Returns:
        Dictionary of trained models
    """
    logger.info(f"Training models with imbalance handling: {handle_imbalance}")

    # Handle class imbalance if requested
    X_train_processed = X_train.copy()
    y_train_processed = y_train.copy()

    if handle_imbalance == "smote":
        X_train_processed, y_train_processed = ImbalanceHandler.apply_smote(
            X_train, y_train
        )
    elif handle_imbalance == "undersample":
        X_train_processed, y_train_processed = (
            ImbalanceHandler.apply_random_undersampling(X_train, y_train)
        )

    # Initialize models
    models = {}

    # Random Forest
    rf_model = RandomForestMaintenanceClassifier(
        use_balanced=(handle_imbalance == "cost_sensitive")
    )
    rf_model.train(X_train_processed, y_train_processed)
    rf_model.evaluate(X_val, y_val)
    models["random_forest"] = rf_model

    # XGBoost
    if handle_imbalance == "cost_sensitive":
        scale_pos_weight = ImbalanceHandler.calculate_scale_pos_weight(y_train)
        xgb_model = XGBoostMaintenanceClassifier(
            use_cost_sensitive=True, scale_pos_weight=scale_pos_weight
        )
    else:
        xgb_model = XGBoostMaintenanceClassifier()

    xgb_model.train(X_train_processed, y_train_processed)
    xgb_model.evaluate(X_val, y_val)
    models["xgboost"] = xgb_model

    return models
