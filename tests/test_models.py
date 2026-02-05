"""Unit tests for models module."""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from src.models import (
    RandomForestMaintenanceClassifier,
    ImbalanceHandler,
)


class TestClassifiers:
    """Test suite for classifier classes."""

    @pytest.fixture
    def sample_data(self):
        """Create sample classification data."""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=42,
        )
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
        y_series = pd.Series(y)

        # Split data
        split_idx = 80
        X_train = X_df[:split_idx]
        y_train = y_series[:split_idx]
        X_test = X_df[split_idx:]
        y_test = y_series[split_idx:]

        return X_train, y_train, X_test, y_test

    def test_random_forest_classifier_training(self, sample_data):
        """Test Random Forest classifier training."""
        X_train, y_train, X_test, y_test = sample_data

        model = RandomForestMaintenanceClassifier()
        model.train(X_train, y_train)

        assert model.model is not None
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)

    def test_predict_proba(self, sample_data):
        """Test probability predictions."""
        X_train, y_train, X_test, y_test = sample_data

        model = RandomForestMaintenanceClassifier()
        model.train(X_train, y_train)

        probas = model.predict_proba(X_test)
        assert probas.shape == (len(y_test), 2)
        assert np.all((probas >= 0) & (probas <= 1))
        assert np.allclose(probas.sum(axis=1), 1.0)

    def test_evaluate(self, sample_data):
        """Test model evaluation."""
        X_train, y_train, X_test, y_test = sample_data

        model = RandomForestMaintenanceClassifier()
        model.train(X_train, y_train)

        metrics = model.evaluate(X_test, y_test)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics


class TestImbalanceHandler:
    """Test suite for ImbalanceHandler class."""

    @pytest.fixture
    def imbalanced_data(self):
        """Create imbalanced dataset."""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
            weights=[0.9, 0.1],  # Imbalanced
            random_state=42,
        )
        X_df = pd.DataFrame(X)
        y_series = pd.Series(y)
        return X_df, y_series

    def test_smote_application(self, imbalanced_data):
        """Test SMOTE application."""
        X, y = imbalanced_data

        X_resampled, y_resampled = ImbalanceHandler.apply_smote(X, y)

        assert len(X_resampled) >= len(X)
        # Should have balanced classes
        assert y_resampled.value_counts()[0] == y_resampled.value_counts()[1]

    def test_random_undersampling(self, imbalanced_data):
        """Test random undersampling."""
        X, y = imbalanced_data

        X_resampled, y_resampled = ImbalanceHandler.apply_random_undersampling(X, y)

        assert len(X_resampled) <= len(X)
        # Should have balanced classes
        assert y_resampled.value_counts()[0] == y_resampled.value_counts()[1]


if __name__ == "__main__":
    pytest.main([__file__])
