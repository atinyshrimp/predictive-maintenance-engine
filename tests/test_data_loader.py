"""Unit tests for data loading module."""

import pytest
import pandas as pd
import numpy as np
from src.data_loader import TurbofanDataLoader, remove_constant_features


class TestTurbofanDataLoader:
    """Test suite for TurbofanDataLoader class."""

    def test_generate_column_headers(self):
        """Test column header generation."""
        loader = TurbofanDataLoader()
        headers = loader._generate_column_headers()

        # 31 columns: 1 unit_number + 1 time_in_cycles + 3 operational_settings + 26 sensor_measurements
        assert len(headers) == 31
        assert "unit_number" in headers
        assert "time_in_cycles" in headers
        assert "operational_setting_1" in headers
        assert "sensor_measurement_1" in headers

    def test_compute_rul(self):
        """Test RUL computation."""
        loader = TurbofanDataLoader()

        # Create sample data
        df = pd.DataFrame({
            "unit_number": [1, 1, 1, 2, 2],
            "time_in_cycles": [1, 2, 3, 1, 2],
        })

        result = loader.compute_rul(df)

        assert "RUL" in result.columns
        assert result.loc[0, "RUL"] == 2  # Max 3 - current 1
        assert result.loc[2, "RUL"] == 0  # Max 3 - current 3

    def test_create_failure_labels(self):
        """Test failure label creation."""
        loader = TurbofanDataLoader()

        df = pd.DataFrame({
            "RUL": [100, 50, 30, 10]
        })

        result = loader.create_failure_labels(df, threshold=50)

        assert "failure" in result.columns
        assert result["failure"].dtype == np.int64
        assert result.loc[0, "failure"] == 0  # RUL=100 > 50
        assert result.loc[3, "failure"] == 1  # RUL=10 <= 50


class TestFeatureRemoval:
    """Test suite for feature removal functions."""

    def test_remove_constant_features(self):
        """Test removal of constant features."""
        train_df = pd.DataFrame({
            "unit_number": [1, 2, 3],
            "const_feature": [1.0, 1.0, 1.0],
            "variable_feature": [1.0, 2.0, 3.0],
            "RUL": [100, 50, 25],
        })

        test_df = train_df.copy()

        train_filtered, test_filtered, removed = remove_constant_features(
            train_df, test_df, threshold=0.01
        )

        assert "const_feature" in removed
        assert "variable_feature" not in removed
        assert "RUL" in train_filtered.columns


if __name__ == "__main__":
    pytest.main([__file__])
