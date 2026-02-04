"""Data loading and preprocessing utilities for turbofan engine dataset."""

import logging
from typing import Tuple, List, Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.config import RAW_DATA_DIR, MODEL_CONFIG, FEATURE_CONFIG

logger = logging.getLogger(__name__)


class TurbofanDataLoader:
    """Load and preprocess NASA Turbofan Engine dataset."""

    def __init__(self, dataset_name: str = "FD001"):
        """
        Initialize the data loader.

        Args:
            dataset_name: Name of the dataset (FD001, FD002, FD003, or FD004)
        """
        self.dataset_name = dataset_name
        self.headers = self._generate_column_headers()
        self.scaler = MinMaxScaler()

    @staticmethod
    def _generate_column_headers() -> List[str]:
        """Generate column headers for the dataset."""
        headers = ["unit_number", "time_in_cycles"]
        headers.extend([f"operational_setting_{i}" for i in range(1, 4)])
        headers.extend([f"sensor_measurement_{i}" for i in range(1, 27)])
        return headers

    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load raw training, test, and RUL data.

        Returns:
            Tuple of (train_df, test_df, rul_df)
        """
        logger.info(f"Loading dataset: {self.dataset_name}")

        train_path = RAW_DATA_DIR / f"train_{self.dataset_name}.txt"
        test_path = RAW_DATA_DIR / f"test_{self.dataset_name}.txt"
        rul_path = RAW_DATA_DIR / f"RUL_{self.dataset_name}.txt"

        try:
            train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=self.headers)
            test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=self.headers)
            rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["RUL"])

            logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
            return train_df, test_df, rul_df

        except FileNotFoundError as e:
            logger.error(f"Data file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def compute_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Remaining Useful Life (RUL) for each record.

        Args:
            df: DataFrame with unit_number and time_in_cycles

        Returns:
            DataFrame with RUL column added
        """
        df = df.copy()
        max_cycles = df.groupby("unit_number")["time_in_cycles"].transform("max")
        df["RUL"] = max_cycles - df["time_in_cycles"]
        logger.info("Computed RUL for all records")
        return df

    def create_failure_labels(
        self, df: pd.DataFrame, threshold: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Create binary failure labels based on RUL threshold.

        Args:
            df: DataFrame with RUL column
            threshold: RUL threshold for failure (defaults to config value)

        Returns:
            DataFrame with failure column added
        """
        if threshold is None:
            threshold = MODEL_CONFIG["failure_threshold"]

        df = df.copy()
        df["failure"] = (df["RUL"] <= threshold).astype(int)

        failure_rate = df["failure"].mean()
        logger.info(
            f"Created failure labels with threshold={threshold}. "
            f"Failure rate: {failure_rate:.2%}"
        )
        return df

    def scale_features(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale numerical features using MinMaxScaler.

        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame

        Returns:
            Tuple of (scaled_train_df, scaled_test_df)
        """
        train_scaled = train_df.copy()
        test_scaled = test_df.copy()

        # Identify columns to scale (exclude identifiers and target)
        columns_to_scale = (
            FEATURE_CONFIG["operational_settings"] + FEATURE_CONFIG["sensor_measurements"]
        )
        columns_to_scale = [col for col in columns_to_scale if col in train_df.columns]

        # Fit scaler on training data and transform both sets
        train_scaled[columns_to_scale] = self.scaler.fit_transform(train_df[columns_to_scale])
        test_scaled[columns_to_scale] = self.scaler.transform(test_df[columns_to_scale])

        logger.info(f"Scaled {len(columns_to_scale)} features")
        return train_scaled, test_scaled

    def prepare_data(
        self, include_test_rul: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete data preparation pipeline.

        Args:
            include_test_rul: Whether to include RUL in test data

        Returns:
            Tuple of (prepared_train_df, prepared_test_df)
        """
        logger.info("Starting data preparation pipeline")

        # Load raw data
        train_df, test_df, rul_df = self.load_raw_data()

        # Add RUL to test data if requested
        if include_test_rul:
            # Assign final RUL to last cycle of each engine
            test_engines = test_df.groupby('unit_number')
            test_df['RUL'] = 0  # Initialize
            
            for unit_id in test_df['unit_number'].unique():
                unit_mask = test_df['unit_number'] == unit_id
                unit_data = test_df[unit_mask]
                
                # Get final RUL from rul_df
                final_rul = rul_df.iloc[unit_id - 1]['RUL']
                
                # Compute RUL for each cycle: final_rul + (max_cycle - current_cycle)
                max_cycle = unit_data['time_in_cycles'].max()
                test_df.loc[unit_mask, 'RUL'] = final_rul + (max_cycle - unit_data['time_in_cycles'])

        # Compute RUL for training data
        train_df = self.compute_rul(train_df)

        # Compute RUL for training data
        train_df = self.compute_rul(train_df)

        # Create failure labels
        train_df = self.create_failure_labels(train_df)
        if include_test_rul:
            test_df = self.create_failure_labels(test_df)

        # Scale features
        train_df, test_df = self.scale_features(train_df, test_df)

        logger.info("Data preparation complete")
        return train_df, test_df


def remove_constant_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame, threshold: float = 0.01
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Remove features with near-zero variance.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        threshold: Variance threshold below which features are removed

    Returns:
        Tuple of (filtered_train_df, filtered_test_df, removed_columns)
    """
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    variances = train_df[numeric_cols].var()

    low_variance_cols = variances[variances < threshold].index.tolist()

    # Don't remove target columns or identifiers
    cols_to_keep = ["unit_number", "time_in_cycles", "RUL", "failure"]
    low_variance_cols = [col for col in low_variance_cols if col not in cols_to_keep]

    if low_variance_cols:
        logger.info(f"Removing {len(low_variance_cols)} low-variance features")
        train_df = train_df.drop(columns=low_variance_cols)
        test_df = test_df.drop(columns=low_variance_cols)

    return train_df, test_df, low_variance_cols
