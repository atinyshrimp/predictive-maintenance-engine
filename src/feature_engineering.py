"""Feature engineering for predictive maintenance."""

import logging
from typing import List, Optional

import pandas as pd

from src.config import FEATURE_CONFIG

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Engineer advanced features from sensor data."""

    def __init__(self, window_sizes: Optional[List[int]] = None):
        """
        Initialize feature engineer.

        Args:
            window_sizes: List of window sizes for rolling statistics
        """
        self.window_sizes = window_sizes or FEATURE_CONFIG["window_sizes"]

    def create_rolling_features(
        self, df: pd.DataFrame, group_col: str = "unit_number"
    ) -> pd.DataFrame:
        """
        Create rolling mean features

        Args:
            df: Input DataFrame
            group_col: Column to group by (typically unit_number)

        Returns:
            DataFrame with rolling features added
        """
        logger.info("Creating rolling mean features")
        df = df.copy()

        # Get columns to scale (operational settings + sensor measurements)
        columns_to_scale = []
        for i in range(1, 4):
            columns_to_scale.append(f"operational_setting_{i}")
        for i in range(1, 27):
            columns_to_scale.append(f"sensor_measurement_{i}")
        
        columns_to_scale = [col for col in columns_to_scale if col in df.columns]

        for window in self.window_sizes:
            for col in columns_to_scale:
                # Rolling mean only
                df[f"{col}_rolling_mean_{window}"] = (
                    df.groupby(group_col)[col]
                    .rolling(window=window)
                    .mean()
                    .reset_index(0, drop=True)
                )

        # Fill NaN values with 0 (as in notebook)
        df.fillna(0, inplace=True)

        logger.info(f"Created rolling mean features for {len(columns_to_scale)} columns "
                   f"with window sizes {self.window_sizes}")
        return df

    def engineer_all_features(
        self,
        df: pd.DataFrame,
        include_rolling: bool = True,
    ) -> pd.DataFrame:
        """
        Apply feature engineering

        Args:
            df: Input DataFrame
            include_rolling: Whether to include rolling features

        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering (rolling statistics)")

        if include_rolling:
            df = self.create_rolling_features(df)

        logger.info(f"Feature engineering complete. Final shape: {df.shape}")
        return df


def select_features_for_training(
    df: pd.DataFrame, exclude_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Select relevant features for model training.

    Args:
        df: Input DataFrame
        exclude_cols: Columns to exclude from features

    Returns:
        DataFrame with only training features
    """
    if exclude_cols is None:
        exclude_cols = ["unit_number", "time_in_cycles", "RUL", "failure"]

    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return df[feature_cols]
