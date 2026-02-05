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
                # Rolling mean
                df[f"{col}_rolling_mean_{window}"] = (
                    df.groupby(group_col)[col]
                    .rolling(window=window)
                    .mean()
                    .reset_index(0, drop=True)
                )
                
                # Rolling std (captures volatility/instability near failure)
                df[f"{col}_rolling_std_{window}"] = (
                    df.groupby(group_col)[col]
                    .rolling(window=window)
                    .std()
                    .reset_index(0, drop=True)
                )
        
        # Exponential moving average (better for trends than simple rolling mean)
        for col in columns_to_scale:
            df[f"{col}_ema"] = (
                df.groupby(group_col)[col]
                .transform(lambda x: x.ewm(span=5, adjust=False).mean())
            )

        # Fill NaN values with 0 (as in notebook)
        df.fillna(0, inplace=True)

        logger.info(f"Created rolling features (mean, std, ema) for {len(columns_to_scale)} columns "
                   f"with window sizes {self.window_sizes}")
        return df
    
    def create_degradation_features(
        self, df: pd.DataFrame, group_col: str = "unit_number"
    ) -> pd.DataFrame:
        """
        Create features that capture degradation patterns
        
        Args:
            df: Input DataFrame
            group_col: Column to group by
            
        Returns:
            DataFrame with degradation features
        """
        logger.info("Creating degradation features")
        df = df.copy()
        
        # Cycle position (0 to 1, captures proximity to end-of-life)
        df['cycle_norm'] = (
            df.groupby(group_col)['time_in_cycles']
            .transform(lambda x: x / x.max() if x.max() > 0 else 0)
        )
        
        # Rate of change for key sensors (captures deterioration velocity)
        sensor_cols = [f"sensor_measurement_{i}" for i in [4, 7, 11, 12, 15, 17, 20, 21]]
        sensor_cols = [col for col in sensor_cols if col in df.columns]
        
        for col in sensor_cols:
            df[f"{col}_rate_of_change"] = (
                df.groupby(group_col)[col]
                .diff()
                .fillna(0)
            )
        
        # Cumulative sum (total degradation accumulation)
        for col in sensor_cols:
            df[f"{col}_cumsum"] = (
                df.groupby(group_col)[col]
                .cumsum()
            )
        
        logger.info(f"Created degradation features: cycle_norm, rate_of_change, cumsum")
        return df

    def engineer_all_features(
        self,
        df: pd.DataFrame,
        include_rolling: bool = True,
        include_degradation: bool = True,
    ) -> pd.DataFrame:
        """
        Apply feature engineering

        Args:
            df: Input DataFrame
            include_rolling: Whether to include rolling features
            include_degradation: Whether to include degradation features

        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering (rolling + degradation)")

        if include_rolling:
            df = self.create_rolling_features(df)
        
        if include_degradation:
            df = self.create_degradation_features(df)

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
