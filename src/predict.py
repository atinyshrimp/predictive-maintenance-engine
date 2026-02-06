"""Prediction script for trained models."""

import logging
import logging.config
import argparse
import sys
from pathlib import Path

import pandas as pd
import joblib

from src.config import LOG_CONFIG, MODELS_DIR, FEATURE_CONFIG, ensure_directories
from src.data_loader import TurbofanDataLoader
from src.feature_engineering import FeatureEngineer, select_features_for_training
from src.reinforcement_learning import MaintenanceScheduler

# Configure logging
logging.config.dictConfig(LOG_CONFIG)
logger = logging.getLogger(__name__)


def load_preprocessing_artifacts(dataset_name: str = "FD001", imbalance_method: str = "cost_sensitive"):
    """
    Load scaler and removed features list.
    
    CRITICAL: These must be applied in the same order as training:
    1. Scale raw features
    2. Remove constant features
    3. Feature engineering
    """
    scaler_path = MODELS_DIR / "scaler.pkl"
    removed_features_path = MODELS_DIR / f"removed_features_{dataset_name}_{imbalance_method}.pkl"
    
    scaler = None
    removed_features = []
    
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        logger.info(f"Loaded scaler from {scaler_path}")
    else:
        logger.warning(f"Scaler not found at {scaler_path}. Predictions may be inaccurate!")
    
    if removed_features_path.exists():
        removed_features = joblib.load(removed_features_path)
        logger.info(f"Loaded {len(removed_features)} removed features from {removed_features_path}")
    else:
        logger.warning(f"Removed features list not found at {removed_features_path}")
    
    return scaler, removed_features


def preprocess_for_prediction(df: pd.DataFrame, scaler, removed_features: list) -> pd.DataFrame:
    """
    Apply preprocessing pipeline (must match training exactly).
    
    Order:
    1. Scale raw features FIRST
    2. Remove constant features
    3. Apply feature engineering
    """
    df = df.copy()
    
    # Step 1: Scale raw features FIRST (same as training pipeline)
    if scaler is not None:
        columns_to_scale = (
            FEATURE_CONFIG["operational_settings"] + FEATURE_CONFIG["sensor_measurements"]
        )
        columns_to_scale = [col for col in columns_to_scale if col in df.columns]
        df[columns_to_scale] = scaler.transform(df[columns_to_scale])
        logger.info(f"Scaled {len(columns_to_scale)} features")
    
    # Step 2: Remove constant features BEFORE feature engineering
    if removed_features:
        cols_to_drop = [col for col in removed_features if col in df.columns]
        df = df.drop(columns=cols_to_drop, errors="ignore")
        logger.info(f"Removed {len(cols_to_drop)} constant features")
    
    # Step 3: Apply feature engineering on scaled data
    feature_engineer = FeatureEngineer()
    df = feature_engineer.engineer_all_features(df)
    
    return df


def predict_with_scheduler(
    model_path: Path,
    dataset_name: str = "FD001",
    imbalance_method: str = "cost_sensitive",
    use_rl_scheduler: bool = True,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Make predictions and generate maintenance schedule.

    Args:
        model_path: Path to trained model
        dataset_name: Dataset to predict on
        imbalance_method: Imbalance method used during training (for loading correct artifacts)
        use_rl_scheduler: Whether to use RL-based scheduler
        output_path: Path to save predictions

    Returns:
        DataFrame with predictions and recommendations
    """
    logger.info("=" * 80)
    logger.info("PREDICTIVE MAINTENANCE PREDICTION PIPELINE")
    logger.info("=" * 80)

    ensure_directories()

    # Load model
    logger.info(f"\nLoading model from {model_path}...")
    model = joblib.load(model_path)
    
    # Load preprocessing artifacts (scaler, removed features)
    logger.info("\nLoading preprocessing artifacts...")
    scaler, removed_features = load_preprocessing_artifacts(dataset_name, imbalance_method)

    # Load RAW data (DO NOT use prepare_data() - it would fit a new scaler!)
    logger.info(f"\nLoading raw dataset: {dataset_name}...")
    data_loader = TurbofanDataLoader(dataset_name=dataset_name)
    train_df_raw, test_df_raw, rul_df = data_loader.load_raw_data()
    
    # Add RUL to test data
    test_df = test_df_raw.copy()
    test_df['RUL'] = 0
    for unit_id in test_df['unit_number'].unique():
        unit_mask = test_df['unit_number'] == unit_id
        unit_data = test_df[unit_mask]
        final_rul = rul_df.iloc[unit_id - 1]['RUL']
        max_cycle = unit_data['time_in_cycles'].max()
        test_df.loc[unit_mask, 'RUL'] = final_rul + (max_cycle - unit_data['time_in_cycles'])
    
    # Create failure labels
    test_df = data_loader.create_failure_labels(test_df)

    # Apply preprocessing pipeline (scale -> remove features -> engineer)
    # using the SAVED scaler from training (not a new one)
    logger.info("\nApplying preprocessing pipeline...")
    test_df = preprocess_for_prediction(test_df, scaler, removed_features)

    # Prepare features for prediction
    X_test = select_features_for_training(test_df)

    # Make predictions
    logger.info("\nMaking predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Compile results
    results_df = test_df[["unit_number", "time_in_cycles", "RUL"]].copy()
    results_df["failure_prediction"] = y_pred
    results_df["failure_probability"] = y_pred_proba

    # Generate maintenance schedule using RL if requested
    if use_rl_scheduler:
        logger.info("\nGenerating maintenance schedule with RL agent...")
        scheduler = MaintenanceScheduler()

        # Train scheduler on predictions
        scheduler.train(results_df, episodes=500)

        # Get maintenance recommendations
        schedule_df = scheduler.get_maintenance_schedule(results_df)

        # Merge with results
        results_df = pd.merge(
            results_df,
            schedule_df[["unit_number", "time_in_cycles", "state", "recommended_action"]],
            on=["unit_number", "time_in_cycles"],
            how="left",
        )

    # Save results
    if output_path is None:
        output_path = MODELS_DIR.parent / "data" / f"predictions_{dataset_name}.csv"

    results_df.to_csv(output_path, index=False)
    logger.info(f"\nPredictions saved to {output_path}")

    # Summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("PREDICTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total records: {len(results_df)}")
    logger.info(f"Predicted failures: {y_pred.sum()} ({y_pred.mean():.2%})")
    logger.info(f"Average failure probability: {y_pred_proba.mean():.4f}")

    if use_rl_scheduler:
        logger.info("\nMaintenance Recommendations:")
        action_counts = results_df["recommended_action"].value_counts()
        for action, count in action_counts.items():
            logger.info(f"  {action}: {count} ({count/len(results_df):.2%})")

    return results_df


def main():
    """Command-line interface for prediction."""
    parser = argparse.ArgumentParser(
        description="Make predictions with trained predictive maintenance model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(MODELS_DIR / "random_forest_(balanced).pkl"),
        help="Path to trained model file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="FD001",
        choices=["FD001", "FD002", "FD003", "FD004"],
        help="Dataset to make predictions on",
    )
    parser.add_argument(
        "--imbalance",
        type=str,
        default="cost_sensitive",
        choices=["none", "smote", "undersample", "cost_sensitive"],
        help="Imbalance method used during training (for loading correct preprocessing artifacts)",
    )
    parser.add_argument(
        "--no-scheduler",
        action="store_true",
        help="Don't use RL-based maintenance scheduler",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for predictions",
    )

    args = parser.parse_args()

    try:
        model_path = Path(args.model)
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            sys.exit(1)

        output_path = Path(args.output) if args.output else None

        predict_with_scheduler(
            model_path=model_path,
            dataset_name=args.dataset,
            imbalance_method=args.imbalance,
            use_rl_scheduler=not args.no_scheduler,
            output_path=output_path,
        )

    except Exception as e:
        logger.exception(f"Prediction pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
