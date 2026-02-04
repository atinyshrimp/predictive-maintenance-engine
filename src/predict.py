"""Prediction script for trained models."""

import logging
import logging.config
import argparse
import sys
from pathlib import Path

import pandas as pd
import joblib

from src.config import LOG_CONFIG, MODELS_DIR, ensure_directories
from src.data_loader import TurbofanDataLoader
from src.feature_engineering import FeatureEngineer, select_features_for_training
from src.reinforcement_learning import MaintenanceScheduler

# Configure logging
logging.config.dictConfig(LOG_CONFIG)
logger = logging.getLogger(__name__)


def predict_with_scheduler(
    model_path: Path,
    dataset_name: str = "FD001",
    use_rl_scheduler: bool = True,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Make predictions and generate maintenance schedule.

    Args:
        model_path: Path to trained model
        dataset_name: Dataset to predict on
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

    # Load and prepare data
    logger.info(f"\nLoading dataset: {dataset_name}...")
    data_loader = TurbofanDataLoader(dataset_name=dataset_name)
    train_df, test_df = data_loader.prepare_data(include_test_rul=True)

    # Feature engineering
    logger.info("\nEngineering features...")
    feature_engineer = FeatureEngineer()
    test_df = feature_engineer.engineer_all_features(test_df, include_rolling=True)

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
        default=str(MODELS_DIR / "best_pipeline.pkl"),
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
            use_rl_scheduler=not args.no_scheduler,
            output_path=output_path,
        )

    except Exception as e:
        logger.exception(f"Prediction pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
