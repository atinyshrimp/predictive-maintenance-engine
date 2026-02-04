"""Main training pipeline for predictive maintenance models."""

import logging
import logging.config
from pathlib import Path
import argparse
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve

from src.config import (
    LOG_CONFIG,
    MODEL_CONFIG,
    ensure_directories,
    MODELS_DIR,
    REPORTS_DIR,
)
from src.data_loader import TurbofanDataLoader, remove_constant_features
from src.feature_engineering import FeatureEngineer, select_features_for_training
from src.models import train_and_evaluate_models
from src.evaluation import ModelEvaluator, save_results_to_csv

# Configure logging
logging.config.dictConfig(LOG_CONFIG)
logger = logging.getLogger(__name__)


def optimize_threshold_for_recall(model, X_val, y_val, min_recall=0.55):
    """
    Find optimal classification threshold by maximizing F1-score while maintaining minimum recall.
    
    This is crucial for maintenance applications where missing failures is costly.
    
    Args:
        model: Trained model with predict_proba method
        X_val: Validation features
        y_val: Validation labels
        min_recall: Minimum acceptable recall (default 0.55 = more lenient for optimization)
        
    Returns:
        Optimal threshold value
    """
    y_proba = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, y_proba)
    
    # precision_recall_curve returns n+1 values, so we need to trim
    thresholds = np.append(thresholds, 1.0)
    
    # Filter for thresholds meeting minimum recall requirement
    valid_mask = recall >= min_recall
    
    if not valid_mask.any():
        logger.warning(f"Could not find threshold with recall >= {min_recall}, using default 0.5")
        return 0.5
    
    # Calculate F1 scores for valid thresholds
    valid_precision = precision[valid_mask]
    valid_recall = recall[valid_mask]
    valid_thresholds = thresholds[valid_mask]
    
    f1_scores = 2 * (valid_precision * valid_recall) / (valid_precision + valid_recall + 1e-10)
    
    # Find threshold that maximizes F1
    best_idx = np.argmax(f1_scores)
    best_threshold = valid_thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    best_recall = valid_recall[best_idx]
    best_precision = valid_precision[best_idx]
    
    logger.info(
        f"Optimal threshold: {best_threshold:.3f} "
        f"(Precision={best_precision:.3f}, Recall={best_recall:.3f}, F1={best_f1:.3f})"
    )
    
    return float(best_threshold)


def train_pipeline(
    dataset_name: str = "FD001",
    imbalance_method: str = "cost_sensitive",
    save_models: bool = True,
) -> None:
    """
    Complete training pipeline for predictive maintenance models.

    Args:
        dataset_name: Name of the dataset to use (FD001, FD002, FD003, FD004)
        imbalance_method: Method to handle class imbalance
        save_models: Whether to save trained models
    """
    logger.info("=" * 80)
    logger.info("PREDICTIVE MAINTENANCE TRAINING PIPELINE")
    logger.info("=" * 80)

    # Ensure directories exist
    ensure_directories()

    # Step 1: Load and preprocess data
    logger.info("\n[Step 1/6] Loading and preprocessing data...")
    data_loader = TurbofanDataLoader(dataset_name=dataset_name)
    train_df, test_df = data_loader.prepare_data(include_test_rul=True)

    # Step 2: Remove constant features
    logger.info("\n[Step 2/6] Removing constant features...")
    train_df, test_df, removed_cols = remove_constant_features(train_df, test_df)
    logger.info(f"Removed {len(removed_cols)} constant features")

    # Step 3: Feature engineering
    logger.info("\n[Step 3/6] Engineering features...")
    feature_engineer = FeatureEngineer()
    train_df = feature_engineer.engineer_all_features(train_df, include_rolling=True)
    test_df = feature_engineer.engineer_all_features(test_df, include_rolling=True)

    # Step 4: Prepare training data
    logger.info("\n[Step 4/6] Preparing training data...")
    X_train_full = select_features_for_training(train_df)
    y_train_full = train_df["failure"]

    X_test = select_features_for_training(test_df)
    y_test = test_df["failure"]
    y_test_array = y_test.to_numpy()

    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=MODEL_CONFIG["validation_size"],
        random_state=MODEL_CONFIG["random_state"],
        stratify=y_train_full,
    )

    logger.info(f"Training set: {X_train.shape}")
    logger.info(f"Validation set: {X_val.shape}")
    logger.info(f"Test set: {X_test.shape}")
    logger.info(f"Class distribution (train): {y_train.value_counts().to_dict()}")

    # Step 5: Train models
    logger.info(f"\n[Step 5/6] Training models with '{imbalance_method}' imbalance handling...")
    models = train_and_evaluate_models(
        X_train, y_train, X_val, y_val, handle_imbalance=imbalance_method
    )

    # Optimize thresholds on validation set
    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZING DECISION THRESHOLDS")
    logger.info("=" * 80)
    
    thresholds = {}
    for model_name, model in models.items():
        logger.info(f"\nOptimizing threshold for {model_name}...")
        threshold = optimize_threshold_for_recall(model, X_val, y_val, min_recall=0.60)
        thresholds[model_name] = threshold

    # Evaluate on test set with optimized thresholds
    logger.info("\n" + "=" * 80)
    logger.info("FINAL EVALUATION ON TEST SET (WITH OPTIMIZED THRESHOLDS)")
    logger.info("=" * 80)

    results = []
    evaluator = ModelEvaluator()

    for model_name, model in models.items():
        logger.info(f"\nEvaluating {model.model_name}...")
        
        # Get probability predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Apply optimized threshold
        threshold = thresholds[model_name]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics with optimized threshold
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        test_metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "threshold": threshold,
        }
        
        logger.info(f"Metrics with threshold={threshold:.3f}:")
        for metric, value in test_metrics.items():
            if metric != "threshold":
                logger.info(f"  {metric}: {value:.4f}")

        results.append(
            {
                "model": model.model_name,
                "dataset": dataset_name,
                "imbalance_method": imbalance_method,
                **test_metrics,
            }
        )

        # Generate visualizations with optimized predictions
        evaluator.plot_confusion_matrix(y_test_array, y_pred, model.model_name)
        evaluator.plot_precision_recall_curve(y_test_array, y_pred_proba, model.model_name)

        # Plot feature importance if available
        if model.model is not None and hasattr(model.model, "feature_importances_"):
            evaluator.plot_feature_importance(
                X_train.columns.tolist(),
                model.model.feature_importances_,
                model.model_name,
            )

        # Save model
        if save_models:
            model.save_model()
    
    # Save optimized thresholds
    if save_models:
        import joblib
        thresholds_path = MODELS_DIR / f"thresholds_{dataset_name}_{imbalance_method}.pkl"
        joblib.dump(thresholds, thresholds_path)
        logger.info(f"\nOptimized thresholds saved to {thresholds_path}")

    # Step 6: Compare models and save results
    logger.info("\n[Step 6/6] Comparing models and saving results...")
    results_df = pd.DataFrame(results)
    save_results_to_csv(results_df, f"results_{dataset_name}_{imbalance_method}.csv")

    # Compare models
    models_metrics = {row["model"]: row.to_dict() for _, row in results_df.iterrows()}
    evaluator.compare_models(models_metrics)

    # Compare ROC curves
    y_pred_probas = [model.predict_proba(X_test)[:, 1] for model in models.values()]
    model_names = [model.model_name for model in models.values()]
    evaluator.plot_roc_curves(y_test_array, y_pred_probas, model_names)

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nResults saved to: {REPORTS_DIR}")
    logger.info(f"Models saved to: {MODELS_DIR}")
    logger.info("\nTop Model Performance:")
    logger.info(results_df.sort_values("f1_score", ascending=False).to_string(index=False))


def main():
    """Command-line interface for training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train predictive maintenance models on NASA Turbofan dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="FD001",
        choices=["FD001", "FD002", "FD003", "FD004"],
        help="Dataset to use for training",
    )
    parser.add_argument(
        "--imbalance",
        type=str,
        default="cost_sensitive",
        choices=["none", "smote", "undersample", "cost_sensitive"],
        help="Method to handle class imbalance",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save trained models",
    )

    args = parser.parse_args()

    try:
        train_pipeline(
            dataset_name=args.dataset,
            imbalance_method=args.imbalance,
            save_models=not args.no_save,
        )
    except Exception as e:
        logger.exception(f"Training pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
