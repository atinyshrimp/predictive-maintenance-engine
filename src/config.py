"""Configuration management for the predictive maintenance engine."""

import os
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
ASSETS_DIR = PROJECT_ROOT / "assets"

# Data paths
RAW_DATA_DIR = DATA_DIR / "CMaps"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_PATH = DATA_DIR / "results.csv"

# Model configuration
MODEL_CONFIG = {
    "failure_threshold": 90,
    "test_size": 0.2,
    "validation_size": 0.2,
    "random_state": 42,
}

# Feature engineering
FEATURE_CONFIG = {
    "operational_settings": ["operational_setting_1", "operational_setting_2", "operational_setting_3"],
    "sensor_measurements": [f"sensor_measurement_{i}" for i in range(1, 27)],
    "window_sizes": [3, 5],  # For rolling statistics (matching notebook)
}

# Model hyperparameters
RANDOM_FOREST_PARAMS = {
    "n_estimators": 500,  # More trees for stability
    "max_depth": 30,  # Deeper trees
    "min_samples_split": 5,  # More aggressive splits
    "min_samples_leaf": 2,  # Smaller leaves to capture minority patterns
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}

# Reinforcement Learning configuration
RL_CONFIG = {
    "states": ["healthy", "moderate_wear", "severe_wear", "failed"],
    "actions": ["no_maintenance", "maintenance"],
    "learning_rate": 0.1,
    "discount_factor": 0.9,
    "epsilon": 0.2,
    "episodes": 1000,
}

# API configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "model_path": MODELS_DIR / "best_pipeline.pkl",
}

# Logging configuration
LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "INFO",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": PROJECT_ROOT / "logs" / "app.log",
            "formatter": "default",
            "level": "DEBUG",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"],
    },
}

RISK_LEVELS = {
    "LOW": {"color": "#28a745", "recommendation": "Continue normal operations."},
    "MEDIUM": {"color": "#ffc107", "recommendation": "Schedule maintenance inspection."},
    "HIGH": {"color": "#fd7e14", "recommendation": "Schedule immediate maintenance."},
    "CRITICAL": {"color": "#dc3545", "recommendation": "URGENT: Emergency maintenance required!"},
}

NOTEBOOK_BASELINE = {
    "recall": 0.9520464263897374,
    "precision": 0.9948470096735572,
    "f1_score": 0.9729762531629744,
    "roc_auc": 0.47297479116356866,
    "accuracy": 0.9520464263897374
}

def ensure_directories() -> None:
    """Create necessary directories if they don't exist."""
    directories = [
        DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        ASSETS_DIR,
        PROJECT_ROOT / "logs",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
