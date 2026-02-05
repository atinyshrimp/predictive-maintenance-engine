"""Utility functions for Streamlit app."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Default removed features for FD001
DEFAULT_REMOVED_FEATURES = [
    'operational_setting_3',
    'sensor_measurement_1',
    'sensor_measurement_5',
    'sensor_measurement_10',
    'sensor_measurement_16',
    'sensor_measurement_18',
    'sensor_measurement_19',
]


def load_model(model_name: str = "random_forest_(balanced).pkl"):
    """Load a trained model from the models directory."""
    model_path = MODELS_DIR / model_name
    if model_path.exists():
        return joblib.load(model_path)
    return None


def load_removed_features(dataset: str = "FD001", method: str = "cost_sensitive"):
    """Load removed features list or return defaults."""
    features_path = MODELS_DIR / f"removed_features_{dataset}_{method}.pkl"
    if features_path.exists():
        return joblib.load(features_path)
    return DEFAULT_REMOVED_FEATURES


def load_scaler():
    """Load the fitted scaler for feature normalization."""
    scaler_path = MODELS_DIR / "scaler.pkl"
    if scaler_path.exists():
        return joblib.load(scaler_path)
    return None


def load_results():
    """Load latest training results."""
    results_files = list(REPORTS_DIR.glob("results_*.csv"))
    if results_files:
        # Get most recent
        latest = max(results_files, key=lambda x: x.stat().st_mtime)
        return pd.read_csv(latest)
    return None


def get_risk_level(probability: float) -> tuple:
    """Get risk level and color based on failure probability."""
    if probability < 0.3:
        return "LOW", "#28a745", "Continue normal operations."
    elif probability < 0.5:
        return "MEDIUM", "#ffc107", "Schedule maintenance inspection."
    elif probability < 0.75:
        return "HIGH", "#fd7e14", "Schedule immediate maintenance."
    else:
        return "CRITICAL", "#dc3545", "URGENT: Emergency maintenance required!"


def create_gauge_chart(probability: float, title: str = "Failure Probability"):
    """Create a gauge chart for failure probability."""
    risk_level, color, _ = get_risk_level(probability)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        number={'suffix': '%', 'font': {'size': 40}},
        title={'text': title, 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#d4edda'},
                {'range': [30, 50], 'color': '#fff3cd'},
                {'range': [50, 75], 'color': '#ffe5d0'},
                {'range': [75, 100], 'color': '#f8d7da'},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 75
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    return fig


def load_sample_data():
    """Load sample sensor data from test set."""
    test_file = DATA_DIR / "CMaps" / "test_FD001.txt"
    if test_file.exists():
        # Match the exact column structure used in training
        columns = ["unit_number", "time_in_cycles"]
        columns.extend([f"operational_setting_{i}" for i in range(1, 4)])
        columns.extend([f"sensor_measurement_{i}" for i in range(1, 27)])  # 26 sensors to match training
        
        df = pd.read_csv(test_file, sep=r'\s+', header=None, names=columns)
        return df
    return None


def prepare_features_for_prediction(df: pd.DataFrame, removed_features: list):
    """
    Apply scaling, feature engineering, and remove low-variance features.
    
    CRITICAL: Must match training pipeline exactly:
    1. Scale raw sensor data FIRST
    2. Remove constant features
    3. Apply feature engineering on scaled data
    """
    from src.feature_engineering import FeatureEngineer
    from src.config import FEATURE_CONFIG
    
    df = df.copy()
    
    # Step 1: Scale raw features FIRST (same as training pipeline)
    scaler = load_scaler()
    if scaler is not None:
        columns_to_scale = (
            FEATURE_CONFIG["operational_settings"] + FEATURE_CONFIG["sensor_measurements"]
        )
        columns_to_scale = [col for col in columns_to_scale if col in df.columns]
        df[columns_to_scale] = scaler.transform(df[columns_to_scale])
    
    # Step 2: Remove constant features BEFORE feature engineering
    for col in removed_features:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Step 3: Apply feature engineering on scaled data
    fe = FeatureEngineer()
    df_engineered = fe.engineer_all_features(df)
    
    # Step 4: Remove non-feature columns
    X = df_engineered.drop(columns=["unit_number", "time_in_cycles", "RUL", "failure"], errors="ignore")
    
    return X


def load_rul_data():
    """Load actual RUL values for test set (ground truth for final cycle of each unit)."""
    rul_file = DATA_DIR / "CMaps" / "RUL_FD001.txt"
    if rul_file.exists():
        rul_df = pd.read_csv(rul_file, sep=r'\s+', header=None, names=["RUL"])
        return rul_df
    return None


def create_prediction_timeline_chart(cycles: list, probabilities: list, title: str = "Failure Risk Over Time"):
    """Create a line chart showing prediction evolution over time."""
    fig = go.Figure()
    
    # Add probability line
    fig.add_trace(go.Scatter(
        x=cycles,
        y=[p * 100 for p in probabilities],
        mode='lines+markers',
        name='Failure Probability',
        line=dict(color='#dc3545', width=3),
        marker=dict(size=8)
    ))
    
    # Add risk threshold lines
    fig.add_hline(y=30, line_dash="dash", line_color="#28a745", 
                  annotation_text="Low Risk", annotation_position="right")
    fig.add_hline(y=50, line_dash="dash", line_color="#ffc107",
                  annotation_text="Medium Risk", annotation_position="right")
    fig.add_hline(y=75, line_dash="dash", line_color="#fd7e14",
                  annotation_text="High Risk", annotation_position="right")
    
    fig.update_layout(
        title=title,
        xaxis_title="Time Cycle",
        yaxis_title="Failure Probability (%)",
        yaxis_range=[0, 100],
        height=400,
        showlegend=True
    )
    
    return fig


def get_delta_results() -> dict | None:
    """Load latest results and compute deltas from notebook baseline."""
    results = load_results()
    if results is not None:
        model_data = results.iloc[0]  # Single model
        notebook_baseline = {
            "recall": 0.9520464263897374,
            "precision": 0.9948470096735572,
            "f1_score": 0.9729762531629744,
            "roc_auc": 0.47297479116356866,
            "accuracy": 0.9520464263897374
        }
        deltas = {metric: model_data[metric] - notebook_baseline[metric] for metric in notebook_baseline}
        return deltas
    return None