"""FastAPI application for predictive maintenance inference."""

import logging
from pathlib import Path
from typing import List, Dict

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import joblib
import numpy as np
import pandas as pd
import uvicorn

from src.feature_engineering import FeatureEngineer, select_features_for_training

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load model at startup
MODEL_PATH = Path(__file__).parent.parent / "models" / "random_forest_(balanced).pkl"
REMOVED_FEATURES_PATH = Path(__file__).parent.parent / "models" / "removed_features_FD001_cost_sensitive.pkl"
SCALER_PATH = Path(__file__).parent.parent / "models" / "scaler.pkl"


class SensorData(BaseModel):
    """Request model for sensor data with time-series history."""

    unit_id: int = Field(..., description="Unique identifier for the equipment unit", ge=1)
    time_steps: List[List[float]] = Field(
        ...,
        description="Time-series sensor data: list of snapshots, each with 29 values (3 operational + 26 sensors). Minimum 5 time steps for rolling features.",
        min_length=5,
    )

    @field_validator("time_steps")
    def validate_time_steps(cls, v):
        """Validate time steps data."""
        if len(v) < 5:
            raise ValueError("At least 5 time steps required for rolling feature computation")
        
        for i, step in enumerate(v):
            if len(step) != 29:
                raise ValueError(f"Time step {i} must have exactly 29 values (3 operational settings + 26 sensors)")
            if any(np.isnan(val) or np.isinf(val) for val in step):
                raise ValueError(f"Time step {i} contains invalid values (NaN or infinite)")
        return v


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    unit_id: int = Field(..., description="Equipment unit identifier")
    failure_probability: float = Field(
        ..., description="Probability of failure (0-1)", ge=0.0, le=1.0
    )
    failure_prediction: bool = Field(
        ..., description="Binary prediction (True if failure predicted)"
    )
    risk_level: str = Field(..., description="Risk level: LOW, MEDIUM, HIGH, CRITICAL")
    recommendation: str = Field(..., description="Maintenance recommendation")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    model_loaded: bool
    model_path: str


# Global model variable
model = None
removed_features = []
scaler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    global model, removed_features, scaler
    try:
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
            logger.info(f"Model loaded successfully from {MODEL_PATH}")
        else:
            logger.error(f"Model file not found at {MODEL_PATH}")
            model = None
        
        # Load scaler (CRITICAL: must scale before feature engineering)
        if SCALER_PATH.exists():
            scaler = joblib.load(SCALER_PATH)
            logger.info(f"Scaler loaded successfully from {SCALER_PATH}")
        else:
            logger.warning(f"Scaler not found at {SCALER_PATH}. Predictions may be inaccurate!")
            scaler = None
            
        # Load removed features list
        if REMOVED_FEATURES_PATH.exists():
            removed_features = joblib.load(REMOVED_FEATURES_PATH)
            logger.info(f"Removed features list loaded: {len(removed_features)} features to drop")
        elif model is not None and hasattr(model, "feature_names_in_"):
            # Derive removed features by comparing model's expected features with full feature set
            model_features = set(model.feature_names_in_)
            
            # Build reference feature set (same columns as feature engineering pipeline)
            base_cols = [f"operational_setting_{i}" for i in range(1, 4)]
            base_cols += [f"sensor_measurement_{i}" for i in range(1, 27)]
            
            # Identify base features not used by the model (they were removed during training)
            removed_features = [col for col in base_cols if col not in model_features]
            
            logger.warning(
                f"Removed features file not found. Derived {len(removed_features)} removed features "
                f"from model's feature_names_in_: {removed_features}"
            )
        else:
            # Cannot safely determine removed features - fail fast to prevent train/inference mismatch
            raise RuntimeError(
                f"Removed features file not found at {REMOVED_FEATURES_PATH} and cannot derive "
                "from model. Please ensure the removed_features file exists or retrain the model."
            )
    except Exception as e:
        logger.exception(f"Error loading model or features: {e}")
        model = None
        removed_features = []
    
    yield
    
    # Shutdown
    logger.info("Application shutdown")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Predictive Maintenance API",
    description="API for predicting equipment failures using sensor data",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Predictive Maintenance API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_path=str(MODEL_PATH),
    )


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict_failure(data: SensorData):
    """
    Predict equipment failure probability based on time-series sensor data.

    Args:
        data: Time-series sensor data from equipment (minimum 5 time steps)

    Returns:
        Prediction results with failure probability and recommendations
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs.",
        )

    try:
        # Create DataFrame from time-series data
        columns = ["unit_number", "time_in_cycles"]
        columns.extend([f"operational_setting_{i}" for i in range(1, 4)])
        columns.extend([f"sensor_measurement_{i}" for i in range(1, 27)])
        
        rows = []
        for idx, step in enumerate(data.time_steps):
            row = [data.unit_id, idx + 1] + step
            rows.append(row)
        
        df = pd.DataFrame(rows, columns=columns)
        
        # Step 1: Scale raw features FIRST (same as training pipeline)
        if scaler is not None:
            from src.config import FEATURE_CONFIG
            columns_to_scale = (
                FEATURE_CONFIG["operational_settings"] + FEATURE_CONFIG["sensor_measurements"]
            )
            columns_to_scale = [col for col in columns_to_scale if col in df.columns]
            df[columns_to_scale] = scaler.transform(df[columns_to_scale])
        
        # Step 2: Remove constant features BEFORE feature engineering
        if removed_features:
            for col in removed_features:
                if col in df.columns:
                    df = df.drop(columns=[col])
        
        # Step 3: Apply feature engineering on scaled data
        feature_engineer = FeatureEngineer()
        df = feature_engineer.engineer_all_features(df)
        
        # Step 4: Get the last time step (most recent) for prediction
        X = select_features_for_training(df.iloc[[-1]])

        # Make predictions
        failure_probability = float(model.predict_proba(X)[0][1])
        failure_prediction = bool(model.predict(X)[0])

        # Determine risk level and recommendation
        if failure_probability < 0.3:
            risk_level = "LOW"
            recommendation = "Continue normal operations. Monitor regularly."
        elif failure_probability < 0.5:
            risk_level = "MEDIUM"
            recommendation = "Schedule maintenance inspection within next cycle."
        elif failure_probability < 0.75:
            risk_level = "HIGH"
            recommendation = "Schedule immediate maintenance. Increase monitoring frequency."
        else:
            risk_level = "CRITICAL"
            recommendation = "URGENT: Schedule emergency maintenance immediately. Consider equipment shutdown."

        logger.info(
            f"Prediction for unit {data.unit_id}: "
            f"probability={failure_probability:.4f}, "
            f"risk_level={risk_level}"
        )

        return PredictionResponse(
            unit_id=data.unit_id,
            failure_probability=failure_probability,
            failure_prediction=failure_prediction,
            risk_level=risk_level,
            recommendation=recommendation,
        )

    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(data_list: List[SensorData]):
    """
    Batch prediction endpoint for multiple equipment units.

    Args:
        data_list: List of sensor data from multiple units

    Returns:
        List of prediction results
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs.",
        )

    results = []
    for data in data_list:
        try:
            result = await predict_failure(data)
            results.append(result)
        except Exception as e:
            logger.error(f"Error in batch prediction for unit {data.unit_id}: {e}")
            # Continue with other predictions

    return results


def main():
    """Run the API server."""
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
