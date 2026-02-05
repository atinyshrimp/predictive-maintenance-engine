# Predictive Maintenance Engine - API Documentation

## Overview

This REST API provides endpoints for predicting equipment failures based on sensor data. It's built with FastAPI and supports both single and batch predictions.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API doesn't require authentication. In production, implement OAuth2 or API key authentication.

## Endpoints

### 1. Root

**GET** `/`

Returns API information.

**Response:**

```json
{
	"name": "Predictive Maintenance API",
	"version": "1.0.0",
	"status": "running",
	"docs": "/docs"
}
```

---

### 2. Health Check

**GET** `/health`

Check API and model status.

**Response:**

```json
{
	"status": "healthy",
	"model_loaded": true,
	"model_path": "/path/to/model.pkl"
}
```

---

### 3. Single Prediction

**POST** `/predict`

Predict failure probability for a single equipment unit using time-series sensor data.

**Request Body:**

```json
{
	"unit_id": 1,
	"time_steps": [
		[0.5, 0.3, -0.2, 0.8, 1.2, -0.5, 0.1, 0.9, -0.3, 0.6, 0.4, -0.1, 0.7, 0.2, -0.4, 1.0, 0.3, -0.2, 0.5, 0.8, 0.1, 0.4, 0.6, -0.1, 0.8, 0.3, 0.5, 0.2, 0.9],
		[0.6, 0.4, -0.1, 0.9, 1.3, -0.4, 0.2, 1.0, -0.2, 0.7, 0.5, 0.0, 0.8, 0.3, -0.3, 1.1, 0.4, -0.1, 0.6, 0.9, 0.2, 0.5, 0.7, 0.0, 0.9, 0.4, 0.6, 0.3, 1.0],
		[0.7, 0.5, 0.0, 1.0, 1.4, -0.3, 0.3, 1.1, -0.1, 0.8, 0.6, 0.1, 0.9, 0.4, -0.2, 1.2, 0.5, 0.0, 0.7, 1.0, 0.3, 0.6, 0.8, 0.1, 1.0, 0.5, 0.7, 0.4, 1.1],
		[0.8, 0.6, 0.1, 1.1, 1.5, -0.2, 0.4, 1.2, 0.0, 0.9, 0.7, 0.2, 1.0, 0.5, -0.1, 1.3, 0.6, 0.1, 0.8, 1.1, 0.4, 0.7, 0.9, 0.2, 1.1, 0.6, 0.8, 0.5, 1.2],
		[0.9, 0.7, 0.2, 1.2, 1.6, -0.1, 0.5, 1.3, 0.1, 1.0, 0.8, 0.3, 1.1, 0.6, 0.0, 1.4, 0.7, 0.2, 0.9, 1.2, 0.5, 0.8, 1.0, 0.3, 1.2, 0.7, 0.9, 0.6, 1.3]
	]
}
```

**Parameters:**

- `unit_id` (integer, required): Unique identifier for the equipment unit
- `time_steps` (array of arrays, required): Time-series sensor data with **minimum 5 time steps**. Each time step contains 29 values (3 operational settings + 26 sensor measurements)

**Response:**

```json
{
	"unit_id": 1,
	"failure_probability": 0.75,
	"failure_prediction": true,
	"risk_level": "HIGH",
	"recommendation": "Schedule immediate maintenance. Increase monitoring frequency."
}
```

**Risk Levels:**

- `LOW` (< 0.3): Continue normal operations
- `MEDIUM` (0.3-0.5): Schedule maintenance inspection
- `HIGH` (0.5-0.75): Schedule immediate maintenance
- `CRITICAL` (> 0.75): Emergency maintenance required

**Status Codes:**

- `200`: Success
- `422`: Validation error (invalid input)
- `500`: Internal server error
- `503`: Service unavailable (model not loaded)

---

### 4. Batch Prediction

**POST** `/predict/batch`

Predict failure probabilities for multiple equipment units.

**Request Body:**

```json
[
  {
    "unit_id": 1,
    "time_steps": [[0.5, 0.3, ...], [0.6, 0.4, ...], [0.7, 0.5, ...], [0.8, 0.6, ...], [0.9, 0.7, ...]]
  },
  {
    "unit_id": 2,
    "time_steps": [[0.2, 0.1, ...], [0.3, 0.2, ...], [0.4, 0.3, ...], [0.5, 0.4, ...], [0.6, 0.5, ...]]
  }
]
```

**Response:**

```json
[
	{
		"unit_id": 1,
		"failure_probability": 0.75,
		"failure_prediction": true,
		"risk_level": "HIGH",
		"recommendation": "Schedule immediate maintenance..."
	},
	{
		"unit_id": 2,
		"failure_probability": 0.25,
		"failure_prediction": false,
		"risk_level": "LOW",
		"recommendation": "Continue normal operations..."
	}
]
```

---

## Error Handling

All endpoints return standard HTTP status codes and error messages:

**Example Error Response:**

```json
{
	"detail": "Sensor values must be finite numbers"
}
```

---

## Usage Examples

### Python (requests)

```python
import requests

# Single prediction with time-series data (minimum 5 time steps)
time_series_data = [
    [0.5, 0.3, -0.2, 0.8, ...],  # Time step 1 (29 values)
    [0.6, 0.4, -0.1, 0.9, ...],  # Time step 2
    [0.7, 0.5, 0.0, 1.0, ...],   # Time step 3
    [0.8, 0.6, 0.1, 1.1, ...],   # Time step 4
    [0.9, 0.7, 0.2, 1.2, ...],   # Time step 5
]

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "unit_id": 1,
        "time_steps": time_series_data
    }
)
result = response.json()
print(f"Failure probability: {result['failure_probability']:.2%}")
print(f"Risk level: {result['risk_level']}")
```

### cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "unit_id": 1,
    "sensor_values": [0.5, 0.3, -0.2, 0.8, ...]
  }'
```

### JavaScript (fetch)

```javascript
fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    unit_id: 1,
    sensor_values: [0.5, 0.3, -0.2, 0.8, ...]
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

---

## Interactive Documentation

Visit the following URLs when the API is running:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Rate Limiting

Currently no rate limiting is implemented. For production, consider adding rate limiting middleware.

## Monitoring

Monitor API performance using:

- FastAPI's built-in metrics
- Application logs in `logs/app.log`
- Health check endpoint for uptime monitoring

---

For more information, see the main [README.md](../README.md).
