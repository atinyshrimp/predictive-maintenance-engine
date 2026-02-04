"""Demo script showing how to make API requests with proper time-series data."""

import requests
import json

# API endpoint
API_URL = "http://localhost:8000/predict"

# Sample time-series data: 5 consecutive time steps for unit 1
# Each time step has 29 values: [op_setting_1, op_setting_2, op_setting_3, sensor_1, ..., sensor_26]
sample_request = {
    "unit_id": 1,
    "time_steps": [
        # Time step 1
        [-0.0007, -0.0004, 100.0, 518.67, 641.82, 1589.7, 1400.6, 14.62, 21.61, 
         554.36, 2388.02, 9046.19, 1.3, 47.47, 521.66, 2388.02, 8138.62, 8.4195, 
         0.03, 392, 2388, 100.0, 39.06, 23.4190, 0.0, 100.0, 0.0, 0.0, 0.0],
        
        # Time step 2
        [-0.0004, -0.0002, 100.0, 518.67, 642.15, 1591.8, 1403.1, 14.62, 21.61,
         553.75, 2388.03, 9044.07, 1.3, 47.49, 522.28, 2388.03, 8131.49, 8.4318,
         0.03, 392, 2388, 100.0, 38.86, 23.3735, 0.0, 100.0, 0.0, 0.0, 0.0],
        
        # Time step 3
        [0.0005, 0.0003, 100.0, 518.67, 642.35, 1587.6, 1404.2, 14.62, 21.61,
         554.26, 2388.08, 9052.94, 1.3, 47.27, 522.42, 2388.08, 8133.23, 8.4178,
         0.03, 391, 2388, 100.0, 38.95, 23.3442, 0.0, 100.0, 0.0, 0.0, 0.0],
        
        # Time step 4
        [-0.0012, -0.0008, 100.0, 518.67, 642.37, 1582.8, 1401.9, 14.62, 21.61,
         554.45, 2388.11, 9049.48, 1.3, 47.13, 522.86, 2388.11, 8133.83, 8.3682,
         0.03, 392, 2388, 100.0, 38.88, 23.3739, 0.0, 100.0, 0.0, 0.0, 0.0],
        
        # Time step 5 (most recent)
        [-0.0013, -0.0010, 100.0, 518.67, 642.27, 1582.8, 1406.2, 14.62, 21.61,
         554.00, 2388.06, 9055.15, 1.3, 47.28, 522.19, 2388.06, 8133.80, 8.4294,
         0.03, 393, 2388, 100.0, 38.90, 23.3594, 0.0, 100.0, 0.0, 0.0, 0.0],
    ]
}


def make_prediction():
    """Make a prediction request to the API."""
    print("Making prediction request...")
    print(f"URL: {API_URL}")
    print(f"\nRequest data:")
    print(f"  Unit ID: {sample_request['unit_id']}")
    print(f"  Time steps: {len(sample_request['time_steps'])}")
    print(f"  Features per step: {len(sample_request['time_steps'][0])}")
    
    response = None
    try:
        response = requests.post(API_URL, json=sample_request)
        response.raise_for_status()
        
        result = response.json()
        print(f"\n✅ Prediction successful!")
        print(f"\nResults:")
        print(f"  Unit ID: {result['unit_id']}")
        print(f"  Failure Probability: {result['failure_probability']:.2%}")
        print(f"  Failure Prediction: {result['failure_prediction']}")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Recommendation: {result['recommendation']}")
        
        return result
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API.")
        print("   Make sure the API is running: uvicorn api.app:app")
        
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ HTTP Error: {e}")
        if response is not None:
            print(f"   Response: {response.text}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


def make_batch_prediction():
    """Make a batch prediction request."""
    batch_request = [
        sample_request,
        {
            "unit_id": 2,
            "time_steps": sample_request["time_steps"]  # Reusing same time steps for demo
        }
    ]
    
    print("\nMaking batch prediction request...")
    try:
        response = requests.post("http://localhost:8000/predict/batch", json=batch_request)
        response.raise_for_status()
        
        results = response.json()
        print(f"\n✅ Batch prediction successful!")
        print(f"\nProcessed {len(results)} units:")
        for result in results:
            print(f"  Unit {result['unit_id']}: {result['risk_level']} risk ({result['failure_probability']:.2%})")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Predictive Maintenance API - Demo Request")
    print("=" * 60)
    
    # Single prediction
    make_prediction()
    
    print("\n" + "=" * 60)
    
    # Batch prediction
    make_batch_prediction()
    
    print("\n" + "=" * 60)
    print("\nFor interactive testing, visit: http://localhost:8000/docs")
