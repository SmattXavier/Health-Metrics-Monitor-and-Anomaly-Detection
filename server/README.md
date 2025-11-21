# Health Risk Prediction Server

This is the server component of the Health Risk Prediction app. It provides an API endpoint for making predictions based on heart rate and body temperature measurements.

## Setup

1. Create a Python virtual environment:

```bash
python -m venv venv
```

2. Activate the virtual environment:

- Windows:

```bash
venv\Scripts\activate
```

- Unix/MacOS:

```bash
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Place your trained model file as `model.pkl` in the server directory.

## Running the Server

1. For development:

```bash
python app.py
```

2. For production (using gunicorn, Unix/MacOS only):

```bash
gunicorn app:app
```

## API Endpoints

### Health Check

- URL: `/health`
- Method: `GET`
- Response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Predict

- URL: `/predict`
- Method: `POST`
- Request Body:

```json
{
  "heart_rate": 75,
  "temperature": 37.0
}
```

- Response:

```json
{
  "prediction": "Low Risk",
  "probabilities": {
    "Low Risk": 0.8,
    "Moderate Risk": 0.15,
    "High Risk": 0.05
  },
  "recommendations": [
    "Continue maintaining healthy habits.",
    "Regular exercise and balanced diet recommended.",
    "Monitor your vital signs periodically."
  ]
}
```

## Model Requirements

The model should be a scikit-learn compatible model that:

1. Takes a 2D array of `[heart_rate, temperature]` as input
2. Has `predict()` and `predict_proba()` methods
3. Outputs predictions for three classes: Low Risk (0), Moderate Risk (1), and High Risk (2)

## Error Handling

The API will return appropriate error messages if:

- The model file is not found
- The input data is invalid
- There's an error during prediction
