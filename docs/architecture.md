# Flight Delay Prediction — Technical Reference

## System Overview

This service predicts the probability of a flight delay (>15 minutes) at Santiago (SCL) airport. It exposes a REST API backed by a machine learning model trained on historical flight data.

```
Client Request
    │
    ▼
┌─────────────────────────────────────────┐
│  FastAPI  (challenge/api.py)            │
│                                         │
│  POST /predict  ──►  DelayModel.predict │
│                           │             │
│                     preprocess()        │
│                     XGBClassifier       │
│                           │             │
│  {"predict": [0, 1, ...]} ◄─────────── │
└─────────────────────────────────────────┘
    │
    ▼
Uvicorn (port 8080)
```

On startup the API attempts to load a pre-trained model from `models/delay_model.pkl`. If that file is absent it trains from `data/data.csv` and persists the artifact.

---

## Model

### Algorithm choice: XGBoost Classifier

Two candidates were evaluated from the DS notebook: Logistic Regression and XGBoost. XGBoost was chosen because it produced meaningfully higher recall on the delayed class (class 1), which is the operationally relevant metric — missing a delayed flight is more costly than a false alarm.

### Feature engineering

Raw inputs are one-hot encoded. From the full encoding, the following **top-10 features** are kept (derived from airline, month, and flight type):

| Feature | Source column |
|---|---|
| `OPERA_Latin American Wings` | `OPERA` |
| `MES_7` | `MES` |
| `MES_10` | `MES` |
| `OPERA_Grupo LATAM` | `OPERA` |
| `MES_12` | `MES` |
| `TIPOVUELO_I` | `TIPOVUELO` |
| `MES_4` | `MES` |
| `MES_11` | `MES` |
| `OPERA_Sky Airline` | `OPERA` |
| `OPERA_Copa Air` | `OPERA` |

### Class imbalance

The dataset is imbalanced (~18% delayed flights). This is addressed via XGBoost's `scale_pos_weight` parameter, set to the ratio of negative to positive samples at training time.

### Hyperparameters

```python
XGBClassifier(
    learning_rate=0.01,
    random_state=1,
    scale_pos_weight=<computed from training data>
)
```

---

## API Reference

### `GET /health`

Health check endpoint.

**Response `200 OK`**
```json
{"status": "OK"}
```

---

### `POST /predict`

Predict delay probability for one or more flights.

**Request body**
```json
{
  "flights": [
    {
      "OPERA": "Grupo LATAM",
      "TIPOVUELO": "N",
      "MES": 3
    }
  ]
}
```

| Field | Type | Constraints |
|---|---|---|
| `OPERA` | `string` | Airline name (free text) |
| `TIPOVUELO` | `"N"` or `"I"` | National or International |
| `MES` | `integer` | 1–12 |

**Response `200 OK`**
```json
{"predict": [0]}
```

Each element in `predict` corresponds positionally to the input flight: `0` = no delay, `1` = delay expected.

**Response `400 Bad Request`** — validation error (invalid field value or missing field)

```json
{"detail": [...]}
```

**Response `500 Internal Server Error`** — prediction failure

---

## Project Structure

```
latam_challenge/
├── challenge/
│   ├── __init__.py          # Exports the FastAPI app instance
│   ├── api.py               # Endpoint definitions, Pydantic schemas, startup logic
│   └── model.py             # DelayModel class (preprocess / fit / predict)
├── data/
│   └── data.csv             # Historical SCL flight data
├── models/
│   └── delay_model.pkl      # Serialized trained model (joblib)
├── tests/
│   ├── model/test_model.py  # Preprocessing and fit/predict correctness
│   ├── api/test_api.py      # Endpoint integration tests
│   └── stress/api_stress.py # Locust load test scenarios
├── docs/
│   ├── challenge.md         # Implementation decisions and caveats
│   └── architecture.md      # This file
├── workflows/               # CI/CD YAML templates (to be moved to .github/workflows/)
│   ├── ci.yml
│   └── cd.yml
├── Dockerfile
├── .dockerignore
├── Makefile
├── requirements.txt         # Runtime dependencies
├── requirements-test.txt    # Test tooling (pytest, locust, coverage)
└── requirements-dev.txt     # Exploration tooling (jupyter, matplotlib)
```

---

## Local Development

```bash
# Create and activate virtual environment
make venv
source .venv/bin/activate

# Install all dependencies
make install

# Start the API server (development)
uvicorn challenge.api:app --reload --port 8080
```

---

## Docker

```bash
# Build the image
docker build -t latam-delay-api .

# Run the container
docker run -p 8080:8080 latam-delay-api
```

The image bundles the training data and model artifact. If the model file is absent at runtime the container trains it on startup (fast, given the dataset size). See `docs/challenge.md` for the production-grade alternative.

---

## Testing

### Model tests

Validate preprocessing output shape, fit metrics (recall/F1 thresholds), and prediction output type.

```bash
make model-test
```

### API tests

Integration tests against the full FastAPI application using `TestClient`. Covers the happy path and validation error cases.

```bash
make api-test
```

### Stress test

Locust-based load test targeting the deployed API. Runs 100 concurrent users, 1 spawn/second, for 60 seconds.

```bash
make stress-test   # targets http://127.0.0.1:8000 by default
# To target a deployed URL, edit STRESS_URL in the Makefile (line 26)
```

---

## CI/CD

To be implemented
