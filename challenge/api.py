import pathlib
from typing import List, Literal

import fastapi
import pandas as pd
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator

from challenge.model import DelayModel

# Paths are resolved relative to this file so they work regardless of cwd.
_DATA_PATH = pathlib.Path(__file__).parent.parent / "data" / "data.csv"
_MODEL_PATH = pathlib.Path(__file__).parent.parent / "models" / "delay_model.pkl"

_model = DelayModel()
if _MODEL_PATH.exists():
    # Fast path: load the already-trained model from disk.
    _model.get_model(str(_MODEL_PATH))
else:
    # First run: train the model and persist it for subsequent starts.
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    _data = pd.read_csv(str(_DATA_PATH), low_memory=False)
    _preprocess_result = _model.preprocess(_data, target_column="delay")
    _features, _target = _preprocess_result
    _model.fit(_features, _target)
    _model.save_model(str(_MODEL_PATH))

app = fastapi.FastAPI()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return HTTP 400 instead of FastAPI's default 422 for request validation errors."""
    return JSONResponse(status_code=400, content={"detail": exc.errors()})


class FlightItem(BaseModel):
    """A single flight record used as input to the delay prediction model."""

    OPERA: str
    TIPOVUELO: Literal["N", "I"]
    MES: int

    @validator("MES")
    def validate_mes(cls, v: int) -> int:
        """Reject months outside the valid 1–12 range."""
        if v < 1 or v > 12:
            raise ValueError("MES must be between 1 and 12")
        return v


class PredictRequest(BaseModel):
    """Request body for the /predict endpoint."""

    flights: List[FlightItem]


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }


@app.post("/predict", status_code=200)
async def post_predict(body: PredictRequest) -> dict:
    """
    Predict flight delays for a list of flights.

    Args:
        body: PredictRequest with a 'flights' list of FlightItem objects.

    Returns:
        dict with a 'predict' key containing a list of 0/1 integers (0 = no delay, 1 = delay).
    """
    df = pd.DataFrame([f.dict() for f in body.flights])
    features = _model.preprocess(df)
    predictions = _model.predict(features)
    return {"predict": predictions}
