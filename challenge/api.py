import logging
import pathlib
from typing import List, Literal

import fastapi
import pandas as pd
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator

from challenge.model import DelayModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths are resolved relative to this file so they work regardless of cwd.
_DATA_PATH = pathlib.Path(__file__).parent.parent / "data" / "data.csv"
_MODEL_PATH = pathlib.Path(__file__).parent.parent / "models" / "delay_model.pkl"

_model = DelayModel()

# TODO: wrap this logic so it only runs on startup
if _MODEL_PATH.exists():
    logger.info("Loading pre-trained model from %s", _MODEL_PATH)
    _model.get_model(str(_MODEL_PATH))
    logger.info("Model loaded successfully")
else:
    logger.warning("No pre-trained model at %s — training from scratch", _MODEL_PATH)
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    _data = pd.read_csv(str(_DATA_PATH), low_memory=False)
    logger.info("Training data loaded: %d rows", len(_data))
    _preprocess_result = _model.preprocess(_data, target_column="delay")
    _features, _target = _preprocess_result
    _model.fit(_features, _target)
    _model.save_model(str(_MODEL_PATH))
    logger.info("Model trained and saved to %s", _MODEL_PATH)

app = fastapi.FastAPI()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return HTTP 400 instead of FastAPI's default 422 for request validation errors."""
    logger.warning("Request validation error: %s", exc.errors())
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
    logger.debug("Health check")
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
    logger.info("Predict request: %d flight(s)", len(body.flights))
    try:
        df = pd.DataFrame([f.dict() for f in body.flights])
        features = _model.preprocess(df)
        logger.debug("Features shape: %s", features.shape)
        predictions = _model.predict(features)
        logger.info("Prediction complete: %d result(s)", len(predictions))
        logger.debug("Prediction results: %s", predictions)
        return {"predict": predictions}
    except Exception as e:
        logger.error("Prediction failed: %s", e, exc_info=True)
        raise fastapi.HTTPException(status_code=500, detail="Something went wrong during prediction")
