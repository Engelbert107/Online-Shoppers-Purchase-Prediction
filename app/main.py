from fastapi import FastAPI, HTTPException, Request
from typing import List, Union
import pandas as pd

from app.schemas import ShopperFeatures, PredictionResponse
from app.model_loader import load_model
from src.logger import get_logger
from config import APP_NAME, APP_VERSION, NUM_FEATURES, CAT_FEATURES, BOOL_FEATURES

logger = get_logger("API")

app = FastAPI(title=APP_NAME, version=APP_VERSION)

# Load model once at startup
model = load_model()


# ----------------------------
# Middleware to log all requests
# ----------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response


@app.get("/health")
def health_check():
    logger.info("Health check requested")
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=List[PredictionResponse])
def predict(data: Union[ShopperFeatures, List[ShopperFeatures]]):
    """
    Predict purchase intention for one or more shoppers.
    """
    try:
        # Normalize to a list if a single object is passed
        if isinstance(data, ShopperFeatures):
            data = [data]

        # Convert to DataFrame
        df = pd.DataFrame([d.model_dump() for d in data])

        # Enforce column order
        expected_columns = NUM_FEATURES + CAT_FEATURES + BOOL_FEATURES
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        df = df[expected_columns]

        # Make predictions
        preds = model.predict(df)
        probs = model.predict_proba(df)[:, 1]

        logger.info(f"Predictions made for {len(data)} samples")

        # Build response
        response = [
            PredictionResponse(prediction=int(p), probability=round(float(prob), 4))
            for p, prob in zip(preds, probs)
        ]
        return response

    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")
