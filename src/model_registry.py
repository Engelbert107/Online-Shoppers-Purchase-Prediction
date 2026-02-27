import os
import joblib
from config import MODEL_DIR, MODEL_NAME, generate_model_version
from src.logger import get_logger

logger = get_logger("MODEL_REGISTRY")


def save_model(model):
    version = generate_model_version()
    model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_{version}.pkl")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, model_path)

    logger.info(f"Model saved: {model_path}")

    # Also update "latest" pointer
    latest_path = os.path.join(MODEL_DIR, "latest.pkl")
    joblib.dump(model, latest_path)

    logger.info("Latest model updated.")

    return model_path


def load_latest_model():
    latest_path = os.path.join(MODEL_DIR, "latest.pkl")
    if not os.path.exists(latest_path):
        raise FileNotFoundError("No latest model found.")
    
    logger.info("Latest model loaded.")
    return joblib.load(latest_path)
