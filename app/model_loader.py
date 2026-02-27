import joblib
from config import MODEL_PATH
from src.logger import get_logger


logger = get_logger("MODEL_LOADER")

_model = None

def load_model():
    """
    Load the model once and return it.
    """
    global _model
    if _model is None:
        try:
            logger.info("Loading model...")
            _model = joblib.load(MODEL_PATH)
            logger.info("Model loaded successfully.")
        except FileNotFoundError:
            logger.error(f"Model file not found at {MODEL_PATH}")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    return _model
