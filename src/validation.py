import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from config import NUM_FEATURES, CAT_FEATURES, BOOL_FEATURES
from src.logger import get_logger

logger = get_logger("VALIDATION")


class FeatureValidator(BaseEstimator, TransformerMixin):
    """
    Ensures input data matches the expected schema
    before entering the ML pipeline.
    """

    def __init__(self):
        self.expected_columns = NUM_FEATURES + CAT_FEATURES + BOOL_FEATURES

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.info("Validating input features...")

        X = pd.DataFrame(X)

        # Check missing columns
        missing_cols = set(self.expected_columns) - set(X.columns)
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Enforce order
        X = X[self.expected_columns]

        # Light type safety
        X[BOOL_FEATURES] = X[BOOL_FEATURES].astype(bool)

        logger.info("Feature validation passed.")
        return X
