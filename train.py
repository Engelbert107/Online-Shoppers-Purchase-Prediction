import os
from datetime import datetime

from src.data_loader import load_data, split_features_target, split_data
from src.models import create_best_decision_tree_pipeline, save_pipeline
from src.logger import get_logger
from config import MODEL_DIR

# ----------------------
# Logger
# ----------------------
logger = get_logger("TRAINING")


# ----------------------
# Train process
# ----------------------
def main():
    logger.info("========== TRAINING STARTED ==========")

    # Load data
    df = load_data()
    logger.info(f"Dataset loaded with shape: {df.shape}")

    # Split features / target
    X, y = split_features_target(df)
    logger.info("Features and target separated.")

    # Train / Test split
    X_train, X_test, y_train, y_test = split_data(X, y)
    logger.info(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")

    # Build pipeline
    pipeline = create_best_decision_tree_pipeline()
    logger.info("Pipeline created.")

    # Train
    logger.info("Training model...")
    pipeline.fit(X_train, y_train)
    logger.info("Model training completed.")

    # Save versioned pipeline
    os.makedirs(MODEL_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_path = os.path.join(MODEL_DIR, f"online_shoppers_dt_{timestamp}.pkl")
    latest_path = os.path.join(MODEL_DIR, "latest.pkl")

    save_pipeline(pipeline, versioned_path)
    save_pipeline(pipeline, latest_path)

    logger.info(f"Versioned model stored at: {versioned_path}")
    logger.info(f"Latest model updated at: {latest_path}")
    logger.info("========== TRAINING FINISHED ==========")


if __name__ == "__main__":
    main()
