import os
import pandas as pd
from sklearn.model_selection import train_test_split

from config import SEED, TEST_SIZE, RAW_DATA_PATH, PROCESSED_DATA_DIR


# -------------------------------------------------
# Load raw dataset
# -------------------------------------------------
def load_data(path=RAW_DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at: {path}")
    
    df = pd.read_csv(path)
    
    return df


# -------------------------------------------------
# Separate features and target
# -------------------------------------------------
def split_features_target(df, target_col="Revenue"):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)  # ensure 0/1
    
    return X, y


# -------------------------------------------------
# Train / Test split
# -------------------------------------------------
def split_data(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=SEED,
    save_splits=False,
    save_dir=PROCESSED_DATA_DIR
):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    if save_splits:
        os.makedirs(save_dir, exist_ok=True)

        X_train.to_csv(f"{save_dir}/X_train.csv", index=False)
        X_test.to_csv(f"{save_dir}/X_test.csv", index=False)
        y_train.to_csv(f"{save_dir}/y_train.csv", index=False)
        y_test.to_csv(f"{save_dir}/y_test.csv", index=False)

    return X_train, X_test, y_train, y_test
