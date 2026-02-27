import pandas as pd
from src.data_loader import load_data, split_features_target
from config import RAW_DATA_PATH

def test_data_loads():
    df = load_data(RAW_DATA_PATH)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0

def test_target_exists():
    df = load_data(RAW_DATA_PATH)
    X, y = split_features_target(df)
    assert "Revenue" not in X.columns
    assert set(y.unique()).issubset({0, 1})
