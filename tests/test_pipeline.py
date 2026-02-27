# import pandas as pd
from src.pipeline import create_full_pipeline
from src.data_loader import load_data, split_features_target

def test_pipeline_fit_predict():
    df = load_data()
    X, y = split_features_target(df)

    X_sample = X.sample(200, random_state=42)
    y_sample = y.loc[X_sample.index]

    pipeline = create_full_pipeline()
    pipeline.fit(X_sample, y_sample)

    preds = pipeline.predict(X_sample)
    assert len(preds) == len(X_sample)
