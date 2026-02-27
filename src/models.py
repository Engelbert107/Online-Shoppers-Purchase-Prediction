import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline 
from imblearn.over_sampling import SMOTE

from config import SEED, BEST_RF_PARAMS, MRMR_PARAMS
from src.preprocessing import create_preprocessor




# =================================================
# MRMR Selector (custom implementation)
# =================================================

class MRMRSelector(BaseEstimator, TransformerMixin):
    """
    Custom MRMR feature selector (difference form)
    """

    def __init__(self, n_features=17, random_state=SEED):
        self.n_features = n_features
        self.random_state = random_state

    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = np.asarray(y)

        features = X.columns.tolist()
        selected = []

        relevance = mutual_info_classif(X, y, random_state=self.random_state)
        relevance = pd.Series(relevance, index=features)

        first_feature = relevance.idxmax()
        selected.append(first_feature)
        remaining = set(features)
        remaining.remove(first_feature)

        while len(selected) < self.n_features and remaining:
            scores = {}

            for feature in remaining:
                redundancy_values = []
                for sel in selected:
                    mi_fs = mutual_info_regression(
                        X[[feature]], X[sel], random_state=self.random_state
                    )[0]
                    redundancy_values.append(mi_fs)

                mean_redundancy = np.mean(redundancy_values)
                score = relevance[feature] - mean_redundancy
                scores[feature] = score

            best_feature = max(scores, key=scores.get)
            selected.append(best_feature)
            remaining.remove(best_feature)

        self.selected_features_ = selected
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X[self.selected_features_]


# =================================================
# Best Decision Tree pipeline 
# =================================================

def create_best_decision_tree_pipeline():
    """
    Preprocessing + MRMR + SMOTE + Decision Tree pipeline:
    1. Preprocessing: numeric scaling, one-hot encoding, passthrough boolean
    2. MRMR feature selection
    3. SMOTE oversampling
    4. Decision Tree classifier
    """

    preprocessor = create_preprocessor()

    mrmr = MRMRSelector(
        n_features=MRMR_PARAMS["n_features"],
        random_state=MRMR_PARAMS["random_state"]
    )

    smote = SMOTE(random_state=SEED)
    
    model = RandomForestClassifier(
        max_depth=BEST_RF_PARAMS["max_depth"],
        min_samples_split=BEST_RF_PARAMS["min_samples_split"],
        min_samples_leaf=BEST_RF_PARAMS["min_samples_leaf"],
        n_estimators=BEST_RF_PARAMS["n_estimators"],
        max_features=BEST_RF_PARAMS["max_features"],
        class_weight=BEST_RF_PARAMS["class_weight"],
        random_state=SEED,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessor),   # numeric scaling + OHE
        ("mrmr", mrmr),                    # MRMR feature selection
        ("smote", smote),                  # oversample
        ("model", model)
    ])

    return pipeline


# =================================================
# Helper functions for saving/loading joblib models
# =================================================
def save_pipeline(pipeline, path: str):
    """
    Save the pipeline with joblib
    """
    dump(pipeline, path)
    return path


def load_pipeline(path: str):
    """
    Load the pipeline with joblib
    """
    return load(path)


