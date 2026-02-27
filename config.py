from datetime import datetime


def generate_model_version():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# -----------------------------
# For Reproductivity
# -----------------------------
SEED = 25

# -----------------------------
# Train / Test split
# -----------------------------
TEST_SIZE = 0.3

# -----------------------------
# Model Hyperparameters
# -----------------------------
BEST_DT_PARAMS = {
    'max_depth': 5,
    'min_samples_split': 11,
    'min_samples_leaf': 6
}

# BEST_RF_PARAMS = {
#     "max_depth": 15,
#     "min_samples_leaf": 9,
#     "min_samples_split": 11,
#     "n_estimators": 245
# }
BEST_RF_PARAMS = {
    "max_depth": 20,
    "min_samples_leaf": 5,
    "min_samples_split": 24,
    "n_estimators": 671,
    "max_features": "sqrt",
    "class_weight": None
}

# n_features 13 -> ROC-AUC=0.9142 Acc=88.32% F1=0.6646  TPR=0.7483 TNR=0.9079 (RobustScaler)
# n_features 22 -> ROC-AUC=0.9286 Acc=89.54% F1=0.6846  TPR=0.7343 TNR=0.9248 
# n_features 25 -> ROC-AUC=0.9303 Acc=89.78% F1=0.6917  TPR=0.7413 TNR=0.9264 
# n_features 32 -> ROC-AUC=0.9337 Acc=89.75% F1=0.6926  TPR=0.7465 TNR=0.9252
# n_features 37 -> ROC-AUC=0.9337 Acc=89.75% F1=0.6926  TPR=0.7465 TNR=0.9252
# n_features 39 -> ROC-AUC=0.9338 Acc=89.73% F1=0.6844  TPR=0.7203 TNR=0.9296 (RobustScaler)
# n_features 39 -> ROC-AUC=0.9338 Acc=89.81% F1=0.6977  TPR=0.7605 TNR=0.9232 (StandardScaler)

MRMR_PARAMS = {
    'n_features': 39,
    'random_state': SEED
}

# -----------------------------
# Feature Groups
# -----------------------------
NUM_FEATURES = [
    'Administrative', 'Administrative_Duration',
    'Informational', 'Informational_Duration',
    'ProductRelated', 'ProductRelated_Duration',
    'BounceRates', 'ExitRates',
    'PageValues', 'SpecialDay',
    'OperatingSystems', 'Browser', 'Region', 'TrafficType'  
]

CAT_FEATURES = [
    'Month', 'VisitorType'  
]

BOOL_FEATURES = ['Weekend']

# -----------------------------
# Paths
# -----------------------------
RAW_DATA_PATH = "data/raw/online_shoppers_intention.csv"
PROCESSED_DATA_DIR = "data/processed"
MODEL_DIR = "models"
MODEL_NAME = "online_shoppers_dt"
REPORT_DIR = "reports"

MODEL_PATH = "models/latest.pkl"
APP_NAME = "Online Shoppers Purchase Prediction API"
APP_VERSION = "1.0.0"
