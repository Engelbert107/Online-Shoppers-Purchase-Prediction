from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from src.models import MRMRSelector
from src.validation import FeatureValidator
from config import BEST_DT_PARAMS, MRMR_PARAMS, SEED
from src.preprocessing import create_preprocessor

def create_full_pipeline():
    # Feature validation
    validator = FeatureValidator()

    # Preprocessing: numeric + categorical + boolean
    preprocessor = create_preprocessor()

    # MRMR → SMOTE → DecisionTree pipeline
    model_pipeline = ImbPipeline([
        ("mrmr", MRMRSelector(n_features=MRMR_PARAMS["n_features"],
                              random_state=MRMR_PARAMS["random_state"])),
        ("smote", SMOTE(random_state=SEED)),
        ("model", DecisionTreeClassifier(
            max_depth=BEST_DT_PARAMS["max_depth"],
            min_samples_split=BEST_DT_PARAMS["min_samples_split"],
            min_samples_leaf=BEST_DT_PARAMS["min_samples_leaf"],
            random_state=MRMR_PARAMS["random_state"]
        ))
    ])

    # Full pipeline: validation → preprocessing → model_pipeline
    full_pipeline = Pipeline([
        ("validation", validator),
        ("preprocessing", preprocessor),   # converts strings to numeric
        ("model_pipeline", model_pipeline)
    ])

    return full_pipeline
