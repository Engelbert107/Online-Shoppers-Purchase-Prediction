from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from config import NUM_FEATURES, CAT_FEATURES, BOOL_FEATURES

def create_preprocessor():
    num_transformer = StandardScaler() 
    cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, NUM_FEATURES),
            ('cat', cat_transformer, CAT_FEATURES),
            ('bool', 'passthrough', BOOL_FEATURES)
        ]
    )

    return preprocessor
