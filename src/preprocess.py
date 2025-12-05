from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
import pandas as pd
import numpy as np

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]

class RareLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.01):
        self.threshold = threshold
        self.mappings = {}

    def fit(self, X, y=None):
        for col in X.columns:
            freqs = X[col].value_counts(normalize=True)
            rare = set(freqs[freqs < self.threshold].index)
            self.mappings[col] = rare
        return self

    def transform(self, X):
        X = X.copy()
        for col, rare in self.mappings.items():
            X[col] = X[col].where(~X[col].isin(rare), other="Rare")
        return X

def build_preprocessor(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "Id"]
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    small_card_cols = [c for c in categorical_cols if X[c].nunique() < 10]
    large_card_cols = [c for c in categorical_cols if X[c].nunique() >= 10]

    numeric_pipeline = Pipeline([
        ("selector", ColumnSelector(numeric_cols)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    small_card_pipeline = Pipeline([
        ("selector", ColumnSelector(small_card_cols)),
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("rare", RareLabelEncoder(threshold=0.01)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    large_card_pipeline = Pipeline([
        ("selector", ColumnSelector(large_card_cols)),
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("rare", RareLabelEncoder(threshold=0.01)),
        # Target encoding applied later during training
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if small_card_cols:
        transformers.append(("small_cat", small_card_pipeline, small_card_cols))
    if large_card_cols:
        transformers.append(("large_cat", large_card_pipeline, large_card_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)
    return preprocessor, large_card_cols

