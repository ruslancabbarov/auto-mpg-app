import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class AutoMPGFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["power_to_weight"] = X["horsepower"] / X["weight"].replace(0, np.nan)
        X["disp_per_cyl"] = X["displacement"] / X["cylinders"].replace(0, np.nan)
        X["weight_per_cyl"] = X["weight"] / X["cylinders"].replace(0, np.nan)
        return X


class RareBrandGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, min_count=5):
        self.min_count = min_count

    def fit(self, X, y=None):
        brands = pd.Series(np.asarray(X).reshape(-1)).astype("string").dropna()
        counts = brands.value_counts()
        self.common_brands_ = set(counts[counts >= self.min_count].index)
        return self

    def transform(self, X):
        brands = pd.Series(np.asarray(X).reshape(-1)).astype("string").fillna("other")
        return brands.where(brands.isin(self.common_brands_), "other").to_numpy().reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        names = input_features if input_features is not None else ["brand"]
        return np.asarray(names, dtype=object)
