
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "TotalCharges" in X.columns:
            X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
        if "TotalCharges" in X.columns and "tenure" in X.columns:
            X["AvgMonthlySpend"] = X["TotalCharges"] / X["tenure"].replace(0, 1)
        service_cols = [
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"
        ]
        available = [c for c in service_cols if c in X.columns]
        if available:
            X["NumOptionalServices"] = X[available].eq("Yes").sum(axis=1).astype(float)
        return X
