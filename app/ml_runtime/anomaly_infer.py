import json
import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path

class AnomalyModel:

    def __init__(self):
        root = Path(__file__).resolve().parents[2]
        model_path = root / "ml_models" / "power_baseline.pkl"
        meta_path = root / "ml_models" / "power_baseline_meta.json"

        obj = joblib.load(model_path)
        if isinstance(obj, dict) and "model" in obj and "features" in obj:
            self.model = obj["model"]
            self.feature_cols = [str(c) for c in obj.get("features", [])]
        else:
            self.model = obj
            with open(meta_path, "r") as f:
                self.feature_cols = json.load(f)["feature_cols"]

        self.explainer = shap.TreeExplainer(self.model)

    def add_time_features(self, df):
        ts = pd.to_datetime(df["timestamp"], utc=True)
        df["hour"] = ts.dt.hour
        df["dow"] = ts.dt.dayofweek
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
        return df

    def score(self, df: pd.DataFrame):

        df = df.copy()
        df = self.add_time_features(df)

        X = df[self.feature_cols]
        y = df["active_power_kw"].values

        y_pred = self.model.predict(X)
        residual = y - y_pred

        mu = residual.mean()
        sigma = residual.std() if residual.std() > 1e-6 else 1.0
        z = (residual - mu) / sigma

        df["expected_power_kw"] = y_pred
        df["z_score"] = z
        df["is_anomaly"] = np.abs(z) > 3

        return df

    def explain_row(self, row_df):
        X = row_df[self.feature_cols]
        shap_vals = self.explainer.shap_values(X)[0]

        contributions = list(zip(self.feature_cols, shap_vals))
        top = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)[:3]

        return [
            {"feature": f, "impact": float(v)}
            for f, v in top
        ]
