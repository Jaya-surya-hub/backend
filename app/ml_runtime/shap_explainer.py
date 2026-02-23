import shap
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "ml_models" / "power_baseline.pkl"

FEATURES = [
    "irradiance_wm2",
    "temp_c",
    "humidity",
    "wind_mps",
    "rain_prob",
    "inv1_ac_kw",
    "inv1_dc_kw",
    "inv1_eff",
    "sin_hour",
    "cos_hour",
    "power_to_irr"
]

class ShapExplainer:

    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.explainer = shap.Explainer(self.model)

    def explain(self, input_dict: dict):

        df = pd.DataFrame([input_dict])

        # ---- Derived features ----
        hour = pd.Timestamp.now().hour
        df["sin_hour"] = np.sin(2 * np.pi * hour / 24)
        df["cos_hour"] = np.cos(2 * np.pi * hour / 24)
        df["power_to_irr"] = df["inv1_ac_kw"] / df["irradiance_wm2"].replace(0, 1)

        df = df[FEATURES]

        prediction = float(self.model.predict(df)[0])
        shap_values = self.explainer(df, check_additivity=False)

        feature_impact = {
            feature: float(shap_values.values[0][i])
            for i, feature in enumerate(FEATURES)
        }

        return {
            "prediction": prediction,
            "feature_impact": feature_impact
        }
