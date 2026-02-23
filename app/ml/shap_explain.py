import datetime as dt
from typing import Dict, Any


def explain_at_timestamp(plant_id: int, tenant_id: int, timestamp: dt.datetime) -> Dict[str, Any]:
    # Stub structure representing SHAP-like contributions
    return {
        "plant_id": plant_id,
        "tenant_id": tenant_id,
        "timestamp": timestamp.isoformat(),
        "prediction_kw": 720.0,
        "contributions": [
            {"feature": "irradiance_wm2", "shap_value": 0.55, "value": 820},
            {"feature": "temp_c", "shap_value": -0.08, "value": 34.2},
            {"feature": "wind_mps", "shap_value": 0.02, "value": 3.1},
            {"feature": "humidity", "shap_value": -0.03, "value": 48.0},
        ],
        "baseline_kw": 500.0
    }

