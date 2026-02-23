from backend.app.ml_runtime.shap_explainer import ShapExplainer
import numpy as np
import datetime as dt

explainer = ShapExplainer()

now = dt.datetime.utcnow()
hour = now.hour

input_data = {
    "irradiance_wm2": 850,
    "temp_c": 32,
    "humidity": 55,
    "wind_mps": 3,
    "rain_prob": 5,
    "inv1_ac_kw": 130,
    "inv1_dc_kw": 135,
    "inv1_eff": 97
}


result = explainer.explain(input_data)

print(result)
