import joblib
import numpy as np
import pandas as pd

def add_time_features_one(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df = df.copy()
    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    return df

def score_row(row: dict, artifact_path="data/power_baseline.pkl"):
    art = joblib.load(artifact_path)
    model = art["model"]
    feats = art["features"]
    irr_min = art["day_irradiance_min"]
    thr95 = art["thr_abs_err_p95"]
    thr99 = art["thr_abs_err_p99"]

    df = pd.DataFrame([row])
    df = add_time_features_one(df)

    # Night handling: you usually don't want “anomalies” at night noise
    if float(df["irradiance_wm2"].iloc[0]) < irr_min:
        return {
            "is_anomaly": False,
            "severity": "none",
            "reason": "night_low_irradiance",
            "pred_kw": None,
            "err_kw": None,
            "score": 0.0,
        }

    pred = float(model.predict(df[feats])[0])
    actual = float(df["active_power_kw"].iloc[0])
    err = actual - pred
    abs_err = abs(err)

    if abs_err >= thr99:
        sev = "high"
        is_anom = True
    elif abs_err >= thr95:
        sev = "medium"
        is_anom = True
    else:
        sev = "low"
        is_anom = False

    # score can be normalized (useful for dashboards)
    score = float(abs_err / max(thr95, 1e-6))

    return {
        "is_anomaly": is_anom,
        "severity": sev,
        "pred_kw": pred,
        "err_kw": err,
        "abs_err_kw": abs_err,
        "score": score,
    }