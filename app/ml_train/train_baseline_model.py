import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

FEATURES = [
    "irradiance_wm2",
    "temp_c",
    "humidity",
    "wind_mps",
    "rain_prob",
    "inv1_ac_kw",
    "inv1_dc_kw",
    "inv1_eff",
]
TARGET = "active_power_kw"

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df = df.copy()
    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    return df

def main():
    inp = r"baseline_training.csv"   # since you are already in ml_train folder
    out_model = r"power_baseline.pkl"

    df = pd.read_csv(inp)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", TARGET, "irradiance_wm2"]).sort_values("timestamp")

    # Train on daytime only (reduces noise)
    df = df[df["irradiance_wm2"] >= 25].copy()
    df = add_time_features(df)

    split_idx = int(len(df) * 0.85)
    train = df.iloc[:split_idx]
    val = df.iloc[split_idx:]

    feats = FEATURES + ["hour_sin", "hour_cos", "dow"]

    X_train = train[feats]
    y_train = train[TARGET]
    X_val = val[feats]
    y_val = val[TARGET]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    pred_val = model.predict(X_val)
    mae = mean_absolute_error(y_val, pred_val)
    print(f"✅ Validation MAE = {mae:.3f} kW")

    pred_train = model.predict(X_train)
    residuals = y_train.values - pred_train

    q95 = float(np.quantile(np.abs(residuals), 0.95))
    q99 = float(np.quantile(np.abs(residuals), 0.99))

    artifact = {
        "model": model,
        "features": feats,
        "day_irradiance_min": 25.0,
        "thr_abs_err_p95": q95,
        "thr_abs_err_p99": q99,
    }

    joblib.dump(artifact, out_model)
    print(f"✅ Saved: {out_model}")
    print(f"✅ Thresholds: p95={q95:.3f} kW, p99={q99:.3f} kW")

if __name__ == "__main__":
    main()