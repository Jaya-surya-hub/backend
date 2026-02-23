import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import timezone
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


# -------------------------
# CONFIG
# -------------------------
CSV_PATH = r"C:\Users\Omen\Documents\trae_projects\project1wiztric\backend\app\ml_train\solar_week_5min.csv"
MODEL_DIR = "backend/app/ml_models"
TARGET_COL = "active_power_kw"

os.makedirs(MODEL_DIR, exist_ok=True)


# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv(CSV_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp").reset_index(drop=True)

# -------------------------
# FEATURE ENGINEERING
# -------------------------

# Time features
df["hour"] = df["timestamp"].dt.hour
df["dayofweek"] = df["timestamp"].dt.dayofweek
df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)

# Lag features
df["power_lag_1"] = df[TARGET_COL].shift(1)
df["power_lag_3"] = df[TARGET_COL].shift(3)
df["power_lag_6"] = df[TARGET_COL].shift(6)

df["irr_lag_1"] = df["irradiance_wm2"].shift(1)
df["irr_lag_3"] = df["irradiance_wm2"].shift(3)

# Rolling
df["power_ma_3"] = df[TARGET_COL].rolling(3).mean()
df["power_ma_12"] = df[TARGET_COL].rolling(12).mean()
df["power_std_12"] = df[TARGET_COL].rolling(12).std()

# Drop NA rows
df = df.dropna().reset_index(drop=True)

# -------------------------
# SELECT FEATURES
# -------------------------
FEATURES = [
    "irradiance_wm2",
    "temp_c",
    "humidity",
    "wind_mps",
    "rain_prob",
    "sin_hour",
    "cos_hour",
    "power_lag_1",
    "power_lag_3",
    "power_lag_6",
    "irr_lag_1",
    "irr_lag_3",
    "power_ma_3",
    "power_ma_12",
    "power_std_12",
]

X = df[FEATURES]
y = df[TARGET_COL]

# -------------------------
# TIME SPLIT (NO SHUFFLE)
# -------------------------
split_idx = int(len(df) * 0.8)

X_train = X.iloc[:split_idx]
y_train = y.iloc[:split_idx]

X_test = X.iloc[split_idx:]
y_test = y.iloc[split_idx:]

# -------------------------
# TRAIN MODEL
# -------------------------
model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------
# EVALUATION
# -------------------------
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# -------------------------
# SAVE MODEL + METADATA
# -------------------------
model_path = os.path.join(MODEL_DIR, "power_forecast_model.pkl")
features_path = os.path.join(MODEL_DIR, "power_forecast_features.json")
meta_path = os.path.join(MODEL_DIR, "power_forecast_meta.json")

joblib.dump(model, model_path)

with open(features_path, "w") as f:
    json.dump(FEATURES, f, indent=2)

with open(meta_path, "w") as f:
    json.dump(
        {
            "mae": float(mae),
            "rmse": float(rmse),
            "target": TARGET_COL,
            "feature_count": len(FEATURES),
        },
        f,
        indent=2,
    )

print("Model training completed successfully.")
print(f"Saved to: {model_path}")

if __name__ == "__main__":
    print("🚀 Starting training...")
    train_model()   # whatever your function name is
    print("✅ Training completed.")

