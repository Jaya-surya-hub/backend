import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

FEATURES = [
    "irradiance_wm2", "temp_c", "humidity", "wind_mps", "rain_prob",
    "inv1_ac_kw", "inv1_dc_kw", "inv1_eff"
]

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df["hour"] = ts.dt.hour + ts.dt.minute / 60.0
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    return df

def build_frame(df: pd.DataFrame):
    df = df.copy()
    df = add_time_features(df)

    # remove night rows
    df = df[df["irradiance_wm2"].fillna(0) > 50].copy()

    # derived feature
    df["power_to_irr"] = df["active_power_kw"] / df["irradiance_wm2"].clip(lower=1)

    feature_cols = FEATURES + ["sin_hour", "cos_hour", "power_to_irr"]

    for c in feature_cols + ["active_power_kw"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=feature_cols + ["active_power_kw"])
    return df, feature_cols

def main():
    print("✅ TRAIN SCRIPT STARTED")

    here = Path(__file__).resolve()
    # ml_train -> app -> backend -> project root
    project_root = here.parents[3]
    csv_path = project_root / "solar_week_5min.csv"

    models_dir = project_root / "backend" / "app" / "ml_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "power_baseline.pkl"
    meta_path = models_dir / "power_baseline_meta.json"

    print("📌 CSV:", csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print("✅ Raw rows:", len(df))

    df, feature_cols = build_frame(df)
    print("✅ Clean rows:", len(df))
    if len(df) < 200:
        raise RuntimeError("Not enough usable rows after cleaning. Generate more data.")

    X = df[feature_cols]
    y = df["active_power_kw"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("✅ Training RandomForest...")
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    r2 = model.score(X_test, y_test)
    print("✅ Done. Test R2:", round(r2, 3))

    joblib.dump(model, model_path)
    with open(meta_path, "w") as f:
        json.dump({"feature_cols": feature_cols}, f, indent=2)

    print("✅ Saved:", model_path)
    print("✅ Saved:", meta_path)

if __name__ == "__main__":
    main()
