import os
from pathlib import Path

import joblib
import pandas as pd
from prophet import Prophet


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]

CSV_PATH = PROJECT_ROOT / "backend" / "app" / "ml_train" / "wiztric_prophet_power_5min.csv"
MODEL_DIR = PROJECT_ROOT / "backend" / "app" / "ml_models"
MODEL_PATH = MODEL_DIR / "prophet_power_model.pkl"
FORECAST_PATH = PROJECT_ROOT / "backend" / "app" / "ml_train" / "prophet_daily_forecast.csv"


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    df["ds"] = pd.to_datetime(df["ds"], utc=True)
    df["ds"] = df["ds"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df = df.sort_values("ds").reset_index(drop=True)

    df = df.dropna(subset=["ds", "y"]).reset_index(drop=True)

    df = df[df["y"] >= 0].reset_index(drop=True)

    df_daily = df.resample("D", on="ds").sum().reset_index()
    df_daily = df_daily.rename(columns={"ds": "ds", "y": "y"})

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
    )

    model.fit(df_daily)

    future = model.make_future_dataframe(periods=30, freq="D")
    forecast = model.predict(future)

    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(FORECAST_PATH, index=False)

    joblib.dump(model, MODEL_PATH)


if __name__ == "__main__":
    main()
