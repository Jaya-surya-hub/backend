from pathlib import Path

import joblib
import pandas as pd
from prophet import Prophet


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]

MODEL_PATH = PROJECT_ROOT / "backend" / "app" / "ml_models" / "prophet_power_model.pkl"
FORECAST_OUT = PROJECT_ROOT / "backend" / "app" / "ml_train" / "prophet_daily_forecast_future.csv"


def main(days_ahead: int = 30) -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Prophet model not found: {MODEL_PATH}")

    model: Prophet = joblib.load(MODEL_PATH)

    future = model.make_future_dataframe(periods=days_ahead, freq="D")
    forecast = model.predict(future)

    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(FORECAST_OUT, index=False)


if __name__ == "__main__":
    main()

