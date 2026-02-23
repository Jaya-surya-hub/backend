from typing import Any, Dict, List, Tuple
import datetime as dt
from pathlib import Path
import csv
import json
import math

import numpy as np
import joblib

from .anomaly_service import detect_anomalies_rule_based
from .prediction_stub import forecast_naive
from .shap_explain import explain_at_timestamp


_MODEL_PATH = Path(__file__).resolve().parent.parent / "ml_models" / "power_baseline.pkl"
_META_PATH = Path(__file__).resolve().parent.parent / "ml_models" / "power_baseline_meta.json"
_POWER_MODEL: Any = None
_FEATURE_COLS: List[str] | None = None
_ANOM_THR_P95: float | None = None
_ANOM_THR_P99: float | None = None
_DAY_IRR_MIN: float | None = None
_PROPHET_FORECAST_PATH = Path(__file__).resolve().parent.parent / "ml_train" / "prophet_daily_forecast_future.csv"
_PROPHET_DAILY_FORECAST: Dict[dt.date, Dict[str, float]] | None = None


def get_power_baseline_model() -> Tuple[Any, List[str]]:
    global _POWER_MODEL, _FEATURE_COLS
    if _POWER_MODEL is None:
        try:
            obj = joblib.load(_MODEL_PATH)
            if isinstance(obj, dict) and "model" in obj and "features" in obj:
                _POWER_MODEL = obj["model"]
                _FEATURE_COLS = [str(c) for c in obj.get("features", [])]
                thr95 = obj.get("thr_abs_err_p95")
                thr99 = obj.get("thr_abs_err_p99")
                day_min = obj.get("day_irradiance_min")
                globals()["_ANOM_THR_P95"] = float(thr95) if thr95 is not None else None
                globals()["_ANOM_THR_P99"] = float(thr99) if thr99 is not None else None
                globals()["_DAY_IRR_MIN"] = float(day_min) if day_min is not None else None
            else:
                _POWER_MODEL = obj
        except Exception:
            _POWER_MODEL = None
    if _FEATURE_COLS is None:
        try:
            with open(_META_PATH, "r") as f:
                meta = json.load(f)
            cols = meta.get("feature_cols") or []
            _FEATURE_COLS = [str(c) for c in cols]
        except Exception:
            _FEATURE_COLS = []
    return _POWER_MODEL, _FEATURE_COLS or []


def get_daily_energy_forecast_map() -> Dict[dt.date, Dict[str, float]]:
    global _PROPHET_DAILY_FORECAST
    if _PROPHET_DAILY_FORECAST is not None:
        return _PROPHET_DAILY_FORECAST
    data: Dict[dt.date, Dict[str, float]] = {}
    try:
        with _PROPHET_FORECAST_PATH.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ds_raw = row.get("ds")
                yhat_raw = row.get("yhat")
                if not ds_raw or not yhat_raw:
                    continue
                try:
                    if "T" in ds_raw:
                        ds = dt.datetime.fromisoformat(ds_raw).date()
                    else:
                        ds = dt.datetime.strptime(ds_raw, "%Y-%m-%d").date()
                    yhat = float(yhat_raw)
                    yhat_lower = float(row.get("yhat_lower", yhat_raw))
                    yhat_upper = float(row.get("yhat_upper", yhat_raw))
                except Exception:
                    continue
                data[ds] = {
                    "yhat": yhat,
                    "yhat_lower": yhat_lower,
                    "yhat_upper": yhat_upper,
                }
    except FileNotFoundError:
        data = {}
    _PROPHET_DAILY_FORECAST = data
    return data


def get_daily_energy_forecast_for_date(target_date: dt.date) -> Dict[str, float] | None:
    data = get_daily_energy_forecast_map()
    return data.get(target_date)


def get_daily_energy_forecast_series(days: int) -> List[Dict[str, Any]]:
    mapping = get_daily_energy_forecast_map()
    if not mapping:
        return []
    today = dt.date.today()
    sorted_dates = sorted(mapping.keys())
    if not sorted_dates:
        return []

    # If the CSV contains future dates relative to today, use them directly.
    future_days = [
        d for d in sorted_dates
        if d >= today and d <= today + dt.timedelta(days=days - 1)
    ]
    out: List[Dict[str, Any]] = []
    if future_days:
        for d in future_days:
            vals = mapping[d]
            out.append(
                {
                    "date": d.isoformat(),
                    "yhat": vals["yhat"],
                    "yhat_lower": vals["yhat_lower"],
                    "yhat_upper": vals["yhat_upper"],
                }
            )
        return out

    # Fallback: reuse the historical daily pattern, rebased onto today.
    for i in range(days):
        src_date = sorted_dates[i % len(sorted_dates)]
        vals = mapping[src_date]
        target_date = today + dt.timedelta(days=i)
        out.append(
            {
                "date": target_date.isoformat(),
                "yhat": vals["yhat"],
                "yhat_lower": vals["yhat_lower"],
                "yhat_upper": vals["yhat_upper"],
            }
        )
    return out


def build_feature_vector(point: Any, feature_cols: List[str]) -> List[float]:
    ts = point.timestamp
    hour_float = ts.hour + ts.minute / 60.0
    sin_hour = math.sin(2 * math.pi * hour_float / 24.0)
    cos_hour = math.cos(2 * math.pi * hour_float / 24.0)
    hour = ts.hour
    dow = ts.weekday()
    hour_sin = math.sin(2 * math.pi * hour / 24.0)
    hour_cos = math.cos(2 * math.pi * hour / 24.0)
    irr = float(point.irradiance_wm2 or 0.0)
    active_kw = float(point.active_power_kw or 0.0)
    inv_ac = float(point.inv1_ac_kw or active_kw / 6.0)
    inv_dc = float(point.inv1_dc_kw or (inv_ac / 0.97 if inv_ac > 0 else 0.0))
    if inv_dc > 0:
        inv_eff = float(point.inv1_eff or inv_ac / inv_dc)
    else:
        inv_eff = float(point.inv1_eff or 0.97)
    if irr > 0:
        power_to_irr = active_kw / (irr / 1000.0)
    else:
        power_to_irr = 0.0

    base = {
        "irradiance_wm2": irr,
        "temp_c": float(point.temp_c or 25.0),
        "humidity": float(point.humidity or 50.0),
        "wind_mps": float(point.wind_mps or 1.0),
        "rain_prob": float(point.rain_prob or 0.0),
        "inv1_ac_kw": inv_ac,
        "inv1_dc_kw": inv_dc,
        "inv1_eff": inv_eff,
        "sin_hour": sin_hour,
        "cos_hour": cos_hour,
        "power_to_irr": power_to_irr,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow": float(dow),
    }
    return [float(base.get(name, 0.0)) for name in feature_cols]


def predict_baseline_kw(point: Any) -> Dict[str, float]:
    model, cols = get_power_baseline_model()
    if model is None or not cols:
        return {"expected_kw": float(point.active_power_kw or 0.0)}
    x = np.array([build_feature_vector(point, cols)], dtype=float)
    y = model.predict(x)
    expected = float(y[0])
    return {"expected_kw": expected}


class MLService:
    @staticmethod
    def get_health_score(plant_id: int, tenant_id: int) -> float:
        try:
            anomalies = detect_anomalies_rule_based(plant_id, tenant_id) or []
            penalty = min(len(anomalies) * 1.5, 30.0)
            return max(0.0, min(100.0, 98.5 - penalty))
        except Exception:
            return 98.4

    @staticmethod
    def get_forecast(plant_id: int, tenant_id: int, horizon: int = 24) -> List[Dict[str, Any]]:
        days = max(1, horizon)
        return get_daily_energy_forecast_series(days)

    @staticmethod
    def detect_anomalies(plant_id: int, tenant_id: int) -> List[Dict[str, Any]]:
        return detect_anomalies_rule_based(plant_id, tenant_id)

    @staticmethod
    def explain(plant_id: int, tenant_id: int, timestamp: dt.datetime) -> Dict[str, Any]:
        return explain_at_timestamp(plant_id, tenant_id, timestamp)
