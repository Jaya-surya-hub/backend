import datetime as dt
import json
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

import httpx

from .config import get_settings

settings = get_settings()

try:
    from redis.asyncio import Redis
    _redis_client: Optional["Redis"] = None

    def get_redis() -> Optional["Redis"]:
        global _redis_client
        if _redis_client is None:
            try:
                _redis_client = Redis.from_url(
                    settings.redis_url, encoding="utf-8", decode_responses=True
                )
            except Exception:
                _redis_client = None
        return _redis_client
except Exception:
    Redis = None

    def get_redis() -> None:
        return None


async def fetch_and_cache_weather(lat: float, lon: float, tz: str, timestamp: dt.datetime) -> Optional[Dict[str, Any]]:
    r = get_redis()
    try:
        tzinfo = ZoneInfo(tz)
    except Exception:
        tzinfo = ZoneInfo("UTC")
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=tzinfo)
    ts_utc = timestamp.astimezone(dt.timezone.utc)
    bucket_epoch_600s = int(ts_utc.timestamp() // 600) * 600
    cache_key = f"weather:{lat:.4f}:{lon:.4f}:{bucket_epoch_600s}"
    if r:
        try:
            cached = await r.get(cache_key)
        except Exception:
            cached = None
        if cached:
            try:
                return json.loads(cached)
            except Exception:
                pass

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "shortwave_radiation,temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation_probability",
        "timezone": tz,
        "past_days": 1,
        "forecast_days": 2,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get("https://api.open-meteo.com/v1/forecast", params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        return None

    hourly = data.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return None

    target_local = timestamp.astimezone(tzinfo)
    best_idx = 0
    best_diff = None
    for i, t in enumerate(times):
        try:
            t_local = dt.datetime.fromisoformat(t)
            t_local = t_local.replace(tzinfo=tzinfo)
        except Exception:
            continue
        diff = abs((t_local - target_local).total_seconds())
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_idx = i

    def pick(name: str) -> Optional[float]:
        seq = hourly.get(name) or []
        if best_idx < len(seq):
            v = seq[best_idx]
            if v is None:
                return None
            return float(v)
        return None

    out = {
        "irradiance_wm2": pick("shortwave_radiation"),
        "temp_c": pick("temperature_2m"),
        "humidity": pick("relative_humidity_2m"),
        "wind_mps": (pick("wind_speed_10m") or 0.0) / 3.6 if pick("wind_speed_10m") is not None else None,
        "rain_prob": pick("precipitation_probability"),
    }

    if r:
        try:
            await r.set(cache_key, json.dumps(out), ex=600)
        except Exception:
            pass

    return out


async def refresh_weather_for_plant(plant) -> Optional[Dict[str, Any]]:
    if plant.latitude is None or plant.longitude is None or not plant.timezone:
        return None
    now_utc = dt.datetime.now(dt.timezone.utc)
    try:
        tzinfo = ZoneInfo(plant.timezone)
    except Exception:
        tzinfo = ZoneInfo("UTC")
    now_local = now_utc.astimezone(tzinfo)
    return await fetch_and_cache_weather(float(plant.latitude), float(plant.longitude), plant.timezone, now_local)
