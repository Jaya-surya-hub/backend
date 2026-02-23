import asyncio
import json
import os
import datetime as dt
from typing import List, Optional
from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from sse_starlette.sse import EventSourceResponse
from ..db import get_session
from ..models import Plant, PlantTimeseries, PlantDailySummary, ReportJob, Alert
from ..auth import get_current_user
from ..ml import MLService, predict_baseline_kw
from ..config import get_settings
from ..weather_service import refresh_weather_for_plant, get_redis
from ..ml_runtime.wiztric_sentinel import WiztricSentinel
import httpx
import json as pyjson
import io
import csv

settings = get_settings()
try:
    from redis.asyncio import Redis
    _redis_client = None

    def get_redis():
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

    def get_redis():
        return None

router = APIRouter()


class AIChatHistoryItem(BaseModel):
    role: str
    content: str


class AIChatRequest(BaseModel):
    plant_id: int
    message: str
    timestamp: Optional[dt.datetime] = None
    request_id: Optional[str] = None
    history: List[AIChatHistoryItem] = []


@router.post("/ai/chat")
async def ai_chat(
    payload: AIChatRequest,
    session: AsyncSession = Depends(get_session),
    user=Depends(get_current_user)
):
    plant_stmt = select(Plant).where(Plant.id == payload.plant_id, Plant.tenant_id == user.tenant_id)
    plant_res = await session.execute(plant_stmt)
    plant = plant_res.scalar_one_or_none()
    if not plant:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Plant not found")

    ts_stmt = select(PlantTimeseries).where(
        PlantTimeseries.plant_id == plant.id,
        PlantTimeseries.tenant_id == user.tenant_id
    )
    if payload.timestamp:
        ts_stmt = ts_stmt.where(PlantTimeseries.timestamp <= payload.timestamp)
    ts_stmt = ts_stmt.order_by(PlantTimeseries.timestamp.desc()).limit(1)
    ts_res = await session.execute(ts_stmt)
    point = ts_res.scalar_one_or_none()

    if not point:
        return {
            "answer": f"I could not find any data yet for plant {plant.name}. Once the simulator ingests a few points, I can analyse performance for you.",
            "timestamp": None,
            "plant_id": plant.id
        }

    baseline = predict_baseline_kw(point)
    expected = baseline.get("expected_kw", float(point.active_power_kw or 0.0))
    actual = float(point.active_power_kw or 0.0)
    residual = actual - expected
    ratio = actual / expected if expected > 0 else 1.0

    ts_str = point.timestamp.isoformat()
    perf_phrase = ""
    if expected < 1.0:
        perf_phrase = "It is effectively night-time, so low power is expected."
    elif ratio >= 0.9 and ratio <= 1.1:
        perf_phrase = "Output is very close to the baseline model expectation."
    elif ratio < 0.8:
        perf_phrase = "Output is significantly below the expected level; this may indicate soiling, temperature losses, or availability issues."
    elif ratio > 1.2:
        perf_phrase = "Output is above the expected level; conditions are better than average or the model is conservative."

    irr = float(point.irradiance_wm2 or 0.0)
    temp = float(point.temp_c or 0.0)
    humidity = float(point.humidity or 0.0)

    env_summary_options = [
        f"Irradiance is about {irr:.0f} W/m², module temperature is around {temp:.1f}°C, and humidity is near {humidity:.0f}%.",
        f"Right now the array sees roughly {irr:.0f} W/m², modules are at {temp:.1f}°C and ambient humidity is close to {humidity:.0f}%.",
    ]

    intro_options = [
        f"For plant {plant.name} at {ts_str}, the actual active power is about {actual:.1f} kW while the baseline model is around {expected:.1f} kW.",
        f"Looking at {plant.name} at {ts_str}, measured active power is roughly {actual:.1f} kW versus a baseline of {expected:.1f} kW.",
    ]
    ratio_phrase_options = [
        f"This corresponds to about {ratio*100:.0f}% of the expected output.",
        f"That works out to roughly {ratio*100:.0f}% of what the model predicts.",
    ]

    simple_summary_options = []
    if expected < 1.0:
        simple_summary_options = [
            "In simple terms: it is night-time for this plant, so zero or very low power is completely normal.",
            "To summarise today right now: the plant is in dark hours, so the inverter output is expected to sit at 0 kW.",
        ]
    elif ratio >= 0.9:
        simple_summary_options = [
            "In simple terms: the plant is performing close to what we expect for the current conditions.",
            "Overall summary: generation is broadly in line with the baseline model right now.",
        ]
    elif ratio < 0.8:
        simple_summary_options = [
            "In simple terms: the plant is underperforming compared to the model; this could be due to soiling, partial shading, high temperature losses or availability issues.",
            "Overall summary: power is noticeably below expectation, so it would be good to review cleaning, outages and inverter alarms.",
        ]
    else:
        simple_summary_options = [
            "In simple terms: the plant is performing slightly off from the model but still within a reasonable band.",
            "Overall summary: performance is acceptable with some deviation from the baseline curve.",
        ]

    import random

    intro = random.choice(intro_options)
    ratio_phrase = random.choice(ratio_phrase_options)
    env_summary = random.choice(env_summary_options)
    simple_summary = random.choice(simple_summary_options) if simple_summary_options else ""

    persona_name = "Wiztric Sentinel"

    fallback_answer = (
        f"I am {persona_name}, your Wiztric performance copilot. "
        f"{intro} "
        f"{ratio_phrase} "
        f"{perf_phrase} {env_summary} "
        f"{simple_summary}"
    )

    plant_context = {
        "plant_id": plant.id,
        "plant_name": plant.name,
        "timestamp": ts_str,
        "actual_kw": actual,
        "expected_kw": expected,
        "residual_kw": residual,
        "ratio": ratio,
        "irradiance_wm2": irr,
        "temp_c": temp,
        "humidity": humidity,
        "wind_mps": float(point.wind_mps or 0.0),
        "rain_prob": float(point.rain_prob or 0.0),
        "alarm_code": point.alarm_code,
        "severity": point.severity,
    }

    answer = fallback_answer
    try:
        sentinel = WiztricSentinel()
        answer = await sentinel.ask(payload.message, plant_context)
    except Exception:
        answer = fallback_answer

    history_key = f"ai_chat:{user.id}:{plant.id}"
    r = get_redis()
    if r:
        try:
            raw = await r.get(history_key)
            existing = pyjson.loads(raw) if raw else []
        except Exception:
            existing = []
        existing.append(
            {
                "role": "user",
                "content": payload.message,
                "timestamp": ts_str,
            }
        )
        existing.append(
            {
                "role": "assistant",
                "content": answer,
                "timestamp": ts_str,
            }
        )
        trimmed = existing[-10:]
        try:
            await r.set(history_key, pyjson.dumps(trimmed), ex=3600)
        except Exception:
            pass

    return {
        "answer": answer,
        "plant_id": plant.id,
        "timestamp": ts_str,
        "actual_kw": actual,
        "expected_kw": expected,
        "residual_kw": residual,
        "ratio": ratio,
        "request_id": payload.request_id,
        "assistant_name": persona_name,
        "conditions": {
            "irradiance_wm2": irr,
            "temp_c": temp,
            "humidity": humidity,
            "wind_mps": float(point.wind_mps or 0.0),
            "rain_prob": float(point.rain_prob or 0.0),
        }
    }


@router.post("/debug/weather/refresh")
async def debug_weather_refresh(
    plant_id: int,
    session: AsyncSession = Depends(get_session),
    user=Depends(get_current_user)
):
    stmt = select(Plant).where(Plant.id == plant_id, Plant.tenant_id == user.tenant_id)
    res = await session.execute(stmt)
    plant = res.scalar_one_or_none()
    if not plant:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Plant not found")

    weather = await refresh_weather_for_plant(plant)

    if plant.latitude is None or plant.longitude is None:
        return {"plant_id": plant.id, "cache_key": None, "weather": weather}

    now_utc = dt.datetime.now(dt.timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        tzinfo = ZoneInfo(plant.timezone or "UTC")
    except Exception:
        tzinfo = dt.timezone.utc
    now_local = now_utc.astimezone(tzinfo)
    ts_utc = now_local.astimezone(dt.timezone.utc)
    bucket_epoch_600s = int(ts_utc.timestamp() // 600) * 600
    lat = float(plant.latitude)
    lon = float(plant.longitude)
    cache_key = f"weather:{lat:.4f}:{lon:.4f}:{bucket_epoch_600s}"

    r = get_redis()
    cached_value = None
    if r:
        try:
            cached_raw = await r.get(cache_key)
            if cached_raw:
                cached_value = pyjson.loads(cached_raw)
        except Exception:
            cached_value = None

    return {
        "plant_id": plant.id,
        "cache_key": cache_key,
        "weather": weather,
        "cached_value": cached_value,
    }

# Analytics Endpoints
@router.get("/analytics/health/{plant_id}")
async def get_plant_health(
    plant_id: int,
    user=Depends(get_current_user)
):
    score = MLService.get_health_score(plant_id, user.tenant_id)
    return {"plant_id": plant_id, "health_score": score}

@router.get("/analytics/forecast/{plant_id}")
async def get_plant_forecast(
    plant_id: int,
    horizon: int = Query(24, ge=1, le=168),
    user=Depends(get_current_user)
):
    forecast = MLService.get_forecast(plant_id, user.tenant_id, horizon=horizon)
    return forecast

@router.get("/analytics/anomalies/{plant_id}")
async def get_plant_anomalies(
    plant_id: int,
    user=Depends(get_current_user)
):
    anomalies = MLService.detect_anomalies(plant_id, user.tenant_id)
    return anomalies

@router.get("/analytics/explain/{plant_id}")
async def explain_point(
    plant_id: int,
    timestamp: dt.datetime = Query(...),
    user=Depends(get_current_user)
):
    explanation = MLService.explain(plant_id, user.tenant_id, timestamp)
    return explanation

# SSE Stream for Live Data
@router.get("/stream/plant/{plant_id}")
async def stream_plant_data(
    request: Request,
    plant_id: int,
    session: AsyncSession = Depends(get_session),
):
    async def event_generator():
        last_id = 0
        while True:
            if await request.is_disconnected():
                break

            # Query for new data points since last_id
            stmt = (
                select(PlantTimeseries)
                .where(PlantTimeseries.plant_id == plant_id)
                .where(PlantTimeseries.id > last_id)
                .order_by(PlantTimeseries.id.desc())
                .limit(1)
            )
            res = await session.execute(stmt)
            point = res.scalar_one_or_none()

            if point:
                last_id = point.id
                yield {
                    "event": "message",
                    "id": str(point.id),
                    "data": json.dumps({
                        "timestamp": point.timestamp.isoformat(),
                        "active_power_kw": point.active_power_kw,
                        "energy_kwh": point.energy_kwh,
                        "daily_yield_kwh": point.daily_yield_kwh,
                        "irradiance_wm2": point.irradiance_wm2,
                        "temp_c": point.temp_c,
                        "humidity": point.humidity,
                        "wind_mps": point.wind_mps,
                        "rain_prob": point.rain_prob,
                        "alarm_code": point.alarm_code,
                        "severity": point.severity
                    })
                }
            
            await asyncio.sleep(5) # Check every 5 seconds

    return EventSourceResponse(event_generator())

# Standard REST Endpoints
@router.get("/plants")
async def list_plants(
    session: AsyncSession = Depends(get_session),
):
    stmt = select(Plant)
    res = await session.execute(stmt)
    return res.scalars().all()

@router.get("/health/db")
async def health_db(
    session: AsyncSession = Depends(get_session),
    user=Depends(get_current_user)
):
    counts = {}
    for model, key in [(PlantTimeseries, "plant_timeseries"), (PlantDailySummary, "plant_daily_summary")]:
        stmt = select(func.count(model.id)).where(model.tenant_id == user.tenant_id)
        res = await session.execute(stmt)
        counts[key] = res.scalar() or 0
    return counts

@router.get("/health/ingestion")
async def health_ingestion(
    session: AsyncSession = Depends(get_session),
    user=Depends(get_current_user)
):
    # Last insert timestamp
    last_ts_stmt = select(func.max(PlantTimeseries.timestamp)).where(
        PlantTimeseries.tenant_id == user.tenant_id
    )
    res = await session.execute(last_ts_stmt)
    last_ts = res.scalar()
    # Rows last hour
    one_hour_ago = dt.datetime.utcnow() - dt.timedelta(hours=1)
    rows_stmt = select(func.count(PlantTimeseries.id)).where(
        PlantTimeseries.tenant_id == user.tenant_id,
        PlantTimeseries.timestamp >= one_hour_ago
    )
    res2 = await session.execute(rows_stmt)
    rows_last_hour = res2.scalar() or 0
    return {
        "last_insert_timestamp": last_ts.isoformat() if last_ts else None,
        "rows_last_hour": int(rows_last_hour)
    }


@router.get("/export/timeseries")
async def export_timeseries_csv(
    plant_id: int,
    session: AsyncSession = Depends(get_session),
    user=Depends(get_current_user)
):
    stmt = (
        select(PlantTimeseries)
        .where(PlantTimeseries.tenant_id == user.tenant_id)
        .where(PlantTimeseries.plant_id == plant_id)
        .order_by(PlantTimeseries.timestamp.asc())
    )
    res = await session.execute(stmt)
    rows = res.scalars().all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "timestamp",
            "active_power_kw",
            "energy_kwh",
            "daily_yield_kwh",
            "irradiance_wm2",
            "temp_c",
            "humidity",
            "wind_mps",
            "rain_prob",
        ]
    )
    for p in rows:
        writer.writerow(
            [
                p.timestamp.isoformat(),
                p.active_power_kw,
                p.energy_kwh,
                p.daily_yield_kwh,
                p.irradiance_wm2,
                p.temp_c,
                p.humidity,
                p.wind_mps,
                p.rain_prob,
            ]
        )
    return output.getvalue()

@router.get("/weather/current")
async def weather_current(
    plant_id: int,
    session: AsyncSession = Depends(get_session),
):
    stmt = select(Plant).where(Plant.id == plant_id)
    res = await session.execute(stmt)
    plant = res.scalar_one_or_none()
    if not plant or plant.latitude is None or plant.longitude is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Plant or coordinates not found")
    lat, lon = float(plant.latitude), float(plant.longitude)
    cache_key = f"weather:current:{lat:.4f}:{lon:.4f}"
    r = get_redis()
    if r:
        try:
            cached = await r.get(cache_key)
        except Exception:
            cached = None
        if cached:
            try:
                return pyjson.loads(cached)
            except Exception:
                cached = None
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": True,
        "hourly": "relative_humidity_2m,temperature_2m,wind_speed_10m,shortwave_radiation,precipitation_probability"
    }
    out = {
        "temperature_c": None,
        "wind_speed_mps": None,
        "humidity_pct": None,
        "rain_prob": None,
        "irradiance_wm2": None
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params)
            data = resp.json()
        out = {
            "temperature_c": data.get("current_weather", {}).get("temperature"),
            "wind_speed_mps": (data.get("current_weather", {}).get("windspeed", 0) / 3.6) if data.get("current_weather") else None,
            "humidity_pct": (data.get("hourly", {}).get("relative_humidity_2m") or [None])[-1],
            "rain_prob": (data.get("hourly", {}).get("precipitation_probability") or [None])[-1],
            "irradiance_wm2": (data.get("hourly", {}).get("shortwave_radiation") or [None])[-1]
        }
    except Exception:
        # Fallback: derive a \"current\" weather snapshot from the latest timeseries point
        ts_stmt = (
            select(PlantTimeseries)
            .where(PlantTimeseries.plant_id == plant.id)
            .order_by(PlantTimeseries.timestamp.desc())
            .limit(1)
        )
        ts_res = await session.execute(ts_stmt)
        point = ts_res.scalar_one_or_none()
        if point:
            out = {
                "temperature_c": float(point.temp_c or 25.0),
                "wind_speed_mps": float(point.wind_mps or 1.0),
                "humidity_pct": float(point.humidity or 50.0),
                "rain_prob": float(point.rain_prob or 0.0),
                "irradiance_wm2": float(point.irradiance_wm2 or 0.0),
            }
    if r:
        try:
            await r.set(cache_key, pyjson.dumps(out), ex=300)
        except Exception:
            pass
    return out

@router.get("/weather/forecast")
async def weather_forecast(
    plant_id: int,
    session: AsyncSession = Depends(get_session),
):
    stmt = select(Plant).where(Plant.id == plant_id)
    res = await session.execute(stmt)
    plant = res.scalar_one_or_none()
    if not plant or plant.latitude is None or plant.longitude is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Plant or coordinates not found")
    lat, lon = float(plant.latitude), float(plant.longitude)
    cache_key = f"weather:forecast:{lat:.4f}:{lon:.4f}"
    r = get_redis()
    if r:
        try:
            cached = await r.get(cache_key)
        except Exception:
            cached = None
        if cached:
            try:
                return pyjson.loads(cached)
            except Exception:
                cached = None
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "relative_humidity_2m,temperature_2m,wind_speed_10m,shortwave_radiation,precipitation_probability"
    }
    series = []
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params)
            data = resp.json()
        hours = data.get("hourly", {})
        timestamps = hours.get("time", []) or []
        for i, t in enumerate(timestamps):
            series.append({
                "timestamp": t,
                "temperature_c": (hours.get("temperature_2m") or [None])[i],
                "wind_speed_mps": ((hours.get("wind_speed_10m") or [0])[i]) / 3.6 if hours.get("wind_speed_10m") else None,
                "humidity_pct": (hours.get("relative_humidity_2m") or [None])[i],
                "rain_prob": (hours.get("precipitation_probability") or [None])[i],
                "irradiance_wm2": (hours.get("shortwave_radiation") or [None])[i],
            })
    except Exception:
        # Fallback: build a simple forecast from the last 24h of simulated data
        ts_stmt = (
            select(PlantTimeseries)
            .where(PlantTimeseries.plant_id == plant.id)
            .order_by(PlantTimeseries.timestamp.desc())
            .limit(24)
        )
        ts_res = await session.execute(ts_stmt)
        rows = list(reversed(ts_res.scalars().all()))
        for p in rows:
            series.append({
                "timestamp": p.timestamp.isoformat(),
                "temperature_c": float(p.temp_c or 25.0),
                "wind_speed_mps": float(p.wind_mps or 1.0),
                "humidity_pct": float(p.humidity or 50.0),
                "rain_prob": float(p.rain_prob or 0.0),
                "irradiance_wm2": float(p.irradiance_wm2 or 0.0),
            })
    out = {"latitude": lat, "longitude": lon, "series": series}
    if r:
        try:
            await r.set(cache_key, pyjson.dumps(out), ex=300)
        except Exception:
            pass
    return out
@router.get("/plant/{plant_id}/timeseries")
async def get_timeseries(
    plant_id: int,
    start: Optional[dt.datetime] = Query(None),
    end: Optional[dt.datetime] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
):
    stmt = select(PlantTimeseries).where(PlantTimeseries.plant_id == plant_id)
    if start:
        stmt = stmt.where(PlantTimeseries.timestamp >= start)
    if end:
        stmt = stmt.where(PlantTimeseries.timestamp <= end)
    
    stmt = stmt.order_by(PlantTimeseries.timestamp.desc()).limit(limit).offset(offset)
    res = await session.execute(stmt)
    return res.scalars().all()

@router.get("/plant/{plant_id}/daily")
async def get_daily_history(
    plant_id: int,
    days: int = Query(30, ge=1, le=365),
    session: AsyncSession = Depends(get_session),
    user=Depends(get_current_user)
):
    start_date = dt.date.today() - dt.timedelta(days=days)
    stmt = (
        select(PlantDailySummary)
        .where(PlantDailySummary.plant_id == plant_id)
        .where(PlantDailySummary.tenant_id == user.tenant_id)
        .where(PlantDailySummary.date >= start_date)
        .order_by(PlantDailySummary.date.desc())
    )
    res = await session.execute(stmt)
    return res.scalars().all()

@router.get("/reports")
async def list_reports(
    session: AsyncSession = Depends(get_session),
    user=Depends(get_current_user)
):
    stmt = select(ReportJob).where(ReportJob.tenant_id == user.tenant_id).order_by(ReportJob.created_at.desc())
    res = await session.execute(stmt)
    return res.scalars().all()

@router.get("/reports/{report_id}/download")
async def download_report(
    report_id: int,
    session: AsyncSession = Depends(get_session),
    user=Depends(get_current_user)
):
    from fastapi.responses import FileResponse
    stmt = select(ReportJob).where(ReportJob.id == report_id).where(ReportJob.tenant_id == user.tenant_id)
    res = await session.execute(stmt)
    report = res.scalar_one_or_none()
    if not report or not os.path.exists(report.file_path):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Report file not found")
    
    return FileResponse(
        report.file_path, 
        media_type='text/csv', 
        filename=os.path.basename(report.file_path)
    )

@router.post("/reports/generate")
async def manual_generate_report(
    plant_id: int,
    start_date: dt.date,
    end_date: dt.date,
    user=Depends(get_current_user)
):
    from ..jobs.tasks import generate_manual_report
    # This should be async in production, but for demo we can run it
    filepath = generate_manual_report(user.tenant_id, plant_id, start_date, end_date)
    return {"message": "Report generated", "file": filepath}

@router.post("/test/trigger-aggregation")
async def trigger_aggregation(
    date: Optional[dt.date] = Query(None),
    user=Depends(get_current_user)
):
    if user.role != "admin":
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="Admin only")
    
    from ..jobs.tasks import run_daily_aggregation
    run_daily_aggregation(target_date=date)
    return {"message": f"Daily aggregation triggered for {date or 'yesterday'}"}

@router.post("/test/trigger-weekly-reports")
async def trigger_weekly_reports(user=Depends(get_current_user)):
    if user.role != "admin":
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="Admin only")
    
    from ..jobs.tasks import run_weekly_reports
    run_weekly_reports()
    return {"message": "Weekly reports triggered"}

@router.get("/alerts")
async def list_alerts(
    plant_id: Optional[int] = Query(None),
    is_acknowledged: Optional[bool] = Query(None),
    session: AsyncSession = Depends(get_session),
    user=Depends(get_current_user)
):
    stmt = select(Alert).where(Alert.tenant_id == user.tenant_id)
    if plant_id:
        stmt = stmt.where(Alert.plant_id == plant_id)
    if is_acknowledged is not None:
        stmt = stmt.where(Alert.is_acknowledged == is_acknowledged)
    
    stmt = stmt.order_by(Alert.timestamp.desc())
    res = await session.execute(stmt)
    return res.scalars().all()
