import datetime as dt
from typing import List, Dict
from sqlalchemy import select, func
from ..db import SessionLocal
from ..models import PlantTimeseries


def forecast_naive(plant_id: int, tenant_id: int, horizon: int = 24) -> List[Dict]:
    db = SessionLocal()
    try:
        # Compute average by hour over last 7 days
        end = dt.datetime.utcnow()
        start = end - dt.timedelta(days=7)
        stmt = select(PlantTimeseries).where(
            PlantTimeseries.plant_id == plant_id,
            PlantTimeseries.tenant_id == tenant_id,
            PlantTimeseries.timestamp >= start,
            PlantTimeseries.timestamp <= end
        )
        rows = db.execute(stmt).scalars().all()
        by_hour = {}
        counts = {}
        for r in rows:
            h = r.timestamp.hour
            by_hour[h] = by_hour.get(h, 0.0) + (r.active_power_kw or 0.0)
            counts[h] = counts.get(h, 0) + 1
        avg_by_hour = {h: (by_hour[h] / counts[h]) for h in by_hour if counts[h] > 0}

        forecast = []
        now = end
        for i in range(1, horizon + 1):
            t = now + dt.timedelta(hours=i)
            val = avg_by_hour.get(t.hour, 0.0)
            forecast.append({
                "timestamp": t.isoformat(),
                "predicted_power_kw": round(val, 2)
            })
        return forecast
    finally:
        db.close()

