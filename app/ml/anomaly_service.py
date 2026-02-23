import datetime as dt
from typing import List, Dict
from sqlalchemy import select, func
from ..db import SessionLocal
from ..models import PlantTimeseries


def detect_anomalies_rule_based(plant_id: int, tenant_id: int) -> List[Dict]:
    db = SessionLocal()
    try:
        one_hour_ago = dt.datetime.utcnow() - dt.timedelta(hours=1)
        stmt = select(PlantTimeseries).where(
            PlantTimeseries.plant_id == plant_id,
            PlantTimeseries.tenant_id == tenant_id,
            PlantTimeseries.timestamp >= one_hour_ago
        ).order_by(PlantTimeseries.timestamp.desc())
        rows = db.execute(stmt).scalars().all()

        anomalies: List[Dict] = []
        for r in rows:
            # Example rules
            if (r.irradiance_wm2 or 0) > 300 and (r.active_power_kw or 0) < 0.05 * (r.irradiance_wm2 / 1000.0):
                anomalies.append({
                    "timestamp": r.timestamp.isoformat(),
                    "type": "Low Output Given Irradiance",
                    "severity": "Warning",
                    "description": "Power is low relative to irradiance."
                })
            if (r.inv1_eff or 0) < 0.85 and (r.active_power_kw or 0) > 50:
                anomalies.append({
                    "timestamp": r.timestamp.isoformat(),
                    "type": "Inverter Efficiency Low",
                    "severity": "Info",
                    "description": "Observed inverter efficiency below threshold."
                })
        return anomalies[-10:]
    finally:
        db.close()

