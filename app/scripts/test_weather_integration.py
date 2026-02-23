import datetime as dt

import redis
from sqlalchemy import select

from app.db import SessionLocal
from app.config import get_settings
from app.models import Plant, PlantTimeseries


def main():
    settings = get_settings()
    r = redis.Redis.from_url(settings.redis_url, decode_responses=True)

    db = SessionLocal()
    try:
        last_row = db.execute(
            select(PlantTimeseries)
            .order_by(PlantTimeseries.timestamp.desc())
        ).scalars().first()
        if not last_row:
            print("No timeseries rows found.")
            return

        plant = db.execute(
            select(Plant).where(Plant.id == last_row.plant_id)
        ).scalars().first()
        if not plant:
            print("Plant not found for last row.")
            return

        ts = last_row.timestamp
        if ts.tzinfo is None:
            ts_utc = ts.replace(tzinfo=dt.timezone.utc)
        else:
            ts_utc = ts.astimezone(dt.timezone.utc)
        bucket_epoch_600s = int(ts_utc.timestamp() // 600) * 600
        lat = float(plant.latitude) if plant.latitude is not None else None
        lon = float(plant.longitude) if plant.longitude is not None else None
        if lat is None or lon is None:
            print("Plant has no coordinates.")
            return

        key = f"weather:{lat:.4f}:{lon:.4f}:{bucket_epoch_600s}"

        latest_key = None
        latest_bucket = None
        for k in r.scan_iter("weather:*"):
            try:
                bucket = int(k.rsplit(":", 1)[-1])
            except ValueError:
                continue
            if latest_bucket is None or bucket > latest_bucket:
                latest_bucket = bucket
                latest_key = k

        print(f"Latest cached weather key: {latest_key}")

        used_real = r.exists(key) == 1 and (last_row.irradiance_wm2 is not None)
        print(f"Simulator used real weather for last row: {bool(used_real)}")
        print(f"Last row timestamp: {last_row.timestamp.isoformat()}, irradiance_wm2={last_row.irradiance_wm2}")
    finally:
        db.close()


if __name__ == "__main__":
    main()

