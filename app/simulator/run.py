import time
import random
import math
import datetime as dt
import os
import structlog
from typing import List, Dict, Any, Optional
from zoneinfo import ZoneInfo

import redis

from sqlalchemy.orm import Session
from sqlalchemy import select, func
from ..db import SessionLocal, sync_engine
from ..models import Plant, PlantTimeseries, Tenant, Alert
from ..config import get_settings

# Configure structlog
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
log = structlog.get_logger()

settings = get_settings()
_redis_sync: Optional["redis.Redis"] = None


def get_sync_redis() -> Optional["redis.Redis"]:
    global _redis_sync
    if _redis_sync is None:
        try:
            _redis_sync = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        except Exception:
            _redis_sync = None
    return _redis_sync

class AdvancedSolarSimulator:
    def __init__(self):
        self.enabled = os.getenv("SIMULATOR_ENABLED", "true").lower() == "true"
        self.speed = float(os.getenv("SIMULATOR_SPEED", "1.0"))
        self.mode = os.getenv("SIMULATOR_MODE", "real") # real or fast
        
        self.weather_state = {}
        self.outage_state = {}

    def get_cached_weather(self, plant: Plant, timestamp: dt.datetime) -> Optional[Dict[str, Any]]:
        r = get_sync_redis()
        if not r:
            return None
        if timestamp.tzinfo is None:
            ts_utc = timestamp.replace(tzinfo=dt.timezone.utc)
        else:
            ts_utc = timestamp.astimezone(dt.timezone.utc)
        bucket_epoch_600s = int(ts_utc.timestamp() // 600) * 600
        lat = float(plant.latitude) if plant.latitude is not None else None
        lon = float(plant.longitude) if plant.longitude is not None else None
        if lat is None or lon is None:
            return None
        cache_key = f"weather:{lat:.4f}:{lon:.4f}:{bucket_epoch_600s}"
        try:
            data = r.get(cache_key)
        except Exception:
            return None
        if not data:
            return None
        try:
            import json
            return json.loads(data)
        except Exception:
            return None

    def get_solar_intensity(self, timestamp: dt.datetime, plant: Plant) -> float:
        try:
            tzinfo = ZoneInfo(plant.timezone or "UTC")
        except Exception:
            tzinfo = ZoneInfo("UTC")
        if timestamp.tzinfo is None:
            ts_local = timestamp.replace(tzinfo=dt.timezone.utc).astimezone(tzinfo)
        else:
            ts_local = timestamp.astimezone(tzinfo)
        day_of_year = ts_local.timetuple().tm_yday
        seasonal_factor = 0.8 + 0.2 * math.cos(2 * math.pi * (day_of_year - 172) / 365)
        hour = ts_local.hour + ts_local.minute / 60.0
        sunrise = 6.0
        sunset = 18.0
        if hour < sunrise or hour > sunset:
            return 0.0
        daylight_fraction = (hour - sunrise) / (sunset - sunrise)
        base = math.sin(math.pi * daylight_fraction)
        midday_boost = math.exp(-((hour - 13.0) ** 2) / (2.0 * 1.0 ** 2))
        scaled = base * seasonal_factor * (1.0 + 0.05 * midday_boost)
        noise = random.gauss(0.0, 0.02)
        intensity = scaled * (1.0 + noise)
        if intensity < 0.0:
            intensity = 0.0
        return intensity

    def simulate_weather(self, plant_id: int, base_intensity: float) -> Dict[str, Any]:
        state = self.weather_state.get(plant_id, {"is_cloudy": False, "duration": 0})
        if state["duration"] <= 0:
            state["is_cloudy"] = random.random() < 0.15
            state["duration"] = random.randint(3, 12)
        else:
            state["duration"] -= 1
        self.weather_state[plant_id] = state
        cloud_factor = random.uniform(0.3, 0.6) if state["is_cloudy"] else random.uniform(0.95, 1.0)
        actual_intensity = base_intensity * cloud_factor
        irradiance = 1000.0 * actual_intensity
        temp_c = 20.0 + 20.0 * base_intensity + (random.uniform(-5.0, 5.0) if state["is_cloudy"] else random.uniform(-2.0, 2.0))
        return {
            "irradiance": irradiance,
            "temp_c": temp_c,
            "humidity": 40 + 40 * (1 - actual_intensity) + random.uniform(-5, 5),
            "wind_mps": 2 + random.uniform(0, 8),
            "rain_prob": 0.8 if state["is_cloudy"] and random.random() < 0.3 else 0.05
        }

    def simulate_inverters(self, plant_id: int, total_power: float) -> List[Dict[str, Any]]:
        """Simulates 6 inverters with random outages."""
        inverters = []
        outages = self.outage_state.get(plant_id, {})
        
        for i in range(1, 7):
            # 0.1% chance of an inverter going down if not already down
            if i not in outages and random.random() < 0.001:
                outages[i] = random.randint(12, 48) # 1-4 hours
                log.warning("inverter_outage_started", plant_id=plant_id, inverter=i)
            
            # Check if outage is over
            if i in outages:
                outages[i] -= 1
                if outages[i] <= 0:
                    del outages[i]
                    log.info("inverter_outage_ended", plant_id=plant_id, inverter=i)
            
            is_down = i in outages
            
            if is_down:
                inverters.append({"ac_kw": 0.0, "dc_kw": 0.0, "eff": 0.0})
            else:
                # Share total power, add small efficiency variation
                ac_kw = (total_power / 6) * random.uniform(0.98, 1.02)
                eff = 0.96 + random.uniform(0.01, 0.02)
                inverters.append({
                    "ac_kw": ac_kw,
                    "dc_kw": ac_kw / eff,
                    "eff": eff
                })
        
        self.outage_state[plant_id] = outages
        return inverters

    def generate_point(self, db: Session, plant: Plant, timestamp: dt.datetime) -> PlantTimeseries:
        weather_real = self.get_cached_weather(plant, timestamp)
        if weather_real and weather_real.get("irradiance_wm2") is not None:
            irr = float(weather_real.get("irradiance_wm2") or 0.0)
            temp_c = float(weather_real.get("temp_c") or 25.0)
            humidity = float(weather_real.get("humidity") or 50.0)
            wind_mps = float(weather_real.get("wind_mps") or 0.0)
            rain_prob = float(weather_real.get("rain_prob") or 0.0)
            capacity_factor = irr / 1000.0
            if capacity_factor <= 0.0:
                active_power = 0.0
                inverters = self.simulate_inverters(plant.id, active_power)
            else:
                temp_coeff = 1.0 - max(0.0, temp_c - 25.0) * 0.004
                system_loss = 0.92 + random.uniform(-0.02, 0.01)
                active_power = plant.capacity_kw * capacity_factor * temp_coeff * system_loss
                if active_power < 0.0:
                    active_power = 0.0
                inverters = self.simulate_inverters(plant.id, active_power)
        else:
            intensity = self.get_solar_intensity(timestamp, plant)
            weather_sim = self.simulate_weather(plant.id, intensity)
            if intensity <= 0.0:
                active_power = 0.0
                inverters = self.simulate_inverters(plant.id, active_power)
            else:
                temp_coeff = 1.0 - max(0.0, weather_sim["temp_c"] - 25.0) * 0.004
                system_loss = 0.92 + random.uniform(-0.02, 0.01)
                capacity_factor = max(0.0, min(1.2, weather_sim["irradiance"] / 1000.0))
                active_power = plant.capacity_kw * capacity_factor * temp_coeff * system_loss
                if active_power < 0.0:
                    active_power = 0.0
                inverters = self.simulate_inverters(plant.id, active_power)
            irr = weather_sim["irradiance"]
            temp_c = weather_sim["temp_c"]
            humidity = weather_sim["humidity"]
            wind_mps = weather_sim["wind_mps"]
            rain_prob = weather_sim["rain_prob"]
        active_power = sum(inv["ac_kw"] for inv in inverters)
        energy_interval = active_power * (5 / 60) # 5 minutes
        
        # Calculate daily yield
        start_of_day = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        daily_sum = db.query(func.sum(PlantTimeseries.energy_kwh)).filter(
            PlantTimeseries.plant_id == plant.id,
            PlantTimeseries.timestamp >= start_of_day
        ).scalar() or 0.0
        
        inv_data = {f"inv{i+1}": inv for i, inv in enumerate(inverters)}
        
        point = PlantTimeseries(
            tenant_id=plant.tenant_id,
            plant_id=plant.id,
            timestamp=timestamp,
            active_power_kw=active_power,
            energy_kwh=energy_interval,
            daily_yield_kwh=float(daily_sum) + energy_interval,
            irradiance_wm2=irr,
            temp_c=temp_c,
            humidity=humidity,
            wind_mps=wind_mps,
            rain_prob=rain_prob,
            inverter_data=inv_data,
            inv1_ac_kw=inverters[0]["ac_kw"],
            inv1_dc_kw=inverters[0]["dc_kw"],
            inv1_eff=inverters[0]["eff"]
        )
        
        # Rare critical alerts
        if random.random() < 0.005:
            alert = Alert(
                tenant_id=plant.tenant_id,
                plant_id=plant.id,
                timestamp=timestamp,
                severity="critical" if random.random() > 0.5 else "warning",
                message=random.choice([
                    "Ground fault detected",
                    "Grid overvoltage trip",
                    "High transformer temperature",
                    "String current mismatch"
                ])
            )
            db.add(alert)
            log.warning("alert_generated", plant_id=plant.id, message=alert.message)

        return point

    async def run_forever(self):
        """Async version of run() to be used as background task."""
        if not self.enabled:
            log.info("simulator_disabled")
            return

        log.info("simulator_started_background", speed=self.speed, mode=self.mode)
        current_time = dt.datetime.now(dt.timezone.utc)
        import asyncio

        while True:
            # Create a synchronous session inside the loop
            db = SessionLocal()
            try:
                # Find all plants to simulate
                plants = db.query(Plant).all()
                if not plants:
                    log.info("no_plants_found_waiting")
                    await asyncio.sleep(10)
                    continue

                # Round current_time to 5 min
                current_time = current_time.replace(second=0, microsecond=0)
                current_time -= dt.timedelta(minutes=current_time.minute % 5)
                
                for plant in plants:
                    point = self.generate_point(db, plant, current_time)
                    db.add(point)
                    log.info("inserted_timeseries_row", plant_id=plant.id, timestamp=point.timestamp.isoformat(), kw=round(point.active_power_kw, 2))
                
                db.commit()
                
                if self.mode == "real":
                    # Sleep for 5 minutes adjusted by speed
                    sleep_time = (5 * 60) / self.speed
                    await asyncio.sleep(sleep_time)
                    current_time = dt.datetime.now(dt.timezone.utc)
                else:
                    # Fast mode: move time forward immediately
                    current_time += dt.timedelta(minutes=5)
                    # Small throttle to not overwhelm DB
                    await asyncio.sleep(0.1 / self.speed)

            except Exception as e:
                log.error("simulator_error", error=str(e))
                db.rollback()
                await asyncio.sleep(5)
            finally:
                db.close()

    def run(self):
        if not self.enabled:
            log.info("simulator_disabled")
            return

        log.info("simulator_started", speed=self.speed, mode=self.mode)
        
        current_time = dt.datetime.now(dt.timezone.utc)
        
        while True:
            db = SessionLocal()
            try:
                # Find all plants to simulate
                plants = db.query(Plant).all()
                if not plants:
                    log.info("no_plants_found_waiting")
                    time.sleep(10)
                    continue

                # Round current_time to 5 min
                current_time = current_time.replace(second=0, microsecond=0)
                current_time -= dt.timedelta(minutes=current_time.minute % 5)
                
                for plant in plants:
                    point = self.generate_point(db, plant, current_time)
                    db.add(point)
                    log.info("inserted_timeseries_row", plant_id=plant.id, timestamp=point.timestamp.isoformat(), kw=round(point.active_power_kw, 2))
                
                db.commit()
                
                if self.mode == "real":
                    # Sleep for 5 minutes adjusted by speed
                    sleep_time = (5 * 60) / self.speed
                    time.sleep(sleep_time)
                    current_time = dt.datetime.now(dt.timezone.utc)
                else:
                    # Fast mode: move time forward immediately
                    current_time += dt.timedelta(minutes=5)
                    # Small throttle to not overwhelm DB
                    time.sleep(0.1 / self.speed)

            except Exception as e:
                log.error("simulator_error", error=str(e))
                db.rollback()
                time.sleep(5)
            finally:
                db.close()

if __name__ == "__main__":
    sim = AdvancedSolarSimulator()
    sim.run()
