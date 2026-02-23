import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, time, timezone
from zoneinfo import ZoneInfo
import pandas as pd


@dataclass
class PlantConfig:
    tenant_id: int = 1
    plant_id: int = 1
    plant_capacity_kw: float = 1000.0  # 1MW
    timezone_name: str = "Asia/Kolkata"
    lat: float = 13.5137496672669
    lon: float = 78.1004985332396
    inv_count: int = 5  # assume 5 inverters total (inv1 is one of them)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def solar_intensity(local_dt: datetime) -> float:
    """
    Returns 0..1 intensity based on time of day.
    sunrise ~06:00, sunset ~18:00 bell curve.
    """
    h = local_dt.hour + local_dt.minute / 60.0
    if h < 6 or h > 18:
        return 0.0
    # Map [6..18] -> [0..pi]
    x = (h - 6) / 12.0 * math.pi
    return max(0.0, math.sin(x))


def temp_model(local_dt: datetime, season_bias: float) -> float:
    """
    Rough temp pattern: cooler morning, hotter midday.
    """
    base = 26.0 + season_bias
    peak_add = 10.0 * solar_intensity(local_dt)
    noise = random.uniform(-0.8, 0.8)
    return clamp(base + peak_add + noise, 10.0, 55.0)


def humidity_model(local_dt: datetime) -> float:
    """
    Humidity tends to be higher early morning/evening.
    """
    h = local_dt.hour + local_dt.minute / 60.0
    morning_boost = 15.0 if h < 9 else 0.0
    evening_boost = 10.0 if h > 17 else 0.0
    noise = random.uniform(-3.0, 3.0)
    return clamp(55.0 + morning_boost + evening_boost + noise, 10.0, 100.0)


def wind_model(local_dt: datetime) -> float:
    """
    Light wind with midday slightly higher.
    """
    base = 2.0 + 1.5 * solar_intensity(local_dt)
    return clamp(base + random.uniform(-0.5, 0.8), 0.0, 20.0)


def rain_prob_model(day_cloudiness: float) -> float:
    """
    rain_prob correlates with cloudiness.
    """
    return clamp(day_cloudiness * 100.0 + random.uniform(-5.0, 10.0), 0.0, 100.0)


def irradiance_model(intensity: float, day_cloudiness: float) -> float:
    """
    0..1100 W/m2 with cloud reduction + noise.
    day_cloudiness: 0 clear, 1 very cloudy
    """
    clear_peak = 1000.0
    cloud_factor = 1.0 - 0.55 * day_cloudiness  # clouds reduce irradiance
    noise = random.uniform(-30.0, 30.0)
    irr = intensity * clear_peak * cloud_factor + noise
    return clamp(irr, 0.0, 1100.0)


def temp_derate_factor(temp_c: float) -> float:
    """
    Simple derate: above 25C reduce ~0.4% per C
    """
    if temp_c <= 25:
        return 1.0
    derate = 1.0 - 0.004 * (temp_c - 25.0)
    return clamp(derate, 0.75, 1.0)


def maybe_alarm(intensity: float) -> tuple[str | None, str | None]:
    """
    Random alarms mainly during generation hours.
    """
    if intensity <= 0:
        return None, None
    r = random.random()
    # very rare critical
    if r < 0.003:
        return "INV_COMM_LOSS", "Critical"
    if r < 0.010:
        return "LOW_EFFICIENCY", "Warning"
    if r < 0.020:
        return "GRID_FLUCT", "Info"
    return None, None


def generate_week_csv(
    start_date_local: str = "2026-02-01",
    days: int = 7,
    step_minutes: int = 5,
    seed: int = 42,
    cfg: PlantConfig = PlantConfig(),
    out_csv: str = "solar_week_5min.csv",
) -> str:
    random.seed(seed)

    tz = ZoneInfo(cfg.timezone_name)
    start_local_dt = datetime.fromisoformat(start_date_local).replace(tzinfo=tz)
    end_local_dt = start_local_dt + timedelta(days=days)

    rows = []
    daily_cum = 0.0
    current_day = start_local_dt.date()

    # day-level cloudiness pattern
    day_cloud = {}

    def cloudiness_for_day(d):
        if d not in day_cloud:
            # Most days partly cloudy; occasionally heavy cloud
            base = random.uniform(0.05, 0.45)
            if random.random() < 0.15:
                base = random.uniform(0.50, 0.85)
            day_cloud[d] = base
        return day_cloud[d]

    dt_local = start_local_dt
    idx = 1

    while dt_local < end_local_dt:
        if dt_local.date() != current_day:
            current_day = dt_local.date()
            daily_cum = 0.0  # reset at local midnight

        intensity = solar_intensity(dt_local)
        cloudiness = cloudiness_for_day(dt_local.date())

        irr = irradiance_model(intensity, cloudiness)
        temp_c = temp_model(dt_local, season_bias=1.0)
        hum = humidity_model(dt_local)
        wind = wind_model(dt_local)
        rain = rain_prob_model(cloudiness)

        # expected power ~ irradiance * capacity, derated by temp and clouds already in irr
        expected_kw = (irr / 1000.0) * cfg.plant_capacity_kw
        expected_kw *= temp_derate_factor(temp_c)

        # additional random noise + occasional dip
        noise_kw = random.uniform(-0.03, 0.03) * cfg.plant_capacity_kw
        power_kw = clamp(expected_kw + noise_kw, 0.0, cfg.plant_capacity_kw * 1.02)

        # interval energy for 5-min
        energy_kwh = power_kw * (step_minutes / 60.0)
        daily_cum += energy_kwh

        # inverter 1 share (one of inv_count)
        inv1_ac = power_kw / cfg.inv_count if cfg.inv_count > 0 else 0.0

        # efficiency as FRACTION (0.92..0.99) -> matches your current inv1_eff field.
        # If later you decide to store percent, change this to 92..99.
        inv1_eff = clamp(random.uniform(0.94, 0.985) - (0.002 * cloudiness), 0.90, 0.99)

        inv1_dc = inv1_ac / inv1_eff if inv1_eff > 0 else inv1_ac

        alarm_code, severity = maybe_alarm(intensity)

        inverter_data = {
            "inv_count": cfg.inv_count,
            "inv1": {
                "ac_kw": round(inv1_ac, 2),
                "dc_kw": round(inv1_dc, 2),
                "eff": round(inv1_eff, 4),
            }
        }

        # store timestamp as UTC ISO, but generation logic is local
        ts_utc = dt_local.astimezone(timezone.utc)

        rows.append({
            "id": idx,
            "tenant_id": cfg.tenant_id,
            "plant_id": cfg.plant_id,
            "timestamp": ts_utc.isoformat().replace("+00:00", "Z"),
            "active_power_kw": round(power_kw, 2),
            "energy_kwh": round(energy_kwh, 4),
            "daily_yield_kwh": round(daily_cum, 2),

            "inv1_ac_kw": round(inv1_ac, 2),
            "inv1_dc_kw": round(inv1_dc, 2),
            "inv1_eff": round(inv1_eff, 4),  # fraction 0.90..0.99

            "inverter_data": inverter_data,

            "irradiance_wm2": round(irr, 1),
            "temp_c": round(temp_c, 1),
            "humidity": round(hum, 1),
            "wind_mps": round(wind, 2),
            "rain_prob": round(rain, 1),

            "alarm_code": alarm_code,
            "severity": severity,
        })

        idx += 1
        dt_local += timedelta(minutes=step_minutes)

    df = pd.DataFrame(rows)

    # Write CSV; keep inverter_data as JSON string
    df["inverter_data"] = df["inverter_data"].apply(lambda x: str(x).replace("'", '"'))
    df.to_csv(out_csv, index=False)
    return out_csv


def generate_three_year_csv(
    start_date_local: str = "2024-01-01",
    years: int = 3,
    step_minutes: int = 5,
    seed: int = 42,
    cfg: PlantConfig = PlantConfig(),
    out_csv: str = "solar_three_years_5min.csv",
) -> str:
    days = 365 * years
    return generate_week_csv(
        start_date_local=start_date_local,
        days=days,
        step_minutes=step_minutes,
        seed=seed,
        cfg=cfg,
        out_csv=out_csv,
    )


if __name__ == "__main__":
    path = generate_three_year_csv()
    print("✅ Generated:", path)
