import csv
import os
import datetime as dt
import structlog
from sqlalchemy import select, func
from sqlalchemy.orm import Session
from ..models import Plant, PlantTimeseries, PlantDailySummary, ReportJob, Tenant, Alert
from ..db import SessionLocal
from zoneinfo import ZoneInfo
from ..ml import get_daily_energy_forecast_for_date

log = structlog.get_logger()
REPORTS_DIR = "/reports/weekly"

def calculate_availability(plant_id: int, target_date: dt.date, db: Session) -> float:
    """Calculates availability based on intervals with non-zero irradiance but zero power."""
    start_ts = dt.datetime.combine(target_date, dt.time(6, 0)) # Daylight start
    end_ts = dt.datetime.combine(target_date, dt.time(18, 0))  # Daylight end
    
    # Total daylight intervals (5-min)
    total_intervals = 12 * 12 # 12 hours * 12 intervals/hour
    
    # Intervals with zero power but non-zero irradiance
    stmt = select(func.count(PlantTimeseries.id)).where(
        PlantTimeseries.plant_id == plant_id,
        PlantTimeseries.timestamp >= start_ts,
        PlantTimeseries.timestamp <= end_ts,
        PlantTimeseries.irradiance_wm2 > 50.0,
        PlantTimeseries.active_power_kw == 0.0
    )
    downtime_intervals = db.execute(stmt).scalar() or 0
    
    availability = ((total_intervals - downtime_intervals) / total_intervals) * 100.0
    return max(0.0, min(100.0, availability))

def calculate_health_score(plant_id: int, target_date: dt.date, db: Session) -> float:
    """Calculates health score based on critical alerts and inverter efficiency."""
    start_ts = dt.datetime.combine(target_date, dt.time.min)
    end_ts = dt.datetime.combine(target_date, dt.time.max)
    
    # Base score
    score = 100.0
    
    # Deduct for critical alerts
    alert_stmt = select(func.count(Alert.id)).where(
        Alert.plant_id == plant_id,
        Alert.timestamp >= start_ts,
        Alert.timestamp <= end_ts,
        Alert.severity == "critical"
    )
    critical_alerts = db.execute(alert_stmt).scalar() or 0
    score -= (critical_alerts * 5.0)
    
    # Deduct for inverter efficiency mismatch (if any inverter is below 90% during peak)
    peak_start = dt.datetime.combine(target_date, dt.time(10, 0))
    peak_end = dt.datetime.combine(target_date, dt.time(15, 0))
    
    eff_stmt = select(PlantTimeseries.inverter_data).where(
        PlantTimeseries.plant_id == plant_id,
        PlantTimeseries.timestamp >= peak_start,
        PlantTimeseries.timestamp <= peak_end,
        PlantTimeseries.active_power_kw > 100.0
    )
    results = db.execute(eff_stmt).scalars().all()
    
    low_eff_count = 0
    total_checks = 0
    for data in results:
        if not data: continue
        for inv_id, inv in data.items():
            total_checks += 1
            if inv.get("eff", 1.0) < 0.90:
                low_eff_count += 1
    
    if total_checks > 0:
        eff_penalty = (low_eff_count / total_checks) * 20.0
        score -= eff_penalty

    return max(0.0, min(100.0, score))

def run_daily_aggregation(target_date: dt.date = None):
    # target_date is interpreted per-plant timezone; when None, compute per plant
    log.info("daily_aggregation_started", date=target_date)
    db = SessionLocal()
    try:
        plants = db.execute(select(Plant)).scalars().all()
        
        for plant in plants:
            # Determine target date in plant timezone
            if target_date is None:
                now_utc = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
                try:
                    tz = ZoneInfo(plant.timezone or "UTC")
                except Exception:
                    tz = ZoneInfo("UTC")
                local_now = now_utc.astimezone(tz)
                tgt_date = local_now.date()
            else:
                tgt_date = target_date
            # Check if summary already exists
            existing = db.execute(
                select(PlantDailySummary)
                .where(PlantDailySummary.plant_id == plant.id)
                .where(PlantDailySummary.date == tgt_date)
            ).scalar_one_or_none()
            
            if existing:
                log.info("summary_already_exists", plant_id=plant.id, date=tgt_date)
                continue

            # Aggregate timeseries
            start_ts = dt.datetime.combine(tgt_date, dt.time.min)
            end_ts = dt.datetime.combine(tgt_date, dt.time.max)
            
            stmt = select(
                func.sum(PlantTimeseries.energy_kwh),
                func.max(PlantTimeseries.active_power_kw),
                func.avg(PlantTimeseries.temp_c),
                func.sum(PlantTimeseries.irradiance_wm2)
            ).where(
                PlantTimeseries.plant_id == plant.id,
                PlantTimeseries.timestamp >= start_ts,
                PlantTimeseries.timestamp <= end_ts
            )
            
            res = db.execute(stmt).one()
            total_energy, peak_power, avg_temp, total_irr = res
            
            if total_energy is None:
                log.warning("no_data_for_aggregation", plant_id=plant.id, date=tgt_date)
                continue

            # PR Calculation
            pr = 0.0
            if plant.capacity_kw and total_irr:
                irr_kwh_m2 = (total_irr / 12) / 1000.0 
                pr = (total_energy / (plant.capacity_kw * irr_kwh_m2)) * 100.0
            
            # CUF Calculation
            cuf = (total_energy / (plant.capacity_kw * 24)) * 100.0
            
            # Availability and Health
            availability = calculate_availability(plant.id, tgt_date, db)
            health = calculate_health_score(plant.id, tgt_date, db)

            forecast_row = get_daily_energy_forecast_for_date(tgt_date)
            predicted_energy = float(forecast_row["yhat"]) if forecast_row else 0.0

            summary = PlantDailySummary(
                tenant_id=plant.tenant_id,
                plant_id=plant.id,
                date=tgt_date,
                total_energy_kwh=total_energy,
                predicted_total_energy_kwh=predicted_energy,
                peak_power_kw=peak_power,
                pr=min(100.0, pr),
                cuf=min(100.0, cuf),
                specific_yield=total_energy / plant.capacity_kw,
                avg_temp=avg_temp or 0.0,
                total_irradiance=total_irr or 0.0,
                availability_score=availability,
                health_score=health
            )
            db.add(summary)
            db.commit()
            log.info("summary_created", plant_id=plant.id, date=tgt_date, pr=round(pr, 2))

    except Exception as e:
        log.error("daily_aggregation_failed", error=str(e))
    finally:
        db.close()

def generate_manual_report(tenant_id: int, plant_id: int, start_date: dt.date, end_date: dt.date):
    db = SessionLocal()
    try:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        filename = f"report_{tenant_id}_{plant_id}_{start_date}_to_{end_date}.csv"
        filepath = os.path.join(REPORTS_DIR, filename)
        
        stmt = select(PlantDailySummary).where(
            PlantDailySummary.plant_id == plant_id,
            PlantDailySummary.date >= start_date,
            PlantDailySummary.date <= end_date
        ).order_by(PlantDailySummary.date.asc())
        
        summaries = db.execute(stmt).scalars().all()
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Date", "Energy (kWh)", "Peak Power (kW)", "PR (%)", "CUF (%)", "Spec. Yield", "Availability (%)", "Health (%)"])
            for s in summaries:
                writer.writerow([
                    s.date, s.total_energy_kwh, s.peak_power_kw, 
                    round(s.pr, 2), round(s.cuf, 2), round(s.specific_yield, 2),
                    round(s.availability_score, 2), round(s.health_score, 2)
                ])
        
        job = ReportJob(
            tenant_id=tenant_id,
            plant_id=plant_id,
            report_type="manual",
            file_path=filepath,
            date_range_start=dt.datetime.combine(start_date, dt.time.min),
            date_range_end=dt.datetime.combine(end_date, dt.time.max)
        )
        db.add(job)
        db.commit()
        log.info("report_generated", path=filepath)
        return filepath
    finally:
        db.close()

def run_weekly_reports():
    log.info("weekly_reports_started")
    db = SessionLocal()
    try:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        plants = db.execute(select(Plant)).scalars().all()
        
        end_date = dt.date.today()
        start_date = end_date - dt.timedelta(days=7)
        
        for plant in plants:
            generate_manual_report(plant.tenant_id, plant.id, start_date, end_date)
            
    except Exception as e:
        log.error("weekly_reports_failed", error=str(e))
    finally:
        db.close()
