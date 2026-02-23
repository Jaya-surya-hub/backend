import datetime as dt
from typing import Optional, List
from sqlalchemy import String, Integer, DateTime, Float, ForeignKey, Index, JSON, UniqueConstraint, Boolean, Date
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .db import Base


class Tenant(Base):
    __tablename__ = "tenants"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    api_key: Mapped[Optional[str]] = mapped_column(String(255), unique=True, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    
    users: Mapped[List["User"]] = relationship(back_populates="tenant", cascade="all, delete")
    plants: Mapped[List["Plant"]] = relationship(back_populates="tenant", cascade="all, delete")


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(50), default="user")
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    
    tenant: Mapped[Tenant] = relationship(back_populates="users")


class Plant(Base):
    __tablename__ = "plants"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    capacity_kw: Mapped[float] = mapped_column(Float)
    latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    timezone: Mapped[str] = mapped_column(String(64), default="UTC")
    tilt: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    azimuth: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    tenant: Mapped[Tenant] = relationship(back_populates="plants")
    timeseries: Mapped[List["PlantTimeseries"]] = relationship(back_populates="plant", cascade="all, delete")
    daily_summaries: Mapped[List["PlantDailySummary"]] = relationship(back_populates="plant", cascade="all, delete")


class PlantTimeseries(Base):
    __tablename__ = "plant_timeseries"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), index=True)
    plant_id: Mapped[int] = mapped_column(ForeignKey("plants.id"), index=True)
    timestamp: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), index=True)
    
    # Aggregated Plant Data
    active_power_kw: Mapped[float] = mapped_column(Float, default=0.0)
    energy_kwh: Mapped[float] = mapped_column(Float, default=0.0) # Interval energy
    daily_yield_kwh: Mapped[float] = mapped_column(Float, default=0.0) # Cumulative daily
    
    # Inverter Details (Simulated for 6 inverters)
    inv1_ac_kw: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    inv1_dc_kw: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    inv1_eff: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # ... Simplified for demo, could use JSON for dynamic inverters
    inverter_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    # Weather/Sensor Data
    irradiance_wm2: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    temp_c: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    humidity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    wind_mps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rain_prob: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Status
    alarm_code: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    severity: Mapped[Optional[str]] = mapped_column(String(32), nullable=True) # Info, Warning, Critical

    plant: Mapped[Plant] = relationship(back_populates="timeseries")

    __table_args__ = (
        UniqueConstraint("plant_id", "timestamp", name="uq_plant_ts"),
        Index("idx_plant_ts_tenant_plant", "tenant_id", "plant_id", "timestamp"),
        Index("idx_plant_ts_timestamp", "timestamp"),
    )


class PlantDailySummary(Base):
    __tablename__ = "plant_daily_summary"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), index=True)
    plant_id: Mapped[int] = mapped_column(ForeignKey("plants.id"), index=True)
    date: Mapped[dt.date] = mapped_column(Date, index=True)
    
    total_energy_kwh: Mapped[float] = mapped_column(Float, default=0.0)
    predicted_total_energy_kwh: Mapped[float] = mapped_column(Float, default=0.0)
    peak_power_kw: Mapped[float] = mapped_column(Float, default=0.0)
    pr: Mapped[float] = mapped_column(Float, default=0.0) # Performance Ratio
    cuf: Mapped[float] = mapped_column(Float, default=0.0) # Capacity Utilization Factor
    specific_yield: Mapped[float] = mapped_column(Float, default=0.0)
    avg_temp: Mapped[float] = mapped_column(Float, default=0.0)
    total_irradiance: Mapped[float] = mapped_column(Float, default=0.0)
    availability_score: Mapped[float] = mapped_column(Float, default=100.0)
    health_score: Mapped[float] = mapped_column(Float, default=100.0)

    plant: Mapped[Plant] = relationship(back_populates="daily_summaries")

    __table_args__ = (
        UniqueConstraint("plant_id", "date", name="uq_plant_date"),
        Index("idx_plant_date_tenant", "date", "tenant_id", "plant_id"),
    )


class ReportJob(Base):
    __tablename__ = "report_jobs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), index=True)
    plant_id: Mapped[int] = mapped_column(ForeignKey("plants.id"), index=True)
    report_type: Mapped[str] = mapped_column(String(64)) # weekly, monthly, custom
    file_path: Mapped[str] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(32), default="completed")
    date_range_start: Mapped[dt.datetime] = mapped_column(DateTime)
    date_range_end: Mapped[dt.datetime] = mapped_column(DateTime)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


class Alert(Base):
    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), index=True)
    plant_id: Mapped[int] = mapped_column(ForeignKey("plants.id"), index=True)
    timestamp: Mapped[dt.datetime] = mapped_column(DateTime, index=True)
    severity: Mapped[str] = mapped_column(String(32)) # Info, Warning, Critical
    message: Mapped[str] = mapped_column(String(512))
    is_acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    acknowledged_by: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    acknowledged_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)
