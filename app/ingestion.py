from typing import Iterable, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from .models import Tenant, SolarData
from .db import SessionLocal


async def reshape_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "ts"})
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"])
    for c in df.columns:
        if c != "ts":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("ts")
    df = df.dropna(subset=["ts"])
    return df


async def bulk_insert_timeseries(session: AsyncSession, tenant_id: int, rows: Iterable[Dict[str, Any]]):
    objects = []
    for row in rows:
        obj = SolarData(tenant_id=tenant_id, **row)
        objects.append(obj)
    session.add_all(objects)
    await session.commit()


def process_historical_csv(file_path: str, data_type: str, tenant_id: int):
    """
    Processes wide-format CSV into long format and stores in DB.
    Wide format: Date column + 96 columns (15-min intervals)
    """
    df = pd.read_csv(file_path)
    # Reshape wide to long
    # Assuming columns are 00:00, 00:15, ..., 23:45
    date_col = df.columns[0]
    melted = df.melt(id_vars=[date_col], var_name="time", value_name="value")
    
    # Create datetime objects
    melted['timestamp'] = pd.to_datetime(melted[date_col] + ' ' + melted['time'])
    
    # Interpolate 15-min to 5-min
    melted = melted.set_index('timestamp').resample('5T').interpolate(method='linear').reset_index()
    
    db = SessionLocal()
    try:
        data_points = []
        for _, row in melted.iterrows():
            data_point = SolarData(
                timestamp=row['timestamp'],
                tenant_id=tenant_id,
                active_power=row['value'] if data_type == 'active_power' else 0.0,
                energy_yield=row['value'] if data_type == 'yield' else 0.0,
            )
            data_points.append(data_point)
            
            # Batch insert every 1000 records
            if len(data_points) >= 1000:
                db.bulk_save_objects(data_points)
                db.commit()
                data_points = []
        
        if data_points:
            db.bulk_save_objects(data_points)
            db.commit()
    finally:
        db.close()

async def simulate_real_time_ingestion():
    """
    Background task to simulate real-time data ingestion every 5 mins.
    """
    print("Simulating real-time ingestion...")
    # In a real app, this would fetch from an IoT gateway or API
    pass

