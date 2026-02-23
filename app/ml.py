import pandas as pd
import numpy as np
import datetime as dt
from typing import List, Dict
from sqlalchemy import select, func
from .db import SessionLocal
from .models import PlantTimeseries, PlantDailySummary

class MLService:
    @staticmethod
    def get_health_score(plant_id: int, tenant_id: int) -> float:
        """
        Calculates health score based on weighted PR, availability, and inverter uptime.
        Placeholder implementation.
        """
        # In a real app, this would query recent data and calculate metrics
        # For now, return a realistic varying score
        base_score = 95.0
        variation = np.random.uniform(-2, 2)
        return round(min(100.0, base_score + variation), 2)

    @staticmethod
    def get_forecast(plant_id: int, tenant_id: int) -> List[Dict]:
        """
        Naive forecast: returns predicted values for next 24 hours.
        Placeholder implementation.
        """
        forecast = []
        now = dt.datetime.utcnow()
        for i in range(1, 25):
            future_time = now + dt.timedelta(hours=i)
            # Simple sine wave simulation for forecast
            hour = future_time.hour
            intensity = max(0, np.sin(np.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0
            forecast.append({
                "timestamp": future_time.isoformat(),
                "predicted_power_kw": round(intensity * 800, 2) # Assuming 1000kW plant
            })
        return forecast

    @staticmethod
    def detect_anomalies(plant_id: int, tenant_id: int) -> List[Dict]:
        """
        Rule-based anomaly detection placeholder.
        """
        anomalies = []
        # Simulate an anomaly if random check fails
        if np.random.random() > 0.8:
            anomalies.append({
                "timestamp": dt.datetime.utcnow().isoformat(),
                "type": "Inverter Clipping",
                "severity": "Low",
                "description": "Output power reaching limit, potential clipping detected."
            })
        return anomalies

