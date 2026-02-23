import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import argparse
import time
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from .db import SessionLocal, sync_engine, Base
from .models import Tenant, User, Plant, PlantTimeseries
from .auth import get_password_hash

def wait_for_db(max_retries=10, delay=2):
    retries = 0
    while retries < max_retries:
        try:
            with sync_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("Database is ready!")
            return True
        except OperationalError:
            retries += 1
            print(f"Database not ready, retrying ({retries}/{max_retries})...")
            time.sleep(delay)
    return False

def seed_database():
    if not wait_for_db():
        print("Could not connect to database. Exiting.")
        return

    print("Seeding database...")
    
    # Ensure tables exist
    Base.metadata.create_all(sync_engine)
    
    db = SessionLocal()
    try:
        # 1. Create Tenant
        tenant = db.query(Tenant).filter((Tenant.name == "Wiztric Technologies") | (Tenant.api_key == "demo-key")).first()
        if not tenant:
            tenant = Tenant(name="Wiztric Technologies", api_key="demo-key")
            db.add(tenant)
            db.commit()
            db.refresh(tenant)
            print("Tenant 'Wiztric Technologies' created.")
        else:
            print(f"Tenant '{tenant.name}' already exists.")

        # 2. Create User
        admin = db.query(User).filter(User.email == "admin@example.com").first()
        if not admin:
            admin = User(
                email="admin@example.com",
                hashed_password=get_password_hash("secret123"),
                role="admin",
                tenant_id=tenant.id
            )
            db.add(admin)
            db.commit()
            print("Admin user 'admin@example.com' created.")
        else:
            print("Admin user already exists.")

        # 3. Create Plant
        plant = db.query(Plant).filter((Plant.name == "Main Solar Plant") & (Plant.tenant_id == tenant.id)).first()
        if not plant:
            plant = Plant(
                name="Main Solar Plant",
                capacity_kw=1000.0,
                latitude=28.6139,
                longitude=77.2090,
                timezone="Asia/Kolkata",
                tenant_id=tenant.id
            )
            db.add(plant)
            db.commit()
            db.refresh(plant)
            print("Main Solar Plant created.")
        else:
            print("Main Solar Plant already exists.")

        # 4. Generate some initial data (last 24 hours)
        latest_point = db.query(PlantTimeseries).filter(PlantTimeseries.plant_id == plant.id).first()
        if not latest_point:
            print("Generating initial 24h data...")
            from .simulator.run import AdvancedSolarSimulator
            sim = AdvancedSolarSimulator()
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=24)
            
            current = start_time
            while current <= end_time:
                # Round to 5 min
                current = current.replace(second=0, microsecond=0)
                current -= timedelta(minutes=current.minute % 5)
                
                point = sim.generate_point(current, plant)
                db.add(point)
                current += timedelta(minutes=5)
            
            db.commit()
            print("Initial 24h data generated.")

    finally:
        db.close()

if __name__ == "__main__":
    seed_database()
