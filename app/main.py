import structlog
import datetime as dt
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from .config import get_settings
from .db import Base, engine, async_session
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from .routes import auth, data
from .jobs.tasks import run_daily_aggregation, run_weekly_reports
from .db import SessionLocal
from .models import Plant
from .weather_service import refresh_weather_for_plant

# Configure structlog
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20), # INFO
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

settings = get_settings()
logger = structlog.get_logger(__name__)

app = FastAPI(title="Wiztric Technologies Solar API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.cors_origins == "*" else [o.strip() for o in settings.cors_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(data.router, prefix="/api", tags=["data"])

# Scheduler
scheduler = AsyncIOScheduler()

@app.on_event("startup")
async def on_startup():
    async with engine.begin() as conn:
        # Note: In production, use Alembic migrations
        await conn.run_sync(Base.metadata.create_all)
    # Ensure default tenant/user/plant exist for demo if DB is empty
    try:
        from .models import Tenant, User
        from .auth import get_password_hash
        db = SessionLocal()
        tenants = db.query(Tenant).count()
        if tenants == 0:
            tenant = Tenant(name="Wiztric Technologies", api_key="demo-key")
            db.add(tenant)
            db.commit()
            db.refresh(tenant)
            admin = User(email="admin@example.com", hashed_password=get_password_hash("secret123"), role="admin", tenant_id=tenant.id)
            db.add(admin)
            db.commit()
            plant = Plant(
                tenant_id=tenant.id,
                name="Rayalacheruvu Plant",
                capacity_kw=1000.0,
                latitude=13.5137496672669,
                longitude=78.1004985332396,
                timezone="Asia/Kolkata"
            )
            db.add(plant)
            db.commit()
            logger.info("seed_created_default_plant", plant_id=plant.id)
        else:
            # If tenants exist but no plants, create a default demo plant
            plants_count = db.query(Plant).count()
            if plants_count == 0:
                tenant = db.query(Tenant).order_by(Tenant.id.asc()).first()
                if tenant:
                    plant = Plant(
                        tenant_id=tenant.id,
                        name="Rayalacheruvu Plant",
                        capacity_kw=1000.0,
                        latitude=13.5137496672669,
                        longitude=78.1004985332396,
                        timezone="Asia/Kolkata"
                    )
                    db.add(plant)
                    db.commit()
                    logger.info("seed_created_default_plant_existing_tenant", plant_id=plant.id)
            else:
                # Ensure at least one plant has the configured coordinates
                plant = db.query(Plant).first()
                if plant and (plant.latitude is None or plant.longitude is None):
                    plant.latitude = 13.5137496672669
                    plant.longitude = 78.1004985332396
                    plant.capacity_kw = 1000.0
                    plant.timezone = "Asia/Kolkata"
                    db.commit()
                    logger.info("updated_default_plant_coordinates", plant_id=plant.id)
    except Exception as e:
        logger.error("seed_failed", error=str(e))
    finally:
        try:
            db.close()
        except Exception:
            pass
    
    scheduler.add_job(run_daily_aggregation, "interval", minutes=10)
    scheduler.add_job(run_daily_aggregation, "date", run_date=dt.datetime.now(dt.timezone.utc))
    scheduler.add_job(run_weekly_reports, 'cron', day_of_week='mon', hour=0, minute=10)
    async def refresh_weather_job():
        db_local = SessionLocal()
        try:
            plants = db_local.query(Plant).all()
            for plant in plants:
                try:
                    await refresh_weather_for_plant(plant)
                except Exception as e:
                    logger.warning("weather_refresh_failed", plant_id=plant.id, error=str(e))
        finally:
            db_local.close()
    scheduler.add_job(refresh_weather_job, "interval", minutes=10)
    
    # Auto-start Simulator in background if enabled
    if settings.simulator_enabled:
        from .simulator.run import AdvancedSolarSimulator
        sim = AdvancedSolarSimulator()
        # Use asyncio.create_task for non-blocking execution
        import asyncio
        asyncio.create_task(sim.run_forever())
        logger.info("simulator_started_background")

    scheduler.start()
    logger.info("startup_completed", environment=settings.environment)

@app.get("/")
async def root():
    return {"message": "Wiztric Technologies Solar API is running"}

@app.get("/health")
async def health():
    async with async_session() as session:
        await session.execute(text("SELECT 1"))
    return {"status": "healthy"}
