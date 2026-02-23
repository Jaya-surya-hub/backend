import os
from functools import lru_cache
from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "Solar PV Analytics"
    environment: str = os.getenv("ENVIRONMENT", "dev")
    secret_key: str = os.getenv("SECRET_KEY", "dev-secret")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
    database_url: str = os.getenv(
        "DATABASE_URL",
        "sqlite+aiosqlite:///./solar.db",
    )
    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    cors_origins: str = os.getenv("CORS_ORIGINS", "*")
    simulator_enabled: bool = os.getenv("SIMULATOR_ENABLED", "true").lower() == "true"
    simulator_speed: float = float(os.getenv("SIMULATOR_SPEED", "1.0"))


@lru_cache
def get_settings() -> Settings:
    return Settings()

