from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy import MetaData
from .config import get_settings

settings = get_settings()

convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=convention)


class Base(DeclarativeBase):
    metadata = metadata


# Async Engine & Session for FastAPI
engine = create_async_engine(settings.database_url, future=True, pool_pre_ping=True)
async_session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# Sync Engine & Session for Data Generator / Background tasks
# Convert asyncpg to psycopg2 for sync engine if needed
if "postgresql+asyncpg://" in settings.database_url:
    sync_url = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
elif "sqlite+aiosqlite://" in settings.database_url:
    sync_url = settings.database_url.replace("sqlite+aiosqlite://", "sqlite://")
else:
    sync_url = settings.database_url

sync_engine = create_engine(sync_url, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session

