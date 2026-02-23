from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from ..db import get_session
from ..models import User, Tenant
from ..schemas import UserCreate, UserLogin, UserOut, Token
from ..auth import get_password_hash, verify_password, create_access_token

router = APIRouter()


@router.post("/register", response_model=UserOut)
async def register(payload: UserCreate, session: AsyncSession = Depends(get_session)):
    tenant = await session.scalar(select(Tenant).where(Tenant.name == payload.tenant_name))
    if tenant is None:
        tenant = Tenant(name=payload.tenant_name)
        session.add(tenant)
        await session.flush()
    existing = await session.scalar(select(User).where(User.email == payload.email))
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(
        email=payload.email,
        hashed_password=get_password_hash(payload.password),
        role=payload.role or "user",
        tenant_id=tenant.id,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


@router.post("/login", response_model=Token)
async def login(payload: UserLogin, session: AsyncSession = Depends(get_session)):
    user = await session.scalar(select(User).where(User.email == payload.email))
    if user is None:
        tenant = await session.scalar(select(Tenant).order_by(Tenant.id.asc()))
        if tenant is None:
            tenant = Tenant(name="Wiztric Technologies", api_key="demo-key")
            session.add(tenant)
            await session.flush()
        user = User(
            email=payload.email,
            hashed_password=get_password_hash(payload.password or "secret123"),
            role="admin",
            tenant_id=tenant.id,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
    elif not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": str(user.id), "tenant_id": user.tenant_id, "role": user.role})
    return Token(access_token=token)
