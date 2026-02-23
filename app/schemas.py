from pydantic import BaseModel, EmailStr
from typing import Optional, List
import datetime as dt


class TenantCreate(BaseModel):
    name: str


class TenantOut(BaseModel):
    id: int
    name: str
    class Config:
        from_attributes = True


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    tenant_name: str
    role: Optional[str] = "admin"


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: int
    email: EmailStr
    role: str
    tenant_id: int
    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class SolarDataIn(BaseModel):
    timestamp: dt.datetime
    active_power: Optional[float] = None
    energy_yield: Optional[float] = None
    irradiance: Optional[float] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    wind_speed: Optional[float] = None
    inverter_id: Optional[str] = None


class SolarDataOut(SolarDataIn):
    id: int
    tenant_id: int
    class Config:
        from_attributes = True


class KPIResponse(BaseModel):
    total_yield: float
    pr: float
    cuf: float
    specific_yield: float
    co2_saved: float
    plant_health: float

