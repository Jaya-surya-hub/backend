from fastapi import APIRouter
from .auth import router as auth_router
from .data import router as data_router

api_router = APIRouter(prefix="/api")
api_router.include_router(auth_router, prefix="/auth", tags=["auth"])
api_router.include_router(data_router, prefix="/data", tags=["data"])

