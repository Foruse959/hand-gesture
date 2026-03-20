from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.routers.blueprint import router as blueprint_router
from app.routers.deep import router as deep_router
from app.routers.health import router as health_router
from app.routers.lightweight import router as lightweight_router
from app.routers.sessions import router as sessions_router
from app.routers.training import router as training_router

FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
FRONTEND_ASSETS = FRONTEND_DIST / "assets"
FRONTEND_LEGACY = FRONTEND_DIST / "legacy"
FRONTEND_FAVICON = FRONTEND_DIST / "favicon.svg"
FRONTEND_ICONS = FRONTEND_DIST / "icons.svg"

app = FastAPI(
    title="Dynamic Gesture Studio API",
    description="Backend services for dataset capture, training orchestration, evaluation, and demo deployment.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(blueprint_router)
app.include_router(sessions_router)
app.include_router(training_router)
app.include_router(lightweight_router)
app.include_router(deep_router)

if FRONTEND_DIST.exists():
    app.mount("/studio", StaticFiles(directory=FRONTEND_DIST, html=True), name="studio_frontend")
    if FRONTEND_ASSETS.exists():
        app.mount("/assets", StaticFiles(directory=FRONTEND_ASSETS), name="studio_assets")
    if FRONTEND_LEGACY.exists():
        app.mount("/legacy", StaticFiles(directory=FRONTEND_LEGACY, html=True), name="studio_legacy")


@app.get("/favicon.svg", include_in_schema=False)
def favicon() -> FileResponse:
    if FRONTEND_FAVICON.exists():
        return FileResponse(FRONTEND_FAVICON)
    raise HTTPException(status_code=404, detail="favicon not found")


@app.get("/icons.svg", include_in_schema=False)
def icons() -> FileResponse:
    if FRONTEND_ICONS.exists():
        return FileResponse(FRONTEND_ICONS)
    raise HTTPException(status_code=404, detail="icons not found")


@app.get("/")
def root() -> dict[str, str]:
    payload = {
        "message": "Dynamic Gesture Studio backend running",
        "docs": "/docs",
    }
    if FRONTEND_DIST.exists():
        payload["studio"] = "/studio"
    return payload
