from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "service": "dynamic-gesture-studio-backend",
        "storage": str(Path(__file__).resolve().parents[2] / "data"),
        "stack": ["FastAPI", "JSON store", "gesture training orchestration"],
    }
