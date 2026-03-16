from __future__ import annotations

from pathlib import Path

from app.services.lightweight_engine import LightweightGestureEngine
from app.services.storage import JsonStore

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
store = JsonStore(DATA_ROOT)
light_engine = LightweightGestureEngine(DATA_ROOT / "lightweight_profiles.json")
