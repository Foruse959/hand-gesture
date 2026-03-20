from __future__ import annotations

from pathlib import Path

from app.services.deep_dataset import DeepDatasetStore
from app.services.deep_model import DeepModelService
from app.services.lightweight_engine import LightweightGestureEngine
from app.services.storage import JsonStore

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
store = JsonStore(DATA_ROOT)
deep_dataset_store = DeepDatasetStore(DATA_ROOT / "deep_dataset.json")
light_engine = LightweightGestureEngine(
    DATA_ROOT / "lightweight_profiles.json",
    dataset_store=deep_dataset_store,
)
deep_engine = DeepModelService(DATA_ROOT / "deep_models", deep_dataset_store, light_engine)
