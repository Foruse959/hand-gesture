from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class DeepTrainRequest(BaseModel):
    profile_id: str
    backbone: Literal["resnet18", "resnet34"] = "resnet18"
    temporal_head: Literal["lstm", "bilstm_attention"] = "bilstm_attention"
    epochs: int = Field(default=12, ge=1, le=200)
    batch_size: int = Field(default=16, ge=1, le=128)
    learning_rate: float = Field(default=1e-3, gt=1e-6, le=1.0)


class DeepTrainResult(BaseModel):
    profile_id: str
    backbone: str
    temporal_head: str
    labels: list[str]
    samples: int
    epochs: int
    train_accuracy: float
    val_accuracy: float
    loss: float
    model_path: str


class DeepModelStatus(BaseModel):
    profile_id: str
    available: bool
    trained_at: datetime | None = None
    backbone: str | None = None
    temporal_head: str | None = None
    labels: list[str] = Field(default_factory=list)
    samples: int = 0
    detail: str = ""


class DeepPredictionRequest(BaseModel):
    profile_id: str
    sequence: list[list[list[float | int | dict]]]
    context: str = "global"
    min_confidence: float = Field(default=0.45, ge=0.0, le=0.99)
