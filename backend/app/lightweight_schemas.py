from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


ActionType = Literal["none", "open_url", "open_app", "hotkey", "type_text"]


class GestureAction(BaseModel):
    action_type: ActionType = "none"
    value: str = ""
    enabled: bool = True
    cooldown_ms: int = Field(default=1500, ge=100, le=60000)
    description: str = ""


class GestureClassState(BaseModel):
    label: str
    samples: int = 0
    prototype: list[float] = Field(default_factory=list)
    prototype_bank: list[list[float]] = Field(default_factory=list)
    prototype_bank_counts: list[int] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class GestureProfile(BaseModel):
    id: str = Field(default_factory=lambda: f"profile_{uuid4().hex[:10]}")
    name: str
    labels: list[str] = Field(default_factory=list)
    sequence_length: int = Field(default=24, ge=8, le=64)
    classes: dict[str, GestureClassState] = Field(default_factory=dict)
    mappings: dict[str, dict[str, GestureAction]] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ProfileCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=80)
    labels: list[str] = Field(default_factory=list)
    sequence_length: int = Field(default=24, ge=8, le=64)


class TrainSequenceRequest(BaseModel):
    profile_id: str
    label: str = Field(..., min_length=1, max_length=40)
    sequence: list[list[list[float | int | dict]]]


class PredictionRequest(BaseModel):
    profile_id: str
    sequence: list[list[list[float | int | dict]]]
    context: str = "global"
    min_confidence: float = Field(default=0.45, ge=0.0, le=0.99)


class PredictionResult(BaseModel):
    label: str | None = None
    confidence: float = 0.0
    distance: float = 0.0
    second_distance: float = 0.0
    accepted: bool = False
    action: GestureAction | None = None


class MappingUpdateRequest(BaseModel):
    profile_id: str
    context: str = Field(default="global", min_length=1, max_length=120)
    label: str = Field(..., min_length=1, max_length=40)
    action_type: ActionType = "none"
    value: str = ""
    enabled: bool = True
    cooldown_ms: int = Field(default=1500, ge=100, le=60000)
    description: str = ""


class ExecuteMappingRequest(BaseModel):
    profile_id: str
    context: str = Field(default="global", min_length=1, max_length=120)
    label: str = Field(..., min_length=1, max_length=40)


class GestureRenameRequest(BaseModel):
    profile_id: str
    old_label: str = Field(..., min_length=1, max_length=40)
    new_label: str = Field(..., min_length=1, max_length=40)


class GestureDeleteRequest(BaseModel):
    profile_id: str
    label: str = Field(..., min_length=1, max_length=40)


class MappingExecutionResult(BaseModel):
    success: bool
    detail: str
    action: GestureAction | None = None


class TrainResult(BaseModel):
    profile_id: str
    label: str
    samples: int
    total_classes: int
    sequence_frames: int


class TrainClipRequest(BaseModel):
    profile_id: str
    label: str = Field(..., min_length=1, max_length=40)
    clip: list[list[list[float | int | dict]]]
    sample_count: int = Field(default=6, ge=2, le=20)


class TrainClipResult(BaseModel):
    profile_id: str
    label: str
    samples_added: int
    samples: int
    total_classes: int
    clip_frames: int
