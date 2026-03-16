from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class SessionCreate(BaseModel):
    name: str = Field(..., min_length=3, max_length=80)
    labels: list[str] = Field(..., min_length=2)
    sequence_length: int = Field(default=16, ge=8, le=64)
    capture_mode: Literal["browser", "hybrid", "backend"] = "hybrid"


class SampleCreate(BaseModel):
    label: str = Field(..., min_length=1, max_length=40)
    frame_count: int = Field(..., ge=1, le=512)
    source: Literal["webcam", "upload", "synthetic"] = "webcam"
    notes: str | None = Field(default=None, max_length=240)


class TrainingJobCreate(BaseModel):
    session_id: str
    backbone: Literal["resnet18", "resnet34", "mobilenetv3"] = "resnet18"
    temporal_head: Literal["bilstm_attention", "lstm", "temporal_conv"] = "bilstm_attention"
    variable_length: bool = True
    augmentation_level: Literal["light", "medium", "strong"] = "medium"


class GestureSample(BaseModel):
    id: str = Field(default_factory=lambda: f"sample_{uuid4().hex[:10]}")
    label: str
    frame_count: int
    source: str
    notes: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class GestureSession(BaseModel):
    id: str = Field(default_factory=lambda: f"session_{uuid4().hex[:10]}")
    name: str
    labels: list[str]
    sequence_length: int
    capture_mode: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    total_samples: int = 0
    total_frames: int = 0
    samples: list[GestureSample] = Field(default_factory=list)


class TrainingMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    latency_ms: float
    robustness_score: float
    cross_user_score: float


class TrainingJob(BaseModel):
    id: str = Field(default_factory=lambda: f"job_{uuid4().hex[:10]}")
    session_id: str
    backbone: str
    temporal_head: str
    variable_length: bool
    augmentation_level: str
    status: Literal["queued", "completed"] = "queued"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    metrics: TrainingMetrics | None = None
    export_formats: list[str] = Field(default_factory=list)


class MetricCard(BaseModel):
    label: str
    value: str
    description: str


class BlueprintModule(BaseModel):
    title: str
    summary: str
    outcomes: list[str]


class DashboardSnapshot(BaseModel):
    sessions: int
    samples: int
    jobs: int
    average_accuracy: float
    best_latency_ms: float
    deployment_targets: list[str]
    metric_cards: list[MetricCard]
