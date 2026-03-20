from __future__ import annotations

import hmac
import os

from fastapi import APIRouter, Depends, Header, HTTPException  # type: ignore[import-not-found]

from app.lightweight_schemas import (
    ExecuteMappingRequest,
    GestureDeleteRequest,
    GestureProfile,
    GestureRenameRequest,
    MappingExecutionResult,
    MappingUpdateRequest,
    PredictionRequest,
    PredictionResult,
    ProfileCreate,
    TrainClipRequest,
    TrainClipResult,
    TrainResult,
    TrainSequenceRequest,
)
from app.state import deep_engine, light_engine


def _verify_light_api_token(x_dgs_token: str | None = Header(default=None, alias="X-DGS-Token")) -> None:
    expected = os.getenv("DGS_API_TOKEN", "").strip()
    if not expected:
        return
    presented = (x_dgs_token or "").strip()
    if not presented or not hmac.compare_digest(expected, presented):
        raise HTTPException(status_code=401, detail="Unauthorized: invalid X-DGS-Token")


router = APIRouter(
    prefix="/api/light",
    tags=["lightweight-gesture"],
    dependencies=[Depends(_verify_light_api_token)],
)


@router.get("/profiles", response_model=list[GestureProfile])
def list_profiles() -> list[GestureProfile]:
    return light_engine.list_profiles()


@router.post("/profiles", response_model=GestureProfile)
def create_profile(payload: ProfileCreate) -> GestureProfile:
    return light_engine.create_profile(payload.name, payload.labels, payload.sequence_length)


@router.get("/profiles/{profile_id}", response_model=GestureProfile)
def get_profile(profile_id: str) -> GestureProfile:
    profile = light_engine.get_profile(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile


@router.delete("/profiles/{profile_id}")
def delete_profile(profile_id: str) -> dict[str, str]:
    profile = light_engine.get_profile(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")

    try:
        deep_engine.delete_profile_artifacts(profile_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not remove local model artifacts: {exc}") from exc

    light_engine.delete_profile(profile_id)
    return {"detail": f"Deleted profile {profile.name} ({profile.id}) and local profile data."}


@router.post("/train", response_model=TrainResult)
def train_sequence(payload: TrainSequenceRequest) -> TrainResult:
    try:
        return light_engine.train(payload.profile_id, payload.label, payload.sequence)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/train_clip", response_model=TrainClipResult)
def train_clip(payload: TrainClipRequest) -> TrainClipResult:
    try:
        return light_engine.train_clip(
            payload.profile_id,
            payload.label,
            payload.clip,
            payload.sample_count,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/predict", response_model=PredictionResult)
def predict_sequence(payload: PredictionRequest) -> PredictionResult:
    try:
        return light_engine.predict(
            payload.profile_id,
            payload.sequence,
            payload.context,
            payload.min_confidence,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/mappings", response_model=GestureProfile)
def update_mapping(payload: MappingUpdateRequest) -> GestureProfile:
    try:
        return light_engine.set_mapping(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/labels/rename", response_model=GestureProfile)
def rename_label(payload: GestureRenameRequest) -> GestureProfile:
    try:
        return light_engine.rename_label(payload.profile_id, payload.old_label, payload.new_label)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/labels/delete", response_model=GestureProfile)
def delete_label(payload: GestureDeleteRequest) -> GestureProfile:
    try:
        return light_engine.delete_label(payload.profile_id, payload.label)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/execute", response_model=MappingExecutionResult)
def execute_mapping(payload: ExecuteMappingRequest) -> MappingExecutionResult:
    try:
        return light_engine.execute_mapping(payload.profile_id, payload.context, payload.label)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
