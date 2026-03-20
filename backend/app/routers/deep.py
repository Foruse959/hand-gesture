from __future__ import annotations

import hmac
import os

from fastapi import APIRouter, Depends, Header, HTTPException  # type: ignore[import-not-found]

from app.deep_schemas import DeepModelStatus, DeepPredictionRequest, DeepTrainRequest, DeepTrainResult
from app.lightweight_schemas import PredictionResult
from app.state import deep_engine


def _verify_deep_api_token(x_dgs_token: str | None = Header(default=None, alias="X-DGS-Token")) -> None:
    expected = os.getenv("DGS_API_TOKEN", "").strip()
    if not expected:
        return

    presented = (x_dgs_token or "").strip()
    if not presented or not hmac.compare_digest(expected, presented):
        raise HTTPException(status_code=401, detail="Unauthorized: invalid X-DGS-Token")


router = APIRouter(
    prefix="/api/deep",
    tags=["deep-gesture"],
    dependencies=[Depends(_verify_deep_api_token)],
)


@router.get("/models/{profile_id}", response_model=DeepModelStatus)
def get_model_status(profile_id: str) -> DeepModelStatus:
    return deep_engine.status(profile_id)


@router.post("/train", response_model=DeepTrainResult)
def train_deep_model(payload: DeepTrainRequest) -> DeepTrainResult:
    try:
        return deep_engine.train(
            payload.profile_id,
            payload.backbone,
            payload.temporal_head,
            payload.epochs,
            payload.batch_size,
            payload.learning_rate,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/predict", response_model=PredictionResult)
def predict_deep(payload: DeepPredictionRequest) -> PredictionResult:
    try:
        return deep_engine.predict(
            payload.profile_id,
            payload.sequence,
            payload.context,
            payload.min_confidence,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
