from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas import DashboardSnapshot, TrainingJob, TrainingJobCreate
from app.services.trainer import build_dashboard, finalize_job
from app.state import store

router = APIRouter(prefix="/api", tags=["training"])


@router.get("/dashboard", response_model=DashboardSnapshot)
def dashboard() -> DashboardSnapshot:
    return build_dashboard(store.list_sessions(), store.list_jobs())


@router.get("/training/jobs", response_model=list[TrainingJob])
def list_jobs() -> list[TrainingJob]:
    return store.list_jobs()


@router.post("/training/jobs", response_model=TrainingJob)
def create_training_job(payload: TrainingJobCreate) -> TrainingJob:
    sessions = store.list_sessions()
    session = next((session for session in sessions if session.id == payload.session_id), None)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.total_samples == 0:
        raise HTTPException(status_code=400, detail="Add samples before launching training")

    jobs = store.list_jobs()
    job = TrainingJob(
        session_id=payload.session_id,
        backbone=payload.backbone,
        temporal_head=payload.temporal_head,
        variable_length=payload.variable_length,
        augmentation_level=payload.augmentation_level,
    )
    completed = finalize_job(session, job)
    jobs.append(completed)
    store.save_jobs(jobs)
    return completed


@router.get("/training/jobs/{job_id}", response_model=TrainingJob)
def get_job(job_id: str) -> TrainingJob:
    for job in store.list_jobs():
        if job.id == job_id:
            return job
    raise HTTPException(status_code=404, detail="Training job not found")
