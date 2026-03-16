from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas import GestureSample, GestureSession, SampleCreate, SessionCreate
from app.state import store

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("", response_model=list[GestureSession])
def list_sessions() -> list[GestureSession]:
    return store.list_sessions()


@router.post("", response_model=GestureSession)
def create_session(payload: SessionCreate) -> GestureSession:
    sessions = store.list_sessions()
    session = GestureSession(
        name=payload.name,
        labels=payload.labels,
        sequence_length=payload.sequence_length,
        capture_mode=payload.capture_mode,
    )
    sessions.append(session)
    store.save_sessions(sessions)
    return session


@router.get("/{session_id}", response_model=GestureSession)
def get_session(session_id: str) -> GestureSession:
    for session in store.list_sessions():
        if session.id == session_id:
            return session
    raise HTTPException(status_code=404, detail="Session not found")


@router.post("/{session_id}/samples", response_model=GestureSession)
def add_sample(session_id: str, payload: SampleCreate) -> GestureSession:
    sessions = store.list_sessions()
    for index, session in enumerate(sessions):
        if session.id != session_id:
            continue
        if payload.label not in session.labels:
            raise HTTPException(status_code=400, detail="Label is not registered in this session")

        sample = GestureSample(
            label=payload.label,
            frame_count=payload.frame_count,
            source=payload.source,
            notes=payload.notes,
        )
        updated = session.model_copy(
            update={
                "samples": [*session.samples, sample],
                "total_samples": session.total_samples + 1,
                "total_frames": session.total_frames + payload.frame_count,
            }
        )
        sessions[index] = updated
        store.save_sessions(sessions)
        return updated

    raise HTTPException(status_code=404, detail="Session not found")
