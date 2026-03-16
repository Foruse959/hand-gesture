from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.schemas import GestureSession, TrainingJob


class JsonStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.sessions_path = self.root / "sessions.json"
        self.jobs_path = self.root / "jobs.json"
        self._ensure_file(self.sessions_path)
        self._ensure_file(self.jobs_path)

    def _ensure_file(self, path: Path) -> None:
        if not path.exists():
            path.write_text("[]", encoding="utf-8")

    def _read_json(self, path: Path) -> list[dict[str, Any]]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

    def _write_json(self, path: Path, payload: list[dict[str, Any]]) -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def list_sessions(self) -> list[GestureSession]:
        return [GestureSession.model_validate(item) for item in self._read_json(self.sessions_path)]

    def save_sessions(self, sessions: list[GestureSession]) -> None:
        self._write_json(
            self.sessions_path,
            [session.model_dump(mode="json") for session in sessions],
        )

    def list_jobs(self) -> list[TrainingJob]:
        return [TrainingJob.model_validate(item) for item in self._read_json(self.jobs_path)]

    def save_jobs(self, jobs: list[TrainingJob]) -> None:
        self._write_json(
            self.jobs_path,
            [job.model_dump(mode="json") for job in jobs],
        )
