from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DeepDatasetStore:
    def __init__(self, path: Path, max_samples_per_label: int = 180):
        self.path = path
        self.max_samples_per_label = max(20, max_samples_per_label)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._payload: dict[str, Any] = {"profiles": {}}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._save()
            return

        try:
            parsed = json.loads(self.path.read_text(encoding="utf-8"))
            profiles = parsed.get("profiles", {})
            if isinstance(profiles, dict):
                self._payload = {"profiles": profiles}
            else:
                self._payload = {"profiles": {}}
        except Exception:
            self._payload = {"profiles": {}}
            self._save()

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._payload, indent=2), encoding="utf-8")

    def _profile_map(self, profile_id: str) -> dict[str, list[list[list[list[float]]]]]:
        profiles = self._payload.setdefault("profiles", {})
        profile_map = profiles.setdefault(profile_id, {})
        return profile_map

    def add_sample(self, profile_id: str, label: str, sequence: list[list[list[float]]]) -> None:
        normalized_label = label.strip().lower()
        if not normalized_label:
            return

        profile_map = self._profile_map(profile_id)
        label_samples = profile_map.setdefault(normalized_label, [])

        sanitized: list[list[list[float]]] = []
        for raw_frame in sequence:
            frame: list[list[float]] = []
            for raw_point in raw_frame[:21]:
                point = list(raw_point)
                x = float(point[0]) if len(point) > 0 else 0.0
                y = float(point[1]) if len(point) > 1 else 0.0
                z = float(point[2]) if len(point) > 2 else 0.0
                frame.append([x, y, z])
            if len(frame) < 21:
                frame.extend([[0.0, 0.0, 0.0] for _ in range(21 - len(frame))])
            sanitized.append(frame)

        if len(sanitized) < 2:
            return

        label_samples.append(sanitized)
        if len(label_samples) > self.max_samples_per_label:
            profile_map[normalized_label] = label_samples[-self.max_samples_per_label :]

        self._save()

    def get_profile_samples(self, profile_id: str) -> dict[str, list[list[list[list[float]]]]]:
        profiles = self._payload.get("profiles", {})
        raw_profile = profiles.get(profile_id, {})
        if not isinstance(raw_profile, dict):
            return {}

        result: dict[str, list[list[list[list[float]]]]] = {}
        for label, sequences in raw_profile.items():
            if not isinstance(sequences, list):
                continue
            copied: list[list[list[list[float]]]] = []
            for sequence in sequences:
                if not isinstance(sequence, list):
                    continue
                seq_copy: list[list[list[float]]] = []
                for frame in sequence:
                    if not isinstance(frame, list):
                        continue
                    frame_copy = []
                    for point in frame:
                        if not isinstance(point, list):
                            continue
                        px = float(point[0]) if len(point) > 0 else 0.0
                        py = float(point[1]) if len(point) > 1 else 0.0
                        pz = float(point[2]) if len(point) > 2 else 0.0
                        frame_copy.append([px, py, pz])
                    if frame_copy:
                        seq_copy.append(frame_copy)
                if seq_copy:
                    copied.append(seq_copy)
            if copied:
                result[str(label)] = copied
        return result

    def profile_sample_count(self, profile_id: str) -> int:
        profile = self.get_profile_samples(profile_id)
        return sum(len(items) for items in profile.values())

    def rename_label(self, profile_id: str, old_label: str, new_label: str) -> None:
        old_norm = old_label.strip().lower()
        new_norm = new_label.strip().lower()
        if not old_norm or not new_norm or old_norm == new_norm:
            return

        profile_map = self._profile_map(profile_id)
        if old_norm not in profile_map:
            return

        existing = profile_map.get(new_norm, [])
        moved = profile_map.pop(old_norm, [])
        profile_map[new_norm] = [*existing, *moved][-self.max_samples_per_label :]
        self._save()

    def delete_label(self, profile_id: str, label: str) -> None:
        normalized = label.strip().lower()
        if not normalized:
            return

        profile_map = self._profile_map(profile_id)
        if normalized in profile_map:
            profile_map.pop(normalized, None)
            self._save()

    def delete_profile(self, profile_id: str) -> None:
        normalized_profile_id = profile_id.strip()
        if not normalized_profile_id:
            return

        profiles = self._payload.get("profiles", {})
        if not isinstance(profiles, dict):
            return

        if normalized_profile_id in profiles:
            profiles.pop(normalized_profile_id, None)
            self._save()
