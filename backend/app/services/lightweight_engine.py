from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from app.lightweight_schemas import (
    GestureAction,
    GestureClassState,
    GestureProfile,
    MappingExecutionResult,
    MappingUpdateRequest,
    PredictionResult,
    TrainResult,
)
from app.services.action_executor import execute_action


class LightweightGestureEngine:
    """
    Lightweight dynamic gesture trainer and predictor.

    Design goals:
    - no heavy pretrained dependency
    - fast custom training from webcam landmarks
    - temporal signal via residual (frame-delta) features
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._profiles: dict[str, GestureProfile] = {}
        self._last_fire: dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._save()
            return

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            profiles: dict[str, Any] = payload.get("profiles", {})
            for profile_id, profile_payload in profiles.items():
                self._profiles[profile_id] = GestureProfile.model_validate(profile_payload)
        except Exception:
            self._profiles = {}
            self._save()

    def _save(self) -> None:
        payload = {
            "profiles": {
                profile_id: profile.model_dump(mode="json")
                for profile_id, profile in self._profiles.items()
            }
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def list_profiles(self) -> list[GestureProfile]:
        return list(self._profiles.values())

    def create_profile(self, name: str, labels: list[str], sequence_length: int) -> GestureProfile:
        cleaned_labels = []
        seen = set()
        for label in labels:
            norm = label.strip().lower()
            if not norm or norm in seen:
                continue
            seen.add(norm)
            cleaned_labels.append(norm)

        profile = GestureProfile(
            name=name.strip(),
            labels=cleaned_labels,
            sequence_length=sequence_length,
            mappings={"global": {}, "browser": {}},
        )
        self._profiles[profile.id] = profile
        self._save()
        return profile

    def get_profile(self, profile_id: str) -> GestureProfile | None:
        return self._profiles.get(profile_id)

    def set_mapping(self, payload: MappingUpdateRequest) -> GestureProfile:
        profile = self._require_profile(payload.profile_id)
        context = payload.context.strip().lower()
        label = payload.label.strip().lower()

        mappings = dict(profile.mappings)
        context_map = dict(mappings.get(context, {}))
        context_map[label] = GestureAction(
            action_type=payload.action_type,
            value=payload.value,
            enabled=payload.enabled,
            cooldown_ms=payload.cooldown_ms,
            description=payload.description,
        )
        mappings[context] = context_map

        updated = profile.model_copy(update={"mappings": mappings, "updated_at": datetime.utcnow()})
        self._profiles[profile.id] = updated
        self._save()
        return updated

    def train(self, profile_id: str, label: str, sequence_payload: list[list[list[float | int | dict]]]) -> TrainResult:
        profile = self._require_profile(profile_id)
        normalized_label = label.strip().lower()

        sequence = self._parse_sequence(sequence_payload)
        sequence = self._resample(sequence, profile.sequence_length)
        embedding = self._build_embedding(sequence)

        classes = dict(profile.classes)
        current = classes.get(normalized_label)

        if current is None:
            classes[normalized_label] = GestureClassState(
                label=normalized_label,
                samples=1,
                prototype=embedding,
                updated_at=datetime.utcnow(),
            )
        else:
            samples = current.samples + 1
            old = current.prototype
            blended = [old[i] + (embedding[i] - old[i]) / samples for i in range(len(embedding))]
            classes[normalized_label] = current.model_copy(
                update={
                    "samples": samples,
                    "prototype": blended,
                    "updated_at": datetime.utcnow(),
                }
            )

        labels = list(profile.labels)
        if normalized_label not in labels:
            labels.append(normalized_label)

        updated = profile.model_copy(
            update={
                "classes": classes,
                "labels": labels,
                "updated_at": datetime.utcnow(),
            }
        )
        self._profiles[profile.id] = updated
        self._save()

        return TrainResult(
            profile_id=profile.id,
            label=normalized_label,
            samples=classes[normalized_label].samples,
            total_classes=len(classes),
            sequence_frames=len(sequence),
        )

    def predict(
        self,
        profile_id: str,
        sequence_payload: list[list[list[float | int | dict]]],
        context: str,
        min_confidence: float,
    ) -> PredictionResult:
        profile = self._require_profile(profile_id)
        if not profile.classes:
            return PredictionResult(accepted=False)

        sequence = self._parse_sequence(sequence_payload)
        sequence = self._resample(sequence, profile.sequence_length)
        embedding = self._build_embedding(sequence)

        distances: list[tuple[str, float]] = []
        for label, cls in profile.classes.items():
            dist = self._distance(embedding, cls.prototype)
            distances.append((label, dist))

        distances.sort(key=lambda item: item[1])
        best_label, best_dist = distances[0]
        second_dist = distances[1][1] if len(distances) > 1 else best_dist + 1.0

        similarity = 1.0 / (1.0 + best_dist)
        margin = max(0.0, second_dist - best_dist) / max(second_dist, 1e-6)
        confidence = min(0.99, max(0.0, similarity * 0.72 + margin * 0.58))

        action = self._lookup_action(profile, context, best_label)
        accepted = confidence >= min_confidence

        return PredictionResult(
            label=best_label,
            confidence=round(confidence, 4),
            distance=round(best_dist, 4),
            second_distance=round(second_dist, 4),
            accepted=accepted,
            action=action,
        )

    def execute_mapping(self, profile_id: str, context: str, label: str) -> MappingExecutionResult:
        profile = self._require_profile(profile_id)
        mapped = self._lookup_action(profile, context, label)
        if mapped is None:
            return MappingExecutionResult(success=False, detail="No mapping configured", action=None)

        cool_key = f"{profile_id}:{context.lower()}:{label.lower()}"
        now = datetime.utcnow().timestamp() * 1000.0
        last = self._last_fire.get(cool_key, 0.0)
        if now - last < mapped.cooldown_ms:
            wait_ms = int(mapped.cooldown_ms - (now - last))
            return MappingExecutionResult(
                success=False,
                detail=f"Cooldown active. Wait {wait_ms} ms",
                action=mapped,
            )

        ok, detail = execute_action(mapped)
        if ok:
            self._last_fire[cool_key] = now

        return MappingExecutionResult(success=ok, detail=detail, action=mapped)

    def _lookup_action(self, profile: GestureProfile, context: str, label: str) -> GestureAction | None:
        ctx = context.strip().lower()
        normalized_label = label.strip().lower()
        ctx_map = profile.mappings.get(ctx, {})
        if normalized_label in ctx_map:
            return ctx_map[normalized_label]
        global_map = profile.mappings.get("global", {})
        return global_map.get(normalized_label)

    def _require_profile(self, profile_id: str) -> GestureProfile:
        profile = self._profiles.get(profile_id)
        if profile is None:
            raise ValueError(f"Profile not found: {profile_id}")
        return profile

    def _parse_sequence(self, payload: list[list[list[float | int | dict]]]) -> list[list[list[float]]]:
        sequence: list[list[list[float]]] = []
        for raw_frame in payload:
            frame = []
            for raw_point in raw_frame[:21]:
                if isinstance(raw_point, dict):
                    x = float(raw_point.get("x", 0.0))
                    y = float(raw_point.get("y", 0.0))
                    z = float(raw_point.get("z", 0.0))
                else:
                    point = list(raw_point)
                    x = float(point[0]) if len(point) > 0 else 0.0
                    y = float(point[1]) if len(point) > 1 else 0.0
                    z = float(point[2]) if len(point) > 2 else 0.0
                frame.append([x, y, z])

            if len(frame) < 21:
                missing = 21 - len(frame)
                frame.extend([[0.0, 0.0, 0.0] for _ in range(missing)])

            sequence.append(self._normalize_frame(frame))

        if len(sequence) < 2:
            raise ValueError("Sequence too short")
        return sequence

    def _normalize_frame(self, frame: list[list[float]]) -> list[list[float]]:
        wrist = frame[0]
        centered = [[p[0] - wrist[0], p[1] - wrist[1], p[2] - wrist[2]] for p in frame]

        ref = centered[9]
        scale = math.sqrt(ref[0] ** 2 + ref[1] ** 2 + ref[2] ** 2)
        if scale < 1e-4:
            scale = 1.0

        return [[p[0] / scale, p[1] / scale, p[2] / scale] for p in centered]

    def _resample(self, sequence: list[list[list[float]]], target: int) -> list[list[list[float]]]:
        if len(sequence) == target:
            return sequence
        if len(sequence) < 2:
            return [sequence[0] for _ in range(target)]

        result: list[list[list[float]]] = []
        max_index = len(sequence) - 1
        for i in range(target):
            pos = i * max_index / max(target - 1, 1)
            low = int(math.floor(pos))
            high = min(low + 1, max_index)
            alpha = pos - low

            frame = []
            for j in range(21):
                p0 = sequence[low][j]
                p1 = sequence[high][j]
                frame.append([
                    p0[0] + (p1[0] - p0[0]) * alpha,
                    p0[1] + (p1[1] - p0[1]) * alpha,
                    p0[2] + (p1[2] - p0[2]) * alpha,
                ])
            result.append(frame)
        return result

    def _build_embedding(self, sequence: list[list[list[float]]]) -> list[float]:
        flattened_frames = [self._flatten(frame) for frame in sequence]
        feature_count = len(flattened_frames[0])

        static_mean = [0.0] * feature_count
        static_std = [0.0] * feature_count

        for feat in range(feature_count):
            values = [frame[feat] for frame in flattened_frames]
            mean = sum(values) / len(values)
            var = sum((v - mean) ** 2 for v in values) / len(values)
            static_mean[feat] = mean
            static_std[feat] = math.sqrt(var)

        residuals: list[list[float]] = []
        for i in range(1, len(flattened_frames)):
            prev_frame = flattened_frames[i - 1]
            cur_frame = flattened_frames[i]
            residuals.append([cur_frame[j] - prev_frame[j] for j in range(feature_count)])

        dyn_mean = [0.0] * feature_count
        dyn_std = [0.0] * feature_count

        for feat in range(feature_count):
            values = [frame[feat] for frame in residuals]
            mean = sum(values) / max(len(values), 1)
            var = sum((v - mean) ** 2 for v in values) / max(len(values), 1)
            dyn_mean[feat] = mean
            dyn_std[feat] = math.sqrt(var)

        return [*static_mean, *static_std, *dyn_mean, *dyn_std]

    def _flatten(self, frame: list[list[float]]) -> list[float]:
        flat = []
        for point in frame:
            flat.extend(point)
        return flat

    def _distance(self, left: list[float], right: list[float]) -> float:
        length = min(len(left), len(right))
        if length == 0:
            return 999.0
        accum = 0.0
        for i in range(length):
            diff = left[i] - right[i]
            accum += diff * diff
        return math.sqrt(accum / length)
