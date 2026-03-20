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
    TrainClipResult,
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

    _MAX_CLASS_PROTOTYPES = 6
    _CLUSTER_ASSIGN_THRESHOLD = 0.26

    def __init__(self, path: Path, dataset_store: Any | None = None):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._profiles: dict[str, GestureProfile] = {}
        self._last_fire: dict[str, float] = {}
        self._dataset_store = dataset_store
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
            mappings={"global": {}, "browser": {}, "presentation": {}},
        )
        self._profiles[profile.id] = profile
        self._save()
        return profile

    def get_profile(self, profile_id: str) -> GestureProfile | None:
        return self._profiles.get(profile_id)

    def delete_profile(self, profile_id: str) -> GestureProfile:
        profile = self._require_profile(profile_id)
        self._profiles.pop(profile_id, None)
        self._save()

        if self._dataset_store is not None:
            try:
                self._dataset_store.delete_profile(profile_id)
            except Exception:
                pass

        return profile

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

    def rename_label(self, profile_id: str, old_label: str, new_label: str) -> GestureProfile:
        profile = self._require_profile(profile_id)
        old_norm = old_label.strip().lower()
        new_norm = new_label.strip().lower()

        if not old_norm or not new_norm:
            raise ValueError("Old and new labels must be non-empty")
        if old_norm == new_norm:
            return profile

        classes = dict(profile.classes)
        if old_norm not in classes and old_norm not in profile.labels:
            raise ValueError(f"Gesture label not found: {old_norm}")
        if new_norm in classes and old_norm != new_norm:
            raise ValueError(f"Target label already exists: {new_norm}")

        if old_norm in classes:
            cls = classes.pop(old_norm)
            classes[new_norm] = cls.model_copy(update={"label": new_norm, "updated_at": datetime.utcnow()})

        labels: list[str] = []
        seen = set()
        for label in profile.labels:
            mapped = new_norm if label == old_norm else label
            if mapped not in seen:
                labels.append(mapped)
                seen.add(mapped)
        if new_norm not in seen:
            labels.append(new_norm)

        mappings: dict[str, dict[str, GestureAction]] = {}
        for context_name, context_map in profile.mappings.items():
            updated_context = dict(context_map)
            if old_norm in updated_context:
                moved_action = updated_context.pop(old_norm)
                if new_norm not in updated_context:
                    updated_context[new_norm] = moved_action
            mappings[context_name] = updated_context

        updated = profile.model_copy(
            update={
                "classes": classes,
                "labels": labels,
                "mappings": mappings,
                "updated_at": datetime.utcnow(),
            }
        )
        self._profiles[profile.id] = updated
        self._save()

        if self._dataset_store is not None:
            try:
                self._dataset_store.rename_label(profile.id, old_norm, new_norm)
            except Exception:
                pass

        return updated

    def delete_label(self, profile_id: str, label: str) -> GestureProfile:
        profile = self._require_profile(profile_id)
        normalized = label.strip().lower()
        if not normalized:
            raise ValueError("Label must be non-empty")

        classes = dict(profile.classes)
        if normalized not in classes and normalized not in profile.labels:
            raise ValueError(f"Gesture label not found: {normalized}")

        classes.pop(normalized, None)
        labels = [item for item in profile.labels if item != normalized]
        mappings: dict[str, dict[str, GestureAction]] = {}
        for context_name, context_map in profile.mappings.items():
            updated_context = dict(context_map)
            updated_context.pop(normalized, None)
            mappings[context_name] = updated_context

        updated = profile.model_copy(
            update={
                "classes": classes,
                "labels": labels,
                "mappings": mappings,
                "updated_at": datetime.utcnow(),
            }
        )
        self._profiles[profile.id] = updated
        self._save()

        if self._dataset_store is not None:
            try:
                self._dataset_store.delete_label(profile.id, normalized)
            except Exception:
                pass

        return updated

    def _record_deep_sample(self, profile_id: str, label: str, sequence: list[list[list[float]]]) -> None:
        if self._dataset_store is None:
            return
        try:
            self._dataset_store.add_sample(profile_id, label, sequence)
        except Exception:
            pass

    def train(self, profile_id: str, label: str, sequence_payload: list[list[list[float | int | dict]]]) -> TrainResult:
        profile = self._require_profile(profile_id)
        normalized_label = label.strip().lower()

        sequence = self._parse_sequence(sequence_payload)
        sequence = self._resample(sequence, profile.sequence_length)
        embedding = self._build_embedding(sequence)

        classes = dict(profile.classes)
        self._merge_embedding_into_class(classes, normalized_label, embedding)

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
        self._record_deep_sample(profile.id, normalized_label, sequence)

        return TrainResult(
            profile_id=profile.id,
            label=normalized_label,
            samples=classes[normalized_label].samples,
            total_classes=len(classes),
            sequence_frames=len(sequence),
        )

    def train_clip(
        self,
        profile_id: str,
        label: str,
        clip_payload: list[list[list[float | int | dict]]],
        sample_count: int,
    ) -> TrainClipResult:
        profile = self._require_profile(profile_id)
        normalized_label = label.strip().lower()

        clip_sequence = self._parse_sequence(clip_payload)
        clip_len = len(clip_sequence)
        seq_len = profile.sequence_length

        if clip_len < seq_len:
            raise ValueError(
                f"Live clip is too short ({clip_len} frames). Keep hand visible longer and retry."
            )

        max_start = clip_len - seq_len
        raw_starts = [int(round(i * max_start / max(sample_count - 1, 1))) for i in range(sample_count)]
        starts = sorted(set(raw_starts))
        if not starts:
            starts = [0]

        classes = dict(profile.classes)
        samples_added = 0

        for start_idx in starts:
            window = clip_sequence[start_idx : start_idx + seq_len]
            if len(window) != seq_len:
                continue
            embedding = self._build_embedding(window)
            self._merge_embedding_into_class(classes, normalized_label, embedding)
            self._record_deep_sample(profile.id, normalized_label, window)
            samples_added += 1

        if samples_added == 0:
            raise ValueError("Could not build training windows from live clip.")

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

        return TrainClipResult(
            profile_id=profile.id,
            label=normalized_label,
            samples_added=samples_added,
            samples=classes[normalized_label].samples,
            total_classes=len(classes),
            clip_frames=clip_len,
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
            candidates = self._class_candidate_vectors(cls, len(embedding))
            if not candidates:
                continue
            dist = min(self._distance(embedding, candidate) for candidate in candidates)
            distances.append((label, dist))

        if not distances:
            return PredictionResult(accepted=False)

        distances.sort(key=lambda item: item[1])
        best_label, best_dist = distances[0]
        second_dist = distances[1][1] if len(distances) > 1 else best_dist + 1.0

        best_state = profile.classes.get(best_label)
        best_samples = best_state.samples if best_state is not None else 0

        similarity = 1.0 / (1.0 + best_dist)
        relative_margin = max(0.0, second_dist - best_dist) / max(second_dist, 1e-6)

        # Confidence blends similarity and winner-vs-runner-up separation.
        sample_bonus = min(0.18, max(0, best_samples - 2) * 0.02)
        sparse_penalty = 0.06 if best_samples < 3 else 0.0
        confidence = min(
            0.995,
            max(0.0, similarity * 0.68 + relative_margin * 0.72 + sample_bonus - sparse_penalty),
        )

        base_margin_floor = 0.14
        if len(distances) > 2:
            base_margin_floor += 0.02
        if best_samples < 5:
            base_margin_floor += 0.04
        margin_ok = len(distances) == 1 or relative_margin >= base_margin_floor

        min_confidence_floor = min(
            0.995,
            min_confidence + (0.07 if best_samples < 3 else 0.02 if best_samples < 6 else 0.0),
        )

        action = self._lookup_action(profile, context, best_label)
        accepted = confidence >= min_confidence_floor and margin_ok

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
        for candidate in self._context_candidates(ctx):
            ctx_map = profile.mappings.get(candidate, {})
            if normalized_label in ctx_map:
                return ctx_map[normalized_label]
        return None

    def _context_candidates(self, context: str) -> list[str]:
        base = (context or "global").strip().lower()
        candidates: list[str] = []

        if base:
            candidates.append(base)

        if base.startswith("site:"):
            site = base.split(":", 1)[1].strip()
            if site.startswith("www."):
                candidates.append(f"site:{site[4:]}")
            candidates.append("browser")

        if base.startswith("browser:"):
            candidates.append("browser")

        if base != "global":
            candidates.append("global")

        dedup: list[str] = []
        seen = set()
        for candidate in candidates:
            if candidate and candidate not in seen:
                dedup.append(candidate)
                seen.add(candidate)
        return dedup

    def _merge_embedding_into_class(
        self,
        classes: dict[str, GestureClassState],
        normalized_label: str,
        embedding: list[float],
    ) -> None:
        current = classes.get(normalized_label)

        if current is None:
            classes[normalized_label] = GestureClassState(
                label=normalized_label,
                samples=1,
                prototype=embedding,
                prototype_bank=[embedding],
                prototype_bank_counts=[1],
                updated_at=datetime.utcnow(),
            )
            return

        samples = current.samples + 1

        bank = [self._resize_vector(item, len(embedding)) for item in current.prototype_bank if item]
        bank_counts = [max(1, int(item)) for item in current.prototype_bank_counts]

        if len(bank) != len(bank_counts):
            bank = bank[: min(len(bank), len(bank_counts))]
            bank_counts = bank_counts[: min(len(bank), len(bank_counts))]

        if not bank:
            bank = [self._resize_vector(current.prototype, len(embedding))]
            bank_counts = [max(1, current.samples)]

        nearest_index = 0
        nearest_distance = self._distance(embedding, bank[0])
        for index in range(1, len(bank)):
            candidate_distance = self._distance(embedding, bank[index])
            if candidate_distance < nearest_distance:
                nearest_distance = candidate_distance
                nearest_index = index

        if nearest_distance <= self._CLUSTER_ASSIGN_THRESHOLD or len(bank) >= self._MAX_CLASS_PROTOTYPES:
            count = bank_counts[nearest_index] + 1
            old_vector = self._resize_vector(bank[nearest_index], len(embedding))
            bank[nearest_index] = [
                old_vector[i] + (embedding[i] - old_vector[i]) / count for i in range(len(embedding))
            ]
            bank_counts[nearest_index] = count
        else:
            bank.append(embedding)
            bank_counts.append(1)

        if len(bank) > self._MAX_CLASS_PROTOTYPES:
            drop_index = bank_counts.index(min(bank_counts))
            bank.pop(drop_index)
            bank_counts.pop(drop_index)

        total_weight = max(1, sum(bank_counts))
        blended = []
        for feat_idx in range(len(embedding)):
            weighted_sum = 0.0
            for proto_idx, proto in enumerate(bank):
                weighted_sum += proto[feat_idx] * bank_counts[proto_idx]
            blended.append(weighted_sum / total_weight)

        classes[normalized_label] = current.model_copy(
            update={
                "samples": samples,
                "prototype": blended,
                "prototype_bank": bank,
                "prototype_bank_counts": bank_counts,
                "updated_at": datetime.utcnow(),
            }
        )

    def _class_candidate_vectors(self, cls: GestureClassState, target_length: int) -> list[list[float]]:
        candidates: list[list[float]] = []
        if cls.prototype:
            candidates.append(self._resize_vector(cls.prototype, target_length))

        for item in cls.prototype_bank:
            if item:
                candidates.append(self._resize_vector(item, target_length))

        dedup: list[list[float]] = []
        seen = set()
        for item in candidates:
            key = tuple(round(value, 6) for value in item)
            if key in seen:
                continue
            seen.add(key)
            dedup.append(item)
        return dedup

    def _resize_vector(self, vector: list[float], target_length: int) -> list[float]:
        if len(vector) == target_length:
            return vector
        if len(vector) > target_length:
            return vector[:target_length]
        return [*vector, *([0.0] * (target_length - len(vector)))]

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

        if len(centered) < 21:
            return centered

        palm_x = self._normalize_vec(self._vec_sub(centered[5], centered[17]))
        palm_y_hint = self._normalize_vec(centered[9])

        if self._vec_norm(palm_x) > 1e-6 and self._vec_norm(palm_y_hint) > 1e-6:
            palm_z = self._normalize_vec(self._cross(palm_x, palm_y_hint))
            if self._vec_norm(palm_z) > 1e-6:
                palm_y = self._normalize_vec(self._cross(palm_z, palm_x))
                if self._vec_norm(palm_y) > 1e-6:
                    oriented = [
                        [
                            self._dot(point, palm_x),
                            self._dot(point, palm_y),
                            self._dot(point, palm_z),
                        ]
                        for point in centered
                    ]

                    ref = oriented[9]
                    scale = self._vec_norm(ref)
                    if scale < 1e-4:
                        scale = 1.0
                    return [[p[0] / scale, p[1] / scale, p[2] / scale] for p in oriented]

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

        # Add motion trajectory descriptors so directional gestures
        # (e.g. swipe-down vs static open hand) are easier to separate.
        trajectory: list[float] = []
        tracked_points = [0, 4, 8, 12, 16, 20]

        for point_index in tracked_points:
            point_series: list[list[float]] = []
            for frame in sequence:
                if point_index < len(frame):
                    point_series.append(frame[point_index])
                else:
                    point_series.append([0.0, 0.0, 0.0])

            start = point_series[0]
            end = point_series[-1]
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            dz = end[2] - start[2]

            cumulative_path = 0.0
            for i in range(1, len(point_series)):
                prev = point_series[i - 1]
                cur = point_series[i]
                ddx = cur[0] - prev[0]
                ddy = cur[1] - prev[1]
                ddz = cur[2] - prev[2]
                cumulative_path += math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz)

            y_values = [point[1] for point in point_series]
            x_values = [point[0] for point in point_series]
            y_range = max(y_values) - min(y_values)
            x_range = max(x_values) - min(x_values)

            trajectory.extend([dx, dy, dz, cumulative_path, x_range, y_range])

        return [*static_mean, *static_std, *dyn_mean, *dyn_std, *trajectory]

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

    def _vec_sub(self, left: list[float], right: list[float]) -> list[float]:
        return [left[0] - right[0], left[1] - right[1], left[2] - right[2]]

    def _dot(self, left: list[float], right: list[float]) -> float:
        return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]

    def _cross(self, left: list[float], right: list[float]) -> list[float]:
        return [
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        ]

    def _vec_norm(self, value: list[float]) -> float:
        return math.sqrt(value[0] ** 2 + value[1] ** 2 + value[2] ** 2)

    def _normalize_vec(self, value: list[float]) -> list[float]:
        norm = self._vec_norm(value)
        if norm < 1e-8:
            return [0.0, 0.0, 0.0]
        return [value[0] / norm, value[1] / norm, value[2] / norm]
