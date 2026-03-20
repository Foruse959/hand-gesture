from __future__ import annotations

import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from app.deep_schemas import DeepModelStatus, DeepTrainResult
from app.lightweight_schemas import PredictionResult
from app.services.deep_dataset import DeepDatasetStore
from app.services.lightweight_engine import LightweightGestureEngine

try:
    import torch  # type: ignore[import-not-found]
    import torch.nn as nn  # type: ignore[import-not-found]
    import torch.nn.functional as F  # type: ignore[import-not-found]
    from torchvision import models  # type: ignore[import-not-found]
except Exception as exc:  # pragma: no cover - runtime dependency check
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    models = None  # type: ignore[assignment]
    TORCH_IMPORT_ERROR = str(exc)
else:
    TORCH_IMPORT_ERROR = ""


if torch is not None:

    class ResNetLSTMGestureModel(nn.Module):
        def __init__(
            self,
            num_classes: int,
            backbone: str,
            temporal_head: str,
            hidden_size: int = 128,
        ):
            super().__init__()
            if backbone == "resnet34":
                base = models.resnet34(weights=None)
            else:
                base = models.resnet18(weights=None)

            base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.frame_encoder = nn.Sequential(
                base.conv1,
                base.bn1,
                base.relu,
                base.maxpool,
                base.layer1,
                base.layer2,
                base.layer3,
                base.layer4,
                base.avgpool,
            )

            self.temporal_head = temporal_head
            self.lstm = nn.LSTM(
                input_size=512,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.attention = nn.Linear(hidden_size * 2, 1) if temporal_head == "bilstm_attention" else None
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_size * 2),
                nn.Dropout(0.25),
                nn.Linear(hidden_size * 2, num_classes),
            )

        def forward(self, x: Any) -> Any:
            # x: [batch, time, 21, 3]
            batch, steps = x.shape[0], x.shape[1]
            frames = x.reshape(batch * steps, 1, 21, 3)
            frames = F.interpolate(frames, size=(64, 64), mode="bilinear", align_corners=False)
            frames = torch.clamp((frames + 2.0) / 4.0, 0.0, 1.0)

            features = self.frame_encoder(frames).flatten(1)
            features = features.view(batch, steps, -1)

            temporal_output, _ = self.lstm(features)
            if self.attention is not None:
                weights = torch.softmax(self.attention(temporal_output).squeeze(-1), dim=1)
                pooled = torch.sum(temporal_output * weights.unsqueeze(-1), dim=1)
            else:
                pooled = temporal_output[:, -1, :]

            return self.classifier(pooled)


class DeepModelService:
    def __init__(
        self,
        root: Path,
        dataset_store: DeepDatasetStore,
        light_engine: LightweightGestureEngine,
    ):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.dataset_store = dataset_store
        self.light_engine = light_engine
        self.device = torch.device("cuda" if (torch is not None and torch.cuda.is_available()) else "cpu") if torch is not None else None
        self._cache: dict[str, tuple[Any, dict[str, Any]]] = {}

    def delete_profile_artifacts(self, profile_id: str) -> None:
        normalized_profile_id = profile_id.strip()
        if not normalized_profile_id:
            return

        self._cache.pop(normalized_profile_id, None)
        profile_dir = self.root / normalized_profile_id
        if profile_dir.exists():
            shutil.rmtree(profile_dir)

    def _profile_dir(self, profile_id: str) -> Path:
        return self.root / profile_id

    def _model_path(self, profile_id: str) -> Path:
        return self._profile_dir(profile_id) / "resnet_lstm.pt"

    def _meta_path(self, profile_id: str) -> Path:
        return self._profile_dir(profile_id) / "metadata.json"

    def _read_meta(self, profile_id: str) -> dict[str, Any] | None:
        path = self._meta_path(profile_id)
        if not path.exists():
            return None

        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write_meta(self, profile_id: str, payload: dict[str, Any]) -> None:
        path = self._meta_path(profile_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def status(self, profile_id: str) -> DeepModelStatus:
        meta = self._read_meta(profile_id)
        sample_count = self.dataset_store.profile_sample_count(profile_id)

        if meta is None:
            detail = "No deep model trained for this profile yet."
            if torch is None:
                detail = f"Deep runtime unavailable: {TORCH_IMPORT_ERROR}"
            return DeepModelStatus(
                profile_id=profile_id,
                available=False,
                labels=[],
                samples=sample_count,
                detail=detail,
            )

        trained_at_raw = str(meta.get("trained_at") or "").strip()
        trained_at = None
        if trained_at_raw:
            try:
                trained_at = datetime.fromisoformat(trained_at_raw)
            except ValueError:
                trained_at = None

        return DeepModelStatus(
            profile_id=profile_id,
            available=self._model_path(profile_id).exists(),
            trained_at=trained_at,
            backbone=str(meta.get("backbone") or ""),
            temporal_head=str(meta.get("temporal_head") or ""),
            labels=list(meta.get("labels") or []),
            samples=int(meta.get("samples") or sample_count),
            detail=str(meta.get("detail") or ""),
        )

    def _split_indices(self, targets: list[int], label_count: int) -> tuple[list[int], list[int]]:
        buckets: dict[int, list[int]] = {index: [] for index in range(label_count)}
        for index, target in enumerate(targets):
            buckets[target].append(index)

        rng = random.Random(42)
        train: list[int] = []
        val: list[int] = []

        for indices in buckets.values():
            rng.shuffle(indices)
            if not indices:
                continue
            if len(indices) == 1:
                train.extend(indices)
                continue

            val_count = max(1, int(round(len(indices) * 0.2)))
            val_count = min(val_count, len(indices) - 1)
            val.extend(indices[:val_count])
            train.extend(indices[val_count:])

        if not train:
            train = list(range(len(targets)))
        if not val:
            val = train[:]

        return train, val

    def _evaluate(self, model: Any, samples: Any, targets: Any) -> float:
        if torch is None:
            return 0.0

        model.eval()
        with torch.no_grad():
            logits = model(samples.to(self.device))
            predicted = torch.argmax(logits, dim=1)
            correct = (predicted == targets.to(self.device)).float().mean().item()
        return float(correct)

    def train(
        self,
        profile_id: str,
        backbone: str,
        temporal_head: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
    ) -> DeepTrainResult:
        if torch is None:
            raise ValueError(
                "PyTorch/torchvision is not available in backend environment. "
                f"Install torch + torchvision first. Import error: {TORCH_IMPORT_ERROR}"
            )

        profile = self.light_engine.get_profile(profile_id)
        if profile is None:
            raise ValueError(f"Profile not found: {profile_id}")

        raw_samples = self.dataset_store.get_profile_samples(profile_id)
        usable_labels: list[str] = []
        examples: list[list[list[list[float]]]] = []
        targets: list[int] = []

        for label in sorted(raw_samples.keys()):
            sequences = raw_samples.get(label, [])
            cleaned: list[list[list[float]]] = []
            for sequence in sequences:
                if len(sequence) < 2:
                    continue
                cleaned.append(self.light_engine._resample(sequence, profile.sequence_length))

            if not cleaned:
                continue

            label_index = len(usable_labels)
            usable_labels.append(label)
            examples.extend(cleaned)
            targets.extend([label_index] * len(cleaned))

        if len(usable_labels) < 2:
            raise ValueError("Deep training needs at least 2 gesture labels with recorded samples.")

        if len(examples) < 8:
            raise ValueError("Deep training needs more recorded sequences. Capture more gesture samples first.")

        train_indices, val_indices = self._split_indices(targets, len(usable_labels))

        x_train = torch.tensor([examples[i] for i in train_indices], dtype=torch.float32)
        y_train = torch.tensor([targets[i] for i in train_indices], dtype=torch.long)
        x_val = torch.tensor([examples[i] for i in val_indices], dtype=torch.float32)
        y_val = torch.tensor([targets[i] for i in val_indices], dtype=torch.long)

        model = ResNetLSTMGestureModel(
            num_classes=len(usable_labels),
            backbone=backbone,
            temporal_head=temporal_head,
        ).to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        epoch_loss = 0.0
        total_steps = max(1, (len(x_train) + batch_size - 1) // batch_size)

        for _ in range(epochs):
            model.train()
            permutation = torch.randperm(len(x_train))
            loss_acc = 0.0

            for step in range(total_steps):
                start = step * batch_size
                stop = min(len(x_train), (step + 1) * batch_size)
                batch_indices = permutation[start:stop]
                batch_x = x_train[batch_indices].to(self.device)
                batch_y = y_train[batch_indices].to(self.device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.2)
                optimizer.step()

                loss_acc += float(loss.item())

            epoch_loss = loss_acc / total_steps

        train_accuracy = self._evaluate(model, x_train, y_train)
        val_accuracy = self._evaluate(model, x_val, y_val)

        model_path = self._model_path(profile_id)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "labels": usable_labels,
                "backbone": backbone,
                "temporal_head": temporal_head,
                "sequence_length": profile.sequence_length,
            },
            model_path,
        )

        metadata = {
            "profile_id": profile_id,
            "backbone": backbone,
            "temporal_head": temporal_head,
            "labels": usable_labels,
            "samples": len(examples),
            "sequence_length": profile.sequence_length,
            "epochs": epochs,
            "train_accuracy": round(train_accuracy, 4),
            "val_accuracy": round(val_accuracy, 4),
            "loss": round(epoch_loss, 6),
            "trained_at": datetime.utcnow().isoformat(),
            "detail": "ResNet frame encoder + LSTM temporal head",
        }
        self._write_meta(profile_id, metadata)

        model.eval()
        self._cache[profile_id] = (model, metadata)

        return DeepTrainResult(
            profile_id=profile_id,
            backbone=backbone,
            temporal_head=temporal_head,
            labels=usable_labels,
            samples=len(examples),
            epochs=epochs,
            train_accuracy=round(train_accuracy, 4),
            val_accuracy=round(val_accuracy, 4),
            loss=round(epoch_loss, 6),
            model_path=str(model_path),
        )

    def _load_or_cache_model(self, profile_id: str) -> tuple[Any, dict[str, Any]]:
        if torch is None:
            raise ValueError("PyTorch is required for deep prediction.")

        meta = self._read_meta(profile_id)
        if meta is None:
            raise ValueError("No deep model metadata found. Train deep model first.")

        cached = self._cache.get(profile_id)
        if cached is not None:
            cached_meta = cached[1]
            if cached_meta.get("trained_at") == meta.get("trained_at"):
                return cached

        model_path = self._model_path(profile_id)
        if not model_path.exists():
            raise ValueError("Deep model weights not found. Train deep model first.")

        labels = list(meta.get("labels") or [])
        if not labels:
            raise ValueError("Deep model labels are missing. Retrain the deep model.")

        backbone = str(meta.get("backbone") or "resnet18")
        temporal_head = str(meta.get("temporal_head") or "bilstm_attention")

        model = ResNetLSTMGestureModel(
            num_classes=len(labels),
            backbone=backbone,
            temporal_head=temporal_head,
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint.get("model_state")
        if not state_dict:
            raise ValueError("Invalid deep model checkpoint. Retrain the deep model.")

        model.load_state_dict(state_dict)
        model.eval()

        self._cache[profile_id] = (model, meta)
        return model, meta

    def predict(
        self,
        profile_id: str,
        sequence_payload: list[list[list[float | int | dict]]],
        context: str,
        min_confidence: float,
    ) -> PredictionResult:
        if torch is None:
            raise ValueError(
                "PyTorch/torchvision is not available in backend environment. "
                f"Install torch + torchvision first. Import error: {TORCH_IMPORT_ERROR}"
            )

        profile = self.light_engine.get_profile(profile_id)
        if profile is None:
            raise ValueError(f"Profile not found: {profile_id}")

        model, meta = self._load_or_cache_model(profile_id)
        labels = list(meta.get("labels") or [])
        sequence_length = int(meta.get("sequence_length") or profile.sequence_length)

        if not labels:
            raise ValueError("Deep model labels are missing. Retrain the deep model.")

        sequence = self.light_engine._parse_sequence(sequence_payload)
        sequence = self.light_engine._resample(sequence, sequence_length)

        with torch.no_grad():
            sample = torch.tensor([sequence], dtype=torch.float32).to(self.device)
            logits = model(sample)
            probabilities = torch.softmax(logits, dim=1)[0]
            if probabilities.numel() > 1:
                top_values, top_indices = torch.topk(probabilities, k=2)
                confidence = float(top_values[0].item())
                second_confidence = float(top_values[1].item())
                best_index = int(top_indices[0].item())
            else:
                confidence = float(probabilities[0].item())
                second_confidence = 0.0
                best_index = 0

        best_label = labels[best_index]
        confidence_margin = max(0.0, confidence - second_confidence)
        required_margin = 0.08 + (0.02 if len(labels) >= 4 else 0.0)
        accepted = confidence >= min_confidence and (len(labels) == 1 or confidence_margin >= required_margin)
        action = self.light_engine._lookup_action(profile, context, best_label)

        return PredictionResult(
            label=best_label,
            confidence=round(confidence, 4),
            distance=round(max(0.0, 1.0 - confidence), 4),
            second_distance=round(max(0.0, 1.0 - second_confidence), 4),
            accepted=accepted,
            action=action,
        )
