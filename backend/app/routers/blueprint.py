from __future__ import annotations

from fastapi import APIRouter

from app.schemas import BlueprintModule

router = APIRouter(prefix="/api", tags=["blueprint"])


@router.get("/blueprint", response_model=list[BlueprintModule])
def blueprint() -> list[BlueprintModule]:
    return [
        BlueprintModule(
            title="Gesture Data Studio",
            summary="Guided browser capture, label planning, sequence control, and preprocessing-aware dataset management.",
            outcomes=["Webcam capture", "Class balancing", "Sequence-length governance"],
        ),
        BlueprintModule(
            title="Training and Evaluation Lab",
            summary="ResNet backbone selection, temporal modeling, augmentation policy, and benchmark reporting.",
            outcomes=["Accuracy tracking", "Latency estimation", "Cross-user robustness"],
        ),
        BlueprintModule(
            title="Live Prediction and API Deployment",
            summary="Run real-time recognition in the browser and expose trained behavior through a clean REST contract.",
            outcomes=["Browser inference", "REST integration", "Export surfaces"],
        ),
        BlueprintModule(
            title="Showcase and Overlay Experience",
            summary="A detachable gesture panel and a polished legacy demo module for presentations, demos, and viva usage.",
            outcomes=["Floating overlay", "Legacy shop embed", "Presentation-ready UX"],
        ),
    ]
