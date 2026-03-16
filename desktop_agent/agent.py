from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
import webbrowser
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import mediapipe as mp
import pyautogui
import requests


HAND_LANDMARKER_MODEL_URLS = [
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
]
HAND_LANDMARKER_MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"


@dataclass
class AgentConfig:
    backend_url: str = "http://127.0.0.1:8000"
    profile_id: str = ""
    context: str = "global"
    min_confidence: float = 0.55
    sequence_length: int = 16
    predict_interval_ms: int = 180
    allow_remote_backend: bool = False

    pointer_enabled: bool = True
    pointer_smoothing: float = 0.72
    click_enabled: bool = True
    pinch_click_threshold: float = 0.055
    scroll_enabled: bool = True
    scroll_sensitivity: int = 1800

    action_cooldown_ms: int = 2000
    show_preview: bool = True

    camera_index: int = -1
    camera_scan_max: int = 5
    camera_backend: str = "auto"
    camera_width: int = 640
    camera_height: int = 480


def _now_ms() -> float:
    return time.time() * 1000.0


def _ensure_hand_landmarker_model(path: Path) -> Path:
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for url in HAND_LANDMARKER_MODEL_URLS:
        try:
            with requests.get(url, timeout=25, stream=True) as response:
                response.raise_for_status()
                temp_path = path.with_suffix(path.suffix + ".tmp")
                with temp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 512):
                        if chunk:
                            handle.write(chunk)
                temp_path.replace(path)
                print(f"Downloaded hand model: {path}")
                return path
        except Exception as exc:
            failures.append(f"{url} -> {exc}")

    detail = "; ".join(failures) if failures else "unknown error"
    raise RuntimeError(f"Unable to download hand landmarker model. {detail}")


def _camera_backend_candidates(preferred: str) -> list[tuple[str, int | None]]:
    normalized = (preferred or "auto").strip().lower()
    valid = {"auto", "dshow", "msmf", "any"}
    if normalized not in valid:
        normalized = "auto"

    backend_map: dict[str, int | None] = {"any": None}
    if hasattr(cv2, "CAP_DSHOW"):
        backend_map["dshow"] = int(cv2.CAP_DSHOW)
    if hasattr(cv2, "CAP_MSMF"):
        backend_map["msmf"] = int(cv2.CAP_MSMF)

    if normalized != "auto":
        if normalized in backend_map:
            return [(normalized, backend_map[normalized])]
        return [("any", None)]

    candidates: list[tuple[str, int | None]] = []
    if os.name == "nt":
        if "dshow" in backend_map:
            candidates.append(("dshow", backend_map["dshow"]))
        if "msmf" in backend_map:
            candidates.append(("msmf", backend_map["msmf"]))
    candidates.append(("any", None))
    return candidates


def _camera_indices(cfg: AgentConfig) -> list[int]:
    if int(cfg.camera_index) >= 0:
        return [int(cfg.camera_index)]
    scan_max = max(1, min(10, int(cfg.camera_scan_max)))
    return list(range(scan_max))


def _open_webcam(cfg: AgentConfig) -> tuple[Any, str]:
    attempts: list[str] = []
    for index in _camera_indices(cfg):
        for backend_name, backend_code in _camera_backend_candidates(cfg.camera_backend):
            cap = cv2.VideoCapture(index, backend_code) if backend_code is not None else cv2.VideoCapture(index)
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                attempts.append(f"{index}/{backend_name}:open_failed")
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg.camera_width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg.camera_height))

            frame_ready = False
            for _ in range(12):
                ok, _ = cap.read()
                if ok:
                    frame_ready = True
                    break
                time.sleep(0.02)

            if not frame_ready:
                cap.release()
                attempts.append(f"{index}/{backend_name}:no_frames")
                continue

            return cap, f"index={index}, backend={backend_name}"

    sampled = ", ".join(attempts[:10])
    if len(attempts) > 10:
        sampled += ", ..."
    raise RuntimeError(
        "Could not open webcam. "
        f"Tried: {sampled or 'no candidates'}. "
        "Close other camera apps (Camera/Zoom/Meet/Teams), then retry. "
        "If needed, set camera_index in config.json (for example 1 or 2) "
        "or run with --camera-index 1."
    )


def _safe_host_check(backend_url: str, allow_remote_backend: bool) -> None:
    parsed = urlparse(backend_url)
    host = parsed.hostname or ""
    if allow_remote_backend:
        return
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError(
            "Remote backend blocked by security policy. Use localhost/127.0.0.1 or set allow_remote_backend=true."
        )


def _decrypt_if_needed(raw_bytes: bytes) -> bytes:
    key = os.getenv("DGS_AGENT_CONFIG_KEY", "").strip()
    if not key:
        return raw_bytes

    try:
        from cryptography.fernet import Fernet
    except Exception as exc:
        raise RuntimeError("DGS_AGENT_CONFIG_KEY is set, but cryptography is not installed.") from exc

    try:
        fernet = Fernet(key.encode("utf-8"))
        return fernet.decrypt(raw_bytes)
    except Exception as exc:
        raise RuntimeError("Failed to decrypt config. Check DGS_AGENT_CONFIG_KEY.") from exc


def load_config(path: Path) -> AgentConfig:
    if not path.exists():
        return AgentConfig()

    raw = path.read_bytes()
    payload = json.loads(_decrypt_if_needed(raw).decode("utf-8"))
    merged = {**AgentConfig().__dict__, **payload}
    return AgentConfig(**merged)


class BackendClient:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.base_url = cfg.backend_url.rstrip("/")
        self.token = os.getenv("DGS_AGENT_TOKEN", "").strip()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["X-DGS-Token"] = self.token
        return headers

    def list_profiles(self) -> list[dict[str, Any]]:
        response = requests.get(
            f"{self.base_url}/api/light/profiles",
            timeout=1.8,
            headers=self._headers(),
        )
        response.raise_for_status()
        return response.json()

    def get_profile(self, profile_id: str) -> dict[str, Any]:
        response = requests.get(
            f"{self.base_url}/api/light/profiles/{profile_id}",
            timeout=1.8,
            headers=self._headers(),
        )
        response.raise_for_status()
        return response.json()

    def predict(self, profile_id: str, sequence: list[list[list[float]]]) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/api/light/predict",
            timeout=1.8,
            headers=self._headers(),
            data=json.dumps(
                {
                    "profile_id": profile_id,
                    "sequence": sequence,
                    "context": self.cfg.context,
                    "min_confidence": self.cfg.min_confidence,
                }
            ),
        )
        response.raise_for_status()
        return response.json()


class ActionDispatcher:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.last_action_fired_ms: dict[str, float] = {}

    def _cooldown_ok(self, key: str, action: dict[str, Any]) -> bool:
        now = _now_ms()
        action_cooldown = int(action.get("cooldown_ms", 0) or 0)
        effective = max(action_cooldown, self.cfg.action_cooldown_ms)
        last = self.last_action_fired_ms.get(key, 0.0)
        if now - last < effective:
            return False
        self.last_action_fired_ms[key] = now
        return True

    def _normalize_hotkey_tokens(self, raw: str) -> list[str]:
        aliases = {
            "control": "ctrl",
            "command": "winleft",
            "win": "winleft",
            "windows": "winleft",
            "option": "alt",
            "return": "enter",
            "esc": "escape",
            "del": "delete",
            "pgup": "pageup",
            "pgdn": "pagedown",
        }
        tokens = [token.strip().lower() for token in raw.split("+") if token.strip()]
        return [aliases.get(token, token) for token in tokens]

    def execute(self, label: str, action: dict[str, Any]) -> tuple[bool, str]:
        if not action or not action.get("enabled", True):
            return False, "No enabled action"

        action_type = str(action.get("action_type", "none"))
        value = str(action.get("value", "")).strip()

        if action_type == "none":
            return False, "Mapped action type is none"

        key = f"{label}:{action_type}:{value}"
        if not self._cooldown_ok(key, action):
            return False, "Cooldown active"

        try:
            if action_type == "open_url":
                webbrowser.open(value, new=2, autoraise=False)
                return True, f"Opened URL: {value}"

            if action_type == "open_app":
                if os.path.exists(value):
                    os.startfile(value)  # type: ignore[attr-defined]
                else:
                    subprocess.Popen(value, shell=True)
                return True, f"Opened app/command: {value}"

            if action_type == "hotkey":
                hotkey_parts = self._normalize_hotkey_tokens(value)
                if not hotkey_parts:
                    return False, "Hotkey value is empty"
                pyautogui.hotkey(*hotkey_parts)
                return True, f"Hotkey: {value}"

            if action_type == "type_text":
                if not value:
                    return False, "type_text value is empty"
                pyautogui.write(value, interval=0.005)
                return True, f"Typed text: {value}"

            return False, f"Unsupported action type: {action_type}"
        except Exception as exc:
            return False, f"Action failed: {exc}"


class PointerController:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.pointer_x = 0.5
        self.pointer_y = 0.5
        self.pinch_latched = False
        self.scroll_anchor_y: float | None = None

    @staticmethod
    def _distance(a: list[float], b: list[float]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    @staticmethod
    def _is_finger_extended(tip: list[float], pip: list[float]) -> bool:
        return tip[1] < pip[1]

    def reset_hand_state(self) -> None:
        self.pinch_latched = False
        self.scroll_anchor_y = None

    def update(self, landmarks: list[list[float]]) -> list[str]:
        events: list[str] = []

        index_tip = landmarks[8]
        target_x = max(0.0, min(1.0, 1.0 - index_tip[0]))
        target_y = max(0.0, min(1.0, index_tip[1]))

        self.pointer_x = self.pointer_x + (target_x - self.pointer_x) * self.cfg.pointer_smoothing
        self.pointer_y = self.pointer_y + (target_y - self.pointer_y) * self.cfg.pointer_smoothing

        if self.cfg.pointer_enabled:
            width, height = pyautogui.size()
            px = int(self.pointer_x * width)
            py = int(self.pointer_y * height)
            pyautogui.moveTo(px, py, duration=0)

        if self.cfg.click_enabled:
            pinch_dist = self._distance(landmarks[4], landmarks[8])
            pinching = pinch_dist < self.cfg.pinch_click_threshold
            if pinching and not self.pinch_latched:
                pyautogui.click()
                self.pinch_latched = True
                events.append("Mouse click")
            elif not pinching:
                self.pinch_latched = False

        if self.cfg.scroll_enabled:
            index_extended = self._is_finger_extended(landmarks[8], landmarks[6])
            middle_extended = self._is_finger_extended(landmarks[12], landmarks[10])
            two_finger_mode = index_extended and middle_extended

            if two_finger_mode:
                avg_tip_y = (landmarks[8][1] + landmarks[12][1]) / 2.0
                if self.scroll_anchor_y is None:
                    self.scroll_anchor_y = avg_tip_y
                else:
                    delta = avg_tip_y - self.scroll_anchor_y
                    if abs(delta) > 0.012:
                        amount = int(-delta * self.cfg.scroll_sensitivity)
                        if amount != 0:
                            pyautogui.scroll(amount)
                            events.append("Scroll")
                            self.scroll_anchor_y = avg_tip_y
            else:
                self.scroll_anchor_y = None

        return events


def draw_preview(
    frame_bgr: Any,
    mp_drawing: Any,
    mp_hands: Any,
    hand_landmarks: Any,
    status: str,
    profile_id: str,
    task_connections: list[Any] | None = None,
) -> None:
    if hand_landmarks is not None:
        if mp_drawing is not None and mp_hands is not None:
            mp_drawing.draw_landmarks(
                frame_bgr,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
            )
        else:
            connections = task_connections or []
            frame_h, frame_w = frame_bgr.shape[:2]
            for connection in connections:
                if hasattr(connection, "start") and hasattr(connection, "end"):
                    start_idx = int(connection.start)
                    end_idx = int(connection.end)
                else:
                    start_idx = int(connection[0])
                    end_idx = int(connection[1])

                if start_idx >= len(hand_landmarks) or end_idx >= len(hand_landmarks):
                    continue

                x1 = int(hand_landmarks[start_idx][0] * frame_w)
                y1 = int(hand_landmarks[start_idx][1] * frame_h)
                x2 = int(hand_landmarks[end_idx][0] * frame_w)
                y2 = int(hand_landmarks[end_idx][1] * frame_h)
                cv2.line(frame_bgr, (x1, y1), (x2, y2), (94, 181, 245), 2)

            for point in hand_landmarks:
                px = int(point[0] * frame_w)
                py = int(point[1] * frame_h)
                cv2.circle(frame_bgr, (px, py), 4, (106, 237, 164), -1)

    cv2.putText(frame_bgr, f"Profile: {profile_id}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (120, 240, 140), 2)
    cv2.putText(frame_bgr, f"Status: {status}", (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (120, 210, 250), 2)
    cv2.putText(frame_bgr, "Press q to quit", (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1)


def choose_profile(client: BackendClient, requested_profile_id: str) -> tuple[str, int]:
    if requested_profile_id:
        profile = client.get_profile(requested_profile_id)
        return str(profile["id"]), int(profile.get("sequence_length", 16))

    profiles = client.list_profiles()
    if not profiles:
        raise RuntimeError("No profiles found. Train a profile in web app first.")

    latest = profiles[-1]
    chosen_id = str(latest["id"])
    chosen_seq = int(latest.get("sequence_length", 16))
    return chosen_id, chosen_seq


def run(cfg: AgentConfig) -> None:
    _safe_host_check(cfg.backend_url, cfg.allow_remote_backend)

    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0

    client = BackendClient(cfg)
    dispatcher = ActionDispatcher(cfg)
    pointer = PointerController(cfg)

    profile_id, backend_seq_len = choose_profile(client, cfg.profile_id)
    cfg.profile_id = profile_id
    cfg.sequence_length = max(8, min(64, backend_seq_len))

    print(f"Using profile: {cfg.profile_id}")
    print(f"Context: {cfg.context}")
    if cfg.show_preview:
        print("Desktop agent is running. Show trained gestures. Press q in preview window to stop.")
    else:
        print("Desktop agent is running in headless mode. Press Ctrl+C in terminal to stop.")

    cap, camera_source = _open_webcam(cfg)
    print(f"Camera source: {camera_source}")

    use_legacy_solutions = bool(getattr(mp, "solutions", None) and hasattr(mp.solutions, "hands"))

    mp_hands = None
    mp_drawing = None
    task_connections: list[Any] = []
    hands = None
    hands_ctx = None

    if use_legacy_solutions:
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        hands_ctx = mp_hands.Hands(
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.4,
        )
        hands = hands_ctx.__enter__()
        print("Hand backend: mediapipe.solutions")
    else:
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        model_path = _ensure_hand_landmarker_model(HAND_LANDMARKER_MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.4,
        )
        hands = vision.HandLandmarker.create_from_options(options)
        task_connections = list(vision.HandLandmarksConnections.HAND_CONNECTIONS)
        print(f"Hand backend: mediapipe.tasks ({model_path})")

    prediction_trail: deque[str] = deque(maxlen=3)
    sequence_buffer: deque[list[list[float]]] = deque(maxlen=max(cfg.sequence_length * 4, 96))

    latched_gesture = ""
    last_predict_at_ms = 0.0
    last_hand_seen_at_ms = 0.0
    status = "WATCHING"

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            now_ms = _now_ms()

            hand_landmarks = None
            frame_points: list[list[float]] | None = None

            if use_legacy_solutions:
                results = hands.process(frame_rgb)
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    frame_points = [
                        [float(point.x), float(point.y), float(point.z)]
                        for point in hand_landmarks.landmark[:21]
                    ]
            else:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                task_result = hands.detect_for_video(mp_image, int(now_ms))
                if task_result.hand_landmarks:
                    hand_landmarks = task_result.hand_landmarks[0]
                    frame_points = [
                        [float(point.x), float(point.y), float(point.z)]
                        for point in hand_landmarks[:21]
                    ]

            if frame_points is not None:
                last_hand_seen_at_ms = now_ms
                sequence_buffer.append(frame_points)

                pointer_events = pointer.update(frame_points)
                for event in pointer_events:
                    print(f"Pointer event: {event}")

                enough_frames = len(sequence_buffer) >= 8
                due_for_predict = now_ms - last_predict_at_ms >= cfg.predict_interval_ms
                if enough_frames and due_for_predict:
                    last_predict_at_ms = now_ms
                    sequence = list(sequence_buffer)[-cfg.sequence_length :]
                    try:
                        prediction = client.predict(cfg.profile_id, sequence)
                    except Exception as exc:
                        status = f"Predict error: {exc}"
                        if cfg.show_preview:
                            draw_preview(
                                frame_bgr,
                                mp_drawing,
                                mp_hands,
                                hand_landmarks if use_legacy_solutions else frame_points,
                                status,
                                cfg.profile_id,
                                task_connections=task_connections,
                            )
                            cv2.imshow("Dynamic Gesture Desktop Agent", frame_bgr)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                break
                        continue

                    accepted = bool(prediction.get("accepted", False))
                    label_raw = str(prediction.get("label") or "").strip().lower()

                    if not accepted or not label_raw:
                        prediction_trail.clear()
                        status = "WATCHING"
                    else:
                        prediction_trail.append(label_raw)
                        stable_votes = sum(1 for item in prediction_trail if item == label_raw)
                        stable = stable_votes >= 2

                        if stable and label_raw != latched_gesture:
                            latched_gesture = label_raw
                            confidence = float(prediction.get("confidence") or 0.0)
                            status = f"DETECTED {label_raw} ({confidence * 100:.1f}%)"
                            print(status)

                            action = prediction.get("action")
                            if action and action.get("enabled", True):
                                ok_action, detail = dispatcher.execute(label_raw, action)
                                if ok_action:
                                    status = f"ACTION: {detail}"
                                    print(status)
                                else:
                                    print(f"Action skipped: {detail}")
                        elif stable and label_raw == latched_gesture:
                            status = f"HOLD {label_raw}"
            else:
                if now_ms - last_hand_seen_at_ms > 650:
                    latched_gesture = ""
                    prediction_trail.clear()
                    sequence_buffer.clear()
                    pointer.reset_hand_state()
                    status = "WATCHING"

            if cfg.show_preview:
                draw_preview(
                    frame_bgr,
                    mp_drawing,
                    mp_hands,
                    hand_landmarks if use_legacy_solutions else frame_points,
                    status,
                    cfg.profile_id,
                    task_connections=task_connections,
                )
                cv2.imshow("Dynamic Gesture Desktop Agent", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if hands_ctx is not None:
            hands_ctx.__exit__(None, None, None)
        elif hands is not None:
            close_method = getattr(hands, "close", None)
            if callable(close_method):
                close_method()
        if cfg.show_preview:
            cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic Gesture Desktop Agent")
    parser.add_argument("--config", default="desktop_agent/config.json", help="Path to config file")
    parser.add_argument("--profile-id", default="", help="Profile id to run. If empty, uses latest profile.")
    parser.add_argument("--context", default="global", help="Mapping context: global/browser/presentation")
    parser.add_argument("--camera-index", type=int, default=None, help="Camera index. -1 means auto scan.")
    parser.add_argument("--camera-backend", default="", help="Camera backend: auto/dshow/msmf/any")
    parser.add_argument("--headless", action="store_true", help="Run without preview window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))

    if args.profile_id:
        cfg.profile_id = args.profile_id.strip()
    if args.context:
        cfg.context = args.context.strip().lower()
    if args.camera_index is not None:
        cfg.camera_index = int(args.camera_index)
    if args.camera_backend:
        cfg.camera_backend = args.camera_backend.strip().lower()
    if args.headless:
        cfg.show_preview = False

    run(cfg)


if __name__ == "__main__":
    main()
