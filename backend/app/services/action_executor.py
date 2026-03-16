from __future__ import annotations

import os
import subprocess
import webbrowser

from app.lightweight_schemas import GestureAction


def execute_action(action: GestureAction) -> tuple[bool, str]:
    if not action.enabled:
        return False, "Action is disabled"

    if action.action_type == "none":
        return True, "No action mapped"

    value = (action.value or "").strip()
    if action.action_type != "none" and not value:
        return False, "Action value is empty"

    try:
        if action.action_type == "open_url":
            webbrowser.open(value)
            return True, f"Opened URL: {value}"

        if action.action_type == "open_app":
            if os.path.exists(value):
                os.startfile(value)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(value, shell=True)
            return True, f"Launched app/command: {value}"

        if action.action_type in ("hotkey", "type_text"):
            try:
                import keyboard  # type: ignore
            except ImportError:
                return False, "keyboard package not installed. Install with: pip install keyboard"

            if action.action_type == "hotkey":
                keyboard.send(value)
                return True, f"Hotkey sent: {value}"

            keyboard.write(value)
            return True, f"Typed text: {value}"

        return False, f"Unsupported action type: {action.action_type}"
    except Exception as exc:
        return False, f"Execution failed: {exc}"
