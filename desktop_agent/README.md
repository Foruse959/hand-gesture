# Desktop Agent (System-Wide Control)

This module runs webcam tracking locally and controls your system cursor/actions using your trained profile.

## Fast first run (copy-paste)

Prerequisites:

- Windows 10/11
- webcam
- backend running at http://127.0.0.1:8000

From repository root, open PowerShell and run:

```powershell
cd desktop_agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item config.example.json config.json -Force
python agent.py --config config.json
```

Stop the agent with `q` in the preview window.

Note: on first run, the agent may download `models/hand_landmarker.task` once.

## Before running desktop agent

You must already have at least one trained profile in Studio:

1. Open `http://127.0.0.1:8000/studio`.
2. Create profile labels.
3. Capture and train 4-8 samples per label.
4. Save at least one gesture action mapping.

If no profile exists, desktop agent will show: `No profiles found. Train a profile in web app first.`

## Run options

```powershell
python agent.py --config config.json
python agent.py --config config.json --context browser
python agent.py --config config.json --profile-id <your_profile_id>
python agent.py --config config.json --camera-index 1
python agent.py --config config.json --camera-index 1 --camera-backend dshow
python agent.py --config config.json --headless
```

PowerShell launcher:

```powershell
./run_agent.ps1
```

## Most common errors (quick fixes)

`Could not find a version that satisfies the requirement mediapipe==0.10.21`

- Cause: old pinned version.
- Fix: use the current `desktop_agent/requirements.txt` and rerun:

```powershell
python -m pip install -r requirements.txt
```

`ModuleNotFoundError: No module named 'cv2'`

- Cause: dependency install stopped before OpenCV was installed.
- Fix:

```powershell
python -m pip install -r requirements.txt
python -c "import cv2, mediapipe; print('ok')"
```

If import check prints `ok`, run the agent again.

`RuntimeError: Could not open webcam.`

- Cause: selected camera index/backend unavailable, or camera is locked by another app.
- Fix order:

```powershell
python agent.py --config config.json --camera-index 1
python agent.py --config config.json --camera-index 2
python agent.py --config config.json --camera-index 0 --camera-backend dshow
python agent.py --config config.json --camera-index 0 --camera-backend msmf
```

- Also close camera-using apps first (Windows Camera, Zoom, Teams, Meet, OBS).
- If needed, set these in `config.json` permanently: `camera_index`, `camera_backend`.

`q is not recognized as a cmdlet`

- `q` is only for the OpenCV preview window when it has focus.
- If the process already crashed, there is nothing to stop and typing `q` in PowerShell will always fail.

## What it controls

- cursor move (index fingertip)
- pinch click (thumb-index pinch)
- two-finger scroll
- mapped actions from backend profile:
	- `open_url`
	- `open_app`
	- `hotkey`
	- `type_text`

Built-in safety:

- stable vote before action fire
- per-action cooldown
- stale-hand reset to avoid repeated accidental triggers

## Optional security

### API token lock

Enable token check for all `/api/light/*` routes:

```powershell
$env:DGS_API_TOKEN="your-long-random-token"
```

Set the same token for desktop agent:

```powershell
$env:DGS_AGENT_TOKEN="your-long-random-token"
```

If `DGS_API_TOKEN` is unset, token auth is disabled.

### Encrypted config

Generate key:

```powershell
python secure_config.py generate-key --out agent.key
```

Encrypt config:

```powershell
python secure_config.py encrypt --in config.json --out config.json.enc --key agent.key
```

Run with encrypted config:

```powershell
$env:DGS_AGENT_CONFIG_KEY=(Get-Content agent.key -Raw).Trim()
python agent.py --config config.json.enc
```

## Privacy notes

- Frames are processed locally on your machine.
- Localhost backend is enforced by default unless `allow_remote_backend=true`.
- Test with safe mappings first because actions affect whatever app is active.
