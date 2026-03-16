# Dynamic Gesture Studio

Dynamic Gesture Studio is a full-stack capstone upgrade for the original Dynamic-Gesture-System repo.

## Why this rebuild

The original repository contains two strong starting points:
- a browser-based gesture-controlled shop demo
- a Python gesture-recognition training/inference pipeline

For a final-year project, that split is too loose. This rebuild turns them into a single platform with:
- dataset session orchestration
- training and evaluation APIs
- a studio-style frontend shell
- a detachable camera overlay
- the original gesture shop embedded as a showcase module

## Architecture

### Frontend
- React + TypeScript + Vite
- Studio dashboard for session setup, sample seeding, and training launch
- Embedded legacy gesture shop under `frontend/public/legacy/gesture-shop.html`
- Floating camera overlay for future realtime inference / assistive controls

### Backend
- FastAPI
- JSON-backed orchestration for sessions and training jobs
- Endpoints for health, blueprint, sessions, sample logging, training jobs, and dashboard metrics
- Lightweight dynamic gesture engine for webcam landmark sequence training and prediction
- Action mapping and execution endpoints (open URL/app, hotkey, text typing)

## Current implementation status

### Implemented now
- frontend studio shell
- backend API foundation
- dataset session provisioning
- sample logging flow
- deterministic training benchmark simulation
- dashboard metrics
- legacy demo embedding
- detachable camera overlay scaffold
- lightweight custom gesture profile creation
- realtime webcam landmark capture using MediaPipe Hands
- dynamic train-from-your-own-gesture pipeline (no heavy pretrained model required)
- residual-temporal feature extractor for sequence classification
- context-based action mapping and execution (global/browser/presentation)

### Next build steps
- true landmark/ROI persistence in dataset sessions
- optional hybrid mode that can switch between lightweight custom model and deep model pipeline
- evaluation pages with confusion matrix and robustness tests
- export/download flow for trained models and configs

## Run locally

### Option A: Fast dev mode (2 local URLs)

Backend:
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --app-dir .
```

Frontend:
```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.
The Vite dev server proxies `/api/*` to `http://127.0.0.1:8000`.

### Option B: Single URL mode (simpler demo)

1. Build frontend once:
```bash
cd frontend
npm install
npm run build
```
2. Start backend:
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --app-dir .
```
3. Open `http://127.0.0.1:8000/studio`.

`/studio` is now served by FastAPI from `frontend/dist` when build output exists.

## Beginner quick use (short)

1. Open `http://127.0.0.1:8000/studio` (single URL mode).
2. In `Live Lab`, follow the `Guided workflow` status row.
3. Step 1: In `Profile`, enter labels and click `Create profile`.
4. Step 2: In `Webcam capture`, click `Start webcam` and keep hand visible.
5. Step 3: Press `Capture + train` for each gesture (4-8 times each).
6. Step 4: Training progress bar (0-100) updates after every captured sample.
7. Step 5: In `Action mapping`, save at least one mapping.
8. Step 6: Click `Start profile` button.
9. Watch `Live activity` logs for detected gesture and action execution messages.

Notes:
- Auto actions are now stability-gated and cooldown-protected.
- If no hand is detected, old frames are cleared to prevent accidental repeated triggers.
- Floating camera panel has an always-visible top-right minimize button.
- `Start profile` works with low-data mode (minimum one sample), but higher progress improves accuracy.
- Same gesture now fires once per hold. To trigger it again, release hand/change gesture, then show gesture again.

## Important: where gesture control works

- Current build controls this app tab (Live Lab + camera panel + virtual pointer).
- Opening `google.com` is supported as an action trigger.
- Browser mode still cannot control OS-level cursor/scroll globally across all tabs/apps.
- For true system-wide control, use the desktop companion app in `desktop_agent/`.

## What profile/session means

- `Profile`: your custom gesture model and label mappings for live use.
- `Session`: backend training history/logging unit used for advanced benchmark tracking.
- For normal usage, you only need `Profile` + `Live Lab`.

## Use profile outside the webpage

You can call backend APIs from any app/script:

- `GET /api/light/profiles`
- `POST /api/light/predict`
- `POST /api/light/execute`

This allows integrating your trained profile with another frontend, local script, or automation app.

## Desktop companion app (background/system-wide)

The `desktop_agent/` module provides practical global controls:

- OS cursor move from hand landmarks
- pinch-to-click and two-finger scroll
- gesture-triggered mappings (URL, app launch, hotkey, type text)
- stable vote + cooldown protection to avoid repeated accidental triggers

Quick start:

```bash
cd desktop_agent
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy config.example.json config.json
python agent.py --config config.json
```

Security options:

- local-only backend check enabled by default in desktop agent
- optional API token lock for `/api/light/*` using `DGS_API_TOKEN` (backend) and `DGS_AGENT_TOKEN` (desktop agent)
- optional encrypted config file via `desktop_agent/secure_config.py`

Windows launcher:

```powershell
./desktop_agent/run_agent.ps1
```

## API quick reference (new lightweight endpoints)

- `GET /api/light/profiles`
- `POST /api/light/profiles`
- `POST /api/light/train`
- `POST /api/light/predict`
- `POST /api/light/mappings`
- `POST /api/light/execute`

## How to test on your laptop (webcam + custom gestures)

1. Start app in either Option A (dev mode) or Option B (single URL mode).
2. Open `Live Lab` and create/select a profile with labels such as `pinch,peace,three,open,fist`.
3. In `Webcam capture`, pick a camera device (or keep `Auto select`).
4. Click `Start webcam` and allow permission.
5. If you get `Requested camera device not found`, switch camera in the dropdown, click `Refresh camera list`, then retry.
6. Hold one gesture steadily and click `Capture + train`.
7. Capture 4-8 samples per label for better stability.
8. In `Action mapping`, save at least one mapping (example: `three -> open_url -> https://www.google.com`).
9. Enable `Live recognition mode` and (optionally) `Auto execute mapped action`.
10. In `Gesture typing + search`, configure which gesture means next/prev/select/backspace/space/submit.
11. Use gestures to type text, then submit gesture to open/search.
12. For direct site opening, try text like `open youtube` or `open github`.

## Notes on lightweight objective

- This path is intentionally lightweight and dynamic:
	- no mandatory heavy pretrained model load for basic usage
	- fast custom training from your own webcam sequence
	- model updates incrementally as you capture new samples
- Temporal behavior is handled with residual (frame-delta) sequence features.
- You can still keep a deep-model track for research comparison in later phases.

## Optional hotkey support

If you want `hotkey` and `type_text` execution modes on Windows:

```bash
cd backend
pip install keyboard
```

Then map gestures to values like:
- `ctrl+shift+b`
- `alt+tab`

Use with care, because hotkeys are sent to the active window.
