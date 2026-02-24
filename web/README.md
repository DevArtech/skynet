# Skyjo Web App

## What it includes

- FastAPI backend with **stateless** endpoints (`state in -> state out`)
- Trained agent inference (`baseline` and `belief`) via MCTS
- React + TypeScript frontend with:
  - Game setup (mode, seats, checkpoints, simulations)
  - Visual board rendering
  - Legal-action controls
  - Action history log
  - Step/play/pause AI controls

## Install

```bash
pip install -e .
cd web/frontend
npm install
cd ../..
```

## Run

From repository root:

```bash
python web/run_dev.py
```

Or run each process manually:

```bash
uvicorn web.backend.main:app --reload --port 8000
cd web/frontend
npm run dev
```

Frontend runs at `http://localhost:5173` and backend at `http://localhost:8000`.

## Session ownership model

- The frontend owns session state entirely.
- Backend does not keep per-game server sessions.
- Frontend sends serialized environment state on each call.
- Backend reconstructs env, applies action/inference, and returns updated state + view.

Key endpoints:

- `POST /api/session/new`
- `POST /api/session/view`
- `POST /api/session/legal-actions`
- `POST /api/session/apply-action`
- `POST /api/session/infer-action`
- `POST /api/session/autoplay`

## Agent checkpoints

- If `checkpoint_path` is not set for a baseline seat, backend tries:
  - `runs/muzero_baseline/checkpoints/checkpoint_iter_*.pt` (latest)
- If `checkpoint_path` is not set for a belief seat, backend tries:
  - `runs/muzero_belief/checkpoints/checkpoint_iter_*.pt` (latest)
- You can provide explicit relative paths in setup, e.g.:
  - `runs/muzero_baseline/checkpoints/checkpoint_iter_50.pt`
  - `runs/muzero_belief/checkpoints/checkpoint_iter_50.pt`

## Smoke tests

```bash
pytest web/backend/tests/test_api.py
cd web/frontend
npm test
```
