# SilentBridge

**Few-Shot Signer-Adaptive Continuous Indian Sign Language Translation**

Real-time ISL-to-text translation using a pose + facial-expression fusion
transformer as a frozen base model, plus a lightweight few-shot adapter
(**BridgeAdapter**) that personalizes to a new signer's style/dialect from
~5 minutes of calibration video — without retraining the base model.

## Repo layout

```
silentbridge/
├── backend/          FastAPI + SQLite backend, model code, adapters
│   ├── app/
│   │   ├── api/       route handlers (auth, translate, calibration, health)
│   │   ├── core/      config + security
│   │   ├── db/        SQLAlchemy session + ORM models
│   │   ├── models/    base_model.py (frozen backbone), bridge_adapter.py (core novelty)
│   │   ├── schemas/   Pydantic request/response models
│   │   └── services/  inference + calibration orchestration
│   ├── tests/
│   ├── requirements.txt
│   └── .env.example
├── frontend/          Static HTML/CSS/JS UI (built separately) + API integration layer
│   └── js/            config.js, api.js — connect existing UI to the backend
├── data/              (empty) place dataset download/prep scripts here
├── render.yaml         Render deployment config for BOTH services
└── .gitignore
```

## Targets this project is designed around

| Metric | Target |
|---|---|
| Calibration time | < 5 minutes |
| Adapter params vs base model | < 2% |
| Inference latency | < 500ms |
| Adapter memory overhead | < 10MB |
| Accuracy gain on unseen signers | 10–20% (base vs base+adapter) |

## Datasets

ISLTranslate, ISL-CSLTR, iSign, INCLUDE — all public ISL datasets.
No download/prep scripts are wired up yet; add them under `data/`.

## Local development

### Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # edit as needed
uvicorn app.main:app --reload
# API docs at http://localhost:8000/docs
```

### Frontend
Drop your built HTML/CSS/JS into `frontend/` (see `frontend/README.md`),
then serve it locally:
```bash
npx serve frontend
```

## Deployment (Render)

`render.yaml` defines two separate services:
- `silentbridge-backend` — FastAPI web service, with a persistent Disk
  mounted at `/var/data` so the SQLite DB survives restarts/redeploys.
- `silentbridge-frontend` — static site serving the `frontend/` folder.

Push this repo to GitHub, then in the Render dashboard: **New > Blueprint**,
point it at the repo, and Render will read `render.yaml` and provision both
services. Update `ALLOWED_ORIGINS` (backend) and `API_BASE_URL` in
`frontend/js/config.js` once you know the actual `.onrender.com` URLs.

## Status / what's NOT built yet

- Base model has **not been trained** — `base_model.py` defines the
  architecture; you still need to run training on the ISL datasets and
  save weights to `backend/app/models/weights/base_model.pt`.
- Keypoint extraction (video/webcam → pose+face landmarks, e.g. via
  MediaPipe Holistic) is **not implemented** — API endpoints expect
  pre-extracted keypoint arrays.
- Confidence-aware calibration is a stretch goal, off by default
  (`BridgeAdapterStack.confidence_aware = False`).
- No RBAC, model versioning/rollback, or rate limiting — deferred by design
  to keep scope realistic for the hackathon timeline.
