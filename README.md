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
ISLTranslate training is wired for the Exploration Lab release; see
`data/README.md` for the expected local layout and trainer command.

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

## Scope (team of 2)

Trimmed deliberately to fit a two-person team:

- Base model stays the small custom pose+face transformer already in
  `base_model.py` (no public pretrained ISL checkpoint exists to fine-tune
  instead) — kept lightweight on purpose so it trains on a small data subset
  in reasonable time on Colab, not a full-scale corpus run.
- Training uses a small subset of ISLTranslate/iSign (a few hundred clips),
  not the full ~228GB release — enough to demonstrate the adapter effect,
  not to chase SOTA translation accuracy.
- Evaluation is a 2-way comparison only: base-only vs base+adapter. No
  broader ablation matrix.
- Confidence-aware calibration is cut from scope entirely, not just
  deferred — the hook has been removed from `bridge_adapter.py`.

## Status / what's NOT built yet

- Base model training is now wired for preprocessed ISLTranslate keypoints via
  `backend/app/training/train_base_model.py`; you still need to download the
  upstream dataset/features and run the trainer to produce
  `backend/app/models/weights/base_model.pt`.
- Keypoint extraction (video/webcam → pose+face landmarks, e.g. via
  MediaPipe Holistic) is **not implemented** — API endpoints expect
  pre-extracted keypoint arrays.
- No RBAC, model versioning/rollback, or rate limiting — deferred by design
  to keep scope realistic for the hackathon timeline.
