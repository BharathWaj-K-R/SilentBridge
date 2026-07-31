from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import auth, calibration, health, translate
from app.core.config import get_settings
from app.db.session import Base, engine

settings = get_settings()

# Create tables on startup. Fine for SQLite + hackathon scope;
# switch to Alembic migrations if the schema needs to evolve post-demo.
Base.metadata.create_all(bind=engine)

app = FastAPI(title=settings.PROJECT_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix=settings.API_V1_PREFIX)
app.include_router(auth.router, prefix=settings.API_V1_PREFIX)
app.include_router(translate.router, prefix=settings.API_V1_PREFIX)
app.include_router(calibration.router, prefix=settings.API_V1_PREFIX)


@app.get("/")
def root():
    return {"message": "SilentBridge API — see /docs for the interactive API explorer"}
