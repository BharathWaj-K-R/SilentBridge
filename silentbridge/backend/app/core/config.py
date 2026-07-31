"""
Application configuration.
Reads from environment variables so the same code runs locally and on Render.
"""
import os
from functools import lru_cache


class Settings:
    # --- General ---
    PROJECT_NAME: str = "SilentBridge"
    API_V1_PREFIX: str = "/api/v1"
    ENV: str = os.getenv("ENV", "development")

    # --- Database ---
    # SQLite by default (file-based, zero setup). On Render, point this at
    # a path inside a mounted Disk so data survives restarts/redeploys, e.g.
    # sqlite:////var/data/silentbridge.db
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./silentbridge.db")

    # --- Auth ---
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-change-me")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
    JWT_ALGORITHM: str = "HS256"

    # --- CORS ---
    # Comma-separated list of allowed origins, e.g. the Render static site URL.
    ALLOWED_ORIGINS: list[str] = [
        origin.strip()
        for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:5500,http://127.0.0.1:5500").split(",")
        if origin.strip()
    ]

    # --- Model / adapter ---
    BASE_MODEL_PATH: str = os.getenv("BASE_MODEL_PATH", "./app/models/weights/base_model.pt")
    ADAPTER_WEIGHTS_DIR: str = os.getenv("ADAPTER_WEIGHTS_DIR", "./app/models/weights/adapters")
    CALIBRATION_MIN_SECONDS: int = int(os.getenv("CALIBRATION_MIN_SECONDS", "300"))  # 5 min target
    MAX_INFERENCE_LATENCY_MS: int = int(os.getenv("MAX_INFERENCE_LATENCY_MS", "500"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
