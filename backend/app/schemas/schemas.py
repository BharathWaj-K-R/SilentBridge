"""
Pydantic request/response models. Kept separate from ORM models so the API
contract can evolve independently of the DB schema.
"""
import datetime as dt

from pydantic import BaseModel, ConfigDict


# ---------- Users / auth ----------

class UserCreate(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    username: str
    created_at: dt.datetime


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ---------- Calibration / adapter ----------

class CalibrationStartRequest(BaseModel):
    user_id: int


class CalibrationResult(BaseModel):
    adapter_id: int
    calibration_seconds: float
    param_count: int
    accuracy_gain_pct: float | None = None


class AdapterOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    owner_id: int
    calibration_seconds: float
    param_count: int | None
    accuracy_gain_pct: float | None
    created_at: dt.datetime


# ---------- Translation ----------

class TranslationRequest(BaseModel):
    user_id: int | None = None
    adapter_id: int | None = None  # if None, base model only


class TranslationResult(BaseModel):
    predicted_text: str
    confidence: float
    latency_ms: float
    used_adapter: bool


# ---------- Ablation / eval ----------

class AblationRow(BaseModel):
    config_name: str  # e.g. "base_only", "base+face", "base+adapter", "base+face+adapter"
    accuracy: float
    calibration_seconds: float | None = None
