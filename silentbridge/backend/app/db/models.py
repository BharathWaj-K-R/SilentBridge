"""
ORM models: kept intentionally minimal for the hackathon scope.
No RBAC, no model-versioning tables, no audit tables — those are future work.
"""
import datetime as dt

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from app.db.session import Base


class User(Base):
    """A signer using the system. JWT auth only kicks in if the demo needs
    multiple simultaneous users; a single hardcoded demo user is fine otherwise."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    adapters = relationship("SignerAdapter", back_populates="owner")


class SignerAdapter(Base):
    """Metadata for one signer's calibrated BridgeAdapter weights.
    The actual weight tensors are stored on disk under ADAPTER_WEIGHTS_DIR;
    this row just tracks which file belongs to which user and how it did."""

    __tablename__ = "signer_adapters"

    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    weights_path = Column(String, nullable=False)
    calibration_seconds = Column(Float, nullable=False)
    param_count = Column(Integer, nullable=True)
    accuracy_gain_pct = Column(Float, nullable=True)  # measured vs base-only, on held-out clips
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    owner = relationship("User", back_populates="adapters")


class TranslationLog(Base):
    """Lightweight log of translation requests, useful for the demo/ablation
    numbers (latency, confidence) without building a full analytics stack."""

    __tablename__ = "translation_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    adapter_id = Column(Integer, ForeignKey("signer_adapters.id"), nullable=True)
    predicted_text = Column(String, nullable=False)
    confidence = Column(Float, nullable=True)
    latency_ms = Column(Float, nullable=True)
    used_adapter = Column(Integer, default=0)  # 0/1 bool for sqlite simplicity
    created_at = Column(DateTime, default=dt.datetime.utcnow)
