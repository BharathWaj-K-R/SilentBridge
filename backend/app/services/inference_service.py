"""
Inference service: loads the frozen base model once (module-level singleton),
optionally applies a signer's BridgeAdapter, and returns decoded text +
confidence + latency — the numbers the ablation study and demo both need.

NOTE: keypoint extraction (e.g. MediaPipe Holistic on raw video/webcam
frames) is NOT implemented here. This service expects pre-extracted pose
and face keypoint tensors. Wire up the extraction step in a separate
preprocessing module once you've picked a keypoint extractor.
"""
import time

import torch

from app.core.config import get_settings
from app.models.base_model import load_frozen_base_model
from app.models.bridge_adapter import BridgeAdapterStack

settings = get_settings()

# Loaded once at process startup, reused across requests.
_base_model = None
_id_to_token: dict[int, str] = {}  # placeholder vocab; replace with real tokenizer output


def get_base_model():
    global _base_model
    if _base_model is None:
        _base_model = load_frozen_base_model(settings.BASE_MODEL_PATH)
    return _base_model


CTC_BLANK_ID = 0


def decode_logits(logits: torch.Tensor) -> tuple[str, float]:
    """Greedy CTC decode: collapse consecutive repeats, then drop blanks.
    This matches the CTC training objective in bridge_adapter.py's
    calibrate() and app/training/train_base_model.py — both use blank=0.
    Returns decoded text + mean top-token confidence.
    Placeholder tokenizer: swap _id_to_token for the real vocab saved
    alongside base_model.pt (base_model.vocab.json) once trained."""
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = probs.max(dim=-1)  # (batch, frames)
    confidence = float(top_probs.mean().item())

    ids = top_ids[0].tolist()
    # CTC collapse: drop consecutive duplicates first, THEN drop blanks
    # (this order matters — it's what distinguishes "aa a" -> "a a" from "aaa" -> "a")
    collapsed = [i for idx, i in enumerate(ids) if idx == 0 or i != ids[idx - 1]]
    token_ids = [i for i in collapsed if i != CTC_BLANK_ID]

    tokens = [_id_to_token.get(i, f"<{i}>") for i in token_ids]
    text = " ".join(tokens)
    return text or "(no confident prediction)", confidence


def run_inference(
    pose: torch.Tensor,
    face: torch.Tensor,
    adapter: BridgeAdapterStack | None = None,
) -> dict:
    """Runs one translation pass, with or without a signer-specific adapter.
    Returns predicted_text, confidence, latency_ms, used_adapter — matching
    schemas.TranslationResult."""
    model = get_base_model()
    start = time.perf_counter()

    with torch.no_grad():
        if adapter is not None:
            logits = adapter.forward_with_base(model, pose, face)
        else:
            logits = model(pose, face)

    latency_ms = (time.perf_counter() - start) * 1000
    text, confidence = decode_logits(logits)

    if latency_ms > settings.MAX_INFERENCE_LATENCY_MS:
        # Don't fail the request — just flag it so it shows up in logs/ablation
        # results rather than silently missing the <500ms target.
        pass

    return {
        "predicted_text": text,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "used_adapter": adapter is not None,
    }
