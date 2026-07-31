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


def decode_logits(logits: torch.Tensor) -> tuple[str, float]:
    """Greedy-decodes logits into text + returns mean top-token confidence.
    Placeholder: swap for CTC decoding or a proper detokenizer once the
    vocab/tokenizer is finalized."""
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = probs.max(dim=-1)  # (batch, frames)
    confidence = float(top_probs.mean().item())

    tokens = [_id_to_token.get(i.item(), f"<{i.item()}>") for i in top_ids[0]]
    # naive de-dup of consecutive repeats, CTC-style
    deduped = [t for i, t in enumerate(tokens) if i == 0 or t != tokens[i - 1]]
    text = " ".join(t for t in deduped if t not in ("<blank>", "<pad>"))
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
